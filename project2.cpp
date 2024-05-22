#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

using namespace std;

// Class to handle data operations
class DataHandler {
public:
    // Constructor to initialize the file path
    DataHandler(const string& filepath) : filepath(filepath) {}

    // Method to read data from a CSV file and store it in a list
    vector<vector<string> > read_csv() {
        ifstream file(filepath); // Open the file
        string line; // Variable to store each line
        vector<vector<string> > dataset; // Vector to store the dataset
        bool header = true; // Flag to skip the header
        while (getline(file, line)) { // Read each line from the file
            if (header) { // Skip the header
                header = false;
                continue;
            }
            vector<string> row; // Vector to store each row
            istringstream ss(line); // String stream to split the line
            string cell; // Variable to store each cell
            while (getline(ss, cell, ',')) { // Split the line by comma
                row.push_back(cell); // Add each cell to the row
            }
            dataset.push_back(row); // Add the row to the dataset
        }
        return dataset; // Return the dataset
    }

    // Method to split the dataset into training and testing sets
    void train_test_split(const vector<vector<string> >& dataset, 
                          vector<vector<string> >& train_set, 
                          vector<vector<string> >& test_set, 
                          double test_size = 0.2) {
        vector<vector<string> > shuffled_dataset = dataset; // Copy the dataset
        random_device rd; // Random device for seeding
        mt19937 g(rd()); // Mersenne Twister random number generator
        shuffle(shuffled_dataset.begin(), shuffled_dataset.end(), g); // Shuffle the dataset
        size_t split_index = static_cast<size_t>(dataset.size() * (1 - test_size)); // Determine the split index
        train_set.assign(shuffled_dataset.begin(), shuffled_dataset.begin() + split_index); // Assign the training set
        test_set.assign(shuffled_dataset.begin() + split_index, shuffled_dataset.end()); // Assign the testing set
    }

    // Method to separate features and labels from the dataset
    void separate_features_labels(const vector<vector<string> >& dataset, 
                                  vector<vector<double> >& features, 
                                  vector<string>& labels) {
        for (size_t i = 0; i < dataset.size(); ++i) { // Iterate over each row in the dataset
            const vector<string>& data = dataset[i];
            vector<double> feature; // Vector to store features
            for (size_t j = 1; j < data.size() - 1; ++j) { // Exclude the first and last columns
                feature.push_back(stod(data[j])); // Convert string to double and add to features
            }
            features.push_back(feature); // Add features to the features vector
            labels.push_back(data.back()); // Add the label (last column) to the labels vector
        }
    }

private:
    string filepath; // Variable to store the file path
};

// Class for the Naive Bayes classifier
class NaiveBayesClassifier {
public:
    // Method to train the classifier
    void fit(const vector<vector<double> >& X, const vector<string>& y) {
        calculate_class_probabilities(y); // Calculate class probabilities
        calculate_means_stds(X, y); // Calculate means and standard deviations
    }

    // Method to predict the class labels for the input features
    vector<string> predict(const vector<vector<double> >& X) {
        vector<string> predictions; // Vector to store predictions
        for (size_t i = 0; i < X.size(); ++i) { // Iterate over each feature set
            predictions.push_back(predict_single(X[i])); // Predict and add to predictions
        }
        return predictions; // Return predictions
    }

    // Method to generate a classification report
    map<string, map<string, double> > classification_report(
        const vector<string>& y_true, const vector<string>& y_pred) {
        map<string, map<string, double> > report; // Map to store the report
        set<string> unique_labels(y_true.begin(), y_true.end()); // Get unique labels
        for (set<string>::iterator it = unique_labels.begin(); it != unique_labels.end(); ++it) { // Iterate over each label
            string label = *it;
            int true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0; // Initialize counts
            for (size_t i = 0; i < y_true.size(); ++i) {
                if (y_true[i] == label && y_pred[i] == label) true_positive++;
                else if (y_true[i] != label && y_pred[i] == label) false_positive++;
                else if (y_true[i] == label && y_pred[i] != label) false_negative++;
                else true_negative++;
            }
            double precision = true_positive / static_cast<double>(true_positive + false_positive); // Calculate precision
            double recall = true_positive / static_cast<double>(true_positive + false_negative); // Calculate recall
            double f1_score = 2 * (precision * recall) / (precision + recall); // Calculate F1-score
            map<string, double> metrics;
            metrics["precision"] = precision;
            metrics["recall"] = recall;
            metrics["f1_score"] = f1_score;
            report[label] = metrics; // Add to report
        }
        double accuracy = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == y_pred[i]) accuracy++;
        }
        accuracy /= y_true.size();
        report["accuracy"]["accuracy"] = accuracy; // Add accuracy to report
        return report; // Return report
    }

private:
    map<string, vector<double> > means; // Map to store means for each class
    map<string, vector<double> > stds; // Map to store standard deviations for each class
    map<string, double> class_probabilities; // Map to store class probabilities

    // Method to calculate class probabilities based on label frequency
    void calculate_class_probabilities(const vector<string>& y) {
        map<string, int> class_counts; // Map to count occurrences of each class
        for (size_t i = 0; i < y.size(); ++i) { // Iterate over each label
            class_counts[y[i]]++;
        }
        int total_count = y.size(); // Get total number of samples
        for (map<string, int>::iterator it = class_counts.begin(); it != class_counts.end(); ++it) { // Iterate over class counts
            string label = it->first;
            int count = it->second;
            class_probabilities[label] = static_cast<double>(count) / total_count; // Calculate probability
        }
    }

    // Method to calculate means and standard deviations for each feature for each class
    void calculate_means_stds(const vector<vector<double> >& X, const vector<string>& y) {
        map<string, vector<vector<double> > > separated_by_class = separate_by_class(X, y); // Separate data by class
        for (map<string, vector<vector<double> > >::iterator it = separated_by_class.begin(); it != separated_by_class.end(); ++it) { // Iterate over each class
            string label = it->first;
            vector<vector<double> > features = it->second;
            vector<double> mean(features[0].size(), 0.0); // Initialize mean vector
            vector<double> std(features[0].size(), 0.0); // Initialize standard deviation vector
            for (size_t i = 0; i < features.size(); ++i) { // Iterate over each feature set
                for (size_t j = 0; j < features[i].size(); ++j) { // Iterate over each feature
                    mean[j] += features[i][j]; // Sum features
                }
            }
            for (size_t i = 0; i < mean.size(); ++i) { // Calculate means
                mean[i] /= features.size();
            }
            for (size_t i = 0; i < features.size(); ++i) { // Calculate standard deviations
                for (size_t j = 0; j < features[i].size(); ++j) {
                    std[j] += (features[i][j] - mean[j]) * (features[i][j] - mean[j]);
                }
            }
            for (size_t i = 0; i < std.size(); ++i) {
                std[i] = sqrt(std[i] / features.size());
            }
            means[label] = mean; // Store means
            stds[label] = std; // Store standard deviations
        }
    }

    // Helper function to separate the dataset by class
    map<string, vector<vector<double> > > separate_by_class(const vector<vector<double> >& X, 
                                                          const vector<string>& y) {
        map<string, vector<vector<double> > > separated; // Map to store separated data
        for (size_t i = 0; i < X.size(); ++i) { // Iterate over each sample
            separated[y[i]].push_back(X[i]); // Add to corresponding class
        }
        return separated; // Return separated data
    }

    // Method to predict the class label for a single input feature set
    string predict_single(const vector<double>& x) {
        map<string, double> probabilities = calculate_class_probabilities_single(x); // Calculate probabilities
        typedef pair<const string, double> pair_type; // Define pair type
        return max_element(probabilities.begin(), probabilities.end(),
            [](const pair_type &a, const pair_type &b) { return a.second < b.second; })->first;
    }

    // Method to calculate class probabilities for a single input feature set
    map<string, double> calculate_class_probabilities_single(const vector<double>& x) {
        map<string, double> probabilities; // Map to store probabilities
        for (map<string, double>::iterator it = class_probabilities.begin(); it != class_probabilities.end(); ++it) { // Iterate over each class
            string label = it->first;
            double class_prob = it->second;
            probabilities[label] = class_prob; // Initialize with class probability
            for (size_t i = 0; i < x.size(); ++i) { // Multiply by feature probabilities
                probabilities[label] *= calculate_probability(x[i], means[label][i], stds[label][i]);
            }
        }
        return probabilities; // Return probabilities
    }

    // Method to calculate the Gaussian probability density function
    double calculate_probability(double x, double mean, double std) {
        double exponent = exp(-((x - mean) * (x - mean) / (2 * std * std))); // Calculate exponent
        return (1 / (sqrt(2 * M_PI) * std)) * exponent; // Return probability
    }
};

int main() {
    string filepath = "banana_quality.csv"; // Define the path to the CSV file containing the dataset

    DataHandler data_handler(filepath); // Initialize the data handler with the filepath

    vector<vector<string> > dataset = data_handler.read_csv(); // Read the dataset

    vector<vector<string> > train_set, test_set; // Vectors to store training and testing sets
    data_handler.train_test_split(dataset, train_set, test_set); // Split the dataset into training and testing sets

    vector<vector<double> > train_features, test_features; // Vectors to store training and testing features
    vector<string> train_labels, test_labels; // Vectors to store training and testing labels
    data_handler.separate_features_labels(train_set, train_features, train_labels); // Separate features and labels for the training set
    data_handler.separate_features_labels(test_set, test_features, test_labels); // Separate features and labels for the testing set

    NaiveBayesClassifier classifier; // Initialize the Naive Bayes Classifier

    classifier.fit(train_features, train_labels); // Fit the classifier on the training data

    vector<string> predictions = classifier.predict(test_features); // Predict the class labels for the test set features

    map<string, map<string, double> > report = classifier.classification_report(test_labels, predictions); // Generate a classification report comparing the true labels and predicted labels

    cout << "Classification Report:" << endl; // Print the classification report
    for (map<string, map<string, double> >::iterator it = report.begin(); it != report.end(); ++it) { // Iterate over each label
        string label = it->first;
        map<string, double> metrics = it->second;
        cout << "Class " << label << ":" << endl;
        for (map<string, double>::iterator mit = metrics.begin(); mit != metrics.end(); ++mit) { // Iterate over each metric
            cout << "  " << mit->first << ": " << mit->second << endl; // Print metric
        }
        cout << endl;
    }

    return 0;
}
