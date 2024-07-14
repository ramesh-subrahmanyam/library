import optuna
import pandas as pd
import matplotlib.pyplot as plt

class OptunaStudy:
    def __init__(self, study_name, objective_f):
        """
        Initialize an Optuna study.

        Args:
        - study_name (str): Name of the study.
        - objective_f (callable): Objective function to optimize.
        """
        self.objective_f = objective_f
        self.study = optuna.create_study(
                direction='maximize',  # Maximize the objective function
                study_name=study_name  # Assign a name to the study
            )
        
    def run(self, n_trials):
        """
        Run the optimization process.

        Args:
        - n_trials (int): Number of optimization trials to run.
        """
        self.study.optimize(self.objective_f, n_trials=n_trials)

    def plot_trial_results(self):
        """
        Plot the objective values over trials.
        """
        # Extract the objective values from all trials
        objective_values = [trial.value for trial in self.study.trials]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(range(1, len(objective_values) + 1), objective_values, marker='o')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Objective Values Over Trials')
        plt.grid(True)
        plt.show()

    def display_top_trials(self, context_fields, top_N=5):
        """
        Display the top trials based on objective value and parameter values.

        Args:
        - context_fields (list): List of fields to include from trial context.

        Notes:
        - Assumes context fields are available in user attributes of trials.
        """
        # Extract and sort trials by objective value (highest to lowest)
        sorted_trials = sorted(self.study.trials, key=lambda x: x.value, reverse=True)
        
        # Extract context information for the top trials
        contexts = [trial.user_attrs["context"] for trial in sorted_trials[:top_N]]

        # Initialize data dictionary for top trials
        top_trials_data = {
            'Objective Value': [trial.value for trial in sorted_trials[:top_N]]  # Objective values
        }

        # Collect all unique parameter names across top trials
        all_params = set()
        for trial in sorted_trials[:top_N]:
            all_params.update(trial.params.keys())

        # Add each parameter to the data dictionary, handling missing parameters
        for param in all_params:
            top_trials_data[param] = [trial.params.get(param, 'N/A') for trial in sorted_trials[:top_N]]

        # Add specified context fields to the data dictionary
        for field in context_fields:
            top_trials_data[field] = [c[field] for c in contexts]

        # Create a DataFrame from the collected data
        top_trials_df = pd.DataFrame(top_trials_data)

        # Display the DataFrame
        print(top_trials_df)
