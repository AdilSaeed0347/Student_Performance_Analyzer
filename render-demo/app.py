from flask import Flask, request, render_template
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Enhanced model loading with error handling
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
            
            # Verify model has required methods
            if not all(hasattr(model, attr) for attr in ['predict', 'predict_proba']):
                raise RuntimeError("Invalid model - missing required methods")
                
            return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate form data with improved checks
        input_data = {
            'IQ': float(request.form.get('IQ', 0)),
            'CGPA': float(request.form.get('CGPA', 0)),
            '10th Marks': float(request.form.get('10th_Marks', 0)),
            '12th Marks': float(request.form.get('12th_Marks', 0)),
            'Communication Skills': float(request.form.get('Communication_Skills', 0))
        }

        # Enhanced validation with specific messages
        validations = [
            (input_data['IQ'], 70, 160, "IQ must be between 70-160"),
            (input_data['CGPA'], 0, 10, "CGPA must be 0.0-10.0"),
            (input_data['10th Marks'], 0, 100, "10th Marks must be 0-100%"),
            (input_data['12th Marks'], 0, 100, "12th Marks must be 0-100%"),
            (input_data['Communication Skills'], 0, 10, "Communication must be 0-10")
        ]

        for value, min_val, max_val, error_msg in validations:
            if not (min_val <= value <= max_val):
                raise ValueError(error_msg)

        # Prepare input with additional checks
        features = np.array(list(input_data.values())).reshape(1, -1)
        
        if features.shape[1] != 5:
            raise ValueError("Exactly 5 features required")

        # Predict with confidence threshold adjustment
        proba = model.predict_proba(features)[0][1] * 100
        prediction = 1 if proba >= 60 else 0  # Using 60% threshold instead of 50%

        # Enhanced result formatting
        if prediction == 1:
            prediction_text = (
                f"🎉 High Placement Potential ({proba:.1f}% confidence)\n\n"
                "Key strengths identified:\n"
                f"- Cognitive Ability: {input_data['IQ']} IQ\n"
                f"- Academic Performance: CGPA {input_data['CGPA']}/10\n"
                f"- Communication: {input_data['Communication Skills']}/10\n\n"
                "Recommendation: Priority candidate for campus recruitment."
            )
        else:
            prediction_text = (
                f"🔍 Needs Improvement ({proba:.1f}% confidence)\n\n"
                "Development areas:\n"
                f"- Current CGPA: {input_data['CGPA']} (Target: 7.0+)\n"
                f"- Communication: {input_data['Communication Skills']}/10 (Target: 7.0+)\n\n"
                "Action Plan:\n"
                "- Complete 2 industry projects\n"
                "- Attend communication workshops"
            )

        return render_template('index.html',
                           prediction_text=prediction_text,
                           input_summary=input_data,
                           prediction=prediction,
                           prediction_prob=proba)

    except ValueError as ve:
        return render_template('index.html', 
                            prediction_text=f"Validation Error: {str(ve)}")
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                            prediction_text="System error. Please try again.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)