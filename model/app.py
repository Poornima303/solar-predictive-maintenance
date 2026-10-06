from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# =========================================================
# LOAD TRAINED RANDOM FOREST MODEL
# =========================================================

model = joblib.load("model/model.pkl")


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # Default values
    prediction = None
    probability = None
    risk_level = None

    reason = None
    recommendation = None

    performance_percentage = None
    performance_status = None

    energy_loss = None

    health_score = None
    health_status = None

    error = None
    maintenance_cost = None
    alert_message = None
    alert_type = None

    # =====================================================
    # POST REQUEST
    # =====================================================

    if request.method == "POST":

        try:

            # -------------------------------------------------
            # GET VALUES FROM HTML FORM
            # -------------------------------------------------

            gsp_id = float(
                request.form["gsp_id"]
            )

            previous_generation = float(
                request.form["previous_generation"]
            )

            previous_generation_2 = float(
                request.form["previous_generation_2"]
            )

            previous_generation_3 = float(
                request.form["previous_generation_3"]
            )

            lcl_mw = float(
                request.form["lcl_mw"]
            )

            ucl_mw = float(
                request.form["ucl_mw"]
            )

            installedcapacity_mwp = float(
                request.form["installedcapacity_mwp"]
            )

            capacity_mwp = float(
                request.form["capacity_mwp"]
            )

            hour = int(
                request.form["hour"]
            )

            day = int(
                request.form["day"]
            )

            month = int(
                request.form["month"]
            )


            # =================================================
            # PERFORMANCE ANALYSIS
            # =================================================

            if capacity_mwp > 0:

                performance_percentage = round(
                    (previous_generation / capacity_mwp) * 100,
                    2
                )

                # Keep performance between 0% and 100%
                performance_percentage = max(
                    0,
                    min(
                        performance_percentage,
                        100
                    )
                )

                # Performance gap
                energy_loss = round(
                    100 - performance_percentage,
                    2
                )

            else:

                performance_percentage = 0
                energy_loss = 100


            # =================================================
            # PERFORMANCE STATUS
            # =================================================

            if performance_percentage >= 80:

                performance_status = "Good Performance"

            elif performance_percentage >= 50:

                performance_status = "Moderate Performance"

            else:

                performance_status = "Poor Performance"


            # =================================================
            # EQUIPMENT HEALTH SCORE
            # =================================================

            # Performance-based health indicator
            health_score = performance_percentage


            # =================================================
            # HEALTH STATUS
            # =================================================

            if health_score >= 80:

                health_status = "Healthy"

            elif health_score >= 50:

                health_status = "Moderate"

            else:

                health_status = "Critical"


            # =================================================
            # CREATE MODEL INPUT
            # =================================================

            input_data = pd.DataFrame([{

                "gsp_id": gsp_id,

                "previous_generation":
                    previous_generation,

                "previous_generation_2":
                    previous_generation_2,

                "previous_generation_3":
                    previous_generation_3,

                "lcl_mw":
                    lcl_mw,

                "ucl_mw":
                    ucl_mw,

                "installedcapacity_mwp":
                    installedcapacity_mwp,

                "capacity_mwp":
                    capacity_mwp,

                "hour":
                    hour,

                "day":
                    day,

                "month":
                    month

            }])


            # =================================================
            # EXACT MODEL FEATURE ORDER
            # =================================================

            model_columns = [

                "gsp_id",

                "previous_generation",

                "previous_generation_2",

                "previous_generation_3",

                "lcl_mw",

                "ucl_mw",

                "installedcapacity_mwp",

                "capacity_mwp",

                "hour",

                "day",

                "month"

            ]


            # Arrange input exactly like training data
            input_data = input_data[
                model_columns
            ]


            # =================================================
            # RANDOM FOREST PREDICTION
            # =================================================

            result = model.predict(
                input_data
            )[0]


            # =================================================
            # MAINTENANCE PROBABILITY
            # =================================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                probability = round(
                    probabilities[1] * 100,
                    2
                )

            else:

                if result == 1:
                    probability = 100.0
                else:
                    probability = 0.0


            # =================================================
            # RISK LEVEL
            # =================================================

            if probability >= 70:

                risk_level = "HIGH RISK"

            elif probability >= 40:

                risk_level = "MEDIUM RISK"

            else:

                risk_level = "LOW RISK"
                # ESTIMATED MAINTENANCE COST
                if probability >= 70:
                  maintenance_cost = 5000
                elif probability >= 40:
                  maintenance_cost = 3000
                else:
                 maintenance_cost = 1000


            # =================================================
            # MAINTENANCE PREDICTION
            # =================================================

            if result == 1:

                prediction = "Maintenance Required"


                # -------------------------------------------------
                # REASON
                # -------------------------------------------------

                reason = (
                    "Abnormal solar generation pattern detected. "
                    "Recent generation values indicate a possible "
                    "performance issue."
                )


                # -------------------------------------------------
                # MAINTENANCE RECOMMENDATION
                # -------------------------------------------------

                if risk_level == "HIGH RISK":

                    recommendation = (
                        "Immediate inspection of the solar "
                        "generation system is recommended."
                    )

                elif risk_level == "MEDIUM RISK":

                    recommendation = (
                        "Schedule a system inspection and "
                        "continue monitoring generation."
                    )

                else:

                    recommendation = (
                        "Continue monitoring the solar "
                        "generation system."
                    )


            # =================================================
            # SYSTEM NORMAL
            # =================================================

            else:

                prediction = "System Normal"

                reason = (
                    "Solar generation pattern appears normal "
                    "based on the provided parameters."
                )

                recommendation = (
                    "No immediate maintenance action is required. "
                    "Continue regular monitoring."
                )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            error = str(e)

            prediction = None
            probability = None
            risk_level = None

            reason = None
            recommendation = None

            performance_percentage = None
            performance_status = None

            energy_loss = None

            health_score = None
            health_status = None


    # =========================================================
    # SEND RESULTS TO INDEX.HTML
    # =========================================================

    return render_template(

        "index.html",

        prediction=prediction,

        probability=probability,

        risk_level=risk_level,

        reason=reason,

        recommendation=recommendation,

        performance_percentage=performance_percentage,

        performance_status=performance_status,

        energy_loss=energy_loss,

        health_score=health_score,

       health_status=health_status,
        maintenance_cost=maintenance_cost,
        alert_message=alert_message,
        alert_type=alert_type,
        error=error

    )


# =========================================================
# RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )