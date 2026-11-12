; RUN: llc < %s

define double @fneg(double %X) {
        %Y = fsub double -0.000000e+00, %X               ; <double> [#uses=1]
        ret double %Y
}

define float @fnegf(float %X) {
        %Y = fsub float -0.000000e+00, %X                ; <float> [#uses=1]
        ret float %Y
}

define double @real_fneg(double %X) {
        %Y = fneg double %X               ; <double> [#uses=1]
        ret double %Y
}

define double @real_fneg_constant() {
        %Y = fneg double -2.0             ; <double> [#uses=1]
        ret double %Y
}

define float @real_fnegf(float %X) {
        %Y = fneg float %X                ; <float> [#uses=1]
        ret float %Y
}

declare double @fabs(double)

declare float @fabsf(float)

define double @fabstest(double %X) {
        %Y = call double @fabs( double %X )             ; <double> [#uses=1]
        ret double %Y
}

define float @fabsftest(float %X) {
        %Y = call float @fabsf( float %X )              ; <float> [#uses=1]
        ret float %Y
}

