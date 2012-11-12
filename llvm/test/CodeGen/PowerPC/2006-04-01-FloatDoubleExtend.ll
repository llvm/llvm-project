; RUN: llc < %s -march=ppc32


define double @CalcSpeed(float %tmp127) {
        %tmp145 = fpext float %tmp127 to double         ; <double> [#uses=1]
        %tmp150 = call double asm "frsqrte $0,$1", "=f,f"( double %tmp145 )             ; <double> [#uses=1]
        ret double %tmp150
}

