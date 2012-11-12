; RUN: llc < %s | not grep ", f1"

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.2.0"

; Dead argument should reserve an FP register.
define double @bar(double %DEAD, double %X, double %Y) {
        %tmp.2 = fadd double %X, %Y              ; <double> [#uses=1]
        ret double %tmp.2
}
