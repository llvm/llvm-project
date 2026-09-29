; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target triple = "powerpc-unknown-linux-gnu"

; Dead argument should reserve an FP register.
define double @bar(double %DEAD, double %X, double %Y) {
; CHECK: fadd 1, 2, 3
        %tmp.2 = fadd double %X, %Y              ; <double> [#uses=1]
        ret double %tmp.2
}
