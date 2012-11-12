; RUN: llc < %s -march=x86-64 -mattr=+sse2 | not grep cvtss2sd
; PR1264

define double @foo(double %x) {
        %y = fmul double %x, 5.000000e-01
        ret double %y
}
