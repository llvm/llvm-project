;; Test that this FP immediate is stored in the constant pool as a float.

; RUN: llc < %s -march=x86 -mattr=-sse2,-sse3 | \
; RUN:   grep ".long.1123418112"

define double @D() {
        ret double 1.230000e+02
}

