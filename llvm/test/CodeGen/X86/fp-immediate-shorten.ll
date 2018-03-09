;; Test that this FP immediate is stored in the constant pool as a float.

; RUN: llc < %s -mtriple=i686-- -mattr=-sse2,-sse3 | FileCheck %s

; CHECK: {{.long.1123418112}}

define double @D() {
        ret double 1.230000e+02
}

