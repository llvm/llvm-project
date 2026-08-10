;; Test that this FP immediate is stored in the constant pool as a float.

; RUN: llc < %s -mtriple=i686-- -mattr=-sse2,-sse3 | FileCheck %s

; CHECK: {{.long.0x42f60000}}

define double @D() {
        ret double 1.230000e+02
}

