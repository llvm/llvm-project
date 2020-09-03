; RUN: not llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=soft-float -O3 2>&1 | FileCheck %s
;
; Verify that inline asms cannot use fp/vector registers with soft-float.

define <2 x i64> @f1() {
  %ret = call <2 x i64> asm "", "=v" ()
  ret <2 x i64> %ret
}

; CHECK: error: couldn't allocate output register for constraint 'v'
