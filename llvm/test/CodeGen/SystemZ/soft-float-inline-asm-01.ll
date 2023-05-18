; RUN: not llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -mattr=soft-float -O3 2>&1 | FileCheck %s
;
; Verify that inline asms cannot use fp/vector registers with soft-float.

define float @f1() {
  %ret = call float asm "", "=f" ()
  ret float %ret
}

; CHECK: error: couldn't allocate output register for constraint 'f'
