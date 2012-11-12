; RUN: llc -O0 -mtriple=powerpc-unknown-linux-gnu   < %s | FileCheck %s
; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; This verifies that the code to update VRSAVE has been removed for SVR4.

define <4 x float> @bar(<4 x float> %v) nounwind {
entry:
  %v.addr = alloca <4 x float>, align 16
  store <4 x float> %v, <4 x float>* %v.addr, align 16
  %0 = load <4 x float>* %v.addr, align 16
  ret <4 x float> %0
}

; CHECK-NOT: mfspr
; CHECK-NOT: mtspr
