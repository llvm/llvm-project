; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: Invalid value for Num32BitValues
; CHECK-NOT: Root Signature Definitions

define void @main() {
entry:
  ret void
}

!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"RootConstants", i32 0, i32 1, i32 2, !"Invalid" }
