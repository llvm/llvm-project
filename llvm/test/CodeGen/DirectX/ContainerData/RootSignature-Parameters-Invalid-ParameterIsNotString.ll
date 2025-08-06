; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

; CHECK: error: Invalid format for Root Element
; CHECK-NOT: Root Signature Definitions

target triple = "dxil-unknown-shadermodel6.0-compute"


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!0}
!0 = !{ ptr @main, !1, i32 2 }
!1 = !{ !2 }
!2 = !{ i32 0 }
