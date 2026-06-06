; RUN: not opt -passes='print<dxil-root-signature>' %s -S -o - 2>&1 | FileCheck %s

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: Invalid Root Signature metadata - expected function, signature, and version.
; CHECK-NOT: Root Signature Definitions


define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!1} ; list of function/root signature pairs
!1= !{ !"RootFlags" } ; function, root signature
