; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s
target triple = "dxil-pc-shadermodel6.0-compute"

; CHECK: Shader Model Version : 6.0
; CHECK-NEXT: DXIL Version : 1.0
; CHECK-NEXT: Shader Stage : compute
; CHECK-NEXT: Validator Version : 0
; CHECK-NEXT: entry
; CHECK-NEXT:   Function Shader Stage : compute
; CHECK-NEXT:   NumThreads: 1,1,1
; CHECK-NEXT: Strip Root Signature: 0
; CHECK-EMPTY:

define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!0, !2} ; list of function/root signature pairs
!0 = !{i1 false} ; don't strip root signature
!2 = !{ ptr @entry, !3, i32 2 } ; function, root signature, version
!3 = !{} ; empty root signature
