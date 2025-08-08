; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.rootsignatures = !{!0, !2} ; list of function/root signature pairs
!0 = !{i1 true} ; strip root signature
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"DescriptorTable", i32 0, !6, !7 }
!6 = !{ !"SRV", i32 1, i32 1, i32 0, i32 -1, i32 4 }
!7 = !{ !"UAV", i32 5, i32 1, i32 10, i32 5, i32 2 }

; Check "required" parts are present
; DXC: - Name:             DXIL
; DXC: - Name:             HASH
; DXC: - Name:             PSV0

; CHECK:     @dx.dxil = private constant
; CHECK:     @dx.hash = private constant
; CHECK:     @dx.psv0 = private constant

; But no RTS0 (root signature) part
; DXC-NOT: - Name:            RTS0
; CHECK-NOT: @dx.rts0 = private constant
