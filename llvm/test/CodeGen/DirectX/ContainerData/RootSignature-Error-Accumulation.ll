; RUN: not llc %s --filetype=obj -o -  2>&1 | FileCheck %s


target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: error: Invalid value for Filter: 666
; CHECK: error: Invalid value for AddressU: 667
; CHECK: error: Invalid value for AddressV: 668
; CHECK: error: Invalid value for AddressW: 669
; CHECK: error: Invalid value for ComparisonFunc: 670
; CHECK: error: Invalid value for ShaderVisibility: 666

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"StaticSampler", i32 666, i32 667, i32 668, i32 669, float 0x3FF6CCCCC0000000, i32 9, i32 670, i32 2, float -1.280000e+02, float 1.280000e+02, i32 42, i32 672, i32 666 }
