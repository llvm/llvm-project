; RUN: opt %s -dxil-embed -dxil-globals -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK: @dx.isg1 = private constant [8 x i8] c"\00\00\00\00\08\00\00\00", section "ISG1", align 4
; CHECK: @dx.osg1 = private constant [8 x i8] c"\00\00\00\00\08\00\00\00", section "OSG1", align 4

define void @main() #0 {
entry:
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!dx.valver = !{!0}

!0 = !{i32 1, i32 7}

; DXC: - Name:            ISG1
; DXC-NEXT:   Size:            8
; DXC-NEXT:   Signature:
; DXC-NEXT:     Parameters:      []
; DXC: - Name:            OSG1
; DXC-NEXT:   Size:            8
; DXC-NEXT:   Signature:
; DXC-NEXT:     Parameters:      []
