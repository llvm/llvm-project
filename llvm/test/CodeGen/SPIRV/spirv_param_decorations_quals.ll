; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

define spir_kernel void @k(i32 addrspace(1)* %a) !kernel_arg_type_qual !7 !spirv.ParameterDecorations !10 {
entry:
  ret void
}

; CHECK-SPIRV: OpDecorate %[[#PId:]] Volatile
; CHECK-SPIRV: OpDecorate %[[#PId]] FuncParamAttr NoAlias
; CHECK-SPIRV: %[[#PId]] = OpFunctionParameter %[[#]]

!7 = !{!"volatile"}
!8 = !{i32 38, i32 4} ; FuncParamAttr NoAlias
!9 = !{!8}
!10 = !{!9}
