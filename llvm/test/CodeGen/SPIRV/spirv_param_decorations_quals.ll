; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpDecorate %[[#PId:]] Volatile
; CHECK-SPIRV-DAG: OpDecorate %[[#PId]] FuncParamAttr NoAlias
; CHECK-SPIRV-DAG: OpDecorate %[[#PId]] FuncParamAttr NoWrite
; CHECK-SPIRV-DAG: OpDecorate %[[#PId2:]] Volatile
; CHECK-SPIRV-DAG: OpDecorate %[[#PId3:]] Volatile
; CHECK-SPIRV-DAG: OpDecorate %[[#PId4:]] Volatile

define spir_kernel void @k(ptr addrspace(1) %a) !kernel_arg_type_qual !7 !spirv.ParameterDecorations !10 {
; CHECK-SPIRV: %[[#PId]] = OpFunctionParameter %[[#]]
entry:
  ret void
}

!7 = !{!"volatile"}
!8 = !{i32 38, i32 4} ; FuncParamAttr NoAlias
!11 = !{i32 38, i32 6} ; FuncParamAttr NoWrite
!9 = !{!8, !11}
!10 = !{!9}

define spir_kernel void @k_const_volatile(ptr addrspace(1) %a) !kernel_arg_type_qual !20 {
; CHECK-SPIRV: %[[#PId2]] = OpFunctionParameter %[[#]]
entry:
  ret void
}

!20 = !{!"const volatile"}

define spir_kernel void @k_restrict_volatile(ptr addrspace(1) %a) !kernel_arg_type_qual !21 {
; CHECK-SPIRV: %[[#PId3]] = OpFunctionParameter %[[#]]
entry:
  ret void
}

!21 = !{!"restrict volatile"}

define spir_kernel void @k_restrict_const_volatile(ptr addrspace(1) %a) !kernel_arg_type_qual !22 {
; CHECK-SPIRV: %[[#PId4]] = OpFunctionParameter %[[#]]
entry:
  ret void
}

!22 = !{!"restrict const volatile"}
