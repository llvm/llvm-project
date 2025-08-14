; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_runtime_aligned %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability RuntimeAlignedAttributeINTEL
; CHECK: OpExtension "SPV_INTEL_runtime_aligned"
; CHECK: OpName %[[#ARGA:]] "a"
; CHECK: OpName %[[#ARGB:]] "b"
; CHECK: OpName %[[#ARGC:]] "c"
; CHECK: OpName %[[#ARGD:]] "d"
; CHECK: OpName %[[#ARGE:]] "e"
; CHECK: OpDecorate %[[#ARGA]] FuncParamAttr RuntimeAlignedINTEL
; CHECK-NOT: OpDecorate %[[#ARGB]] FuncParamAttr RuntimeAlignedINTEL
; CHECK: OpDecorate %[[#ARGC]] FuncParamAttr RuntimeAlignedINTEL
; CHECK-NOT: OpDecorate %[[#ARGD]] FuncParamAttr RuntimeAlignedINTEL
; CHECK-NOT: OpDecorate %[[#ARGE]] FuncParamAttr RuntimeAlignedINTEL

; CHECK: OpFunction
; CHECK: %[[#ARGA]] = OpFunctionParameter %[[#]]
; CHECK: %[[#ARGB]] = OpFunctionParameter %[[#]]
; CHECK: %[[#ARGC]] = OpFunctionParameter %[[#]]
; CHECK: %[[#ARGD]] = OpFunctionParameter %[[#]]
; CHECK: %[[#ARGE]] = OpFunctionParameter %[[#]]

define spir_kernel void @test(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %d, i32 %e) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !9 !kernel_arg_runtime_aligned !10 {
entry:
  ret void
}

!0 = !{i32 2, i32 2}
!1 = !{i32 0, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{}
!4 = !{i16 6, i16 14}
!5 = !{i32 1, i32 1, i32 1, i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none", !"none"}
!7 = !{!"int*", !"float*", !"int*"}
!8 = !{!"", !"", !"", !"", !""}
!9 = !{!"int*", !"float*", !"int*", !"int", !"int"}
!10 = !{i1 true, i1 false, i1 true, i1 false, i1 false}
