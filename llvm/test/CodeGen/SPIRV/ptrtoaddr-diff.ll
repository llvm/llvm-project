; Test sub(ptrtoaddr, ptrtoaddr) -> OpPtrDiff for SPIR-V 1.4+
; RUN: llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-14
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.4-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; Test fallback to OpConvertPtrToU + ISub for SPIR-V < 1.4
; RUN: llc -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-13
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-14: %[[#Arg1:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-14: %[[#Arg2:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-14: %[[#]] = OpPtrDiff %[[#]] %[[#Arg1]] %[[#Arg2]]
; CHECK-SPIRV-14-NOT: OpConvertPtrToU

; CHECK-SPIRV-13: %[[#Arg1:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-13: %[[#Arg2:]] = OpFunctionParameter %[[#]]
; CHECK-SPIRV-13: %[[#Conv1:]] = OpConvertPtrToU %[[#]] %[[#Arg1]]
; CHECK-SPIRV-13: %[[#Conv2:]] = OpConvertPtrToU %[[#]] %[[#Arg2]]
; CHECK-SPIRV-13: %[[#]] = OpISub %[[#]] %[[#Conv1]] %[[#Conv2]]
; CHECK-SPIRV-13-NOT: OpPtrDiff

define spir_kernel void @test_ptrdiff(ptr addrspace(1) %p1, ptr addrspace(1) %p2, ptr addrspace(1) %res) {
entry:
  %a = ptrtoaddr ptr addrspace(1) %p1 to i64
  %b = ptrtoaddr ptr addrspace(1) %p2 to i64
  %diff = sub i64 %a, %b
  store i64 %diff, ptr addrspace(1) %res
  ret void
}
