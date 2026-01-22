; RUN: llc -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck --check-prefixes=CHECK,SPIRV %s
; RUN: llc -verify-machineinstrs -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck --check-prefixes=CHECK,AMDGCNSPIRV %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
;
; Verify that we lower the embedded bitcode

@llvm.embedded.module = private addrspace(1) constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
@llvm.compiler.used = appending addrspace(1) global [1 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.embedded.module to ptr addrspace(4))], section "llvm.metadata"

; CHECK: OpName %[[#LLVM_EMBEDDED_MODULE:]] "llvm.embedded.module"
; CHECK: OpDecorate %[[#LLVM_EMBEDDED_MODULE]] Constant
; CHECK: %[[#UCHAR:]] = OpTypeInt 8 0
; AMDGCNSPIRV: %[[#UINT64:]] = OpTypeInt 64 0
; SPIRV: %[[#UCHAR_PTR:]] = OpTypePointer Generic %[[#UCHAR]]
; AMDGCNSPIRV: %[[#UINT64_MAX:]] = OpConstant %[[#UINT64]] 18446744073709551615
; AMDGCNSPIRV: %[[#UCHAR_ARR_UINT64_MAX:]] = OpTypeArray %[[#UCHAR]] %[[#UINT64_MAX]]
; AMDGCNSPIRV: %[[#UCHAR_ARR_UINT64_MAX_PTR:]] = OpTypePointer CrossWorkgroup %[[#UCHAR_ARR_UINT64_MAX]]
; AMDGCNSPIRV: %[[#CONST_UCHAR_ARR_UINT64_MAX:]] = OpConstantNull %[[#UCHAR_ARR_UINT64_MAX]]
; SPIRV: %[[#CONST_UCHAR_NULL_PTR:]] = OpConstantNull %[[#UCHAR_PTR]]
; AMDGCNSPIRV: %[[#LLVM_EMBEDDED_MODULE]] = OpVariable %[[#UCHAR_ARR_UINT64_MAX_PTR]] CrossWorkgroup %[[#CONST_UCHAR_ARR_UINT64_MAX]]
; SPIRV: %[[#LLVM_EMBEDDED_MODULE]] = OpVariable %[[#]] CrossWorkgroup %[[#CONST_UCHAR_NULL_PTR]]

define spir_kernel void @foo() {
entry:
  ret void
}
