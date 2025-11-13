; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-intel --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-intel %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK: OpName %[[F1:.*]] "f1"
; CHECK: OpName %[[F2:.*]] "f2"

; CHECK: %[[TyBool:.*]] = OpTypeBool

; CHECK %[[F1Ptr:.*]] = OpConstantFunctionPointerINTEL %{{.*}} %[[F2]]
; CHECK %[[F2Ptr:.*]] = OpConstantFunctionPointerINTEL %{{.*}} %[[F2]]

; CHECK OpPtrEqual %[[TyBool]] %[[F1Ptr]] %[[F2Ptr]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1-P9-A0"
target triple = "spirv64-intel"

define spir_func void @f1() addrspace(9) {
entry:
  ret void
}

define spir_func void @f2() addrspace(9) {
entry:
  ret void
}

define spir_kernel void @foo() addrspace(9) {
entry:
  %a = icmp eq ptr addrspace(9) @f1, @f2
  ret void
}
