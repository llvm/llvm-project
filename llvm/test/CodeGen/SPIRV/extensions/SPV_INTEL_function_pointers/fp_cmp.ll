; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK: OpName %[[F1:.*]] "f1"
; CHECK: OpName %[[ARG:.*]] "arg"

; CHECK: %[[TyBool:.*]] = OpTypeBool

; CHECK: %[[F1Ptr:.*]] = OpConstantFunctionPointerINTEL %{{.*}} %[[F1]]

; CHECK: OpPtrEqual %[[TyBool]] %[[F1Ptr]] %[[ARG]]

target triple = "spirv64"

define spir_func void @f1() addrspace(9) {
entry:
  ret void
}

define spir_func i1 @foo(ptr addrspace(9) %arg) addrspace(9) {
entry:
  %a = icmp eq ptr addrspace(9) @f1, %arg
  ret i1 %a
}
