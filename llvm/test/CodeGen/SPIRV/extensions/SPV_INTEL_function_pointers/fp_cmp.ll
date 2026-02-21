; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck --check-prefix=SPIRV-VAL %s %}

; spirv-val poorly supports the SPV_INTEL_function_pointers extension.
; In this case the function is declared after the constant so it fails.
; SPIRV-VAL: ID '{{[0-9]+}}[%f1]' has not been defined
; SPIRV-VAL: OpConstantFunctionPointerINTEL %_ptr_CodeSectionINTEL_5 %f1

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
