; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; This test verifies that indirect calls with functions lacking return statements
; are don't cause an error.

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK: OpExtension "SPV_INTEL_function_pointers"

; CHECK-DAG: %[[TyVoid:.*]] = OpTypeVoid
; CHECK-DAG: %[[TyInt32:.*]] = OpTypeInt 32 0

; CHECK-DAG: %[[TyFun:.*]] = OpTypeFunction %[[TyVoid]] %[[TyInt32]] %[[TyInt32]]
; CHECK-DAG: %[[TyPtrFun:.*]] = OpTypePointer CodeSectionINTEL %[[TyFun]]

; CHECK: OpFunction
; CHECK: %[[Load:.*]] = OpLoad %[[TyPtrFun]]
; CHECK: FunctionPointerCallINTEL %[[TyVoid]] %[[Load]]

%struct.ident_t = type { i32, i32, i32, i32, ptr addrspace(4) }

@0 = addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr addrspace(4) addrspacecast (ptr addrspace(1) null to ptr addrspace(4)) }

define spir_func void @foo(ptr addrspace(4) %Ident) addrspace(9) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %0 = load ptr addrspace(9), ptr addrspace(4) null, align 8
  call spir_func addrspace(9) void %0(i32 0, i32 0)
  br label %do.body
}

define spir_func void @bar() addrspace(9) {
entry:
  ret void
}
