; XFAIL: asan
; RUN: llc -mtriple=spirv32-unknown-unknown < %s --spirv-ext=+SPV_INTEL_function_pointers | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv32-unknown-unknown < %s -filetype=obj | not spirv-val 2>&1 | FileCheck --check-prefix=SPIRV-VAL %s %}

; spirv-val poorly supports the SPV_INTEL_function_pointers extension.
; In this case the function type used in a pointer type fails.
; SPIRV-VAL: Invalid use of function type result id '[[#ID:]][%{{.*}}]'.
; SPIRV-VAL: %_ptr_Generic_4 = OpTypePointer Generic %[[#ID]]

; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_function_pointers"
; CHECK-DAG: %[[#Void:]] = OpTypeVoid
; CHECK-DAG: %[[#FnTy:]] = OpTypeFunction %[[#Void]]
; CHECK-DAG: %[[#GenPtrTy:]] = OpTypePointer Generic %[[#FnTy]]
; CHECK-DAG: %[[#GenPtrPtrTy:]] = OpTypePointer Function %[[#GenPtrTy]]
; CHECK-DAG: %[[#Int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int8PtrTy:]] = OpTypePointer Function %[[#Int8]]
; CHECK-DAG: %[[#CodePtrTy:]] = OpTypePointer CodeSectionINTEL %[[#FnTy]]
; CHECK-DAG: %[[#Null:]] = OpConstantNull %[[#Int8PtrTy]]
; CHECK-DAG: %[[#FnPtr:]] = OpConstantFunctionPointerINTEL %[[#CodePtrTy]] %[[#FnDef:]]
; CHECK:     %[[#FnDef]] = OpFunction %[[#Void]] None %[[#FnTy]]
; CHECK:     %[[#Cast:]] = OpPtrCastToGeneric %[[#GenPtrTy]] %[[#FnPtr]]
; CHECK:     %[[#BC:]] = OpBitcast %[[#GenPtrPtrTy]] %[[#Null]]
; CHECK:     OpStore %[[#BC]] %[[#Cast]] Aligned 8
; CHECK:     OpReturn
; CHECK:     OpFunctionEnd

define void @foo() {
entry:
  store ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4)), ptr null, align 8
  ret void
}

