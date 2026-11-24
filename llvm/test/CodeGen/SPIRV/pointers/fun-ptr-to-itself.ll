; RUN: llc -mtriple=spirv32-unknown-unknown -O0 %s -o - --spirv-ext=+SPV_INTEL_function_pointers | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - --spirv-ext=+SPV_INTEL_function_pointers -filetype=obj | spirv-val %}

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
; The OpVariable for @fp must use CodeSectionINTEL to match its OpConstantFunctionPointerINTEL initializer.
; CHECK-DAG: %[[#Int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#FunTy2:]] = OpTypeFunction %[[#Int32]] %[[#Int32]]
; CHECK-DAG: %[[#CodePtrTy2:]] = OpTypePointer CodeSectionINTEL %[[#FunTy2]]
; CHECK-DAG: %[[#GenPtrTy2:]] = OpTypePointer Generic %[[#FunTy2]]
; CHECK-DAG: %[[#GenPtrPtrTy2:]] = OpTypePointer Function %[[#GenPtrTy2]]
; CHECK-DAG: %[[#VarTy2:]] = OpTypePointer Function %[[#CodePtrTy2]]
; CHECK-DAG: %[[#FnPtr2:]] = OpConstantFunctionPointerINTEL %[[#CodePtrTy2]] %[[#Callback:]]
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

; CHECK:     %[[#Fp:]] = OpVariable %[[#VarTy2]] Function %[[#FnPtr2]]
; CHECK:     %[[#BC2:]] = OpBitcast %[[#GenPtrPtrTy2]] %[[#Fp]]
; CHECK:     %[[#Ptr:]] = OpLoad %[[#GenPtrTy2]] %[[#BC2]]
; CHECK:     OpFunctionPointerCallINTEL %[[#Int32]] %[[#Ptr]]

@fp = global ptr addrspace(4) @callback

define void @caller() {
  %ptr = load ptr addrspace(4), ptr @fp
  %r = call addrspace(4) i32 %ptr(i32 0)
  ret void
}

; CHECK: %[[#Callback]] = OpFunction %[[#Int32]] None %[[#FunTy2]]
define i32 @callback(i32 %x) addrspace(4) {
  ret i32 %x
}
