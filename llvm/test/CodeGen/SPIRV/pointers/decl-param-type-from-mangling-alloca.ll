; An alloca passed to an external declaration whose mangled name gives the
; pointer parameter a different pointee type keeps its allocated type, and the
; pointer is cast at the call. The allocated type decides the size of the
; variable, both for fixed-size and variable-length allocations.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array,+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_variable_length_array %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Foo:]] "_Z3fooPj"
; CHECK-DAG: OpName %[[#Bar:]] "_Z3barPDv4_j"
; CHECK-DAG: OpName %[[#Counted:]] "counted"
; CHECK-DAG: OpName %[[#Array:]] "array"
; CHECK-DAG: OpName %[[#VecCounted:]] "vec_counted"
; CHECK-DAG: OpName %[[#Same:]] "same"
; CHECK-DAG: OpName %[[#DynVec:]] "dyn_vec"
; CHECK-DAG: OpName %[[#DynChar:]] "dyn_char"
; CHECK-DAG: OpName %[[#DynVecParam:]] "dyn_vec_param"
; CHECK-DAG: OpName %[[#NotEntry:]] "not_entry"

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4Int:]] = OpTypeVector %[[#Int]] 4
; CHECK-DAG: %[[#Two:]] = OpConstant %[[#Int]] 2{{$}}
; CHECK-DAG: %[[#Eight:]] = OpConstant %[[#Int]] 8{{$}}
; CHECK-DAG: %[[#IntArr8:]] = OpTypeArray %[[#Int]] %[[#Eight]]
; CHECK-DAG: %[[#V4IntArr2:]] = OpTypeArray %[[#V4Int]] %[[#Two]]

; TYPED-DAG: %[[#PtrInt:]] = OpTypePointer Function %[[#Int]]
; TYPED-DAG: %[[#PtrV4Int:]] = OpTypePointer Function %[[#V4Int]]
; TYPED-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]
; TYPED-DAG: %[[#PtrIntArr8:]] = OpTypePointer Function %[[#IntArr8]]
; TYPED-DAG: %[[#PtrV4IntArr2:]] = OpTypePointer Function %[[#V4IntArr2]]
; UNTYPED-DAG: %[[#Ptr:]] = OpTypeUntypedPointerKHR Function

declare spir_func void @_Z3fooPj(ptr)
declare spir_func void @_Z3barPDv4_j(ptr)

; CHECK: %[[#Counted]] = OpFunction
; TYPED: %[[#Var:]] = OpVariable %[[#PtrIntArr8]] Function
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableKHR %[[#Ptr]] Function %[[#IntArr8]]
; CHECK: OpFunctionEnd
define spir_kernel void @counted() {
  %buf = alloca i32, i32 8, align 4
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; CHECK: %[[#Array]] = OpFunction
; TYPED: %[[#Var:]] = OpVariable %[[#PtrIntArr8]] Function
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableKHR %[[#Ptr]] Function %[[#IntArr8]]
; CHECK: OpFunctionEnd
define spir_kernel void @array() {
  %buf = alloca [8 x i32], align 4
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; CHECK: %[[#VecCounted]] = OpFunction
; TYPED: %[[#Var:]] = OpVariable %[[#PtrV4IntArr2]] Function
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableKHR %[[#Ptr]] Function %[[#V4IntArr2]]
; CHECK: OpFunctionEnd
define spir_kernel void @vec_counted() {
  %buf = alloca <4 x i32>, i32 2, align 16
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; The allocated type already matches the parameter, so no cast is needed.
; CHECK: %[[#Same]] = OpFunction
; TYPED: %[[#Var:]] = OpVariable %[[#PtrInt]] Function
; TYPED-NOT: OpBitcast
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Var]]
; UNTYPED: OpUntypedVariableKHR %[[#Ptr]] Function %[[#Int]]
; CHECK: OpFunctionEnd
define spir_kernel void @same() {
  %buf = alloca i32, align 4
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; CHECK: %[[#DynVec]] = OpFunction
; CHECK: %[[#N:]] = OpFunctionParameter %[[#Int]]
; TYPED: %[[#Var:]] = OpVariableLengthArrayINTEL %[[#PtrV4Int]] %[[#N]]
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableLengthArrayINTEL %[[#Ptr]] %[[#V4Int]] %[[#N]]
; CHECK: OpFunctionEnd
define spir_kernel void @dyn_vec(i32 %n) {
  %buf = alloca <4 x i32>, i32 %n, align 16
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; CHECK: %[[#DynChar]] = OpFunction
; CHECK: %[[#N:]] = OpFunctionParameter %[[#Int]]
; TYPED: %[[#Var:]] = OpVariableLengthArrayINTEL %[[#PtrChar]] %[[#N]]
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableLengthArrayINTEL %[[#Ptr]] %[[#Char]] %[[#N]]
; CHECK: OpFunctionEnd
define spir_kernel void @dyn_char(i32 %n) {
  %buf = alloca i8, i32 %n, align 4
  call spir_func void @_Z3fooPj(ptr %buf)
  ret void
}

; CHECK: %[[#DynVecParam]] = OpFunction
; CHECK: %[[#N:]] = OpFunctionParameter %[[#Int]]
; TYPED: %[[#Var:]] = OpVariableLengthArrayINTEL %[[#PtrInt]] %[[#N]]
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrV4Int]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Bar]] %[[#Cast]]
; UNTYPED: OpUntypedVariableLengthArrayINTEL %[[#Ptr]] %[[#Int]] %[[#N]]
; CHECK: OpFunctionEnd
define spir_kernel void @dyn_vec_param(i32 %n) {
  %buf = alloca i32, i32 %n, align 16
  call spir_func void @_Z3barPDv4_j(ptr %buf)
  ret void
}

; A constant count outside the entry block keeps the variable-length lowering.
; CHECK: %[[#NotEntry]] = OpFunction
; TYPED: %[[#Var:]] = OpVariableLengthArrayINTEL %[[#PtrV4Int]] %[[#Two]]
; TYPED: %[[#Cast:]] = OpBitcast %[[#PtrInt]] %[[#Var]]
; TYPED: OpFunctionCall %[[#]] %[[#Foo]] %[[#Cast]]
; UNTYPED: OpUntypedVariableLengthArrayINTEL %[[#Ptr]] %[[#V4Int]] %[[#Two]]
; CHECK: OpFunctionEnd
define spir_kernel void @not_entry(i1 %c) {
entry:
  br i1 %c, label %body, label %exit
body:
  %buf = alloca <4 x i32>, i32 2, align 16
  call spir_func void @_Z3fooPj(ptr %buf)
  br label %exit
exit:
  ret void
}
