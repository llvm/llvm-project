; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#ArrayTy:]] = OpTypeArray %[[#Char]] %[[#]]
; CHECK-SPIRV-DAG: %[[#CharPtrTy:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-SPIRV-DAG: %[[#Const1:]] = OpConstantComposite %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV-DAG: %[[#CharPtrPtrTy:]] = OpTypePointer CrossWorkgroup %[[#CharPtrTy]]
; CHECK-SPIRV-DAG: %[[#PtrArrayTy:]] = OpTypePointer CrossWorkgroup %[[#ArrayTy]]
; CHECK-SPIRV-DAG: OpVariable %[[#PtrArrayTy]] CrossWorkgroup %[[#Const1]]
; CHECK-SPIRV-DAG: OpVariable %[[#CharPtrPtrTy]] CrossWorkgroup %[[#]]

@a_var = addrspace(1) global [2 x i8] c"\01\01"
@p_var = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([2 x i8], ptr addrspace(1) @a_var, i32 0, i64 1)

define spir_func zeroext i8 @foo() {
entry:
  ret i8 1
}

define spir_func zeroext i8 @bar() {
entry:
  ret i8 1
}
