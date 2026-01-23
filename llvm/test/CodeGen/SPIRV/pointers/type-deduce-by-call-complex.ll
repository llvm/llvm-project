; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[Long:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[Void:.*]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[Struct:.*]] = OpTypeStruct %[[Long]]
; CHECK-SPIRV-DAG: %[[StructPtr:.*]] = OpTypePointer Generic %[[Struct]]
; CHECK-SPIRV-DAG: %[[Function:.*]] = OpTypeFunction %[[Void]] %[[StructPtr]]
; CHECK-SPIRV-DAG: %[[Const:.*]] = OpConstantNull %[[Struct]]
; CHECK-SPIRV-DAG: %[[CrossStructPtr:.*]] = OpTypePointer CrossWorkgroup %[[Struct]]
; CHECK-SPIRV-DAG: %[[Var:.*]] = OpVariable %[[CrossStructPtr]] CrossWorkgroup %[[Const]]
; CHECK-SPIRV: %[[Foo:.*]] = OpFunction %[[Void]] None %[[Function]]
; CHECK-SPIRV-NEXT: OpFunctionParameter %[[StructPtr]]
; CHECK-SPIRV: %[[Casted:.*]] = OpPtrCastToGeneric %[[StructPtr]] %[[Var]]
; CHECK-SPIRV-NEXT: OpFunctionCall %[[Void]] %[[Foo]] %[[Casted]]

%struct.global_ctor_dtor = type { i32 }
@g1 = addrspace(1) global %struct.global_ctor_dtor zeroinitializer

define linkonce_odr spir_func void @foo(ptr addrspace(4) %this) {
entry:
  ret void
}

define internal spir_func void @bar() {
entry:
  call spir_func void @foo(ptr addrspace(4) addrspacecast (ptr addrspace(1) @g1 to ptr addrspace(4)))
  ret void
}
