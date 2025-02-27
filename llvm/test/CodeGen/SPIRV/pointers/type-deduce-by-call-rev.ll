; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpName %[[FooArg:.*]] "known_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Foo:.*]] "foo"
; CHECK-SPIRV-DAG: OpName %[[ArgToDeduce:.*]] "unknown_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Bar:.*]] "bar"
; CHECK-SPIRV-DAG: %[[Long:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[Void:.*]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[LongPtr:.*]] = OpTypePointer CrossWorkgroup %[[Long]]
; CHECK-SPIRV-DAG: %[[Fun:.*]] = OpTypeFunction %[[Void]] %[[LongPtr]]
; CHECK-SPIRV: %[[Bar]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[ArgToDeduce]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionCall %[[Void]] %[[Foo]] %[[ArgToDeduce]]
; CHECK-SPIRV: %[[Foo]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[FooArg]] = OpFunctionParameter %[[LongPtr]]

define spir_kernel void @bar(ptr addrspace(1) %unknown_type_ptr) {
entry:
  call spir_func void @foo(ptr addrspace(1) %unknown_type_ptr)
  ret void
}

define void @foo(ptr addrspace(1) %known_type_ptr) {
entry:
  %elem = getelementptr inbounds i32, ptr addrspace(1) %known_type_ptr, i64 0
  ret void
}
