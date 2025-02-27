; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpName %[[FooArg:.*]] "unknown_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Foo:.*]] "foo"
; CHECK-SPIRV-DAG: OpName %[[BarArg:.*]] "known_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Bar:.*]] "bar"
; CHECK-SPIRV-DAG: OpName %[[UntypedArg:.*]] "arg"
; CHECK-SPIRV-DAG: OpName %[[FunUntypedArg:.*]] "foo_untyped_arg"
; CHECK-SPIRV-DAG: OpName %[[UnusedArg1:.*]] "unused_arg1"
; CHECK-SPIRV-DAG: OpName %[[Foo2Arg:.*]] "unknown_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Foo2:.*]] "foo2"
; CHECK-SPIRV-DAG: OpName %[[Bar2Arg:.*]] "known_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Bar2:.*]] "bar2"
; CHECK-SPIRV-DAG: OpName %[[Foo5Arg1:.*]] "unknown_type_ptr1"
; CHECK-SPIRV-DAG: OpName %[[Foo5Arg2:.*]] "unknown_type_ptr2"
; CHECK-SPIRV-DAG: OpName %[[Foo5:.*]] "foo5"
; CHECK-SPIRV-DAG: OpName %[[Bar5Arg:.*]] "known_type_ptr"
; CHECK-SPIRV-DAG: OpName %[[Bar5:.*]] "bar5"
; CHECK-SPIRV-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[Long:.*]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[Half:.*]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[Void:.*]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[HalfConst:.*]] = OpConstant %[[Half]] 15360
; CHECK-SPIRV-DAG: %[[CharPtr:.*]] = OpTypePointer CrossWorkgroup %[[Char]]
; CHECK-SPIRV-DAG: %[[LongPtr:.*]] = OpTypePointer CrossWorkgroup %[[Long]]
; CHECK-SPIRV-DAG: %[[Fun:.*]] = OpTypeFunction %[[Void]] %[[LongPtr]]
; CHECK-SPIRV-DAG: %[[Fun2:.*]] = OpTypeFunction %[[Void]] %[[Half]] %[[LongPtr]]
; CHECK-SPIRV-DAG: %[[Fun5:.*]] = OpTypeFunction %[[Void]] %[[Half]] %[[LongPtr]] %[[Half]] %[[LongPtr]] %[[Half]]
; CHECK-SPIRV-DAG: %[[FunUntyped:.*]] = OpTypeFunction %[[Void]] %[[CharPtr]]

; CHECK-SPIRV: %[[Foo]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[FooArg]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: %[[Bar]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[BarArg]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionCall %[[Void]] %[[Foo]] %[[BarArg]]

; CHECK-SPIRV: %[[FunUntypedArg]] = OpFunction %[[Void]] None %[[FunUntyped]]
; CHECK-SPIRV: %[[UntypedArg]] = OpFunctionParameter %[[CharPtr]]

; CHECK-SPIRV: %[[Foo2]] = OpFunction %[[Void]] None %[[Fun2]]
; CHECK-SPIRV: %[[UnusedArg1]] = OpFunctionParameter %[[Half]]
; CHECK-SPIRV: %[[Foo2Arg]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: %[[Bar2]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[Bar2Arg]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionCall %[[Void]] %[[Foo2]] %[[HalfConst]] %[[Bar2Arg]]

; CHECK-SPIRV: %[[Foo5]] = OpFunction %[[Void]] None %[[Fun5]]
; CHECK-SPIRV: OpFunctionParameter %[[Half]]
; CHECK-SPIRV: %[[Foo5Arg1]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionParameter %[[Half]]
; CHECK-SPIRV: %[[Foo5Arg2]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionParameter %[[Half]]
; CHECK-SPIRV: %[[Bar5]] = OpFunction %[[Void]] None %[[Fun]]
; CHECK-SPIRV: %[[Bar5Arg]] = OpFunctionParameter %[[LongPtr]]
; CHECK-SPIRV: OpFunctionCall %[[Void]] %[[Foo5]] %[[HalfConst]] %[[Bar5Arg]] %[[HalfConst]] %[[Bar5Arg]] %[[HalfConst]]

define void @foo(ptr addrspace(1) %unknown_type_ptr) {
entry:
  ret void
}

define spir_kernel void @bar(ptr addrspace(1) %known_type_ptr) {
entry:
  %elem = getelementptr inbounds i32, ptr addrspace(1) %known_type_ptr, i64 0
  call void @foo(ptr addrspace(1) %known_type_ptr)
  ret void
}

define void @foo_untyped_arg(ptr addrspace(1) %arg) {
entry:
  ret void
}

define void @foo2(half %unused_arg1, ptr addrspace(1) %unknown_type_ptr) {
entry:
  ret void
}

define spir_kernel void @bar2(ptr addrspace(1) %known_type_ptr) {
entry:
  %elem = getelementptr inbounds i32, ptr addrspace(1) %known_type_ptr, i64 0
  call void @foo2(half 1.0, ptr addrspace(1) %known_type_ptr)
  ret void
}

define void @foo5(half %unused_arg1, ptr addrspace(1) %unknown_type_ptr1, half %unused_arg2, ptr addrspace(1) %unknown_type_ptr2, half %unused_arg3) {
entry:
  ret void
}

define spir_kernel void @bar5(ptr addrspace(1) %known_type_ptr) {
entry:
  %elem = getelementptr inbounds i32, ptr addrspace(1) %known_type_ptr, i64 0
  call void @foo5(half 1.0, ptr addrspace(1) %known_type_ptr, half 1.0, ptr addrspace(1) %known_type_ptr, half 1.0)
  ret void
}
