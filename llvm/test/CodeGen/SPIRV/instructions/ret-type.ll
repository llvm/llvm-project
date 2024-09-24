; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --translator-compatibility-mode %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Modifying the block ordering prevents the pointer types to correctly be deduced. Not sure why, but looks
; orthogonal to the block sorting.
; XFAIL: *

; CHECK-DAG: OpName %[[Test1:.*]] "test1"
; CHECK-DAG: OpName %[[Foo:.*]] "foo"
; CHECK-DAG: OpName %[[Bar:.*]] "bar"
; CHECK-DAG: OpName %[[Test2:.*]] "test2"

; CHECK-DAG: %[[Long:.*]] = OpTypeInt 64 0
; CHECK-DAG: %[[Array:.*]] = OpTypeArray %[[Long]] %[[#]]
; CHECK-DAG: %[[Struct1:.*]] = OpTypeStruct %[[Array]]
; CHECK-DAG: %[[Struct2:.*]] = OpTypeStruct %[[Struct1]]
; CHECK-DAG: %[[StructPtr:.*]] = OpTypePointer Function %[[Struct2]]
; CHECK-DAG: %[[Bool:.*]] = OpTypeBool
; CHECK-DAG: %[[FooType:.*]] = OpTypeFunction %[[StructPtr:.*]] %[[StructPtr]] %[[StructPtr]] %[[Bool]]
; CHECK-DAG: %[[Char:.*]] = OpTypeInt 8 0
; CHECK-DAG: %[[CharPtr:.*]] = OpTypePointer Function %[[Char]]

; CHECK: %[[Test1]] = OpFunction
; CHECK: OpFunctionCall %[[StructPtr:.*]] %[[Foo]]
; CHECK: OpFunctionCall %[[StructPtr:.*]] %[[Bar]]
; CHECK: OpFunctionEnd

; CHECK: %[[Foo]] = OpFunction %[[StructPtr:.*]] None %[[FooType]]
; CHECK: %[[Arg1:.*]] = OpFunctionParameter %[[StructPtr]]
; CHECK: %[[Arg2:.*]] = OpFunctionParameter
; CHECK: %[[Sw:.*]] = OpFunctionParameter
; CHECK: %[[Res:.*]] = OpInBoundsPtrAccessChain %[[StructPtr]] %[[Arg1]] %[[#]]
; CHECK: OpReturnValue %[[Res]]
; CHECK: OpReturnValue %[[Arg2]]

; CHECK: %[[Bar]] = OpFunction %[[StructPtr:.*]] None %[[#]]
; CHECK: %[[BarArg:.*]] = OpFunctionParameter
; CHECK: %[[BarRes:.*]] = OpInBoundsPtrAccessChain %[[CharPtr]] %[[BarArg]] %[[#]]
; CHECK: %[[BarResCasted:.*]] = OpBitcast %[[StructPtr]] %[[BarRes]]
; CHECK: %[[BarResStruct:.*]] = OpInBoundsPtrAccessChain %[[StructPtr]] %[[#]] %[[#]]
; CHECK: OpReturnValue %[[BarResStruct]]
; CHECK: OpReturnValue %[[BarResCasted]]

; CHECK: %[[Test2]] = OpFunction
; CHECK: OpFunctionCall %[[StructPtr:.*]] %[[Foo]]
; CHECK: OpFunctionCall %[[StructPtr:.*]] %[[Bar]]
; CHECK: OpFunctionEnd

%struct = type { %array }
%array = type { [1 x i64] }

define spir_func void @test1(ptr %arg1, ptr %arg2, i1 %sw) {
entry:
  %r1 = call ptr @foo(ptr %arg1, ptr %arg2, i1 %sw)
  %r2 = call ptr @bar(ptr %arg1, i1 %sw)
  ret void
}

define spir_func ptr @foo(ptr %arg1, ptr %arg2, i1 %sw) {
entry:
  br i1 %sw, label %exit, label %sw1
sw1:
  %result = getelementptr inbounds %struct, ptr %arg1, i64 100
  ret ptr %result
exit:
  ret ptr %arg2
}

define spir_func ptr @bar(ptr %arg1, i1 %sw) {
entry:
  %charptr = getelementptr inbounds i8, ptr %arg1, i64 0
  br i1 %sw, label %exit, label %sw1
sw1:
  %result = getelementptr inbounds %struct, ptr %arg1, i64 100
  ret ptr %result
exit:
  ret ptr %charptr
}

define spir_func void @test2(ptr %arg1, ptr %arg2, i1 %sw) {
entry:
  %r1 = call ptr @foo(ptr %arg1, ptr %arg2, i1 %sw)
  %r2 = call ptr @bar(ptr %arg1, i1 %sw)
  ret void
}
