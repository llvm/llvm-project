; The goal of the test case is to ensure that correct types are applied to PHI's as arguments of other PHI's.
; Pass criterion is that spirv-val considers output valid.

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#FooVal1:]] "val1"
; CHECK-DAG: OpName %[[#FooVal2:]] "val2"
; CHECK-DAG: OpName %[[#FooVal3:]] "val3"
; CHECK-DAG: OpName %[[#Bar:]] "bar"
; CHECK-DAG: OpName %[[#BarVal1:]] "val1"
; CHECK-DAG: OpName %[[#BarVal2:]] "val2"
; CHECK-DAG: OpName %[[#BarVal3:]] "val3"

; CHECK-DAG: %[[#Short:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#ShortGenPtr:]] = OpTypePointer Generic %[[#Short]]
; CHECK-DAG: %[[#ShortWrkPtr:]] = OpTypePointer Workgroup %[[#Short]]
; CHECK-DAG: %[[#G1:]] = OpVariable %[[#ShortWrkPtr]] Workgroup

; CHECK: %[[#Foo:]] = OpFunction %[[#]] None %[[#]]
; CHECK: %[[#FooArgP:]] = OpFunctionParameter %[[#ShortGenPtr]]
; CHECK: OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#FooG1:]] = OpPtrCastToGeneric %[[#ShortGenPtr]] %[[#G1]]
; CHECK: %[[#FooVal2]] = OpPhi %[[#ShortGenPtr]] %[[#FooArgP]] %[[#]] %[[#FooVal3]] %[[#]]
; CHECK: %[[#FooVal1]] = OpPhi %[[#ShortGenPtr]] %[[#FooG1]] %[[#]] %[[#FooVal2]] %[[#]]
; CHECK: %[[#FooVal3]] = OpLoad %[[#ShortGenPtr]] %[[#]]

; CHECK: %[[#Bar:]] = OpFunction %[[#]] None %[[#]]
; CHECK: %[[#BarArgP:]] = OpFunctionParameter %[[#ShortGenPtr]]
; CHECK: OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#BarVal3]] = OpLoad %[[#ShortGenPtr]] %[[#]]
; CHECK: %[[#BarG1:]] = OpPtrCastToGeneric %[[#ShortGenPtr]] %[[#G1]]
; CHECK: %[[#BarVal1]] = OpPhi %[[#ShortGenPtr]] %[[#BarG1]] %[[#]] %[[#BarVal2]] %[[#]]
; CHECK: %[[#BarVal2]] = OpPhi %[[#ShortGenPtr]] %[[#BarArgP]] %[[#]] %[[#BarVal3]] %[[#]]

@G1 = internal addrspace(3) global i16 undef, align 8
@G2 = internal unnamed_addr addrspace(3) global ptr addrspace(4) undef, align 8

define spir_kernel void @foo(ptr addrspace(4) %p, i1 %f1, i1 %f2, i1 %f3) {
entry:
  br label %l1

l1:
  br i1 %f1, label %l2, label %exit

l2:
  %val2 = phi ptr addrspace(4) [ %p, %l1 ], [ %val3, %l3 ]
  %val1 = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @G1 to ptr addrspace(4)), %l1 ], [ %val2, %l3 ]
  br i1 %f2, label %l3, label %exit

l3:
  %val3 = load ptr addrspace(4), ptr addrspace(3) @G2, align 8
  br i1 %f3, label %l2, label %exit

exit:
  ret void
}

define spir_kernel void @bar(ptr addrspace(4) %p, i1 %f1, i1 %f2, i1 %f3) {
entry:
  %val3 = load ptr addrspace(4), ptr addrspace(3) @G2, align 8
  br label %l1

l3:
  br i1 %f3, label %l2, label %exit

l1:
  br i1 %f1, label %l2, label %exit

l2:
  %val1 = phi ptr addrspace(4) [ addrspacecast (ptr addrspace(3) @G1 to ptr addrspace(4)), %l1 ], [ %val2, %l3 ]
  %val2 = phi ptr addrspace(4) [ %p, %l1 ], [ %val3, %l3 ]
  br i1 %f2, label %l3, label %exit

exit:
  ret void
}
