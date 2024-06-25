; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName %[[#Foo:]] "foo"
; CHECK-DAG: OpName %[[#Ptr1:]] "_arg1"
; CHECK-DAG: OpName %[[#Ptr2:]] "_arg2"
; CHECK-DAG: OpName %[[#Ptr3:]] "_arg3"
; CHECK-DAG: OpName %[[#Ptr4:]] "_arg4"
; CHECK-DAG: OpName %[[#Ptr5:]] "_arg5"
; CHECK-DAG: OpDecorate %[[#Ptr1]] NonReadable
; CHECK-DAG: OpDecorate %[[#Ptr2]] Alignment 128
; CHECK-DAG: OpDecorate %[[#Ptr2]] NonReadable
; CHECK-DAG: OpDecorate %[[#Ptr3]] Alignment 128
; CHECK-DAG: OpDecorate %[[#Ptr3]] NonReadable
; CHECK-DAG: OpDecorate %[[#Ptr4]] Alignment 128
; CHECK-DAG: OpDecorate %[[#Ptr4]] NonReadable
; CHECK-DAG: OpDecorate %[[#Ptr5]] UserSemantic "Unknown format"
; CHECK: %[[#Foo]] = OpFunction
; CHECK-NEXT: %[[#Ptr1]] = OpFunctionParameter
; CHECK-NEXT: %[[#Ptr2]] = OpFunctionParameter
; CHECK-NEXT: %[[#Ptr3]] = OpFunctionParameter
; CHECK-NEXT: %[[#Ptr4]] = OpFunctionParameter
; CHECK-NEXT: %[[#Ptr5]] = OpFunctionParameter
; CHECK: OpFunctionEnd

@.str.0 = private unnamed_addr addrspace(1) constant [16 x i8] c"../prefetch.hpp\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [5 x i8] c"{25}\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [13 x i8] c"{44:128}{25}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [15 x i8] c"{44:\22128\22}{25}\00", section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [13 x i8] c"{44,128}{25}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr addrspace(1) constant [15 x i8] c"Unknown format\00", section "llvm.metadata"

define spir_kernel void @foo(ptr addrspace(1) %_arg1, ptr addrspace(1) %_arg2, ptr addrspace(1) %_arg3, ptr addrspace(1) %_arg4, ptr addrspace(1) %_arg5) {
entry:
  %r1 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %_arg1, ptr addrspace(1) @.str.1, ptr addrspace(1) @.str.0, i32 80, ptr addrspace(1) null)
  %r2 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %_arg2, ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.0, i32 80, ptr addrspace(1) null)
  %r3 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %_arg3, ptr addrspace(1) @.str.3, ptr addrspace(1) @.str.0, i32 80, ptr addrspace(1) null)
  %r4 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %_arg4, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.0, i32 80, ptr addrspace(1) null)
  %r5 = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) %_arg5, ptr addrspace(1) @.str.5, ptr addrspace(1) @.str.0, i32 80, ptr addrspace(1) null)
  ret void
}
