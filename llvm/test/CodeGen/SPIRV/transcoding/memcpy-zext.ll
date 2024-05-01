; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-32
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:    %[[#i64:]] = OpTypeInt 64 0
; CHECK-DAG:    %[[#i8:]] = OpTypeInt 8 0
; CHECK-DAG:    %[[#i32:]] = OpTypeInt 32 0
; CHECK-DAG:    %[[#one:]] = OpConstant %[[#i32]] 1
; CHECK-DAG:    %[[#two:]] = OpConstant %[[#i32]] 2
; CHECK-DAG:    %[[#three:]] = OpConstant %[[#i32]] 3
; CHECK-DAG:    %[[#i32x3:]] = OpTypeArray %[[#i32]] %[[#three]]
; CHECK-DAG:    %[[#test_arr_init:]] = OpConstantComposite %[[#i32x3]] %[[#one]] %[[#two]] %[[#three]]
; CHECK-DAG:    %[[#szconst1024:]] = OpConstant %[[#i32]] 1024
; CHECK-DAG:    %[[#szconst42:]] = OpConstant %[[#i8]] 42
; CHECK-DAG:    %[[#szconst123:]] = OpConstant %[[#i64]] 123
; CHECK-DAG:    %[[#const_i32x3_ptr:]] = OpTypePointer UniformConstant %[[#i32x3]]
; CHECK-DAG:    %[[#test_arr:]] = OpVariable %[[#const_i32x3_ptr]] UniformConstant %[[#test_arr_init]]
; CHECK-DAG:    %[[#i32x3_ptr:]] = OpTypePointer Function %[[#i32x3]]
; CHECK:        %[[#arr:]] = OpVariable %[[#i32x3_ptr]] Function

; CHECK-32:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconst1024]]
; CHECK-32:     %[[#szconstext42:]] = OpUConvert %[[#i32:]] %[[#szconst42:]]
; CHECK-32:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconstext42]]
; CHECK-32:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconst123]]

; If/when Backend stoped rewrite actual reg size of i8/i16/i32/i64 with i32,
; i32 = G_TRUNC i64 would appear for the 32-bit target, switching the following
; TODO patterns instead of the last line above.
; TODO:         %[[#szconstext123:]] = OpUConvert %[[#i32:]] %[[#szconst123:]]
; TODO:         OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconst123]]

; CHECK-64:     %[[#szconstext1024:]] = OpUConvert %[[#i64:]] %[[#szconst1024:]]
; CHECK-64:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconstext1024]]
; CHECK-64:     %[[#szconstext42:]] = OpUConvert %[[#i64:]] %[[#szconst42:]]
; CHECK-64:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconstext42]]
; CHECK-64:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#szconst123]]

@__const.test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3]

define spir_func void @test() {
entry:
  %arr = alloca [3 x i32], align 4
  %dest = bitcast ptr %arr to ptr
  call void @llvm.memcpy.p0.p2.i32(ptr align 4 %dest, ptr addrspace(2) align 4 @__const.test.arr, i32 1024, i1 false)
  call void @llvm.memcpy.p0.p2.i8(ptr align 4 %dest, ptr addrspace(2) align 4 @__const.test.arr, i8 42, i1 false)
  call void @llvm.memcpy.p0.p2.i64(ptr align 4 %dest, ptr addrspace(2) align 4 @__const.test.arr, i64 123, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p2.i32(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i32, i1)
declare void @llvm.memcpy.p0.p2.i8(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i8, i1)
declare void @llvm.memcpy.p0.p2.i64(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1)
