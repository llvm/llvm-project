; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
;
; CHECK-SPIRV-DAG: %[[#i32:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#one:]] = OpConstant %[[#i32]] 1
; CHECK-SPIRV-DAG: %[[#two:]] = OpConstant %[[#i32]] 2
; CHECK-SPIRV-DAG: %[[#three:]] = OpConstant %[[#i32]] 3
; CHECK-SPIRV-DAG: %[[#i32x3:]] = OpTypeArray %[[#i32]] %[[#three]]
; CHECK-SPIRV-DAG: %[[#test_arr_init:]] = OpConstantComposite %[[#i32x3]] %[[#one]] %[[#two]] %[[#three]]
; CHECK-SPIRV-DAG: %[[#twelve:]] = OpConstant %[[#i32]] 12
; CHECK-SPIRV-DAG: %[[#const_i32x3_ptr:]] = OpTypePointer UniformConstant %[[#i32x3]]

; CHECK-SPIRV:     %[[#test_arr2:]] = OpVariable %[[#const_i32x3_ptr]] UniformConstant %[[#test_arr_init]]
; CHECK-SPIRV:     %[[#test_arr:]] = OpVariable %[[#const_i32x3_ptr]] UniformConstant %[[#test_arr_init]]

; CHECK-SPIRV-DAG: %[[#i32x3_ptr:]] = OpTypePointer Function %[[#i32x3]]

; CHECK-SPIRV:     %[[#arr:]] = OpVariable %[[#i32x3_ptr]] Function
; CHECK-SPIRV:     %[[#arr2:]] = OpVariable %[[#i32x3_ptr]] Function
; CHECK-SPIRV:     OpCopyMemorySized %[[#arr]] %[[#test_arr]] %[[#twelve]] Aligned 4
; CHECK-SPIRV:     OpCopyMemorySized %[[#arr2]] %[[#test_arr2]] %[[#twelve]] Aligned 4


@__const.test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4
@__const.test.arr2 = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

define spir_func void @test() {
entry:
  %arr = alloca [3 x i32], align 4
  %arr2 = alloca [3 x i32], align 4
  %0 = bitcast [3 x i32]* %arr to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %0, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr to i8 addrspace(2)*), i32 12, i1 false)
  %1 = bitcast [3 x i32]* %arr2 to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %1, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr2 to i8 addrspace(2)*), i32 12, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p2i8.i32(i8* nocapture writeonly, i8 addrspace(2)* nocapture readonly, i32, i1)
