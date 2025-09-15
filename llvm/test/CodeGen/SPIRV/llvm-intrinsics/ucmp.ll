; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[TypeI8:.*]] = OpTypeInt 8 0 
; CHECK-DAG: %[[TypeI16:.*]] = OpTypeInt 16 0 
; CHECK-DAG: %[[TypeI32:.*]] = OpTypeInt 32 0 
; CHECK-DAG: %[[TypeI64:.*]] = OpTypeInt 64 0 

; CHECK-DAG: %[[CmpI64ConstMinusOne:.*]] = OpConstant %[[TypeI64]] 18446744073709551615

; CHECK-DAG: %[[CmpI8ConstOne:.*]] = OpConstant %[[TypeI8]] 1 
; CHECK-DAG: %[[CmpI8ConstZero:.*]] = OpConstantNull %[[TypeI8]]
; CHECK-DAG: %[[CmpI8ConstMinusOne:.*]] = OpConstant %[[TypeI8]] 255 

; CHECK-DAG: %[[CmpI16ConstOne:.*]] = OpConstant %[[TypeI16]] 1 
; CHECK-DAG: %[[CmpI16ConstZero:.*]] = OpConstantNull %[[TypeI16]]
; CHECK-DAG: %[[CmpI16ConstMinusOne:.*]] = OpConstant %[[TypeI16]] 65535 

; CHECK-DAG: %[[CmpI32ConstOne:.*]] = OpConstant %[[TypeI32]] 1 
; CHECK-DAG: %[[CmpI32ConstZero:.*]] = OpConstantNull %[[TypeI32]]
; CHECK-DAG: %[[CmpI32ConstMinusOne:.*]] = OpConstant %[[TypeI32]] 4294967295 

; CHECK-DAG: %[[CmpI64ConstOne:.*]] = OpConstant %[[TypeI64]] 1 
; CHECK-DAG: %[[CmpI64ConstZero:.*]] = OpConstantNull %[[TypeI64]]

; CHECK-DAG: %[[TypeBool:.*]] = OpTypeBool
; CHECK-DAG: %[[TypeVBool:.*]] = OpTypeVector %[[TypeBool]] 4

; CHECK-DAG: %[[TypeV4I8:.*]] = OpTypeVector %[[TypeI8]] 4 
; CHECK-DAG: %[[TypeV4I16:.*]] = OpTypeVector %[[TypeI16]] 4 
; CHECK-DAG: %[[TypeV4I32:.*]] = OpTypeVector %[[TypeI32]] 4 
; CHECK-DAG: %[[TypeV4I64:.*]] = OpTypeVector %[[TypeI64]] 4 

; CHECK-DAG: %[[V4I8ConstOne:.*]] = OpConstantComposite %[[TypeV4I8]] %[[CmpI8ConstOne]] %[[CmpI8ConstOne]] %[[CmpI8ConstOne]] %[[CmpI8ConstOne]]
; CHECK-DAG: %[[V4I8ConstZero:.*]] = OpConstantNull %[[TypeV4I8]]
; CHECK-DAG: %[[V4I8ConstMinusOne:.*]] = OpConstantComposite %[[TypeV4I8]] %[[CmpI8ConstMinusOne]] %[[CmpI8ConstMinusOne]] %[[CmpI8ConstMinusOne]] %[[CmpI8ConstMinusOne]]

; CHECK-DAG: %[[V4I16ConstOne:.*]] = OpConstantComposite %[[TypeV4I16]] %[[CmpI16ConstOne]] %[[CmpI16ConstOne]] %[[CmpI16ConstOne]] %[[CmpI16ConstOne]]
; CHECK-DAG: %[[V4I16ConstZero:.*]] = OpConstantNull %[[TypeV4I16]]
; CHECK-DAG: %[[V4I16ConstMinusOne:.*]] = OpConstantComposite %[[TypeV4I16]] %[[CmpI16ConstMinusOne]] %[[CmpI16ConstMinusOne]] %[[CmpI16ConstMinusOne]] %[[CmpI16ConstMinusOne]]

; CHECK-DAG: %[[V4I32ConstOne:.*]] = OpConstantComposite %[[TypeV4I32]] %[[CmpI32ConstOne]] %[[CmpI32ConstOne]] %[[CmpI32ConstOne]] %[[CmpI32ConstOne]]
; CHECK-DAG: %[[V4I32ConstZero:.*]] = OpConstantNull %[[TypeV4I32]]
; CHECK-DAG: %[[V4I32ConstMinusOne:.*]] = OpConstantComposite %[[TypeV4I32]] %[[CmpI32ConstMinusOne]] %[[CmpI32ConstMinusOne]] %[[CmpI32ConstMinusOne]] %[[CmpI32ConstMinusOne]]

; CHECK-DAG: %[[V4I64ConstOne:.*]] = OpConstantComposite %[[TypeV4I64]] %[[CmpI64ConstOne]] %[[CmpI64ConstOne]] %[[CmpI64ConstOne]] %[[CmpI64ConstOne]]
; CHECK-DAG: %[[V4I64ConstZero:.*]] = OpConstantNull %[[TypeV4I64]]
; CHECK-DAG: %[[V4I64ConstMinusOne:.*]] = OpConstantComposite %[[TypeV4I64]] %[[CmpI64ConstMinusOne]] %[[CmpI64ConstMinusOne]] %[[CmpI64ConstMinusOne]] %[[CmpI64ConstMinusOne]]

; CHECK: OpFunction
; CHECK: %[[CmpI8R1:.*]] = OpULessThanEqual %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[CmpI8R2:.*]] = OpULessThan %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[SelI8R1:.*]] = OpSelect %[[TypeI8]] %[[CmpI8R2]] %[[CmpI8ConstMinusOne]] %[[CmpI8ConstZero]]
; CHECK: %[[SelI8R2:.*]] = OpSelect %[[TypeI8]] %[[CmpI8R1]] %[[SelI8R1]] %[[CmpI8ConstOne]] 
; CHECK: OpReturnValue %[[SelI8R2]] 
define range(i8 -1, 2) i8 @test_i8(i8 noundef %0, i8 noundef %1) {
  %3 = tail call i8 @llvm.ucmp.i8.i8(i8 %0, i8 %1)
  ret i8 %3
}

; CHECK: OpFunction
; CHECK: %[[CmpI16R1:.*]] = OpULessThanEqual %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[CmpI16R2:.*]] = OpULessThan %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[SelI16R1:.*]] = OpSelect %[[TypeI16]] %[[CmpI16R2]] %[[CmpI16ConstMinusOne]] %[[CmpI16ConstZero]]
; CHECK: %[[SelI16R2:.*]] = OpSelect %[[TypeI16]] %[[CmpI16R1]] %[[SelI16R1]] %[[CmpI16ConstOne]] 
; CHECK: OpReturnValue %[[SelI16R2]] 
define range(i16 -1, 2) i16 @test_i16(i16 noundef %0, i16 noundef %1) {
  %3 = tail call i16 @llvm.ucmp.i16.i16(i16 %0, i16 %1)
  ret i16 %3
}

; CHECK: OpFunction
; CHECK: %[[CmpI32R1:.*]] = OpULessThanEqual %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[CmpI32R2:.*]] = OpULessThan %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[SelI32R1:.*]] = OpSelect %[[TypeI32]] %[[CmpI32R2]] %[[CmpI32ConstMinusOne]] %[[CmpI32ConstZero]]
; CHECK: %[[SelI32R2:.*]] = OpSelect %[[TypeI32]] %[[CmpI32R1]] %[[SelI32R1]] %[[CmpI32ConstOne]] 
; CHECK: OpReturnValue %[[SelI32R2]] 
define range(i32 -1, 2) i32 @test_i32(i32 noundef %0, i32 noundef %1) {
  %3 = tail call i32 @llvm.ucmp.i32.i32(i32 %0, i32 %1)
  ret i32 %3
}

; CHECK: OpFunction
; CHECK: %[[CmpI64R1:.*]] = OpULessThanEqual %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[CmpI64R2:.*]] = OpULessThan %[[TypeBool]] %[[#]] %[[#]]
; CHECK: %[[SelI64R1:.*]] = OpSelect %[[TypeI64]] %[[CmpI64R2]] %[[CmpI64ConstMinusOne]] %[[CmpI64ConstZero]]
; CHECK: %[[SelI64R2:.*]] = OpSelect %[[TypeI64]] %[[CmpI64R1]] %[[SelI64R1]] %[[CmpI64ConstOne]] 
; CHECK: OpReturnValue %[[SelI64R2]] 
define range(i64 -1, 2) i64 @test_i64(i64 noundef %0, i64 noundef %1) {
  %3 = tail call i64 @llvm.ucmp.i64.i64(i64 %0, i64 %1)
  ret i64 %3
}

; CHECK: OpFunction
; CHECK: %[[V4I8R1:.*]] = OpULessThanEqual %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[V4I8R2:.*]] = OpULessThan %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[SelectV4I8R1:.*]] = OpSelect %[[TypeV4I8]] %[[V4I8R2]] %[[V4I8ConstMinusOne]] %[[V4I8ConstZero]]
; CHECK: %[[SelectV4I8R2:.*]] = OpSelect %[[TypeV4I8]] %[[V4I8R1]] %[[SelectV4I8R1]] %[[V4I8ConstOne]] 
; CHECK: OpReturnValue %[[SelectV4I8R2]] 
define range(i8 -1, 2) <4 x i8> @test_v4i8(<4 x i8> noundef %0, <4 x i8> noundef %1) {
  %3 = tail call <4 x i8> @llvm.ucmp.v4i8.v4i8(<4 x i8> %0, <4 x i8> %1)
  ret <4 x i8> %3
}

; CHECK: OpFunction
; CHECK: %[[V4I16R1:.*]] = OpULessThanEqual %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[V4I16R2:.*]] = OpULessThan %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[SelectV4I16R1:.*]] = OpSelect %[[TypeV4I16]] %[[V4I16R2]] %[[V4I16ConstMinusOne]] %[[V4I16ConstZero]]
; CHECK: %[[SelectV4I16R2:.*]] = OpSelect %[[TypeV4I16]] %[[V4I16R1]] %[[SelectV4I16R1]] %[[V4I16ConstOne]] 
; CHECK: OpReturnValue %[[SelectV4I16R2]] 
define range(i16 -1, 2) <4 x i16> @test_v4i16(<4 x i16> noundef %0, <4 x i16> noundef %1) {
  %3 = tail call <4 x i16> @llvm.ucmp.v4i16.v4i16(<4 x i16> %0, <4 x i16> %1)
  ret <4 x i16> %3
}

; CHECK: OpFunction
; CHECK: %[[V4I32R1:.*]] = OpULessThanEqual %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[V4I32R2:.*]] = OpULessThan %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[SelectV4I32R1:.*]] = OpSelect %[[TypeV4I32]] %[[V4I32R2]] %[[V4I32ConstMinusOne]] %[[V4I32ConstZero]]
; CHECK: %[[SelectV4I32R2:.*]] = OpSelect %[[TypeV4I32]] %[[V4I32R1]] %[[SelectV4I32R1]] %[[V4I32ConstOne]] 
; CHECK: OpReturnValue %[[SelectV4I32R2]] 
define range(i32 -1, 2) <4 x i32> @test_v4i32(<4 x i32> noundef %0, <4 x i32> noundef %1) {
  %3 = tail call <4 x i32> @llvm.ucmp.v4i32.v4i32(<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %3
}

; CHECK: OpFunction
; CHECK: %[[V4I64R1:.*]] = OpULessThanEqual %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[V4I64R2:.*]] = OpULessThan %[[TypeVBool]] %[[#]] %[[#]]
; CHECK: %[[SelectV4I64R1:.*]] = OpSelect %[[TypeV4I64]] %[[V4I64R2]] %[[V4I64ConstMinusOne]] %[[V4I64ConstZero]]
; CHECK: %[[SelectV4I64R2:.*]] = OpSelect %[[TypeV4I64]] %[[V4I64R1]] %[[SelectV4I64R1]] %[[V4I64ConstOne]] 
; CHECK: OpReturnValue %[[SelectV4I64R2]] 
define range(i64 -1, 2) <4 x i64> @test_v4i64(<4 x i64> noundef %0, <4 x i64> noundef %1) {
  %3 = tail call <4 x i64> @llvm.ucmp.v4i64.v4i64(<4 x i64> %0, <4 x i64> %1)
  ret <4 x i64> %3
}

declare i8 @llvm.ucmp.i8.i8(i8, i8)
declare i16 @llvm.ucmp.i16.i16(i16, i16)
declare i32 @llvm.ucmp.i32.i32(i32, i32)
declare i64 @llvm.ucmp.i64.i64(i64, i64)
declare <4 x i8> @llvm.ucmp.v4i8.v4i8(<4 x i8>, <4 x i8>)
declare <4 x i16> @llvm.ucmp.v4i16.v4i16(<4 x i16>, <4 x i16>)
declare <4 x i32> @llvm.ucmp.v4i32.v4i32(<4 x i32>, <4 x i32>)
declare <4 x i64> @llvm.ucmp.v4i64.v4i64(<4 x i64>, <4 x i64>)
