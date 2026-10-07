; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

@scalar = addrspace(3) global i64 0, align 8
@array = addrspace(3) global [4 x i64] zeroinitializer, align 8

define i32 @scalar_constexpr() {
; CHECK-LABEL: define i32 @scalar_constexpr(
; CHECK: [[GEP:%.*]] = getelementptr [1 x i32], ptr addrspace(3) @scalar, i32 0, i32 1
; CHECK-NEXT: [[LOAD:%.*]] = load i32, ptr addrspace(3) [[GEP]], align 4
; CHECK-NEXT: ret i32 [[LOAD]]
  %value = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @scalar, i32 4), align 4
  ret i32 %value
}

define i64 @scalar_instruction() {
; CHECK-LABEL: define i64 @scalar_instruction(
; CHECK: [[GEP:%.*]] = getelementptr [1 x i64], ptr addrspace(3) @scalar, i32 0, i32 0
; CHECK-NEXT: [[LOAD:%.*]] = load i64, ptr addrspace(3) [[GEP]], align 8
; CHECK-NEXT: ret i64 [[LOAD]]
  %address = getelementptr i8, ptr addrspace(3) @scalar, i32 0
  %value = load i64, ptr addrspace(3) %address, align 8
  ret i64 %value
}

define i64 @array_constexpr() {
; CHECK-LABEL: define i64 @array_constexpr(
; CHECK: [[GEP:%.*]] = getelementptr inbounds nuw [4 x i64], ptr addrspace(3) @array, i32 0, i32 2
; CHECK-NEXT: [[LOAD:%.*]] = load i64, ptr addrspace(3) [[GEP]], align 8
; CHECK-NEXT: ret i64 [[LOAD]]
  %value = load i64, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @array, i64 16), align 8
  ret i64 %value
}

define i64 @array_instruction() {
; CHECK-LABEL: define i64 @array_instruction(
; CHECK: [[GEP:%.*]] = getelementptr inbounds nuw [4 x i64], ptr addrspace(3) @array, i32 0, i32 2
; CHECK-NEXT: [[LOAD:%.*]] = load i64, ptr addrspace(3) [[GEP]], align 8
; CHECK-NEXT: ret i64 [[LOAD]]
  %address = getelementptr inbounds nuw i8, ptr addrspace(3) @array, i64 16
  %value = load i64, ptr addrspace(3) %address, align 8
  ret i64 %value
}

define i16 @array_i16_instruction() {
; CHECK-LABEL: define i16 @array_i16_instruction(
; CHECK: [[SLOT:%.*]] = alloca [4 x i16], align 2
; CHECK: [[GEP:%.*]] = getelementptr nuw [4 x i16], ptr [[SLOT]], i32 0, i32 2
; CHECK-NEXT: [[LOAD:%.*]] = load i16, ptr [[GEP]], align 2
; CHECK-NEXT: ret i16 [[LOAD]]
  %slot = alloca [4 x i16], align 2
  %address = getelementptr nuw i8, ptr %slot, i32 4
  %value = load i16, ptr %address, align 2
  ret i16 %value
}

define i16 @replacement_alloca() {
; CHECK-LABEL: define i16 @replacement_alloca(
; CHECK: [[SLOT:%.*]] = alloca i16, align 2
; CHECK: [[GEP:%.*]] = getelementptr [1 x i16], ptr [[SLOT]], i32 0, i32 0
; CHECK-NEXT: [[LOAD:%.*]] = load i16, ptr [[GEP]], align 2
; CHECK: ret i16
  %slot = alloca i8, align 2
  %address = getelementptr i8, ptr %slot, i32 0
  %value = load i8, ptr %address, align 2
  %result = zext i8 %value to i16
  ret i16 %result
}
