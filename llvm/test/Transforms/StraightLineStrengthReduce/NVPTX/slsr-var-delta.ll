; RUN: opt < %s -passes=slsr -S | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_75 | FileCheck %s --check-prefix=PTX

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; Test SLSR can reuse the computation by complex variable delta.
; The original program needs 4 mul.wide.s32, after SLSR with 
; variable-delta, it can reduce to 1 mul.wide.s32.
; PTX-COUNT-1: mul.wide.s32
; PTX-NOT: mul.wide.s32
define void @foo(ptr %a, ptr %b, i32 %j) {
  %i.0 = load i32, ptr %a, align 8
  %i = add i32 %i.0, %j
  ; CHECK: [[L:%.*]] = load i32, ptr %a, align 8
  ; CHECK: [[I:%.*]] = add i32 [[L]], %j
  %gep.24 = getelementptr float, ptr %b, i32 %i
  ; CHECK: [[GEP0:%.*]] = getelementptr float, ptr %b, i32 [[I]]
  ; CHECK: store i32 0, ptr [[GEP0]]
  store i32 0, ptr %gep.24
  %gep.24.sum1 = add i32 %i, %i
  %gep.25 = getelementptr float, ptr %b, i32 %gep.24.sum1
  ; CHECK: [[EXT1:%.*]] = sext i32 [[I]] to i64
  ; CHECK: [[MUL1:%.*]] = shl i64 [[EXT1]], 2
  ; CHECK: [[GEP1:%.*]] = getelementptr i8, ptr [[GEP0]], i64 [[MUL1]]
  ; CHECK: store i32 1, ptr [[GEP1]]
  store i32 1, ptr %gep.25
  %gep.26.sum3 = add i32 1, %i
  %gep.27.sum = add i32 %gep.26.sum3, %i
  %gep.28 = getelementptr float, ptr %b, i32 %gep.27.sum
  ; CHECK: [[GEP2:%.*]] = getelementptr i8, ptr [[GEP1]], i64 4
  ; CHECK: store i32 2, ptr [[GEP2]]
  store i32 2, ptr %gep.28
  %gep.28.sum = add i32 %gep.27.sum, %i
  %gep.29 = getelementptr float, ptr %b, i32 %gep.28.sum
  ; CHECK: [[EXT2:%.*]] = sext i32 [[I]] to i64
  ; CHECK: [[MUL2:%.*]] = shl i64 [[EXT2]], 2
  ; CHECK: [[GEP3:%.*]] = getelementptr i8, ptr [[GEP2]], i64 [[MUL2]]
  ; CHECK: store i32 3, ptr [[GEP3]]
  store i32 3, ptr %gep.29
  %gep.29.sum = add i32 %gep.28.sum, %i
  %gep.30 = getelementptr float, ptr %b, i32 %gep.29.sum
  ; CHECK: [[EXT3:%.*]] = sext i32 [[I]] to i64
  ; CHECK: [[MUL3:%.*]] = shl i64 [[EXT3]], 2
  ; CHECK: [[GEP4:%.*]] = getelementptr i8, ptr [[GEP3]], i64 [[MUL3]]
  ; CHECK: store i32 4, ptr [[GEP4]]
  store i32 4, ptr %gep.30
  ret void
}
