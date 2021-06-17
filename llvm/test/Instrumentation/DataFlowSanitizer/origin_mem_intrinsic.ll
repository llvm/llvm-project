; RUN: opt < %s -dfsan -dfsan-track-origins=1  -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

define void @memcpy(i8* %d, i8* %s, i32 %l) {
  ; CHECK: @memcpy.dfsan
  ; CHECK: [[L64:%.*]] = zext i32 %l to i64
  ; CHECK: call void @__dfsan_mem_origin_transfer(i8* %d, i8* %s, i64 [[L64]])
  ; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align [[#SBYTES]] {{.*}}, i8* align [[#SBYTES]] {{.*}}, i32 {{.*}}, i1 false)
  ; CHECK: call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d, i8* %s, i32 %l, i1 false)

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %d, i8* %s, i32 %l, i1 0)
  ret void
}

define void @memmove(i8* %d, i8* %s, i32 %l) {
  ; CHECK: @memmove.dfsan
  ; CHECK: [[L64:%.*]] = zext i32 %l to i64
  ; CHECK: call void @__dfsan_mem_origin_transfer(i8* %d, i8* %s, i64 [[L64]])
  ; CHECK: call void @llvm.memmove.p0i8.p0i8.i32(i8* align [[#SBYTES]] {{.*}}, i8* align [[#SBYTES]] {{.*}}, i32 {{.*}}, i1 false)
  ; CHECK: call void @llvm.memmove.p0i8.p0i8.i32(i8* %d, i8* %s, i32 %l, i1 false)

  call void @llvm.memmove.p0i8.p0i8.i32(i8* %d, i8* %s, i32 %l, i1 0)
  ret void
}

define void @memset(i8* %p, i8 %v) {
  ; CHECK: @memset.dfsan
  ; CHECK: [[O:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[S:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; CHECK: call void @__dfsan_set_label(i[[#SBITS]] [[S]], i32 [[O]], i8* %p, i64 1)
  call void @llvm.memset.p0i8.i64(i8* %p, i8 %v, i64 1, i1 1)
  ret void
}