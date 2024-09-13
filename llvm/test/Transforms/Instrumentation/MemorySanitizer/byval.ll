; RUN: opt < %s -S -passes="msan<track-origins=1>" 2>&1 | FileCheck %s --implicit-check-not "call void @llvm.mem" --implicit-check-not " load" --implicit-check-not " store"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @FnByVal(ptr byval(i128) %p);
declare void @Fn(ptr %p);

define i128 @ByValArgument(i32, ptr byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValArgument(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[#]], ptr align 8 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         [[X:%.*]] = load i128, ptr %p, align 8
; CHECK:         [[_MSLD:%.*]] = load i128, ptr %[[#]], align 8
; CHECK:         %[[#]] = load i32, ptr %[[#]], align 8
; CHECK:         store i128 [[_MSLD]], ptr @__msan_retval_tls, align 8
; CHECK:         store i32 %[[#]], ptr @__msan_retval_origin_tls, align 4
; CHECK:         ret i128 [[X]]
;
entry:
  %x = load i128, ptr %p
  ret i128 %x
}

define i128 @ByValArgumentNoSanitize(i32, ptr byval(i128) %p) {
; CHECK-LABEL: @ByValArgumentNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 8 %[[#]], i8 0, i64 16, i1 false)
; CHECK:         [[X:%.*]] = load i128, ptr %p, align 8
; CHECK:         store i128 0, ptr @__msan_retval_tls, align 8
; CHECK:         store i32 0, ptr @__msan_retval_origin_tls, align 4
; CHECK:         ret i128 [[X]]
;
entry:
  %x = load i128, ptr %p
  ret i128 %x
}

define void @ByValForward(i32, ptr byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValForward(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[#]], ptr align 8 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         store i64 0, ptr @__msan_param_tls, align 8
; CHECK:         call void @Fn(ptr %p)
; CHECK:         ret void
;
entry:
  call void @Fn(ptr %p)
  ret void
}

define void @ByValForwardNoSanitize(i32, ptr byval(i128) %p) {
; CHECK-LABEL: @ByValForwardNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 8 %[[#]], i8 0, i64 16, i1 false)
; CHECK:         store i64 0, ptr @__msan_param_tls, align 8
; CHECK:         call void @Fn(ptr %p)
; CHECK:         ret void
;
entry:
  call void @Fn(ptr %p)
  ret void
}

define void @ByValForwardByVal(i32, ptr byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValForwardByVal(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[#]], ptr align 8 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr @__msan_param_tls, ptr %[[#]], i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 @__msan_param_origin_tls, ptr align 4 %[[#]], i64 16, i1 false)
; CHECK:         call void @FnByVal(ptr byval(i128) %p)
; CHECK:         ret void
;
entry:
  call void @FnByVal(ptr byval(i128) %p)
  ret void
}

define void @ByValForwardByValNoSanitize(i32, ptr byval(i128) %p) {
; CHECK-LABEL: @ByValForwardByValNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 8 %[[#]], i8 0, i64 16, i1 false)
; CHECK:         call void @llvm.memset.p0.i64(ptr @__msan_param_tls, i8 0, i64 16, i1 false)
; CHECK:         call void @FnByVal(ptr byval(i128) %p)
; CHECK:         ret void
;
entry:
  call void @FnByVal(ptr byval(i128) %p)
  ret void
}

declare void @FnByVal8(ptr byval(i8) %p);
declare void @Fn8(ptr %p);

define i8 @ByValArgument8(i32, ptr byval(i8) %p) sanitize_memory {
; CHECK-LABEL: @ByValArgument8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[#]], ptr align 1 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 1, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 4, i1 false)
; CHECK:         [[X:%.*]] = load i8, ptr %p, align 1
; CHECK:         [[_MSLD:%.*]] = load i8, ptr %[[#]], align 1
; CHECK:         %[[#]] = load i32, ptr %[[#]], align 4
; CHECK:         store i8 [[_MSLD]], ptr @__msan_retval_tls, align 8
; CHECK:         store i32 %[[#]], ptr @__msan_retval_origin_tls, align 4
; CHECK:         ret i8 [[X]]
;
entry:
  %x = load i8, ptr %p
  ret i8 %x
}

define i8 @ByValArgumentNoSanitize8(i32, ptr byval(i8) %p) {
; CHECK-LABEL: @ByValArgumentNoSanitize8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 1 %[[#]], i8 0, i64 1, i1 false)
; CHECK:         [[X:%.*]] = load i8, ptr %p, align 1
; CHECK:         store i8 0, ptr @__msan_retval_tls, align 8
; CHECK:         store i32 0, ptr @__msan_retval_origin_tls, align 4
; CHECK:         ret i8 [[X]]
;
entry:
  %x = load i8, ptr %p
  ret i8 %x
}

define void @ByValForward8(i32, ptr byval(i8) %p) sanitize_memory {
; CHECK-LABEL: @ByValForward8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[#]], ptr align 1 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 1, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 4, i1 false)
; CHECK:         store i64 0, ptr @__msan_param_tls, align 8
; CHECK:         call void @Fn8(ptr %p)
; CHECK:         ret void
;
entry:
  call void @Fn8(ptr %p)
  ret void
}

define void @ByValForwardNoSanitize8(i32, ptr byval(i8) %p) {
; CHECK-LABEL: @ByValForwardNoSanitize8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 1 %[[#]], i8 0, i64 1, i1 false)
; CHECK:         store i64 0, ptr @__msan_param_tls, align 8
; CHECK:         call void @Fn8(ptr %p)
; CHECK:         ret void
;
entry:
  call void @Fn8(ptr %p)
  ret void
}

define void @ByValForwardByVal8(i32, ptr byval(i8) %p) sanitize_memory {
; CHECK-LABEL: @ByValForwardByVal8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[#]], ptr align 1 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), i64 1, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[#]], ptr align 4 inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_origin_tls to i64), i64 8) to ptr), i64 4, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr @__msan_param_tls, ptr %[[#]], i64 1, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr align 4 @__msan_param_origin_tls, ptr align 4 %[[#]], i64 4, i1 false)
; CHECK:         call void @FnByVal8(ptr byval(i8) %p)
; CHECK:         ret void
;
entry:
  call void @FnByVal8(ptr byval(i8) %p)
  ret void
}

define void @ByValForwardByValNoSanitize8(i32, ptr byval(i8) %p) {
; CHECK-LABEL: @ByValForwardByValNoSanitize8(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0.i64(ptr align 1 %[[#]], i8 0, i64 1, i1 false)
; CHECK:         call void @llvm.memset.p0.i64(ptr @__msan_param_tls, i8 0, i64 1, i1 false)
; CHECK:         call void @FnByVal8(ptr byval(i8) %p)
; CHECK:         ret void
;
entry:
  call void @FnByVal8(ptr byval(i8) %p)
  ret void
}

