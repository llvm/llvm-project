; RUN: opt < %s -S -passes="msan<track-origins=1>" 2>&1 | FileCheck %s --implicit-check-not "call void @llvm.mem" --implicit-check-not " load" --implicit-check-not " store"
; RUN: opt < %s -S -msan -msan-track-origins=1 | FileCheck %s --implicit-check-not "call void @llvm.mem" --implicit-check-not " load" --implicit-check-not " store"

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @FnByVal(i128* byval(i128) %p);
declare void @Fn(i128* %p);

define i128 @ByValArgument(i32, i128* byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValArgument(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} @__msan_param_tls to i64), i64 8) to i8*), i64 16, i1 false)
; CHECK:         %x = load i128, i128* %p
; CHECK:         load i128
; CHECK:         load i32
; CHECK:         store i128 {{.*}}, i128* bitcast ([100 x i64]* @__msan_retval_tls to i128*)
; CHECK-NEXT:    store i32 {{.*}}, i32* @__msan_retval_origin_tls
; CHECK-NEXT:    ret i128
;
entry:
  %x = load i128, i128* %p
  ret i128 %x
}

define i128 @ByValArgumentNoSanitize(i32, i128* byval(i128) %p) {
; CHECK-LABEL: @ByValArgumentNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 16, i1 false)
; CHECK:         %x = load i128, i128* %p
; CHECK:         store i128 0, i128* bitcast ([100 x i64]* @__msan_retval_tls to i128*)
; CHECK-NEXT:    store i32 0, i32* @__msan_retval_origin_tls
; CHECK-NEXT:    ret i128
;
entry:
  %x = load i128, i128* %p
  ret i128 %x
}

; FIXME: Origin of byval pointee is not propagated.
define void @ByValForward(i32, i128* byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValForward(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} @__msan_param_tls to i64), i64 8) to i8*), i64 16, i1 false)
; CHECK:         store i64 0, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @__msan_param_tls, i32 0, i32 0)
; CHECK-NEXT:    call void @Fn(
; CHECK-NEXT:    ret void
;
entry:
  call void @Fn(i128* %p)
  ret void
}

define void @ByValForwardNoSanitize(i32, i128* byval(i128) %p) {
; CHECK-LABEL: @ByValForwardNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 16, i1 false)
; CHECK:         store i64 0, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @__msan_param_tls, i32 0, i32 0)
; CHECK-NEXT:    call void @Fn(
; CHECK-NEXT:    ret void
;
entry:
  call void @Fn(i128* %p)
  ret void
}

; FIXME: Origin of %p byval pointee is not propagated.
define void @ByValForwardByVal(i32, i128* byval(i128) %p) sanitize_memory {
; CHECK-LABEL: @ByValForwardByVal(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64({{.*}} @__msan_param_tls to i64), i64 8) to i8*), i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* bitcast ([100 x i64]* @__msan_param_tls to i8*), i8* {{.*}}, i64 16, i1 false)
; CHECK:         store i32 {{.*}}, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__msan_param_origin_tls, i32 0, i32 0)
; CHECK-NEXT:    call void @FnByVal(
; CHECK-NEXT:    ret void
;
entry:
  call void @FnByVal(i128* byval(i128) %p)
  ret void
}

; FIXME: Shadow for byval should be reset not copied before the call.
define void @ByValForwardByValNoSanitize(i32, i128* byval(i128) %p) {
; CHECK-LABEL: @ByValForwardByValNoSanitize(
; CHECK-NEXT:  entry:
; CHECK:         call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 16, i1 false)
; CHECK:         call void @llvm.memcpy.p0i8.p0i8.i64(i8* bitcast ([100 x i64]* @__msan_param_tls to i8*), i8* {{.*}}, i64 16, i1 false) 
; CHECK:         store i32 0, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__msan_param_origin_tls, i32 0, i32 0)
; CHECK-NEXT:    call void @FnByVal(
; CHECK-NEXT:    ret void
;
entry:
  call void @FnByVal(i128* byval(i128) %p)
  ret void
}

