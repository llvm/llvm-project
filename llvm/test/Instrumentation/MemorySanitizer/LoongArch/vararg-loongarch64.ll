; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "loongarch64-unknown-linux-gnu"

;; First, check allocation of the save area.
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.va_start(ptr) #2
declare void @llvm.va_end(ptr) #2
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1
define i32 @foo(i32 %guard, ...) {
; CHECK-LABEL: @foo
; CHECK:    [[TMP1:%.*]] = load {{.*}} @__msan_va_arg_overflow_size_tls
; CHECK:    [[TMP2:%.*]] = add i64 0, [[TMP1]]
; CHECK:    [[TMP3:%.*]] = alloca {{.*}} [[TMP2]]
; CHECK:    call void @llvm.memset.p0.i64(ptr align 8 [[TMP3]], i8 0, i64 [[TMP2]], i1 false)
; CHECK:    [[TMP4:%.*]] = call i64 @llvm.umin.i64(i64 [[TMP2]], i64 800)
; CHECK:    call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[TMP3]], ptr align 8 @__msan_va_arg_tls, i64 [[TMP4]], i1 false)
;
  %vl = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr %vl)
  call void @llvm.va_start(ptr %vl)
  call void @llvm.va_end(ptr %vl)
  call void @llvm.lifetime.end.p0(i64 32, ptr %vl)
  ret i32 0
}

;; Save the incoming shadow value from the arguments in the __msan_va_arg_tls
;; array.
define i32 @bar() {
; CHECK-LABEL: @bar
; CHECK:    store i32 0, ptr @__msan_va_arg_tls, align 8
; CHECK:    store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 8) to ptr), align 8
; CHECK:    store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 16) to ptr), align 8
; CHECK:    store {{.*}} 24, {{.*}} @__msan_va_arg_overflow_size_tls
;
  %1 = call i32 (i32, ...) @foo(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

;; Check multiple fixed arguments.
declare i32 @foo2(i32 %g1, i32 %g2, ...)
define i32 @bar2() {
; CHECK-LABEL: @bar2
; CHECK:    store i64 0, ptr @__msan_va_arg_tls, align 8
; CHECK:    store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 8) to ptr), align 8
; CHECK:    store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls
;
  %1 = call i32 (i32, i32, ...) @foo2(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

;; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
;; passed to a variadic function.
declare i64 @sum(i64 %n, ...)
define dso_local i64 @many_args() {
;; If the size of __msan_va_arg_tls changes the second argument of `add` must also be changed.
; CHECK-LABEL: @many_args
; CHECK:    i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 792)
; CHECK-NOT: i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 800)
;
entry:
  %ret = call i64 (i64, ...) @sum(i64 120,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1,
  i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1
  )
  ret i64 %ret
}
