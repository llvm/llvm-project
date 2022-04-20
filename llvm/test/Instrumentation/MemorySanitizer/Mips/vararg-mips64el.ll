; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:m-i8:8:32-i16:16:32-i64:64-n32:64-S128"
target triple = "mips64el--linux"

define i32 @foo(i32 %guard, ...) {
  %vl = alloca i8*, align 8
  %1 = bitcast i8** %vl to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %1)
  call void @llvm.va_start(i8* %1)
  call void @llvm.va_end(i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %1)
  ret i32 0
}

; First, check allocation of the save area.

; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} @__msan_va_arg_overflow_size_tls
; CHECK: [[B:%.*]] = add i64 0, [[A]]
; CHECK: [[C:%.*]] = alloca {{.*}} [[B]]

; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[C]], i8* align 8 bitcast ({{.*}} @__msan_va_arg_tls to i8*), i64 [[B]], i1 false)

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.va_start(i8*) #2
declare void @llvm.va_end(i8*) #2
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

define i32 @bar() {
  %1 = call i32 (i32, ...) @foo(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

; Save the incoming shadow value from the arguments in the __msan_va_arg_tls
; array.
; CHECK-LABEL: @bar
; CHECK: store i32 0, i32* bitcast ([100 x i64]* @__msan_va_arg_tls to i32*), align 8
; CHECK: store i64 0, i64* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 8) to i64*), align 8
; CHECK: store i64 0, i64* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 16) to i64*), align 8
; CHECK: store {{.*}} 24, {{.*}} @__msan_va_arg_overflow_size_tls

; Check multiple fixed arguments.
declare i32 @foo2(i32 %g1, i32 %g2, ...)
define i32 @bar2() {
  %1 = call i32 (i32, i32, ...) @foo2(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

; CHECK-LABEL: @bar2
; CHECK: store i64 0, i64* getelementptr inbounds ([100 x i64], [100 x i64]* @__msan_va_arg_tls, i32 0, i32 0), align 8
; CHECK: store i64 0, i64* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 8) to i64*), align 8
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
; passed to a variadic function.
define dso_local i64 @many_args() {
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

; If the size of __msan_va_arg_tls changes the second argument of `add` must also be changed.
; CHECK-LABEL: @many_args
; CHECK: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 792)
; CHECK-NOT: i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 800)
declare i64 @sum(i64 %n, ...)
