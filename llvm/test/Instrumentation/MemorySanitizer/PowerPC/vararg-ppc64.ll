; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -S | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64--linux"

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
; array.  The first argument is stored at position 4, since it's right
; justified.
; CHECK-LABEL: @bar
; CHECK: store i32 0, i32* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 4) to i32*), align 8
; CHECK: store i64 0, i64* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 8) to i64*), align 8
; CHECK: store i64 0, i64* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 16) to i64*), align 8
; CHECK: store {{.*}} 24, {{.*}} @__msan_va_arg_overflow_size_tls

; Check vector argument.
define i32 @bar2() {
  %1 = call i32 (i32, ...) @foo(i32 0, <2 x i64> <i64 1, i64 2>)
  ret i32 %1
}

; The vector is at offset 16 of parameter save area, but __msan_va_arg_tls
; corresponds to offset 8+ of parameter save area - so the offset from
; __msan_va_arg_tls is actually misaligned.
; CHECK-LABEL: @bar2
; CHECK: store <2 x i64> zeroinitializer, <2 x i64>* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 8) to <2 x i64>*), align 8
; CHECK: store {{.*}} 24, {{.*}} @__msan_va_arg_overflow_size_tls

; Check i64 array.
define i32 @bar4() {
  %1 = call i32 (i32, ...) @foo(i32 0, [2 x i64] [i64 1, i64 2])
  ret i32 %1
}

; CHECK-LABEL: @bar4
; CHECK: store [2 x i64] zeroinitializer, [2 x i64]* bitcast ([100 x i64]* @__msan_va_arg_tls to [2 x i64]*), align 8
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Check i128 array.
define i32 @bar5() {
  %1 = call i32 (i32, ...) @foo(i32 0, [2 x i128] [i128 1, i128 2])
  ret i32 %1
}

; CHECK-LABEL: @bar5
; CHECK: store [2 x i128] zeroinitializer, [2 x i128]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_va_arg_tls to i64), i64 8) to [2 x i128]*), align 8
; CHECK: store {{.*}} 40, {{.*}} @__msan_va_arg_overflow_size_tls

; Check 8-aligned byval.
define i32 @bar6([2 x i64]* %arg) {
  %1 = call i32 (i32, ...) @foo(i32 0, [2 x i64]* byval align 8 %arg)
  ret i32 %1
}

; CHECK-LABEL: @bar6
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 bitcast ([100 x i64]* @__msan_va_arg_tls to i8*), i8* align 8 {{.*}}, i64 16, i1 false)
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Check 16-aligned byval.
define i32 @bar7([4 x i64]* %arg) {
  %1 = call i32 (i32, ...) @foo(i32 0, [4 x i64]* byval align 16 %arg)
  ret i32 %1
}

; CHECK-LABEL: @bar7
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__msan_param_tls to i64), i64 8) to i8*), i8* align 8 {{.*}}, i64 32, i1 false)
; CHECK: store {{.*}} 40, {{.*}} @__msan_va_arg_overflow_size_tls


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
