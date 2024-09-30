; RUN: opt < %s -S -passes=msan 2>&1 -msan-origin-base=0x40000000 -msan-and-mask=0x80000000 | FileCheck %s

target datalayout = "E-m:m-i8:8:32-i16:16:32-i32:32-n32-S64"
target triple = "mips--linux"

define i32 @foo(i32 %guard, ...) {
  %vl = alloca ptr, align 4
  call void @llvm.lifetime.start.p0(i64 32, ptr %vl)
  call void @llvm.va_start(ptr %vl)
  call void @llvm.va_end(ptr %vl)
  call void @llvm.lifetime.end.p0(i64 32, ptr %vl)
  ret i32 0
}

; First, check allocation of the save area.
; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} @__msan_va_arg_overflow_size_tls
; CHECK: [[B:%.*]] = add i64 0, [[A]]
; CHECK: [[C:%.*]] = alloca {{.*}} [[B]]

; CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[C]], i8 0, i64 [[B]], i1 false)

; CHECK: [[D:%.*]] = call i64 @llvm.umin.i64(i64 [[B]], i64 800)
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[C]], ptr align 8 @__msan_va_arg_tls, i64 [[D]], i1 false)


declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1
declare void @llvm.va_start(ptr) #2
declare void @llvm.va_end(ptr) #2
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

define i32 @bar() {
  %1 = call i32 (i32, ...) @foo(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

; Save the incoming shadow value from the arguments in the __msan_va_arg_tls
; array.  The first argument is stored at position 4, since it's right
; justified.
; CHECK-LABEL: @bar
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 4) to ptr), align 8
; CHECK: store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 8) to ptr), align 8
; CHECK: store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 16) to ptr), align 8
; CHECK: store {{.*}} 24, {{.*}} @__msan_va_arg_overflow_size_tls

; Check multiple fixed arguments.
declare i32 @foo2(i32 %g1, i32 %g2, ...)
define i32 @bar2() {
  %1 = call i32 (i32, i32, ...) @foo2(i32 0, i32 1, i64 2, double 3.000000e+00)
  ret i32 %1
}

; CHECK-LABEL: @bar2
; CHECK: store i64 0, ptr @__msan_va_arg_tls, align 8
; CHECK: store i64 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 8) to ptr), align 8
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
; passed to a variadic function.
define dso_local i32 @many_args() {
; CHECK-LABEL: @many_args
; CHECK: i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 796)
; CHECK-NOT: i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 800)
entry:
  %ret = call i32 (i32, ...) @sum(i32 120,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1,
  i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1
  )
  ret i32 %ret
}

declare i32 @sum(i32 %n, ...)

; CHECK: declare void @__msan_maybe_warning_1(i8 signext, i32 signext)
; CHECK: declare void @__msan_maybe_store_origin_1(i8 signext, ptr, i32 signext)
; CHECK: declare void @__msan_maybe_warning_2(i16 signext, i32 signext)
; CHECK: declare void @__msan_maybe_store_origin_2(i16 signext, ptr, i32 signext)
; CHECK: declare void @__msan_maybe_warning_4(i32 signext, i32 signext)
; CHECK: declare void @__msan_maybe_store_origin_4(i32 signext, ptr, i32 signext)
; CHECK: declare void @__msan_maybe_warning_8(i64 signext, i32 signext)
; CHECK: declare void @__msan_maybe_store_origin_8(i64 signext, ptr, i32 signext)
