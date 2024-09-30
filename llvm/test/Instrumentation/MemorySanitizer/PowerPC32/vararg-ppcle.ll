; RUN: opt < %s -S -passes=msan 2>&1 -msan-origin-base=0x40000000 -msan-and-mask=0x80000000 | FileCheck %s

target datalayout = "E-m:e-i32:32-n32:32"
target triple = "powerpcle--linux"

%struct.__va_list_tag = type { i8, i8, i16, ptr, ptr }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo(i32 noundef %guard, ...) #0 {
entry:
  %guard.addr = alloca i32, align 4
  %va = alloca [1 x %struct.__va_list_tag], align 4
  store i32 %guard, ptr %guard.addr, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %va, i32 0, i32 0
  call void @llvm.va_start.p0(ptr %arraydecay)
  %arraydecay1 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %va, i32 0, i32 0
  call void @llvm.va_end.p0(ptr %arraydecay1)
  ret i32 0
}

; First, check allocation of the save area.
; CHECK-LABEL: @foo
; CHECK: [[A:%.*]] = load {{.*}} ptr @__msan_va_arg_overflow_size_tls, align 4
; CHECK: [[B:%.*]] = add i64 0, [[A]]
; CHECK: [[C:%.*]] = alloca i8, i64 [[B]], align 8
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

; Check vector argument.
define i32 @bar2() {
  %1 = call i32 (i32, ...) @foo(i32 0, <2 x i64> <i64 1, i64 2>)
  ret i32 %1
}

; The vector is at offset 16 of parameter save area, but __msan_va_arg_tls
; corresponds to offset 8+ of parameter save area - so the offset from
; __msan_va_arg_tls is actually misaligned.
; CHECK-LABEL: @bar2
; CHECK: store <2 x i64> zeroinitializer, ptr @__msan_va_arg_tls, align 8
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls


; Check i64 array.
define i32 @bar4() {
  %1 = call i32 (i32, ...) @foo(i32 0, [2 x i64] [i64 1, i64 2])
  ret i32 %1
}

; CHECK-LABEL: @bar4
; CHECK: store [2 x i64] zeroinitializer, ptr @__msan_va_arg_tls, align 8
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Check i128 array.
define i32 @bar5() {
  %1 = call i32 (i32, ...) @foo(i32 0, [2 x i128] [i128 1, i128 2])
  ret i32 %1
}

; CHECK-LABEL: @bar5
; CHECK: store [2 x i128] zeroinitializer, ptr @__msan_va_arg_tls, align 8
; CHECK: store {{.*}} 32, {{.*}} @__msan_va_arg_overflow_size_tls

; Check 8-aligned byval.
define i32 @bar6(ptr %arg) {
  %1 = call i32 (i32, ...) @foo(i32 0, ptr byval([2 x i64]) align 8 %arg)
  ret i32 %1
}

; CHECK-LABEL: @bar6
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 @__msan_va_arg_tls, ptr align 8 {{.*}}, i64 16, i1 false)
; CHECK: store {{.*}} 16, {{.*}} @__msan_va_arg_overflow_size_tls

; Check 16-aligned byval.
define i32 @bar7(ptr %arg) {
  %1 = call i32 (i32, ...) @foo(i32 0, ptr byval([4 x i64]) align 16 %arg)
  ret i32 %1
}

; CHECK-LABEL: @bar7
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 @__msan_va_arg_tls, ptr align 8 {{.*}}, i64 32, i1 false)
; CHECK: store {{.*}} 32, {{.*}} @__msan_va_arg_overflow_size_tls

; Test that MSan doesn't generate code overflowing __msan_va_arg_tls when too many arguments are
; passed to a variadic function.
define dso_local i64 @many_args() {
; CHECK-LABEL: @many_args
; CHECK: i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 792)
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

; If the size of __msan_va_arg_tls changes the second argument of `add` must also be changed.
declare i64 @sum(i64 %n, ...)
