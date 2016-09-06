; RUN: opt -S -guard-widening < %s        | FileCheck %s
; RUN: opt -S -passes=guard-widening < %s | FileCheck %s

declare void @llvm.experimental.guard(i1,...)

; Basic test case: we wide the first check to check both the
; conditions.
define void @f_0(i1 %cond_0, i1 %cond_1) {
; CHECK-LABEL: @f_0(
entry:
; CHECK:  %wide.chk = and i1 %cond_0, %cond_1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  ret void

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void
}

; Same as @f_0, but with using a more general notion of postdominance.
define void @f_1(i1 %cond_0, i1 %cond_1) {
; CHECK-LABEL: @f_1(
entry:
; CHECK:  %wide.chk = and i1 %cond_0, %cond_1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
; CHECK: merge:
; CHECK-NOT: call void (i1, ...) @llvm.experimental.guard(
; CHECK: ret void
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void
}

; Like @f_1, but we have some code we need to hoist before we can
; widen a dominanting check.
define void @f_2(i32 %a, i32 %b) {
; CHECK-LABEL: @f_2(
entry:
; CHECK:  %cond_0 = icmp ult i32 %a, 10
; CHECK:  %cond_1 = icmp ult i32 %b, 10
; CHECK:  %wide.chk = and i1 %cond_0, %cond_1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_0 = icmp ult i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  %cond_1 = icmp ult i32 %b, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void
}

; Negative test: don't hoist stuff out of control flow
; indiscriminately, since that can make us do more work than needed.
define void @f_3(i32 %a, i32 %b) {
; CHECK-LABEL: @f_3(
entry:
; CHECK:  %cond_0 = icmp ult i32 %a, 10
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_0 = icmp ult i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
; CHECK: left:
; CHECK:   %cond_1 = icmp ult i32 %b, 10
; CHECK:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
; CHECK:   ret void

  %cond_1 = icmp ult i32 %b, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void

right:
  ret void
}

; But hoisting out of control flow is fine if it makes a loop computed
; condition loop invariant.  This behavior may require some tuning in
; the future.
define void @f_4(i32 %a, i32 %b) {
; CHECK-LABEL: @f_4(
entry:
; CHECK:  %cond_0 = icmp ult i32 %a, 10
; CHECK:  %cond_1 = icmp ult i32 %b, 10
; CHECK:  %wide.chk = and i1 %cond_0, %cond_1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %loop, label %leave

  %cond_0 = icmp ult i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %loop, label %leave

loop:
  %cond_1 = icmp ult i32 %b, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  br i1 undef, label %loop, label %leave

leave:
  ret void
}

; Hoisting out of control flow is also fine if we can widen the
; dominating check without doing any extra work.
define void @f_5(i32 %a) {
; CHECK-LABEL: @f_5(
entry:
; CHECK:  %wide.chk = icmp uge i32 %a, 11
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_0 = icmp ugt i32 %a, 7
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
  %cond_1 = icmp ugt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void

right:
  ret void
}

; Negative test: the load from %a can be safely speculated to before
; the first guard, but there is no guarantee that it will produce the
; same value.
define void @f_6(i1* dereferenceable(32) %a, i1* %b, i1 %unknown) {
; CHECK-LABEL: @f_6(
; CHECK: call void (i1, ...) @llvm.experimental.guard(
; CHECK: call void (i1, ...) @llvm.experimental.guard(
; CHECK: ret void
entry:
  %cond_0 = load i1, i1* %a
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  store i1 %unknown, i1* %b
  %cond_1 = load i1, i1* %a
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void
}

; All else equal, we try to widen the earliest guard we can.  This
; heuristic can use some tuning.
define void @f_7(i32 %a, i1* %cond_buf) {
; CHECK-LABEL: @f_7(
entry:
; CHECK:  %cond_1 = load volatile i1, i1* %cond_buf
; CHECK:  %cond_3 = icmp ult i32 %a, 7
; CHECK:  %wide.chk = and i1 %cond_1, %cond_3
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  %cond_2 = load volatile i1, i1* %cond_buf
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_2) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_1 = load volatile i1, i1* %cond_buf
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  %cond_2 = load volatile i1, i1* %cond_buf
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_2) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
  %cond_3 = icmp ult i32 %a, 7
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_3) [ "deopt"() ]
  br label %left

right:
  ret void
}

; In this case the earliest dominating guard is in a loop, and we
; don't want to put extra work in there.  This heuristic can use some
; tuning.
define void @f_8(i32 %a, i1 %cond_1, i1 %cond_2) {
; CHECK-LABEL: @f_8(
entry:
  br label %loop

loop:
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  br i1 undef, label %loop, label %leave

leave:
; CHECK: leave:
; CHECK:  %cond_3 = icmp ult i32 %a, 7
; CHECK:  %wide.chk = and i1 %cond_2, %cond_3
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %loop2, label %leave2

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_2) [ "deopt"() ]
  br i1 undef, label %loop2, label %leave2

loop2:
  %cond_3 = icmp ult i32 %a, 7
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_3) [ "deopt"() ]
  br label %loop2

leave2:
  ret void
}

; In cases like these where there isn't any "obviously profitable"
; widening sites, we refuse to do anything.
define void @f_9(i32 %a, i1 %cond_0, i1 %cond_1) {
; CHECK-LABEL: @f_9(
entry:
  br label %first_loop

first_loop:
; CHECK: first_loop:
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
; CHECK:  br i1 undef, label %first_loop, label %second_loop

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %first_loop, label %second_loop

second_loop:
; CHECK: second_loop:
; CHECK:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
; CHECK:   br label %second_loop

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  br label %second_loop
}

; Same situation as in @f_9: no "obviously profitable" widening sites,
; so we refuse to do anything.
define void @f_10(i32 %a, i1 %cond_0, i1 %cond_1) {
; CHECK-LABEL: @f_10(
entry:
  br label %loop

loop:
; CHECK: loop:
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
; CHECK:  br i1 undef, label %loop, label %no_loop

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %loop, label %no_loop

no_loop:
; CHECK: no_loop:
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
; CHECK:  ret void
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void
}

; With guards in loops, we're okay hoisting out the guard into the
; containing loop.
define void @f_11(i32 %a, i1 %cond_0, i1 %cond_1) {
; CHECK-LABEL: @f_11(
entry:
  br label %inner

inner:
; CHECK: inner:
; CHECK:  %wide.chk = and i1 %cond_0, %cond_1
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %inner, label %outer

  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %inner, label %outer

outer:
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  br label %inner
}

; Checks that we are adequately guarded against exponential-time
; behavior when hoisting code.
define void @f_12(i32 %a0) {
; CHECK-LABEL: @f_12

; Eliding the earlier 29 multiplications for brevity
; CHECK:  %a30 = mul i32 %a29, %a29
; CHECK-NEXT:  %cond = trunc i32 %a30 to i1
; CHECK-NEXT:  %wide.chk = and i1 true, %cond
; CHECK-NEXT:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK-NEXT:  ret void

entry:
  call void(i1, ...) @llvm.experimental.guard(i1 true) [ "deopt"() ]
  %a1 = mul i32 %a0, %a0
  %a2 = mul i32 %a1, %a1
  %a3 = mul i32 %a2, %a2
  %a4 = mul i32 %a3, %a3
  %a5 = mul i32 %a4, %a4
  %a6 = mul i32 %a5, %a5
  %a7 = mul i32 %a6, %a6
  %a8 = mul i32 %a7, %a7
  %a9 = mul i32 %a8, %a8
  %a10 = mul i32 %a9, %a9
  %a11 = mul i32 %a10, %a10
  %a12 = mul i32 %a11, %a11
  %a13 = mul i32 %a12, %a12
  %a14 = mul i32 %a13, %a13
  %a15 = mul i32 %a14, %a14
  %a16 = mul i32 %a15, %a15
  %a17 = mul i32 %a16, %a16
  %a18 = mul i32 %a17, %a17
  %a19 = mul i32 %a18, %a18
  %a20 = mul i32 %a19, %a19
  %a21 = mul i32 %a20, %a20
  %a22 = mul i32 %a21, %a21
  %a23 = mul i32 %a22, %a22
  %a24 = mul i32 %a23, %a23
  %a25 = mul i32 %a24, %a24
  %a26 = mul i32 %a25, %a25
  %a27 = mul i32 %a26, %a26
  %a28 = mul i32 %a27, %a27
  %a29 = mul i32 %a28, %a28
  %a30 = mul i32 %a29, %a29
  %cond = trunc i32 %a30 to i1
  call void(i1, ...) @llvm.experimental.guard(i1 %cond) [ "deopt"() ]
  ret void
}

define void @f_13(i32 %a) {
; CHECK-LABEL: @f_13(
entry:
; CHECK:  %wide.chk = icmp ult i32 %a, 10
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %wide.chk) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_0 = icmp ult i32 %a, 14
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
  %cond_1 = icmp slt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void

right:
  ret void
}

define void @f_14(i32 %a) {
; CHECK-LABEL: @f_14(
entry:
; CHECK:  %cond_0 = icmp ult i32 %a, 14
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
; CHECK:  br i1 undef, label %left, label %right

  %cond_0 = icmp ult i32 %a, 14
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_0) [ "deopt"() ]
  br i1 undef, label %left, label %right

left:
; CHECK: left:
; CHECK:  %cond_1 = icmp sgt i32 %a, 10
; CHECK:  call void (i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]

  %cond_1 = icmp sgt i32 %a, 10
  call void(i1, ...) @llvm.experimental.guard(i1 %cond_1) [ "deopt"() ]
  ret void

right:
  ret void
}
