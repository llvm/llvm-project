; RUN: opt -passes=newgvn -S < %s | FileCheck %s

define i32 @f1(i32 %x) {
  ; CHECK-LABEL: define i32 @f1(
bb0:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f2(i32 %x) {
  ; CHECK-LABEL: define i32 @f2(
bb0:
  %cmp = icmp ne i32 %x, 0
  br i1 %cmp, label %bb1, label %bb2
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f3(i32 %x) {
  ; CHECK-LABEL: define i32 @f3(
bb0:
  switch i32 %x, label %bb1 [ i32 0, label %bb2]
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

declare void @g(i1)
define void @f4(i8 * %x)  {
; CHECK-LABEL: define void @f4(
bb0:
  %y = icmp eq i8* null, %x
  br i1 %y, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %zed = icmp eq i8* null, %x
  call void @g(i1 %zed)
; CHECK: call void @g(i1 %y)
  ret void
}

define double @fcmp_oeq_not_zero(double %x, double %y) {
entry:
  %cmp = fcmp oeq double %y, 2.0
  br i1 %cmp, label %if, label %return

if:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval = phi double [ %div, %if ], [ %x, %entry ]
  ret double %retval

; CHECK-LABEL: define double @fcmp_oeq_not_zero(
; CHECK: %div = fdiv double %x, 2.0
}

define double @fcmp_une_not_zero(double %x, double %y) {
entry:
  %cmp = fcmp une double %y, 2.0
  br i1 %cmp, label %return, label %else

else:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval = phi double [ %div, %else ], [ %x, %entry ]
  ret double %retval

; CHECK-LABEL: define double @fcmp_une_not_zero(
; CHECK: %div = fdiv double %x, 2.0
}

; PR22376 - We can't propagate zero constants because -0.0 
; compares equal to 0.0. If %y is -0.0 in this test case,
; we would produce the wrong sign on the infinity return value.
define double @fcmp_oeq_zero(double %x, double %y) {
entry:
  %cmp = fcmp oeq double %y, 0.0
  br i1 %cmp, label %if, label %return

if:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval = phi double [ %div, %if ], [ %x, %entry ]
  ret double %retval

; CHECK-LABEL: define double @fcmp_oeq_zero(
; CHECK: %div = fdiv double %x, %y
}

define double @fcmp_une_zero(double %x, double %y) {
entry:
  %cmp = fcmp une double %y, -0.0
  br i1 %cmp, label %return, label %else

else:
  %div = fdiv double %x, %y
  br label %return

return:
  %retval = phi double [ %div, %else ], [ %x, %entry ]
  ret double %retval

; CHECK-LABEL: define double @fcmp_une_zero(
; CHECK: %div = fdiv double %x, %y
}

; We also cannot propagate a value if it's not a constant.
; This is because the value could be 0.0 or -0.0.

define double @fcmp_oeq_maybe_zero(double %x, double %y, double %z1, double %z2) {
entry:
 %z = fadd double %z1, %z2
 %cmp = fcmp oeq double %y, %z
 br i1 %cmp, label %if, label %return

if:
 %div = fdiv double %x, %z
 br label %return

return:
 %retval = phi double [ %div, %if ], [ %x, %entry ]
 ret double %retval

; CHECK-LABEL: define double @fcmp_oeq_maybe_zero(
; CHECK: %div = fdiv double %x, %z
}

define double @fcmp_une_maybe_zero(double %x, double %y, double %z1, double %z2) {
entry:
 %z = fadd double %z1, %z2
 %cmp = fcmp une double %y, %z
 br i1 %cmp, label %return, label %else

else:
 %div = fdiv double %x, %z
 br label %return

return:
 %retval = phi double [ %div, %else ], [ %x, %entry ]
 ret double %retval

; CHECK-LABEL: define double @fcmp_une_maybe_zero(
; CHECK: %div = fdiv double %x, %z
}
