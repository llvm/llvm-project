; RUN: opt < %s -nary-reassociate -S | FileCheck %s
; RUN: opt < %s -passes='nary-reassociate' -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

declare void @foo(i32)

; foo(a + c);
; foo((a + (b + c));
;   =>
; t = a + c;
; foo(t);
; foo(t + b);
define void @left_reassociate(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @left_reassociate(
  %1 = add i32 %a, %c
; CHECK: [[BASE:%[a-zA-Z0-9]+]] = add i32 %a, %c
  call void @foo(i32 %1)
  %2 = add i32 %b, %c
  %3 = add i32 %a, %2
; CHECK: [[RESULT:%[a-zA-Z0-9]+]] = add i32 [[BASE]], %b
  call void @foo(i32 %3)
; CHECK-NEXT: call void @foo(i32 [[RESULT]])
  ret void
}

; foo(a + c);
; foo((a + b) + c);
;   =>
; t = a + c;
; foo(t);
; foo(t + b);
define void @right_reassociate(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @right_reassociate(
  %1 = add i32 %a, %c
; CHECK: [[BASE:%[a-zA-Z0-9]+]] = add i32 %a, %c
  call void @foo(i32 %1)
  %2 = add i32 %a, %b
  %3 = add i32 %2, %c
; CHECK: [[RESULT:%[a-zA-Z0-9]+]] = add i32 [[BASE]], %b
  call void @foo(i32 %3)
; CHECK-NEXT: call void @foo(i32 [[RESULT]])
  ret void
}

; t1 = a + c;
; foo(t1);
; t2 = a + b;
; foo(t2);
; t3 = t2 + c;
; foo(t3);
;
; Do not rewrite t3 into t1 + b because t2 is used elsewhere and is likely free.
define void @no_reassociate(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @no_reassociate(
  %1 = add i32 %a, %c
; CHECK: add i32 %a, %c
  call void @foo(i32 %1)
  %2 = add i32 %a, %b
; CHECK: add i32 %a, %b
  call void @foo(i32 %2)
  %3 = add i32 %2, %c
; CHECK: add i32 %2, %c
  call void @foo(i32 %3)
  ret void
}

; if (p1)
;   foo(a + c);
; if (p2)
;   foo(a + c);
; if (p3)
;   foo((a + b) + c);
;
; No action because (a + c) does not dominate ((a + b) + c).
define void @conditional(i1 %p1, i1 %p2, i1 %p3, i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @conditional(
entry:
  br i1 %p1, label %then1, label %branch1

then1:
  %0 = add i32 %a, %c
; CHECK: add i32 %a, %c
  call void @foo(i32 %0)
  br label %branch1

branch1:
  br i1 %p2, label %then2, label %branch2

then2:
  %1 = add i32 %a, %c
; CHECK: add i32 %a, %c
  call void @foo(i32 %1)
  br label %branch2

branch2:
  br i1 %p3, label %then3, label %return

then3:
  %2 = add i32 %a, %b
; CHECK: %2 = add i32 %a, %b
  %3 = add i32 %2, %c
; CHECK: add i32 %2, %c
  call void @foo(i32 %3)
  br label %return

return:
  ret void
}

; This test involves more conditional reassociation candidates. It exercises
; the stack optimization in tryReassociatedAdd that pops the candidates that
; do not dominate the current instruction.
;
;       def1
;      cond1
;      /  \
;     /    \
;   cond2  use2
;   /  \
;  /    \
; def2  def3
;      cond3
;       /  \
;      /    \
;    def4   use1
;
; NaryReassociate should match use1 with def3, and use2 with def1.
define void @conditional2(i32 %a, i32 %b, i32 %c, i1 %cond1, i1 %cond2, i1 %cond3) {
entry:
  %def1 = add i32 %a, %b
  br i1 %cond1, label %bb1, label %bb6
bb1:
  br i1 %cond2, label %bb2, label %bb3
bb2:
  %def2 = add i32 %a, %b
  call void @foo(i32 %def2)
  ret void
bb3:
  %def3 = add i32 %a, %b
  br i1 %cond3, label %bb4, label %bb5
bb4:
  %def4 = add i32 %a, %b
  call void @foo(i32 %def4)
  ret void
bb5:
  %0 = add i32 %a, %c
  %1 = add i32 %0, %b
; CHECK: [[t1:%[0-9]+]] = add i32 %def3, %c
  call void @foo(i32 %1) ; foo((a + c) + b);
; CHECK-NEXT: call void @foo(i32 [[t1]])
  ret void
bb6:
  %2 = add i32 %a, %c
  %3 = add i32 %2, %b
; CHECK: [[t2:%[0-9]+]] = add i32 %def1, %c
  call void @foo(i32 %3) ; foo((a + c) + b);
; CHECK-NEXT: call void @foo(i32 [[t2]])
  ret void
}

; foo((a + b) + c)
; foo(((a + d) + b) + c)
;   =>
; t = (a + b) + c;
; foo(t);
; foo(t + d);
define void @quaternary(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK-LABEL: @quaternary(
  %1 = add i32 %a, %b
  %2 = add i32 %1, %c
  call void @foo(i32 %2)
; CHECK: call void @foo(i32 [[TMP1:%[a-zA-Z0-9]]])
  %3 = add i32 %a, %d
  %4 = add i32 %3, %b
  %5 = add i32 %4, %c
; CHECK: [[TMP2:%[a-zA-Z0-9]]] = add i32 [[TMP1]], %d
  call void @foo(i32 %5)
; CHECK: call void @foo(i32 [[TMP2]]
  ret void
}

define void @iterative(i32 %a, i32 %b, i32 %c) {
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  call void @foo(i32 %abc)

  %ab2 = add i32 %ab, %b
  %ab2c = add i32 %ab2, %c
; CHECK: %ab2c = add i32 %abc, %b
  call void @foo(i32 %ab2c)
; CHECK-NEXT: call void @foo(i32 %ab2c)

  %ab3 = add i32 %ab2, %b
  %ab3c = add i32 %ab3, %c
; CHECK-NEXT: %ab3c = add i32 %ab2c, %b
  call void @foo(i32 %ab3c)
; CHECK-NEXT: call void @foo(i32 %ab3c)

  ret void
}

define void @avoid_infinite_loop(i32 %a, i32 %b) {
; CHECK-LABEL: @avoid_infinite_loop
  %ab = add i32 %a, %b
; CHECK-NEXT: %ab
  %ab2 = add i32 %ab, %b
; CHECK-NEXT: %ab2
  call void @foo(i32 %ab2)
; CHECK-NEXT: @foo(i32 %ab2)
  ret void
}
