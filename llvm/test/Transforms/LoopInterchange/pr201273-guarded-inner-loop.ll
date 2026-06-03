; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -pass-remarks-missed='loop-interchange' -disable-output \
; RUN:     -pass-remarks-output=%t
; RUN: FileCheck -input-file=%t %s

; The middle loop header (%for.j) guards the inner loop %for.k: the inner loop
; only runs when %j != 0, and its exit condition (%k.next == %j) is only
; well-defined under that guard. Hoisting %for.k out of the guard would make it
; spin when %j == 0, so the pass must not interchange this nest.

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            NotTightlyNested
; CHECK-NEXT: Function:        main

@x = global [3 x [3 x [3 x i32]]] zeroinitializer
@w = global [3 x [3 x [3 x i32]]] zeroinitializer
@y = global [3 x [3 x [3 x i32]]] zeroinitializer

define i32 @main() {
entry:
  br label %for.i

for.i:
  %j = phi i32 [ %j.next, %for.i.inc ], [ 0, %entry ]
  %j.is.zero = icmp eq i32 %j, 0
  %xbase = getelementptr [9 x i32], ptr @x, i32 %j
  %wbase = getelementptr [9 x i32], ptr @w, i32 %j
  %ybase = getelementptr [3 x i32], ptr @y, i32 %j
  br label %for.j

for.j:
  %i = phi i32 [ %i.next, %for.j.inc ], [ 0, %for.i ]
  br i1 %j.is.zero, label %for.j.inc, label %for.k.ph

for.k.ph:
  %xp = getelementptr i32, ptr %xbase, i32 %i
  %wp = getelementptr i32, ptr %wbase, i32 %i
  %yp = getelementptr [9 x i32], ptr %ybase, i32 %i
  br label %for.k

for.k:
  %k = phi i32 [ 0, %for.k.ph ], [ %k.next, %for.k ]
  %xk = getelementptr [3 x i32], ptr %xp, i32 %k
  %xv = load i32, ptr %xk, align 4
  %wk = getelementptr [3 x i32], ptr %wp, i32 %k
  %wv = load i32, ptr %wk, align 4
  %add = add i32 %xv, %wv
  %yk = getelementptr i32, ptr %yp, i32 %k
  store i32 %add, ptr %yk, align 4
  %k.next = add i32 %k, 1
  %k.done = icmp eq i32 %k.next, %j
  br i1 %k.done, label %for.j.inc, label %for.k

for.j.inc:
  %i.next = add i32 %i, 1
  %i.done = icmp eq i32 %i, 0
  br i1 %i.done, label %for.j, label %for.i.inc

for.i.inc:
  %j.next = add i32 %j, 1
  %j.done = icmp eq i32 %j.next, 3
  br i1 %j.done, label %exit, label %for.i

exit:
  ret i32 0
}
