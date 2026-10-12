; RUN: opt -passes='function(loop(loop-interchange),print<loops>)' \
; RUN:     -loop-interchange-profitabilities=ignore -disable-output < %s 2>&1 | \
; RUN:     FileCheck %s
;
; A parent loop 'outer' contains an interchangeable perfect pair (pair.j/pair.k)
; as its first subloop, followed by two sibling loops (sib1, sib2). After
; interchanging the pair, LoopInfo must keep it in its original slot -- ahead of
; the siblings -- so that the preserved LoopAnalysis matches a rebuilt one and
; later loop passes see a consistent sibling traversal order.
;
;   for (i = 0; i < 64; i++) {
;     for (j = 0; j < 64; j++)      // block %pair.j \  interchangeable pair
;       for (k = 0; k < 64; k++)    // block %pair.k /  (first subloop of the i-loop)
;         A[k][j] = 0;
;     for (s1 = 0; s1 < 4; s1++);   // sibling
;     for (s2 = 0; s2 < 4; s2++);   // sibling
;   }

; The interchanged pair (new outer header %pair.k, new inner header %pair.j)
; stays first, ahead of sib1 and sib2.
; CHECK-LABEL: Loop info for function 'f':
; CHECK:         Loop at depth 1 containing: %outer.header<header>
; CHECK-NEXT:      Loop at depth 2 containing: %pair.k<header>
; CHECK-NEXT:        Loop at depth 3 containing: %pair.j<header>
; CHECK-NEXT:      Loop at depth 2 containing: %sib1.header<header>
; CHECK-NEXT:      Loop at depth 2 containing: %sib2.header<header>

define void @f(ptr noalias %A) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %pair.j

pair.j:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %pair.j.latch ]
  br label %pair.k

pair.k:
  %k = phi i64 [ 0, %pair.j ], [ %k.next, %pair.k ]
  %idx = getelementptr [64 x i8], ptr %A, i64 %k, i64 %j
  store i8 0, ptr %idx, align 1
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 64
  br i1 %k.done, label %pair.j.latch, label %pair.k

pair.j.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 64
  br i1 %j.done, label %sib1.header, label %pair.j

sib1.header:
  %s1 = phi i64 [ 0, %pair.j.latch ], [ %s1.next, %sib1.header ]
  %s1.next = add i64 %s1, 1
  %s1.done = icmp eq i64 %s1.next, 4
  br i1 %s1.done, label %sib2.header, label %sib1.header

sib2.header:
  %s2 = phi i64 [ 0, %sib1.header ], [ %s2.next, %sib2.header ]
  %s2.next = add i64 %s2, 1
  %s2.done = icmp eq i64 %s2.next, 4
  br i1 %s2.done, label %outer.latch, label %sib2.header

outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 64
  br i1 %i.done, label %exit, label %outer.header

exit:
  ret void
}
