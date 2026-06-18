; RUN: opt < %s -passes=loop-interchange -loop-interchange-profitabilities=ignore -debug-only=loop-interchange -disable-output -S 2>%t
; RUN: FileCheck --input-file=%t %s

; This test documents a currently missed optimization opportunity.
;
; collectPerfectNests() walks up from each innermost loop and stops as soon as
; it reaches a loop with more than one subloop. But because [loop.i, loop.j] is never put in
; any LoopList, the pass never analyses or attempts that interchange — the
; opportunity is silently missed.
;
; Corresponding C code:
;
;   for (int i = 0; i < 8; ++i)
;     for (int j = 0; j < 8; ++j) {
;       A[i][j] = 0;                // missed: i/j interchange
;       for (int k = 0; k < 8; ++k) {
;         for (int l = 0; l < 8; ++l)
;           for (int m = 0; m < 8; ++m)
;             Left[m][l] = 0;       // interchanged: l/m swapped
;         for (int n = 0; n < 8; ++n)
;           for (int o = 0; o < 8; ++o)
;             Right[o][n] = 0;      // interchanged: n/o swapped
;       }
;     }
;
;
; CHECK:      Processing LoopList of size = 2 containing the following loops:
; CHECK-NEXT:   - Loop at depth 4 containing: %loop.l.header<header>,%loop.m.header,%loop.m.latch,%loop.l.latch<latch><exiting>
; CHECK-NEXT:     Loop at depth 5 containing: %loop.m.header<header>,%loop.m.latch<latch><exiting>
; CHECK-NEXT:   - Loop at depth 5 containing: %loop.m.header<header>,%loop.m.latch<latch><exiting>
; CHECK: Loops interchanged: outer loop 'loop.l.header' and inner loop 'loop.m.header'
;
; CHECK:      Processing LoopList of size = 2 containing the following loops:
; CHECK-NEXT:   - Loop at depth 4 containing: %loop.n.header<header>,%loop.o.header,%loop.o.latch,%loop.n.latch<latch><exiting>
; CHECK-NEXT:     Loop at depth 5 containing: %loop.o.header<header>,%loop.o.latch<latch><exiting>
; CHECK-NEXT:   - Loop at depth 5 containing: %loop.o.header<header>,%loop.o.latch<latch><exiting>
; CHECK: Loops interchanged: outer loop 'loop.n.header' and inner loop 'loop.o.header'
;
;
; CHECK-NOT: loop.i.header
; CHECK-NOT: loop.j.header

define void @missed_outer_prefix_subnest(ptr noalias %Left, ptr noalias %Right,
                                         ptr noalias %A) {
entry:
  br label %loop.i.header

loop.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.i.latch ]
  br label %loop.j.header

loop.j.header:
  %j = phi i64 [ 0, %loop.i.header ], [ %j.next, %loop.j.latch ]
  %a.row = mul nuw nsw i64 %i, 8
  %a.idx = add nuw nsw i64 %a.row, %j
  %a.ptr = getelementptr i8, ptr %A, i64 %a.idx
  store i8 0, ptr %a.ptr, align 1
  br label %loop.k.header

loop.k.header:
  %k = phi i64 [ 0, %loop.j.header ], [ %k.next, %loop.k.latch ]
  br label %loop.l.header

loop.l.header:
  %l = phi i64 [ 0, %loop.k.header ], [ %l.next, %loop.l.latch ]
  br label %loop.m.header

loop.m.header:
  %m = phi i64 [ 0, %loop.l.header ], [ %m.next, %loop.m.latch ]
  %left.row.base = mul nuw nsw i64 %m, 8
  %left.index = add nuw nsw i64 %left.row.base, %l
  %left.element.ptr = getelementptr i8, ptr %Left, i64 %left.index
  store i8 0, ptr %left.element.ptr, align 1
  br label %loop.m.latch

loop.m.latch:
  %m.next = add nuw nsw i64 %m, 1
  %m.done = icmp eq i64 %m.next, 8
  br i1 %m.done, label %loop.l.latch, label %loop.m.header

loop.l.latch:
  %l.next = add nuw nsw i64 %l, 1
  %l.done = icmp eq i64 %l.next, 8
  br i1 %l.done, label %loop.n.header, label %loop.l.header

loop.n.header:
  %n = phi i64 [ 0, %loop.l.latch ], [ %n.next, %loop.n.latch ]
  br label %loop.o.header

loop.o.header:
  %o = phi i64 [ 0, %loop.n.header ], [ %o.next, %loop.o.latch ]
  %right.row.base = mul nuw nsw i64 %o, 8
  %right.index = add nuw nsw i64 %right.row.base, %n
  %right.element.ptr = getelementptr i8, ptr %Right, i64 %right.index
  store i8 0, ptr %right.element.ptr, align 1
  br label %loop.o.latch

loop.o.latch:
  %o.next = add nuw nsw i64 %o, 1
  %o.done = icmp eq i64 %o.next, 8
  br i1 %o.done, label %loop.n.latch, label %loop.o.header

loop.n.latch:
  %n.next = add nuw nsw i64 %n, 1
  %n.done = icmp eq i64 %n.next, 8
  br i1 %n.done, label %loop.k.latch, label %loop.n.header

loop.k.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, 8
  br i1 %k.done, label %loop.j.latch, label %loop.k.header

loop.j.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %loop.i.latch, label %loop.j.header

loop.i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %loop.i.header

exit:
  ret void
}
