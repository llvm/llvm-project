; RUN: opt < %s -passes=loop-interchange -loop-interchange-profitabilities=ignore \
; RUN:     -pass-remarks-missed=loop-interchange -pass-remarks=loop-interchange \
; RUN:     -disable-output 2>&1 | FileCheck %s

; Reproducer for the confused-dependence path through partially-perfect
; subnests. The [j, k] subnest comes from:
;
;   void f(double *A, double *C, int n, int m, int p) {
;     for (int i = 0; i < n; i++) {
;       for (int j = 0; j < m; j++)
;         for (int k = 0; k < m; k++)
;           A[k] = C[k] + 1.0;        // may alias -> confused
;       for (int x = 0; x < p; x++)
;         for (int y = 0; y < p; y++)
;           A[(long)y * p + x] += 2.0;
;     }
;   }
;
; After fixing the confused-dependence width, the [j, k] subnest bails out
; during dependency-matrix construction (its direction vector is all '*'),
; so it cannot be interchanged. The disjoint [x, y] sibling subnest is still
; interchanged.
;
; CHECK: remark: {{.*}}All loops have dependencies in all directions.
; CHECK: remark: {{.*}}Loop interchanged with enclosing loop.

define void @confused_subnest(ptr %A, ptr %C) {
entry:
  br label %loop.i.header

loop.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.i.latch ]
  br label %loop.j.header

loop.j.header:
  %j = phi i64 [ 0, %loop.i.header ], [ %j.next, %loop.j.latch ]
  br label %loop.k.header

loop.k.header:
  %k = phi i64 [ 0, %loop.j.header ], [ %k.next, %loop.k.latch ]
  %c.ptr = getelementptr double, ptr %C, i64 %k
  %c.val = load double, ptr %c.ptr, align 8
  %sum = fadd double %c.val, 1.000000e+00
  %a.ptr = getelementptr double, ptr %A, i64 %k
  store double %sum, ptr %a.ptr, align 8
  br label %loop.k.latch

loop.k.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, 8
  br i1 %k.done, label %loop.j.latch, label %loop.k.header

loop.j.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %loop.x.header, label %loop.j.header

loop.x.header:
  %x = phi i64 [ 0, %loop.j.latch ], [ %x.next, %loop.x.latch ]
  br label %loop.y.header

loop.y.header:
  %y = phi i64 [ 0, %loop.x.header ], [ %y.next, %loop.y.latch ]
  %row = mul nuw nsw i64 %y, 8
  %idx = add nuw nsw i64 %row, %x
  %axy.ptr = getelementptr double, ptr %A, i64 %idx
  %old = load double, ptr %axy.ptr, align 8
  %new = fadd double %old, 2.000000e+00
  store double %new, ptr %axy.ptr, align 8
  br label %loop.y.latch

loop.y.latch:
  %y.next = add nuw nsw i64 %y, 1
  %y.done = icmp eq i64 %y.next, 8
  br i1 %y.done, label %loop.x.latch, label %loop.y.header

loop.x.latch:
  %x.next = add nuw nsw i64 %x, 1
  %x.done = icmp eq i64 %x.next, 8
  br i1 %x.done, label %loop.i.latch, label %loop.x.header

loop.i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %loop.i.header

exit:
  ret void
}
