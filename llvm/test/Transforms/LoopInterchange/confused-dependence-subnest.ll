; RUN: opt < %s -passes=loop-interchange -loop-interchange-profitabilities=ignore \
; RUN:     -pass-remarks-missed=loop-interchange -pass-remarks=loop-interchange \
; RUN:     -disable-output 2>&1 | FileCheck %s

; Reproducer for the confused-dependence path through partially-perfect
; subnests.
; Here, confused [j,k] subnest at depth 3.
; If the confused-dependence logic doesn't use the absolute depth,
; the pass wrongly interchanges the potentially aliasing [j,k] pair.
;
;   void f(double *A, double *C) {
;     for (int h = 0; h < 8; h++)
;       for (int i = 0; i < 8; i++) {
;         for (int j = 0; j < 8; j++)
;           for (int k = 0; k < 8; k++)
;             A[j*8 + k] = C[k] + 1.0;      // may alias -> confused, [* *]
;         for (int x = 0; x < 8; x++)
;           for (int y = 0; y < 8; y++)
;             A[(long)y*8 + x] += 2.0;
;       }
;   }
;
; After fixing the confused-dependence width, the [j, k] subnest bails out
; during dependency-matrix construction (its direction vector is all '*'),
; so it cannot be interchanged. The disjoint [x, y] sibling subnest is still
; interchanged.
;
; CHECK: remark: {{.*}}All loops have dependencies in all directions.
; CHECK: remark: {{.*}}Loop interchanged with enclosing loop.


define void @confused_subnest_depth3(ptr %A, ptr %C){
entry:
  br label %loop.h.header


loop.h.header:
  %h = phi i64 [ 0, %entry ], [ %h.next, %loop.h.latch ]
  br label %loop.i2.header


loop.i2.header:
  %i = phi i64 [ 0, %loop.h.header ], [ %i.next, %loop.i2.latch ]
  br label %loop.j2.header


loop.j2.header:
  %j = phi i64 [ 0, %loop.i2.header ], [ %j.next, %loop.j2.latch ]
  br label %loop.k2.header


loop.k2.header:
  %k = phi i64 [ 0, %loop.j2.header ], [ %k.next, %loop.k2.latch ]
  %c.ptr = getelementptr double, ptr %C, i64 %k
  %c.val = load double, ptr %c.ptr, align 8
  %sum = fadd double %c.val, 1.000000e+00
  %jrow = mul nuw nsw i64 %j, 8
  %aidx = add nuw nsw i64 %jrow, %k
  %a.ptr = getelementptr double, ptr %A, i64 %aidx
  store double %sum, ptr %a.ptr, align 8
  br label %loop.k2.latch


loop.k2.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, 8
  br i1 %k.done, label %loop.j2.latch, label %loop.k2.header


loop.j2.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %loop.x2.header, label %loop.j2.header


loop.x2.header:
  %x = phi i64 [ 0, %loop.j2.latch ], [ %x.next, %loop.x2.latch ]
  br label %loop.y2.header


loop.y2.header:
  %y = phi i64 [ 0, %loop.x2.header ], [ %y.next, %loop.y2.latch ]
  %row = mul nuw nsw i64 %y, 8
  %idx = add nuw nsw i64 %row, %x
  %axy.ptr = getelementptr double, ptr %A, i64 %idx
  %old = load double, ptr %axy.ptr, align 8
  %new = fadd double %old, 2.000000e+00
  store double %new, ptr %axy.ptr, align 8
  br label %loop.y2.latch


loop.y2.latch:
  %y.next = add nuw nsw i64 %y, 1
  %y.done = icmp eq i64 %y.next, 8
  br i1 %y.done, label %loop.x2.latch, label %loop.y2.header


loop.x2.latch:
  %x.next = add nuw nsw i64 %x, 1
  %x.done = icmp eq i64 %x.next, 8
  br i1 %x.done, label %loop.i2.latch, label %loop.x2.header


loop.i2.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %loop.h.latch, label %loop.i2.header


loop.h.latch:
  %h.next = add nuw nsw i64 %h, 1
  %h.done = icmp eq i64 %h.next, 8
  br i1 %h.done, label %exit, label %loop.h.header


exit:
  ret void
}

