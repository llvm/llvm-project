; RUN: opt < %s -passes=loop-interchange -loop-interchange-profitabilities=ignore -debug-only=loop-interchange -disable-output -S 2>%t
; RUN: FileCheck --input-file=%t %s

; There is no partially-perfect subnest here. Every innermost loop has a parent
; with multiple child loops, so collectPerfectNests() should return an empty
; list and loop-interchange should bail out immediately without attempting or
; performing any interchange.
;
; Corresponding C code:
;
;   for (int i = 0; i < 16; ++i) {
;     for (int j = 0; j < 16; ++j)
;       A[i][j] = 0;
;
;     for (int k = 0; k < 16; ++k)
;       B[i][k] = 0;
;   }
;
; CHECK: No Valid candidates for loop interchange.

define void @no_partially_perfect_subnest(ptr noalias %A, ptr noalias %B) {
entry:
  br label %loop.i.header

loop.i.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.i.latch ]
  br label %loop.j.header

loop.j.header:
  %j = phi i64 [ 0, %loop.i.header ], [ %j.next, %loop.j.latch ]
  %a.row.base = mul nuw nsw i64 %i, 16
  %a.index = add nuw nsw i64 %a.row.base, %j
  %a.element.ptr = getelementptr i8, ptr %A, i64 %a.index
  store i8 1, ptr %a.element.ptr, align 1
  br label %loop.j.latch

loop.j.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %loop.k.header, label %loop.j.header

loop.k.header:
  %k = phi i64 [ 0, %loop.j.latch ], [ %k.next, %loop.k.latch ]
  %b.row.base = mul nuw nsw i64 %i, 16
  %b.index = add nuw nsw i64 %b.row.base, %k
  %b.element.ptr = getelementptr i8, ptr %B, i64 %b.index
  store i8 2, ptr %b.element.ptr, align 1
  br label %loop.k.latch

loop.k.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %loop.i.latch, label %loop.k.header

loop.i.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %loop.i.header

exit:
  ret void
}
