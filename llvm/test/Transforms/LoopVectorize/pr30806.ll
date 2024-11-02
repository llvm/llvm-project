; RUN: opt -passes=loop-vectorize -S < %s 2>&1 | FileCheck %s

; Produced from test-case:
;
; void testGuardedInnerLoop(uint32_t *ptr, uint32_t denom, uint32_t numer, uint32_t outer_lim)
; {
;   for(uint32_t outer_i = 0; outer_i < outer_lim; ++outer_i) {
;     if (denom > 0) {
;       const uint32_t lim = numer / denom;
;
;       for (uint32_t i = 0; i < lim; ++i)
;         ptr[i] = 1;
;     }
;   }
; }


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

define void @testGuardedInnerLoop(i32* %ptr, i32 %denom, i32 %numer, i32 %outer_lim) {
entry:
  %cmp1 = icmp eq i32 %outer_lim, 0
  br i1 %cmp1, label %exit, label %loop1.preheader

; Verify that a 'udiv' does not appear between the 'loop1.preheader' label, and
; whatever label comes next.
loop1.preheader:
; CHECK-LABEL: loop1.preheader:
; CHECK-NOT: udiv
; CHECK-LABEL: :
  br label %loop1

loop1:
  %outer_i = phi i32 [ %inc1, %loop2.exit ], [ 0, %loop1.preheader ]
  %0 = add i32 %denom, -1
  %1 = icmp ult i32 %0, %numer
  br i1 %1, label %loop2.preheader, label %loop2.exit

; Verify that a 'udiv' does appear between the 'loop2.preheader' label, and
; whatever label comes next.
loop2.preheader:
; CHECK-LABEL: loop2.preheader:
; CHECK: udiv
; CHECK-LABEL: :
  %lim = udiv i32 %numer, %denom
  %2 = zext i32 %lim to i64
  br label %loop2

loop2:
  %indvar.loop2 = phi i64 [ 0, %loop2.preheader ], [ %indvar.loop2.next, %loop2 ]
  %arrayidx = getelementptr inbounds i32, i32* %ptr, i64 %indvar.loop2
  store i32 1, i32* %arrayidx, align 4
  %indvar.loop2.next = add nuw nsw i64 %indvar.loop2, 1
  %cmp2 = icmp ult i64 %indvar.loop2.next, %2
  br i1 %cmp2, label %loop2, label %loop2.exit

loop2.exit:
  %inc1 = add nuw i32 %outer_i, 1
  %exitcond = icmp eq i32 %inc1, %outer_lim
  br i1 %exitcond, label %exit, label %loop1

exit:
  ret void
}
