; RUN: opt -S -passes=loop-vectorize -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; This test is checking that a scalable load inside a loop does not trigger a
; TypeSize error in the loop vectorization legality analysis. It is possible for
; a scalable/vector load to appear inside a loop at vectorization legality
; analysis if, for example, the ACLE are used. If we encounter a scalable/vector
; load, it should not be considered for analysis, and we should not see a
; TypeSize error.

; #include <arm_sve.h>
;
; void scalable_load_in_loop(long n, int *a, int *b, svuint32_t *x,
;                            svuint32_t *y) {
;     for (unsigned i = 0; i < n; i++) {
;         if (i % 2 == 0) continue;
;         a[i] = 2 * b[i];
;         *x = *y;
;     }
; }

; CHECK-LABEL: @scalable_load_in_loop
; CHECK-NOT: vector.body
define void @scalable_load_in_loop(i64 %n, ptr %x, ptr %y) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %rem = and i32 %i, 1
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %for.inc, label %if.end

if.end:
  %0 = load <vscale x 4 x i32>, ptr %y
  store <vscale x 4 x i32> %0, ptr %x
  br label %for.inc

for.inc:
  %inc = add i32 %i, 1
  %cmp2 = icmp slt i64 0, %n
  br i1 %cmp2, label %for.body, label %for.cleanup

for.cleanup:
  ret void
}
