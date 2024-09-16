; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -verify-dom-info -verify-loop-info \
; RUN:     -S -debug 2>&1 | FileCheck %s

;; Test that a confused dependency in loop should prevent interchange in
;; loops i and j.
;;
;; void test_deps() {
;;   for (int i = 0; i <= 3; i++)
;;     for (int j = 0; j <= 3; j++) {
;;       *f ^= 0x1000;
;;       c[j][i] = *f;
;;     }
;; }


; CHECK:  Confused dependency between:
; CHECK:    store i32 %xor, ptr %arrayidx6, align 4
; CHECK:    %1 = load i32, ptr %0, align 4
; CHECK-NOT: Loops interchanged.

@a = global i32 0, align 4
@f = global ptr @a, align 8
@c = global [4 x [4 x i32]] zeroinitializer, align 8

define void @test_deps() {
entry:
  %0 = load ptr, ptr @f, align 8
  br label %for.cond1.preheader

; Loop:
for.cond1.preheader:                              ; preds = %entry, %for.cond.cleanup3
  %indvars.iv16 = phi i64 [ 0, %entry ], [ %indvars.iv.next17, %for.cond.cleanup3 ]
  br label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1
  %exitcond18 = icmp ne i64 %indvars.iv.next17, 4
  br i1 %exitcond18, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:                                        ; preds = %for.cond1.preheader, %for.body4
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %1 = load i32, ptr %0, align 4
  %xor = xor i32 %1, 4096
  store i32 %xor, ptr %0, align 4
  %arrayidx6 = getelementptr inbounds [4 x [4 x i32]], ptr @c, i64 0, i64 %indvars.iv, i64 %indvars.iv16
  store i32 %xor, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 4
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

; Exit blocks
for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void
}
