; RUN: opt -passes=loop-unroll -debug-only=loop-unroll -S < %s  2>&1 | FileCheck %s
; REQUIRES: asserts


; CHECK: Loop Unroll: F[test]
; CHECK-NEXT: Not unrolling loop since parent loop has llvm.loop.unroll_and_jam
; CHECK: Loop Unroll: F[test]
; CHECK-NEXT: Not unrolling loop since it has llvm.loop.unroll_and_jam

define i32 @test() {
for.body4.preheader.preheader:
  br label %for.body4.preheader4

for.body4.preheader4:                             ; preds = %for.cond.cleanup3, %for.body4.preheader.preheader
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %for.cond.cleanup3 ], [ 0, %for.body4.preheader.preheader ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader4, %for.body4
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 0, %for.body4.preheader4 ]
  tail call void @test2(i64 %indvars.iv28, i64 %indvars.iv)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31.not = icmp eq i64 %indvars.iv.next29, 10
  br i1 %exitcond31.not, label %for.cond.cleanup, label %for.body4.preheader4, !llvm.loop !1

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 55
}


; CHECK: F[test_02]
; CHECK-NOT: Not unrolling loop since {{.*}}  llvm.loop.unroll_and_jam
define i32 @test_02() {
for.body4.preheader.preheader:
  br label %for.body4.preheader4

for.body4.preheader4:                             ; preds = %for.cond.cleanup3, %for.body4.preheader.preheader
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %for.cond.cleanup3 ], [ 0, %for.body4.preheader.preheader ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader4, %for.body4
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 0, %for.body4.preheader4 ]
  tail call void @test2(i64 %indvars.iv28, i64 %indvars.iv)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31.not = icmp eq i64 %indvars.iv.next29, 10
  br i1 %exitcond31.not, label %for.cond.cleanup, label %for.body4.preheader4, !llvm.loop !2

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 55
}

; CHECK:F[test_03]
; CHECK-NOT: Not unrolling loop since {{.*}}  llvm.loop.unroll_and_jam
define i32 @test_03() {
for.body4.preheader.preheader:
  br label %for.body4.preheader4

for.body4.preheader4:                             ; preds = %for.cond.cleanup3, %for.body4.preheader.preheader
  %indvars.iv28 = phi i64 [ %indvars.iv.next29, %for.cond.cleanup3 ], [ 0, %for.body4.preheader.preheader ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4.preheader4, %for.body4
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4 ], [ 0, %for.body4.preheader4 ]
  tail call void @test2(i64 %indvars.iv28, i64 %indvars.iv)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 10
  br i1 %exitcond.not, label %for.cond.cleanup3, label %for.body4

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31.not = icmp eq i64 %indvars.iv.next29, 10
  br i1 %exitcond31.not, label %for.cond.cleanup, label %for.body4.preheader4, !llvm.loop !3

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 55
}

declare void @test2(i64 signext, i64 signext)

!1 = distinct !{!1, !4}
!2 = distinct !{!1, !5}
!3 = distinct !{!1, !6}
!4 = !{!"llvm.loop.unroll_and_jam.count", i32 4}
!5 = !{!"llvm.loop.unroll_and_jam.count", i32 1}
!6 = !{!"llvm.loop.unroll_and_jam.disable"}
