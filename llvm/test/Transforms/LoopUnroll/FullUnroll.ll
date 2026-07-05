; RUN: opt -passes='loop-unroll-full' -disable-verify -disable-loop-unrolling=true --mtriple x86_64-pc-linux-gnu -S -o - %s | FileCheck %s

; This checks that the loop full unroller will fire in the new pass manager
; when forced via #pragma in the source (or annotation in the code).

; Completely unroll the inner loop
; CHECK-LABEL: @foo
; CHECK: br i1
; CHECK-NOT: br i1

; Function Attrs: noinline nounwind optnone uwtable
define void @foo() local_unnamed_addr #0 {
bb:
  %tmp = alloca [5 x ptr], align 16
  br label %bb7.preheader

bb3.loopexit:                                     ; preds = %bb10
  %spec.select.lcssa = phi i32 [ %spec.select, %bb10 ]
  %tmp5.not = icmp eq i32 %spec.select.lcssa, 0
  br i1 %tmp5.not, label %bb24, label %bb7.preheader

bb7.preheader:                                    ; preds = %bb3.loopexit, %bb
  %tmp1.06 = phi i32 [ 5, %bb ], [ %spec.select.lcssa, %bb3.loopexit ]
  br label %bb10

bb10:                                             ; preds = %bb10, %bb7.preheader
  %indvars.iv = phi i64 [ 0, %bb7.preheader ], [ %indvars.iv.next, %bb10 ]
  %tmp1.14 = phi i32 [ %tmp1.06, %bb7.preheader ], [ %spec.select, %bb10 ]
  %tmp13 = getelementptr inbounds [5 x ptr], ptr %tmp, i64 0, i64 %indvars.iv
  %tmp14 = load ptr, ptr %tmp13, align 8
  %tmp15.not = icmp ne ptr %tmp14, null
  %tmp18 = sext i1 %tmp15.not to i32
  %spec.select = add nsw i32 %tmp1.14, %tmp18
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 5
  br i1 %exitcond.not, label %bb3.loopexit, label %bb10, !llvm.loop !1

bb24:                                             ; preds = %bb3.loopexit
  ret void
}

attributes #0 = { nounwind uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.full"}
