; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -enable-vplan-native-path -force-vector-width=2 -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s
; RUN: opt -S -passes=loop-vectorize -scalable-vectorization=on -force-vector-width=4 -force-vector-interleave=1 -force-target-supports-scalable-vectors=true -debug-only=loop-vectorize < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCALABLE

; Verify that llvm.pseudoprobe present in the outer loop does not crash the 
; VPlan native path and is not treated as a widenable intrinsic.
;
; CHECK-NOT: WIDEN-INTRINSIC

define i32 @main() {
entry:
  br label %outer.header

outer.header:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %inner

outer.latch:
  %j.next = add i64 %j, 1
  call void @llvm.pseudoprobe(i64 0, i64 0, i32 0, i64 0)
  %outer.cond = icmp eq i64 %j, 100
  br i1 %outer.cond, label %exit, label %outer.header, !llvm.loop !0

inner:
  %k = phi i64 [ 0, %outer.header ], [ %k.next, %inner ]
  %k.next = add i64 %k, 1
  %inner.cond = icmp eq i64 %k.next, 0
  br i1 %inner.cond, label %outer.latch, label %inner

exit:
  ret i32 0
}

; Verify that pseudoprobe does not crash with scalable vector vectorization
;
; CHECK-SCALABLE: LV: Found uniform instruction:{{.*}}pseudoprobe
; CHECK-SCALABLE: CLONE call @llvm.pseudoprobe
; CHECK-SCALABLE-NOT: WIDEN-INTRINSIC{{.*}}pseudoprobe
;
; CHECK-SCALABLE-LABEL: @test_scalable
; CHECK-SCALABLE: vector.body:
; CHECK-SCALABLE: call void @llvm.pseudoprobe
; CHECK-SCALABLE: br i1

define void @test_scalable() {
entry:
    br label %loop

loop:
    %iv = phi i64 [ 1, %entry ], [ %iv.next, %loop ]
    %iv.next = add i64 %iv, 1
    call void @llvm.pseudoprobe(i64 0, i64 0, i32 0, i64 0)
    %done = icmp eq i64 %iv.next, 0
    br i1 %done, label %exit, label %loop

exit:
    ret void
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
