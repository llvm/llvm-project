; RUN: opt < %s -simple-loop-unswitch-inject-invariant-conditions=true -passes='loop(simple-loop-unswitch<nontrivial>,loop-instsimplify)' -S | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

define void @test() {
; CHECK-LABEL: test
bb:
  %tmp = call i1 @llvm.experimental.widenable.condition()
  %tmp1 = load atomic i32, ptr addrspace(1) poison unordered, align 8
  %tmp2 = load atomic i32, ptr addrspace(1) poison unordered, align 8
  br label %bb3

bb3:                                              ; preds = %bb15, %bb
  br label %bb4

bb4:                                              ; preds = %bb13, %bb3
  %tmp5 = phi i32 [ poison, %bb3 ], [ %tmp14, %bb13 ]
  %tmp6 = phi i32 [ poison, %bb3 ], [ %tmp5, %bb13 ]
  %tmp7 = add nuw nsw i32 %tmp6, 2
  %tmp8 = icmp ult i32 %tmp7, %tmp2
  br i1 %tmp8, label %bb9, label %bb16, !prof !0

bb9:                                              ; preds = %bb4
  %tmp10 = icmp ult i32 %tmp7, %tmp1
  %tmp11 = and i1 %tmp10, %tmp
  br i1 %tmp11, label %bb12, label %bb17, !prof !0

bb12:                                             ; preds = %bb9
  br i1 poison, label %bb15, label %bb13

bb13:                                             ; preds = %bb12
  %tmp14 = add nuw nsw i32 %tmp5, 1
  br label %bb4

bb15:                                             ; preds = %bb12
  br label %bb3

bb16:                                             ; preds = %bb4
  ret void

bb17:                                             ; preds = %bb9
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite)
declare i1 @llvm.experimental.widenable.condition()

!0 = !{!"branch_weights", i32 1048576, i32 1}

