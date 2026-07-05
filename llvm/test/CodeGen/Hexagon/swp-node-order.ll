; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Test that we include all the nodes in the final node ordering
; computation. This test creates two set of nodes that are processed
; by computeNodeOrder().

; Function Attrs: nounwind
define void @f0(i32 %a0) #0 {
b0:
  %v0 = add nsw i32 undef, 4
  %v1 = ashr i32 %a0, 1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi i64 [ %v5, %b1 ], [ 0, %b0 ]
  %v3 = phi i64 [ %v9, %b1 ], [ undef, %b0 ]
  %v4 = phi i32 [ %v10, %b1 ], [ 0, %b0 ]
  %v5 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v2, i64 %v3, i64 undef)
  %v6 = tail call i64 @llvm.hexagon.A2.combinew(i32 0, i32 0)
  %v7 = tail call i64 @llvm.hexagon.S2.shuffeh(i64 %v6, i64 undef)
  %v8 = trunc i64 %v7 to i32
  %v9 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v8, i32 undef)
  %v10 = add nsw i32 %v4, 1
  %v11 = icmp eq i32 %v10, %v1
  br i1 %v11, label %b2, label %b1

b2:                                               ; preds = %b1
  %v12 = trunc i64 %v5 to i32
  %v13 = inttoptr i32 %v0 to ptr
  store i32 %v12, ptr %v13, align 4, !tbaa !0
  call void @llvm.trap()
  unreachable
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.shuffeh(i64, i64) #1

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
