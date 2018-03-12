; RUN: llc  -march=hexagon -mno-pairing -mno-compound < %s
; REQUIRES: asserts

; Test that the SWP doesn't assert when generating new phis. In this example, a
; phi references another phi and phi as well as the phi loop value are all
; defined in different stages.
;  v3 =             stage 2
;  v2 = phi(vb, v3) stage 1
;  v1 = phi(va, v2) stage 0

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  br i1 undef, label %b3, label %b4

b4:                                               ; preds = %b4, %b3
  %v0 = phi i32 [ %v17, %b4 ], [ undef, %b3 ]
  %v1 = phi i64 [ %v13, %b4 ], [ undef, %b3 ]
  %v2 = phi i32 [ %v19, %b4 ], [ undef, %b3 ]
  %v3 = phi i32 [ %v4, %b4 ], [ undef, %b3 ]
  %v4 = phi i32 [ %v14, %b4 ], [ undef, %b3 ]
  %v5 = phi i32 [ %v18, %b4 ], [ undef, %b3 ]
  %v6 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v3, i32 %v3)
  %v7 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v0, i32 undef)
  %v8 = tail call i64 @llvm.hexagon.S2.valignib(i64 %v7, i64 undef, i32 2)
  %v9 = inttoptr i32 %v5 to i16*
  %v10 = load i16, i16* %v9, align 2, !tbaa !0
  %v11 = sext i16 %v10 to i32
  %v12 = add nsw i32 %v5, -8
  %v13 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v1, i64 %v6, i64 %v8)
  %v14 = tail call i32 @llvm.hexagon.A2.combine.ll(i32 %v11, i32 %v0)
  %v15 = inttoptr i32 %v12 to i16*
  %v16 = load i16, i16* %v15, align 2, !tbaa !0
  %v17 = sext i16 %v16 to i32
  %v18 = add nsw i32 %v5, -16
  %v19 = add nsw i32 %v2, 1
  %v20 = icmp eq i32 %v19, 0
  br i1 %v20, label %b5, label %b4

b5:                                               ; preds = %b4
  %v21 = phi i64 [ %v13, %b4 ]
  unreachable
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.combine.ll(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.valignib(i64, i64, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
