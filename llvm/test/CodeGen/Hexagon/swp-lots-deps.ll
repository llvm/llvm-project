; RUN: llc -march=hexagon -enable-pipeliner -stats -o /dev/null < %s 2>&1 -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=STATS
; REQUIRES: asserts

; STATS: 1 pipeliner        - Number of loops software pipelined

; Function Attrs: nounwind
define i64 @f0(i32 %a0, ptr %a1) #0 {
b0:
  %v0 = icmp slt i32 %a0, 123469
  br i1 %v0, label %b1, label %b4

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i64 [ undef, %b1 ], [ %v12, %b2 ]
  %v2 = phi i64 [ undef, %b1 ], [ %v10, %b2 ]
  %v3 = phi i32 [ 0, %b1 ], [ %v13, %b2 ]
  %v4 = phi i32 [ undef, %b1 ], [ %v9, %b2 ]
  %v5 = phi i64 [ undef, %b1 ], [ %v7, %b2 ]
  %v6 = phi i64 [ undef, %b1 ], [ %v11, %b2 ]
  %v7 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v5, i64 %v6, i64 %v6)
  %v8 = tail call i64 @llvm.hexagon.S2.packhl(i32 undef, i32 %v4)
  %v9 = load i32, ptr %a1, align 4, !tbaa !0
  %v10 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v2, i64 %v6, i64 %v8)
  %v11 = tail call i64 @llvm.hexagon.S2.packhl(i32 %v9, i32 undef)
  %v12 = tail call i64 @llvm.hexagon.M2.vdmacs.s0(i64 %v1, i64 %v6, i64 %v11)
  %v13 = add nsw i32 %v3, 1
  %v14 = icmp eq i32 %v13, undef
  br i1 %v14, label %b3, label %b2

b3:                                               ; preds = %b2
  %v15 = lshr i64 %v12, 32
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v16 = phi i64 [ %v10, %b3 ], [ undef, %b0 ]
  %v17 = phi i64 [ %v7, %b3 ], [ undef, %b0 ]
  %v18 = add i64 %v16, %v17
  ret i64 %v18
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M2.vdmacs.s0(i64, i64, i64) #1

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.packhl(i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
