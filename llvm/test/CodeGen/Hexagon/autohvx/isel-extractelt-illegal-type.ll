; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Make sure this doesn't crash.
; CHECK: = mem

target triple = "hexagon"

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32, i32) #0

; Function Attrs: nounwind readonly
define dso_local signext i16 @f0(ptr nocapture readonly %a0) local_unnamed_addr #1 {
b0:
  %v0 = load <8 x i16>, ptr undef, align 2, !tbaa !0
  %v1 = shufflevector <8 x i16> %v0, <8 x i16> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %v2 = shufflevector <8 x i16> %v1, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %v3 = lshr <16 x i16> %v2, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %v4 = and <16 x i16> %v3, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %v5 = extractelement <16 x i16> %v4, i32 0
  %v6 = sext i16 %v5 to i32
  %v7 = tail call i32 @llvm.hexagon.A2.subh.l16.sat.ll(i32 %v6, i32 16)
  %v8 = trunc i32 %v7 to i16
  %v9 = icmp sgt i16 %v8, -1
  %v10 = select i1 %v9, i16 0, i16 1
  ret i16 %v10
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
