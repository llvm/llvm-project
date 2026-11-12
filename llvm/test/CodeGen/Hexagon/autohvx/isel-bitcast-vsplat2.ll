; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: vsplat

target triple = "hexagon"

; Function Attrs: norecurse nounwind
define dso_local i32 @f0(ptr nocapture %a0, ptr nocapture readonly %a1, ptr nocapture readonly %a2, i32 %a3) local_unnamed_addr #0 {
b0:
  %v0 = insertelement <16 x i32> undef, i32 %a3, i32 0
  %v1 = shufflevector <16 x i32> %v0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v2 = add i32 %a3, 64
  %v3 = add <16 x i32> %v1, <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  %v4 = sdiv <16 x i32> %v3, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
  %v5 = add nsw <16 x i32> %v4, <i32 1000, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %v6 = shufflevector <16 x i32> %v5, <16 x i32> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v7 = add <16 x i32> %v5, %v6
  %v8 = extractelement <16 x i32> %v7, i32 0
  %v9 = add nsw i32 %v2, 1
  %v10 = sdiv i32 %v9, 23
  %v11 = add i32 %v8, %v10
  ret i32 %v11
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
