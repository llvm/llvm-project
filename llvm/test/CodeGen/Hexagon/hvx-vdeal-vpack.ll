; REQUIRES: asserts
; RUN: llc -mtriple=hexagon -O2  < %s | FileCheck %s

;; Test that a vdeal + vpack can be lowered.

; CHECK: v[[#V1:]]:[[#V0:]] = vdeal(v[[#]],v[[#]],r[[#]])
; CHECK: v[[#]].h = vpacke(v[[#V1]].w,v[[#V0]].w)

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local <64 x i16> @f(<128 x i16> noundef %index) local_unnamed_addr #1 {
entry:
  %b = shufflevector <128 x i16> %index, <128 x i16> poison, <64 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28, i32 32, i32 36, i32 40, i32 44, i32 48, i32 52, i32 56, i32 60, i32 64, i32 68, i32 72, i32 76, i32 80, i32 84, i32 88, i32 92, i32 96, i32 100, i32 104, i32 108, i32 112, i32 116, i32 120, i32 124, i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30, i32 34, i32 38, i32 42, i32 46, i32 50, i32 54, i32 58, i32 62, i32 66, i32 70, i32 74, i32 78, i32 82, i32 86, i32 90, i32 94, i32 98, i32 102, i32 106, i32 110, i32 114, i32 118, i32 122, i32 126>
  ret <64 x i16> %b
}

attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn "target-cpu"="hexagonv75" "target-features"="+hvx-length128b,+hvx" }
