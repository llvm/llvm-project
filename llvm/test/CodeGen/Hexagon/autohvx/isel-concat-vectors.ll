; RUN: llc -march=hexagon < %s | FileCheck %s

; Check for a non-crashing output.
; CHECK: vsplat

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #0
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #0
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #0

define void @crash(ptr %a0) #1 {
b0:
  %v1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 0) #0
  %v2 = bitcast <16 x i32> %v1 to <32 x i16>
  %v3 = shufflevector <32 x i16> %v2, <32 x i16> undef, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v4 = shufflevector <128 x i16> %v3, <128 x i16> undef, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %v5 = bitcast <64 x i16> %v4 to <32 x i32>
  %v6 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v5) #0
  %v7 = tail call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> undef, <16 x i32> %v6, i32 -2) #0
  %v8 = tail call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %v7)
  store <16 x i32> %v8, ptr %a0, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
