; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: dealloc_return

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local fastcc void @f0(ptr %a0) unnamed_addr #0 {
b0:
  %v0 = load <96 x float>, ptr poison, align 4
  %v1 = shufflevector <96 x float> %v0, <96 x float> poison, <32 x i32> <i32 1, i32 4, i32 7, i32 10, i32 13, i32 16, i32 19, i32 22, i32 25, i32 28, i32 31, i32 34, i32 37, i32 40, i32 43, i32 46, i32 49, i32 52, i32 55, i32 58, i32 61, i32 64, i32 67, i32 70, i32 73, i32 76, i32 79, i32 82, i32 85, i32 88, i32 91, i32 94>
  %v2 = fptrunc <32 x float> %v1 to <32 x half>
  %v3 = getelementptr half, ptr %a0, i32 0
  %v4 = shufflevector <32 x half> zeroinitializer, <32 x half> %v2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %v5 = shufflevector <64 x half> %v4, <64 x half> poison, <96 x i32> <i32 0, i32 32, i32 64, i32 1, i32 33, i32 65, i32 2, i32 34, i32 66, i32 3, i32 35, i32 67, i32 4, i32 36, i32 68, i32 5, i32 37, i32 69, i32 6, i32 38, i32 70, i32 7, i32 39, i32 71, i32 8, i32 40, i32 72, i32 9, i32 41, i32 73, i32 10, i32 42, i32 74, i32 11, i32 43, i32 75, i32 12, i32 44, i32 76, i32 13, i32 45, i32 77, i32 14, i32 46, i32 78, i32 15, i32 47, i32 79, i32 16, i32 48, i32 80, i32 17, i32 49, i32 81, i32 18, i32 50, i32 82, i32 19, i32 51, i32 83, i32 20, i32 52, i32 84, i32 21, i32 53, i32 85, i32 22, i32 54, i32 86, i32 23, i32 55, i32 87, i32 24, i32 56, i32 88, i32 25, i32 57, i32 89, i32 26, i32 58, i32 90, i32 27, i32 59, i32 91, i32 28, i32 60, i32 92, i32 29, i32 61, i32 93, i32 30, i32 62, i32 94, i32 31, i32 63, i32 95>
  store <96 x half> %v5, ptr %v3, align 2
  ret void
}

attributes #0 = { "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp" }
