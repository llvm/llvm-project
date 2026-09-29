; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: dealloc_return

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(ptr %a0, ptr %a1) local_unnamed_addr #0 {
b0:
  %v2 = load <160 x half>, ptr %a1, align 4
  %v3 = shufflevector <160 x half> %v2, <160 x half> poison, <32 x i32> <i32 4, i32 9, i32 14, i32 19, i32 24, i32 29, i32 34, i32 39, i32 44, i32 49, i32 54, i32 59, i32 64, i32 69, i32 74, i32 79, i32 84, i32 89, i32 94, i32 99, i32 104, i32 109, i32 114, i32 119, i32 124, i32 129, i32 134, i32 139, i32 144, i32 149, i32 154, i32 159>
  %v4 = fadd nnan nsz <32 x half> %v3, zeroinitializer
  %v7 = shufflevector <32 x half> %v4, <32 x half> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v8 = shufflevector <128 x half> undef, <128 x half> %v7, <160 x i32> <i32 0, i32 32, i32 64, i32 96, i32 128, i32 1, i32 33, i32 65, i32 97, i32 129, i32 2, i32 34, i32 66, i32 98, i32 130, i32 3, i32 35, i32 67, i32 99, i32 131, i32 4, i32 36, i32 68, i32 100, i32 132, i32 5, i32 37, i32 69, i32 101, i32 133, i32 6, i32 38, i32 70, i32 102, i32 134, i32 7, i32 39, i32 71, i32 103, i32 135, i32 8, i32 40, i32 72, i32 104, i32 136, i32 9, i32 41, i32 73, i32 105, i32 137, i32 10, i32 42, i32 74, i32 106, i32 138, i32 11, i32 43, i32 75, i32 107, i32 139, i32 12, i32 44, i32 76, i32 108, i32 140, i32 13, i32 45, i32 77, i32 109, i32 141, i32 14, i32 46, i32 78, i32 110, i32 142, i32 15, i32 47, i32 79, i32 111, i32 143, i32 16, i32 48, i32 80, i32 112, i32 144, i32 17, i32 49, i32 81, i32 113, i32 145, i32 18, i32 50, i32 82, i32 114, i32 146, i32 19, i32 51, i32 83, i32 115, i32 147, i32 20, i32 52, i32 84, i32 116, i32 148, i32 21, i32 53, i32 85, i32 117, i32 149, i32 22, i32 54, i32 86, i32 118, i32 150, i32 23, i32 55, i32 87, i32 119, i32 151, i32 24, i32 56, i32 88, i32 120, i32 152, i32 25, i32 57, i32 89, i32 121, i32 153, i32 26, i32 58, i32 90, i32 122, i32 154, i32 27, i32 59, i32 91, i32 123, i32 155, i32 28, i32 60, i32 92, i32 124, i32 156, i32 29, i32 61, i32 93, i32 125, i32 157, i32 30, i32 62, i32 94, i32 126, i32 158, i32 31, i32 63, i32 95, i32 127, i32 159>
  store <160 x half> %v8, ptr %a0, align 4
  ret void
}

attributes #0 = { "target-features"="+hvxv69,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp" }
