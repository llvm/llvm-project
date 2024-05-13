; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this does not crash.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @f0() local_unnamed_addr #0 {
b0:
  %v0 = load i32, ptr undef, align 4
  %v1 = select i1 undef, i32 0, i32 1073741823
  %v2 = shl i32 %v1, 0
  %v3 = sext i32 %v0 to i64
  %v4 = sext i32 %v2 to i64
  %v5 = mul nsw i64 %v4, %v3
  %v6 = lshr i64 %v5, 32
  %v7 = trunc i64 %v6 to i32
  %v8 = sext i32 %v7 to i64
  %v9 = insertelement <32 x i64> undef, i64 %v8, i32 0
  %v10 = shufflevector <32 x i64> %v9, <32 x i64> undef, <32 x i32> zeroinitializer
  %v11 = getelementptr i32, ptr null, i32 32
  %v13 = load <32 x i32>, ptr %v11, align 4
  %v14 = shl <32 x i32> %v13, zeroinitializer
  %v15 = sext <32 x i32> %v14 to <32 x i64>
  %v16 = mul nsw <32 x i64> %v10, %v15
  %v17 = lshr <32 x i64> %v16, <i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32, i64 32>
  %v18 = trunc <32 x i64> %v17 to <32 x i32>
  store <32 x i32> %v18, ptr %v11, align 4
  ret void
}

attributes #0 = { "target-features"="+hvx-length128b,+hvxv67,+v67,-long-calls" }
