; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; Truncating a type-to-be-widenened to a legal type (v8i8).
; Check that this compiles successfully.
; CHECK-LABEL: f0:
; CHECK: dealloc_return

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(ptr %a0) local_unnamed_addr #0 {
b0:
  %v0 = load i8, ptr undef, align 1
  %v1 = zext i8 %v0 to i16
  %v2 = add i16 0, %v1
  %v3 = icmp sgt i16 %v2, 1
  %v4 = select i1 %v3, i16 %v2, i16 1
  %v5 = udiv i16 -32768, %v4
  %v6 = zext i16 %v5 to i32
  %v7 = insertelement <8 x i32> undef, i32 %v6, i32 0
  %v8 = shufflevector <8 x i32> %v7, <8 x i32> undef, <8 x i32> zeroinitializer
  %v9 = load <8 x i16>, ptr undef, align 2
  %v10 = sext <8 x i16> %v9 to <8 x i32>
  %v11 = mul nsw <8 x i32> %v8, %v10
  %v12 = add nsw <8 x i32> %v11, <i32 16384, i32 16384, i32 16384, i32 16384, i32 16384, i32 16384, i32 16384, i32 16384>
  %v13 = lshr <8 x i32> %v12, <i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15, i32 15>
  %v14 = trunc <8 x i32> %v13 to <8 x i8>
  %v15 = getelementptr inbounds i8, ptr %a0, i32 undef
  store <8 x i8> %v14, ptr %v15, align 1
  ret void
}

attributes #0 = { "target-features"="+hvx,+hvx-length128b" }
