; RUN: llc -march=hexagon < %s | FileCheck %s

; This used to crash because SelectionDAG::isSplatValue did not set UndefElts
; for ISD::SPLAT_VECTOR.
; CHECK: vmemu

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @f0(ptr %a0) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr inbounds i16, ptr %a0, i32 undef
  %v1 = load <64 x i16>, ptr undef, align 2
  %v2 = shufflevector <64 x i16> %v1, <64 x i16> undef, <8 x i32> <i32 3, i32 11, i32 19, i32 27, i32 35, i32 43, i32 51, i32 59>
  %v3 = sext <8 x i16> %v2 to <8 x i32>
  %v4 = mul nsw <8 x i32> %v3, <i32 54492, i32 54492, i32 54492, i32 54492, i32 54492, i32 54492, i32 54492, i32 54492>
  %v5 = add nsw <8 x i32> %v4, <i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768>
  %v6 = add <8 x i32> %v5, zeroinitializer
  %v7 = ashr <8 x i32> %v6, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v8 = add nsw <8 x i32> zeroinitializer, %v7
  %v9 = shl <8 x i32> %v8, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v10 = ashr exact <8 x i32> %v9, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v11 = add nsw <8 x i32> %v10, zeroinitializer
  %v12 = trunc <8 x i32> %v11 to <8 x i16>
  %v13 = extractelement <8 x i16> %v12, i32 0
  store i16 %v13, ptr %v0, align 2
  ret void
}

attributes #0 = { "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
