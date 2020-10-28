; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: vmemu

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dso_local void @f0(i16* %a0) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr inbounds i16, i16* %a0, i32 undef
  %v1 = load <64 x i16>, <64 x i16>* undef, align 2
  %v2 = shufflevector <64 x i16> %v1, <64 x i16> undef, <8 x i32> <i32 2, i32 10, i32 18, i32 26, i32 34, i32 42, i32 50, i32 58>
  %v3 = shufflevector <64 x i16> %v1, <64 x i16> undef, <8 x i32> <i32 6, i32 14, i32 22, i32 30, i32 38, i32 46, i32 54, i32 62>
  %v4 = sext <8 x i16> %v2 to <8 x i32>
  %v5 = mul nsw <8 x i32> %v4, <i32 60548, i32 60548, i32 60548, i32 60548, i32 60548, i32 60548, i32 60548, i32 60548>
  %v6 = sext <8 x i16> %v3 to <8 x i32>
  %v7 = mul nsw <8 x i32> %v6, <i32 25080, i32 25080, i32 25080, i32 25080, i32 25080, i32 25080, i32 25080, i32 25080>
  %v8 = add nsw <8 x i32> %v5, <i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768, i32 32768>
  %v9 = add <8 x i32> %v8, %v7
  %v10 = ashr <8 x i32> %v9, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v11 = add nsw <8 x i32> %v10, zeroinitializer
  %v12 = shl <8 x i32> %v11, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v13 = ashr exact <8 x i32> %v12, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  %v14 = add nsw <8 x i32> zeroinitializer, %v13
  %v15 = trunc <8 x i32> %v14 to <8 x i16>
  %v16 = extractelement <8 x i16> %v15, i32 0
  store i16 %v16, i16* %v0, align 2
  ret void
}

attributes #0 = { "target-features"="+v66,+hvxv66,+hvx,+hvx-length64b" }
