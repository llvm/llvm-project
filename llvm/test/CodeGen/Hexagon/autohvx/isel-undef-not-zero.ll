; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; Check that we don't generate lots of vinserts (of 0 that should be undef).
; CHECK: vinsert
; CHECK: vinsert
; CHECK-NOT: vinsert

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(ptr noalias align 128 %a0) #0 {
b0:
  %v1 = getelementptr inbounds i32, ptr %a0, i32 undef
  br label %b1

b1:                                               ; preds = %b0
  %v3 = load i8, ptr undef, align 1
  %v4 = insertelement <7 x i8> undef, i8 %v3, i32 0
  %v5 = shufflevector <7 x i8> %v4, <7 x i8> undef, <7 x i32> zeroinitializer
  %v6 = zext <7 x i8> %v5 to <7 x i32>
  %v7 = load <7 x i8>, ptr undef, align 1
  %v8 = zext <7 x i8> %v7 to <7 x i32>
  %v9 = mul nsw <7 x i32> %v6, %v8
  %v10 = add nsw <7 x i32> %v9, zeroinitializer
  store <7 x i32> %v10, ptr %v1, align 4
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv66" "target-features"="+hvx,+hvx-length128b" }

