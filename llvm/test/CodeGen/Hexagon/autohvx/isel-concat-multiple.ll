; RUN: llc -march=hexagon < %s | FileCheck %s

; This code generates a concat_vectors with more than 2 inputs. Make sure
; that this compiles successfully.
; CHECK: lsr

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(ptr %a0, ptr %a1, ptr %a2) #0 {
b0:
  %v1 = load i32, ptr %a0, align 4
  %v2 = mul nsw i32 %v1, -15137
  %v3 = add nsw i32 0, %v2
  %v4 = sub nsw i32 0, %v3
  %v5 = load i32, ptr %a1, align 4
  %v6 = insertelement <2 x i32> undef, i32 %v5, i32 1
  %v7 = add nsw <2 x i32> %v6, %v6
  %v8 = extractelement <2 x i32> %v7, i32 0
  %v9 = insertelement <4 x i32> undef, i32 %v4, i32 2
  %v10 = insertelement <4 x i32> %v9, i32 undef, i32 3
  %v11 = add <4 x i32> %v10, %v10
  %v12 = sub <4 x i32> %v11, zeroinitializer
  %v13 = shufflevector <4 x i32> %v12, <4 x i32> undef, <8 x i32> <i32 undef, i32 0, i32 undef, i32 1, i32 undef, i32 2, i32 undef, i32 3>
  %v14 = shufflevector <8 x i32> undef, <8 x i32> %v13, <8 x i32> <i32 0, i32 9, i32 2, i32 11, i32 4, i32 13, i32 6, i32 15>
  %v15 = lshr <8 x i32> %v14, <i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18, i32 18>
  %v16 = and <8 x i32> %v15, %v14
  %v17 = extractelement <8 x i32> %v16, i32 5
  %v18 = getelementptr inbounds i8, ptr null, i32 %v17
  %v19 = load i8, ptr %v18, align 1
  store i8 %v19, ptr %a2, align 1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
