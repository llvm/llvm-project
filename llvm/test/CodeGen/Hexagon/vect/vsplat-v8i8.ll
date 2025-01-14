; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this compiles successfully. Used to crash with "cannot select
; v8i8 = vsplat ..."
; CHECK: jumpr r31

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nounwind
define i32 @fred() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = xor <16 x i8> undef, undef
  %v3 = icmp eq i32 undef, undef
  br i1 %v3, label %b4, label %b1

b4:                                               ; preds = %b1
  %v5 = xor <16 x i8> %v2, zeroinitializer
  %v6 = xor <16 x i8> %v5, zeroinitializer
  %v7 = xor <16 x i8> %v6, zeroinitializer
  %v8 = xor <16 x i8> %v7, zeroinitializer
  %v9 = shufflevector <16 x i8> %v8, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %v10 = xor <16 x i8> %v8, %v9
  %v11 = extractelement <16 x i8> %v10, i32 0
  br i1 undef, label %b14, label %b12

b12:                                              ; preds = %b4
  %v13 = xor i8 undef, %v11
  br label %b14

b14:                                              ; preds = %b12, %b4
  %v15 = phi i8 [ %v11, %b4 ], [ %v13, %b12 ]
  %v16 = zext i8 %v15 to i32
  ret i32 %v16
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
