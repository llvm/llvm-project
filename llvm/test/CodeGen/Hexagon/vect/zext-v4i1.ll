; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: vcmph.eq

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define i32 @fred(<8 x i16>* %a0) #0 {
b0:
  switch i32 undef, label %b14 [
    i32 5, label %b2
    i32 3, label %b1
  ]

b1:                                               ; preds = %b0
  br label %b14

b2:                                               ; preds = %b0
  %v2 = load <8 x i16>, <8 x i16>* %a0, align 64
  %v3 = icmp eq <8 x i16> %v2, zeroinitializer
  %v4 = zext <8 x i1> %v3 to <8 x i16>
  %v5 = add <8 x i16> zeroinitializer, %v4
  %v6 = add <8 x i16> %v5, zeroinitializer
  %v7 = add <8 x i16> %v6, zeroinitializer
  %v8 = extractelement <8 x i16> %v7, i32 0
  %v9 = add i16 %v8, 0
  %v10 = add i16 %v9, 0
  %v11 = add i16 %v10, 0
  %v12 = icmp eq i16 %v11, 11
  br i1 %v12, label %b14, label %b13

b13:                                              ; preds = %b2
  ret i32 1

b14:                                              ; preds = %b2, %b1, %b0
  ret i32 0
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
