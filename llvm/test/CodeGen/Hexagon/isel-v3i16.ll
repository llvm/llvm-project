; RUN: llc -march=hexagon < %s | FileCheck %s

; Check for a successful compilation.
; CHECK: callr

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = external dllexport global i8* (i32, i32, i64, i32, i32)*, align 4

define hidden void @f0(i32 %a0, <3 x i16>* %a1) #0 {
b0:
  %v0 = load i8* (i32, i32, i64, i32, i32)*, i8* (i32, i32, i64, i32, i32)** @g0, align 4
  %v1 = call i8* %v0(i32 1, i32 %a0, i64 314646, i32 0, i32 16)
  br label %b1

b1:                                               ; preds = %b2, %b0
  br label %b2

b2:                                               ; preds = %b5, %b1
  %v2 = icmp slt i32 0, 229
  br i1 %v2, label %b3, label %b1

b3:                                               ; preds = %b2
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b3
  %v3 = load <3 x i16>, <3 x i16>* %a1, align 2
  br label %b5

b5:                                               ; preds = %b4, %b3
  %v4 = phi <3 x i16> [ %v3, %b4 ], [ zeroinitializer, %b3 ]
  %v5 = bitcast i8* %v1 to i16*
  %v6 = getelementptr inbounds i16, i16* %v5, i32 undef
  %v7 = bitcast i16* %v6 to <3 x i16>*
  store <3 x i16> %v4, <3 x i16>* %v7, align 2
  br label %b2
}

attributes #0 = { "target-cpu"="hexagonv68" }
