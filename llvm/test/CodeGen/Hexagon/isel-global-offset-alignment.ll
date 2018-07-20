; RUN: llc -march=hexagon < %s | FileCheck %s

; This should compile without errors, and the offsets with respect to the
; beginning of the global "array" don't need to be multiples of 8.
;
; CHECK-DAG: memd(r0+##array+174)
; CHECK-DAG: memd(r0+##array+182)

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@array = external global [1000000 x i16], align 8

define void @fred() #0 {
b0:
  br i1 undef, label %b3, label %b1

b1:                                               ; preds = %b0
  %v2 = add i32 0, 512
  br label %b3

b3:                                               ; preds = %b1, %b0
  %v4 = phi i32 [ 0, %b0 ], [ %v2, %b1 ]
  %v5 = or i32 %v4, 1
  %v6 = add nsw i32 %v5, -1
  %v7 = getelementptr inbounds [1000000 x i16], [1000000 x i16]* @array, i32 0, i32 %v6
  %v8 = getelementptr i16, i16* %v7, i32 88
  %v9 = bitcast i16* %v8 to <8 x i16>*
  store <8 x i16> zeroinitializer, <8 x i16>* %v9, align 8
  unreachable
}

attributes #0 = { norecurse nounwind }
