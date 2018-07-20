; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that the parsing succeeded.
; CHECK: f0

target triple = "hexagon"

@g0 = external global [0 x i16], align 8

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br label %b2

b1:                                               ; preds = %b2
  ret void

b2:                                               ; preds = %b2, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 4
  %v2 = getelementptr [0 x i16], [0 x i16]* @g0, i32 0, i32 %v0
  %v3 = bitcast i16* %v2 to <4 x i16>*
  %v4 = load <4 x i16>, <4 x i16>* %v3, align 2
  %v5 = icmp slt <4 x i16> %v4, zeroinitializer
  %v6 = xor <4 x i16> %v4, <i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %v7 = select <4 x i1> %v5, <4 x i16> %v6, <4 x i16> zeroinitializer
  store <4 x i16> %v7, <4 x i16>* %v3, align 2
  %v8 = icmp slt i32 %v1, 32768
  br i1 %v8, label %b2, label %b1
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
