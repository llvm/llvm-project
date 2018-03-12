; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that we do not ICE with a cannot select message when
; generating a v16i32 constant pool node.

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b1 ], [ 0, %b0 ]
  store <16 x i32> zeroinitializer, <16 x i32>* null, align 64
  br i1 false, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
