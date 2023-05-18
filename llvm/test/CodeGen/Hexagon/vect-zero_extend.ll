; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; Used to fail with "Cannot select: 0x16cb2d0: v4i16 = zero_extend"

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = load <3 x i8>, ptr undef, align 8
  %v1 = zext <3 x i8> %v0 to <3 x i16>
  store <3 x i16> %v1, ptr undef, align 8
  br label %b2

b3:                                               ; preds = %b0
  ret void
}

attributes #0 = { nounwind }
