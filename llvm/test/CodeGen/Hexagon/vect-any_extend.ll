; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
; Used to fail with "Cannot select: 0x17300f0: v2i32 = any_extend"

target triple = "hexagon-unknown-linux-gnu"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = load <4 x i8>, <4 x i8>* undef, align 8
  %v1 = zext <4 x i8> %v0 to <4 x i32>
  store <4 x i32> %v1, <4 x i32>* undef, align 8
  unreachable
}

attributes #0 = { nounwind }
