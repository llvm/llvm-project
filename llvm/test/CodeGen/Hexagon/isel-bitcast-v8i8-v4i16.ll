; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this doesn't fail to select instructions.
; CHECK: vsplath

define <8 x i8> @fred(i16 %a0) #0 {
  %t0 = insertelement <4 x i16> undef, i16 %a0, i32 0
  %t1 = shufflevector <4 x i16> %t0, <4 x i16> undef, <4 x i32> zeroinitializer
  %t2 = bitcast <4 x i16> %t1 to <8 x i8>
  ret <8 x i8> %t2
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv62" }
