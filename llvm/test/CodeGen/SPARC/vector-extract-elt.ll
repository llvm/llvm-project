; RUN: llc -march=sparc < %s | FileCheck %s


; If computeKnownSignBits (in SelectionDAG) can do a simple
; look-thru for extractelement then we know that the add will yield a
; non-negative result.
define i1 @test1(<4 x i16>* %in) {
; CHECK-LABEL: ! %bb.0:
; CHECK-NEXT:        retl
; CHECK-NEXT:        sethi 0, %o0
  %vec2 = load <4 x i16>, <4 x i16>* %in, align 1
  %vec3 = lshr <4 x i16> %vec2, <i16 2, i16 2, i16 2, i16 2>
  %vec4 = sext <4 x i16> %vec3 to <4 x i32>
  %elt0 = extractelement <4 x i32> %vec4, i32 0
  %elt1 = extractelement <4 x i32> %vec4, i32 1
  %sum = add i32 %elt0, %elt1
  %bool = icmp slt i32 %sum, 0
  ret i1 %bool
}
