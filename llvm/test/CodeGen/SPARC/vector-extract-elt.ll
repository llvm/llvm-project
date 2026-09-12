; RUN: llc -mtriple=sparc < %s | FileCheck %s


; If computeKnownSignBits (in SelectionDAG) can do a simple
; look-thru for extractelement then we know that the add will yield a
; non-negative result.
define i1 @test1(ptr %in) {
; CHECK-LABEL: ! %bb.0:
; CHECK-NEXT:        retl
; CHECK-NEXT:        mov %g0, %o0
  %vec2 = load <4 x i16>, ptr %in, align 1
  %vec3 = lshr <4 x i16> %vec2, <i16 2, i16 2, i16 2, i16 2>
  %vec4 = sext <4 x i16> %vec3 to <4 x i32>
  %elt0 = extractelement <4 x i32> %vec4, i32 0
  %elt1 = extractelement <4 x i32> %vec4, i32 1
  %sum = add i32 %elt0, %elt1
  %bool = icmp slt i32 %sum, 0
  ret i1 %bool
}

; A legal v2i32 is used internally to represent 64-bit integer register pairs.
; Make sure a dynamic extract produced while legalizing a smaller vector does
; not reach instruction selection, which only handles constant indices.
define i8 @extract_v2i8(<2 x i8> %v, i32 %idx) {
; CHECK-LABEL: extract_v2i8:
; CHECK:       cmp %o2, 0
; CHECK:       be
; CHECK:       mov %o1, %o0
; CHECK:       retl
  %elt = extractelement <2 x i8> %v, i32 %idx
  ret i8 %elt
}
