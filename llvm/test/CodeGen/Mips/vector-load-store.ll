; RUN: llc -mtriple=mipsel -mattr=+dsp < %s | FileCheck %s

@g1 = common global <2 x i16> zeroinitializer, align 4
@g0 = common global <2 x i16> zeroinitializer, align 4
@g3 = common global <4 x i8> zeroinitializer, align 4
@g2 = common global <4 x i8> zeroinitializer, align 4

define void @func_v2i16() nounwind {
entry:
; CHECK: lw
; CHECK: sw

  %0 = load <2 x i16>, ptr @g1, align 4
  store <2 x i16> %0, ptr @g0, align 4
  ret void
}

define void @func_v4i8() nounwind {
entry:
; CHECK: lw
; CHECK: sw

  %0 = load <4 x i8>, ptr @g3, align 4
  store <4 x i8> %0, ptr @g2, align 4
  ret void
}

