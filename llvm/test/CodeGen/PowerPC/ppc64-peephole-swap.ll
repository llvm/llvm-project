; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-PWR8

; The strightforward expansion of this code will result in a swap followed by a
;  splat. However, the swap is not needed since in this case the splat is the
;  only use.
; We want to check that we are not using the swap and that we have indexed the
;  splat to the correct location.
; 8 Bit Signed Version of the test.
; Function Attrs: norecurse nounwind readnone
define <16 x i8> @splat_8_plus(<16 x i8> %v, i8 signext %c) local_unnamed_addr {
entry:
  %splat.splatinsert.i = insertelement <16 x i8> undef, i8 %c, i32 0
  %splat.splat.i = shufflevector <16 x i8> %splat.splatinsert.i, <16 x i8> undef, <16 x i32> zeroinitializer
  %add = add <16 x i8> %splat.splat.i, %v
  ret <16 x i8> %add
; CHECK-LABEL: splat_8_plus
; CHECK-NOT: xxswapd
; CHECK: vspltb {{[0-9]+}}, {{[0-9]+}}, 7
; CHECK: blr
; CHECK-PWR8-LABEL: splat_8_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: vspltb {{[0-9]+}}, {{[0-9]+}}, 7
; CHECK-PWR8: blr
}

; 8 Bit Unsigned Version of the test.
; Function Attrs: norecurse nounwind readnone
define <16 x i8> @splat_u8_plus(<16 x i8> %v, i8 zeroext %c) local_unnamed_addr {
entry:
  %splat.splatinsert.i = insertelement <16 x i8> undef, i8 %c, i32 0
  %splat.splat.i = shufflevector <16 x i8> %splat.splatinsert.i, <16 x i8> undef, <16 x i32> zeroinitializer
  %add = add <16 x i8> %splat.splat.i, %v
  ret <16 x i8> %add
; CHECK-LABEL: splat_u8_plus
; CHECK-NOT: xxswapd
; CHECK: vspltb {{[0-9]+}}, {{[0-9]+}}, 7
; CHECK: blr
; CHECK-PWR8-LABEL: splat_u8_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: vspltb {{[0-9]+}}, {{[0-9]+}}, 7
; CHECK-PWR8: blr
}

; 16 Bit Signed Version of the test.
; Function Attrs: norecurse nounwind readnone
define <8 x i16> @splat_16_plus(<8 x i16> %v, i16 signext %c) local_unnamed_addr {
entry:
  %0 = shl i16 %c, 8
  %conv.i = ashr exact i16 %0, 8
  %splat.splatinsert.i = insertelement <8 x i16> undef, i16 %conv.i, i32 0
  %splat.splat.i = shufflevector <8 x i16> %splat.splatinsert.i, <8 x i16> undef, <8 x i32> zeroinitializer
  %add = add <8 x i16> %splat.splat.i, %v
  ret <8 x i16> %add
; CHECK-LABEL: splat_16_plus
; CHECK-NOT: xxswapd
; CHECK: vsplth {{[0-9]+}}, {{[0-9]+}}, 3
; CHECK: blr
; CHECK-PWR8-LABEL: splat_16_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: vsplth {{[0-9]+}}, {{[0-9]+}}, 3
; CHECK-PWR8: blr
}

; 16 Bit Unsigned Version of the test.
; Function Attrs: norecurse nounwind readnone
define <8 x i16> @splat_u16_plus(<8 x i16> %v, i16 zeroext %c) local_unnamed_addr {
entry:
  %0 = shl i16 %c, 8
  %conv.i = ashr exact i16 %0, 8
  %splat.splatinsert.i = insertelement <8 x i16> undef, i16 %conv.i, i32 0
  %splat.splat.i = shufflevector <8 x i16> %splat.splatinsert.i, <8 x i16> undef, <8 x i32> zeroinitializer
  %add = add <8 x i16> %splat.splat.i, %v
  ret <8 x i16> %add
; CHECK-LABEL: splat_u16_plus
; CHECK-NOT: xxswapd
; CHECK: vsplth {{[0-9]+}}, {{[0-9]+}}, 3
; CHECK: blr
; CHECK-PWR8-LABEL: splat_u16_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: vsplth {{[0-9]+}}, {{[0-9]+}}, 3
; CHECK-PWR8: blr
}

; 32 Bit Signed Version of the test.
; The 32 bit examples work differently than the 8 and 16 bit versions of the
;  test. On Power 9 we have the mtvsrws instruction that does both the move to
;  register and the splat so it does not really test the newly implemented code.
; On Power 9 for the 32 bit case we don't need the new simplification. It is
;  just here for completeness.
; Function Attrs: norecurse nounwind readnone
define <4 x i32> @splat_32_plus(<4 x i32> %v, i32 signext %c) local_unnamed_addr {
entry:
  %sext = shl i32 %c, 24
  %conv.i = ashr exact i32 %sext, 24
  %splat.splatinsert.i = insertelement <4 x i32> undef, i32 %conv.i, i32 0
  %splat.splat.i = shufflevector <4 x i32> %splat.splatinsert.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %add = add <4 x i32> %splat.splat.i, %v
  ret <4 x i32> %add
; CHECK-LABEL: splat_32_plus
; CHECK-NOT: xxswapd
; CHECK: mtvsrws {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
; CHECK-PWR8-LABEL: splat_32_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: xxspltw {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK-PWR8: blr
}

; 32 Bit Unsigned Version of the test.
; The 32 bit examples work differently than the 8 and 16 bit versions of the
;  test. On Power 9 we have the mtvsrws instruction that does both the move to
;  register and the splat so it does not really test the newly implemented code.
; On Power 9 for the 32 bit case we don't need the new simplification. It is
;  just here for completeness.
; Function Attrs: norecurse nounwind readnone
define <4 x i32> @splat_u32_plus(<4 x i32> %v, i32 zeroext %c) local_unnamed_addr {
entry:
  %sext = shl i32 %c, 24
  %conv.i = ashr exact i32 %sext, 24
  %splat.splatinsert.i = insertelement <4 x i32> undef, i32 %conv.i, i32 0
  %splat.splat.i = shufflevector <4 x i32> %splat.splatinsert.i, <4 x i32> undef, <4 x i32> zeroinitializer
  %add = add <4 x i32> %splat.splat.i, %v
  ret <4 x i32> %add
; CHECK-LABEL: splat_u32_plus
; CHECK-NOT: xxswapd
; CHECK: mtvsrws {{[0-9]+}}, {{[0-9]+}}
; CHECK: blr
; CHECK-PWR8-LABEL: splat_u32_plus
; CHECK-PWR8-NOT: xxswapd
; CHECK-PWR8: xxspltw {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK-PWR8: blr
}

