; Test that the v1024i16 mulhu/mulhs patterns
;   zext -> mul -> lshr -> trunc   (unsigned high multiply)
;   sext -> mul -> ashr -> trunc   (signed   high multiply)
; are recognised and lowered to the HLX widening-multiply instructions
; LX_mpy_uw_uh and LX_mpy_w_h respectively, with the high half extracted
; via vsub_hi.  No explicit shift or truncate should appear in the output.
;
; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx \
; RUN:   -stop-after="hexagon-isel" -verify-machineinstrs < %s | FileCheck %s

target triple = "hexagon"

; Unsigned high multiply: zext -> mul -> lshr(16) -> trunc
; Expected lowering: LX_mpy_uw_uh (widening unsigned 16x16->32) + vsub_hi
;
; CHECK-LABEL: name: test_mulhu
; CHECK: %[[P:[0-9]+]]:hlxllr = LX_mpy_uw_uh
; CHECK: hlxlr = COPY %[[P]].vsub_hi
; CHECK-NOT: LX_shr
define <1024 x i16> @test_mulhu(<1024 x i16> %a, <1024 x i16> %b) {
  %wa = zext <1024 x i16> %a to <1024 x i32>
  %wb = zext <1024 x i16> %b to <1024 x i32>
  %p  = mul  <1024 x i32> %wa, %wb
  %s  = lshr <1024 x i32> %p, splat (i32 16)
  %r  = trunc <1024 x i32> %s to <1024 x i16>
  ret <1024 x i16> %r
}

; Signed high multiply: sext -> mul -> ashr(16) -> trunc
; Expected lowering: LX_mpy_w_h (widening signed 16x16->32) + vsub_hi
;
; CHECK-LABEL: name: test_mulhs
; CHECK: %[[Q:[0-9]+]]:hlxllr = LX_mpy_w_h
; CHECK: hlxlr = COPY %[[Q]].vsub_hi
; CHECK-NOT: LX_shr
define <1024 x i16> @test_mulhs(<1024 x i16> %a, <1024 x i16> %b) {
  %wa = sext <1024 x i16> %a to <1024 x i32>
  %wb = sext <1024 x i16> %b to <1024 x i32>
  %p  = mul  <1024 x i32> %wa, %wb
  %s  = ashr <1024 x i32> %p, splat (i32 16)
  %r  = trunc <1024 x i32> %s to <1024 x i16>
  ret <1024 x i16> %r
}
