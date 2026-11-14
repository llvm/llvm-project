; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx < %s | FileCheck %s

; Test that v1024i32 shift operations with a general (non-splat) v1024i32
; shift amount compile without crashing.  The HLX shift instructions take a
; v1024i16 shift amount; the fix extracts the low half via LX_lo.

; CHECK-LABEL: test_srl_v1024i32:
; CHECK: .uw = shr(
define <1024 x i32> @test_srl_v1024i32(<1024 x i32> %a, <1024 x i32> %b) #0 {
  %r = lshr <1024 x i32> %a, %b
  ret <1024 x i32> %r
}

; CHECK-LABEL: test_sra_v1024i32:
; CHECK: .w = shr(
define <1024 x i32> @test_sra_v1024i32(<1024 x i32> %a, <1024 x i32> %b) #0 {
  %r = ashr <1024 x i32> %a, %b
  ret <1024 x i32> %r
}

; CHECK-LABEL: test_shl_v1024i32:
; CHECK: = sub(
; CHECK: .w = shr(
define <1024 x i32> @test_shl_v1024i32(<1024 x i32> %a, <1024 x i32> %b) #0 {
  %r = shl <1024 x i32> %a, %b
  ret <1024 x i32> %r
}

attributes #0 = { "target-features"="+hlx,+v85" }
