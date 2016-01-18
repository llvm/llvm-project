; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefix=CHECK

; First, a simple example from Clang. The registers could plausibly be
; different, but probably won't be.

%struct.foo = type { i8, [2 x i8], i8 }

define [1 x i64] @from_clang([1 x i64] %f.coerce, i32 %n) nounwind readnone {
; CHECK-LABEL: from_clang:
; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #3, #4

entry:
  %f.coerce.fca.0.extract = extractvalue [1 x i64] %f.coerce, 0
  %tmp.sroa.0.0.extract.trunc = trunc i64 %f.coerce.fca.0.extract to i32
  %bf.value = shl i32 %n, 3
  %0 = and i32 %bf.value, 120
  %f.sroa.0.0.insert.ext.masked = and i32 %tmp.sroa.0.0.extract.trunc, 135
  %1 = or i32 %f.sroa.0.0.insert.ext.masked, %0
  %f.sroa.0.0.extract.trunc = zext i32 %1 to i64
  %tmp1.sroa.1.1.insert.insert = and i64 %f.coerce.fca.0.extract, 4294967040
  %tmp1.sroa.0.0.insert.insert = or i64 %f.sroa.0.0.extract.trunc, %tmp1.sroa.1.1.insert.insert
  %.fca.0.insert = insertvalue [1 x i64] undef, i64 %tmp1.sroa.0.0.insert.insert, 0
  ret [1 x i64] %.fca.0.insert
}

define void @test_whole32(i32* %existing, i32* %new) {
; CHECK-LABEL: test_whole32:

; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #26, #5

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 2214592511 ; =0x83ffffff

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 26
  %newval_masked = and i32 %newval_shifted, 2080374784 ; = 0x7c000000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

define void @test_whole64(i64* %existing, i64* %new) {
; CHECK-LABEL: test_whole64:
; CHECK: bfi {{x[0-9]+}}, {{x[0-9]+}}, #26, #14
; CHECK-NOT: and
; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 18446742974265032703 ; = 0xffffff0003ffffffL

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 26
  %newval_masked = and i64 %newval_shifted, 1099444518912 ; = 0xfffc000000

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

define void @test_whole32_from64(i64* %existing, i64* %new) {
; CHECK-LABEL: test_whole32_from64:


; CHECK: bfxil {{x[0-9]+}}, {{x[0-9]+}}, #0, #16

; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 4294901760 ; = 0xffff0000

  %newval = load volatile i64, i64* %new
  %newval_masked = and i64 %newval, 65535 ; = 0xffff

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

define void @test_32bit_masked(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_masked:

; CHECK: and
; CHECK: bfi [[INSERT:w[0-9]+]], {{w[0-9]+}}, #3, #4

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 135 ; = 0x87

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 120 ; = 0x78

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

define void @test_64bit_masked(i64 *%existing, i64 *%new) {
; CHECK-LABEL: test_64bit_masked:
; CHECK: and
; CHECK: bfi [[INSERT:x[0-9]+]], {{x[0-9]+}}, #40, #8

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 1095216660480 ; = 0xff_0000_0000

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 40
  %newval_masked = and i64 %newval_shifted, 280375465082880 ; = 0xff00_0000_0000

  %combined = or i64 %newval_masked, %oldval_keep
  store volatile i64 %combined, i64* %existing

  ret void
}

; Mask is too complicated for literal ANDwwi, make sure other avenues are tried.
define void @test_32bit_complexmask(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_complexmask:

; CHECK: and
; CHECK: bfi {{w[0-9]+}}, {{w[0-9]+}}, #3, #4

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 647 ; = 0x287

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 120 ; = 0x278

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

; Neither mask is is a contiguous set of 1s. BFI can't be used
define void @test_32bit_badmask(i32 *%existing, i32 *%new) {
; CHECK-LABEL: test_32bit_badmask:
; CHECK-NOT: bfi
; CHECK-NOT: bfm
; CHECK: ret

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 135 ; = 0x87

  %newval = load volatile i32, i32* %new
  %newval_shifted = shl i32 %newval, 3
  %newval_masked = and i32 %newval_shifted, 632 ; = 0x278

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing

  ret void
}

; Ditto
define void @test_64bit_badmask(i64 *%existing, i64 *%new) {
; CHECK-LABEL: test_64bit_badmask:
; CHECK-NOT: bfi
; CHECK-NOT: bfm
; CHECK: ret

  %oldval = load volatile i64, i64* %existing
  %oldval_keep = and i64 %oldval, 135 ; = 0x87

  %newval = load volatile i64, i64* %new
  %newval_shifted = shl i64 %newval, 3
  %newval_masked = and i64 %newval_shifted, 664 ; = 0x278

  %combined = or i64 %oldval_keep, %newval_masked
  store volatile i64 %combined, i64* %existing

  ret void
}

; Bitfield insert where there's a left-over shr needed at the beginning
; (e.g. result of str.bf1 = str.bf2)
define void @test_32bit_with_shr(i32* %existing, i32* %new) {
; CHECK-LABEL: test_32bit_with_shr:

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 2214592511 ; =0x83ffffff

  %newval = load i32, i32* %new
  %newval_shifted = shl i32 %newval, 12
  %newval_masked = and i32 %newval_shifted, 2080374784 ; = 0x7c000000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing
; CHECK: lsr [[BIT:w[0-9]+]], {{w[0-9]+}}, #14
; CHECK: bfi {{w[0-9]+}}, [[BIT]], #26, #5

  ret void
}

; Bitfield insert where the second or operand is a better match to be folded into the BFM
define void @test_32bit_opnd1_better(i32* %existing, i32* %new) {
; CHECK-LABEL: test_32bit_opnd1_better:

  %oldval = load volatile i32, i32* %existing
  %oldval_keep = and i32 %oldval, 65535 ; 0x0000ffff

  %newval = load i32, i32* %new
  %newval_shifted = shl i32 %newval, 16
  %newval_masked = and i32 %newval_shifted, 16711680 ; 0x00ff0000

  %combined = or i32 %oldval_keep, %newval_masked
  store volatile i32 %combined, i32* %existing
; CHECK: and [[BIT:w[0-9]+]], {{w[0-9]+}}, #0xffff
; CHECK: bfi [[BIT]], {{w[0-9]+}}, #16, #8

  ret void
}
