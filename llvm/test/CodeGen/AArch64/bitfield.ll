; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@var32 = dso_local global i32 0
@var64 = dso_local global i64 0

define dso_local void @test_extendb32(i8 %var) {
; CHECK-LABEL: test_extendb32:

  %sxt32 = sext i8 %var to i32
  store volatile i32 %sxt32, i32* @var32
; CHECK: sxtb {{w[0-9]+}}, {{w[0-9]+}}

; N.b. this doesn't actually produce a bitfield instruction at the
; moment, but it's still a good test to have and the semantics are
; correct.
  %uxt32 = zext i8 %var to i32
  store volatile i32 %uxt32, i32* @var32
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, #0xff
  ret void
}

define dso_local void @test_extendb64(i8 %var) {
; CHECK-LABEL: test_extendb64:

  %sxt64 = sext i8 %var to i64
  store volatile i64 %sxt64, i64* @var64
; CHECK: sxtb {{x[0-9]+}}, {{w[0-9]+}}

; N.b. this doesn't actually produce a bitfield instruction at the
; moment, but it's still a good test to have and the semantics are
; correct.
  %uxt64 = zext i8 %var to i64
  store volatile i64 %uxt64, i64* @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xff
  ret void
}

define dso_local void @test_extendh32(i16 %var) {
; CHECK-LABEL: test_extendh32:

  %sxt32 = sext i16 %var to i32
  store volatile i32 %sxt32, i32* @var32
; CHECK: sxth {{w[0-9]+}}, {{w[0-9]+}}

; N.b. this doesn't actually produce a bitfield instruction at the
; moment, but it's still a good test to have and the semantics are
; correct.
  %uxt32 = zext i16 %var to i32
  store volatile i32 %uxt32, i32* @var32
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, #0xffff
  ret void
}

define dso_local void @test_extendh64(i16 %var) {
; CHECK-LABEL: test_extendh64:

  %sxt64 = sext i16 %var to i64
  store volatile i64 %sxt64, i64* @var64
; CHECK: sxth {{x[0-9]+}}, {{w[0-9]+}}

; N.b. this doesn't actually produce a bitfield instruction at the
; moment, but it's still a good test to have and the semantics are
; correct.
  %uxt64 = zext i16 %var to i64
  store volatile i64 %uxt64, i64* @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xffff
  ret void
}

define dso_local void @test_extendw(i32 %var) {
; CHECK-LABEL: test_extendw:

  %sxt64 = sext i32 %var to i64
  store volatile i64 %sxt64, i64* @var64
; CHECK: sxtw {{x[0-9]+}}, {{w[0-9]+}}

  %uxt64 = zext i32 %var to i64
  store volatile i64 %uxt64, i64* @var64
; CHECK: mov {{w[0-9]+}}, w0
  ret void
}

define dso_local void @test_shifts(i32 %val32, i64 %val64) {
; CHECK-LABEL: test_shifts:

  %shift1 = ashr i32 %val32, 31
  store volatile i32 %shift1, i32* @var32
; CHECK: asr {{w[0-9]+}}, {{w[0-9]+}}, #31

  %shift2 = lshr i32 %val32, 8
  store volatile i32 %shift2, i32* @var32
; CHECK: lsr {{w[0-9]+}}, {{w[0-9]+}}, #8

  %shift3 = shl i32 %val32, 1
  store volatile i32 %shift3, i32* @var32
; CHECK: lsl {{w[0-9]+}}, {{w[0-9]+}}, #1

  %shift4 = ashr i64 %val64, 31
  store volatile i64 %shift4, i64* @var64
; CHECK: asr {{x[0-9]+}}, {{x[0-9]+}}, #31

  %shift5 = lshr i64 %val64, 8
  store volatile i64 %shift5, i64* @var64
; CHECK: lsr {{x[0-9]+}}, {{x[0-9]+}}, #8

  %shift6 = shl i64 %val64, 63
  store volatile i64 %shift6, i64* @var64
; CHECK: lsl {{x[0-9]+}}, {{x[0-9]+}}, #63

  %shift7 = ashr i64 %val64, 63
  store volatile i64 %shift7, i64* @var64
; CHECK: asr {{x[0-9]+}}, {{x[0-9]+}}, #63

  %shift8 = lshr i64 %val64, 63
  store volatile i64 %shift8, i64* @var64
; CHECK: lsr {{x[0-9]+}}, {{x[0-9]+}}, #63

  %shift9 = lshr i32 %val32, 31
  store volatile i32 %shift9, i32* @var32
; CHECK: lsr {{w[0-9]+}}, {{w[0-9]+}}, #31

  %shift10 = shl i32 %val32, 31
  store volatile i32 %shift10, i32* @var32
; CHECK: lsl {{w[0-9]+}}, {{w[0-9]+}}, #31

  ret void
}

; LLVM can produce in-register extensions taking place entirely with
; 64-bit registers too.
define dso_local void @test_sext_inreg_64(i64 %in) {
; CHECK-LABEL: test_sext_inreg_64:

; i1 doesn't have an official alias, but crops up and is handled by
; the bitfield ops.
  %trunc_i1 = trunc i64 %in to i1
  %sext_i1 = sext i1 %trunc_i1 to i64
  store volatile i64 %sext_i1, i64* @var64
; CHECK: sbfx {{x[0-9]+}}, {{x[0-9]+}}, #0, #1

  %trunc_i8 = trunc i64 %in to i8
  %sext_i8 = sext i8 %trunc_i8 to i64
  store volatile i64 %sext_i8, i64* @var64
; CHECK: sxtb {{x[0-9]+}}, {{w[0-9]+}}

  %trunc_i16 = trunc i64 %in to i16
  %sext_i16 = sext i16 %trunc_i16 to i64
  store volatile i64 %sext_i16, i64* @var64
; CHECK: sxth {{x[0-9]+}}, {{w[0-9]+}}

  %trunc_i32 = trunc i64 %in to i32
  %sext_i32 = sext i32 %trunc_i32 to i64
  store volatile i64 %sext_i32, i64* @var64
; CHECK: sxtw {{x[0-9]+}}, {{w[0-9]+}}
  ret void
}

; These instructions don't actually select to official bitfield
; operations, but it's important that we select them somehow:
define dso_local void @test_zext_inreg_64(i64 %in) {
; CHECK-LABEL: test_zext_inreg_64:

  %trunc_i8 = trunc i64 %in to i8
  %zext_i8 = zext i8 %trunc_i8 to i64
  store volatile i64 %zext_i8, i64* @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xff

  %trunc_i16 = trunc i64 %in to i16
  %zext_i16 = zext i16 %trunc_i16 to i64
  store volatile i64 %zext_i16, i64* @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xffff

  %trunc_i32 = trunc i64 %in to i32
  %zext_i32 = zext i32 %trunc_i32 to i64
  store volatile i64 %zext_i32, i64* @var64
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, #0xffffffff

  ret void
}

define dso_local i64 @test_sext_inreg_from_32(i32 %in) {
; CHECK-LABEL: test_sext_inreg_from_32:

  %small = trunc i32 %in to i1
  %ext = sext i1 %small to i64

  ; Different registers are of course, possible, though suboptimal. This is
  ; making sure that a 64-bit "(sext_inreg (anyext GPR32), i1)" uses the 64-bit
  ; sbfx rather than just 32-bits.
; CHECK: sbfx x0, x0, #0, #1
  ret i64 %ext
}


define dso_local i32 @test_ubfx32(i32* %addr) {
; CHECK-LABEL: test_ubfx32:
; CHECK: ubfx {{w[0-9]+}}, {{w[0-9]+}}, #23, #3

   %fields = load i32, i32* %addr
   %shifted = lshr i32 %fields, 23
   %masked = and i32 %shifted, 7
   ret i32 %masked
}

define dso_local i64 @test_ubfx64(i64* %addr) {
; CHECK-LABEL: test_ubfx64:
; CHECK: ubfx {{x[0-9]+}}, {{x[0-9]+}}, #25, #10
   %fields = load i64, i64* %addr
   %shifted = lshr i64 %fields, 25
   %masked = and i64 %shifted, 1023
   ret i64 %masked
}

define dso_local i32 @test_sbfx32(i32* %addr) {
; CHECK-LABEL: test_sbfx32:
; CHECK: sbfx {{w[0-9]+}}, {{w[0-9]+}}, #6, #3

   %fields = load i32, i32* %addr
   %shifted = shl i32 %fields, 23
   %extended = ashr i32 %shifted, 29
   ret i32 %extended
}

define dso_local i64 @test_sbfx64(i64* %addr) {
; CHECK-LABEL: test_sbfx64:
; CHECK: sbfx {{x[0-9]+}}, {{x[0-9]+}}, #0, #63

   %fields = load i64, i64* %addr
   %shifted = shl i64 %fields, 1
   %extended = ashr i64 %shifted, 1
   ret i64 %extended
}

define i32 @test_ubfx_mask(i32 %lhs, i32 %rhs) {
; CHECK-LABEL: test_ubfx_mask:
; CHECK: lsr w0, w1, #20
  %mask = and i32 %lhs, 20
  %i7 = add i32 %mask, 1
  %i8 = xor i32 %lhs, 20
  %i9 = xor i32 %i8, %i7
  %i10 = and i32 %i9, 20
  %shift = lshr i32 %rhs, %i10
  %shift.masked = and i32 %shift, 65535
  ret i32 %shift.masked
}
