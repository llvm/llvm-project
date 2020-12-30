; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s

@var_8bit = dso_local global i8 0
@var_16bit = dso_local global i16 0
@var_32bit = dso_local global i32 0
@var_64bit = dso_local global i64 0

@var_float = dso_local global float 0.0
@var_double = dso_local global double 0.0

define dso_local void @ldst_8bit() {
; CHECK-LABEL: ldst_8bit:

; No architectural support for loads to 16-bit or 8-bit since we
; promote i8 during lowering.

; match a sign-extending load 8-bit -> 32-bit
   %val8_sext32 = load volatile i8, i8* @var_8bit
   %val32_signed = sext i8 %val8_sext32 to i32
   store volatile i32 %val32_signed, i32* @var_32bit
; CHECK: adrp {{x[0-9]+}}, var_8bit
; CHECK: ldrsb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; match a zero-extending load volatile 8-bit -> 32-bit
  %val8_zext32 = load volatile i8, i8* @var_8bit
  %val32_unsigned = zext i8 %val8_zext32 to i32
  store volatile i32 %val32_unsigned, i32* @var_32bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; match an any-extending load volatile 8-bit -> 32-bit
  %val8_anyext = load volatile i8, i8* @var_8bit
  %newval8 = add i8 %val8_anyext, 1
  store volatile i8 %newval8, i8* @var_8bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; match a sign-extending load volatile 8-bit -> 64-bit
  %val8_sext64 = load volatile i8, i8* @var_8bit
  %val64_signed = sext i8 %val8_sext64 to i64
  store volatile i64 %val64_signed, i64* @var_64bit
; CHECK: ldrsb {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; match a zero-extending load volatile 8-bit -> 64-bit.
; This uses the fact that ldrb w0, [x0] will zero out the high 32-bits
; of x0 so it's identical to load volatileing to 32-bits.
  %val8_zext64 = load volatile i8, i8* @var_8bit
  %val64_unsigned = zext i8 %val8_zext64 to i64
  store volatile i64 %val64_unsigned, i64* @var_64bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; truncating store volatile 32-bits to 8-bits
  %val32 = load volatile i32, i32* @var_32bit
  %val8_trunc32 = trunc i32 %val32 to i8
  store volatile i8 %val8_trunc32, i8* @var_8bit
; CHECK: strb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

; truncating store volatile 64-bits to 8-bits
  %val64 = load volatile i64, i64* @var_64bit
  %val8_trunc64 = trunc i64 %val64 to i8
  store volatile i8 %val8_trunc64, i8* @var_8bit
; CHECK: strb {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_8bit]

   ret void
}

define dso_local void @ldst_16bit() {
; CHECK-LABEL: ldst_16bit:

; No architectural support for load volatiles to 16-bit promote i16 during
; lowering.

; match a sign-extending load volatile 16-bit -> 32-bit
  %val16_sext32 = load volatile i16, i16* @var_16bit
  %val32_signed = sext i16 %val16_sext32 to i32
  store volatile i32 %val32_signed, i32* @var_32bit
; CHECK: adrp {{x[0-9]+}}, var_16bit
; CHECK: ldrsh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; match a zero-extending load volatile 16-bit -> 32-bit
  %val16_zext32 = load volatile i16, i16* @var_16bit
  %val32_unsigned = zext i16 %val16_zext32 to i32
  store volatile i32 %val32_unsigned, i32* @var_32bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; match an any-extending load volatile 16-bit -> 32-bit
  %val16_anyext = load volatile i16, i16* @var_16bit
  %newval16 = add i16 %val16_anyext, 1
  store volatile i16 %newval16, i16* @var_16bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; match a sign-extending load volatile 16-bit -> 64-bit
  %val16_sext64 = load volatile i16, i16* @var_16bit
  %val64_signed = sext i16 %val16_sext64 to i64
  store volatile i64 %val64_signed, i64* @var_64bit
; CHECK: ldrsh {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; match a zero-extending load volatile 16-bit -> 64-bit.
; This uses the fact that ldrb w0, [x0] will zero out the high 32-bits
; of x0 so it's identical to load volatileing to 32-bits.
  %val16_zext64 = load volatile i16, i16* @var_16bit
  %val64_unsigned = zext i16 %val16_zext64 to i64
  store volatile i64 %val64_unsigned, i64* @var_64bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; truncating store volatile 32-bits to 16-bits
  %val32 = load volatile i32, i32* @var_32bit
  %val16_trunc32 = trunc i32 %val32 to i16
  store volatile i16 %val16_trunc32, i16* @var_16bit
; CHECK: strh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

; truncating store volatile 64-bits to 16-bits
  %val64 = load volatile i64, i64* @var_64bit
  %val16_trunc64 = trunc i64 %val64 to i16
  store volatile i16 %val16_trunc64, i16* @var_16bit
; CHECK: strh {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_16bit]

  ret void
}

define dso_local void @ldst_32bit() {
; CHECK-LABEL: ldst_32bit:

; Straight 32-bit load/store
  %val32_noext = load volatile i32, i32* @var_32bit
  store volatile i32 %val32_noext, i32* @var_32bit
; CHECK: adrp {{x[0-9]+}}, var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_32bit]
; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_32bit]

; Zero-extension to 64-bits
  %val32_zext = load volatile i32, i32* @var_32bit
  %val64_unsigned = zext i32 %val32_zext to i64
  store volatile i64 %val64_unsigned, i64* @var_64bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_32bit]
; CHECK: str {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_64bit]

; Sign-extension to 64-bits
  %val32_sext = load volatile i32, i32* @var_32bit
  %val64_signed = sext i32 %val32_sext to i64
  store volatile i64 %val64_signed, i64* @var_64bit
; CHECK: ldrsw {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_32bit]
; CHECK: str {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_64bit]

; Truncation from 64-bits
  %val64_trunc = load volatile i64, i64* @var_64bit
  %val32_trunc = trunc i64 %val64_trunc to i32
  store volatile i32 %val32_trunc, i32* @var_32bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_64bit]
; CHECK: str {{w[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_32bit]

  ret void
}

@arr8 = dso_local global i8* null
@arr16 = dso_local global i16* null
@arr32 = dso_local global i32* null
@arr64 = dso_local global i64* null

; Now check that our selection copes with accesses more complex than a
; single symbol. Permitted offsets should be folded into the loads and
; stores. Since all forms use the same Operand it's only necessary to
; check the various access-sizes involved.

define dso_local void @ldst_complex_offsets() {
; CHECK: ldst_complex_offsets
  %arr8_addr = load volatile i8*, i8** @arr8
; CHECK: adrp {{x[0-9]+}}, arr8
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:arr8]

  %arr8_sub1_addr = getelementptr i8, i8* %arr8_addr, i64 1
  %arr8_sub1 = load volatile i8, i8* %arr8_sub1_addr
  store volatile i8 %arr8_sub1, i8* @var_8bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, #1]

  %arr8_sub4095_addr = getelementptr i8, i8* %arr8_addr, i64 4095
  %arr8_sub4095 = load volatile i8, i8* %arr8_sub4095_addr
  store volatile i8 %arr8_sub4095, i8* @var_8bit
; CHECK: ldrb {{w[0-9]+}}, [{{x[0-9]+}}, #4095]


  %arr16_addr = load volatile i16*, i16** @arr16
; CHECK: adrp {{x[0-9]+}}, arr16
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:arr16]

  %arr16_sub1_addr = getelementptr i16, i16* %arr16_addr, i64 1
  %arr16_sub1 = load volatile i16, i16* %arr16_sub1_addr
  store volatile i16 %arr16_sub1, i16* @var_16bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, #2]

  %arr16_sub4095_addr = getelementptr i16, i16* %arr16_addr, i64 4095
  %arr16_sub4095 = load volatile i16, i16* %arr16_sub4095_addr
  store volatile i16 %arr16_sub4095, i16* @var_16bit
; CHECK: ldrh {{w[0-9]+}}, [{{x[0-9]+}}, #8190]


  %arr32_addr = load volatile i32*, i32** @arr32
; CHECK: adrp {{x[0-9]+}}, arr32
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:arr32]

  %arr32_sub1_addr = getelementptr i32, i32* %arr32_addr, i64 1
  %arr32_sub1 = load volatile i32, i32* %arr32_sub1_addr
  store volatile i32 %arr32_sub1, i32* @var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, #4]

  %arr32_sub4095_addr = getelementptr i32, i32* %arr32_addr, i64 4095
  %arr32_sub4095 = load volatile i32, i32* %arr32_sub4095_addr
  store volatile i32 %arr32_sub4095, i32* @var_32bit
; CHECK: ldr {{w[0-9]+}}, [{{x[0-9]+}}, #16380]


  %arr64_addr = load volatile i64*, i64** @arr64
; CHECK: adrp {{x[0-9]+}}, arr64
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:arr64]

  %arr64_sub1_addr = getelementptr i64, i64* %arr64_addr, i64 1
  %arr64_sub1 = load volatile i64, i64* %arr64_sub1_addr
  store volatile i64 %arr64_sub1, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, #8]

  %arr64_sub4095_addr = getelementptr i64, i64* %arr64_addr, i64 4095
  %arr64_sub4095 = load volatile i64, i64* %arr64_sub4095_addr
  store volatile i64 %arr64_sub4095, i64* @var_64bit
; CHECK: ldr {{x[0-9]+}}, [{{x[0-9]+}}, #32760]

  ret void
}

define dso_local void @ldst_float() {
; CHECK-LABEL: ldst_float:

   %valfp = load volatile float, float* @var_float
; CHECK: adrp {{x[0-9]+}}, var_float
; CHECK: ldr {{s[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_float]
; CHECK-NOFP-NOT: ldr {{s[0-9]+}},

  store volatile float %valfp, float* @var_float
; CHECK: str {{s[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_float]
; CHECK-NOFP-NOT: str {{s[0-9]+}},

   ret void
}

define dso_local void @ldst_double() {
; CHECK-LABEL: ldst_double:

   %valfp = load volatile double, double* @var_double
; CHECK: adrp {{x[0-9]+}}, var_double
; CHECK: ldr {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_double]
; CHECK-NOFP-NOT: ldr {{d[0-9]+}},

  store volatile double %valfp, double* @var_double
; CHECK: str {{d[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_double]
; CHECK-NOFP-NOT: str {{d[0-9]+}},

   ret void
}
