; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s

@var_8bit = dso_local global i8 0
@var_16bit = dso_local global i16 0
@var_32bit = dso_local global i32 0
@var_64bit = dso_local global i64 0

@var_float = dso_local global float 0.0
@var_double = dso_local global double 0.0

@varptr = dso_local global ptr null

define dso_local void @ldst_8bit() {
; CHECK-LABEL: ldst_8bit:

; No architectural support for loads to 16-bit or 8-bit since we
; promote i8 during lowering.
  %addr_8bit = load ptr, ptr @varptr

; match a sign-extending load 8-bit -> 32-bit
   %addr_sext32 = getelementptr i8, ptr %addr_8bit, i64 -256
   %val8_sext32 = load volatile i8, ptr %addr_sext32
   %val32_signed = sext i8 %val8_sext32 to i32
   store volatile i32 %val32_signed, ptr @var_32bit
; CHECK: ldursb {{w[0-9]+}}, [{{x[0-9]+}}, #-256]

; match a zero-extending load volatile 8-bit -> 32-bit
  %addr_zext32 = getelementptr i8, ptr %addr_8bit, i64 -12
  %val8_zext32 = load volatile i8, ptr %addr_zext32
  %val32_unsigned = zext i8 %val8_zext32 to i32
  store volatile i32 %val32_unsigned, ptr @var_32bit
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-12]

; match an any-extending load volatile 8-bit -> 32-bit
  %addr_anyext = getelementptr i8, ptr %addr_8bit, i64 -1
  %val8_anyext = load volatile i8, ptr %addr_anyext
  %newval8 = add i8 %val8_anyext, 1
  store volatile i8 %newval8, ptr @var_8bit
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-1]

; match a sign-extending load volatile 8-bit -> 64-bit
  %addr_sext64 = getelementptr i8, ptr %addr_8bit, i64 -5
  %val8_sext64 = load volatile i8, ptr %addr_sext64
  %val64_signed = sext i8 %val8_sext64 to i64
  store volatile i64 %val64_signed, ptr @var_64bit
; CHECK: ldursb {{x[0-9]+}}, [{{x[0-9]+}}, #-5]

; match a zero-extending load volatile 8-bit -> 64-bit.
; This uses the fact that ldrb w0, [x0] will zero out the high 32-bits
; of x0 so it's identical to load volatileing to 32-bits.
  %addr_zext64 = getelementptr i8, ptr %addr_8bit, i64 -9
  %val8_zext64 = load volatile i8, ptr %addr_zext64
  %val64_unsigned = zext i8 %val8_zext64 to i64
  store volatile i64 %val64_unsigned, ptr @var_64bit
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-9]

; truncating store volatile 32-bits to 8-bits
  %addr_trunc32 = getelementptr i8, ptr %addr_8bit, i64 -256
  %val32 = load volatile i32, ptr @var_32bit
  %val8_trunc32 = trunc i32 %val32 to i8
  store volatile i8 %val8_trunc32, ptr %addr_trunc32
; CHECK: sturb {{w[0-9]+}}, [{{x[0-9]+}}, #-256]

; truncating store volatile 64-bits to 8-bits
  %addr_trunc64 = getelementptr i8, ptr %addr_8bit, i64 -1
  %val64 = load volatile i64, ptr @var_64bit
  %val8_trunc64 = trunc i64 %val64 to i8
  store volatile i8 %val8_trunc64, ptr %addr_trunc64
; CHECK: sturb {{w[0-9]+}}, [{{x[0-9]+}}, #-1]

   ret void
}

define dso_local void @ldst_16bit() {
; CHECK-LABEL: ldst_16bit:

; No architectural support for loads to 16-bit or 16-bit since we
; promote i16 during lowering.
  %addr_8bit = load ptr, ptr @varptr

; match a sign-extending load 16-bit -> 32-bit
   %addr8_sext32 = getelementptr i8, ptr %addr_8bit, i64 -256
   %val16_sext32 = load volatile i16, ptr %addr8_sext32
   %val32_signed = sext i16 %val16_sext32 to i32
   store volatile i32 %val32_signed, ptr @var_32bit
; CHECK: ldursh {{w[0-9]+}}, [{{x[0-9]+}}, #-256]

; match a zero-extending load volatile 16-bit -> 32-bit. With offset that would be unaligned.
  %addr8_zext32 = getelementptr i8, ptr %addr_8bit, i64 15
  %val16_zext32 = load volatile i16, ptr %addr8_zext32
  %val32_unsigned = zext i16 %val16_zext32 to i32
  store volatile i32 %val32_unsigned, ptr @var_32bit
; CHECK: ldurh {{w[0-9]+}}, [{{x[0-9]+}}, #15]

; match an any-extending load volatile 16-bit -> 32-bit
  %addr8_anyext = getelementptr i8, ptr %addr_8bit, i64 -1
  %val16_anyext = load volatile i16, ptr %addr8_anyext
  %newval16 = add i16 %val16_anyext, 1
  store volatile i16 %newval16, ptr @var_16bit
; CHECK: ldurh {{w[0-9]+}}, [{{x[0-9]+}}, #-1]

; match a sign-extending load volatile 16-bit -> 64-bit
  %addr8_sext64 = getelementptr i8, ptr %addr_8bit, i64 -5
  %val16_sext64 = load volatile i16, ptr %addr8_sext64
  %val64_signed = sext i16 %val16_sext64 to i64
  store volatile i64 %val64_signed, ptr @var_64bit
; CHECK: ldursh {{x[0-9]+}}, [{{x[0-9]+}}, #-5]

; match a zero-extending load volatile 16-bit -> 64-bit.
; This uses the fact that ldrb w0, [x0] will zero out the high 32-bits
; of x0 so it's identical to load volatileing to 32-bits.
  %addr8_zext64 = getelementptr i8, ptr %addr_8bit, i64 9
  %val16_zext64 = load volatile i16, ptr %addr8_zext64
  %val64_unsigned = zext i16 %val16_zext64 to i64
  store volatile i64 %val64_unsigned, ptr @var_64bit
; CHECK: ldurh {{w[0-9]+}}, [{{x[0-9]+}}, #9]

; truncating store volatile 32-bits to 16-bits
  %addr8_trunc32 = getelementptr i8, ptr %addr_8bit, i64 -256
  %val32 = load volatile i32, ptr @var_32bit
  %val16_trunc32 = trunc i32 %val32 to i16
  store volatile i16 %val16_trunc32, ptr %addr8_trunc32
; CHECK: sturh {{w[0-9]+}}, [{{x[0-9]+}}, #-256]

; truncating store volatile 64-bits to 16-bits
  %addr8_trunc64 = getelementptr i8, ptr %addr_8bit, i64 -1
  %val64 = load volatile i64, ptr @var_64bit
  %val16_trunc64 = trunc i64 %val64 to i16
  store volatile i16 %val16_trunc64, ptr %addr8_trunc64
; CHECK: sturh {{w[0-9]+}}, [{{x[0-9]+}}, #-1]

   ret void
}

define dso_local void @ldst_32bit() {
; CHECK-LABEL: ldst_32bit:

  %addr_8bit = load ptr, ptr @varptr

; Straight 32-bit load/store
  %addr32_8_noext = getelementptr i8, ptr %addr_8bit, i64 1
  %val32_noext = load volatile i32, ptr %addr32_8_noext
  store volatile i32 %val32_noext, ptr %addr32_8_noext
; CHECK: ldur {{w[0-9]+}}, [{{x[0-9]+}}, #1]
; CHECK: stur {{w[0-9]+}}, [{{x[0-9]+}}, #1]

; Zero-extension to 64-bits
  %addr32_8_zext = getelementptr i8, ptr %addr_8bit, i64 -256
  %val32_zext = load volatile i32, ptr %addr32_8_zext
  %val64_unsigned = zext i32 %val32_zext to i64
  store volatile i64 %val64_unsigned, ptr @var_64bit
; CHECK: ldur {{w[0-9]+}}, [{{x[0-9]+}}, #-256]
; CHECK: str {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_64bit]

; Sign-extension to 64-bits
  %addr32_8_sext = getelementptr i8, ptr %addr_8bit, i64 -12
  %val32_sext = load volatile i32, ptr %addr32_8_sext
  %val64_signed = sext i32 %val32_sext to i64
  store volatile i64 %val64_signed, ptr @var_64bit
; CHECK: ldursw {{x[0-9]+}}, [{{x[0-9]+}}, #-12]
; CHECK: str {{x[0-9]+}}, [{{x[0-9]+}}, {{#?}}:lo12:var_64bit]

; Truncation from 64-bits
  %addr64_8_trunc = getelementptr i8, ptr %addr_8bit, i64 255
  %addr32_8_trunc = getelementptr i8, ptr %addr_8bit, i64 -20

  %val64_trunc = load volatile i64, ptr %addr64_8_trunc
  %val32_trunc = trunc i64 %val64_trunc to i32
  store volatile i32 %val32_trunc, ptr %addr32_8_trunc
; CHECK: ldur {{x[0-9]+}}, [{{x[0-9]+}}, #255]
; CHECK: stur {{w[0-9]+}}, [{{x[0-9]+}}, #-20]

  ret void
}

define dso_local void @ldst_float() {
; CHECK-LABEL: ldst_float:

  %addr_8bit = load ptr, ptr @varptr
  %addrfp_8 = getelementptr i8, ptr %addr_8bit, i64 -5

  %valfp = load volatile float, ptr %addrfp_8
; CHECK: ldur {{s[0-9]+}}, [{{x[0-9]+}}, #-5]
; CHECK-NOFP-NOT: ldur {{s[0-9]+}},

  store volatile float %valfp, ptr %addrfp_8
; CHECK: stur {{s[0-9]+}}, [{{x[0-9]+}}, #-5]
; CHECK-NOFP-NOT: stur {{s[0-9]+}},

  ret void
}

define dso_local void @ldst_double() {
; CHECK-LABEL: ldst_double:

  %addr_8bit = load ptr, ptr @varptr
  %addrfp_8 = getelementptr i8, ptr %addr_8bit, i64 4

  %valfp = load volatile double, ptr %addrfp_8
; CHECK: ldur {{d[0-9]+}}, [{{x[0-9]+}}, #4]
; CHECK-NOFP-NOT: ldur {{d[0-9]+}},

  store volatile double %valfp, ptr %addrfp_8
; CHECK: stur {{d[0-9]+}}, [{{x[0-9]+}}, #4]
; CHECK-NOFP-NOT: stur {{d[0-9]+}},

   ret void
}
