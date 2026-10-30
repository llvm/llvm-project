; sitofp iN->bfloat is decomposed via float intermediate + bit shuffling: the
; magnitude is rounded to bfloat's 8 significant bits (to nearest even) in the
; integer domain, converted exactly, and the sign set in the bits. Truncating
; an f32 conversion instead would round toward zero (found on an M2 Max).
; fptosi from half or bfloat goes through f32 (Metal reads a bfloat operand
; as half; its compiler can hang on half -> i64).
; sitofp i8->float is widened through i32.
; RUN: llc -mtriple=air %s -o - | FileCheck %s
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s --check-prefix=PORTABLE

; CHECK-LABEL: define bfloat @to_bf16
; CHECK: [[NEG:%.*]] = icmp slt i32 %x, 0
; CHECK: [[MAG:%.*]] = select i1 [[NEG]]
; CHECK: [[UP:%.*]] = zext i1 {{.*}} to i32
; CHECK: [[QR:%.*]] = add i32 {{%.*}}, [[UP]]
; CHECK: [[QF:%.*]] = uitofp i32 [[QR]] to float
; CHECK: %to_f32 = fmul float [[QF]],
; CHECK: %f32_bits = bitcast float %to_f32 to i32
; CHECK: [[SIGN:%.*]] = select i1 [[NEG]], i32 -2147483648, i32 0
; CHECK: [[BITS:%.*]] = or i32 %f32_bits, [[SIGN]]
; CHECK: lshr i32 [[BITS]], 16
; CHECK: trunc i32 {{.*}} to i16
; CHECK: bitcast i16 {{.*}} to bfloat


; CHECK-LABEL: define float @to_f32_from_i8
; CHECK: sext i8 %x to i32
; CHECK: sitofp i32 {{.*}} to float

; CHECK-LABEL: define i64 @half_to_i64
; CHECK: [[W:%.*]] = fpext half {{.*}} to float
; CHECK: fptosi float [[W]] to i64
; CHECK-LABEL: define i8 @bf16_to_u8
; CHECK-NOT: fptoui bfloat
; CHECK: fptoui float {{.*}} to i8
; PORTABLE-LABEL: define i64 @half_to_i64
; PORTABLE: fptosi half {{%.*}} to i64
; PORTABLE-LABEL: define i8 @bf16_to_u8
; PORTABLE: fptoui bfloat {{%.*}} to i8

define bfloat @to_bf16(i32 %x) {
  %r = sitofp i32 %x to bfloat
  ret bfloat %r
}

define float @to_f32_from_i8(i8 %x) {
  %r = sitofp i8 %x to float
  ret float %r
}

define i64 @half_to_i64(half %x) {
  %r = fptosi half %x to i64
  ret i64 %r
}

define i8 @bf16_to_u8(bfloat %x) {
  %r = fptoui bfloat %x to i8
  ret i8 %r
}
