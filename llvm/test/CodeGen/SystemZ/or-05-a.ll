; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; or-05.ll tests the multiclass RMWIByte for i8 types. 
; This test verifies the DAGToDag implementation for i16, i32, and i64 types,
; covering both 12-bit unsigned (OI) and 20-bit signed (OIY) displacements.

; There are certain liomitations:
; 1. Displacement Range Limitations:
;    - Minimum: For multi-byte types, the absolute minimum 20-bit displacement 
;      (-524288) cannot be folded. A GEP offset < -524288 triggers AGFI 
;      legalization in the address matcher before folding can occur.
;    - Maximum: Similarly, the absolute maximum displacement (524287) cannot 
;      be reached via a starting GEP offset of 524287, as the Big-Endian 
;      adjustment (+1, +3, or +7) would overflow the 20-bit signed range.
;
; 2. Negative Constant Folding Failures:
;    Negative constants fail to fold for multi-byte types because they are
;    sign-extended (e.g., i16 -2 is 0xFFFE). These values have more than 8
;    active bits, whereas NI/OI/XI instructions only accept an 8-bit immediate.

; Check i16 (short): Offset + 1.
define void @or_i16_oi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: or_i16_oi_low_unsigned:
; CHECK: oi 1(%r2), 1
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check i32 (int): Offset + 3.
define void @or_i32_oi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: or_i32_oi_low_unsigned:
; CHECK: oi 3(%r2), 1
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check i64 (int): Offset + 7.
define void @or_i64_oi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: or_i64_oi_low_unsigned:
; CHECK: oi 7(%r2), 1
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check High: 128 (0x80) - The "signed bit" of a byte.
; Verifies that patterns with the 8th bit set still fold.
define void @or_i32_oi_boundary_128(ptr %ptr) {
; CHECK-LABEL: or_i32_oi_boundary_128:
; CHECK: oi 3(%r2), 128
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 128
  store i32 %or, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
define void @or_i32_oi_max_byte(ptr %ptr) {
; CHECK-LABEL: or_i32_oi_max_byte:
; CHECK: oi 3(%r2), 255
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 255
  store i32 %or, ptr %ptr
  ret void
}

; Check bit-width constraint with negative signed value.
; This test case with -2 (0xFF...FE) fails to fold because the immediate value 
; has more than 8 active bits (due to sign extension in i16/i32/i64). SystemZ 
; logical-to-memory instructions (NI, OI, XI) only operate on a single byte.
define void @or_i16_neg_2(ptr %src) {
; CHECK-LABEL: or_i16_neg_2:
; CHECK-NOT: oi {{.*}}, 254
; CHECK: lh   [[REG:%r[0-9]+]], 0(%r2)
; CHECK: oill [[REG]], 65534
; CHECK: sth  [[REG]], 0(%r2)
; CHECK: br   %r14
  %val = load i16, ptr %src
  %or = or i16 %val, -2
  store i16 %or, ptr %src
  ret void
}

define void @or_i32_neg_2(ptr %src) {
; CHECK-LABEL: or_i32_neg_2:
; CHECK-NOT: oiy {{.*}}, 254
; CHECK: o [[REG:%r[0-9]+]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %src
  %or = or i32 %val, -2
  store i32 %or, ptr %src
  ret void
}

define void @or_i64_neg_2(ptr %src) {
; CHECK-LABEL: or_i64_neg_2:
; CHECK-NOT: oiy {{.*}}, 254
; CHECK: lg [[REG:%r[0-9]+]], 0(%r2)
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %src
  %or = or i64 %val, -2
  store i64 %or, ptr %src
  ret void
}

; Check out of Range: 256 (0x100) - Requires 9 bits.
; Should not fold to oi/oiy.
define void @or_i32_too_wide(ptr %ptr) {
; CHECK-LABEL: or_i32_too_wide:
; CHECK-NOT: oi{{y?}} {{.*}}, 256
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 256
  store i32 %or, ptr %ptr
  ret void
}

; Check high: 128 (0x80) - The "signed bit" of a byte.
define void @or_i32_oiy_boundary_128(ptr %src) {
; CHECK-LABEL: or_i32_oiy_boundary_128:
; CHECK: oiy 4096(%r2), 128
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %or = or i32 %val, 128
  store i32 %or, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
define void @or_i32_oiy_max_byte(ptr %src) {
; CHECK-LABEL: or_i32_oiy_max_byte:
; CHECK: oiy 4096(%r2), 255
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %or = or i32 %val, 255
  store i32 %or, ptr %ptr
  ret void
}

; Check i16 high end of OI range:  Original 4095 + 1 (LSB) = 4095.
define void @or_i16_oi_high_disp(ptr %src) {
; CHECK-LABEL: or_i16_oi_high_disp:
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4094
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check i32 high end of OI range:  Original 4092 + 3 (LSB) = 4095.
define void @or_i32_oi_high_disp(ptr %src) {
; CHECK-LABEL: or_i32_oi_high_disp:
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4092
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check i64 high end of OI range:  Original 4088 + 7 (LSB) = 4095.
define void @or_i64_oi_high_disp(ptr %src) {
; CHECK-LABEL: or_i64_oi_high_disp:
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4088
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check i16 transisition to oiy: Original 4095 + 1 (LSB) = 4096 (triggers OIY).
define void @or_i16_oiy_transition_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_transition_disp:
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check i32 transisition to oiy: Original 4093 + 3 (LSB) = 4096 (triggers OIY).
define void @or_i32_oiy_transition_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_transition_disp:
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check i64  transisition to oiy: Original 4089 + 7 (LSB) = 4096 (triggers OIY).
define void @or_i64_oiy_transition_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_transition_disp:
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4089
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check i16 high end of the OIY range: Original 524286 + 1 (LSB) = 524287.
define void @or_i16_oiy_high_end_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_high_end_disp:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524286
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check i32  high end of the OIY range: Original 524284 + 3 (LSB) = 524287.
define void @or_i32_oiy_high_end_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_high_end_disp:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524284
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check i64 high end of the OIY range: Original 524280 + 7 (LSB) = 524287.
define void @or_i64_oiy_high_end_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_high_end_disp:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524280
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the next byte up - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i16_oiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: oi 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the next byte up - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i32_oiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: oi 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the next byte up - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i64_oiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: oi 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check i16: High end of the negative Displacement (Signed 20-bit).
define void @or_i16_oiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_neg_max_disp:
; CHECK: oiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -2
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check i32: High end of the negative Displacement (Signed 20-bit).
define void @or_i32_oiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_neg_max_disp:
; CHECK: oiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -4
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check i64: High end of the negative Displacement (Signed 20-bit).
define void @or_i64_oiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_neg_max_disp:
; CHECK: oiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -8
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check i16: Low end of the negative Displacement (Signed 20-bit).
define void @or_i16_oiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_neg_min_disp:
; CHECK: oiy -524287(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check i32: Low end of the negative Displacement (Signed 20-bit).
define void @or_i32_oiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_neg_min_disp:
; CHECK: oiy -524285(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check i64: Low end of the negative Displacement (Signed 20-bit).
define void @or_i64_oiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_neg_min_disp:
; CHECK: oiy -524281(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check the next byte down - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i16_oiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i16_oiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: oi 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check the next byte down - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i32_oiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i32_oiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: oi 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check the next byte down - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @or_i64_oiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: or_i64_oiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: oi 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check that OI does not allow an index for i16.
; Original 4094 + 1 (LSB) = 4095 (Max range for OI).
define void @or_i16_oi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i16_oi_no_index:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check that OI does not allow an index for i32.
; Original 4092 + 3 (LSB) = 4095 (Max range for OI).
define void @or_i32_oi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i32_oi_no_index:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check that OI does not allow an index for i64.
; Original 4088 + 7 (LSB) = 4095 (Max range for OI).
define void @or_i64_oi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i64_oi_no_index:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check that OIY does not allow an index for i16.
; Original 4095 + 1 (LSB) = 4096 (triggers OIY).
define void @or_i16_oiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i16_oiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check that OIY does not allow an index for i32.
; Original 4093 + 3 (LSB) = 4096 (triggers OIY).
define void @or_i32_oiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i32_oiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4093
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check that OIY does not allow an index for i64.
; Original 4089 + 7 (LSB) = 4096 (triggers OIY).
define void @or_i64_oiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: or_i64_oiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4089
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Volatile memory should not be folded into OI.
define i32 @test_volatile(ptr %ptr) {
; CHECK-LABEL: test_volatile:
; CHECK-NOT: oi {{.*}}, 1
; CHECK: l {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load volatile i32, ptr %ptr
  %or = or i32 %val, 1
  store volatile i32 %or, ptr %ptr
  ret i32 %val
}

; Multiple uses of the loaded value should not be folded.
define i32 @test_multi_use(ptr %ptr) {
; CHECK-LABEL: test_multi_use:
; CHECK: l [[REG:%r[0-9]+]], 0(%r2)
; CHECK: oill {{%r[0-9]+}}, 1
; CHECK: st {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret i32 %val
}
