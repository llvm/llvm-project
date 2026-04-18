; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; xor-05.ll tests the multiclass RMWIByte for i8 types. 
; This test verifies the DAGToDag implementation for i16, i32, and i64 types,
; covering both 12-bit unsigned (XI) and 20-bit signed (XIY) displacements.

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
define void @xor_i16_xi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: xor_i16_xi_low_unsigned:
; CHECK: xi 1(%r2), 1
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32 (int): Offset + 3.
define void @xor_i32_xi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: xor_i32_xi_low_unsigned:
; CHECK: xi 3(%r2), 1
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64 (int): Offset + 7.
define void @xor_i64_xi_low_unsigned(ptr %ptr) {
; CHECK-LABEL: xor_i64_xi_low_unsigned:
; CHECK: xi 7(%r2), 1
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check High: 128 (0x80) - The "signed bit" of a byte.
; Verifies that patterns with the 8th bit set still fold.
define void @xor_i32_xi_boundary_128(ptr %ptr) {
; CHECK-LABEL: xor_i32_xi_boundary_128:
; CHECK: xi 3(%r2), 128
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 128
  store i32 %xor, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
define void @xor_i32_xi_max_byte(ptr %ptr) {
; CHECK-LABEL: xor_i32_xi_max_byte:
; CHECK: xi 3(%r2), 255
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 255
  store i32 %xor, ptr %ptr
  ret void
}

; Check bit-width constraint with negative signed value.
; This test case with -2 (0xFF...FE) fails to fold because the immediate value 
; has more than 8 active bits (due to sign extension in i16/i32/i64). SystemZ 
; logical-to-memory instructions (NI, OI, XI) only operate on a single byte.
define void @xor_i16_neg_2(ptr %src) {
; CHECK-LABEL: xor_i16_neg_2:
; CHECK-NOT: xi {{.*}}, 254
; CHECK: lh   [[REG:%r[0-9]+]], 0(%r2)
; CHECK: xilf [[REG]], 65534
; CHECK: sth  [[REG]], 0(%r2)
; CHECK: br   %r14
  %val = load i16, ptr %src
  %xor = xor i16 %val, -2
  store i16 %xor, ptr %src
  ret void
}

define void @xor_i32_neg_2(ptr %src) {
; CHECK-LABEL: xor_i32_neg_2:
; CHECK-NOT: xiy {{.*}}, 254
; CHECK: x [[REG:%r[0-9]+]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %src
  %xor = xor i32 %val, -2
  store i32 %xor, ptr %src
  ret void
}

define void @xor_i64_neg_2(ptr %src) {
; CHECK-LABEL: xor_i64_neg_2:
; CHECK-NOT: xiy {{.*}}, 254
; CHECK: lg [[REG:%r[0-9]+]], 0(%r2)
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %src
  %xor = xor i64 %val, -2
  store i64 %xor, ptr %src
  ret void
}

; Check out of Range: 256 (0x100) - Requires 9 bits.
; Should not fold to xi/xiy.
define void @xor_i32_too_wide(ptr %ptr) {
; CHECK-LABEL: xor_i32_too_wide:
; CHECK-NOT: xi{{y?}} {{.*}}, 256
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 256
  store i32 %xor, ptr %ptr
  ret void
}

; Check high: 128 (0x80) - The "signed bit" of a byte.
define void @xor_i32_xiy_boundary_128(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_boundary_128:
; CHECK: xiy 4096(%r2), 128
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 128
  store i32 %xor, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
define void @xor_i32_xiy_max_byte(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_max_byte:
; CHECK: xiy 4096(%r2), 255
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 255
  store i32 %xor, ptr %ptr
  ret void
}

; Check i16 high end of XI range:  Original 4095 + 1 (LSB) = 4095.
define void @xor_i16_xi_high_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xi_high_disp:
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4094
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32 high end of XI range:  Original 4092 + 3 (LSB) = 4095.
define void @xor_i32_xi_high_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xi_high_disp:
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4092
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64 high end of XI range:  Original 4088 + 7 (LSB) = 4095.
define void @xor_i64_xi_high_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xi_high_disp:
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4088
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check i16 transisition to xiy: Original 4095 + 1 (LSB) = 4096 (triggers XIY).
define void @xor_i16_xiy_transition_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_transition_disp:
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32 transisition to xiy: Original 4093 + 3 (LSB) = 4096 (triggers XIY).
define void @xor_i32_xiy_transition_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_transition_disp:
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64  transisition to xiy: Original 4089 + 7 (LSB) = 4096 (triggers XIY).
define void @xor_i64_xiy_transition_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_transition_disp:
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4089
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check i16 high end of the XIY range: Original 524286 + 1 (LSB) = 524287.
define void @xor_i16_xiy_high_end_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_high_end_disp:
; CHECK: xiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524286
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 127
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32  high end of the XIY range: Original 524284 + 3 (LSB) = 524287.
define void @xor_i32_xiy_high_end_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_high_end_disp:
; CHECK: xiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524284
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 127
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64 high end of the XIY range: Original 524280 + 7 (LSB) = 524287.
define void @xor_i64_xiy_high_end_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_high_end_disp:
; CHECK: xiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524280
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 127
  store i64 %xor, ptr %ptr
  ret void
}

; Check the next byte up - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i16_xiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: xi 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 127
  store i16 %xor, ptr %ptr
  ret void
}

; Check the next byte up - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i32_xiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: xi 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 127
  store i32 %xor, ptr %ptr
  ret void
}

; Check the next byte up - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i64_xiy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: xi 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 127
  store i64 %xor, ptr %ptr
  ret void
}

; Check i16: High end of the negative Displacement (Signed 20-bit).
define void @xor_i16_xiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_neg_max_disp:
; CHECK: xiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -2
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32: High end of the negative Displacement (Signed 20-bit).
define void @xor_i32_xiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_neg_max_disp:
; CHECK: xiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -4
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64: High end of the negative Displacement (Signed 20-bit).
define void @xor_i64_xiy_neg_max_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_neg_max_disp:
; CHECK: xiy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -8
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check i16: Low end of the negative Displacement (Signed 20-bit).
define void @xor_i16_xiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_neg_min_disp:
; CHECK: xiy -524287(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check i32: Low end of the negative Displacement (Signed 20-bit).
define void @xor_i32_xiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_neg_min_disp:
; CHECK: xiy -524285(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check i64: Low end of the negative Displacement (Signed 20-bit).
define void @xor_i64_xiy_neg_min_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_neg_min_disp:
; CHECK: xiy -524281(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check the next byte down - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i16_xiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i16_xiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: xi 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check the next byte down - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i32_xiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i32_xiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: xi 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check the next byte down - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @xor_i64_xiy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: xor_i64_xiy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: xi 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i16.
; Original 4094 + 1 (LSB) = 4095 (Max range for XI).
define void @xor_i16_xi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i16_xi_no_index:
; CHECK: agr %r2, %r3
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i32.
; Original 4092 + 3 (LSB) = 4095 (Max range for XI).
define void @xor_i32_xi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i32_xi_no_index:
; CHECK: agr %r2, %r3
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i64.
; Original 4088 + 7 (LSB) = 4095 (Max range for XI).
define void @xor_i64_xi_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i64_xi_no_index:
; CHECK: agr %r2, %r3
; CHECK: xi 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i16.
; Original 4095 + 1 (LSB) = 4096 (triggers XIY).
define void @xor_i16_xiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i16_xiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %xor = xor i16 %val, 1
  store i16 %xor, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i32.
; Original 4093 + 3 (LSB) = 4096 (triggers XIY).
define void @xor_i32_xiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i32_xiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4093
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i64.
; Original 4089 + 7 (LSB) = 4096 (triggers XIY).
define void @xor_i64_xiy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: xor_i64_xiy_no_index:
; CHECK: agr %r2, %r3
; CHECK: xiy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4089
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %xor = xor i64 %val, 1
  store i64 %xor, ptr %ptr
  ret void
}

; Volatile memory should not be folded into XI.
define i32 @test_volatile(ptr %ptr) {
; CHECK-LABEL: test_volatile:
; CHECK-NOT: xi {{.*}}, 1
; CHECK: l {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load volatile i32, ptr %ptr
  %xor = xor i32 %val, 1
  store volatile i32 %xor, ptr %ptr
  ret i32 %val
}

; Multiple uses of the loaded value should not be folded.
define i32 @test_multi_use(ptr %ptr) {
; CHECK-LABEL: test_multi_use:
; CHECK: l [[REG:%r[0-9]+]], 0(%r2)
; CHECK: xilf {{%r[0-9]+}}, 1
; CHECK: st {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %xor = xor i32 %val, 1
  store i32 %xor, ptr %ptr
  ret i32 %val
}
