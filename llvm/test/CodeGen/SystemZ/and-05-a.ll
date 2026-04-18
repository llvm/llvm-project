; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; and-05.ll tests the multiclass RMWIByte for i8 types. 
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
define void @and_i16_ni_low_unsigned(ptr %ptr) {
; CHECK-LABEL: and_i16_ni_low_unsigned:
; CHECK: ni 1(%r2), 1
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check i32 (int): Offset + 3.
define void @and_i32_ni_low_unsigned(ptr %ptr) {
; CHECK-LABEL: and_i32_ni_low_unsigned:
; CHECK: ni 3(%r2), 1
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check i64 (int): Offset + 7.
define void @and_i64_ni_low_unsigned(ptr %ptr) {
; CHECK-LABEL: and_i64_ni_low_unsigned:
; CHECK: ni 7(%r2), 1
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check High: 128 (0x80) - The "signed bit" of a byte.
; Verifies that patterns with the 8th bit set still fold.
define void @and_i32_ni_boundary_128(ptr %ptr) {
; CHECK-LABEL: and_i32_ni_boundary_128:
; CHECK: ni 3(%r2), 128
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 128
  store i32 %and, ptr %ptr
  ret void
}

; Check (max - 1): 254 (0xFE) - All but one bit set in the target byte.
define void @and_i32_ni_max_byte_minus_1(ptr %ptr) {
; CHECK-LABEL: and_i32_ni_max_byte_minus_1:
; CHECK: ni 3(%r2), 254
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 254
  store i32 %and, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
; Phase ordering, DAG combiner identifies 'and 255' on 32-bit load as
; zero-extending byte load (llc).
define void @and_i32_ni_max_byte(ptr %ptr) {
; CHECK-LABEL: and_i32_ni_max_byte:
; CHECK: llc [[REG:%r[0-9]+]], 3(%r2)
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 255
  store i32 %and, ptr %ptr
  ret void
}

; Check bit-width constraint with negative signed value.
; This test case with -2 (0xFF...FE) fails to fold because the immediate value 
; has more than 8 active bits (due to sign extension in i16/i32/i64). SystemZ 
; logical-to-memory instructions (NI, OI, XI) only operate on a single byte.
define void @and_i16_neg_2(ptr %src) {
; CHECK-LABEL: and_i16_neg_2:
; CHECK-NOT: ni {{.*}}, 254
; CHECK: lh   [[REG:%r[0-9]+]], 0(%r2)
; CHECK: nilf [[REG]], 65534
; CHECK: sth  [[REG]], 0(%r2)
; CHECK: br   %r14
  %val = load i16, ptr %src
  %and = and i16 %val, -2
  store i16 %and, ptr %src
  ret void
}

define void @and_i32_neg_2(ptr %src) {
; CHECK-LABEL: and_i32_neg_2:
; CHECK-NOT:niy {{.*}}, 254
; CHECK: n [[REG:%r[0-9]+]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %src
  %and = and i32 %val, -2
  store i32 %and, ptr %src
  ret void
}

; DAG Combiner already combines it into a single And Memory (ng) instruction.
define void @and_i64_neg_2(ptr %src) {
; CHECK-LABEL: and_i64_neg_2:
; CHECK: ng %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %src
  %and = and i64 %val, -2
  store i64 %and, ptr %src
  ret void
}

; Check out of Range: 256 (0x100) - Requires 9 bits.
; Should not fold to ni/niy.
define void @and_i32_too_wide(ptr %ptr) {
; CHECK-LABEL: and_i32_too_wide:
; CHECK-NOT: ni{{y?}} {{.*}}, 256
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 256
  store i32 %and, ptr %ptr
  ret void
}

; Check high: 128 (0x80) - The "signed bit" of a byte.
define void @and_i32_niy_boundary_128(ptr %src) {
; CHECK-LABEL: and_i32_niy_boundary_128:
; CHECK: niy 4096(%r2), 128
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %and = and i32 %val, 128
  store i32 %and, ptr %ptr
  ret void
}

; Check (max - 1): 254 (0xFE) - All but one bit set in the target byte.
define void @and_i32_niy_max_byte_minus_1(ptr %src) {
; CHECK-LABEL: and_i32_niy_max_byte_minus_1:
; CHECK: niy 4096(%r2), 254
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %and = and i32 %val, 254
  store i32 %and, ptr %ptr
  ret void
}

; Check max: 255 (0xFF) - All bits set in the target byte.
; DAG Combiner transforms the 4-byte load into an 8-bit load logical (llc).
define void @and_i32_niy_max_byte(ptr %src) {
; CHECK-LABEL: and_i32_niy_max_byte:
; CHECK: llc [[REG:%r[0-9]+]], 4096(%r2)
; CHECK: st  [[REG]], 4093(%r2)
; CHECK: br  %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %xor = and i32 %val, 255
  store i32 %xor, ptr %ptr
  ret void
}

; Check i16 high end of XI range:  Original 4095 + 1 (LSB) = 4095.
define void @and_i16_ni_high_disp(ptr %src) {
; CHECK-LABEL: and_i16_ni_high_disp:
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4094
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check i32 high end of XI range:  Original 4092 + 3 (LSB) = 4095.
define void @and_i32_ni_high_disp(ptr %src) {
; CHECK-LABEL: and_i32_ni_high_disp:
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4092
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check i64 high end of XI range:  Original 4088 + 7 (LSB) = 4095.
define void @and_i64_ni_high_disp(ptr %src) {
; CHECK-LABEL: and_i64_ni_high_disp:
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4088
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check i16 transisition to niy: Original 4095 + 1 (LSB) = 4096 (triggers XIY).
define void @and_i16_niy_transition_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_transition_disp:
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check i32 transisition to niy: Original 4093 + 3 (LSB) = 4096 (triggers XIY).
define void @and_i32_niy_transition_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_transition_disp:
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check i64  transisition to niy: Original 4089 + 7 (LSB) = 4096 (triggers XIY).
define void @and_i64_niy_transition_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_transition_disp:
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4089
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check i16 high end of the XIY range: Original 524286 + 1 (LSB) = 524287.
define void @and_i16_niy_high_end_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_high_end_disp:
; CHECK:niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524286
  %val = load i16, ptr %ptr
  %and = and i16 %val, 127
  store i16 %and, ptr %ptr
  ret void
}

; Check i32  high end of the XIY range: Original 524284 + 3 (LSB) = 524287.
define void @and_i32_niy_high_end_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_high_end_disp:
; CHECK:niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524284
  %val = load i32, ptr %ptr
  %and = and i32 %val, 127
  store i32 %and, ptr %ptr
  ret void
}

; Check i64 high end of the XIY range: Original 524280 + 7 (LSB) = 524287.
define void @and_i64_niy_high_end_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_high_end_disp:
; CHECK:niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524280
  %val = load i64, ptr %ptr
  %and = and i64 %val, 127
  store i64 %and, ptr %ptr
  ret void
}

; Check the next byte up - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i16_niy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: ni 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i16, ptr %ptr
  %and = and i16 %val, 127
  store i16 %and, ptr %ptr
  ret void
}

; Check the next byte up - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i32_niy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: ni 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i32, ptr %ptr
  %and = and i32 %val, 127
  store i32 %and, ptr %ptr
  ret void
}

; Check the next byte up - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i64_niy_pos_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_pos_out_of_range_disp:
; CHECK: agfi %r2, 524288
; CHECK: ni 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i64, ptr %ptr
  %and = and i64 %val, 127
  store i64 %and, ptr %ptr
  ret void
}

; Check i16: High end of the negative Displacement (Signed 20-bit).
define void @and_i16_niy_neg_max_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_neg_max_disp:
; CHECK:niy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -2
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check i32: High end of the negative Displacement (Signed 20-bit).
define void @and_i32_niy_neg_max_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_neg_max_disp:
; CHECK:niy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -4
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check i64: High end of the negative Displacement (Signed 20-bit).
define void @and_i64_niy_neg_max_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_neg_max_disp:
; CHECK:niy -1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -8
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check i16: Low end of the negative Displacement (Signed 20-bit).
define void @and_i16_niy_neg_min_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_neg_min_disp:
; CHECK:niy -524287(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check i32: Low end of the negative Displacement (Signed 20-bit).
define void @and_i32_niy_neg_min_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_neg_min_disp:
; CHECK:niy -524285(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check i64: Low end of the negative Displacement (Signed 20-bit).
define void @and_i64_niy_neg_min_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_neg_min_disp:
; CHECK:niy -524281(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check the next byte down - i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i16_niy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i16_niy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: ni 1(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check the next byte down - i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i32_niy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i32_niy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: ni 3(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check the next byte down - i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @and_i64_niy_neg_out_of_range_disp(ptr %src) {
; CHECK-LABEL: and_i64_niy_neg_out_of_range_disp:
; CHECK: agfi %r2, -524289
; CHECK: ni 7(%r2), 1
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i16.
; Original 4094 + 1 (LSB) = 4095 (Max range for XI).
define void @and_i16_ni_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i16_ni_no_index:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i32.
; Original 4092 + 3 (LSB) = 4095 (Max range for XI).
define void @and_i32_ni_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i32_ni_no_index:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check that XI does not allow an index for i64.
; Original 4088 + 7 (LSB) = 4095 (Max range for XI).
define void @and_i64_ni_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i64_ni_no_index:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i16.
; Original 4095 + 1 (LSB) = 4096 (triggers XIY).
define void @and_i16_niy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i16_niy_no_index:
; CHECK: agr %r2, %r3
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i32.
; Original 4093 + 3 (LSB) = 4096 (triggers XIY).
define void @and_i32_niy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i32_niy_no_index:
; CHECK: agr %r2, %r3
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4093
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check that XIY does not allow an index for i64.
; Original 4089 + 7 (LSB) = 4096 (triggers XIY).
define void @and_i64_niy_no_index(i64 %src, i64 %index) {
; CHECK-LABEL: and_i64_niy_no_index:
; CHECK: agr %r2, %r3
; CHECK:niy 4096(%r2), 1
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4089
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Volatile memory should not be folded into XI.
define i32 @test_volatile(ptr %ptr) {
; CHECK-LABEL: test_volatile:
; CHECK-NOT: ni {{.*}}, 1
; CHECK: l {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load volatile i32, ptr %ptr
  %and = and i32 %val, 1
  store volatile i32 %and, ptr %ptr
  ret i32 %val
}

; Multiple uses of the loaded value should not be folded.
define i32 @test_multi_use(ptr %ptr) {
; CHECK-LABEL: test_multi_use:
; CHECK: l [[REG:%r[0-9]+]], 0(%r2)
; CHECK: nilf {{%r[0-9]+}}, 1
; CHECK: st {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret i32 %val
}
