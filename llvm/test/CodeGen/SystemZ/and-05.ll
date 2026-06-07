; Test ANDs of a constant into a byte of memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful constant for i8, expressed as a signed integer.
define void @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: ni 0(%r2), 1
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %and = and i8 %val, -255
  store i8 %and, ptr %ptr
  ret void
}

; Check lowest useful constant for i16, expressed as a signed integer.
define void @f1_i16(ptr %ptr) {
; CHECK-LABEL: f1_i16:
; CHECK: ni 1(%r2), 1
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, -255
  store i16 %and, ptr %ptr
  ret void
}

; Check lowest useful constant for i32, expressed as a signed integer.
define void @f1_i32(ptr %ptr) {
; CHECK-LABEL: f1_i32:
; CHECK: ni 3(%r2), 1
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, -255
  store i32 %and, ptr %ptr
  ret void
}

; Check lowest useful constant for i64, expressed as a signed integer.
define void @f1_i64(ptr %ptr) {
; CHECK-LABEL: f1_i64:
; CHECK: ni 7(%r2), 1
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, -255
  store i64 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i8, expressed as a signed integer.
define void @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: ni 0(%r2), 254
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %and = and i8 %val, -2
  store i8 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i16, expressed as a signed integer.
define void @f2_i16(ptr %ptr) {
; CHECK-LABEL: f2_i16:
; CHECK: ni 1(%r2), 254
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, -2
  store i16 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i32, expressed as a signed integer.
define void @f2_i32(ptr %ptr) {
; CHECK-LABEL: f2_i32:
; CHECK: ni 3(%r2), 254
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, -2
  store i32 %and, ptr %ptr
  ret void
}

; Check i32 with zero-extended i16 pattern (0x0000FFxx).
define void @f2_i32_zext(ptr %ptr) {
; CHECK-LABEL: f2_i32_zext:
; CHECK: n %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 65534  ; 0x0000FFFE
  store i32 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i64, expressed as a signed integer.
define void @f2_i64(ptr %ptr) {
; CHECK-LABEL: f2_i64:
; CHECK: ni 7(%r2), 254
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, -2
  store i64 %and, ptr %ptr
  ret void
}

; Check the lowest useful constant for i8, expressed as an unsigned integer.
define void @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: ni 0(%r2), 1
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %and = and i8 %val, 1
  store i8 %and, ptr %ptr
  ret void
}

; Check the lowest useful constant for i16, expressed as an unsigned integer.
; Constant 1 should not fold for i16, as it requires clearing the high byte.
define void @f3_i16(ptr %ptr) {
; CHECK-LABEL: f3_i16:
; CHECK-NOT: ni 1(%r2), 1
; CHECK: llh %r0, 0(%r2)
; CHECK: nilf %r0, 1
; CHECK: sth %r0, 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, 1
  store i16 %and, ptr %ptr
  ret void
}

; Check the lowest useful constant for i32, expressed as an unsigned integer.
; Constant 1 should not fold for i32, as it requires clearing the high bytes.
define void @f3_i32(ptr %ptr) {
; CHECK-LABEL: f3_i32:
; CHECK-NOT: ni 3(%r2), 1
; CHECK: lhi %r0, 1
; CHECK: n %r0, 0(%r2)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 1
  store i32 %and, ptr %ptr
  ret void
}

; Check the lowest useful constant for i64, expressed as an unsigned integer.
; Constant 1 should not fold for i64, as it requires clearing the high bytes.
define void @f3_i64(ptr %ptr) {
; CHECK-LABEL: f3_i64:
; CHECK-NOT: ni 7(%r2), 1
; CHECK: lg %r0, 0(%r2)
; CHECK: risbg %r0, %r0, 63, 191, 0
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, 1
  store i64 %and, ptr %ptr
  ret void
}

; Check the highest useful constant, expressed as a unsigned integer.
define void @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: ni 0(%r2), 254
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %and = and i8 %val, 254
  store i8 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i16, expressed as a unsigned integer.
; Constant 254 (0x00FE) should not fold for i16 because the high byte must be
; cleared.
define void @f4_i16(ptr %ptr) {
; CHECK-LABEL: f4_i16:
; CHECK-NOT: ni 1(%r2), 254
; CHECK: llh %r0, 0(%r2)
; CHECK: nilf %r0, 254
; CHECK: sth %r0, 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, 254
  store i16 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i32, expressed as a unsigned integer.
; Constant 254 should not fold for i32 because the high byte must be cleared.
define void @f4_i32(ptr %ptr) {
; CHECK-LABEL: f4_i32:
; CHECK-NOT: ni 3(%r2), 254
; CHECK: lhi %r0, 254
; CHECK: n %r0, 0(%r2)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 254
  store i32 %and, ptr %ptr
  ret void
}

; Check the highest useful constant for i64, expressed as a unsigned integer.
; Constant 254 should not fold for i64 because the high byte must be cleared.
define void @f4_i64(ptr %ptr) {
; CHECK-LABEL: f4_i64:
; CHECK-NOT: ni 7(%r2), 254
; CHECK: lg %r0, 0(%r2)
; CHECK: risbg %r0, %r0, 56, 190, 0
; CHECK: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, 254
  store i64 %and, ptr %ptr
  ret void
}

; Check the high end of the NI range for i8.
define void @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the high end of the NI range for i16.
define void @f5_i16(ptr %src) {
; CHECK-LABEL: f5_i16:
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4094
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the high end of the NI range for i32.
define void @f5_i32(ptr %src) {
; CHECK-LABEL: f5_i32:
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4092
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the high end of the NI range for i64.
define void @f5_i64(ptr %src) {
; CHECK-LABEL: f5_i64:
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4088
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the next byte up for i8, which should use NIY instead of NI.
define void @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4096
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the next byte up for i16, which should use NIY instead of NI.
define void @f6_i16(ptr %src) {
; CHECK-LABEL: f6_i16:
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the next byte up for i32, which should use NIY instead of NI.
define void @f6_i32(ptr %src) {
; CHECK-LABEL: f6_i32:
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the next byte up for i64, which should use NIY instead of NI.
define void @f6_i64(ptr %src) {
; CHECK-LABEL: f6_i64:
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4089
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the high end of the NIY range for i8.
define void @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524287
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the high end of the NIY range for i16.
define void @f7_i16(ptr %src) {
; CHECK-LABEL: f7_i16:
; CHECK: niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524286
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the high end of the NIY range for i32.
define void @f7_i32(ptr %src) {
; CHECK-LABEL: f7_i32:
; CHECK: niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524284
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the high end of the NIY range for i64.
define void @f7_i64(ptr %src) {
; CHECK-LABEL: f7_i64:
; CHECK: niy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524280
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the next byte up for i8, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, 524288
; CHECK: ni 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the next byte up for i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i16(ptr %src) {
; CHECK-LABEL: f8_i16:
; CHECK: agfi %r2, 524288
; CHECK: ni 1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the next byte up for i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i32(ptr %src) {
; CHECK-LABEL: f8_i32:
; CHECK: agfi %r2, 524288
; CHECK: ni 3(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the next byte up for i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i64(ptr %src) {
; CHECK-LABEL: f8_i64:
; CHECK: agfi %r2, 524288
; CHECK: ni 7(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the high end of the negative NIY range for i8.
define void @f9(ptr %src) {
; CHECK-LABEL: f9:
; CHECK: niy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -1
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the high end of the negative NIY range for i16.
define void @f9_i16(ptr %src) {
; CHECK-LABEL: f9_i16:
; CHECK: niy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -2
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the high end of the negative NIY range for i32.
define void @f9_i32(ptr %src) {
; CHECK-LABEL: f9_i32:
; CHECK: niy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -4
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the high end of the negative NIY range for i64.
define void @f9_i64(ptr %src) {
; CHECK-LABEL: f9_i64:
; CHECK: niy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -8
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the low end of the NIY range for i8.
define void @f10(ptr %src) {
; CHECK-LABEL: f10:
; CHECK: niy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the low end of the NIY range for i16.
define void @f10_i16(ptr %src) {
; CHECK-LABEL: f10_i16:
; CHECK: niy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the low end of the NIY range for i32.
define void @f10_i32(ptr %src) {
; CHECK-LABEL: f10_i32:
; CHECK: niy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524291
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the low end of the NIY range for i64.
define void @f10_i64(ptr %src) {
; CHECK-LABEL: f10_i64:
; CHECK: niy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524295
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check the next byte down for i8, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11(ptr %src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r2, -524289
; CHECK: ni 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the next byte down for i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i16(ptr %src) {
; CHECK-LABEL: f11_i16:
; CHECK: agfi %r2, -524290
; CHECK: ni 1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524290
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the next byte down for i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i32(ptr %src) {
; CHECK-LABEL: f11_i32:
; CHECK: agfi %r2, -524292
; CHECK: ni 3(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524292
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the next byte down for i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i64(ptr %src) {
; CHECK-LABEL: f11_i64:
; CHECK: agfi %r2, -524296
; CHECK: ni 7(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524296
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check that NI does not allow an index for i8.
define void @f12(i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check that NI does not allow an index for i16.
define void @f12_i16(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i16:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the NI does not allow an index for i32.
define void @f12_i32(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i32:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the NI does not allow an index for i64.
define void @f12_i64(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i64:
; CHECK: agr %r2, %r3
; CHECK: ni 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check that NIY does not allow an index for i8.
define void @f13(i64 %src, i64 %index) {
; CHECK-LABEL: f13:
; CHECK: agr %r2, %r3
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i8, ptr %ptr
  %and = and i8 %val, 127
  store i8 %and, ptr %ptr
  ret void
}

; Check the NIY does not allow an index for i16.
define void @f13_i16(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i16:
; CHECK: agr %r2, %r3
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65407
  store i16 %and, ptr %ptr
  ret void
}

; Check the NIY does not allow an index for i32.
define void @f13_i32(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i32:
; CHECK: agr %r2, %r3
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4093
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967167
  store i32 %and, ptr %ptr
  ret void
}

; Check the NIY does not allow an index for i64.
define void @f13_i64(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i64:
; CHECK: agr %r2, %r3
; CHECK: niy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4089
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %and = and i64 %val, -129
  store i64 %and, ptr %ptr
  ret void
}

; Check folding of multi-byte 'and' operations into byte-immediate memory
; operation 'ni'.
; Additional tests for immAndLSB8 PatLeaf logic - preservation of upper bytes
; for i16, i32, and i64. Low/high constant (signed/unsigned) tests have already
; been covered (f1 to f4).

; Check i16 - should not fold. High byte has bit cleared.
define void @f_const_i16_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i16_no_fold:
; CHECK-NOT: ni 1(%r2)
; CHECK: llh [[REG:%r[0-5]]], 0(%r2)
; CHECK: nilf [[REG]], 65278
; CHECK: sth [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65278  ; 0xFEFE
  store i16 %and, ptr %ptr
  ret void
}

; Check i32 - should not fold. Upper bytes have bit cleared.
define void @f_const_i32_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i32_no_fold:
; CHECK-NOT: ni 3(%r2)
; CHECK: lhi [[REG:%r[0-5]]], -258
; CHECK: n [[REG]], 0(%r2)
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294967038  ; 0xFFFFFEFE
  store i32 %and, ptr %ptr
  ret void
}

; Check i64 - should not fold. Upper bytes have bit cleared.
define void @f_const_i64_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i64_no_fold:
; CHECK-NOT: ni 7(%r2)
; CHECK: lghi [[REG:%r[0-5]]], -258
; CHECK: ng [[REG]], 0(%r2)
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, -258  ; 0xFFFFFFFFFFFFFFFE
  store i64 %and, ptr %ptr
  ret void
}

; Check i16 - should not fold. Constant 0xFE00 affects more than just the LSB.
define void @f_const_i16_multi_byte_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i16_multi_byte_no_fold:
; CHECK-NOT: ni 1(%r2)
; CHECK: llh [[REG:%r[0-5]]], 0(%r2)
; CHECK: nilf [[REG]], 65024
; CHECK: sth [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %and = and i16 %val, 65024  ; 0xFE00
  store i16 %and, ptr %ptr
  ret void
}

; Check i32 - should not fold. Constant 0xFFFFFE00 affects more than just the
; LSB.
define void @f_const_i32_multi_byte_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i32_multi_byte_no_fold:
; CHECK-NOT: ni 3(%r2)
; CHECK: lhi %r0, -512
; CHECK: n %r0, 0(%r2)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294966784  ; 0xFFFFFE00
  store i32 %and, ptr %ptr
  ret void
}

; Check i32 - should not fold. Constant affects a non-LSB byte.
define void @f_const_i32_wrong_byte_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i32_wrong_byte_no_fold:
; CHECK-NOT: ni 3(%r2)
; CHECK: iilf [[REG:%r[0-5]]], 4294902015
; CHECK: n [[REG]], 0(%r2)
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %and = and i32 %val, 4294902015  ; 0xFFFF00FF
  store i32 %and, ptr %ptr
  ret void
}

; Check i64 - should not fold. Constant affects more than just the LSB.
define void @f_const_i64_multi_byte_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i64_multi_byte_no_fold:
; CHECK-NOT: ni 7(%r2)
; CHECK: lghi [[REG:%r[0-5]]], -512
; CHECK: ng [[REG]], 0(%r2)
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, -512  ; 0xFFFFFFFFFFFFFE00
  store i64 %and, ptr %ptr
  ret void
}

; Check i64 - should not fold. Constant affects a non-LSB byte.
define void @f_const_i64_wrong_byte_no_fold(ptr %ptr) {
; CHECK-LABEL: f_const_i64_wrong_byte_no_fold:
; CHECK-NOT: ni 7(%r2)
; CHECK: lgfi [[REG:%r[0-5]]], -65281
; CHECK: ng [[REG]], 0(%r2)
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %and = and i64 %val, -65281  ; 0xFFFFFFFFFFFF00FF
  store i64 %and, ptr %ptr
  ret void
}
