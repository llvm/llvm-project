; Test ORs of a constant into a byte of memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the lowest useful constant for i8, expressed as a signed integer.
define void @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: oi 0(%r2), 1
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %or = or i8 %val, -255
  store i8 %or, ptr %ptr
  ret void
}

; Check lowest useful constant for i16, expressed as a signed integer.
define void @f1_i16(ptr %ptr) {
; CHECK-LABEL: f1_i16:
; CHECK-NOT: oi 1(%r2)
; CHECK: lh %r0, 0(%r2)
; CHECK: oill %r0, 65281
; CHECK: sth %r0, 0(%r2)
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %or = or i16 %val, -255
  store i16 %or, ptr %ptr
  ret void
}

; Check lowest useful constant for i32, expressed as a signed integer.
define void @f1_i32(ptr %ptr) {
; CHECK-LABEL: f1_i32:
; CHECK-NOT: oi 3(%r2)
; CHECK: lhi %r0, -255
; CHECK: o %r0, 0(%r2)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, -255
  store i32 %or, ptr %ptr
  ret void
}

; Check lowest useful constant for i64, expressed as a signed integer.
define void @f1_i64(ptr %ptr) {
; CHECK-LABEL: f1_i64:
; CHECK-NOT: oi 7(%r2)
; CHECK: lg {{%r[0-9]+}}, 0(%r2)
; CHECK: oihf    %r0, 4294967295
; CHECK: stg {{%r[0-9]+}}, 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %or = or i64 %val, -255
  store i64 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i8, expressed as a signed integer.
define void @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %or = or i8 %val, -2
  store i8 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i16, expressed as a signed integer.
define void @f2_i16(ptr %ptr) {
; CHECK-LABEL: f2_i16:
; CHECK-NOT: oi 1(%r2), 254
; CHECK: lh %r0, 0(%r2)
; CHECK: oill %r0, 65534
; CHECK: sth %r0, 0(%r2)
  %val = load i16, ptr %ptr
  %or = or i16 %val, -2
  store i16 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i32, expressed as a signed integer.
define void @f2_i32(ptr %ptr) {
; CHECK-LABEL: f2_i32:
; CHECK-NOT: oi 3(%r2)
; CHECK: lhi %r0, -2
; CHECK: o %r0, 0(%r2)
; CHECK: st %r0, 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, -2
  store i32 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i64, expressed as a signed integer.
define void @f2_i64(ptr %ptr) {
; CHECK-LABEL: f2_i64:
; CHECK-NOT: oi 7(%r2)
; CHECK: lg [[REG:%r[0-9]+]], 0(%r2)
; CHECK: oilf [[REG]], 4294967294
; CHECK: stg [[REG]], 0(%r2)
  %val = load i64, ptr %ptr
  %or = or i64 %val, -2
  store i64 %or, ptr %ptr
  ret void
}

; Check the lowest useful constant for i8, expressed as an unsigned integer.
define void @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: oi 0(%r2), 1
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %or = or i8 %val, 1
  store i8 %or, ptr %ptr
  ret void
}

; Check the lowest useful constant for i16, expressed as an unsigned integer.
define void @f3_i16(ptr %ptr) {
; CHECK-LABEL: f3_i16:
; CHECK: oi 1(%r2), 1
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %or = or i16 %val, 1
  store i16 %or, ptr %ptr
  ret void
}

; Check the lowest useful constant for i32, expressed as an unsigned integer.
define void @f3_i32(ptr %ptr) {
; CHECK-LABEL: f3_i32:
; CHECK: oi 3(%r2), 1
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 1
  store i32 %or, ptr %ptr
  ret void
}

; Check the lowest useful constant for i64, expressed as an unsigned integer.
define void @f3_i64(ptr %ptr) {
; CHECK-LABEL: f3_i64:
; CHECK: oi 7(%r2), 1
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %or = or i64 %val, 1
  store i64 %or, ptr %ptr
  ret void
}

; Check the highest useful constant i8, expressed as a unsigned integer.
define void @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: oi 0(%r2), 254
; CHECK: br %r14
  %val = load i8, ptr %ptr
  %or = or i8 %val, 254
  store i8 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i16, expressed as a unsigned integer.
define void @f4_i16(ptr %ptr) {
; CHECK-LABEL: f4_i16:
; CHECK: oi 1(%r2), 254
; CHECK: br %r14
  %val = load i16, ptr %ptr
  %or = or i16 %val, 254
  store i16 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i32, expressed as a unsigned integer.
define void @f4_i32(ptr %ptr) {
; CHECK-LABEL: f4_i32:
; CHECK: oi 3(%r2), 254
; CHECK: br %r14
  %val = load i32, ptr %ptr
  %or = or i32 %val, 254
  store i32 %or, ptr %ptr
  ret void
}

; Check the highest useful constant for i64, expressed as a unsigned integer.
define void @f4_i64(ptr %ptr) {
; CHECK-LABEL: f4_i64:
; CHECK: oi 7(%r2), 254
; CHECK: br %r14
  %val = load i64, ptr %ptr
  %or = or i64 %val, 254
  store i64 %or, ptr %ptr
  ret void
}

; Check the high end of the OI range for i8.
define void @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the high end of the OI range for i16.
define void @f5_i16(ptr %src) {
; CHECK-LABEL: f5_i16:
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4094
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the high end of the OI range for i32.
define void @f5_i32(ptr %src) {
; CHECK-LABEL: f5_i32:
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4092
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the high end of the OI range for i64.
define void @f5_i64(ptr %src) {
; CHECK-LABEL: f5_i64:
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4088
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the next byte up for i8, which should use OIY instead of OI.
define void @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4096
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the next byte up for i16, which should use OIY instead of OI.
define void @f6_i16(ptr %src) {
; CHECK-LABEL: f6_i16:
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4095
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the next byte up for i32, which should use OIY instead of OI.
define void @f6_i32(ptr %src) {
; CHECK-LABEL: f6_i32:
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4093
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the next byte up for i64, which should use OIY instead of OI.
define void @f6_i64(ptr %src) {
; CHECK-LABEL: f6_i64:
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 4089
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the high end of the OIY range for i8.
define void @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524287
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the high end of the OIY range for i16.
define void @f7_i16(ptr %src) {
; CHECK-LABEL: f7_i16:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524286
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the high end of the OIY range for i32.
define void @f7_i32(ptr %src) {
; CHECK-LABEL: f7_i32:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524284
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the high end of the OIY range for i64.
define void @f7_i64(ptr %src) {
; CHECK-LABEL: f7_i64:
; CHECK: oiy 524287(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524280
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the next byte up for i8, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, 524288
; CHECK: oi 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the next byte up for i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i16(ptr %src) {
; CHECK-LABEL: f8_i16:
; CHECK: agfi %r2, 524288
; CHECK: oi 1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the next byte up for i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i32(ptr %src) {
; CHECK-LABEL: f8_i32:
; CHECK: agfi %r2, 524288
; CHECK: oi 3(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the next byte up for i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8_i64(ptr %src) {
; CHECK-LABEL: f8_i64:
; CHECK: agfi %r2, 524288
; CHECK: oi 7(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 524288
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the high end of the negative OIY range for i8.
define void @f9(ptr %src) {
; CHECK-LABEL: f9:
; CHECK: oiy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -1
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the high end of the negative OIY range for i16.
define void @f9_i16(ptr %src) {
; CHECK-LABEL: f9_i16:
; CHECK: oiy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -2
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the high end of the negative OIY range for i32.
define void @f9_i32(ptr %src) {
; CHECK-LABEL: f9_i32:
; CHECK: oiy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -4
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the high end of the negative OIY range for i64.
define void @f9_i64(ptr %src) {
; CHECK-LABEL: f9_i64:
; CHECK: oiy -1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -8
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the low end of the OIY range for i8.
define void @f10(ptr %src) {
; CHECK-LABEL: f10:
; CHECK: oiy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524288
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the low end of the OIY range for i16.
define void @f10_i16(ptr %src) {
; CHECK-LABEL: f10_i16:
; CHECK: oiy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the low end of the OIY range for i32.
define void @f10_i32(ptr %src) {
; CHECK-LABEL: f10_i32:
; CHECK: oiy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524291
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the low end of the OIY range for i64.
define void @f10_i64(ptr %src) {
; CHECK-LABEL: f10_i64:
; CHECK: oiy -524288(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524295
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check the next byte down for i8, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11(ptr %src) {
; CHECK-LABEL: f11:
; CHECK: agfi %r2, -524289
; CHECK: oi 0(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524289
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the next byte down for i16, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i16(ptr %src) {
; CHECK-LABEL: f11_i16:
; CHECK: agfi %r2, -524290
; CHECK: oi 1(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524290
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the next byte down for i32, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i32(ptr %src) {
; CHECK-LABEL: f11_i32:
; CHECK: agfi %r2, -524292
; CHECK: oi 3(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524292
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the next byte down for i64, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f11_i64(ptr %src) {
; CHECK-LABEL: f11_i64:
; CHECK: agfi %r2, -524296
; CHECK: oi 7(%r2), 127
; CHECK: br %r14
  %ptr = getelementptr i8, ptr %src, i64 -524296
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check that OI does not allow an index for i8.
define void @f12(i64 %src, i64 %index) {
; CHECK-LABEL: f12:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the OI does not allow an index for i16.
define void @f12_i16(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i16:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4094
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the OI does not allow an index for i32.
define void @f12_i32(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i32:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the OI does not allow an index for i64.
define void @f12_i64(i64 %src, i64 %index) {
; CHECK-LABEL: f12_i64:
; CHECK: agr %r2, %r3
; CHECK: oi 4095(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}

; Check that OIY does not allow an index for i8.
define void @f13(i64 %src, i64 %index) {
; CHECK-LABEL: f13:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i8, ptr %ptr
  %or = or i8 %val, 127
  store i8 %or, ptr %ptr
  ret void
}

; Check the OIY does not allow an index for i16.
define void @f13_i16(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i16:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i16, ptr %ptr
  %or = or i16 %val, 127
  store i16 %or, ptr %ptr
  ret void
}

; Check the OIY does not allow an index for i32.
define void @f13_i32(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i32:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4093
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %or = or i32 %val, 127
  store i32 %or, ptr %ptr
  ret void
}

; Check the OIY does not allow an index for i64.
define void @f13_i64(i64 %src, i64 %index) {
; CHECK-LABEL: f13_i64:
; CHECK: agr %r2, %r3
; CHECK: oiy 4096(%r2), 127
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4089
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i64, ptr %ptr
  %or = or i64 %val, 127
  store i64 %or, ptr %ptr
  ret void
}
