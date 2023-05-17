; RUN: llc -march=mips -mcpu=mips32r2 -mattr=dsp < %s | FileCheck %s -check-prefix=R1
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=dspr2 < %s | FileCheck %s -check-prefix=R2

; R1-LABEL: test_lbux:
; R1: lbux ${{[0-9]+}}

define zeroext i8 @test_lbux(ptr nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %b, i32 %i
  %0 = load i8, ptr %add.ptr, align 1
  ret i8 %0
}

; R1-LABEL: test_lhx:
; R1: lhx ${{[0-9]+}}

define signext i16 @test_lhx(ptr nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i16, ptr %b, i32 %i
  %0 = load i16, ptr %add.ptr, align 2
  ret i16 %0
}

; R1-LABEL: test_lwx:
; R1: lwx ${{[0-9]+}}

define i32 @test_lwx(ptr nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i32, ptr %b, i32 %i
  %0 = load i32, ptr %add.ptr, align 4
  ret i32 %0
}

; R1-LABEL: test_add_v2q15_:
; R1: addq.ph ${{[0-9]+}}

define { i32 } @test_add_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %add = add <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %add to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: test_sub_v2q15_:
; R1: subq.ph ${{[0-9]+}}

define { i32 } @test_sub_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %sub = sub <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %sub to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R2-LABEL: test_mul_v2q15_:
; R2: mul.ph ${{[0-9]+}}

; mul.ph is an R2 instruction. Check that multiply node gets expanded.
; R1-LABEL: test_mul_v2q15_:
; R1: mul ${{[0-9]+}}
; R1: mul ${{[0-9]+}}

define { i32 } @test_mul_v2q15_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %mul = mul <2 x i16> %0, %1
  %2 = bitcast <2 x i16> %mul to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: test_add_v4i8_:
; R1: addu.qb ${{[0-9]+}}

define { i32 } @test_add_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %add = add <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %add to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: test_sub_v4i8_:
; R1: subu.qb ${{[0-9]+}}

define { i32 } @test_sub_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %sub = sub <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %sub to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; DSP-ASE doesn't have a v4i8 multiply instruction. Check that multiply node gets expanded.
; R2-LABEL: test_mul_v4i8_:
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}
; R2: mul ${{[0-9]+}}

define { i32 } @test_mul_v4i8_(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %1 = bitcast i32 %b.coerce to <4 x i8>
  %mul = mul <4 x i8> %0, %1
  %2 = bitcast <4 x i8> %mul to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: test_addsc:
; R1: addsc ${{[0-9]+}}
; R1: addwc ${{[0-9]+}}

define i64 @test_addsc(i64 %a, i64 %b) {
entry:
  %add = add nsw i64 %b, %a
  ret i64 %add
}

; R1-LABEL: shift1_v2i16_shl_:
; R1: shll.ph ${{[0-9]+}}, ${{[0-9]+}}, 15

define { i32 } @shift1_v2i16_shl_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %shl = shl <2 x i16> %0, <i16 15, i16 15>
  %1 = bitcast <2 x i16> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: shift1_v2i16_sra_:
; R1: shra.ph ${{[0-9]+}}, ${{[0-9]+}}, 15

define { i32 } @shift1_v2i16_sra_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %shr = ashr <2 x i16> %0, <i16 15, i16 15>
  %1 = bitcast <2 x i16> %shr to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: shift1_v2ui16_srl_:
; R1-NOT: shrl.ph
; R2-LABEL: shift1_v2ui16_srl_:
; R2: shrl.ph ${{[0-9]+}}, ${{[0-9]+}}, 15

define { i32 } @shift1_v2ui16_srl_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %shr = lshr <2 x i16> %0, <i16 15, i16 15>
  %1 = bitcast <2 x i16> %shr to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: shift1_v4i8_shl_:
; R1: shll.qb ${{[0-9]+}}, ${{[0-9]+}}, 7

define { i32 } @shift1_v4i8_shl_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %shl = shl <4 x i8> %0, <i8 7, i8 7, i8 7, i8 7>
  %1 = bitcast <4 x i8> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: shift1_v4i8_sra_:
; R1-NOT: shra.qb
; R2-LABEL: shift1_v4i8_sra_:
; R2: shra.qb ${{[0-9]+}}, ${{[0-9]+}}, 7

define { i32 } @shift1_v4i8_sra_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %shr = ashr <4 x i8> %0, <i8 7, i8 7, i8 7, i8 7>
  %1 = bitcast <4 x i8> %shr to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; R1-LABEL: shift1_v4ui8_srl_:
; R1: shrl.qb ${{[0-9]+}}, ${{[0-9]+}}, 7

define { i32 } @shift1_v4ui8_srl_(i32 %a0.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %shr = lshr <4 x i8> %0, <i8 7, i8 7, i8 7, i8 7>
  %1 = bitcast <4 x i8> %shr to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; Check that shift node is expanded if splat element size is not 16-bit.
;
; R1-LABEL: test_vector_splat_imm_v2q15:
; R1-NOT: shll.ph

define { i32 } @test_vector_splat_imm_v2q15(i32 %a.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %shl = shl <2 x i16> %0, <i16 0, i16 2>
  %1 = bitcast <2 x i16> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; Check that shift node is expanded if splat element size is not 8-bit.
;
; R1-LABEL: test_vector_splat_imm_v4i8:
; R1-NOT: shll.qb

define { i32 } @test_vector_splat_imm_v4i8(i32 %a.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %shl = shl <4 x i8> %0, <i8 0, i8 2, i8 0, i8 2>
  %1 = bitcast <4 x i8> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; Check that shift node is expanded if shift amount doesn't fit in 4-bit sa field.
;
; R1-LABEL: test_shift_amount_v2q15:
; R1-NOT: shll.ph

define { i32 } @test_shift_amount_v2q15(i32 %a.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %shl = shl <2 x i16> %0, <i16 16, i16 16>
  %1 = bitcast <2 x i16> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}

; Check that shift node is expanded if shift amount doesn't fit in 3-bit sa field.
;
; R1-LABEL: test_shift_amount_v4i8:
; R1-NOT: shll.qb

define { i32 } @test_shift_amount_v4i8(i32 %a.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <4 x i8>
  %shl = shl <4 x i8> %0, <i8 8, i8 8, i8 8, i8 8>
  %1 = bitcast <4 x i8> %shl to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %1, 0
  ret { i32 } %.fca.0.insert
}
