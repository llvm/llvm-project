; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

@var1_32 = global i32 0
@var2_32 = global i32 0

@var1_64 = global i64 0
@var2_64 = global i64 0

define void @logical_32bit() minsize {
; CHECK-LABEL: logical_32bit:
  %val1 = load i32, i32* @var1_32
  %val2 = load i32, i32* @var2_32

  ; First check basic and/bic/or/orn/eor/eon patterns with no shift
  %neg_val2 = xor i32 -1, %val2

  %and_noshift = and i32 %val1, %val2
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %and_noshift, i32* @var1_32
  %bic_noshift = and i32 %neg_val2, %val1
; CHECK: bic {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %bic_noshift, i32* @var1_32

  %or_noshift = or i32 %val1, %val2
; CHECK: orr {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %or_noshift, i32* @var1_32
  %orn_noshift = or i32 %neg_val2, %val1
; CHECK: orn {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %orn_noshift, i32* @var1_32

  %xor_noshift = xor i32 %val1, %val2
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %xor_noshift, i32* @var1_32
  %xorn_noshift = xor i32 %neg_val2, %val1
; CHECK: eon {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  store volatile i32 %xorn_noshift, i32* @var1_32

  ; Check the maximum shift on each
  %operand_lsl31 = shl i32 %val2, 31
  %neg_operand_lsl31 = xor i32 -1, %operand_lsl31

  %and_lsl31 = and i32 %val1, %operand_lsl31
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %and_lsl31, i32* @var1_32
  %bic_lsl31 = and i32 %val1, %neg_operand_lsl31
; CHECK: bic {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %bic_lsl31, i32* @var1_32

  %or_lsl31 = or i32 %val1, %operand_lsl31
; CHECK: orr {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %or_lsl31, i32* @var1_32
  %orn_lsl31 = or i32 %val1, %neg_operand_lsl31
; CHECK: orn {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %orn_lsl31, i32* @var1_32

  %xor_lsl31 = xor i32 %val1, %operand_lsl31
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %xor_lsl31, i32* @var1_32
  %xorn_lsl31 = xor i32 %val1, %neg_operand_lsl31
; CHECK: eon {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsl #31
  store volatile i32 %xorn_lsl31, i32* @var1_32

  ; Check other shifts on a subset
  %operand_asr10 = ashr i32 %val2, 10
  %neg_operand_asr10 = xor i32 -1, %operand_asr10

  %bic_asr10 = and i32 %val1, %neg_operand_asr10
; CHECK: bic {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #10
  store volatile i32 %bic_asr10, i32* @var1_32
  %xor_asr10 = xor i32 %val1, %operand_asr10
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, asr #10
  store volatile i32 %xor_asr10, i32* @var1_32

  %operand_lsr1 = lshr i32 %val2, 1
  %neg_operand_lsr1 = xor i32 -1, %operand_lsr1

  %orn_lsr1 = or i32 %val1, %neg_operand_lsr1
; CHECK: orn {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #1
  store volatile i32 %orn_lsr1, i32* @var1_32
  %xor_lsr1 = xor i32 %val1, %operand_lsr1
; CHECK: eor {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, lsr #1
  store volatile i32 %xor_lsr1, i32* @var1_32

  %operand_ror20_big = shl i32 %val2, 12
  %operand_ror20_small = lshr i32 %val2, 20
  %operand_ror20 = or i32 %operand_ror20_big, %operand_ror20_small
  %neg_operand_ror20 = xor i32 -1, %operand_ror20

  %xorn_ror20 = xor i32 %val1, %neg_operand_ror20
; CHECK: eon {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, ror #20
  store volatile i32 %xorn_ror20, i32* @var1_32
  %and_ror20 = and i32 %val1, %operand_ror20
; CHECK: and {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, ror #20
  store volatile i32 %and_ror20, i32* @var1_32

  ret void
}

define void @logical_64bit() minsize {
; CHECK-LABEL: logical_64bit:
  %val1 = load i64, i64* @var1_64
  %val2 = load i64, i64* @var2_64

  ; First check basic and/bic/or/orn/eor/eon patterns with no shift
  %neg_val2 = xor i64 -1, %val2

  %and_noshift = and i64 %val1, %val2
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %and_noshift, i64* @var1_64
  %bic_noshift = and i64 %neg_val2, %val1
; CHECK: bic {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %bic_noshift, i64* @var1_64

  %or_noshift = or i64 %val1, %val2
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %or_noshift, i64* @var1_64
  %orn_noshift = or i64 %neg_val2, %val1
; CHECK: orn {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %orn_noshift, i64* @var1_64

  %xor_noshift = xor i64 %val1, %val2
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %xor_noshift, i64* @var1_64
  %xorn_noshift = xor i64 %neg_val2, %val1
; CHECK: eon {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  store volatile i64 %xorn_noshift, i64* @var1_64

  ; Check the maximum shift on each
  %operand_lsl63 = shl i64 %val2, 63
  %neg_operand_lsl63 = xor i64 -1, %operand_lsl63

  %and_lsl63 = and i64 %val1, %operand_lsl63
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %and_lsl63, i64* @var1_64
  %bic_lsl63 = and i64 %val1, %neg_operand_lsl63
; CHECK: bic {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %bic_lsl63, i64* @var1_64

  %or_lsl63 = or i64 %val1, %operand_lsl63
; CHECK: orr {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %or_lsl63, i64* @var1_64
  %orn_lsl63 = or i64 %val1, %neg_operand_lsl63
; CHECK: orn {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %orn_lsl63, i64* @var1_64

  %xor_lsl63 = xor i64 %val1, %operand_lsl63
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %xor_lsl63, i64* @var1_64
  %xorn_lsl63 = xor i64 %val1, %neg_operand_lsl63
; CHECK: eon {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
  store volatile i64 %xorn_lsl63, i64* @var1_64

  ; Check other shifts on a subset
  %operand_asr10 = ashr i64 %val2, 10
  %neg_operand_asr10 = xor i64 -1, %operand_asr10

  %bic_asr10 = and i64 %val1, %neg_operand_asr10
; CHECK: bic {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #10
  store volatile i64 %bic_asr10, i64* @var1_64
  %xor_asr10 = xor i64 %val1, %operand_asr10
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, asr #10
  store volatile i64 %xor_asr10, i64* @var1_64

  %operand_lsr1 = lshr i64 %val2, 1
  %neg_operand_lsr1 = xor i64 -1, %operand_lsr1

  %orn_lsr1 = or i64 %val1, %neg_operand_lsr1
; CHECK: orn {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #1
  store volatile i64 %orn_lsr1, i64* @var1_64
  %xor_lsr1 = xor i64 %val1, %operand_lsr1
; CHECK: eor {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, lsr #1
  store volatile i64 %xor_lsr1, i64* @var1_64

  ; Construct a rotate-right from a bunch of other logical
  ; operations. DAGCombiner should ensure we the ROTR during
  ; selection
  %operand_ror20_big = shl i64 %val2, 44
  %operand_ror20_small = lshr i64 %val2, 20
  %operand_ror20 = or i64 %operand_ror20_big, %operand_ror20_small
  %neg_operand_ror20 = xor i64 -1, %operand_ror20

  %xorn_ror20 = xor i64 %val1, %neg_operand_ror20
; CHECK: eon {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, ror #20
  store volatile i64 %xorn_ror20, i64* @var1_64
  %and_ror20 = and i64 %val1, %operand_ror20
; CHECK: and {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}, ror #20
  store volatile i64 %and_ror20, i64* @var1_64

  ret void
}

define void @flag_setting() {
; CHECK-LABEL: flag_setting:
  %val1 = load i64, i64* @var1_64
  %val2 = load i64, i64* @var2_64

; CHECK: tst {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: b.gt .L
  %simple_and = and i64 %val1, %val2
  %tst1 = icmp sgt i64 %simple_and, 0
  br i1 %tst1, label %ret, label %test2, !prof !1

test2:
; CHECK: tst {{x[0-9]+}}, {{x[0-9]+}}, lsl #63
; CHECK: b.lt .L
  %shifted_op = shl i64 %val2, 63
  %shifted_and = and i64 %val1, %shifted_op
  %tst2 = icmp slt i64 %shifted_and, 0
  br i1 %tst2, label %ret, label %test3, !prof !1

test3:
; CHECK: tst {{x[0-9]+}}, {{x[0-9]+}}, asr #12
; CHECK: b.gt .L
  %asr_op = ashr i64 %val2, 12
  %asr_and = and i64 %asr_op, %val1
  %tst3 = icmp sgt i64 %asr_and, 0
  br i1 %tst3, label %ret, label %other_exit, !prof !1

other_exit:
  store volatile i64 %val1, i64* @var1_64
  ret void
ret:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 1}
