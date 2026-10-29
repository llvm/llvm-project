; RUN: opt -S -mtriple=s390x-unknown-linux -mcpu=z17 -passes='print<cost-model>' \
; RUN:  -disable-output %s 2>&1 | FileCheck %s
;
; Test the SystemZ cost model for scalar Read-Modify-Write (RMW) folding.
; A load, a scalar arithmetic operation (ADD, SUB, AND, OR, XOR) with
; an immediate, and a store all target the same memory location and have
; no external uses, the cost of the arithmetic and store insn should be 0.

define void @test_and(ptr %p) {
; CHECK-LABEL: 'test_and'
; CHECK: cost of 0 {{.*}} and i8
; CHECK: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = and i8 %v, 1
  store i8 %res, ptr %p
  ret void
}

define void @test_or(ptr %p) {
; CHECK-LABEL: 'test_or'
; CHECK: cost of 0 {{.*}} or i8
; CHECK: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = or i8 %v, 1
  store i8 %res, ptr %p
  ret void
}

define void @test_xor(ptr %p) {
; CHECK-LABEL: 'test_xor'
; CHECK: cost of 0 {{.*}} xor i8
; CHECK: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = xor i8 %v, 1
  store i8 %res, ptr %p
  ret void
}

define void @test_add(ptr %p) {
; CHECK-LABEL: 'test_add'
; CHECK: cost of 0 {{.*}} add i32
; CHECK: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = add i32 %v, 1
  store i32 %res, ptr %p
  ret void
}

define void @test_sub(ptr %p) {
; CHECK-LABEL: 'test_sub'
; CHECK: cost of 0 {{.*}} sub i32
; CHECK: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = sub i32 %v, 1
  store i32 %res, ptr %p
  ret void
}

; Subtraction is not commutative.
define void @test_sub_neg_imm_lhs(ptr %p) {
; CHECK-LABEL: 'test_sub_neg_imm_lhs'
; CHECK-NOT: cost of 0 {{.*}} sub i32
; CHECK-NOT: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = sub i32 1, %v
  store i32 %res, ptr %p
  ret void
}

; Different Addresses.
define void @test_diff_addr(ptr %p, ptr %q) {
; CHECK-LABEL: 'test_diff_addr'
; CHECK-NOT: cost of 0 {{.*}} add i32
; CHECK-NOT: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = add i32 %v, 1
  store i32 %res, ptr %q
  ret void
}

; Multi-use of Arithmetic Result.
define i32 @test_multi_use_arith(ptr %p) {
; CHECK-LABEL: 'test_multi_use_arith'
; CHECK-NOT: cost of 0 {{.*}} add i8
; CHECK-NOT: cost of 0 {{.*}} store i8
  %v = load i32, ptr %p
  %res = add i32 %v, 1
  store i32 %res, ptr %p
  ret i32 %res
}

; Multi-use of Load Result.
define i8 @test_multi_use_load(ptr %p) {
; CHECK-LABEL: 'test_multi_use_load'
; CHECK-NOT: cost of 0 {{.*}} and i8
; CHECK-NOT: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = and i8 %v, 1
  store i8 %res, ptr %p
  %use2 = and i8 %v, %res
  ret i8 %use2
}

; Neither Operand is Immediate.
define void @test_no_immediate(ptr %p, i8 %reg) {
; CHECK-LABEL: 'test_no_immediate'
; CHECK-NOT: cost of 0 {{.*}} and i8
; CHECK-NOT: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = and i8 %v, %reg
  store i8 %res, ptr %p
  ret void
}

; Both Operands are Immediate.
define void @test_both_immediate(ptr %p) {
; CHECK-LABEL: 'test_both_immediate'
; CHECK-NOT: cost of 0 {{.*}} or i8
; CHECK-NOT: cost of 0 {{.*}} store i8
  %v = load i8, ptr %p
  %res = or i8 5, 10
  store i8 %res, ptr %p
  ret void
}

; Volatile Load/Store.
define void @test_volatile(ptr %p) {
; CHECK-LABEL: 'test_volatile'
; CHECK-NOT: cost of 0 {{.*}} xor i8
; CHECK-NOT: cost of 0 {{.*}} store i8
  %v = load volatile i8, ptr %p
  %res = xor i8 %v, 1
  store volatile i8 %res, ptr %p
  ret void
}

; Vector types cost should not be changed.
define void @test_vector_no_fold(<4 x i32> %val, ptr %p) {
; CHECK-LABEL: 'test_vector_no_fold'
; CHECK-NOT: cost of 0 {{.*}} add <4 x i32>
; CHECK-NOT: cost of 0 {{.*}} store <4 x i32>
  %v = load <4 x i32>, ptr %p
  %res = add <4 x i32> %v, %val
  store <4 x i32> %res, ptr %p
  ret void
}

; Logical operations for i16/i32/i64 types.

; Modifies only LSB bits.
define void @test_and_i16_valid(ptr %p) {
; CHECK-LABEL: 'test_and_i16_valid'
; CHECK: cost of 0 {{.*}} and i16 %v, -255
; CHECK: cost of 0 {{.*}} store i16
  %v = load i16, ptr %p
  %res = and i16 %v, 65281 ; 0xff01
  store i16 %res, ptr %p
  ret void
}

; Clears bit 8, just above LSB window.
define void @test_and_i16_invalid_above(ptr %p) {
; CHECK-LABEL: 'test_and_i16_invalid_above'
; CHECK: cost of 1 {{.*}} and i16 %v, -511
; CHECK: cost of 1 {{.*}} store i16
  %v = load i16, ptr %p
  %res = and i16 %v, 65025 ; 0xfe01
  store i16 %res, ptr %p
  ret void
}

; Active bits <=8.
define void @test_or_i16_valid(ptr %p) {
; CHECK-LABEL: 'test_or_i16_valid'
; CHECK: cost of 0 {{.*}} or i16 %v, 255
; CHECK: cost of 0 {{.*}} store i16
  %v = load i16, ptr %p
  %res = or i16 %v, 255 ; 0x00ff
  store i16 %res, ptr %p
  ret void
}

; Active bits > 8.
define void @test_or_i16_invalid(ptr %p) {
; CHECK-LABEL: 'test_or_i16_invalid'
; CHECK: cost of 1 {{.*}} or i16 %v, 256
; CHECK: cost of 1 {{.*}} store i16
  %v = load i16, ptr %p
  %res = or i16 %v, 256 ; 0x0100
  store i16 %res, ptr %p
  ret void
}

; Modifies only LSB bits.
define void @test_and_i32_valid(ptr %p) {
; CHECK-LABEL: 'test_and_i32_valid'
; CHECK: cost of 0 {{.*}} and i32 %v, -255
; CHECK: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = and i32 %v, 4294967041 ; 0xffffff01
  store i32 %res, ptr %p
  ret void
}

; Zero-extended i16 context.
define void @test_and_i32_zext_invalid(ptr %p) {
; CHECK-LABEL: 'test_and_i32_zext_invalid'
; CHECK: cost of 1 {{.*}} and i32 %v, 65281
; CHECK: cost of 1 {{.*}} store i32
  %v = load i32, ptr %p
  %res = and i32 %v, 65281 ; 0x0000ff01
  store i32 %res, ptr %p
  ret void
}

; One bit down from zero-extended threshold.
define void @test_and_i32_zext_below_invalid(ptr %p) {
; CHECK-LABEL: 'test_and_i32_zext_below_invalid'
; CHECK: cost of 1 {{.*}} and i32 %v, 65025
; CHECK: cost of 1 {{.*}} store i32
  %v = load i32, ptr %p
  %res = and i32 %v, 65025 ; 0x0000fe01
  store i32 %res, ptr %p
  ret void
}

; One bit above zero-extended threshold.
define void @test_and_i32_zext_above_invalid(ptr %p) {
; CHECK-LABEL: 'test_and_i32_zext_above_invalid'
; CHECK: cost of 1 {{.*}} and i32 %v, 130817
; CHECK: cost of 1 {{.*}} store i32
  %v = load i32, ptr %p
  %res = and i32 %v, 130817 ; 0x0001ff01
  store i32 %res, ptr %p
  ret void
}

; Modifies bits outside LSB.
define void @test_and_i32_invalid(ptr %p) {
; CHECK-LABEL: 'test_and_i32_invalid'
; CHECK: cost of 1 {{.*}} and i32 %v, 16
; CHECK: cost of 1 {{.*}} store i32
  %v = load i32, ptr %p
  %res = and i32 %v, 16
  store i32 %res, ptr %p
  ret void
}

; Active bits <=8.
define void @test_xor_i32_valid(ptr %p) {
; CHECK-LABEL: 'test_xor_i32_valid'
; CHECK: cost of 0 {{.*}} xor i32 %v, 255
; CHECK: cost of 0 {{.*}} store i32
  %v = load i32, ptr %p
  %res = xor i32 %v, 255 ; 0x000000ff
  store i32 %res, ptr %p
  ret void
}

; Active bits > 8.
define void @test_xor_i32_invalid_above(ptr %p) {
; CHECK-LABEL: 'test_xor_i32_invalid_above'
; CHECK: cost of 1 {{.*}} xor i32 %v, 256
; CHECK: cost of 1 {{.*}} store i32
  %v = load i32, ptr %p
  %res = xor i32 %v, 256 ; 0x00000100
  store i32 %res, ptr %p
  ret void
}

; Modifies only LSB bits.
define void @test_and_i64_valid(ptr %p) {
; CHECK-LABEL: 'test_and_i64_valid'
; CHECK: cost of 0 {{.*}} and i64 %v, -255
; CHECK: cost of 0 {{.*}} store i64
  %v = load i64, ptr %p
  %res = and i64 %v, -255 ; 0xffffffffffffff01
  store i64 %res, ptr %p
  ret void
}

; Clears bit 8.
define void @test_and_i64_invalid_above(ptr %p) {
; CHECK-LABEL: 'test_and_i64_invalid_above'
; CHECK: cost of 1 {{.*}} and i64 %v, -511
; CHECK: cost of 1 {{.*}} store i64
  %v = load i64, ptr %p
  %res = and i64 %v, -511 ; 0xfffffffffffffffe01
  store i64 %res, ptr %p
  ret void
}

; Active bits <=8.
define void @test_or_i64_valid(ptr %p) {
; CHECK-LABEL: 'test_or_i64_valid'
; CHECK: cost of 0 {{.*}} or i64 %v, 255
; CHECK: cost of 0 {{.*}} store i64
  %v = load i64, ptr %p
  %res = or i64 %v, 255 ; 0x00000000000000ff
  store i64 %res, ptr %p
  ret void
}

; Active bits > 8.
define void @test_or_i64_invalid(ptr %p) {
; CHECK-LABEL: 'test_or_i64_invalid'
; CHECK: cost of 1 {{.*}} or i64 %v, 256
  %v = load i64, ptr %p
  %res = or i64 %v, 256 ; 0x0000000000000100
  store i64 %res, ptr %p
  ret void
}
