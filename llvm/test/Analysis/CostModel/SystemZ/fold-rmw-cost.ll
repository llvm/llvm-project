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

