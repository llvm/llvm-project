; NOTE: This test case is borrowed from undef-ops.ll
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define i32 @add_poison_rhs(i32 %x) {
; CHECK-LABEL: add_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = add i32 %x, poison
  ret i32 %r
}

define <4 x i32> @add_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: add_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = add <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @add_poison_lhs(i32 %x) {
; CHECK-LABEL: add_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = add i32 poison, %x
  ret i32 %r
}

define <4 x i32> @add_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: add_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = add <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @sub_poison_rhs(i32 %x) {
; CHECK-LABEL: sub_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sub i32 %x, poison
  ret i32 %r
}

define <4 x i32> @sub_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: sub_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sub <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @sub_poison_lhs(i32 %x) {
; CHECK-LABEL: sub_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sub i32 poison, %x
  ret i32 %r
}

define <4 x i32> @sub_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: sub_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sub <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @mul_poison_rhs(i32 %x) {
; CHECK-LABEL: mul_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = mul i32 %x, poison
  ret i32 %r
}

define <4 x i32> @mul_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: mul_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = mul <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @mul_poison_lhs(i32 %x) {
; CHECK-LABEL: mul_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = mul i32 poison, %x
  ret i32 %r
}

define <4 x i32> @mul_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: mul_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = mul <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @sdiv_poison_rhs(i32 %x) {
; CHECK-LABEL: sdiv_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sdiv i32 %x, poison
  ret i32 %r
}

define <4 x i32> @sdiv_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: sdiv_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = sdiv <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @sdiv_poison_lhs(i32 %x) {
; CHECK-LABEL: sdiv_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = sdiv i32 poison, %x
  ret i32 %r
}

define <4 x i32> @sdiv_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: sdiv_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = sdiv <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @udiv_poison_rhs(i32 %x) {
; CHECK-LABEL: udiv_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = udiv i32 %x, poison
  ret i32 %r
}

define <4 x i32> @udiv_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: udiv_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = udiv <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @udiv_poison_lhs(i32 %x) {
; CHECK-LABEL: udiv_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = udiv i32 poison, %x
  ret i32 %r
}

define <4 x i32> @udiv_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: udiv_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = udiv <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @srem_poison_rhs(i32 %x) {
; CHECK-LABEL: srem_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = srem i32 %x, poison
  ret i32 %r
}

define <4 x i32> @srem_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: srem_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = srem <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @srem_poison_lhs(i32 %x) {
; CHECK-LABEL: srem_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = srem i32 poison, %x
  ret i32 %r
}

define <4 x i32> @srem_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: srem_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = srem <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @urem_poison_rhs(i32 %x) {
; CHECK-LABEL: urem_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = urem i32 %x, poison
  ret i32 %r
}

define <4 x i32> @urem_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: urem_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = urem <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @urem_poison_lhs(i32 %x) {
; CHECK-LABEL: urem_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = urem i32 poison, %x
  ret i32 %r
}

define <4 x i32> @urem_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: urem_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = urem <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @ashr_poison_rhs(i32 %x) {
; CHECK-LABEL: ashr_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = ashr i32 %x, poison
  ret i32 %r
}

define <4 x i32> @ashr_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: ashr_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = ashr <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @ashr_poison_lhs(i32 %x) {
; CHECK-LABEL: ashr_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = ashr i32 poison, %x
  ret i32 %r
}

define <4 x i32> @ashr_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: ashr_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = ashr <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @lshr_poison_rhs(i32 %x) {
; CHECK-LABEL: lshr_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = lshr i32 %x, poison
  ret i32 %r
}

define <4 x i32> @lshr_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: lshr_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = lshr <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @lshr_poison_lhs(i32 %x) {
; CHECK-LABEL: lshr_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = lshr i32 poison, %x
  ret i32 %r
}

define <4 x i32> @lshr_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: lshr_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = lshr <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @shl_poison_rhs(i32 %x) {
; CHECK-LABEL: shl_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = shl i32 %x, poison
  ret i32 %r
}

define <4 x i32> @shl_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: shl_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = shl <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @shl_poison_lhs(i32 %x) {
; CHECK-LABEL: shl_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = shl i32 poison, %x
  ret i32 %r
}

define <4 x i32> @shl_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: shl_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = shl <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @and_poison_rhs(i32 %x) {
; CHECK-LABEL: and_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = and i32 %x, poison
  ret i32 %r
}

define <4 x i32> @and_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: and_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = and <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @and_poison_lhs(i32 %x) {
; CHECK-LABEL: and_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorl %eax, %eax
; CHECK-NEXT:    retq
  %r = and i32 poison, %x
  ret i32 %r
}

define <4 x i32> @and_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: and_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    xorps %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = and <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @or_poison_rhs(i32 %x) {
; CHECK-LABEL: or_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl $-1, %eax
; CHECK-NEXT:    retq
  %r = or i32 %x, poison
  ret i32 %r
}

define <4 x i32> @or_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: or_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = or <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @or_poison_lhs(i32 %x) {
; CHECK-LABEL: or_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl $-1, %eax
; CHECK-NEXT:    retq
  %r = or i32 poison, %x
  ret i32 %r
}

define <4 x i32> @or_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: or_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pcmpeqd %xmm0, %xmm0
; CHECK-NEXT:    retq
  %r = or <4 x i32> poison, %x
  ret <4 x i32> %r
}

define i32 @xor_poison_rhs(i32 %x) {
; CHECK-LABEL: xor_poison_rhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = xor i32 %x, poison
  ret i32 %r
}

define <4 x i32> @xor_poison_rhs_vec(<4 x i32> %x) {
; CHECK-LABEL: xor_poison_rhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = xor <4 x i32> %x, poison
  ret <4 x i32> %r
}

define i32 @xor_poison_lhs(i32 %x) {
; CHECK-LABEL: xor_poison_lhs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = xor i32 poison, %x
  ret i32 %r
}

define <4 x i32> @xor_poison_lhs_vec(<4 x i32> %x) {
; CHECK-LABEL: xor_poison_lhs_vec:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %r = xor <4 x i32> poison, %x
  ret <4 x i32> %r
}

; This would crash because the shift amount is an i8 operand,
; but the result of the shift is i32. We can't just propagate
; the existing poison as the result.

define i1 @poison_operand_size_not_same_as_result() {
; CHECK-LABEL: poison_operand_size_not_same_as_result:
; CHECK:       # %bb.0:
; CHECK-NEXT:    retq
  %sh = shl i32 7, poison
  %cmp = icmp eq i32 0, %sh
  ret i1 %cmp
}

