; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; SPIR-V (without sub-byte int extensions) widens sub-pow2 scalars to the next
; legal width by relabeling the LLT only, without inserting any sign-extension.
; Sign-sensitive ops (icmp slt/sle/sgt/sge, ashr, sdiv, srem) on such operands
; would then read the sign bit at the wrong position. The pre-legalizer must
; emit a sign-extend-in-register before the widening so the wide-width signed
; op observes the correct sign bit.

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#K4:]] = OpConstant %[[#I8]] 4
; CHECK-DAG: %[[#K8:]] = OpConstant %[[#I32]] 8

; ----------------------------------------------------------------------------
; icmp slt i4 against zero (the canonical XLA F4E2M1FN sign-bit-check pattern).
; CHECK: OpFunction
; CHECK: %[[#X1:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL1:]] = OpShiftLeftLogical %[[#I8]] %[[#X1]] %[[#K4]]
; CHECK: %[[#SX1:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL1]] %[[#K4]]
; CHECK: OpSLessThan {{%[0-9]+}} %[[#SX1]] {{%[0-9]+}}
define spir_kernel void @icmp_slt_i4_zero(i4 %x, ptr addrspace(1) %out) {
  %c = icmp slt i4 %x, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; icmp sle i4: same widening as slt, different SPIR-V opcode.
; CHECK: OpFunction
; CHECK: %[[#X_SLE:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL_SLE:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SLE]] %[[#K4]]
; CHECK: %[[#SX_SLE:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_SLE]] %[[#K4]]
; CHECK: OpSLessThanEqual {{%[0-9]+}} %[[#SX_SLE]] {{%[0-9]+}}
define spir_kernel void @icmp_sle_i4_zero(i4 %x, ptr addrspace(1) %out) {
  %c = icmp sle i4 %x, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; icmp sgt i4.
; CHECK: OpFunction
; CHECK: %[[#X_SGT:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL_SGT:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SGT]] %[[#K4]]
; CHECK: %[[#SX_SGT:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_SGT]] %[[#K4]]
; CHECK: OpSGreaterThan {{%[0-9]+}} %[[#SX_SGT]] {{%[0-9]+}}
define spir_kernel void @icmp_sgt_i4_zero(i4 %x, ptr addrspace(1) %out) {
  %c = icmp sgt i4 %x, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; icmp sge i4.
; CHECK: OpFunction
; CHECK: %[[#X_SGE:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL_SGE:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SGE]] %[[#K4]]
; CHECK: %[[#SX_SGE:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_SGE]] %[[#K4]]
; CHECK: OpSGreaterThanEqual {{%[0-9]+}} %[[#SX_SGE]] {{%[0-9]+}}
define spir_kernel void @icmp_sge_i4_zero(i4 %x, ptr addrspace(1) %out) {
  %c = icmp sge i4 %x, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; icmp slt i4 between two registers: both operands must be sign-extended.
; CHECK: OpFunction
; CHECK: %[[#X2:]] = OpFunctionParameter
; CHECK: %[[#Y2:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHLA2:]] = OpShiftLeftLogical %[[#I8]] %[[#X2]] %[[#K4]]
; CHECK: %[[#SXA2:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLA2]] %[[#K4]]
; CHECK: %[[#SHLB2:]] = OpShiftLeftLogical %[[#I8]] %[[#Y2]] %[[#K4]]
; CHECK: %[[#SXB2:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLB2]] %[[#K4]]
; CHECK: OpSLessThan {{%[0-9]+}} %[[#SXA2]] %[[#SXB2]]
define spir_kernel void @icmp_slt_i4_reg(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %c = icmp slt i4 %x, %y
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; ashr i4: arithmetic right shift on a widened operand needs the sign bit at
; the top of the wider register.
; CHECK: OpFunction
; CHECK: %[[#X3:]] = OpFunctionParameter
; CHECK: %[[#Y3:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHLA3:]] = OpShiftLeftLogical %[[#I8]] %[[#X3]] %[[#K4]]
; CHECK: %[[#SXA3:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLA3]] %[[#K4]]
; CHECK: %[[#SHLB3:]] = OpShiftLeftLogical %[[#I8]] %[[#Y3]] %[[#K4]]
; CHECK: %[[#SXB3:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLB3]] %[[#K4]]
; CHECK: OpShiftRightArithmetic %[[#I8]] %[[#SXA3]] %[[#SXB3]]
define spir_kernel void @ashr_i4(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %r = ashr i4 %x, %y
  %z = sext i4 %r to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; sdiv i4: signed division.
; CHECK: OpFunction
; CHECK: %[[#X4:]] = OpFunctionParameter
; CHECK: %[[#Y4:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHLA4:]] = OpShiftLeftLogical %[[#I8]] %[[#X4]] %[[#K4]]
; CHECK: %[[#SXA4:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLA4]] %[[#K4]]
; CHECK: %[[#SHLB4:]] = OpShiftLeftLogical %[[#I8]] %[[#Y4]] %[[#K4]]
; CHECK: %[[#SXB4:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLB4]] %[[#K4]]
; CHECK: OpSDiv %[[#I8]] %[[#SXA4]] %[[#SXB4]]
define spir_kernel void @sdiv_i4(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %r = sdiv i4 %x, %y
  %z = sext i4 %r to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; srem i4: signed remainder.
; CHECK: OpFunction
; CHECK: %[[#X5:]] = OpFunctionParameter
; CHECK: %[[#Y5:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHLA5:]] = OpShiftLeftLogical %[[#I8]] %[[#X5]] %[[#K4]]
; CHECK: %[[#SXA5:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLA5]] %[[#K4]]
; CHECK: %[[#SHLB5:]] = OpShiftLeftLogical %[[#I8]] %[[#Y5]] %[[#K4]]
; CHECK: %[[#SXB5:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHLB5]] %[[#K4]]
; CHECK: OpSRem %[[#I8]] %[[#SXA5]] %[[#SXB5]]
define spir_kernel void @srem_i4(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %r = srem i4 %x, %y
  %z = sext i4 %r to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; A non-pow2 width that widens to a different legal size: i24 -> i32, k = 8.
; CHECK: OpFunction
; CHECK: %[[#X6:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL6:]] = OpShiftLeftLogical %[[#I32]] %[[#X6]] %[[#K8]]
; CHECK: %[[#SX6:]] = OpShiftRightArithmetic %[[#I32]] %[[#SHL6]] %[[#K8]]
; CHECK: OpSLessThan {{%[0-9]+}} %[[#SX6]] {{%[0-9]+}}
define spir_kernel void @icmp_slt_i24_zero(i24 %x, ptr addrspace(1) %out) {
  %c = icmp slt i24 %x, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; Same vreg on both operands of one instruction: only one sign-extension pair
; is emitted and both operand slots reference it.
; CHECK: OpFunction
; CHECK: %[[#X_SAME:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL_SAME:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SAME]] %[[#K4]]
; CHECK: %[[#SX_SAME:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_SAME]] %[[#K4]]
; CHECK-NOT: OpShiftRightArithmetic
; CHECK: OpSDiv %[[#I8]] %[[#SX_SAME]] %[[#SX_SAME]]
define spir_kernel void @sdiv_i4_same_operand(i4 %x, ptr addrspace(1) %out) {
  %r = sdiv i4 %x, %x
  %z = sext i4 %r to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; Same vreg feeding two separate sign-sensitive instructions: each instruction
; needs its own sign-extension since G_SEXT_INREG is not hoisted.
; CHECK: OpFunction
; CHECK: %[[#X_SHARED:]] = OpFunctionParameter
; CHECK: %[[#Y_SHARED:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#SHL_A:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SHARED]] %[[#K4]]
; CHECK: %[[#SXA:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_A]] %[[#K4]]
; CHECK: %[[#SHL_B:]] = OpShiftLeftLogical %[[#I8]] %[[#Y_SHARED]] %[[#K4]]
; CHECK: %[[#SXB:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_B]] %[[#K4]]
; CHECK: OpSDiv %[[#I8]] %[[#SXA]] %[[#SXB]]
; CHECK: %[[#SHL_A2:]] = OpShiftLeftLogical %[[#I8]] %[[#X_SHARED]] %[[#K4]]
; CHECK: %[[#SXA2:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_A2]] %[[#K4]]
; CHECK: %[[#SHL_B2:]] = OpShiftLeftLogical %[[#I8]] %[[#Y_SHARED]] %[[#K4]]
; CHECK: %[[#SXB2:]] = OpShiftRightArithmetic %[[#I8]] %[[#SHL_B2]] %[[#K4]]
; CHECK: OpSRem %[[#I8]] %[[#SXA2]] %[[#SXB2]]
define spir_kernel void @sdiv_and_srem_i4_shared(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %q = sdiv i4 %x, %y
  %r = srem i4 %x, %y
  %s = add i4 %q, %r
  %z = sext i4 %s to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; trunc i64 to i24 feeding icmp slt: the i24 value is widened to i32 and must
; be sign-extended in the widened register before the signed compare.
; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK: %[[#T:]] = OpUConvert %[[#I32]] {{%[0-9]+}}
; CHECK: %[[#SHL_T:]] = OpShiftLeftLogical %[[#I32]] %[[#T]] %[[#K8]]
; CHECK: %[[#SX_T:]] = OpShiftRightArithmetic %[[#I32]] %[[#SHL_T]] %[[#K8]]
; CHECK: OpSLessThan {{%[0-9]+}} %[[#SX_T]] {{%[0-9]+}}
define spir_kernel void @trunc_i24_icmp_slt(i64 %a, ptr addrspace(1) %out) {
  %t = trunc i64 %a to i24
  %c = icmp slt i24 %t, 0
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; Negative test: unsigned compare must NOT emit sign-extension shifts.
; CHECK: OpFunction
; CHECK: %[[#X7:]] = OpFunctionParameter
; CHECK: OpFunctionParameter
; CHECK-NOT: OpShiftRightArithmetic
; CHECK: OpULessThan {{%[0-9]+}} %[[#X7]] {{%[0-9]+}}
define spir_kernel void @icmp_ult_i4_one(i4 %x, ptr addrspace(1) %out) {
  %c = icmp ult i4 %x, 1
  %r = sext i1 %c to i8
  store i8 %r, ptr addrspace(1) %out
  ret void
}

; ----------------------------------------------------------------------------
; Negative test: logical right shift must NOT emit sign-extension shifts.
; CHECK: OpFunction
; CHECK-NOT: OpShiftRightArithmetic
; CHECK: OpShiftRightLogical
define spir_kernel void @lshr_i4(i4 %x, i4 %y, ptr addrspace(1) %out) {
  %r = lshr i4 %x, %y
  %z = zext i4 %r to i32
  store i32 %z, ptr addrspace(1) %out
  ret void
}
