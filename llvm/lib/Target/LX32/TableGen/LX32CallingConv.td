//===-- LX32CallingConv.td - LX32 Calling Convention ----------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines three things:
//
//   1. Callee-saved register sets (CSR_*) — used by the register allocator
//      and frame lowering to know which registers a function must preserve
//      across calls.
//
//   2. RetCC_LX32 — the return-value convention.  Governs which registers
//      carry values back from callee to caller.
//
//   3. CC_LX32 — the argument-passing convention.  Governs which registers
//      (or stack slots) carry values from caller to callee.
//
// The convention implemented here is ILP32: 32-bit pointers, 32-bit int,
// 64-bit long long passed in register pairs.  There is no hardware FPU,
// so floating-point values must be converted to integer before reaching
// this layer (handled in LX32ISelLowering).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Callee-saved register sets
//===----------------------------------------------------------------------===//

/// CSR_LX32_ILP32 — standard ILP32 callee-saved set.
///
/// A function that modifies any of these registers must save the original
/// value on entry and restore it before returning, so that the caller sees
/// them unchanged.
///
///   X1  (ra)     : return address — saved so nested calls can overwrite it
///   X8  (s0/fp)  : frame pointer / callee-saved s0
///   X9  (s1)     : callee-saved s1
///   X18-X27 (s2-s11) : callee-saved s2 through s11
def CSR_LX32_ILP32 : CalleeSavedRegs<(add X1, X8, X9, (sequence "X%u", 18, 27))>;

/// CSR_NoRegs — empty set.
/// Used for functions that are known to clobber no registers (e.g., naked
/// functions or explicit no-callee-save calling conventions).
def CSR_NoRegs : CalleeSavedRegs<(add)>;

/// CSR_IPRA — minimal set for Inter-Procedural Register Allocation.
/// When IPRA is active, LLVM computes per-function clobber sets and uses
/// this as the baseline: only ra (X1) is unconditionally preserved.
def CSR_IPRA : CalleeSavedRegs<(add X1)>;

//===----------------------------------------------------------------------===//
// Return-value convention  —  RetCC_LX32
//
// Promotion: i1/i8/i16 are widened to i32 before register assignment.
//
// i32 return values: up to two values fit in a0 (X10) and a1 (X11).
//   Single i32  → a0
//   Pair i32    → a0, a1  (e.g., struct { int x, y; } returned by value)
//
// i64 return value: the low 32 bits go in a0 (X10), the high 32 bits go in
//   a1 (X11).  CCAssignToRegWithShadow<[X10], [X11]> models this pair:
//   allocating X10 also reserves X11 as its shadow, preventing any other
//   value from landing there.
//
// Overflow: anything that does not fit in registers goes to the stack at
//   4-byte alignment.  This case is unusual for return values (most ABIs
//   handle large returns via a hidden pointer argument), but the rule is
//   included for completeness.
//===----------------------------------------------------------------------===//

def RetCC_LX32 : CallingConv<[
  // Widen sub-word integer types to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i32 return values: a0, then a1.
  CCIfType<[i32], CCAssignToReg<[X10, X11]>>,

  // i64 return value: low word in a0 (X10), high word in a1 (X11).
  CCIfType<[i64], CCAssignToRegWithShadow<[X10], [X11]>>,

  // Overflow to stack: 4-byte size, 4-byte alignment.
  CCAssignToStack<4, 4>
]>;

//===----------------------------------------------------------------------===//
// Argument-passing convention  —  CC_LX32
//
// Promotion: i1/i8/i16 are widened to i32 before register assignment,
//   matching the ILP32 ABI requirement that sub-word arguments occupy a
//   full register.
//
// i32 arguments: up to eight values are passed in a0-a7 (X10-X17) in order.
//   The ninth and subsequent i32 arguments spill to the stack at 4-byte
//   alignment.
//
// i64 arguments: passed in aligned register pairs — only even-indexed
//   argument registers are valid low-word slots (a0, a2, a4, a6).
//   CCAssignToRegWithShadow pairs each even register with the next odd
//   register as its shadow.  If a pair cannot fit (e.g., only a7 remains),
//   the i64 spills entirely to the stack at 8-byte alignment.
//
// Note on floats: lx32 has no FPU.  Float arguments are converted to
//   integer in LX32ISelLowering before reaching this convention, so no
//   float rules are needed here.
//===----------------------------------------------------------------------===//

def CC_LX32 : CallingConv<[
  // Widen sub-word integer types to i32.
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i32 arguments: a0-a7 in order, then stack.
  CCIfType<[i32], CCAssignToReg<[X10, X11, X12, X13, X14, X15, X16, X17]>>,

  // i64 arguments: aligned register pairs (a0+a1, a2+a3, a4+a5, a6+a7).
  CCIfType<[i64], CCAssignToRegWithShadow<[X10, X12, X14, X16],
                                          [X11, X13, X15, X17]>>,

  // Stack overflow rules: i32 at 4-byte, i64 at 8-byte alignment.
  CCIfType<[i32], CCAssignToStack<4, 4>>,
  CCIfType<[i64], CCAssignToStack<8, 8>>
]>;
