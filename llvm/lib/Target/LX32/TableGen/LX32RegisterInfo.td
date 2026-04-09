//===-- LX32RegisterInfo.td - LX32 Register Definitions -------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the LX32 register file: 32 general-purpose 32-bit
// registers (X0-X31), their ABI aliases, two register classes (GPR and
// GPRNOX0), and an alternate-name index for ABI names.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Register base class
//===----------------------------------------------------------------------===//

let Namespace = "LX32" in {
class LX32Reg<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}
}

//===----------------------------------------------------------------------===//
// Alternate name index
//
// ABIRegAltName provides the standard ABI aliases (zero, ra, sp, ...).
// FallbackRegAltNameIndex = NoRegAltName means that registers without an
// explicit ABI alias fall back to the canonical hardware name.
//===----------------------------------------------------------------------===//

let FallbackRegAltNameIndex = NoRegAltName in
def ABIRegAltName : RegAltNameIndex;

//===----------------------------------------------------------------------===//
// Register definitions
//
// Registers are listed in hardware-encoding order (X0-X31).
// ABI aliases follow standard ILP32 conventions.
//   X0        : hardwired zero — marked isConstant so the allocator never
//               assigns it as a destination.
//   X1  (ra)  : return address
//   X2  (sp)  : stack pointer
//   X3  (gp)  : global pointer
//   X4  (tp)  : thread pointer
//   X5-X7     : temporaries t0-t2 (caller-saved)
//   X8  (s0/fp): frame pointer / callee-saved s0
//   X9  (s1)  : callee-saved
//   X10-X17   : arguments a0-a7 / return values a0-a1
//   X18-X27   : callee-saved s2-s11
//   X28-X31   : temporaries t3-t6 (caller-saved)
//===----------------------------------------------------------------------===//

let RegAltNameIndices = [ABIRegAltName] in {
  let isConstant = true in
  def X0  : LX32Reg<0,  "x0",  ["zero"]>;

  def X1  : LX32Reg<1,  "x1",  ["ra"]>;
  def X2  : LX32Reg<2,  "x2",  ["sp"]>;
  def X3  : LX32Reg<3,  "x3",  ["gp"]>;
  def X4  : LX32Reg<4,  "x4",  ["tp"]>;
  def X5  : LX32Reg<5,  "x5",  ["t0"]>;
  def X6  : LX32Reg<6,  "x6",  ["t1"]>;
  def X7  : LX32Reg<7,  "x7",  ["t2"]>;
  def X8  : LX32Reg<8,  "x8",  ["s0", "fp"]>;
  def X9  : LX32Reg<9,  "x9",  ["s1"]>;
  def X10 : LX32Reg<10, "x10", ["a0"]>;
  def X11 : LX32Reg<11, "x11", ["a1"]>;
  def X12 : LX32Reg<12, "x12", ["a2"]>;
  def X13 : LX32Reg<13, "x13", ["a3"]>;
  def X14 : LX32Reg<14, "x14", ["a4"]>;
  def X15 : LX32Reg<15, "x15", ["a5"]>;
  def X16 : LX32Reg<16, "x16", ["a6"]>;
  def X17 : LX32Reg<17, "x17", ["a7"]>;
  def X18 : LX32Reg<18, "x18", ["s2"]>;
  def X19 : LX32Reg<19, "x19", ["s3"]>;
  def X20 : LX32Reg<20, "x20", ["s4"]>;
  def X21 : LX32Reg<21, "x21", ["s5"]>;
  def X22 : LX32Reg<22, "x22", ["s6"]>;
  def X23 : LX32Reg<23, "x23", ["s7"]>;
  def X24 : LX32Reg<24, "x24", ["s8"]>;
  def X25 : LX32Reg<25, "x25", ["s9"]>;
  def X26 : LX32Reg<26, "x26", ["s10"]>;
  def X27 : LX32Reg<27, "x27", ["s11"]>;
  def X28 : LX32Reg<28, "x28", ["t3"]>;
  def X29 : LX32Reg<29, "x29", ["t4"]>;
  def X30 : LX32Reg<30, "x30", ["t5"]>;
  def X31 : LX32Reg<31, "x31", ["t6"]>;
}

//===----------------------------------------------------------------------===//
// Register classes
//
// GPR — all 32 registers, ordered by allocation priority:
//   1. Argument / return-value registers (a0-a7): highest priority so that
//      the allocator places values in calling-convention registers first,
//      minimizing moves at call boundaries.
//   2. Caller-saved temporaries (t0-t2, t3-t6): free to use without saving.
//   3. Callee-saved registers (s0-s11): usable but must be saved/restored.
//   4. Special pointers (ra, sp, gp, tp): the allocator avoids these by
//      convention; they are reserved in LX32RegisterInfo::getReservedRegs.
//   5. Zero register (x0): listed last so the allocator never picks it as
//      a destination; it is also marked isConstant above.
//
// GPRNOX0 — same as GPR minus X0. Used for instruction operands that must
//   not target the zero register (e.g., JALR destination when used as call).
//   X0 is excluded via (sub GPR, X0) so TableGen keeps allocation order.
//===----------------------------------------------------------------------===//

def GPR : RegisterClass<"LX32", [i32], 32, (add
  // Arguments and return values — highest priority
  X10, X11, X12, X13, X14, X15, X16, X17,
  // Caller-saved temporaries
  X5, X6, X7,
  X28, X29, X30, X31,
  // Callee-saved
  X8, X9,
  X18, X19, X20, X21, X22, X23, X24, X25, X26, X27,
  // Special pointers — allocator avoids these (reserved in RegisterInfo)
  X1, X2, X3, X4,
  // Zero register — never allocatable
  X0
)>;

def GPRNOX0 : RegisterClass<"LX32", [i32], 32, (sub GPR, X0)>;
