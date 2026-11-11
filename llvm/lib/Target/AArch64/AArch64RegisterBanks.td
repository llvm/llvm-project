//=- AArch64RegisterBank.td - Describe the AArch64 Banks -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// General Purpose Registers: W, X.
def GPRRegBank : RegisterBank<"GPR", [XSeqPairsClass]>;

/// Floating Point, Vector, Scalable Vector Registers: B, H, S, D, Q, Z.
def FPRRegBank : RegisterBank<"FPR", [QQQQ, ZPR]> {
  let ScalableMappingSize = 128;
  let IgnoredPartialMappings=[PartialMapping<8>, PartialMapping<192>, PartialMapping<384>];
}

/// Conditional register: NZCV.
def CCRegBank : RegisterBank<"CC", [CCR]>;

def RegBankCustomMappingID : RegBankMappingID<"CustomMappingID", 1>;

def : RegBankInstrMapping<[
    // G_{F|S|U}REM are not listed because they are not legal.
    // Arithmetic ops.
    G_ADD, G_SUB, G_MUL, G_SDIV, G_UDIV,
    // Bitwise ops.
    G_AND, G_OR, G_XOR,
    // Floating point ops.
    G_FADD, G_FSUB, G_FMUL, G_FDIV,
    G_FMAXIMUM, G_FMINIMUM
  ], [
  RegBankMappingRule<[RegBankTypeMapping<FPRRegBank, type0>],
    predicate=[{Type0.isVector() || isPreISelGenericFloatingPointOpcode(Opc)}]>,
  RegBankMappingRule<[RegBankTypeMapping<GPRRegBank, type0>]>
]>;

def : RegBankInstrMapping<[G_PTR_ADD], [
  RegBankMappingRule<[RegBankTypeMapping<FPRRegBank, ptype0>, RegBankTypeMapping<GPRRegBank, type1>],
    predicate=[{PType0.isVector()}]>,
  RegBankMappingRule<[RegBankTypeMapping<GPRRegBank, ptype0>, RegBankTypeMapping<GPRRegBank, type1>]>
]>;

def : RegBankInstrMapping<[G_FPEXT], [
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>
  ]>
]>;

def : RegBankInstrMapping<[
    // Shifts.
    G_SHL, G_LSHR, G_ASHR
  ], [
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>,
    RegBankTypeMapping<GPRRegBank, type1>
  ], predicate=[{Type0Size == 64 && Type1Size == 32}]>,
  RegBankMappingRule<[
      RegBankTypeMapping<FPRRegBank, type0>,
      RegBankTypeMapping<FPRRegBank, type1>
    ],
    predicate=[{Type0.isVector()}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>,
    RegBankTypeMapping<GPRRegBank, type1>
  ]>
]>;

def : RegBankInstrMapping<[G_CONSTANT], [
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>
  ], predicate=[{Type0.isPointer() || (Type0.isScalar() && Type0Size >= 32 && Type0Size <= 64)}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>
  ], id=RegBankCustomMappingID, predicate=[{Type0.isScalar() && Type0Size < 32}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>
  ]>
]>;

def : RegBankInstrMapping<[G_BRCOND], [
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>
  ]>
]>;

def : RegBankInstrMapping<[G_FRAME_INDEX], [
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, ptype0>
  ]>
]>;

def : RegBankInstrMapping<[G_FCONSTANT], [
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>,
  ], id=RegBankCustomMappingID,
      predicate=[{preferGPRForFPImm(MI, MRI, static_cast<const AArch64Subtarget&>(STI))}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>
  ]>
]>;

def : RegBankInstrMapping<[G_FCMP], [
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>,
  ], predicate=[{Type0.isVector()}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<GPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>,
  ]>
]>;

def : RegBankInstrMapping<[
    G_VECREDUCE_FADD, G_VECREDUCE_FMUL, G_VECREDUCE_FMAX,
    G_VECREDUCE_FMIN, G_VECREDUCE_FMAXIMUM, G_VECREDUCE_FMINIMUM, G_VECREDUCE_ADD,
    G_VECREDUCE_MUL, G_VECREDUCE_AND, G_VECREDUCE_OR, G_VECREDUCE_XOR, G_VECREDUCE_SMAX,
    G_VECREDUCE_SMIN, G_VECREDUCE_UMAX, G_VECREDUCE_UMIN
  ], [
  // Reductions produce a scalar value from a vector, the scalar should be on
  // FPR bank.
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>
  ]>
]>;

def : RegBankInstrMapping<[G_INSERT_VECTOR_ELT], [
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>,
    RegBankTypeMapping<GPRRegBank, type2>
  ],
    // The element may be either a GPR or FPR. Preserve that behaviour.
    predicate=[{getRegBank(EltReg, MRI, TRI) == &AArch64::FPRRegBank}]>,
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<GPRRegBank, type1>,
    RegBankTypeMapping<GPRRegBank, type2>
  ]>
]>;

def : RegBankInstrMapping<[G_VECREDUCE_SEQ_FADD, G_VECREDUCE_SEQ_FMUL], [
  RegBankMappingRule<[
    RegBankTypeMapping<FPRRegBank, type0>,
    RegBankTypeMapping<FPRRegBank, type1>,
    RegBankTypeMapping<FPRRegBank, type2>
  ]>
]>;

