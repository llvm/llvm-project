//===-- PPCOperands.td - PowerPC instruction operands ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PowerPC instruction operands, including immediate operands
// and addressing modes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Immediate operand base classes
//===----------------------------------------------------------------------===//

// Base class for immediate AsmOperandClass definitions
class ImmediateAsmOperand : AsmOperandClass {
  let Name = NAME;
  let RenderMethod = "addImmOperands";
}

// Base class for immediate AsmOperandClass with custom predicate
class ImmediateAsmOperandWithPredicate<string predicate> : ImmediateAsmOperand {
  let PredicateMethod = predicate;
}

// Base class for unsigned immediate AsmOperandClass with templated predicate
class UnsignedImmAsmOperand<int width> : ImmediateAsmOperand {
  let PredicateMethod = "isUImm<" # width # ">";
}

// Base class for signed immediate AsmOperandClass with templated predicate
class SignedImmAsmOperand<int width> : ImmediateAsmOperand {
  let PredicateMethod = "isSImm<" # width # ">";
}

// Base class for immediate operands
class ImmediateOp<ValueType vt, string asmop, string decoder> : Operand<vt> {
  let PrintMethod = "print"#asmop#"Operand";
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
  let DecoderMethod = decoder;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Unsigned immediate operand
class UnsignedImmOp<ValueType vt, string asmop, int width>
  : ImmediateOp<vt, asmop, "decodeUImmOperand<"#width#">">;

// Unsigned immediate operand with encoder
class UnsignedImmOpWithEncoder<ValueType vt, string asmop, int width, string fixup>
  : UnsignedImmOp<vt, asmop, width> {
  let EncoderMethod = "getImmEncoding<" # fixup # ">";
}

// Signed immediate operand
class SignedImmOp<ValueType vt, string asmop, int width>
  : ImmediateOp<vt, asmop, "decodeSImmOperand<"#width#">">;

// Signed immediate operand with encoder
class SignedImmOpWithEncoder<ValueType vt, string asmop, int width, string fixup>
  : SignedImmOp<vt, asmop, width> {
  let EncoderMethod = "getImmEncoding<" # fixup # ">";
}

// Multiclass for unsigned immediate operands with both regular and PC-relative versions
multiclass UnsignedImmOpWithPCRel<ValueType vt, string asmop, int width,
                                   string fixup_imm, string fixup_pcrel> {
  def "" : UnsignedImmOpWithEncoder<vt, asmop, width, fixup_imm>;
  def _pcrel : UnsignedImmOpWithEncoder<vt, asmop, width, fixup_pcrel>;
}

// Multiclass for signed immediate operands with both regular and PC-relative versions
multiclass SignedImmOpWithPCRel<ValueType vt, string asmop, int width,
                                 string fixup_imm, string fixup_pcrel> {
  def "" : SignedImmOpWithEncoder<vt, asmop, width, fixup_imm>;
  def _pcrel : SignedImmOpWithEncoder<vt, asmop, width, fixup_pcrel>;
}

//===----------------------------------------------------------------------===//
// Immediate AsmOperand definitions
//===----------------------------------------------------------------------===//

def U1Imm  : UnsignedImmAsmOperand<1>;
def U2Imm  : UnsignedImmAsmOperand<2>;
def U3Imm  : UnsignedImmAsmOperand<3>;
def U4Imm  : UnsignedImmAsmOperand<4>;
def U5Imm  : UnsignedImmAsmOperand<5>;
def U6Imm  : UnsignedImmAsmOperand<6>;
def U7Imm  : UnsignedImmAsmOperand<7>;
def U8Imm  : UnsignedImmAsmOperand<8>;
def U10Imm : UnsignedImmAsmOperand<10>;
def U12Imm : UnsignedImmAsmOperand<12>;

def S5Imm  : SignedImmAsmOperand<5>;

// Special case: S32Imm and S34Imm have custom predicates that accept expressions
def S32Imm : ImmediateAsmOperandWithPredicate<"isS32Imm">;
def S34Imm : ImmediateAsmOperandWithPredicate<"isS34Imm">;

// Special case: S16Imm has custom render method
def S16Imm : ImmediateAsmOperandWithPredicate<"isS16Imm"> {
  let RenderMethod = "addS16ImmOperands";
}

// Special case: U16Imm has custom render method
def U16Imm : ImmediateAsmOperandWithPredicate<"isU16Imm"> {
  let RenderMethod = "addU16ImmOperands";
}

// Special case: S17Imm has custom predicate and render method
def S17Imm : ImmediateAsmOperandWithPredicate<"isS17Imm"> {
  let RenderMethod = "addS16ImmOperands";
}

// Special case: ATBitsAsHint has custom predicate
def ATBitsAsHint : ImmediateAsmOperandWithPredicate<"isATBitsAsHint">;

// Special case: ImmZero has custom predicate
def ImmZero : ImmediateAsmOperandWithPredicate<"isImmZero">;

//===----------------------------------------------------------------------===//
// i32 immediate operands
//===----------------------------------------------------------------------===//

def u1imm   : UnsignedImmOp<i32, "U1Imm", 1>;
def u2imm   : UnsignedImmOp<i32, "U2Imm", 2>;
def u3imm   : UnsignedImmOp<i32, "U3Imm", 3>;
def u4imm   : UnsignedImmOp<i32, "U4Imm", 4>;
def u5imm   : UnsignedImmOp<i32, "U5Imm", 5>;
def u6imm   : UnsignedImmOp<i32, "U6Imm", 6>;
def u7imm   : UnsignedImmOp<i32, "U7Imm", 7>;
def u8imm   : UnsignedImmOp<i32, "U8Imm", 8>;
def u10imm  : UnsignedImmOp<i32, "U10Imm", 10>;
def u12imm  : UnsignedImmOp<i32, "U12Imm", 12>;

def s5imm   : SignedImmOp<i32, "S5Imm", 5>;

// Special case: atimm has custom print method and no decoder
def atimm : Operand<i32> {
  let PrintMethod = "printATBitsAsHint";
  let ParserMatchClass = ATBitsAsHint;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Special case: immZero has custom print and decoder methods
def immZero : Operand<i32> {
  let PrintMethod = "printImmZeroOperand";
  let ParserMatchClass = ImmZero;
  let DecoderMethod = "decodeImmZeroOperand";
  let OperandType = "OPERAND_IMMEDIATE";
}

// Special cases: s16imm and u16imm have custom encoder methods
def s16imm : SignedImmOpWithEncoder<i32, "S16Imm", 16, "PPC::fixup_ppc_half16">;

def u16imm : UnsignedImmOpWithEncoder<i32, "U16Imm", 16, "PPC::fixup_ppc_half16">;

//===----------------------------------------------------------------------===//
// i64 immediate operands
//===----------------------------------------------------------------------===//

def s16imm64 : SignedImmOpWithEncoder<i64, "S16Imm", 16, "PPC::fixup_ppc_half16">;

def u16imm64 : UnsignedImmOpWithEncoder<i64, "U16Imm", 16, "PPC::fixup_ppc_half16">;

// Special case: s17imm uses S16Imm print method but accepts wider range
def s17imm : SignedImmOpWithEncoder<i32, "S17Imm", 16, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}

def s17imm64 : SignedImmOpWithEncoder<i64, "S17Imm", 16, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}

defm s32imm : SignedImmOpWithPCRel<i64, "S32Imm", 32,
                                    "PPC::fixup_ppc_imm32", "PPC::fixup_ppc_pcrel32">;

defm s34imm : SignedImmOpWithPCRel<i64, "S34Imm", 34,
                                    "PPC::fixup_ppc_imm34", "PPC::fixup_ppc_pcrel34">;
