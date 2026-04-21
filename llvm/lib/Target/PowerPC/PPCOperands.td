//===-- PPCOperands.td - PowerPC instruction operands -------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PowerPC instruction operands, including immediate
// operands and addressing modes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Immediate operand base classes
//===----------------------------------------------------------------------===//

// Base class for immediate AsmOperandClass definitions.
class ImmediateAsmOperand<string predicate, string render="addImmOperands">
    : AsmOperandClass {
  let Name = NAME;
  let PredicateMethod = predicate;
  let RenderMethod = render;
}

// Base class for signed immediate operands.
class ImmediateOp<ValueType vt, string asmop, int width> : Operand<vt> {
  let PrintMethod = "printSImmOperand<"#width#">";
  let DecoderMethod = "decodeSImmOperand<"#width#">";
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
  let OperandType = "OPERAND_IMMEDIATE";
}

// Base class for unsigned immediate operands.
class UImmediateOp<ValueType vt, string asmop, int width>
    : ImmediateOp<vt, asmop, width> {
  let PrintMethod = "printUImmOperand<"#width#">";
  let DecoderMethod = "decodeUImmOperand<"#width#">";
}

// Base class for signed immediate operands with relocation.
class ImmediateRelocOp<ValueType vt, string asmop, int width, string fixup,
                       string decoder = ""> : Operand<vt> {
  let PrintMethod = "print"#asmop#"Operand";
  let DecoderMethod = !if(!eq(decoder, ""),
                          "decodeSImmOperand<"#width#">",
                          decoder);
  let ParserMatchClass = !cast<AsmOperandClass>(asmop);
  let EncoderMethod = "getImmEncoding<" # fixup # ">";
  let OperandType = "OPERAND_IMMEDIATE";
}

// Base class for unsigned immediate operands with relocation.
class UImmediateRelocOp<ValueType vt, string asmop, int width, string fixup,
                        string decoder = "">
    : ImmediateRelocOp<vt, asmop, width, fixup> {
  let DecoderMethod = !if(!eq(decoder, ""),
                          "decodeUImmOperand<"#width#">",
                          decoder);
}
//===----------------------------------------------------------------------===//
// Multiclasses for complete immediate definitions
// (AsmOperand + Operand + ImmLeaf).
//===----------------------------------------------------------------------===//

multiclass ImmPatterns<ValueType vt, code pred, SDNodeXForm xform> {
  def _pat : ImmLeaf<vt, pred, xform>;
  def _timm : TImmLeaf<vt, pred, xform>;
}

multiclass SignedImmediate<ValueType vt, code pred, SDNodeXForm xform,
                           string asmop, int width >
    : ImmPatterns<vt, pred, xform> {
  def "" : ImmediateOp<vt, asmop, width>;
}

multiclass UnsignedImmediate<ValueType vt, code pred, SDNodeXForm xform,
                             string asmop, int width >
    : ImmPatterns<vt, pred, xform> {
  def "" : UImmediateOp<vt, asmop, width>;
}

multiclass SignedImmediateReloc<ValueType vt, code pred, SDNodeXForm xform,
                                string asmop, int width, string fixup>
    : ImmPatterns<vt, pred, xform> {
  def "" : ImmediateRelocOp<vt, asmop, width, fixup>;
}

// Helper multiclass for unsigned immediates with relocation fixup string.
multiclass UnsignedImmediateReloc<ValueType vt, code pred, SDNodeXForm xform,
                                  string asmop, int width, string fixup>
    : ImmPatterns<vt, pred, xform> {
  def "" : UImmediateRelocOp<vt, asmop, width, fixup>;
}

// Multiclass for signed immediate operands with both regular and
// PC-relative versions.
multiclass SignedImmediateWithPCRel<ValueType vt, code pred, SDNodeXForm xform,
                                    string asmop, int width, string fixup_imm,
                                    string fixup_pcrel>
      : SignedImmediateReloc<vt, pred, xform, asmop, width, fixup_imm> {
  // PC-relative immediate: instantiate with PC-relative fixup
  defm _pcrel : SignedImmediateReloc<vt, pred, xform, asmop, width, fixup_pcrel>;
}

//===----------------------------------------------------------------------===//
// Immediate AsmOperand definitions
//===----------------------------------------------------------------------===//
def U1Imm : ImmediateAsmOperand<"isUImm<1>">;
def U2Imm : ImmediateAsmOperand<"isUImm<2>">;
def U3Imm : ImmediateAsmOperand<"isUImm<3>">;
def U4Imm : ImmediateAsmOperand<"isUImm<4>">;
def U5Imm : ImmediateAsmOperand<"isUImm<5>">;
def U6Imm : ImmediateAsmOperand<"isUImm<6>">;
def U7Imm : ImmediateAsmOperand<"isUImm<7>">;
def U8Imm : ImmediateAsmOperand<"isUImm<8>">;
def U10Imm : ImmediateAsmOperand<"isUImm<10>">;
def U12Imm : ImmediateAsmOperand<"isUImm<12>">;
def U32Imm : ImmediateAsmOperand<"isUImm<32>">;
def S5Imm : ImmediateAsmOperand<"isSImm<5>">;

// Special cases that have custom predicat and/or render method.
def ATBitsAsHint : ImmediateAsmOperand<"isATBitsAsHint">; //Predicate always fails.
def ImmZero : ImmediateAsmOperand<"isImmZero">;
def U16Imm : ImmediateAsmOperand<"isU16Imm","addU16ImmOperands">;
def S16Imm : ImmediateAsmOperand<"isS16Imm","addS16ImmOperands">;
def S17Imm : ImmediateAsmOperand<"isS17Imm","addS16ImmOperands">;
def S32Imm : ImmediateAsmOperand<"isS32Imm">;
def S34Imm : ImmediateAsmOperand<"isS34Imm">;

//===----------------------------------------------------------------------===//
// i32 immediate operands
//===----------------------------------------------------------------------===//

defm u1imm : UnsignedImmediate<i32,
  [{ return isUInt<1>(Imm); }], NOOP_SDNodeXForm,
  "U1Imm", 1>;
defm u2imm : UnsignedImmediate<i32,
  [{ return isUInt<2>(Imm); }], NOOP_SDNodeXForm,
  "U2Imm", 2>;
defm u3imm : UnsignedImmediate<i32,
  [{ return isUInt<3>(Imm); }], NOOP_SDNodeXForm,
  "U3Imm", 3>;
defm u4imm : UnsignedImmediate<i32,
  [{ return isUInt<4>(Imm); }], NOOP_SDNodeXForm,
  "U4Imm", 4>;
defm u5imm : UnsignedImmediate<i32,
  [{ return isUInt<5>(Imm); }], NOOP_SDNodeXForm,
  "U5Imm", 5>;
defm u6imm : UnsignedImmediate<i32,
  [{ return isUInt<6>(Imm); }], NOOP_SDNodeXForm,
  "U6Imm", 6>;
defm u7imm : UnsignedImmediate<i32,
  [{ return isUInt<7>(Imm); }], NOOP_SDNodeXForm,
  "U7Imm", 7>;
defm u8imm : UnsignedImmediate<i32,
  [{ return isUInt<8>(Imm); }], NOOP_SDNodeXForm,
  "U8Imm", 8>;

// Truncating version for BUILD_VECTOR operands that may be sign-extended.
// Uses the same parser class as u8imm but with a truncating print method.
def u8imm_trunc : UImmediateOp<i32, "U8Imm", 8> {
  let PrintMethod = "printU8ImmOperandTrunc";
}
defm u10imm : UnsignedImmediate<i32,
  [{ return isUInt<10>(Imm); }], NOOP_SDNodeXForm,
  "U10Imm", 10>;
defm u12imm : UnsignedImmediate<i32,
  [{ return isUInt<12>(Imm); }], NOOP_SDNodeXForm,
  "U12Imm", 12>;
defm u32imm : UnsignedImmediate<i32,
  [{ return isUInt<32>(Imm); }], NOOP_SDNodeXForm,
  "U32Imm", 32>;

defm s5imm : SignedImmediate<i32,
  [{ return isInt<5>(Imm); }], NOOP_SDNodeXForm,
  "S5Imm", 5>;

defm s32imm : SignedImmediateWithPCRel<i32,
  [{ return isInt<32>(Imm); }], NOOP_SDNodeXForm,
  "S32Imm", 32, "PPC::fixup_ppc_imm32", "PPC::fixup_ppc_pcrel32">;

defm s34imm : SignedImmediateReloc<i32,
  [{ return isInt<34>(Imm); }], NOOP_SDNodeXForm,
  "S34Imm", 34, "PPC::fixup_ppc_imm34">;

//===----------------------------------------------------------------------===//
// i64 immediate operands
//===----------------------------------------------------------------------===//

defm s32imm64 : SignedImmediateWithPCRel<i64,
  [{ return isInt<32>(Imm); }], NOOP_SDNodeXForm,
  "S32Imm", 32, "PPC::fixup_ppc_imm32", "PPC::fixup_ppc_pcrel32">;
defm s34imm64 : SignedImmediateWithPCRel<i64,
  [{ return isInt<34>(Imm); }], NOOP_SDNodeXForm,
  "S34Imm", 34, "PPC::fixup_ppc_imm34", "PPC::fixup_ppc_pcrel34">;

//===----------------------------------------------------------------------===//
// Special case immediate operands
//===----------------------------------------------------------------------===//

// immZero represents a hardcoded zero register encoding, it is NOT used in DAG
// patterns. It is only used as an instruction operand for assembly/disassembly,
// specifically to represent a hardcoded zero register value in PC-relative
// addressing modes.
def immZero : Operand<i32> {
  let PrintMethod = "printImmZeroOperand";
  let ParserMatchClass = ImmZero;
  let DecoderMethod = "decodeImmZeroOperand";
  let OperandType = "OPERAND_IMMEDIATE";
}

// atimm is used to represent branch prediction hints, not a general immediate
// value. It's a 2-bit AT (Address Translation) field in PPC branch instructions
// and is an assembly-only operand that prints as `+` or `-` symbols, not
// numeric values.
def atimm : Operand<i32> {
  let PrintMethod = "printATBitsAsHint";
  let ParserMatchClass = ATBitsAsHint;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Special cases: s16imm and u16imm have custom encoder methods.
defm s16imm : SignedImmediateReloc<i32,
  [{ return isInt<16>(Imm); }], NOOP_SDNodeXForm,
  "S16Imm", 16, "PPC::fixup_ppc_half16">;
defm u16imm : UnsignedImmediateReloc<i32,
  [{ return isUInt<16>(Imm); }], NOOP_SDNodeXForm,
  "U16Imm", 16, "PPC::fixup_ppc_half16">;

// s16imm64 uses imm64SExt16 pattern to match the operand type.
def s16imm64 : ImmediateRelocOp<i64, "S16Imm", 16, "PPC::fixup_ppc_half16">;

// u16imm64 uses two different patterns depending on the instruction context:
// * immZExt16 - For low 16-bit immediates
// * imm16ShiftedZExt - For high 16-bit immediates (shifted)
def u16imm64 : UImmediateRelocOp<i64, "U16Imm", 16, "PPC::fixup_ppc_half16">;

// Special case: s17imm uses S16Imm print method but accepts wider range.
// This operand type is used for addis/lis to allow the assembler parser
// to accept immediates in the range -65536..65535 for compatibility
// with the GNU assembler. The operand is treated as 16-bit otherwise.
def s17imm : ImmediateRelocOp<i32, "S17Imm", 16, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}
def s17imm64 : ImmediateRelocOp<i64, "S17Imm", 16, "PPC::fixup_ppc_half16"> {
  let PrintMethod = "printS16ImmOperand";
}

//===----------------------------------------------------------------------===//
// Branch target operands
//===----------------------------------------------------------------------===//

def PPCDirectBrAsmOperand : AsmOperandClass {
  let Name = "DirectBr"; let PredicateMethod = "isDirectBr";
  let RenderMethod = "addBranchTargetOperands";
}
def directbrtarget : Operand<OtherVT> {
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getDirectBrEncoding";
  let DecoderMethod = "decodeDirectBrTarget";
  let ParserMatchClass = PPCDirectBrAsmOperand;
  let OperandType = "OPERAND_PCREL";
}
def absdirectbrtarget : Operand<OtherVT> {
  let PrintMethod = "printAbsBranchOperand";
  let EncoderMethod = "getAbsDirectBrEncoding";
  let DecoderMethod = "decodeDirectBrTarget";
  let ParserMatchClass = PPCDirectBrAsmOperand;
}
def PPCCondBrAsmOperand : AsmOperandClass {
  let Name = "CondBr"; let PredicateMethod = "isCondBr";
  let RenderMethod = "addBranchTargetOperands";
}
def condbrtarget : Operand<OtherVT> {
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getCondBrEncoding";
  let DecoderMethod = "decodeCondBrTarget";
  let ParserMatchClass = PPCCondBrAsmOperand;
  let OperandType = "OPERAND_PCREL";
}
def abscondbrtarget : Operand<OtherVT> {
  let PrintMethod = "printAbsBranchOperand";
  let EncoderMethod = "getAbsCondBrEncoding";
  let DecoderMethod = "decodeCondBrTarget";
  let ParserMatchClass = PPCCondBrAsmOperand;
}
def calltarget : Operand<iPTR> {
  let PrintMethod = "printBranchOperand";
  let EncoderMethod = "getDirectBrEncoding";
  let DecoderMethod = "decodeDirectBrTarget";
  let ParserMatchClass = PPCDirectBrAsmOperand;
  let OperandType = "OPERAND_PCREL";
}
def abscalltarget : Operand<iPTR> {
  let PrintMethod = "printAbsBranchOperand";
  let EncoderMethod = "getAbsDirectBrEncoding";
  let DecoderMethod = "decodeDirectBrTarget";
  let ParserMatchClass = PPCDirectBrAsmOperand;
}

//===----------------------------------------------------------------------===//
// CR bit mask operand
//===----------------------------------------------------------------------===//

def PPCCRBitMaskOperand : AsmOperandClass {
 let Name = "CRBitMask"; let PredicateMethod = "isCRBitMask";
}
def crbitm: Operand<i8> {
  let PrintMethod = "printcrbitm";
  let EncoderMethod = "get_crbitm_encoding";
  let DecoderMethod = "decodeCRBitMOperand";
  let ParserMatchClass = PPCCRBitMaskOperand;
}

//===----------------------------------------------------------------------===//
// Displacement operands
//===----------------------------------------------------------------------===//

def PPCDispRI34Operand : AsmOperandClass {
  let Name = "DispRI34"; let PredicateMethod = "isS34Imm";
  let RenderMethod = "addImmOperands";
}
def dispRI34 : Operand<iPTR> {
  let ParserMatchClass = PPCDispRI34Operand;
  let EncoderMethod = "getDispRI34Encoding";
  let DecoderMethod = "decodeSImmOperand<34>";
}
def dispRI34_pcrel : Operand<iPTR> {
  let ParserMatchClass = PPCDispRI34Operand;
  let EncoderMethod = "getDispRI34PCRelEncoding";
  let DecoderMethod = "decodeSImmOperand<34>";
}
def PPCDispRIOperand : AsmOperandClass {
 let Name = "DispRI"; let PredicateMethod = "isS16Imm";
 let RenderMethod = "addS16ImmOperands";
}
def dispRI : Operand<iPTR> {
  let ParserMatchClass = PPCDispRIOperand;
  let EncoderMethod = "getDispRIEncoding";
}
def PPCDispRIXOperand : AsmOperandClass {
 let Name = "DispRIX"; let PredicateMethod = "isS16ImmX4";
 let RenderMethod = "addS16ImmOperands";
}
def dispRIX : Operand<iPTR> {
  let ParserMatchClass = PPCDispRIXOperand;
  let EncoderMethod = "getDispRIXEncoding";
  let DecoderMethod = "decodeDispRIXOperand";
}
def PPCDispRIHashOperand : AsmOperandClass {
  let Name = "DispRIHash"; let PredicateMethod = "isHashImmX8";
  let RenderMethod = "addImmOperands";
}
def dispRIHash : Operand<iPTR> {
  let ParserMatchClass = PPCDispRIHashOperand;
  let EncoderMethod = "getDispRIHashEncoding";
  let DecoderMethod = "decodeDispRIHashOperand";
}
def PPCDispRIX16Operand : AsmOperandClass {
 let Name = "DispRIX16"; let PredicateMethod = "isS16ImmX16";
 let RenderMethod = "addS16ImmOperands";
}
def dispRIX16 : Operand<iPTR> {
  let ParserMatchClass = PPCDispRIX16Operand;
  let EncoderMethod = "getDispRIX16Encoding";
  let DecoderMethod = "decodeDispRIX16Operand";
}
def PPCDispSPE8Operand : AsmOperandClass {
 let Name = "DispSPE8"; let PredicateMethod = "isU8ImmX8";
 let RenderMethod = "addImmOperands";
}
def dispSPE8 : Operand<iPTR> {
  let ParserMatchClass = PPCDispSPE8Operand;
  let DecoderMethod = "decodeDispSPE8Operand";
  let EncoderMethod = "getDispSPE8Encoding";
}
def PPCDispSPE4Operand : AsmOperandClass {
 let Name = "DispSPE4"; let PredicateMethod = "isU7ImmX4";
 let RenderMethod = "addImmOperands";
}
def dispSPE4 : Operand<iPTR> {
  let ParserMatchClass = PPCDispSPE4Operand;
  let DecoderMethod = "decodeDispSPE4Operand";
  let EncoderMethod = "getDispSPE4Encoding";
}
def PPCDispSPE2Operand : AsmOperandClass {
 let Name = "DispSPE2"; let PredicateMethod = "isU6ImmX2";
 let RenderMethod = "addImmOperands";
}
def dispSPE2 : Operand<iPTR> {
  let ParserMatchClass = PPCDispSPE2Operand;
  let DecoderMethod = "decodeDispSPE2Operand";
  let EncoderMethod = "getDispSPE2Encoding";
}

//===----------------------------------------------------------------------===//
// TLS operands
//===----------------------------------------------------------------------===//

def PPCTLSRegOperand : AsmOperandClass {
  let Name = "TLSReg"; let PredicateMethod = "isTLSReg";
  let RenderMethod = "addTLSRegOperands";
}
def tlsreg32 : Operand<i32> {
  let EncoderMethod = "getTLSRegEncoding";
  let ParserMatchClass = PPCTLSRegOperand;
}
def tlsgd32 : Operand<i32> {}
def tlscall32 : Operand<i32> {
  let PrintMethod = "printTLSCall";
  let MIOperandInfo = (ops calltarget:$func, tlsgd32:$sym);
  let EncoderMethod = "getTLSCallEncoding";
}
