//===-- AGXOperands.td - AGX operand definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operands of the real (encoded) instructions and of the pseudo instructions.
//
// Encoded ALU sources are complex operands made of a modifier immediate and a
// value. The value is a general purpose register, a uniform register or an
// 8-bit immediate; the register class carries the access size. The modifier
// immediate holds AGXSrcMods flags (abs, neg, sx, cache, discard).
//
// The encoder methods produce the operand fields laid out as follows (bit
// numbers within the encoded operand value):
//
//   Src, Src64, Src16  [5:0] value, [9:6] type, [11:10] extension
//   SrcSX, SrcSX64     [5:0] value, [9:6] type, [10] sign extend,
//                      [12:11] extension
//   SrcF               [5:0] value, [9:6] type, [11:10] abs/neg,
//                      [13:12] extension
//   SrcF16             [5:0] value, [8:6] type, [10:9] abs/neg,
//                      [12:11] extension
//   CmpselSrc          [5:0] value, [8:6] type, [10:9] extension
//   Dst*               [0] cache, [1] 32-bit, [7:2] value, [9:8] extension
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Assembler operand classes
//===----------------------------------------------------------------------===//

class AGXAsmOperand<string name, string parser, string render> :
    AsmOperandClass {
  let Name = name;
  let PredicateMethod = "is"#name;
  let ParserMethod = parser;
  let RenderMethod = render;
  let DiagnosticType = "Invalid"#name;
}

class AGXSrcAsmOperand<string name>
    : AGXAsmOperand<name, "parseSrc", "addSrcOperands">;

def AGXSrcAsm : AGXSrcAsmOperand<"Src">;
def AGXSrc64Asm : AGXSrcAsmOperand<"Src64">;
def AGXSrc16Asm : AGXSrcAsmOperand<"Src16">;
def AGXSrcSXAsm : AGXSrcAsmOperand<"SrcSX">;
def AGXSrcSX64Asm : AGXSrcAsmOperand<"SrcSX64">;
def AGXSrcFAsm : AGXSrcAsmOperand<"SrcF">;
def AGXSrcF16Asm : AGXSrcAsmOperand<"SrcF16">;
def AGXCmpselSrcAsm : AGXSrcAsmOperand<"CmpselSrc">;

class AGXDstAsmOperand<string name>
    : AGXAsmOperand<name, "parseDst", "addDstOperands">;

def AGXDstAsm : AGXDstAsmOperand<"Dst">;
def AGXDst64Asm : AGXDstAsmOperand<"Dst64">;
def AGXDst16Asm : AGXDstAsmOperand<"Dst16">;
def AGXDst32Asm : AGXDstAsmOperand<"Dst32">;

def AGXImmAsm : AGXAsmOperand<"AGXImm", "parseImm", "addImmOperands">;

// Optional operands report a mismatch as a plain invalid operand, so that the
// matcher can skip them and match the next one (a store flag without a shift).
def AGXShiftAsm : AGXAsmOperand<"Shift", "parseShift", "addImmOperands"> {
  let IsOptional = 1;
  let DefaultMethod = "defaultZeroOperand";
  let DiagnosticType = "";
}

def AGXStoreFlagAsm : AGXAsmOperand<"StoreFlag", "parseImm", "addImmOperands"> {
  let IsOptional = 1;
  let DefaultMethod = "defaultZeroOperand";
  let DiagnosticType = "";
}

def AGXBitfieldMaskAsm : AGXAsmOperand<"BitfieldMask", "parseBitfieldMask",
                                       "addImmOperands"> {
  let IsOptional = 1;
  let DefaultMethod = "defaultZeroOperand";
}

def AGXTruthTableAsm : AGXAsmOperand<"TruthTable", "parseTruthTable",
                                     "addImmOperands">;

class AGXEnumAsmOperand<string name>
    : AGXAsmOperand<name, "parse"#name, "addImmOperands">;

def AGXMemRegAsm : AGXAsmOperand<"MemReg", "parseMemReg", "addRegOperands">;
def AGXAtomicDstAsm : AGXAsmOperand<"AtomicDst", "parseMemReg",
                                    "addRegOperands">;
def AGXMemBaseAsm : AGXAsmOperand<"MemBase", "parseMemReg", "addRegOperands">;
def AGXAtomicSrcAsm : AGXAsmOperand<"AtomicSrc", "parseMemReg",
                                    "addRegOperands">;
def AGXMemIndexAsm : AGXAsmOperand<"MemIndex", "parseMemIndex",
                                   "addRegOrImmOperands">;
def AGXLocalBaseAsm : AGXAsmOperand<"LocalBase", "parseMemIndex",
                                    "addRegOrImmOperands">;
def AGXLocalIndexAsm : AGXAsmOperand<"LocalIndex", "parseMemIndex",
                                     "addRegOrImmOperands">;
def AGXBrTargetAsm : AGXAsmOperand<"BrTarget", "parseBrTarget",
                                   "addBrTargetOperands">;
def AGXNestCounterAsm : AGXAsmOperand<"NestCounter", "parseNestCounter",
                                      "addImmOperands">;

//===----------------------------------------------------------------------===//
// ALU operands
//===----------------------------------------------------------------------===//

// Sub-operand of a real instruction that holds either a register or an
// immediate.
def AGXAnyValue : Operand<i32>;

class AGXSrcOperandBase<string kind, AsmOperandClass asm> : Operand<i32> {
  let MIOperandInfo = (ops i32imm:$mods, AGXAnyValue:$val);
  let PrintMethod = "print"#kind;
  let EncoderMethod = "get"#kind#"Encoding";
  let DecoderMethod = "decode"#kind;
  let ParserMatchClass = asm;
  let OperandType = "OPERAND_SRC";
  let OperandNamespace = "AGX";
}

// Zero extended 16/32-bit source.
def AGXSrc : AGXSrcOperandBase<"Src", AGXSrcAsm>;
// Zero extended 16/32/64-bit source.
def AGXSrc64 : AGXSrcOperandBase<"Src64", AGXSrc64Asm>;
// 16-bit only source.
def AGXSrc16 : AGXSrcOperandBase<"Src16", AGXSrc16Asm>;
// Sign-extendable 16/32-bit source (imadd A/B).
def AGXSrcSX : AGXSrcOperandBase<"SrcSX", AGXSrcSXAsm>;
// Sign-extendable 16/32/64-bit source (iadd, imadd C).
def AGXSrcSX64 : AGXSrcOperandBase<"SrcSX64", AGXSrcSX64Asm>;
// Floating point 16/32-bit source with abs/neg; immediates are 8-bit floats.
def AGXSrcF : AGXSrcOperandBase<"SrcF", AGXSrcFAsm>;
// Floating point 16-bit source with abs/neg.
def AGXSrcF16 : AGXSrcOperandBase<"SrcF16", AGXSrcF16Asm>;
// Compare-and-select value source, sized like the destination.
def AGXCmpselSrc : AGXSrcOperandBase<"CmpselSrc", AGXCmpselSrcAsm>;

class AGXDstOperandBase<string kind, AsmOperandClass asm> : Operand<i32> {
  let MIOperandInfo = (ops AGXAnyValue:$reg, i32imm:$cache);
  let PrintMethod = "printDst";
  let EncoderMethod = "getDstEncoding";
  let DecoderMethod = "decode"#kind;
  let ParserMatchClass = asm;
  let OperandType = "OPERAND_DST";
  let OperandNamespace = "AGX";
}

// 16/32-bit destination.
def AGXDst : AGXDstOperandBase<"Dst", AGXDstAsm>;
// 16/32/64-bit destination.
def AGXDst64 : AGXDstOperandBase<"Dst64", AGXDst64Asm>;
// Destinations whose size bit is part of the opcode.
def AGXDst16 : AGXDstOperandBase<"Dst16", AGXDst16Asm>;
def AGXDst32 : AGXDstOperandBase<"Dst32", AGXDst32Asm>;

//===----------------------------------------------------------------------===//
// Immediate operands
//===----------------------------------------------------------------------===//

def AGXImm : Operand<i32> {
  let PrintMethod = "printImm";
  let ParserMatchClass = AGXImmAsm;
  let DecoderMethod = "decodeImm";
}

def AGXImm64 : Operand<i64> {
  let PrintMethod = "printImm";
  let ParserMatchClass = AGXImmAsm;
  let DecoderMethod = "decodeImm";
}

// Pseudo instruction operand holding AGXSrcMods flags.
def SrcMods : Operand<i32> {
  let PrintMethod = "printImm";
}

def AGXShift : Operand<i32> {
  let PrintMethod = "printShift";
  let ParserMatchClass = AGXShiftAsm;
  let DecoderMethod = "decodeImm";
}

// Optional trailing flag bit, printed only when set.
def AGXStoreFlag : Operand<i32> {
  let PrintMethod = "printStoreFlag";
  let ParserMatchClass = AGXStoreFlagAsm;
  let DecoderMethod = "decodeImm";
}

def AGXBitfieldMask : Operand<i32> {
  let PrintMethod = "printBitfieldMask";
  let ParserMatchClass = AGXBitfieldMaskAsm;
  let DecoderMethod = "decodeImm";
}

def AGXTruthTable : Operand<i32> {
  let PrintMethod = "printTruthTable";
  let ParserMatchClass = AGXTruthTableAsm;
  let DecoderMethod = "decodeImm";
}

multiclass AGXEnum<string name> {
  def Asm : AGXEnumAsmOperand<name>;
  def "" : Operand<i32> {
    let PrintMethod = "print"#name;
    let ParserMatchClass = !cast<AsmOperandClass>(NAME#"Asm");
    let DecoderMethod = "decodeImm";
  }
}

// Integer conditions including the inverted forms (4 bits).
defm AGXICond : AGXEnum<"ICond">;
// Float conditions including the inverted forms (4 bits).
defm AGXFCond : AGXEnum<"FCond">;
// Conditions without inversion (3 bits), for compare-and-select.
defm AGXICond3 : AGXEnum<"ICond3">;
defm AGXFCond3 : AGXEnum<"FCond3">;
defm AGXConvertMode : AGXEnum<"ConvertMode">;
defm AGXRoundMode : AGXEnum<"RoundMode">;
defm AGXMemFormat : AGXEnum<"MemFormat">;
defm AGXMemMask : AGXEnum<"MemMask">;
defm AGXSignedness : AGXEnum<"Signedness">;
defm AGXAtomicOp : AGXEnum<"AtomicOp">;
defm AGXSpecialReg : AGXEnum<"SpecialReg">;

//===----------------------------------------------------------------------===//
// Memory operands
//===----------------------------------------------------------------------===//

// Register (or first register of the register tuple) read or written by a
// memory instruction. The element count is the popcount of the mask.
def AGXMemReg : Operand<i32> {
  let PrintMethod = "printMemReg";
  let EncoderMethod = "getMemRegEncoding";
  let DecoderMethod = "decodeMemReg";
  let ParserMatchClass = AGXMemRegAsm;
}

// Optional atomic result register ('_' for none).
// Bits 45-46 of the device atomic. Operand bit 0 is bit 45 (the coherency bit,
// or the float / 64-bit variant on G14); operand bit 1 clears bit 46, which is
// set in every encoding Mesa emits. 0 and 1 are therefore Mesa's encodings,
// and 3 with cmpxchg is the 64-bit unsigned min.
def AGXAtomicMode : Operand<i32> {
  let PrintMethod = "printImm";
  let ParserMatchClass = AGXImmAsm;
  let EncoderMethod = "getAtomicModeEncoding";
  let DecoderMethod = "decodeAtomicMode";
}

def AGXAtomicDst : Operand<i32> {
  let PrintMethod = "printAtomicDst";
  let EncoderMethod = "getAtomicDstEncoding";
  let DecoderMethod = "decodeAtomicDst";
  let ParserMatchClass = AGXAtomicDstAsm;
}

// Optional threadgroup atomic result register ('_' for none): a 16-bit or
// 32-bit register.
def AGXLocalAtomicDstAsm : AGXAsmOperand<"LocalAtomicDst", "parseMemReg",
                                         "addRegOperands">;
def AGXLocalAtomicDst : Operand<i32> {
  let PrintMethod = "printAtomicDst";
  let EncoderMethod = "getLocalAtomicDstEncoding";
  let DecoderMethod = "decodeLocalAtomicDst";
  let ParserMatchClass = AGXLocalAtomicDstAsm;
}

// 64-bit device memory base: general purpose or uniform register pair.
def AGXMemBase : Operand<i64> {
  let PrintMethod = "printMemBase";
  let EncoderMethod = "getMemBaseEncoding";
  let DecoderMethod = "decodeMemBase";
  let ParserMatchClass = AGXMemBaseAsm;
}

// Device memory index: 32-bit register or signed 16-bit immediate.
def AGXMemIndex : Operand<i32> {
  let PrintMethod = "printMemIndex";
  let EncoderMethod = "getMemIndexEncoding";
  let DecoderMethod = "decodeMemIndex";
  let ParserMatchClass = AGXMemIndexAsm;
}

// Threadgroup memory base: 16-bit register, 16-bit uniform or zero.
def AGXLocalBase : Operand<i32> {
  let PrintMethod = "printLocalBase";
  let EncoderMethod = "getLocalBaseEncoding";
  let DecoderMethod = "decodeLocalBase";
  let ParserMatchClass = AGXLocalBaseAsm;
}

// Threadgroup memory index: 16-bit register or 16-bit immediate.
def AGXLocalIndex : Operand<i32> {
  let PrintMethod = "printLocalIndex";
  let EncoderMethod = "getLocalIndexEncoding";
  let DecoderMethod = "decodeLocalIndex";
  let ParserMatchClass = AGXLocalIndexAsm;
}

// Atomic data source: 32-bit register (64-bit pair for cmpxchg).
def AGXAtomicSrc : Operand<i32> {
  let PrintMethod = "printAtomicSrc";
  let EncoderMethod = "getAtomicSrcEncoding";
  let DecoderMethod = "decodeAtomicSrc";
  let ParserMatchClass = AGXAtomicSrcAsm;
}

//===----------------------------------------------------------------------===//
// Branch targets
//===----------------------------------------------------------------------===//

def AGXBrTarget : Operand<OtherVT> {
  let PrintMethod = "printBrTarget";
  let EncoderMethod = "getBrTargetEncoding";
  let DecoderMethod = "decodeBrTarget";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = AGXBrTargetAsm;
}

// The implicit nesting counter operand of execution mask instructions. The
// immediate is its cache hint; it is printed as r0l.
def AGXNestCounter : Operand<i32> {
  let PrintMethod = "printNestCounter";
  let ParserMatchClass = AGXNestCounterAsm;
  let DecoderMethod = "decodeImm";
}

//===----------------------------------------------------------------------===//
// Pseudo instruction register operands
//===----------------------------------------------------------------------===//

class AGXRegOperand<RegisterClass rc, string type, string ns = "AGX">
    : RegisterOperand<rc> {
  let OperandNamespace = ns;
  let OperandType = type;
}

// ALU sources accept general purpose registers, uniform registers and,
// after operand folding, 8-bit immediates.
def Src16Op : AGXRegOperand<SRC16, "OPERAND_SRC">;
def Src32Op : AGXRegOperand<SRC32, "OPERAND_SRC">;
// 64-bit sources cannot be uniform registers.
def Src64Op : AGXRegOperand<GPR64, "OPERAND_SRC">;
def Base64Op : AGXRegOperand<BASE64, "OPERAND_REGISTER", "MCOI">;
