//===-- RISCVInstrInfoZb.td - RISC-V Bitmanip instructions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Bitmanip
// extensions, versions:
//   Zba - 1.0
//   Zbb - 1.0
//   Zbc - 1.0
//   Zbs - 1.0
//
// The experimental extensions appeared in an earlier draft of the Bitmanip
// extensions. They are not ratified and subject to change.
//
// This file also describes RISC-V instructions from the Zbk* extensions in
// Cryptography Extensions Volume I: Scalar & Entropy Source Instructions,
// versions:
//   Zbkb - 1.0
//   Zbkc - 1.0
//   Zbkx - 1.0
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def riscv_clzw   : SDNode<"RISCVISD::CLZW",   SDT_RISCVIntUnaryOpW>;
def riscv_ctzw   : SDNode<"RISCVISD::CTZW",   SDT_RISCVIntUnaryOpW>;
def riscv_rolw   : SDNode<"RISCVISD::ROLW",   SDT_RISCVIntBinOpW>;
def riscv_rorw   : SDNode<"RISCVISD::RORW",   SDT_RISCVIntBinOpW>;
def riscv_brev8  : SDNode<"RISCVISD::BREV8",  SDTIntUnaryOp>;
def riscv_orc_b  : SDNode<"RISCVISD::ORC_B",  SDTIntUnaryOp>;
def riscv_zip    : SDNode<"RISCVISD::ZIP",    SDTIntUnaryOp>;
def riscv_unzip  : SDNode<"RISCVISD::UNZIP",  SDTIntUnaryOp>;
def riscv_absw   : SDNode<"RISCVISD::ABSW",   SDTIntUnaryOp>;

def UImmLog2XLenHalfAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLenHalf";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLenHalf";
}

def shfl_uimm : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<5>(Imm);
  return isUInt<4>(Imm);
}]> {
  let ParserMatchClass = UImmLog2XLenHalfAsmOperand;
  let DecoderMethod = "decodeUImmOperand<5>";
  let OperandType = "OPERAND_UIMM_SHFL";
  let OperandNamespace = "RISCVOp";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    if (STI.getTargetTriple().isArch64Bit())
      return  isUInt<5>(Imm);
    return isUInt<4>(Imm);
  }];
}

def BCLRXForm : SDNodeXForm<imm, [{
  // Find the lowest 0.
  return CurDAG->getTargetConstant(countTrailingOnes(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

def SingleBitSetMaskToIndex : SDNodeXForm<imm, [{
  // Find the lowest 1.
  return CurDAG->getTargetConstant(countTrailingZeros(N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Checks if this mask has a single 0 bit and cannot be used with ANDI.
def BCLRMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(~Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}], BCLRXForm>;

// Checks if this mask has a single 1 bit and cannot be used with ORI/XORI.
def SingleBitSetMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(Imm);
}], SingleBitSetMaskToIndex>;

// Check if (or r, i) can be optimized to (BSETI (BSETI r, i0), i1),
// in which i = (1 << i0) | (1 << i1).
def BSETINVTwoBitsMask : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  // The immediate should not be a simm12.
  if (isInt<12>(N->getSExtValue()))
    return false;
  // The immediate must have exactly two bits set.
  return countPopulation(N->getZExtValue()) == 2;
}]>;

def BSETINVTwoBitsMaskHigh : SDNodeXForm<imm, [{
  uint64_t I = N->getZExtValue();
  return CurDAG->getTargetConstant(63 - countLeadingZeros(I), SDLoc(N),
                                   N->getValueType(0));
}]>;

// Check if (or r, imm) can be optimized to (BSETI (ORI r, i0), i1),
// in which imm = i0 | (1 << i1).
def BSETINVORIMask : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  // The immediate should not be a simm12.
  if (isInt<12>(N->getSExtValue()))
    return false;
  // There should be only one set bit from bit 11 to the top.
  return isPowerOf2_64(N->getZExtValue() & ~0x7ff);
}]>;

def BSETINVORIMaskLow : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0x7ff,
                                   SDLoc(N), N->getValueType(0));
}]>;

// Check if (and r, i) can be optimized to (BCLRI (BCLRI r, i0), i1),
// in which i = ~((1<<i0) | (1<<i1)).
def BCLRITwoBitsMask : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  // The immediate should not be a simm12.
  if (isInt<12>(N->getSExtValue()))
    return false;
  // The immediate must have exactly two bits clear.
  return countPopulation(N->getZExtValue()) == Subtarget->getXLen() - 2;
}]>;

def BCLRITwoBitsMaskLow : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(countTrailingZeros(~N->getZExtValue()),
                                   SDLoc(N), N->getValueType(0));
}]>;

def BCLRITwoBitsMaskHigh : SDNodeXForm<imm, [{
  uint64_t I = N->getSExtValue();
  if (!Subtarget->is64Bit())
    I |= 0xffffffffull << 32;
  return CurDAG->getTargetConstant(63 - countLeadingZeros(~I), SDLoc(N),
                                   N->getValueType(0));
}]>;

// Check if (and r, i) can be optimized to (BCLRI (ANDI r, i0), i1),
// in which i = i0 & ~(1<<i1).
def BCLRIANDIMask : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  // The immediate should not be a simm12.
  if (isInt<12>(N->getSExtValue()))
    return false;
  // There should be only one clear bit from bit 11 to the top.
  uint64_t I = N->getZExtValue() | 0x7ff;
  return Subtarget->is64Bit() ? isPowerOf2_64(~I) : isPowerOf2_32(~I);
}]>;

def BCLRIANDIMaskLow : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() & 0x7ff) | ~0x7ffull,
                                   SDLoc(N), N->getValueType(0));
}]>;

def C3LeftShift : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  return C > 3 && ((C % 3) == 0) && isPowerOf2_64(C / 3);
}]>;

def C5LeftShift : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  return C > 5 && ((C % 5) == 0) && isPowerOf2_64(C / 5);
}]>;

def C9LeftShift : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  return C > 9 && ((C % 9) == 0) && isPowerOf2_64(C / 9);
}]>;

// Constant of the form (3 << C) where C is less than 32.
def C3LeftShiftUW : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  if (C <= 3 || (C % 3) != 0)
    return false;
  C /= 3;
  return isPowerOf2_64(C) && C < (1ULL << 32);
}]>;

// Constant of the form (5 << C) where C is less than 32.
def C5LeftShiftUW : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  if (C <= 5 || (C % 5) != 0)
    return false;
  C /= 5;
  return isPowerOf2_64(C) && C < (1ULL << 32);
}]>;

// Constant of the form (9 << C) where C is less than 32.
def C9LeftShiftUW : PatLeaf<(imm), [{
  uint64_t C = N->getZExtValue();
  if (C <= 9 || (C % 9) != 0)
    return false;
  C /= 9;
  return isPowerOf2_64(C) && C < (1ULL << 32);
}]>;

def CSImm12MulBy4 : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  int64_t C = N->getSExtValue();
  // Skip if C is simm12, an lui, or can be optimized by the PatLeaf AddiPair.
  return !isInt<13>(C) && !isShiftedInt<20, 12>(C) && isShiftedInt<12, 2>(C);
}]>;

def CSImm12MulBy8 : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  int64_t C = N->getSExtValue();
  // Skip if C is simm12, an lui or can be optimized by the PatLeaf AddiPair or
  // CSImm12MulBy4.
  return !isInt<14>(C) && !isShiftedInt<20, 12>(C) && isShiftedInt<12, 3>(C);
}]>;

def SimmShiftRightBy2XForm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() >> 2, SDLoc(N),
                                   N->getValueType(0));
}]>;

def SimmShiftRightBy3XForm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() >> 3, SDLoc(N),
                                   N->getValueType(0));
}]>;

// Pattern to exclude simm12 immediates from matching.
def non_imm12 : PatLeaf<(XLenVT GPR:$a), [{
  auto *C = dyn_cast<ConstantSDNode>(N);
  return !C || !isInt<12>(C->getSExtValue());
}]>;

def Shifted32OnesMask : PatLeaf<(imm), [{
  uint64_t Imm = N->getZExtValue();
  if (!isShiftedMask_64(Imm))
    return false;

  unsigned TrailingZeros = countTrailingZeros(Imm);
  return TrailingZeros > 0 && TrailingZeros < 32 &&
         Imm == UINT64_C(0xFFFFFFFF) << TrailingZeros;
}], TrailingZeros>;

def sh1add_op : ComplexPattern<XLenVT, 1, "selectSHXADDOp<1>", [], [], 6>;
def sh2add_op : ComplexPattern<XLenVT, 1, "selectSHXADDOp<2>", [], [], 6>;
def sh3add_op : ComplexPattern<XLenVT, 1, "selectSHXADDOp<3>", [], [], 6>;

def sh1add_uw_op : ComplexPattern<XLenVT, 1, "selectSHXADD_UWOp<1>", [], [], 6>;
def sh2add_uw_op : ComplexPattern<XLenVT, 1, "selectSHXADD_UWOp<2>", [], [], 6>;
def sh3add_uw_op : ComplexPattern<XLenVT, 1, "selectSHXADD_UWOp<3>", [], [], 6>;

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

// Some of these templates should be moved to RISCVInstrFormats.td once the B
// extension has been ratified.

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBUnary<bits<7> funct7, bits<5> funct5, bits<3> funct3,
               RISCVOpcode opcode, string opcodestr>
    : RVInstR<funct7, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1"> {
  let rs2 = funct5;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShift_ri<bits<5> imm11_7, bits<3> funct3, RISCVOpcode opcode,
                  string opcodestr>
    : RVInstIShift<imm11_7, funct3, opcode, (outs GPR:$rd),
                   (ins GPR:$rs1, uimmlog2xlen:$shamt), opcodestr,
                   "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShiftW_ri<bits<7> imm11_5, bits<3> funct3, RISCVOpcode opcode,
                   string opcodestr>
    : RVInstIShiftW<imm11_5, funct3, opcode, (outs GPR:$rd),
                    (ins GPR:$rs1, uimm5:$shamt), opcodestr,
                    "$rd, $rs1, $shamt">;

// Using RVInstIShiftW since it allocates 5 bits instead of 6 to shamt.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShfl_ri<bits<7> imm11_5, bits<3> funct3, RISCVOpcode opcode,
                 string opcodestr>
    : RVInstIShiftW<imm11_5, funct3, opcode, (outs GPR:$rd),
                    (ins GPR:$rs1, shfl_uimm:$shamt), opcodestr,
                    "$rd, $rs1, $shamt">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryR<bits<2> funct2, bits<3> funct3, RISCVOpcode opcode,
                  string opcodestr, string argstr>
    : RVInstR4<funct2, funct3, opcode, (outs GPR:$rd),
               (ins GPR:$rs1, GPR:$rs2, GPR:$rs3), opcodestr, argstr>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbkb] in {
def ANDN  : ALU_rr<0b0100000, 0b111, "andn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def ORN   : ALU_rr<0b0100000, 0b110, "orn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def XNOR  : ALU_rr<0b0100000, 0b100, "xnor">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbbOrZbkb]

let Predicates = [HasStdExtZba] in {
def SH1ADD : ALU_rr<0b0010000, 0b010, "sh1add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
def SH2ADD : ALU_rr<0b0010000, 0b100, "sh2add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
def SH3ADD : ALU_rr<0b0010000, 0b110, "sh3add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
} // Predicates = [HasStdExtZba]

let Predicates = [HasStdExtZba, IsRV64] in {
def SLLI_UW : RVBShift_ri<0b00001, 0b001, OPC_OP_IMM_32, "slli.uw">,
              Sched<[WriteShiftImm32, ReadShiftImm32]>;
def ADD_UW : ALUW_rr<0b0000100, 0b000, "add.uw">,
             Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>;
def SH1ADD_UW : ALUW_rr<0b0010000, 0b010, "sh1add.uw">,
                Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
def SH2ADD_UW : ALUW_rr<0b0010000, 0b100, "sh2add.uw">,
                Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
def SH3ADD_UW : ALUW_rr<0b0010000, 0b110, "sh3add.uw">,
                Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbbOrZbkb] in {
def ROL   : ALU_rr<0b0110000, 0b001, "rol">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;
def ROR   : ALU_rr<0b0110000, 0b101, "ror">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;

def RORI  : RVBShift_ri<0b01100, 0b101, OPC_OP_IMM, "rori">,
            Sched<[WriteRotateImm, ReadRotateImm]>;
} // Predicates = [HasStdExtZbbOrZbkb]

let Predicates = [HasStdExtZbbOrZbkb, IsRV64] in {
def ROLW  : ALUW_rr<0b0110000, 0b001, "rolw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;
def RORW  : ALUW_rr<0b0110000, 0b101, "rorw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;

def RORIW : RVBShiftW_ri<0b0110000, 0b101, OPC_OP_IMM_32, "roriw">,
            Sched<[WriteRotateImm32, ReadRotateImm32]>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV64]

let Predicates = [HasStdExtZbs] in {
def BCLR : ALU_rr<0b0100100, 0b001, "bclr">,
           Sched<[WriteSingleBit, ReadSingleBit, ReadSingleBit]>;
def BSET : ALU_rr<0b0010100, 0b001, "bset">,
           Sched<[WriteSingleBit, ReadSingleBit, ReadSingleBit]>;
def BINV : ALU_rr<0b0110100, 0b001, "binv">,
           Sched<[WriteSingleBit, ReadSingleBit, ReadSingleBit]>;
def BEXT : ALU_rr<0b0100100, 0b101, "bext">,
           Sched<[WriteSingleBit, ReadSingleBit, ReadSingleBit]>;

def BCLRI : RVBShift_ri<0b01001, 0b001, OPC_OP_IMM, "bclri">,
            Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
def BSETI : RVBShift_ri<0b00101, 0b001, OPC_OP_IMM, "bseti">,
            Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
def BINVI : RVBShift_ri<0b01101, 0b001, OPC_OP_IMM, "binvi">,
            Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
def BEXTI : RVBShift_ri<0b01001, 0b101, OPC_OP_IMM, "bexti">,
            Sched<[WriteSingleBitImm, ReadSingleBitImm]>;
} // Predicates = [HasStdExtZbs]

// These instructions were named xperm.n and xperm.b in the last version of
// the draft bit manipulation specification they were included in. However, we
// use the mnemonics given to them in the ratified Zbkx extension.
let Predicates = [HasStdExtZbkx] in {
def XPERM4 : ALU_rr<0b0010100, 0b010, "xperm4">,
             Sched<[WriteXPERM, ReadXPERM, ReadXPERM]>;
def XPERM8 : ALU_rr<0b0010100, 0b100, "xperm8">,
             Sched<[WriteXPERM, ReadXPERM, ReadXPERM]>;
} // Predicates = [HasStdExtZbkx]

let Predicates = [HasStdExtZbb] in {
def CLZ  : RVBUnary<0b0110000, 0b00000, 0b001, OPC_OP_IMM, "clz">,
           Sched<[WriteCLZ, ReadCLZ]>;
def CTZ  : RVBUnary<0b0110000, 0b00001, 0b001, OPC_OP_IMM, "ctz">,
           Sched<[WriteCTZ, ReadCTZ]>;
def CPOP : RVBUnary<0b0110000, 0b00010, 0b001, OPC_OP_IMM, "cpop">,
           Sched<[WriteCPOP, ReadCPOP]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV64] in {
def CLZW   : RVBUnary<0b0110000, 0b00000, 0b001, OPC_OP_IMM_32, "clzw">,
             Sched<[WriteCLZ32, ReadCLZ32]>;
def CTZW   : RVBUnary<0b0110000, 0b00001, 0b001, OPC_OP_IMM_32, "ctzw">,
             Sched<[WriteCTZ32, ReadCTZ32]>;
def CPOPW  : RVBUnary<0b0110000, 0b00010, 0b001, OPC_OP_IMM_32, "cpopw">,
             Sched<[WriteCPOP32, ReadCPOP32]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbb] in {
def SEXT_B : RVBUnary<0b0110000, 0b00100, 0b001, OPC_OP_IMM, "sext.b">,
             Sched<[WriteIALU, ReadIALU]>;
def SEXT_H : RVBUnary<0b0110000, 0b00101, 0b001, OPC_OP_IMM, "sext.h">,
             Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbc] in {
def CLMULR : ALU_rr<0b0000101, 0b010, "clmulr", /*Commutable*/1>,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
} // Predicates = [HasStdExtZbc]

let Predicates = [HasStdExtZbcOrZbkc] in {
def CLMUL  : ALU_rr<0b0000101, 0b001, "clmul", /*Commutable*/1>,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
def CLMULH : ALU_rr<0b0000101, 0b011, "clmulh", /*Commutable*/1>,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
} // Predicates = [HasStdExtZbcOrZbkc]

let Predicates = [HasStdExtZbb] in {
def MIN  : ALU_rr<0b0000101, 0b100, "min", /*Commutable*/1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MINU : ALU_rr<0b0000101, 0b101, "minu", /*Commutable*/1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MAX  : ALU_rr<0b0000101, 0b110, "max", /*Commutable*/1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MAXU : ALU_rr<0b0000101, 0b111, "maxu", /*Commutable*/1>,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbkb] in {
def PACK  : ALU_rr<0b0000100, 0b100, "pack">,
            Sched<[WritePACK, ReadPACK, ReadPACK]>;
def PACKH : ALU_rr<0b0000100, 0b111, "packh">,
            Sched<[WritePACK, ReadPACK, ReadPACK]>;
} // Predicates = [HasStdExtZbkb]

let Predicates = [HasStdExtZbkb, IsRV64] in
def PACKW  : ALUW_rr<0b0000100, 0b100, "packw">,
             Sched<[WritePACK32, ReadPACK32, ReadPACK32]>;

let Predicates = [HasStdExtZbb, IsRV32] in {
def ZEXT_H_RV32 : RVBUnary<0b0000100, 0b00000, 0b100, OPC_OP, "zext.h">,
                  Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb, IsRV32]

let Predicates = [HasStdExtZbb, IsRV64] in {
def ZEXT_H_RV64 : RVBUnary<0b0000100, 0b00000, 0b100, OPC_OP_32, "zext.h">,
                  Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbkb, IsRV32] in {
def REV8_RV32 : RVBUnary<0b0110100, 0b11000, 0b101, OPC_OP_IMM, "rev8">,
                Sched<[WriteREV8, ReadREV8]>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV32]

let Predicates = [HasStdExtZbbOrZbkb, IsRV64] in {
def REV8_RV64 : RVBUnary<0b0110101, 0b11000, 0b101, OPC_OP_IMM, "rev8">,
                Sched<[WriteREV8, ReadREV8]>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV64]

let Predicates = [HasStdExtZbb] in {
def ORC_B : RVBUnary<0b0010100, 0b00111, 0b101, OPC_OP_IMM, "orc.b">,
            Sched<[WriteORCB, ReadORCB]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbkb] in
def BREV8 : RVBUnary<0b0110100, 0b00111, 0b101, OPC_OP_IMM, "brev8">,
            Sched<[WriteBREV8, ReadBREV8]>;

let Predicates = [HasStdExtZbkb, IsRV32] in {
def ZIP_RV32   : RVBUnary<0b0000100, 0b01111, 0b001, OPC_OP_IMM, "zip">,
                 Sched<[WriteZIP, ReadZIP]>;
def UNZIP_RV32 : RVBUnary<0b0000100, 0b01111, 0b101, OPC_OP_IMM, "unzip">,
                 Sched<[WriteZIP, ReadZIP]>;
} // Predicates = [HasStdExtZbkb, IsRV32]


//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZba, IsRV64] in {
def : InstAlias<"zext.w $rd, $rs", (ADD_UW GPR:$rd, GPR:$rs, X0)>;
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbb] in {
def : InstAlias<"ror $rd, $rs1, $shamt",
                (RORI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : InstAlias<"rorw $rd, $rs1, $shamt",
                (RORIW  GPR:$rd, GPR:$rs1, uimm5:$shamt), 0>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbs] in {
def : InstAlias<"bset $rd, $rs1, $shamt",
                (BSETI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
def : InstAlias<"bclr $rd, $rs1, $shamt",
                (BCLRI GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
def : InstAlias<"binv $rd, $rs1, $shamt",
                (BINVI GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
def : InstAlias<"bext $rd, $rs1, $shamt",
                (BEXTI GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
} // Predicates = [HasStdExtZbs]

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbkb] in {
def : Pat<(and GPR:$rs1, (not GPR:$rs2)), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(or  GPR:$rs1, (not GPR:$rs2)), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, (not GPR:$rs2)), (XNOR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbkb]

let Predicates = [HasStdExtZbbOrZbkb] in {
def : PatGprGpr<shiftop<rotl>, ROL>;
def : PatGprGpr<shiftop<rotr>, ROR>;

def : PatGprImm<rotr, RORI, uimmlog2xlen>;
// There's no encoding for roli in the the 'B' extension as it can be
// implemented with rori by negating the immediate.
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbbOrZbkb]

let Predicates = [HasStdExtZbbOrZbkb, IsRV64] in {
def : PatGprGpr<shiftopw<riscv_rolw>, ROLW>;
def : PatGprGpr<shiftopw<riscv_rorw>, RORW>;
def : PatGprImm<riscv_rorw, RORIW, uimm5>;
def : Pat<(riscv_rolw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, (ImmSubFrom32 uimm5:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV64]

let Predicates = [HasStdExtZbs] in {
def : Pat<(and (not (shiftop<shl> 1, GPR:$rs2)), GPR:$rs1),
          (BCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (rotl -2, GPR:$rs2), GPR:$rs1), (BCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (BSET GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (BINV GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (shiftop<srl> GPR:$rs1, GPR:$rs2), 1),
          (BEXT GPR:$rs1, GPR:$rs2)>;

def : Pat<(shiftop<shl> 1, GPR:$rs2),
          (BSET X0, GPR:$rs2)>;

def : Pat<(and GPR:$rs1, BCLRMask:$mask),
          (BCLRI GPR:$rs1, BCLRMask:$mask)>;
def : Pat<(or GPR:$rs1, SingleBitSetMask:$mask),
          (BSETI GPR:$rs1, SingleBitSetMask:$mask)>;
def : Pat<(xor GPR:$rs1, SingleBitSetMask:$mask),
          (BINVI GPR:$rs1, SingleBitSetMask:$mask)>;

def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), (XLenVT 1)),
          (BEXTI GPR:$rs1, uimmlog2xlen:$shamt)>;

def : Pat<(seteq (and GPR:$rs1, SingleBitSetMask:$mask), 0),
          (BEXTI (XORI GPR:$rs1, -1), SingleBitSetMask:$mask)>;

def : Pat<(or GPR:$r, BSETINVTwoBitsMask:$i),
          (BSETI (BSETI GPR:$r, (TrailingZeros BSETINVTwoBitsMask:$i)),
                 (BSETINVTwoBitsMaskHigh BSETINVTwoBitsMask:$i))>;
def : Pat<(xor GPR:$r, BSETINVTwoBitsMask:$i),
          (BINVI (BINVI GPR:$r, (TrailingZeros BSETINVTwoBitsMask:$i)),
                 (BSETINVTwoBitsMaskHigh BSETINVTwoBitsMask:$i))>;
def : Pat<(or GPR:$r, BSETINVORIMask:$i),
          (BSETI (ORI GPR:$r, (BSETINVORIMaskLow BSETINVORIMask:$i)),
                 (BSETINVTwoBitsMaskHigh BSETINVORIMask:$i))>;
def : Pat<(xor GPR:$r, BSETINVORIMask:$i),
          (BINVI (XORI GPR:$r, (BSETINVORIMaskLow BSETINVORIMask:$i)),
                 (BSETINVTwoBitsMaskHigh BSETINVORIMask:$i))>;
def : Pat<(and GPR:$r, BCLRITwoBitsMask:$i),
          (BCLRI (BCLRI GPR:$r, (BCLRITwoBitsMaskLow BCLRITwoBitsMask:$i)),
                 (BCLRITwoBitsMaskHigh BCLRITwoBitsMask:$i))>;
def : Pat<(and GPR:$r, BCLRIANDIMask:$i),
          (BCLRI (ANDI GPR:$r, (BCLRIANDIMaskLow BCLRIANDIMask:$i)),
                 (BCLRITwoBitsMaskHigh BCLRIANDIMask:$i))>;
} // Predicates = [HasStdExtZbs]

let Predicates = [HasStdExtZbb] in {
def : Pat<(riscv_orc_b GPR:$rs1), (ORC_B GPR:$rs1)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbkb] in {
def : Pat<(riscv_brev8 GPR:$rs1), (BREV8 GPR:$rs1)>;
} // Predicates = [HasStdExtZbkb]

let Predicates = [HasStdExtZbkb, IsRV32] in {
// We treat zip and unzip as separate instructions, so match it directly.
def : Pat<(i32 (riscv_zip GPR:$rs1)), (ZIP_RV32 GPR:$rs1)>;
def : Pat<(i32 (riscv_unzip GPR:$rs1)), (UNZIP_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbkb, IsRV32]

let Predicates = [HasStdExtZbb] in {
def : PatGpr<ctlz, CLZ>;
def : PatGpr<cttz, CTZ>;
def : PatGpr<ctpop, CPOP>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : PatGpr<riscv_clzw, CLZW>;
def : PatGpr<riscv_ctzw, CTZW>;
def : Pat<(i64 (ctpop (i64 (zexti32 (i64 GPR:$rs1))))), (CPOPW GPR:$rs1)>;

def : Pat<(i64 (riscv_absw GPR:$rs1)),
          (MAX GPR:$rs1, (SUBW X0, GPR:$rs1))>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbb] in {
def : Pat<(sext_inreg GPR:$rs1, i8), (SEXT_B GPR:$rs1)>;
def : Pat<(sext_inreg GPR:$rs1, i16), (SEXT_H GPR:$rs1)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb] in {
def : PatGprGpr<smin, MIN>;
def : PatGprGpr<smax, MAX>;
def : PatGprGpr<umin, MINU>;
def : PatGprGpr<umax, MAXU>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbbOrZbkb, IsRV32] in {
def : Pat<(i32 (bswap GPR:$rs1)), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV32]

let Predicates = [HasStdExtZbbOrZbkb, IsRV64] in {
def : Pat<(i64 (bswap GPR:$rs1)), (REV8_RV64 GPR:$rs1)>;
} // Predicates = [HasStdExtZbbOrZbkb, IsRV64]

let Predicates = [HasStdExtZbkb] in {
def : Pat<(or (and (shl GPR:$rs2, (XLenVT 8)), 0xFFFF),
              (zexti8 GPR:$rs1)),
          (PACKH GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (shl (zexti8 GPR:$rs2), (XLenVT 8)),
              (zexti8 GPR:$rs1)),
          (PACKH GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (or (shl GPR:$rs2, (XLenVT 8)),
                   (zexti8 GPR:$rs1)), 0xFFFF),
          (PACKH GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbkb]

let Predicates = [HasStdExtZbkb, IsRV32] in
def : Pat<(i32 (or (zexti16 GPR:$rs1), (shl GPR:$rs2, (i32 16)))),
          (PACK GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbkb, IsRV64] in {
def : Pat<(i64 (or (zexti32 GPR:$rs1), (shl GPR:$rs2, (i64 32)))),
          (PACK GPR:$rs1, GPR:$rs2)>;

def : Pat<(binop_allwusers<or> (shl GPR:$rs2, (i64 16)),
                               (zexti16 GPR:$rs1)),
          (PACKW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (or (sext_inreg (shl GPR:$rs2, (i64 16)), i32),
                   (zexti16 GPR:$rs1))),
          (PACKW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbkb, IsRV64]

let Predicates = [HasStdExtZbb, IsRV32] in
def : Pat<(i32 (and GPR:$rs, 0xFFFF)), (ZEXT_H_RV32 GPR:$rs)>;
let Predicates = [HasStdExtZbb, IsRV64] in
def : Pat<(i64 (and GPR:$rs, 0xFFFF)), (ZEXT_H_RV64 GPR:$rs)>;

let Predicates = [HasStdExtZba] in {
def : Pat<(add (shl GPR:$rs1, (XLenVT 1)), non_imm12:$rs2),
          (SH1ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 2)), non_imm12:$rs2),
          (SH2ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 3)), non_imm12:$rs2),
          (SH3ADD GPR:$rs1, GPR:$rs2)>;

// More complex cases use a ComplexPattern.
def : Pat<(add sh1add_op:$rs1, non_imm12:$rs2),
          (SH1ADD sh1add_op:$rs1, GPR:$rs2)>;
def : Pat<(add sh2add_op:$rs1, non_imm12:$rs2),
          (SH2ADD sh2add_op:$rs1, GPR:$rs2)>;
def : Pat<(add sh3add_op:$rs1, non_imm12:$rs2),
          (SH3ADD sh3add_op:$rs1, GPR:$rs2)>;

def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 6)), GPR:$rs2),
          (SH1ADD (SH1ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 10)), GPR:$rs2),
          (SH1ADD (SH2ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 18)), GPR:$rs2),
          (SH1ADD (SH3ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 12)), GPR:$rs2),
          (SH2ADD (SH1ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 20)), GPR:$rs2),
          (SH2ADD (SH2ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 36)), GPR:$rs2),
          (SH2ADD (SH3ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 24)), GPR:$rs2),
          (SH3ADD (SH1ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 40)), GPR:$rs2),
          (SH3ADD (SH2ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;
def : Pat<(add (mul_oneuse GPR:$rs1, (XLenVT 72)), GPR:$rs2),
          (SH3ADD (SH3ADD GPR:$rs1, GPR:$rs1), GPR:$rs2)>;

def : Pat<(add GPR:$r, CSImm12MulBy4:$i),
          (SH2ADD (ADDI X0, (SimmShiftRightBy2XForm CSImm12MulBy4:$i)),
                  GPR:$r)>;
def : Pat<(add GPR:$r, CSImm12MulBy8:$i),
          (SH3ADD (ADDI X0, (SimmShiftRightBy3XForm CSImm12MulBy8:$i)),
                  GPR:$r)>;

def : Pat<(mul GPR:$r, C3LeftShift:$i),
          (SLLI (SH1ADD GPR:$r, GPR:$r),
                (TrailingZeros C3LeftShift:$i))>;
def : Pat<(mul GPR:$r, C5LeftShift:$i),
          (SLLI (SH2ADD GPR:$r, GPR:$r),
                (TrailingZeros C5LeftShift:$i))>;
def : Pat<(mul GPR:$r, C9LeftShift:$i),
          (SLLI (SH3ADD GPR:$r, GPR:$r),
                (TrailingZeros C9LeftShift:$i))>;

def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 11)),
          (SH1ADD (SH2ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 19)),
          (SH1ADD (SH3ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 13)),
          (SH2ADD (SH1ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 21)),
          (SH2ADD (SH2ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 37)),
          (SH2ADD (SH3ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 25)),
          (SH3ADD (SH1ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 41)),
          (SH3ADD (SH2ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 73)),
          (SH3ADD (SH3ADD GPR:$r, GPR:$r), GPR:$r)>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 27)),
          (SH1ADD (SH3ADD GPR:$r, GPR:$r), (SH3ADD GPR:$r, GPR:$r))>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 45)),
          (SH2ADD (SH3ADD GPR:$r, GPR:$r), (SH3ADD GPR:$r, GPR:$r))>;
def : Pat<(mul_const_oneuse GPR:$r, (XLenVT 81)),
          (SH3ADD (SH3ADD GPR:$r, GPR:$r), (SH3ADD GPR:$r, GPR:$r))>;
} // Predicates = [HasStdExtZba]

let Predicates = [HasStdExtZba, IsRV64] in {
def : Pat<(i64 (shl (and GPR:$rs1, 0xFFFFFFFF), uimm5:$shamt)),
          (SLLI_UW GPR:$rs1, uimm5:$shamt)>;
// Match a shifted 0xffffffff mask. Use SRLI to clear the LSBs and SLLI_UW to
// mask and shift.
def : Pat<(i64 (and GPR:$rs1, Shifted32OnesMask:$mask)),
          (SLLI_UW (SRLI GPR:$rs1, Shifted32OnesMask:$mask),
                   Shifted32OnesMask:$mask)>;

def : Pat<(i64 (add (and GPR:$rs1, 0xFFFFFFFF), non_imm12:$rs2)),
          (ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (and GPR:$rs, 0xFFFFFFFF)), (ADD_UW GPR:$rs, X0)>;

def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 1)), non_imm12:$rs2)),
          (SH1ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 2)), non_imm12:$rs2)),
          (SH2ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 3)), non_imm12:$rs2)),
          (SH3ADD_UW GPR:$rs1, GPR:$rs2)>;

def : Pat<(i64 (add (and (shl GPR:$rs1, (i64 1)), 0x1FFFFFFFF), non_imm12:$rs2)),
          (SH1ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (and (shl GPR:$rs1, (i64 2)), 0x3FFFFFFFF), non_imm12:$rs2)),
          (SH2ADD_UW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (and (shl GPR:$rs1, (i64 3)), 0x7FFFFFFFF), non_imm12:$rs2)),
          (SH3ADD_UW GPR:$rs1, GPR:$rs2)>;

// More complex cases use a ComplexPattern.
def : Pat<(add sh1add_uw_op:$rs1, non_imm12:$rs2),
          (SH1ADD_UW sh1add_uw_op:$rs1, GPR:$rs2)>;
def : Pat<(add sh2add_uw_op:$rs1, non_imm12:$rs2),
          (SH2ADD_UW sh2add_uw_op:$rs1, GPR:$rs2)>;
def : Pat<(add sh3add_uw_op:$rs1, non_imm12:$rs2),
          (SH3ADD_UW sh3add_uw_op:$rs1, GPR:$rs2)>;

def : Pat<(i64 (add (and GPR:$rs1, 0xFFFFFFFE), non_imm12:$rs2)),
          (SH1ADD (SRLIW GPR:$rs1, 1), GPR:$rs2)>;
def : Pat<(i64 (add (and GPR:$rs1, 0xFFFFFFFC), non_imm12:$rs2)),
          (SH2ADD (SRLIW GPR:$rs1, 2), GPR:$rs2)>;
def : Pat<(i64 (add (and GPR:$rs1, 0xFFFFFFF8), non_imm12:$rs2)),
          (SH3ADD (SRLIW GPR:$rs1, 3), GPR:$rs2)>;

// Use SRLI to clear the LSBs and SHXADD_UW to mask and shift.
def : Pat<(i64 (add (and GPR:$rs1, 0x1FFFFFFFE), non_imm12:$rs2)),
          (SH1ADD_UW (SRLI GPR:$rs1, 1), GPR:$rs2)>;
def : Pat<(i64 (add (and GPR:$rs1, 0x3FFFFFFFC), non_imm12:$rs2)),
          (SH2ADD_UW (SRLI GPR:$rs1, 2), GPR:$rs2)>;
def : Pat<(i64 (add (and GPR:$rs1, 0x7FFFFFFF8), non_imm12:$rs2)),
          (SH3ADD_UW (SRLI GPR:$rs1, 3), GPR:$rs2)>;

def : Pat<(mul (binop_oneuse<and> GPR:$r, 0xFFFFFFFF), C3LeftShiftUW:$i),
          (SH1ADD (SLLI_UW GPR:$r, (TrailingZeros C3LeftShiftUW:$i)),
                  (SLLI_UW GPR:$r, (TrailingZeros C3LeftShiftUW:$i)))>;
def : Pat<(mul (binop_oneuse<and> GPR:$r, 0xFFFFFFFF), C5LeftShiftUW:$i),
          (SH2ADD (SLLI_UW GPR:$r, (TrailingZeros C5LeftShiftUW:$i)),
                  (SLLI_UW GPR:$r, (TrailingZeros C5LeftShiftUW:$i)))>;
def : Pat<(mul (binop_oneuse<and> GPR:$r, 0xFFFFFFFF), C9LeftShiftUW:$i),
          (SH3ADD (SLLI_UW GPR:$r, (TrailingZeros C9LeftShiftUW:$i)),
                  (SLLI_UW GPR:$r, (TrailingZeros C9LeftShiftUW:$i)))>;
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbcOrZbkc] in {
def : PatGprGpr<int_riscv_clmul, CLMUL>;
def : PatGprGpr<int_riscv_clmulh, CLMULH>;
} // Predicates = [HasStdExtZbcOrZbkc]

let Predicates = [HasStdExtZbc] in
def : PatGprGpr<int_riscv_clmulr, CLMULR>;

let Predicates = [HasStdExtZbkx] in {
def : PatGprGpr<int_riscv_xperm4, XPERM4>;
def : PatGprGpr<int_riscv_xperm8, XPERM8>;
} // Predicates = [HasStdExtZbkx]
