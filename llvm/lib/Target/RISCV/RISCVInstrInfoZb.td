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
//   Zbe - 0.93 *experimental
//   Zbf - 0.93 *experimental
//   Zbm - 0.93 *experimental
//   Zbp - 0.93 *experimental
//   Zbr - 0.93 *experimental
//   Zbt - 0.93 *experimental
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
def riscv_fslw   : SDNode<"RISCVISD::FSLW",   SDT_RISCVIntShiftDOpW>;
def riscv_fsrw   : SDNode<"RISCVISD::FSRW",   SDT_RISCVIntShiftDOpW>;
def riscv_fsl    : SDNode<"RISCVISD::FSL",    SDTIntShiftDOp>;
def riscv_fsr    : SDNode<"RISCVISD::FSR",    SDTIntShiftDOp>;
def riscv_grev   : SDNode<"RISCVISD::GREV",   SDTIntBinOp>;
def riscv_grevw  : SDNode<"RISCVISD::GREVW",  SDT_RISCVIntBinOpW>;
def riscv_gorc   : SDNode<"RISCVISD::GORC",   SDTIntBinOp>;
def riscv_gorcw  : SDNode<"RISCVISD::GORCW",  SDT_RISCVIntBinOpW>;
def riscv_shfl   : SDNode<"RISCVISD::SHFL",   SDTIntBinOp>;
def riscv_shflw  : SDNode<"RISCVISD::SHFLW",  SDT_RISCVIntBinOpW>;
def riscv_unshfl : SDNode<"RISCVISD::UNSHFL", SDTIntBinOp>;
def riscv_unshflw: SDNode<"RISCVISD::UNSHFLW",SDT_RISCVIntBinOpW>;
def riscv_bfp    : SDNode<"RISCVISD::BFP",    SDTIntBinOp>;
def riscv_bfpw   : SDNode<"RISCVISD::BFPW",   SDT_RISCVIntBinOpW>;
def riscv_bcompress    : SDNode<"RISCVISD::BCOMPRESS",   SDTIntBinOp>;
def riscv_bcompressw   : SDNode<"RISCVISD::BCOMPRESSW",  SDT_RISCVIntBinOpW>;
def riscv_bdecompress  : SDNode<"RISCVISD::BDECOMPRESS", SDTIntBinOp>;
def riscv_bdecompressw : SDNode<"RISCVISD::BDECOMPRESSW",SDT_RISCVIntBinOpW>;

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
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingOnes(),
                                   SDLoc(N), N->getValueType(0));
}]>;

def BSETINVXForm : SDNodeXForm<imm, [{
  // Find the lowest 1.
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingZeros(),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Checks if this mask has a single 0 bit and cannot be used with ANDI.
def BCLRMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(~Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}], BCLRXForm>;

// Checks if this mask has a single 1 bit and cannot be used with ORI/XORI.
def BSETINVMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(Imm);
}], BSETINVXForm>;

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

def TrailingZerosXForm : SDNodeXForm<imm, [{
  uint64_t I = N->getZExtValue();
  return CurDAG->getTargetConstant(countTrailingZeros(I), SDLoc(N),
                                   N->getValueType(0));
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

def CSImm12MulBy4 : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  int64_t C = N->getSExtValue();
  // Skip if C is simm12 or can be optimized by the PatLeaf AddiPair.
  return !isInt<13>(C) && isInt<14>(C) && (C & 3) == 0;
}]>;

def CSImm12MulBy8 : PatLeaf<(imm), [{
  if (!N->hasOneUse())
    return false;
  int64_t C = N->getSExtValue();
  // Skip if C is simm12 or can be optimized by the PatLeaf AddiPair.
  return !isInt<13>(C) && isInt<15>(C) && (C & 7) == 0;
}]>;

def SimmShiftRightBy2XForm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() >> 2, SDLoc(N),
                                   N->getValueType(0));
}]>;

def SimmShiftRightBy3XForm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue() >> 3, SDLoc(N),
                                   N->getValueType(0));
}]>;

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

// Currently used by FSRI only
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryImm6<bits<3> funct3, RISCVOpcode opcode,
                     string opcodestr, string argstr>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt),
             opcodestr, argstr, [], InstFormatR4> {
  bits<5> rs3;
  bits<6> shamt;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-27} = rs3;
  let Inst{26} = 1;
  let Inst{25-20} = shamt;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Opcode = opcode.Value;
}

// Currently used by FSRIW only
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryImm5<bits<2> funct2, bits<3> funct3, RISCVOpcode opcode,
                     string opcodestr, string argstr>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs3, uimm5:$shamt),
             opcodestr, argstr, [], InstFormatR4> {
  bits<5> rs3;
  bits<5> shamt;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-27} = rs3;
  let Inst{26-25} = funct2;
  let Inst{24-20} = shamt;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Opcode = opcode.Value;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbpOrZbkb] in {
def ANDN  : ALU_rr<0b0100000, 0b111, "andn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def ORN   : ALU_rr<0b0100000, 0b110, "orn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def XNOR  : ALU_rr<0b0100000, 0b100, "xnor">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb]

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
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbpOrZbkb] in {
def ROL   : ALU_rr<0b0110000, 0b001, "rol">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;
def ROR   : ALU_rr<0b0110000, 0b101, "ror">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;

def RORI  : RVBShift_ri<0b01100, 0b101, OPC_OP_IMM, "rori">,
            Sched<[WriteRotateImm, ReadRotateImm]>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb]

let Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV64] in {
def ROLW  : ALUW_rr<0b0110000, 0b001, "rolw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;
def RORW  : ALUW_rr<0b0110000, 0b101, "rorw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;

def RORIW : RVBShiftW_ri<0b0110000, 0b101, OPC_OP_IMM_32, "roriw">,
            Sched<[WriteRotateImm32, ReadRotateImm32]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

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

let Predicates = [HasStdExtZbp] in {
def GORC : ALU_rr<0b0010100, 0b101, "gorc">, Sched<[]>;
def GREV : ALU_rr<0b0110100, 0b101, "grev">, Sched<[]>;

def GREVI : RVBShift_ri<0b01101, 0b101, OPC_OP_IMM, "grevi">, Sched<[]>;
def GORCI : RVBShift_ri<0b00101, 0b101, OPC_OP_IMM, "gorci">, Sched<[]>;

def SHFL   : ALU_rr<0b0000100, 0b001, "shfl">, Sched<[]>;
def UNSHFL : ALU_rr<0b0000100, 0b101, "unshfl">, Sched<[]>;

def SHFLI   : RVBShfl_ri<0b0000100, 0b001, OPC_OP_IMM, "shfli">, Sched<[]>;
def UNSHFLI : RVBShfl_ri<0b0000100, 0b101, OPC_OP_IMM, "unshfli">, Sched<[]>;

def XPERM_H : ALU_rr<0b0010100, 0b110, "xperm.h">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV64] in {
def GORCW  : ALUW_rr<0b0010100, 0b101, "gorcw">, Sched<[]>;
def GREVW  : ALUW_rr<0b0110100, 0b101, "grevw">, Sched<[]>;

def GORCIW : RVBShiftW_ri<0b0010100, 0b101, OPC_OP_IMM_32, "gorciw">, Sched<[]>;
def GREVIW : RVBShiftW_ri<0b0110100, 0b101, OPC_OP_IMM_32, "greviw">, Sched<[]>;

def SHFLW   : ALUW_rr<0b0000100, 0b001, "shflw">, Sched<[]>;
def UNSHFLW : ALUW_rr<0b0000100, 0b101, "unshflw">, Sched<[]>;

def XPERM_W : ALU_rr<0b0010100, 0b000, "xperm.w">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

// These instructions were named xperm.n and xperm.b in the last version of
// the draft bit manipulation specification they were included in. However, we
// use the mnemonics given to them in the ratified Zbkx extension.
let Predicates = [HasStdExtZbpOrZbkx] in {
def XPERM4 : ALU_rr<0b0010100, 0b010, "xperm4">, Sched<[]>;
def XPERM8 : ALU_rr<0b0010100, 0b100, "xperm8">, Sched<[]>;
} // Predicates = [HasStdExtZbpOrZbkx]

let Predicates = [HasStdExtZbt] in {
def CMIX : RVBTernaryR<0b11, 0b001, OPC_OP, "cmix", "$rd, $rs2, $rs1, $rs3">,
           Sched<[]>;
def CMOV : RVBTernaryR<0b11, 0b101, OPC_OP, "cmov", "$rd, $rs2, $rs1, $rs3">,
           Sched<[]>;
def FSL  : RVBTernaryR<0b10, 0b001, OPC_OP, "fsl", "$rd, $rs1, $rs3, $rs2">,
           Sched<[]>;
def FSR  : RVBTernaryR<0b10, 0b101, OPC_OP, "fsr", "$rd, $rs1, $rs3, $rs2">,
           Sched<[]>;
def FSRI : RVBTernaryImm6<0b101, OPC_OP_IMM, "fsri",
                          "$rd, $rs1, $rs3, $shamt">, Sched<[]>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbt, IsRV64] in {
def FSLW  : RVBTernaryR<0b10, 0b001, OPC_OP_32,
                        "fslw", "$rd, $rs1, $rs3, $rs2">, Sched<[]>;
def FSRW  : RVBTernaryR<0b10, 0b101, OPC_OP_32, "fsrw",
                        "$rd, $rs1, $rs3, $rs2">, Sched<[]>;
def FSRIW : RVBTernaryImm5<0b10, 0b101, OPC_OP_IMM_32,
                           "fsriw", "$rd, $rs1, $rs3, $shamt">, Sched<[]>;
} // Predicates = [HasStdExtZbt, IsRV64]

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

let Predicates = [HasStdExtZbr] in {
def CRC32_B : RVBUnary<0b0110000, 0b10000, 0b001, OPC_OP_IMM, "crc32.b">,
              Sched<[]>;
def CRC32_H : RVBUnary<0b0110000, 0b10001, 0b001, OPC_OP_IMM, "crc32.h">,
              Sched<[]>;
def CRC32_W : RVBUnary<0b0110000, 0b10010, 0b001, OPC_OP_IMM, "crc32.w">,
              Sched<[]>;

def CRC32C_B : RVBUnary<0b0110000, 0b11000, 0b001, OPC_OP_IMM, "crc32c.b">,
               Sched<[]>;
def CRC32C_H : RVBUnary<0b0110000, 0b11001, 0b001, OPC_OP_IMM, "crc32c.h">,
               Sched<[]>;
def CRC32C_W : RVBUnary<0b0110000, 0b11010, 0b001, OPC_OP_IMM, "crc32c.w">,
               Sched<[]>;
} // Predicates = [HasStdExtZbr]

let Predicates = [HasStdExtZbr, IsRV64] in {
def CRC32_D  : RVBUnary<0b0110000, 0b10011, 0b001, OPC_OP_IMM, "crc32.d">,
               Sched<[]>;

def CRC32C_D : RVBUnary<0b0110000, 0b11011, 0b001, OPC_OP_IMM, "crc32c.d">,
               Sched<[]>;
} // Predicates = [HasStdExtZbr, IsRV64]

let Predicates = [HasStdExtZbc] in {
def CLMULR : ALU_rr<0b0000101, 0b010, "clmulr">,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
} // Predicates = [HasStdExtZbc]

let Predicates = [HasStdExtZbcOrZbkc] in {
def CLMUL  : ALU_rr<0b0000101, 0b001, "clmul">,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
def CLMULH : ALU_rr<0b0000101, 0b011, "clmulh">,
             Sched<[WriteCLMUL, ReadCLMUL, ReadCLMUL]>;
} // Predicates = [HasStdExtZbcOrZbkc]

let Predicates = [HasStdExtZbb] in {
def MIN  : ALU_rr<0b0000101, 0b100, "min">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MINU : ALU_rr<0b0000101, 0b101, "minu">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MAX  : ALU_rr<0b0000101, 0b110, "max">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def MAXU : ALU_rr<0b0000101, 0b111, "maxu">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbe] in {
// NOTE: These mnemonics are from the 0.94 spec. There is a name conflict with
// bext in the 0.93 spec.
def BDECOMPRESS : ALU_rr<0b0100100, 0b110, "bdecompress">, Sched<[]>;
def BCOMPRESS   : ALU_rr<0b0000100, 0b110, "bcompress">, Sched<[]>;
} // Predicates = [HasStdExtZbe]

let Predicates = [HasStdExtZbe, IsRV64] in {
// NOTE: These mnemonics are from the 0.94 spec. There is a name conflict with
// bextw in the 0.93 spec.
def BDECOMPRESSW : ALUW_rr<0b0100100, 0b110, "bdecompressw">, Sched<[]>;
def BCOMPRESSW   : ALUW_rr<0b0000100, 0b110, "bcompressw">, Sched<[]>;
} // Predicates = [HasStdExtZbe, IsRV64]

let Predicates = [HasStdExtZbpOrZbkb] in {
def PACK  : ALU_rr<0b0000100, 0b100, "pack">, Sched<[]>;
def PACKH : ALU_rr<0b0000100, 0b111, "packh">, Sched<[]>;
} // Predicates = [HasStdExtZbpOrZbkb]

let Predicates = [HasStdExtZbpOrZbkb, IsRV64] in 
def PACKW  : ALUW_rr<0b0000100, 0b100, "packw">, Sched<[]>;

let Predicates = [HasStdExtZbp] in 
def PACKU : ALU_rr<0b0100100, 0b100, "packu">, Sched<[]>;

let Predicates = [HasStdExtZbp, IsRV64] in 
def PACKUW : ALUW_rr<0b0100100, 0b100, "packuw">, Sched<[]>;

let Predicates = [HasStdExtZbm, IsRV64] in {
def BMATFLIP : RVBUnary<0b0110000, 0b00011, 0b001, OPC_OP_IMM, "bmatflip">,
               Sched<[]>;

def BMATOR   : ALU_rr<0b0000100, 0b011, "bmator">, Sched<[]>;
def BMATXOR  : ALU_rr<0b0100100, 0b011, "bmatxor">, Sched<[]>;
} // Predicates = [HasStdExtZbm, IsRV64]

let Predicates = [HasStdExtZbf] in
def BFP : ALU_rr<0b0100100, 0b111, "bfp">,
          Sched<[WriteBFP, ReadBFP, ReadBFP]>;

let Predicates = [HasStdExtZbf, IsRV64] in
def BFPW : ALUW_rr<0b0100100, 0b111, "bfpw">,
           Sched<[WriteBFP32, ReadBFP32, ReadBFP32]>;

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in {
def ZEXT_H_RV32 : RVBUnary<0b0000100, 0b00000, 0b100, OPC_OP, "zext.h">,
                  Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV32]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def ZEXT_H_RV64 : RVBUnary<0b0000100, 0b00000, 0b100, OPC_OP_32, "zext.h">,
                  Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

// We treat rev8 and orc.b as standalone instructions even though they use a
// portion of the encodings for grevi and gorci. This allows us to support only
// those encodings when only Zbb is enabled. We do this even when grevi and
// gorci are available with Zbp. Trying to use 'HasStdExtZbb, NotHasStdExtZbp'
// causes diagnostics to suggest that Zbp rather than Zbb is required for rev8
// or gorci. Since Zbb is closer to being finalized than Zbp this will be
// misleading to users.
let Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV32] in {
def REV8_RV32 : RVBUnary<0b0110100, 0b11000, 0b101, OPC_OP_IMM, "rev8">,
                Sched<[WriteREV8, ReadREV8]>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV32]

let Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV64] in {
def REV8_RV64 : RVBUnary<0b0110101, 0b11000, 0b101, OPC_OP_IMM, "rev8">,
                Sched<[WriteREV8, ReadREV8]>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp] in {
def ORC_B : RVBUnary<0b0010100, 0b00111, 0b101, OPC_OP_IMM, "orc.b">,
            Sched<[WriteORCB, ReadORCB]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbpOrZbkb] in 
def BREV8 : RVBUnary<0b0110100, 0b00111, 0b101, OPC_OP_IMM, "brev8">;

let Predicates = [HasStdExtZbpOrZbkb, IsRV32] in {
def ZIP_RV32   : RVBUnary<0b0000100, 0b01111, 0b001, OPC_OP_IMM, "zip">;
def UNZIP_RV32 : RVBUnary<0b0000100, 0b01111, 0b101, OPC_OP_IMM, "unzip">;
} // Predicates = [HasStdExtZbpOrZbkb, IsRV32]


//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZba, IsRV64] in {
def : InstAlias<"zext.w $rd, $rs", (ADD_UW GPR:$rd, GPR:$rs, X0)>;
}

let Predicates = [HasStdExtZbp] in {
def : InstAlias<"rev.p $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00001)>;
def : InstAlias<"rev2.n $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00010)>;
def : InstAlias<"rev.n $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00011)>;
def : InstAlias<"rev4.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00100)>;
def : InstAlias<"rev2.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00110)>;
def : InstAlias<"rev8.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"rev4.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"rev2.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"rev.h $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b01111)>;
def : InstAlias<"rev.b $rd, $rs",  (BREV8 GPR:$rd, GPR:$rs)>;

def : InstAlias<"zip.n $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0001)>;
def : InstAlias<"unzip.n $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0001)>;
def : InstAlias<"zip2.b $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0010)>;
def : InstAlias<"unzip2.b $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0010)>;
def : InstAlias<"zip.b $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0011)>;
def : InstAlias<"unzip.b $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0011)>;
def : InstAlias<"zip4.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0100)>;
def : InstAlias<"unzip4.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0100)>;
def : InstAlias<"zip2.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0110)>;
def : InstAlias<"unzip2.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0110)>;
def : InstAlias<"zip.h $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0111)>;
def : InstAlias<"unzip.h $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0111)>;

def : InstAlias<"orc.p $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00001)>;
def : InstAlias<"orc2.n $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00010)>;
def : InstAlias<"orc.n $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00011)>;
def : InstAlias<"orc4.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00100)>;
def : InstAlias<"orc2.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00110)>;
// orc.b is considered an instruction rather than an alias.
def : InstAlias<"orc8.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"orc4.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"orc2.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"orc.h $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b01111)>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : InstAlias<"rev16 $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b10000)>;
// rev8 is considered an instruction rather than an alias.
def : InstAlias<"rev4 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"rev2 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"rev $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b11111)>;

def : InstAlias<"zip8 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1000)>;
def : InstAlias<"unzip8 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1000)>;
def : InstAlias<"zip4 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1100)>;
def : InstAlias<"unzip4 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1100)>;
def : InstAlias<"zip2 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1110)>;
def : InstAlias<"unzip2 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1110)>;
// zip and unzip are considered instructions rather than an alias.

def : InstAlias<"orc16 $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"orc8 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"orc4 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"orc2 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"orc $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b11111)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : InstAlias<"rev16.w $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b010000)>;
def : InstAlias<"rev8.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011000)>;
def : InstAlias<"rev4.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011100)>;
def : InstAlias<"rev2.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011110)>;
def : InstAlias<"rev.w $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b011111)>;
def : InstAlias<"rev32 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b100000)>;
def : InstAlias<"rev16 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b110000)>;
// rev8 is considered an instruction rather than an alias.
def : InstAlias<"rev4 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111100)>;
def : InstAlias<"rev2 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111110)>;
def : InstAlias<"rev $rd, $rs",     (GREVI GPR:$rd, GPR:$rs, 0b111111)>;

def : InstAlias<"zip8.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"unzip8.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"zip4.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"unzip4.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"zip2.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"unzip2.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"zip.w $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b01111)>;
def : InstAlias<"unzip.w $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b01111)>;
def : InstAlias<"zip16 $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"unzip16 $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b10000)>;
def : InstAlias<"zip8 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"unzip8 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11000)>;
def : InstAlias<"zip4 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"unzip4 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11100)>;
def : InstAlias<"zip2 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"unzip2 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11110)>;
def : InstAlias<"zip $rd, $rs",      (SHFLI   GPR:$rd, GPR:$rs, 0b11111)>;
def : InstAlias<"unzip $rd, $rs",    (UNSHFLI GPR:$rd, GPR:$rs, 0b11111)>;

def : InstAlias<"orc16.w $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b010000)>;
def : InstAlias<"orc8.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011000)>;
def : InstAlias<"orc4.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011100)>;
def : InstAlias<"orc2.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011110)>;
def : InstAlias<"orc.w $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b011111)>;
def : InstAlias<"orc32 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b100000)>;
def : InstAlias<"orc16 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b110000)>;
def : InstAlias<"orc8 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111000)>;
def : InstAlias<"orc4 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111100)>;
def : InstAlias<"orc2 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111110)>;
def : InstAlias<"orc $rd, $rs",     (GORCI GPR:$rd, GPR:$rs, 0b111111)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbbOrZbp] in {
def : InstAlias<"ror $rd, $rs1, $shamt",
                (RORI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : InstAlias<"rorw $rd, $rs1, $shamt",
                (RORIW  GPR:$rd, GPR:$rs1, uimm5:$shamt), 0>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbp] in {
def : InstAlias<"grev $rd, $rs1, $shamt",
                (GREVI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
def : InstAlias<"gorc $rd, $rs1, $shamt",
                (GORCI  GPR:$rd, GPR:$rs1, uimmlog2xlen:$shamt), 0>;
def : InstAlias<"shfl $rd, $rs1, $shamt",
                (SHFLI  GPR:$rd, GPR:$rs1, shfl_uimm:$shamt), 0>;
def : InstAlias<"unshfl $rd, $rs1, $shamt",
                (UNSHFLI  GPR:$rd, GPR:$rs1, shfl_uimm:$shamt), 0>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : InstAlias<"grevw $rd, $rs1, $shamt",
                (GREVIW  GPR:$rd, GPR:$rs1, uimm5:$shamt), 0>;
def : InstAlias<"gorcw $rd, $rs1, $shamt",
                (GORCIW  GPR:$rd, GPR:$rs1, uimm5:$shamt), 0>;
} // Predicates = [HasStdExtZbp, IsRV64]

// Zbp is unratified and that it would likely adopt the already ratified Zbkx names.
// Thus current Zbp instructions are defined as aliases for Zbkx instructions.
let Predicates = [HasStdExtZbp] in {
  def : InstAlias<"xperm.b $rd, $rs1, $rs2",
                  (XPERM8 GPR:$rd, GPR:$rs1, GPR:$rs2)>;
  def : InstAlias<"xperm.n $rd, $rs1, $rs2",
                  (XPERM4 GPR:$rd, GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbp]

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

let Predicates = [HasStdExtZbbOrZbpOrZbkb] in {
def : Pat<(and GPR:$rs1, (not GPR:$rs2)), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(or  GPR:$rs1, (not GPR:$rs2)), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, (not GPR:$rs2)), (XNOR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb]

let Predicates = [HasStdExtZbbOrZbpOrZbkb] in {
def : PatGprGpr<rotl, ROL>;
def : PatGprGpr<rotr, ROR>;

def : PatGprImm<rotr, RORI, uimmlog2xlen>;
// There's no encoding for roli in the the 'B' extension as it can be
// implemented with rori by negating the immediate.
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb]

let Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV64] in {
def : PatGprGpr<riscv_rolw, ROLW>;
def : PatGprGpr<riscv_rorw, RORW>;
def : PatGprImm<riscv_rorw, RORIW, uimm5>;
def : Pat<(riscv_rolw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, (ImmSubFrom32 uimm5:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbpOrZbkb, IsRV64]

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
def : Pat<(or GPR:$rs1, BSETINVMask:$mask),
          (BSETI GPR:$rs1, BSETINVMask:$mask)>;
def : Pat<(xor GPR:$rs1, BSETINVMask:$mask),
          (BINVI GPR:$rs1, BSETINVMask:$mask)>;

def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), (XLenVT 1)),
          (BEXTI GPR:$rs1, uimmlog2xlen:$shamt)>;

def : Pat<(and (not (srl GPR:$rs1, uimmlog2xlen:$shamt)), (XLenVT 1)),
          (XORI (BEXTI GPR:$rs1, uimmlog2xlen:$shamt), (XLenVT 1))>;

def : Pat<(or GPR:$r, BSETINVTwoBitsMask:$i),
          (BSETI (BSETI GPR:$r, (TrailingZerosXForm BSETINVTwoBitsMask:$i)),
                 (BSETINVTwoBitsMaskHigh BSETINVTwoBitsMask:$i))>;
def : Pat<(xor GPR:$r, BSETINVTwoBitsMask:$i),
          (BINVI (BINVI GPR:$r, (TrailingZerosXForm BSETINVTwoBitsMask:$i)),
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
}

let Predicates = [HasStdExtZbbOrZbp] in {
// We treat orc.b as a separate instruction, so match it directly. We also
// lower the Zbb orc.b intrinsic to this.
def : Pat<(riscv_gorc GPR:$rs1, 7), (ORC_B GPR:$rs1)>;
}

let Predicates = [HasStdExtZbpOrZbkb] in {
// We treat brev8 as a separate instruction, so match it directly. We also
// use this for brev8 when lowering bitreverse with Zbkb.
def : Pat<(riscv_grev GPR:$rs1, 7), (BREV8 GPR:$rs1)>;
}

let Predicates = [HasStdExtZbpOrZbkb, IsRV32] in {
// We treat zip and unzip as separate instructions, so match it directly.
def : Pat<(i32 (riscv_shfl GPR:$rs1, 15)), (ZIP_RV32 GPR:$rs1)>;
def : Pat<(i32 (riscv_unshfl GPR:$rs1, 15)), (UNZIP_RV32 GPR:$rs1)>;
}

let Predicates = [HasStdExtZbp] in {
def : PatGprGpr<riscv_grev, GREV>;
def : PatGprGpr<riscv_gorc, GORC>;
def : PatGprImm<riscv_grev, GREVI, uimmlog2xlen>;
def : PatGprImm<riscv_gorc, GORCI, uimmlog2xlen>;

def : PatGprGpr<riscv_shfl, SHFL>;
def : PatGprGpr<riscv_unshfl, UNSHFL>;
def : PatGprImm<riscv_shfl, SHFLI, shfl_uimm>;
def : PatGprImm<riscv_unshfl, UNSHFLI, shfl_uimm>;

def : PatGprGpr<int_riscv_xperm_n, XPERM4>;
def : PatGprGpr<int_riscv_xperm_b, XPERM8>;
def : PatGprGpr<int_riscv_xperm_h, XPERM_H>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : PatGprGpr<riscv_grevw, GREVW>;
def : PatGprGpr<riscv_gorcw, GORCW>;
def : PatGprImm<riscv_grevw, GREVIW, uimm5>;
def : PatGprImm<riscv_gorcw, GORCIW, uimm5>;

def : PatGprGpr<riscv_shflw, SHFLW>;
def : PatGprGpr<riscv_unshflw, UNSHFLW>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in
def : PatGprGpr<int_riscv_xperm_w, XPERM_W>;

let Predicates = [HasStdExtZbp, IsRV32] in {
// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(i32 (riscv_grev GPR:$rs1, 24)), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(i64 (riscv_grev GPR:$rs1, 56)), (REV8_RV64 GPR:$rs1)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt] in {
def : Pat<(or (and (not GPR:$rs2), GPR:$rs3), (and GPR:$rs2, GPR:$rs1)),
          (CMIX GPR:$rs1, GPR:$rs2, GPR:$rs3)>;

def : Pat<(select (XLenVT (setne GPR:$rs2, 0)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$rs2, 0)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (setne GPR:$x, simm12_plus1:$y)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$x, simm12_plus1:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (setne GPR:$x, GPR:$y)), GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setuge GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLTU GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setule GPR:$y, GPR:$x)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLTU GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setge GPR:$x, GPR:$y)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setle GPR:$y, GPR:$x)), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select GPR:$rs2, GPR:$rs1, GPR:$rs3),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbt] in {
def : Pat<(riscv_fsl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsr GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSR GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsr GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt)>;
// We can use FSRI for FSL by immediate if we subtract the immediate from
// XLen and swap the operands.
def : Pat<(riscv_fsl GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(riscv_fslw GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSLW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSRW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs1, GPR:$rs3, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, uimm5:$shamt)>;
// We can use FSRIW for FSLW by immediate if we subtract the immediate from
// 32 and swap the operands.
def : Pat<(riscv_fslw GPR:$rs3, GPR:$rs1, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, (ImmSubFrom32 uimm5:$shamt))>;
} // Predicates = [HasStdExtZbt, IsRV64]

let Predicates = [HasStdExtZbb] in {
def : PatGpr<ctlz, CLZ>;
def : PatGpr<cttz, CTZ>;
def : PatGpr<ctpop, CPOP>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : PatGpr<riscv_clzw, CLZW>;
def : PatGpr<riscv_ctzw, CTZW>;
def : Pat<(i64 (ctpop (i64 (zexti32 (i64 GPR:$rs1))))), (CPOPW GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbb] in {
def : Pat<(sext_inreg GPR:$rs1, i8), (SEXT_B GPR:$rs1)>;
def : Pat<(sext_inreg GPR:$rs1, i16), (SEXT_H GPR:$rs1)>;
}

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

let Predicates = [HasStdExtZbpOrZbkb] in {
def : Pat<(or (and (shl GPR:$rs2, (XLenVT 8)), 0xFFFF),
              (and GPR:$rs1, 0x00FF)),
          (PACKH GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (shl (and GPR:$rs2, 0x00FF), (XLenVT 8)),
              (and GPR:$rs1, 0x00FF)),
          (PACKH GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbpOrZbkb]

let Predicates = [HasStdExtZbpOrZbkb, IsRV32] in
def : Pat<(i32 (or (and GPR:$rs1, 0x0000FFFF), (shl GPR:$rs2, (i32 16)))),
          (PACK GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbpOrZbkb, IsRV64] in {
def : Pat<(i64 (or (and GPR:$rs1, 0x00000000FFFFFFFF), (shl GPR:$rs2, (i64 32)))),
          (PACK GPR:$rs1, GPR:$rs2)>;

def : Pat<(i64 (sext_inreg (or (shl GPR:$rs2, (i64 16)),
                               (and GPR:$rs1, 0x000000000000FFFF)),
                           i32)),
          (PACKW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (or (sext_inreg (shl GPR:$rs2, (i64 16)), i32),
                   (and GPR:$rs1, 0x000000000000FFFF))),
          (PACKW GPR:$rs1, GPR:$rs2)>;
}

let Predicates = [HasStdExtZbp, IsRV32] in
def : Pat<(i32 (or (and GPR:$rs2, 0xFFFF0000), (srl GPR:$rs1, (i32 16)))),
          (PACKU GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(i64 (or (and GPR:$rs2, 0xFFFFFFFF00000000), (srl GPR:$rs1, (i64 32)))),
          (PACKU GPR:$rs1, GPR:$rs2)>;

def : Pat<(i64 (or (and (assertsexti32 GPR:$rs2), 0xFFFFFFFFFFFF0000),
                   (srl (and GPR:$rs1, 0xFFFFFFFF), (i64 16)))),
          (PACKUW GPR:$rs1, GPR:$rs2)>;
}

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(i32 (and GPR:$rs, 0xFFFF)), (ZEXT_H_RV32 GPR:$rs)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in
def : Pat<(i64 (and GPR:$rs, 0xFFFF)), (ZEXT_H_RV64 GPR:$rs)>;

// Pattern to exclude simm12 immediates from matching.
def non_imm12 : PatLeaf<(XLenVT GPR:$a), [{
  auto *C = dyn_cast<ConstantSDNode>(N);
  return !C || !isInt<12>(C->getSExtValue());
}]>;

let Predicates = [HasStdExtZba] in {
def : Pat<(add (shl GPR:$rs1, (XLenVT 1)), non_imm12:$rs2),
          (SH1ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 2)), non_imm12:$rs2),
          (SH2ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 3)), non_imm12:$rs2),
          (SH3ADD GPR:$rs1, GPR:$rs2)>;

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
                (TrailingZerosXForm C3LeftShift:$i))>;
def : Pat<(mul GPR:$r, C5LeftShift:$i),
          (SLLI (SH2ADD GPR:$r, GPR:$r),
                (TrailingZerosXForm C5LeftShift:$i))>;
def : Pat<(mul GPR:$r, C9LeftShift:$i),
          (SLLI (SH3ADD GPR:$r, GPR:$r),
                (TrailingZerosXForm C9LeftShift:$i))>;

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
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbcOrZbkc] in {
def : PatGprGpr<int_riscv_clmul, CLMUL>;
def : PatGprGpr<int_riscv_clmulh, CLMULH>;
} // Predicates = [HasStdExtZbcOrZbkc]

let Predicates = [HasStdExtZbc] in
def : PatGprGpr<int_riscv_clmulr, CLMULR>;

let Predicates = [HasStdExtZbe] in {
def : PatGprGpr<riscv_bcompress, BCOMPRESS>;
def : PatGprGpr<riscv_bdecompress, BDECOMPRESS>;
} // Predicates = [HasStdExtZbe]

let Predicates = [HasStdExtZbe, IsRV64] in {
def : PatGprGpr<riscv_bcompressw, BCOMPRESSW>;
def : PatGprGpr<riscv_bdecompressw, BDECOMPRESSW>;
} // Predicates = [HasStdExtZbe, IsRV64]

let Predicates = [HasStdExtZbr] in {
def : PatGpr<int_riscv_crc32_b, CRC32_B>;
def : PatGpr<int_riscv_crc32_h, CRC32_H>;
def : PatGpr<int_riscv_crc32_w, CRC32_W>;
def : PatGpr<int_riscv_crc32c_b, CRC32C_B>;
def : PatGpr<int_riscv_crc32c_h, CRC32C_H>;
def : PatGpr<int_riscv_crc32c_w, CRC32C_W>;
} // Predicates = [HasStdExtZbr]

let Predicates = [HasStdExtZbr, IsRV64] in {
def : PatGpr<int_riscv_crc32_d, CRC32_D>;
def : PatGpr<int_riscv_crc32c_d, CRC32C_D>;
} // Predicates = [HasStdExtZbr, IsRV64]

let Predicates = [HasStdExtZbf] in
def : PatGprGpr<riscv_bfp, BFP>;

let Predicates = [HasStdExtZbf, IsRV64] in
def : PatGprGpr<riscv_bfpw, BFPW>;

let Predicates = [HasStdExtZbkx] in {
def : PatGprGpr<int_riscv_xperm4, XPERM4>;
def : PatGprGpr<int_riscv_xperm8, XPERM8>;
}
