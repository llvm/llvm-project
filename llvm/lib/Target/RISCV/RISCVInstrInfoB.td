//===-- RISCVInstrInfoB.td - RISC-V 'B' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'B' Bitmanip
// extension, version 0.93.
// This version is still experimental as the 'B' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def riscv_clzw   : SDNode<"RISCVISD::CLZW",   SDTIntUnaryOp>;
def riscv_ctzw   : SDNode<"RISCVISD::CTZW",   SDTIntUnaryOp>;
def riscv_rolw   : SDNode<"RISCVISD::ROLW",   SDTIntShiftOp>;
def riscv_rorw   : SDNode<"RISCVISD::RORW",   SDTIntShiftOp>;
def riscv_fslw   : SDNode<"RISCVISD::FSLW",   SDTIntShiftDOp>;
def riscv_fsrw   : SDNode<"RISCVISD::FSRW",   SDTIntShiftDOp>;
def riscv_fsl    : SDNode<"RISCVISD::FSL",    SDTIntShiftDOp>;
def riscv_fsr    : SDNode<"RISCVISD::FSR",    SDTIntShiftDOp>;
def riscv_grevi  : SDNode<"RISCVISD::GREVI",  SDTIntBinOp>;
def riscv_greviw : SDNode<"RISCVISD::GREVIW", SDTIntBinOp>;
def riscv_gorci  : SDNode<"RISCVISD::GORCI",  SDTIntBinOp>;
def riscv_gorciw : SDNode<"RISCVISD::GORCIW", SDTIntBinOp>;
def riscv_shfli  : SDNode<"RISCVISD::SHFLI",  SDTIntBinOp>;

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
  let Inst{24-20} = funct5;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShift_ri<bits<5> funct5, bits<3> funct3, RISCVOpcode opcode,
                  string opcodestr>
    : RVInstI<funct3, opcode, (outs GPR:$rd),
              (ins GPR:$rs1, uimmlog2xlen:$shamt), opcodestr,
              "$rd, $rs1, $shamt"> {
  bits<6> shamt;

  let Inst{31-27} = funct5;
  // NOTE: the bit op(26)=1 is used to select funnel shifts. All other
  // shifts operations and operations that live in the encoding space
  // of the shifts (single bit operations, grev, gorc) use op(26) = 0
  let Inst{26} = 0;
  let Inst{25-20} = shamt;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShiftW_ri<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode,
                   string opcodestr>
    : RVInstI<funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1, uimm5:$shamt),
              opcodestr, "$rd, $rs1, $shamt"> {
  bits<5> shamt;

  let Inst{31-25} = funct7;
  let Inst{24-20} = shamt;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBShfl_ri<bits<6> funct6, bits<3> funct3, RISCVOpcode opcode,
                 string opcodestr>
    : RVInstI<funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1, shfl_uimm:$shamt),
              opcodestr, "$rd, $rs1, $shamt"> {
  bits<6> shamt;

  let Inst{31-26} = funct6;
  let Inst{25-20} = shamt;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryR<bits<2> funct2, bits<3> funct3_b, RISCVOpcode opcode,
                  string opcodestr, string argstr>
    : RVInstR4<funct2, opcode, (outs GPR:$rd),
               (ins GPR:$rs1, GPR:$rs2, GPR:$rs3), opcodestr, argstr> {
  let Inst{14-12} = funct3_b;
}

// Currently used by FSRI only
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryImm6<bits<3> funct3_b, RISCVOpcode opcode,
                     string opcodestr, string argstr>
    : RVInstR4<0b10, opcode, (outs GPR:$rd),
               (ins GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt),
               opcodestr, argstr> {
  bits<6> shamt;

  // NOTE: the first argument of RVInstR4 is hardcoded to 0b10 like the other
  // funnel shift instructions. The second bit of the argument though is
  // overwritten by the shamt as the encoding of this particular instruction
  // requires. This is to obtain op(26) = 1 as required by funnel shift
  // instructions without the need of a confusing argument in the definition
  // of the instruction.
  let Inst{25-20} = shamt;
  let Inst{14-12} = funct3_b;
}

// Currently used by FSRIW only
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBTernaryImm5<bits<2> funct2, bits<3> funct3_b, RISCVOpcode opcode,
                     string opcodestr, string argstr>
    : RVInstR4<funct2, opcode, (outs GPR:$rd),
               (ins GPR:$rs1, GPR:$rs3, uimm5:$shamt), opcodestr, argstr> {
  bits<5> shamt;

  let Inst{24-20} = shamt;
  let Inst{14-12} = funct3_b;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbp] in {
def ANDN  : ALU_rr<0b0100000, 0b111, "andn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def ORN   : ALU_rr<0b0100000, 0b110, "orn">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def XNOR  : ALU_rr<0b0100000, 0b100, "xnor">,
            Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZba] in {
def SH1ADD : ALU_rr<0b0010000, 0b010, "sh1add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
def SH2ADD : ALU_rr<0b0010000, 0b100, "sh2add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
def SH3ADD : ALU_rr<0b0010000, 0b110, "sh3add">,
             Sched<[WriteSHXADD, ReadSHXADD, ReadSHXADD]>;
} // Predicates = [HasStdExtZba]

let Predicates = [HasStdExtZbbOrZbp] in {
def ROL   : ALU_rr<0b0110000, 0b001, "rol">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;
def ROR   : ALU_rr<0b0110000, 0b101, "ror">,
            Sched<[WriteRotateReg, ReadRotateReg, ReadRotateReg]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbs] in {
def BCLR : ALU_rr<0b0100100, 0b001, "bclr">, Sched<[]>;
def BSET : ALU_rr<0b0010100, 0b001, "bset">, Sched<[]>;
def BINV : ALU_rr<0b0110100, 0b001, "binv">, Sched<[]>;
def BEXT : ALU_rr<0b0100100, 0b101, "bext">, Sched<[]>;
} // Predicates = [HasStdExtZbs]

let Predicates = [HasStdExtZbp] in {
def GORC : ALU_rr<0b0010100, 0b101, "gorc">, Sched<[]>;
def GREV : ALU_rr<0b0110100, 0b101, "grev">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp] in {
def XPERMN : ALU_rr<0b0010100, 0b010, "xperm.n">, Sched<[]>;
def XPERMB : ALU_rr<0b0010100, 0b100, "xperm.b">, Sched<[]>;
def XPERMH : ALU_rr<0b0010100, 0b110, "xperm.h">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbbOrZbp] in
def RORI  : RVBShift_ri<0b01100, 0b101, OPC_OP_IMM, "rori">,
            Sched<[WriteRotateImm, ReadRotateImm]>;

let Predicates = [HasStdExtZbs] in {
def BCLRI : RVBShift_ri<0b01001, 0b001, OPC_OP_IMM, "bclri">, Sched<[]>;
def BSETI : RVBShift_ri<0b00101, 0b001, OPC_OP_IMM, "bseti">, Sched<[]>;
def BINVI : RVBShift_ri<0b01101, 0b001, OPC_OP_IMM, "binvi">, Sched<[]>;
def BEXTI : RVBShift_ri<0b01001, 0b101, OPC_OP_IMM, "bexti">, Sched<[]>;
} // Predicates = [HasStdExtZbs]

let Predicates = [HasStdExtZbp] in {
def GREVI : RVBShift_ri<0b01101, 0b101, OPC_OP_IMM, "grevi">, Sched<[]>;
def GORCI : RVBShift_ri<0b00101, 0b101, OPC_OP_IMM, "gorci">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

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

let Predicates = [HasStdExtZbb] in {
def CLZ  : RVBUnary<0b0110000, 0b00000, 0b001, RISCVOpcode<0b0010011>, "clz">,
           Sched<[WriteCLZ, ReadCLZ]>;
def CTZ  : RVBUnary<0b0110000, 0b00001, 0b001, RISCVOpcode<0b0010011>, "ctz">,
           Sched<[WriteCTZ, ReadCTZ]>;
def CPOP : RVBUnary<0b0110000, 0b00010, 0b001, RISCVOpcode<0b0010011>, "cpop">,
           Sched<[WriteCPOP, ReadCPOP]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbm, IsRV64] in
def BMATFLIP : RVBUnary<0b0110000, 0b00011, 0b001, RISCVOpcode<0b0010011>,
                        "bmatflip">, Sched<[]>;

let Predicates = [HasStdExtZbb] in {
def SEXTB : RVBUnary<0b0110000, 0b00100, 0b001, RISCVOpcode<0b0010011>,
                     "sext.b">, Sched<[WriteIALU, ReadIALU]>;
def SEXTH : RVBUnary<0b0110000, 0b00101, 0b001, RISCVOpcode<0b0010011>,
                     "sext.h">, Sched<[WriteIALU, ReadIALU]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbr] in {
def CRC32B : RVBUnary<0b0110000, 0b10000, 0b001, RISCVOpcode<0b0010011>,
                      "crc32.b">, Sched<[]>;
def CRC32H : RVBUnary<0b0110000, 0b10001, 0b001, RISCVOpcode<0b0010011>,
                      "crc32.h">, Sched<[]>;
def CRC32W : RVBUnary<0b0110000, 0b10010, 0b001, RISCVOpcode<0b0010011>,
                      "crc32.w">, Sched<[]>;
} // Predicates = [HasStdExtZbr]

let Predicates = [HasStdExtZbr, IsRV64] in
def CRC32D  : RVBUnary<0b0110000, 0b10011, 0b001, RISCVOpcode<0b0010011>,
                       "crc32.d">, Sched<[]>;

let Predicates = [HasStdExtZbr] in {
def CRC32CB : RVBUnary<0b0110000, 0b11000, 0b001, RISCVOpcode<0b0010011>,
                       "crc32c.b">, Sched<[]>;
def CRC32CH : RVBUnary<0b0110000, 0b11001, 0b001, RISCVOpcode<0b0010011>,
                       "crc32c.h">, Sched<[]>;
def CRC32CW : RVBUnary<0b0110000, 0b11010, 0b001, RISCVOpcode<0b0010011>,
                       "crc32c.w">, Sched<[]>;
} // Predicates = [HasStdExtZbr]

let Predicates = [HasStdExtZbr, IsRV64] in
def CRC32CD : RVBUnary<0b0110000, 0b11011, 0b001, RISCVOpcode<0b0010011>,
                       "crc32c.d">, Sched<[]>;

let Predicates = [HasStdExtZbc] in {
def CLMUL  : ALU_rr<0b0000101, 0b001, "clmul">, Sched<[]>;
def CLMULR : ALU_rr<0b0000101, 0b010, "clmulr">, Sched<[]>;
def CLMULH : ALU_rr<0b0000101, 0b011, "clmulh">, Sched<[]>;
} // Predicates = [HasStdExtZbc]

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

let Predicates = [HasStdExtZbp] in {
def SHFL   : ALU_rr<0b0000100, 0b001, "shfl">, Sched<[]>;
def UNSHFL : ALU_rr<0b0000100, 0b101, "unshfl">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbe] in {
// NOTE: These mnemonics are from the 0.94 spec. There is a name conflict with
// bext in the 0.93 spec.
def BDECOMPRESS : ALU_rr<0b0100100, 0b110, "bdecompress">, Sched<[]>;
def BCOMPRESS   : ALU_rr<0b0000100, 0b110, "bcompress">, Sched<[]>;
} // Predicates = [HasStdExtZbe]

let Predicates = [HasStdExtZbp] in {
def PACK  : ALU_rr<0b0000100, 0b100, "pack">, Sched<[]>;
def PACKU : ALU_rr<0b0100100, 0b100, "packu">, Sched<[]>;
def PACKH : ALU_rr<0b0000100, 0b111, "packh">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbm, IsRV64] in {
def BMATOR   : ALU_rr<0b0000100, 0b011, "bmator">, Sched<[]>;
def BMATXOR  : ALU_rr<0b0100100, 0b011, "bmatxor">, Sched<[]>;
} // Predicates = [HasStdExtZbm, IsRV64]

let Predicates = [HasStdExtZbf] in
def BFP : ALU_rr<0b0100100, 0b111, "bfp">, Sched<[]>;

let Predicates = [HasStdExtZbp] in {
def SHFLI   : RVBShfl_ri<0b000010, 0b001, OPC_OP_IMM, "shfli">, Sched<[]>;
def UNSHFLI : RVBShfl_ri<0b000010, 0b101, OPC_OP_IMM, "unshfli">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZba, IsRV64] in {
def SLLIUW : RVBShift_ri<0b00001, 0b001, OPC_OP_IMM_32, "slli.uw">,
             Sched<[WriteShiftImm32, ReadShiftImm32]>;
def ADDUW : ALUW_rr<0b0000100, 0b000, "add.uw">,
            Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>;
def SH1ADDUW : ALUW_rr<0b0010000, 0b010, "sh1add.uw">,
               Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
def SH2ADDUW : ALUW_rr<0b0010000, 0b100, "sh2add.uw">,
               Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
def SH3ADDUW : ALUW_rr<0b0010000, 0b110, "sh3add.uw">,
               Sched<[WriteSHXADD32, ReadSHXADD32, ReadSHXADD32]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def ROLW  : ALUW_rr<0b0110000, 0b001, "rolw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;
def RORW  : ALUW_rr<0b0110000, 0b101, "rorw">,
            Sched<[WriteRotateReg32, ReadRotateReg32, ReadRotateReg32]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbs, IsRV64] in {
// NOTE: These instructions have been removed from the 0.94 spec. As a result
// we have no isel patterns for them.
def BCLRW : ALUW_rr<0b0100100, 0b001, "bclrw">, Sched<[]>;
def BSETW : ALUW_rr<0b0010100, 0b001, "bsetw">, Sched<[]>;
def BINVW : ALUW_rr<0b0110100, 0b001, "binvw">, Sched<[]>;
def BEXTW : ALUW_rr<0b0100100, 0b101, "bextw">, Sched<[]>;
} // Predicates = [HasStdExtZbs, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def GORCW  : ALUW_rr<0b0010100, 0b101, "gorcw">, Sched<[]>;
def GREVW  : ALUW_rr<0b0110100, 0b101, "grevw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def XPERMW : ALU_rr<0b0010100, 0b000, "xperm.w">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in
def RORIW : RVBShiftW_ri<0b0110000, 0b101, OPC_OP_IMM_32, "roriw">,
            Sched<[WriteRotateImm32, ReadRotateImm32]>;

let Predicates = [HasStdExtZbs, IsRV64] in {
// NOTE: These instructions have been removed from the 0.94 spec. As a result
// we have no isel patterns for them.
def BCLRIW : RVBShiftW_ri<0b0100100, 0b001, OPC_OP_IMM_32, "bclriw">,
              Sched<[]>;
def BSETIW : RVBShiftW_ri<0b0010100, 0b001, OPC_OP_IMM_32, "bsetiw">,
              Sched<[]>;
def BINVIW : RVBShiftW_ri<0b0110100, 0b001, OPC_OP_IMM_32, "binviw">,
              Sched<[]>;
} // Predicates = [HasStdExtZbs, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def GORCIW : RVBShiftW_ri<0b0010100, 0b101, OPC_OP_IMM_32, "gorciw">, Sched<[]>;
def GREVIW : RVBShiftW_ri<0b0110100, 0b101, OPC_OP_IMM_32, "greviw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt, IsRV64] in {
def FSLW  : RVBTernaryR<0b10, 0b001, OPC_OP_32,
                        "fslw", "$rd, $rs1, $rs3, $rs2">, Sched<[]>;
def FSRW  : RVBTernaryR<0b10, 0b101, OPC_OP_32, "fsrw",
                        "$rd, $rs1, $rs3, $rs2">, Sched<[]>;
def FSRIW : RVBTernaryImm5<0b10, 0b101, OPC_OP_IMM_32,
                           "fsriw", "$rd, $rs1, $rs3, $shamt">, Sched<[]>;
} // Predicates = [HasStdExtZbt, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def CLZW   : RVBUnary<0b0110000, 0b00000, 0b001, RISCVOpcode<0b0011011>,
                      "clzw">, Sched<[WriteCLZ32, ReadCLZ32]>;
def CTZW   : RVBUnary<0b0110000, 0b00001, 0b001, RISCVOpcode<0b0011011>,
                      "ctzw">, Sched<[WriteCTZ32, ReadCTZ32]>;
def CPOPW  : RVBUnary<0b0110000, 0b00010, 0b001, RISCVOpcode<0b0011011>,
                      "cpopw">, Sched<[WriteCPOP32, ReadCPOP32]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def SHFLW   : ALUW_rr<0b0000100, 0b001, "shflw">, Sched<[]>;
def UNSHFLW : ALUW_rr<0b0000100, 0b101, "unshflw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbe, IsRV64] in {
// NOTE: These mnemonics are from the 0.94 spec. There is a name conflict with
// bextw in the 0.93 spec.
def BDECOMPRESSW : ALUW_rr<0b0100100, 0b110, "bdecompressw">, Sched<[]>;
def BCOMPRESSW   : ALUW_rr<0b0000100, 0b110, "bcompressw">, Sched<[]>;
} // Predicates = [HasStdExtZbe, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def PACKW  : ALUW_rr<0b0000100, 0b100, "packw">, Sched<[]>;
def PACKUW : ALUW_rr<0b0100100, 0b100, "packuw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbf, IsRV64] in
def BFPW : ALUW_rr<0b0100100, 0b111, "bfpw">, Sched<[]>;

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def ZEXTH_RV32 : RVInstR<0b0000100, 0b100, OPC_OP, (outs GPR:$rd),
                         (ins GPR:$rs1), "zext.h", "$rd, $rs1">,
                 Sched<[WriteIALU, ReadIALU]> {
  let rs2 = 0b00000;
}
} // Predicates = [HasStdExtZbbOrZbp, IsRV32]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def ZEXTH_RV64 : RVInstR<0b0000100, 0b100, OPC_OP_32, (outs GPR:$rd),
                         (ins GPR:$rs1), "zext.h", "$rd, $rs1">,
                 Sched<[WriteIALU, ReadIALU]> {
  let rs2 = 0b00000;
}
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

// We treat rev8 and orc.b as standalone instructions even though they use a
// portion of the encodings for grevi and gorci. This allows us to support only
// those encodings when only Zbb is enabled. We do this even when grevi and
// gorci are available with Zbp. Trying to use 'HasStdExtZbb, NotHasStdExtZbp'
// causes diagnostics to suggest that Zbp rather than Zbb is required for rev8
// or gorci. Since Zbb is closer to being finalized than Zbp this will be
// misleading to users.
let Predicates = [HasStdExtZbbOrZbp, IsRV32] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def REV8_RV32 : RVInstI<0b101, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1),
                        "rev8", "$rd, $rs1">, Sched<[WriteREV8, ReadREV8]> {
  let imm12 = { 0b01101, 0b0011000 };
}
} // Predicates = [HasStdExtZbbOrZbp, IsRV32]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def REV8_RV64 : RVInstI<0b101, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1),
                        "rev8", "$rd, $rs1">, Sched<[WriteREV8, ReadREV8]> {
  let imm12 = { 0b01101, 0b0111000 };
}
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbbOrZbp] in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def ORCB : RVInstI<0b101, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1),
                   "orc.b", "$rd, $rs1">, Sched<[WriteORCB, ReadORCB]> {
  let imm12 = { 0b00101, 0b0000111 };
}
} // Predicates = [HasStdExtZbbOrZbp]

//===----------------------------------------------------------------------===//
// Future compressed instructions
//===----------------------------------------------------------------------===//

// The presence of these instructions in the B extension is purely experimental
// and they should be moved to the C extension as soon as they are ratified.

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBInstC<bits<2> funct2, string opcodestr>
    : RVInst16<(outs GPRC:$rs_wb), (ins GPRC:$rs), opcodestr, "$rs", [],
               InstFormatCR> {
  bits<3> rs;
  let Constraints = "$rs = $rs_wb";

  let Inst{15-12} = 0b0110;
  let Inst{11-10} = funct2;
  let Inst{9-7} = rs;
  let Inst{6-0} = 0b0000001;
}

// The namespace RVBC exists to avoid encoding conflicts with the compressed
// instructions c.addi16sp and c.lui already implemented in the C extension.

let DecoderNamespace = "RVBC", Predicates = [HasStdExtZbproposedc, HasStdExtC] in {
def C_NOT : RVBInstC<0b00, "c.not">, Sched<[]>;
def C_NEG : RVBInstC<0b01, "c.neg">, Sched<[]>;
} // DecoderNamespace = "RVBC", Predicates = [HasStdExtZbproposedc, HasStdExtC]

let DecoderNamespace = "RVBC", Predicates = [HasStdExtZbproposedc, HasStdExtZba, HasStdExtC, IsRV64] in
def C_ZEXTW : RVBInstC<0b10, "c.zext.w">, Sched<[]>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZba, IsRV64] in {
// NOTE: The 0.93 spec shows zext.w as an alias of pack/packw. It has been
// changed to add.uw in a draft after 0.94.
def : InstAlias<"zext.w $rd, $rs", (ADDUW GPR:$rd, GPR:$rs, X0)>;
}

let Predicates = [HasStdExtZbp] in {
def : InstAlias<"rev.p $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00001)>;
def : InstAlias<"rev2.n $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00010)>;
def : InstAlias<"rev.n $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00011)>;
def : InstAlias<"rev4.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00100)>;
def : InstAlias<"rev2.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00110)>;
def : InstAlias<"rev.b $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00111)>;
def : InstAlias<"rev8.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01000)>;
def : InstAlias<"rev4.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01100)>;
def : InstAlias<"rev2.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01110)>;
def : InstAlias<"rev.h $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b01111)>;

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
def : InstAlias<"zip $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b1111)>;
def : InstAlias<"unzip $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b1111)>;

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

//===----------------------------------------------------------------------===//
// Compressed Instruction patterns
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZbproposedc, HasStdExtC] in {
def : CompressPat<(XORI GPRC:$rs1, GPRC:$rs1, -1),
                  (C_NOT GPRC:$rs1)>;
def : CompressPat<(SUB GPRC:$rs1, X0, GPRC:$rs1),
                  (C_NEG GPRC:$rs1)>;
} // Predicates = [HasStdExtZbproposedc, HasStdExtC]

let Predicates = [HasStdExtZbproposedc, HasStdExtZba, HasStdExtC, IsRV64] in {
def : CompressPat<(ADDUW GPRC:$rs1, GPRC:$rs1, X0),
                  (C_ZEXTW GPRC:$rs1)>;
} // Predicates = [HasStdExtZbproposedc, HasStdExtC, IsRV64]

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(and GPR:$rs1, (not GPR:$rs2)), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(or  GPR:$rs1, (not GPR:$rs2)), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, (not GPR:$rs2)), (XNOR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotl GPR:$rs1, GPR:$rs2), (ROL GPR:$rs1, GPR:$rs2)>;
def : Pat<(rotr GPR:$rs1, GPR:$rs2), (ROR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

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
}

// There's no encoding for roli in the the 'B' extension as it can be
// implemented with rori by negating the immediate.
let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotr GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, (ImmSubFromXLen uimmlog2xlen:$shamt))>;

// We treat orc.b as a separate instruction, so match it directly. We also
// lower the Zbb orc.b intrinsic to this.
def : Pat<(riscv_gorci GPR:$rs1, 7), (ORCB GPR:$rs1)>;
}

let Predicates = [HasStdExtZbp] in {
def : Pat<(riscv_shfli GPR:$rs1, timm:$shamt), (SHFLI GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_grevi GPR:$rs1, timm:$shamt), (GREVI GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_gorci GPR:$rs1, timm:$shamt), (GORCI GPR:$rs1, timm:$shamt)>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(i32 (rotr (riscv_grevi GPR:$rs1, 24), (i32 16))), (GREVI GPR:$rs1, 8)>;
def : Pat<(i32 (rotl (riscv_grevi GPR:$rs1, 24), (i32 16))), (GREVI GPR:$rs1, 8)>;

// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(i32 (riscv_grevi GPR:$rs1, 24)), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
// We treat rev8 as a separate instruction, so match it directly.
def : Pat<(i64 (riscv_grevi GPR:$rs1, 56)), (REV8_RV64 GPR:$rs1)>;
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

// fshl and fshr concatenate their operands in the same order. fsr and fsl
// instruction use different orders. fshl will return its first operand for
// shift of zero, fshr will return its second operand. fsl and fsr both return
// $rs1 so the patterns need to have different operand orders.
let Predicates = [HasStdExtZbt] in {
def : Pat<(riscv_fsl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsr GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSR GPR:$rs1, GPR:$rs2, GPR:$rs3)>;

def : Pat<(fshr GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt)>;
// We can use FSRI for fshl by immediate if we subtract the immediate from
// XLen and swap the operands.
def : Pat<(fshl GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, (ImmSubFromXLen uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbb] in {
def : Pat<(ctlz GPR:$rs1), (CLZ GPR:$rs1)>;
def : Pat<(cttz GPR:$rs1), (CTZ GPR:$rs1)>;
def : Pat<(ctpop GPR:$rs1), (CPOP GPR:$rs1)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb] in {
def : Pat<(sext_inreg GPR:$rs1, i8), (SEXTB GPR:$rs1)>;
def : Pat<(sext_inreg GPR:$rs1, i16), (SEXTH GPR:$rs1)>;
}

let Predicates = [HasStdExtZbb] in {
def : Pat<(smin GPR:$rs1, GPR:$rs2), (MIN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(smax GPR:$rs1, GPR:$rs2), (MAX  GPR:$rs1, GPR:$rs2)>;
def : Pat<(umin GPR:$rs1, GPR:$rs2), (MINU GPR:$rs1, GPR:$rs2)>;
def : Pat<(umax GPR:$rs1, GPR:$rs2), (MAXU GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbb, IsRV32] in {
def : Pat<(i32 (bswap GPR:$rs1)), (REV8_RV32 GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV32]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(i64 (bswap GPR:$rs1)), (REV8_RV64 GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(i32 (or (and GPR:$rs1, 0x0000FFFF), (shl GPR:$rs2, (i32 16)))),
          (PACK GPR:$rs1, GPR:$rs2)>;
def : Pat<(i32 (or (and GPR:$rs2, 0xFFFF0000), (srl GPR:$rs1, (i32 16)))),
          (PACKU GPR:$rs1, GPR:$rs2)>;

}
let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(i64 (or (and GPR:$rs1, 0x00000000FFFFFFFF), (shl GPR:$rs2, (i64 32)))),
          (PACK GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (or (and GPR:$rs2, 0xFFFFFFFF00000000), (srl GPR:$rs1, (i64 32)))),
          (PACKU GPR:$rs1, GPR:$rs2)>;
}
let Predicates = [HasStdExtZbp] in
def : Pat<(or (and (shl GPR:$rs2, (XLenVT 8)), 0xFF00),
              (and GPR:$rs1, 0x00FF)),
          (PACKH GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(i32 (and GPR:$rs, 0xFFFF)), (ZEXTH_RV32 GPR:$rs)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(i64 (and GPR:$rs, 0xFFFF)), (ZEXTH_RV64 GPR:$rs)>;
}

let Predicates = [HasStdExtZba] in {
def : Pat<(add (shl GPR:$rs1, (XLenVT 1)), GPR:$rs2),
          (SH1ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 2)), GPR:$rs2),
          (SH2ADD GPR:$rs1, GPR:$rs2)>;
def : Pat<(add (shl GPR:$rs1, (XLenVT 3)), GPR:$rs2),
          (SH3ADD GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZba]

let Predicates = [HasStdExtZba, IsRV64] in {
def : Pat<(i64 (SLLIUWPat GPR:$rs1, uimm5:$shamt)),
          (SLLIUW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(i64 (shl (and GPR:$rs1, 0xFFFFFFFF), uimm5:$shamt)),
          (SLLIUW GPR:$rs1, uimm5:$shamt)>;
def : Pat<(i64 (add (and GPR:$rs1, 0xFFFFFFFF), GPR:$rs2)),
          (ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (and GPR:$rs, 0xFFFFFFFF)), (ADDUW GPR:$rs, X0)>;

def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 1)), GPR:$rs2)),
          (SH1ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 2)), GPR:$rs2)),
          (SH2ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (shl (and GPR:$rs1, 0xFFFFFFFF), (i64 3)), GPR:$rs2)),
          (SH3ADDUW GPR:$rs1, GPR:$rs2)>;

def : Pat<(i64 (add (SLLIUWPat GPR:$rs1, (i64 1)), GPR:$rs2)),
          (SH1ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (SLLIUWPat GPR:$rs1, (i64 2)), GPR:$rs2)),
          (SH2ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (add (SLLIUWPat GPR:$rs1, (i64 3)), GPR:$rs2)),
          (SH3ADDUW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(i64 (riscv_rolw GPR:$rs1, GPR:$rs2)),
          (ROLW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (riscv_rorw GPR:$rs1, GPR:$rs2)),
          (RORW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (riscv_rorw GPR:$rs1, uimm5:$rs2)),
          (RORIW GPR:$rs1, uimm5:$rs2)>;
def : Pat<(i64 (riscv_rolw GPR:$rs1, uimm5:$rs2)),
          (RORIW GPR:$rs1, (ImmSubFrom32 uimm5:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(i64 (riscv_rorw (riscv_greviw GPR:$rs1, 24), (i64 16))), (GREVIW GPR:$rs1, 8)>;
def : Pat<(i64 (riscv_rolw (riscv_greviw GPR:$rs1, 24), (i64 16))), (GREVIW GPR:$rs1, 8)>;
def : Pat<(i64 (riscv_greviw GPR:$rs1, timm:$shamt)), (GREVIW GPR:$rs1, timm:$shamt)>;
def : Pat<(i64 (riscv_gorciw GPR:$rs1, timm:$shamt)), (GORCIW GPR:$rs1, timm:$shamt)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(i64 (riscv_fslw GPR:$rs1, GPR:$rs3, GPR:$rs2)),
          (FSLW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(i64 (riscv_fsrw GPR:$rs3, GPR:$rs1, GPR:$rs2)),
          (FSRW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(i64 (riscv_fsrw GPR:$rs3, GPR:$rs1, uimm5:$shamt)),
          (FSRIW GPR:$rs1, GPR:$rs3, uimm5:$shamt)>;
def : Pat<(i64 (riscv_fslw GPR:$rs3, GPR:$rs1, uimm5:$shamt)),
          (FSRIW GPR:$rs1, GPR:$rs3, (ImmSubFrom32 uimm5:$shamt))>;
} // Predicates = [HasStdExtZbt, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(i64 (riscv_clzw GPR:$rs1)), (CLZW GPR:$rs1)>;
def : Pat<(i64 (riscv_ctzw GPR:$rs1)), (CTZW GPR:$rs1)>;
def : Pat<(i64 (ctpop (and GPR:$rs1, 0xFFFFFFFF))), (CPOPW GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(i64 (sext_inreg (or (shl GPR:$rs2, (i64 16)),
                               (and GPR:$rs1, 0x000000000000FFFF)),
                           i32)),
          (PACKW GPR:$rs1, GPR:$rs2)>;
def : Pat<(i64 (or (and (assertsexti32 GPR:$rs2), 0xFFFFFFFFFFFF0000),
                   (srl (and GPR:$rs1, 0xFFFFFFFF), (i64 16)))),
          (PACKUW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbc] in {
def : PatGprGpr<int_riscv_clmul, CLMUL>;
def : PatGprGpr<int_riscv_clmulh, CLMULH>;
def : PatGprGpr<int_riscv_clmulr, CLMULR>;
} // Predicates = [HasStdExtZbc]

let Predicates = [HasStdExtZbr] in {
def : PatGpr<int_riscv_crc32_b, CRC32B>;
def : PatGpr<int_riscv_crc32_h, CRC32H>;
def : PatGpr<int_riscv_crc32_w, CRC32W>;
def : PatGpr<int_riscv_crc32c_b, CRC32CB>;
def : PatGpr<int_riscv_crc32c_h, CRC32CH>;
def : PatGpr<int_riscv_crc32c_w, CRC32CW>;
} // Predicates = [HasStdExtZbr]

let Predicates = [HasStdExtZbr, IsRV64] in {
def : PatGpr<int_riscv_crc32_d, CRC32D>;
def : PatGpr<int_riscv_crc32c_d, CRC32CD>;
} // Predicates = [HasStdExtZbr, IsRV64]
