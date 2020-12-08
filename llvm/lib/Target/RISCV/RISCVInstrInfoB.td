//===-- RISCVInstrInfoB.td - RISC-V 'B' instructions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'B' Bitmanip
// extension, version 0.92.
// This version is still experimental as the 'B' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def riscv_rolw : SDNode<"RISCVISD::ROLW", SDTIntShiftOp>;
def riscv_rorw : SDNode<"RISCVISD::RORW", SDTIntShiftOp>;
def riscv_fslw : SDNode<"RISCVISD::FSLW", SDTIntShiftDOp>;
def riscv_fsrw : SDNode<"RISCVISD::FSRW", SDTIntShiftDOp>;

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


// Convert rotl immediate to a rotr immediate for XLen instructions.
def ImmROTL2R : SDNodeXForm<imm, [{
  uint64_t XLen = Subtarget->getXLen();
  return CurDAG->getTargetConstant(XLen - N->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;

// Convert rotl immediate to a rotr immediate for W instructions.
def ImmROTL2RW : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(32 - N->getZExtValue(), SDLoc(N),
                                   N->getValueType(0));
}]>;

// Checks if this mask has a single 0 bit and cannot be used with ANDI.
def SBCLRMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(~Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}]>;

// Checks if this mask has a single 1 bit and cannot be used with ORI/XORI.
def SBSETINVMask : ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return !isInt<12>(Imm) && isPowerOf2_64(Imm);
  return !isInt<12>(Imm) && isPowerOf2_32(Imm);
}]>;

def SBCLRXForm : SDNodeXForm<imm, [{
  // Find the lowest 0.
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingOnes(),
                                   SDLoc(N), N->getValueType(0));
}]>;

def SBSETINVXForm : SDNodeXForm<imm, [{
  // Find the lowest 1.
  return CurDAG->getTargetConstant(N->getAPIntValue().countTrailingZeros(),
                                   SDLoc(N), N->getValueType(0));
}]>;

// Similar to above, but makes sure the immediate has 33 sign bits. When used
// with an AND/OR/XOR where the other operand has at least 33 sign bits, the
// result will have 33 sign bits. This can match SBCLRIW/SBSETIW/SBINVIW.
def SBCLRWMask : ImmLeaf<i64, [{
  // After checking the sign bits, truncate to 32 bits for power of 2 check.
  return isInt<32>(Imm) && !isInt<12>(Imm) && isPowerOf2_32(~Imm);
}]>;

def SBSETINVWMask : ImmLeaf<i64, [{
  return isInt<32>(Imm) && !isInt<12>(Imm) && isPowerOf2_32(Imm);
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
  let Inst{24-20} = funct5;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVBALUW_ri<bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP_IMM_32, (outs GPR:$rd),
              (ins GPR:$rs1, simm12:$imm12), opcodestr, "$rd, $rs1, $imm12">;

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
def ANDN  : ALU_rr<0b0100000, 0b111, "andn">, Sched<[]>;
def ORN   : ALU_rr<0b0100000, 0b110, "orn">, Sched<[]>;
def XNOR  : ALU_rr<0b0100000, 0b100, "xnor">, Sched<[]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbb] in {
def SLO  : ALU_rr<0b0010000, 0b001, "slo">, Sched<[]>;
def SRO  : ALU_rr<0b0010000, 0b101, "sro">, Sched<[]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbbOrZbp] in {
def ROL   : ALU_rr<0b0110000, 0b001, "rol">, Sched<[]>;
def ROR   : ALU_rr<0b0110000, 0b101, "ror">, Sched<[]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbs] in {
def SBCLR : ALU_rr<0b0100100, 0b001, "sbclr">, Sched<[]>;
def SBSET : ALU_rr<0b0010100, 0b001, "sbset">, Sched<[]>;
def SBINV : ALU_rr<0b0110100, 0b001, "sbinv">, Sched<[]>;
def SBEXT : ALU_rr<0b0100100, 0b101, "sbext">, Sched<[]>;
} // Predicates = [HasStdExtZbs]

let Predicates = [HasStdExtZbp] in {
def GORC : ALU_rr<0b0010100, 0b101, "gorc">, Sched<[]>;
def GREV : ALU_rr<0b0110100, 0b101, "grev">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbb] in {
def SLOI : RVBShift_ri<0b00100, 0b001, OPC_OP_IMM, "sloi">, Sched<[]>;
def SROI : RVBShift_ri<0b00100, 0b101, OPC_OP_IMM, "sroi">, Sched<[]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbbOrZbp] in
def RORI  : RVBShift_ri<0b01100, 0b101, OPC_OP_IMM, "rori">, Sched<[]>;

let Predicates = [HasStdExtZbs] in {
def SBCLRI : RVBShift_ri<0b01001, 0b001, OPC_OP_IMM, "sbclri">, Sched<[]>;
def SBSETI : RVBShift_ri<0b00101, 0b001, OPC_OP_IMM, "sbseti">, Sched<[]>;
def SBINVI : RVBShift_ri<0b01101, 0b001, OPC_OP_IMM, "sbinvi">, Sched<[]>;
def SBEXTI : RVBShift_ri<0b01001, 0b101, OPC_OP_IMM, "sbexti">, Sched<[]>;
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
           Sched<[]>;
def CTZ  : RVBUnary<0b0110000, 0b00001, 0b001, RISCVOpcode<0b0010011>, "ctz">,
           Sched<[]>;
def PCNT : RVBUnary<0b0110000, 0b00010, 0b001, RISCVOpcode<0b0010011>, "pcnt">,
           Sched<[]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbm, IsRV64] in
def BMATFLIP : RVBUnary<0b0110000, 0b00011, 0b001, RISCVOpcode<0b0010011>,
                        "bmatflip">, Sched<[]>;

let Predicates = [HasStdExtZbb] in {
def SEXTB : RVBUnary<0b0110000, 0b00100, 0b001, RISCVOpcode<0b0010011>,
                     "sext.b">, Sched<[]>;
def SEXTH : RVBUnary<0b0110000, 0b00101, 0b001, RISCVOpcode<0b0010011>,
                     "sext.h">, Sched<[]>;
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
def MIN  : ALU_rr<0b0000101, 0b100, "min">, Sched<[]>;
def MAX  : ALU_rr<0b0000101, 0b101, "max">, Sched<[]>;
def MINU : ALU_rr<0b0000101, 0b110, "minu">, Sched<[]>;
def MAXU : ALU_rr<0b0000101, 0b111, "maxu">, Sched<[]>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbp] in {
def SHFL   : ALU_rr<0b0000100, 0b001, "shfl">, Sched<[]>;
def UNSHFL : ALU_rr<0b0000100, 0b101, "unshfl">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbe] in {
def BDEP : ALU_rr<0b0100100, 0b110, "bdep">, Sched<[]>;
def BEXT : ALU_rr<0b0000100, 0b110, "bext">, Sched<[]>;
} // Predicates = [HasStdExtZbe]

let Predicates = [HasStdExtZbbOrZbp] in {
def PACK  : ALU_rr<0b0000100, 0b100, "pack">, Sched<[]>;
def PACKU : ALU_rr<0b0100100, 0b100, "packu">, Sched<[]>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbm, IsRV64] in {
def BMATOR   : ALU_rr<0b0000100, 0b011, "bmator">, Sched<[]>;
def BMATXOR  : ALU_rr<0b0100100, 0b011, "bmatxor">, Sched<[]>;
} // Predicates = [HasStdExtZbm, IsRV64]

let Predicates = [HasStdExtZbbOrZbp] in
def PACKH : ALU_rr<0b0000100, 0b111, "packh">, Sched<[]>;

let Predicates = [HasStdExtZbf] in
def BFP : ALU_rr<0b0100100, 0b111, "bfp">, Sched<[]>;

let Predicates = [HasStdExtZbp] in {
def SHFLI   : RVBShfl_ri<0b000010, 0b001, OPC_OP_IMM, "shfli">, Sched<[]>;
def UNSHFLI : RVBShfl_ri<0b000010, 0b101, OPC_OP_IMM, "unshfli">, Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbb, IsRV64] in {
def ADDIWU : RVBALUW_ri<0b100, "addiwu">, Sched<[]>;
def SLLIUW : RVBShift_ri<0b00001, 0b001, OPC_OP_IMM_32, "slliu.w">, Sched<[]>;
def ADDWU : ALUW_rr<0b0000101, 0b000, "addwu">, Sched<[]>;
def SUBWU : ALUW_rr<0b0100101, 0b000, "subwu">, Sched<[]>;
def ADDUW : ALUW_rr<0b0000100, 0b000, "addu.w">, Sched<[]>;
def SUBUW : ALUW_rr<0b0100100, 0b000, "subu.w">, Sched<[]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def SLOW   : ALUW_rr<0b0010000, 0b001, "slow">, Sched<[]>;
def SROW   : ALUW_rr<0b0010000, 0b101, "srow">, Sched<[]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def ROLW  : ALUW_rr<0b0110000, 0b001, "rolw">, Sched<[]>;
def RORW  : ALUW_rr<0b0110000, 0b101, "rorw">, Sched<[]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbs, IsRV64] in {
def SBCLRW : ALUW_rr<0b0100100, 0b001, "sbclrw">, Sched<[]>;
def SBSETW : ALUW_rr<0b0010100, 0b001, "sbsetw">, Sched<[]>;
def SBINVW : ALUW_rr<0b0110100, 0b001, "sbinvw">, Sched<[]>;
def SBEXTW : ALUW_rr<0b0100100, 0b101, "sbextw">, Sched<[]>;
} // Predicates = [HasStdExtZbs, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def GORCW  : ALUW_rr<0b0010100, 0b101, "gorcw">, Sched<[]>;
def GREVW  : ALUW_rr<0b0110100, 0b101, "grevw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def SLOIW  : RVBShiftW_ri<0b0010000, 0b001, OPC_OP_IMM_32, "sloiw">, Sched<[]>;
def SROIW  : RVBShiftW_ri<0b0010000, 0b101, OPC_OP_IMM_32, "sroiw">, Sched<[]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in
def RORIW : RVBShiftW_ri<0b0110000, 0b101, OPC_OP_IMM_32, "roriw">, Sched<[]>;

let Predicates = [HasStdExtZbs, IsRV64] in {
def SBCLRIW : RVBShiftW_ri<0b0100100, 0b001, OPC_OP_IMM_32, "sbclriw">,
              Sched<[]>;
def SBSETIW : RVBShiftW_ri<0b0010100, 0b001, OPC_OP_IMM_32, "sbsetiw">,
              Sched<[]>;
def SBINVIW : RVBShiftW_ri<0b0110100, 0b001, OPC_OP_IMM_32, "sbinviw">,
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
                      "clzw">, Sched<[]>;
def CTZW   : RVBUnary<0b0110000, 0b00001, 0b001, RISCVOpcode<0b0011011>,
                      "ctzw">, Sched<[]>;
def PCNTW  : RVBUnary<0b0110000, 0b00010, 0b001, RISCVOpcode<0b0011011>,
                      "pcntw">, Sched<[]>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbc, IsRV64] in {
def CLMULW  : ALUW_rr<0b0000101, 0b001, "clmulw">, Sched<[]>;
def CLMULRW : ALUW_rr<0b0000101, 0b010, "clmulrw">, Sched<[]>;
def CLMULHW : ALUW_rr<0b0000101, 0b011, "clmulhw">, Sched<[]>;
} // Predicates = [HasStdExtZbc, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def SHFLW   : ALUW_rr<0b0000100, 0b001, "shflw">, Sched<[]>;
def UNSHFLW : ALUW_rr<0b0000100, 0b101, "unshflw">, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbe, IsRV64] in {
def BDEPW : ALUW_rr<0b0100100, 0b110, "bdepw">, Sched<[]>;
def BEXTW : ALUW_rr<0b0000100, 0b110, "bextw">, Sched<[]>;
} // Predicates = [HasStdExtZbe, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def PACKW  : ALUW_rr<0b0000100, 0b100, "packw">, Sched<[]>;
def PACKUW : ALUW_rr<0b0100100, 0b100, "packuw">, Sched<[]>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbf, IsRV64] in
def BFPW : ALUW_rr<0b0100100, 0b111, "bfpw">, Sched<[]>;

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

let DecoderNamespace = "RVBC", Predicates = [HasStdExtZbproposedc, HasStdExtZbbOrZbp, HasStdExtC, IsRV64] in
def C_ZEXTW : RVBInstC<0b10, "c.zext.w">, Sched<[]>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZbb, IsRV32] in {
def : InstAlias<"zext.b $rd, $rs", (ANDI GPR:$rd, GPR:$rs, 0xFF)>;
def : InstAlias<"zext.h $rd, $rs", (PACK GPR:$rd, GPR:$rs, X0)>;
} // Predicates = [HasStdExtZbb, IsRV32]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : InstAlias<"zext.b $rd, $rs", (ANDI GPR:$rd, GPR:$rs, 0xFF)>;
def : InstAlias<"zext.h $rd, $rs", (PACKW GPR:$rd, GPR:$rs, X0)>;
def : InstAlias<"zext.w $rd, $rs", (PACK GPR:$rd, GPR:$rs, X0)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp] in {
def : InstAlias<"rev.p $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00001)>,
      Sched<[]>;
def : InstAlias<"rev2.n $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00010)>,
      Sched<[]>;
def : InstAlias<"rev.n $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00011)>,
      Sched<[]>;
def : InstAlias<"rev4.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00100)>,
      Sched<[]>;
def : InstAlias<"rev2.b $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b00110)>,
      Sched<[]>;
def : InstAlias<"rev.b $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b00111)>,
      Sched<[]>;
def : InstAlias<"rev8.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01000)>,
      Sched<[]>;
def : InstAlias<"rev4.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01100)>,
      Sched<[]>;
def : InstAlias<"rev2.h $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b01110)>,
      Sched<[]>;
def : InstAlias<"rev.h $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b01111)>,
      Sched<[]>;

def : InstAlias<"zip.n $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0001)>,
      Sched<[]>;
def : InstAlias<"unzip.n $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0001)>,
      Sched<[]>;
def : InstAlias<"zip2.b $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0010)>,
      Sched<[]>;
def : InstAlias<"unzip2.b $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0010)>,
      Sched<[]>;
def : InstAlias<"zip.b $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0011)>,
      Sched<[]>;
def : InstAlias<"unzip.b $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0011)>,
      Sched<[]>;
def : InstAlias<"zip4.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0100)>,
      Sched<[]>;
def : InstAlias<"unzip4.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0100)>,
      Sched<[]>;
def : InstAlias<"zip2.h $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b0110)>,
      Sched<[]>;
def : InstAlias<"unzip2.h $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b0110)>,
      Sched<[]>;
def : InstAlias<"zip.h $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b0111)>,
      Sched<[]>;
def : InstAlias<"unzip.h $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b0111)>,
      Sched<[]>;

def : InstAlias<"orc.p $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00001)>,
      Sched<[]>;
def : InstAlias<"orc2.n $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00010)>,
      Sched<[]>;
def : InstAlias<"orc.n $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00011)>,
      Sched<[]>;
def : InstAlias<"orc4.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00100)>,
      Sched<[]>;
def : InstAlias<"orc2.b $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b00110)>,
      Sched<[]>;
def : InstAlias<"orc.b $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b00111)>,
      Sched<[]>;
def : InstAlias<"orc8.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01000)>,
      Sched<[]>;
def : InstAlias<"orc4.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01100)>,
      Sched<[]>;
def : InstAlias<"orc2.h $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b01110)>,
      Sched<[]>;
def : InstAlias<"orc.h $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b01111)>,
      Sched<[]>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : InstAlias<"rev16 $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b10000)>, Sched<[]>;
def : InstAlias<"rev8 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11000)>, Sched<[]>;
def : InstAlias<"rev4 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11100)>, Sched<[]>;
def : InstAlias<"rev2 $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b11110)>, Sched<[]>;
def : InstAlias<"rev $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b11111)>, Sched<[]>;

def : InstAlias<"zip8 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1000)>,
      Sched<[]>;
def : InstAlias<"unzip8 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1000)>,
      Sched<[]>;
def : InstAlias<"zip4 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1100)>,
      Sched<[]>;
def : InstAlias<"unzip4 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1100)>,
      Sched<[]>;
def : InstAlias<"zip2 $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b1110)>,
      Sched<[]>;
def : InstAlias<"unzip2 $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b1110)>,
      Sched<[]>;
def : InstAlias<"zip $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b1111)>,
      Sched<[]>;
def : InstAlias<"unzip $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b1111)>,
      Sched<[]>;

def : InstAlias<"orc16 $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b10000)>, Sched<[]>;
def : InstAlias<"orc8 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11000)>, Sched<[]>;
def : InstAlias<"orc4 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11100)>, Sched<[]>;
def : InstAlias<"orc2 $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b11110)>, Sched<[]>;
def : InstAlias<"orc $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b11111)>, Sched<[]>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : InstAlias<"rev16.w $rd, $rs", (GREVI GPR:$rd, GPR:$rs, 0b010000)>,
      Sched<[]>;
def : InstAlias<"rev8.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011000)>,
      Sched<[]>;
def : InstAlias<"rev4.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011100)>,
      Sched<[]>;
def : InstAlias<"rev2.w $rd, $rs",  (GREVI GPR:$rd, GPR:$rs, 0b011110)>,
      Sched<[]>;
def : InstAlias<"rev.w $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b011111)>,
      Sched<[]>;
def : InstAlias<"rev32 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b100000)>,
      Sched<[]>;
def : InstAlias<"rev16 $rd, $rs",   (GREVI GPR:$rd, GPR:$rs, 0b110000)>,
      Sched<[]>;
def : InstAlias<"rev8 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111000)>,
      Sched<[]>;
def : InstAlias<"rev4 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111100)>,
      Sched<[]>;
def : InstAlias<"rev2 $rd, $rs",    (GREVI GPR:$rd, GPR:$rs, 0b111110)>,
      Sched<[]>;
def : InstAlias<"rev $rd, $rs",     (GREVI GPR:$rd, GPR:$rs, 0b111111)>,
      Sched<[]>;

def : InstAlias<"zip8.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01000)>,
      Sched<[]>;
def : InstAlias<"unzip8.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01000)>,
      Sched<[]>;
def : InstAlias<"zip4.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01100)>,
      Sched<[]>;
def : InstAlias<"unzip4.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01100)>,
      Sched<[]>;
def : InstAlias<"zip2.w $rd, $rs",   (SHFLI   GPR:$rd, GPR:$rs, 0b01110)>,
      Sched<[]>;
def : InstAlias<"unzip2.w $rd, $rs", (UNSHFLI GPR:$rd, GPR:$rs, 0b01110)>,
      Sched<[]>;
def : InstAlias<"zip.w $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b01111)>,
      Sched<[]>;
def : InstAlias<"unzip.w $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b01111)>,
      Sched<[]>;
def : InstAlias<"zip16 $rd, $rs",    (SHFLI   GPR:$rd, GPR:$rs, 0b10000)>,
      Sched<[]>;
def : InstAlias<"unzip16 $rd, $rs",  (UNSHFLI GPR:$rd, GPR:$rs, 0b10000)>,
      Sched<[]>;
def : InstAlias<"zip8 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11000)>,
      Sched<[]>;
def : InstAlias<"unzip8 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11000)>,
      Sched<[]>;
def : InstAlias<"zip4 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11100)>,
      Sched<[]>;
def : InstAlias<"unzip4 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11100)>,
      Sched<[]>;
def : InstAlias<"zip2 $rd, $rs",     (SHFLI   GPR:$rd, GPR:$rs, 0b11110)>,
      Sched<[]>;
def : InstAlias<"unzip2 $rd, $rs",   (UNSHFLI GPR:$rd, GPR:$rs, 0b11110)>,
      Sched<[]>;
def : InstAlias<"zip $rd, $rs",      (SHFLI   GPR:$rd, GPR:$rs, 0b11111)>,
      Sched<[]>;
def : InstAlias<"unzip $rd, $rs",    (UNSHFLI GPR:$rd, GPR:$rs, 0b11111)>,
      Sched<[]>;

def : InstAlias<"orc16.w $rd, $rs", (GORCI GPR:$rd, GPR:$rs, 0b010000)>,
      Sched<[]>;
def : InstAlias<"orc8.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011000)>,
      Sched<[]>;
def : InstAlias<"orc4.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011100)>,
      Sched<[]>;
def : InstAlias<"orc2.w $rd, $rs",  (GORCI GPR:$rd, GPR:$rs, 0b011110)>,
      Sched<[]>;
def : InstAlias<"orc.w $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b011111)>,
      Sched<[]>;
def : InstAlias<"orc32 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b100000)>,
      Sched<[]>;
def : InstAlias<"orc16 $rd, $rs",   (GORCI GPR:$rd, GPR:$rs, 0b110000)>,
      Sched<[]>;
def : InstAlias<"orc8 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111000)>,
      Sched<[]>;
def : InstAlias<"orc4 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111100)>,
      Sched<[]>;
def : InstAlias<"orc2 $rd, $rs",    (GORCI GPR:$rd, GPR:$rs, 0b111110)>,
      Sched<[]>;
def : InstAlias<"orc $rd, $rs",     (GORCI GPR:$rd, GPR:$rs, 0b111111)>,
      Sched<[]>;
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

let Predicates = [HasStdExtZbproposedc, HasStdExtZbbOrZbp, HasStdExtC, IsRV64] in {
def : CompressPat<(PACK GPRC:$rs1, GPRC:$rs1, X0),
                  (C_ZEXTW GPRC:$rs1)>;
} // Predicates = [HasStdExtZbproposedc, HasStdExtC, IsRV64]

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//
def SLOIPat   : ComplexPattern<XLenVT, 2, "SelectSLOI", [or]>;
def SROIPat   : ComplexPattern<XLenVT, 2, "SelectSROI", [or]>;
def SLLIUWPat : ComplexPattern<i64, 2, "SelectSLLIUW", [and]>;
def SLOIWPat  : ComplexPattern<i64, 2, "SelectSLOIW", [sext_inreg]>;
def SROIWPat  : ComplexPattern<i64, 2, "SelectSROIW", [or]>;

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(and GPR:$rs1, (not GPR:$rs2)), (ANDN GPR:$rs1, GPR:$rs2)>;
def : Pat<(or  GPR:$rs1, (not GPR:$rs2)), (ORN  GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor GPR:$rs1, (not GPR:$rs2)), (XNOR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbb] in {
def : Pat<(not (shl (not GPR:$rs1), GPR:$rs2)),
          (SLO GPR:$rs1, GPR:$rs2)>;
def : Pat<(not (srl (not GPR:$rs1), GPR:$rs2)),
          (SRO GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbb]

let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotl GPR:$rs1, GPR:$rs2), (ROL GPR:$rs1, GPR:$rs2)>;
def : Pat<(rotr GPR:$rs1, GPR:$rs2), (ROR GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp]

let Predicates = [HasStdExtZbs] in {
def : Pat<(and (not (shiftop<shl> 1, GPR:$rs2)), GPR:$rs1),
          (SBCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (rotl -2, GPR:$rs2), GPR:$rs1), (SBCLR GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (SBSET GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor (shiftop<shl> 1, GPR:$rs2), GPR:$rs1),
          (SBINV GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (shiftop<srl> GPR:$rs1, GPR:$rs2), 1),
          (SBEXT GPR:$rs1, GPR:$rs2)>;

def : Pat<(shiftop<shl> 1, GPR:$rs2),
          (SBSET X0, GPR:$rs2)>;

def : Pat<(and GPR:$rs1, SBCLRMask:$mask),
          (SBCLRI GPR:$rs1, (SBCLRXForm imm:$mask))>;
def : Pat<(or GPR:$rs1, SBSETINVMask:$mask),
          (SBSETI GPR:$rs1, (SBSETINVXForm imm:$mask))>;
def : Pat<(xor GPR:$rs1, SBSETINVMask:$mask),
          (SBINVI GPR:$rs1, (SBSETINVXForm imm:$mask))>;
}

let Predicates = [HasStdExtZbb] in {
def : Pat<(SLOIPat GPR:$rs1, uimmlog2xlen:$shamt),
          (SLOI GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(SROIPat GPR:$rs1, uimmlog2xlen:$shamt),
          (SROI GPR:$rs1, uimmlog2xlen:$shamt)>;
} // Predicates = [HasStdExtZbb]

// There's no encoding for roli in the current version of the 'B' extension
// (v0.92) as it can be implemented with rori by negating the immediate.
let Predicates = [HasStdExtZbbOrZbp] in {
def : Pat<(rotr GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(rotl GPR:$rs1, uimmlog2xlen:$shamt),
          (RORI GPR:$rs1, (ImmROTL2R uimmlog2xlen:$shamt))>;
}

// We don't pattern-match sbclri[w], sbseti[w], sbinvi[w] because they are
// pattern-matched by simple andi, ori, and xori.
let Predicates = [HasStdExtZbs] in
def : Pat<(and (srl GPR:$rs1, uimmlog2xlen:$shamt), (XLenVT 1)),
          (SBEXTI GPR:$rs1, uimmlog2xlen:$shamt)>;

def riscv_grevi    : SDNode<"RISCVISD::GREVI", SDTIntBinOp, []>;
def riscv_greviw   : SDNode<"RISCVISD::GREVIW", SDTIntBinOp, []>;
def riscv_gorci    : SDNode<"RISCVISD::GORCI", SDTIntBinOp, []>;
def riscv_gorciw   : SDNode<"RISCVISD::GORCIW", SDTIntBinOp, []>;

let Predicates = [HasStdExtZbp] in {
def : Pat<(riscv_grevi GPR:$rs1, timm:$shamt), (GREVI GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_gorci GPR:$rs1, timm:$shamt), (GORCI GPR:$rs1, timm:$shamt)>;
} // Predicates = [HasStdExtZbp]

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(rotr (riscv_grevi GPR:$rs1, (i32 24)), (i32 16)), (GREVI GPR:$rs1, 8)>;
def : Pat<(rotl (riscv_grevi GPR:$rs1, (i32 24)), (i32 16)), (GREVI GPR:$rs1, 8)>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbt] in {
def : Pat<(or (and (not GPR:$rs2), GPR:$rs3), (and GPR:$rs2, GPR:$rs1)),
          (CMIX GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_selectcc GPR:$rs2, (XLenVT 0), (XLenVT 17), GPR:$rs3, GPR:$rs1),
          (CMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
} // Predicates = [HasStdExtZbt]

// fshl and fshr concatenate their operands in the same order. fsr and fsl
// instruction use different orders. fshl will return its first operand for
// shift of zero, fshr will return its second operand. fsl and fsr both return
// $rs1 so the patterns need to have different operand orders.
//
// fshl and fshr only read the lower log2(xlen) bits of the shift amount, but
// fsl/fsr instructions read log2(xlen)+1 bits. DAG combine may have removed
// an AND mask on the shift amount that we need to add back to avoid a one in
// the extra bit.
// FIXME: If we can prove that the extra bit in the shift amount is zero, we
// don't need this mask.
let Predicates = [HasStdExtZbt, IsRV32] in {
def : Pat<(fshl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, (ANDI GPR:$rs2, 31), GPR:$rs3)>;
def : Pat<(fshr GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSR GPR:$rs1, (ANDI GPR:$rs2, 31), GPR:$rs3)>;
}
let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(fshl GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSL GPR:$rs1, (ANDI GPR:$rs2, 63), GPR:$rs3)>;
def : Pat<(fshr GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSR GPR:$rs1, (ANDI GPR:$rs2, 63), GPR:$rs3)>;
}
let Predicates = [HasStdExtZbt] in {
def : Pat<(fshr GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, uimmlog2xlen:$shamt)>;
// We can use FSRI for fshl by immediate if we subtract the immediate from
// XLen and swap the operands.
def : Pat<(fshl GPR:$rs3, GPR:$rs1, uimmlog2xlen:$shamt),
          (FSRI GPR:$rs1, GPR:$rs3, (ImmROTL2R uimmlog2xlen:$shamt))>;
} // Predicates = [HasStdExtZbt]

let Predicates = [HasStdExtZbb] in {
def : Pat<(ctlz GPR:$rs1), (CLZ GPR:$rs1)>;
def : Pat<(cttz GPR:$rs1), (CTZ GPR:$rs1)>;
def : Pat<(ctpop GPR:$rs1), (PCNT GPR:$rs1)>;
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

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(or (and GPR:$rs1, 0x0000FFFF), (shl GPR:$rs2, (i32 16))),
          (PACK GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in
def : Pat<(or (and GPR:$rs1, 0x00000000FFFFFFFF), (shl GPR:$rs2, (i64 32))),
          (PACK GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(or (and GPR:$rs2, 0xFFFF0000), (srl GPR:$rs1, (i32 16))),
          (PACKU GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in
def : Pat<(or (and GPR:$rs2, 0xFFFFFFFF00000000), (srl GPR:$rs1, (i64 32))),
          (PACKU GPR:$rs1, GPR:$rs2)>;
let Predicates = [HasStdExtZbbOrZbp] in
def : Pat<(or (and (shl GPR:$rs2, (XLenVT 8)), 0xFF00),
              (and GPR:$rs1, 0x00FF)),
          (PACKH GPR:$rs1, GPR:$rs2)>;

let Predicates = [HasStdExtZbbOrZbp, IsRV32] in
def : Pat<(and GPR:$rs, 0x0000FFFF), (PACK GPR:$rs, X0)>;
let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(and GPR:$rs, 0x000000000000FFFF), (PACKW GPR:$rs, X0)>;
def : Pat<(and GPR:$rs, 0x00000000FFFFFFFF), (PACK GPR:$rs, X0)>;
}

let Predicates = [HasStdExtZbp, IsRV32] in {
def : Pat<(or (or (and (shl GPR:$rs1, (i32 8)), (i32 0x00FF0000)),
                  (and GPR:$rs1, (i32 0xFF0000FF))),
              (and (srl GPR:$rs1, (i32 8)), (i32 0x0000FF00))),
          (SHFLI GPR:$rs1, (i32 8))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 4)), (i32 0x0F000F00)),
                  (and GPR:$rs1, (i32 0xF00FF00F))),
              (and (srl GPR:$rs1, (i32 4)), (i32 0x00F000F0))),
          (SHFLI GPR:$rs1, (i32 4))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 2)), (i32 0x30303030)),
                  (and GPR:$rs1, (i32 0xC3C3C3C3))),
              (and (srl GPR:$rs1, (i32 2)), (i32 0x0C0C0C0C))),
          (SHFLI GPR:$rs1, (i32 2))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i32 1)), (i32 0x44444444)),
                  (and GPR:$rs1, (i32 0x99999999))),
              (and (srl GPR:$rs1, (i32 1)), (i32 0x22222222))),
          (SHFLI GPR:$rs1, (i32 1))>;
} // Predicates = [HasStdExtZbp, IsRV32]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(or (or (and (shl GPR:$rs1, (i64 16)), (i64 0x0000FFFF00000000)),
                  (and GPR:$rs1, (i64 0xFFFF00000000FFFF))),
              (and (srl GPR:$rs1, (i64 16)), (i64 0x00000000FFFF0000))),
          (SHFLI GPR:$rs1, (i64 16))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 8)), (i64 0x00FF000000FF0000)),
                  (and GPR:$rs1, (i64 0xFF0000FFFF0000FF))),
              (and (srl GPR:$rs1, (i64 8)), (i64 0x0000FF000000FF00))),
          (SHFLI GPR:$rs1, (i64 8))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 4)), (i64 0x0F000F000F000F00)),
                  (and GPR:$rs1, (i64 0xF00FF00FF00FF00F))),
              (and (srl GPR:$rs1, (i64 4)), (i64 0x00F000F000F000F0))),
          (SHFLI GPR:$rs1, (i64 4))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 2)), (i64 0x3030303030303030)),
                  (and GPR:$rs1, (i64 0xC3C3C3C3C3C3C3C3))),
              (and (srl GPR:$rs1, (i64 2)), (i64 0x0C0C0C0C0C0C0C0C))),
          (SHFLI GPR:$rs1, (i64 2))>;
def : Pat<(or (or (and (shl GPR:$rs1, (i64 1)), (i64 0x4444444444444444)),
                  (and GPR:$rs1, (i64 0x9999999999999999))),
              (and (srl GPR:$rs1, (i64 1)), (i64 0x2222222222222222))),
          (SHFLI GPR:$rs1, (i64 1))>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(and (add GPR:$rs, simm12:$simm12), (i64 0xFFFFFFFF)),
          (ADDIWU GPR:$rs, simm12:$simm12)>;
def : Pat<(SLLIUWPat GPR:$rs1, uimmlog2xlen:$shamt),
          (SLLIUW GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(and (add GPR:$rs1, GPR:$rs2), (i64 0xFFFFFFFF)),
          (ADDWU GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (sub GPR:$rs1, GPR:$rs2), (i64 0xFFFFFFFF)),
          (SUBWU GPR:$rs1, GPR:$rs2)>;
def : Pat<(add GPR:$rs1, (and GPR:$rs2, (i64 0xFFFFFFFF))),
          (ADDUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sub GPR:$rs1, (and GPR:$rs2, (i64 0xFFFFFFFF))),
          (SUBUW GPR:$rs1, GPR:$rs2)>;
def : Pat<(not (riscv_sllw (not GPR:$rs1), GPR:$rs2)),
          (SLOW GPR:$rs1, GPR:$rs2)>;
def : Pat<(not (riscv_srlw (not GPR:$rs1), GPR:$rs2)),
          (SROW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(riscv_rolw GPR:$rs1, GPR:$rs2),
          (ROLW GPR:$rs1, GPR:$rs2)>;
def : Pat<(riscv_rorw GPR:$rs1, GPR:$rs2),
          (RORW GPR:$rs1, GPR:$rs2)>;
def : Pat<(riscv_rorw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, uimm5:$rs2)>;
def : Pat<(riscv_rolw GPR:$rs1, uimm5:$rs2),
          (RORIW GPR:$rs1, (ImmROTL2RW uimm5:$rs2))>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]

let Predicates = [HasStdExtZbs, IsRV64] in {
def : Pat<(and (not (riscv_sllw 1, GPR:$rs2)), (assertsexti32 GPR:$rs1)),
          (SBCLRW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (and (not (riscv_sllw 1, GPR:$rs2)), GPR:$rs1), i32),
          (SBCLRW GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (riscv_sllw 1, GPR:$rs2), (assertsexti32 GPR:$rs1)),
          (SBSETW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (or (riscv_sllw 1, GPR:$rs2), GPR:$rs1), i32),
          (SBSETW GPR:$rs1, GPR:$rs2)>;
def : Pat<(xor (riscv_sllw 1, GPR:$rs2), (assertsexti32 GPR:$rs1)),
          (SBINVW GPR:$rs1, GPR:$rs2)>;
def : Pat<(sext_inreg (xor (riscv_sllw 1, GPR:$rs2), GPR:$rs1), i32),
          (SBINVW GPR:$rs1, GPR:$rs2)>;
def : Pat<(and (riscv_srlw GPR:$rs1, GPR:$rs2), 1),
          (SBEXTW GPR:$rs1, GPR:$rs2)>;

def : Pat<(riscv_sllw 1, GPR:$rs2),
          (SBSETW X0, GPR:$rs2)>;

def : Pat<(and (assertsexti32 GPR:$rs1), SBCLRWMask:$mask),
          (SBCLRIW GPR:$rs1, (SBCLRXForm imm:$mask))>;
def : Pat<(or (assertsexti32 GPR:$rs1), SBSETINVWMask:$mask),
          (SBSETIW GPR:$rs1, (SBSETINVXForm imm:$mask))>;
def : Pat<(xor (assertsexti32 GPR:$rs1), SBSETINVWMask:$mask),
          (SBINVIW GPR:$rs1, (SBSETINVXForm imm:$mask))>;

} // Predicates = [HasStdExtZbs, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(SLOIWPat GPR:$rs1, uimmlog2xlen:$shamt),
          (SLOIW GPR:$rs1, uimmlog2xlen:$shamt)>;
def : Pat<(SROIWPat GPR:$rs1, uimmlog2xlen:$shamt),
          (SROIW GPR:$rs1, uimmlog2xlen:$shamt)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbp, IsRV64] in {
def : Pat<(riscv_rorw (riscv_greviw GPR:$rs1, 24), (i64 16)), (GREVIW GPR:$rs1, 8)>;
def : Pat<(riscv_rolw (riscv_greviw GPR:$rs1, 24), (i64 16)), (GREVIW GPR:$rs1, 8)>;
def : Pat<(riscv_greviw GPR:$rs1, timm:$shamt), (GREVIW GPR:$rs1, timm:$shamt)>;
def : Pat<(riscv_gorciw GPR:$rs1, timm:$shamt), (GORCIW GPR:$rs1, timm:$shamt)>;
} // Predicates = [HasStdExtZbp, IsRV64]

let Predicates = [HasStdExtZbt, IsRV64] in {
def : Pat<(riscv_fslw GPR:$rs1, GPR:$rs3, GPR:$rs2),
          (FSLW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs3, GPR:$rs1, GPR:$rs2),
          (FSRW GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(riscv_fsrw GPR:$rs3, GPR:$rs1, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, uimm5:$shamt)>;
def : Pat<(riscv_fslw GPR:$rs3, GPR:$rs1, uimm5:$shamt),
          (FSRIW GPR:$rs1, GPR:$rs3, (ImmROTL2RW uimm5:$shamt))>;
} // Predicates = [HasStdExtZbt, IsRV64]

let Predicates = [HasStdExtZbb, IsRV64] in {
def : Pat<(add (ctlz (and GPR:$rs1, (i64 0xFFFFFFFF))), (i64 -32)),
          (CLZW GPR:$rs1)>;
// computeKnownBits can't figure out that the and mask on the add result is
// unnecessary so we need to pattern match it away.
def : Pat<(and (add (ctlz (and GPR:$rs1, (i64 0xFFFFFFFF))), (i64 -32)),
               (i64 0xFFFFFFFF)),
          (CLZW GPR:$rs1)>;
def : Pat<(cttz (or GPR:$rs1, (i64 0x100000000))),
          (CTZW GPR:$rs1)>;
def : Pat<(ctpop (and GPR:$rs1, (i64 0xFFFFFFFF))), (PCNTW GPR:$rs1)>;
} // Predicates = [HasStdExtZbb, IsRV64]

let Predicates = [HasStdExtZbbOrZbp, IsRV64] in {
def : Pat<(sext_inreg (or (shl GPR:$rs2, (i64 16)),
                          (and GPR:$rs1, 0x000000000000FFFF)),
                      i32),
          (PACKW GPR:$rs1, GPR:$rs2)>;
def : Pat<(or (and (assertsexti32 GPR:$rs2), 0xFFFFFFFFFFFF0000),
              (srl (and GPR:$rs1, 0x00000000FFFF0000),
                   (i64 16))),
          (PACKUW GPR:$rs1, GPR:$rs2)>;
} // Predicates = [HasStdExtZbbOrZbp, IsRV64]
