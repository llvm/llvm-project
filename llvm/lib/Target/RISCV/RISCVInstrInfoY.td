//===-- RISCVInstrInfoY.td - RISCV instructions -------------*- tblgen-*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Base Y'
// instruction set.
//
// This version is still experimental as the 'Y' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

// A 9-bit unsigned immediate where the least significant four bits are zero.
def uimm9_lsb0000 : RISCVUImmLsbZeroOp<9, 4>;
// A 10-bit unsigned immediate where the least significant four bits are zero.
def uimm10_lsb0000 : RISCVUImmLsbZeroOp<10, 4>;

def YBNDSWImmOperand : AsmOperandClass {
  let Name = "YBNDSWImm";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidYBNDSWImm";
}

def ybndsw_imm : RISCVOp<XLenVT>, ImmLeaf<XLenVT, [{
  return llvm::RISCV::isValidYBNDSWImm(Imm);
}]> {
  let EncoderMethod = "getYBNDSWImmOpValue";
  let ParserMatchClass = YBNDSWImmOperand;
  let DecoderMethod = "DecodeYBNDSWImm";
  let OperandType = "OPERAND_UIMM9_YBNDSWI";
}

def uimm7_eq_xlen : RISCVUImmOp<7>, ImmLeaf<XLenVT, [{
  return Imm == Subtarget->getXLen();
}]> {
  let ParserMatchClass = UImmAsmOperand<7, "EqXLen">;
  let DecoderMethod = "decodeUImm7EqXLenOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return Imm == Subtarget->getXLen()
  }];
  let OperandType = "OPERAND_UIMM7_EQ_XLEN";
}

//===----------------------------------------------------------------------===//
// Instruction Formats and class templates
//===----------------------------------------------------------------------===//

defvar OPC_RVY_A = OPC_CUSTOM_3;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// All regular 3-operand RVY instructions use opc=CUSTOM-3/RVY-A with funct3=0
class RVY_rr<bits<7> funct7, string opcodestr, DAGOperand rdOp,
             DAGOperand rs1Op, DAGOperand rs2Op>
    : RVInstR<funct7, 0b000, OPC_RVY_A, (outs rdOp:$rd),
              (ins rs1Op:$rs1, rs2Op:$rs2), opcodestr, "$rd, $rs1, $rs2">;

// Like RVY_rr, except rs2 is now an additional function code.
class RVYInst2Op<bits<7> funct7, bits<5> funct5, DAGOperand rdOp,
                 DAGOperand rs1Op, string opcodestr, string argstr>
    : RVInst<(outs rdOp:$rd), (ins rs1Op:$rs1), opcodestr, argstr, [],
             InstFormatR> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = funct5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000; // funct3
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_RVY_A.Value;
}

class RVY_r_xx<bits<5> funct5, string opcodestr>
    : RVYInst2Op<0b1111000, funct5, GPR, GPR, opcodestr, "$rd, $rs1">;
class RVY_r_xy<bits<5> funct5, string opcodestr>
    : RVYInst2Op<0b1111010, funct5, GPR, YGPR, opcodestr, "$rd, $rs1">;
class RVY_ri<bits<3> funct3, string opcodestr, DAGOperand rdOp = YGPR,
             DAGOperand rs1Op = YGPR>
    : RVInstI<funct3, OPC_RVY_A, (outs rdOp:$rd),
              (ins rs1Op:$rs1, simm12_lo:$imm12), opcodestr,
              "$rd, $rs1, $imm12">;
class RVY_bndswi<string opcodestr>
   : RVInstIBase<0b101, OPC_RVY_A, (outs YGPR:$rd),
                 (ins YGPR:$rs1, ybndsw_imm:$imm), opcodestr,
                 "$rd, $rs1, $imm"> {
  bits<5> rd;
  bits<5> rs1;
  bits<9> imm;

  let Inst{31-29} = 0b111;
  let Inst{28-20} = imm;
}
class RVY_srli<string opcodestr>
    : RVInstIBase<0b101, OPC_RVY_A, (outs GPR:$rd),
                  (ins YGPR:$rs1, uimm7_eq_xlen:$shamt), opcodestr,
                  "$rd, $rs1, $shamt"> {
  bits<7> shamt;

  let Inst{31-27} = 0b00000;
  let Inst{26-20} = shamt;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Predicates = [HasStdExtY] in {
//
// Instructions to Update The Capability Pointer
//
let isReMaterializable = true, isAsCheapAsAMove = true in {
def YADD : RVY_rr<0b0000011, "yadd", YGPR, YGPR, GPRNoX0>;
def YADDI : RVY_ri<0b100, "yaddi">;
}
def : InstAlias<"yadd $rd, $rs1, $imm12",
                (YADDI YGPR:$rd, YGPR:$rs1, simm12_lo:$imm12), 0>;
def YADDRW : RVY_rr<0b0001011, "yaddrw", YGPR, YGPR, GPR>;

//
// Instructions to Manipulate Capabilities
//
def YPERMC : RVY_rr<0b0010011, "ypermc", YGPR, YGPR, GPR>;
// YMV is encoded as YADD with rs2=x0
let isReMaterializable = true, isAsCheapAsAMove = true, isMoveReg = true in
def YMV : RVYInst2Op<0b0000011, 0b00000, YGPR, YGPR, "ymv", "$rd, $rs1">;
def PACKY : RVY_rr<0b0000001, "packy", YGPR, YGPR, GPR>;
def : MnemonicAlias<"yhiw", "packy">;
def YBNDSW : RVY_rr<0b0011011, "ybndsw", YGPR, YGPR, GPR>;
def YBNDSWI : RVY_bndswi<"ybndswi">;
def : InstAlias<"ybndsw $rd, $rs1, $imm",
                (YBNDSWI YGPR:$rd, YGPR:$rs1, ybndsw_imm:$imm), 0>;
def YBNDSRW : RVY_rr<0b0100011, "ybndsrw", YGPR, YGPR, GPR>;
def YSUNSEAL : RVY_rr<0b0000111, "ysunseal", YGPR, YGPR, YGPR>;
def YBLD : RVY_rr<0b0001111, "ybld", YGPR, YGPR, YGPR>;

//
// Instructions to Decode Capability Bounds
//
def YBASER : RVY_r_xy<0b00000, "ybaser">;
def YLENR : RVY_r_xy<0b00011, "ylenr">;
def YTOPR : RVY_r_xy<0b00010, "ytopr">;

//
// Instructions to Extract Capability Fields
//
def YTAGR : RVY_r_xy<0b00100, "ytagr">;
def YPERMR : RVY_r_xy<0b00001, "ypermr">;
def YTYPER : RVY_r_xy<0b00101, "ytyper">;
// Extracting the high bits of capabilities is effectively a XLEN-bit right
// shift. It is encoded this way so that a future "YLEN arithmetic" extension
// can define shifts with arbitrary shift amounts. YHIR is a pseudo that expands
// to the XLEN-wide shift value depending on RV32/RV64.
def SRLIY : RVY_srli<"srliy">;
let Predicates = [HasStdExtY, IsRV64] in {
def : InstAlias<"yhir $rd, $rs1", (SRLIY GPR:$rd, YGPR:$rs1, 64)>;
}
let Predicates = [HasStdExtY, IsRV32] in {
def : InstAlias<"yhir $rd, $rs1", (SRLIY GPR:$rd, YGPR:$rs1, 32)>;
}
//
// Miscellaneous Instructions to Handle Capability Data
//
def YEQ : RVY_rr<0b0000110, "yeq", GPR, YGPR, YGPR>;
def YSS : RVY_rr<0b0001110, "yss", GPR, YGPR, YGPR>;
def YAMASK : RVY_r_xx<0b00000, "yamask">;

//
// Instructions to Load and Store Capability Data
//
def LY : Load_ri<0b001, "ly", YGPR, opcode=OPC_RVY_A>,
         Sched<[/* TODO: WriteLDY, */ ReadMemBase]>;
def SY : Store_rri<0b010, "sy", YGPR, opcode=OPC_RVY_A>,
         Sched<[/* TODO: WriteSTY, */ ReadStoreData, ReadMemBase]>;
} // Predicates = [HasStdExtY]

//
// Compressed capability loads/stores using Zcf/Zcd encodings.
// These only exist RVY mode, not in RVI compatibility mode.
//
let Predicates = [HasStdExtZca, IsStdExtYCapMode, IsRV32] in {
// DecoderNamespace needed since RVY32 reuses Zcf/Zclsd encodings
let DecoderNamespace = "RVY32Only" in {
def C_LY_RV32 : CLoad_ri<0b011, "c.ly", YGPRC, uimm8_lsb000>,
              Sched<[/*TODO: WriteLDY,*/ ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}
def C_SY_RV32 : CStore_rri<0b111, "c.sy", YGPRC, uimm8_lsb000>,
                Sched<[/*TODO: WriteSTY,*/ ReadStoreData, ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}
def C_LYSP_RV32 : CStackLoad<0b011, "c.lysp", YGPRNoX0, uimm9_lsb000>,
                  Sched<[/*TODO: WriteLDY,*/ ReadMemBase]> {
  let Inst{4-2} = imm{8-6};
}
def C_SYSP_RV32 : CStackStore<0b111, "c.sysp", YGPR, uimm9_lsb000>,
                  Sched<[/*TODO: WriteSTY,*/ ReadStoreData, ReadMemBase]> {
  let Inst{9-7} = imm{8-6};
}
} // let DecoderNamespace = "RVY32Only"
def : CompressPat<(LY YGPRC:$rd, GPRCMem:$rs1, uimm8_lsb000:$imm),
                  (C_LY_RV32 YGPRC:$rd, GPRCMem:$rs1, uimm8_lsb000:$imm)>;
def : CompressPat<(SY YGPRC:$rs2, GPRCMem:$rs1, uimm8_lsb000:$imm),
                  (C_SY_RV32 YGPRC:$rs2, GPRCMem:$rs1, uimm8_lsb000:$imm)>;
def : CompressPat<(LY YGPRNoX0:$rd, SPMem:$rs1,  uimm9_lsb000:$imm),
                  (C_LYSP_RV32 YGPRNoX0:$rd, SPMem:$rs1, uimm9_lsb000:$imm)>;
def : CompressPat<(SY YGPR:$rs2, SPMem:$rs1, uimm9_lsb000:$imm),
                  (C_SYSP_RV32 YGPR:$rs2, SPMem:$rs1, uimm9_lsb000:$imm)>;
} // Predicates = [HasStdExtZca, IsStdExtYCapMode, IsRV32]

let Predicates = [HasStdExtZca, IsStdExtYCapMode, IsRV64] in {
// DecoderNamespace needed since RVY64 reuses Zcd/Zcm* encodings
let DecoderNamespace = "RVY64Only" in {
def C_LY_RV64 : CLoad_ri<0b001, "c.ly", YGPRC, uimm9_lsb0000>,
                Sched<[/* TODO: WriteLDY, */ ReadMemBase]> {
  bits<9> imm;
  let Inst{12-11} = imm{5-4};
  let Inst{10} = imm{8};
  let Inst{6-5} = imm{7-6};
}
def C_SY_RV64 : CStore_rri<0b101, "c.sy", YGPRC, uimm9_lsb0000>,
                Sched<[/*TODO: WriteSTY,*/ ReadStoreData, ReadMemBase]> {
  bits<9> imm;
  let Inst{12-11} = imm{5-4};
  let Inst{10} = imm{8};
  let Inst{6-5} = imm{7-6};
}
def C_LYSP_RV64 : CStackLoad<0b001, "c.lysp", YGPRNoX0, uimm10_lsb0000>,
                  Sched<[/*TODO: WriteLDY,*/ ReadMemBase]> {
  let Inst{4-2} = imm{8-6};
}
def C_SYSP_RV64 : CStackStore<0b101, "c.sysp", YGPR, uimm10_lsb0000>,
                  Sched<[/*TODO: WriteSTY,*/ ReadStoreData, ReadMemBase]> {
  let Inst{10-7} = imm{9-6};
}
} // let DecoderNamespace = "RVY64Only"
def : CompressPat<(LY YGPRC:$rd, GPRCMem:$rs1, uimm9_lsb0000:$imm),
                  (C_LY_RV64 YGPRC:$rd, GPRCMem:$rs1, uimm9_lsb0000:$imm)>;
def : CompressPat<(SY YGPRC:$rs2, GPRCMem:$rs1, uimm9_lsb0000:$imm),
                  (C_SY_RV64 YGPRC:$rs2, GPRCMem:$rs1, uimm9_lsb0000:$imm)>;
def : CompressPat<(LY YGPRNoX0:$rd, SPMem:$rs1,  uimm10_lsb0000:$imm),
                  (C_LYSP_RV64 YGPRNoX0:$rd, SPMem:$rs1, uimm10_lsb0000:$imm)>;
def : CompressPat<(SY YGPR:$rs2, SPMem:$rs1, uimm10_lsb0000:$imm),
                  (C_SYSP_RV64 YGPR:$rs2, SPMem:$rs1, uimm10_lsb0000:$imm)>;
} // Predicates = [HasStdExtZca, IsStdExtYCapMode, IsRV64]
