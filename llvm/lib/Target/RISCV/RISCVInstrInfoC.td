//===- RISCVInstrInfoC.td - Compressed RISCV instructions -*- tblgen-*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsC.td"

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

def UImmLog2XLenNonZeroAsmOperand : AsmOperandClass {
  let Name = "UImmLog2XLenNonZero";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidUImmLog2XLenNonZero";
}

def uimmlog2xlennonzero : Operand<XLenVT>, ImmLeaf<XLenVT, [{
  if (Subtarget->is64Bit())
    return isUInt<6>(Imm) && (Imm != 0);
  return isUInt<5>(Imm) && (Imm != 0);
}]> {
  let ParserMatchClass = UImmLog2XLenNonZeroAsmOperand;
  // TODO: should ensure invalid shamt is rejected when decoding.
  let DecoderMethod = "decodeUImmOperand<6>";
}

def simm6 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isInt<6>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<6>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<6>";
}

def uimm6nonzero : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isUInt<6>(Imm) && (Imm != 0);}]> {
  let ParserMatchClass = UImmAsmOperand<6, "NonZero">;
  let DecoderMethod = "decodeUImmOperand<6>";
}

// A 7-bit unsigned immediate where the least significant two bits are zero.
def uimm7_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<5, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<7, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<7>";
}

// A 8-bit unsigned immediate where the least significant two bits are zero.
def uimm8_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<6, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
}

// A 8-bit unsigned immediate where the least significant three bits are zero.
def uimm8_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<5, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
}

// A 9-bit signed immediate where the least significant bit is zero.
def simm9_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<9, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<9>";
}

// A 9-bit unsigned immediate where the least significant three bits are zero.
def uimm9_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<6, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<9, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<9>";
}

// A 10-bit unsigned immediate where the least significant two bits are zero
// and the immediate can't be zero.
def uimm10_lsb00nonzero : Operand<XLenVT>,
                          ImmLeaf<XLenVT,
                          [{return isShiftedUInt<8, 2>(Imm) && (Imm != 0);}]> {
  let ParserMatchClass = UImmAsmOperand<10, "Lsb00NonZero">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<10>";
}

// A 10-bit signed immediate where the least significant four bits are zero.
def simm10_lsb0000 : Operand<XLenVT>,
                     ImmLeaf<XLenVT, [{return isShiftedInt<6, 4>(Imm);}]> {
  let ParserMatchClass = SImmAsmOperand<10, "Lsb0000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeSImmOperand<10>";
}

// A 12-bit signed immediate where the least significant bit is zero.
def simm12_lsb0 : Operand<OtherVT> {
  let ParserMatchClass = SImmAsmOperand<12, "Lsb0">;
  let EncoderMethod = "getImmOpValueAsr1";
  let DecoderMethod = "decodeSImmOperandAndLsl1<12>";
}

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class CStackLoad<bits<3> funct3, string OpcodeStr,
                 RegisterClass cls, DAGOperand opnd>
    : RVInst16CI<funct3, 0b10, (outs cls:$rd), (ins SP:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CStackStore<bits<3> funct3, string OpcodeStr,
                  RegisterClass cls, DAGOperand opnd>
    : RVInst16CSS<funct3, 0b10, (outs), (ins cls:$rs2, SP:$rs1, opnd:$imm),
                  OpcodeStr, "$rs2, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class CLoad_ri<bits<3> funct3, string OpcodeStr,
               RegisterClass cls, DAGOperand opnd>
    : RVInst16CL<funct3, 0b00, (outs cls:$rd), (ins GPRC:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CStore_rri<bits<3> funct3, string OpcodeStr,
                 RegisterClass cls, DAGOperand opnd>
    : RVInst16CS<funct3, 0b00, (outs), (ins cls:$rs2, GPRC:$rs1, opnd:$imm),
                 OpcodeStr, "$rs2, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Bcz<bits<3> funct3, string OpcodeStr, PatFrag CondOp,
          RegisterClass cls>
    : RVInst16CB<funct3, 0b01, (outs), (ins cls:$rs1, simm9_lsb0:$imm),
                 OpcodeStr, "$rs1, $imm"> {
  let isBranch = 1;
  let isTerminator = 1;
  let Inst{12} = imm{7};
  let Inst{11-10} = imm{3-2};
  let Inst{6-5} = imm{6-5};
  let Inst{4-3} = imm{1-0};
  let Inst{2} = imm{4};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Shift_right<bits<2> funct2, string OpcodeStr, RegisterClass cls,
                  Operand ImmOpnd>
    : RVInst16CB<0b100, 0b01, (outs cls:$rs1_wb), (ins cls:$rs1, ImmOpnd:$imm),
                 OpcodeStr, "$rs1, $imm"> {
  let Constraints = "$rs1 = $rs1_wb";
  let Inst{12} = imm{5};
  let Inst{11-10} = funct2;
  let Inst{6-2} = imm{4-0};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class CS_ALU<bits<2> funct2, string OpcodeStr, RegisterClass cls,
             bit RV64only>
    : RVInst16CS<0b100, 0b01, (outs cls:$rd_wb), (ins cls:$rd, cls:$rs2),
                 OpcodeStr, "$rd, $rs2"> {
  bits<3> rd;
  let Constraints = "$rd = $rd_wb";
  let Inst{12} = RV64only;
  let Inst{11-10} = 0b11;
  let Inst{9-7} = rd;
  let Inst{6-5} = funct2;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtC] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADDI4SPN : RVInst16CIW<0b000, 0b00, (outs GPRC:$rd),
                             (ins SP:$rs1, uimm10_lsb00nonzero:$imm),
                             "c.addi4spn", "$rd, $rs1, $imm"> {
  bits<5> rs1;
  let Inst{12-11} = imm{5-4};
  let Inst{10-7} = imm{9-6};
  let Inst{6} = imm{2};
  let Inst{5} = imm{3};
}

def C_FLD  : CLoad_ri<0b001, "c.fld", FPR64C, uimm8_lsb000>,
             Requires<[HasStdExtD]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def C_LW : CLoad_ri<0b010, "c.lw", GPRC, uimm7_lsb00> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

let DecoderNamespace = "RISCV32Only_" in
def C_FLW  : CLoad_ri<0b011, "c.flw", FPR32C, uimm7_lsb00>,
             Requires<[HasStdExtF, IsRV32]> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

def C_LD : CLoad_ri<0b011, "c.ld", GPRC, uimm8_lsb000>,
           Requires<[IsRV64]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def C_FSD  : CStore_rri<0b101, "c.fsd", FPR64C, uimm8_lsb000>,
             Requires<[HasStdExtD]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def C_SW : CStore_rri<0b110, "c.sw", GPRC, uimm7_lsb00> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

let DecoderNamespace = "RISCV32Only_" in
def C_FSW  : CStore_rri<0b111, "c.fsw", FPR32C, uimm7_lsb00>,
             Requires<[HasStdExtF, IsRV32]> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

def C_SD : CStore_rri<0b111, "c.sd", GPRC, uimm8_lsb000>,
           Requires<[IsRV64]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

let rd = 0, imm = 0, hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_NOP : RVInst16CI<0b000, 0b01, (outs), (ins), "c.nop", "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADDI : RVInst16CI<0b000, 0b01, (outs GPRNoX0:$rd_wb),
                        (ins GPRNoX0:$rd, simm6:$imm),
                        "c.addi", "$rd, $imm"> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = imm{4-0};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCall = 1,
    DecoderNamespace = "RISCV32Only_" in
def C_JAL : RVInst16CJ<0b001, 0b01, (outs), (ins simm12_lsb0:$offset),
                       "c.jal", "$offset">,
            Requires<[IsRV32]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADDIW : RVInst16CI<0b001, 0b01, (outs GPRNoX0:$rd_wb),
                         (ins GPRNoX0:$rd, simm6:$imm),
                         "c.addiw", "$rd, $imm">,
              Requires<[IsRV64]> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = imm{4-0};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_LI : RVInst16CI<0b010, 0b01, (outs GPRNoX0:$rd), (ins simm6:$imm),
                      "c.li", "$rd, $imm"> {
  let Inst{6-2} = imm{4-0};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADDI16SP : RVInst16CI<0b011, 0b01, (outs SP:$rd_wb),
                            (ins SP:$rd, simm10_lsb0000:$imm),
                            "c.addi16sp", "$rd, $imm"> {
  let Constraints = "$rd = $rd_wb";
  let Inst{12} = imm{9};
  let Inst{11-7} = 2;
  let Inst{6} = imm{4};
  let Inst{5} = imm{6};
  let Inst{4-3} = imm{8-7};
  let Inst{2} = imm{5};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_LUI : RVInst16CI<0b011, 0b01, (outs GPRNoX0X2:$rd),
                       (ins uimm6nonzero:$imm),
                       "c.lui", "$rd, $imm"> {
  let Inst{6-2} = imm{4-0};
}

def C_SRLI : Shift_right<0b00, "c.srli", GPRC, uimmlog2xlennonzero>;
def C_SRAI : Shift_right<0b01, "c.srai", GPRC, uimmlog2xlennonzero>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ANDI : RVInst16CB<0b100, 0b01, (outs GPRC:$rs1_wb), (ins GPRC:$rs1, simm6:$imm),
                        "c.andi", "$rs1, $imm"> {
  let Constraints = "$rs1 = $rs1_wb";
  let Inst{12} = imm{5};
  let Inst{11-10} = 0b10;
  let Inst{6-2} = imm{4-0};
}

def C_SUB  : CS_ALU<0b00, "c.sub", GPRC, 0>;
def C_XOR  : CS_ALU<0b01, "c.xor", GPRC, 0>;
def C_OR   : CS_ALU<0b10, "c.or" , GPRC, 0>;
def C_AND  : CS_ALU<0b11, "c.and", GPRC, 0>;

def C_SUBW : CS_ALU<0b00, "c.subw", GPRC, 1>, Requires<[IsRV64]>;
def C_ADDW : CS_ALU<0b01, "c.addw", GPRC, 1>, Requires<[IsRV64]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_J : RVInst16CJ<0b101, 0b01, (outs), (ins simm12_lsb0:$offset),
                     "c.j", "$offset"> {
  let isBranch = 1;
  let isTerminator=1;
  let isBarrier=1;
}

def C_BEQZ : Bcz<0b110, "c.beqz",  seteq, GPRC>;
def C_BNEZ : Bcz<0b111, "c.bnez",  setne, GPRC>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_SLLI : RVInst16CI<0b000, 0b10, (outs GPRNoX0:$rd_wb),
                        (ins GPRNoX0:$rd, uimmlog2xlennonzero:$imm),
                        "c.slli" ,"$rd, $imm"> {
  let Constraints = "$rd = $rd_wb";
  let Inst{6-2} = imm{4-0};
}

def C_FLDSP  : CStackLoad<0b001, "c.fldsp", FPR64, uimm9_lsb000>,
               Requires<[HasStdExtD]> {
  let Inst{6-5} = imm{4-3};
  let Inst{4-2} = imm{8-6};
}

def C_LWSP : CStackLoad<0b010, "c.lwsp", GPRNoX0, uimm8_lsb00> {
  let Inst{6-4} = imm{4-2};
  let Inst{3-2} = imm{7-6};
}

let DecoderNamespace = "RISCV32Only_" in
def C_FLWSP  : CStackLoad<0b011, "c.flwsp", FPR32, uimm8_lsb00>,
               Requires<[HasStdExtF, IsRV32]> {
  let Inst{6-4} = imm{4-2};
  let Inst{3-2} = imm{7-6};
}

def C_LDSP : CStackLoad<0b011, "c.ldsp", GPRNoX0, uimm9_lsb000>,
             Requires<[IsRV64]> {
  let Inst{6-5} = imm{4-3};
  let Inst{4-2} = imm{8-6};
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_JR : RVInst16CR<0b1000, 0b10, (outs), (ins GPRNoX0:$rs1),
                      "c.jr", "$rs1"> {
  let isBranch = 1;
  let isBarrier = 1;
  let isTerminator = 1;
  let isIndirectBranch = 1;
  let rs2 = 0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_MV : RVInst16CR<0b1000, 0b10, (outs GPRNoX0:$rs1), (ins GPRNoX0:$rs2),
                      "c.mv", "$rs1, $rs2">;

let rs1 = 0, rs2 = 0, hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_EBREAK : RVInst16CR<0b1001, 0b10, (outs), (ins), "c.ebreak", "">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    isCall=1, Defs=[X1], rs2 = 0 in
def C_JALR : RVInst16CR<0b1001, 0b10, (outs), (ins GPRNoX0:$rs1),
                        "c.jalr", "$rs1">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def C_ADD : RVInst16CR<0b1001, 0b10, (outs GPRNoX0:$rs1_wb),
                       (ins GPRNoX0:$rs1, GPRNoX0:$rs2),
                       "c.add", "$rs1, $rs2"> {
  let Constraints = "$rs1 = $rs1_wb";
}

def C_FSDSP  : CStackStore<0b101, "c.fsdsp", FPR64, uimm9_lsb000>,
               Requires<[HasStdExtD]> {
  let Inst{12-10} = imm{5-3};
  let Inst{9-7}   = imm{8-6};
}

def C_SWSP : CStackStore<0b110, "c.swsp", GPR, uimm8_lsb00> {
  let Inst{12-9} = imm{5-2};
  let Inst{8-7}  = imm{7-6};
}

let DecoderNamespace = "RISCV32Only_" in
def C_FSWSP  : CStackStore<0b111, "c.fswsp", FPR32, uimm8_lsb00>,
               Requires<[HasStdExtF, IsRV32]> {
  let Inst{12-9} = imm{5-2};
  let Inst{8-7}  = imm{7-6};
}

def C_SDSP : CStackStore<0b111, "c.sdsp", GPR, uimm9_lsb000>,
             Requires<[IsRV64]> {
  let Inst{12-10} = imm{5-3};
  let Inst{9-7}   = imm{8-6};
}

} // Predicates = [HasStdExtC]
