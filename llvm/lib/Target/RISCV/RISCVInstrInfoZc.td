//===-- RISCVInstrInfoZc.td - RISC-V 'Zc*' instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from the 'Zc*' compressed
/// instruction extensions, version 1.0.1.
/// This version is still experimental as the 'Zc*' extensions haven't been
/// ratified yet.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def uimm2_lsb0 : Operand<XLenVT>,
                 ImmLeaf<XLenVT, [{return isShiftedUInt<1, 1>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<2, "Lsb0">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<2>";
  let OperandType = "OPERAND_UIMM2_LSB0";
  let OperandNamespace = "RISCVOp";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isShiftedUInt<1, 1>(Imm);
  }];
}

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVZcArith_r<bits<5> funct5, string OpcodeStr> :
  RVInst16CU<0b100111, funct5, 0b01, (outs GPRC:$rd_wb), (ins GPRC:$rd),
             OpcodeStr, "$rd"> {
  let Constraints = "$rd = $rd_wb";
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZcb, HasStdExtZba, IsRV64] in
def C_ZEXT_W  : RVZcArith_r<0b11100 , "c.zext.w">,
                Sched<[WriteIALU32, ReadIALU32, ReadIALU32]>;

let Predicates = [HasStdExtZcb, HasStdExtZbb] in {
def C_ZEXT_H  : RVZcArith_r<0b11010 , "c.zext.h">,
                Sched<[WriteIALU, ReadIALU]>;
def C_SEXT_B  : RVZcArith_r<0b11001 , "c.sext.b">,
                Sched<[WriteIALU, ReadIALU]>;
def C_SEXT_H  : RVZcArith_r<0b11011 , "c.sext.h">,
                Sched<[WriteIALU, ReadIALU]>;
}

let Predicates = [HasStdExtZcb] in
def C_ZEXT_B  : RVZcArith_r<0b11000 , "c.zext.b">,
                Sched<[WriteIALU, ReadIALU]>;

let Predicates = [HasStdExtZcb, HasStdExtMOrZmmul] in
def C_MUL     : CA_ALU<0b100111, 0b10, "c.mul", GPRC>,
                Sched<[WriteIMul, ReadIMul, ReadIMul]>;

let Predicates = [HasStdExtZcb] in {
def C_NOT : RVZcArith_r<0b11101 , "c.not">,
            Sched<[WriteIALU, ReadIALU]>;

def C_LBU : CLoad_ri<0b100, "c.lbu", GPRC, uimm2>,
            Sched<[WriteLDB, ReadMemBase]> {
  bits<2> imm;

  let Inst{12-10} = 0b000;
  let Inst{6-5} = imm{0,1};
}

def C_LHU : CLoad_ri<0b100, "c.lhu", GPRC, uimm2_lsb0>,
            Sched<[WriteLDH, ReadMemBase]> {
  bits<2> imm;

  let Inst{12-10} = 0b001;
  let Inst{6} = 0b0;
  let Inst{5} = imm{1};
}

def C_LH  : CLoad_ri<0b100, "c.lh", GPRC, uimm2_lsb0>,
            Sched<[WriteLDH, ReadMemBase]> {
  bits<2> imm;

  let Inst{12-10} = 0b001;
  let Inst{6} = 0b1;
  let Inst{5} = imm{1};
}

def C_SB : CStore_rri<0b100, "c.sb", GPRC, uimm2>,
           Sched<[WriteSTB, ReadStoreData, ReadMemBase]> {
  bits<2> imm;

  let Inst{12-10} = 0b010;
  let Inst{6-5} = imm{0,1};
}

def C_SH : CStore_rri<0b100, "c.sh", GPRC, uimm2_lsb0>, 
           Sched<[WriteSTH, ReadStoreData, ReadMemBase]> {
  bits<2> imm;

  let Inst{12-10} = 0b011;
  let Inst{6} = 0b1;
  let Inst{5} = imm{1};
}
}

let Predicates = [HasStdExtZcb, HasStdExtMOrZmmul] in{
def : CompressPat<(MUL GPRC:$rs1, GPRC:$rs1, GPRC:$rs2),
                  (C_MUL GPRC:$rs1, GPRC:$rs2)>;
let isCompressOnly = true in
def : CompressPat<(MUL GPRC:$rs1, GPRC:$rs2, GPRC:$rs1),
                  (C_MUL GPRC:$rs1, GPRC:$rs2)>;
} // Predicates = [HasStdExtZcb, HasStdExtMOrZmmul]

let Predicates = [HasStdExtZcb, HasStdExtZbb] in{
def : CompressPat<(SEXT_B GPRC:$rs1, GPRC:$rs1),
                  (C_SEXT_B GPRC:$rs1, GPRC:$rs1)>;
def : CompressPat<(SEXT_H GPRC:$rs1, GPRC:$rs1),
                  (C_SEXT_H GPRC:$rs1, GPRC:$rs1)>;
} // Predicates = [HasStdExtZcb, HasStdExtZbb]

let Predicates = [HasStdExtZcb, HasStdExtZbb] in{
def : CompressPat<(ZEXT_H_RV32 GPRC:$rs1, GPRC:$rs1),
                  (C_ZEXT_H GPRC:$rs1, GPRC:$rs1)>;
def : CompressPat<(ZEXT_H_RV64 GPRC:$rs1, GPRC:$rs1),
                  (C_ZEXT_H GPRC:$rs1, GPRC:$rs1)>;
} // Predicates = [HasStdExtZcb, HasStdExtZbb]

let Predicates = [HasStdExtZcb] in{
def : CompressPat<(ANDI GPRC:$rs1, GPRC:$rs1, 255),
                  (C_ZEXT_B GPRC:$rs1, GPRC:$rs1)>;
} // Predicates = [HasStdExtZcb]

let Predicates = [HasStdExtZcb, HasStdExtZba, IsRV64] in{
def : CompressPat<(ADD_UW GPRC:$rs1, GPRC:$rs1, X0),
                  (C_ZEXT_W GPRC:$rs1, GPRC:$rs1)>;
} // Predicates = [HasStdExtZcb, HasStdExtZba, IsRV64]

let Predicates = [HasStdExtZcb] in{
def : CompressPat<(XORI GPRC:$rs1, GPRC:$rs1, -1),
                  (C_NOT GPRC:$rs1, GPRC:$rs1)>;
}

let Predicates = [HasStdExtZcb] in{
def : CompressPat<(LBU GPRC:$rd, GPRCMem:$rs1, uimm2:$imm),
                  (C_LBU GPRC:$rd, GPRCMem:$rs1, uimm2:$imm)>;
def : CompressPat<(LHU GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_LHU GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
def : CompressPat<(LH GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_LH GPRC:$rd, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
def : CompressPat<(SB GPRC:$rs2, GPRCMem:$rs1, uimm2:$imm),
                  (C_SB GPRC:$rs2, GPRCMem:$rs1, uimm2:$imm)>;
def : CompressPat<(SH GPRC:$rs2, GPRCMem:$rs1, uimm2_lsb0:$imm),
                  (C_SH GPRC:$rs2, GPRCMem:$rs1, uimm2_lsb0:$imm)>;
}// Predicates = [HasStdExtZcb]


//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZcb] in {
def : InstAlias<"c.lbu $rd, (${rs1})",(C_LBU GPRC:$rd, GPRC:$rs1, 0)>;
def : InstAlias<"c.lhu $rd, (${rs1})",(C_LHU GPRC:$rd, GPRC:$rs1, 0)>;
def : InstAlias<"c.lh $rd, (${rs1})", (C_LH GPRC:$rd, GPRC:$rs1, 0)>;
def : InstAlias<"c.sb $rd, (${rs1})", (C_SB GPRC:$rd, GPRC:$rs1, 0)>;
def : InstAlias<"c.sh $rd, (${rs1})", (C_SH GPRC:$rd, GPRC:$rs1, 0)>;
}
