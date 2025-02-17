//===-- RISCVInstrInfoZibimm.td - 'Zibimm' instructions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zibimm' (branch with imm).
///
//===----------------------------------------------------------------------===//
// A 5-bit unsigned immediate representing 1-31 and -1. 00000 represents -1.
def uimm5_zibimm : RISCVOp, ImmLeaf<XLenVT, [{
    return (Imm != 0 && isUInt<5>(Imm)) || Imm == -1;
}]> {
  let ParserMatchClass = UImmAsmOperand<5, "Zibimm">;
  let EncoderMethod = "getImmOpValueZibimm";
  let DecoderMethod = "decodeUImmZibimmOperand<5>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return (Imm >= 1 && Imm <= 31) || Imm == -1;
  }];
  let OperandType = "OPERAND_UIMM5_ZIBIMM";
}
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Branch_imm<bits<3> funct3, string opcodestr>
    : RVInstBIMM<funct3, OPC_BRANCH, (outs),
              (ins GPR:$rs1, uimm5_zibimm:$cimm, simm13_lsb0:$imm12),
              opcodestr, "$rs1, $cimm, $imm12">,
      Sched<[WriteJmp, ReadJmp]> {
  let isBranch = 1;
  let isTerminator = 1;
}
let Predicates = [HasStdExtZibimm] in {
def BEQI  : Branch_imm<0b010, "beqi">;
def BNEI  : Branch_imm<0b011, "bnei">;
} // Predicates = [HasStdExtZibimm]

let Predicates = [HasStdExtZibimm] in {
multiclass BccImmPat<CondCode Cond, Branch_imm Inst> {
  def : Pat<(riscv_brcc (XLenVT GPR:$rs1), uimm5_zibimm:$cimm, Cond, bb:$imm12),
            (Inst GPR:$rs1, uimm5_zibimm:$cimm, simm13_lsb0:$imm12)>;
}
defm : BccImmPat<SETEQ, BEQI>;
defm : BccImmPat<SETNE, BNEI>;
}// Predicates = [HasStdExtZibimm]