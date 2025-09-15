//===-- RISCVInstrInfoZibi.td - 'Zibi' instructions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zibi' (branch with imm).
///
//===----------------------------------------------------------------------===//

// A 5-bit unsigned immediate representing 1-31 and -1. 00000 represents -1.
def imm5_zibi : RISCVOp<XLenVT>, ImmLeaf<XLenVT, [{
    return (Imm != 0 && isUInt<5>(Imm)) || Imm == -1;
}]> {
  let ParserMatchClass = ImmAsmOperand<"", 5, "Zibi">;
  let EncoderMethod = "getImmOpValueZibi";
  let DecoderMethod = "decodeImmZibiOperand";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return (Imm >= 1 && Imm <= 31) || Imm == -1;
  }];
  let OperandType = "OPERAND_IMM5_ZIBI";
}

class Branch_imm<bits<3> funct3, string opcodestr>
    : RVInstBIMM<funct3, OPC_BRANCH, (outs),
                 (ins GPR:$rs1, imm5_zibi:$cimm, bare_simm13_lsb0:$imm12),
                 opcodestr, "$rs1, $cimm, $imm12">,
      Sched<[WriteJmp, ReadJmp]> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

let Predicates = [HasStdExtZibi] in {
  def BEQI : Branch_imm<0b010, "beqi">;
  def BNEI : Branch_imm<0b011, "bnei">;
} // Predicates = [HasStdExtZibi]
