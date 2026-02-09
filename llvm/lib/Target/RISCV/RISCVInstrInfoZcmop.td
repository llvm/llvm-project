//===-- RISCVInstrInfoZcmop.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Compressed
// May-Be-Operations Extension (Zcmop).
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class CMOPInst<bits<3> imm3, string opcodestr>
    : RVInst16CI<0b011, OPC_C1, (outs), (ins), opcodestr, ""> {
  let Inst{6-2} = 0;
  let Inst{7} = 1;
  let Inst{10-8} = imm3;
  let Inst{12-11} = 0;
}

// c.mop.1 and c.mop.5 are aliases for c.sspush and c.sspopchk respectively,
// defined in RISCVInstrInfoZicfiss.td.
foreach n = [3, 7, 9, 11, 13, 15] in {
  let Predicates = [HasStdExtZcmop] in
  def C_MOP_ # n : CMOPInst<!srl(n, 1), "c.mop." # n>, Sched<[]>;
}
