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
// This version is still experimental as the 'Zcmop' extension hasn't been
// ratified yet. It is based on v0.2 of the specification.
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class CMOPInst<bits<3> imm3, string opcodestr>
    : RVInst16CI<0b011, 0b01, (outs), (ins), opcodestr, ""> {
  let Inst{6-2} = 0;
  let Inst{7} = 1;
  let Inst{10-8} = imm3;
  let Inst{12-11} = 0;
}

// CMOP1, CMOP5 is used by Zicfiss.
let Predicates = [HasStdExtZcmop, NoHasStdExtZicfiss] in {
  def CMOP1 : CMOPInst<0, "cmop.1">, Sched<[]>;
  def CMOP5 : CMOPInst<2, "cmop.5">, Sched<[]>;
}

foreach n = [3, 7, 9, 11, 13, 15] in {
  let Predicates = [HasStdExtZcmop] in
  def CMOP # n : CMOPInst<!srl(n, 1), "cmop." # n>, Sched<[]>;
}
