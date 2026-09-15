//===-- RISCVInstrInfoSmcsps.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class CSPSInst<bits<7> funct7, bits<5> funct5, string opcodestr>
    : RVInstR<funct7, 0b000, OPC_SYSTEM, (outs SP:$rd), (ins SP:$rs1),
              opcodestr, "$rd, $rs1">, Sched<[]> {
  let rs2 = funct5;
}

let Predicates = [HasStdExtSmcsps] in {
def MCSPSPUSH : CSPSInst<0b0011000, 0b01001, "mcspspush">;
def MCSPSPOP  : CSPSInst<0b0011000, 0b01100, "mcspspop">;
}

let Predicates = [HasStdExtSscsps] in {
def SCSPSPUSH : CSPSInst<0b0001000, 0b01001, "scspspush">;
def SCSPSPOP  : CSPSInst<0b0001000, 0b01100, "scspspop">;
}
