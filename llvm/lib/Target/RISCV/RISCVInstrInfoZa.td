//===-- RISCVInstrInfoZa.td - RISC-V Atomic instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard atomic 'Za*'
// extensions:
//   - Zawrs (v1.0) : Wait-on-Reservation-Set.
//   - Zacas (v1.0-rc1) : Atomic Compare-and-Swap.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Zacas (Atomic Compare-and-Swap)
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZacas] in {
defm AMOCAS_W : AMO_rr_aq_rl<0b00101, 0b010, "amocas.w">;
defm AMOCAS_D : AMO_rr_aq_rl<0b00101, 0b011, "amocas.d">;
} // Predicates = [HasStdExtZacas]

let Predicates = [HasStdExtZacas, IsRV64] in {
defm AMOCAS_Q : AMO_rr_aq_rl<0b00101, 0b100, "amocas.q">;
} // Predicates = [HasStdExtZacas, IsRV64]

//===----------------------------------------------------------------------===//
// Zawrs (Wait-on-Reservation-Set)
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in
class WRSInst<bits<12> funct12, string opcodestr>
    : RVInstI<0b000, OPC_SYSTEM, (outs), (ins), opcodestr, ""> {
  let rs1 = 0;
  let rd = 0;
  let imm12 = funct12;
}

let Predicates = [HasStdExtZawrs] in {
def WRS_NTO : WRSInst<0b000000001101, "wrs.nto">, Sched<[]>;
def WRS_STO : WRSInst<0b000000011101, "wrs.sto">, Sched<[]>;
} // Predicates = [HasStdExtZawrs]
