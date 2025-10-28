//===-- RISCVInstrInfoZalasr.td  ---------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the Zalasr (Load-Acquire
// and Store-Release) extension
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class LAQ_r<bit aq, bit rl, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<0b00110, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins GPRMemZeroOffset:$rs1),
                    opcodestr, "$rd, $rs1"> {
  let rs2 = 0;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class SRL_r<bit aq, bit rl, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<0b00111, aq, rl, funct3, OPC_AMO,
                    (outs ), (ins GPRMemZeroOffset:$rs1, GPR:$rs2),
                    opcodestr, "$rs2, $rs1"> {
  let rd = 0;
}
multiclass LAQ_r_aq_rl<bits<3> funct3, string opcodestr> {
  def _AQ    : LAQ_r<1, 0, funct3, opcodestr # ".aq">;
  def _AQ_RL : LAQ_r<1, 1, funct3, opcodestr # ".aqrl">;
}

multiclass SRL_r_aq_rl<bits<3> funct3, string opcodestr> {
  def _RL    : SRL_r<0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : SRL_r<1, 1, funct3, opcodestr # ".aqrl">;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZalasr] in {
defm LB : LAQ_r_aq_rl<0b000, "lb">;
defm LH : LAQ_r_aq_rl<0b001, "lh">;
defm LW : LAQ_r_aq_rl<0b010, "lw">;
defm SB : SRL_r_aq_rl<0b000, "sb">;
defm SH : SRL_r_aq_rl<0b001, "sh">;
defm SW : SRL_r_aq_rl<0b010, "sw">;
} // Predicates = [HasStdExtZalasr]

let Predicates = [HasStdExtZalasr, IsRV64] in {
defm LD : LAQ_r_aq_rl<0b011, "ld">;
defm SD : SRL_r_aq_rl<0b011, "sd">;
} // Predicates = [HasStdExtZalasr, IsRV64]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

class PatLAQ<SDPatternOperator OpNode, RVInst Inst, ValueType vt = XLenVT>
    : Pat<(vt (OpNode (XLenVT GPRMemZeroOffset:$rs1))),
          (Inst GPRMemZeroOffset:$rs1)>;

// n.b. this switches order of arguments
//  to deal with the fact that SRL has addr, data
//  while atomic_store has data, addr
class PatSRL<SDPatternOperator OpNode, RVInst Inst, ValueType vt = XLenVT>
    : Pat<(OpNode (vt GPR:$rs2), (XLenVT GPRMemZeroOffset:$rs1)),
          (Inst GPRMemZeroOffset:$rs1, GPR:$rs2)>;


let Predicates = [HasStdExtZalasr] in {
  // the sequentially consistent loads use
  //  .aq instead of .aqrl to match the psABI/A.7
  def : PatLAQ<acquiring_load<atomic_load_asext_8>, LB_AQ>;
  def : PatLAQ<seq_cst_load<atomic_load_asext_8>, LB_AQ>;

  def : PatLAQ<acquiring_load<atomic_load_asext_16>, LH_AQ>;
  def : PatLAQ<seq_cst_load<atomic_load_asext_16>, LH_AQ>;

  // the sequentially consistent stores use
  //  .rl instead of .aqrl to match the psABI/A.7
  def : PatSRL<releasing_store<atomic_store_8>, SB_RL>;
  def : PatSRL<seq_cst_store<atomic_store_8>, SB_RL>;

  def : PatSRL<releasing_store<atomic_store_16>, SH_RL>;
  def : PatSRL<seq_cst_store<atomic_store_16>, SH_RL>;

  def : PatSRL<releasing_store<atomic_store_32>, SW_RL>;
  def : PatSRL<seq_cst_store<atomic_store_32>, SW_RL>;
} // Predicates = [HasStdExtZalasr]

let Predicates = [HasStdExtZalasr, IsRV32] in {
  def : PatLAQ<acquiring_load<atomic_load_nonext_32>, LW_AQ>;
  def : PatLAQ<seq_cst_load<atomic_load_nonext_32>, LW_AQ>;
} // Predicates = [HasStdExtZalasr, IsRV32]

let Predicates = [HasStdExtZalasr, IsRV64] in {
  def : PatLAQ<acquiring_load<atomic_load_asext_32>, LW_AQ, i64>;
  def : PatLAQ<seq_cst_load<atomic_load_asext_32>, LW_AQ, i64>;

  def : PatLAQ<acquiring_load<atomic_load_nonext_64>, LD_AQ, i64>;
  def : PatLAQ<seq_cst_load<atomic_load_nonext_64>, LD_AQ, i64>;

  def : PatSRL<releasing_store<atomic_store_64>, SD_RL, i64>;
  def : PatSRL<seq_cst_store<atomic_store_64>, SD_RL, i64>;
} // Predicates = [HasStdExtZalasr, IsRV64]
