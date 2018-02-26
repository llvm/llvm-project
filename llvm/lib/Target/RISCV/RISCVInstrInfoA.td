//===-- RISCVInstrInfoA.td - RISC-V 'A' instructions -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'A', Atomic
// Instructions extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class LR_r<bit aq, bit rl, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<0b00010, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins GPR:$rs1),
                    opcodestr, "$rd, (${rs1})"> {
  let rs2 = 0;
}

multiclass LR_r_aq_rl<bits<3> funct3, string opcodestr> {
  def ""     : LR_r<0, 0, funct3, opcodestr>;
  def _AQ    : LR_r<1, 0, funct3, opcodestr # ".aq">;
  def _RL    : LR_r<0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : LR_r<1, 1, funct3, opcodestr # ".aqrl">;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 1 in
class AMO_rr<bits<5> funct5, bit aq, bit rl, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<funct5, aq, rl, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                    opcodestr, "$rd, $rs2, (${rs1})">;

multiclass AMO_rr_aq_rl<bits<5> funct5, bits<3> funct3, string opcodestr> {
  def ""     : AMO_rr<funct5, 0, 0, funct3, opcodestr>;
  def _AQ    : AMO_rr<funct5, 1, 0, funct3, opcodestr # ".aq">;
  def _RL    : AMO_rr<funct5, 0, 1, funct3, opcodestr # ".rl">;
  def _AQ_RL : AMO_rr<funct5, 1, 1, funct3, opcodestr # ".aqrl">;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtA] in {
defm LR_W       : LR_r_aq_rl<0b010, "lr.w">;
defm SC_W       : AMO_rr_aq_rl<0b00011, 0b010, "sc.w">;
defm AMOSWAP_W  : AMO_rr_aq_rl<0b00001, 0b010, "amoswap.w">;
defm AMOADD_W   : AMO_rr_aq_rl<0b00000, 0b010, "amoadd.w">;
defm AMOXOR_W   : AMO_rr_aq_rl<0b00100, 0b010, "amoxor.w">;
defm AMOAND_W   : AMO_rr_aq_rl<0b01100, 0b010, "amoand.w">;
defm AMOOR_W    : AMO_rr_aq_rl<0b01000, 0b010, "amoor.w">;
defm AMOMIN_W   : AMO_rr_aq_rl<0b10000, 0b010, "amomin.w">;
defm AMOMAX_W   : AMO_rr_aq_rl<0b10100, 0b010, "amomax.w">;
defm AMOMINU_W  : AMO_rr_aq_rl<0b11000, 0b010, "amominu.w">;
defm AMOMAXU_W  : AMO_rr_aq_rl<0b11100, 0b010, "amomaxu.w">;
} // Predicates = [HasStdExtA]

let Predicates = [HasStdExtA, IsRV64] in {
defm LR_D       : LR_r_aq_rl<0b011, "lr.d">;
defm SC_D       : AMO_rr_aq_rl<0b00011, 0b011, "sc.d">;
defm AMOSWAP_D  : AMO_rr_aq_rl<0b00001, 0b011, "amoswap.d">;
defm AMOADD_D   : AMO_rr_aq_rl<0b00000, 0b011, "amoadd.d">;
defm AMOXOR_D   : AMO_rr_aq_rl<0b00100, 0b011, "amoxor.d">;
defm AMOAND_D   : AMO_rr_aq_rl<0b01100, 0b011, "amoand.d">;
defm AMOOR_D    : AMO_rr_aq_rl<0b01000, 0b011, "amoor.d">;
defm AMOMIN_D   : AMO_rr_aq_rl<0b10000, 0b011, "amomin.d">;
defm AMOMAX_D   : AMO_rr_aq_rl<0b10100, 0b011, "amomax.d">;
defm AMOMINU_D  : AMO_rr_aq_rl<0b11000, 0b011, "amominu.d">;
defm AMOMAXU_D  : AMO_rr_aq_rl<0b11100, 0b011, "amomaxu.d">;
} // Predicates = [HasStedExtA, IsRV64]
