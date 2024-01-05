//===-- RISCVInstrInfoZimop.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard
// May-Be-Operations Extension (Zimop).
// This version is still experimental as the 'Zimop' extension hasn't been
// ratified yet. It is based on v0.1 of the specification.
//
//===----------------------------------------------------------------------===//

class RVInstIMopr<bits<7> imm7, bits<5> imm5, bits<3> funct3, RISCVOpcode opcode,
                   dag outs, dag ins, string opcodestr, string argstr>
    : RVInstIBase<funct3, opcode, outs, ins, opcodestr, argstr> {
  let Inst{31} = imm7{6};
  let Inst{30} = imm5{4};
  let Inst{29-28} = imm7{5-4};
  let Inst{27-26} = imm5{3-2};
  let Inst{25-22} = imm7{3-0};
  let Inst{21-20} = imm5{1-0};
}

class RVInstRMoprr<bits<4> imm4, bits<3> imm3, bits<3> funct3, RISCVOpcode opcode,
                   dag outs, dag ins, string opcodestr, string argstr>
    : RVInstRBase<funct3, opcode, outs, ins, opcodestr, argstr> {
  let Inst{31} = imm4{3};
  let Inst{30} = imm3{2};
  let Inst{29-28} = imm4{2-1};
  let Inst{27-26} = imm3{1-0};
  let Inst{25} = imm4{0};
}

foreach i = 0...31 in {
  defvar riscvisd_moprx = "RISCVISD::MOPR"#i;
  def riscv_mopr#i : SDNode<riscvisd_moprx,  SDTIntUnaryOp>;
}
foreach i = 0...7 in {
  defvar riscvisd_moprrx = "RISCVISD::MOPRR"#i;
  def riscv_moprr#i : SDNode<riscvisd_moprrx,  SDTIntBinOp>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVMopr<bits<7> imm7, bits<5> imm5, bits<3> funct3,
             RISCVOpcode opcode, string opcodestr>
    : RVInstIMopr<imm7, imm5, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1),
                   opcodestr, "$rd, $rs1">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVMoprr<bits<4> imm4, bits<3> imm3, bits<3> funct3,
             RISCVOpcode opcode, string opcodestr>
    : RVInstRMoprr<imm4, imm3, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
                   opcodestr, "$rd, $rs1, $rs2">;

foreach i = 0...31 in {
  let Predicates = [HasStdExtZimop] in
  def MOPR#i : RVMopr<0b1000111, i, 0b100, OPC_SYSTEM, "mop.r."#i>,
               Sched<[]>;
}

foreach i = 0...7 in {
  let Predicates = [HasStdExtZimop] in
  def MOPRR#i : RVMoprr<0b1001, i, 0b100, OPC_SYSTEM, "mop.rr."#i>,
                Sched<[]>;
}

// Zimop instructions
foreach i = 0...31 in {
    defvar moprx = !cast<Instruction>("MOPR"#i);
    defvar riscv_moprx = !cast<SDNode>("riscv_mopr"#i);
    let Predicates = [HasStdExtZimop] in {
    def : Pat<(XLenVT (riscv_moprx (XLenVT GPR:$rs1))),
              (moprx GPR:$rs1)>;
    } // Predicates = [HasStdExtZimop]
}

foreach i = 0...7 in {
    defvar moprrx = !cast<Instruction>("MOPRR"#i);
    defvar riscv_moprrx = !cast<SDNode>("riscv_moprr"#i);
    let Predicates = [HasStdExtZimop] in {
    def : Pat<(XLenVT (riscv_moprrx (XLenVT GPR:$rs1), (XLenVT GPR:$rs2))),
              (moprrx GPR:$rs1, GPR:$rs2)>;
    } // Predicates = [HasStdExtZimop]
}
