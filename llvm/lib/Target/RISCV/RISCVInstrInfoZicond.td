//===-- RISCVInstrInfoZicond.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard Integer
// Conditional operations extension (Zicond).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

// Branchless select operations, matching the semantics of the instructions
// defined in Zicond or XVentanaCondOps
def riscv_czero_eqz : RVSDNode<"CZERO_EQZ", SDTIntBinOp>; // vt.maskc for XVentanaCondOps.
def riscv_czero_nez : RVSDNode<"CZERO_NEZ", SDTIntBinOp>; // vt.maskcn for XVentanaCondOps.

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicond] in {
def CZERO_EQZ : ALU_rr<0b0000111, 0b101, "czero.eqz">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def CZERO_NEZ : ALU_rr<0b0000111, 0b111, "czero.nez">,
                Sched<[WriteIALU, ReadIALU, ReadIALU]>;
} // Predicates = [HasStdExtZicond]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

// Zero (eqz) is instantiated with CZERO_EQZ or a compatible custom
// instruction, and nez with CZERO_NEZ or a compatible custom instruction.
multiclass CondOpsPat<Instruction eqz, Instruction nez> {
  def : Pat<(XLenVT (riscv_czero_eqz GPR:$rs1, GPR:$rc)),
            (eqz GPR:$rs1, GPR:$rc)>;
  def : Pat<(XLenVT (riscv_czero_nez GPR:$rs1, GPR:$rc)),
            (nez GPR:$rs1, GPR:$rc)>;

  def : Pat<(XLenVT (riscv_czero_eqz GPR:$rs1, (riscv_setne (XLenVT GPR:$rc)))),
            (eqz GPR:$rs1, GPR:$rc)>;
  def : Pat<(XLenVT (riscv_czero_eqz GPR:$rs1, (riscv_seteq (XLenVT GPR:$rc)))),
            (nez GPR:$rs1, GPR:$rc)>;
  def : Pat<(XLenVT (riscv_czero_nez GPR:$rs1, (riscv_setne (XLenVT GPR:$rc)))),
            (nez GPR:$rs1, GPR:$rc)>;
  def : Pat<(XLenVT (riscv_czero_nez GPR:$rs1, (riscv_seteq (XLenVT GPR:$rc)))),
            (eqz GPR:$rs1, GPR:$rc)>;

  // (and (setcc a, b, ne), c) -> (eqz c, (xor a, b))
  // (and (setcc a, b, eq), c) -> (nez c, (xor a, b))
  // where c is known to be 0 or 1.
  def : Pat<(XLenVT (and (riscv_setne_oneuse (XLenVT GPR:$rs1)), zero_or_one:$rc)),
            (eqz GPR:$rc, GPR:$rs1)>;
  def : Pat<(XLenVT (and (riscv_seteq_oneuse (XLenVT GPR:$rs1)), zero_or_one:$rc)),
            (nez GPR:$rc, GPR:$rs1)>;
}

let Predicates = [HasStdExtZicond] in
defm : CondOpsPat<CZERO_EQZ, CZERO_NEZ>;
