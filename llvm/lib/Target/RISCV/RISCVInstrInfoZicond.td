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
// This version is still experimental as the 'Zicond' extension hasn't been
// ratified yet. It is based on v1.0-rc1 of the specification.
//
//===----------------------------------------------------------------------===//

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

let Predicates = [HasStdExtZicond] in {
// Directly use CZERO_EQZ/CZERO_NEZ in case of any of the operands being 0.
def : Pat<(select GPR:$rc, GPR:$rs1, 0),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;
def : Pat<(select GPR:$rc, 0, GPR:$rs1),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;

def : Pat<(select (riscv_setne GPR:$rc), GPR:$rs1, 0),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;
def : Pat<(select (riscv_seteq GPR:$rc), GPR:$rs1, 0),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;
def : Pat<(select (riscv_setne GPR:$rc), 0, GPR:$rs1),
          (CZERO_NEZ GPR:$rs1, GPR:$rc)>;
def : Pat<(select (riscv_seteq GPR:$rc), 0, GPR:$rs1),
          (CZERO_EQZ GPR:$rs1, GPR:$rc)>;

// Conditional AND operation patterns.
def : Pat<(select GPR:$rc, (and GPR:$rs1, GPR:$rs2), GPR:$rs1),
          (OR (AND $rs1, $rs2), (CZERO_NEZ $rs1, $rc))>;
def : Pat<(select GPR:$rc, GPR:$rs1, (and GPR:$rs1, GPR:$rs2)),
          (OR (AND $rs1, $rs2), (CZERO_EQZ $rs1, $rc))>;

// Basic select pattern that selects between 2 registers.
def : Pat<(select GPR:$rc, GPR:$rs1, GPR:$rs2),
          (OR (CZERO_EQZ $rs1, $rc), (CZERO_NEZ $rs2, $rc))>;

def : Pat<(select (riscv_setne GPR:$rc), GPR:$rs1, GPR:$rs2),
          (OR (CZERO_EQZ GPR:$rs1, GPR:$rc), (CZERO_NEZ GPR:$rs2, GPR:$rc))>;
def : Pat<(select (riscv_seteq GPR:$rc), GPR:$rs2, GPR:$rs1),
          (OR (CZERO_EQZ GPR:$rs1, GPR:$rc), (CZERO_NEZ GPR:$rs2, GPR:$rc))>;
} // Predicates = [HasStdExtZicond]
