//===-- RISCVInstrInfoXVentana.td --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Ventana Micro Systems.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XVentanaCondOps
//===----------------------------------------------------------------------===//

let Predicates = [IsRV64, HasVendorXVentanaCondOps], hasSideEffects = 0,
  mayLoad = 0, mayStore = 0, isCodeGenOnly = 0, DecoderNamespace = "Ventana" in
class VTMaskedMove<bits<3> funct3, string opcodestr>
    : RVInstR<0b0000000, funct3, OPC_CUSTOM_3, (outs GPR:$rd),
              (ins GPR:$rs1, GPR:$rs2), opcodestr,
              "$rd, $rs1, $rs2"> {
}

def VT_MASKC : VTMaskedMove<0b110, "vt.maskc">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;

def VT_MASKCN : VTMaskedMove<0b111, "vt.maskcn">,
           Sched<[WriteIALU, ReadIALU, ReadIALU]>;

multiclass XVentanaCondops_pats<SDPatternOperator Op, RVInst MI> {
  def: Pat<(select i64:$rc, (Op i64:$rs1, i64:$rs2), i64:$rs1),
           (MI $rs1, (VT_MASKC $rs2, $rc))>;
  def: Pat<(select i64:$rc, i64:$rs1, (Op i64:$rs1, i64:$rs2)),
           (MI $rs1, (VT_MASKCN $rs2, $rc))>;
}

// Directly use MASKC/MASKCN in case of any of the operands being 0.
def: Pat<(select i64:$rc, i64:$rs1, (i64 0)),
         (VT_MASKC $rs1, $rc)>;
def: Pat<(select i64:$rc, (i64 0), i64:$rs1),
         (VT_MASKCN $rs1, $rc)>;

// Conditional operations patterns.
defm: XVentanaCondops_pats<add, ADD>;
defm: XVentanaCondops_pats<sub, SUB>;
defm: XVentanaCondops_pats<or, OR>;
defm: XVentanaCondops_pats<xor, XOR>;

// Conditional AND operation patterns.
def: Pat<(select i64:$rc, (and i64:$rs1, i64:$rs2), i64:$rs1),
         (OR (AND $rs1, $rs2), (VT_MASKCN $rs1, $rc))>;
def: Pat<(select i64:$rc, i64:$rs1, (and i64:$rs1, i64:$rs2)),
         (OR (AND $rs1, $rs2), (VT_MASKC $rs1, $rc))>;

// Basic select pattern that selects between 2 registers.
def: Pat<(select i64:$rc, i64:$rs1, i64:$rs2),
         (OR (VT_MASKC $rs1, $rc), (VT_MASKCN $rs2, $rc))>;
