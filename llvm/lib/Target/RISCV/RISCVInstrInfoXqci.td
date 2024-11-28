//===---------------- RISCVInstrInfoXQci.td ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by QUALCOMM.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Formats
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXqcicsr, IsRV32], DecoderNamespace = "Xqcicsr" in {
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def QC_CSRRWR : RVInstR<0b1000110, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                          (ins GPR:$rs1, GPRNoX0:$rs2), "qc.csrrwr",
                          "$rd, $rs1, $rs2">;

  def QC_CSRRWRI : RVInstR<0b1000111, 0b000, OPC_SYSTEM, (outs GPR:$rd),
                           (ins uimm5:$rs1, GPRNoX0:$rs2), "qc.csrrwri",
                           "$rd, $rs1, $rs2">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0
} // Predicates = [HasVendorXqcicsr, IsRV32], DecoderNamespace = "Xqcicsr"
