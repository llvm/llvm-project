//===-- RISCVInstrInfoZilsd.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zilsd',
// Load/Store pair instructions extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class PairLoad_ri<string opcodestr, DAGOperand RC>
    : RVInstI<0b011, OPC_LOAD, (outs RC:$rd), 
              (ins GPRMem:$rs1, simm12:$imm12),
              opcodestr, "${rd}, ${imm12}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class PairStore_rri<string opcodestr, DAGOperand RC>
    : RVInstS<0b011, OPC_STORE, (outs),
              (ins RC:$rs2, GPRMem:$rs1, simm12:$imm12),
              opcodestr, "${rs2}, ${imm12}(${rs1})">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilsd, IsRV32], DecoderNamespace = "RV32Only" in {
def LD_RV32 : PairLoad_ri<"ld", GPRPairRV32>, Sched<[WriteLDD, ReadMemBase]>;
def SD_RV32 : PairStore_rri<"sd", GPRPairRV32>, Sched<[WriteSTD, ReadStoreData,
                            ReadMemBase]>;
} // Predicates = [HasStdExtZilsd, IsRV32], DecoderNamespace = "RV32Only"

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilsd, IsRV32] in {
def : InstAlias<"ld $rd, (${rs1})", (LD_RV32 GPRPairRV32:$rd, GPR:$rs1, 0), 0>;
def : InstAlias<"sd $rs2, (${rs1})", (SD_RV32 GPRPairRV32:$rs2, GPR:$rs1, 0), 0>;
}
