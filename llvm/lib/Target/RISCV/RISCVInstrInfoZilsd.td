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
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//

def SDT_RISCV_LD_RV32
    : SDTypeProfile<2, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisPtrTy<2>]>;
def SDT_RISCV_SD_RV32
    : SDTypeProfile<0, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisPtrTy<2>]>;

def riscv_ld_rv32 : RVSDNode<"LD_RV32", SDT_RISCV_LD_RV32,
                             [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_st_rv32 : RVSDNode<"SD_RV32", SDT_RISCV_SD_RV32,
                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilsd, IsRV32], DecoderNamespace = "RV32Only" in {
def LD_RV32 : Load_ri<0b011, "ld", GPRPairRV32>, Sched<[WriteLDD, ReadMemBase]>;
def SD_RV32 : Store_rri<0b011, "sd", GPRPairRV32>,
              Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;
} // Predicates = [HasStdExtZilsd, IsRV32], DecoderNamespace = "RV32Only"

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilsd, IsRV32] in {
def PseudoLD_RV32 : PseudoLoad<"ld", GPRPairRV32>;
def PseudoSD_RV32 : PseudoStore<"sd", GPRPairRV32>;

// Pseudo instructions for load/store optimization with 2 separate registers
def PseudoLD_RV32_OPT :
    Pseudo<(outs GPR:$rd1, GPR:$rd2),
           (ins GPR:$rs1, simm12_lo:$imm12), [], "", ""> {
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
}

def PseudoSD_RV32_OPT :
    Pseudo<(outs),
           (ins GPR:$rs1, GPR:$rs2, GPR:$rs3, simm12_lo:$imm12), [], "", ""> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
}

def : InstAlias<"ld $rd, (${rs1})", (LD_RV32 GPRPairRV32:$rd, GPR:$rs1, 0), 0>;
def : InstAlias<"sd $rs2, (${rs1})", (SD_RV32 GPRPairRV32:$rs2, GPR:$rs1, 0), 0>;
}
