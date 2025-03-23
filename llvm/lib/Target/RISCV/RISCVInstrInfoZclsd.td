//===-- RISCVInstrInfoZclsd.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zclsd',
// Compressed Load/Store pair instructions extension.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

def GPRPairNoX0RV32Operand : AsmOperandClass {
  let Name = "GPRPairNoX0RV32";
  let ParserMethod = "parseGPRPair<false>";
  let PredicateMethod = "isGPRPairNoX0";
  let RenderMethod = "addRegOperands";
}

def GPRPairNoX0RV32 : RegisterOperand<GPRPairNoX0> {
  let ParserMatchClass = GPRPairNoX0RV32Operand;
}

def GPRPairCRV32Operand : AsmOperandClass {
  let Name = "GPRPairCRV32";
  let ParserMethod = "parseGPRPair<false>";
  let PredicateMethod = "isGPRPairC";
  let RenderMethod = "addRegOperands";
}

def GPRPairCRV32 : RegisterOperand<GPRPairC> {
  let ParserMatchClass = GPRPairCRV32Operand;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class PairCStackLoad<bits<3> funct3, string OpcodeStr,
                     DAGOperand RC, DAGOperand opnd>
    : RVInst16CI<funct3, 0b10, (outs RC:$rd), (ins SPMem:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class PairCStackStore<bits<3> funct3, string OpcodeStr,
                      DAGOperand RC, DAGOperand opnd>
    : RVInst16CSS<funct3, 0b10, (outs), (ins RC:$rs2, SPMem:$rs1, opnd:$imm),
                  OpcodeStr, "$rs2, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class PairCLoad_ri<bits<3> funct3, string OpcodeStr,
                  DAGOperand RC, DAGOperand opnd>
    : RVInst16CL<funct3, 0b00, (outs RC:$rd), (ins GPRCMem:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class PairCStore_rri<bits<3> funct3, string OpcodeStr,
                     DAGOperand RC, DAGOperand opnd>
    : RVInst16CS<funct3, 0b00, (outs), (ins RC:$rs2,GPRCMem:$rs1, opnd:$imm),
                 OpcodeStr, "$rs2, ${imm}(${rs1})">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZclsd, IsRV32], DecoderNamespace = "ZcOverlap" in {
def C_LDSP_RV32 : PairCStackLoad<0b011, "c.ldsp", GPRPairNoX0RV32, uimm9_lsb000>,
                  Sched<[WriteLDD, ReadMemBase]> {
  let Inst{4-2} = imm{8-6};
}

def C_SDSP_RV32 : PairCStackStore<0b111, "c.sdsp", GPRPairRV32, uimm9_lsb000>,
                  Sched<[WriteSTD, ReadStoreData, ReadMemBase]> {
  let Inst{9-7} = imm{8-6};
}

def C_LD_RV32 : PairCLoad_ri<0b011, "c.ld", GPRPairCRV32, uimm8_lsb000>,
                Sched<[WriteLDD, ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def C_SD_RV32 : PairCStore_rri<0b111, "c.sd", GPRPairCRV32, uimm8_lsb000>,
                Sched<[WriteSTD, ReadStoreData, ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}
}// Predicates = [HasStdExtZclsd, IsRV32], DecoderNamespace = "ZcOverlap"

//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZclsd, IsRV32] in {
def : InstAlias<"c.ld $rd, (${rs1})",
                (C_LD_RV32 GPRPairCRV32:$rd, GPRCMem:$rs1, 0), 0>;
def : InstAlias<"c.sd $rs2, (${rs1})",
                (C_SD_RV32 GPRPairCRV32:$rs2, GPRCMem:$rs1, 0), 0>;
def : InstAlias<"c.ldsp $rd, (${rs1})",
                (C_LDSP_RV32 GPRPairNoX0RV32:$rd, SPMem:$rs1, 0), 0>;
def : InstAlias<"c.sdsp $rs2, (${rs1})",
                (C_SDSP_RV32 GPRPairRV32:$rs2, SPMem:$rs1, 0), 0>;
}

//===----------------------------------------------------------------------===//
// Compress Instruction tablegen backend.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZclsd, IsRV32] in {
def : CompressPat<(LD_RV32 GPRPairNoX0RV32:$rd, SPMem:$rs1, uimm9_lsb000:$imm),
                  (C_LDSP_RV32 GPRPairNoX0RV32:$rd, SPMem:$rs1, uimm9_lsb000:$imm)>;
def : CompressPat<(SD_RV32 GPRPairRV32:$rs2, SPMem:$rs1, uimm9_lsb000:$imm),
                  (C_SDSP_RV32 GPRPairRV32:$rs2, SPMem:$rs1, uimm9_lsb000:$imm)>;
def : CompressPat<(LD_RV32 GPRPairCRV32:$rd, GPRCMem:$rs1, uimm8_lsb000:$imm),
                  (C_LD_RV32 GPRPairCRV32:$rd, GPRCMem:$rs1, uimm8_lsb000:$imm)>;
def : CompressPat<(SD_RV32 GPRPairCRV32:$rs2, GPRCMem:$rs1, uimm8_lsb000:$imm),
                  (C_SD_RV32 GPRPairCRV32:$rs2, GPRCMem:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtZclsd, IsRV32]
