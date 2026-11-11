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
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZclsd, IsRV32], DecoderNamespace = "ZcOverlap" in {
def C_LDSP_RV32 : CStackLoad<0b011, "c.ldsp", GPRPairNoX0, uimm9_lsb000_optional>,
                  Sched<[WriteLDD, ReadMemBase]> {
  let Inst{4-2} = imm{8-6};
}

def C_SDSP_RV32 : CStackStore<0b111, "c.sdsp", GPRPair, uimm9_lsb000_optional>,
                  Sched<[WriteSTD, ReadStoreData, ReadMemBase]> {
  let Inst{9-7} = imm{8-6};
}

def C_LD_RV32 : CLoad_ri<0b011, "c.ld", GPRPairC, uimm8_lsb000_optional>,
                Sched<[WriteLDD, ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def C_SD_RV32 : CStore_rri<0b111, "c.sd", GPRPairC, uimm8_lsb000_optional>,
                Sched<[WriteSTD, ReadStoreData, ReadMemBase]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}
}// Predicates = [HasStdExtZclsd, IsRV32], DecoderNamespace = "ZcOverlap"

//===----------------------------------------------------------------------===//
// Compress Instruction tablegen backend.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZclsd, IsRV32] in {
def : CompressPat<(LD_RV32 GPRPairNoX0:$rd, SPMem:$rs1, uimm9_lsb000:$imm),
                  (C_LDSP_RV32 GPRPairNoX0:$rd, SPMem:$rs1, uimm9_lsb000:$imm)>;
def : CompressPat<(SD_RV32 GPRPair:$rs2, SPMem:$rs1, uimm9_lsb000:$imm),
                  (C_SDSP_RV32 GPRPair:$rs2, SPMem:$rs1, uimm9_lsb000:$imm)>;
def : CompressPat<(LD_RV32 GPRPairC:$rd, BasePtrC:$rs1, uimm8_lsb000:$imm),
                  (C_LD_RV32 GPRPairC:$rd, BasePtrC:$rs1, uimm8_lsb000:$imm)>;
def : CompressPat<(SD_RV32 GPRPairC:$rs2, BasePtrC:$rs1, uimm8_lsb000:$imm),
                  (C_SD_RV32 GPRPairC:$rs2, BasePtrC:$rs1, uimm8_lsb000:$imm)>;
} // Predicates = [HasStdExtZclsd, IsRV32]
