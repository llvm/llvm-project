//===-- RISCVInstrInfoXMips.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by MIPS.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// MIPS extensions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorMIPSCMove], hasSideEffects = 0, mayLoad = 0, mayStore = 0, DecoderNamespace = "Xmipscomve" in {
def CCMOV : RVInstR4<0b11, 0b011, OPC_CUSTOM_0, (outs GPR:$rd),
                    (ins GPR:$rs1, GPR:$rs2, GPR:$rs3),
                    "ccmov", "$rd, $rs2, $rs1, $rs3">,
           Sched<[]>;
}

let Predicates = [HasVendorMIPSCMove] in {
def : Pat<(select (XLenVT (setne (XLenVT GPR:$rs2), (XLenVT 0))),
                  (XLenVT GPR:$rs1), (XLenVT GPR:$rs3)),
          (CCMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (setne (XLenVT GPR:$x), (XLenVT simm12_plus1:$y))),
                  (XLenVT GPR:$rs1), (XLenVT GPR:$rs3)),
          (CCMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (setne (XLenVT GPR:$x), (XLenVT GPR:$y))),
                  (XLenVT GPR:$rs1), (XLenVT GPR:$rs3)),
          (CCMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq (XLenVT GPR:$rs2), (XLenVT 0))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq (XLenVT GPR:$x), (XLenVT simm12_plus1:$y))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, (ADDI GPR:$x, (NegImm simm12_plus1:$y)), GPR:$rs3)>;
def : Pat<(select (XLenVT (seteq (XLenVT GPR:$x), (XLenVT GPR:$y))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, (XOR GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setuge (XLenVT GPR:$x), (XLenVT GPR:$y))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, (SLTU GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setge (XLenVT GPR:$x), (XLenVT GPR:$y))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT (setle (XLenVT GPR:$y), (XLenVT GPR:$x))),
                  (XLenVT GPR:$rs3), (XLenVT GPR:$rs1)),
          (CCMOV GPR:$rs1, (SLT GPR:$x, GPR:$y), GPR:$rs3)>;
def : Pat<(select (XLenVT GPR:$rs2), (XLenVT GPR:$rs1), (XLenVT GPR:$rs3)),
          (CCMOV GPR:$rs1, GPR:$rs2, GPR:$rs3)>;
}

let Predicates = [HasVendorMIPSLoadStorePairs], hasSideEffects = 0, DecoderNamespace = "Xmipslsp" in {
def LWP : LWPFormat<(outs GPR:$rd1, GPR:$rd2), (ins GPR:$rs1, uimm7_lsb00:$imm7),
                    "lwp", "$rd1, $rd2, ${imm7}(${rs1})">,
                    Sched<[WriteLDW, WriteLDW, ReadMemBase]> {
let mayLoad = 1;
let mayStore = 0;
}
def LDP : LDPFormat<(outs GPR:$rd1, GPR:$rd2), (ins GPR:$rs1, uimm7_lsb000:$imm7),
                    "ldp", "$rd1, $rd2, ${imm7}(${rs1})">,
                    Sched<[WriteLDD, WriteLDD, ReadMemBase]> {
let mayLoad = 1;
let mayStore = 0;
}
def SWP : SWPFormat<(outs), (ins GPR:$rs2, GPR:$rs3, GPR:$rs1, uimm7_lsb00:$imm7),
                    "swp", "$rs2, $rs3, ${imm7}(${rs1})">,
                    Sched<[WriteSTW, ReadStoreData, ReadStoreData, ReadMemBase]> {
let mayLoad = 0;
let mayStore = 1;
}
def SDP : SDPFormat<(outs), (ins GPR:$rs2, GPR:$rs3, GPR:$rs1, uimm7_lsb000:$imm7),
                    "sdp", "$rs2, $rs3, ${imm7}(${rs1})">,
                    Sched<[WriteSTD, ReadStoreData, ReadStoreData, ReadMemBase]> {
let mayLoad = 0;
let mayStore = 1;
}
}
