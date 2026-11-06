//=- AArch64SchedFirestorm.td - Firestorm Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for AArch64 Firestorm to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def FirestormModel : SchedMachineModel {
  let IssueWidth = 6; // 6 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 192; // Based on the reorder buffer.
  let LoadLatency = 4; // Optimistic load latency.
  let MispredictPenalty = 13;
  let CompleteModel = 1;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Firestorm.

// 4 integer pipes
def FsUnitI : ProcResource<4> {
  let BufferSize = 48;
}

// 2 branch units: I[0..1]
def FsUnitB : ProcResource<2> {
  let Super  = FsUnitI;
  let BufferSize = 24;
}

// 1 indirect-branch unit: I[0]
def FsUnitBR : ProcResource<1> {
  let Super  = FsUnitB;
}

// 2 shifter pipes: I[2..3]
// When an instruction consumes a FsUnitIS, it also consumes a FsUnitI
def FsUnitIS : ProcResource<2> {
  let Super = FsUnitI;
  let BufferSize = 24;
}

// 1 mul pipe: I[0]
def FsUnitIM : ProcResource<1> {
  let Super = FsUnitBR;
  let BufferSize = 32;
}

// 1 div pipe: I[1]
def FsUnitID : ProcResource<1> {
  let Super = FsUnitB;
  let BufferSize = 16;
}

// 1 integer division unit. This is driven by the ID pipe, but only
// consumes the pipe for one cycle at issue and another cycle at writeback.
def FsUnitIntDiv : ProcResource<1>;

// 2 ld/st pipes.
def FsUnitLS : ProcResource<2> {
  let BufferSize = 28;
}

// 3 fp/vector pipes.
def FsUnitV : ProcResource<3> {
  let BufferSize = 48;
}
// 2 fp/vector arithmetic and multiply pipes: V[0-1]
def FsUnitVM : ProcResource<2> {
  let Super = FsUnitV;
  let BufferSize = 32;
}
// 1 fp/vector division/sqrt pipe: V[2]
def FsUnitVD : ProcResource<1> {
  let Super = FsUnitV;
  let BufferSize = 16;
}
// 1 fp compare pipe: V[0]
def FsUnitVC : ProcResource<1> {
  let Super = FsUnitVM;
  let BufferSize = 16;
}

// 2 fp division/square-root units.  These are driven by the VD pipe,
// but only consume the pipe for one cycle at issue and a cycle at writeback.
def FsUnitFloatDiv : ProcResource<2>;

//===----------------------------------------------------------------------===//
// Define scheduler read/write resources and latency on Firestorm.
// This mirrors sections 7.7-7.9 of the Tuning Guide v1.0.1.

let SchedModel = FirestormModel in {

//---
// 7.8.1. Moves
//---

// A single nop micro-op (uX).
def FsWriteX : SchedWriteRes<[]> { let Latency = 0; }

// Move zero is a register rename (to machine register zero).
// The move is replaced by a single nop micro-op.
// MOVZ Rd, #0
// AND Rd, Rzr, #imm
def FsWriteZPred : SchedPredicate<[{TII->isGPRZero(*MI)}]>;
def FsWriteImmZ  : SchedWriteVariant<[
                   SchedVar<FsWriteZPred, [FsWriteX]>,
                   SchedVar<NoSchedPred, [WriteImm]>]>;
def : InstRW<[FsWriteImmZ], (instrs MOVZWi,MOVZXi,ANDWri,ANDXri)>;

// Move GPR is a register rename and single nop micro-op.
// ORR Xd, XZR, Xm
// ADD Xd, Xn, #0
def FsWriteIMovPred : SchedPredicate<[{TII->isGPRCopy(*MI)}]>;
def FsWriteVMovPred : SchedPredicate<[{TII->isFPRCopy(*MI)}]>;
def FsWriteMov      : SchedWriteVariant<[
                      SchedVar<FsWriteIMovPred, [FsWriteX]>,
                      SchedVar<FsWriteVMovPred, [FsWriteX]>,
                      SchedVar<NoSchedPred,   [WriteI]>]>;
def : InstRW<[FsWriteMov], (instrs COPY,ORRXrr,ADDXrr)>;

// Move non-zero immediate is an integer ALU op.
// MOVN,MOVZ,MOVK
def : WriteRes<WriteImm, [FsUnitI]>;

//---
// 7.8.2-7.8.5. Arithmetic and Logical, Comparison, Conditional,
//              Shifts and Bitfield Operations
//---

// ADR,ADRP
// ADD(S)ri,SUB(S)ri,AND(S)ri,EORri,ORRri
// ADD(S)rr,SUB(S)rr,AND(S)rr,BIC(S)rr,EONrr,EORrr,ORNrr,ORRrr
// ADC(S),SBC(S)
// Aliases: CMN, CMP, TST
//
// Conditional operations.
// CCMNi,CCMPi,CCMNr,CCMPr,
// CSEL,CSINC,CSINV,CSNEG
//
// Bit counting and reversal operations.
// CLS,CLZ,RBIT,REV,REV16,REV32
def : WriteRes<WriteI, [FsUnitI]>;

// ADD with shifted register operand is a single micro-op that
// consumes a shift pipeline for two cycles.
// ADD(S)rs,SUB(S)rs,AND(S)rs,BIC(S)rs,EONrs,EORrs,ORNrs,ORRrs
// EXAMPLE: ADDrs Xn, Xm LSL #imm
def : WriteRes<WriteISReg, [FsUnitIS]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
}

// ADD with extended register operand is the same as shifted reg operand.
// ADD(S)re,SUB(S)re
// EXAMPLE: ADDXre Xn, Xm, UXTB #1
def : WriteRes<WriteIEReg, [FsUnitIS]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
}

// Variable shift and bitfield operations.
// ASRV,LSLV,LSRV,RORV,BFM,SBFM,UBFM
def : WriteRes<WriteIS, [FsUnitIS]>;

// EXTR Shifts a pair of registers and requires two micro-ops.
// The second micro-op is delayed, as modeled by ReadExtrHi.
// EXTR Xn, Xm, #imm
def : WriteRes<WriteExtr, [FsUnitIS, FsUnitIS]> {
  let Latency = 2;
  let NumMicroOps = 2;
}

// EXTR's first register read is delayed by one cycle, effectively
// shortening its writer's latency.
// EXTR Xn, Xm, #imm
def : ReadAdvance<ReadExtrHi, 1>;

//---
// 7.8.6. Multiplies
//---

// MUL/MNEG are aliases for MADD/MSUB.
// MADDW,MSUBW,SMADDL,SMSUBL,UMADDL,UMSUBL
def : WriteRes<WriteIM32, [FsUnitIM]> {
  let Latency = 4;
}
// MADDX,MSUBX,SMULH,UMULH
def : WriteRes<WriteIM64, [FsUnitIM]> {
  let Latency = 5;
}

//---
// 7.8.7. Divide
//---

// 32-bit divide takes 7-13 cycles. 10 cycles covers a 20-bit quotient.
// The ID pipe is consumed for 2 cycles: issue and writeback.
// SDIVW,UDIVW
def : WriteRes<WriteID32, [FsUnitID, FsUnitIntDiv]> {
  let Latency = 10;
  let ReleaseAtCycles = [2, 10];
}
// 64-bit divide takes 7-21 cycles. 13 cycles covers a 32-bit quotient.
// The ID pipe is consumed for 2 cycles: issue and writeback.
// SDIVX,UDIVX
def : WriteRes<WriteID64, [FsUnitID, FsUnitIntDiv]> {
  let Latency = 13;
  let ReleaseAtCycles = [2, 13];
}

//---
// 7.8.8,7.8.10. Load/Store, single element
//---

// Integer loads take 4 cycles and use one LS unit for one cycle.
def : WriteRes<WriteLD, [FsUnitLS]> {
  let Latency = 4;
}

// Store-load forwarding is 4 cycles.
//
// Note: The store-exclusive sequence incorporates this
// latency. However, general heuristics should not model the
// dependence between a store and subsequent may-alias load because
// hardware speculation works.
def : WriteRes<WriteST, [FsUnitLS]> {
  let Latency = 4;
}

// Load from base address plus an optionally scaled register offset.
// Rt latency is latency WriteIS + WriteLD.
// EXAMPLE: LDR Xn, Xm [, lsl 3]
def FsWriteLDIdx : SchedWriteVariant<[
  SchedVar<ScaledIdxPred, [WriteIS, WriteLD]>, // Load from scaled register.
  SchedVar<NoSchedPred,   [WriteLD]>]>;        // Load from register offset.
def : SchedAlias<WriteLDIdx, FsWriteLDIdx>;    // Map AArch64->Firestorm type.

// EXAMPLE: STR Xn, Xm [, lsl 3]
def FsWriteSTIdx : SchedWriteVariant<[
  SchedVar<ScaledIdxPred, [WriteIS, WriteST]>, // Store to scaled register.
  SchedVar<NoSchedPred,   [WriteST]>]>;        // Store to register offset.
def : SchedAlias<WriteSTIdx, FsWriteSTIdx>;    // Map AArch64->Firestorm type.

// Read the (unshifted) base register Xn in the second micro-op one cycle later.
// EXAMPLE: LDR Xn, Xm [, lsl 3]
def FsReadBaseRS : SchedReadAdvance<1>;
def FsReadAdrBase : SchedReadVariant<[
  SchedVar<ScaledIdxPred, [FsReadBaseRS]>, // Read base reg after shifting offset.
  SchedVar<NoSchedPred,   [ReadDefault]>]>;   // Read base reg with no shift.
def : SchedAlias<ReadAdrBase, FsReadAdrBase>; // Map AArch64->Firestorm type.
def : ReadAdvance<ReadST, 0>;

//---
// 7.8.9,7.8.11. Load/Store, paired
//---

// Address pre/post increment is a simple ALU op with one cycle latency.
def : WriteRes<WriteAdr, [FsUnitI]>;

// LDP high register write is fused with the load, but a nop micro-op remains.
def : WriteRes<WriteLDHi, []> {
  let Latency = 4;
}

// STP is a vector op and store, except for QQ, which is just two stores.
def FsWriteSTP : SchedWriteRes<[FsUnitV, FsUnitLS]> {
  let Latency = 6;
  let NumMicroOps = 2;
}

def : SchedAlias<WriteSTP, FsWriteSTP>;
def : InstRW<[WriteST, WriteST], (instrs STPQi)>;

//---
// 7.8.13. Branches
//---

// Branches take a single micro-op.
// The misprediction penalty is defined as a SchedMachineModel property.
def : WriteRes<WriteBr,    [FsUnitB]>  {let Latency = 0;}
def : WriteRes<WriteBrReg, [FsUnitBR]> {let Latency = 0;}

//---
// 7.8.14. Never-issued Instructions, Barrier and Hint Operations
//---

// NOP,SEV,SEVL,WFE,WFI,YIELD
def : WriteRes<WriteHint, []> {let Latency = 0;}
// ISB
def : InstRW<[WriteI], (instrs ISB)>;
// SLREX,DMB,DSB
def : WriteRes<WriteBarrier, [FsUnitLS]>;

// System instructions get an invalid latency because the latency of
// other operations across them is meaningless.
def : WriteRes<WriteSys, []> {let Latency = -1;}

//===----------------------------------------------------------------------===//
// 7.9 Vector Unit Instructions

// Simple vector operations take 2 cycles.
def : WriteRes<WriteVd, [FsUnitV]> {let Latency = 2;}
def : WriteRes<WriteVq, [FsUnitV]> {let Latency = 2;}

// Define some longer latency vector op types for Firestorm.
def FsWriteV3 : SchedWriteRes<[FsUnitV]> {let Latency = 3;}
def FsWriteV4 : SchedWriteRes<[FsUnitV]> {let Latency = 4;}
def FsWriteV5 : SchedWriteRes<[FsUnitV]> {let Latency = 5;}
def FsWriteV6 : SchedWriteRes<[FsUnitV]> {let Latency = 6;}

// Simple floating-point operations take 2 cycles.
def : WriteRes<WriteF, [FsUnitV]> {let Latency = 2;}

//---
// 7.9.1 Vector Moves
//---

// TODO: Add Firestorm-specific zero-cycle zeros. LLVM currently
// generates expensive int-float conversion instead:
// FMOVDi Dd, #0.0
// FMOVv2f64ns Vd.2d, #0.0

// FMOVSi,FMOVDi
def : WriteRes<WriteFImm, [FsUnitV]> {let Latency = 2;}

// MOVI,MVNI are WriteV
// FMOVv2f32ns,FMOVv2f64ns,FMOVv4f32ns are WriteV

// Move FPR is a register rename and single nop micro-op.
// ORR.16b Vd,Vn,Vn
// COPY is handled above in the FsWriteMov Variant.
def FsWriteVMov    : SchedWriteVariant<[
                     SchedVar<FsWriteVMovPred, [FsWriteX]>,
                     SchedVar<NoSchedPred,   [WriteVq]>]>;
def : InstRW<[FsWriteVMov], (instrs ORRv16i8)>;

// FMOVSr,FMOVDr are WriteF.

// MOV V,V is a WriteV.

// CPY D,V[x] is a WriteV

// INS V[x],V[y] is a WriteV.

// FMOVWSr,FMOVXDr,FMOVXDHighr
def : WriteRes<WriteFCopy, [FsUnitLS]> {
  let Latency = 5;
}

// FMOVSWr,FMOVDXr
def : InstRW<[WriteLD], (instrs FMOVSWr,FMOVDXr,FMOVDXHighr)>;

// INS V[x],R
def FsWriteCopyToFPR : WriteSequence<[WriteVLD, WriteVq]>;
def : InstRW<[FsWriteCopyToFPR], (instregex "INSv")>;

// SMOV,UMOV R,V[x]
def FsWriteCopyToGPR : WriteSequence<[WriteLD, WriteI]>;
def : InstRW<[FsWriteCopyToGPR], (instregex "SMOVv","UMOVv")>;

// DUP V,R
def : InstRW<[FsWriteCopyToFPR], (instregex "DUPv")>;

// DUP V,V[x] is a WriteV.

//---
// 7.9.2 Integer Arithmetic, Logical, and Comparisons
//---

// BIC,ORR V,#imm are WriteV

def : InstRW<[FsWriteV3], (instregex "ABSv")>;

// MVN,NEG,NOT are WriteV

def : InstRW<[FsWriteV3], (instregex "SQABSv","SQNEGv")>;

// ADDP is a WriteV.
def FsWriteVADDLP : SchedWriteRes<[FsUnitV]> {let Latency = 2;}
def : InstRW<[FsWriteVADDLP], (instregex "SADDLPv","UADDLPv")>;

def : InstRW<[FsWriteV3],
             (instregex "ADDVv","SMAXVv","UMAXVv","SMINVv","UMINVv")>;

def : InstRW<[FsWriteV3], (instregex "SADDLV","UADDLV")>;

// ADD,SUB are WriteV

// Forward declare.
def FsWriteVABD : SchedWriteRes<[FsUnitV]> {let Latency = 3;}

// Add/Diff and accumulate uses the vector multiply unit.
def FsWriteVAccum : SchedWriteRes<[FsUnitVM]> {let Latency = 3;}
def FsReadVAccum  : SchedReadAdvance<1,
                    [FsWriteVAccum, FsWriteVADDLP, FsWriteVABD]>;

def : InstRW<[FsWriteVAccum, FsReadVAccum],
             (instregex "SADALP","UADALP")>;

def : InstRW<[FsWriteVAccum, FsReadVAccum],
             (instregex "SABAv","UABAv","SABALv","UABALv")>;

def : InstRW<[FsWriteV3], (instregex "SQADDv","SQSUBv","UQADDv","UQSUBv")>;

def : InstRW<[FsWriteV3], (instregex "SUQADDv","USQADDv")>;

def : InstRW<[FsWriteV4], (instregex "ADDHNv","RADDHNv", "RSUBHNv", "SUBHNv")>;

// WriteV includes:
// AND,BIC,CMTST,EOR,ORN,ORR
// ADDP
// SHADD,SHSUB,SRHADD,UHADD,UHSUB,URHADD
// SADDL,SSUBL,UADDL,USUBL
// SADDW,SSUBW,UADDW,USUBW

def : InstRW<[FsWriteV3], (instregex "CMEQv","CMGEv","CMGTv",
                                     "CMLEv","CMLTv",
                                     "CMHIv","CMHSv")>;

def : InstRW<[FsWriteV3], (instregex "SMAXv","SMINv","UMAXv","UMINv",
                                     "SMAXPv","SMINPv","UMAXPv","UMINPv")>;

def : InstRW<[FsWriteVABD], (instregex "SABDv","UABDv",
                                       "SABDLv","UABDLv")>;

//---
// 7.9.3 Floating Point Arithmetic and Comparisons
//---

// FABS,FNEG are WriteF

def : InstRW<[FsWriteV4], (instrs FADDPv2i32p)>;
def : InstRW<[FsWriteV5], (instrs FADDPv2i64p)>;

def : InstRW<[FsWriteV3], (instregex "FMAXPv2i","FMAXNMPv2i",
                                     "FMINPv2i","FMINNMPv2i")>;

def : InstRW<[FsWriteV4], (instregex "FMAXVv","FMAXNMVv","FMINVv","FMINNMVv")>;

def : InstRW<[FsWriteV4], (instrs FADDSrr,FADDv2f32,FADDv4f32,
                                  FSUBSrr,FSUBv2f32,FSUBv4f32,
                                  FADDPv2f32,FADDPv4f32,
                                  FABD32,FABDv2f32,FABDv4f32)>;
def : InstRW<[FsWriteV5], (instrs FADDDrr,FADDv2f64,
                                  FSUBDrr,FSUBv2f64,
                                  FADDPv2f64,
                                  FABD64,FABDv2f64)>;

def : InstRW<[FsWriteV3], (instregex "FCMEQ","FCMGT","FCMLE","FCMLT")>;

def : InstRW<[FsWriteV3], (instregex "FACGE","FACGT",
                                     "FMAXS","FMAXD","FMAXv",
                                     "FMINS","FMIND","FMINv",
                                     "FMAXNMS","FMAXNMD","FMAXNMv",
                                     "FMINNMS","FMINNMD","FMINNMv",
                                     "FMAXPv2f","FMAXPv4f",
                                     "FMINPv2f","FMINPv4f",
                                     "FMAXNMPv2f","FMAXNMPv4f",
                                     "FMINNMPv2f","FMINNMPv4f")>;

// FCMP,FCMPE,FCCMP,FCCMPE
def : WriteRes<WriteFCmp, [FsUnitVC]> {let Latency = 4;}

// FCSEL is a WriteF.

//---
// 7.9.4 Shifts and Bitfield Operations
//---

// SHL is a WriteV

def FsWriteVSHR : SchedWriteRes<[FsUnitV]> {let Latency = 2;}
def : InstRW<[FsWriteVSHR], (instregex "SSHRv","USHRv")>;

def FsWriteVSRSHR : SchedWriteRes<[FsUnitV]> {let Latency = 3;}
def : InstRW<[FsWriteVSRSHR], (instregex "SRSHRv","URSHRv")>;

// Shift and accumulate uses the vector multiply unit.
def FsWriteVShiftAcc : SchedWriteRes<[FsUnitVM]> {let Latency = 3;}
def FsReadVShiftAcc  : SchedReadAdvance<1,
                        [FsWriteVShiftAcc, FsWriteVSHR, FsWriteVSRSHR]>;
def : InstRW<[FsWriteVShiftAcc, FsReadVShiftAcc],
             (instregex "SRSRAv","SSRAv","URSRAv","USRAv")>;

// SSHL,USHL are WriteV.

def : InstRW<[FsWriteV3], (instregex "SRSHLv","URSHLv")>;

// SQSHL,SQSHLU,UQSHL are WriteV.

def : InstRW<[FsWriteV3], (instregex "SQRSHLv","UQRSHLv")>;

// WriteV includes:
// SHLL,SSHLL,USHLL
// SLI,SRI
// BIF,BIT,BSL,BSP
// EXT
// CLS,CLZ,CNT,RBIT,REV16,REV32,REV64,XTN
// XTN2

def : InstRW<[FsWriteV4],
             (instregex "RSHRNv","SHRNv",
                        "SQRSHRNv","SQRSHRUNv","SQSHRNv","SQSHRUNv",
                        "UQRSHRNv","UQSHRNv","SQXTNv","SQXTUNv","UQXTNv")>;

//---
// 7.9.5 Multiplication
//---

def FsWriteVMul : SchedWriteRes<[FsUnitVM]> { let Latency = 4;}
def : InstRW<[FsWriteVMul], (instregex "MULv","SMULLv","UMULLv",
                             "SQDMULLv","SQDMULHv","SQRDMULHv")>;

// FMUL,FMULX,FNMUL default to WriteFMul.
def : WriteRes<WriteFMul, [FsUnitVM]> { let Latency = 4;}

def FsWriteV64Mul : SchedWriteRes<[FsUnitVM]> { let Latency = 5;}
def : InstRW<[FsWriteV64Mul], (instrs FMULDrr,FMULv2f64,FMULv2i64_indexed,
                               FNMULDrr,FMULX64,FMULXv2f64,FMULXv2i64_indexed)>;

def FsReadVMulAcc : SchedReadAdvance<1, [FsWriteVMul, FsWriteV64Mul]>;
def : InstRW<[FsWriteVMul, FsReadVMulAcc],
             (instregex "MLA","MLS","SMLAL","SMLSL","UMLAL","UMLSL",
              "SQDMLAL","SQDMLSL")>;

def FsWriteSMul : SchedWriteRes<[FsUnitVM]> { let Latency = 8;}
def FsWriteDMul : SchedWriteRes<[FsUnitVM]> { let Latency = 10;}
def FsReadSMul : SchedReadAdvance<4, [FsWriteSMul]>;
def FsReadDMul : SchedReadAdvance<5, [FsWriteDMul]>;

def : InstRW<[FsWriteSMul, FsReadSMul],
             (instrs FMADDSrrr,FMSUBSrrr,FNMADDSrrr,FNMSUBSrrr,
              FMLAv2f32,FMLAv4f32,
              FMLAv1i32_indexed,FMLAv1i64_indexed,FMLAv2i32_indexed)>;
def : InstRW<[FsWriteDMul, FsReadDMul],
             (instrs FMADDDrrr,FMSUBDrrr,FNMADDDrrr,FNMSUBDrrr,
              FMLAv2f64,FMLAv2i64_indexed,
              FMLSv2f64,FMLSv2i64_indexed)>;

def FsWritePMUL : SchedWriteRes<[FsUnitVD]> { let Latency = 3; }
def : InstRW<[FsWritePMUL], (instregex "PMULv", "PMULLv")>;

//---
// 7.9.6 Divide and Square Root
//---

// FDIV,FSQRT
// TODO: Add 64-bit variant with 19 cycle latency.
// TODO: Specialize FSQRT for longer latency.
def : WriteRes<WriteFDiv, [FsUnitVD, FsUnitFloatDiv]> {
  let Latency = 17;
  let ReleaseAtCycles = [2, 17];
}

def : InstRW<[FsWriteV4], (instregex "FRECPEv","FRECPXv","URECPEv","URSQRTEv")>;

def FsWriteFRSQRTE : SchedWriteRes<[FsUnitVM]> { let Latency = 4; }
def : InstRW<[FsWriteFRSQRTE], (instregex "FRSQRTEv")>;

def FsWriteFRECPS : SchedWriteRes<[FsUnitVM]> { let Latency = 8; }
def FsWriteFRSQRTS : SchedWriteRes<[FsUnitVM]> { let Latency = 10; }
def : InstRW<[FsWriteFRECPS],  (instregex "FRECPSv")>;
def : InstRW<[FsWriteFRSQRTS], (instregex "FRSQRTSv")>;

//---
// 7.9.7 Integer-FP Conversions
//---

// FCVT lengthen f16/s32
def : InstRW<[WriteVq], (instrs FCVTSHr,FCVTDHr,FCVTDSr)>;

// FCVT,FCVTN,FCVTXN
// SCVTF,UCVTF V,V
// FRINT(AIMNPXZ) V,V
def : WriteRes<WriteFCvt, [FsUnitV]> {let Latency = 4;}

// SCVT/UCVT S/D, Rd = VLD5+V4: 9 cycles.
def FsWriteCvtToFPR : WriteSequence<[WriteVLD, FsWriteV4]>;
def : InstRW<[FsWriteCopyToFPR], (instregex "FCVT[AMNPZ][SU][SU][WX][SD]r")>;

// FCVT Rd, S/D = V6+LD4: 10 cycles
def FsWriteCvtToGPR : WriteSequence<[FsWriteV6, WriteLD]>;
def : InstRW<[FsWriteCvtToGPR], (instregex "[SU]CVTF[SU][WX][SD]r")>;

// FCVTL is a WriteV

//---
// 7.9.8-7.9.10 Cryptography, Data Transposition, Table Lookup
//---

def FsWriteCrypto2 : SchedWriteRes<[FsUnitVD]> {let Latency = 2;}
def : InstRW<[FsWriteCrypto2], (instrs AESIMCrr, AESMCrr, SHA1Hrr,
                                       AESDrr, AESErr, SHA1SU1rr, SHA256SU0rr,
                                       SHA1SU0rrr)>;

def FsWriteCrypto3 : SchedWriteRes<[FsUnitVD]> {let Latency = 3;}
def : InstRW<[FsWriteCrypto3], (instrs SHA256SU1rrr)>;

def FsWriteCrypto6 : SchedWriteRes<[FsUnitVD]> {let Latency = 6;}
def : InstRW<[FsWriteCrypto6], (instrs SHA1Crrr, SHA1Mrrr, SHA1Prrr,
                                       SHA256Hrrr,SHA256H2rrr)>;

// TRN,UZP,ZUP are WriteV.

// TBL,TBX are WriteV.

//---
// 7.9.11-7.9.14 Load/Store, single element and paired
//---

// Loading into the vector unit takes 5 cycles vs 4 for integer loads.
def : WriteRes<WriteVLD, [FsUnitLS]> {
  let Latency = 5;
}

// Store-load forwarding is 4 cycles.
def : WriteRes<WriteVST, [FsUnitLS]> {
  let Latency = 4;
}

// WriteVLDPair/VSTPair sequences are expanded by the target description.

//---
// 7.9.15 Load, element operations
//---

// Only the first WriteVLD and WriteAdr for writeback matches def operands.
// Subsequent WriteVLDs consume resources. Since all loaded values have the
// same latency, this is acceptable.

// Vd is read 5 cycles after issuing the vector load.
def : ReadAdvance<ReadVLD, 5>;

def : InstRW<[WriteVLD],
             (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr],
             (instregex "LD1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST")>;

// Register writes from the load's high half are fused micro-ops.
def : InstRW<[WriteVLD],
             (instregex "LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteVLD, WriteAdr],
             (instregex "LD1Twov(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVLD, WriteVLD],
             (instregex "LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr, WriteVLD],
             (instregex "LD1Twov(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLD, WriteVLD],
             (instregex "LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteVLD, WriteAdr, WriteVLD],
             (instregex "LD1Threev(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVLD, WriteVLD, WriteVLD],
             (instregex "LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr, WriteVLD, WriteVLD],
             (instregex "LD1Threev(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLD, WriteVLD],
             (instregex "LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteVLD, WriteAdr, WriteVLD],
             (instregex "LD1Fourv(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVLD, WriteVLD, WriteVLD, WriteVLD],
             (instregex "LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLD, WriteAdr, WriteVLD, WriteVLD, WriteVLD],
             (instregex "LD1Fourv(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD],
             (instregex "LD1i(8|16|32)$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr],
             (instregex "LD1i(8|16|32)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD],          (instrs LD1i64)>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr],(instrs LD1i64_POST)>;

def : InstRW<[WriteVLDShuffle],
             (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr],
             (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

def : InstRW<[WriteVLDShuffle, WriteVq],
             (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVq],
             (instregex "LD2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteVLDShuffle, WriteVLDShuffle],
             (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle],
             (instregex "LD2Twov(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVq],
             (instregex "LD2i(8|16|32)$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVq],
             (instregex "LD2i(8|16|32)_POST")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVq],
             (instregex "LD2i64$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVq],
             (instregex "LD2i64_POST")>;

def : InstRW<[WriteVLDShuffle, WriteVq],
             (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVq],
             (instregex "LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLDShuffle, WriteVLDShuffle, WriteVq],
             (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle, WriteVq],
             (instregex "LD3Threev(8b|4h|2s)_POST")>;
def : InstRW<[WriteVLDShuffle, WriteVLDShuffle, WriteVLDShuffle],
             (instregex "LD3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle, WriteVLDShuffle],
             (instregex "LD3Threev(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVq, WriteVq],
             (instregex "LD3i(8|16|32)$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVq, WriteVq],
             (instregex "LD3i(8|16|32)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVLDShuffle, WriteVq],
             (instregex "LD3i64$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVLDShuffle, WriteVq],
             (instregex "LD3i64_POST")>;

def : InstRW<[WriteVLDShuffle, WriteVq, WriteVq],
             (instregex "LD3Rv(8b|4h|2s|16b|8h|4s)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVq, WriteVq],
             (instregex "LD3Rv(8b|4h|2s|16b|8h|4s)_POST")>;

def : InstRW<[WriteVLDShuffle, WriteVLDShuffle, WriteVq],
             (instrs LD3Rv1d,LD3Rv2d)>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle, WriteVq],
             (instrs LD3Rv1d_POST,LD3Rv2d_POST)>;

def : InstRW<[WriteVLDShuffle, WriteVLDShuffle, WriteVq, WriteVq],
             (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle, WriteVq, WriteVq],
             (instregex "LD4Fourv(8b|4h|2s)_POST")>;
def : InstRW<[WriteVLDPairShuffle, WriteVLDPairShuffle,
              WriteVLDPairShuffle, WriteVLDPairShuffle],
             (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteVLDPairShuffle, WriteAdr, WriteVLDPairShuffle,
              WriteVLDPairShuffle, WriteVLDPairShuffle],
             (instregex "LD4Fourv(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVq, WriteVq, WriteVq],
             (instregex "LD4i(8|16|32)$")>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVq, WriteVq, WriteVq],
             (instregex "LD4i(8|16|32)_POST")>;


def : InstRW<[WriteVLDShuffle, ReadVLD, WriteVLDShuffle, WriteVq, WriteVq],
             (instrs LD4i64)>;
def : InstRW<[WriteVLDShuffle, ReadVLD, WriteAdr, WriteVLDShuffle, WriteVq],
             (instrs LD4i64_POST)>;

def : InstRW<[WriteVLDShuffle, WriteVq, WriteVq, WriteVq],
             (instregex "LD4Rv(8b|4h|2s|16b|8h|4s)$")>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVq, WriteVq, WriteVq],
             (instregex "LD4Rv(8b|4h|2s|16b|8h|4s)_POST")>;

def : InstRW<[WriteVLDShuffle, WriteVLDShuffle, WriteVq, WriteVq],
             (instrs LD4Rv1d,LD4Rv2d)>;
def : InstRW<[WriteVLDShuffle, WriteAdr, WriteVLDShuffle, WriteVq, WriteVq],
             (instrs LD4Rv1d_POST,LD4Rv2d_POST)>;

//---
// 7.9.16 Store, element operations
//---

// Only the WriteAdr for writeback matches a def operands.
// Subsequent WriteVLDs only consume resources.

def : InstRW<[WriteVST],
             (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVST],
             (instregex "ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle],
             (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle],
             (instregex "ST1Twov(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVST, WriteVST],
             (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVST, WriteVST],
             (instregex "ST1Twov(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle, WriteVST],
             (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVST],
             (instregex "ST1Threev(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVST, WriteVST, WriteVST],
             (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVST, WriteVST, WriteVST],
             (instregex "ST1Threev(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST1Fourv(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVST, WriteVST, WriteVST, WriteVST],
             (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVST, WriteVST, WriteVST, WriteVST],
             (instregex "ST1Fourv(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle],           (instregex "ST1i(8|16|32)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle], (instregex "ST1i(8|16|32)_POST")>;

def : InstRW<[WriteVSTShuffle],           (instrs ST1i64)>;
def : InstRW<[WriteAdr, WriteVSTShuffle], (instrs ST1i64_POST)>;

def : InstRW<[WriteVSTShuffle],
             (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle],
             (instregex "ST2Twov(8b|4h|2s)_POST")>;
def : InstRW<[WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST2Twov(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle],           (instregex "ST2i(8|16|32)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle], (instregex "ST2i(8|16|32)_POST")>;
def : InstRW<[WriteVSTShuffle],           (instrs ST2i64)>;
def : InstRW<[WriteAdr, WriteVSTShuffle], (instrs ST2i64_POST)>;

def : InstRW<[WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST3Threev(8b|4h|2s)_POST")>;
def : InstRW<[WriteVSTShuffle, WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle, WriteVSTShuffle],
             (instregex "ST3Threev(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTShuffle],           (instregex "ST3i(8|16|32)$")>;
def : InstRW<[WriteAdr, WriteVSTShuffle], (instregex "ST3i(8|16|32)_POST")>;

def :InstRW<[WriteVSTShuffle, WriteVSTShuffle],           (instrs ST3i64)>;
def :InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle], (instrs ST3i64_POST)>;

def : InstRW<[WriteVSTPairShuffle, WriteVSTPairShuffle],
            (instregex "ST4Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, WriteVSTPairShuffle, WriteVSTPairShuffle],
            (instregex "ST4Fourv(8b|4h|2s|1d)_POST")>;
def : InstRW<[WriteVSTPairShuffle, WriteVSTPairShuffle,
              WriteVSTPairShuffle, WriteVSTPairShuffle],
             (instregex "ST4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, WriteVSTPairShuffle, WriteVSTPairShuffle,
              WriteVSTPairShuffle, WriteVSTPairShuffle],
             (instregex "ST4Fourv(16b|8h|4s|2d)_POST")>;

def : InstRW<[WriteVSTPairShuffle],           (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, WriteVSTPairShuffle], (instregex "ST4i(8|16|32)_POST")>;

def : InstRW<[WriteVSTShuffle, WriteVSTShuffle],          (instrs ST4i64)>;
def : InstRW<[WriteAdr, WriteVSTShuffle, WriteVSTShuffle],(instrs ST4i64_POST)>;

// Atomic operations are not supported.
def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

//---
// Unused SchedRead types
//---

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 0>;
def : ReadAdvance<ReadID, 0>;

} // SchedModel = FirestormModel
