//==- AArch64SchedCortexA320.td - ARM Cortex-A320 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for the ARM Cortex-A320 processor.
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the per-operand machine model.
// This works with MachineScheduler. See MCSchedModel.h for details.

// Cortex-A320 machine model for scheduling and other instruction cost heuristics.
def CortexA320Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // Cortex-A320 is an in-order processor
  let IssueWidth = 1;         // Cortex-A320 is a single-issue processor
  let LoadLatency = 5;
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // Covers instructions applicable to Cortex-A320.

  let FullInstRWOverlapCheck = 0;
}


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types

let SchedModel = CortexA320Model in {

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since the
// Cortex-A320 is in-order.
let BufferSize = 0 in {
  def CortexA320UnitALU    : ProcResource<1>;    // Int ALU
  def CortexA320UnitMAC    : ProcResource<1>;    // Int MAC, 64-bit wide
  def CortexA320UnitDiv    : ProcResource<1>;    // Int Division, not pipelined
  def CortexA320UnitLdSt   : ProcResource<1>;    // Load/Store shared pipe
  def CortexA320UnitB      : ProcResource<1>;    // Branch
  def CortexA320UnitPAC    : ProcResource<1>;    // Pointer Authentication (PAC) pipe

  // The FP DIV/SQRT instructions execute totally differently from the FP ALU
  // instructions; that's why for now we model them with 2 resources.
  def CortexA320UnitVALU  : ProcResource<1>;    // SIMD/FP/SVE ALU
  def CortexA320UnitVMAC   : ProcResource<1>;    // SIMD/FP/SVE MAC
  def CortexA320UnitVMC    : ProcResource<1>;    // SIMD/FP/SVE multicycle instrs  (e.g Div, SQRT, cryptography)
}

// These latencies are modeled without taking into account forwarding paths
// (the software optimisation guide lists latencies taking into account
// typical forwarding paths).
def : WriteRes<WriteImm,   [CortexA320UnitALU]> { let Latency = 1; }  // MOVN, MOVZ
def : WriteRes<WriteI,     [CortexA320UnitALU]> { let Latency = 1; }  // ALU
def : WriteRes<WriteISReg, [CortexA320UnitALU]> { let Latency = 2; }  // ALU of Shifted-Reg
def : WriteRes<WriteIEReg, [CortexA320UnitALU]> { let Latency = 2; }  // ALU of Extended-Reg
def : WriteRes<WriteExtr, [CortexA320UnitALU]> { let Latency = 2; }   // EXTR from a reg pair
def : WriteRes<WriteIS, [CortexA320UnitALU]> { let Latency = 2; }     // Shift/Scale

// MAC
def : WriteRes<WriteIM32, [CortexA320UnitMAC]> { let Latency = 3; }   // 32-bit Multiply
def : WriteRes<WriteIM64, [CortexA320UnitMAC]> { let Latency = 5; let ReleaseAtCycles = [2];}   // 64-bit Multiply

// Div
def : WriteRes<WriteID32, [CortexA320UnitDiv]> {
  let Latency = 12; let ReleaseAtCycles = [12];
}
def : WriteRes<WriteID64, [CortexA320UnitDiv]> {
  let Latency = 20; let ReleaseAtCycles = [20];
}

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to Cortex-A320

//===----------------------------------------------------------------------===//
class CortexA320Write<int n, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = n;
}

class CortexA320MCWrite<int n, int m, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = n;
  let ReleaseAtCycles = [m];
  let BeginGroup = 1;
}

class CortexA320MC_RC0Write<int n, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = n;
  let BeginGroup = 1;
}

//===----------------------------------------------------------------------===//

// Define generic 2 micro-op types
def CortexA320Write_11cyc_1VMAC_1VALU : SchedWriteRes<[CortexA320UnitVALU, CortexA320UnitVMAC]> {
  let Latency     = 11;
  let NumMicroOps = 2;
}

def CortexA320Write_16cyc_1VMAC_1VALU : SchedWriteRes<[CortexA320UnitVALU, CortexA320UnitVMAC]> {
  let Latency     = 16;
  let NumMicroOps = 2;
}

class CortexA320Write_PAC_B <int lat> : SchedWriteRes<[CortexA320UnitPAC, CortexA320UnitB]> {
  let Latency = lat;
  let NumMicroOps = 2;
}

// Load
def : WriteRes<WriteLD, [CortexA320UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDIdx, [CortexA320UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDHi, [CortexA320UnitLdSt]> { let Latency = 4; }

def CortexA320WriteVLD1 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 3; }
def CortexA320WriteVLD1SI : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 3; let SingleIssue = 1; }

def CortexA320WriteVLD2 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4;
                                                  let ReleaseAtCycles = [2]; }

def CortexA320WriteVLD3 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [3]; }

def CortexA320WriteVLD4 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [4]; }

def CortexA320WriteVLD6 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [3]; }

def CortexA320WriteVLD8 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [4]; }

def CortexA320WriteLDP1 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; }
def CortexA320WriteLDP2 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; }
def CortexA320WriteLDP4 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; }

// Pre/Post Indexing - Performed as part of address generation
def : WriteRes<WriteAdr, []> { let Latency = 0; }

// Store
let RetireOOO = 1 in {
def : WriteRes<WriteST, [CortexA320UnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTP, [CortexA320UnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [CortexA320UnitLdSt]> { let Latency = 1; }
}
def : WriteRes<WriteSTX, [CortexA320UnitLdSt]> { let Latency = 3; }

// Vector Store - Similar to vector loads, can take 1-3 cycles to issue.
def : WriteRes<WriteVST, [CortexA320UnitLdSt]> { let Latency = 5;
                                          let ReleaseAtCycles = [2];}
def CortexA320WriteVST1 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; }
def CortexA320WriteVST2 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def CortexA320WriteVST3 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [3]; }
def CortexA320WriteVST4 : SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [4]; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Branch
def : WriteRes<WriteBr, [CortexA320UnitB]>;
def : WriteRes<WriteBrReg, [CortexA320UnitB]>;
def : WriteRes<WriteSys, [CortexA320UnitB]>;
def : WriteRes<WriteBarrier, [CortexA320UnitB]>;
def : WriteRes<WriteHint, [CortexA320UnitB]>;

// FP ALU
//   As WriteF result is produced in F5 and it can be mostly forwarded
//   to consumer at F1, the effectively Latency is set as 4.
def : WriteRes<WriteF, [CortexA320UnitVALU]> { let Latency = 4; }
def : WriteRes<WriteFCmp, [CortexA320UnitVALU]> { let Latency = 3; }
def : WriteRes<WriteFCvt, [CortexA320UnitVALU]> { let Latency = 4; }
def : WriteRes<WriteFCopy, [CortexA320UnitVALU]> { let Latency = 3; }
def : WriteRes<WriteFImm, [CortexA320UnitVALU]> { let Latency = 3; }

class CortexA320VSt<int n> : SchedWriteRes<[CortexA320UnitLdSt]> {
  let RetireOOO = 1;
  let ReleaseAtCycles = [n];
}

def CortexA320VSt0      : SchedWriteRes<[CortexA320UnitLdSt]> {
  let RetireOOO = 1;
}

def : SchedAlias<WriteVd, CortexA320Write<4, CortexA320UnitVALU>>;
def : SchedAlias<WriteVq, CortexA320Write<4, CortexA320UnitVALU>>;

// FP ALU specific new schedwrite definitions
def CortexA320WriteFPALU_F3 : SchedWriteRes<[CortexA320UnitVALU]> { let Latency = 3;}
def CortexA320WriteFPALU_F4 : SchedWriteRes<[CortexA320UnitVALU]> { let Latency = 4;}

// FP Mul, Div, Sqrt. Div/Sqrt are not pipelined
def : WriteRes<WriteFMul, [CortexA320UnitVMAC]> { let Latency = 4; }

let RetireOOO = 1 in {
def : WriteRes<WriteFDiv, [CortexA320UnitVMC]> { let Latency = 22;
                                            let ReleaseAtCycles = [29]; }
def CortexA320WriteVMAC : SchedWriteRes<[CortexA320UnitVMAC]> { let Latency = 4; }
def CortexA320WriteFDivHP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 8;
                                                     let ReleaseAtCycles = [5]; }
def CortexA320WriteFDivSP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 13;
                                                     let ReleaseAtCycles = [10]; }
def CortexA320WriteFDivDP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 22;
                                                     let ReleaseAtCycles = [19]; }
def CortexA320WriteFSqrtHP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 8;
                                                      let ReleaseAtCycles = [5]; }
def CortexA320WriteFSqrtSP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 12;
                                                      let ReleaseAtCycles = [9]; }
def CortexA320WriteFSqrtDP : SchedWriteRes<[CortexA320UnitVMC]> { let Latency = 22;
                                                      let ReleaseAtCycles = [19]; }
}

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types.

def : ReadAdvance<ReadVLD, 0>;
def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadST, 1>;

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;


// MUL
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 2>;

// Div
def : ReadAdvance<ReadID, 0>;

//===----------------------------------------------------------------------===//
// Subtarget-specific InstRWs.

def CortexA320WriteISReg : SchedWriteVariant<[
       SchedVar<RegShiftedPred, [WriteISReg]>,
       SchedVar<NoSchedPred, [WriteI]>]>;
def : InstRW<[CortexA320WriteISReg], (instregex ".*rs$")>;
def : InstRW<[WriteIS], (instrs RBITWr, RBITXr)>;

// Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code, using generic key
// Compute pointer authentication code for instruction address
def : InstRW<[CortexA320Write<4, CortexA320UnitPAC>], (instregex "^AUT", "^PAC")>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[CortexA320Write_PAC_B<1>], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA,
                                            BRAAZ, BRAB, BRABZ, RETAA, RETAB,
                                            ERETAA, ERETAB)>;

// Load register, with pointer authentication
def : InstRW<[CortexA320Write<2, CortexA320UnitPAC>], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[CortexA320Write<5, CortexA320UnitPAC>], (instrs XPACD, XPACI, XPACLRI)>;
//---
// Miscellaneous
//---
def : InstRW<[CortexA320WriteVLD1SI,CortexA320WriteLDP1], (instregex "LDPS?Wi")>;
def : InstRW<[CortexA320WriteVLD1,CortexA320WriteLDP1], (instregex "LDPSi")>;
def : InstRW<[CortexA320WriteVLD1,CortexA320WriteLDP2], (instregex "LDP(X|D)i")>;
def : InstRW<[CortexA320WriteVLD1,CortexA320WriteLDP4], (instregex "LDPQi")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1SI,CortexA320WriteLDP1], (instregex "LDPS?W(pre|post)")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1,CortexA320WriteLDP1], (instregex "LDPS(pre|post)")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1,CortexA320WriteLDP2], (instregex "LDP(X|D)(pre|post)")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1,CortexA320WriteLDP4], (instregex "LDPQ(pre|post)")>;
def : InstRW<[WriteI], (instrs COPY)>;
//---
// Vector Loads - 128-bit per cycle
//---
//   1-element structures
def CortexA320WriteVLD1Latency3: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 3; let ReleaseAtCycles = [1]; }
def CortexA320WriteVLD1Latency4: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def CortexA320WriteVLD1Latency5: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [3]; }
def CortexA320WriteVLD1Latency6: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 6; let ReleaseAtCycles = [4]; }

def : InstRW<[CortexA320WriteVLD1Latency3], (instregex "LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency3], (instregex "LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency4], (instregex "LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency4], (instregex "LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency5], (instregex "LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency4], (instregex "LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency6], (instregex "LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVLD1Latency3], (instregex "LD1i(8|16|32|64)$")>;                // single element
def : InstRW<[CortexA320WriteVLD1Latency3], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>; // replicate

def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency3], (instregex "LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency3], (instregex "LD1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency4], (instregex "LD1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency4], (instregex "LD1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency5], (instregex "LD1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency4], (instregex "LD1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency6], (instregex "LD1Fourv(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency3], (instregex "LD1i(8|16|32|64)_POST$")>;                // single element
def : InstRW<[WriteAdr, CortexA320WriteVLD1Latency3], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>; // replicate

//    2-element structures
def CortexA320WriteVLD2Latency3: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 3; let ReleaseAtCycles = [2]; }
def CortexA320WriteVLD2Latency4Release1: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [1]; }
def CortexA320WriteVLD2Latency4Release2: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def CortexA320WriteVLD2Latency4Release6: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [6]; }

def : InstRW<[CortexA320WriteVLD2Latency4Release1], (instregex "LD2Twov(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVLD2Latency4Release2], (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVLD2Latency4Release6], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVLD2Latency3], (instregex "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;

def : InstRW<[WriteAdr, CortexA320WriteVLD2Latency4Release1], (instregex "LD2Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD2Latency4Release2], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD2Latency4Release6], (instregex "LD2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD2Latency3], (instregex "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

//    3-element structures
def CortexA320WriteVLD3Latency4: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [3]; }
def CortexA320WriteVLD3Latency5Release6: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [6]; }
def CortexA320WriteVLD3Latency5Release7: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [7]; }

def : InstRW<[CortexA320WriteVLD3Latency5Release6], (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[CortexA320WriteVLD3Latency5Release7], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVLD3Latency4], (instregex "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;

def : InstRW<[WriteAdr, CortexA320WriteVLD3Latency5Release6], (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD3Latency5Release7], (instregex "LD3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD3Latency4], (instregex "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

//    4-element structures
def CortexA320WriteVLD4Latency4: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [4]; }
def CortexA320WriteVLD4Latency5Release7: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [7]; }
def CortexA320WriteVLD4Latency5Release8: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [8]; }
def CortexA320WriteVLD4Latency6: SchedWriteRes<[CortexA320UnitLdSt]> { let Latency = 6; let ReleaseAtCycles = [7]; }

def : InstRW<[CortexA320WriteVLD4Latency5Release7], (instregex "LD4Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVLD4Latency5Release8], (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVLD4Latency6], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVLD4Latency4], (instregex "LD4Rv(8b|16b|4h|8b|2s|4s|1d|2d)$")>;
 
def : InstRW<[WriteAdr, CortexA320WriteVLD4Latency5Release7], (instregex "LD4Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD4Latency5Release8], (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>; 
def : InstRW<[WriteAdr, CortexA320WriteVLD4Latency6], (instregex "LD4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVLD4Latency4], (instregex "LD4Rv(8b|16b|4h|8b|2s|4s|1d|2d)_POST$")>;
//---
// Vector Stores
//---
// 1 Element structures
def : InstRW<[CortexA320WriteVST1], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVST2], (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[CortexA320WriteVST2], (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[CortexA320WriteVST4], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST1], (instregex "ST1i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST1], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST1], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST4], (instregex "ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// 2 Element structures
def : InstRW<[CortexA320WriteVST2], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVST2], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[CortexA320WriteVST4], (instregex "ST2Twov(16b|8h|4s|2d)$")>;

def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST2i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST4], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// 3 Element structures
def : InstRW<[CortexA320WriteVST2], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVST4], (instregex "ST3Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;

def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST3i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST4], (instregex "ST3Threev(8b|4h|2s|1d|2d|16b|8h|4s|4d)_POST$")>;

// 4 Element structures
def : InstRW<[CortexA320WriteVST2], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[CortexA320WriteVST4], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;

def : InstRW<[WriteAdr, CortexA320WriteVST2], (instregex "ST4i(8|16|32|64)_POST$")>;
def : InstRW<[WriteAdr, CortexA320WriteVST4], (instregex "ST4Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

//---
// Floating Point Conversions, MAC, DIV, SQRT
//---
def : InstRW<[CortexA320WriteFPALU_F3], (instregex "^DUP(v2i64|v4i32|v8i16|v16i8)")>;
def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^XTN")>;
def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^FCVT[ALMNPZ][SU](S|U)?(W|X)")>;
def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^FCVT(X)?[ALMNPXZ](S|U|N)?v")>;

def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^(S|U)CVTF(S|U)(W|X)(H|S|D)")>;
def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^(S|U)CVTF(h|s|d)")>;
def : InstRW<[CortexA320WriteFPALU_F4], (instregex "^(S|U)CVTFv")>;

def : InstRW<[CortexA320WriteVMAC], (instregex "^FN?M(ADD|SUB).*")>;
def : InstRW<[CortexA320WriteVMAC], (instregex "^FML(A|S)v.*")>;
def : InstRW<[CortexA320WriteFDivHP], (instrs FDIVHrr)>;
def : InstRW<[CortexA320WriteFDivSP], (instrs FDIVSrr)>;
def : InstRW<[CortexA320WriteFDivDP], (instrs FDIVDrr)>;
def : InstRW<[CortexA320WriteFDivHP], (instregex "^FDIVv.*16$")>;
def : InstRW<[CortexA320WriteFDivSP], (instregex "^FDIVv.*32$")>;
def : InstRW<[CortexA320WriteFDivDP], (instregex "^FDIVv.*64$")>;
def : InstRW<[CortexA320WriteFSqrtHP], (instregex "^.*SQRT.*16$")>;
def : InstRW<[CortexA320WriteFSqrtSP], (instregex "^.*SQRT.*32$")>;
def : InstRW<[CortexA320WriteFSqrtDP], (instregex "^.*SQRT.*64$")>;

def : InstRW<[CortexA320WriteFPALU_F3], (instrs FCSELHrrr, FCSELSrrr, FCSELDrrr)>;

// Advanced SIMD integer instructions
// ASIMD absolute diff
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]ABDv(2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]ABDv(16i8|4i32|8i16)")>;
// ASIMD absolute diff accum
def : InstRW<[CortexA320Write<6, CortexA320UnitVALU>], (instregex "[SU]ABAL?v")>;
// ASIMD absolute diff long
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]ABDLv")>;
// ASIMD arith #1
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "(ADD|SUB|NEG)v",
  "[SU]R?HADDv", "[SU]HSUBv")>;
// ASIMD arith #2
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "ABSv(1i64|2i32|4i16|8i8)$",
  "[SU]ADDLPv(2i32_v1i64|4i16_v2i32|8i8_v4i16)$",
  "ADDPv(2i32|4i16|8i8)$")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(1i16|1i32|1i64|1i8|2i32|4i16|8i8)$")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "ABSv(2i64|4i32|8i16|16i8)$",
  "[SU]ADDLPv(16i8_v8i16|4i32_v2i64|8i16_v4i32)$",
  "ADDPv(16i8|2i64|4i32|8i16)$")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(16i8|2i64|4i32|8i16)$")>;
// ASIMD arith #3
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex  "SADDLv", "UADDLv", "SADDWv",
  "UADDWv", "SSUBLv", "USUBLv", "SSUBWv", "USUBWv")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex  "ADDHNv", "SUBHNv")>;
// ASIMD arith #5
def : InstRW<[CortexA320Write<8, CortexA320UnitVALU>], (instregex "RADDHNv", "RSUBHNv")>;
// ASIMD arith, reduce
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex  "ADDVv")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex  "SADDLVv", "UADDLVv")>;
// ASIMD compare #1
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(2i64|4i32|8i16|16i8)")>;
// ASIMD compare #2
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "CMTSTv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "CMTSTv(2i64|4i32|8i16|16i8)")>;
// ASIMD logical $1
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "(AND|EOR|NOT|ORN)v8i8",
  "(ORR|BIC)v(2i32|4i16|8i8)$", "MVNIv(2i|2s|4i16)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "(AND|EOR|NOT|ORN)v16i8",
  "(ORR|BIC)v(16i8|4i32|8i16)$", "MVNIv(4i32|4s|8i16)")>;
// ASIMD max/min, basic
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU](MIN|MAX)P?v(2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU](MIN|MAX)P?v(16i8|4i132|8i16)")>;
// SIMD max/min, reduce
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU](MAX|MIN)Vv")>;
// ASIMD multiply, by element
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "MULv(2i32|4i16|4i32|8i16)_indexed$",
  "SQR?DMULHv(1i16|1i32|2i32|4i16|4i32|8i16)_indexed$")>;
// ASIMD multiply
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs PMULv8i8)>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs PMULv16i8)>;
// ASIMD multiply accumulate
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "ML[AS]v(2i32|4i16|8i8)$")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "ML[AS]v(16i8|4i32|8i16)$")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "ML[AS]v(2i32|4i16|4i32|8i16)_indexed$")>;
// ASIMD multiply accumulate half
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "SQRDML[AS]H[vi]")>;
// ASIMD multiply accumulate long
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]ML[AS]Lv")>;
// ASIMD multiply accumulate long #2
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "SQDML[AS]L[iv]")>;
// ASIMD dot product
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]DOTv8i8")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]DOTv16i8")>;
// ASIMD dot product, by scalar
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]DOTlanev")>;
// ASIMD multiply long
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]MULLv", "SQDMULL[iv]")>;
// ASIMD polynomial (8x8) multiply long
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs PMULLv8i8, PMULLv16i8)>;
// ASIMD pairwise add and accumulate
def : InstRW<[CortexA320MCWrite<7, 2, CortexA320UnitVALU>], (instregex "[SU]ADALPv")>;
// ASIMD shift accumulate
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]SRA(d|v2i32|v4i16|v8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]SRAv(16i8|2i64|4i32|8i16)")>;
// ASIMD shift accumulate #2
def : InstRW<[CortexA320MCWrite<7, 2, CortexA320UnitVALU>], (instregex "[SU]RSRA[vd]")>;
// ASIMD shift by immed
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "SHLd$", "SHLv",
  "SLId$", "SRId$", "[SU]SHR[vd]", "SHRNv")>;
// ASIMD shift by immed
// SXTL and UXTL are aliases for SHLL
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[US]?SHLLv")>;
// ASIMD shift by immed #2
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]RSHR(d|v2i32|v4i16|v8i8)",
  "[SU]RSHRv(16i8|2i64|4i32|8i16)")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "RSHRNv(2i32|4i16|8i8)",
  "RSHRNv(16i8|4i32|8i16)")>;
// ASIMD shift by register
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]SHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]SHLv(2i64|4i32|8i16|16i8)")>;
// ASIMD shift by register #2
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]RSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "[SU]RSHLv(2i64|4i32|8i16|16i8)")>;

def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]QSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]QSHLv(2i64|4i32|8i16|16i8)")>;

def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]QRSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "[SU]QRSHLv(2i64|4i32|8i16|16i8)")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[CortexA320MCWrite<4, 0, CortexA320UnitVMC>], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^SHA1(H|SU0|SU1)")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[CortexA320MCWrite<4, 0, CortexA320UnitVMC>], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[CortexA320MCWrite<4, 0, CortexA320UnitVMC>], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[CortexA320MCWrite<9, 0, CortexA320UnitVMC>], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instrs BCAX, EOR3)>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs XAR)>;
def : InstRW<[CortexA320MCWrite<9, 0, CortexA320UnitVMC>], (instrs RAX1)>;


// Crypto SM3 ops
def : InstRW<[CortexA320MCWrite<9, 0, CortexA320UnitVMC>], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                                            "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[CortexA320MCWrite<9, 0, CortexA320UnitVMC>], (instrs SM4E, SM4ENCKEY)>;

// CRC
// -----------------------------------------------------------------------------

def : InstRW<[CortexA320MCWrite<2, 0, CortexA320UnitMAC>], (instregex "^CRC32")>;

// SVE Predicate instructions

// Loop control, based on predicate
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs BRKA_PPmP, BRKA_PPzP,
                                                  BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs BRKN_PPzP, BRKPA_PPzPP, BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs BRKNS_PPzP)>;
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs BRKPAS_PPzPP, BRKPBS_PPzPP)>;


// Loop control, based on GPR
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]")>;

def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Loop terminate
def : InstRW<[CortexA320Write<1, CortexA320UnitALU>], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// Predicate counting scalar
def : InstRW<[CortexA320Write<1, CortexA320UnitALU>], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;

def : InstRW<[CortexA320Write<3, CortexA320UnitALU>],
             (instregex "^CNT[BHWD]_XPiI")>;

def : InstRW<[CortexA320Write<3, CortexA320UnitALU>],
             (instregex "^(INC|DEC)[BHWD]_XPiI")>;

def : InstRW<[CortexA320Write<5, CortexA320UnitALU>],
             (instregex "^(SQINC|SQDEC|UQINC|UQDEC)[BHWD]_[XW]Pi(Wd)?I")>;

// Predicate counting scalar, active predicate
def : InstRW<[CortexA320Write<1, CortexA320UnitVALU>],
             (instregex "^CNTP_XPP_[BHSD]")>;

def : InstRW<[CortexA320Write<1, CortexA320UnitVALU>],
             (instregex "^(DEC|INC)P_XP_[BHSD]")>;

def : InstRW<[CortexA320Write<9, CortexA320UnitVALU>],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]",
                        "^(SQDEC|SQINC|UQDEC|UQINC)P_XPWd_[BHSD]")>;


// Predicate counting vector, active predicate
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;

// Predicate logical
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;

// Predicate test
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs PTEST_PP)>;

// Predicate transpose
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^TRN[12]_PPP_[BHSDQ]")>;

// Predicate unpack and widen
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[CortexA320Write<2, CortexA320UnitVALU>], (instregex "^(ZIP|UZP)[12]_PPP_[BHSDQ]")>;


// SVE integer instructions
// -----------------------------------------------------------------------------
// Arithmetic, absolute diff
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^[SU]ABD_(ZPmZ|ZPZZ)_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[CortexA320MCWrite<6, 2, CortexA320UnitVALU>], (instregex "^[SU]ABA_ZZZ_[BHSD]")>;

// Arithmetic, absolute diff accum long
def : InstRW<[CortexA320MCWrite<6, 2, CortexA320UnitVALU>], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]")>;

// Arithmetic, absolute diff long
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]")>;

// Arithmetic, basic
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(ABS|CNOT|NEG)_ZPmZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPZZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZI_[BHSD]",
                        "^ADR_[SU]XTW_ZZZ_D_[0123]",
                        "^ADR_LSL_ZZZ_[SD]_[0123]",
                        "^[SU]H(ADD|SUB|SUBR)_ZPmZ_[BHSD]")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^[SU](ADD|SUB)[LW][BT]_ZZZ_[HSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]")>;

// Arithmetic, complex
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^SQ(ABS|NEG)_ZPmZ_[BHSD]",
                        "^SQ(ADD|SUB|SUBR)_ZPmZ_?[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;
def : InstRW<[CortexA320Write<8, CortexA320UnitVALU>],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]")>;

// Arithmetic, large integer
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]")>;

// Arithmetic, pairwise add
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^ADDP_ZPmZ_[BHSD]")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[CortexA320MCWrite<7, 2, CortexA320UnitVALU>], (instregex "^[SU]ADALP_ZPmZ_[HSD]")>;

// Arithmetic, shift
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;
// Arithmetic, shift right for divide
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^ASRD_ZPmI_[BHSD]",
                        "^ASRD_ZPZI_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^(SSRA|USRA)_ZZI_[BHSD]")>;

def : InstRW<[CortexA320MCWrite<7, 2, CortexA320UnitVALU>],
             (instregex "^(SRSRA|URSRA)_ZZI_[BHSD]")>;


// Arithmetic, shift by immediate
// Arithmetic, shift by immediate and insert
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(SHRNB|SHRNT|SSHLLB|SSHLLT|USHLLB|USHLLT|SLI|SRI)_ZZI_[BHSD]")>;

// Arithmetic, shift complex
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift rounding
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^(SRSHL|SRSHR|SRSHLR|URSHL|URSHLR|URSHR)_(ZPmZ|ZPZZ|ZPZI)_[BHSD]",
                        "^[SU]RSHR_ZPmI_[BHSD]")>;

// Bit manipulation
def : InstRW<[CortexA320MCWrite<13, 12, CortexA320UnitVMC>],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_B")>;

def : InstRW<[CortexA320MCWrite<21, 20, CortexA320UnitVMC>],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_H")>;

def : InstRW<[CortexA320MCWrite<37, 36, CortexA320UnitVMC>],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_S")>;

def : InstRW<[CortexA320MCWrite<68, 67, CortexA320UnitVMC>],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_D")>;


// Bitwise select
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ")>;

// Count/reverse bits
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^(CLS|CLZ|RBIT)_ZPmZ_[BHSD]")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^CNT_ZPmZ_[BH]")>;
def : InstRW<[CortexA320Write<8, CortexA320UnitVALU>], (instregex "^CNT_ZPmZ_S")>;
def : InstRW<[CortexA320Write<12, CortexA320UnitVALU>], (instregex "^CNT_ZPmZ_D")>;
// Broadcast logical bitmask immediate to vector
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[CortexA320Write<5, CortexA320UnitVALU>],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;

// Complex add
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^CADD_ZZI_[BHSD]")>;

def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^SQCADD_ZZI_[BHSD]")>;

// Complex dot product 8-bit element
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^CMLA_ZZZ_[BHS]",
                                            "^CMLA_ZZZI_[HS]")>;

// Complex multiply-add D element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form
def : InstRW<[CortexA320MCWrite<8, 2, CortexA320UnitVALU>], (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]",
                                            "^COMPACT_ZPZ_[SD]",
                                            "^SPLICE_ZPZZ?_[BHSD]")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]CVTF_ZPmZ_Dto[SD]")>;

// Convert to floating point, 64b to half
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]CVTF_ZPmZ_DtoH")>;

// Convert to floating point, 32b to single or half
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 32b to double
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 16b to half
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],(instregex "^CPY_ZPmR_[BHSD]")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^CPY_ZPm[IV]_[BHSD]",
                                           "^CPY_ZPzI_[BHSD]")>;

// Divides, 32 bit
def : InstRW<[CortexA320MCWrite<15, 12, CortexA320UnitVMC>], (instregex "^[SU]DIVR?_(ZPmZ|ZPZZ)_S")>;

// Divides, 64 bit
def : InstRW<[CortexA320MCWrite<26, 23, CortexA320UnitVMC>], (instregex "^[SU]DIVR?_(ZPmZ|ZPZZ)_D")>;

// Dot product, 8 bit
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^[SU]DOT_ZZZI?_S")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^[SU]DOT_ZZZI?_D")>;

// Duplicate, immediate and indexed form
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^DUP_ZI_[BHSD]",
                                           "^DUP_ZZI_[BHSDQ]")>;

// Duplicate, scalar form
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^DUP_ZR_[BHSD]")>;

// Extend, sign or zero
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                            "^[SU]XTH_ZPmZ_[SD]",
                                            "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instrs EXT_ZZI, EXT_ZZI_CONSTRUCTIVE, EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]",
                                            "^SQXTUN[BT]_ZZ_[BHS]")>;

// Extract/insert operation, SIMD and FP scalar form
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^LAST[AB]_VPZ_[BHSD]",
                                            "^INSR_ZV_[BHSD]")>;

// Extract/insert operation, scalar
def : InstRW<[CortexA320MCWrite<8, 2, CortexA320UnitVALU>], (instregex "^LAST[AB]_RPZ_[BHSD]",
                                                "^INSR_ZR_[BHSD]")>;

// Histogram operations
def : InstRW<[CortexA320MCWrite<8, 2, CortexA320UnitVALU>], (instregex "^HISTCNT_ZPzZZ_[SD]",
                                                  "^HISTSEG_ZZZ")>;

// Horizontal operations, B, H, S form, immediate operands only
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^INDEX_II_[BHS]")>;

// Horizontal operations, B, H, S form, scalar, immediate operands/ scalar
// operands only / immediate, scalar operands
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^INDEX_(IR|RI|RR)_[BHS]")>;

// Horizontal operations, D form, immediate operands only
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs INDEX_II_D)>;

// Horizontal operations, D form, scalar, immediate operands)/ scalar operands
// only / immediate, scalar operands
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^INDEX_(IR|RI|RR)_D")>;

// Logical
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|EOR|ORR)_ZZZ",
                        "^(AND|BIC|EOR|NOT|ORR)_ZPmZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_ZPZZ_[BHSD]")>;

def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^EOR(BT|TB)_ZZZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                           "^[SU](MAX|MIN)P?_(ZPmZ|ZPZZ)_[BHSD]")>;

// Matching operations
def : InstRW<[CortexA320MCWrite<9, 2, CortexA320UnitVALU>], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Matrix multiply-accumulate
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]",
                                           "^MOVPRFX_ZZ")>;

// Multiply, B, H, S element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ|ZPZZ)_[BHS]",
                                            "^[SU]MULH_(ZPmZ|ZZZ|ZPZZ)_[BHS]")>;

// Multiply, D element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ|ZPZZ)_D",
                                            "^[SU]MULH_(ZPmZ|ZZZ|ZPZZ)_D")>;

// Multiply long
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^[SU]MULL[BT]_ZZZI_[SD]",
                                            "^[SU]MULL[BT]_ZZZ_[HSD]")>;

// Multiply accumulate, B, H, S element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^ML[AS]_(ZZZI|ZPZZZ)_[BHS]",
                                            "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]")>;

// Multiply accumulate, D element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^ML[AS]_(ZZZI|ZPZZZ)_D",
                                            "^(ML[AS]|MAD|MSB)_ZPmZZ_D")>;

// Multiply accumulate long
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]",
                                            "^[SU]ML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQDML[AS](LB|LT|LBT)_ZZZ_[HSD]",
                                            "^SQDML[AS](LB|LT)_ZZZI_[SD]")>;

// Multiply saturating doubling high, B, H, S element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQDMULH_ZZZ_[BHS]",
                                            "^SQDMULH_ZZZI_[HS]")>;

// Multiply saturating doubling high, D element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQDMULL[BT]_ZZZ_[HSD]",
                                            "^SQDMULL[BT]_ZZZI_[SD]")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S
// element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQRDML[AS]H_ZZZ_[BHS]",
                                            "^SQRDCMLAH_ZZZ_[BHS]",
                                            "^SQRDML[AS]H_ZZZI_[HS]",
                                            "^SQRDCMLAH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element
// size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQRDML[AS]H_ZZZI?_D",
                                            "^SQRDCMLAH_ZZZ_D")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQRDMULH_ZZZ_[BHS]",
                                            "^SQRDMULH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex, D element size
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^SQRDMULH_ZZZI?_D")>;

// Multiply/multiply long, (8x8) polynomial
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^PMUL_ZZZ_B")>;

def : InstRW<[CortexA320Write<9, CortexA320UnitVMC>], (instregex "^PMULL[BT]_ZZZ_[HDQ]")>;


// Predicate counting vector
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>],
             (instregex "^(DEC|INC)[HWD]_ZPiI")>;
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)[HWD]_ZPiI")>;

// Reciprocal estimate
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_B")>;

// Reduction, arithmetic, H form
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^(ANDV|EORV|ORV)_VPZ_[BHSD]")>;

// Reverse, vector
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^REV_ZZ_[BHSD]",
                                           "^REVB_ZPmZ_[HSD]",
                                           "^REVH_ZPmZ_[SD]",
                                           "^REVW_ZPmZ_D")>;

// Select, vector form
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^SEL_ZPZZ_[BHSD]")>;

// Table lookup
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^TBL_ZZZZ?_[BHSD]")>;

// Table lookup extension
def : InstRW<[CortexA320Write<8, CortexA320UnitVALU>], (instregex "^TBX_ZZZ_[BHSD]")>;

// Transpose, vector form
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^TRN[12]_ZZZ_[BHSDQ]")>;

// Unpack and extend
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]")>;

// Zip/unzip
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]")>;

// SVE floating-point instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FAB[SD]_ZPmZ_[HSD]",
                                                                  "^FAB[SD]_ZPZZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ|ZPZI|ZPZZ)_[HSD]",
                                           "^FADDP_ZPmZZ_[HSD]",
                                           "^FNEG_ZPmZ_[HSD]",
                                           "^FSUBR_(ZPm[IZ]|ZPZ[IZ])_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[CortexA320MCWrite<32, 29, CortexA320UnitVALU>], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[CortexA320MCWrite<16, 13, CortexA320UnitVALU>], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[CortexA320MCWrite<8, 5, CortexA320UnitVALU>], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[CortexA320Write<5, CortexA320UnitVALU>], (instregex "^FACG[ET]_PPzZZ_[HSD]",
                                            "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]",
                                            "^FCM(LE|LT)_PPzZ0_[HSD]",
                                            "^FCMUO_PPzZZ_[HSD]")>;

// Floating point complex add
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCADD_ZPmZ_[HSD]")>;

// Floating point complex multiply add
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FCMLA_ZPmZZ_[HSD]",
                                           "^FCMLA_ZZZI_[HS]")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                            "^FCVTLT_ZPmZ_HtoS",
                                            "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32
// or F64 to F16)
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                            "^FCVTLT_ZPmZ_StoD",
                                            "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCVTX_ZPmZ_DtoS", "FCVTXNT_ZPmZ_DtoS")>;

// Floating point base2 log, F16
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^FCPY_ZPmI_[HSD]",
                                           "^FDUP_ZI_[HSD]")>;

// Floating point divide, F16
def : InstRW<[CortexA320MCWrite<8, 5, CortexA320UnitVMC>], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[CortexA320MCWrite<13, 10, CortexA320UnitVMC>], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[CortexA320MCWrite<22, 19, CortexA320UnitVMC>], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^F(MAX|MIN)(NM)?_(ZPm[IZ]|ZPZZ|ZPZI)_[HSD]")>;

// Floating point multiply
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^(FSCALE|FMULX)_(ZPmZ|ZPZZ)_[HSD]",
                                           "^FMUL_(ZPm[IZ]|ZZZI?|ZPZI|ZPZZ)_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>],
             (instregex "^FML[AS]_(ZPmZZ|ZZZI|ZPZZZ)_[HSD]",
                        "^(FMAD|FNMAD|FNML[AS]|FN?MSB)_(ZPmZZ|ZPZZZ)_[HSD]")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FML[AS]L[BT]_ZZZI?_SHH")>;

// Floating point reciprocal estimate, F16
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FRECPE_ZZ_H", "^FRECPX_ZPmZ_H",
                                         "^FRSQRTE_ZZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FRECPE_ZZ_S", "^FRECPX_ZPmZ_S",
                                         "^FRSQRTE_ZZ_S")>;
// Floating point reciprocal estimate, F64
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>],(instregex "^FRECPE_ZZ_D", "^FRECPX_ZPmZ_D",
                                         "^FRSQRTE_ZZ_D")>;

// Floating point reciprocal step
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]")>;

// Floating point reduction, F16
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^(FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_[HSD]")>;

// Floating point reduction, F32
def : InstRW<[CortexA320MCWrite<12, 11, CortexA320UnitVALU>],
             (instregex "^FADDV_VPZ_H")>;

def : InstRW<[CortexA320MCWrite<8, 5, CortexA320UnitVALU>],
             (instregex "^FADDV_VPZ_S")>;

def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>],
             (instregex "^FADDV_VPZ_D")>;


// Floating point round to integral, F16
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[CortexA320MCWrite<11, 5, CortexA320UnitVMC>], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[CortexA320MCWrite<14, 9, CortexA320UnitVMC>], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64
def : InstRW<[CortexA320MCWrite<25, 19, CortexA320UnitVMC>], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FEXPA_ZZ_[HSD]")>;

// Floating point trigonometric multiply add
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FTMAD_ZZI_[HSD]")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^FTSMUL_ZZZ_[HSD]")>;
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^FTSSEL_ZZZ_[HSD]")>;


// SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[CortexA320Write_11cyc_1VMAC_1VALU], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[CortexA320Write_16cyc_1VMAC_1VALU], (instrs BFMMLA_ZZZ)>;

// Multiply accumulate long
def : InstRW<[CortexA320Write<4, CortexA320UnitVMAC>], (instregex "^BFMLAL[BT]_ZZZ(I)?")>;

// SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LD1[BHWD]_IMM$",
                                           "^LD1S?B_[HSD]_IMM$",
                                           "^LD1S?H_[SD]_IMM$",
                                           "^LD1S?W_D_IMM$" )>;
// Contiguous load, scalar + scalar
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LD1[BHWD]$",
                                             "^LD1S?B_[HSD]$",
                                             "^LD1S?H_[SD]$",
                                             "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LD1R[BHWD]_IMM$",
                                           "^LD1RSW_IMM$",
                                           "^LD1RS?B_[HSD]_IMM$",
                                           "^LD1RS?H_[SD]_IMM$",
                                           "^LD1RS?W_D_IMM$",
                                           "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LDNT1[BHWD]_ZRI$")>;

// Non temporal load, scalar + scalar
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LDNT1[BHWD]_ZRR$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[CortexA320MCWrite<9, 9, CortexA320UnitLdSt>], (instregex "^LDNT1[BHW]_ZZR_S$",
                                              "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LDFF1[BHWD]$",
                                              "^LDFF1S?B_[HSD]$",
                                              "^LDFF1S?H_[SD]$",
                                              "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[CortexA320Write<3, CortexA320UnitLdSt>], (instregex "^LDNF1[BHWD]_IMM$",
                                           "^LDNF1S?B_[HSD]_IMM$",
                                           "^LDNF1S?H_[SD]_IMM$",
                                           "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[CortexA320MCWrite<3, 1, CortexA320UnitLdSt>], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[CortexA320MCWrite<3, 2, CortexA320UnitLdSt>], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[CortexA320MCWrite<5, 3, CortexA320UnitLdSt>], (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[CortexA320MCWrite<5, 3, CortexA320UnitLdSt>], (instregex "^LD3[BHWD]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[CortexA320MCWrite<5, 3, CortexA320UnitLdSt>], (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[CortexA320MCWrite<5, 3, CortexA320UnitLdSt>], (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[CortexA320MCWrite<9, 9, CortexA320UnitLdSt>], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                              "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                              "^GLD(FF)?1D_IMM$")>;

// Gather load, 64-bit element size
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>],
             (instregex "^GLD(FF)?1S?[BHW]_D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1S?[BHW]_D(_SCALED)?$",
                        "^GLD(FF)?1D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1D(_SCALED)?$")>;

// Gather load, 32-bit scaled offset
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>],
             (instregex "^GLD(FF)?1S?[HW]_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;

// Gather load, 32-bit unpacked unscaled offset
def : InstRW<[CortexA320MCWrite<7, 7, CortexA320UnitLdSt>], (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                                              "^GLD(FF)?1W_[SU]XTW$")>;

def : InstRW<[CortexA320Write<0, CortexA320UnitVALU>], (instregex "^PRF(B|H|W|D).*")>;
// SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[CortexA320VSt0], (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[CortexA320VSt0], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[CortexA320VSt0], (instregex "^ST1[BHWD]_IMM$",
                                                "^ST1B_[HSD]_IMM$",
                                                "^ST1H_[SD]_IMM$",
                                                "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[CortexA320VSt0], (instregex "^ST1H(_[SD])?$")>;
def : InstRW<[CortexA320VSt0], (instregex "^ST1[BWD]$",
                                                "^ST1B_[HSD]$",
                                                "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[CortexA320VSt<11>], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[CortexA320VSt<11>], (instrs ST2H)>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[CortexA320VSt<11>], (instregex "^ST2[BWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[CortexA320VSt<25>], (instregex "^ST3[BHW]_IMM$")>;
def : InstRW<[CortexA320VSt<14>], (instregex "^ST3D_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[CortexA320VSt<25>], (instregex "^ST3[BHW]$")>;
def : InstRW<[CortexA320VSt<14>], (instregex "^ST3D$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[CortexA320VSt<50>], (instregex "^ST4[BHW]_IMM$")>;
def : InstRW<[CortexA320VSt<25>], (instregex "^ST4D_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[CortexA320VSt<50>], (instregex "^ST4[BHW]$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[CortexA320VSt<25>], (instregex "^ST4D$")>;

// Non temporal store, scalar + imm
def : InstRW<[CortexA320VSt0], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[CortexA320VSt0], (instrs STNT1H_ZRR)>;
def : InstRW<[CortexA320VSt0], (instregex "^STNT1[BWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[CortexA320VSt<9>], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[CortexA320VSt<7>], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[CortexA320VSt<9>], (instregex "^SST1[BH]_S_IMM$",
                                                "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[CortexA320VSt<7>], (instregex "^SST1[BHW]_D_IMM$",
                                                "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[CortexA320VSt<8>],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[CortexA320VSt<8>], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                                "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[CortexA320VSt<8>], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                                "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[CortexA320VSt<8>], (instregex "^SST1[BH]_S_[SU]XTW$",
                                                "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[CortexA320VSt<8>], (instregex "^SST1[HW]_D_SCALED$",
                                                "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[CortexA320VSt<8>], (instregex "^SST1[BHW]_D$",
                                                "^SST1D$")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[CortexA320Write<1, CortexA320UnitALU>], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[CortexA320Write<3, CortexA320UnitALU>], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[CortexA320Write<3, CortexA320UnitALU>], (instrs RDFFRS_PPz)>;

// Set first fault register
// Write to first fault register
def : InstRW<[CortexA320Write<1, CortexA320UnitALU>], (instrs SETFFR, WRFFR)>;

// SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[CortexA320Write<3, CortexA320UnitVALU>], (instregex "^AES[DE]_ZZZ_B$",
                                           "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[CortexA320Write<4, CortexA320UnitVALU>], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                            "^XAR_ZZZI_[BHSD]$")>;

def : InstRW<[CortexA320MC_RC0Write<9, CortexA320UnitVMC>], (instregex "^RAX1_ZZZ_D$")>;

// Crypto SM4 ops
def : InstRW<[CortexA320MC_RC0Write<9, CortexA320UnitVMC>], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

}
