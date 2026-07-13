//==- AArch64Sched1Nano.td - ARM C1-Nano Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for the ARM C1-Nano processor.
// Information is taken from the Arm C1-Nano Core Software Optimization Guide
//
// https://developer.arm.com/documentation/109590/0001
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//

// C1-Nano machine model for scheduling and other instruction cost heuristics.
def C1NanoModel : SchedMachineModel {
  let MicroOpBufferSize = 0;  // The C1-Nano is an in-order processor
  let IssueWidth = 3;         // It dual-issues under most circumstances
  let LoadLatency = 3;        // Cycles for loads to access the cache.
                              // 2 is best case, 4 is normal case.
                              // 3 seems to be a good tradeoff
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 1;      // Covers instructions applicable to C1-Nano.

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
                                                    SVE2p1Unsupported.F,
                                                    [HasSVE2p1, HasSVEB16B16,
                                                     HasCPA, HasCSSC]);
}


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types

let SchedModel = C1NanoModel in {

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since the
// C1-Nano is in-order.
let BufferSize = 0 in {
  def C1NanoUnitALU0   : ProcResource<1>;    // Int ALU0
  def C1NanoUnitALU1   : ProcResource<1>;    // Int ALU1
  def C1NanoUnitMAC    : ProcResource<1>;    // Int MAC, 64-bit wide
  def C1NanoUnitDiv    : ProcResource<1>;    // Int Division, not pipelined
  // There are 2 LS pipes, 1 for Load/Store; 1 for Load only
  def C1NanoUnitLdSt   : ProcResource<1>;    // Load/Store shared pipe
  def C1NanoUnitLd1    : ProcResource<1>;    // Load pipe
  def C1NanoUnitB      : ProcResource<1>;    // Branch
  def C1NanoUnitPAC    : ProcResource<1>;    // Pointer Authentication (PAC) pipe

  // The FP DIV/SQRT instructions execute totally differently from the FP ALU
  // instructions, which can mostly be dual-issued; that's why for now we model
  // them with 2 resources.
  def C1NanoUnitVALU0  : ProcResource<1>;    // SIMD/FP/SVE ALU0
  def C1NanoUnitVALU1  : ProcResource<1>;    // SIMD/FP/SVE ALU1
  def C1NanoUnitVMAC0  : ProcResource<1>;    // SIMD/FP/SVE MAC0
  def C1NanoUnitVMAC1  : ProcResource<1>;    // SIMD/FP/SVE MAC1
  def C1NanoUnitVMC    : ProcResource<1>;    // SIMD/FP/SVE multicycle instrs  (e.g Div, SQRT, cryptography)
}

def C1NanoUnitLd     : ProcResGroup<[C1NanoUnitLdSt, C1NanoUnitLd1]>;
def C1NanoUnitVALU   : ProcResGroup<[C1NanoUnitVALU0, C1NanoUnitVALU1]>;
def C1NanoUnitALU    : ProcResGroup<[C1NanoUnitALU0, C1NanoUnitALU1]>;
def C1NanoUnitVMAC   : ProcResGroup<[C1NanoUnitVMAC0, C1NanoUnitVMAC1]>;

// Workaround for throughput being doubled on a single resource
def C1NanoUnit2VMC : ProcResGroup<[C1NanoUnitVMC, C1NanoUnitVMC]>;
def C1NanoUnit4VMC : ProcResGroup<[C1NanoUnitVMC, C1NanoUnitVMC, C1NanoUnitVMC, C1NanoUnitVMC]>;
def C1NanoUnit2VALU0: ProcResGroup<[C1NanoUnitVALU0, C1NanoUnitVALU0]>;

// These latencies are modeled without taking into account forwarding paths
// (the software optimisation guide lists latencies taking into account
// typical forwarding paths).
def : WriteRes<WriteImm, [C1NanoUnitALU]> { let Latency = 1; }    // MOVN, MOVZ
def : WriteRes<WriteI, [C1NanoUnitALU]> { let Latency = 1; }      // ALU
def : WriteRes<WriteISReg, [C1NanoUnitALU]> { let Latency = 1; }  // ALU of Shifted-Reg
def : WriteRes<WriteIEReg, [C1NanoUnitALU]> { let Latency = 1; }  // ALU of Extended-Reg
def : WriteRes<WriteExtr, [C1NanoUnitALU]> { let Latency = 2; }   // EXTR from a reg pair
def : WriteRes<WriteIS, [C1NanoUnitALU]> { let Latency = 2; }     // Shift/Scale

// Define commonly used read types.
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

// -----------------------------------------------------------------------------
// Forwarding

def C1NanoWrite_ADRP  : SchedWriteRes<[C1NanoUnitALU]> { let Latency = 1; }
def C1NanoWrite_LDR   : SchedWriteRes<[C1NanoUnitLd]> { let Latency = 2; }
def C1NanoRd_LDR   : SchedReadAdvance<2, [C1NanoWrite_ADRP, C1NanoWrite_LDR]>;
def : SchedAlias<ReadAdrBase, C1NanoRd_LDR>;

class C1NanoWriteVMACBypass : SchedWriteRes<[C1NanoUnitVMAC]> {
  let Latency = 4;
}

def C1NanoWriteVMACB64   : C1NanoWriteVMACBypass;
def C1NanoWriteVMACB128  : C1NanoWriteVMACBypass;
def C1NanoWriteVMACH64   : C1NanoWriteVMACBypass;
def C1NanoWriteVMACH128  : C1NanoWriteVMACBypass;
def C1NanoWriteVMACS64   : C1NanoWriteVMACBypass;
def C1NanoWriteVMACS128  : C1NanoWriteVMACBypass;
def C1NanoWriteVMACD128  : C1NanoWriteVMACBypass;

// SIMD MAC forwarding - reduces latency to 1 cycle when conditions are met
// for accumulator dependencies with matching destination element/register size.
def C1NanoReadVMACB64   : SchedReadAdvance<3, [C1NanoWriteVMACB64]>;
def C1NanoReadVMACB128  : SchedReadAdvance<3, [C1NanoWriteVMACB128]>;
def C1NanoReadVMACH64   : SchedReadAdvance<3, [C1NanoWriteVMACH64]>;
def C1NanoReadVMACH128  : SchedReadAdvance<3, [C1NanoWriteVMACH128]>;
def C1NanoReadVMACS64   : SchedReadAdvance<3, [C1NanoWriteVMACS64]>;
def C1NanoReadVMACS128  : SchedReadAdvance<3, [C1NanoWriteVMACS128]>;
def C1NanoReadVMACD128  : SchedReadAdvance<3, [C1NanoWriteVMACD128]>;
def : ReadAdvance<ReadVMAC, 0>;
def : ReadAdvance<ReadVMACAccum, 0>;

// -----------------------------------------------------------------------------

// Define generic 2 micro-op types
def C1NanoWrite_10c_1VMAC_1VALU : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVMAC]> {
  let Latency     = 10;
  let NumMicroOps = 2;
}

def C1NanoWrite_14c_2VMAC_2VALU : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU, C1NanoUnitVMAC, C1NanoUnitVMAC]> {
  let Latency     = 14;
  let NumMicroOps = 2;
}

def C1NanoWrite_0c        : SchedWriteRes<[]> { let Latency = 0; }
def C1NanoWrite_1B        : SchedWriteRes<[C1NanoUnitB]> {}
def C1NanoWrite_1c_1ALU   : SchedWriteRes<[C1NanoUnitALU]> { let Latency = 1; }
def C1NanoWrite_2c_1ALU   : SchedWriteRes<[C1NanoUnitALU]> { let Latency = 2; }
def C1NanoWrite_1c_1ALU0  : SchedWriteRes<[C1NanoUnitALU0]> { let Latency = 1; }
def C1NanoWrite_2c_1ALU0  : SchedWriteRes<[C1NanoUnitALU0]> { let Latency = 2; }
def C1NanoWrite_3c_1ALU0  : SchedWriteRes<[C1NanoUnitALU0]> { let Latency = 3; }
def C1NanoWrite_5c_1ALU0  : SchedWriteRes<[C1NanoUnitALU0]> { let Latency = 5; }
def C1NanoWrite_2c_1Ld    : SchedWriteRes<[C1NanoUnitLd]> { let Latency = 2; }
def C1NanoWrite_2c_1Ld_SI : SchedWriteRes<[C1NanoUnitLd]> { let Latency = 2; let SingleIssue = 1; }
def C1NanoWrite_3c_1Ld    : SchedWriteRes<[C1NanoUnitLd]> { let Latency = 3; }
def C1NanoWrite_1c_1LdSt  : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 1; }
def C1NanoWrite_3c_1LdSt  : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 3; }
def C1NanoWrite_2c_1MAC   : SchedWriteRes<[C1NanoUnitMAC]> { let Latency = 2; }
def C1NanoWrite_2c_1PAC   : SchedWriteRes<[C1NanoUnitPAC]> { let Latency = 2; }
def C1NanoWrite_4c_1PAC   : SchedWriteRes<[C1NanoUnitPAC]> { let Latency = 4; }
def C1NanoWrite_5c_1PAC   : SchedWriteRes<[C1NanoUnitPAC]> { let Latency = 5; }
def C1NanoWrite_2c_1VALU  : SchedWriteRes<[C1NanoUnitVALU]> { let Latency = 2; }
def C1NanoWrite_3c_1VALU  : SchedWriteRes<[C1NanoUnitVALU]> { let Latency = 3; }
def C1NanoWrite_3c_2VALU  : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 3; }
def C1NanoWrite_4c_1VALU  : SchedWriteRes<[C1NanoUnitVALU]> { let Latency = 4; }
def C1NanoWrite_4c_2VALU  : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 4; }
def C1NanoWrite_4c_1VALU0 : SchedWriteRes<[C1NanoUnitVALU0]> { let Latency = 4; }
def C1NanoWrite_4c_1VMAC  : SchedWriteRes<[C1NanoUnitVMAC]> { let Latency = 4; }

let BeginGroup = 1 in {
  def C1NanoWrite_1c_2ALU_1rc     : SchedWriteRes<[C1NanoUnitALU, C1NanoUnitALU]> { let Latency = 1; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_1c_2ALU_2rc     : SchedWriteRes<[C1NanoUnitALU, C1NanoUnitALU]> { let Latency = 1; let ReleaseAtCycles = [2, 2]; }
  def C1NanoWrite_2c_2ALU_2rc     : SchedWriteRes<[C1NanoUnitALU, C1NanoUnitALU]> { let Latency = 2; let ReleaseAtCycles = [2, 2]; }
  def C1NanoWrite_4c_2ALU_3rc     : SchedWriteRes<[C1NanoUnitALU,C1NanoUnitALU]> { let Latency = 4; let ReleaseAtCycles = [3, 3]; }
  def C1NanoWrite_6c_1MAC_4rc     : SchedWriteRes<[C1NanoUnitMAC]> { let Latency = 6; let ReleaseAtCycles = [4]; }
  def C1NanoWrite_2c_2Ld_4rc      : SchedWriteRes<[C1NanoUnitLd, C1NanoUnitLd]> { let Latency = 2; let ReleaseAtCycles = [4, 4]; }
  def C1NanoWrite_3c_2Ld         : SchedWriteRes<[C1NanoUnitLd, C1NanoUnitLd]> { let Latency = 3; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_1c_1LdSt_2rc    : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [2]; }
  def C1NanoWrite_3c_1LdSt_2rc    : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 3; let ReleaseAtCycles = [2]; }
  def C1NanoWrite_7c_1LdSt_6rc    : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 7; let ReleaseAtCycles = [6]; }
  def C1NanoWrite_7c_1LdSt_7rc    : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 7; let ReleaseAtCycles = [7]; }
  def C1NanoWrite_9c_1LdSt_7rc    : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 9; let ReleaseAtCycles = [7]; }
  def C1NanoWrite_1c_2VALU_1rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 1; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_2c_2VALU_1rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 2; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_3c_1VALU_0rc   : SchedWriteRes<[C1NanoUnitVALU]> { let Latency = 3; }
  def C1NanoWrite_3c_2VALU_1rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 3; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_4c_2VALU_1rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 4; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_4c_2VALU_4rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 4; let ReleaseAtCycles = [4, 4]; }
  def C1NanoWrite_5c_2VALU_1rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 5; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_5c_2VALU_2rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 5; let ReleaseAtCycles = [2, 2]; }
  def C1NanoWrite_5c_2VALU_3rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 5; let ReleaseAtCycles = [3, 3]; }
  def C1NanoWrite_5c_2VALU_5rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 5; let ReleaseAtCycles = [5, 5]; }
  def C1NanoWrite_6c_2VALU_3rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 6; let ReleaseAtCycles = [3, 3]; }
  def C1NanoWrite_6c_2VALU_4rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 6; let ReleaseAtCycles = [4, 4]; }
  def C1NanoWrite_7c_2VALU_4rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 7; let ReleaseAtCycles = [4, 4]; }
  def C1NanoWrite_8c_2VALU_4rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 8; let ReleaseAtCycles = [4, 4]; }
  def C1NanoWrite_8c_1VALU_5rc    : SchedWriteRes<[C1NanoUnitVALU]> { let Latency = 8; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_8c_2VALU_5rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 8; let ReleaseAtCycles = [5, 5]; }
  def C1NanoWrite_9c_2VALU_7rc    : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 9; let ReleaseAtCycles = [7, 7]; }
  def C1NanoWrite_16c_2VALU_9rc   : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 16; let ReleaseAtCycles = [9, 9]; }
  def C1NanoWrite_23c_2VALU_25rc  : SchedWriteRes<[C1NanoUnitVALU, C1NanoUnitVALU]> { let Latency = 32; let ReleaseAtCycles = [25, 25]; }
  def C1NanoWrite_4c_2VALU_1rc0_2 : SchedWriteRes<[C1NanoUnit2VALU0]> { let Latency = 4; let ReleaseAtCycles = [1]; }
  def C1NanoWrite_6c_2VALU_4rc0   : SchedWriteRes<[C1NanoUnitVALU0]> { let Latency = 6; let ReleaseAtCycles = [4]; }
  def C1NanoWrite_8c_1VALU_5rc0_2 : SchedWriteRes<[C1NanoUnit2VALU0]> { let Latency = 8; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_12c_1VALU0_5rc  : SchedWriteRes<[C1NanoUnitVALU0]> { let Latency = 12; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_4c_2VMAC_1rc    : SchedWriteRes<[C1NanoUnitVMAC, C1NanoUnitVMAC]> { let Latency = 4; let ReleaseAtCycles = [1, 1]; }
  def C1NanoWrite_3c_1VMC_2rc     : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 3; let ReleaseAtCycles = [2]; }
  def C1NanoWrite_8c_1VMC_5rc_2   : SchedWriteRes<[C1NanoUnit2VMC]> { let Latency = 8; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_8c_1VMC_5rc_4   : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 8; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_8c_1VMC_5rc     : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 8; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_8c_2VMC_5rc_2   : SchedWriteRes<[C1NanoUnit2VMC, C1NanoUnit2VMC]> { let Latency = 8; let ReleaseAtCycles = [5, 5]; }
  def C1NanoWrite_9c_1VMC_7rc     : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 9; let ReleaseAtCycles = [7]; }
  def C1NanoWrite_11c_1VMC_5rc_4  : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 11; let ReleaseAtCycles = [5]; }
  def C1NanoWrite_12c_1VMC_9rc    : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 12; let ReleaseAtCycles = [9]; }
  def C1NanoWrite_12c_1VMC_9rc_2  : SchedWriteRes<[C1NanoUnit2VMC]> { let Latency = 12; let ReleaseAtCycles = [9]; }
  def C1NanoWrite_12c_2VMC_9rc_2  : SchedWriteRes<[C1NanoUnit2VMC, C1NanoUnit2VMC]> { let Latency = 12; let ReleaseAtCycles = [9, 9]; }
  def C1NanoWrite_13c_2VMC_5rc_2  : SchedWriteRes<[C1NanoUnit2VMC, C1NanoUnit2VMC]> { let Latency = 13; let ReleaseAtCycles = [5, 5]; }
  def C1NanoWrite_13c_1VMC_10rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 13; let ReleaseAtCycles = [10]; }
  def C1NanoWrite_13c_1VMC_10rc_4 : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 13; let ReleaseAtCycles = [10]; }
  def C1NanoWrite_13c_2VMC_10rc   : SchedWriteRes<[C1NanoUnit2VMC, C1NanoUnit2VMC]> { let Latency = 13; let ReleaseAtCycles = [10, 10]; }
  def C1NanoWrite_13c_1VMC_11rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 13; let ReleaseAtCycles = [11]; }
  def C1NanoWrite_14c_1VMC_9rc_4  : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 14; let ReleaseAtCycles = [9]; }
  def C1NanoWrite_15c_1VMC_12rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 15; let ReleaseAtCycles = [12]; }
  def C1NanoWrite_21c_1VMC_19rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 21; let ReleaseAtCycles = [19]; }
  def C1NanoWrite_22c_1VMC_19rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 22; let ReleaseAtCycles = [19]; }
  def C1NanoWrite_22c_1VMC_19rc_4 : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 22; let ReleaseAtCycles = [19]; }
  def C1NanoWrite_22c_2VMC_19rc_2 : SchedWriteRes<[C1NanoUnit2VMC, C1NanoUnit2VMC]> { let Latency = 22; let ReleaseAtCycles = [19, 19]; }
  def C1NanoWrite_25c_1VMC_19rc_4 : SchedWriteRes<[C1NanoUnit4VMC]> { let Latency = 25; let ReleaseAtCycles = [19]; }
  def C1NanoWrite_26c_1VMC_23rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 26; let ReleaseAtCycles = [23]; }
  def C1NanoWrite_37c_1VMC_35rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 37; let ReleaseAtCycles = [35]; }
  def C1NanoWrite_68c_1VMC_66rc   : SchedWriteRes<[C1NanoUnitVMC]> { let Latency = 68; let ReleaseAtCycles = [66]; }
}

def C1NanoWrite_3c_1LdSt_1rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 3; let ReleaseAtCycles  = [1]; }
def C1NanoWrite_4c_1LdSt_1rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 4; let ReleaseAtCycles  = [1]; }
def C1NanoWrite_4c_1LdSt_2rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 4; let ReleaseAtCycles  = [2]; }
def C1NanoWrite_4c_1LdSt_3rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 4; let ReleaseAtCycles  = [3]; }
def C1NanoWrite_4c_1LdSt_4rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 4; let ReleaseAtCycles  = [4]; }
def C1NanoWrite_5c_1LdSt_1rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [1]; }
def C1NanoWrite_5c_1LdSt_3rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [3]; }
def C1NanoWrite_5c_1LdSt_4rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [4]; }
def C1NanoWrite_5c_1LdSt_5rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [5]; }
def C1NanoWrite_5c_1LdSt_6rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [6]; }
def C1NanoWrite_5c_1LdSt_2rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [2]; }
def C1NanoWrite_5c_1LdSt_8rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [8]; }
def C1NanoWrite_6c_1LdSt_5rc : SchedWriteRes<[C1NanoUnitLdSt]> { let Latency = 6; let ReleaseAtCycles = [5]; }


def C1NanoWrite_1c_2ALU  : SchedWriteRes<[C1NanoUnitALU, C1NanoUnitALU]> { let Latency = 1; }

def C1NanoWrite_1c_1PAC_1B: SchedWriteRes<[C1NanoUnitPAC, C1NanoUnitB]> { let Latency = 1; let NumMicroOps = 2; }
def C1NanoWrite_1c_1ALU_1LdSt_1rc :  SchedWriteRes<[C1NanoUnitALU, C1NanoUnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [1, 1]; }
def C1NanoWrite_2c_1ALU_1LdSt_2rc :  SchedWriteRes<[C1NanoUnitALU, C1NanoUnitLdSt]> { let Latency = 2; let ReleaseAtCycles = [2, 2]; }
def C1NanoWrite_3c_1ALU_1LdSt_3rc :  SchedWriteRes<[C1NanoUnitALU, C1NanoUnitLdSt]> { let Latency = 3; let ReleaseAtCycles = [3, 3]; }

let RetireOOO = 1 in {
  def C1NanoWrite_1LdSt_0rc : SchedWriteRes<[C1NanoUnitLdSt]> { }
  def C1NanoWrite_1LdSt_2rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [2]; }
  def C1NanoWrite_1LdSt_3rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [3]; }
  def C1NanoWrite_1LdSt_4rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [4]; }
  def C1NanoWrite_1LdSt_6rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [6]; }
  def C1NanoWrite_1LdSt_7rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [7]; }
  def C1NanoWrite_1LdSt_8rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [8]; }
  def C1NanoWrite_1LdSt_9rc : SchedWriteRes<[C1NanoUnitLdSt]> { let ReleaseAtCycles  = [9]; }
  def C1NanoWrite_3c_1VMAC : SchedWriteRes<[C1NanoUnitVMAC]> { let Latency = 4; }
}

// Pre/Post Indexing - Performed as part of address generation
def : SchedAlias<WriteAdr, C1NanoWrite_0c>;

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

def : SchedAlias<WriteSys, C1NanoWrite_1B>;
def : SchedAlias<WriteBarrier, C1NanoWrite_1B>;
def : SchedAlias<WriteHint, C1NanoWrite_1B>;
def : SchedAlias<WriteVd, C1NanoWrite_4c_1VALU>;
def : SchedAlias<WriteVq, C1NanoWrite_4c_1VALU>;

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to C1 Nano.
//===----------------------------------------------------------------------===//

def C1NanoWriteISReg : SchedWriteVariant<[
       SchedVar<RegShiftedPred, [WriteISReg]>,
       SchedVar<NoSchedPred, [WriteI]>]>;
def : InstRW<[C1NanoWriteISReg], (instregex ".*rs$")>;

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the C1-Nano
//===----------------------------------------------------------------------===//

class C1NanoWrite<int n, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = n;
}

//===----------------------------------------------------------------------===//
// Instruction scheduling classes.
//===----------------------------------------------------------------------===//

// Branch instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr, C1NanoWrite_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, C1NanoWrite_1B>;

// Arithmetic and logical operations
// -----------------------------------------------------------------------------

// Arithmetic, basic, flagset
def : InstRW<[C1NanoWrite_1c_2ALU], (instregex "^(ADCS|SBCS)(W|X)(r|i)$")>;

// CMP/CMN are implemented via ADDS/SUBS with Rd = ZR.
def C1NanoWriteISRegCmp : SchedWriteVariant<[
       SchedVar<IsReg0ZeroPred, [WriteI]>,
       SchedVar<RegShiftedPred, [WriteISReg]>,
       SchedVar<NoSchedPred, [WriteI]>]>;

// Conditional compare
def : InstRW<[C1NanoWrite_1c_2ALU], (instregex "^(CCMN|CCMP)(W|X)(r|i)$")>;

// Divide and multiply instructions
// -----------------------------------------------------------------------------

// SDIV, UDIV
def : WriteRes<WriteID32, [C1NanoUnitDiv]> {
  let Latency = 12; let ReleaseAtCycles = [12];
}
def : WriteRes<WriteID64, [C1NanoUnitDiv]> {
  let Latency = 20; let ReleaseAtCycles = [20];
}

// MAC
def : WriteRes<WriteIM32, [C1NanoUnitMAC]> { let Latency = 3; }   // 32-bit Multiply
def : WriteRes<WriteIM64, [C1NanoUnitMAC]> { let Latency = 4; let ReleaseAtCycles = [2];}   // 64-bit Multiply

// Multiply accumulate long
def : InstRW<[C1NanoWrite_2c_1MAC], (instregex "[SU]M(ADD|SUB)L")>;

// Multiply high
def : InstRW<[C1NanoWrite_6c_1MAC_4rc], (instregex "[SU]MULHr")>;

// Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Compute pointer authentication code, using generic key
def : InstRW<[C1NanoWrite_5c_1PAC], (instrs PACGA)>;
// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code for instruction address
def : InstRW<[C1NanoWrite_4c_1PAC], (instregex "^AUT", "^PAC[DI]")>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[C1NanoWrite_1c_1PAC_1B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA,
                                            BRAAZ, BRAB, BRABZ, RETAA, RETAB,
                                            ERETAA, ERETAB)>;

// Load register, with pointer authentication
def : InstRW<[C1NanoWrite_2c_1PAC], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[C1NanoWrite_4c_1PAC], (instrs XPACD, XPACI, XPACLRI)>;

// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Address generation
def : InstRW<[C1NanoWrite_ADRP], (instrs ADR, ADRP)>;

// Convert floating-point condition flags
def : InstRW<[C1NanoWrite_1c_2ALU_2rc], (instregex "^(AX|XA)FLAG")>;

// Flag set instructions
def : InstRW<[C1NanoWrite_2c_2ALU_2rc], (instregex "^SETF(8|16)")>;

// Flag manipulation instructions, rotate and select
def : InstRW<[C1NanoWrite_1c_2ALU_1rc], (instrs RMIF)>;

// Flag manipulation instructions, invert carry
def : InstRW<[C1NanoWrite_1c_2ALU_2rc], (instrs CFINV)>;

// Variable shift
def : InstRW<[WriteI], (instregex "(ASR|LSL|LSR|ROR)V[WX]r")>;

// Reverse bits
def : InstRW<[WriteI], (instrs RBITWr, RBITXr)>;

// ROR (immediate) is implemented as an EXTR (EXTR Rd, Rs, Rs, #Imm).
// Keep EXTR's base latency, but model the ROR (immediate) as 1-cycle.
def C1NanoWriteExtrRORImm : SchedWriteVariant<[
       SchedVar<IsRORImmIdiomPred, [WriteI]>,
       SchedVar<NoSchedPred, [WriteExtr]>]>;
def : InstRW<[C1NanoWriteExtrRORImm], (instrs EXTRWrri, EXTRXrri)>;

// LSL/LSR (immediate) and UXTB are implemented as UBFM aliases.
// Keep UBFM's base latency, but model these aliases as 1-cycle.
def C1NanoWriteISFastUBFMImm : SchedWriteVariant<[
       SchedVar<IsFastUBFMImmPred, [WriteI]>,
       SchedVar<NoSchedPred, [WriteIS]>]>;
def : InstRW<[C1NanoWriteISFastUBFMImm], (instrs UBFMWri, UBFMXri)>;

// ASR (immediate) is implemented as an SBFM alias.
// Keep SBFM's base latency, but model ASR (immediate) as 1-cycle.
def C1NanoWriteISFastSBFMImm : SchedWriteVariant<[
       SchedVar<IsFastSBFMImmPred, [WriteI]>,
       SchedVar<NoSchedPred, [WriteIS]>]>;
def : InstRW<[C1NanoWriteISFastSBFMImm], (instrs SBFMWri, SBFMXri)>;

// Load instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteLD, C1NanoWrite_2c_1Ld>;
def : SchedAlias<WriteLDIdx, C1NanoWrite_2c_1Ld>;
def : SchedAlias<WriteLDHi, C1NanoWrite_2c_1Ld>;

// Integer loads, immediate and register offset.
def : InstRW<[C1NanoWrite_LDR, C1NanoRd_LDR], (instregex "^LDR[WX]ui$")>;
def : InstRW<[C1NanoWrite_LDR, C1NanoRd_LDR], (instregex "^LDR[WX]ro[WX]$")>;

// Load register, immediate pre-index / post-index
def : InstRW<[WriteAdr, C1NanoWrite_LDR, C1NanoWrite_LDR], (instregex "LDR[WX](pre|post)")>;

// Load vector register, immediate pre-index / post-index
def : InstRW<[WriteAdr, C1NanoWrite_3c_1Ld, C1NanoWrite_3c_1Ld], (instregex "LDR[BHSDQ](pre|post)")>;

def : InstRW<[C1NanoWrite_2c_1Ld], (instregex "LDPSWi")>;
def : InstRW<[C1NanoWrite_2c_1Ld], (instregex "LDN?P[WX]i")>;
def : InstRW<[C1NanoWrite_3c_1Ld, C1NanoWrite_3c_1Ld], (instregex "LDN?P[SDQ]i")>;
def : InstRW<[WriteAdr, C1NanoWrite_2c_1Ld, C1NanoWrite_2c_1Ld], (instregex "LDPSW(pre|post)")>;
def : InstRW<[WriteAdr, C1NanoWrite_2c_1Ld_SI, C1NanoWrite_2c_1Ld], (instregex "LDPW(pre|post)")>;
def : InstRW<[WriteAdr, C1NanoWrite_2c_1Ld_SI, C1NanoWrite_2c_1Ld], (instregex "LDPX(pre|post)")>;
def : InstRW<[WriteAdr, C1NanoWrite_3c_1Ld, C1NanoWrite_2c_1Ld], (instregex "LDP[SDQ](pre|post)")>;
def : InstRW<[WriteI], (instrs COPY)>;

// Store instructions
// -----------------------------------------------------------------------------

let RetireOOO = 1 in {
def : WriteRes<WriteST, [C1NanoUnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTP, [C1NanoUnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [C1NanoUnitLdSt]> { let Latency = 1; }
}

// Vector Store - Similar to vector loads, can take 1-3 cycles to issue.
def : WriteRes<WriteVST, [C1NanoUnitLdSt]> { let Latency = 5;
                                          let ReleaseAtCycles = [2];}

// Tag Data processing
// -----------------------------------------------------------------------------
// Arithmetic, immediate to logical address tag
def : InstRW<[C1NanoWrite_2c_1ALU], (instrs ADDG, SUBG)>;

// Insert Random Tags
def : InstRW<[C1NanoWrite_4c_2ALU_3rc], (instrs IRG, IRGstack)>;

// Insert Tag Mask
// Subtract Pointer
// Subtract Pointer, flagset
def : InstRW<[C1NanoWrite_2c_1ALU], (instrs GMI, SUBP, SUBPS)>;

// Tag Load instructions
// -----------------------------------------------------------------------------
// Load allocation tag
def : InstRW<[C1NanoWrite_2c_1Ld], (instrs LDG)>;

// Load multiple allocation tags
def : InstRW<[C1NanoWrite_2c_2Ld_4rc], (instrs LDGM)>;

// Tag store instructions
// -----------------------------------------------------------------------------
// Store allocation tags to one granule, post-index
// Store allocation tags to one granule, pre-index
// Store allocation tag to one granule, zeroing, post-index
// Store Allocation Tag to one granule, zeroing, pre-index
// Store allocation tags to one granule, signed offset
// Store allocation tag and reg pair to memory, signed offset
// Store allocation tag and reg pair to memory, post-Index
// Store allocation tag and reg pair to memory, pre-Index
// Store multiple allocation tags
def : InstRW<[C1NanoWrite_1c_1LdSt], (instrs STGPreIndex, STGPostIndex,
                                                STZGPreIndex, STZGPostIndex,
                                                STGPpre, STGPpost,
                                                STGi, STZGi,
                                                STGPi, STGM, STZGM)>;
// Store allocation tags to two granules, post-index
// Store allocation tags to two granules, pre-index
// Store allocation tag to two granules, zeroing, post-index
// Store Allocation Tag to two granules, zeroing, pre-index
// Store allocation tag to two granules, zeroing, signed offset
def : InstRW<[C1NanoWrite_1c_1LdSt_2rc], (instrs ST2GPreIndex, ST2GPostIndex,
                                                STZ2GPreIndex, STZ2GPostIndex,
                                                ST2Gi, STZ2Gi)>;

// FP scalar data processing instructions
// -----------------------------------------------------------------------------

// FP conditional compare
def : InstRW<[C1NanoWrite_3c_2VALU_1rc], (instregex "FCCMP")>;

// FP divide, H-form
def : InstRW<[C1NanoWrite_8c_1VMC_5rc_4], (instrs FDIVHrr)>;

// FP divide, S-form
def : InstRW<[C1NanoWrite_13c_1VMC_10rc_4], (instrs FDIVSrr)>;

// FP divide, D-form
def : InstRW<[C1NanoWrite_22c_1VMC_19rc_4], (instrs FDIVDrr)>;

// FP square root, H-form
def : InstRW<[C1NanoWrite_11c_1VMC_5rc_4], (instrs FSQRTHr)>;

// FP square root, S-form
def : InstRW<[C1NanoWrite_14c_1VMC_9rc_4], (instrs FSQRTSr)>;

// FP square root, D-form
def : InstRW<[C1NanoWrite_25c_1VMC_19rc_4], (instrs FSQRTDr)>;

// FP data processing instructions
// -----------------------------------------------------------------------------
//   As WriteF result is produced in F5 and it can be mostly forwarded
//   to consumer at F1, the effectively Latency is set as 4.

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF, C1NanoWrite_4c_1VALU>;

// FP compare
def : SchedAlias<WriteFCmp, C1NanoWrite_3c_2VALU>;

// FP Mul, Div, Sqrt. Div/Sqrt are not pipelined
def : WriteRes<WriteFMul, [C1NanoUnitVMAC]> { let Latency = 4; }

let RetireOOO = 1 in {
def : WriteRes<WriteFDiv, [C1NanoUnitVMC]> { let Latency = 22;
                                            let ReleaseAtCycles = [29]; }
}

// FP scalar miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, Javascript from vec to gen reg
def : SchedAlias<WriteFCvt, C1NanoWrite_4c_1VALU>;

// FP move, immed
// FP move, register
def : SchedAlias<WriteFImm, C1NanoWrite_3c_1VALU>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, C1NanoWrite_3c_1VALU>;

// FP scalar load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal
def : InstRW<[C1NanoWrite_3c_1Ld],
             (instregex "^LDR[SDQ]l$")>;

// Load vector reg, unscaled immediate
def : InstRW<[C1NanoWrite_3c_1Ld],
             (instregex "LDUR[BHSDQ]i")>;

// Load vector register, unsigned immediate
def : InstRW<[C1NanoWrite_3c_1Ld],
             (instregex "LDR[BHSDQ]ui")>;

// Load vector register, register offset
def : InstRW<[C1NanoWrite_3c_1Ld],
             (instregex "LDR[BHSDQ]ro[WX]")>;

// FP scalar store instructions
// -----------------------------------------------------------------------------

// Store vector pair, immediate offset, Q-form
def : InstRW<[C1NanoWrite_1c_1LdSt_2rc],
             (instregex "STN?PQ(i|post|pre)")>;

//---
// Floating Point Conversions, MAC, DIV, SQRT
//---
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^DUP(v2i64|v2i32|v4i32|v4i16|v8i16|v8i8|v16i8)")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^XTN")>;

// FP convert, from vec to gen reg
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^FCVT[ALMNPZ][SU](S|U)?(W|X)")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^FCVT(X)?[ALMNPXZ](S|U|N)?v")>;

// FP convert, Javascript from vec to gen reg
def : InstRW<[C1NanoWrite_4c_2VALU], (instrs FJCVTZS)>;

def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^(S|U)CVTF(S|U)(W|X)(H|S|D)")>;
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^(S|U)CVTF(h|s|d)")>;
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^(S|U)CVTFv")>;

// ASIMD integer instructions
// -----------------------------------------------------------------------------

// ASIMD absolute diff
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "[SU]ABDv(2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "[SU]ABDv(16i8|4i32|8i16)")>;

// ASIMD move, integer immediate
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^MOVI(v|D)")>;

// ASIMD reverse
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^REV(16|32|64)v")>;

// ASIMD absolute diff accum
def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "[SU]ABAL?v")>;

// ASIMD absolute diff long
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]ABDLv")>;

// ASIMD arith, basic
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "(ABS|ADD|SUB|NEG)v",
  "[SU](HADDv|HSUBv)")>;

// ASIMD, arith, basic, long, saturate
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex  "SADDLv", "UADDLv", "SADDWv",
  "UADDWv", "SSUBLv", "USUBLv", "SSUBWv", "USUBWv")>;

// ASIMD, arith, complex
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex  "ADDHNv", "SUBHNv")>;
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(16i8|2i64|4i32|8i16)$")>;

// ASIMD, arith, complex, rounding, add and subtract
def : InstRW<[C1NanoWrite_6c_2VALU_3rc], (instregex "RADDHNv", "RSUBHNv")>;

// ASIMD, arith, complex, rounding halving addition
def : InstRW<[C1NanoWrite_2c_1VALU], (instregex "[SU]RHADDv")>;

// ASIMD, arith, pair-wise
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "[SU]ADDLPv", "ADDPv")>;

// ASIMD arith, reduce
def : InstRW<[C1NanoWrite_3c_2VALU_1rc], (instregex  "ADDVv(8|16)")>;

// ASIMD, arith, reduce 4H/4S
def : InstRW<[C1NanoWrite_4c_2VALU_1rc], (instregex  "SADDLVv", "UADDLVv", "ADDVv4")>;

// ASIMD compare
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(2i64|4i32|8i16|16i8)")>;

// ASIMD compare test
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "CMTSTv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "CMTSTv(2i64|4i32|8i16|16i8)")>;

// ASIMD unzip/zip/transpose
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^(TRN|UZP|ZIP)[12]v")>;

// ASIMD extract
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^EXTv")>;

// ASIMD table lookup / table lookup extension
def : InstRW<[C1NanoWrite_4c_1VALU], (instrs TBLv8i8One, TBLv16i8One)>;
def : InstRW<[C1NanoWrite_5c_2VALU_2rc], (instrs TBLv8i8Two, TBLv16i8Two)>;
def : InstRW<[C1NanoWrite_6c_2VALU_3rc], (instrs TBLv8i8Three, TBLv16i8Three)>;
def : InstRW<[C1NanoWrite_7c_2VALU_4rc], (instrs TBLv8i8Four, TBLv16i8Four)>;
def : InstRW<[C1NanoWrite_5c_2VALU_2rc], (instrs TBXv8i8One, TBXv16i8One)>;
def : InstRW<[C1NanoWrite_6c_2VALU_3rc], (instrs TBXv8i8Two, TBXv16i8Two)>;
def : InstRW<[C1NanoWrite_7c_2VALU_4rc], (instrs TBXv8i8Three, TBXv16i8Three)>;
def : InstRW<[C1NanoWrite_8c_2VALU_5rc], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD logical
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "(AND|EOR|NOT|ORN)v8i8",
  "(ORR|BIC)v(2i32|4i16|8i8)$", "MVNIv(2i|2s|4i16)")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "(AND|EOR|NOT|ORN)v16i8",
  "(ORR|BIC)v(16i8|4i32|8i16)$", "MVNIv(4i32|4s|8i16)")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^(BIF|BIT|BSL)v")>;

// ASIMD max/min, basic
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU](MIN|MAX)P?v(2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU](MIN|MAX)P?v(16i8|4i32|8i16)")>;

// ASIMD max/min, reduce
def : InstRW<[C1NanoWrite_4c_2VALU_1rc],
             (instregex "[SU](MAX|MIN)Vv")>;

// ASIMD FP max/min, reduce.
def : InstRW<[C1NanoWrite_4c_2VALU_1rc],
             (instregex "^F(MAX|MIN)(NM)?Vv")>;

// ASIMD multiply, by element
def : InstRW<[C1NanoWriteVMACH64], (instregex "^MULv4i16_indexed$")>;
def : InstRW<[C1NanoWriteVMACH128], (instregex "^MULv8i16_indexed$")>;
def : InstRW<[C1NanoWriteVMACS64], (instregex "^MULv2i32_indexed$")>;
def : InstRW<[C1NanoWriteVMACS128], (instregex "^MULv4i32_indexed$")>;
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "SQR?DMULHv(1i16|1i32|2i32|4i16|4i32|8i16)(_indexed)?$")>;

// ASIMD multiply
def : InstRW<[C1NanoWriteVMACB64], (instregex "^MULv8i8$")>;
def : InstRW<[C1NanoWriteVMACB128], (instregex "^MULv16i8$")>;
def : InstRW<[C1NanoWriteVMACH64], (instregex "^MULv4i16$")>;
def : InstRW<[C1NanoWriteVMACH128], (instregex "^MULv8i16$")>;
def : InstRW<[C1NanoWriteVMACS64], (instregex "^MULv2i32$")>;
def : InstRW<[C1NanoWriteVMACS128], (instregex "^MULv4i32$")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^PMULv(8i8|16i8)")>;

// ASIMD multiply accumulate
def : InstRW<[C1NanoWriteVMACB64, C1NanoReadVMACB64], (instregex "^ML[AS]v8i8$")>;
def : InstRW<[C1NanoWriteVMACB128, C1NanoReadVMACB128], (instregex "^ML[AS]v16i8$")>;
def : InstRW<[C1NanoWriteVMACH64, C1NanoReadVMACH64], (instregex "^ML[AS]v4i16$")>;
def : InstRW<[C1NanoWriteVMACH128, C1NanoReadVMACH128], (instregex "^ML[AS]v8i16$")>;
def : InstRW<[C1NanoWriteVMACS64, C1NanoReadVMACS64], (instregex "^ML[AS]v2i32$")>;
def : InstRW<[C1NanoWriteVMACS128, C1NanoReadVMACS128], (instregex "^ML[AS]v4i32$")>;
def : InstRW<[C1NanoWriteVMACH64, C1NanoReadVMACH64], (instregex "^ML[AS]v4i16_indexed$")>;
def : InstRW<[C1NanoWriteVMACH128, C1NanoReadVMACH128], (instregex "^ML[AS]v8i16_indexed$")>;
def : InstRW<[C1NanoWriteVMACS64, C1NanoReadVMACS64], (instregex "^ML[AS]v2i32_indexed$")>;
def : InstRW<[C1NanoWriteVMACS128, C1NanoReadVMACS128], (instregex "^ML[AS]v4i32_indexed$")>;

// ASIMD multiply accumulate half
def : InstRW<[C1NanoWrite_4c_1VMAC,
             ReadVMACAccum], (instregex "SQRDML[AS]H[vi]")>;

// ASIMD multiply accumulate long
def : InstRW<[C1NanoWriteVMACH128, C1NanoReadVMACH128],
             (instregex "^[SU]ML[AS]Lv(8i8|16i8)_(v8i16|indexed)$")>;
def : InstRW<[C1NanoWriteVMACS128, C1NanoReadVMACS128],
             (instregex "^[SU]ML[AS]Lv(4i16|8i16)_(v4i32|indexed)$")>;
def : InstRW<[C1NanoWriteVMACD128, C1NanoReadVMACD128],
             (instregex "^[SU]ML[AS]Lv(2i32|4i32)_(v2i64|indexed)$")>;

// ASIMD multiply accumulate long #2
def : InstRW<[C1NanoWrite_4c_1VMAC, ReadVMACAccum],
             (instregex "SQDML[AS]L[iv]")>;

// ASIMD dot product
def : InstRW<[C1NanoWriteVMACS64, C1NanoReadVMACS64], (instregex "^(S|U|SU|US)DOTv8i8$")>;
def : InstRW<[C1NanoWriteVMACS128, C1NanoReadVMACS128], (instregex "^(S|U|SU|US)DOTv16i8$")>;

// ASIMD dot product, by scalar
def : InstRW<[C1NanoWriteVMACS64, C1NanoReadVMACS64], (instregex "^(S|U|SU|US)DOTlanev8i8$")>;
def : InstRW<[C1NanoWriteVMACS128, C1NanoReadVMACS128], (instregex "^(S|U|SU|US)DOTlanev16i8$")>;

// ASIMD multiply long
def : InstRW<[C1NanoWriteVMACH128], (instregex "^[SU]MULLv(8i8|16i8)_(v8i16|indexed)$")>;
def : InstRW<[C1NanoWriteVMACS128], (instregex "^[SU]MULLv(4i16|8i16)_(v4i32|indexed)$")>;
def : InstRW<[C1NanoWriteVMACD128], (instregex "^[SU]MULLv(2i32|4i32)_(v2i64|indexed)$")>;
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "SQDMULL[iv]")>;

// ASIMD polynomial (8x8) multiply long
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instrs PMULLv8i8, PMULLv16i8)>;

// ASIMD pairwise add and accumulate
def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "[SU]ADALPv")>;

// ASIMD shift accumulate
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]SRA(d|v2i32|v4i16|v8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]SRAv(16i8|2i64|4i32|8i16)")>;

// ASIMD shift accumulate #2
def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "[SU]RSRA[vd]")>;

// ASIMD shift by immed
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "SHLd$", "SHLv",
  "SLId$", "SRId$", "[SU]SHR[vd]", "SHRNv(8i8|4i16|2i32)")>;

// ASIMD shift by immediate and insert, basic
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^SLIv.*_shift", "^SRIv.*_shift")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "SHRNv(16i8|8i16|4i32)")>;

// ASIMD shift by immed
// SXTL and UXTL are aliases for SHLL
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[US]?SHLLv")>;

// ASIMD shift by immed #2
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]RSHR(d|v2i32|v4i16|v8i8)",
  "[SU]RSHRv(16i8|2i64|4i32|8i16)")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "RSHRNv(2i32|4i16|8i8)",
  "RSHRNv(16i8|4i32|8i16)")>;

// ASIMD shift by register
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]SHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]SHLv(2i64|4i32|8i16|16i8)")>;

// ASIMD shift by register #2
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]RSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "[SU]RSHLv(2i64|4i32|8i16|16i8)")>;

def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "[SU]QSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "[SU]QSHLv(2i64|4i32|8i16|16i8)")>;

def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "[SU]QRSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "[SU]QRSHLv(2i64|4i32|8i16|16i8)")>;

// ASIMD FP data processing instructions
// -----------------------------------------------------------------------------

// ASIMD FP, arith, normal
def : InstRW<[C1NanoWrite_3c_1VMAC],
             (instregex "^FN?M(ADD|SUB)")>;

// ASIMD FP compare
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(FACGE|FACGT|FCMEQ|FCMGE|FCMGT|FCMLE|FCMLT)(v|16|32|64)")>;

// ASIMD FP, complex add
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^FCADDv")>;

// ASIMD FP, complex multiply add
def : InstRW<[C1NanoWrite_3c_1VMAC],
             (instregex "^FCMLAv")>;

// ASIMD FP, multiply accumulate
def : InstRW<[C1NanoWrite_3c_1VMAC],
             (instregex "^FML(A|S)v")>;

// ASIMD FP divide, D-form, F61
def : InstRW<[C1NanoWrite_8c_1VMC_5rc_2],
             (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
def : InstRW<[C1NanoWrite_13c_2VMC_5rc_2],
             (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[C1NanoWrite_8c_2VMC_5rc_2],
             (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[C1NanoWrite_13c_2VMC_10rc],
             (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[C1NanoWrite_22c_2VMC_19rc_2],
             (instrs FDIVv2f64)>;

// ASIMD FP square root, D-form, F16
def : InstRW<[C1NanoWrite_8c_1VMC_5rc_2],
             (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
def : InstRW<[C1NanoWrite_12c_1VMC_9rc_2],
             (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[C1NanoWrite_8c_2VMC_5rc_2],
             (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
def : InstRW<[C1NanoWrite_12c_2VMC_9rc_2],
             (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[C1NanoWrite_22c_2VMC_19rc_2],
             (instrs FSQRTv2f64)>;

def : InstRW<[C1NanoWrite_3c_2VALU],
             (instrs FCSELHrrr, FCSELSrrr, FCSELDrrr)>;

// ASIMD FP multiply
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "FMULX?(16|32|64|v)")>;

// ASIMD FP multiply accumulate long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "FML[AS]L2?(v|lane)")>;

// ASIMD BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
// Scalar convert, F32 to BF16
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instrs BFCVTN, BFCVTN2, BFCVT)>;

// ASIMD dot product
def : InstRW<[C1NanoWrite_10c_1VMAC_1VALU],
             (instregex "^BFDOTv", "^BF16DOT")>;

// ASIMD matrix multiply accumulate
def : InstRW<[C1NanoWrite_14c_2VMAC_2VALU],
             (instregex "^BFMMLA$")>;

// ASIMD multiply accumulate long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^BFMLAL[BT]$", "^BFMLAL[BT]Idx$")>;

// ASIMD miscellaneous instructions
// -----------------------------------------------------------------------------

// ASIMD count
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^(CLS|CLZ|CNT)v")>;

// ASIMD scalar DUP (asm mnemonic is "mov", e.g. "mov b0, v0.b[1]").
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "DUPi")>;

// ASIMD transfer, element to gen reg
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "[SU]MOVvi")>;

// ASIMD transfer from vector element to GPR with sign-extension.
// ASIMD move between vector elements (asm mnemonic is "mov", e.g. "mov v2.b[0], v0.b[0]").
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "INSvi(8|16|32|64)lane")>;

// ASIMD transfer, gen reg to element
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "INSvi(8|16|32|64)gpr")>;

// ASIMD move, FP immediate
def : InstRW<[C1NanoWrite_2c_1VALU], (instrs FMOVSr, FMOVDr)>;

// ASIMD move, FP transfer, from reg to vec reg
def : InstRW<[C1NanoWrite_3c_1VALU], (instrs FMOVv2f32_ns, FMOVv4f32_ns, FMOVv2f64_ns, FMOVv4f16_ns, FMOVv8f16_ns)>;

// ASIMD reciprocal estimate
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "FRECP[EX]v", "URECPEv", "[FU]RSQRTEv")>;

// ASIMD reciprocal step
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "FR(ECPS|SQRTS)(16|32|64|v)")>;

// ASIMD reverse bits.
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "RBITv")>;

// ASIMD load instructions
// -----------------------------------------------------------------------------

// ASIMD load, 1 element, multiple, 1 reg, D-form
// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[C1NanoWrite_3c_1Ld], (instregex "LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_3c_1Ld], (instregex "LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[C1NanoWrite_3c_1LdSt_1rc], (instregex "LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_3c_1LdSt_1rc], (instregex "LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "LD1Fourv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "LD1Fourv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[C1NanoWrite_3c_1Ld], (instregex "LD1i(8|16|32|64)$")>;                // single element
def : InstRW<[WriteAdr, C1NanoWrite_3c_1Ld], (instregex "LD1i(8|16|32|64)_POST$")>;                // single element

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
def : InstRW<[C1NanoWrite_3c_1Ld], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>; // replicate
def : InstRW<[WriteAdr, C1NanoWrite_3c_1Ld], (instregex "LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>; // replicate

// ASIMD load, 2 element, multiple, D-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_4c_1LdSt_1rc], (instregex "LD2Twov(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_1rc], (instregex "LD2Twov(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[C1NanoWrite_4c_1LdSt_4rc], (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_4rc], (instregex "LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[C1NanoWrite_3c_1LdSt_1rc], (instregex "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_3c_1LdSt_1rc], (instregex "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_5c_1LdSt_3rc], (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_3rc], (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 3 element, one lane, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[C1NanoWrite_5c_1LdSt_5rc], (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_5rc], (instregex "LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_5c_1LdSt_3rc], (instregex "LD4Fourv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_3rc], (instregex "LD4Fourv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[C1NanoWrite_6c_1LdSt_5rc], (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_6c_1LdSt_5rc], (instregex "LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD store instructions
// -----------------------------------------------------------------------------

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
// TODO: Handle the special case of ASIMD store, 1 element, multiple, 3 reg, D-form when:
//   Throughput=1/3 when the access is aligned and crosses 16B boundary, one more cycle is needed
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_3rc], (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_3rc], (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[C1NanoWrite_4c_1LdSt_2rc], (instregex "ST1Fourv(8b|4h|2s|1d)$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[C1NanoWrite_4c_1LdSt_4rc], (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_4rc], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_4c_1LdSt_1rc], (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
def : InstRW<[C1NanoWrite_5c_1LdSt_2rc], (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_2rc], (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_5c_1LdSt_1rc], (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_1rc], (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[C1NanoWrite_5c_1LdSt_2rc], (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_2rc], (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[C1NanoWrite_5c_1LdSt_4rc], (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_4rc], (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_5c_1LdSt_6rc], (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_6rc], (instregex "ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[C1NanoWrite_5c_1LdSt_4rc], (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_4rc], (instregex "ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[C1NanoWrite_5c_1LdSt_4rc], (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_4rc], (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[C1NanoWrite_5c_1LdSt_8rc], (instregex "ST4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_8rc], (instregex "ST4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H/S
// ASIMD store, 4 element, one lane, D
def : InstRW<[C1NanoWrite_5c_1LdSt_8rc], (instregex "ST4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, C1NanoWrite_5c_1LdSt_8rc], (instregex "ST4i(8|16|32|64)_POST$")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[C1NanoWrite_3c_1VMC_2rc], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
def : InstRW<[C1NanoWrite_3c_2VALU_1rc], (instregex "^SHA1H")>;

// Crypto SHA1 schedule acceleration ops
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^SHA1(SU0|SU1)")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[C1NanoWrite_9c_1VMC_7rc], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[C1NanoWrite_3c_1VALU], (instrs BCAX, EOR3, RAX1)>;
def : InstRW<[C1NanoWrite_4c_1VALU], (instrs XAR)>;

// Crypto SM3 ops
def : InstRW<[C1NanoWrite_9c_1VMC_7rc], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                                            "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[C1NanoWrite_9c_1VMC_7rc], (instrs SM4E, SM4ENCKEY)>;

// CRC
// -----------------------------------------------------------------------------

def : InstRW<[C1NanoWrite_2c_1MAC], (instregex "^CRC32")>;

// SVE Predicate instructions
// -----------------------------------------------------------------------------

// Note Correction to what is stated in the Arm C1-Nano Core Software Optimization
//      Guide issue 4, section 22.4 SVE Predicate instructions:
//      - PALU should be ALU0

// Loop control, based on predicate
def : InstRW<[C1NanoWrite_2c_1ALU0], (instrs BRKA_PPmP, BRKA_PPzP,
                                                  BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[C1NanoWrite_2c_1ALU0], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[C1NanoWrite_2c_1ALU0], (instrs BRKN_PPzP, BRKPA_PPzPP, BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[C1NanoWrite_2c_1ALU0], (instrs BRKNS_PPzP, BRKPAS_PPzPP, BRKPBS_PPzPP)>;

// Loop control, based on GPR
def : InstRW<[C1NanoWrite_2c_1ALU0],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]")>;

def : InstRW<[C1NanoWrite_2c_1ALU0], (instregex "^WHILE(RW|WR)_PXX_[BHSD]")>;

// Loop terminate
def : InstRW<[C1NanoWrite<1, C1NanoUnitALU1>], (instregex "^CTERM(EQ|NE)_(WW|XX)")>;

// Predicate counting scalar
def : InstRW<[C1NanoWrite_1c_1ALU], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;

def : InstRW<[C1NanoWrite_1c_1ALU0],
             (instregex "^CNT[BHWD]_XPiI")>;

def : InstRW<[C1NanoWrite<1, C1NanoUnitALU1>],
             (instregex "^(INC|DEC)[BHWD]_XPiI")>;

def : InstRW<[C1NanoWrite_5c_1ALU0],
             (instregex "^(SQINC|SQDEC|UQINC|UQDEC)[BHWD]_[XW]Pi(Wd)?I")>;

// Predicate counting scalar, active predicate
def : InstRW<[C1NanoWrite_1c_1ALU0],
             (instregex "^CNTP_XPP_[BHSD]")>;

def : InstRW<[C1NanoWrite_1c_1ALU0],
             (instregex "^(DEC|INC)P_XP_[BHSD]")>;

def : InstRW<[C1NanoWrite_2c_2VALU_1rc],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]",
                        "^(UQDEC|UQINC)P_WP_[BHSD]")>;

// Predicate counting scalar, active predicate, saturating, 32-bit.
def : InstRW<[C1NanoWrite_1c_2VALU_1rc],
             (instregex "^(SQDEC|SQINC)P_XPWd_[BHSD]")>;

// Predicate counting vector, active predicate
def : InstRW<[C1NanoWrite_3c_1ALU0],
             (instregex "^(DEC|INC)P_ZP_[HSD]")>;

def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]")>;

// Predicate logical
def : InstRW<[C1NanoWrite_2c_1ALU0],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP")>;

// Predicate logical, flag setting
def : InstRW<[C1NanoWrite_2c_1ALU0],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP")>;

// Predicate reverse
def : InstRW<[C1NanoWrite_1c_1ALU0], (instregex "^REV_PP_[BHSD]")>;

// Predicate select
def : InstRW<[C1NanoWrite_2c_1ALU0], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[C1NanoWrite_1c_1ALU0], (instregex "^PFALSE", "^PTRUE_[BHSD]")>;

// Predicate set/initialize, set flags
def : InstRW<[C1NanoWrite_2c_1ALU0], (instregex "^PTRUES_[BHSD]")>;

// Predicate find first/next
def : InstRW<[C1NanoWrite_2c_1ALU0], (instregex "^PFIRST_B", "^PNEXT_[BHSD]")>;

// Predicate test
def : InstRW<[C1NanoWrite_1c_1ALU0], (instrs PTEST_PP, PTEST_PP_ANY, PTEST_PP_FIRST)>;

// Predicate transpose
def : InstRW<[C1NanoWrite_1c_1ALU0], (instregex "^TRN[12]_PPP_[BHSDQ]")>;

// Predicate unpack and widen
def : InstRW<[C1NanoWrite_1c_1ALU0], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[C1NanoWrite_1c_1ALU0], (instregex "^(ZIP|UZP)[12]_PPP_[BHSDQ]")>;

// SVE integer instructions
// -----------------------------------------------------------------------------
// Arithmetic, absolute diff
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^[SU]ABD_(ZPmZ|ZPZZ)_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "^[SU]ABA_ZZZ_[BHSD]")>;

// Arithmetic, absolute diff accum long
def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]")>;

// Arithmetic, absolute diff long
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]")>;

// Arithmetic, basic
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(ABS|CNOT|NEG)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZPZZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB)_ZI_[BHSD]",
                        "^ADR_[SU]XTW_ZZZ_D_[0123]",
                        "^ADR_LSL_ZZZ_[SD]_[0123]",
                        "^[SU]H(ADD|SUB|SUBR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^UADDW[BT]_ZZZ_[HSD]",
                        "^[SU]RHADD_ZPmZ_[BHSD]")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^SADDW[BT]_ZZZ_[HSD]",
                        "^[SU]ADDL[BT]_ZZZ_[HSD]",
                        "^[SU]SUB[LW][BT]_ZZZ_[HSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]",
                        "^SUBR_(ZPmZ|ZPZZ|ZI)_[BHSD]")>;

// Arithmetic, complex
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^SQ(ABS|NEG)_ZPmZ_[BHSD]",
                        "^SQ(ADD|SUB|SUBR)_ZPmZ_?[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^(SUQ|UQ|USQ)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;
def : InstRW<[C1NanoWrite_6c_2VALU_3rc],
             (instregex "^R(ADD|SUB)HN[BT]_ZZZ_[BHS]")>;

// Arithmetic, large integer
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]")>;

// Arithmetic, pairwise add
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^ADDP_ZPmZ_[BHSD]")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[C1NanoWrite_6c_2VALU_4rc],
             (instregex "^[SU]ADALP_ZPmZ_[HSD]")>;

// Arithmetic, shift
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;
// Arithmetic, shift right for divide
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^ASRD_ZPmI_[BHSD]",
                        "^ASRD_ZPZI_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(SSRA|USRA)_ZZI_[BHSD]")>;

def : InstRW<[C1NanoWrite_5c_2VALU_3rc],
             (instregex "^(SRSRA|URSRA)_ZZI_[BHSD]")>;


// Arithmetic, shift by immediate
// Arithmetic, shift by immediate and insert
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(SHRNB|SHRNT|SSHLLB|SSHLLT|USHLLB|USHLLT|SLI|SRI)_ZZI_[BHSD]")>;

// Arithmetic, shift complex
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift rounding
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(SRSHL|SRSHR|SRSHLR|URSHL|URSHLR|URSHR)_(ZPmZ|ZPZZ|ZPZI)_[BHSD]",
                        "^[SU]RSHR_ZPmI_[BHSD]")>;

// Bit manipulation
def : InstRW<[C1NanoWrite_13c_1VMC_11rc],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_B")>;

def : InstRW<[C1NanoWrite_21c_1VMC_19rc],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_H")>;

def : InstRW<[C1NanoWrite_37c_1VMC_35rc],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_S")>;

def : InstRW<[C1NanoWrite_68c_1VMC_66rc],
             (instregex "^(BDEP|BEXT|BGRP)_ZZZ_D")>;


// Bitwise select
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ")>;

// Count/reverse bits
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(CLS|CLZ|RBIT)_ZPmZ_[BHSD]")>;
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^CNT_ZPmZ_[BH]")>;
def : InstRW<[C1NanoWrite_6c_2VALU_4rc],
             (instregex "^CNT_ZPmZ_S")>;
def : InstRW<[C1NanoWrite_9c_2VALU_7rc],
             (instregex "^CNT_ZPmZ_D")>;
// Broadcast logical bitmask immediate to vector.
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[C1NanoWrite_5c_2VALU_1rc],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]")>;

// Complex add
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^CADD_ZZI_[BHSD]")>;

def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^SQCADD_ZZI_[BHSD]")>;

// Complex dot product 8-bit element
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^CMLA_ZZZ_[BHS]", "^CMLA_ZZZI_[HS]")>;

// Complex multiply-add D element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form
def : InstRW<[C1NanoWrite_4c_2VALU_4rc],
             (instregex "^CLAST[AB]_RPZ_[BHSD]")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]",
                        "^COMPACT_ZPZ_[SD]",
                        "^SPLICE_ZPZZ?_[BHSD]")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]CVTF_ZPmZ_Dto[SD]")>;

// Convert to floating point, 64b to half
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]CVTF_ZPmZ_DtoH")>;

// Convert to floating point, 32b to single or half
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 32b to double
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 16b to half
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^CPY_ZPmR_[BHSD]")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^CPY_ZPm[IV]_[BHSD]", "^CPY_ZPzI_[BHSD]")>;

// Divides, 32 bit
def : InstRW<[C1NanoWrite_15c_1VMC_12rc],
             (instregex "^[SU]DIVR?_(ZPmZ|ZPZZ)_S")>;

// Divides, 64 bit
def : InstRW<[C1NanoWrite_26c_1VMC_23rc],
             (instregex "^[SU]DIVR?_(ZPmZ|ZPZZ)_D")>;

// Dot product, 8 bit
def : InstRW<[C1NanoWrite_4c_1VMAC, ReadVMACAccum],
             (instregex "^[SU]DOT_ZZZI?_BtoS")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[C1NanoWrite_4c_1VMAC, ReadVMACAccum],
             (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[C1NanoWrite_4c_1VMAC, ReadVMACAccum],
             (instregex "^[SU]DOT_ZZZI?_HtoD")>;

// Duplicate, immediate and indexed form
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^DUP_ZI_[BHSD]", "^DUP_ZZI_[BHSDQ]")>;

// Duplicate, scalar form
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^DUP_ZR_[BHSD]")>;

// Extend, sign or zero
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^[SU]XTB_ZPmZ_[HSD]",
                        "^[SU]XTH_ZPmZ_[SD]",
                        "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instrs EXT_ZZI, EXT_ZZI_CONSTRUCTIVE, EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]QXTN[BT]_ZZ_[BHS]", "^SQXTUN[BT]_ZZ_[BHS]")>;

// Extract/insert operation, SIMD and FP scalar form
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^LAST[AB]_VPZ_[BHSD]", "^INSR_ZV_[BHSD]")>;

// Extract operation, scalar
def : InstRW<[C1NanoWrite_8c_2VALU_4rc],
             (instregex "^LAST[AB]_RPZ_[BHSD]")>;

// Insert operation, scalar
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^INSR_ZR_[BHSD]")>;

// Histogram operations
def : InstRW<[C1NanoWrite_6c_2VALU_4rc0],
             (instregex "^HISTCNT_ZPzZZ_[SD]", "^HISTSEG_ZZZ")>;

// Horizontal operations, B, H, S form, immediate operands only
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^INDEX_II_[BHS]")>;

// Horizontal operations, B, H, S form, scalar, immediate operands/ scalar
// operands only / immediate, scalar operands
def : InstRW<[C1NanoWrite_4c_2VMAC_1rc],
             (instregex "^INDEX_(IR|RI|RR)_[BHS]")>;

// Horizontal operations, D form, immediate operands only
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instrs INDEX_II_D)>;

// Horizontal operations, D form, scalar, immediate operands / scalar operands
// only / immediate, scalar operands
def : InstRW<[C1NanoWrite_4c_2VMAC_1rc],
             (instregex "^INDEX_(IR|RI|RR)_D")>;

// Logical
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|EON|ORR|NAND|NOR)_ZZZ",
                        "^(AND|BIC|EOR|NOT|ORR)_ZPmZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_ZPZZ_[BHSD]")>;

def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^EOR(BT|TB)_ZZZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                        "^[SU](MAX|MIN)P?_(ZPmZ|ZPZZ)_[BHSD]")>;

// Matching operations
def : InstRW<[C1NanoWrite_8c_2VALU_4rc], (instregex "^N?MATCH_PPzZZ_[BH]")>;

// Matrix multiply-accumulate
def : InstRW<[C1NanoWrite_4c_1VMAC], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[C1NanoWrite_3c_1VALU],
            (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]", "^MOVPRFX_ZZ")>;

// Multiply, B, H, S element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ|ZPZZ)_[BHS]",
                        "^[SU]MULH_(ZPmZ|ZZZ|ZPZZ)_[BHS]")>;

// Multiply, D element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ|ZPZZ)_D",
                        "^[SU]MULH_(ZPmZ|ZZZ|ZPZZ)_D")>;

// Multiply long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^[SU]MULL[BT]_ZZZI_[SD]",
                        "^[SU]MULL[BT]_ZZZ_[HSD]")>;

// Multiply accumulate, B, H, S element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^ML[AS]_(ZZZI|ZPZZZ)_[BHS]",
                        "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]")>;

// Multiply accumulate, D element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^ML[AS]_(ZZZI|ZPZZZ)_D",
                        "^(ML[AS]|MAD|MSB)_ZPmZZ_D")>;

// Multiply accumulate long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]",
                        "^[SU]ML[AS]L[BT]_ZZZI_[SD]")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQDML[AS](LB|LT|LBT)_ZZZ_[HSD]",
                        "^SQDML[AS](LB|LT)_ZZZI_[SD]")>;

// Multiply saturating doubling high, B, H, S element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQDMULH_ZZZ_[BHS]",
                        "^SQDMULH_ZZZI_[HS]")>;

// Multiply saturating doubling high, D element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQDMULL[BT]_ZZZ_[HSD]",
                        "^SQDMULL[BT]_ZZZI_[SD]")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S
// element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQRDML[AS]H_ZZZ_[BHS]",
                        "^SQRDCMLAH_ZZZ_[BHS]",
                        "^SQRDML[AS]H_ZZZI_[HS]",
                        "^SQRDCMLAH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element
// size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQRDML[AS]H_ZZZI?_D",
                        "^SQRDCMLAH_ZZZ_D")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQRDMULH_ZZZ_[BHS]",
                        "^SQRDMULH_ZZZI_[HS]")>;

// Multiply saturating rounding doubling regular/complex, D element size
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^SQRDMULH_ZZZI?_D")>;

// Multiply/multiply long, (8x8) polynomial
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^PMUL_ZZZ_B", "^PMULL[BT]_ZZZ_H")>;

def : InstRW<[C1NanoWrite_9c_1VMC_7rc], (instregex "^PMULL[BT]_ZZZ_[DQ]")>;


// Predicate counting vector
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(DEC|INC)[HWD]_ZPiI")>;
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^(SQDEC|SQINC|UQDEC|UQINC)[HWD]_ZPiI")>;

// Reciprocal estimate
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[C1NanoWrite_4c_1VALU0],
             (instregex "^[SU](ADD|MAX|MIN)V_VPZ_B")>;

// Reduction, arithmetic, H form
def : InstRW<[C1NanoWrite_4c_1VALU0],
             (instregex "^[SU](ADD|MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[C1NanoWrite_4c_1VALU0],
             (instregex "^[SU](ADD|MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[C1NanoWrite_4c_1VALU0],
             (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[C1NanoWrite_4c_1VALU0],
             (instregex "^(ANDV|EORV|ORV)_VPZ_[BHSD]")>;

// Reverse, vector
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^REV_ZZ_[BHSD]",
                        "^REVB_ZPmZ_[HSD]",
                        "^REVH_ZPmZ_[SD]",
                        "^REVW_ZPmZ_D")>;

// Select, vector form
def : InstRW<[C1NanoWrite_2c_1VALU],
             (instregex "^SEL_ZPZZ_[BHSD]")>;

// Table lookup
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^TBL_ZZZ_[BHSD]")>;

// Table lookup, double table
def : InstRW<[C1NanoWrite_8c_2VALU_5rc],
             (instregex "^TBL_ZZZZ_[BHSD]")>;

// Table lookup extension
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^TBX_ZZZ_[BHSD]")>;

// Transpose, vector form
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^TRN[12]_ZZZ_[BHSDQ]")>;

// Unpack and extend
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]")>;

// Zip/unzip
def : InstRW<[C1NanoWrite_3c_1VALU],
             (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]")>;

// SVE FP data processing instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^FAB[SD]_ZPmZ_[HSD]", "^FAB[SD]_ZPZZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ|ZPZI|ZPZZ)_[HSD]",
                        "^FADDP_ZPmZZ_[HSD]",
                        "^FNEG_ZPmZ_[HSD]",
                        "^FSUBR_(ZPm[IZ]|ZPZ[IZ])_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[C1NanoWrite_23c_2VALU_25rc], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[C1NanoWrite_16c_2VALU_9rc], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[C1NanoWrite_8c_1VALU_5rc], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[C1NanoWrite_4c_2VALU_1rc],
               (instregex "^FACG[ET]_PPzZZ_[HSD]",
                          "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]",
                          "^FCM(LE|LT)_PPzZ0_[HSD]",
                          "^FCMUO_PPzZZ_[HSD]")>;

// Floating point complex add
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCADD_ZPmZ_[HSD]")>;

// Floating point complex multiply add
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FCMLA_ZPmZZ_[HSD]", "^FCMLA_ZZZI_[HS]")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                          "^FCVTLT_ZPmZ_HtoS",
                          "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32
// or F64 to F16)
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                          "^FCVTLT_ZPmZ_StoD",
                          "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVTX_ZPmZ_DtoS", "FCVTXNT_ZPmZ_DtoS")>;

// Floating point base2 log, F16
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[C1NanoWrite_3c_1VALU],
               (instregex "^FCPY_ZPmI_[HSD]", "^FDUP_ZI_[HSD]")>;

// Floating point divide, F16
def : InstRW<[C1NanoWrite_8c_1VMC_5rc],
               (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[C1NanoWrite_13c_1VMC_10rc],
               (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[C1NanoWrite_22c_1VMC_19rc],
               (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^F(MAX|MIN)(NM)?_(ZPm[IZ]|ZPZZ|ZPZI)_[HSD]")>;

// Floating point multiply
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^(FSCALE|FMULX)_(ZPmZ|ZPZZ)_[HSD]",
                          "^FMUL_(ZPm[IZ]|ZZZI?|ZPZI|ZPZZ)_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FML[AS]_(ZPmZZ|ZZZI|ZPZZZ)_[HSD]",
                          "^(FMAD|FNMAD|FNML[AS]|FN?MSB)_(ZPmZZ|ZPZZZ)_[HSD]")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FML[AS]L[BT]_ZZZI?_SHH")>;

// Floating point reciprocal estimate, F16
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FR(ECP|SQRT)E_ZZ_H", "^FRECPX_ZPmZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FR(ECP|SQRT)E_ZZ_S", "^FRECPX_ZPmZ_S")>;

// Floating point reciprocal estimate, F64
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^FR(ECP|SQRT)E_ZZ_D", "^FRECPX_ZPmZ_D")>;

// Floating point reciprocal step
def : InstRW<[C1NanoWrite_4c_1VMAC],
               (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]")>;

// Floating point reduction
def : InstRW<[C1NanoWrite_4c_1VALU0],
               (instregex "^F(MAX|MIN)(NM)?V_VPZ_[HSD]")>;

// Floating point reduction, F16
def : InstRW<[C1NanoWrite_12c_1VALU0_5rc],
               (instregex "^FADDV_VPZ_H")>;

// Floating point reduction, F32
def : InstRW<[C1NanoWrite_8c_1VALU_5rc0_2],
               (instregex "^FADDV_VPZ_S")>;

// Floating point reduction, F64
def : InstRW<[C1NanoWrite_4c_2VALU_1rc0_2],
               (instregex "^FADDV_VPZ_D")>;

// Floating point round to integral, F16
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[C1NanoWrite_4c_1VALU],
               (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[C1NanoWrite_8c_1VMC_5rc],
               (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[C1NanoWrite_12c_1VMC_9rc],
               (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64
def : InstRW<[C1NanoWrite_22c_1VMC_19rc],
               (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "^FEXPA_ZZ_[HSD]")>;

// Floating point trigonometric multiply add
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "^FTMAD_ZZI_[HSD]")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[C1NanoWrite_4c_1VMAC], (instregex "^FTSMUL_ZZZ_[HSD]")>;
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^FTSSEL_ZZZ_[HSD]")>;

// SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[C1NanoWrite_4c_1VALU],
             (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[C1NanoWrite_10c_1VMAC_1VALU],
             (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[C1NanoWrite_14c_2VMAC_2VALU],
             (instregex "^BFMMLA_ZZZ")>;

// Multiply accumulate long
def : InstRW<[C1NanoWrite_4c_1VMAC],
             (instregex "^BFMLAL[BT]_ZZZ(I)?")>;

// SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[C1NanoWrite_3c_1LdSt],
             (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LD1[BHWD]_IMM$",
                        "^LD1S?B_[HSD]_IMM$",
                        "^LD1S?H_[SD]_IMM$",
                        "^LD1S?W_D_IMM$" )>;
// Contiguous load, scalar + scalar
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LD1[BHWD]$",
                        "^LD1S?B_[HSD]$",
                        "^LD1S?H_[SD]$",
                        "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LD1R[BHWD]_IMM$",
                        "^LD1RSW_IMM$",
                        "^LD1RS?B_[HSD]_IMM$",
                        "^LD1RS?H_[SD]_IMM$",
                        "^LD1RS?W_D_IMM$",
                        "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LDNT1[BHWD]_ZRI$")>;

// Non temporal load, scalar + scalar
def : InstRW<[C1NanoWrite_3c_2Ld],
             (instregex "^LDNT1[BHWD]_ZRR$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[C1NanoWrite_9c_1LdSt_7rc], (instregex "^LDNT1[BHW]_ZZR_S$",
                                              "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[C1NanoWrite_7c_1LdSt_6rc], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[C1NanoWrite_7c_1LdSt_6rc], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[C1NanoWrite_3c_2Ld], (instregex "^LDFF1[BHWD]$",
                                              "^LDFF1S?B_[HSD]$",
                                              "^LDFF1S?H_[SD]$",
                                              "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[C1NanoWrite_3c_2Ld], (instregex "^LDNF1[BHWD]_IMM$",
                                           "^LDNF1S?B_[HSD]_IMM$",
                                           "^LDNF1S?H_[SD]_IMM$",
                                           "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[C1NanoWrite_3c_1LdSt],
               (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[C1NanoWrite_3c_1LdSt_2rc],
               (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[C1NanoWrite_5c_1LdSt_3rc],
               (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[C1NanoWrite_5c_1LdSt_4rc],
               (instregex "^LD3[BHWD]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[C1NanoWrite_5c_1LdSt_3rc],
               (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[C1NanoWrite_5c_1LdSt_4rc],
               (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[C1NanoWrite_9c_1LdSt_7rc], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                              "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[C1NanoWrite_7c_1LdSt_6rc], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                              "^GLD(FF)?1D_IMM$")>;

// Gather load, 64-bit element size
def : InstRW<[C1NanoWrite_7c_1LdSt_6rc],
             (instregex "^GLD(FF)?1S?[BHW]_D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1S?[BHW]_D(_SCALED)?$",
                        "^GLD(FF)?1D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1D(_SCALED)?$")>;

// Gather load, 32-bit scaled offset
def : InstRW<[C1NanoWrite_7c_1LdSt_7rc],
             (instregex "^GLD(FF)?1S?[HW]_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;

// Gather load, 32-bit unpacked unscaled offset
def : InstRW<[C1NanoWrite_7c_1LdSt_6rc], (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                                              "^GLD(FF)?1W_[SU]XTW$")>;

def : InstRW<[C1NanoWrite<0, C1NanoUnitVALU>], (instregex "^PRF(B|H|W|D).*")>;

// SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instregex "^ST1[BHWD]_IMM$",
                        "^ST1B_[HSD]_IMM$",
                        "^ST1H_[SD]_IMM$",
                        "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instregex "^ST1H(_[SD])?$",
                        "^ST1[BWD]$",
                        "^ST1B_[HSD]$",
                        "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[C1NanoWrite_1LdSt_2rc],
             (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_2rc],
             (instrs ST2H)>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_2rc],
             (instregex "^ST2[BWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[C1NanoWrite_1LdSt_6rc],
             (instregex "^ST3[BHW]_IMM$")>;
def : InstRW<[C1NanoWrite_1LdSt_3rc],
             (instregex "^ST3D_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_6rc],
             (instregex "^ST3[BHW]$")>;
def : InstRW<[C1NanoWrite_1LdSt_3rc],
             (instregex "^ST3D$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[C1NanoWrite_1LdSt_8rc],
             (instregex "^ST4[BHW]_IMM$")>;
def : InstRW<[C1NanoWrite_1LdSt_4rc],
             (instregex "^ST4D_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_8rc],
             (instregex "^ST4[BHW]$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_4rc],
             (instregex "^ST4D$")>;

// Non temporal store, scalar + imm
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instrs STNT1H_ZRR)>;
def : InstRW<[C1NanoWrite_1LdSt_0rc],
             (instregex "^STNT1[BWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[C1NanoWrite_1LdSt_9rc], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[C1NanoWrite_1LdSt_9rc], (instregex "^SST1[BH]_S_IMM$",
                                                "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^SST1[BHW]_D_IMM$",
                                                "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[C1NanoWrite_1LdSt_9rc],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                                "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                                "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[C1NanoWrite_1LdSt_9rc], (instregex "^SST1[BH]_S_[SU]XTW$",
                                                "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^SST1[HW]_D_SCALED$",
                                                "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[C1NanoWrite_1LdSt_7rc], (instregex "^SST1[BHW]_D$",
                                                "^SST1D$")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[C1NanoWrite_1c_1LdSt], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[C1NanoWrite_3c_1LdSt], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[C1NanoWrite_3c_1LdSt], (instrs RDFFRS_PPz)>;

// Set first fault register
// Write to first fault register
def : InstRW<[C1NanoWrite_1c_1LdSt], (instrs SETFFR, WRFFR)>;

// SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^AES[DE]_ZZZ_B$",
                                           "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[C1NanoWrite_3c_1VALU], (instregex "^(BCAX|EOR3)_ZZZZ$")>;
def : InstRW<[C1NanoWrite_4c_1VALU], (instregex "^XAR_ZZZI_[BHSD]$")>;

def : InstRW<[C1NanoWrite_3c_1VALU_0rc], (instregex "^RAX1_ZZZ_D$")>;

// Crypto SM4 ops
def : InstRW<[C1NanoWrite_9c_1VMC_7rc], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

// MOPS instructions
// -----------------------------------------------------------------------------

//Memory Copy Forward-only Prologue
def : InstRW<[C1NanoWrite_2c_1ALU_1LdSt_2rc], (instregex "^CPYFP")>;

// Memory Copy Forward-only Main
// Note: we model the base latency here.

def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^CPYFM")>;

// Memory Copy Forward-only Epilogue
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^CPYFE")>;

// Memory Copy Prologue
def : InstRW<[C1NanoWrite_3c_1ALU_1LdSt_3rc], (instregex "^CPYP")>;

// Memory Copy Main
// Note: we model the base latency here.
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^CPYM")>;

// Memory Copy Epilogue
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^CPYE")>;

// Memory Set Prologue
def : InstRW<[C1NanoWrite_2c_1ALU_1LdSt_2rc], (instregex "^SETP")>;

// Memory Set Main
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^SETM")>;

// Memory Set Epilogue
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^SETE")>;

// Memory Set with tag setting Prologue
def : InstRW<[C1NanoWrite_2c_1ALU_1LdSt_2rc], (instregex "^SETGP")>;

// Memory Set with tag setting Main
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^SETGM")>;

// Memory Set with tag setting Epilogue
def : InstRW<[C1NanoWrite_1c_1ALU_1LdSt_1rc], (instregex "^MOPSSETGE")>;

}
