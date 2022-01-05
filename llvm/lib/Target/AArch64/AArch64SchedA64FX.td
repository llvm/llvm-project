//=- AArch64SchedA64FX.td - Fujitsu A64FX Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Fujitsu A64FX processors.
//
//===----------------------------------------------------------------------===//

def A64FXModel : SchedMachineModel {
  let IssueWidth            =   6; // 6 micro-ops dispatched at a time.
  let MicroOpBufferSize     = 180; // 180 entries in micro-op re-order buffer.
  let LoadLatency           =   5; // Optimistic load latency.
  let MispredictPenalty     =  12; // Extra cycles for mispredicted branch.
  // Determined via a mix of micro-arch details and experimentation.
  let LoopMicroOpBufferSize = 128;
  let PostRAScheduler       =   1; // Using PostRA sched.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures =
    [HasSVE2, HasSVE2AES, HasSVE2SM4, HasSVE2SHA3, HasSVE2BitPerm, HasPAuth,
     HasSVE2orStreamingSVE];

  let FullInstRWOverlapCheck = 0;
}

let SchedModel = A64FXModel in {

// Define the issue ports.

// A64FXIP*

// Port 0
def A64FXIPFLA : ProcResource<1>;

// Port 1
def A64FXIPPR : ProcResource<1>;

// Port 2
def A64FXIPEXA : ProcResource<1>;

// Port 3
def A64FXIPFLB : ProcResource<1>;

// Port 4
def A64FXIPEXB : ProcResource<1>;

// Port 5
def A64FXIPEAGA : ProcResource<1>;

// Port 6
def A64FXIPEAGB : ProcResource<1>;

// Port 7
def A64FXIPBR : ProcResource<1>;

// Define groups for the functional units on each issue port.  Each group
// created will be used by a WriteRes later on.

def A64FXGI7 : ProcResGroup<[A64FXIPBR]>;

def A64FXGI0 : ProcResGroup<[A64FXIPFLA]>;

def A64FXGI1 : ProcResGroup<[A64FXIPPR]>;

def A64FXGI2 : ProcResGroup<[A64FXIPEXA]>;

def A64FXGI3 : ProcResGroup<[A64FXIPFLB]>;

def A64FXGI4 : ProcResGroup<[A64FXIPEXB]>;

def A64FXGI5 : ProcResGroup<[A64FXIPEAGA]>;

def A64FXGI6 : ProcResGroup<[A64FXIPEAGB]>;

def A64FXGI03 : ProcResGroup<[A64FXIPFLA, A64FXIPFLB]>;

def A64FXGI01 : ProcResGroup<[A64FXIPFLA, A64FXIPPR]>;

def A64FXGI02 : ProcResGroup<[A64FXIPFLA, A64FXIPEXA]>;

def A64FXGI12 : ProcResGroup<[A64FXIPEXA, A64FXIPPR]>;

def A64FXGI15 : ProcResGroup<[A64FXIPEAGA, A64FXIPPR]>;

def A64FXGI05 : ProcResGroup<[A64FXIPFLA, A64FXIPEAGA]>;

def A64FXGI24 : ProcResGroup<[A64FXIPEXA, A64FXIPEXB]>;

def A64FXGI124 : ProcResGroup<[A64FXIPEXA, A64FXIPEXB, A64FXIPPR]>;

def A64FXGI056 : ProcResGroup<[A64FXIPFLA, A64FXIPEAGA, A64FXIPEAGB]>;

def A64FXGI0256 : ProcResGroup<[A64FXIPFLA, A64FXIPEXA, A64FXIPEAGA, A64FXIPEAGB]>;

def A64FXGI56 : ProcResGroup<[A64FXIPEAGA, A64FXIPEAGB]>;

def A64FXGI2456 : ProcResGroup<[A64FXIPEXA, A64FXIPEXB, A64FXIPEAGA, A64FXIPEAGB]>;

def A64FXAny : ProcResGroup<[A64FXIPFLA, A64FXIPPR, A64FXIPEXA, A64FXIPFLB,
                             A64FXIPEXB, A64FXIPEAGA, A64FXIPEAGB, A64FXIPBR]> {
  let BufferSize = 60;
}

def A64FXWrite_6Cyc : SchedWriteRes<[]> {
  let Latency = 6;
}

def A64FXWrite_1Cyc_GI7 : SchedWriteRes<[A64FXGI7]> {
  let Latency = 1;
}

def A64FXWrite_2Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 2;
}

def A64FXWrite_4Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 4;
}

def A64FXWrite_5Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 5;
}

def A64FXWrite_6Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 6;
}

def A64FXWrite_8Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 8;
}

def A64FXWrite_9Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 9;
}

def A64FXWrite_13Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 13;
}

def A64FXWrite_37Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 37;
}

def A64FXWrite_98Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 98;
}

def A64FXWrite_134Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 134;
}

def A64FXWrite_154Cyc_GI0 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 154;
}

def A64FXWrite_4Cyc_GI01 : SchedWriteRes<[A64FXGI01]> {
  let Latency = 4;
}

def A64FXWrite_6Cyc_GI01 : SchedWriteRes<[A64FXGI01]> {
  let Latency = 6;
}

def A64FXWrite_8Cyc_GI01 : SchedWriteRes<[A64FXGI01]> {
  let Latency = 8;
}

def A64FXWrite_12Cyc_GI01 : SchedWriteRes<[A64FXGI01]> {
  let Latency = 12;
}

def A64FXWrite_10Cyc_GI02 : SchedWriteRes<[A64FXGI02]> {
  let Latency = 10;
}

def A64FXWrite_17Cyc_GI02 : SchedWriteRes<[A64FXGI02]> {
  let Latency = 17;
}

def A64FXWrite_21Cyc_GI02 : SchedWriteRes<[A64FXGI02]> {
  let Latency = 21;
}

def A64FXWrite_3Cyc_GI1 : SchedWriteRes<[A64FXGI1]> {
  let Latency = 3;
}

def A64FXWrite_6Cyc_NGI1 : SchedWriteRes<[A64FXGI1]> {
  let Latency = 3;
  let NumMicroOps = 2;
}

def A64FXWrite_4Cyc_GI12 : SchedWriteRes<[A64FXGI12]> {
  let Latency = 4;
}

def A64FXWrite_3Cyc_GI2 : SchedWriteRes<[A64FXGI2]> {
  let Latency = 3;
}

def A64FXWrite_5Cyc_GI2 : SchedWriteRes<[A64FXGI2]> {
  let Latency = 5;
}

def A64FXWrite_6Cyc_GI2 : SchedWriteRes<[A64FXGI2]> {
  let Latency = 6;
}

def A64FXWrite_4Cyc_GI3 : SchedWriteRes<[A64FXGI3]> {
  let Latency = 4;
}

def A64FXWrite_6Cyc_GI3 : SchedWriteRes<[A64FXGI3]> {
  let Latency = 6;
}

def A64FXWrite_6Cyc_GI15 : SchedWriteRes<[A64FXGI15]> {
  let Latency = 6;
}

def A64FXWrite_3Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 3;
}

def A64FXWrite_4Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 4;
}

def A64FXWrite_6Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 6;
}

def A64FXWrite_8Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 8;
}

def A64FXWrite_9Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 9;
}

def A64FXWrite_10Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
}

def A64FXWrite_12Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 12;
}

def A64FXWrite_14Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
}

def A64FXWrite_15Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 15;
}

def A64FXWrite_15Cyc_NGI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 15;
  let NumMicroOps = 2;
}

def A64FXWrite_18Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 18;
}

def A64FXWrite_45Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 45;
}

def A64FXWrite_60Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 60;
}

def A64FXWrite_75Cyc_GI03 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 75;
}

def A64FXWrite_6Cyc_GI05 : SchedWriteRes<[A64FXGI05]> {
  let Latency = 6;
}

def A64FXWrite_10Cyc_GI4 : SchedWriteRes<[A64FXGI4]> {
  let Latency = 10;
}

def A64FXWrite_12Cyc_GI4 : SchedWriteRes<[A64FXGI4]> {
  let Latency = 12;
}

def A64FXWrite_20Cyc_GI4 : SchedWriteRes<[A64FXGI4]> {
  let Latency = 20;
}

def A64FXWrite_5Cyc_GI5 : SchedWriteRes<[A64FXGI5]> {
  let Latency = 5;
}

def A64FXWrite_11Cyc_GI5 : SchedWriteRes<[A64FXGI5]> {
  let Latency = 11;
}

def A64FXWrite_5Cyc_GI6 : SchedWriteRes<[A64FXGI6]> {
  let Latency = 5;
}

def A64FXWrite_1Cyc_GI24 : SchedWriteRes<[A64FXGI24]> {
  let Latency = 1;
}

def A64FXWrite_2Cyc_GI24 : SchedWriteRes<[A64FXGI24]> {
  let Latency = 2;
}

def A64FXWrite_4Cyc_NGI24 : SchedWriteRes<[A64FXGI24]> {
  let Latency = 4;
  let NumMicroOps = 4;
}

def A64FXWrite_6Cyc_GI124: SchedWriteRes<[A64FXGI124]> {
  let Latency = 6;
}

def A64FXWrite_8Cyc_GI124 : SchedWriteRes<[A64FXGI124]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

def A64FXWrite_6Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_1Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 1;
}

def A64FXWrite_5Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 5;
}

def A64FXWrite_8Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
}

def A64FXWrite_11Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
}

def A64FXWrite_44Cyc_GI56 : SchedWriteRes<[A64FXGI56]> {
  let Latency = 44;
}

def A64FXWrite_10Cyc_GI056 : SchedWriteRes<[A64FXGI056]> {
  let Latency = 10;
}

def A64FXWrite_15Cyc_GI056 : SchedWriteRes<[A64FXGI056]> {
  let Latency = 15;
}

def A64FXWrite_19Cyc_GI056 : SchedWriteRes<[A64FXGI056]> {
  let Latency = 19;
}

def A64FXWrite_25Cyc_GI056 : SchedWriteRes<[A64FXGI056]> {
  let Latency = 25;
}

def A64FXWrite_14Cyc_GI0256 : SchedWriteRes<[A64FXGI0256]> {
  let Latency = 14;
}

def A64FXWrite_19Cyc_GI0256 : SchedWriteRes<[A64FXGI0256]> {
  let Latency = 19;
}

def A64FXWrite_29Cyc_GI0256 : SchedWriteRes<[A64FXGI0256]> {
  let Latency = 29;
}

def A64FXWrite_LDNP: SchedWriteRes<[A64FXGI56]> {
  let Latency = 5;
  let NumMicroOps = 2;
}

def A64FXWrite_LDP01: SchedWriteRes<[A64FXGI2456]> {
  let Latency = 5;
  let NumMicroOps = 3;
}

def A64FXWrite_LDR01: SchedWriteRes<[A64FXGI2456]> {
  let Latency = 5;
  let NumMicroOps = 2;
}

def A64FXWrite_LD102: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

def A64FXWrite_LD103: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 2;

}

def A64FXWrite_LD104: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 3;
}

def A64FXWrite_LD105: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 3;
}

def A64FXWrite_LD106: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 4;
}

def A64FXWrite_LD107: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 4;
}

def A64FXWrite_LD108: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

def A64FXWrite_LD109: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 2;
}

def A64FXWrite_LD110: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 3;
}

def A64FXWrite_LD111: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 3;
}

def A64FXWrite_LD112: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 4;
}

def A64FXWrite_LD113: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 4;
}

def A64FXWrite_LD114: SchedWriteRes<[A64FXGI56]> {
  let Latency = 8;
  let NumMicroOps = 5;
}

def A64FXWrite_LD115: SchedWriteRes<[A64FXGI56]> {
  let Latency = 11;
  let NumMicroOps = 5;
}

def A64FXWrite_LD1I0: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

def A64FXWrite_LD1I1: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 3;
}

def A64FXWrite_LD2I0: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 4;
}

def A64FXWrite_LD2I1: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 5;
}

def A64FXWrite_LD3I0: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 6;
}

def A64FXWrite_LD3I1: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 7;
}

def A64FXWrite_LD4I0: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 8;
}

def A64FXWrite_LD4I1: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
  let NumMicroOps = 9;
}

def A64FXWrite_1Cyc_GI2456 : SchedWriteRes<[A64FXGI2456]> {
  let Latency = 1;
}

def A64FXWrite_FMOV_GV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
}

def A64FXWrite_FMOV_VG14 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
}

def A64FXWrite_FMOV_VG : SchedWriteRes<[A64FXGI03]> {
  let Latency = 25;
}

def A64FXWrite_ADDLV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 12;
}

def A64FXWrite_MULLE : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
}

def A64FXWrite_MULLV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
}

def A64FXWrite_MADDL : SchedWriteRes<[A64FXGI03]> {
  let Latency = 6;
}

def A64FXWrite_ABA : SchedWriteRes<[A64FXGI03]> {
  let Latency = 8;
}

def A64FXWrite_ABAL : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
}

def A64FXWrite_ADDLV1 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 12;
  let NumMicroOps = 6;
}

def A64FXWrite_MINMAXV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
  let NumMicroOps = 6;
}

def A64FXWrite_SQRDMULH : SchedWriteRes<[A64FXGI03]> {
  let Latency = 9;
}

def A64FXWrite_PMUL : SchedWriteRes<[A64FXGI03]> {
  let Latency = 8;
}


def A64FXWrite_SRSRAV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 8;
  let NumMicroOps = 3;
}

def A64FXWrite_SSRAV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 8;
  let NumMicroOps = 2;
}

def A64FXWrite_RSHRN : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 3;
}

def A64FXWrite_SHRN : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 2;
}


def A64FXWrite_ADDP : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 3;
}

def A64FXWrite_FMULXE : SchedWriteRes<[A64FXGI03]> {
  let Latency = 15;
  let NumMicroOps = 2;
}

def A64FXWrite_FADDPV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 15;
  let NumMicroOps = 3;
}

def A64FXWrite_SADALP : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 3;
}

def A64FXWrite_SADDLP : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 2;
}

def A64FXWrite_FCVTXNV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 15;
  let NumMicroOps = 2;
}

def A64FXWrite_FMAXVVH : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
  let NumMicroOps = 7;
}

def A64FXWrite_FMAXVVS : SchedWriteRes<[A64FXGI03]> {
  let Latency = 14;
}

def A64FXWrite_BIF : SchedWriteRes<[A64FXGI03]> {
  let Latency = 5;
}

def A64FXWrite_DUPGENERAL : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
}

def A64FXWrite_SHA00 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 9;
}

def A64FXWrite_SHA01 : SchedWriteRes<[A64FXGI0]> {
  let Latency = 12;
}

def A64FXWrite_SMOV : SchedWriteRes<[A64FXGI03]> {
  let Latency = 25;
}

def A64FXWrite_TBX1 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 3;
}

def A64FXWrite_TBX2 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 5;
}

def A64FXWrite_TBX3 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 7;
}

def A64FXWrite_TBX4 : SchedWriteRes<[A64FXGI03]> {
  let Latency = 10;
  let NumMicroOps = 9;
}

def A64FXWrite_PREF0: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_PREF1: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_SWP: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_STUR: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_STNP: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_STP01: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_ST10: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_ST11: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_ST12: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_ST13: SchedWriteRes<[A64FXGI56]> {
  let Latency = 0;
}

def A64FXWrite_ST14: SchedWriteRes<[A64FXGI56]> {
  let Latency = 1;
}

def A64FXWrite_ST15: SchedWriteRes<[A64FXGI56]> {
  let Latency = 1;
}

def A64FXWrite_ST16: SchedWriteRes<[A64FXGI56]> {
  let Latency = 1;
}

def A64FXWrite_ST17: SchedWriteRes<[A64FXGI56]> {
  let Latency = 1;
}

def A64FXWrite_ST1W_6: SchedWriteRes<[A64FXGI056]> {
  let Latency = 6;
}

def A64FXWrite_ST2W_7: SchedWriteRes<[A64FXGI056]> {
  let Latency = 7;
}

def A64FXWrite_ST3W_8: SchedWriteRes<[A64FXGI056]> {
  let Latency = 8;
}

def A64FXWrite_ST4W_9: SchedWriteRes<[A64FXGI056]> {
  let Latency = 9;
}

def A64FXWrite_ST1W_15: SchedWriteRes<[A64FXGI056]> {
  let Latency = 15;
}

def A64FXWrite_ST1W_19: SchedWriteRes<[A64FXGI056]> {
  let Latency = 19;
}

def A64FXWrite_CAS: SchedWriteRes<[A64FXGI56]> {
  let Latency = 7;
}

// Define commonly used read types.

// No forwarding is provided for these types.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadST,      0>;
def : ReadAdvance<ReadVLD,     0>;

//===----------------------------------------------------------------------===//
// 3. Instruction Tables.

//---
// 3.1 Branch Instructions
//---

// Branch, immed
// Branch and link, immed
// Compare and branch
def : WriteRes<WriteBr,      [A64FXGI7]> {
  let Latency = 1;
}

// Branch, register
// Branch and link, register != LR
// Branch and link, register = LR
def : WriteRes<WriteBrReg,   [A64FXGI7]> {
  let Latency = 1;
}

def : WriteRes<WriteSys,     []> { let Latency = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }

def : WriteRes<WriteAtomic,  []> {
  let Latency = 4;
}

//---
// Branch
//---
def : InstRW<[A64FXWrite_1Cyc_GI7], (instrs B, BL, BR, BLR)>;
def : InstRW<[A64FXWrite_1Cyc_GI7], (instrs RET)>;
def : InstRW<[A64FXWrite_1Cyc_GI7], (instregex "^B..$")>;
def : InstRW<[A64FXWrite_1Cyc_GI7],
            (instregex "^CBZ", "^CBNZ", "^TBZ", "^TBNZ")>;

//---
// 3.2 Arithmetic and Logical Instructions
// 3.3 Move and Shift Instructions
//---

// ALU, basic
// Conditional compare
// Conditional select
// Address generation
def : WriteRes<WriteI,       [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

def : InstRW<[WriteI],
            (instregex "ADD?(W|X)r(i|r|s|x)",   "ADDS?(W|X)r(i|r|s|x)(64)?",
                       "AND?(W|X)r(i|r|s|x)",   "ANDS?(W|X)r(i|r|s|x)",
                       "ADC(W|X)r",
                       "BIC?(W|X)r(i|r|s|x)",   "BICS?(W|X)r(i|r|s|x)",
                       "EON?(W|X)r(i|r|s|x)",   "ORN?(W|X)r(i|r|s|x)",
                       "ORR?(W|X)r(i|r|s|x)",   "SUB?(W|X)r(i|r|s|x)",
                       "SUBS?(W|X)r(i|r|s|x)",  "SBC(W|X)r",
                       "SBCS(W|X)r",            "CCMN(W|X)(i|r)",
                       "CCMP(W|X)(i|r)",        "CSEL(W|X)r",
                       "CSINC(W|X)r",           "CSINV(W|X)r",
                       "CSNEG(W|X)r")>;

def : InstRW<[WriteI], (instrs COPY)>;

// ALU, extend and/or shift
def : WriteRes<WriteISReg,   [A64FXGI2456]> {
  let Latency = 2;
  let ResourceCycles = [1];
}

def : InstRW<[WriteISReg],
            (instregex "ADD?(W|X)r(i|r|s|x)",   "ADDS?(W|X)r(i|r|s|x)(64)?",
                       "AND?(W|X)r(i|r|s|x)",   "ANDS?(W|X)r(i|r|s|x)",
                       "ADC(W|X)r",
                       "BIC?(W|X)r(i|r|s|x)",   "BICS?(W|X)r(i|r|s|x)",
                       "EON?(W|X)r(i|r|s|x)",   "ORN?(W|X)r(i|r|s|x)",
                       "ORR?(W|X)r(i|r|s|x)",   "SUB?(W|X)r(i|r|s|x)",
                       "SUBS?(W|X)r(i|r|s|x)",  "SBC(W|X)r",
                       "SBCS(W|X)r",            "CCMN(W|X)(i|r)",
                       "CCMP(W|X)(i|r)",        "CSEL(W|X)r",
                       "CSINC(W|X)r",           "CSINV(W|X)r",
                       "CSNEG(W|X)r")>;

def : WriteRes<WriteIEReg,   [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

def : InstRW<[WriteIEReg],
            (instregex "ADD?(W|X)r(i|r|s|x)",   "ADDS?(W|X)r(i|r|s|x)(64)?",
                       "AND?(W|X)r(i|r|s|x)",   "ANDS?(W|X)r(i|r|s|x)",
                       "ADC(W|X)r",
                       "BIC?(W|X)r(i|r|s|x)",   "BICS?(W|X)r(i|r|s|x)",
                       "EON?(W|X)r(i|r|s|x)",   "ORN?(W|X)r(i|r|s|x)",
                       "ORR?(W|X)r(i|r|s|x)",   "SUB?(W|X)r(i|r|s|x)",
                       "SUBS?(W|X)r(i|r|s|x)",  "SBC(W|X)r",
                       "SBCS(W|X)r",            "CCMN(W|X)(i|r)",
                       "CCMP(W|X)(i|r)",        "CSEL(W|X)r",
                       "CSINC(W|X)r",           "CSINV(W|X)r",
                       "CSNEG(W|X)r")>;

// Move immed
def : WriteRes<WriteImm,     [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

def : InstRW<[A64FXWrite_1Cyc_GI2456],
            (instrs MOVKWi, MOVKXi, MOVNWi, MOVNXi, MOVZWi, MOVZXi)>;

def : InstRW<[A64FXWrite_2Cyc_GI24],
            (instrs ASRVWr, ASRVXr, LSLVWr, LSLVXr, RORVWr, RORVXr)>;

// Variable shift
def : WriteRes<WriteIS,      [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

//---
// 3.4 Divide and Multiply Instructions
//---

// Divide, W-form
def : WriteRes<WriteID32,    [A64FXGI4]> {
  let Latency = 39;
  let ResourceCycles = [39];
}

// Divide, X-form
def : WriteRes<WriteID64,    [A64FXGI4]> {
  let Latency = 23;
  let ResourceCycles = [23];
}

// Multiply accumulate, W-form
def : WriteRes<WriteIM32,    [A64FXGI2456]> {
  let Latency = 5;
  let ResourceCycles = [1];
}

// Multiply accumulate, X-form
def : WriteRes<WriteIM64,    [A64FXGI2456]> {
  let Latency = 5;
  let ResourceCycles = [1];
}

def : InstRW<[WriteIM32], (instrs MADDWrrr, MSUBWrrr)>;
def : InstRW<[WriteIM32], (instrs MADDXrrr, MSUBXrrr)>;
def : InstRW<[A64FXWrite_MADDL],
            (instregex "(S|U)(MADDL|MSUBL)rrr")>;

def : InstRW<[WriteID32], (instrs SDIVWr, UDIVWr)>;
def : InstRW<[WriteID64], (instrs SDIVXr, UDIVXr)>;

// Bitfield extract, two reg
def : WriteRes<WriteExtr,    [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

// Multiply high
def : InstRW<[A64FXWrite_5Cyc_GI2], (instrs SMULHrr, UMULHrr)>;

// Miscellaneous Data-Processing Instructions
// Bitfield extract
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs EXTRWrri, EXTRXrri)>;

// Bitifield move - basic
def : InstRW<[A64FXWrite_1Cyc_GI24],
            (instrs SBFMWri, SBFMXri, UBFMWri, UBFMXri)>;

// Bitfield move, insert
def : InstRW<[A64FXWrite_4Cyc_NGI24], (instregex "^BFM")>;
def : InstRW<[A64FXWrite_1Cyc_GI24], (instregex "(S|U)?BFM.*")>;

// Count leading
def : InstRW<[A64FXWrite_2Cyc_GI0], (instregex "^CLS(W|X)r$",
                                               "^CLZ(W|X)r$")>;

// Reverse bits
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs RBITWr, RBITXr)>;

// Cryptography Extensions
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^AES[DE]")>;
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^AESI?MC")>;
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^PMULL")>;
def : InstRW<[A64FXWrite_SHA00], (instregex "^SHA1SU0")>;
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^SHA1(H|SU1)")>;
def : InstRW<[A64FXWrite_SHA01], (instregex "^SHA1[CMP]")>;
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^SHA256SU0")>;
def : InstRW<[A64FXWrite_8Cyc_GI0], (instregex "^SHA256SU1")>;
def : InstRW<[A64FXWrite_SHA01], (instregex "^SHA256(H|H2)")>;

// CRC Instructions
def : InstRW<[A64FXWrite_10Cyc_GI4], (instrs CRC32Brr, CRC32Hrr)>;
def : InstRW<[A64FXWrite_12Cyc_GI4], (instrs CRC32Wrr)>;
def : InstRW<[A64FXWrite_20Cyc_GI4], (instrs CRC32Xrr)>;

def : InstRW<[A64FXWrite_10Cyc_GI4], (instrs CRC32CBrr, CRC32CHrr)>;
def : InstRW<[A64FXWrite_12Cyc_GI4], (instrs CRC32CWrr)>;
def : InstRW<[A64FXWrite_20Cyc_GI4], (instrs CRC32CXrr)>;

// Reverse bits/bytes
// NOTE: Handled by WriteI.

//---
// 3.6 Load Instructions
// 3.10 FP Load Instructions
//---

// Load register, literal
// Load register, unscaled immed
// Load register, immed unprivileged
// Load register, unsigned immed
def : WriteRes<WriteLD,      [A64FXGI56]> {
  let Latency = 4;
  let ResourceCycles = [3];
}

// Load register, immed post-index
// NOTE: Handled by WriteLD, WriteI.
// Load register, immed pre-index
// NOTE: Handled by WriteLD, WriteAdr.
def : WriteRes<WriteAdr,     [A64FXGI2456]> {
  let Latency = 1;
  let ResourceCycles = [1];
}

// Load pair, immed offset, normal
// Load pair, immed offset, signed words, base != SP
// Load pair, immed offset signed words, base = SP
// LDP only breaks into *one* LS micro-op.  Thus
// the resources are handled by WriteLD.
def : WriteRes<WriteLDHi,    []> {
  let Latency = 5;
}

// Load register offset, basic
// Load register, register offset, scale by 4/8
// Load register, register offset, scale by 2
// Load register offset, extend
// Load register, register offset, extend, scale by 4/8
// Load register, register offset, extend, scale by 2
def A64FXWriteLDIdx : SchedWriteVariant<[
  SchedVar<ScaledIdxPred, [A64FXWrite_1Cyc_GI56]>,
  SchedVar<NoSchedPred,   [A64FXWrite_1Cyc_GI56]>]>;
def : SchedAlias<WriteLDIdx, A64FXWriteLDIdx>;

def A64FXReadAdrBase : SchedReadVariant<[
  SchedVar<ScaledIdxPred, [ReadDefault]>,
  SchedVar<NoSchedPred,   [ReadDefault]>]>;
def : SchedAlias<ReadAdrBase, A64FXReadAdrBase>;

// Load pair, immed pre-index, normal
// Load pair, immed pre-index, signed words
// Load pair, immed post-index, normal
// Load pair, immed post-index, signed words
// NOTE: Handled by WriteLD, WriteLDHi, WriteAdr.

def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDNPDi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDNPQi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDNPSi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDNPWi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDNPXi)>;

def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPDi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPQi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPSi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPSWi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPWi)>;
def : InstRW<[A64FXWrite_LDNP, WriteLDHi], (instrs LDPXi)>;

def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDRBui)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDRDui)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDRHui)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDRQui)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDRSui)>;

def : InstRW<[A64FXWrite_5Cyc_GI6], (instrs LDRDl)>;
def : InstRW<[A64FXWrite_5Cyc_GI6], (instrs LDRQl)>;
def : InstRW<[A64FXWrite_5Cyc_GI6], (instrs LDRWl)>;
def : InstRW<[A64FXWrite_5Cyc_GI6], (instrs LDRXl)>;

def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRBi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRHi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRWi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRXi)>;

def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRSBWi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRSBXi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRSHWi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRSHXi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDTRSWi)>;

def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPDpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPQpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPSpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPWpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPWpre)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRBpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRDpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRHpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRQpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRWpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRXpre)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSBWpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSBXpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSBWpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSBXpost)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSHWpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSHXpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSHWpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSHXpost)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRBBpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRBBpost)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRHHpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRHHpost)>;

def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPDpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPQpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPSpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPWpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPXpost)>;

def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRBpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRDpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRHpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRQpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRSpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRWpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRXpost)>;

def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPDpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPQpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPSpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPWpre)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPXpre)>;

def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRBpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRDpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRHpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRQpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRSpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRWpre)>;
def : InstRW<[A64FXWrite_LDR01, WriteAdr], (instrs LDRXpre)>;

def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPDpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPQpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPSpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPWpost)>;
def : InstRW<[A64FXWrite_LDP01, WriteLDHi, WriteAdr],
            (instrs LDPXpost)>;

def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRBpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRDpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRHpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRQpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRSpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRWpost)>;
def : InstRW<[A64FXWrite_LDR01, WriteI], (instrs LDRXpost)>;

def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRBroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRDroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRHroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRHHroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRQroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSHWroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSHXroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRWroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRXroW)>;

def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRBroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRDroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRHHroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRHroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRQroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSHWroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRSHXroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRWroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase], (instrs LDRXroX)>;

def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRBroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRBroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
             (instrs LDRDroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRHroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRHHroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRQroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSHWroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSHXroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRWroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRXroW)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRBroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRDroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRHroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRHHroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRQroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSHWroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRSHXroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRWroX)>;
def : InstRW<[A64FXWrite_5Cyc_GI56, ReadAdrBase],
            (instrs LDRXroX)>;

def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURBi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURBBi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURDi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURHi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURHHi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURQi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURXi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSBWi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSBXi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSHWi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSHXi)>;
def : InstRW<[A64FXWrite_5Cyc_GI56], (instrs LDURSWi)>;

//---
// Prefetch
//---
def : InstRW<[A64FXWrite_PREF0], (instrs PRFMl)>;
def : InstRW<[A64FXWrite_PREF1], (instrs PRFUMi)>;
def : InstRW<[A64FXWrite_PREF1], (instrs PRFMui)>;
def : InstRW<[A64FXWrite_PREF1], (instrs PRFMroW)>;
def : InstRW<[A64FXWrite_PREF1], (instrs PRFMroX)>;

//--
// 3.7 Store Instructions
// 3.11 FP Store Instructions
//--

// Store register, unscaled immed
// Store register, immed unprivileged
// Store register, unsigned immed
def : WriteRes<WriteST,      [A64FXGI56]> {
  let Latency = 1;
}

// Store register, immed post-index
// NOTE: Handled by WriteAdr, WriteST, ReadAdrBase

// Store register, immed pre-index
// NOTE: Handled by WriteAdr, WriteST

// Store register, register offset, basic
// Store register, register offset, scaled by 4/8
// Store register, register offset, scaled by 2
// Store register, register offset, extend
// Store register, register offset, extend, scale by 4/8
// Store register, register offset, extend, scale by 1
def : WriteRes<WriteSTIdx, [A64FXGI56, A64FXGI2456]> {
  let Latency = 1;
}

// Store pair, immed offset, W-form
// Store pair, immed offset, X-form
def : WriteRes<WriteSTP,     [A64FXGI56]> {
  let Latency = 1;
}

// Store pair, immed post-index, W-form
// Store pair, immed post-index, X-form
// Store pair, immed pre-index, W-form
// Store pair, immed pre-index, X-form
// NOTE: Handled by WriteAdr, WriteSTP.

def : InstRW<[A64FXWrite_STUR], (instrs STURBi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURBBi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURDi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURHi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURHHi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURQi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURSi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURWi)>;
def : InstRW<[A64FXWrite_STUR], (instrs STURXi)>;

def : InstRW<[WriteAdr, A64FXWrite_STUR], (instrs STTRBi)>;
def : InstRW<[WriteAdr, A64FXWrite_STUR], (instrs STTRHi)>;
def : InstRW<[WriteAdr, A64FXWrite_STUR], (instrs STTRWi)>;
def : InstRW<[WriteAdr, A64FXWrite_STUR], (instrs STTRXi)>;

def : InstRW<[A64FXWrite_STNP], (instrs STNPDi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STNPQi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STNPXi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STNPWi)>;

def : InstRW<[A64FXWrite_STNP], (instrs STPDi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STPQi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STPXi)>;
def : InstRW<[A64FXWrite_STNP], (instrs STPWi)>;

def : InstRW<[A64FXWrite_STUR], (instrs STRBui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRBui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRDui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRDui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRHui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRHui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRQui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRQui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRXui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRXui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRWui)>;
def : InstRW<[A64FXWrite_STUR], (instrs STRWui)>;

def : InstRW<[A64FXWrite_STP01],
            (instrs STPDpre, STPDpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPDpre, STPDpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPDpre, STPDpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPDpre, STPDpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPQpre, STPQpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPQpre, STPQpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPQpre, STPQpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPQpre, STPQpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPSpre, STPSpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPSpre, STPSpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPSpre, STPSpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPSpre, STPSpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPWpre, STPWpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPWpre, STPWpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPWpre, STPWpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPWpre, STPWpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPXpre, STPXpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPXpre, STPXpost)>;
def : InstRW<[A64FXWrite_STP01],
            (instrs STPXpre, STPXpost)>;
def : InstRW<[A64FXWrite_STP01, ReadAdrBase],
            (instrs STPXpre, STPXpost)>;

def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRBpre, STRBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRBpre, STRBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRBpre, STRBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRBpre, STRBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRBBpre, STRBBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRBBpre, STRBBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRBBpre, STRBBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRBBpre, STRBBpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRDpre, STRDpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRDpre, STRDpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRDpre, STRDpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRDpre, STRDpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRHpre, STRHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRHpre, STRHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRHpre, STRHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRHpre, STRHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRHHpre, STRHHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRHHpre, STRHHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRHHpre, STRHHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRHHpre, STRHHpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRQpre, STRQpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRQpre, STRQpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRQpre, STRQpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRQpre, STRQpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRSpre, STRSpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRSpre, STRSpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRSpre, STRSpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRSpre, STRSpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRWpre, STRWpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRWpre, STRWpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRWpre, STRWpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRWpre, STRWpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRXpre, STRXpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRXpre, STRXpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01],
            (instrs STRXpre, STRXpost)>;
def : InstRW<[WriteAdr, A64FXWrite_STP01, ReadAdrBase],
            (instrs STRXpre, STRXpost)>;

def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRBroW, STRBroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRBroW, STRBroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRBBroW, STRBBroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRBBroW, STRBBroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRDroW, STRDroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRDroW, STRDroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRHroW, STRHroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRHroW, STRHroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRHHroW, STRHHroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRHHroW, STRHHroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRQroW, STRQroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRQroW, STRQroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRSroW, STRSroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRSroW, STRSroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRWroW, STRWroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRWroW, STRWroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRXroW, STRXroX)>;
def : InstRW<[A64FXWrite_STUR, ReadAdrBase],
            (instrs STRXroW, STRXroX)>;

//---
// 3.8 FP Data Processing Instructions
//---

// FP absolute value
// FP min/max
// FP negate
def : WriteRes<WriteF,       [A64FXGI03]> {
  let Latency = 4;
  let ResourceCycles = [2];
}

// FP arithmetic

def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FADDDrr, FADDHrr)>;
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FSUBDrr, FSUBHrr)>;

// FP compare
def : WriteRes<WriteFCmp,    [A64FXGI03]> {
  let Latency = 4;
  let ResourceCycles = [2];
}

// FP Div, Sqrt
def : WriteRes<WriteFDiv, [A64FXGI0]> {
  let Latency = 43;
}

def A64FXXWriteFDiv : SchedWriteRes<[A64FXGI0]> {
  let Latency = 38;
}

def A64FXXWriteFDivSP : SchedWriteRes<[A64FXGI0]> {
  let Latency = 29;
}

def A64FXXWriteFDivDP : SchedWriteRes<[A64FXGI0]> {
  let Latency = 43;
}

def A64FXXWriteFSqrtSP : SchedWriteRes<[A64FXGI0]> {
  let Latency = 29;
}

def A64FXXWriteFSqrtDP : SchedWriteRes<[A64FXGI0]> {
  let Latency = 43;
}

// FP divide, S-form
// FP square root, S-form
def : InstRW<[A64FXXWriteFDivSP], (instrs FDIVSrr)>;
def : InstRW<[A64FXXWriteFSqrtSP], (instrs FSQRTSr)>;
def : InstRW<[A64FXXWriteFDivSP], (instregex "^FDIVv.*32$")>;
def : InstRW<[A64FXXWriteFSqrtSP], (instregex "^.*SQRT.*32$")>;
def : InstRW<[A64FXXWriteFDivSP], (instregex "^FDIVSrr")>;
def : InstRW<[A64FXXWriteFSqrtSP], (instregex "^FSQRTSr")>;

// FP divide, D-form
// FP square root, D-form
def : InstRW<[A64FXXWriteFDivDP], (instrs FDIVDrr)>;
def : InstRW<[A64FXXWriteFSqrtDP], (instrs FSQRTDr)>;
def : InstRW<[A64FXXWriteFDivDP], (instregex "^FDIVv.*64$")>;
def : InstRW<[A64FXXWriteFSqrtDP], (instregex "^.*SQRT.*64$")>;
def : InstRW<[A64FXXWriteFDivDP], (instregex "^FDIVDrr")>;
def : InstRW<[A64FXXWriteFSqrtDP], (instregex "^FSQRTDr")>;

// FP multiply
// FP multiply accumulate
def : WriteRes<WriteFMul, [A64FXGI03]> {
  let Latency = 9;
  let ResourceCycles = [2];
}

def A64FXXWriteFMul : SchedWriteRes<[A64FXGI03]> {
  let Latency = 9;
  let ResourceCycles = [2];
}

def A64FXXWriteFMulAcc : SchedWriteRes<[A64FXGI03]> {
  let Latency = 9;
  let ResourceCycles = [2];
}

def : InstRW<[A64FXXWriteFMul], (instregex "^FMUL", "^FNMUL")>;
def : InstRW<[A64FXXWriteFMulAcc],
            (instregex "^FMADD", "^FMSUB", "^FNMADD", "^FNMSUB")>;

// FP round to integral
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^FRINT(A|I|M|N|P|X|Z)(Sr|Dr)")>;

// FP select
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^FCSEL")>;

//---
// 3.9 FP Miscellaneous Instructions
//---

// FP convert, from vec to vec reg
// FP convert, from gen to vec reg
// FP convert, from vec to gen reg
def : WriteRes<WriteFCvt, [A64FXGI03]> {
  let Latency = 9;
  let ResourceCycles = [2];
}

// FP move, immed
// FP move, register
def : WriteRes<WriteFImm, [A64FXGI0]> {
  let Latency = 4;
  let ResourceCycles = [2];
}

// FP transfer, from gen to vec reg
// FP transfer, from vec to gen reg
def : WriteRes<WriteFCopy, [A64FXGI0]> {
  let Latency = 4;
  let ResourceCycles = [2];
}

def : InstRW<[A64FXWrite_FMOV_GV], (instrs FMOVXDHighr)>;
def : InstRW<[A64FXWrite_FMOV_VG14], (instrs FMOVDXHighr)>;

//---
// 3.12 ASIMD Integer Instructions
//---

// ASIMD absolute diff, D-form
// ASIMD absolute diff, Q-form
// ASIMD absolute diff accum, D-form
// ASIMD absolute diff accum, Q-form
// ASIMD absolute diff accum long
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD compare
// ASIMD logical (AND, BIC, EOR)
// ASIMD max/min, basic
// ASIMD max/min, reduce, 4H/4S
// ASIMD max/min, reduce, 8B/8H
// ASIMD max/min, reduce, 16B
// ASIMD multiply, D-form
// ASIMD multiply, Q-form
// ASIMD multiply accumulate long
// ASIMD multiply accumulate saturating long
// ASIMD multiply long
// ASIMD pairwise add and accumulate
// ASIMD shift accumulate
// ASIMD shift by immed, basic
// ASIMD shift by immed and insert, basic, D-form
// ASIMD shift by immed and insert, basic, Q-form
// ASIMD shift by immed, complex
// ASIMD shift by register, basic, D-form
// ASIMD shift by register, basic, Q-form
// ASIMD shift by register, complex, D-form
// ASIMD shift by register, complex, Q-form
def : WriteRes<WriteVd, [A64FXGI03]> {
  let Latency = 4;
  let ResourceCycles = [1];
}
def : WriteRes<WriteVq, [A64FXGI03]> {
  let Latency = 4;
  let ResourceCycles = [1];
}

// ASIMD arith, reduce, 4H/4S
// ASIMD arith, reduce, 8B/8H
// ASIMD arith, reduce, 16B

// ASIMD logical (MVN (alias for NOT), ORN, ORR)
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^ANDv", "^BICv", "^EORv", "^ORRv", "^ORNv", "^NOTv")>;

// ASIMD arith, reduce
def : InstRW<[A64FXWrite_ADDLV],
            (instregex "^ADDVv", "^SADDLVv", "^UADDLVv")>;

// ASIMD polynomial (8x8) multiply long
def : InstRW<[A64FXWrite_MULLE], (instregex "^(S|U|SQD)MULL")>;
def : InstRW<[A64FXWrite_MULLV],
            (instregex "(S|U|SQD)(MLAL|MLSL|MULL)v.*")>;
def : InstRW<[A64FXWrite_8Cyc_GI03], (instregex "^PMULL(v8i8|v16i8)")>;
def : InstRW<[A64FXWrite_8Cyc_GI03], (instregex "^PMULL(v1i64|v2i64)")>;

// ASIMD absolute diff accum, D-form
def : InstRW<[A64FXWrite_ABA],
            (instregex "^[SU]ABA(v8i8|v4i16|v2i32)$")>;
// ASIMD absolute diff accum, Q-form
def : InstRW<[A64FXWrite_ABA],
            (instregex "^[SU]ABA(v16i8|v8i16|v4i32)$")>;
// ASIMD absolute diff accum long
def : InstRW<[A64FXWrite_ABAL],
            (instregex "^[SU]ABAL")>;
// ASIMD arith, reduce, 4H/4S
def : InstRW<[A64FXWrite_ADDLV1],
            (instregex "^[SU]?ADDL?V(v8i8|v4i16|v2i32)v$")>;
// ASIMD arith, reduce, 8B
def : InstRW<[A64FXWrite_ADDLV1],
            (instregex "^[SU]?ADDL?V(v8i16|v4i32)v$")>;
// ASIMD arith, reduce, 16B/16H
def : InstRW<[A64FXWrite_ADDLV1],
            (instregex "^[SU]?ADDL?Vv16i8v$")>;
// ASIMD max/min, reduce, 4H/4S
def : InstRW<[A64FXWrite_MINMAXV],
            (instregex "^[SU](MIN|MAX)V(v4i16|v4i32)v$")>;
// ASIMD max/min, reduce, 8B/8H
def : InstRW<[A64FXWrite_MINMAXV],
            (instregex "^[SU](MIN|MAX)V(v8i8|v8i16)v$")>;
// ASIMD max/min, reduce, 16B/16H
def : InstRW<[A64FXWrite_MINMAXV],
            (instregex "^[SU](MIN|MAX)Vv16i8v$")>;
// ASIMD multiply, D-form
def : InstRW<[A64FXWrite_PMUL],
            (instregex "^(P?MUL|SQR?DMUL)" #
                       "(v8i8|v4i16|v2i32|v1i8|v1i16|v1i32|v1i64)" #
                       "(_indexed)?$")>;

// ASIMD multiply, Q-form
def : InstRW<[A64FXWrite_PMUL],
            (instregex "^(P?MUL)(v16i8|v8i16|v4i32)(_indexed)?$")>;

// ASIMD multiply, Q-form
def : InstRW<[A64FXWrite_SQRDMULH],
            (instregex "^(SQR?DMULH)(v16i8|v8i16|v4i32)(_indexed)?$")>;

// ASIMD multiply accumulate, D-form
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^ML[AS](v8i8|v4i16|v2i32)(_indexed)?$")>;
// ASIMD multiply accumulate, Q-form
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^ML[AS](v16i8|v8i16|v4i32)(_indexed)?$")>;
// ASIMD shift accumulate
def : InstRW<[A64FXWrite_SRSRAV],
            (instregex "SRSRAv", "URSRAv")>;
def : InstRW<[A64FXWrite_SSRAV],
            (instregex "SSRAv", "USRAv")>;

// ASIMD shift by immed, basic
def : InstRW<[A64FXWrite_RSHRN],
            (instregex "RSHRNv", "SQRSHRNv", "SQRSHRUNv", "UQRSHRNv")>;
def : InstRW<[A64FXWrite_SHRN],
            (instregex "SHRNv", "SQSHRNv", "SQSHRUNv", "UQSHRNv")>;

def : InstRW<[A64FXWrite_6Cyc_GI3],
            (instregex "SQXTNv", "SQXTUNv", "UQXTNv")>;

// ASIMD shift by immed, complex
def : InstRW<[A64FXWrite_ABA], (instregex "^[SU]?(Q|R){1,2}SHR")>;
def : InstRW<[A64FXWrite_6Cyc_GI3], (instregex "^SQSHLU")>;
// ASIMD shift by register, basic, Q-form
def : InstRW<[A64FXWrite_6Cyc_GI3],
            (instregex "^[SU]SHL(v16i8|v8i16|v4i32|v2i64)")>;
// ASIMD shift by register, complex, D-form
def : InstRW<[A64FXWrite_6Cyc_GI3],
            (instregex "^[SU][QR]{1,2}SHL" #
                       "(v1i8|v1i16|v1i32|v1i64|v8i8|v4i16|v2i32|b|d|h|s)")>;
// ASIMD shift by register, complex, Q-form
def : InstRW<[A64FXWrite_6Cyc_GI3],
            (instregex "^[SU][QR]{1,2}SHL(v16i8|v8i16|v4i32|v2i64)")>;

// ASIMD Arithmetic
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "(ADD|SUB)(v8i8|v4i16|v2i32|v1i64)")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "(ADD|SUB)(v16i8|v8i16|v4i32|v2i64)")>;
def : InstRW<[A64FXWrite_SHRN], (instregex "(ADD|SUB)HNv.*")>;
def : InstRW<[A64FXWrite_RSHRN], (instregex "(RADD|RSUB)HNv.*")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^SQADD", "^SQNEG", "^SQSUB", "^SRHADD",
                       "^SUQADD", "^UQADD", "^UQSUB", "^URHADD", "^USQADD")>;
def : InstRW<[A64FXWrite_ADDP],
            (instregex "ADDP(v16i8|v8i16|v4i32|v2i64)")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "((AND|ORN|EOR|EON)S?(Xr[rsi]|v16i8|v8i16|v4i32)|" #
                       "(ORR|BIC)S?(Xr[rs]|v16i8|v8i16|v4i32))")>;
def : InstRW<[A64FXWrite_4Cyc_GI0],
            (instregex "(CLS|CLZ|CNT)(v4i32|v8i16|v16i8)")>;
def : InstRW<[A64FXWrite_SADALP], (instregex "^SADALP", "^UADALP")>;
def : InstRW<[A64FXWrite_SADDLP], (instregex "^SADDLPv", "^UADDLPv")>;
def : InstRW<[A64FXWrite_ADDLV1], (instregex "^SADDLV", "^UADDLV")>;
def : InstRW<[A64FXWrite_MINMAXV],
             (instregex "^ADDVv", "^SMAXVv", "^UMAXVv", "^SMINVv", "^UMINVv")>;
def : InstRW<[A64FXWrite_ABA],
             (instregex "^SABAv", "^UABAv", "^SABALv", "^UABALv")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^SQADDv", "^SQSUBv", "^UQADDv", "^UQSUBv")>;
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^SUQADDv", "^USQADDv")>;
def : InstRW<[A64FXWrite_SHRN],
            (instregex "^ADDHNv", "^SUBHNv")>;
def : InstRW<[A64FXWrite_RSHRN],
            (instregex "^RADDHNv", "^RSUBHNv")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^SQABS", "^SQADD", "^SQNEG", "^SQSUB",
                       "^SRHADD", "^SUQADD", "^UQADD", "^UQSUB",
                      "^URHADD", "^USQADD")>;

def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^CMEQv", "^CMGEv", "^CMGTv",
                       "^CMLEv", "^CMLTv", "^CMHIv", "^CMHSv")>;
def : InstRW<[A64FXWrite_MINMAXV],
            (instregex "^SMAXv", "^SMINv", "^UMAXv", "^UMINv")>;
def : InstRW<[A64FXWrite_ADDP],
            (instregex "^SMAXPv", "^SMINPv", "^UMAXPv", "^UMINPv")>;
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^SABDv", "^UABDv")>;
def : InstRW<[A64FXWrite_TBX1],
            (instregex "^SABDLv", "^UABDLv")>;

//---
// 3.13 ASIMD Floating-point Instructions
//---

// ASIMD FP absolute value
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^FABSv")>;

// ASIMD FP arith, normal, D-form
// ASIMD FP arith, normal, Q-form
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^FABDv", "^FADDv", "^FSUBv")>;

// ASIMD FP arith, pairwise, D-form
// ASIMD FP arith, pairwise, Q-form
def : InstRW<[A64FXWrite_FADDPV], (instregex "^FADDPv")>;

// ASIMD FP compare, D-form
// ASIMD FP compare, Q-form
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^FACGEv", "^FACGTv")>;
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^FCMEQv", "^FCMGEv",
                                                 "^FCMGTv", "^FCMLEv",
                                                 "^FCMLTv")>;
// ASIMD FP round, D-form
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^FRINT[AIMNPXZ](v2f32)")>;
// ASIMD FP round, Q-form
def : InstRW<[A64FXWrite_9Cyc_GI03],
            (instregex "^FRINT[AIMNPXZ](v4f32|v2f64)")>;

// ASIMD FP convert, long
// ASIMD FP convert, narrow
// ASIMD FP convert, other, D-form
// ASIMD FP convert, other, Q-form

// ASIMD FP convert, long and narrow
def : InstRW<[A64FXWrite_FCVTXNV], (instregex "^FCVT(L|N|XN)v")>;
// ASIMD FP convert, other, D-form
def : InstRW<[A64FXWrite_FCVTXNV],
      (instregex "^[FVSU]CVT([AMNPZ][SU])?(_Int)?(v2f32|v1i32|v2i32|v1i64)")>;
// ASIMD FP convert, other, Q-form
def : InstRW<[A64FXWrite_FCVTXNV],
      (instregex "^[FVSU]CVT([AMNPZ][SU])?(_Int)?(v4f32|v2f64|v4i32|v2i64)")>;

// ASIMD FP divide, D-form, F32
def : InstRW<[A64FXXWriteFDivSP], (instrs FDIVv2f32)>;
def : InstRW<[A64FXXWriteFDivSP], (instregex "FDIVv2f32")>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[A64FXXWriteFDiv], (instrs FDIVv4f32)>;
def : InstRW<[A64FXXWriteFDiv], (instregex "FDIVv4f32")>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[A64FXXWriteFDivDP], (instrs FDIVv2f64)>;
def : InstRW<[A64FXXWriteFDivDP], (instregex "FDIVv2f64")>;

// ASIMD FP max/min, normal, D-form
// ASIMD FP max/min, normal, Q-form
def : InstRW<[A64FXWrite_4Cyc_GI0], (instregex "^FMAXv", "^FMAXNMv",
                                               "^FMINv", "^FMINNMv")>;

// ASIMD FP max/min, pairwise, D-form
// ASIMD FP max/min, pairwise, Q-form
def : InstRW<[A64FXWrite_ADDP], (instregex "^FMAXPv", "^FMAXNMPv",
                                           "^FMINPv", "^FMINNMPv")>;

// ASIMD FP max/min, reduce
def : InstRW<[A64FXWrite_FMAXVVH], (instregex "^FMAXVv", "^FMAXNMVv",
                                              "^FMINVv", "^FMINNMVv")>;

// ASIMD FP multiply, D-form, FZ
// ASIMD FP multiply, D-form, no FZ
// ASIMD FP multiply, Q-form, FZ
// ASIMD FP multiply, Q-form, no FZ
def : InstRW<[A64FXWrite_9Cyc_GI03], (instregex "^FMULv", "^FMULXv")>;
def : InstRW<[A64FXWrite_FMULXE],
            (instregex "^FMULX?(v2f32|v1i32|v2i32|v1i64|32|64)")>;
def : InstRW<[A64FXWrite_FMULXE],
            (instregex "^FMULX?(v4f32|v2f64|v4i32|v2i64)")>;

// ASIMD FP multiply accumulate, Dform, FZ
// ASIMD FP multiply accumulate, Dform, no FZ
// ASIMD FP multiply accumulate, Qform, FZ
// ASIMD FP multiply accumulate, Qform, no FZ
def : InstRW<[A64FXWrite_9Cyc_GI03], (instregex "^FMLAv", "^FMLSv")>;
def : InstRW<[A64FXWrite_FMULXE],
            (instregex "^FML[AS](v2f32|v1i32|v2i32|v1i64)")>;
def : InstRW<[A64FXWrite_FMULXE],
            (instregex "^FML[AS](v4f32|v2f64|v4i32|v2i64)")>;

// ASIMD FP negate
def : InstRW<[A64FXWrite_4Cyc_GI03], (instregex "^FNEGv")>;

//--
// 3.14 ASIMD Miscellaneous Instructions
//--

// ASIMD bit reverse
def : InstRW<[A64FXWrite_1Cyc_GI2456], (instregex "^RBITv")>;

// ASIMD bitwise insert, D-form
// ASIMD bitwise insert, Q-form
def : InstRW<[A64FXWrite_BIF],
            (instregex "^BIFv", "^BITv", "^BSLv")>;

// ASIMD count, D-form
// ASIMD count, Q-form
def : InstRW<[A64FXWrite_4Cyc_GI0],
            (instregex "^CLSv", "^CLZv", "^CNTv")>;

// ASIMD duplicate, gen reg
// ASIMD duplicate, element
def : InstRW<[A64FXWrite_DUPGENERAL], (instregex "^DUPv")>;
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^DUP(i8|i16|i32|i64)$")>;
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^DUPv.+gpr")>;

// ASIMD extract
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^EXTv")>;

// ASIMD extract narrow
def : InstRW<[A64FXWrite_6Cyc_GI3], (instregex "^XTNv")>;

// ASIMD extract narrow, saturating
def : InstRW<[A64FXWrite_6Cyc_GI3],
            (instregex "^SQXTNv", "^SQXTUNv", "^UQXTNv")>;

// ASIMD insert, element to element
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^INSv")>;

// ASIMD transfer, element to gen reg
def : InstRW<[A64FXWrite_SMOV], (instregex "^[SU]MOVv")>;

// ASIMD move, integer immed
def : InstRW<[A64FXWrite_4Cyc_GI0], (instregex "^MOVIv")>;

// ASIMD move, FP immed
def : InstRW<[A64FXWrite_4Cyc_GI0], (instregex "^FMOVv")>;

// ASIMD table lookup, D-form
def : InstRW<[A64FXWrite_6Cyc_GI3], (instregex "^TBLv8i8One")>;
def : InstRW<[A64FXWrite_TBX1], (instregex "^TBLv8i8Two")>;
def : InstRW<[A64FXWrite_TBX2], (instregex "^TBLv8i8Three")>;
def : InstRW<[A64FXWrite_TBX3], (instregex "^TBLv8i8Four")>;
def : InstRW<[A64FXWrite_TBX1], (instregex "^TBXv8i8One")>;
def : InstRW<[A64FXWrite_TBX2], (instregex "^TBXv8i8Two")>;
def : InstRW<[A64FXWrite_TBX3], (instregex "^TBXv8i8Three")>;
def : InstRW<[A64FXWrite_TBX4], (instregex "^TBXv8i8Four")>;

// ASIMD table lookup, Q-form
def : InstRW<[A64FXWrite_6Cyc_GI3], (instregex "^TBLv16i8One")>;
def : InstRW<[A64FXWrite_TBX1], (instregex "^TBLv16i8Two")>;
def : InstRW<[A64FXWrite_TBX2], (instregex "^TBLv16i8Three")>;
def : InstRW<[A64FXWrite_TBX3], (instregex "^TBLv16i8Four")>;
def : InstRW<[A64FXWrite_TBX1], (instregex "^TBXv16i8One")>;
def : InstRW<[A64FXWrite_TBX2], (instregex "^TBXv16i8Two")>;
def : InstRW<[A64FXWrite_TBX3], (instregex "^TBXv16i8Three")>;
def : InstRW<[A64FXWrite_TBX4], (instregex "^TBXv16i8Four")>;

// ASIMD transpose
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^TRN1", "^TRN2")>;

// ASIMD unzip/zip
def : InstRW<[A64FXWrite_6Cyc_GI0],
            (instregex "^UZP1", "^UZP2", "^ZIP1", "^ZIP2")>;

// ASIMD reciprocal estimate, D-form
// ASIMD reciprocal estimate, Q-form
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^FRECPEv", "^FRECPXv", "^URECPEv",
                       "^FRSQRTEv", "^URSQRTEv")>;

// ASIMD reciprocal step, D-form, FZ
// ASIMD reciprocal step, D-form, no FZ
// ASIMD reciprocal step, Q-form, FZ
// ASIMD reciprocal step, Q-form, no FZ
def : InstRW<[A64FXWrite_9Cyc_GI0], (instregex "^FRECPSv", "^FRSQRTSv")>;

// ASIMD reverse
def : InstRW<[A64FXWrite_4Cyc_GI03],
            (instregex "^REV16v", "^REV32v", "^REV64v")>;

// ASIMD table lookup, D-form
// ASIMD table lookup, Q-form
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^TBLv", "^TBXv")>;

// ASIMD transfer, element to word or word
def : InstRW<[A64FXWrite_SMOV], (instregex "^[SU]MOVv")>;

// ASIMD transfer, element to gen reg
def : InstRW<[A64FXWrite_SMOV], (instregex "(S|U)MOVv.*")>;

// ASIMD transfer gen reg to element
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^INSv")>;

// ASIMD transpose
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^TRN1v", "^TRN2v",
                                                 "^UZP1v", "^UZP2v")>;

// ASIMD unzip/zip
def : InstRW<[A64FXWrite_6Cyc_GI0], (instregex "^ZIP1v", "^ZIP2v")>;

//--
// 3.15 ASIMD Load Instructions
//--

// ASIMD load, 1 element, multiple, 1 reg, D-form
// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[A64FXWrite_8Cyc_GI56],
            (instregex "^LD1Onev(8b|4h|2s|1d|2d)$")>;
def : InstRW<[A64FXWrite_11Cyc_GI56],
            (instregex "^LD1Onev(16b|8h|4s)$")>;
def : InstRW<[A64FXWrite_LD108, WriteAdr],
            (instregex "^LD1Onev(8b|4h|2s|1d|2d)_POST$")>;
def : InstRW<[A64FXWrite_LD109, WriteAdr],
            (instregex "^LD1Onev(16b|8h|4s)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[A64FXWrite_LD102],
            (instregex "^LD1Twov(8b|4h|2s|1d|2d)$")>;
def : InstRW<[A64FXWrite_LD103],
            (instregex "^LD1Twov(16b|8h|4s)$")>;
def : InstRW<[A64FXWrite_LD110, WriteAdr],
            (instregex "^LD1Twov(8b|4h|2s|1d|2d)_POST$")>;
def : InstRW<[A64FXWrite_LD111, WriteAdr],
            (instregex "^LD1Twov(16b|8h|4s)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[A64FXWrite_LD104],
            (instregex "^LD1Threev(8b|4h|2s|1d|2d)$")>;
def : InstRW<[A64FXWrite_LD105],
            (instregex "^LD1Threev(16b|8h|4s)$")>;
def : InstRW<[A64FXWrite_LD112, WriteAdr],
            (instregex "^LD1Threev(8b|4h|2s|1d|2d)_POST$")>;
def : InstRW<[A64FXWrite_LD113, WriteAdr],
            (instregex "^LD1Threev(16b|8h|4s)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[A64FXWrite_LD106],
            (instregex "^LD1Fourv(8b|4h|2s|1d|2d)$")>;
def : InstRW<[A64FXWrite_LD107],
            (instregex "^LD1Fourv(16b|8h|4s)$")>;
def : InstRW<[A64FXWrite_LD114, WriteAdr],
            (instregex "^LD1Fourv(8b|4h|2s|1d|2d)_POST$")>;
def : InstRW<[A64FXWrite_LD115, WriteAdr],
            (instregex "^LD1Fourv(16b|8h|4s)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[A64FXWrite_LD1I0], (instregex "^LD1i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_LD1I1, WriteAdr],
            (instregex "^LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[A64FXWrite_8Cyc_GI03],
            (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD108, WriteAdr],
            (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_LD103],
            (instregex "^LD2Twov(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD111, WriteAdr],
            (instregex "^LD2Twov(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[A64FXWrite_LD2I0], (instregex "^LD2i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_LD2I1, WriteAdr],
            (instregex "^LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[A64FXWrite_LD102],
            (instregex "^LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD110, WriteAdr],
            (instregex "^LD2Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, B/H/S
// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_LD105],
            (instregex "^LD3Threev(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD113, WriteAdr],
            (instregex "^LD3Threev(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, one lone, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[A64FXWrite_LD3I0], (instregex "^LD3i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_LD3I1, WriteAdr],
            (instregex "^LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[A64FXWrite_LD104],
            (instregex "^LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD112, WriteAdr],
            (instregex "^LD3Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_LD107],
            (instregex "^LD4Fourv(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD115, WriteAdr],
            (instregex "^LD4Fourv(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[A64FXWrite_LD4I0], (instregex "^LD4i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_LD4I1, WriteAdr],
            (instregex "^LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[A64FXWrite_LD106],
            (instregex "^LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_LD114, WriteAdr],
            (instregex "^LD4Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

//--
// 3.16 ASIMD Store Instructions
//--

// ASIMD store, 1 element, multiple, 1 reg, D-form
// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[A64FXWrite_ST10],
            (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST14, WriteAdr],
            (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[A64FXWrite_ST11],
            (instregex "^ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST15, WriteAdr],
            (instregex "^ST1Twov(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[A64FXWrite_ST12],
            (instregex "^ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST16, WriteAdr],
            (instregex "^ST1Threev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[A64FXWrite_ST13],
            (instregex "^ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST17, WriteAdr],
            (instregex "^ST1Fourv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[A64FXWrite_ST10],
            (instregex "^ST1i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_ST14, WriteAdr],
            (instregex "^ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_ST11],
            (instregex "^ST2Twov(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST15, WriteAdr],
            (instregex "^ST2Twov(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[A64FXWrite_ST11],
            (instregex "^ST2i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_ST15, WriteAdr],
            (instregex "^ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_ST12],
            (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST16, WriteAdr],
            (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[A64FXWrite_ST12], (instregex "^ST3i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_ST16, WriteAdr],
            (instregex "^ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
// ASIMD store, 4 element, multiple, Q-form, B/H/S
// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[A64FXWrite_ST13],
            (instregex "^ST4Fourv(8b|4h|2s|16b|8h|4s|2d)$")>;
def : InstRW<[A64FXWrite_ST17, WriteAdr],
            (instregex "^ST4Fourv(8b|4h|2s|16b|8h|4s|2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H
// ASIMD store, 4 element, one lane, S
// ASIMD store, 4 element, one lane, D
def : InstRW<[A64FXWrite_ST13], (instregex "^ST4i(8|16|32|64)$")>;
def : InstRW<[A64FXWrite_ST17, WriteAdr],
            (instregex "^ST4i(8|16|32|64)_POST$")>;

// V8.1a Atomics (LSE)
def : InstRW<[A64FXWrite_CAS, WriteAtomic],
            (instrs CASB, CASH, CASW, CASX)>;

def : InstRW<[A64FXWrite_CAS, WriteAtomic],
            (instrs CASAB, CASAH, CASAW, CASAX)>;

def : InstRW<[A64FXWrite_CAS, WriteAtomic],
            (instrs CASLB, CASLH, CASLW, CASLX)>;

def : InstRW<[A64FXWrite_CAS, WriteAtomic],
            (instrs CASALB, CASALH, CASALW, CASALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDLARB, LDLARH, LDLARW, LDLARX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDADDB, LDADDH, LDADDW, LDADDX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDADDAB, LDADDAH, LDADDAW, LDADDAX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDADDLB, LDADDLH, LDADDLW, LDADDLX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDADDALB, LDADDALH, LDADDALW, LDADDALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDCLRB, LDCLRH, LDCLRW, LDCLRX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDCLRAB, LDCLRAH, LDCLRAW, LDCLRAX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDCLRLB, LDCLRLH, LDCLRLW, LDCLRLX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDCLRALB, LDCLRALH, LDCLRALW, LDCLRALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDEORB, LDEORH, LDEORW, LDEORX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDEORAB, LDEORAH, LDEORAW, LDEORAX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDEORLB, LDEORLH, LDEORLW, LDEORLX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDEORALB, LDEORALH, LDEORALW, LDEORALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSETB, LDSETH, LDSETW, LDSETX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSETAB, LDSETAH, LDSETAW, LDSETAX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSETLB, LDSETLH, LDSETLW, LDSETLX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSETALB, LDSETALH, LDSETALW, LDSETALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSMAXB, LDSMAXH, LDSMAXW, LDSMAXX,
             LDSMAXAB, LDSMAXAH, LDSMAXAW, LDSMAXAX,
             LDSMAXLB, LDSMAXLH, LDSMAXLW, LDSMAXLX,
             LDSMAXALB, LDSMAXALH, LDSMAXALW, LDSMAXALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDSMINB, LDSMINH, LDSMINW, LDSMINX,
             LDSMINAB, LDSMINAH, LDSMINAW, LDSMINAX,
             LDSMINLB, LDSMINLH, LDSMINLW, LDSMINLX,
             LDSMINALB, LDSMINALH, LDSMINALW, LDSMINALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDUMAXB, LDUMAXH, LDUMAXW, LDUMAXX,
             LDUMAXAB, LDUMAXAH, LDUMAXAW, LDUMAXAX,
             LDUMAXLB, LDUMAXLH, LDUMAXLW, LDUMAXLX,
             LDUMAXALB, LDUMAXALH, LDUMAXALW, LDUMAXALX)>;

def : InstRW<[A64FXWrite_5Cyc_GI5, WriteAtomic],
            (instrs LDUMINB, LDUMINH, LDUMINW, LDUMINX,
             LDUMINAB, LDUMINAH, LDUMINAW, LDUMINAX,
             LDUMINLB, LDUMINLH, LDUMINLW, LDUMINLX,
             LDUMINALB, LDUMINALH, LDUMINALW, LDUMINALX)>;

def : InstRW<[A64FXWrite_SWP, WriteAtomic],
            (instrs SWPB, SWPH, SWPW, SWPX)>;

def : InstRW<[A64FXWrite_SWP, WriteAtomic],
            (instrs SWPAB, SWPAH, SWPAW, SWPAX)>;

def : InstRW<[A64FXWrite_SWP, WriteAtomic],
            (instrs SWPLB, SWPLH, SWPLW, SWPLX)>;

def : InstRW<[A64FXWrite_SWP, WriteAtomic],
            (instrs SWPALB, SWPALH, SWPALW, SWPALX)>;

def : InstRW<[A64FXWrite_STUR, WriteAtomic],
            (instrs STLLRB, STLLRH, STLLRW, STLLRX)>;

// [ 1]   "abs  $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ABS_ZPmZ_B, ABS_ZPmZ_D, ABS_ZPmZ_H, ABS_ZPmZ_S)>;

// [ 2]   "add  $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ADD_ZZZ_B, ADD_ZZZ_D, ADD_ZZZ_H, ADD_ZZZ_S)>;

// [ 3]   "add  $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ADD_ZPmZ_B, ADD_ZPmZ_D, ADD_ZPmZ_H, ADD_ZPmZ_S)>;

// [ 4]   "add  $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ADD_ZI_B, ADD_ZI_D, ADD_ZI_H, ADD_ZI_S)>;

// [ 5]   "addpl        $Rd, $Rn, $imm6";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs ADDPL_XXI)>;

// [ 6]   "addvl        $Rd, $Rn, $imm6";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs ADDVL_XXI)>;

// [ 7]   "adr  $Zd, [$Zn, $Zm]";
def : InstRW<[A64FXWrite_5Cyc_GI0], (instrs ADR_LSL_ZZZ_D_0, ADR_LSL_ZZZ_D_1, ADR_LSL_ZZZ_D_2, ADR_LSL_ZZZ_D_3, ADR_LSL_ZZZ_S_0, ADR_LSL_ZZZ_S_1, ADR_LSL_ZZZ_S_2, ADR_LSL_ZZZ_S_3, ADR_SXTW_ZZZ_D_0, ADR_SXTW_ZZZ_D_1, ADR_SXTW_ZZZ_D_2, ADR_SXTW_ZZZ_D_3, ADR_UXTW_ZZZ_D_0, ADR_UXTW_ZZZ_D_1, ADR_UXTW_ZZZ_D_2, ADR_UXTW_ZZZ_D_3)>;

// [ 8]   "and  $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs AND_PPzPP)>;

// [ 9]   "and  $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs AND_ZZZ)>;

// [10]   "and  $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs AND_ZPmZ_B, AND_ZPmZ_D, AND_ZPmZ_H, AND_ZPmZ_S)>;

// [11]   "and  $Zdn, $_Zdn, $imms13";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs AND_ZI)>;

// [12]   "ands $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs ANDS_PPzPP)>;

// [13]   "andv $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs ANDV_VPZ_B, ANDV_VPZ_D, ANDV_VPZ_H, ANDV_VPZ_S)>;

// [14]   "asr  $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASR_WIDE_ZZZ_B, ASR_WIDE_ZZZ_H, ASR_WIDE_ZZZ_S)>;

// [15]   "asr  $Zd, $Zn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASR_ZZI_B, ASR_ZZI_D, ASR_ZZI_H, ASR_ZZI_S)>;

// [16]   "asr  $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASR_WIDE_ZPmZ_B, ASR_WIDE_ZPmZ_H, ASR_WIDE_ZPmZ_S, ASR_ZPmZ_B, ASR_ZPmZ_D, ASR_ZPmZ_H, ASR_ZPmZ_S)>;

// [17]   "asr  $Zdn, $Pg/m, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASR_ZPmI_B, ASR_ZPmI_D, ASR_ZPmI_H, ASR_ZPmI_S)>;

// [18]   "asrd $Zdn, $Pg/m, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASRD_ZPmI_B, ASRD_ZPmI_D, ASRD_ZPmI_H, ASRD_ZPmI_S)>;

// [19]   "asrr $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ASRR_ZPmZ_B, ASRR_ZPmZ_D, ASRR_ZPmZ_H, ASRR_ZPmZ_S)>;

// [20]   "bic  $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BIC_PPzPP)>;

// [21]   "bic  $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs BIC_ZZZ)>;

// [22]   "bic  $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs BIC_ZPmZ_B, BIC_ZPmZ_D, BIC_ZPmZ_H, BIC_ZPmZ_S)>;

// [23]   "bics $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BICS_PPzPP)>;

// [24]   "brka $Pd, $Pg/m, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKA_PPmP)>;

// [25]   "brka $Pd, $Pg/z, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKA_PPzP)>;

// [26]   "brkas        $Pd, $Pg/z, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKAS_PPzP)>;

// [27]   "brkb $Pd, $Pg/m, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKB_PPmP)>;

// [28]   "brkb $Pd, $Pg/z, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKB_PPzP)>;

// [29]   "brkbs        $Pd, $Pg/z, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKBS_PPzP)>;

// [30]   "brkn $Pdm, $Pg/z, $Pn, $_Pdm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKN_PPzP)>;

// [31]   "brkns        $Pdm, $Pg/z, $Pn, $_Pdm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKNS_PPzP)>;

// [32]   "brkpa        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKPA_PPzPP)>;

// [33]   "brkpas       $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKPAS_PPzPP)>;

// [34]   "brkpb        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKPB_PPzPP)>;

// [35]   "brkpbs       $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs BRKPBS_PPzPP)>;

// [36]   "clasta       $Rdn, $Pg, $_Rdn, $Zm";
def : InstRW<[A64FXWrite_29Cyc_GI0256], (instrs CLASTA_RPZ_B, CLASTA_RPZ_D, CLASTA_RPZ_H, CLASTA_RPZ_S)>;

// [37]   "clasta       $Vdn, $Pg, $_Vdn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CLASTA_VPZ_B, CLASTA_VPZ_D, CLASTA_VPZ_H, CLASTA_VPZ_S)>;

// [38]   "clasta       $Zdn, $Pg, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CLASTA_ZPZ_B, CLASTA_ZPZ_D, CLASTA_ZPZ_H, CLASTA_ZPZ_S)>;

// [39]   "clastb       $Rdn, $Pg, $_Rdn, $Zm";
def : InstRW<[A64FXWrite_29Cyc_GI0256], (instrs CLASTB_RPZ_B, CLASTB_RPZ_D, CLASTB_RPZ_H, CLASTB_RPZ_S)>;

// [40]   "clastb       $Vdn, $Pg, $_Vdn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CLASTB_VPZ_B, CLASTB_VPZ_D, CLASTB_VPZ_H, CLASTB_VPZ_S)>;

// [41]   "clastb       $Zdn, $Pg, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CLASTB_ZPZ_B, CLASTB_ZPZ_D, CLASTB_ZPZ_H, CLASTB_ZPZ_S)>;

// [42]   "cls  $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs CLS_ZPmZ_B, CLS_ZPmZ_D, CLS_ZPmZ_H, CLS_ZPmZ_S)>;

// [43]   "clz  $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs CLZ_ZPmZ_B, CLZ_ZPmZ_D, CLZ_ZPmZ_H, CLZ_ZPmZ_S)>;

// [44]   "cmpeq        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPEQ_PPzZZ_B, CMPEQ_PPzZZ_D, CMPEQ_PPzZZ_H, CMPEQ_PPzZZ_S, CMPEQ_WIDE_PPzZZ_B, CMPEQ_WIDE_PPzZZ_H, CMPEQ_WIDE_PPzZZ_S)>;

// [45]   "cmpeq        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPEQ_PPzZI_B, CMPEQ_PPzZI_D, CMPEQ_PPzZI_H, CMPEQ_PPzZI_S)>;

// [46]   "cmpge        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPGE_PPzZZ_B, CMPGE_PPzZZ_D, CMPGE_PPzZZ_H, CMPGE_PPzZZ_S, CMPGE_WIDE_PPzZZ_B, CMPGE_WIDE_PPzZZ_H, CMPGE_WIDE_PPzZZ_S)>;

// [47]   "cmpge        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPGE_PPzZI_B, CMPGE_PPzZI_D, CMPGE_PPzZI_H, CMPGE_PPzZI_S)>;

// [48]   "cmpgt        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPGT_PPzZZ_B, CMPGT_PPzZZ_D, CMPGT_PPzZZ_H, CMPGT_PPzZZ_S, CMPGT_WIDE_PPzZZ_B, CMPGT_WIDE_PPzZZ_H, CMPGT_WIDE_PPzZZ_S)>;

// [49]   "cmpgt        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPGT_PPzZI_B, CMPGT_PPzZI_D, CMPGT_PPzZI_H, CMPGT_PPzZI_S)>;

// [50]   "cmphi        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPHI_PPzZZ_B, CMPHI_PPzZZ_D, CMPHI_PPzZZ_H, CMPHI_PPzZZ_S, CMPHI_WIDE_PPzZZ_B, CMPHI_WIDE_PPzZZ_H, CMPHI_WIDE_PPzZZ_S)>;

// [51]   "cmphi        $Pd, $Pg/z, $Zn, $imm7";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPHI_PPzZI_B, CMPHI_PPzZI_D, CMPHI_PPzZI_H, CMPHI_PPzZI_S)>;

// [52]   "cmphs        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPHS_PPzZZ_B, CMPHS_PPzZZ_D, CMPHS_PPzZZ_H, CMPHS_PPzZZ_S, CMPHS_WIDE_PPzZZ_B, CMPHS_WIDE_PPzZZ_H, CMPHS_WIDE_PPzZZ_S)>;

// [53]   "cmphs        $Pd, $Pg/z, $Zn, $imm7";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPHS_PPzZI_B, CMPHS_PPzZI_D, CMPHS_PPzZI_H, CMPHS_PPzZI_S)>;

// [54]   "cmple        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLE_WIDE_PPzZZ_B, CMPLE_WIDE_PPzZZ_H, CMPLE_WIDE_PPzZZ_S)>;

// [55]   "cmple        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLE_PPzZI_B, CMPLE_PPzZI_D, CMPLE_PPzZI_H, CMPLE_PPzZI_S)>;

// [56]   "cmplo        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLO_WIDE_PPzZZ_B, CMPLO_WIDE_PPzZZ_H, CMPLO_WIDE_PPzZZ_S)>;

// [57]   "cmplo        $Pd, $Pg/z, $Zn, $imm7";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLO_PPzZI_B, CMPLO_PPzZI_D, CMPLO_PPzZI_H, CMPLO_PPzZI_S)>;

// [58]   "cmpls        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLS_WIDE_PPzZZ_B, CMPLS_WIDE_PPzZZ_H, CMPLS_WIDE_PPzZZ_S)>;

// [59]   "cmpls        $Pd, $Pg/z, $Zn, $imm7";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLS_PPzZI_B, CMPLS_PPzZI_D, CMPLS_PPzZI_H, CMPLS_PPzZI_S)>;

// [60]   "cmplt        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLT_WIDE_PPzZZ_B, CMPLT_WIDE_PPzZZ_H, CMPLT_WIDE_PPzZZ_S)>;

// [61]   "cmplt        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPLT_PPzZI_B, CMPLT_PPzZI_D, CMPLT_PPzZI_H, CMPLT_PPzZI_S)>;

// [62]   "cmpne        $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPNE_PPzZZ_B, CMPNE_PPzZZ_D, CMPNE_PPzZZ_H, CMPNE_PPzZZ_S, CMPNE_WIDE_PPzZZ_B, CMPNE_WIDE_PPzZZ_H, CMPNE_WIDE_PPzZZ_S)>;

// [63]   "cmpne        $Pd, $Pg/z, $Zn, $imm5";
def : InstRW<[A64FXWrite_4Cyc_GI01], (instrs CMPNE_PPzZI_B, CMPNE_PPzZI_D, CMPNE_PPzZI_H, CMPNE_PPzZI_S)>;

// [64]   "cnot $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs CNOT_ZPmZ_B, CNOT_ZPmZ_D, CNOT_ZPmZ_H, CNOT_ZPmZ_S)>;

// [65]   "cnt  $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI3], (instrs CNT_ZPmZ_B, CNT_ZPmZ_D, CNT_ZPmZ_H, CNT_ZPmZ_S)>;

// [66]   "cntb $Rd, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs CNTB_XPiI)>;

// [67]   "cntd $Rd, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs CNTD_XPiI)>;

// [68]   "cnth $Rd, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs CNTH_XPiI)>;

// [69]   "cntp $Rd, $Pg, $Pn";
def : InstRW<[A64FXWrite_6Cyc_GI01], (instrs CNTP_XPP_B, CNTP_XPP_D, CNTP_XPP_H, CNTP_XPP_S)>;

// [70]   "cntw $Rd, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs CNTW_XPiI)>;

// [71]   "compact      $Zd, $Pg, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs COMPACT_ZPZ_D, COMPACT_ZPZ_S)>;

// [72]   "cpy  $Zd, $Pg/m, $Rn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CPY_ZPmR_B, CPY_ZPmR_D, CPY_ZPmR_H, CPY_ZPmR_S)>;

// [73]   "cpy  $Zd, $Pg/m, $Vn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CPY_ZPmV_B, CPY_ZPmV_D, CPY_ZPmV_H, CPY_ZPmV_S)>;

// [74]   "cpy  $Zd, $Pg/m, $imm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CPY_ZPmI_B, CPY_ZPmI_D, CPY_ZPmI_H, CPY_ZPmI_S)>;

// [75]   "cpy  $Zd, $Pg/z, $imm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs CPY_ZPzI_B, CPY_ZPzI_D, CPY_ZPzI_H, CPY_ZPzI_S)>;

// [76]   "ctermeq      $Rn, $Rm";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs CTERMEQ_WW, CTERMEQ_XX)>;

// [77]   "ctermne      $Rn, $Rm";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs CTERMNE_WW, CTERMNE_XX)>;

// [78]   "decb $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs DECB_XPiI)>;

// [79]   "decd $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs DECD_XPiI)>;

// [80]   "decd $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs DECD_ZPiI)>;

// [81]   "dech $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs DECH_XPiI)>;

// [82]   "dech $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs DECH_ZPiI)>;

// [83]   "decp $Rdn, $Pg";
def : InstRW<[A64FXWrite_6Cyc_GI124], (instrs DECP_XP_B, DECP_XP_D, DECP_XP_H, DECP_XP_S)>;

// [84]   "decp $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs DECP_ZP_D, DECP_ZP_H, DECP_ZP_S)>;

// [85]   "decw $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs DECW_XPiI)>;

// [86]   "decw $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs DECW_ZPiI)>;

// [87]   "dup  $Zd, $Rn";
def : InstRW<[A64FXWrite_8Cyc_GI01], (instrs DUP_ZR_B, DUP_ZR_D, DUP_ZR_H, DUP_ZR_S)>;

// [88]   "dup  $Zd, $Zn$idx";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs DUP_ZZI_B, DUP_ZZI_D, DUP_ZZI_H, DUP_ZZI_Q, DUP_ZZI_S)>;

// [89]   "dup  $Zd, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs DUP_ZI_B, DUP_ZI_D, DUP_ZI_H, DUP_ZI_S)>;

// [90]   "dupm $Zd, $imms";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs DUPM_ZI)>;

// [91]   "eor  $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs EOR_PPzPP)>;

// [92]   "eor  $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs EOR_ZZZ)>;

// [93]   "eor  $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs EOR_ZPmZ_B, EOR_ZPmZ_D, EOR_ZPmZ_H, EOR_ZPmZ_S)>;

// [94]   "eor  $Zdn, $_Zdn, $imms13";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs EOR_ZI)>;

// [95]   "eors $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs EORS_PPzPP)>;

// [96]   "eorv $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs EORV_VPZ_B, EORV_VPZ_D, EORV_VPZ_H, EORV_VPZ_S)>;

// [97]   "ext  $Zdn, $_Zdn, $Zm, $imm8";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs EXT_ZZI)>;

// [99]   "fabd $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FABD_ZPmZ_D, FABD_ZPmZ_H, FABD_ZPmZ_S)>;

// [100]   "fabs        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FABS_ZPmZ_D, FABS_ZPmZ_H, FABS_ZPmZ_S)>;

// [101]   "facge       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FACGE_PPzZZ_D, FACGE_PPzZZ_H, FACGE_PPzZZ_S)>;

// [102]   "facgt       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FACGT_PPzZZ_D, FACGT_PPzZZ_H, FACGT_PPzZZ_S)>;

// [103]   "fadd        $Zd, $Zn, $Zm"; def is line 1638
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FADD_ZZZ_D, FADD_ZZZ_H, FADD_ZZZ_S)>;

// [104]   "fadd        $Zdn, $Pg/m, $_Zdn, $Zm"; def is line 1638
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FADD_ZPmZ_D, FADD_ZPmZ_H, FADD_ZPmZ_S)>;

// [105]   "fadd        $Zdn, $Pg/m, $_Zdn, $i1"; def is line 1638
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FADD_ZPmI_D, FADD_ZPmI_H, FADD_ZPmI_S)>;

// [106]   "fadda       $Vdn, $Pg, $_Vdn, $Zm";
def : InstRW<[A64FXWrite_18Cyc_GI03], (instrs FADDA_VPZ_D, FADDA_VPZ_H, FADDA_VPZ_S)>;

// [107]   "faddv       $Vd, $Pg, $Zn";
// H : 4 / 6 / ([1,2]9 / [1]6) x 4 / [1,2]9 = 75 cycle
// S : 4 / 6 / ([1,2]9 / [1]6) x 3 / [1,2]9 = 60 cycle
// D : 4 / 6 / ([1,2]9 / [1]6) x 2 / [1,2]9 = 45 cycle
def : InstRW<[A64FXWrite_75Cyc_GI03], (instrs FADDV_VPZ_H)>;
def : InstRW<[A64FXWrite_60Cyc_GI03], (instrs FADDV_VPZ_S)>;
def : InstRW<[A64FXWrite_45Cyc_GI03], (instrs FADDV_VPZ_D)>;

// [108]   "fcadd       $Zdn, $Pg/m, $_Zdn, $Zm, $imm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FCADD_ZPmZ_D, FCADD_ZPmZ_H, FCADD_ZPmZ_S)>;

// [109]   "fcmeq       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMEQ_PPzZ0_D, FCMEQ_PPzZ0_H, FCMEQ_PPzZ0_S)>;

// [110]   "fcmeq       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMEQ_PPzZZ_D, FCMEQ_PPzZZ_H, FCMEQ_PPzZZ_S)>;

// [111]   "fcmge       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMGE_PPzZ0_D, FCMGE_PPzZ0_H, FCMGE_PPzZ0_S)>;

// [112]   "fcmge       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMGE_PPzZZ_D, FCMGE_PPzZZ_H, FCMGE_PPzZZ_S)>;

// [113]   "fcmgt       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMGT_PPzZ0_D, FCMGT_PPzZ0_H, FCMGT_PPzZ0_S)>;

// [114]   "fcmgt       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMGT_PPzZZ_D, FCMGT_PPzZZ_H, FCMGT_PPzZZ_S)>;

// [115]   "fcmla       $Zda, $Pg/m, $Zn, $Zm, $imm";
def : InstRW<[A64FXWrite_15Cyc_GI03], (instrs FCMLA_ZPmZZ_D, FCMLA_ZPmZZ_H, FCMLA_ZPmZZ_S)>;

// [116]   "fcmla       $Zda, $Zn, $Zm$iop, $imm";
def : InstRW<[A64FXWrite_15Cyc_GI03], (instrs FCMLA_ZZZI_H, FCMLA_ZZZI_S)>;

// [117]   "fcmle       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMLE_PPzZ0_D, FCMLE_PPzZ0_H, FCMLE_PPzZ0_S)>;

// [118]   "fcmlt       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMLT_PPzZ0_D, FCMLT_PPzZ0_H, FCMLT_PPzZ0_S)>;

// [119]   "fcmne       $Pd, $Pg/z, $Zn, #0.0";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMNE_PPzZ0_D, FCMNE_PPzZ0_H, FCMNE_PPzZ0_S)>;

// [120]   "fcmne       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMNE_PPzZZ_D, FCMNE_PPzZZ_H, FCMNE_PPzZZ_S)>;

// [121]   "fcmuo       $Pd, $Pg/z, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCMUO_PPzZZ_D, FCMUO_PPzZZ_H, FCMUO_PPzZZ_S)>;

// [122]   "fcpy        $Zd, $Pg/m, $imm8";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FCPY_ZPmI_D, FCPY_ZPmI_H, FCPY_ZPmI_S)>;

// [123]   "fcvt        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FCVT_ZPmZ_DtoH, FCVT_ZPmZ_DtoS, FCVT_ZPmZ_HtoD, FCVT_ZPmZ_HtoS, FCVT_ZPmZ_StoD, FCVT_ZPmZ_StoH)>;

// [124]   "fcvtzs      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FCVTZS_ZPmZ_DtoD, FCVTZS_ZPmZ_DtoS, FCVTZS_ZPmZ_HtoD, FCVTZS_ZPmZ_HtoH, FCVTZS_ZPmZ_HtoS, FCVTZS_ZPmZ_StoD, FCVTZS_ZPmZ_StoS)>;

// [125]   "fcvtzu      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FCVTZU_ZPmZ_DtoD, FCVTZU_ZPmZ_DtoS, FCVTZU_ZPmZ_HtoD, FCVTZU_ZPmZ_HtoH, FCVTZU_ZPmZ_HtoS, FCVTZU_ZPmZ_StoD, FCVTZU_ZPmZ_StoS)>;

// [126]   "fdiv        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_154Cyc_GI0], (instrs FDIV_ZPmZ_D)>;
def : InstRW<[A64FXWrite_134Cyc_GI0], (instrs FDIV_ZPmZ_H)>;
def : InstRW<[A64FXWrite_98Cyc_GI0], (instrs FDIV_ZPmZ_S)>;

// [127]   "fdivr       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_154Cyc_GI0], (instrs FDIVR_ZPmZ_D)>;
def : InstRW<[A64FXWrite_134Cyc_GI0], (instrs FDIVR_ZPmZ_H)>;
def : InstRW<[A64FXWrite_98Cyc_GI0], (instrs FDIVR_ZPmZ_S)>;

// [128]   "fdup        $Zd, $imm8";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FDUP_ZI_D, FDUP_ZI_H, FDUP_ZI_S)>;

// [129]   "fexpa       $Zd, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FEXPA_ZZ_D, FEXPA_ZZ_H, FEXPA_ZZ_S)>;

// [130]   "fmad        $Zdn, $Pg/m, $Zm, $Za";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FMAD_ZPmZZ_D, FMAD_ZPmZZ_H, FMAD_ZPmZZ_S)>;

// [131]   "fmax        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FMAX_ZPmZ_D, FMAX_ZPmZ_H, FMAX_ZPmZ_S)>;

// [132]   "fmax        $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FMAX_ZPmI_D, FMAX_ZPmI_H, FMAX_ZPmI_S)>;

// [133]   "fmaxnm      $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FMAXNM_ZPmZ_D, FMAXNM_ZPmZ_H, FMAXNM_ZPmZ_S)>;

// [134]   "fmaxnm      $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FMAXNM_ZPmI_D, FMAXNM_ZPmI_H, FMAXNM_ZPmI_S)>;

// [135]   "fmaxnmv     $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_10Cyc_GI03], (instrs FMAXNMV_VPZ_D, FMAXNMV_VPZ_H, FMAXNMV_VPZ_S)>;

// [136]   "fmaxv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_10Cyc_GI03], (instrs FMAXV_VPZ_D, FMAXV_VPZ_H, FMAXV_VPZ_S)>;

// [137]   "fmin        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FMIN_ZPmZ_D, FMIN_ZPmZ_H, FMIN_ZPmZ_S)>;

// [138]   "fmin        $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FMIN_ZPmI_D, FMIN_ZPmI_H, FMIN_ZPmI_S)>;

// [139]   "fminnm      $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FMINNM_ZPmZ_D, FMINNM_ZPmZ_H, FMINNM_ZPmZ_S)>;

// [140]   "fminnm      $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs FMINNM_ZPmI_D, FMINNM_ZPmI_H, FMINNM_ZPmI_S)>;

// [141]   "fminnmv     $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_10Cyc_GI03], (instrs FMINNMV_VPZ_D, FMINNMV_VPZ_H, FMINNMV_VPZ_S)>;

// [142]   "fminv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_10Cyc_GI03], (instrs FMINV_VPZ_D, FMINV_VPZ_H, FMINV_VPZ_S)>;

// [143]   "fmla        $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FMLA_ZPmZZ_D, FMLA_ZPmZZ_H, FMLA_ZPmZZ_S)>;

// [144]   "fmla        $Zda, $Zn, $Zm$iop";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FMLA_ZZZI_D, FMLA_ZZZI_H, FMLA_ZZZI_S)>;

// [145]   "fmls        $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FMLS_ZPmZZ_D, FMLS_ZPmZZ_H, FMLS_ZPmZZ_S)>;

// [146]   "fmls        $Zda, $Zn, $Zm$iop";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FMLS_ZZZI_D, FMLS_ZZZI_H, FMLS_ZZZI_S)>;

// [147]   "fmsb        $Zdn, $Pg/m, $Zm, $Za";

// [148]   "fmul        $Zd, $Zn, $Zm";

// [149]   "fmul        $Zd, $Zn, $Zm$iop";

// [150]   "fmul        $Zdn, $Pg/m, $_Zdn, $Zm";

// [151]   "fmul        $Zdn, $Pg/m, $_Zdn, $i1";

// [152]   "fmulx       $Zdn, $Pg/m, $_Zdn, $Zm";

// [153]   "fneg        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FNEG_ZPmZ_D, FNEG_ZPmZ_H, FNEG_ZPmZ_S)>;

// [154]   "fnmad       $Zdn, $Pg/m, $Zm, $Za";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FNMAD_ZPmZZ_D, FNMAD_ZPmZZ_H, FNMAD_ZPmZZ_S)>;

// [155]   "fnmla       $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FNMLA_ZPmZZ_D, FNMLA_ZPmZZ_H, FNMLA_ZPmZZ_S)>;

// [156]   "fnmls       $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FNMLS_ZPmZZ_D, FNMLS_ZPmZZ_H, FNMLS_ZPmZZ_S)>;

// [157]   "fnmsb       $Zdn, $Pg/m, $Zm, $Za";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FNMSB_ZPmZZ_D, FNMSB_ZPmZZ_H, FNMSB_ZPmZZ_S)>;

// [158]   "frecpe      $Zd, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FRECPE_ZZ_D, FRECPE_ZZ_H, FRECPE_ZZ_S)>;

// [159]   "frecps      $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRECPS_ZZZ_D, FRECPS_ZZZ_H, FRECPS_ZZZ_S)>;

// [160]   "frecpx      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FRECPX_ZPmZ_D, FRECPX_ZPmZ_H, FRECPX_ZPmZ_S)>;

// [161]   "frinta      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTA_ZPmZ_D, FRINTA_ZPmZ_H, FRINTA_ZPmZ_S)>;

// [162]   "frinti      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTI_ZPmZ_D, FRINTI_ZPmZ_H, FRINTI_ZPmZ_S)>;

// [163]   "frintm      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTM_ZPmZ_D, FRINTM_ZPmZ_H, FRINTM_ZPmZ_S)>;

// [164]   "frintn      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTN_ZPmZ_D, FRINTN_ZPmZ_H, FRINTN_ZPmZ_S)>;

// [165]   "frintp      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTP_ZPmZ_D, FRINTP_ZPmZ_H, FRINTP_ZPmZ_S)>;

// [166]   "frintx      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTX_ZPmZ_D, FRINTX_ZPmZ_H, FRINTX_ZPmZ_S)>;

// [167]   "frintz      $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRINTZ_ZPmZ_D, FRINTZ_ZPmZ_H, FRINTZ_ZPmZ_S)>;

// [168]   "frsqrte     $Zd, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FRSQRTE_ZZ_D, FRSQRTE_ZZ_H, FRSQRTE_ZZ_S)>;

// [169]   "frsqrts     $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs FRSQRTS_ZZZ_D, FRSQRTS_ZZZ_H, FRSQRTS_ZZZ_S)>;

// [170]   "fscale      $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FSCALE_ZPmZ_D, FSCALE_ZPmZ_H, FSCALE_ZPmZ_S)>;

// [171]   "fsqrt       $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_154Cyc_GI0], (instrs FSQRT_ZPmZ_D)>;
def : InstRW<[A64FXWrite_134Cyc_GI0], (instrs FSQRT_ZPmZ_H)>;
def : InstRW<[A64FXWrite_98Cyc_GI0], (instrs FSQRT_ZPmZ_S)>;

// [172]   "fsub        $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FSUB_ZZZ_D, FSUB_ZZZ_H, FSUB_ZZZ_S)>;

// [173]   "fsub        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FSUB_ZPmZ_D, FSUB_ZPmZ_H, FSUB_ZPmZ_S)>;

// [174]   "fsub        $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_9Cyc_GI0], (instrs FSUB_ZPmI_D, FSUB_ZPmI_H, FSUB_ZPmI_S)>;

// [175]   "fsubr       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FSUBR_ZPmZ_D, FSUBR_ZPmZ_H, FSUBR_ZPmZ_S)>;

// [176]   "fsubr       $Zdn, $Pg/m, $_Zdn, $i1";
def : InstRW<[A64FXWrite_9Cyc_GI0], (instrs FSUBR_ZPmI_D, FSUBR_ZPmI_H, FSUBR_ZPmI_S)>;

// [177]   "ftmad       $Zdn, $_Zdn, $Zm, $imm3";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FTMAD_ZZI_D, FTMAD_ZZI_H, FTMAD_ZZI_S)>;

// [178]   "ftsmul      $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs FTSMUL_ZZZ_D, FTSMUL_ZZZ_H, FTSMUL_ZZZ_S)>;

// [180]   "incb        $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs INCB_XPiI)>;

// [181]   "incd        $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs INCD_XPiI)>;

// [182]   "incd        $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs INCD_ZPiI)>;

// [183]   "inch        $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs INCH_XPiI)>;

// [184]   "inch        $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs INCH_ZPiI)>;

// [185]   "incp        $Rdn, $Pg";
def : InstRW<[A64FXWrite_6Cyc_GI124], (instrs INCP_XP_B, INCP_XP_D, INCP_XP_H, INCP_XP_S)>;

// [186]   "incp        $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs INCP_ZP_D, INCP_ZP_H, INCP_ZP_S)>;

// [187]   "incw        $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs INCW_XPiI)>;

// [188]   "incw        $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs INCW_ZPiI)>;

// [189]   "index       $Zd, $Rn, $Rm";
def : InstRW<[A64FXWrite_17Cyc_GI02], (instrs INDEX_RR_B, INDEX_RR_D, INDEX_RR_H, INDEX_RR_S)>;

// [190]   "index       $Zd, $Rn, $imm5";
def : InstRW<[A64FXWrite_21Cyc_GI02], (instrs INDEX_RI_B, INDEX_RI_D, INDEX_RI_H, INDEX_RI_S)>;

// [191]   "index       $Zd, $imm5, $Rm";
def : InstRW<[A64FXWrite_21Cyc_GI02], (instrs INDEX_IR_B, INDEX_IR_D, INDEX_IR_H, INDEX_IR_S)>;

// [192]   "index       $Zd, $imm5, $imm5b";
def : InstRW<[A64FXWrite_13Cyc_GI0], (instrs INDEX_II_B, INDEX_II_D, INDEX_II_H, INDEX_II_S)>;

// [193]   "insr        $Zdn, $Rm";
def : InstRW<[A64FXWrite_10Cyc_GI02], (instrs INSR_ZR_B, INSR_ZR_D, INSR_ZR_H, INSR_ZR_S)>;

// [194]   "insr        $Zdn, $Vm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs INSR_ZV_B, INSR_ZV_D, INSR_ZV_H, INSR_ZV_S)>;

// [195]   "lasta       $Rd, $Pg, $Zn";
def : InstRW<[A64FXWrite_25Cyc_GI056], (instrs LASTA_RPZ_B, LASTA_RPZ_D, LASTA_RPZ_H, LASTA_RPZ_S)>;

// [196]   "lasta       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs LASTA_VPZ_B, LASTA_VPZ_D, LASTA_VPZ_H, LASTA_VPZ_S)>;

// [197]   "lastb       $Rd, $Pg, $Zn";
def : InstRW<[A64FXWrite_25Cyc_GI056], (instrs LASTB_RPZ_B, LASTB_RPZ_D, LASTB_RPZ_H, LASTB_RPZ_S)>;

// [198]   "lastb       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs LASTB_VPZ_B, LASTB_VPZ_D, LASTB_VPZ_H, LASTB_VPZ_S)>;

// [199]   "ld1b        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1B, LD1B_D, LD1B_H, LD1B_S)>;

// [200]   "ld1b        $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1B_D_REAL, GLD1B_D_SXTW_REAL, GLD1B_D_UXTW_REAL, GLD1B_S_SXTW_REAL, GLD1B_S_UXTW_REAL)>;

// [201]   "ld1b        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1B_D_IMM_REAL, LD1B_H_IMM_REAL, LD1B_IMM_REAL, LD1B_S_IMM_REAL)>;

// [202]   "ld1b        $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1B_D_IMM_REAL, GLD1B_S_IMM_REAL)>;

// [203]   "ld1d        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1D)>;

// [204]   "ld1d        $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1D_REAL, GLD1D_SCALED_REAL, GLD1D_SXTW_REAL, GLD1D_SXTW_SCALED_REAL, GLD1D_UXTW_REAL, GLD1D_UXTW_SCALED_REAL)>;

// [205]   "ld1d        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1D_IMM_REAL)>;

// [206]   "ld1d        $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1D_IMM_REAL)>;

// [207]   "ld1h        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1H, LD1H_D, LD1H_S)>;

// [208]   "ld1h        $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1H_D_REAL, GLD1H_D_SCALED_REAL, GLD1H_D_SXTW_REAL, GLD1H_D_SXTW_SCALED_REAL, GLD1H_D_UXTW_REAL, GLD1H_D_UXTW_SCALED_REAL, GLD1H_S_SXTW_REAL, GLD1H_S_SXTW_SCALED_REAL, GLD1H_S_UXTW_REAL, GLD1H_S_UXTW_SCALED_REAL)>;

// [209]   "ld1h        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1H_D_IMM_REAL, LD1H_IMM_REAL, LD1H_S_IMM_REAL)>;

// [210]   "ld1h        $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1H_D_IMM_REAL, GLD1H_S_IMM_REAL)>;

// [211]   "ld1rb       $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RB_D_IMM, LD1RB_H_IMM, LD1RB_IMM, LD1RB_S_IMM)>;

// [212]   "ld1rd       $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RD_IMM)>;

// [213]   "ld1rh       $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RH_D_IMM, LD1RH_IMM, LD1RH_S_IMM)>;

// [214]   "ld1rqb      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_B)>;

// [215]   "ld1rqb      $Zt, $Pg/z, [$Rn, $imm4]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_B_IMM)>;

// [216]   "ld1rqd      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_D)>;

// [217]   "ld1rqd      $Zt, $Pg/z, [$Rn, $imm4]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_D_IMM)>;

// [218]   "ld1rqh      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_H)>;

// [219]   "ld1rqh      $Zt, $Pg/z, [$Rn, $imm4]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_H_IMM)>;

// [220]   "ld1rqw      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_W)>;

// [221]   "ld1rqw      $Zt, $Pg/z, [$Rn, $imm4]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RQ_W_IMM)>;

// [222]   "ld1rsb      $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RSB_D_IMM, LD1RSB_H_IMM, LD1RSB_S_IMM)>;

// [223]   "ld1rsh      $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RSH_D_IMM, LD1RSH_S_IMM)>;

// [224]   "ld1rsw      $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RSW_IMM)>;

// [225]   "ld1rw       $Zt, $Pg/z, [$Rn, $imm6]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1RW_D_IMM, LD1RW_IMM)>;

// [226]   "ld1sb       $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SB_D, LD1SB_H, LD1SB_S)>;

// [227]   "ld1sb       $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1SB_D_REAL, GLD1SB_D_SXTW_REAL, GLD1SB_D_UXTW_REAL, GLD1SB_S_SXTW_REAL, GLD1SB_S_UXTW_REAL)>;

// [228]   "ld1sb       $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SB_D_IMM_REAL, LD1SB_H_IMM_REAL, LD1SB_S_IMM_REAL)>;

// [229]   "ld1sb       $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1SB_D_IMM_REAL, GLD1SB_S_IMM_REAL)>;

// [230]   "ld1sh       $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SH_D, LD1SH_S)>;

// [231]   "ld1sh       $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1SH_D_REAL, GLD1SH_D_SCALED_REAL, GLD1SH_D_SXTW_REAL, GLD1SH_D_SXTW_SCALED_REAL, GLD1SH_D_UXTW_REAL, GLD1SH_D_UXTW_SCALED_REAL, GLD1SH_S_SXTW_REAL, GLD1SH_S_SXTW_SCALED_REAL, GLD1SH_S_UXTW_REAL, GLD1SH_S_UXTW_SCALED_REAL)>;

// [232]   "ld1sh       $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SH_D_IMM_REAL, LD1SH_S_IMM_REAL)>;

// [233]   "ld1sh       $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1SH_D_IMM_REAL, GLD1SH_S_IMM_REAL)>;

// [234]   "ld1sw       $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SW_D)>;

// [235]   "ld1sw       $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1SW_D_REAL, GLD1SW_D_SCALED_REAL, GLD1SW_D_SXTW_REAL, GLD1SW_D_SXTW_SCALED_REAL, GLD1SW_D_UXTW_REAL, GLD1SW_D_UXTW_SCALED_REAL)>;

// [236]   "ld1sw       $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1SW_D_IMM_REAL)>;

// [237]   "ld1sw       $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1SW_D_IMM_REAL)>;

// [238]   "ld1w        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1W, LD1W_D)>;

// [239]   "ld1w        $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLD1W_D_REAL, GLD1W_D_SCALED_REAL, GLD1W_D_SXTW_REAL, GLD1W_D_SXTW_SCALED_REAL, GLD1W_D_UXTW_REAL, GLD1W_D_UXTW_SCALED_REAL, GLD1W_SXTW_REAL, GLD1W_SXTW_SCALED_REAL, GLD1W_UXTW_REAL, GLD1W_UXTW_SCALED_REAL)>;

// [240]   "ld1w        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD1W_D_IMM_REAL, LD1W_IMM_REAL)>;

// [241]   "ld1w        $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLD1W_D_IMM_REAL, GLD1W_IMM_REAL)>;

// [242]   "ld2b        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD2B)>;

// [243]   "ld2b        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD2B_IMM)>;

// [244]   "ld2d        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD2D)>;

// [245]   "ld2d        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD2D_IMM)>;

// [246]   "ld2h        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD2H)>;

// [247]   "ld2h        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD2H_IMM)>;

// [248]   "ld2w        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD2W)>;

// [249]   "ld2w        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD2W_IMM)>;

// [250]   "ld3b        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD3B)>;

// [251]   "ld3b        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD3B_IMM)>;

// [252]   "ld3d        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD3D)>;

// [253]   "ld3d        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD3D_IMM)>;

// [254]   "ld3h        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD3H)>;

// [255]   "ld3h        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD3H_IMM)>;

// [256]   "ld3w        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD3W)>;

// [257]   "ld3w        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD3W_IMM)>;

// [258]   "ld4b        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD4B)>;

// [259]   "ld4b        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_44Cyc_GI56], (instrs LD4B_IMM)>;

// [260]   "ld4d        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4D)>;

// [261]   "ld4d        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4D_IMM)>;

// [262]   "ld4h        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4H)>;

// [263]   "ld4h        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4H_IMM)>;

// [264]   "ld4w        $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4W)>;

// [265]   "ld4w        $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LD4W_IMM)>;

// [266]   "ldff1b      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1B_D_REAL, LDFF1B_H_REAL, LDFF1B_REAL, LDFF1B_S_REAL)>;

// [267]   "ldff1b      $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1B_D_REAL, GLDFF1B_D_SXTW_REAL, GLDFF1B_D_UXTW_REAL, GLDFF1B_S_SXTW_REAL, GLDFF1B_S_UXTW_REAL)>;

// [268]   "ldff1b      $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1B_D_IMM_REAL, GLDFF1B_S_IMM_REAL)>;

// [269]   "ldff1d      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1D_REAL)>;

// [270]   "ldff1d      $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1D_REAL, GLDFF1D_SCALED_REAL, GLDFF1D_SXTW_REAL, GLDFF1D_SXTW_SCALED_REAL, GLDFF1D_UXTW_REAL, GLDFF1D_UXTW_SCALED_REAL)>;

// [271]   "ldff1d      $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1D_IMM_REAL)>;

// [272]   "ldff1h      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1H_D_REAL, LDFF1H_REAL, LDFF1H_S_REAL)>;

// [273]   "ldff1h      $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1H_D_REAL, GLDFF1H_D_SCALED_REAL, GLDFF1H_D_SXTW_REAL, GLDFF1H_D_SXTW_SCALED_REAL, GLDFF1H_D_UXTW_REAL, GLDFF1H_D_UXTW_SCALED_REAL, GLDFF1H_S_SXTW_REAL, GLDFF1H_S_SXTW_SCALED_REAL, GLDFF1H_S_UXTW_REAL, GLDFF1H_S_UXTW_SCALED_REAL)>;

// [274]   "ldff1h      $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1H_D_IMM_REAL, GLDFF1H_S_IMM_REAL)>;

// [275]   "ldff1sb     $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1SB_D_REAL, LDFF1SB_H_REAL, LDFF1SB_S_REAL)>;

// [276]   "ldff1sb     $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1SB_D_REAL, GLDFF1SB_D_SXTW_REAL, GLDFF1SB_D_UXTW_REAL, GLDFF1SB_S_SXTW_REAL, GLDFF1SB_S_UXTW_REAL)>;

// [277]   "ldff1sb     $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1SB_D_IMM_REAL, GLDFF1SB_S_IMM_REAL)>;

// [278]   "ldff1sh     $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1SH_D_REAL, LDFF1SH_S_REAL)>;

// [279]   "ldff1sh     $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1SH_D_REAL, GLDFF1SH_D_SCALED_REAL, GLDFF1SH_D_SXTW_REAL, GLDFF1SH_D_SXTW_SCALED_REAL, GLDFF1SH_D_UXTW_REAL, GLDFF1SH_D_UXTW_SCALED_REAL, GLDFF1SH_S_SXTW_REAL, GLDFF1SH_S_SXTW_SCALED_REAL, GLDFF1SH_S_UXTW_REAL, GLDFF1SH_S_UXTW_SCALED_REAL)>;

// [280]   "ldff1sh     $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1SH_D_IMM_REAL, GLDFF1SH_S_IMM_REAL)>;

// [281]   "ldff1sw     $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1SW_D_REAL)>;

// [282]   "ldff1sw     $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1SW_D_REAL, GLDFF1SW_D_SCALED_REAL, GLDFF1SW_D_SXTW_REAL, GLDFF1SW_D_SXTW_SCALED_REAL, GLDFF1SW_D_UXTW_REAL, GLDFF1SW_D_UXTW_SCALED_REAL)>;

// [283]   "ldff1sw     $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1SW_D_IMM_REAL)>;

// [284]   "ldff1w      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDFF1W_D_REAL, LDFF1W_REAL)>;

// [285]   "ldff1w      $Zt, $Pg/z, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_19Cyc_GI0256], (instrs GLDFF1W_D_REAL, GLDFF1W_D_SCALED_REAL, GLDFF1W_D_SXTW_REAL, GLDFF1W_D_SXTW_SCALED_REAL, GLDFF1W_D_UXTW_REAL, GLDFF1W_D_UXTW_SCALED_REAL, GLDFF1W_SXTW_REAL, GLDFF1W_SXTW_SCALED_REAL, GLDFF1W_UXTW_REAL, GLDFF1W_UXTW_SCALED_REAL)>;

// [286]   "ldff1w      $Zt, $Pg/z, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_15Cyc_GI056], (instrs GLDFF1W_D_IMM_REAL, GLDFF1W_IMM_REAL)>;

// [287]   "ldnf1b      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1B_D_IMM_REAL, LDNF1B_H_IMM_REAL, LDNF1B_IMM_REAL, LDNF1B_S_IMM_REAL)>;

// [288]   "ldnf1d      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1D_IMM_REAL)>;

// [289]   "ldnf1h      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1H_D_IMM_REAL, LDNF1H_IMM_REAL, LDNF1H_S_IMM_REAL)>;

// [290]   "ldnf1sb     $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1SB_D_IMM_REAL, LDNF1SB_H_IMM_REAL, LDNF1SB_S_IMM_REAL)>;

// [291]   "ldnf1sh     $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1SH_D_IMM_REAL, LDNF1SH_S_IMM_REAL)>;

// [292]   "ldnf1sw     $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1SW_D_IMM_REAL)>;

// [293]   "ldnf1w      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNF1W_D_IMM_REAL, LDNF1W_IMM_REAL)>;

// [294]   "ldnt1b      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1B_ZRR)>;

// [295]   "ldnt1b      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1B_ZRI)>;

// [296]   "ldnt1d      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1D_ZRR)>;

// [297]   "ldnt1d      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1D_ZRI)>;

// [298]   "ldnt1h      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1H_ZRR)>;

// [299]   "ldnt1h      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1H_ZRI)>;

// [300]   "ldnt1w      $Zt, $Pg/z, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1W_ZRR)>;

// [301]   "ldnt1w      $Zt, $Pg/z, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI56], (instrs LDNT1W_ZRI)>;

// [302]   "ldr $Pt, [$Rn, $imm9, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI5], (instrs LDR_PXI)>;

// [303]   "ldr $Zt, [$Rn, $imm9, mul vl]";
def : InstRW<[A64FXWrite_11Cyc_GI5], (instrs LDR_ZXI)>;

// [304]   "lsl $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSL_WIDE_ZZZ_B, LSL_WIDE_ZZZ_H, LSL_WIDE_ZZZ_S)>;

// [305]   "lsl $Zd, $Zn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSL_ZZI_B, LSL_ZZI_D, LSL_ZZI_H, LSL_ZZI_S)>;

// [306]   "lsl $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSL_WIDE_ZPmZ_B, LSL_WIDE_ZPmZ_H, LSL_WIDE_ZPmZ_S, LSL_ZPmZ_B, LSL_ZPmZ_D, LSL_ZPmZ_H, LSL_ZPmZ_S)>;

// [307]   "lsl $Zdn, $Pg/m, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSL_ZPmI_B, LSL_ZPmI_D, LSL_ZPmI_H, LSL_ZPmI_S)>;

// [308]   "lslr        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSLR_ZPmZ_B, LSLR_ZPmZ_D, LSLR_ZPmZ_H, LSLR_ZPmZ_S)>;

// [309]   "lsr $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSR_WIDE_ZZZ_B, LSR_WIDE_ZZZ_H, LSR_WIDE_ZZZ_S)>;

// [310]   "lsr $Zd, $Zn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSR_ZZI_B, LSR_ZZI_D, LSR_ZZI_H, LSR_ZZI_S)>;

// [311]   "lsr $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSR_WIDE_ZPmZ_B, LSR_WIDE_ZPmZ_H, LSR_WIDE_ZPmZ_S, LSR_ZPmZ_B, LSR_ZPmZ_D, LSR_ZPmZ_H, LSR_ZPmZ_S)>;

// [312]   "lsr $Zdn, $Pg/m, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSR_ZPmI_B, LSR_ZPmI_D, LSR_ZPmI_H, LSR_ZPmI_S)>;

// [313]   "lsrr        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs LSRR_ZPmZ_B, LSRR_ZPmZ_D, LSRR_ZPmZ_H, LSRR_ZPmZ_S)>;

// [314]   "mad $Zdn, $Pg/m, $Zm, $Za";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs MAD_ZPmZZ_B, MAD_ZPmZZ_D, MAD_ZPmZZ_H, MAD_ZPmZZ_S)>;

// [315]   "mla $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs MLA_ZPmZZ_B, MLA_ZPmZZ_D, MLA_ZPmZZ_H, MLA_ZPmZZ_S)>;

// [316]   "mls $Zda, $Pg/m, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs MLS_ZPmZZ_B, MLS_ZPmZZ_D, MLS_ZPmZZ_H, MLS_ZPmZZ_S)>;

// [317]   "movprfx     $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs MOVPRFX_ZPmZ_B, MOVPRFX_ZPmZ_D, MOVPRFX_ZPmZ_H, MOVPRFX_ZPmZ_S)>;

// [318]   "movprfx     $Zd, $Pg/z, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs MOVPRFX_ZPzZ_B, MOVPRFX_ZPzZ_D, MOVPRFX_ZPzZ_H, MOVPRFX_ZPzZ_S)>;

// [319]   "movprfx     $Zd, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs MOVPRFX_ZZ)>;

// [320]   "msb $Zdn, $Pg/m, $Zm, $Za";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs MSB_ZPmZZ_B, MSB_ZPmZZ_D, MSB_ZPmZZ_H, MSB_ZPmZZ_S)>;

// [321]   "mul $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs MUL_ZPmZ_B, MUL_ZPmZ_D, MUL_ZPmZ_H, MUL_ZPmZ_S)>;

// [322]   "mul $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_9Cyc_GI0], (instrs MUL_ZI_B, MUL_ZI_D, MUL_ZI_H, MUL_ZI_S)>;

// [323]   "nand        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs NAND_PPzPP)>;

// [324]   "nands       $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs NANDS_PPzPP)>;

// [325]   "neg $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs NEG_ZPmZ_B, NEG_ZPmZ_D, NEG_ZPmZ_H, NEG_ZPmZ_S)>;

// [326]   "nor $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs NOR_PPzPP)>;

// [327]   "nors        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs NORS_PPzPP)>;

// [328]   "not $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs NOT_ZPmZ_B, NOT_ZPmZ_D, NOT_ZPmZ_H, NOT_ZPmZ_S)>;

// [329]   "orn $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs ORN_PPzPP)>;

// [330]   "orns        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs ORNS_PPzPP)>;

// [331]   "orr $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs ORR_PPzPP)>;

// [332]   "orr $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ORR_ZZZ)>;

// [333]   "orr $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs ORR_ZPmZ_B, ORR_ZPmZ_D, ORR_ZPmZ_H, ORR_ZPmZ_S)>;

// [334]   "orr $Zdn, $_Zdn, $imms13";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs ORR_ZI)>;

// [335]   "orrs        $Pd, $Pg/z, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs ORRS_PPzPP)>;

// [336]   "orv $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs ORV_VPZ_B, ORV_VPZ_D, ORV_VPZ_H, ORV_VPZ_S)>;

// [337]   "pfalse      $Pd";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PFALSE)>;

// [338]   "pnext       $Pdn, $Pg, $_Pdn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PNEXT_B, PNEXT_D, PNEXT_H, PNEXT_S)>;

// [339]   "prfb        $prfop, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFB_PRR)>;

// [340]   "prfb        $prfop, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_14Cyc_GI0256], (instrs PRFB_D_SCALED, PRFB_D_SXTW_SCALED, PRFB_D_UXTW_SCALED, PRFB_S_SXTW_SCALED, PRFB_S_UXTW_SCALED)>;

// [341]   "prfb        $prfop, $Pg, [$Rn, $imm6, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFB_PRI)>;

// [342]   "prfb        $prfop, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_10Cyc_GI056], (instrs PRFB_D_PZI, PRFB_S_PZI)>;

// [343]   "prfd        $prfop, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFD_PRR)>;

// [344]   "prfd        $prfop, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_14Cyc_GI0256], (instrs PRFD_D_SCALED, PRFD_D_SXTW_SCALED, PRFD_D_UXTW_SCALED, PRFD_S_SXTW_SCALED, PRFD_S_UXTW_SCALED)>;

// [345]   "prfd        $prfop, $Pg, [$Rn, $imm6, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFD_PRI)>;

// [346]   "prfd        $prfop, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_10Cyc_GI056], (instrs PRFD_D_PZI, PRFD_S_PZI)>;

// [347]   "prfh        $prfop, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFH_PRR)>;

// [348]   "prfh        $prfop, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_14Cyc_GI0256], (instrs PRFH_D_SCALED, PRFH_D_SXTW_SCALED, PRFH_D_UXTW_SCALED, PRFH_S_SXTW_SCALED, PRFH_S_UXTW_SCALED)>;

// [349]   "prfh        $prfop, $Pg, [$Rn, $imm6, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFH_PRI)>;

// [350]   "prfh        $prfop, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_10Cyc_GI056], (instrs PRFH_D_PZI, PRFH_S_PZI)>;

// [351]   "prfw        $prfop, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFS_PRR)>;

// [352]   "prfw        $prfop, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_14Cyc_GI0256], (instrs PRFW_D_SCALED, PRFW_D_SXTW_SCALED, PRFW_D_UXTW_SCALED, PRFW_S_SXTW_SCALED, PRFW_S_UXTW_SCALED)>;

// [353]   "prfw        $prfop, $Pg, [$Rn, $imm6, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI56], (instrs PRFW_PRI)>;

// [354]   "prfw        $prfop, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_10Cyc_GI056], (instrs PRFW_D_PZI, PRFW_S_PZI)>;

// [355]   "ptest       $Pg, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PTEST_PP)>;

// [356]   "ptrue       $Pd, $pattern";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PTRUE_B, PTRUE_D, PTRUE_H, PTRUE_S)>;

// [357]   "ptrues      $Pd, $pattern";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PTRUES_B, PTRUES_D, PTRUES_H, PTRUES_S)>;

// [358]   "punpkhi     $Pd, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PUNPKHI_PP)>;

// [359]   "punpklo     $Pd, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs PUNPKLO_PP)>;

// [360]   "rbit        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs RBIT_ZPmZ_B, RBIT_ZPmZ_D, RBIT_ZPmZ_H, RBIT_ZPmZ_S)>;

// [361]   "rdffr       $Pd";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs RDFFR_P)>;

// [362]   "rdffr       $Pd, $Pg/z";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs RDFFR_PPz)>;

// [363]   "rdffrs      $Pd, $Pg/z";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs RDFFRS_PPz)>;

// [364]   "rdvl        $Rd, $imm6";
def : InstRW<[A64FXWrite_1Cyc_GI24], (instrs RDVLI_XI)>;

// [365]   "rev $Pd, $Pn";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs REV_PP_B, REV_PP_D, REV_PP_H, REV_PP_S)>;

// [366]   "rev $Zd, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs REV_ZZ_B, REV_ZZ_D, REV_ZZ_H, REV_ZZ_S)>;

// [367]   "revb        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs REVB_ZPmZ_D, REVB_ZPmZ_H, REVB_ZPmZ_S)>;

// [368]   "revh        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs REVH_ZPmZ_D, REVH_ZPmZ_S)>;

// [369]   "revw        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs REVW_ZPmZ_D)>;

// [370]   "sabd        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SABD_ZPmZ_B, SABD_ZPmZ_D, SABD_ZPmZ_H, SABD_ZPmZ_S)>;

// [371]   "saddv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_12Cyc_GI03], (instrs SADDV_VPZ_B, SADDV_VPZ_H, SADDV_VPZ_S)>;

// [372]   "scvtf       $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs SCVTF_ZPmZ_DtoD, SCVTF_ZPmZ_DtoH, SCVTF_ZPmZ_DtoS, SCVTF_ZPmZ_HtoH, SCVTF_ZPmZ_StoD, SCVTF_ZPmZ_StoH, SCVTF_ZPmZ_StoS)>;

// [373]   "sdiv        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_37Cyc_GI0], (instrs SDIV_ZPmZ_D, SDIV_ZPmZ_S)>;

// [374]   "sdivr       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_37Cyc_GI0], (instrs SDIVR_ZPmZ_D, SDIVR_ZPmZ_S)>;

// [375]   "sdot        $Zda, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs SDOT_ZZZ_D, SDOT_ZZZ_S)>;

// [376]   "sdot        $Zda, $Zn, $Zm$iop";
def : InstRW<[A64FXWrite_15Cyc_NGI03], (instrs SDOT_ZZZI_D, SDOT_ZZZI_S)>;

// [377]   "sel $Pd, $Pg, $Pn, $Pm";
def : InstRW<[A64FXWrite_3Cyc_GI1], (instrs SEL_PPPP)>;

// [378]   "sel $Zd, $Pg, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SEL_ZPZZ_B, SEL_ZPZZ_D, SEL_ZPZZ_H, SEL_ZPZZ_S)>;

// [379]   "setffr";
def : InstRW<[A64FXWrite_6Cyc], (instrs SETFFR)>;

// [380]   "smax        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SMAX_ZPmZ_B, SMAX_ZPmZ_D, SMAX_ZPmZ_H, SMAX_ZPmZ_S)>;

// [381]   "smax        $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs SMAX_ZI_B, SMAX_ZI_D, SMAX_ZI_H, SMAX_ZI_S)>;

// [382]   "smaxv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs SMAXV_VPZ_B, SMAXV_VPZ_D, SMAXV_VPZ_H, SMAXV_VPZ_S)>;

// [383]   "smin        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SMIN_ZPmZ_B, SMIN_ZPmZ_D, SMIN_ZPmZ_H, SMIN_ZPmZ_S)>;

// [384]   "smin        $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs SMIN_ZI_B, SMIN_ZI_D, SMIN_ZI_H, SMIN_ZI_S)>;

// [385]   "sminv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs SMINV_VPZ_B, SMINV_VPZ_D, SMINV_VPZ_H, SMINV_VPZ_S)>;

// [386]   "smulh       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs SMULH_ZPmZ_B, SMULH_ZPmZ_D, SMULH_ZPmZ_H, SMULH_ZPmZ_S)>;

// [387]   "splice      $Zdn, $Pg, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs SPLICE_ZPZ_B, SPLICE_ZPZ_D, SPLICE_ZPZ_H, SPLICE_ZPZ_S)>;

// [388]   "sqadd       $Zd, $Zn, $Zm";

// [389]   "sqadd       $Zdn, $_Zdn, $imm";

// [390]   "sqdecb      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECB_XPiWdI)>;

// [391]   "sqdecb      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECB_XPiI)>;

// [392]   "sqdecd      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECD_XPiWdI)>;

// [393]   "sqdecd      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECD_XPiI)>;

// [394]   "sqdecd      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQDECD_ZPiI)>;

// [395]   "sqdech      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECH_XPiWdI)>;

// [396]   "sqdech      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECH_XPiI)>;

// [397]   "sqdech      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQDECH_ZPiI)>;

// [398]   "sqdecp      $Rdn, $Pg";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs SQDECP_XP_B, SQDECP_XP_D, SQDECP_XP_H, SQDECP_XP_S)>;

// [399]   "sqdecp      $Rdn, $Pg, $_Rdn";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs SQDECP_XPWd_B, SQDECP_XPWd_D, SQDECP_XPWd_H, SQDECP_XPWd_S)>;

// [400]   "sqdecp      $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs SQDECP_ZP_D, SQDECP_ZP_H, SQDECP_ZP_S)>;

// [401]   "sqdecw      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECW_XPiWdI)>;

// [402]   "sqdecw      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQDECW_XPiI)>;

// [403]   "sqdecw      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQDECW_ZPiI)>;

// [404]   "sqincb      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCB_XPiWdI)>;

// [405]   "sqincb      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCB_XPiI)>;

// [406]   "sqincd      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCD_XPiWdI)>;

// [407]   "sqincd      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCD_XPiI)>;

// [408]   "sqincd      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQINCD_ZPiI)>;

// [409]   "sqinch      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCH_XPiWdI)>;

// [410]   "sqinch      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCH_XPiI)>;

// [411]   "sqinch      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQINCH_ZPiI)>;

// [412]   "sqincp      $Rdn, $Pg";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs SQINCP_XP_B, SQINCP_XP_D, SQINCP_XP_H, SQINCP_XP_S)>;

// [413]   "sqincp      $Rdn, $Pg, $_Rdn";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs SQINCP_XPWd_B, SQINCP_XPWd_D, SQINCP_XPWd_H, SQINCP_XPWd_S)>;

// [414]   "sqincp      $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs SQINCP_ZP_D, SQINCP_ZP_H, SQINCP_ZP_S)>;

// [415]   "sqincw      $Rdn, $_Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCW_XPiWdI)>;

// [416]   "sqincw      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs SQINCW_XPiI)>;

// [417]   "sqincw      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SQINCW_ZPiI)>;

// [418]   "sqsub       $Zd, $Zn, $Zm";

// [419]   "sqsub       $Zdn, $_Zdn, $imm";

// [420]   "st1b        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1B, ST1B_D, ST1B_H, ST1B_S)>;

// [421]   "st1b        $Zt, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_ST1W_19], (instrs SST1B_D_REAL, SST1B_D_SXTW, SST1B_D_UXTW, SST1B_S_SXTW, SST1B_S_UXTW)>;

// [422]   "st1b        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1B_D_IMM, ST1B_H_IMM, ST1B_IMM, ST1B_S_IMM)>;

// [423]   "st1b        $Zt, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_ST1W_15], (instrs SST1B_D_IMM, SST1B_S_IMM)>;

// [424]   "st1d        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1D)>;

// [425]   "st1d        $Zt, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_ST1W_19], (instrs SST1D_REAL, SST1D_SCALED_SCALED_REAL, SST1D_SXTW, SST1D_SXTW_SCALED, SST1D_UXTW, SST1D_UXTW_SCALED)>;

// [426]   "st1d        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1D_IMM)>;

// [427]   "st1d        $Zt, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_ST1W_15], (instrs SST1D_IMM)>;

// [428]   "st1h        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1H, ST1H_D, ST1H_S)>;

// [429]   "st1h        $Zt, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_ST1W_19], (instrs SST1H_D_REAL, SST1H_D_SCALED_SCALED_REAL, SST1H_D_SXTW, SST1H_D_SXTW_SCALED, SST1H_D_UXTW, SST1H_D_UXTW_SCALED, SST1H_S_SXTW, SST1H_S_SXTW_SCALED, SST1H_S_UXTW, SST1H_S_UXTW_SCALED)>;

// [430]   "st1h        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1H_D_IMM, ST1H_IMM, ST1H_S_IMM)>;

// [431]   "st1h        $Zt, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_ST1W_15], (instrs SST1H_D_IMM, SST1H_S_IMM)>;

// [432]   "st1w        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1W, ST1W_D)>;

// [433]   "st1w        $Zt, $Pg, [$Rn, $Zm]";
def : InstRW<[A64FXWrite_ST1W_19], (instrs SST1W_D_REAL, SST1W_D_SCALED_SCALED_REAL, SST1W_D_SXTW, SST1W_D_SXTW_SCALED, SST1W_D_UXTW, SST1W_D_UXTW_SCALED, SST1W_SXTW, SST1W_SXTW_SCALED, SST1W_UXTW, SST1W_UXTW_SCALED)>;

// [434]   "st1w        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs ST1W_D_IMM, ST1W_IMM)>;

// [435]   "st1w        $Zt, $Pg, [$Zn, $imm5]";
def : InstRW<[A64FXWrite_ST1W_15], (instrs SST1W_D_IMM, SST1W_IMM)>;

// [436]   "st2b        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2B)>;

// [437]   "st2b        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2B_IMM)>;

// [438]   "st2d        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2D)>;

// [439]   "st2d        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2D_IMM)>;

// [440]   "st2h        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2H)>;

// [441]   "st2h        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2H_IMM)>;

// [442]   "st2w        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2W)>;

// [443]   "st2w        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST2W_7], (instrs ST2W_IMM)>;

// [444]   "st3b        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3B)>;

// [445]   "st3b        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3B_IMM)>;

// [446]   "st3d        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3D)>;

// [447]   "st3d        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3D_IMM)>;

// [448]   "st3h        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3H)>;

// [449]   "st3h        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3H_IMM)>;

// [450]   "st3w        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3W)>;

// [451]   "st3w        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST3W_8], (instrs ST3W_IMM)>;

// [452]   "st4b        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4B)>;

// [453]   "st4b        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4B_IMM)>;

// [454]   "st4d        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4D)>;

// [455]   "st4d        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4D_IMM)>;

// [456]   "st4h        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4H)>;

// [457]   "st4h        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4H_IMM)>;

// [458]   "st4w        $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4W)>;

// [459]   "st4w        $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST4W_9], (instrs ST4W_IMM)>;

// [460]   "stnt1b      $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1B_ZRR)>;

// [461]   "stnt1b      $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1B_ZRI)>;

// [462]   "stnt1d      $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1D_ZRR)>;

// [463]   "stnt1d      $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1D_ZRI)>;

// [464]   "stnt1h      $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1H_ZRR)>;

// [465]   "stnt1h      $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1H_ZRI)>;

// [466]   "stnt1w      $Zt, $Pg, [$Rn, $Rm]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1W_ZRR)>;

// [467]   "stnt1w      $Zt, $Pg, [$Rn, $imm4, mul vl]";
def : InstRW<[A64FXWrite_ST1W_6], (instrs STNT1W_ZRI)>;

// [468]   "str $Pt, [$Rn, $imm9, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI15], (instrs STR_PXI)>;

// [469]   "str $Zt, [$Rn, $imm9, mul vl]";
def : InstRW<[A64FXWrite_6Cyc_GI05], (instrs STR_ZXI)>;

// [470]   "sub $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SUB_ZZZ_B, SUB_ZZZ_D, SUB_ZZZ_H, SUB_ZZZ_S)>;

// [471]   "sub $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SUB_ZPmZ_B, SUB_ZPmZ_D, SUB_ZPmZ_H, SUB_ZPmZ_S)>;

// [472]   "sub $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SUB_ZI_B, SUB_ZI_D, SUB_ZI_H, SUB_ZI_S)>;

// [473]   "subr        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SUBR_ZPmZ_B, SUBR_ZPmZ_D, SUBR_ZPmZ_H, SUBR_ZPmZ_S)>;

// [474]   "subr        $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs SUBR_ZI_B, SUBR_ZI_D, SUBR_ZI_H, SUBR_ZI_S)>;

// [475]   "sunpkhi     $Zd, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs SUNPKHI_ZZ_D, SUNPKHI_ZZ_H, SUNPKHI_ZZ_S)>;

// [476]   "sunpklo     $Zd, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs SUNPKLO_ZZ_D, SUNPKLO_ZZ_H, SUNPKLO_ZZ_S)>;

// [477]   "sxtb        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SXTB_ZPmZ_D, SXTB_ZPmZ_H, SXTB_ZPmZ_S)>;

// [478]   "sxth        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SXTH_ZPmZ_D, SXTH_ZPmZ_S)>;

// [479]   "sxtw        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs SXTW_ZPmZ_D)>;

// [480]   "tbl $Zd, $Zn, $Zm";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs TBL_ZZZ_B, TBL_ZZZ_D, TBL_ZZZ_H, TBL_ZZZ_S)>;

// [481]   "trn1        $Pd, $Pn, $Pm";

// [482]   "trn1        $Zd, $Zn, $Zm";

// [483]   "trn2        $Pd, $Pn, $Pm";

// [484]   "trn2        $Zd, $Zn, $Zm";

// [486]   "uabd        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UABD_ZPmZ_B, UABD_ZPmZ_D, UABD_ZPmZ_H, UABD_ZPmZ_S)>;

// [487]   "uaddv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_12Cyc_GI03], (instrs UADDV_VPZ_B, UADDV_VPZ_D, UADDV_VPZ_H, UADDV_VPZ_S)>;

// [488]   "ucvtf       $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs UCVTF_ZPmZ_DtoD, UCVTF_ZPmZ_DtoH, UCVTF_ZPmZ_DtoS, UCVTF_ZPmZ_HtoH, UCVTF_ZPmZ_StoD, UCVTF_ZPmZ_StoH, UCVTF_ZPmZ_StoS)>;

// [489]   "udiv        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_37Cyc_GI0], (instrs UDIV_ZPmZ_D, UDIV_ZPmZ_S)>;

// [490]   "udivr       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_37Cyc_GI0], (instrs UDIVR_ZPmZ_D, UDIVR_ZPmZ_S)>;

// [491]   "udot        $Zda, $Zn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs UDOT_ZZZ_D, UDOT_ZZZ_S)>;

// [492]   "udot        $Zda, $Zn, $Zm$iop";
def : InstRW<[A64FXWrite_15Cyc_NGI03], (instrs UDOT_ZZZI_D, UDOT_ZZZI_S)>;

// [493]   "umax        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UMAX_ZPmZ_B, UMAX_ZPmZ_D, UMAX_ZPmZ_H, UMAX_ZPmZ_S)>;

// [494]   "umax        $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs UMAX_ZI_B, UMAX_ZI_D, UMAX_ZI_H, UMAX_ZI_S)>;

// [495]   "umaxv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs UMAXV_VPZ_B, UMAXV_VPZ_D, UMAXV_VPZ_H, UMAXV_VPZ_S)>;

// [496]   "umin        $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UMIN_ZPmZ_B, UMIN_ZPmZ_D, UMIN_ZPmZ_H, UMIN_ZPmZ_S)>;

// [497]   "umin        $Zdn, $_Zdn, $imm";
def : InstRW<[A64FXWrite_4Cyc_GI0], (instrs UMIN_ZI_B, UMIN_ZI_D, UMIN_ZI_H, UMIN_ZI_S)>;

// [498]   "uminv       $Vd, $Pg, $Zn";
def : InstRW<[A64FXWrite_14Cyc_GI03], (instrs UMINV_VPZ_B, UMINV_VPZ_D, UMINV_VPZ_H, UMINV_VPZ_S)>;

// [499]   "umulh       $Zdn, $Pg/m, $_Zdn, $Zm";
def : InstRW<[A64FXWrite_9Cyc_GI03], (instrs UMULH_ZPmZ_B, UMULH_ZPmZ_D, UMULH_ZPmZ_H, UMULH_ZPmZ_S)>;

// [500]   "uqadd       $Zd, $Zn, $Zm";

// [501]   "uqadd       $Zdn, $_Zdn, $imm";

// [502]   "uqdecb      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQDECB_WPiI, UQDECB_XPiI)>;

// [503]   "uqdecd      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQDECD_WPiI, UQDECD_XPiI)>;

// [504]   "uqdecd      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQDECD_ZPiI)>;

// [505]   "uqdech      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQDECH_WPiI, UQDECH_XPiI)>;

// [506]   "uqdech      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQDECH_ZPiI)>;

// [507]   "uqdecp      $Rdn, $Pg";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs UQDECP_WP_B, UQDECP_WP_D, UQDECP_WP_H, UQDECP_WP_S, UQDECP_XP_B, UQDECP_XP_D, UQDECP_XP_H, UQDECP_XP_S)>;

// [508]   "uqdecp      $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs UQDECP_ZP_D, UQDECP_ZP_H, UQDECP_ZP_S)>;

// [509]   "uqdecw      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQDECW_WPiI, UQDECW_XPiI)>;

// [510]   "uqdecw      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQDECW_ZPiI)>;

// [511]   "uqincb      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQINCB_WPiI, UQINCB_XPiI)>;

// [512]   "uqincd      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQINCD_WPiI, UQINCD_XPiI)>;

// [513]   "uqincd      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQINCD_ZPiI)>;

// [514]   "uqinch      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQINCH_WPiI, UQINCH_XPiI)>;

// [515]   "uqinch      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQINCH_ZPiI)>;

// [516]   "uqincp      $Rdn, $Pg";
def : InstRW<[A64FXWrite_8Cyc_GI124], (instrs UQINCP_WP_B, UQINCP_WP_D, UQINCP_WP_H, UQINCP_WP_S, UQINCP_XP_B, UQINCP_XP_D, UQINCP_XP_H, UQINCP_XP_S)>;

// [517]   "uqincp      $Zdn, $Pg";
def : InstRW<[A64FXWrite_12Cyc_GI01], (instrs UQINCP_ZP_D, UQINCP_ZP_H, UQINCP_ZP_S)>;

// [518]   "uqincw      $Rdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_2Cyc_GI24], (instrs UQINCW_WPiI, UQINCW_XPiI)>;

// [519]   "uqincw      $Zdn, $pattern, mul $imm4";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQINCW_ZPiI)>;

// [520]   "uqsub       $Zd, $Zn, $Zm";
//@@@ def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQSUB_ZZZ_B, UQSUB_ZZZ_D, UQSUB_ZZZ_H, UQSUB_ZZZ_S)>;

// [521]   "uqsub       $Zdn, $_Zdn, $imm";
//@@@ def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UQSUB_ZI_B, UQSUB_ZI_D, UQSUB_ZI_H, UQSUB_ZI_S)>;

// [522]   "uunpkhi     $Zd, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs UUNPKHI_ZZ_D, UUNPKHI_ZZ_H, UUNPKHI_ZZ_S)>;

// [523]   "uunpklo     $Zd, $Zn";
def : InstRW<[A64FXWrite_6Cyc_GI0], (instrs UUNPKLO_ZZ_D, UUNPKLO_ZZ_H, UUNPKLO_ZZ_S)>;

// [524]   "uxtb        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UXTB_ZPmZ_D, UXTB_ZPmZ_H, UXTB_ZPmZ_S)>;

// [525]   "uxth        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UXTH_ZPmZ_D, UXTH_ZPmZ_S)>;

// [526]   "uxtw        $Zd, $Pg/m, $Zn";
def : InstRW<[A64FXWrite_4Cyc_GI03], (instrs UXTW_ZPmZ_D)>;

// [527]   "uzp1        $Pd, $Pn, $Pm";

// [528]   "uzp1        $Zd, $Zn, $Zm";

// [529]   "uzp2        $Pd, $Pn, $Pm";

// [530]   "uzp2        $Zd, $Zn, $Zm";

// [531]   "whilele     $Pd, $Rn, $Rm";
def : InstRW<[A64FXWrite_4Cyc_GI12], (instrs WHILELE_PWW_B, WHILELE_PWW_D, WHILELE_PWW_H, WHILELE_PWW_S, WHILELE_PXX_B, WHILELE_PXX_D, WHILELE_PXX_H, WHILELE_PXX_S)>;

// [532]   "whilelo     $Pd, $Rn, $Rm";
def : InstRW<[A64FXWrite_4Cyc_GI12], (instrs WHILELO_PWW_B, WHILELO_PWW_D, WHILELO_PWW_H, WHILELO_PWW_S, WHILELO_PXX_B, WHILELO_PXX_D, WHILELO_PXX_H, WHILELO_PXX_S)>;

// [533]   "whilels     $Pd, $Rn, $Rm";
def : InstRW<[A64FXWrite_4Cyc_GI12], (instrs WHILELS_PWW_B, WHILELS_PWW_D, WHILELS_PWW_H, WHILELS_PWW_S, WHILELS_PXX_B, WHILELS_PXX_D, WHILELS_PXX_H, WHILELS_PXX_S)>;

// [534]   "whilelt     $Pd, $Rn, $Rm";
def : InstRW<[A64FXWrite_4Cyc_GI12], (instrs WHILELT_PWW_B, WHILELT_PWW_D, WHILELT_PWW_H, WHILELT_PWW_S, WHILELT_PXX_B, WHILELT_PXX_D, WHILELT_PXX_H, WHILELT_PXX_S)>;

// [535]   "wrffr       $Pn";
def : InstRW<[A64FXWrite_6Cyc_NGI1], (instrs WRFFR)>;

// [536]   "zip1        $Pd, $Pn, $Pm";

// [537]   "zip1        $Zd, $Zn, $Zm";

// [538]   "zip2        $Pd, $Pn, $Pm";

// [539]   "zip2        $Zd, $Zn, $Zm";

} // SchedModel = A64FXModel
