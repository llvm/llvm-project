//=- X86ScheduleC864GM4.td - X86 C86-4G-M4 Scheduling --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for C86-4G-M4 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def C864GM4Model : SchedMachineModel {
  let IssueWidth = 4;
  let MicroOpBufferSize = 224;
  // The maximum dispatch is 6 ops/cycle, and a mispredict cost is 16 cycle from
  // the op-cache, so the loop buffer is limited to 6*16 = 96.
  let LoopMicroOpBufferSize = 96;
  // The common case of branch misprediction penalty is 16 cycles.
  let MispredictPenalty = 16;

  let LoadLatency = 4;
  int VecLoadLatency = 7;
  int StoreLatency = 1;
  let HighLatency = 25;

  let PostRAScheduler = 1;

  let CompleteModel = 1;
}

let SchedModel = C864GM4Model in {

//===----------------------------------------------------------------------===//
// Integer Execution Unit
//===----------------------------------------------------------------------===//

// The C864GM4 has 4 ALUs.
def 4GM4ALU0 : ProcResource<1>;
def 4GM4ALU1 : ProcResource<1>;
def 4GM4ALU2 : ProcResource<1>;
def 4GM4ALU3 : ProcResource<1>;

def 4GM4ALU23  : ProcResGroup<[4GM4ALU2, 4GM4ALU3]>;

// BRU on ALU0 and ALU3.
defvar 4GM4BRU0 = 4GM4ALU0;
defvar 4GM4BRU1 = 4GM4ALU3;

def 4GM4BRU01 : ProcResGroup<[4GM4BRU0, 4GM4BRU1]>;

// 72 Entry (4x18 entries) integer Scheduler.
def 4GM4ALU : ProcResGroup<[4GM4ALU0, 4GM4BRU0,
                            4GM4ALU1,
                            4GM4ALU2,
                            4GM4ALU3, 4GM4BRU1]> {
  let BufferSize = 72;
}

// The integer physical register file consists of 192 registers.
def 4GM4IntegerPRF : RegisterFile<192,
                                  [GR64, CCR],
                                  [1, 1],
                                  [1, 0],
                                  6, // Max moves that can be eliminated per cycle.
                                  0>; // Restrict move elimination to zero regs.

// AGU
// The C864GM4 has 3 AGUs.
def 4GM4AGU0 : ProcResource<1>;
def 4GM4AGU1 : ProcResource<1>;
def 4GM4AGU2 : ProcResource<1>;
// 28 Entry (1x28 entries) AGU group.
def 4GM4AGU : ProcResGroup<[4GM4AGU0, 4GM4AGU1, 4GM4AGU2]> {
  let BufferSize = 28;
}

//===----------------------------------------------------------------------===//
// Floating-Point Unit
//===----------------------------------------------------------------------===//

// The C864GM4 has 4 FPUs.
def 4GM4FPU0 : ProcResource<1>;
def 4GM4FPU1 : ProcResource<1>;
def 4GM4FPU2 : ProcResource<1>;
def 4GM4FPU3 : ProcResource<1>;

def 4GM4FPU01  : ProcResGroup<[4GM4FPU0, 4GM4FPU1]>;
def 4GM4FPU13  : ProcResGroup<[4GM4FPU1, 4GM4FPU3]>;
def 4GM4FPU23  : ProcResGroup<[4GM4FPU2, 4GM4FPU3]>;
def 4GM4FPU02  : ProcResGroup<[4GM4FPU0, 4GM4FPU2]>;

// 48 Entry (4x12 entries) floating-point Scheduler.
def 4GM4FPU : ProcResGroup<[4GM4FPU0, 4GM4FPU1, 4GM4FPU2, 4GM4FPU3]> {
  let BufferSize = 48;
}

// The floating point physical register file consists of 176 registers.
def 4GM4FpuPRF : RegisterFile<176,
                              [VR64, VR128, VR256],
                              [1, 1, 1],
                              [0, 1, 1],
                              6, // Max moves that can be eliminated per cycle.
                              0>; // Restrict move elimination to zero regs.

//===----------------------------------------------------------------------===//
// Load-Store Unit
//===----------------------------------------------------------------------===//

// Load/Store Units and Memory Queues
// The C864GM4 has 3 LS Units.
def 4GM4LSU : ProcResource<3>;

// The two of LSU can be loads.
let Super = 4GM4LSU in
def 4GM4Load : ProcResource<2> {
  // The LDQ is 50.
  let BufferSize = 50;
}
def 4GM4LoadQueue : LoadQueue<4GM4Load>;

// All three of LSU can be loads.
let Super = 4GM4LSU in
def 4GM4Store : ProcResource<3> {
  // The STQ is 52.
  let BufferSize = 52;
}
def 4GM4StoreQueue : StoreQueue<4GM4Store>;

def : ReadAdvance<ReadAfterLd, C864GM4Model.LoadLatency>;
def : ReadAdvance<ReadAfterVecLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecXLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecYLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadInt2Fpu, 0>;

//===----------------------------------------------------------------------===//
// Retire Control Unit
//===----------------------------------------------------------------------===//

def 4GM4RCU : RetireControlUnit<C864GM4Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Basic helper classes.
//===----------------------------------------------------------------------===//

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.


multiclass __4GM4WriteRes<SchedWrite SchedRW, list<ProcResourceKind> ExePorts,
                         int Lat = 1, list<int> Res = [], int UOps = 1> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }
}

multiclass __4GM4WriteResPair<X86FoldableSchedWrite SchedRW,
                             list<ProcResourceKind> ExePorts, int Lat,
                             list<int> Res, int UOps, int LoadLat, int LoadUOps,
                             ProcResourceKind AGU, int LoadRes> {
  defm : __4GM4WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;

  defm : __4GM4WriteRes<SchedRW.Folded,
                       !listconcat([AGU, 4GM4Load], ExePorts),
                       !add(LoadLat, Lat),
                       !if(!and(!empty(Res), !eq(LoadRes, 1)),
                         [],
                         !listconcat([1, LoadRes],
                           !if(!empty(Res),
                             !listsplat(1, !size(ExePorts)),
                             Res))),
                       !add(UOps, LoadUOps)>;
}

// For classes without folded loads.
multiclass 4GM4WriteRes<SchedWrite SchedRW,
                        list<ProcResourceKind> ExePorts, int Lat = 1,
                        list<int> Res = [], int UOps = 1> {
  defm : __4GM4WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

// For classes with folded loads.
multiclass 4GM4WriteResIntPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __4GM4WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM4Model.LoadLatency,
                           LoadUOps, 4GM4AGU, LoadRes>;
}

multiclass 4GM4WriteResFPPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __4GM4WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM4Model.VecLoadLatency,
                           LoadUOps, 4GM4AGU, LoadRes>;
}

// Microcoded Instructions
def 4GM4WriteMicrocoded : SchedWriteRes<[]> {
  let Latency = 100;
}

def : SchedAlias<WriteMicrocoded, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteSystem, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteFCMOV, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteLDMXCSR, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteSTMXCSR, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPS, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSY, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSLd, 4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSYLd,4GM4WriteMicrocoded>;

def 4GM4WritePDEP_PEXT : SchedWriteRes<[]>;
def : InstRW<[4GM4WritePDEP_PEXT], (instregex "PDEP(32|64)rr",
                                              "PEXT(32|64)rr")>;
def : SchedAlias<4GM4WritePDEP_PEXT, 4GM4WriteMicrocoded>;
def 4GM4WritePDEP_PEXTLd : SchedWriteRes<[]>;
def : InstRW<[4GM4WritePDEP_PEXTLd], (instregex "PDEP(32|64)rm",
                                                "PEXT(32|64)rm")>;
def : SchedAlias<4GM4WritePDEP_PEXTLd, 4GM4WriteMicrocoded>;

// Integer Instructions

defm : 4GM4WriteRes<WriteRMW, [4GM4AGU, 4GM4Store]>;

defm : 4GM4WriteRes<WriteStore, [4GM4AGU, 4GM4Store]>;
defm : 4GM4WriteRes<WriteStoreNT, [4GM4AGU, 4GM4Store]>;
defm : 4GM4WriteRes<WriteMove, [4GM4ALU]>;
defm : 4GM4WriteRes<WriteLoad, [4GM4AGU, 4GM4Load], !add(1, C864GM4Model.LoadLatency)>;

defm : 4GM4WriteRes<WriteVecMaskedGatherWriteback, [], !add(1, C864GM4Model.VecLoadLatency)>;

defm : 4GM4WriteRes<WriteZero, [], 0, [], 1>;
defm : 4GM4WriteRes<WriteLEA, [4GM4ALU]>;
defm : 4GM4WriteResIntPair<WriteALU, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteADC, [4GM4ALU], 1>;

// This write is used for slow LEA instructions.
def 4GM4Write3OpsLEA : SchedWriteRes<[4GM4ALU23]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
// a slow LEA is either a 3Ops LEA (base, index, offset),
// or an LEA with a `Scale` value different than 1.
def 4GM4SlowLEAPredicate : MCSchedPredicate<
  CheckAny<[
    IsThreeOperandsLEAFn,
    CheckAll<[
      CheckIsImmOperand<2>,
      CheckNot<CheckImmOperand<2, 1>>
    ]>
  ]>
>;

def 4GM4WriteLEA : SchedWriteVariant<[
    SchedVar<4GM4SlowLEAPredicate, [4GM4Write3OpsLEA]>,
    SchedVar<NoSchedPred,          [WriteLEA]>
]>;

def : InstRW<[4GM4WriteLEA], (instrs LEA64r, LEA64_32r)>;

// MUL IMUL MULX
defm : 4GM4WriteResIntPair<WriteIMul8, [4GM4ALU1], 3, [3], 1>;
defm : 4GM4WriteResIntPair<WriteIMul16, [4GM4ALU1], 3, [3], 1>;
defm : 4GM4WriteResIntPair<WriteIMul16Imm, [4GM4ALU1], 4, [4], 2>;
defm : 4GM4WriteResIntPair<WriteIMul16Reg, [4GM4ALU1], 3>;
defm : 4GM4WriteResIntPair<WriteIMul32, [4GM4ALU1], 3, [3], 1>;
defm : 4GM4WriteResIntPair<WriteIMul32Imm, [4GM4ALU1], 3, [3], 2>;
defm : 4GM4WriteResIntPair<WriteIMul32Reg, [4GM4ALU1], 3>;
defm : 4GM4WriteResIntPair<WriteIMul64, [4GM4ALU1], 3, [3], 1>;
defm : 4GM4WriteResIntPair<WriteIMul64Imm, [4GM4ALU1], 3, [3], 2>;
defm : 4GM4WriteResIntPair<WriteIMul64Reg, [4GM4ALU1], 3>;

defm : 4GM4WriteResIntPair<WriteMULX32, [4GM4ALU1], 3>;
defm : 4GM4WriteResIntPair<WriteMULX64, [4GM4ALU1], 3>;
defm : 4GM4WriteRes<WriteIMulHLd, [4GM4ALU1], !add(3, C864GM4Model.LoadLatency), [], 0>;
defm : 4GM4WriteRes<WriteIMulH, [4GM4ALU1], 3, [], 0>;

def 4GM4WriteMULX:SchedWriteRes<[4GM4ALU1]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteMULX], (instregex "MULX(32|64)rr")>;

def 4GM4WriteMULXLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4ALU1]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteMULX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteMULXLd], (instregex "MULX(32|64)rm")>;

// DIV IDIV
defm : 4GM4WriteResIntPair<WriteDiv8, [4GM4ALU2], 12, [12], 2>;
defm : 4GM4WriteResIntPair<WriteDiv16, [4GM4ALU2], 17, [17], 2>;
defm : 4GM4WriteResIntPair<WriteDiv32, [4GM4ALU2], 25, [25], 2>;
defm : 4GM4WriteResIntPair<WriteDiv64, [4GM4ALU2], 41, [41], 1>;
defm : 4GM4WriteResIntPair<WriteIDiv8, [4GM4ALU2], 12, [12], 2>;
defm : 4GM4WriteResIntPair<WriteIDiv16, [4GM4ALU2], 17, [17], 2>;
defm : 4GM4WriteResIntPair<WriteIDiv32, [4GM4ALU2], 25, [25], 2>;
defm : 4GM4WriteResIntPair<WriteIDiv64, [4GM4ALU2], 41, [41], 1>;

defm : 4GM4WriteRes<WriteBSWAP32, [4GM4ALU], 1, [4], 1>;
defm : 4GM4WriteRes<WriteBSWAP64, [4GM4ALU], 1, [4], 1>;
defm : 4GM4WriteRes<WriteCMPXCHG, [4GM4ALU], 3>;
defm : 4GM4WriteRes<WriteCMPXCHGRMW,[4GM4ALU, 4GM4AGU], 5, [1,1], 3>;

def 4GM4WriteXCHGrm : SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4ALU]> {
  let Latency = !add(C864GM4Model.LoadLatency, 1);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteXCHGrm], (instregex "XCHG(8|16|32|64)rm")>;

defm : 4GM4WriteResIntPair<WriteShift, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteShiftCL, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteRotate, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteRotateCL, [4GM4ALU], 1>;

def 4GM4WriteRCR:SchedWriteRes<[4GM4ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteRCR], (instregex "RCR(8|16|32|64)rCL")>;

def 4GM4WriteRCRLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4ALU]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteRCR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteRCRLd], (instregex "RCR(8|16|32|64)mCL")>;

def 4GM4WriteRCL:SchedWriteRes<[4GM4ALU]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteRCL], (instregex "RCL(8|16|32|64)rCL")>;

def 4GM4WriteRCLLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4ALU]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteRCL.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteRCLLd], (instregex "RCL(8|16|32|64)mCL")>;


defm : 4GM4WriteRes<WriteSHDrri, [4GM4ALU], 3, [3], 6>;
defm : 4GM4WriteRes<WriteSHDrrcl, [4GM4ALU], 3, [5], 7>;
defm : 4GM4WriteRes<WriteSHDmri, [4GM4ALU], 5, [5], 8>;
defm : 4GM4WriteRes<WriteSHDmrcl, [4GM4ALU], 5, [5], 8>;

defm : 4GM4WriteResIntPair<WriteJump, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteCRC32, [4GM4ALU3], 3, [3], 3>;

defm : 4GM4WriteResIntPair<WriteCMOV, [4GM4ALU], 1>;
defm : 4GM4WriteRes<WriteSETCC, [4GM4ALU]>;
defm : 4GM4WriteRes<WriteSETCCStore, [4GM4ALU, 4GM4AGU, 4GM4Store]>;
defm : 4GM4WriteRes<WriteLAHFSAHF, [4GM4ALU], 2, [1], 2>;

defm : 4GM4WriteRes<WriteBitTest, [4GM4ALU], 1>;
defm : 4GM4WriteRes<WriteBitTestImmLd, [4GM4AGU, 4GM4Load, 4GM4ALU], !add(C864GM4Model.LoadLatency, 1), [1,1,1], 2>;
defm : 4GM4WriteRes<WriteBitTestRegLd, [4GM4AGU, 4GM4Load, 4GM4ALU], !add(C864GM4Model.LoadLatency, 1), [1,1,1], 7>;
defm : 4GM4WriteRes<WriteBitTestSet, [4GM4ALU], 2, [1], 2>;
defm : 4GM4WriteRes<WriteBitTestSetImmLd, [4GM4AGU, 4GM4Load, 4GM4ALU], !add(C864GM4Model.LoadLatency, 2), [1, 1, 1], 2>;
defm : 4GM4WriteRes<WriteBitTestSetRegLd, [4GM4AGU, 4GM4Load, 4GM4ALU], !add(C864GM4Model.LoadLatency, 2), [1, 1, 1], 7>;

// Bit counts.
defm : 4GM4WriteResIntPair<WriteBSF, [4GM4ALU], 3, [12], 6, 4, 2>;
defm : 4GM4WriteResIntPair<WriteBSR, [4GM4ALU], 4, [16], 6, 4, 2>;
defm : 4GM4WriteResIntPair<WriteLZCNT, [4GM4ALU], 1>;
defm : 4GM4WriteResIntPair<WriteTZCNT, [4GM4ALU], 2, [2], 2, 4, 0>;
defm : 4GM4WriteResIntPair<WritePOPCNT, [4GM4ALU], 1, [1], 1, 4, 0>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

// BMI1 BEXTR, BMI2 BZHI
defm : 4GM4WriteResIntPair<WriteBEXTR, [4GM4ALU], 1, [1], 1, 4, 1>;
defm : 4GM4WriteResIntPair<WriteBLS, [4GM4ALU], 2, [2], 2, 4, 1>;
defm : 4GM4WriteResIntPair<WriteBZHI, [4GM4ALU], 1>;

// Floating Point Instructions

defm : 4GM4WriteRes<WriteFLD0, [4GM4AGU, 4GM4AGU, 4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM4WriteRes<WriteFLD1, [4GM4AGU, 4GM4AGU, 4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM4WriteRes<WriteFLDC, [4GM4AGU, 4GM4AGU, 4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM4WriteRes<WriteFLoad, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : 4GM4WriteRes<WriteFLoadX, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : 4GM4WriteRes<WriteFLoadY, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : 4GM4WriteRes<WriteFMaskedLoad, [4GM4AGU, 4GM4Load, 4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : 4GM4WriteRes<WriteFMaskedLoadY, [4GM4AGU, 4GM4Load, 4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 1], 2>;

defm : 4GM4WriteRes<WriteFStore, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFStoreX, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFStoreY, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFStoreNT, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : 4GM4WriteRes<WriteFStoreNTX, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : 4GM4WriteRes<WriteFStoreNTY, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : 4GM4WriteRes<WriteFMaskedStore32, [4GM4FPU02, 4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFMaskedStore32Y, [4GM4FPU02, 4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFMaskedStore64, [4GM4FPU02, 4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteFMaskedStore64Y, [4GM4FPU02, 4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;

defm : 4GM4WriteResFPPair<WriteFAdd, [4GM4FPU13], 3>;
defm : 4GM4WriteResFPPair<WriteFAddX, [4GM4FPU13], 3>;
defm : 4GM4WriteResFPPair<WriteFAddY, [4GM4FPU13], 3>;
defm : X86WriteResPairUnsupported<WriteFAddZ>;
defm : 4GM4WriteResFPPair<WriteFAdd64, [4GM4FPU13], 3>;
defm : 4GM4WriteResFPPair<WriteFAdd64X, [4GM4FPU13], 3>;
defm : 4GM4WriteResFPPair<WriteFAdd64Y, [4GM4FPU13], 3>;
defm : X86WriteResPairUnsupported<WriteFAdd64Z>;
defm : 4GM4WriteResFPPair<WriteFHAdd, [4GM4FPU13], 7, [8], 4, 3>;
defm : 4GM4WriteResFPPair<WriteFHAddY, [4GM4FPU13], 7, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WriteFHAddZ>;
defm : 4GM4WriteResFPPair<WritePHAdd, [4GM4FPU13], 3, [8], 4, 3>;
defm : 4GM4WriteResFPPair<WritePHAddX, [4GM4FPU13], 3, [8], 4, 3>;
defm : 4GM4WriteResFPPair<WritePHAddY, [4GM4FPU13], 3, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WritePHAddZ>;
defm : 4GM4WriteResFPPair<WriteFCmp, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFCmpX, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFCmpY, [4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFCmpZ>;
defm : 4GM4WriteResFPPair<WriteFCmp64, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFCmp64X, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFCmp64Y, [4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFCmp64Z>;
defm : 4GM4WriteResFPPair<WriteFCom, [4GM4FPU02,4GM4FPU1], 5>;
defm : 4GM4WriteResFPPair<WriteFComX, [4GM4FPU02,4GM4FPU1], 5>;
defm : 4GM4WriteResFPPair<WriteFBlend, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFBlendY, [4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFBlendZ>;
defm : 4GM4WriteResFPPair<WriteFVarBlend, [4GM4FPU02], 1>;
defm : 4GM4WriteResFPPair<WriteFVarBlendY,[4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFVarBlendZ>;
defm : 4GM4WriteResFPPair<WriteCvtSS2I, [4GM4FPU1, 4GM4FPU1], 2, [1, 1], 2>;
defm : 4GM4WriteResFPPair<WriteCvtPS2I, [4GM4FPU1], 4, [1], 2>;
defm : 4GM4WriteResFPPair<WriteCvtPS2IY, [4GM4FPU1], 4, [1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPS2IZ>;
defm : 4GM4WriteResFPPair<WriteCvtSD2I, [4GM4FPU1, 4GM4FPU1], 2, [1, 2], 2>;
defm : 4GM4WriteResFPPair<WriteCvtPD2I, [4GM4FPU01, 4GM4FPU1], 7, [1, 2], 2>;
defm : 4GM4WriteResFPPair<WriteCvtPD2IY, [4GM4FPU01, 4GM4FPU1], 9, [1, 2], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPD2IZ>;
defm : 4GM4WriteResFPPair<WriteCvtSS2SD, [4GM4FPU1], 4, [1], 1>;
defm : 4GM4WriteResFPPair<WriteCvtPS2PD, [4GM4FPU13], 3, [1], 1>;
defm : 4GM4WriteResFPPair<WriteCvtPS2PDY, [4GM4FPU13], 3, [1], 1>;
defm : X86WriteResPairUnsupported<WriteCvtPS2PDZ>;
defm : 4GM4WriteResFPPair<WriteCvtSD2SS, [4GM4FPU1], 4, [1], 1>;
defm : 4GM4WriteResFPPair<WriteCvtPD2PS, [4GM4FPU1, 4GM4FPU13], 3, [1, 1], 2>;
defm : 4GM4WriteResFPPair<WriteCvtPD2PSY, [4GM4FPU1, 4GM4FPU13], 3, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPD2PSZ>;
defm : 4GM4WriteResFPPair<WriteCvtI2SS, [4GM4ALU2, 4GM4FPU1], 4, [1, 1], 2>;
defm : 4GM4WriteResFPPair<WriteCvtI2PS, [4GM4FPU1], 4>;
defm : 4GM4WriteResFPPair<WriteCvtI2PSY, [4GM4FPU1], 4>;
defm : X86WriteResPairUnsupported<WriteCvtI2PSZ>;
defm : 4GM4WriteResFPPair<WriteCvtI2SD, [4GM4ALU2, 4GM4FPU1], 4, [1, 2], 2>;
defm : 4GM4WriteResFPPair<WriteCvtI2PD, [4GM4FPU01, 4GM4FPU1], 7, [1, 2], 2>;
defm : 4GM4WriteResFPPair<WriteCvtI2PDY, [4GM4FPU01, 4GM4FPU1], 9, [1, 2], 2>;
defm : X86WriteResPairUnsupported<WriteCvtI2PDZ>;
defm : 4GM4WriteResFPPair<WriteCvtPH2PS, [4GM4FPU1], 7, [2], 1>;
defm : 4GM4WriteResFPPair<WriteCvtPH2PSY, [4GM4FPU0], 9, [1], 1>;
defm : X86WriteResPairUnsupported<WriteCvtPH2PSZ>;
defm : 4GM4WriteRes<WriteCvtPS2PH, [4GM4FPU0], 7, [1], 1>;
defm : 4GM4WriteRes<WriteCvtPS2PHY, [4GM4FPU1], 9, [2], 1>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZ>;
defm : 4GM4WriteRes<WriteCvtPS2PHSt, [4GM4FPU0, 4GM4AGU, 4GM4Store], !add(7, C864GM4Model.StoreLatency), [1, 1, 1], 2>;
defm : 4GM4WriteRes<WriteCvtPS2PHYSt, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(9, C864GM4Model.StoreLatency), [2, 1, 1], 3>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZSt>;

defm : 4GM4WriteResFPPair<WriteFDiv, [4GM4FPU1], 10, [5], 1>;
defm : 4GM4WriteResFPPair<WriteFDivX, [4GM4FPU1], 10, [5], 1>;
defm : 4GM4WriteResFPPair<WriteFDivY, [4GM4FPU1], 10, [5], 1>;
defm : X86WriteResPairUnsupported<WriteFDivZ>;
defm : 4GM4WriteResFPPair<WriteFDiv64, [4GM4FPU1], 8, [6], 1>;
defm : 4GM4WriteResFPPair<WriteFDiv64X, [4GM4FPU1], 8, [6], 1>;
defm : 4GM4WriteResFPPair<WriteFDiv64Y, [4GM4FPU1], 8, [6], 1>;
defm : X86WriteResPairUnsupported<WriteFDiv64Z>;
defm : 4GM4WriteResFPPair<WriteFSign, [4GM4FPU13], 1>;
defm : 4GM4WriteResFPPair<WriteFRnd, [4GM4FPU1], 4, [1], 1, C864GM4Model.VecLoadLatency, 0>;
defm : 4GM4WriteResFPPair<WriteFRndY, [4GM4FPU1], 4, [1], 1, C864GM4Model.VecLoadLatency, 0>;
defm : X86WriteResPairUnsupported<WriteFRndZ>;
defm : 4GM4WriteResFPPair<WriteFLogic, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteFLogicY, [4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteFLogicZ>;
defm : 4GM4WriteResFPPair<WriteFTest, [4GM4FPU01, 4GM4FPU1], 4, [1, 2], 1, C864GM4Model.VecLoadLatency, 1>;
defm : 4GM4WriteResFPPair<WriteFTestY, [4GM4FPU01, 4GM4FPU1], 8, [1, 2], 1, C864GM4Model.VecLoadLatency, 1>;
defm : X86WriteResPairUnsupported<WriteFTestZ>;
defm : 4GM4WriteResFPPair<WriteFShuffle, [4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteFShuffleY, [4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteFShuffleZ>;
defm : 4GM4WriteResFPPair<WriteFVarShuffle, [4GM4FPU01], 3>;
defm : 4GM4WriteResFPPair<WriteFVarShuffleY,[4GM4FPU01], 3>;
defm : X86WriteResPairUnsupported<WriteFVarShuffleZ>;
defm : 4GM4WriteResFPPair<WriteFMul, [4GM4FPU02], 3>;
defm : 4GM4WriteResFPPair<WriteFMulX, [4GM4FPU02], 3>;
defm : 4GM4WriteResFPPair<WriteFMulY, [4GM4FPU02], 3>;
defm : X86WriteResPairUnsupported<WriteFMulZ>;
defm : 4GM4WriteResFPPair<WriteFMul64, [4GM4FPU02], 4>;
defm : 4GM4WriteResFPPair<WriteFMul64X, [4GM4FPU02], 4>;
defm : 4GM4WriteResFPPair<WriteFMul64Y, [4GM4FPU02], 4>;
defm : X86WriteResPairUnsupported<WriteFMul64Z>;
defm : 4GM4WriteResFPPair<WriteFMA, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFMAX, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFMAY, [4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFMAZ>;
defm : 4GM4WriteResFPPair<WriteFRcp, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFRcpX, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFRcpY, [4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFRcpZ>;
defm : 4GM4WriteResFPPair<WriteFRsqrt, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFRsqrtX, [4GM4FPU02], 5>;
defm : 4GM4WriteResFPPair<WriteFRsqrtY, [4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFRsqrtZ>;
defm : 4GM4WriteResFPPair<WriteFSqrt, [4GM4FPU1], 8, [7]>;
defm : 4GM4WriteResFPPair<WriteFSqrtX, [4GM4FPU1], 8, [7]>;
defm : 4GM4WriteResFPPair<WriteFSqrtY, [4GM4FPU1], 8, [7]>;
defm : X86WriteResPairUnsupported<WriteFSqrtZ>;
defm : 4GM4WriteResFPPair<WriteFSqrt64, [4GM4FPU1], 8, [10]>;
defm : 4GM4WriteResFPPair<WriteFSqrt64X, [4GM4FPU1], 8, [10]>;
defm : 4GM4WriteResFPPair<WriteFSqrt64Y, [4GM4FPU1], 8, [10]>;
defm : X86WriteResPairUnsupported<WriteFSqrt64Z>;
defm : 4GM4WriteResFPPair<WriteFSqrt80, [4GM4FPU1], 22, [22]>;
defm : 4GM4WriteResFPPair<WriteFShuffle256, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WriteFVarShuffle256, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WriteDPPD, [4GM4FPU1], 10, [8], 4>;

def 4GM4WriteVPERM:SchedWriteRes<[4GM4FPU01]> {
  let Latency = 7;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 3;
}
def : InstRW<[4GM4WriteVPERM], (instrs VPERM2I128rri, VPERM2F128rri)>;

def 4GM4WriteVPERMLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteVPERM.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 3;
}
def : InstRW<[4GM4WriteVPERMLd], (instrs VPERM2I128rmi, VPERM2F128rmi)>;

def 4GM4WriteVPERMDY:SchedWriteRes<[4GM4FPU0]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 3;
}
def : InstRW<[4GM4WriteVPERMDY], (instrs VPERMDYrr)>;

def 4GM4WriteVPERMDYLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU0]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteVPERMDY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteVPERMDYLd], (instrs VPERMDYrm)>;

defm : 4GM4WriteRes<WriteVecLoad, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM4WriteRes<WriteVecLoadX, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM4WriteRes<WriteVecLoadY, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM4WriteRes<WriteVecLoadNT, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM4WriteRes<WriteVecLoadNTY, [4GM4AGU, 4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM4WriteRes<WriteVecMaskedLoad, [4GM4AGU, 4GM4Load, 4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : 4GM4WriteRes<WriteVecMaskedLoadY, [4GM4AGU, 4GM4Load, 4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : 4GM4WriteRes<WriteVecStore, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteVecStoreX, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteVecStoreY, [4GM4AGU, 4GM4Store], C864GM4Model.StoreLatency>;
defm : 4GM4WriteRes<WriteVecStoreNT, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : 4GM4WriteRes<WriteVecStoreNTY, [4GM4FPU1, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : 4GM4WriteRes<WriteVecMaskedStore32, [4GM4FPU02, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : 4GM4WriteRes<WriteVecMaskedStore32Y, [4GM4FPU02, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : 4GM4WriteRes<WriteVecMaskedStore64, [4GM4FPU02, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : 4GM4WriteRes<WriteVecMaskedStore64Y, [4GM4FPU02, 4GM4AGU, 4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : 4GM4WriteRes<WriteVecMoveToGpr, [4GM4FPU1], 1>;
defm : 4GM4WriteRes<WriteVecMoveFromGpr, [4GM4FPU1], 1>;
defm : 4GM4WriteRes<WriteEMMS, [4GM4FPU], 0>;

defm : 4GM4WriteResFPPair<WriteVecShift, [4GM4FPU1], 1>;
defm : 4GM4WriteResFPPair<WriteVecShiftX, [4GM4FPU1], 1>;
defm : 4GM4WriteResFPPair<WriteVecShiftY, [4GM4FPU1], 1>;
defm : X86WriteResPairUnsupported<WriteVecShiftZ>;
defm : 4GM4WriteResFPPair<WriteVecShiftImm, [4GM4FPU1], 1>;
defm : 4GM4WriteResFPPair<WriteVecShiftImmX, [4GM4FPU1], 1>;
defm : 4GM4WriteResFPPair<WriteVecShiftImmY, [4GM4FPU1], 1>;
defm : X86WriteResPairUnsupported<WriteVecShiftImmZ>;
defm : 4GM4WriteResFPPair<WriteVarVecShift, [4GM4FPU0], 3, [2], 1>;
defm : 4GM4WriteResFPPair<WriteVarVecShiftY, [4GM4FPU0], 3, [2], 1>;
defm : X86WriteResPairUnsupported<WriteVarVecShiftZ>;
defm : 4GM4WriteResFPPair<WriteVecLogic, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteVecLogicX, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteVecLogicY, [4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteVecLogicZ>;
defm : 4GM4WriteResFPPair<WriteVecTest, [4GM4FPU01, 4GM4FPU1], 4, [1, 2], 1, 7, 1>;
defm : 4GM4WriteResFPPair<WriteVecTestY, [4GM4FPU01, 4GM4FPU1], 8, [1, 2], 1, 7, 1>;
defm : X86WriteResPairUnsupported<WriteVecTestZ>;
defm : 4GM4WriteResFPPair<WriteVecALU, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteVecALUX, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteVecALUY, [4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteVecALUZ>;
defm : 4GM4WriteResFPPair<WriteVecIMul, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WriteVecIMulX, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WriteVecIMulY, [4GM4FPU0], 3>;
defm : 4GM4WriteRes<WriteCLMul, [4GM4FPU3], 3>;
defm : 4GM4WriteRes<WriteCLMulLd, [4GM4AGU, 4GM4Load, 4GM4FPU3], !add(C864GM4Model.VecLoadLatency, 3)>;
defm : X86WriteResPairUnsupported<WriteVecIMulZ>;
defm : 4GM4WriteResFPPair<WritePMULLD, [4GM4FPU0], 4, [2]>;
defm : 4GM4WriteResFPPair<WritePMULLDY, [4GM4FPU0], 4, [2]>;
defm : X86WriteResPairUnsupported<WritePMULLDZ>;
defm : 4GM4WriteResFPPair<WriteShuffle, [4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteShuffleX, [4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteShuffleY, [4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteShuffleZ>;
defm : 4GM4WriteResFPPair<WriteVarShuffle, [4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteVarShuffleX,[4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteVarShuffleY,[4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteVarShuffleZ>;
defm : 4GM4WriteResFPPair<WriteBlend, [4GM4FPU], 1>;
defm : 4GM4WriteResFPPair<WriteBlendY, [4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteBlendZ>;
defm : 4GM4WriteResFPPair<WriteVarBlend, [4GM4FPU0], 1>;
defm : 4GM4WriteResFPPair<WriteVarBlendY, [4GM4FPU0], 1>;
defm : X86WriteResPairUnsupported<WriteVarBlendZ>;
defm : 4GM4WriteResFPPair<WriteShuffle256, [4GM4FPU01], 3>;
defm : 4GM4WriteResFPPair<WriteVPMOV256, [4GM4FPU01], 1>;
defm : 4GM4WriteResFPPair<WriteVarShuffle256, [4GM4FPU01], 2>;
defm : 4GM4WriteResFPPair<WritePSADBW, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WritePSADBWX, [4GM4FPU0], 3>;
defm : 4GM4WriteResFPPair<WritePSADBWY, [4GM4FPU0], 3>;
defm : X86WriteResPairUnsupported<WritePSADBWZ>;
defm : 4GM4WriteResFPPair<WriteMPSAD, [4GM4FPU], 4, [8], 1>;
defm : 4GM4WriteResFPPair<WriteMPSADY, [4GM4FPU], 4, [8], 1>;
defm : 4GM4WriteResFPPair<WriteMPSADZ, [4GM4FPU], 4, [16], 1>;
defm : 4GM4WriteResFPPair<WritePHMINPOS, [4GM4FPU0], 3>;

// EXTRQ INSERTQ
def 4GM4WriteEXTRQ_INSERTQ:SchedWriteRes<[4GM4FPU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteEXTRQ_INSERTQ], (instrs EXTRQ, EXTRQI, INSERTQ, INSERTQI)>;

def 4GM4WriteVPMOVY:SchedWriteRes<[4GM4FPU01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteVPMOVY], (instregex "VPMOV(S|Z)X(BD|BQ|BW|DQ|WD|WQ)Yrr")>;

def 4GM4WriteVPMOVYLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteVPMOVY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteVPMOVYLd], (instregex "VPMOV(S|Z)X(BD|BQ|BW|DQ|WD|WQ)Yrm")>;

def 4GM4WriteVPS:SchedWriteRes<[4GM4FPU1]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteVPS], (instregex "VPS(LL|RA|RL)(D|Q|W)Yrr")>;

def 4GM4WriteVPSLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU1]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteVPS.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteVPSLd], (instregex "VPS(LL|RA|RL)(D|Q|W)Yrm")>;

def 4GM4WritePMADDUBSW_PMULHRSW:SchedWriteRes<[4GM4FPU0]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WritePMADDUBSW_PMULHRSW], (instregex "(V?)PMADDUBSW(Y)?rr",
                                                       "(V?)PMULHRSW(Y)?rr")>;

def 4GM4WritePMADDUBSW_PMULHRSWLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU0]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WritePMADDUBSW_PMULHRSW.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WritePMADDUBSW_PMULHRSWLd], (instregex "(V?)PMADDUBSW(Y)?rm",
                                                         "(V?)PMULHRSW(Y)?rm")>;

// Vector insert/extract operations.
defm : 4GM4WriteResFPPair<WriteVecInsert, [4GM4FPU01], 1>;
defm : 4GM4WriteRes<WriteVecExtract, [4GM4FPU01, 4GM4FPU1], 2, [1, 2], 2>;
defm : 4GM4WriteRes<WriteVecExtractSt, [4GM4FPU01, 4GM4FPU1, 4GM4AGU, 4GM4Store], 2, [1, 1, 1, 2], 2>;

def 4GM4WritePINSR:SchedWriteRes<[4GM4FPU01]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WritePINSR], (instregex "(V?)PINSR(B|D|Q|W)rri")>;

def 4GM4WritePINSRLd:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WritePINSR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WritePINSRLd], (instregex "(V?)PINSR(B|D|Q|W)rmi")>;


// MOVMSK Instructions.
defm : 4GM4WriteRes<WriteFMOVMSK, [4GM4FPU1], 1>;
defm : 4GM4WriteRes<WriteMMXMOVMSK, [4GM4FPU1], 1>;
defm : 4GM4WriteRes<WriteVecMOVMSK, [4GM4FPU1], 1>;
defm : 4GM4WriteRes<WriteVecMOVMSKY, [4GM4FPU1], 1>;

// Strings instructions.
defm : 4GM4WriteResFPPair<WritePCmpIStrM, [4GM4FPU], 7, [25], 3>;
defm : 4GM4WriteResFPPair<WritePCmpEStrM, [4GM4FPU], 8, [25], 8>;
defm : 4GM4WriteResFPPair<WritePCmpIStrI, [4GM4FPU], 1, [8], 2>;
defm : 4GM4WriteResFPPair<WritePCmpEStrI, [4GM4FPU], 2, [13], 8>;

// AES Instructions.
defm : 4GM4WriteResFPPair<WriteAESDecEnc, [4GM4FPU23], 4>;
defm : 4GM4WriteResFPPair<WriteAESIMC, [4GM4FPU23], 4>;
defm : 4GM4WriteResFPPair<WriteAESKeyGen, [4GM4FPU23], 4>;

defm : 4GM4WriteRes<WriteFence, [4GM4AGU]>;
defm : 4GM4WriteRes<WriteNop, []>;

// SHA Instructions.

def 4GM4WriteSHA1:SchedWriteRes<[4GM4FPU2]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA1], (instrs SHA1MSG1rr, SHA1MSG2rr, SHA1NEXTErr)>;

def 4GM4WriteSHA1Ld:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteSHA1.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA1Ld], (instrs SHA1MSG1rm, SHA1MSG2rm, SHA1NEXTErm)>;

def 4GM4WriteSHA1RNDS4rri:SchedWriteRes<[4GM4FPU2]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA1RNDS4rri], (instrs SHA1RNDS4rri)>;

def 4GM4WriteSHA1RNDS4rmi:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteSHA1RNDS4rri.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA1RNDS4rmi], (instrs SHA1RNDS4rmi)>;

def 4GM4WriteSHA256MSG1rr:SchedWriteRes<[4GM4FPU2]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA256MSG1rr], (instrs SHA256MSG1rr)>;

def 4GM4WriteSHA256MSG1rm:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteSHA256MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA256MSG1rm], (instrs SHA256MSG1rm)>;

def 4GM4WriteSHA256:SchedWriteRes<[4GM4FPU2]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA256], (instrs SHA256MSG2rr, SHA256RNDS2rr)>;

def 4GM4WriteSHA256Ld:SchedWriteRes<[4GM4AGU, 4GM4Load, 4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, 4GM4WriteSHA256.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteSHA256Ld], (instrs SHA256MSG2rm, SHA256RNDS2rm)>;

//===----------------------------------------------------------------------===//
// Zero Cycle Move
//===----------------------------------------------------------------------===//

def 4GM4WriteZeroLatency : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}
def : InstRW<[4GM4WriteZeroLatency], (instrs MOV32rr, MOV32rr_REV,
                                             MOV64rr, MOV64rr_REV,
                                             MOVSX32rr32)>;

def 4GM4WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[4GM4WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                               XCHG64rr, XCHG64ar)>;

defm : 4GM4WriteRes<WriteXCHG, [4GM4ALU], 1, [2], 1>;

defm : 4GM4WriteRes<WriteFMoveX, [], 0, [], 1>;
defm : 4GM4WriteRes<WriteFMoveY, [], 0, [], 1>;
defm : X86WriteResUnsupported<WriteFMoveZ>;

defm : 4GM4WriteRes<WriteVecMove, [], 0, [], 1>;
defm : 4GM4WriteRes<WriteVecMoveX, [], 0, [], 1>;
defm : 4GM4WriteRes<WriteVecMoveY, [], 0, [], 1>;
defm : X86WriteResUnsupported<WriteVecMoveZ>;

def : IsOptimizableRegisterMove<[
  InstructionEquivalenceClass<[
    // GPR variants.
    MOV32rr, MOV32rr_REV,
    MOV64rr, MOV64rr_REV,
    MOVSX32rr32,
    XCHG32rr, XCHG32ar,
    XCHG64rr, XCHG64ar,

    // MMX variants.
    // MMX moves are *NOT* eliminated.

    // SSE variants.
    MOVAPSrr, MOVAPSrr_REV,
    MOVUPSrr, MOVUPSrr_REV,
    MOVAPDrr, MOVAPDrr_REV,
    MOVUPDrr, MOVUPDrr_REV,
    MOVDQArr, MOVDQArr_REV,
    MOVDQUrr, MOVDQUrr_REV,

    // AVX variants.
    VMOVAPSrr, VMOVAPSrr_REV,
    VMOVUPSrr, VMOVUPSrr_REV,
    VMOVAPDrr, VMOVAPDrr_REV,
    VMOVUPDrr, VMOVUPDrr_REV,
    VMOVDQArr, VMOVDQArr_REV,
    VMOVDQUrr, VMOVDQUrr_REV,

    // AVX YMM variants.
    VMOVAPSYrr, VMOVAPSYrr_REV,
    VMOVUPSYrr, VMOVUPSYrr_REV,
    VMOVAPDYrr, VMOVAPDYrr_REV,
    VMOVUPDYrr, VMOVUPDYrr_REV,
    VMOVDQAYrr, VMOVDQAYrr_REV,
    VMOVDQUYrr, VMOVDQUYrr_REV,
  ], TruePred >
]>;

//===----------------------------------------------------------------------===//
// Dependency breaking instructions.
//===----------------------------------------------------------------------===//

def 4GM4WriteZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteALU]>
]>;
def : InstRW<[4GM4WriteZeroIdiom], (instrs XOR32rr, XOR32rr_REV,
                                           XOR64rr, XOR64rr_REV,
                                           SUB32rr, SUB32rr_REV,
                                           SUB64rr, SUB64rr_REV)>;

def 4GM4WriteZeroIdiomEFLAGS : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<CheckSameRegOperand<0, 1>>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                                 [WriteALU]>
]>;
def : InstRW<[4GM4WriteZeroIdiomEFLAGS], (instrs CMP8rr,  CMP8rr_REV,
                                                 CMP16rr, CMP16rr_REV,
                                                 CMP32rr, CMP32rr_REV,
                                                 CMP64rr, CMP64rr_REV)>;

def 4GM4WriteFZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogic]>
]>;

def : InstRW<[4GM4WriteFZeroIdiom], (instrs VXORPSrr, VXORPDrr,
                                            VANDNPSrr, VANDNPDrr)>;

def 4GM4WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[4GM4WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                             VANDNPSYrr, VANDNPDYrr)>;

def 4GM4WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;

def : InstRW<[4GM4WriteVZeroIdiomLogicX], (instrs VPXORrr, VPANDNrr)>;

def 4GM4WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[4GM4WriteVZeroIdiomLogicY], (instrs VPXORYrr, VPANDNYrr)>;

def 4GM4WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;


def : InstRW<[4GM4WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr)>;

def 4GM4WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[4GM4WriteVZeroIdiomALUY],
             (instrs VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
                     VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr)>;

def : IsZeroIdiomFunction<[
  // GPR Zero-idioms.
  DepBreakingClass<[ XOR32rr, XOR32rr_REV,
                     XOR64rr, XOR64rr_REV,
                     SUB32rr, SUB32rr_REV,
                     SUB64rr, SUB64rr_REV ], ZeroIdiomPredicate>,

  // SSE XMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    XORPSrr, XORPDrr,
    ANDNPSrr, ANDNPDrr,

    // int variants.
    PXORrr,
    PANDNrr,
    PSUBBrr, PSUBWrr, PSUBDrr, PSUBQrr,
    PSUBSBrr, PSUBSWrr,
    PSUBUSBrr, PSUBUSWrr,
    PCMPGTBrr, PCMPGTWrr, PCMPGTDrr, PCMPGTQrr
  ], ZeroIdiomPredicate>,

  // AVX XMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSrr, VXORPDrr,
    VANDNPSrr, VANDNPDrr,

    // int variants.
    VPXORrr,
    VPANDNrr,
    VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
    VPSUBSBrr, VPSUBSWrr,
    VPSUBUSBrr, VPSUBUSWrr,
    VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr,
  ], ZeroIdiomPredicate>,

  // AVX YMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSYrr, VXORPDYrr,
    VANDNPSYrr, VANDNPDYrr,

    // int variants.
    VPXORYrr,
    VPANDNYrr,
    VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
    VPSUBSBYrr, VPSUBSWYrr,
    VPSUBUSBYrr, VPSUBUSWYrr,
    VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr
  ], ZeroIdiomPredicate>,
]>;

def : IsDepBreakingFunction<[
  // GPR
  DepBreakingClass<[ SBB32rr, SBB32rr_REV,
                     SBB64rr, SBB64rr_REV ], ZeroIdiomPredicate>,
  DepBreakingClass<[ CMP8rr,  CMP8rr_REV,
                     CMP16rr, CMP16rr_REV,
                     CMP32rr, CMP32rr_REV,
                     CMP64rr, CMP64rr_REV ], CheckSameRegOperand<0, 1> >,

  // MMX
  DepBreakingClass<[
    MMX_PCMPEQBrr, MMX_PCMPEQWrr, MMX_PCMPEQDrr
  ], ZeroIdiomPredicate>,

  // SSE
  DepBreakingClass<[
    PCMPEQBrr, PCMPEQWrr, PCMPEQDrr, PCMPEQQrr
  ], ZeroIdiomPredicate>,

  // AVX XMM
  DepBreakingClass<[
    VPCMPEQBrr, VPCMPEQWrr, VPCMPEQDrr, VPCMPEQQrr
  ], ZeroIdiomPredicate>,

  // AVX YMM
  DepBreakingClass<[
    VPCMPEQBYrr, VPCMPEQWYrr, VPCMPEQDYrr, VPCMPEQQYrr
  ], ZeroIdiomPredicate>,
]>;

} // SchedModel