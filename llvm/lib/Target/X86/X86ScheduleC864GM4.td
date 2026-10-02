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
def C4GM4ALU0 : ProcResource<1>;
def C4GM4ALU1 : ProcResource<1>;
def C4GM4ALU2 : ProcResource<1>;
def C4GM4ALU3 : ProcResource<1>;

def C4GM4ALU23  : ProcResGroup<[C4GM4ALU2, C4GM4ALU3]>;

// BRU on ALU0 and ALU3.
defvar C4GM4BRU0 = C4GM4ALU0;
defvar C4GM4BRU1 = C4GM4ALU3;

def C4GM4BRU01 : ProcResGroup<[C4GM4BRU0, C4GM4BRU1]>;

// 72 Entry (4x18 entries) integer Scheduler.
def C4GM4ALU : ProcResGroup<[C4GM4ALU0, C4GM4BRU0,
                             C4GM4ALU1,
                             C4GM4ALU2,
                             C4GM4ALU3, C4GM4BRU1]> {
  let BufferSize = 72;
}

// The integer physical register file consists of 192 registers.
def C4GM4IntegerPRF : RegisterFile<192,
                                   [GR64, CCR],
                                   [1, 1],
                                   [1, 0],
                                   6, // Max moves that can be eliminated per cycle.
                                   0>; // Restrict move elimination to zero regs.

// AGU
// The C864GM4 has 3 AGUs.
def C4GM4AGU0 : ProcResource<1>;
def C4GM4AGU1 : ProcResource<1>;
def C4GM4AGU2 : ProcResource<1>;
// 28 Entry (1x28 entries) AGU group.
def C4GM4AGU : ProcResGroup<[C4GM4AGU0, C4GM4AGU1, C4GM4AGU2]> {
  let BufferSize = 28;
}

//===----------------------------------------------------------------------===//
// Floating-Point Unit
//===----------------------------------------------------------------------===//

// The C864GM4 has 4 FPUs.
def C4GM4FPU0 : ProcResource<1>;
def C4GM4FPU1 : ProcResource<1>;
def C4GM4FPU2 : ProcResource<1>;
def C4GM4FPU3 : ProcResource<1>;

def C4GM4FPU01  : ProcResGroup<[C4GM4FPU0, C4GM4FPU1]>;
def C4GM4FPU13  : ProcResGroup<[C4GM4FPU1, C4GM4FPU3]>;
def C4GM4FPU23  : ProcResGroup<[C4GM4FPU2, C4GM4FPU3]>;
def C4GM4FPU02  : ProcResGroup<[C4GM4FPU0, C4GM4FPU2]>;

// 48 Entry (4x12 entries) floating-point Scheduler.
def C4GM4FPU : ProcResGroup<[C4GM4FPU0, C4GM4FPU1, C4GM4FPU2, C4GM4FPU3]> {
  let BufferSize = 48;
}

// The floating point physical register file consists of 176 registers.
def C4GM4FpuPRF : RegisterFile<176,
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
def C4GM4LSU : ProcResource<3>;

// The two of LSU can be loads.
let Super = C4GM4LSU in
def C4GM4Load : ProcResource<2> {
  // The LDQ is 50.
  let BufferSize = 50;
}
def C4GM4LoadQueue : LoadQueue<C4GM4Load>;

// All three of LSU can be loads.
let Super = C4GM4LSU in
def C4GM4Store : ProcResource<3> {
  // The STQ is 52.
  let BufferSize = 52;
}
def C4GM4StoreQueue : StoreQueue<C4GM4Store>;

def : ReadAdvance<ReadAfterLd, C864GM4Model.LoadLatency>;
def : ReadAdvance<ReadAfterVecLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecXLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecYLd, C864GM4Model.VecLoadLatency>;
def : ReadAdvance<ReadInt2Fpu, 0>;

//===----------------------------------------------------------------------===//
// Retire Control Unit
//===----------------------------------------------------------------------===//

def C4GM4RCU : RetireControlUnit<C864GM4Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Basic helper classes.
//===----------------------------------------------------------------------===//

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.


multiclass __C4GM4WriteRes<SchedWrite SchedRW, list<ProcResourceKind> ExePorts,
                         int Lat = 1, list<int> Res = [], int UOps = 1> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }
}

multiclass __C4GM4WriteResPair<X86FoldableSchedWrite SchedRW,
                             list<ProcResourceKind> ExePorts, int Lat,
                             list<int> Res, int UOps, int LoadLat, int LoadUOps,
                             ProcResourceKind AGU, int LoadRes> {
  defm : __C4GM4WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;

  defm : __C4GM4WriteRes<SchedRW.Folded,
                       !listconcat([AGU, C4GM4Load], ExePorts),
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
multiclass C4GM4WriteRes<SchedWrite SchedRW,
                        list<ProcResourceKind> ExePorts, int Lat = 1,
                        list<int> Res = [], int UOps = 1> {
  defm : __C4GM4WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

// For classes with folded loads.
multiclass C4GM4WriteResIntPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __C4GM4WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM4Model.LoadLatency,
                           LoadUOps, C4GM4AGU, LoadRes>;
}

multiclass C4GM4WriteResFPPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __C4GM4WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM4Model.VecLoadLatency,
                           LoadUOps, C4GM4AGU, LoadRes>;
}

// Microcoded Instructions
def C4GM4WriteMicrocoded : SchedWriteRes<[]> {
  let Latency = 100;
}

def : SchedAlias<WriteMicrocoded, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteSystem, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteFCMOV, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteLDMXCSR, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteSTMXCSR, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPS, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSY, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSLd, C4GM4WriteMicrocoded>;
def : SchedAlias<WriteDPPSYLd,C4GM4WriteMicrocoded>;

def C4GM4WritePDEP_PEXT : SchedWriteRes<[]>;
def : InstRW<[C4GM4WritePDEP_PEXT], (instregex "PDEP(32|64)rr",
                                               "PEXT(32|64)rr")>;
def : SchedAlias<C4GM4WritePDEP_PEXT, C4GM4WriteMicrocoded>;
def C4GM4WritePDEP_PEXTLd : SchedWriteRes<[]>;
def : InstRW<[C4GM4WritePDEP_PEXTLd], (instregex "PDEP(32|64)rm",
                                                 "PEXT(32|64)rm")>;
def : SchedAlias<C4GM4WritePDEP_PEXTLd, C4GM4WriteMicrocoded>;

// Integer Instructions

defm : C4GM4WriteRes<WriteRMW, [C4GM4AGU, C4GM4Store]>;

defm : C4GM4WriteRes<WriteStore, [C4GM4AGU, C4GM4Store]>;
defm : C4GM4WriteRes<WriteStoreNT, [C4GM4AGU, C4GM4Store]>;
defm : C4GM4WriteRes<WriteMove, [C4GM4ALU]>;
defm : C4GM4WriteRes<WriteLoad, [C4GM4AGU, C4GM4Load], !add(1, C864GM4Model.LoadLatency)>;

defm : C4GM4WriteRes<WriteVecMaskedGatherWriteback, [], !add(1, C864GM4Model.VecLoadLatency)>;

defm : C4GM4WriteRes<WriteZero, [], 0, [], 1>;
defm : C4GM4WriteRes<WriteLEA, [C4GM4ALU]>;
defm : C4GM4WriteResIntPair<WriteALU, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteADC, [C4GM4ALU], 1>;

// This write is used for slow LEA instructions.
def C4GM4Write3OpsLEA : SchedWriteRes<[C4GM4ALU23]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
// a slow LEA is either a 3Ops LEA (base, index, offset),
// or an LEA with a `Scale` value different than 1.
def C4GM4SlowLEAPredicate : MCSchedPredicate<
  CheckAny<[
    IsThreeOperandsLEAFn,
    CheckAll<[
      CheckIsImmOperand<2>,
      CheckNot<CheckImmOperand<2, 1>>
    ]>
  ]>
>;

def C4GM4WriteLEA : SchedWriteVariant<[
    SchedVar<C4GM4SlowLEAPredicate, [C4GM4Write3OpsLEA]>,
    SchedVar<NoSchedPred,           [WriteLEA]>
]>;

def : InstRW<[C4GM4WriteLEA], (instrs LEA64r, LEA64_32r)>;

// MUL IMUL MULX
defm : C4GM4WriteResIntPair<WriteIMul8, [C4GM4ALU1], 3, [3], 1>;
defm : C4GM4WriteResIntPair<WriteIMul16, [C4GM4ALU1], 3, [3], 1>;
defm : C4GM4WriteResIntPair<WriteIMul16Imm, [C4GM4ALU1], 4, [4], 2>;
defm : C4GM4WriteResIntPair<WriteIMul16Reg, [C4GM4ALU1], 3>;
defm : C4GM4WriteResIntPair<WriteIMul32, [C4GM4ALU1], 3, [3], 1>;
defm : C4GM4WriteResIntPair<WriteIMul32Imm, [C4GM4ALU1], 3, [3], 2>;
defm : C4GM4WriteResIntPair<WriteIMul32Reg, [C4GM4ALU1], 3>;
defm : C4GM4WriteResIntPair<WriteIMul64, [C4GM4ALU1], 3, [3], 1>;
defm : C4GM4WriteResIntPair<WriteIMul64Imm, [C4GM4ALU1], 3, [3], 2>;
defm : C4GM4WriteResIntPair<WriteIMul64Reg, [C4GM4ALU1], 3>;

defm : C4GM4WriteResIntPair<WriteMULX32, [C4GM4ALU1], 3>;
defm : C4GM4WriteResIntPair<WriteMULX64, [C4GM4ALU1], 3>;
defm : C4GM4WriteRes<WriteIMulHLd, [C4GM4ALU1], !add(3, C864GM4Model.LoadLatency), [], 0>;
defm : C4GM4WriteRes<WriteIMulH, [C4GM4ALU1], 3, [], 0>;

def C4GM4WriteMULX:SchedWriteRes<[C4GM4ALU1]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteMULX], (instregex "MULX(32|64)rr")>;

def C4GM4WriteMULXLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4ALU1]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteMULX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteMULXLd], (instregex "MULX(32|64)rm")>;

// DIV IDIV
defm : C4GM4WriteResIntPair<WriteDiv8, [C4GM4ALU2], 12, [12], 2>;
defm : C4GM4WriteResIntPair<WriteDiv16, [C4GM4ALU2], 17, [17], 2>;
defm : C4GM4WriteResIntPair<WriteDiv32, [C4GM4ALU2], 25, [25], 2>;
defm : C4GM4WriteResIntPair<WriteDiv64, [C4GM4ALU2], 41, [41], 1>;
defm : C4GM4WriteResIntPair<WriteIDiv8, [C4GM4ALU2], 12, [12], 2>;
defm : C4GM4WriteResIntPair<WriteIDiv16, [C4GM4ALU2], 17, [17], 2>;
defm : C4GM4WriteResIntPair<WriteIDiv32, [C4GM4ALU2], 25, [25], 2>;
defm : C4GM4WriteResIntPair<WriteIDiv64, [C4GM4ALU2], 41, [41], 1>;

defm : C4GM4WriteRes<WriteBSWAP32, [C4GM4ALU], 1, [4], 1>;
defm : C4GM4WriteRes<WriteBSWAP64, [C4GM4ALU], 1, [4], 1>;
defm : C4GM4WriteRes<WriteCMPXCHG, [C4GM4ALU], 3>;
defm : C4GM4WriteRes<WriteCMPXCHGRMW,[C4GM4ALU, C4GM4AGU], 5, [1,1], 3>;

def C4GM4WriteXCHGrm : SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4ALU]> {
  let Latency = !add(C864GM4Model.LoadLatency, 1);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteXCHGrm], (instregex "XCHG(8|16|32|64)rm")>;

defm : C4GM4WriteResIntPair<WriteShift, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteShiftCL, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteRotate, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteRotateCL, [C4GM4ALU], 1>;

def C4GM4WriteRCR:SchedWriteRes<[C4GM4ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteRCR], (instregex "RCR(8|16|32|64)rCL")>;

def C4GM4WriteRCRLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4ALU]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteRCR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteRCRLd], (instregex "RCR(8|16|32|64)mCL")>;

def C4GM4WriteRCL:SchedWriteRes<[C4GM4ALU]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteRCL], (instregex "RCL(8|16|32|64)rCL")>;

def C4GM4WriteRCLLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4ALU]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteRCL.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteRCLLd], (instregex "RCL(8|16|32|64)mCL")>;


defm : C4GM4WriteRes<WriteSHDrri, [C4GM4ALU], 3, [3], 6>;
defm : C4GM4WriteRes<WriteSHDrrcl, [C4GM4ALU], 3, [5], 7>;
defm : C4GM4WriteRes<WriteSHDmri, [C4GM4ALU], 5, [5], 8>;
defm : C4GM4WriteRes<WriteSHDmrcl, [C4GM4ALU], 5, [5], 8>;

defm : C4GM4WriteResIntPair<WriteJump, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteCRC32, [C4GM4ALU3], 3, [3], 3>;

defm : C4GM4WriteResIntPair<WriteCMOV, [C4GM4ALU], 1>;
defm : C4GM4WriteRes<WriteSETCC, [C4GM4ALU]>;
defm : C4GM4WriteRes<WriteSETCCStore, [C4GM4ALU, C4GM4AGU, C4GM4Store]>;
defm : C4GM4WriteRes<WriteLAHFSAHF, [C4GM4ALU], 2, [1], 2>;

defm : C4GM4WriteRes<WriteBitTest, [C4GM4ALU], 1>;
defm : C4GM4WriteRes<WriteBitTestImmLd, [C4GM4AGU, C4GM4Load, C4GM4ALU], !add(C864GM4Model.LoadLatency, 1), [1,1,1], 2>;
defm : C4GM4WriteRes<WriteBitTestRegLd, [C4GM4AGU, C4GM4Load, C4GM4ALU], !add(C864GM4Model.LoadLatency, 1), [1,1,1], 7>;
defm : C4GM4WriteRes<WriteBitTestSet, [C4GM4ALU], 2, [1], 2>;
defm : C4GM4WriteRes<WriteBitTestSetImmLd, [C4GM4AGU, C4GM4Load, C4GM4ALU], !add(C864GM4Model.LoadLatency, 2), [1, 1, 1], 2>;
defm : C4GM4WriteRes<WriteBitTestSetRegLd, [C4GM4AGU, C4GM4Load, C4GM4ALU], !add(C864GM4Model.LoadLatency, 2), [1, 1, 1], 7>;

// Bit counts.
defm : C4GM4WriteResIntPair<WriteBSF, [C4GM4ALU], 3, [12], 6, 4, 2>;
defm : C4GM4WriteResIntPair<WriteBSR, [C4GM4ALU], 4, [16], 6, 4, 2>;
defm : C4GM4WriteResIntPair<WriteLZCNT, [C4GM4ALU], 1>;
defm : C4GM4WriteResIntPair<WriteTZCNT, [C4GM4ALU], 2, [2], 2, 4, 0>;
defm : C4GM4WriteResIntPair<WritePOPCNT, [C4GM4ALU], 1, [1], 1, 4, 0>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

// BMI1 BEXTR, BMI2 BZHI
defm : C4GM4WriteResIntPair<WriteBEXTR, [C4GM4ALU], 1, [1], 1, 4, 1>;
defm : C4GM4WriteResIntPair<WriteBLS, [C4GM4ALU], 2, [2], 2, 4, 1>;
defm : C4GM4WriteResIntPair<WriteBZHI, [C4GM4ALU], 1>;

// Floating Point Instructions

defm : C4GM4WriteRes<WriteFLD0, [C4GM4AGU, C4GM4AGU, C4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM4WriteRes<WriteFLD1, [C4GM4AGU, C4GM4AGU, C4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM4WriteRes<WriteFLDC, [C4GM4AGU, C4GM4AGU, C4GM4FPU1], !add(C864GM4Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM4WriteRes<WriteFLoad, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : C4GM4WriteRes<WriteFLoadX, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : C4GM4WriteRes<WriteFLoadY, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : C4GM4WriteRes<WriteFMaskedLoad, [C4GM4AGU, C4GM4Load, C4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1)>;
defm : C4GM4WriteRes<WriteFMaskedLoadY, [C4GM4AGU, C4GM4Load, C4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 1], 2>;

defm : C4GM4WriteRes<WriteFStore, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFStoreX, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFStoreY, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFStoreNT, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : C4GM4WriteRes<WriteFStoreNTX, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : C4GM4WriteRes<WriteFStoreNTY, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : C4GM4WriteRes<WriteFMaskedStore32, [C4GM4FPU02, C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFMaskedStore32Y, [C4GM4FPU02, C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFMaskedStore64, [C4GM4FPU02, C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteFMaskedStore64Y, [C4GM4FPU02, C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;

defm : C4GM4WriteResFPPair<WriteFAdd, [C4GM4FPU13], 3>;
defm : C4GM4WriteResFPPair<WriteFAddX, [C4GM4FPU13], 3>;
defm : C4GM4WriteResFPPair<WriteFAddY, [C4GM4FPU13], 3>;
defm : X86WriteResPairUnsupported<WriteFAddZ>;
defm : C4GM4WriteResFPPair<WriteFAdd64, [C4GM4FPU13], 3>;
defm : C4GM4WriteResFPPair<WriteFAdd64X, [C4GM4FPU13], 3>;
defm : C4GM4WriteResFPPair<WriteFAdd64Y, [C4GM4FPU13], 3>;
defm : X86WriteResPairUnsupported<WriteFAdd64Z>;
defm : C4GM4WriteResFPPair<WriteFHAdd, [C4GM4FPU13], 7, [8], 4, 3>;
defm : C4GM4WriteResFPPair<WriteFHAddY, [C4GM4FPU13], 7, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WriteFHAddZ>;
defm : C4GM4WriteResFPPair<WritePHAdd, [C4GM4FPU13], 3, [8], 4, 3>;
defm : C4GM4WriteResFPPair<WritePHAddX, [C4GM4FPU13], 3, [8], 4, 3>;
defm : C4GM4WriteResFPPair<WritePHAddY, [C4GM4FPU13], 3, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WritePHAddZ>;
defm : C4GM4WriteResFPPair<WriteFCmp, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFCmpX, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFCmpY, [C4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFCmpZ>;
defm : C4GM4WriteResFPPair<WriteFCmp64, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFCmp64X, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFCmp64Y, [C4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFCmp64Z>;
defm : C4GM4WriteResFPPair<WriteFCom, [C4GM4FPU02,C4GM4FPU1], 5>;
defm : C4GM4WriteResFPPair<WriteFComX, [C4GM4FPU02,C4GM4FPU1], 5>;
defm : C4GM4WriteResFPPair<WriteFBlend, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFBlendY, [C4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFBlendZ>;
defm : C4GM4WriteResFPPair<WriteFVarBlend, [C4GM4FPU02], 1>;
defm : C4GM4WriteResFPPair<WriteFVarBlendY,[C4GM4FPU02], 1>;
defm : X86WriteResPairUnsupported<WriteFVarBlendZ>;
defm : C4GM4WriteResFPPair<WriteCvtSS2I, [C4GM4FPU1, C4GM4FPU1], 2, [1, 1], 2>;
defm : C4GM4WriteResFPPair<WriteCvtPS2I, [C4GM4FPU1], 4, [1], 2>;
defm : C4GM4WriteResFPPair<WriteCvtPS2IY, [C4GM4FPU1], 4, [1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPS2IZ>;
defm : C4GM4WriteResFPPair<WriteCvtSD2I, [C4GM4FPU1, C4GM4FPU1], 2, [1, 2], 2>;
defm : C4GM4WriteResFPPair<WriteCvtPD2I, [C4GM4FPU01, C4GM4FPU1], 7, [1, 2], 2>;
defm : C4GM4WriteResFPPair<WriteCvtPD2IY, [C4GM4FPU01, C4GM4FPU1], 9, [1, 2], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPD2IZ>;
defm : C4GM4WriteResFPPair<WriteCvtSS2SD, [C4GM4FPU1], 4, [1], 1>;
defm : C4GM4WriteResFPPair<WriteCvtPS2PD, [C4GM4FPU13], 3, [1], 1>;
defm : C4GM4WriteResFPPair<WriteCvtPS2PDY, [C4GM4FPU13], 3, [1], 1>;
defm : X86WriteResPairUnsupported<WriteCvtPS2PDZ>;
defm : C4GM4WriteResFPPair<WriteCvtSD2SS, [C4GM4FPU1], 4, [1], 1>;
defm : C4GM4WriteResFPPair<WriteCvtPD2PS, [C4GM4FPU1, C4GM4FPU13], 3, [1, 1], 2>;
defm : C4GM4WriteResFPPair<WriteCvtPD2PSY, [C4GM4FPU1, C4GM4FPU13], 3, [1, 1], 2>;
defm : X86WriteResPairUnsupported<WriteCvtPD2PSZ>;
defm : C4GM4WriteResFPPair<WriteCvtI2SS, [C4GM4ALU2, C4GM4FPU1], 4, [1, 1], 2>;
defm : C4GM4WriteResFPPair<WriteCvtI2PS, [C4GM4FPU1], 4>;
defm : C4GM4WriteResFPPair<WriteCvtI2PSY, [C4GM4FPU1], 4>;
defm : X86WriteResPairUnsupported<WriteCvtI2PSZ>;
defm : C4GM4WriteResFPPair<WriteCvtI2SD, [C4GM4ALU2, C4GM4FPU1], 4, [1, 2], 2>;
defm : C4GM4WriteResFPPair<WriteCvtI2PD, [C4GM4FPU01, C4GM4FPU1], 7, [1, 2], 2>;
defm : C4GM4WriteResFPPair<WriteCvtI2PDY, [C4GM4FPU01, C4GM4FPU1], 9, [1, 2], 2>;
defm : X86WriteResPairUnsupported<WriteCvtI2PDZ>;
defm : C4GM4WriteResFPPair<WriteCvtPH2PS, [C4GM4FPU1], 7, [2], 1>;
defm : C4GM4WriteResFPPair<WriteCvtPH2PSY, [C4GM4FPU0], 9, [1], 1>;
defm : X86WriteResPairUnsupported<WriteCvtPH2PSZ>;
defm : C4GM4WriteRes<WriteCvtPS2PH, [C4GM4FPU0], 7, [1], 1>;
defm : C4GM4WriteRes<WriteCvtPS2PHY, [C4GM4FPU1], 9, [2], 1>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZ>;
defm : C4GM4WriteRes<WriteCvtPS2PHSt, [C4GM4FPU0, C4GM4AGU, C4GM4Store], !add(7, C864GM4Model.StoreLatency), [1, 1, 1], 2>;
defm : C4GM4WriteRes<WriteCvtPS2PHYSt, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(9, C864GM4Model.StoreLatency), [2, 1, 1], 3>;
defm : X86WriteResUnsupported<WriteCvtPS2PHZSt>;

defm : C4GM4WriteResFPPair<WriteFDiv, [C4GM4FPU1], 10, [5], 1>;
defm : C4GM4WriteResFPPair<WriteFDivX, [C4GM4FPU1], 10, [5], 1>;
defm : C4GM4WriteResFPPair<WriteFDivY, [C4GM4FPU1], 10, [5], 1>;
defm : X86WriteResPairUnsupported<WriteFDivZ>;
defm : C4GM4WriteResFPPair<WriteFDiv64, [C4GM4FPU1], 8, [6], 1>;
defm : C4GM4WriteResFPPair<WriteFDiv64X, [C4GM4FPU1], 8, [6], 1>;
defm : C4GM4WriteResFPPair<WriteFDiv64Y, [C4GM4FPU1], 8, [6], 1>;
defm : X86WriteResPairUnsupported<WriteFDiv64Z>;
defm : C4GM4WriteResFPPair<WriteFSign, [C4GM4FPU13], 1>;
defm : C4GM4WriteResFPPair<WriteFRnd, [C4GM4FPU1], 4, [1], 1, C864GM4Model.VecLoadLatency, 0>;
defm : C4GM4WriteResFPPair<WriteFRndY, [C4GM4FPU1], 4, [1], 1, C864GM4Model.VecLoadLatency, 0>;
defm : X86WriteResPairUnsupported<WriteFRndZ>;
defm : C4GM4WriteResFPPair<WriteFLogic, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteFLogicY, [C4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteFLogicZ>;
defm : C4GM4WriteResFPPair<WriteFTest, [C4GM4FPU01, C4GM4FPU1], 4, [1, 2], 1, C864GM4Model.VecLoadLatency, 1>;
defm : C4GM4WriteResFPPair<WriteFTestY, [C4GM4FPU01, C4GM4FPU1], 8, [1, 2], 1, C864GM4Model.VecLoadLatency, 1>;
defm : X86WriteResPairUnsupported<WriteFTestZ>;
defm : C4GM4WriteResFPPair<WriteFShuffle, [C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteFShuffleY, [C4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteFShuffleZ>;
defm : C4GM4WriteResFPPair<WriteFVarShuffle, [C4GM4FPU01], 3>;
defm : C4GM4WriteResFPPair<WriteFVarShuffleY,[C4GM4FPU01], 3>;
defm : X86WriteResPairUnsupported<WriteFVarShuffleZ>;
defm : C4GM4WriteResFPPair<WriteFMul, [C4GM4FPU02], 3>;
defm : C4GM4WriteResFPPair<WriteFMulX, [C4GM4FPU02], 3>;
defm : C4GM4WriteResFPPair<WriteFMulY, [C4GM4FPU02], 3>;
defm : X86WriteResPairUnsupported<WriteFMulZ>;
defm : C4GM4WriteResFPPair<WriteFMul64, [C4GM4FPU02], 4>;
defm : C4GM4WriteResFPPair<WriteFMul64X, [C4GM4FPU02], 4>;
defm : C4GM4WriteResFPPair<WriteFMul64Y, [C4GM4FPU02], 4>;
defm : X86WriteResPairUnsupported<WriteFMul64Z>;
defm : C4GM4WriteResFPPair<WriteFMA, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFMAX, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFMAY, [C4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFMAZ>;
defm : C4GM4WriteResFPPair<WriteFRcp, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFRcpX, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFRcpY, [C4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFRcpZ>;
defm : C4GM4WriteResFPPair<WriteFRsqrt, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFRsqrtX, [C4GM4FPU02], 5>;
defm : C4GM4WriteResFPPair<WriteFRsqrtY, [C4GM4FPU02], 5>;
defm : X86WriteResPairUnsupported<WriteFRsqrtZ>;
defm : C4GM4WriteResFPPair<WriteFSqrt, [C4GM4FPU1], 8, [7]>;
defm : C4GM4WriteResFPPair<WriteFSqrtX, [C4GM4FPU1], 8, [7]>;
defm : C4GM4WriteResFPPair<WriteFSqrtY, [C4GM4FPU1], 8, [7]>;
defm : X86WriteResPairUnsupported<WriteFSqrtZ>;
defm : C4GM4WriteResFPPair<WriteFSqrt64, [C4GM4FPU1], 8, [10]>;
defm : C4GM4WriteResFPPair<WriteFSqrt64X, [C4GM4FPU1], 8, [10]>;
defm : C4GM4WriteResFPPair<WriteFSqrt64Y, [C4GM4FPU1], 8, [10]>;
defm : X86WriteResPairUnsupported<WriteFSqrt64Z>;
defm : C4GM4WriteResFPPair<WriteFSqrt80, [C4GM4FPU1], 22, [22]>;
defm : C4GM4WriteResFPPair<WriteFShuffle256, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WriteFVarShuffle256, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WriteDPPD, [C4GM4FPU1], 10, [8], 4>;

def C4GM4WriteVPERM:SchedWriteRes<[C4GM4FPU01]> {
  let Latency = 7;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM4WriteVPERM], (instrs VPERM2I128rri, VPERM2F128rri)>;

def C4GM4WriteVPERMLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteVPERM.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM4WriteVPERMLd], (instrs VPERM2I128rmi, VPERM2F128rmi)>;

def C4GM4WriteVPERMDY:SchedWriteRes<[C4GM4FPU0]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM4WriteVPERMDY], (instrs VPERMDYrr)>;

def C4GM4WriteVPERMDYLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU0]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteVPERMDY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteVPERMDYLd], (instrs VPERMDYrm)>;

defm : C4GM4WriteRes<WriteVecLoad, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM4WriteRes<WriteVecLoadX, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM4WriteRes<WriteVecLoadY, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM4WriteRes<WriteVecLoadNT, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM4WriteRes<WriteVecLoadNTY, [C4GM4AGU, C4GM4Load], !add(C864GM4Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM4WriteRes<WriteVecMaskedLoad, [C4GM4AGU, C4GM4Load, C4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : C4GM4WriteRes<WriteVecMaskedLoadY, [C4GM4AGU, C4GM4Load, C4GM4FPU02], !add(C864GM4Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : C4GM4WriteRes<WriteVecStore, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteVecStoreX, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteVecStoreY, [C4GM4AGU, C4GM4Store], C864GM4Model.StoreLatency>;
defm : C4GM4WriteRes<WriteVecStoreNT, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : C4GM4WriteRes<WriteVecStoreNTY, [C4GM4FPU1, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 4)>;
defm : C4GM4WriteRes<WriteVecMaskedStore32, [C4GM4FPU02, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : C4GM4WriteRes<WriteVecMaskedStore32Y, [C4GM4FPU02, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : C4GM4WriteRes<WriteVecMaskedStore64, [C4GM4FPU02, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : C4GM4WriteRes<WriteVecMaskedStore64Y, [C4GM4FPU02, C4GM4AGU, C4GM4Store], !add(C864GM4Model.StoreLatency, 1)>;
defm : C4GM4WriteRes<WriteVecMoveToGpr, [C4GM4FPU1], 1>;
defm : C4GM4WriteRes<WriteVecMoveFromGpr, [C4GM4FPU1], 1>;
defm : C4GM4WriteRes<WriteEMMS, [C4GM4FPU], 0>;

defm : C4GM4WriteResFPPair<WriteVecShift, [C4GM4FPU1], 1>;
defm : C4GM4WriteResFPPair<WriteVecShiftX, [C4GM4FPU1], 1>;
defm : C4GM4WriteResFPPair<WriteVecShiftY, [C4GM4FPU1], 1>;
defm : X86WriteResPairUnsupported<WriteVecShiftZ>;
defm : C4GM4WriteResFPPair<WriteVecShiftImm, [C4GM4FPU1], 1>;
defm : C4GM4WriteResFPPair<WriteVecShiftImmX, [C4GM4FPU1], 1>;
defm : C4GM4WriteResFPPair<WriteVecShiftImmY, [C4GM4FPU1], 1>;
defm : X86WriteResPairUnsupported<WriteVecShiftImmZ>;
defm : C4GM4WriteResFPPair<WriteVarVecShift, [C4GM4FPU0], 3, [2], 1>;
defm : C4GM4WriteResFPPair<WriteVarVecShiftY, [C4GM4FPU0], 3, [2], 1>;
defm : X86WriteResPairUnsupported<WriteVarVecShiftZ>;
defm : C4GM4WriteResFPPair<WriteVecLogic, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteVecLogicX, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteVecLogicY, [C4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteVecLogicZ>;
defm : C4GM4WriteResFPPair<WriteVecTest, [C4GM4FPU01, C4GM4FPU1], 4, [1, 2], 1, 7, 1>;
defm : C4GM4WriteResFPPair<WriteVecTestY, [C4GM4FPU01, C4GM4FPU1], 8, [1, 2], 1, 7, 1>;
defm : X86WriteResPairUnsupported<WriteVecTestZ>;
defm : C4GM4WriteResFPPair<WriteVecALU, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteVecALUX, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteVecALUY, [C4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteVecALUZ>;
defm : C4GM4WriteResFPPair<WriteVecIMul, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WriteVecIMulX, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WriteVecIMulY, [C4GM4FPU0], 3>;
defm : C4GM4WriteRes<WriteCLMul, [C4GM4FPU3], 3>;
defm : C4GM4WriteRes<WriteCLMulLd, [C4GM4AGU, C4GM4Load, C4GM4FPU3], !add(C864GM4Model.VecLoadLatency, 3)>;
defm : X86WriteResPairUnsupported<WriteVecIMulZ>;
defm : C4GM4WriteResFPPair<WritePMULLD, [C4GM4FPU0], 4, [2]>;
defm : C4GM4WriteResFPPair<WritePMULLDY, [C4GM4FPU0], 4, [2]>;
defm : X86WriteResPairUnsupported<WritePMULLDZ>;
defm : C4GM4WriteResFPPair<WriteShuffle, [C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteShuffleX, [C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteShuffleY, [C4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteShuffleZ>;
defm : C4GM4WriteResFPPair<WriteVarShuffle, [C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteVarShuffleX,[C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteVarShuffleY,[C4GM4FPU01], 1>;
defm : X86WriteResPairUnsupported<WriteVarShuffleZ>;
defm : C4GM4WriteResFPPair<WriteBlend, [C4GM4FPU], 1>;
defm : C4GM4WriteResFPPair<WriteBlendY, [C4GM4FPU], 1>;
defm : X86WriteResPairUnsupported<WriteBlendZ>;
defm : C4GM4WriteResFPPair<WriteVarBlend, [C4GM4FPU0], 1>;
defm : C4GM4WriteResFPPair<WriteVarBlendY, [C4GM4FPU0], 1>;
defm : X86WriteResPairUnsupported<WriteVarBlendZ>;
defm : C4GM4WriteResFPPair<WriteShuffle256, [C4GM4FPU01], 3>;
defm : C4GM4WriteResFPPair<WriteVPMOV256, [C4GM4FPU01], 1>;
defm : C4GM4WriteResFPPair<WriteVarShuffle256, [C4GM4FPU01], 2>;
defm : C4GM4WriteResFPPair<WritePSADBW, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WritePSADBWX, [C4GM4FPU0], 3>;
defm : C4GM4WriteResFPPair<WritePSADBWY, [C4GM4FPU0], 3>;
defm : X86WriteResPairUnsupported<WritePSADBWZ>;
defm : C4GM4WriteResFPPair<WriteMPSAD, [C4GM4FPU], 4, [8], 1>;
defm : C4GM4WriteResFPPair<WriteMPSADY, [C4GM4FPU], 4, [8], 1>;
defm : C4GM4WriteResFPPair<WriteMPSADZ, [C4GM4FPU], 4, [16], 1>;
defm : C4GM4WriteResFPPair<WritePHMINPOS, [C4GM4FPU0], 3>;

// EXTRQ INSERTQ
def C4GM4WriteEXTRQ_INSERTQ:SchedWriteRes<[C4GM4FPU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteEXTRQ_INSERTQ], (instrs EXTRQ, EXTRQI, INSERTQ, INSERTQI)>;

def C4GM4WriteVPMOVY:SchedWriteRes<[C4GM4FPU01]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteVPMOVY], (instregex "VPMOV(S|Z)X(BD|BQ|BW|DQ|WD|WQ)Yrr")>;

def C4GM4WriteVPMOVYLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteVPMOVY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteVPMOVYLd], (instregex "VPMOV(S|Z)X(BD|BQ|BW|DQ|WD|WQ)Yrm")>;

def C4GM4WriteVPS:SchedWriteRes<[C4GM4FPU1]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteVPS], (instregex "VPS(LL|RA|RL)(D|Q|W)Yrr")>;

def C4GM4WriteVPSLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU1]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteVPS.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteVPSLd], (instregex "VPS(LL|RA|RL)(D|Q|W)Yrm")>;

def C4GM4WritePMADDUBSW_PMULHRSW:SchedWriteRes<[C4GM4FPU0]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WritePMADDUBSW_PMULHRSW], (instregex "(V?)PMADDUBSW(Y)?rr",
                                                        "(V?)PMULHRSW(Y)?rr")>;

def C4GM4WritePMADDUBSW_PMULHRSWLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU0]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WritePMADDUBSW_PMULHRSW.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WritePMADDUBSW_PMULHRSWLd], (instregex "(V?)PMADDUBSW(Y)?rm",
                                                          "(V?)PMULHRSW(Y)?rm")>;

// Vector insert/extract operations.
defm : C4GM4WriteResFPPair<WriteVecInsert, [C4GM4FPU01], 1>;
defm : C4GM4WriteRes<WriteVecExtract, [C4GM4FPU01, C4GM4FPU1], 2, [1, 2], 2>;
defm : C4GM4WriteRes<WriteVecExtractSt, [C4GM4FPU01, C4GM4FPU1, C4GM4AGU, C4GM4Store], 2, [1, 1, 1, 2], 2>;

def C4GM4WritePINSR:SchedWriteRes<[C4GM4FPU01]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WritePINSR], (instregex "(V?)PINSR(B|D|Q|W)rri")>;

def C4GM4WritePINSRLd:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU01]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WritePINSR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WritePINSRLd], (instregex "(V?)PINSR(B|D|Q|W)rmi")>;


// MOVMSK Instructions.
defm : C4GM4WriteRes<WriteFMOVMSK, [C4GM4FPU1], 1>;
defm : C4GM4WriteRes<WriteMMXMOVMSK, [C4GM4FPU1], 1>;
defm : C4GM4WriteRes<WriteVecMOVMSK, [C4GM4FPU1], 1>;
defm : C4GM4WriteRes<WriteVecMOVMSKY, [C4GM4FPU1], 1>;

// Strings instructions.
defm : C4GM4WriteResFPPair<WritePCmpIStrM, [C4GM4FPU], 7, [25], 3>;
defm : C4GM4WriteResFPPair<WritePCmpEStrM, [C4GM4FPU], 8, [25], 8>;
defm : C4GM4WriteResFPPair<WritePCmpIStrI, [C4GM4FPU], 1, [8], 2>;
defm : C4GM4WriteResFPPair<WritePCmpEStrI, [C4GM4FPU], 2, [13], 8>;

// AES Instructions.
defm : C4GM4WriteResFPPair<WriteAESDecEnc, [C4GM4FPU23], 4>;
defm : C4GM4WriteResFPPair<WriteAESIMC, [C4GM4FPU23], 4>;
defm : C4GM4WriteResFPPair<WriteAESKeyGen, [C4GM4FPU23], 4>;

defm : C4GM4WriteRes<WriteFence, [C4GM4AGU]>;
defm : C4GM4WriteRes<WriteNop, []>;

// SHA Instructions.

def C4GM4WriteSHA1:SchedWriteRes<[C4GM4FPU2]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA1], (instrs SHA1MSG1rr, SHA1MSG2rr, SHA1NEXTErr)>;

def C4GM4WriteSHA1Ld:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteSHA1.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA1Ld], (instrs SHA1MSG1rm, SHA1MSG2rm, SHA1NEXTErm)>;

def C4GM4WriteSHA1RNDS4rri:SchedWriteRes<[C4GM4FPU2]> {
  let Latency = 6;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA1RNDS4rri], (instrs SHA1RNDS4rri)>;

def C4GM4WriteSHA1RNDS4rmi:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteSHA1RNDS4rri.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA1RNDS4rmi], (instrs SHA1RNDS4rmi)>;

def C4GM4WriteSHA256MSG1rr:SchedWriteRes<[C4GM4FPU2]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA256MSG1rr], (instrs SHA256MSG1rr)>;

def C4GM4WriteSHA256MSG1rm:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteSHA256MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA256MSG1rm], (instrs SHA256MSG1rm)>;

def C4GM4WriteSHA256:SchedWriteRes<[C4GM4FPU2]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA256], (instrs SHA256MSG2rr, SHA256RNDS2rr)>;

def C4GM4WriteSHA256Ld:SchedWriteRes<[C4GM4AGU, C4GM4Load, C4GM4FPU2]> {
  let Latency = !add(C864GM4Model.VecLoadLatency, C4GM4WriteSHA256.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteSHA256Ld], (instrs SHA256MSG2rm, SHA256RNDS2rm)>;

//===----------------------------------------------------------------------===//
// Zero Cycle Move
//===----------------------------------------------------------------------===//

def C4GM4WriteZeroLatency : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM4WriteZeroLatency], (instrs MOV32rr, MOV32rr_REV,
                                              MOV64rr, MOV64rr_REV,
                                              MOVSX32rr32)>;

def C4GM4WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM4WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                                 XCHG64rr, XCHG64ar)>;

defm : C4GM4WriteRes<WriteXCHG, [C4GM4ALU], 1, [2], 1>;

defm : C4GM4WriteRes<WriteFMoveX, [], 0, [], 1>;
defm : C4GM4WriteRes<WriteFMoveY, [], 0, [], 1>;
defm : X86WriteResUnsupported<WriteFMoveZ>;

defm : C4GM4WriteRes<WriteVecMove, [], 0, [], 1>;
defm : C4GM4WriteRes<WriteVecMoveX, [], 0, [], 1>;
defm : C4GM4WriteRes<WriteVecMoveY, [], 0, [], 1>;
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

def C4GM4WriteZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteALU]>
]>;
def : InstRW<[C4GM4WriteZeroIdiom], (instrs XOR32rr, XOR32rr_REV,
                                            XOR64rr, XOR64rr_REV,
                                            SUB32rr, SUB32rr_REV,
                                            SUB64rr, SUB64rr_REV)>;

def C4GM4WriteZeroIdiomEFLAGS : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<CheckSameRegOperand<0, 1>>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                                 [WriteALU]>
]>;
def : InstRW<[C4GM4WriteZeroIdiomEFLAGS], (instrs CMP8rr,  CMP8rr_REV,
                                                  CMP16rr, CMP16rr_REV,
                                                  CMP32rr, CMP32rr_REV,
                                                  CMP64rr, CMP64rr_REV)>;

def C4GM4WriteFZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogic]>
]>;

def : InstRW<[C4GM4WriteFZeroIdiom], (instrs VXORPSrr, VXORPDrr,
                                             VANDNPSrr, VANDNPDrr)>;

def C4GM4WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[C4GM4WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                              VANDNPSYrr, VANDNPDYrr)>;

def C4GM4WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;

def : InstRW<[C4GM4WriteVZeroIdiomLogicX], (instrs VPXORrr, VPANDNrr)>;

def C4GM4WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[C4GM4WriteVZeroIdiomLogicY], (instrs VPXORYrr, VPANDNYrr)>;

def C4GM4WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;


def : InstRW<[C4GM4WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr)>;

def C4GM4WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM4WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[C4GM4WriteVZeroIdiomALUY],
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
