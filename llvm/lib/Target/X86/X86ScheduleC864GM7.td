//=- X86ScheduleC864GM7.td - X86 C86-4G-M7 Scheduling --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for C86-4G-M7 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def C864GM7Model : SchedMachineModel {
  let IssueWidth = 4;
  let MicroOpBufferSize = 256;
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

let SchedModel = C864GM7Model in {

//===----------------------------------------------------------------------===//
// Integer Execution Unit
//===----------------------------------------------------------------------===//

// The C864GM7 has 4 ALUs.
def 4GM7ALU0 : ProcResource<1>;
def 4GM7ALU1 : ProcResource<1>;
def 4GM7ALU2 : ProcResource<1>;
def 4GM7ALU3 : ProcResource<1>;

def 4GM7ALU : ProcResGroup<[4GM7ALU0, 4GM7ALU1, 4GM7ALU2, 4GM7ALU3]>;
def 4GM7ALU12 : ProcResGroup<[4GM7ALU1, 4GM7ALU2]>;
def 4GM7ALU03 : ProcResGroup<[4GM7ALU0, 4GM7ALU3]>;

// BRU0 on ALU0.
defvar 4GM7BRU0 = 4GM7ALU0;
// BRU1 is a separate branch execution unit.
def 4GM7BRU1 : ProcResource<1>;

def 4GM7BRU : ProcResGroup<[4GM7BRU0, 4GM7BRU1]>;

// The C864GM7 has 3 AGUs.
def 4GM7AGU0 : ProcResource<1>;
def 4GM7AGU1 : ProcResource<1>;
def 4GM7AGU2 : ProcResource<1>;

def 4GM7AGU : ProcResGroup<[4GM7AGU0, 4GM7AGU1, 4GM7AGU2]>;

// 96 Entry (4x24 entries) integer Scheduler.
def 4GM7Int : ProcResGroup<[4GM7ALU0, 4GM7AGU0, 4GM7BRU0,
                            4GM7ALU1, 4GM7AGU1,
                            4GM7ALU2, 4GM7AGU2,
                            4GM7ALU3,           4GM7BRU1]> {
  let BufferSize = 96;
}

// The integer physical register file consists of 224 registers.
def 4GM7IntegerPRF : RegisterFile<224,
                                  [GR64, CCR],
                                  [1, 1],
                                  [1, 0],
                                  6,  // Max moves that can be eliminated per cycle.
                                  0>; // Restrict move elimination to zero regs.

//===----------------------------------------------------------------------===//
// Floating-Point Unit
//===----------------------------------------------------------------------===//

// The C864GM7 has 4 FPUs.
def 4GM7FPU0 : ProcResource<1>;
def 4GM7FPU1 : ProcResource<1>;
def 4GM7FPU2 : ProcResource<1>;
def 4GM7FPU3 : ProcResource<1>;

def 4GM7FPU013 : ProcResGroup<[4GM7FPU0, 4GM7FPU1, 4GM7FPU3]>;
def 4GM7FPU01  : ProcResGroup<[4GM7FPU0, 4GM7FPU1]>;
def 4GM7FPU12  : ProcResGroup<[4GM7FPU1, 4GM7FPU2]>;
def 4GM7FPU13  : ProcResGroup<[4GM7FPU1, 4GM7FPU3]>;
def 4GM7FPU23  : ProcResGroup<[4GM7FPU2, 4GM7FPU3]>;
def 4GM7FPU02  : ProcResGroup<[4GM7FPU0, 4GM7FPU2]>;
def 4GM7FPU03  : ProcResGroup<[4GM7FPU0, 4GM7FPU3]>;

// 48 Entry (4x12 entries) floating-point Scheduler.
def 4GM7FPU : ProcResGroup<[4GM7FPU0, 4GM7FPU1, 4GM7FPU2, 4GM7FPU3]> {
  let BufferSize = 48;
}

// The floating point physical register file consists of 256 bits x 208 registers.
def 4GM7FpuPRF : RegisterFile<208,
                              [VR64, VR128, VR256, VR512],
                              [1, 1, 1, 2],
                              [0, 1, 1, 0],
                              6,  // Max moves that can be eliminated per cycle.
                              0>; // Restrict move elimination to zero regs.


//===----------------------------------------------------------------------===//
// Load-Store Unit
//===----------------------------------------------------------------------===//

// Load/Store Units and Memory Queues
// The C864GM7 has 4 LS Units.
def 4GM7LSU : ProcResource<4>;

// The three of LSU can be loads.
let Super = 4GM7LSU in
def 4GM7Load : ProcResource<3> {
  // The LDQ is 54.
  let BufferSize = 54;
}
def 4GM7LoadQueue : LoadQueue<4GM7Load>;

// All four of LSU can be loads.
let Super = 4GM7LSU in
def 4GM7Store : ProcResource<4> {
  // The STQ is 52.
  let BufferSize = 52;
}
def 4GM7StoreQueue : StoreQueue<4GM7Store>;

def : ReadAdvance<ReadAfterLd, C864GM7Model.LoadLatency>;
def : ReadAdvance<ReadAfterVecLd, C864GM7Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecXLd, C864GM7Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecYLd, C864GM7Model.VecLoadLatency>;
def : ReadAdvance<ReadInt2Fpu, 0>;


//===----------------------------------------------------------------------===//
// Retire Control Unit
//===----------------------------------------------------------------------===//
def 4GM7RCU : RetireControlUnit<C864GM7Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Basic helper classes.
//===----------------------------------------------------------------------===//

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.


multiclass __4GM7WriteRes<SchedWrite SchedRW, list<ProcResourceKind> ExePorts,
                         int Lat = 1, list<int> Res = [], int UOps = 1> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }
}

multiclass __4GM7WriteResPair<X86FoldableSchedWrite SchedRW,
                             list<ProcResourceKind> ExePorts, int Lat,
                             list<int> Res, int UOps, int LoadLat, int LoadUOps,
                             ProcResourceKind AGU, int LoadRes> {
  defm : __4GM7WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;

  defm : __4GM7WriteRes<SchedRW.Folded,
                       !listconcat([AGU, 4GM7Load], ExePorts),
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
multiclass 4GM7WriteRes<SchedWrite SchedRW,
                        list<ProcResourceKind> ExePorts, int Lat = 1,
                        list<int> Res = [], int UOps = 1> {
  defm : __4GM7WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

// For classes with folded loads.
multiclass 4GM7WriteResIntPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __4GM7WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM7Model.LoadLatency,
                           LoadUOps, 4GM7AGU, LoadRes>;
}

multiclass 4GM7WriteResFPPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __4GM7WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM7Model.VecLoadLatency,
                           LoadUOps, 4GM7AGU, LoadRes>;
}

// Microcoded Instructions
def 4GM7WriteMicrocoded : SchedWriteRes<[]> {
  let Latency = 100;
}

def : SchedAlias<WriteMicrocoded, 4GM7WriteMicrocoded>;
def : SchedAlias<WriteSystem, 4GM7WriteMicrocoded>;
def : SchedAlias<WriteFCMOV, 4GM7WriteMicrocoded>;
def : SchedAlias<WriteLDMXCSR, 4GM7WriteMicrocoded>;
def : SchedAlias<WriteSTMXCSR, 4GM7WriteMicrocoded>;

// Integer Instructions

defm : 4GM7WriteRes<WriteRMW, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency, [1, 1], 0>;

defm : 4GM7WriteRes<WriteStore, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency, [1, 2], 1>;
defm : 4GM7WriteRes<WriteStoreNT, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency, [1, 2], 1>;
defm : 4GM7WriteRes<WriteLoad, [4GM7AGU, 4GM7Load], !add(C864GM7Model.LoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteMove, [4GM7ALU], 1, [4], 1>;

def : InstRW<[WriteMove], (instrs COPY, MOVSX16rr16, MOVSX16rr32,
                                  MOVZX16rr16, MOVSX16rr8, MOVZX16rr8)>;

defm : 4GM7WriteRes<WriteVecMaskedGatherWriteback, [], !add(1, C864GM7Model.VecLoadLatency)>;

defm : 4GM7WriteRes<WriteZero, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteLEA, [4GM7AGU]>;
defm : 4GM7WriteResIntPair<WriteALU, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteADC, [4GM7ALU], 1>;

// This write is used for slow LEA instructions.
def 4GM7Write3OpsLEA : SchedWriteRes<[4GM7ALU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
// a slow LEA is either a 3Ops LEA (base, index, offset),
// or an LEA with a `Scale` value different than 1.
def 4GM7SlowLEAPredicate : MCSchedPredicate<
  CheckAny<[
    IsThreeOperandsLEAFn,
    CheckAll<[
      CheckIsImmOperand<2>,
      CheckNot<CheckImmOperand<2, 1>>
    ]>
  ]>
>;

def 4GM7WriteLEA : SchedWriteVariant<[
    SchedVar<4GM7SlowLEAPredicate, [4GM7Write3OpsLEA]>,
    SchedVar<NoSchedPred,          [WriteLEA]>
]>;

def : InstRW<[4GM7WriteLEA], (instrs LEA64r, LEA64_32r)>;

// MUL IMUL MULX
defm : 4GM7WriteResIntPair<WriteIMul8, [4GM7ALU1], 3, [3], 1>;
defm : 4GM7WriteResIntPair<WriteIMul16, [4GM7ALU1], 3, [3], 1>;
defm : 4GM7WriteResIntPair<WriteIMul16Imm, [4GM7ALU1], 4, [3], 2>;
defm : 4GM7WriteResIntPair<WriteIMul16Reg, [4GM7ALU1], 3>;
defm : 4GM7WriteResIntPair<WriteIMul32, [4GM7ALU1], 3, [3], 1>;
defm : 4GM7WriteResIntPair<WriteIMul32Imm, [4GM7ALU1], 3, [3], 2>;
defm : 4GM7WriteResIntPair<WriteIMul32Reg, [4GM7ALU1], 3>;
defm : 4GM7WriteResIntPair<WriteIMul64, [4GM7ALU1], 3, [4], 1>;
defm : 4GM7WriteResIntPair<WriteIMul64Imm, [4GM7ALU1], 3, [4], 2>;
defm : 4GM7WriteResIntPair<WriteIMul64Reg, [4GM7ALU1], 3>;

defm : 4GM7WriteResIntPair<WriteMULX32, [4GM7ALU1], 3, [2], 2>;
defm : 4GM7WriteResIntPair<WriteMULX64, [4GM7ALU1], 4, [2], 2>;
defm : 4GM7WriteRes<WriteIMulHLd, [4GM7ALU1], !add(3, C864GM7Model.LoadLatency), [], 0>;
defm : 4GM7WriteRes<WriteIMulH, [4GM7ALU1], 3, [], 0>;

// IDIV
defm : 4GM7WriteResIntPair<WriteDiv8, [4GM7ALU3], 9, [4], 2>;
defm : 4GM7WriteResIntPair<WriteDiv16, [4GM7ALU3], 10, [4], 2>;
defm : 4GM7WriteResIntPair<WriteDiv32, [4GM7ALU3], 12, [7], 2>;
defm : 4GM7WriteResIntPair<WriteDiv64, [4GM7ALU3], 18, [10], 2>;
defm : 4GM7WriteResIntPair<WriteIDiv8, [4GM7ALU3], 9, [4], 2>;
defm : 4GM7WriteResIntPair<WriteIDiv16, [4GM7ALU3], 10, [4], 2>;
defm : 4GM7WriteResIntPair<WriteIDiv32, [4GM7ALU3], 12, [7], 2>;
defm : 4GM7WriteResIntPair<WriteIDiv64, [4GM7ALU3], 18, [10], 2>;

defm : 4GM7WriteRes<WriteBSWAP32, [4GM7ALU], 1, [1], 1>;
defm : 4GM7WriteRes<WriteBSWAP64, [4GM7ALU], 1, [1], 1>;
defm : 4GM7WriteRes<WriteCMPXCHG, [4GM7ALU], 3, [12], 5>;
defm : 4GM7WriteRes<WriteCMPXCHGRMW,[4GM7ALU], 3, [12], 6>;

def 4GM7WriteXADD : SchedWriteRes<[4GM7ALU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteXADD], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr)>;

def 4GM7WriteXADDLd : SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7ALU]> {
  let Latency = !add(C864GM7Model.LoadLatency, 4GM7WriteXADD.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteXADDLd], (instrs XADD8rm, XADD16rm, XADD32rm, XADD64rm)>;

def 4GM7WriteCMPXCHG8rr : SchedWriteRes<[4GM7ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 3;
}
def : InstRW<[4GM7WriteCMPXCHG8rr], (instrs CMPXCHG8rr)>;

def 4GM7WriteCMPXCHG8mr : SchedWriteRes<[4GM7ALU, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteCMPXCHG8rr.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [12, 1, 1];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteCMPXCHG8mr], (instrs CMPXCHG8rm, LCMPXCHG8)>;

def 4GM7WriteXCHG : SchedWriteRes<[4GM7ALU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteXCHG], (instrs XCHG8rr, XCHG16ar, XCHG16rr)>;

def 4GM7WriteXCHGLd : SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7ALU]> {
  let Latency = !add(C864GM7Model.LoadLatency, 4GM7WriteXCHG.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteXCHGLd], (instrs XCHG8rm, XCHG16rm)>;

defm : 4GM7WriteResIntPair<WriteShift, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteShiftCL, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteRotate, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteRotateCL, [4GM7ALU], 1>;

def 4GM7WriteRCR:SchedWriteRes<[4GM7ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRCR], (instregex "RCR(8|16|32|64)rCL")>;

def 4GM7WriteRCRLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7ALU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRCR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRCRLd], (instregex "RCR(8|16|32|64)mCL")>;

def 4GM7WriteRCL:SchedWriteRes<[4GM7ALU]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRCL], (instregex "RCL(8|16|32|64)rCL")>;

def 4GM7WriteRCLLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7ALU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRCL.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRCLLd], (instregex "RCL(8|16|32|64)mCL")>;

defm : 4GM7WriteRes<WriteSHDrri, [4GM7ALU], 3, [4], 6>;
defm : 4GM7WriteRes<WriteSHDrrcl, [4GM7ALU], 3, [4], 7>;
defm : 4GM7WriteRes<WriteSHDmri, [4GM7ALU, 4GM7AGU, 4GM7Store], !add(3, C864GM7Model.StoreLatency), [4, 1, 1], 8>;
defm : 4GM7WriteRes<WriteSHDmrcl, [4GM7ALU, 4GM7AGU, 4GM7Store], !add(3, C864GM7Model.StoreLatency), [4, 1, 1], 8>;

defm : 4GM7WriteResIntPair<WriteJump, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteCRC32, [4GM7ALU], 3, [12], 3>;

defm : 4GM7WriteResIntPair<WriteCMOV, [4GM7ALU], 1>;
defm : 4GM7WriteRes<WriteSETCC, [4GM7ALU], 1, [1], 1>;
defm : 4GM7WriteRes<WriteSETCCStore, [4GM7ALU, 4GM7AGU, 4GM7Store], 1, [2, 1, 1], 2>;
defm : 4GM7WriteRes<WriteLAHFSAHF, [4GM7ALU], 2, [2], 4>;

defm : 4GM7WriteRes<WriteBitTest, [4GM7ALU], 1, [1], 1>;
defm : 4GM7WriteRes<WriteBitTestImmLd, [4GM7AGU, 4GM7Load, 4GM7ALU], !add(C864GM7Model.LoadLatency, 1), [1,1,1], 2>;
defm : 4GM7WriteRes<WriteBitTestRegLd, [4GM7AGU, 4GM7Load, 4GM7ALU], !add(C864GM7Model.LoadLatency, 1), [1,1,1], 6>;
defm : 4GM7WriteRes<WriteBitTestSet, [4GM7ALU], 2, [1], 2>;
defm : 4GM7WriteRes<WriteBitTestSetImmLd, [4GM7AGU, 4GM7Load, 4GM7ALU], !add(C864GM7Model.LoadLatency, 2), [1, 1, 1], 4>;
defm : 4GM7WriteRes<WriteBitTestSetRegLd, [4GM7AGU, 4GM7Load, 4GM7ALU], !add(C864GM7Model.LoadLatency, 2), [1, 1, 1], 8>;

// Bit counts.
defm : 4GM7WriteResIntPair<WriteBSF, [4GM7ALU], 3, [12], 6, -2>;
defm : 4GM7WriteResIntPair<WriteBSR, [4GM7ALU], 4, [16], 6, -3>;
defm : 4GM7WriteResIntPair<WriteLZCNT, [4GM7ALU], 1>;
defm : 4GM7WriteResIntPair<WriteTZCNT, [4GM7ALU], 2, [2], 2>;
defm : 4GM7WriteResIntPair<WritePOPCNT, [4GM7ALU], 1>;

// PLZCNT
def 4GM7WritePLZCNTXY:SchedWriteRes<[4GM7FPU]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePLZCNTXY], (instregex "VPLZCNT(D|Q)(Z128|Z256)rr(k|kz)")>;

def 4GM7WritePLZCNTXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePLZCNTXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePLZCNTXYLd], (instregex "VPLZCNT(D|Q)(Z128|Z256)rm(k|kz)")>;

def 4GM7WritePLZCNTZ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePLZCNTZ], (instregex "VPLZCNT(D|Q)Zrr(k|kz)")>;

def 4GM7WritePLZCNTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePLZCNTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePLZCNTZLd], (instregex "VPLZCNT(D|Q)Zrm(k|kz)")>;

// BMI1 BEXTR, BMI2 BZHI
defm : 4GM7WriteResIntPair<WriteBEXTR, [4GM7ALU], 1, [1], 1>;
defm : 4GM7WriteResIntPair<WriteBLS, [4GM7ALU], 2, [2], 3>;
defm : 4GM7WriteResIntPair<WriteBZHI, [4GM7ALU], 1>;


// Floating Point Instructions
defm : 4GM7WriteRes<WriteFLD0, [4GM7AGU, 4GM7Load, 4GM7FPU1], !add(C864GM7Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM7WriteRes<WriteFLD1, [4GM7AGU, 4GM7Load, 4GM7FPU1], !add(C864GM7Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM7WriteRes<WriteFLDC, [4GM7AGU, 4GM7Load, 4GM7FPU1], !add(C864GM7Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : 4GM7WriteRes<WriteFLoad, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1)>;
defm : 4GM7WriteRes<WriteFLoadX, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1)>;
defm : 4GM7WriteRes<WriteFLoadY, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1)>;
defm : 4GM7WriteRes<WriteFMaskedLoad, [4GM7AGU, 4GM7Load, 4GM7FPU01], !add(C864GM7Model.VecLoadLatency, 1)>;
defm : 4GM7WriteRes<WriteFMaskedLoadY, [4GM7AGU, 4GM7Load, 4GM7FPU01], !add(C864GM7Model.VecLoadLatency, 1), [1, 1, 1], 2>;

defm : 4GM7WriteRes<WriteFStore, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFStoreX, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFStoreY, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFStoreNT, [4GM7FPU1, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteFStoreNTX, [4GM7FPU1, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteFStoreNTY, [4GM7FPU1, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteFMaskedStore32, [4GM7FPU01, 4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFMaskedStore32Y, [4GM7FPU01, 4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFMaskedStore64, [4GM7FPU01, 4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteFMaskedStore64Y, [4GM7FPU01, 4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;

defm : 4GM7WriteResFPPair<WriteFAdd, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAddX, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAddY, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAddZ, [4GM7FPU13], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFAdd64, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAdd64X, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAdd64Y, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteFAdd64Z, [4GM7FPU13], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFHAdd, [4GM7FPU13], 7, [8], 4, 3>;
defm : 4GM7WriteResFPPair<WriteFHAddY, [4GM7FPU13], 7, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WriteFHAddZ>;
defm : 4GM7WriteResFPPair<WritePHAdd, [4GM7FPU13], 3, [8], 4, 3>;
defm : 4GM7WriteResFPPair<WritePHAddX, [4GM7FPU13], 3, [8], 4, 3>;
defm : 4GM7WriteResFPPair<WritePHAddY, [4GM7FPU13], 3, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WritePHAddZ>;
defm : 4GM7WriteResFPPair<WriteFMul, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMulX, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMulY, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMulZ, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFMul64, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMul64X, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMul64Y, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFMul64Z, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFMA, [4GM7FPU02], 4>;
defm : 4GM7WriteResFPPair<WriteFMAX, [4GM7FPU02], 4>;
defm : 4GM7WriteResFPPair<WriteFMAY, [4GM7FPU02], 4>;
defm : 4GM7WriteResFPPair<WriteFMAZ, [4GM7FPU02], 4, [2], 2>;
defm : 4GM7WriteResFPPair<WriteFRcp, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRcpX, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRcpY, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRcpZ, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRsqrt, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRsqrtX, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRsqrtY, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFRsqrtZ, [4GM7FPU02], 5, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFSqrt, [4GM7FPU13], 8, [13]>;
defm : 4GM7WriteResFPPair<WriteFSqrtX, [4GM7FPU13], 8, [13]>;
defm : 4GM7WriteResFPPair<WriteFSqrtY, [4GM7FPU13], 8, [13]>;
defm : 4GM7WriteResFPPair<WriteFSqrtZ, [4GM7FPU13], 16, [13]>;
defm : 4GM7WriteResFPPair<WriteFSqrt64, [4GM7FPU13], 8, [19]>;
defm : 4GM7WriteResFPPair<WriteFSqrt64X, [4GM7FPU13], 8, [19]>;
defm : 4GM7WriteResFPPair<WriteFSqrt64Y, [4GM7FPU13], 8, [19]>;
defm : 4GM7WriteResFPPair<WriteFSqrt64Z, [4GM7FPU13], 16, [19]>;
defm : 4GM7WriteResFPPair<WriteFSqrt80, [4GM7FPU13], 22, [23]>;
defm : 4GM7WriteResFPPair<WriteFDiv, [4GM7FPU13], 10, [9], 1>;
defm : 4GM7WriteResFPPair<WriteFDivX, [4GM7FPU13], 10, [9], 1>;
defm : 4GM7WriteResFPPair<WriteFDivY, [4GM7FPU13], 10, [9], 1>;
defm : 4GM7WriteResFPPair<WriteFDivZ, [4GM7FPU13], 18, [18], 2>;
defm : 4GM7WriteResFPPair<WriteFDiv64, [4GM7FPU13], 8, [12], 1>;
defm : 4GM7WriteResFPPair<WriteFDiv64X, [4GM7FPU13], 8, [12], 1>;
defm : 4GM7WriteResFPPair<WriteFDiv64Y, [4GM7FPU13], 8, [12], 1>;
defm : 4GM7WriteResFPPair<WriteFDiv64Z, [4GM7FPU13], 16, [24], 2>;
defm : 4GM7WriteResFPPair<WriteDPPD, [4GM7FPU13], 9, [8], 4>;
defm : 4GM7WriteResFPPair<WriteDPPS, [4GM7FPU13], 14, [8], 8>;
defm : 4GM7WriteResFPPair<WriteDPPSY, [4GM7FPU13], 14, [8], 8>;

// PMADD
def 4GM7WriteVPMADDXYrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDXYrrk], (instrs VPMADDWDZ128rrk, VPMADDWDZ256rrk)>;

def 4GM7WriteVPMADDXYrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADDXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDXYrmk], (instrs VPMADDWDZ128rmk, VPMADDWDZ256rmk)>;

def 4GM7WriteVPMADDZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDZrrk], (instrs VPMADDWDZrrk)>;

def 4GM7WriteVPMADDZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADDZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDZrmk], (instrs VPMADDWDZrmk)>;


// PMADD52
def 4GM7WriteVPMADD52XY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52XY], (instregex "VPMADD52(L|H)UQ(Z128|Z256)(r|rkz)$")>;

def 4GM7WriteVPMADD52XYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADD52XY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52XYLd], (instregex "VPMADD52(L|H)UQ(Z128|Z256)(m|mkz)$")>;

def 4GM7WriteVPMADD52XYrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52XYrk], (instregex "VPMADD52(L|H)UQ(Z128|Z256)rk$")>;

def 4GM7WriteVPMADD52XYmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADD52XYrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52XYmk], (instregex "VPMADD52(L|H)UQ(Z128|Z256)mk$")>;


def 4GM7WriteVPMADD52Z:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52Z], (instregex "VPMADD52(L|H)UQZ(r|rkz)$")>;

def 4GM7WriteVPMADD52ZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADD52Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52ZLd], (instregex "VPMADD52(L|H)UQZ(m|mkz)$")>;

def 4GM7WriteVPMADD52Zrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52Zrk], (instregex "VPMADD52(L|H)UQZrk$")>;

def 4GM7WriteVPMADD52Zmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADD52Zrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADD52Zmk], (instregex "VPMADD52(L|H)UQZmk$")>;


// VMULZ VPMULZ
def 4GM7WriteVMULZ_VPMULZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVMULZ_VPMULZ], (instrs VMULPSZ128rrk, VMULPDZ128rrk,
                                              VMULPSZ256rrk, VMULPDZ256rrk,
                                              VMULPSZrrbk, VMULPSZrrk,
                                              VMULPDZrrbk, VMULPDZrrk,
                                              VMULSSZrrbk_Int, VMULSSZrrk_Int,
                                              VMULSDZrrbk_Int, VMULSDZrrk_Int,
                                              VPMULDQZrrk, VPMULDQZ128rrk, VPMULDQZ256rrk,
                                              VPMULHRSWZrrk, VPMULHRSWZ128rrk, VPMULHRSWZ256rrk,
                                              VPMULHUWZrrk, VPMULHUWZ128rrk, VPMULHUWZ256rrk,
                                              VPMULHWZrrk, VPMULHWZ128rrk, VPMULHWZ256rrk,
                                              VPMULLDZrrk, VPMULLDZ128rrk, VPMULLDZ256rrk,
                                              VPMULLQZrr, VPMULLQZrrkz,
                                              VPMULLQZ128rr, VPMULLQZ128rrkz,
                                              VPMULLQZ256rr, VPMULLQZ256rrkz,
                                              VPMULLWZrrk, VPMULLWZ128rrk, VPMULLWZ256rrk,
                                              VPMULUDQZrrk, VPMULUDQZ128rrk, VPMULUDQZ256rrk)>;

def 4GM7WriteVMULZ_VPMULZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVMULZ_VPMULZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVMULZ_VPMULZLd], (instrs VMULPSZ128rmk, VMULPDZ128rmk,
                                                VMULPSZ256rmk, VMULPDZ256rmk,
                                                VMULPSZrmbk, VMULPSZrmk,
                                                VMULPDZrmbk, VMULPDZrmk,
                                                VPMULDQZrmk, VPMULDQZ128rmk, VPMULDQZ256rmk,
                                                VPMULHRSWZrmk, VPMULHRSWZ128rmk, VPMULHRSWZ256rmk,
                                                VPMULHUWZrmk, VPMULHUWZ128rmk, VPMULHUWZ256rmk,
                                                VPMULHWZrmk, VPMULHWZ128rmk, VPMULHWZ256rmk,
                                                VPMULLDZrmk, VPMULLDZ128rmk, VPMULLDZ256rmk,
                                                VPMULLQZrm, VPMULLQZrmkz,
                                                VPMULLQZ128rm, VPMULLQZ128rmkz,
                                                VPMULLQZ256rm, VPMULLQZ256rmkz,
                                                VPMULLWZrmk, VPMULLWZ128rmk, VPMULLWZ256rmk,
                                                VPMULUDQZrmk, VPMULUDQZ128rmk, VPMULUDQZ256rmk)>;

def 4GM7WriteVPMULLQZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULLQZrrk], (instrs VPMULLQZrrk, VPMULLQZ128rrk, VPMULLQZ256rrk)>;

def 4GM7WriteVPMULLQZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMULLQZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULLQZrmk], (instrs VPMULLQZrmk, VPMULLQZ128rmk, VPMULLQZ256rmk)>;


// VPMULTISHIFTQB
def 4GM7WriteVPMULTISHIFTQBZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULTISHIFTQBZ], (instrs VPMULTISHIFTQBZ128rr, VPMULTISHIFTQBZ128rrkz,
                                                 VPMULTISHIFTQBZ256rr, VPMULTISHIFTQBZ256rrkz)>;

def 4GM7WriteVPMULTISHIFTQBZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMULTISHIFTQBZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULTISHIFTQBZLd], (instrs VPMULTISHIFTQBZ128rm, VPMULTISHIFTQBZ128rmkz,
                                                   VPMULTISHIFTQBZ256rm, VPMULTISHIFTQBZ256rmkz)>;

def 4GM7WriteVPMULTISHIFTQBZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULTISHIFTQBZrrk], (instrs VPMULTISHIFTQBZ128rrk, VPMULTISHIFTQBZ256rrk,
                                                    VPMULTISHIFTQBZrr, VPMULTISHIFTQBZrrk, VPMULTISHIFTQBZrrkz)>;

def 4GM7WriteVPMULTISHIFTQBZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMULTISHIFTQBZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMULTISHIFTQBZrmk], (instrs VPMULTISHIFTQBZ128rmk, VPMULTISHIFTQBZ256rmk,
                                                    VPMULTISHIFTQBZrm, VPMULTISHIFTQBZrmk, VPMULTISHIFTQBZrmkz)>;

// VPMADDUBSWZ
def 4GM7WriteVPMADDUBSWZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDUBSWZrrk], (instrs VPMADDUBSWZrrk, VPMADDUBSWZ128rrk, VPMADDUBSWZ256rrk)>;

def 4GM7WriteVPMADDUBSWZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPMADDUBSWZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPMADDUBSWZrmk], (instrs VPMADDUBSWZrmk, VPMADDUBSWZ128rmk, VPMADDUBSWZ256rmk)>;

// FADD rrb rrbk
def 4GM7WriteFADD:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteFADD], (instregex "VADD(S|P)(S|D)(Z|Z128|Z256)rr(bk|k)$",
                                         "VSUB(S|P)(S|D)(Z|Z128|Z256)rr(bk|k)$")>;

def 4GM7WriteFADDLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteFADD.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteFADDLd], (instregex "VADD(S|P)(S|D)(Z|Z128|Z256)rm(bk|k)$",
                                           "VSUB(S|P)(S|D)(Z|Z128|Z256)rm(bk|k)$")>;

// FADD rrb rrbk Int
def 4GM7WriteFADD_Int:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteFADD_Int], (instregex "VADD(SD|SS)Zrr(bk|k)_Int",
                                             "VSUB(SD|SS)Zrr(bk|k)_Int")>;

def 4GM7WriteFADDLd_Int:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteFADD_Int.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteFADDLd_Int], (instregex "VADD(SD|SS)Zrm(bk|k)_Int",
                                               "VSUB(SD|SS)Zrm(bk|k)_Int")>;

// RCP14
def 4GM7WriteVRCP14:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14], (instrs VRCP14SSZrr, VRCP14SSZrrk, VRCP14SSZrrkz,
                                        VRCP14SDZrr, VRCP14SDZrrk, VRCP14SDZrrkz)>;

def 4GM7WriteVRCP14Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRCP14.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14Ld], (instrs VRCP14SDZrm, VRCP14SDZrmk, VRCP14SDZrmkz,
                                          VRCP14SSZrm, VRCP14SSZrmk, VRCP14SSZrmkz)>;

def 4GM7WriteVRCP14X:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14X], (instrs VRCP14PSZ128r, VRCP14PSZ128rk, VRCP14PSZ128rkz,
                                         VRCP14PDZ128r, VRCP14PDZ128rk, VRCP14PDZ128rkz)>;

def 4GM7WriteVRCP14XLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRCP14X.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14XLd], (instrs VRCP14PDZ128m, VRCP14PDZ128mb, VRCP14PDZ128mbk,
                                           VRCP14PDZ128mbkz, VRCP14PDZ128mk, VRCP14PDZ128mkz,
                                           VRCP14PSZ128m, VRCP14PSZ128mb, VRCP14PSZ128mbk,
                                           VRCP14PSZ128mbkz, VRCP14PSZ128mk, VRCP14PSZ128mkz)>;

def 4GM7WriteVRCP14Y:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14Y], (instrs VRCP14PSZ256r, VRCP14PSZ256rk, VRCP14PSZ256rkz,
                                         VRCP14PDZ256r, VRCP14PDZ256rk, VRCP14PDZ256rkz)>;

def 4GM7WriteVRCP14YLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRCP14Y.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14YLd], (instrs VRCP14PDZ256m, VRCP14PDZ256mb, VRCP14PDZ256mbk,
                                           VRCP14PDZ256mbkz, VRCP14PDZ256mk, VRCP14PDZ256mkz,
                                           VRCP14PSZ256m, VRCP14PSZ256mb, VRCP14PSZ256mbk,
                                           VRCP14PSZ256mbkz, VRCP14PSZ256mk, VRCP14PSZ256mkz)>;

def 4GM7WriteVRCP14Z:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14Z], (instrs VRCP14PDZr, VRCP14PDZrk, VRCP14PDZrkz,
                                         VRCP14PSZr, VRCP14PSZrk, VRCP14PSZrkz)>;

def 4GM7WriteVRCP14ZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRCP14Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRCP14ZLd], (instrs VRCP14PDZm, VRCP14PDZmb, VRCP14PDZmbk,
                                           VRCP14PDZmbkz, VRCP14PDZmk, VRCP14PDZmkz,
                                           VRCP14PSZm, VRCP14PSZmb, VRCP14PSZmbk,
                                           VRCP14PSZmbkz, VRCP14PSZmk, VRCP14PSZmkz)>;

// RSQRT14
def 4GM7WriteRSQRT14:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14], (instrs VRSQRT14SDZrr, VRSQRT14SDZrrk, VRSQRT14SDZrrkz,
                                         VRSQRT14SSZrr, VRSQRT14SSZrrk, VRSQRT14SSZrrkz)>;

def 4GM7WriteRSQRT14Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRSQRT14.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14Ld], (instrs VRSQRT14SDZrm, VRSQRT14SDZrmk, VRSQRT14SDZrmkz,
                                           VRSQRT14SSZrm, VRSQRT14SSZrmk, VRSQRT14SSZrmkz)>;

def 4GM7WriteRSQRT14X:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14X], (instrs VRSQRT14PDZ128r, VRSQRT14PDZ128rk, VRSQRT14PDZ128rkz,
                                          VRSQRT14PSZ128r, VRSQRT14PSZ128rk, VRSQRT14PSZ128rkz)>;

def 4GM7WriteRSQRT14XLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRSQRT14X.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14XLd], (instrs VRSQRT14PDZ128m, VRSQRT14PDZ128mb, VRSQRT14PDZ128mbk,
                                            VRSQRT14PDZ128mbkz, VRSQRT14PDZ128mk, VRSQRT14PDZ128mkz,
                                            VRSQRT14PSZ128m, VRSQRT14PSZ128mb, VRSQRT14PSZ128mbk,
                                            VRSQRT14PSZ128mbkz, VRSQRT14PSZ128mk, VRSQRT14PSZ128mkz)>;

def 4GM7WriteRSQRT14Y:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14Y], (instrs VRSQRT14PDZ256r, VRSQRT14PDZ256rk, VRSQRT14PDZ256rkz,
                                          VRSQRT14PSZ256r, VRSQRT14PSZ256rk, VRSQRT14PSZ256rkz)>;

def 4GM7WriteRSQRT14YLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRSQRT14Y.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14YLd], (instrs VRSQRT14PDZ256m, VRSQRT14PDZ256mb, VRSQRT14PDZ256mbk,
                                            VRSQRT14PDZ256mbkz, VRSQRT14PDZ256mk, VRSQRT14PDZ256mkz,
                                            VRSQRT14PSZ256m, VRSQRT14PSZ256mb, VRSQRT14PSZ256mbk,
                                            VRSQRT14PSZ256mbkz, VRSQRT14PSZ256mk, VRSQRT14PSZ256mkz)>;

def 4GM7WriteRSQRT14Z:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14Z], (instrs VRSQRT14PDZr, VRSQRT14PDZrk, VRSQRT14PDZrkz,
                                          VRSQRT14PSZr, VRSQRT14PSZrk, VRSQRT14PSZrkz)>;

def 4GM7WriteRSQRT14ZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteRSQRT14Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteRSQRT14ZLd], (instrs VRSQRT14PDZm, VRSQRT14PDZmb, VRSQRT14PDZmbk,
                                            VRSQRT14PDZmbkz, VRSQRT14PDZmk, VRSQRT14PDZmkz,
                                            VRSQRT14PSZm, VRSQRT14PSZmb, VRSQRT14PSZmbk,
                                            VRSQRT14PSZmbkz, VRSQRT14PSZmk, VRSQRT14PSZmkz)>;

defm : 4GM7WriteResFPPair<WriteFCmp, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmpX, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmpY, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmpZ, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmp64, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmp64X, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmp64Y, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCmp64Z, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFCom, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteFComX, [4GM7FPU], 1, [2], 2>;
defm : 4GM7WriteResFPPair<WriteFBlend, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFBlendY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFBlendZ, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFVarBlend, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFVarBlendY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFVarBlendZ, [4GM7FPU], 1>;

def 4GM7WriteFCOMI_FCOMIP:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteFCOMI_FCOMIP], (instrs COM_FIPr, COM_FIr)>;

// VMAX VMIN
def 4GM7WriteVMAXXY_VMINXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVMAXXY_VMINXY], (instrs VMAXPSZ128rrk, VMAXPDZ128rrk,
                                               VMINPSZ128rrk, VMINPDZ128rrk,
                                               VMAXCPSZ128rrk, VMAXCPDZ128rrk,
                                               VMINCPSZ128rrk, VMINCPDZ128rrk,
                                               VMAXPSZ256rrk, VMAXPDZ256rrk,
                                               VMINPSZ256rrk, VMINPDZ256rrk,
                                               VMAXCPSZ256rrk, VMAXCPDZ256rrk,
                                               VMINCPSZ256rrk, VMINCPDZ256rrk)>;

def 4GM7WriteVMAXXY_VMINXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVMAXXY_VMINXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVMAXXY_VMINXYLd], (instrs VMAXPSZ128rmk, VMAXPDZ128rmk,
                                                 VMINPSZ128rmk, VMINPDZ128rmk,
                                                 VMAXCPSZ128rmk, VMAXCPDZ128rmk,
                                                 VMINCPSZ128rmk, VMINCPDZ128rmk,
                                                 VMAXPSZ256rmk, VMAXPDZ256rmk,
                                                 VMINPSZ256rmk, VMINPDZ256rmk,
                                                 VMAXCPSZ256rmk, VMAXCPDZ256rmk,
                                                 VMINCPSZ256rmk, VMINCPDZ256rmk)>;

def 4GM7WriteVMAXZ_VMINZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVMAXZ_VMINZ], (instrs VMAXPSZrrk, VMAXPSZrrbk,
                                             VMAXPDZrrk, VMAXPDZrrbk,
                                             VMINPSZrrk, VMINPSZrrbk,
                                             VMINPDZrrk, VMINPDZrrbk,
                                             VMAXCPSZrrk, VMAXCPDZrrk,
                                             VMINCPSZrrk, VMINCPDZrrk,
                                             VMAXSSZrrk_Int, VMAXSSZrrbk_Int,
                                             VMAXSDZrrk_Int, VMAXSDZrrbk_Int,
                                             VMINSSZrrk_Int, VMINSSZrrbk_Int,
                                             VMINSDZrrk_Int, VMINSDZrrbk_Int)>;

def 4GM7WriteVMAXZ_VMINZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVMAXZ_VMINZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVMAXZ_VMINZLd], (instrs VMAXPSZrmk, VMAXPSZrmbk,
                                               VMAXPDZrmk, VMAXPDZrmbk,
                                               VMINPSZrmk, VMINPSZrmbk,
                                               VMINPDZrmk, VMINPDZrmbk,
                                               VMAXCPSZrmk, VMAXCPDZrmk,
                                               VMINCPSZrmk, VMINCPDZrmk,
                                               VMAXSSZrmk_Int, VMAXSDZrmk_Int,
                                               VMINSSZrmk_Int, VMINSDZrmk_Int)>;

// VBLEND
def 4GM7WriteVBLENDrrk:SchedWriteRes<[4GM7FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBLENDrrk], (instregex "VPBLENDM(B|D|W|Q)Z(128|256)?rrk$",
                                              "VBLENDM(PS|PD)Zrrk$")>;

def 4GM7WriteVBLENDrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBLENDrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBLENDrmk], (instregex "VPBLENDM(B|D|W|Q)Z(128|256)?rmk$",
                                              "VBLENDM(PS|PD)Zrmk$")>;

defm : 4GM7WriteResFPPair<WriteCvtSS2I, [4GM7FPU13], 2, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtPS2I, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPS2IY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPS2IZ, [4GM7FPU13], 4, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtSD2I, [4GM7FPU13], 2, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtPD2I, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPD2IY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPD2IZ, [4GM7FPU13], 8, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtSS2SD, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPS2PD, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPS2PDY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPS2PDZ, [4GM7FPU13], 5, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtSD2SS, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPD2PS, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPD2PSY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPD2PSZ, [4GM7FPU13], 8, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtI2SS, [4GM7FPU13], 4, [3], 2>;
defm : 4GM7WriteResFPPair<WriteCvtI2PS, [4GM7FPU13], 4>;
defm : 4GM7WriteResFPPair<WriteCvtI2PSY, [4GM7FPU13], 4>;
defm : 4GM7WriteResFPPair<WriteCvtI2PSZ, [4GM7FPU13], 4, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtI2SD, [4GM7FPU13], 4, [3], 2>;
defm : 4GM7WriteResFPPair<WriteCvtI2PD, [4GM7FPU13], 4>;
defm : 4GM7WriteResFPPair<WriteCvtI2PDY, [4GM7FPU13], 4>;
defm : 4GM7WriteResFPPair<WriteCvtI2PDZ, [4GM7FPU13], 5, [2], 2>;
defm : 4GM7WriteResFPPair<WriteCvtPH2PS, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPH2PSY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteCvtPH2PSZ, [4GM7FPU13], 5, [2], 2>;
defm : 4GM7WriteRes<WriteCvtPS2PH, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteRes<WriteCvtPS2PHY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteRes<WriteCvtPS2PHZ, [4GM7FPU13], 8, [2], 2>;
defm : 4GM7WriteRes<WriteCvtPS2PHSt, [4GM7FPU13, 4GM7AGU, 4GM7Store], !add(4, C864GM7Model.StoreLatency), [1, 1, 1], 2>;
defm : 4GM7WriteRes<WriteCvtPS2PHYSt, [4GM7FPU13, 4GM7AGU, 4GM7Store], !add(4, C864GM7Model.StoreLatency), [1, 1, 1], 2>;
defm : 4GM7WriteRes<WriteCvtPS2PHZSt, [4GM7FPU13, 4GM7AGU, 4GM7Store], !add(8, C864GM7Model.StoreLatency), [2, 1, 1], 3>;

// CVTPD2QQ
def 4GM7WritePD2QQ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WritePD2QQ], (instregex "VCVT(T?)PD2(U?)QQ(Z|Z128|Z256)(rr$|rrkz)")>;

def 4GM7WritePD2QQLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePD2QQ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WritePD2QQLd], (instregex "VCVT(T?)PD2(U?)QQ(Z|Z128|Z256)(rm$|rmkz)")>;

// CVTSI2SS
def 4GM7WriteSI2SSrr:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSI2SSrr], (instregex "VCVT(U?)SI2SS(Z?)rr")>;

def 4GM7WriteSI2SSrm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSI2SSrr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSI2SSrm], (instregex "VCVT(U?)SI2SS(Z?)rm")>;

// SI2SD
def 4GM7WriteSI2SD:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSI2SD], (instregex "VCVT(U?)SI2SD(Z?)rr")>;

def 4GM7WriteSI2SDLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSI2SD.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSI2SDLd], (instregex "VCVT(U?)SI2SD(Z?)rm")>;

// CVT BF16
def 4GM7WriteCVTBF16:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16], (instrs VCVTNE2PS2BF16Z256rr, VCVTNE2PS2BF16Z256rrkz,
                                         VCVTNE2PS2BF16Zrrkz)>;

def 4GM7WriteCVTBF16Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteCVTBF16.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16Ld], (instrs VCVTNE2PS2BF16Z256rm, VCVTNE2PS2BF16Z256rmkz,
                                           VCVTNE2PS2BF16Zrmkz)>;

// CVT BF16
def 4GM7WriteCVTBF16rrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16rrk], (instrs VCVTNE2PS2BF16Z128rrk, VCVTNEPS2BF16Zrrk,
                                            VCVTNE2PS2BF16Zrrk)>;

def 4GM7WriteCVTBF16rmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteCVTBF16rrk.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16rmk], (instrs VCVTNE2PS2BF16Z128rmk, VCVTNEPS2BF16Zrmk)>;

// CVT BF16
def 4GM7WriteCVTBF16rr:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16rr], (instrs VCVTNE2PS2BF16Zrr)>;

def 4GM7WriteCVTBF16rm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteCVTBF16rr.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[4GM7WriteCVTBF16rm], (instrs VCVTNE2PS2BF16Zrm)>;

// VDPBF16
def 4GM7WriteVDPBF16:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 8;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVDPBF16], (instrs VDPBF16PSZr, VDPBF16PSZrkz,
                                         VDPBF16PSZ128r, VDPBF16PSZ128rkz,
                                         VDPBF16PSZ256r, VDPBF16PSZ256rkz)>;

def 4GM7WriteVDPBF16Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVDPBF16.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVDPBF16Ld], (instrs VDPBF16PSZm, VDPBF16PSZmkz,
                                           VDPBF16PSZ128m, VDPBF16PSZ128mkz,
                                           VDPBF16PSZ256m, VDPBF16PSZ256mkz)>;

// VDPBF16
def 4GM7WriteVDPBF16rk:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVDPBF16rk], (instrs VDPBF16PSZrk, VDPBF16PSZ128rk, VDPBF16PSZ256rk)>;

def 4GM7WriteVDPBF16mk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVDPBF16rk.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVDPBF16mk], (instrs VDPBF16PSZmk, VDPBF16PSZ128mk, VDPBF16PSZ256mk)>;

// VCVT rrk
def 4GM7WriteVCVTrrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTrrk], (instrs VCVTPD2PSZ128rrk, VCVTPD2PSZ128rrk, VCVTSD2SSZrrbk_Int,
                                        VCVTSD2SSZrrk_Int, VCVTSS2SDZrrbk_Int, VCVTSS2SDZrrk_Int,
                                        VCVTDQ2PDZ128rrk, VCVTUDQ2PDZ128rrk, VCVTDQ2PDZ256rrk,
                                        VCVTUDQ2PDZ256rrk, VCVTDQ2PSZ128rrk, VCVTUDQ2PSZ128rrk,
                                        VCVTDQ2PSZ256rrk, VCVTUDQ2PSZ256rrk, VCVTDQ2PSZrrbk,
                                        VCVTDQ2PSZrrk, VCVTUDQ2PSZrrbk, VCVTUDQ2PSZrrk,
                                        VCVTPD2DQZ128rrk, VCVTPD2UDQZ128rrk, VCVTTPD2DQZ128rrk,
                                        VCVTTPD2UDQZ128rrk, VCVTPD2DQZ256rrk, VCVTPD2UDQZ256rrk,
                                        VCVTTPD2DQZ256rrk, VCVTTPD2UDQZ256rrk, VCVTPD2DQZrrbk,
                                        VCVTPD2DQZrrk, VCVTPD2UDQZrrbk, VCVTPD2UDQZrrk,
                                        VCVTTPD2DQZrrbk, VCVTTPD2DQZrrk, VCVTTPD2UDQZrrbk,
                                        VCVTTPD2UDQZrrk, VCVTPD2PSZ256rrk, VCVTPD2PSZrrbk,
                                        VCVTPD2PSZrrk, VCVTPD2QQZ128rrk, VCVTPD2UQQZ128rrk,
                                        VCVTTPD2QQZ128rrk, VCVTTPD2UQQZ128rrk, VCVTPD2QQZ256rrk,
                                        VCVTPD2UQQZ256rrk, VCVTTPD2QQZ256rrk, VCVTTPD2UQQZ256rrk,
                                        VCVTPD2QQZrrbk, VCVTPD2QQZrrk, VCVTPD2UQQZrrbk,
                                        VCVTPD2UQQZrrk, VCVTTPD2QQZrrbk, VCVTTPD2QQZrrk,
                                        VCVTTPD2UQQZrrbk, VCVTTPD2UQQZrrk, VCVTPH2PSZ128rrk,
                                        VCVTPH2PSZ256rrk, VCVTPS2DQZ256rrk, VCVTPS2UDQZ256rrk,
                                        VCVTTPS2DQZ256rrk, VCVTTPS2UDQZ256rrk, VCVTPS2DQZ128rrk,
                                        VCVTPS2UDQZ128rrk, VCVTTPS2DQZ128rrk, VCVTTPS2UDQZ128rrk,
                                        VCVTPS2DQZrrbk, VCVTPS2DQZrrk, VCVTPS2UDQZrrbk,
                                        VCVTPS2UDQZrrk, VCVTTPS2DQZrrbk, VCVTTPS2DQZrrk,
                                        VCVTTPS2UDQZrrbk, VCVTTPS2UDQZrrk, VCVTPS2PDZ256rrk,
                                        VCVTPS2PHZ128rrk, VCVTPS2PHZ256rrk, VCVTPS2PHZrrbk,
                                        VCVTPS2PHZrrk, VCVTPS2QQZ128rrk, VCVTPS2UQQZ128rrk,
                                        VCVTTPS2QQZ128rrk, VCVTTPS2UQQZ128rrk, VCVTPS2QQZ256rrk,
                                        VCVTPS2UQQZ256rrk, VCVTTPS2QQZ256rrk, VCVTTPS2UQQZ256rrk,
                                        VCVTQQ2PDZ128rrk, VCVTUQQ2PDZ128rrk, VCVTQQ2PDZ256rrk,
                                        VCVTUQQ2PDZ256rrk, VCVTQQ2PDZrrbk, VCVTQQ2PDZrrk,
                                        VCVTUQQ2PDZrrbk, VCVTUQQ2PDZrrk, VCVTQQ2PSZ128rrk,
                                        VCVTUQQ2PSZ128rrk, VCVTQQ2PSZ256rrk, VCVTUQQ2PSZ256rrk,
                                        VCVTQQ2PSZrrbk, VCVTQQ2PSZrrk, VCVTUQQ2PSZrrbk,
                                        VCVTUQQ2PSZrrk, VCVTPS2PDZ128rrk, VCVTPS2QQZrr,
                                        VCVTPS2UQQZrr, VCVTTPS2QQZrr, VCVTTPS2UQQZrr,
                                        VCVTPS2QQZrrb, VCVTPS2QQZrrbkz, VCVTPS2QQZrrkz,
                                        VCVTPS2UQQZrrb, VCVTPS2UQQZrrbkz, VCVTPS2UQQZrrkz,
                                        VCVTTPS2QQZrrb, VCVTTPS2QQZrrbkz, VCVTTPS2QQZrrkz,
                                        VCVTTPS2UQQZrrb, VCVTTPS2UQQZrrbkz, VCVTTPS2UQQZrrkz,
                                        VCVTPS2QQZrrbk, VCVTPS2QQZrrk, VCVTPS2UQQZrrbk,
                                        VCVTPS2UQQZrrk, VCVTTPS2QQZrrbk, VCVTTPS2QQZrrk,
                                        VCVTTPS2UQQZrrbk, VCVTTPS2UQQZrrk)>;

def 4GM7WriteVCVTrrkLD:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCVTrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTrrkLD], (instrs VCVTPD2PSZ128rmk, VCVTPD2PSZ128rmk,
                                           VCVTSD2SSZrmk_Int, VCVTSS2SDZrmk_Int,
                                           VCVTDQ2PDZ128rmk, VCVTUDQ2PDZ128rmk, VCVTDQ2PDZ256rmk,
                                           VCVTUDQ2PDZ256rmk, VCVTDQ2PSZ128rmk, VCVTUDQ2PSZ128rmk,
                                           VCVTDQ2PSZ256rmk, VCVTUDQ2PSZ256rmk, VCVTDQ2PSZrmbk,
                                           VCVTDQ2PSZrmk, VCVTUDQ2PSZrmbk, VCVTUDQ2PSZrmk,
                                           VCVTPD2DQZ128rmk, VCVTPD2UDQZ128rmk, VCVTTPD2DQZ128rmk,
                                           VCVTTPD2UDQZ128rmk, VCVTPD2DQZ256rmk, VCVTPD2UDQZ256rmk,
                                           VCVTTPD2DQZ256rmk, VCVTTPD2UDQZ256rmk, VCVTPD2DQZrmbk,
                                           VCVTPD2DQZrmk, VCVTPD2UDQZrmbk, VCVTPD2UDQZrmk,
                                           VCVTTPD2DQZrmbk, VCVTTPD2DQZrmk, VCVTTPD2UDQZrmbk,
                                           VCVTTPD2UDQZrmk, VCVTPD2PSZ256rmk, VCVTPD2PSZrmbk,
                                           VCVTPD2PSZrmk, VCVTPD2QQZ128rmk, VCVTPD2UQQZ128rmk,
                                           VCVTTPD2QQZ128rmk, VCVTTPD2UQQZ128rmk, VCVTPD2QQZ256rmk,
                                           VCVTPD2UQQZ256rmk, VCVTTPD2QQZ256rmk, VCVTTPD2UQQZ256rmk,
                                           VCVTPD2QQZrmbk, VCVTPD2QQZrmk, VCVTPD2UQQZrmbk,
                                           VCVTPD2UQQZrmk, VCVTTPD2QQZrmbk, VCVTTPD2QQZrmk,
                                           VCVTTPD2UQQZrmbk, VCVTTPD2UQQZrmk, VCVTPH2PSZ128rmk,
                                           VCVTPH2PSZ256rmk, VCVTPS2DQZ256rmk, VCVTPS2UDQZ256rmk,
                                           VCVTTPS2DQZ256rmk, VCVTTPS2UDQZ256rmk, VCVTPS2DQZ128rmk,
                                           VCVTPS2UDQZ128rmk, VCVTTPS2DQZ128rmk, VCVTTPS2UDQZ128rmk,
                                           VCVTPS2DQZrmbk, VCVTPS2DQZrmk, VCVTPS2UDQZrmbk,
                                           VCVTPS2UDQZrmk, VCVTTPS2DQZrmbk, VCVTTPS2DQZrmk,
                                           VCVTTPS2UDQZrmbk, VCVTTPS2UDQZrmk, VCVTPS2PDZ256rmk,
                                           VCVTPS2QQZ128rmk, VCVTPS2UQQZ128rmk,
                                           VCVTTPS2QQZ128rmk, VCVTTPS2UQQZ128rmk, VCVTPS2QQZ256rmk,
                                           VCVTPS2UQQZ256rmk, VCVTTPS2QQZ256rmk, VCVTTPS2UQQZ256rmk,
                                           VCVTQQ2PDZ128rmk, VCVTUQQ2PDZ128rmk, VCVTQQ2PDZ256rmk,
                                           VCVTUQQ2PDZ256rmk, VCVTQQ2PDZrmbk, VCVTQQ2PDZrmk,
                                           VCVTUQQ2PDZrmbk, VCVTUQQ2PDZrmk, VCVTQQ2PSZ128rmk,
                                           VCVTUQQ2PSZ128rmk, VCVTQQ2PSZ256rmk, VCVTUQQ2PSZ256rmk,
                                           VCVTQQ2PSZrmbk, VCVTQQ2PSZrmk, VCVTUQQ2PSZrmbk,
                                           VCVTUQQ2PSZrmk, VCVTPS2PDZ128rmk, VCVTPS2QQZrm,
                                           VCVTPS2UQQZrm, VCVTTPS2QQZrm, VCVTTPS2UQQZrm,
                                           VCVTPS2QQZrmb, VCVTPS2QQZrmbkz, VCVTPS2QQZrmkz,
                                           VCVTPS2UQQZrmb, VCVTPS2UQQZrmbkz, VCVTPS2UQQZrmkz,
                                           VCVTTPS2QQZrmb, VCVTTPS2QQZrmbkz, VCVTTPS2QQZrmkz,
                                           VCVTTPS2UQQZrmb, VCVTTPS2UQQZrmbkz, VCVTTPS2UQQZrmkz,
                                           VCVTPS2QQZrmbk, VCVTPS2QQZrmk, VCVTPS2UQQZrmbk,
                                           VCVTPS2UQQZrmk, VCVTTPS2QQZrmbk, VCVTTPS2QQZrmk,
                                           VCVTTPS2UQQZrmbk, VCVTTPS2UQQZrmk)>;

// VCVT
def 4GM7WriteVCVTrr:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTrr], (instrs VCVTPD2QQZrrb, VCVTPD2QQZrrbkz, VCVTPD2UQQZrrb,
                                        VCVTPD2UQQZrrbkz, VCVTTPD2QQZrrb, VCVTTPD2QQZrrbkz,
                                        VCVTTPD2UQQZrrb, VCVTTPD2UQQZrrbkz, VCVTQQ2PDZrr,
                                        VCVTQQ2PDZrrb, VCVTQQ2PDZrrbkz, VCVTQQ2PDZrrkz,
                                        VCVTUQQ2PDZrr, VCVTUQQ2PDZrrb, VCVTUQQ2PDZrrbkz,
                                        VCVTUQQ2PDZrrkz)>;

def 4GM7WriteVCVTrm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCVTrr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTrm], (instrs VCVTPD2QQZrmb, VCVTPD2QQZrmbkz, VCVTPD2UQQZrmb,
                                        VCVTPD2UQQZrmbkz, VCVTTPD2QQZrmb, VCVTTPD2QQZrmbkz,
                                        VCVTTPD2UQQZrmb, VCVTTPD2UQQZrmbkz, VCVTQQ2PDZrm,
                                        VCVTQQ2PDZrmb, VCVTQQ2PDZrmbkz, VCVTQQ2PDZrmkz,
                                        VCVTUQQ2PDZrm, VCVTUQQ2PDZrmb, VCVTUQQ2PDZrmbkz,
                                        VCVTUQQ2PDZrmkz)>;

// VCVTQQ2PS
def 4GM7WriteVCVTQQ2PS:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTQQ2PS], (instrs VCVTQQ2PSZrr, VCVTQQ2PSZrrb, VCVTQQ2PSZrrbkz, VCVTQQ2PSZrrkz,
                                           VCVTUQQ2PSZrr, VCVTUQQ2PSZrrb, VCVTUQQ2PSZrrbkz, VCVTUQQ2PSZrrkz)>;

def 4GM7WriteVCVTQQ2PSLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCVTQQ2PS.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCVTQQ2PSLd], (instrs VCVTQQ2PSZrm, VCVTQQ2PSZrmb, VCVTQQ2PSZrmbkz, VCVTQQ2PSZrmkz,
                                             VCVTUQQ2PSZrm, VCVTUQQ2PSZrmb, VCVTUQQ2PSZrmbkz, VCVTUQQ2PSZrmkz)>;


// VCVTNE2PS2BF16Z128
def 4GM7WriteVCVTNE2PS2BF16Z128:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 9;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVCVTNE2PS2BF16Z128], (instrs VCVTNE2PS2BF16Z128rr,
                                                    VCVTNE2PS2BF16Z128rrkz)>;

def 4GM7WriteVCVTNE2PS2BF16Z128Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCVTNE2PS2BF16Z128.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVCVTNE2PS2BF16Z128Ld], (instrs VCVTNE2PS2BF16Z128rm,
                                                      VCVTNE2PS2BF16Z128rmkz)>;

// VCVTNE2PS2BF16Z VCVTNE2PS2BF16Z256 VCVTNEPS2BF16
def 4GM7WriteVCVTNEPS2BF16:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVCVTNEPS2BF16], (instrs VCVTNE2PS2BF16Z256rrk,
                                               VCVTNEPS2BF16Z128rrk, VCVTNEPS2BF16Z256rrk)>;

def 4GM7WriteVCVTNEPS2BF16Ld:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCVTNEPS2BF16.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVCVTNEPS2BF16Ld], (instrs VCVTNE2PS2BF16Z256rmk,
                                                 VCVTNEPS2BF16Z128rmk, VCVTNEPS2BF16Z256rmk)>;


defm : 4GM7WriteResFPPair<WriteFSign, [4GM7FPU1], 1>;
defm : 4GM7WriteResFPPair<WriteFRnd, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFRndY, [4GM7FPU13], 4, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFRndZ, [4GM7FPU13], 4, [2], 2>;
defm : 4GM7WriteResFPPair<WriteFLogic, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFLogicY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFLogicZ, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteFTest, [4GM7FPU13], 1, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFTestY, [4GM7FPU13], 1, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFTestZ, [4GM7FPU13], 1, [2], 2>;
defm : 4GM7WriteResFPPair<WriteFShuffle, [4GM7FPU13], 1, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFShuffleY, [4GM7FPU13], 1, [1], 1>;
defm : 4GM7WriteResFPPair<WriteFShuffleZ, [4GM7FPU13], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteFVarShuffle, [4GM7FPU02], 3, [4], 1>;
defm : 4GM7WriteResFPPair<WriteFVarShuffleY,[4GM7FPU02], 3, [4], 1>;
defm : 4GM7WriteResFPPair<WriteFVarShuffleZ,[4GM7FPU02], 5, [8], 1>;
defm : 4GM7WriteResFPPair<WriteFShuffle256, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteFVarShuffle256, [4GM7FPU02], 3>;

defm : 4GM7WriteRes<WriteVecLoad, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteVecLoadX, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteVecLoadY, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteVecLoadNT, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteVecLoadNTY, [4GM7AGU, 4GM7Load], !add(C864GM7Model.VecLoadLatency, 1), [1, 1], 1>;
defm : 4GM7WriteRes<WriteVecMaskedLoad, [4GM7AGU, 4GM7Load, 4GM7FPU01], !add(C864GM7Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : 4GM7WriteRes<WriteVecMaskedLoadY, [4GM7AGU, 4GM7Load, 4GM7FPU01], !add(C864GM7Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : 4GM7WriteRes<WriteVecStore, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteVecStoreX, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteVecStoreY, [4GM7AGU, 4GM7Store], C864GM7Model.StoreLatency>;
defm : 4GM7WriteRes<WriteVecStoreNT, [4GM7FPU1, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecStoreNTY, [4GM7FPU1, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecMaskedStore32, [4GM7FPU01, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecMaskedStore32Y, [4GM7FPU01, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecMaskedStore64, [4GM7FPU01, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecMaskedStore64Y, [4GM7FPU01, 4GM7AGU, 4GM7Store], !add(1, C864GM7Model.StoreLatency)>;
defm : 4GM7WriteRes<WriteVecMoveToGpr, [4GM7FPU13], 1, [2], 1>;
defm : 4GM7WriteRes<WriteVecMoveFromGpr, [4GM7FPU], 1, [4], 2>;
defm : 4GM7WriteRes<WriteEMMS, [], 0, [], 1>;

defm : 4GM7WriteResFPPair<WriteVecShift, [4GM7FPU02], 1>; // from llvm-exegesis
defm : 4GM7WriteResFPPair<WriteVecShiftX, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVecShiftY, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVecShiftZ, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVecShiftImm, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteVecShiftImmX, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteVecShiftImmY, [4GM7FPU02], 1>;
defm : 4GM7WriteResFPPair<WriteVecShiftImmZ, [4GM7FPU02], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVarVecShift, [4GM7FPU02], 1, [4], 1>;
defm : 4GM7WriteResFPPair<WriteVarVecShiftY, [4GM7FPU02], 1, [4], 1>;
defm : 4GM7WriteResFPPair<WriteVarVecShiftZ, [4GM7FPU02], 1, [8], 1>;
defm : 4GM7WriteResFPPair<WriteVecLogic, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecLogicX, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecLogicY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecLogicZ, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecTest, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteVecTestY, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteVecTestZ, [4GM7FPU13], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVecALU, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecALUX, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecALUY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVecALUZ, [4GM7FPU], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVecIMul, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVecIMulX, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVecIMulY, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVecIMulZ, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteRes<WriteCLMul, [4GM7FPU02], 3>;
defm : 4GM7WriteRes<WriteCLMulLd, [4GM7AGU, 4GM7Load, 4GM7FPU02], !add(C864GM7Model.VecLoadLatency, 3)>;
defm : 4GM7WriteResFPPair<WritePMULLD, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WritePMULLDY, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WritePMULLDZ, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WriteShuffle, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteShuffleX, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteShuffleY, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteShuffleZ, [4GM7FPU13], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVarShuffle, [4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteVarShuffleX,[4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteVarShuffleY,[4GM7FPU13], 1>;
defm : 4GM7WriteResFPPair<WriteVarShuffleZ,[4GM7FPU13], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteBlend, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteBlendY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteBlendZ, [4GM7FPU], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteVarBlend, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVarBlendY, [4GM7FPU], 1>;
defm : 4GM7WriteResFPPair<WriteVarBlendZ, [4GM7FPU], 1, [2], 1>;
defm : 4GM7WriteResFPPair<WriteShuffle256, [4GM7FPU01], 3>;
defm : 4GM7WriteResFPPair<WriteVPMOV256, [4GM7FPU02], 3>;
defm : 4GM7WriteResFPPair<WriteVarShuffle256, [4GM7FPU3, 4GM7FPU02], 2, [2, 2], 2>;
defm : 4GM7WriteResFPPair<WritePSADBW, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WritePSADBWX, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WritePSADBWY, [4GM7FPU02], 3, [2], 1>;
defm : 4GM7WriteResFPPair<WritePSADBWZ, [4GM7FPU02], 3, [4], 1>;
defm : 4GM7WriteResFPPair<WriteMPSAD, [4GM7FPU], 4, [8], 1>;
defm : 4GM7WriteResFPPair<WriteMPSADY, [4GM7FPU], 4, [8], 1>;
defm : 4GM7WriteResFPPair<WriteMPSADZ, [4GM7FPU], 4, [16], 1>;
defm : 4GM7WriteResFPPair<WritePHMINPOS, [4GM7FPU02], 3, [2], 1>;

// EXTRQ INSERTQ
def 4GM7WriteEXTRQ_INSERTQ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXTRQ_INSERTQ], (instrs EXTRQ, EXTRQI, INSERTQ, INSERTQI)>;

// VPCMP
def 4GM7WriteVPCMPXY:SchedWriteRes<[4GM7FPU, 4GM7FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [2, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPCMPXY], (instregex "VPCMPEQ(B|D|Q|W)(Z128|Z256)rrk$",
                                            "VPCMPGT(B|D|Q|W)(Z128|Z256)rrk$",
                                            "VPCMP(U)?(B|D|Q|W)(Z128|Z256)rrik$")>;

def 4GM7WriteVPCMPXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCMPXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPCMPXYLd], (instregex "VPCMPEQ(B|D|Q|W)(Z128|Z256)rmk$",
                                              "VPCMPGT(B|D|Q|W)(Z128|Z256)rmk$",
                                              "VPCMP(U)?(B|D|Q|W)(Z128|Z256)rmik$")>;

def 4GM7WriteVPCMPZ:SchedWriteRes<[4GM7FPU, 4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [12, 6];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCMPZ], (instregex "VPCMPEQ(B|D|Q|W)Zrrk$",
                                           "VPCMPGT(B|D|Q|W)Zrrk$",
                                           "VPCMP(U)?(B|D|Q|W)Zrrik$")>;

def 4GM7WriteVPCMPZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCMPZ.Latency);
  let ReleaseAtCycles = [1, 1, 12, 6];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCMPZLd], (instregex "VPCMPEQ(B|D|Q|W)Zrmk$",
                                             "VPCMPGT(B|D|Q|W)Zrmk$",
                                             "VPCMP(U)?(B|D|Q|W)Zrmik$")>;

// VPSHUFBIT
def 4GM7WriteVPSHUFBIT:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPSHUFBIT], (instrs VPSHUFBITQMBZ128rrk, VPSHUFBITQMBZ256rrk)>;

def 4GM7WriteVPSHUFBITLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPSHUFBIT.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPSHUFBITLd], (instrs VPSHUFBITQMBZ128rmk, VPSHUFBITQMBZ256rmk)>;

def 4GM7WriteVPSHUFBITZ:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 10;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPSHUFBITZ], (instrs VPSHUFBITQMBZrrk)>;

def 4GM7WriteVPSHUFBITZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPSHUFBITZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 3;
}
def : InstRW<[4GM7WriteVPSHUFBITZLd], (instrs VPSHUFBITQMBZrmk)>;

// VSHUF VPSHUFB VPSHUFLW VPSHUFHW
def 4GM7WriteVPSHUFXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPSHUFXY], (instrs VSHUFPSZ128rrik, VSHUFPDZ128rrik,
                                          VSHUFPSZ256rrik, VSHUFPDZ256rrik,
                                          VPSHUFBZ128rrk, VPSHUFBZ128rrkz,
                                          VPSHUFHWZ128rik, VPSHUFHWZ128rikz,
                                          VPSHUFLWZ128rik, VPSHUFLWZ128rikz,
                                          VPSHUFBZ256rrk, VPSHUFBZ256rrkz,
                                          VPSHUFHWZ256rik, VPSHUFHWZ256rikz,
                                          VPSHUFLWZ256rik, VPSHUFLWZ256rikz)>;

def 4GM7WriteVPSHUFXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPSHUFXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPSHUFXYLd], (instrs VSHUFPSZ128rmik, VSHUFPDZ128rmik,
                                            VSHUFPSZ256rmik, VSHUFPDZ256rmik,
                                            VPSHUFBZ128rmk, VPSHUFBZ128rmkz,
                                            VPSHUFHWZ128mik, VPSHUFHWZ128mikz,
                                            VPSHUFLWZ128mik, VPSHUFLWZ128mikz,
                                            VPSHUFBZ256rmk, VPSHUFBZ256rmkz,
                                            VPSHUFHWZ256mik, VPSHUFHWZ256mikz,
                                            VPSHUFLWZ256mik, VPSHUFLWZ256mikz)>;

def 4GM7WriteVPSHUFZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPSHUFZ], (instrs VSHUFPSZrrik, VSHUFPDZrrik,
                                         VPSHUFBZrrk, VPSHUFBZrrkz,
                                         VPSHUFHWZrik, VPSHUFHWZrikz,
                                         VPSHUFLWZrik, VPSHUFLWZrikz)>;

def 4GM7WriteVPSHUFZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPSHUFZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPSHUFZLd], (instrs VSHUFPSZrmik, VSHUFPDZrmik,
                                           VPSHUFBZrmk, VPSHUFBZrmkz,
                                           VPSHUFHWZmik, VPSHUFHWZmikz,
                                           VPSHUFLWZmik, VPSHUFLWZmikz)>;

// VSHUFF VSHUFI
def 4GM7WriteVSHUFFY_VSHUFIY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSHUFFY_VSHUFIY], (instrs VSHUFF32X4Z256rrik, VSHUFF64X2Z256rrik,
                                                 VSHUFI32X4Z256rrik, VSHUFI64X2Z256rrik)>;

def 4GM7WriteVSHUFFY_VSHUFIYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSHUFFY_VSHUFIY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSHUFFY_VSHUFIYLd], (instrs VSHUFF32X4Z256rmik, VSHUFF64X2Z256rmik,
                                                   VSHUFI32X4Z256rmik, VSHUFI64X2Z256rmik)>;

def 4GM7WriteVSHUFFZ_VSHUFIZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVSHUFFZ_VSHUFIZ], (instrs VSHUFF32X4Zrri, VSHUFF32X4Zrrik, VSHUFF32X4Zrrikz,
                                                 VSHUFF64X2Zrri, VSHUFF64X2Zrrik, VSHUFF64X2Zrrikz,
                                                 VSHUFI32X4Zrri, VSHUFI32X4Zrrik, VSHUFI32X4Zrrikz,
                                                 VSHUFI64X2Zrri, VSHUFI64X2Zrrik, VSHUFI64X2Zrrikz)>;

def 4GM7WriteVSHUFFZ_VSHUFIZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSHUFFZ_VSHUFIZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVSHUFFZ_VSHUFIZLd], (instrs VSHUFF32X4Zrmi, VSHUFF32X4Zrmik, VSHUFF32X4Zrmikz,
                                                   VSHUFF64X2Zrmi, VSHUFF64X2Zrmik, VSHUFF64X2Zrmikz,
                                                   VSHUFI32X4Zrmi, VSHUFI32X4Zrmik, VSHUFI32X4Zrmikz,
                                                   VSHUFI64X2Zrmi, VSHUFI64X2Zrmik, VSHUFI64X2Zrmikz)>;

// UNPCK
def 4GM7WriteUNPCKXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteUNPCKXY], (instrs VUNPCKLPSZ128rrk, VUNPCKLPDZ128rrk,
                                         VUNPCKHPSZ128rrk, VUNPCKHPDZ128rrk,
                                         VUNPCKLPSZ256rrk, VUNPCKLPDZ256rrk,
                                         VUNPCKHPSZ256rrk, VUNPCKHPDZ256rrk,
                                         VPUNPCKHBWZ128rrk, VPUNPCKHBWZ128rrkz,
                                         VPUNPCKHWDZ128rrk, VPUNPCKHWDZ128rrkz,
                                         VPUNPCKLBWZ128rrk, VPUNPCKLBWZ128rrkz,
                                         VPUNPCKLWDZ128rrk, VPUNPCKLWDZ128rrkz,
                                         VPUNPCKHBWZ256rrk, VPUNPCKHBWZ256rrkz,
                                         VPUNPCKHWDZ256rrk, VPUNPCKHWDZ256rrkz,
                                         VPUNPCKLBWZ256rrk, VPUNPCKLBWZ256rrkz,
                                         VPUNPCKLWDZ256rrk, VPUNPCKLWDZ256rrkz)>;

def 4GM7WriteUNPCKXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteUNPCKXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteUNPCKXYLd], (instrs VUNPCKLPSZ128rmk, VUNPCKLPDZ128rmk,
                                           VUNPCKHPSZ128rmk, VUNPCKHPDZ128rmk,
                                           VUNPCKLPSZ256rmk, VUNPCKLPDZ256rmk,
                                           VUNPCKHPSZ256rmk, VUNPCKHPDZ256rmk,
                                           VPUNPCKHBWZ128rmk, VPUNPCKHBWZ128rmkz,
                                           VPUNPCKHWDZ128rmk, VPUNPCKHWDZ128rmkz,
                                           VPUNPCKLBWZ128rmk, VPUNPCKLBWZ128rmkz,
                                           VPUNPCKLWDZ128rmk, VPUNPCKLWDZ128rmkz,
                                           VPUNPCKHBWZ256rmk, VPUNPCKHBWZ256rmkz,
                                           VPUNPCKHWDZ256rmk, VPUNPCKHWDZ256rmkz,
                                           VPUNPCKLBWZ256rmk, VPUNPCKLBWZ256rmkz,
                                           VPUNPCKLWDZ256rmk, VPUNPCKLWDZ256rmkz)>;


def 4GM7WriteUNPCKZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteUNPCKZ], (instrs VUNPCKLPSZrrk, VUNPCKLPDZrrk,
                                        VUNPCKHPSZrrk, VUNPCKHPDZrrk,
                                        VPUNPCKHBWZrrk, VPUNPCKHBWZrrkz,
                                        VPUNPCKHWDZrrk, VPUNPCKHWDZrrkz,
                                        VPUNPCKLBWZrrk, VPUNPCKLBWZrrkz,
                                        VPUNPCKLWDZrrk, VPUNPCKLWDZrrkz)>;

def 4GM7WriteUNPCKZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteUNPCKZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteUNPCKZLd], (instrs VUNPCKLPSZrmk, VUNPCKLPDZrmk,
                                          VUNPCKHPSZrmk, VUNPCKHPDZrmk,
                                          VPUNPCKHBWZrmk, VPUNPCKHBWZrmkz,
                                          VPUNPCKHWDZrmk, VPUNPCKHWDZrmkz,
                                          VPUNPCKLBWZrmk, VPUNPCKLBWZrmkz,
                                          VPUNPCKLWDZrmk, VPUNPCKLWDZrmkz)>;

// POPCNT
def 4GM7WritePOPCNTZ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePOPCNTZ], (instrs VPOPCNTBZrrk, VPOPCNTBZrrkz,
                                         VPOPCNTWZrrk, VPOPCNTWZrrkz,
                                         VPOPCNTBZ128rrk, VPOPCNTBZ128rrkz,
                                         VPOPCNTWZ128rrk, VPOPCNTWZ128rrkz,
                                         VPOPCNTBZ256rrk, VPOPCNTBZ256rrkz,
                                         VPOPCNTWZ256rrk, VPOPCNTWZ256rrkz)>;

def 4GM7WritePOPCNTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePOPCNTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePOPCNTZLd], (instrs VPOPCNTBZrmk, VPOPCNTBZrmkz,
                                           VPOPCNTWZrmk, VPOPCNTWZrmkz,
                                           VPOPCNTBZ128rmk, VPOPCNTBZ128rmkz,
                                           VPOPCNTWZ128rmk, VPOPCNTWZ128rmkz,
                                           VPOPCNTBZ256rmk, VPOPCNTBZ256rmkz,
                                           VPOPCNTWZ256rmk, VPOPCNTWZ256rmkz)>;

// VPABS VPADDS VPADDUS VPAVG VPMAX VPMIN VPADD VPSUB VPTERNLOG
def 4GM7WriteVecALUZ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecALUZ], (instregex "VPABS(B|W)Z(128|256)?rrk$",
                                            "VPADD(U)?S(B|W)Z(128|256)?rrk$",
                                            "VPAVG(B|W)Z(128|256)?rrk$",
                                            "VPSUB(U)?S(B|W)Z(128|256)?rrk$",
                                            "VPMAX(SD|UD|SB|SW|UB|UW)Z(256)?rrk$",
                                            "VPMIN(SD|UD|SB|SW|UB|UW)Z(256)?rrk$",
                                            "VPADD(B|W)Z(128)?rrk$",
                                            "VPADD(D|Q)Zrr(k)?",
                                            "VPSUB(B|W)Z(128)?rrk$",
                                            "VPSUB(D|Q)Zrrk$",
                                            "VPTERNLOG(D|Q)Zrri(k)?")>;

def 4GM7WriteVecALUZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVecALUZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecALUZLd], (instregex "VPABS(B|W)Z(128|256)?rmk$",
                                              "VPADD(U)?S(B|W)Z(128|256)?rmk$",
                                              "VPAVG(B|W)Z(128|256)?rmk$",
                                              "VPSUB(U)?S(B|W)Z(128|256)?rmk$",
                                              "VPMAX(SD|UD|SB|SW|UB|UW)Z(256)?rmk$",
                                              "VPMIN(SD|UD|SB|SW|UB|UW)Z(256)?rmk$",
                                              "VPADD(B|W)Z(128)?rmk$",
                                              "VPADD(D|Q)Zrm(k)?$",
                                              "VPSUB(B|W)Z(128)?rmk$",
                                              "VPSUB(D|Q)Zrmk$",
                                              "VPTERNLOG(D|Q)Zrmi(k)?")>;

// WriteVecShift WriteVecShiftImm
def 4GM7WriteVecShiftZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShiftZ], (instrs VPSHLDDZ128rrik, VPSHLDDZ128rrikz, VPSHLDDZ256rrik,
                                           VPSHLDDZ256rrikz, VPSHLDDZrrik, VPSHLDDZrrikz,
                                           VPSHLDQZ128rrik, VPSHLDQZ128rrikz, VPSHLDQZ256rrik,
                                           VPSHLDQZ256rrikz, VPSHLDQZrrik, VPSHLDQZrrikz,
                                           VPSHLDVWZ128rk, VPSHLDVWZ128rkz, VPSHLDVWZ256rk,
                                           VPSHLDVWZ256rkz, VPSHLDVWZrk, VPSHLDVWZrkz,
                                           VPSHLDWZ128rrik, VPSHLDWZ128rrikz, VPSHLDWZ256rrik,
                                           VPSHLDWZ256rrikz, VPSHLDWZrrik, VPSHLDWZrrikz,
                                           VPSHRDDZ128rrik, VPSHRDDZ128rrikz, VPSHRDDZ256rrik,
                                           VPSHRDDZ256rrikz, VPSHRDDZrrik, VPSHRDDZrrikz,
                                           VPSHRDQZ128rrik, VPSHRDQZ128rrikz, VPSHRDQZ256rrik,
                                           VPSHRDQZ256rrikz, VPSHRDQZrrik, VPSHRDQZrrikz,
                                           VPSHRDVWZ128rk, VPSHRDVWZ128rkz, VPSHRDVWZ256rk,
                                           VPSHRDVWZ256rkz, VPSHRDVWZrk, VPSHRDVWZrkz,
                                           VPSHRDWZ128rrik, VPSHRDWZ128rrikz, VPSHRDWZ256rrik,
                                           VPSHRDWZ256rrikz, VPSHRDWZrrik, VPSHRDWZrrikz,
                                           VPSLLVWZ128rrk, VPSLLVWZ128rrkz, VPSLLVWZ256rrk,
                                           VPSLLVWZ256rrkz, VPSLLVWZrrk, VPSLLVWZrrkz,
                                           VPSLLWZ128rik, VPSLLWZ128rikz, VPSLLWZ128rrk,
                                           VPSLLWZ128rrkz, VPSLLWZ256rik, VPSLLWZ256rikz,
                                           VPSLLWZrik, VPSLLWZrikz, VPSRAVWZ128rrk,
                                           VPSRAVWZ128rrkz, VPSRAVWZ256rrk, VPSRAVWZ256rrkz,
                                           VPSRAVWZrrk, VPSRAVWZrrkz, VPSRAWZ128rik,
                                           VPSRAWZ128rikz, VPSRAWZ128rrk, VPSRAWZ128rrkz,
                                           VPSRAWZ256rik, VPSRAWZ256rikz, VPSRAWZrik,
                                           VPSRAWZrikz, VPSRLVWZ128rrk, VPSRLVWZ128rrkz,
                                           VPSRLVWZ256rrk, VPSRLVWZ256rrkz, VPSRLVWZrrk,
                                           VPSRLVWZrrkz, VPSRLWZ128rik, VPSRLWZ128rikz,
                                           VPSRLWZ128rrk, VPSRLWZ128rrkz, VPSRLWZ256rik,
                                           VPSRLWZ256rikz, VPSRLWZrik, VPSRLWZrikz)>;

def 4GM7WriteVecShiftZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVecShiftZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShiftZLd], (instrs VPSHLDDZ128rmik, VPSHLDDZ128rmikz, VPSHLDDZ256rmik,
                                             VPSHLDDZ256rmikz, VPSHLDDZrmik, VPSHLDDZrmikz,
                                             VPSHLDQZ128rmik, VPSHLDQZ128rmikz, VPSHLDQZ256rmik,
                                             VPSHLDQZ256rmikz, VPSHLDQZrmik, VPSHLDQZrmikz,
                                             VPSHLDVWZ128mk, VPSHLDVWZ128mkz, VPSHLDVWZ256mk,
                                             VPSHLDVWZ256mkz, VPSHLDVWZmk, VPSHLDVWZmkz,
                                             VPSHLDWZ128rmik, VPSHLDWZ128rmikz, VPSHLDWZ256rmik,
                                             VPSHLDWZ256rmikz, VPSHLDWZrmik, VPSHLDWZrmikz,
                                             VPSHRDDZ128rmik, VPSHRDDZ128rmikz, VPSHRDDZ256rmik,
                                             VPSHRDDZ256rmikz, VPSHRDDZrmik, VPSHRDDZrmikz,
                                             VPSHRDQZ128rmik, VPSHRDQZ128rmikz, VPSHRDQZ256rmik,
                                             VPSHRDQZ256rmikz, VPSHRDQZrmik, VPSHRDQZrmikz,
                                             VPSHRDVWZ128mk, VPSHRDVWZ128mkz, VPSHRDVWZ256mk,
                                             VPSHRDVWZ256mkz, VPSHRDVWZmk, VPSHRDVWZmkz,
                                             VPSHRDWZ128rmik, VPSHRDWZ128rmikz, VPSHRDWZ256rmik,
                                             VPSHRDWZ256rmikz, VPSHRDWZrmik, VPSHRDWZrmikz,
                                             VPSLLVWZ128rmk, VPSLLVWZ128rmkz, VPSLLVWZ256rmk,
                                             VPSLLVWZ256rmkz, VPSLLVWZrmk, VPSLLVWZrmkz,
                                             VPSLLWZ128mik, VPSLLWZ128mikz, VPSLLWZ128rmk,
                                             VPSLLWZ128rmkz, VPSLLWZ256mik, VPSLLWZ256mikz,
                                             VPSLLWZmik, VPSLLWZmikz, VPSRAVWZ128rmk,
                                             VPSRAVWZ128rmkz, VPSRAVWZ256rmk, VPSRAVWZ256rmkz,
                                             VPSRAVWZrmk, VPSRAVWZrmkz, VPSRAWZ128mik,
                                             VPSRAWZ128mikz, VPSRAWZ128rmk, VPSRAWZ128rmkz,
                                             VPSRAWZ256mik, VPSRAWZ256mikz, VPSRAWZmik,
                                             VPSRAWZmikz, VPSRLVWZ128rmk, VPSRLVWZ128rmkz,
                                             VPSRLVWZ256rmk, VPSRLVWZ256rmkz, VPSRLVWZrmk,
                                             VPSRLVWZrmkz, VPSRLWZ128mik, VPSRLWZ128mikz,
                                             VPSRLWZ128rmk, VPSRLWZ128rmkz, VPSRLWZ256mik,
                                             VPSRLWZ256mikz, VPSRLWZmik, VPSRLWZmikz)>;

def 4GM7WriteVecShiftZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShiftZrrk], (instrs VPSLLDZrrk, VPSLLQZrrk, VPSRADZrrk,
                                              VPSRAQZrrk, VPSRLDZrrk, VPSRLQZrrk,
                                              VPSLLWZrrk, VPSRAWZrrk, VPSRLWZrrk,
                                              VPSLLDZ256rrk, VPSLLQZ256rrk, VPSRADZ256rrk,
                                              VPSRAQZ256rrk, VPSRLDZ256rrk, VPSRLQZ256rrk,
                                              VPSLLWZ256rrk, VPSRAWZ256rrk, VPSRLWZ256rrk)>;

def 4GM7WriteVecShiftZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVecShiftZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShiftZrmk], (instrs VPSLLDZrmk, VPSLLQZrmk, VPSRADZrmk,
                                              VPSRAQZrmk, VPSRLDZrmk, VPSRLQZrmk,
                                              VPSLLWZrmk, VPSRAWZrmk, VPSRLWZrmk,
                                              VPSLLDZ256rmk, VPSLLQZ256rmk, VPSRADZ256rmk,
                                              VPSRAQZ256rmk, VPSRLDZ256rmk, VPSRLQZ256rmk,
                                              VPSLLWZ256rmk, VPSRAWZ256rmk, VPSRLWZ256rmk)>;

def 4GM7WriteVecShift:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShift], (instrs PSLLDrr, PSLLQrr, PSLLWrr, PSRADrr,
                                                   PSRAWrr, PSRLDrr,PSRLQrr, PSRLWrr,
                                                   VPSHLDDZ128rri, VPSHLDDZ256rri, VPSHLDDZrri,
                                                   VPSHLDQZ128rri, VPSHLDQZ256rri, VPSHLDQZrri,
                                                   VPSHLDVDZ128r, VPSHLDVDZ128rk, VPSHLDVDZ128rkz,
                                                   VPSHLDVDZ256r, VPSHLDVDZ256rk, VPSHLDVDZ256rkz,
                                                   VPSHLDVDZr, VPSHLDVDZrk, VPSHLDVDZrkz,
                                                   VPSHLDVQZ128r, VPSHLDVQZ128rk, VPSHLDVQZ128rkz,
                                                   VPSHLDVQZ256r, VPSHLDVQZ256rk, VPSHLDVQZ256rkz,
                                                   VPSHLDVQZr, VPSHLDVQZrk, VPSHLDVQZrkz,
                                                   VPSHLDVWZ128r, VPSHLDVWZ256r, VPSHLDVWZr,
                                                   VPSHLDWZ128rri, VPSHLDWZ256rri, VPSHLDWZrri,
                                                   VPSHRDDZ128rri, VPSHRDDZ256rri, VPSHRDDZrri,
                                                   VPSHRDQZ128rri, VPSHRDQZ256rri, VPSHRDQZrri,
                                                   VPSHRDVDZ128r, VPSHRDVDZ128rk, VPSHRDVDZ128rkz,
                                                   VPSHRDVDZ256r, VPSHRDVDZ256rk, VPSHRDVDZ256rkz,
                                                   VPSHRDVDZr, VPSHRDVDZrk, VPSHRDVDZrkz,
                                                   VPSHRDVQZ128r, VPSHRDVQZ128rk, VPSHRDVQZ128rkz,
                                                   VPSHRDVQZ256r, VPSHRDVQZ256rk, VPSHRDVQZ256rkz,
                                                   VPSHRDVQZr, VPSHRDVQZrk, VPSHRDVQZrkz,
                                                   VPSHRDVWZ128r, VPSHRDVWZ256r, VPSHRDVWZr,
                                                   VPSHRDWZ128rri, VPSHRDWZ256rri, VPSHRDWZrri,
                                                   VPSLLDrr, VPSLLDZ128rr, VPSLLDZ128rrk,
                                                   VPSLLDZ128rrkz, VPSLLQrr, VPSLLQZ128rr,
                                                   VPSLLQZ128rrk, VPSLLQZ128rrkz, VPSLLWrr,
                                                   VPSLLWZ128rr, VPSRADrr, VPSRADZ128rr,
                                                   VPSRADZ128rrk, VPSRADZ128rrkz, VPSRAQZ128rr,
                                                   VPSRAQZ128rrk, VPSRAQZ128rrkz, VPSRAWrr,
                                                   VPSRAWZ128rr, VPSRLDrr, VPSRLDZ128rr,
                                                   VPSRLDZ128rrk, VPSRLDZ128rrkz, VPSRLQrr,
                                                   VPSRLQZ128rr, VPSRLQZ128rrk, VPSRLQZ128rrkz,
                                                   VPSRLWrr, VPSRLWZ128rr)>;

def 4GM7WriteVecShiftLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVecShift.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVecShiftLd], (instrs PSLLDrm, PSLLQrm, PSLLWrm, PSRADrm,
                                                  PSRAWrm, PSRLDrm,PSRLQrm, PSRLWrm,
                                                  VPSHLDDZ128rmi, VPSHLDDZ256rmi, VPSHLDDZrmi,
                                                  VPSHLDQZ128rmi, VPSHLDQZ256rmi, VPSHLDQZrmi,
                                                  VPSHLDVDZ128m, VPSHLDVDZ128mk, VPSHLDVDZ128mkz,
                                                  VPSHLDVDZ256m, VPSHLDVDZ256mk, VPSHLDVDZ256mkz,
                                                  VPSHLDVDZm, VPSHLDVDZmk, VPSHLDVDZmkz,
                                                  VPSHLDVQZ128m, VPSHLDVQZ128mk, VPSHLDVQZ128mkz,
                                                  VPSHLDVQZ256m, VPSHLDVQZ256mk, VPSHLDVQZ256mkz,
                                                  VPSHLDVQZm, VPSHLDVQZmk, VPSHLDVQZmkz,
                                                  VPSHLDVWZ128m, VPSHLDVWZ256m, VPSHLDVWZm,
                                                  VPSHLDWZ128rmi, VPSHLDWZ256rmi, VPSHLDWZrmi,
                                                  VPSHRDDZ128rmi, VPSHRDDZ256rmi, VPSHRDDZrmi,
                                                  VPSHRDQZ128rmi, VPSHRDQZ256rmi, VPSHRDQZrmi,
                                                  VPSHRDVDZ128m, VPSHRDVDZ128mk, VPSHRDVDZ128mkz,
                                                  VPSHRDVDZ256m, VPSHRDVDZ256mk, VPSHRDVDZ256mkz,
                                                  VPSHRDVDZm, VPSHRDVDZmk, VPSHRDVDZmkz,
                                                  VPSHRDVQZ128m, VPSHRDVQZ128mk, VPSHRDVQZ128mkz,
                                                  VPSHRDVQZ256m, VPSHRDVQZ256mk, VPSHRDVQZ256mkz,
                                                  VPSHRDVQZm, VPSHRDVQZmk, VPSHRDVQZmkz,
                                                  VPSHRDVWZ128m, VPSHRDVWZ256m, VPSHRDVWZm,
                                                  VPSHRDWZ128rmi, VPSHRDWZ256rmi, VPSHRDWZrmi,
                                                  VPSLLDrm, VPSLLDZ128rm, VPSLLDZ128rmk,
                                                  VPSLLDZ128rmkz, VPSLLQrm, VPSLLQZ128rm,
                                                  VPSLLQZ128rmk, VPSLLQZ128rmkz, VPSLLWrm,
                                                  VPSLLWZ128rm, VPSRADrm, VPSRADZ128rm,
                                                  VPSRADZ128rmk, VPSRADZ128rmkz, VPSRAQZ128rm,
                                                  VPSRAQZ128rmk, VPSRAQZ128rmkz, VPSRAWrm,
                                                  VPSRAWZ128rm, VPSRLDrm, VPSRLDZ128rm,
                                                  VPSRLDZ128rmk, VPSRLDZ128rmkz, VPSRLQrm,
                                                  VPSRLQZ128rm, VPSRLQZ128rmk, VPSRLQZ128rmkz,
                                                  VPSRLWrm, VPSRLWZ128rm)>;

// VCMP
def 4GM7WriteVCMPXY:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCMPXY], (instrs VCMPPDZ128rrik, VCMPPSZ128rrik,
                                        VCMPPDZ256rrik, VCMPPSZ256rrik)>;

def 4GM7WriteVCMPXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCMPXY.Latency);
  let ReleaseAtCycles = [1, 1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVCMPXYLd], (instrs VCMPPDZ128rmik, VCMPPSZ128rmik,
                                          VCMPPDZ256rmik, VCMPPSZ256rmik)>;

def 4GM7WriteVCMPZ:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [10, 10];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVCMPZ], (instrs VCMPPSZrribk, VCMPPSZrrik,
                                       VCMPPDZrribk, VCMPPDZrrik,
                                       VCMPSSZrrik_Int, VCMPSSZrribk_Int,
                                       VCMPSDZrribk_Int, VCMPSDZrrik_Int)>;

def 4GM7WriteVCMPZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVCMPZ.Latency);
  let ReleaseAtCycles = [1, 1, 10, 10];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVCMPZLd], (instrs VCMPPSZrmik, VCMPPDZrmik,
                                         VCMPSSZrmik_Int, VCMPSDZrmik_Int)>;

// VFPCLASS
def 4GM7WriteVFPCLASSXY:SchedWriteRes<[4GM7FPU13, 4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFPCLASSXY], (instrs VFPCLASSPDZ128rik, VFPCLASSPSZ128rik,
                                            VFPCLASSPDZ256rik, VFPCLASSPSZ256rik)>;

def 4GM7WriteVFPCLASSXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFPCLASSXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFPCLASSXYLd], (instrs VFPCLASSPDZ128mik, VFPCLASSPSZ128mik,
                                              VFPCLASSPDZ256mik, VFPCLASSPSZ256mik)>;

def 4GM7WriteVFPCLASSZ:SchedWriteRes<[4GM7FPU13, 4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVFPCLASSZ], (instrs VFPCLASSSDZrik, VFPCLASSSSZrik)>;

def 4GM7WriteVFPCLASSZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFPCLASSZ.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVFPCLASSZLd], (instrs VFPCLASSSDZmik, VFPCLASSSSZmik)>;

def 4GM7WriteVFPCLASSZrik:SchedWriteRes<[4GM7FPU13, 4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVFPCLASSZrik], (instrs VFPCLASSPDZrik, VFPCLASSPSZrik)>;

def 4GM7WriteVFPCLASSZrikLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFPCLASSZrik.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVFPCLASSZrikLd], (instrs VFPCLASSPDZmik, VFPCLASSPSZmik)>;

// VRNDSCALE
def 4GM7WriteVRNDSCALEXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRNDSCALEXY], (instrs VRNDSCALEPDZ128rrik, VRNDSCALEPSZ128rrik,
                                             VRNDSCALEPDZ256rrik, VRNDSCALEPSZ256rrik)>;

def 4GM7WriteVRNDSCALEXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRNDSCALEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRNDSCALEXYLd], (instrs VRNDSCALEPDZ128rmik, VRNDSCALEPSZ128rmik,
                                               VRNDSCALEPDZ256rmik, VRNDSCALEPSZ256rmik)>;

def 4GM7WriteVRNDSCALEZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRNDSCALEZ], (instrs VRNDSCALESDZrribk_Int, VRNDSCALESDZrrik_Int,
                                            VRNDSCALESSZrribk_Int, VRNDSCALESSZrrik_Int,
                                            VRNDSCALEPDZrribk, VRNDSCALEPDZrrik,
                                            VRNDSCALEPSZrribk, VRNDSCALEPSZrrik)>;

def 4GM7WriteVRNDSCALEZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRNDSCALEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRNDSCALEZLd], (instrs VRNDSCALESDZrmik_Int, VRNDSCALESSZrmik_Int,
                                              VRNDSCALEPDZrmik, VRNDSCALEPSZrmik)>;

// VREDUCE
def 4GM7WriteVREDUCEXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVREDUCEXY], (instrs VREDUCEPDZ128rrik, VREDUCEPSZ128rrik,
                                           VREDUCEPDZ256rrik, VREDUCEPSZ256rrik)>;

def 4GM7WriteVREDUCEXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVREDUCEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVREDUCEXYLd], (instrs VREDUCEPDZ128rmik, VREDUCEPSZ128rmik,
                                             VREDUCEPDZ256rmik, VREDUCEPSZ256rmik)>;

def 4GM7WriteVREDUCEZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVREDUCEZ], (instrs VREDUCESDZrribk, VREDUCESDZrrik,
                                          VREDUCESSZrribk, VREDUCESSZrrik,
                                          VREDUCEPDZrribk, VREDUCEPDZrrik,
                                          VREDUCEPSZrribk, VREDUCEPSZrrik)>;

def 4GM7WriteVREDUCEZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVREDUCEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVREDUCEZLd], (instrs VREDUCESSZrmik, VREDUCESDZrmik,
                                            VREDUCEPSZrmik, VREDUCEPDZrmik)>;

// VGETEXP VGETMANT
def 4GM7WriteVGETEXPXY_VGETMANTXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVGETEXPXY_VGETMANTXY], (instrs VGETEXPPSZ128rk, VGETEXPPDZ128rk,
                                                      VGETEXPPSZ256rk, VGETEXPPDZ256rk,
                                                      VGETMANTPSZ128rrik, VGETMANTPDZ128rrik,
                                                      VGETMANTPSZ256rrik, VGETMANTPDZ256rrik)>;

def 4GM7WriteVGETEXPXY_VGETMANTXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVGETEXPXY_VGETMANTXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVGETEXPXY_VGETMANTXYLd], (instrs VGETEXPPSZ128mk, VGETEXPPDZ128mk,
                                                        VGETEXPPSZ256mk, VGETEXPPDZ256mk,
                                                        VGETMANTPSZ128rmik, VGETMANTPDZ128rmik,
                                                        VGETMANTPSZ256rmik, VGETMANTPDZ256rmik)>;

def 4GM7WriteVGETEXPZ_VGETMANTZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVGETEXPZ_VGETMANTZ], (instrs VGETEXPSSZrbk, VGETEXPSSZrk,
                                                    VGETEXPSDZrbk, VGETEXPSDZrk,
                                                    VGETEXPPSZrbk, VGETEXPPSZrk,
                                                    VGETEXPPDZrbk, VGETEXPPDZrk,
                                                    VGETMANTSSZrribk, VGETMANTSSZrrik,
                                                    VGETMANTSDZrribk, VGETMANTSDZrrik,
                                                    VGETMANTPSZrribk, VGETMANTPSZrrik,
                                                    VGETMANTPDZrribk, VGETMANTPDZrrik)>;

def 4GM7WriteVGETEXPZ_VGETMANTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVGETEXPZ_VGETMANTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVGETEXPZ_VGETMANTZLd], (instrs VGETEXPSSZmk, VGETEXPSDZmk,
                                                      VGETEXPPSZmbk, VGETEXPPSZmk,
                                                      VGETEXPPDZmbk, VGETEXPPDZmk,
                                                      VGETMANTSSZrmik, VGETMANTSDZrmik,
                                                      VGETMANTPSZrmik, VGETMANTPDZrmik)>;

// VSCALEF
def 4GM7WriteVSCALEFXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFXY], (instrs VSCALEFPSZ128rr, VSCALEFPSZ128rrkz,
                                           VSCALEFPSZ256rr, VSCALEFPSZ256rrkz,
                                           VSCALEFPDZ128rr, VSCALEFPDZ128rrkz,
                                           VSCALEFPDZ256rr, VSCALEFPDZ256rrkz)>;

def 4GM7WriteVSCALEFXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSCALEFXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFXYLd], (instrs VSCALEFPSZ128rm, VSCALEFPSZ128rmkz,
                                             VSCALEFPSZ256rm, VSCALEFPSZ256rmkz,
                                             VSCALEFPDZ128rm, VSCALEFPDZ128rmkz,
                                             VSCALEFPDZ256rm, VSCALEFPDZ256rmkz)>;

def 4GM7WriteVSCALEFZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFZ], (instrs VSCALEFSSZrr, VSCALEFSSZrrkz,
                                          VSCALEFSSZrrb_Int, VSCALEFSSZrrbkz_Int,
                                          VSCALEFSDZrr, VSCALEFSDZrrkz,
                                          VSCALEFSDZrrb_Int, VSCALEFSDZrrbkz_Int,
                                          VSCALEFPSZrr, VSCALEFPSZrrkz,
                                          VSCALEFPSZrrb, VSCALEFPSZrrbkz,
                                          VSCALEFPDZrr, VSCALEFPDZrrkz,
                                          VSCALEFPDZrrb, VSCALEFPDZrrbkz)>;

def 4GM7WriteVSCALEFZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSCALEFZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFZLd], (instrs VSCALEFSSZrm, VSCALEFSSZrmkz,
                                            VSCALEFSDZrm,
                                            VSCALEFPSZrm, VSCALEFPSZrmkz,
                                            VSCALEFPSZrmb, VSCALEFPSZrmbkz,
                                            VSCALEFPDZrm, VSCALEFPDZrmkz,
                                            VSCALEFPDZrmb, VSCALEFPDZrmbkz)>;

def 4GM7WriteVSCALEFXYrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFXYrrk], (instrs VSCALEFPSZ128rrk, VSCALEFPDZ128rrk,
                                              VSCALEFPSZ256rrk, VSCALEFPDZ256rrk)>;

def 4GM7WriteVSCALEFXYrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSCALEFXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFXYrmk], (instrs VSCALEFPSZ128rmk, VSCALEFPDZ128rmk,
                                              VSCALEFPSZ256rmk, VSCALEFPDZ256rmk)>;

def 4GM7WriteVSCALEFZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFZrrk], (instrs VSCALEFSSZrrk, VSCALEFSSZrrbk_Int,
                                             VSCALEFSDZrrk, VSCALEFSDZrrbk_Int,
                                             VSCALEFPSZrrk, VSCALEFPSZrrbk,
                                             VSCALEFPDZrrk, VSCALEFPDZrrbk)>;

def 4GM7WriteVSCALEFZrmkLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVSCALEFZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVSCALEFZrmkLd], (instrs VSCALEFSSZrmk, VSCALEFSDZrmk,
                                               VSCALEFPSZrmk, VSCALEFPSZrmbk,
                                               VSCALEFPDZrmk, VSCALEFPDZrmbk)>;

// VFMA
def 4GM7WriteVFMAXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVFMAXY], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)(Z128|Z256)r(bk|k)$")>;

def 4GM7WriteVFMAXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFMAXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVFMAXYLd], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)(Z128|Z256)m(bk|k)$")>;

def 4GM7WriteVFMAZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFMAZ], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)Zr(bk|k)$",
                                          "VF(MADD|MSUB|NMADD|NMSUB)(132|213|231)(SD|SS)Zr(bk|k)_Int")>;

def 4GM7WriteVFMAZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFMAZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFMAZLd], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)Zm(bk|k)$",
                                            "VF(MADD|MSUB|NMADD|NMSUB)(132|213|231)(SD|SS)Zm(bk|k)_Int")>;

// VPROL VPROR
def 4GM7WriteVPROLXY_VPRORXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLXY_VPRORXY], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)(rr|rrkz)$")>;

def 4GM7WriteVPROLXY_VPRORXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPROLXY_VPRORXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLXY_VPRORXYLd], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)(rm|rmkz)$")>;

def 4GM7WriteVPROLXY_VPRORXYrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLXY_VPRORXYrrk], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)rrk$")>;

def 4GM7WriteVPROLXY_VPRORXYrmkLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPROLXY_VPRORXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLXY_VPRORXYrmkLd], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)rmk$")>;

def 4GM7WriteVPROLZ_VPRORZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLZ_VPRORZ], (instregex "VPRO(L|R)V(D|Q)Z(rr|rrk|rrkz)")>;

def 4GM7WriteVPROLZ_VPRORZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPROLZ_VPRORZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLZ_VPRORZLd], (instregex "VPRO(L|R)V(D|Q)Z(rm|rmk|rmkz)")>;

def 4GM7WriteVPROLImmZ_VPRORImmZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLImmZ_VPRORImmZ], (instregex "VPRO(L|R)(D|Q)Z(ri|rik|rikz)")>;

def 4GM7WriteVPROLImmZ_VPRORImmZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPROLImmZ_VPRORImmZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPROLImmZ_VPRORImmZLd], (instregex "VPRO(L|R)(D|Q)Z(mi|mik|mikz)")>;

// VPTEST
def 4GM7WriteVPTESTXY:SchedWriteRes<[4GM7FPU, 4GM7FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [3, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPTESTXY], (instregex "VPTEST(N)?M(B|D|Q|W)(Z128|Z256)rrk")>;

def 4GM7WriteVPTESTXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPTESTXY.Latency);
  let ReleaseAtCycles = [1, 1, 3, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVPTESTXYLd], (instregex "VPTEST(N)?M(B|D|Q|W)(Z128|Z256)rmk")>;

def 4GM7WriteVPTESTZ:SchedWriteRes<[4GM7FPU, 4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [8, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPTESTZ], (instregex "VPTEST(N)?M(B|D|Q|W)Zrrk")>;

def 4GM7WriteVPTESTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPTESTZ.Latency);
  let ReleaseAtCycles = [1, 1, 12, 6];
  let NumMicroOps = 3;
}
def : InstRW<[4GM7WriteVPTESTZLd], (instregex "VPTEST(N)?M(B|D|Q|W)Zrmk")>;

// VPACK
def 4GM7WriteVPACKXY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPACKXY], (instrs VPACKSSDWZ128rrk, VPACKSSWBZ128rrk,
                                         VPACKUSDWZ128rrk, VPACKUSWBZ128rrk,
                                         VPACKSSDWZ256rrk, VPACKSSWBZ256rrk,
                                         VPACKUSDWZ256rrk, VPACKUSWBZ256rrk)>;

def 4GM7WriteVPACKXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPACKXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPACKXYLd], (instrs VPACKSSDWZ128rmk, VPACKSSWBZ128rmk,
                                           VPACKUSDWZ128rmk, VPACKUSWBZ128rmk,
                                           VPACKSSDWZ256rmk, VPACKSSWBZ256rmk,
                                           VPACKUSDWZ256rmk, VPACKUSWBZ256rmk)>;

def 4GM7WriteVPACKZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPACKZ], (instrs VPACKSSDWZrrk, VPACKSSWBZrrk,
                                        VPACKUSDWZrrk, VPACKUSWBZrrk)>;

def 4GM7WriteVPACKZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPACKZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPACKZLd], (instrs VPACKSSDWZrmk, VPACKSSWBZrmk,
                                          VPACKUSDWZrmk, VPACKUSWBZrmk)>;

// VRANGE
def 4GM7WriteVRANGEXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEXY], (instregex "VRANGE(PS|PD)(Z128|Z256)(rri|rrikz)$",
                                             "VRANGE(SS|SD)Z(rri|rrib|rribkz|rrikz)$")>;

def 4GM7WriteVRANGEXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRANGEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEXYLd], (instregex "VRANGE(PS|PD)(Z128|Z256)(rmi|rmikz)$",
                                               "VRANGE(SS|SD)Z(rmi|rmib|rmibkz|rmikz)$")>;

def 4GM7WriteVRANGEZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEZ], (instregex "VRANGE(PS|PD)Z(rri|rrib|rribkz|rrikz)$")>;

def 4GM7WriteVRANGEZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRANGEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEZLd], (instregex "VRANGE(PS|PD)Z(rmi|rmib|rmibkz|rmikz)$")>;

def 4GM7WriteVRANGEXYrrik:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEXYrrik], (instregex "VRANGE(PS|PD)(Z128|Z256)rrik$",
                                                 "VRANGE(SS|SD)Z(rrik|rribk)$")>;

def 4GM7WriteVRANGEXYrmikLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRANGEXYrrik.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEXYrmikLd], (instregex "VRANGE(PS|PD)(Z128|Z256)rmik$",
                                                   "VRANGE(SS|SD)Z(rmik|rmibk)$")>;

def 4GM7WriteVRANGEZrrik:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEZrrik], (instregex "VRANGE(PS|PD)Z(rrik|rribk)$")>;

def 4GM7WriteVRANGEZrmikLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVRANGEZrrik.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVRANGEZrmikLd], (instregex "VRANGE(PS|PD)Z(rmik|rmibk)$")>;

// VFIXUPIMM
def 4GM7WriteVFIXUPIMMXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVFIXUPIMMXY], (instregex "VFIXUPIMM(PS|PD)(Z128|Z256)rrik$")>;

def 4GM7WriteVFIXUPIMMXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFIXUPIMMXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVFIXUPIMMXYLd], (instregex "VFIXUPIMM(PS|PD)(Z128|Z256)rmik$")>;

def 4GM7WriteVFIXUPIMMZ:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFIXUPIMMZ], (instregex "VFIXUPIMM(SS|SD|PS|PD)Z(rrik|rribk)$",
                                               "VFIXUPIMM(SS|SD)Z(rrik|rribk)$")>;

def 4GM7WriteVFIXUPIMMZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVFIXUPIMMZ.Latency);
  let ReleaseAtCycles = [1, 1, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFIXUPIMMZLd], (instregex "VFIXUPIMM(SS|SD|PS|PD)Z(rmik|rmibk)$",
                                                 "VFIXUPIMM(SS|SD)Z(rmik|rmibk)$")>;

def 4GM7WriteVFIXUPIMMZrribkz:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVFIXUPIMMZrribkz], (instregex "VFIXUPIMM(SS|SD)Z(rrib|rribkz)$")>;

// VCOMPRESS VPCOMPRESS
def 4GM7WriteCOMPRESSXYrrkz:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteCOMPRESSXYrrkz], (instrs VCOMPRESSPSZ128rrkz, VCOMPRESSPDZ128rrkz,
                                                VPCOMPRESSBZ128rrkz, VPCOMPRESSWZ128rrkz,
                                                VPCOMPRESSDZ128rrkz, VPCOMPRESSQZ128rrkz,
                                                VCOMPRESSPSZ256rrkz, VCOMPRESSPDZ256rrkz,
                                                VPCOMPRESSBZ256rrkz, VPCOMPRESSWZ256rrkz,
                                                VPCOMPRESSDZ256rrkz, VPCOMPRESSQZ256rrkz)>;

def 4GM7WriteCOMPRESSXYrrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteCOMPRESSXYrrk], (instrs VCOMPRESSPSZ128rrk, VCOMPRESSPDZ128rrk,
                                               VPCOMPRESSBZ128rrk, VPCOMPRESSWZ128rrk,
                                               VPCOMPRESSDZ128rrk, VPCOMPRESSQZ128rrk,
                                               VCOMPRESSPSZ256rrk, VCOMPRESSPDZ256rrk,
                                               VPCOMPRESSBZ256rrk, VPCOMPRESSWZ256rrk,
                                               VPCOMPRESSDZ256rrk, VPCOMPRESSQZ256rrk)>;

def 4GM7WriteCOMPRESSXYmrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteCOMPRESSXYrrk.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [2, 4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteCOMPRESSXYmrk], (instrs VCOMPRESSPSZ128mrk, VCOMPRESSPDZ128mrk,
                                               VPCOMPRESSBZ128mrk, VPCOMPRESSWZ128mrk,
                                               VPCOMPRESSDZ128mrk, VPCOMPRESSQZ128mrk,
                                               VCOMPRESSPSZ256mrk, VCOMPRESSPDZ256mrk,
                                               VPCOMPRESSBZ256mrk, VPCOMPRESSWZ256mrk,
                                               VPCOMPRESSDZ256mrk, VPCOMPRESSQZ256mrk)>;

def 4GM7WriteCOMPRESSXY:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteCOMPRESSXY], (instrs VCOMPRESSPSZ128rr, VCOMPRESSPDZ128rr,
                                            VPCOMPRESSBZ128rr, VPCOMPRESSWZ128rr,
                                            VPCOMPRESSDZ128rr, VPCOMPRESSQZ128rr,
                                            VCOMPRESSPSZ256rr, VCOMPRESSPDZ256rr,
                                            VPCOMPRESSBZ256rr, VPCOMPRESSWZ256rr,
                                            VPCOMPRESSDZ256rr, VPCOMPRESSQZ256rr)>;

def 4GM7WriteCOMPRESSXYLd:SchedWriteRes<[4GM7FPU3, 4GM7FPU02, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteCOMPRESSXY.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [2, 4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteCOMPRESSXYLd], (instrs VCOMPRESSPSZ128mr, VCOMPRESSPDZ128mr,
                                              VPCOMPRESSBZ128mr, VPCOMPRESSWZ128mr,
                                              VPCOMPRESSDZ128mr, VPCOMPRESSQZ128mr,
                                              VCOMPRESSPSZ256mr, VCOMPRESSPDZ256mr,
                                              VPCOMPRESSBZ256mr, VPCOMPRESSWZ256mr,
                                              VPCOMPRESSDZ256mr, VPCOMPRESSQZ256mr)>;

def 4GM7WriteCOMPRESSZrrkz:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteCOMPRESSZrrkz], (instrs VCOMPRESSPSZrrkz, VCOMPRESSPDZrrkz,
                                               VPCOMPRESSBZrrkz, VPCOMPRESSWZrrkz,
                                               VPCOMPRESSDZrrkz, VPCOMPRESSQZrrkz)>;

def 4GM7WriteCOMPRESSZrrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 7;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteCOMPRESSZrrk], (instrs VCOMPRESSPSZrrk, VCOMPRESSPDZrrk,
                                              VPCOMPRESSBZrrk, VPCOMPRESSDZrrk,
                                              VPCOMPRESSWZrrk, VPCOMPRESSQZrrk)>;

def 4GM7WriteCOMPRESSZmrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteCOMPRESSZrrk.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [6, 12, 1, 1];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteCOMPRESSZmrk], (instrs VCOMPRESSPSZmrk, VCOMPRESSPDZmrk,
                                              VPCOMPRESSBZmrk, VPCOMPRESSDZmrk,
                                              VPCOMPRESSWZmrk, VPCOMPRESSQZmrk)>;

def 4GM7WriteCOMPRESSZ:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteCOMPRESSZ], (instrs VCOMPRESSPSZrr, VCOMPRESSPDZrr,
                                           VPCOMPRESSBZrr, VPCOMPRESSWZrr,
                                           VPCOMPRESSDZrr, VPCOMPRESSQZrr)>;

def 4GM7WriteCOMPRESSZLd:SchedWriteRes<[4GM7FPU3, 4GM7FPU02, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteCOMPRESSZ.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [6, 12, 1, 1];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteCOMPRESSZLd], (instrs VCOMPRESSPSZmr, VCOMPRESSPDZmr,
                                             VPCOMPRESSBZmr, VPCOMPRESSWZmr,
                                             VPCOMPRESSDZmr, VPCOMPRESSQZmr)>;

// EXPAND
def 4GM7WriteEXPANDXYrrkz:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXYrrkz], (instrs VEXPANDPSZ128rrkz, VEXPANDPDZ128rrkz,
                                              VPEXPANDBZ128rrkz, VPEXPANDWZ128rrkz,
                                              VPEXPANDDZ128rrkz, VPEXPANDQZ128rrkz,
                                              VEXPANDPSZ256rrkz, VEXPANDPDZ256rrkz,
                                              VPEXPANDBZ256rrkz, VPEXPANDWZ256rrkz,
                                              VPEXPANDDZ256rrkz, VPEXPANDQZ256rrkz)>;

def 4GM7WriteEXPANDXYrmkz:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDXYrrkz.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXYrmkz], (instrs VEXPANDPSZ128rmkz, VEXPANDPDZ128rmkz,
                                              VPEXPANDBZ128rmkz, VPEXPANDWZ128rmkz,
                                              VPEXPANDDZ128rmkz, VPEXPANDQZ128rmkz,
                                              VEXPANDPSZ256rmkz, VEXPANDPDZ256rmkz,
                                              VPEXPANDBZ256rmkz, VPEXPANDWZ256rmkz,
                                              VPEXPANDDZ256rmkz, VPEXPANDQZ256rmkz)>;

def 4GM7WriteEXPANDXYrrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXYrrk], (instrs VEXPANDPSZ128rrk, VEXPANDPDZ128rrk,
                                             VPEXPANDBZ128rrk, VPEXPANDWZ128rrk,
                                             VPEXPANDDZ128rrk, VPEXPANDQZ128rrk,
                                             VEXPANDPSZ256rrk, VEXPANDPDZ256rrk,
                                             VPEXPANDBZ256rrk, VPEXPANDWZ256rrk,
                                             VPEXPANDDZ256rrk, VPEXPANDQZ256rrk)>;

def 4GM7WriteEXPANDXYrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXYrmk], (instrs VEXPANDPSZ128rmk, VEXPANDPDZ128rmk,
                                             VPEXPANDBZ128rmk, VPEXPANDWZ128rmk,
                                             VPEXPANDDZ128rmk, VPEXPANDQZ128rmk,
                                             VEXPANDPSZ256rmk, VEXPANDPDZ256rmk,
                                             VPEXPANDBZ256rmk, VPEXPANDWZ256rmk,
                                             VPEXPANDDZ256rmk, VPEXPANDQZ256rmk)>;

def 4GM7WriteEXPANDXY:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXY], (instrs VEXPANDPSZ128rr, VEXPANDPDZ128rr,
                                          VPEXPANDBZ128rr, VPEXPANDWZ128rr,
                                          VPEXPANDDZ128rr, VPEXPANDQZ128rr,
                                          VEXPANDPSZ256rr, VEXPANDPDZ256rr,
                                          VPEXPANDBZ256rr, VPEXPANDWZ256rr,
                                          VPEXPANDDZ256rr, VPEXPANDQZ256rr)>;

def 4GM7WriteEXPANDXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteEXPANDXYLd], (instrs VEXPANDPSZ128rm, VEXPANDPDZ128rm,
                                            VPEXPANDBZ128rm, VPEXPANDWZ128rm,
                                            VPEXPANDDZ128rm, VPEXPANDQZ128rm,
                                            VEXPANDPSZ256rm, VEXPANDPDZ256rm,
                                            VPEXPANDBZ256rm, VPEXPANDWZ256rm,
                                            VPEXPANDDZ256rm, VPEXPANDQZ256rm)>;

def 4GM7WriteEXPANDZrrkz:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZrrkz], (instrs VEXPANDPSZrrkz, VEXPANDPDZrrkz,
                                             VPEXPANDBZrrkz, VPEXPANDWZrrkz,
                                             VPEXPANDDZrrkz, VPEXPANDQZrrkz)>;

def 4GM7WriteEXPANDZrmkz:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDZrrkz.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZrmkz], (instrs VEXPANDPSZrmkz, VEXPANDPDZrmkz,
                                             VPEXPANDBZrmkz, VPEXPANDWZrmkz,
                                             VPEXPANDDZrmkz, VPEXPANDQZrmkz)>;

def 4GM7WriteEXPANDZrrk:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 8;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZrrk], (instrs VEXPANDPSZrrk, VEXPANDPDZrrk,
                                            VPEXPANDBZrrk, VPEXPANDDZrrk,
                                            VPEXPANDWZrrk, VPEXPANDQZrrk)>;

def 4GM7WriteEXPANDZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZrmk], (instrs VEXPANDPSZrmk, VEXPANDPDZrmk,
                                            VPEXPANDBZrmk, VPEXPANDDZrmk,
                                            VPEXPANDWZrmk, VPEXPANDQZrmk)>;

def 4GM7WriteEXPANDZ:SchedWriteRes<[4GM7FPU3, 4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZ], (instrs VEXPANDPSZrr, VEXPANDPDZrr,
                                         VPEXPANDBZrr, VPEXPANDWZrr,
                                         VPEXPANDDZrr, VPEXPANDQZrr)>;

def 4GM7WriteEXPANDZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU3, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteEXPANDZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[4GM7WriteEXPANDZLd], (instrs VEXPANDPSZrm, VEXPANDPDZrm,
                                           VPEXPANDBZrm, VPEXPANDWZrm,
                                           VPEXPANDDZrm, VPEXPANDQZrm)>;

// VINSERT
def 4GM7WriteVINSERT256Z:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERT256Z], (instregex "VINSERT(F|I)(32X8|64X4)Zrri(k|kz)?")>;

def 4GM7WriteVINSERT256ZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVINSERT256Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERT256ZLd], (instregex "VINSERT(F|I)(32X8|64X4)Zrmi(k|kz)?")>;

def 4GM7WriteVINSERTY:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVINSERTY], (instregex "VINSERT(F|I)(32X4|64X2)Z256rrik")>;

def 4GM7WriteVINSERTYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVINSERTY.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVINSERTYLd], (instregex "VINSERT(F|I)(32X4|64X2)Z256rmik")>;

def 4GM7WriteVINSERTZ:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERTZ], (instregex "VINSERT(F|I)(32X4|64X2)Zrrik")>;

def 4GM7WriteVINSERTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVINSERTZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERTZLd], (instregex "VINSERT(F|I)(32X4|64X2)Zrmik")>;

def 4GM7WriteVINSERTZrri:SchedWriteRes<[4GM7FPU02, 4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERTZrri], (instregex "VINSERT(F|I)(32X4|64X2)Zrri$")>;

def 4GM7WriteVINSERTZrmi:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVINSERTZrri.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVINSERTZrmi], (instregex "VINSERT(F|I)(32X4|64X2)Zrmi$")>;

// VEXTRACT
def 4GM7WriteVEXTRACTZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVEXTRACTZ], (instregex "VEXTRACT(F|I)(32X8|64X4)Z(rri|rrikz)$")>;

def 4GM7WriteVEXTRACTZLd:SchedWriteRes<[4GM7FPU13, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteVEXTRACTZ.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVEXTRACTZLd], (instregex "VEXTRACT(F|I)(32X8|64X4)Z(mri|mrikz)$")>;

def 4GM7WriteVEXTRACTZrrik:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVEXTRACTZrrik], (instregex "VEXTRACT(F|I)(32X8|64X4)Zrrik$")>;

def 4GM7WriteVEXTRACTZmrik:SchedWriteRes<[4GM7FPU13, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteVEXTRACTZrrik.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVEXTRACTZmrik], (instregex "VEXTRACT(F|I)(32X8|64X4)Zmrik$")>;

def 4GM7WriteVEXTRACTY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVEXTRACTY], (instregex "VEXTRACT(F|I)(32X4|64X2)(Z|Z256)rrik$")>;

def 4GM7WriteVEXTRACTYLd:SchedWriteRes<[4GM7FPU02, 4GM7AGU, 4GM7Store]> {
  let Latency = !add(4GM7WriteVEXTRACTY.Latency, C864GM7Model.StoreLatency);
  let ReleaseAtCycles = [4, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVEXTRACTYLd], (instregex "VEXTRACT(F|I)(32X4|64X2)(Z|Z256)mrik$")>;

// VPCONFLIC
def 4GM7WriteVPCONFLICX:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCONFLICX], (instregex "VPCONFLICT(D|Q)Z128(rr|rrk|rrkz)")>;

def 4GM7WriteVPCONFLICXLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCONFLICX.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCONFLICXLd], (instregex "VPCONFLICT(D|Q)Z128(rm|rmk|rmkz)")>;

def 4GM7WriteVPCONFLICY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteVPCONFLICY], (instregex "VPCONFLICT(D|Q)Z256(rr|rrkz)$")>;

def 4GM7WriteVPCONFLICYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCONFLICY.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 8;
}
def : InstRW<[4GM7WriteVPCONFLICYLd], (instregex "VPCONFLICT(D|Q)Z256(rm|rmkz)$")>;

def 4GM7WriteVPCONFLICYrrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCONFLICYrrk], (instregex "VPCONFLICT(D|Q)Z256rrk$")>;

def 4GM7WriteVPCONFLICYrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCONFLICYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPCONFLICYrmk], (instregex "VPCONFLICT(D|Q)Z256rmk$")>;

def 4GM7WriteVPCONFLICZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 7;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPCONFLICZ], (instregex "VPCONFLICT(D|Q)Z(rr|rrkz)$")>;

def 4GM7WriteVPCONFLICZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCONFLICZ.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPCONFLICZLd], (instregex "VPCONFLICT(D|Q)Z(rm|rmkz)$")>;

def 4GM7WriteVPCONFLICZrrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPCONFLICZrrk], (instregex "VPCONFLICT(D|Q)Zrrk$")>;

def 4GM7WriteVPCONFLICZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPCONFLICZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPCONFLICZrmk], (instregex "VPCONFLICT(D|Q)Zrmk$")>;

// VPDP
def 4GM7WriteVPDPXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPDPXY], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)(Z128|Z256)rrk$")>;

def 4GM7WriteVPDPXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPDPXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPDPXYLd], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)(Z128|Z256)rmk$")>;

def 4GM7WriteVPDPZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPDPZ], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)Zrrk$")>;

def 4GM7WriteVPDPZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPDPZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPDPZLd], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)Zrmk$")>;

// VBROADCAST VPBROADCAST
def 4GM7WriteVBROADCASTX:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTX], (instregex "V(P)?BROADCAST(SS|B|W)Z128(rr|rrkz)$")>;

def 4GM7WriteVBROADCASTXLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBROADCASTX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTXLd], (instregex "V(P)?BROADCAST(SS|B|W)Z128(rm|rmkz)$")>;

def 4GM7WriteVBROADCASTXrrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTXrrk], (instregex "V(P)?BROADCAST(SS|SD|I32X2|B|W|D|Q)(r)?Z128rrk$")>;

def 4GM7WriteVBROADCASTXrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBROADCASTXrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTXrmk], (instregex "V(P)?BROADCAST(SS|SD|I32X2|B|W|D|Q)(r)?Z128rmk$")>;

def 4GM7WriteVBROADCASTY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTY], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Z256rrk$")>;

def 4GM7WriteVBROADCASTYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBROADCASTY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVBROADCASTYLd], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Z256rmk$")>;

def 4GM7WriteVBROADCASTZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVBROADCASTZ], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)Z(rr|rrkz)$")>;

def 4GM7WriteVBROADCASTZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBROADCASTZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVBROADCASTZLd], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)Z(rm|rmkz)$")>;

def 4GM7WriteVBROADCASTZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVBROADCASTZrrk], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Zrrk$")>;

def 4GM7WriteVBROADCASTZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVBROADCASTZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVBROADCASTZrmk], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Zrmk$")>;

// VPERM
def 4GM7WriteVPERMXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMXY], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)(rr|rrkz)$")>;

def 4GM7WriteVPERMXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMXYLd], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)(rm|rmkz)$")>;


def 4GM7WriteVPERMXYrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMXYrrk], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)rrk$")>;

def 4GM7WriteVPERMXYrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMXYrmk], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)rmk$")>;


def 4GM7WriteVPERMZ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 8;
  let ReleaseAtCycles = [32];
  let NumMicroOps = 8;
}
def : InstRW<[4GM7WriteVPERMZ], (instregex "VPERM(PS|PD|B|W|D|Q)Zrr(k|kz)?",
                                           "VPERM2(F|I)128rri")>;

def 4GM7WriteVPERMZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMZ.Latency);
  let ReleaseAtCycles = [1, 1, 32];
  let NumMicroOps = 8;
}
def : InstRW<[4GM7WriteVPERMZLd], (instregex "VPERM(PS|PD|B|W|D|Q)Zrm(k|kz)?",
                                             "VPERM2(F|I)128rmi")>;

def 4GM7WriteVPERMImmY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmY], (instregex "VPERM(PD|Q)Z256(ri|rikz)$")>;

def 4GM7WriteVPERMImmYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMImmY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmYLd], (instregex "VPERM(PD|Q)Z256(mi|mikz)$")>;

def 4GM7WriteVPERMImmYrik:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmYrik], (instregex "VPERM(PD|Q)Z256rik$")>;

def 4GM7WriteVPERMImmYmik:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMImmYrik.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmYmik], (instregex "VPERM(PD|Q)Z256mik$")>;

def 4GM7WriteVPERMImmZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmZ], (instregex "VPERM(PD|Q)Zri(k|kz)?")>;

def 4GM7WriteVPERMImmZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMImmZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMImmZLd], (instregex "VPERM(PD|Q)Zmi(k|kz)?")>;

// VPERMI2 VPERMT2
def 4GM7WriteVPERMI2X_VPERMT2X:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMI2X_VPERMT2X], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Z128(rr|rrkz)$")>;

def 4GM7WriteVPERMI2X_VPERMT2XLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMI2X_VPERMT2X.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMI2X_VPERMT2XLd], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Z128(rm|rmkz)$")>;

def 4GM7WriteVPERMI2X_VPERMT2Xrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMI2X_VPERMT2Xrrk], (instregex "VPERM(I|T)2(PS|PD|B|D|W|Q)Z128rrk$")>;

def 4GM7WriteVPERMI2X_VPERMT2Xrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMI2X_VPERMT2Xrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMI2X_VPERMT2Xrmk], (instregex "VPERM(I|T)2(PS|PD|B|D|W|Q)Z128rmk$")>;

def 4GM7WriteVPERMT2Y:SchedWriteRes<[4GM7FPU]> {
  let Latency = 4;
  let ReleaseAtCycles = [16];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPERMT2Y], (instregex "VPERMT2(PS|PD|B|W|D|Q)Z256rr(kz)?")>;

def 4GM7WriteVPERMT2YLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMT2Y.Latency);
  let ReleaseAtCycles = [1, 1, 16];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPERMT2YLd], (instregex "VPERMT2(PS|PD|B|W|D|Q)Z256rm(kz)?")>;

def 4GM7WriteVPERMI2Y:SchedWriteRes<[4GM7FPU]> {
  let Latency = 6;
  let ReleaseAtCycles = [16];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPERMI2Y], (instregex "VPERMI2(PS|PD|B|W|D|Q)Z256rr(kz)?")>;

def 4GM7WriteVPERMI2YLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMI2Y.Latency);
  let ReleaseAtCycles = [1, 1, 16];
  let NumMicroOps = 4;
}
def : InstRW<[4GM7WriteVPERMI2YLd], (instregex "VPERMI2(PS|PD|B|W|D|Q)Z256rm(kz)?")>;

def 4GM7WriteVPERMI2Z_VPERMT2Z:SchedWriteRes<[4GM7FPU]> {
  let Latency = 16;
  let ReleaseAtCycles = [64];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPERMI2Z_VPERMT2Z], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Zrr(kz)?")>;

def 4GM7WriteVPERMI2Z_VPERMT2ZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMI2Z_VPERMT2Z.Latency);
  let ReleaseAtCycles = [1, 1, 64];
  let NumMicroOps = 16;
}
def : InstRW<[4GM7WriteVPERMI2Z_VPERMT2ZLd], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Zrm(kz)?")>;

// VPERMIL
def 4GM7WriteVPERMILXY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILXY], (instregex "VPERMIL(PS|PD)(Z128|Z256)rrk$")>;

def 4GM7WriteVPERMILXYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMILXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILXYLd], (instregex "VPERMIL(PS|PD)(Z128|Z256)rmk$")>;

def 4GM7WriteVPERMILZ:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZ], (instregex "VPERMIL(PS|PD)Z(rr|rrkz)$")>;

def 4GM7WriteVPERMILZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMILZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZLd], (instregex "VPERMIL(PS|PD)Z(rm|rmkz)$")>;

def 4GM7WriteVPERMILZrrk:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZrrk], (instregex "VPERMIL(PS|PD)Zrrk$")>;

def 4GM7WriteVPERMILZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMILZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZrmk], (instregex "VPERMIL(PS|PD)Zrmk$")>;

def 4GM7WriteVPERMILZrik:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZrik], (instregex "VPERMIL(PS|PD)Zri(k|kz)?")>;

def 4GM7WriteVPERMILZmik:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVPERMILZrik.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteVPERMILZmik], (instregex "VPERMIL(PS|PD)Zmi(k|kz)?")>;

// ALIGN
def 4GM7WriteALIGNX:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNX], (instrs VALIGNDZ128rrik, VALIGNDZ128rrikz,
                                        VALIGNQZ128rrik, VALIGNQZ128rrikz)>;

def 4GM7WriteALIGNXLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteALIGNX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNXLd], (instrs VALIGNDZ128rmik, VALIGNDZ128rmikz,
                                          VALIGNQZ128rmik, VALIGNQZ128rmikz)>;

def 4GM7WriteALIGNY:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNY], (instrs VALIGNDZ256rri, VALIGNDZ256rrikz,
                                        VALIGNQZ256rri, VALIGNQZ256rrikz)>;

def 4GM7WriteALIGNYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteALIGNY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNYLd], (instrs VALIGNDZ256rmi, VALIGNDZ256rmikz,
                                          VALIGNQZ256rmi, VALIGNQZ256rmikz)>;

def 4GM7WriteALIGNYrrik:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNYrrik], (instrs VALIGNDZ256rrik, VALIGNQZ256rrik)>;

def 4GM7WriteALIGNYrmik:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteALIGNYrrik.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteALIGNYrmik], (instrs VALIGNDZ256rmik, VALIGNQZ256rmik)>;

def 4GM7WriteALIGNZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 8;
}
def : InstRW<[4GM7WriteALIGNZ], (instrs VALIGNDZrri, VALIGNQZrri,
                                        VALIGNDZrrik, VALIGNQZrrik,
                                        VALIGNDZrrikz, VALIGNQZrrikz)>;

def 4GM7WriteALIGNZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteALIGNZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 8;
}
def : InstRW<[4GM7WriteALIGNZLd], (instrs VALIGNDZrmi, VALIGNQZrmi,
                                          VALIGNDZrmik, VALIGNQZrmik,
                                          VALIGNDZrmikz, VALIGNQZrmikz)>;

// PALIGN
def 4GM7WritePALIGNX:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNX], (instrs VPALIGNRZrrik, VPALIGNRZrrikz)>;

def 4GM7WritePALIGNXLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePALIGNX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNXLd], (instrs VPALIGNRZrmik, VPALIGNRZrmikz)>;

def 4GM7WritePALIGNY:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNY], (instrs VPALIGNRZ128rrik, VPALIGNRZ128rrikz)>;

def 4GM7WritePALIGNYLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePALIGNY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNYLd], (instrs VPALIGNRZ128rmik, VPALIGNRZ128rmikz)>;

def 4GM7WritePALIGNZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNZ], (instrs VPALIGNRZ256rrik, VPALIGNRZ256rrikz)>;

def 4GM7WritePALIGNZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WritePALIGNZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WritePALIGNZLd], (instrs VPALIGNRZ256rmik, VPALIGNRZ256rmikz)>;

// DBPSADBWZ
def 4GM7WriteDBPSADBWZ:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteDBPSADBWZ], (instrs VDBPSADBWZrri, VDBPSADBWZrrikz,
                                           VDBPSADBWZ128rri, VDBPSADBWZ128rrikz,
                                           VDBPSADBWZ256rri, VDBPSADBWZ256rrikz)>;

def 4GM7WriteDBPSADBWZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteDBPSADBWZ.Latency);
  let ReleaseAtCycles = [1, 1, 6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteDBPSADBWZLd], (instrs VDBPSADBWZrmi, VDBPSADBWZrmikz,
                                             VDBPSADBWZ128rmi, VDBPSADBWZ128rmikz,
                                             VDBPSADBWZ256rmi, VDBPSADBWZ256rmikz)>;

def 4GM7WriteDBPSADBWZrrik:SchedWriteRes<[4GM7FPU02, 4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteDBPSADBWZrrik], (instrs VDBPSADBWZrrik, VDBPSADBWZ128rrik, VDBPSADBWZ256rrik)>;

def 4GM7WriteDBPSADBWZrmikLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteDBPSADBWZrrik.Latency);
  let ReleaseAtCycles = [1, 1, 6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[4GM7WriteDBPSADBWZrmikLd], (instrs VDBPSADBWZrmik, VDBPSADBWZ128rmik, VDBPSADBWZ256rmik)>;

// VecLogicZ rrk
def 4GM7WriteVecLogicZ:SchedWriteRes<[4GM7FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVecLogicZ], (instregex "VPAND(D|Q)Z(128|256)?rrk$",
                                               "VPANDN(D|Q)Z(128|256)?rrk$",
                                               "VPOR(D|Q)Z(128|256)?rrk$",
                                               "VPXOR(D|Q)Z(128|256)?rrk$",
                                               "VANDNP(D|S)Z(128|256)?rrk$",
                                               "VANDP(D|S)Z(128|256)?rrk$",
                                               "VORP(D|S)Z(128|256)?rrk$",
                                               "VXORP(D|S)Z(128|256)?rrk$")>;

def 4GM7WriteVecLogicZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVecLogicZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVecLogicZLd], (instregex "VPAND(D|Q)Z(128|256)?rmk$",
                                                  "VPANDN(D|Q)Z(128|256)?rmk$",
                                                  "VPOR(D|Q)Z(128|256)?rmk$",
                                                  "VPXOR(D|Q)Z(128|256)?rmk$",
                                                  "VANDNP(D|S)Z(128|256)?rmk$",
                                                  "VANDP(D|S)Z(128|256)?rmk$",
                                                  "VORP(D|S)Z(128|256)?rmk$",
                                                  "VXORP(D|S)Z(128|256)?rmk$")>;

// GF2P8MULB
def 4GM7WriteGF2P8MULB:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteGF2P8MULB], (instrs GF2P8MULBrr, VGF2P8MULBrr, VGF2P8MULBYrr)>;

def 4GM7WriteGF2P8MULBLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteGF2P8MULB.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteGF2P8MULBLd], (instrs GF2P8MULBrm, VGF2P8MULBrm, VGF2P8MULBYrm)>;

def 4GM7WriteGF2P8MULBXYZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteGF2P8MULBXYZ], (instrs VGF2P8MULBZrr, VGF2P8MULBZrrkz,
                                              VGF2P8MULBZ128rr, VGF2P8MULBZ128rrkz,
                                              VGF2P8MULBZ256rr, VGF2P8MULBZ256rrkz)>;

def 4GM7WriteGF2P8MULBXYZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteGF2P8MULBXYZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteGF2P8MULBXYZLd], (instrs VGF2P8MULBZrm, VGF2P8MULBZrmkz,
                                                VGF2P8MULBZ128rm, VGF2P8MULBZ128rmkz,
                                                VGF2P8MULBZ256rm, VGF2P8MULBZ256rmkz)>;

def 4GM7WriteGF2P8MULBXYZrrk:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteGF2P8MULBXYZrrk], (instrs VGF2P8MULBZrrk, VGF2P8MULBZ128rrk, VGF2P8MULBZ256rrk)>;

def 4GM7WriteGF2P8MULBXYZrmk:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteGF2P8MULBXYZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteGF2P8MULBXYZrmk], (instrs VGF2P8MULBZrmk, VGF2P8MULBZ128rmk, VGF2P8MULBZ256rmk)>;

// VGF2P8AFFINE
def 4GM7WriteVGF2P8AFFINEZ:SchedWriteRes<[4GM7FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVGF2P8AFFINEZ], (instrs VGF2P8AFFINEQBZrrik, VGF2P8AFFINEQBZ128rrik, VGF2P8AFFINEQBZ256rrik,
                                               VGF2P8AFFINEINVQBZrrik, VGF2P8AFFINEINVQBZ128rrik, VGF2P8AFFINEINVQBZ256rrik)>;

def 4GM7WriteVGF2P8AFFINEZLd:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU13]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteVGF2P8AFFINEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteVGF2P8AFFINEZLd], (instrs VGF2P8AFFINEQBZrmik, VGF2P8AFFINEQBZ128rmik, VGF2P8AFFINEQBZ256rmik,
                                                 VGF2P8AFFINEINVQBZrmik, VGF2P8AFFINEINVQBZ128rmik, VGF2P8AFFINEINVQBZ256rmik)>;

// Vector insert/extract operations.
defm : 4GM7WriteResFPPair<WriteVecInsert, [4GM7FPU13], 2>;
defm : 4GM7WriteRes<WriteVecExtract, [4GM7FPU02], 1, [4], 1>;
defm : 4GM7WriteRes<WriteVecExtractSt, [4GM7FPU02, 4GM7AGU, 4GM7Store], 3, [4, 1, 1], 1>;

// MOVMSK Instructions.
defm : 4GM7WriteRes<WriteFMOVMSK, [4GM7FPU13], 1>;
defm : 4GM7WriteRes<WriteMMXMOVMSK, [4GM7FPU13], 1>;
defm : 4GM7WriteRes<WriteVecMOVMSK, [4GM7FPU13], 1>;
defm : 4GM7WriteRes<WriteVecMOVMSKY, [4GM7FPU13], 1>;

// Strings instructions.
defm : 4GM7WriteResFPPair<WritePCmpIStrM, [4GM7FPU], 7, [25], 3>;
defm : 4GM7WriteResFPPair<WritePCmpEStrM, [4GM7FPU], 7, [25], 8>;
defm : 4GM7WriteResFPPair<WritePCmpIStrI, [4GM7FPU], 2, [8], 2>;
defm : 4GM7WriteResFPPair<WritePCmpEStrI, [4GM7FPU], 8, [13], 8>;

// AES Instructions.
defm : 4GM7WriteResFPPair<WriteAESDecEnc, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteAESIMC, [4GM7FPU13], 3>;
defm : 4GM7WriteResFPPair<WriteAESKeyGen, [4GM7FPU13], 3>;

// SHA Instructions.

def 4GM7WriteSHA1rr:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA1rr], (instrs SHA1MSG1rr, SHA1MSG2rr, SHA1NEXTErr)>;

def 4GM7WriteSHA1rm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSHA1rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA1rm], (instrs SHA1MSG1rm, SHA1MSG2rm, SHA1NEXTErm)>;

def 4GM7WriteSHA256MSG1rr:SchedWriteRes<[4GM7FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteSHA256MSG1rr], (instrs SHA256MSG1rr)>;

def 4GM7WriteSHA256MSG1rm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSHA256MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteSHA256MSG1rm], (instrs SHA256MSG1rm)>;

def 4GM7WriteSHA256rr:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA256rr], (instrs SHA256MSG2rr, SHA256RNDS2rr)>;

def 4GM7WriteSHA256rm:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSHA256rr.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA256rm], (instrs SHA256MSG2rm, SHA256RNDS2rm)>;

def 4GM7WriteSHA1RNDS4rri:SchedWriteRes<[4GM7FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA1RNDS4rri], (instrs SHA1RNDS4rri)>;

def 4GM7WriteSHA1RNDS4rmi:SchedWriteRes<[4GM7AGU, 4GM7Load, 4GM7FPU02]> {
  let Latency = !add(C864GM7Model.VecLoadLatency, 4GM7WriteSHA1RNDS4rri.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[4GM7WriteSHA1RNDS4rmi], (instrs SHA1RNDS4rmi)>;

defm : 4GM7WriteRes<WriteFence, [4GM7AGU]>;
defm : 4GM7WriteRes<WriteNop, []>;

//===----------------------------------------------------------------------===//
// Zero Cycle Move
//===----------------------------------------------------------------------===//

def 4GM7WriteZeroLatency : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}

def : InstRW<[4GM7WriteZeroLatency], (instrs MOV32rr, MOV32rr_REV,
                                             MOV64rr, MOV64rr_REV,
                                             MOVSX32rr32)>;

def 4GM7WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[4GM7WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                                XCHG64rr, XCHG64ar)>;

defm : 4GM7WriteRes<WriteXCHG, [4GM7ALU], 0, [8], 2>;

defm : 4GM7WriteRes<WriteFMoveX, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteFMoveY, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteFMoveZ, [], 0, [], 1>;

defm : 4GM7WriteRes<WriteVecMove, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteVecMoveX, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteVecMoveY, [], 0, [], 1>;
defm : 4GM7WriteRes<WriteVecMoveZ, [], 0, [], 1>;

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

def 4GM7WriteZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteALU]>
]>;
def : InstRW<[4GM7WriteZeroIdiom], (instrs XOR32rr, XOR32rr_REV,
                                           XOR64rr, XOR64rr_REV,
                                           SUB32rr, SUB32rr_REV,
                                           SUB64rr, SUB64rr_REV)>;

def 4GM7WriteZeroIdiomEFLAGS : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<CheckSameRegOperand<0, 1>>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                                 [WriteALU]>
]>;
def : InstRW<[4GM7WriteZeroIdiomEFLAGS], (instrs CMP8rr,  CMP8rr_REV,
                                                 CMP16rr, CMP16rr_REV,
                                                 CMP32rr, CMP32rr_REV,
                                                 CMP64rr, CMP64rr_REV)>;

def 4GM7WriteFZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogic]>
]>;

def : InstRW<[4GM7WriteFZeroIdiom], (instrs VXORPSrr, VXORPDrr,
                                            VXORPSZ128rr,
                                            VXORPDZ128rr,
                                            VANDNPSrr, VANDNPDrr,
                                            VANDNPSZ128rr,
                                            VANDNPDZ128rr)>;

def 4GM7WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[4GM7WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                             VXORPSZ256rr,
                                             VXORPDZ256rr,
                                             VANDNPSYrr, VANDNPDYrr,
                                             VANDNPSZ256rr,
                                             VANDNPDZ256rr)>;

def 4GM7WriteFZeroIdiomZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicZ]>
]>;
def : InstRW<[4GM7WriteFZeroIdiomZ], (instrs VXORPSZrr, VXORPDZrr,
                                             VANDNPSZrr, VANDNPDZrr)>;

def 4GM7WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;

def : InstRW<[4GM7WriteVZeroIdiomLogicX], (instrs VPXORrr,
                                                  VPXORDZ128rr,
                                                  VPXORQZ128rr,
                                                  VPANDNrr,
                                                  VPANDNDZ128rr,
                                                  VPANDNQZ128rr)>;

def 4GM7WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[4GM7WriteVZeroIdiomLogicY], (instrs VPXORYrr,
                                                  VPXORDZ256rr,
                                                  VPXORQZ256rr,
                                                  VPANDNYrr,
                                                  VPANDNDZ256rr,
                                                  VPANDNQZ256rr)>;

def 4GM7WriteVZeroIdiomLogicZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicZ]>
]>;
def : InstRW<[4GM7WriteVZeroIdiomLogicZ], (instrs VPXORDZrr, VPXORQZrr,
                                                  VPANDNDZrr, VPANDNQZrr)>;

def 4GM7WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;


def : InstRW<[4GM7WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPSUBBZ128rr, VPSUBWZ128rr, VPSUBDZ128rr, VPSUBQZ128rr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr,
                     VPCMPGTBZ128rr, VPCMPGTWZ128rr,
                     VPCMPGTDZ128rr, VPCMPGTQZ128rr)>;

def 4GM7WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[4GM7WriteVZeroIdiomALUY],
             (instrs VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
                     VPSUBBZ256rr, VPSUBWZ256rr, VPSUBDZ256rr, VPSUBQZ256rr,
                     VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr,
                     VPCMPGTBZ256rr, VPCMPGTWZ256rr,
                     VPCMPGTDZ256rr, VPCMPGTQZ256rr)>;

def 4GM7WriteVZeroIdiomALUZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [4GM7WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUZ]>
]>;
def : InstRW<[4GM7WriteVZeroIdiomALUY],
             (instrs VPSUBBZrr, VPSUBWZrr, VPSUBDZrr, VPSUBQZrr,
                     VPCMPGTBZrr, VPCMPGTWZrr, VPCMPGTDZrr, VPCMPGTQZrr)>;

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

  // AVX ZMM Zero-idioms.
  DepBreakingClass<[
    // fp variants.
    VXORPSZrr, VXORPDZrr,
    VXORPSZ128rr, VXORPDZ128rr, VXORPSZ256rr, VXORPDZ256rr,
    VANDNPSZrr, VANDNPDZrr,
    VANDNPSZ128rr, VANDNPDZ128rr, VANDNPSZ256rr, VANDNPDZ256rr,

    // int variants.
    VPCMPGTBZrr, VPCMPGTWZrr, VPCMPGTDZrr, VPCMPGTQZrr,
    VPCMPGTBZ128rr, VPCMPGTWZ128rr, VPCMPGTDZ128rr, VPCMPGTQZ128rr,
    VPCMPGTBZ256rr, VPCMPGTWZ256rr, VPCMPGTDZ256rr, VPCMPGTQZ256rr,
    VPANDNDZrr, VPANDNQZrr,
    VPANDNDZ128rr, VPANDNQZ128rr, VPANDNDZ256rr, VPANDNQZ256rr,
    VPXORDZrr, VPXORQZrr,
    VPXORDZ128rr, VPXORQZ128rr, VPXORDZ256rr, VPXORQZ256rr,
    VPSUBBZrr, VPSUBWZrr, VPSUBDZrr, VPSUBQZrr,
    VPSUBBZ128rr, VPSUBWZ128rr, VPSUBDZ128rr, VPSUBQZ128rr,
    VPSUBBZ256rr, VPSUBWZ256rr, VPSUBDZ256rr, VPSUBQZ256rr,
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