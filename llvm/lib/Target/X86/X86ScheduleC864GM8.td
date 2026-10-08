//=- X86ScheduleC864GM8.td - X86 C86-4G-M8 Scheduling --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for C86-4G-M8 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def C864GM8Model : SchedMachineModel {
  let IssueWidth = 6;
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

let SchedModel = C864GM8Model in {

//===----------------------------------------------------------------------===//
// Integer Execution Unit
//===----------------------------------------------------------------------===//

// The C864GM8 has 4 ALUs.
def C4GM8ALU0 : ProcResource<1>;
def C4GM8ALU1 : ProcResource<1>;
def C4GM8ALU2 : ProcResource<1>;
def C4GM8ALU3 : ProcResource<1>;

def C4GM8ALU : ProcResGroup<[C4GM8ALU0, C4GM8ALU1, C4GM8ALU2, C4GM8ALU3]>;
def C4GM8ALU12 : ProcResGroup<[C4GM8ALU1, C4GM8ALU2]>;
def C4GM8ALU03 : ProcResGroup<[C4GM8ALU0, C4GM8ALU3]>;

// BRU0 on ALU0.
defvar C4GM8BRU0 = C4GM8ALU0;
// BRU1 is a separate branch execution unit.
def C4GM8BRU1 : ProcResource<1>;

def C4GM8BRU : ProcResGroup<[C4GM8BRU0, C4GM8BRU1]>;

// The C864GM8 has 3 AGUs.
def C4GM8AGU0 : ProcResource<1>;
def C4GM8AGU1 : ProcResource<1>;
def C4GM8AGU2 : ProcResource<1>;

def C4GM8AGU : ProcResGroup<[C4GM8AGU0, C4GM8AGU1, C4GM8AGU2]>;

// 96 Entry (4x24 entries) integer Scheduler.
def C4GM8Int : ProcResGroup<[C4GM8ALU0, C4GM8AGU0, C4GM8BRU0,
                             C4GM8ALU1, C4GM8AGU1,
                             C4GM8ALU2, C4GM8AGU2,
                             C4GM8ALU3,            C4GM8BRU1]> {
  let BufferSize = 96;
}

// The integer physical register file consists of 224 registers.
def C4GM8IntegerPRF : RegisterFile<224,
                                   [GR64, CCR],
                                   [1, 1],
                                   [1, 0],
                                   6,  // Max moves that can be eliminated per cycle.
                                   0>; // Restrict move elimination to zero regs.

//===----------------------------------------------------------------------===//
// Floating-Point Unit
//===----------------------------------------------------------------------===//

// The C864GM8 has 4 FPUs.
def C4GM8FPU0 : ProcResource<1>;
def C4GM8FPU1 : ProcResource<1>;
def C4GM8FPU2 : ProcResource<1>;
def C4GM8FPU3 : ProcResource<1>;

def C4GM8FPU013 : ProcResGroup<[C4GM8FPU0, C4GM8FPU1, C4GM8FPU3]>;
def C4GM8FPU01  : ProcResGroup<[C4GM8FPU0, C4GM8FPU1]>;
def C4GM8FPU12  : ProcResGroup<[C4GM8FPU1, C4GM8FPU2]>;
def C4GM8FPU13  : ProcResGroup<[C4GM8FPU1, C4GM8FPU3]>;
def C4GM8FPU23  : ProcResGroup<[C4GM8FPU2, C4GM8FPU3]>;
def C4GM8FPU02  : ProcResGroup<[C4GM8FPU0, C4GM8FPU2]>;
def C4GM8FPU03  : ProcResGroup<[C4GM8FPU0, C4GM8FPU3]>;

// 48 floating-point Scheduler Queue.
def C4GM8FPU : ProcResGroup<[C4GM8FPU0, C4GM8FPU1, C4GM8FPU2, C4GM8FPU3]> {
  let BufferSize = 48;
}

// 28 floating-point store Scheduler Queue.
def C4GM8FPU45 : ProcResource<2> {
  let BufferSize = 28;
}

// FP store and Int2FP
defvar C4GM8FPLd01 = C4GM8FPU45;

// 1 FP store per cycle. 
let Super = C4GM8FPLd01 in
def C4GM8FPSt : ProcResource<1>;

// The floating point physical register file consists of 256 bits x 208 registers.
def C4GM8FpuPRF : RegisterFile<224,
                               [VR64, VR128, VR256, VR512],
                               [1, 1, 1, 2],
                               [0, 1, 1, 0],
                               6,  // Max moves that can be eliminated per cycle.
                               0>; // Restrict move elimination to zero regs.


//===----------------------------------------------------------------------===//
// Load-Store Unit
//===----------------------------------------------------------------------===//

// Load/Store Units and Memory Queues
// The C864GM8 has 4 LS Units.
def C4GM8LSU : ProcResource<4>;

// The three of LSU can be loads.
let Super = C4GM8LSU in
def C4GM8Load : ProcResource<3> {
  // The LDQ is 54.
  let BufferSize = 54;
}
def C4GM8LoadQueue : LoadQueue<C4GM8Load>;

// All four of LSU can be loads.
let Super = C4GM8LSU in
def C4GM8Store : ProcResource<4> {
  // The STQ is 52.
  let BufferSize = 52;
}
def C4GM8StoreQueue : StoreQueue<C4GM8Store>;

def : ReadAdvance<ReadAfterLd, C864GM8Model.LoadLatency>;
def : ReadAdvance<ReadAfterVecLd, C864GM8Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecXLd, C864GM8Model.VecLoadLatency>;
def : ReadAdvance<ReadAfterVecYLd, C864GM8Model.VecLoadLatency>;
def : ReadAdvance<ReadInt2Fpu, -1>;


//===----------------------------------------------------------------------===//
// Retire Control Unit
//===----------------------------------------------------------------------===//
def C4GM8RCU : RetireControlUnit<C864GM8Model.MicroOpBufferSize, 8>;

//===----------------------------------------------------------------------===//
// Basic helper classes.
//===----------------------------------------------------------------------===//

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.


multiclass __C4GM8WriteRes<SchedWrite SchedRW, list<ProcResourceKind> ExePorts,
                         int Lat = 1, list<int> Res = [], int UOps = 1> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ReleaseAtCycles = Res;
    let NumMicroOps = UOps;
  }
}

multiclass __C4GM8WriteResPair<X86FoldableSchedWrite SchedRW,
                             list<ProcResourceKind> ExePorts, int Lat,
                             list<int> Res, int UOps, int LoadLat, int LoadUOps,
                             ProcResourceKind AGU, int LoadRes> {
  defm : __C4GM8WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;

  defm : __C4GM8WriteRes<SchedRW.Folded,
                       !listconcat([AGU, C4GM8Load], ExePorts),
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
multiclass C4GM8WriteRes<SchedWrite SchedRW,
                        list<ProcResourceKind> ExePorts, int Lat = 1,
                        list<int> Res = [], int UOps = 1> {
  defm : __C4GM8WriteRes<SchedRW, ExePorts, Lat, Res, UOps>;
}

// For classes with folded loads.
multiclass C4GM8WriteResIntPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __C4GM8WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM8Model.LoadLatency,
                           LoadUOps, C4GM8AGU, LoadRes>;
}

multiclass C4GM8WriteResFPPair<X86FoldableSchedWrite SchedRW,
                              list<ProcResourceKind> ExePorts, int Lat = 1,
                              list<int> Res = [], int UOps = 1,
                              int LoadUOps = 0, int LoadRes = 1> {
  defm : __C4GM8WriteResPair<SchedRW, ExePorts, Lat, Res, UOps,
                           C864GM8Model.VecLoadLatency,
                           LoadUOps, C4GM8FPLd01, LoadRes>;
}

// Microcoded Instructions
def C4GM8WriteMicrocoded : SchedWriteRes<[]> {
  let Latency = 100;
}

def : SchedAlias<WriteMicrocoded, C4GM8WriteMicrocoded>;
def : SchedAlias<WriteSystem, C4GM8WriteMicrocoded>;
def : SchedAlias<WriteFCMOV, C4GM8WriteMicrocoded>;
def : SchedAlias<WriteLDMXCSR, C4GM8WriteMicrocoded>;
def : SchedAlias<WriteSTMXCSR, C4GM8WriteMicrocoded>;

// Integer Instructions

defm : C4GM8WriteRes<WriteRMW, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency, [1, 1], 0>;

defm : C4GM8WriteRes<WriteStore, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency, [1, 2], 1>;
defm : C4GM8WriteRes<WriteStoreNT, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency, [1, 2], 1>;
defm : C4GM8WriteRes<WriteLoad, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.LoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteMove, [C4GM8ALU], 1, [4], 1>;

def : InstRW<[WriteMove], (instrs COPY, MOVSX16rr16, MOVSX16rr32,
                                  MOVZX16rr16, MOVSX16rr8, MOVZX16rr8)>;

defm : C4GM8WriteRes<WriteVecMaskedGatherWriteback, [], !add(1, C864GM8Model.VecLoadLatency)>;

defm : C4GM8WriteRes<WriteZero, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteLEA, [C4GM8AGU]>;
defm : C4GM8WriteResIntPair<WriteALU, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteADC, [C4GM8ALU], 1>;

// This write is used for slow LEA instructions.
def C4GM8Write3OpsLEA : SchedWriteRes<[C4GM8ALU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
// a slow LEA is either a 3Ops LEA (base, index, offset),
// or an LEA with a `Scale` value different than 1.
def C4GM8SlowLEAPredicate : MCSchedPredicate<
  CheckAny<[
    IsThreeOperandsLEAFn,
    CheckAll<[
      CheckIsImmOperand<2>,
      CheckNot<CheckImmOperand<2, 1>>
    ]>
  ]>
>;

def C4GM8WriteLEA : SchedWriteVariant<[
    SchedVar<C4GM8SlowLEAPredicate, [C4GM8Write3OpsLEA]>,
    SchedVar<NoSchedPred,          [WriteLEA]>
]>;

def : InstRW<[C4GM8WriteLEA], (instrs LEA64r, LEA64_32r)>;

// MUL IMUL MULX
defm : C4GM8WriteResIntPair<WriteIMul8, [C4GM8ALU1], 3, [3], 1>;
defm : C4GM8WriteResIntPair<WriteIMul16, [C4GM8ALU1], 3, [3], 1>;
defm : C4GM8WriteResIntPair<WriteIMul16Imm, [C4GM8ALU1], 4, [3], 2>;
defm : C4GM8WriteResIntPair<WriteIMul16Reg, [C4GM8ALU1], 3>;
defm : C4GM8WriteResIntPair<WriteIMul32, [C4GM8ALU1], 3, [3], 1>;
defm : C4GM8WriteResIntPair<WriteIMul32Imm, [C4GM8ALU1], 3, [3], 2>;
defm : C4GM8WriteResIntPair<WriteIMul32Reg, [C4GM8ALU1], 3>;
defm : C4GM8WriteResIntPair<WriteIMul64, [C4GM8ALU1], 3, [4], 1>;
defm : C4GM8WriteResIntPair<WriteIMul64Imm, [C4GM8ALU1], 3, [4], 2>;
defm : C4GM8WriteResIntPair<WriteIMul64Reg, [C4GM8ALU1], 3>;

defm : C4GM8WriteResIntPair<WriteMULX32, [C4GM8ALU1], 3, [2], 2>;
defm : C4GM8WriteResIntPair<WriteMULX64, [C4GM8ALU1], 4, [2], 2>;
defm : C4GM8WriteRes<WriteIMulHLd, [C4GM8ALU1], !add(3, C864GM8Model.LoadLatency), [], 0>;
defm : C4GM8WriteRes<WriteIMulH, [C4GM8ALU1], 3, [], 0>;

// IDIV
defm : C4GM8WriteResIntPair<WriteDiv8, [C4GM8ALU3], 9, [4], 2>;
defm : C4GM8WriteResIntPair<WriteDiv16, [C4GM8ALU3], 10, [4], 2>;
defm : C4GM8WriteResIntPair<WriteDiv32, [C4GM8ALU3], 12, [7], 2>;
defm : C4GM8WriteResIntPair<WriteDiv64, [C4GM8ALU3], 18, [10], 2>;
defm : C4GM8WriteResIntPair<WriteIDiv8, [C4GM8ALU3], 9, [4], 2>;
defm : C4GM8WriteResIntPair<WriteIDiv16, [C4GM8ALU3], 10, [4], 2>;
defm : C4GM8WriteResIntPair<WriteIDiv32, [C4GM8ALU3], 12, [7], 2>;
defm : C4GM8WriteResIntPair<WriteIDiv64, [C4GM8ALU3], 18, [10], 2>;

defm : C4GM8WriteRes<WriteBSWAP32, [C4GM8ALU], 1, [1], 1>;
defm : C4GM8WriteRes<WriteBSWAP64, [C4GM8ALU], 1, [1], 1>;
defm : C4GM8WriteRes<WriteCMPXCHG, [C4GM8ALU], 3, [12], 5>;
defm : C4GM8WriteRes<WriteCMPXCHGRMW,[C4GM8ALU], 3, [12], 6>;

def C4GM8WriteXADD : SchedWriteRes<[C4GM8ALU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteXADD], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr)>;

def C4GM8WriteXADDLd : SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8ALU]> {
  let Latency = !add(C864GM8Model.LoadLatency, C4GM8WriteXADD.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteXADDLd], (instrs XADD8rm, XADD16rm, XADD32rm, XADD64rm)>;

def C4GM8WriteCMPXCHG8rr : SchedWriteRes<[C4GM8ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM8WriteCMPXCHG8rr], (instrs CMPXCHG8rr)>;

def C4GM8WriteCMPXCHG8mr : SchedWriteRes<[C4GM8ALU, C4GM8AGU, C4GM8Store]> {
  let Latency = !add(C4GM8WriteCMPXCHG8rr.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [12, 1, 1];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteCMPXCHG8mr], (instrs CMPXCHG8rm, LCMPXCHG8)>;

def C4GM8WriteXCHG : SchedWriteRes<[C4GM8ALU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteXCHG], (instrs XCHG8rr, XCHG16ar, XCHG16rr)>;

def C4GM8WriteXCHGLd : SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8ALU]> {
  let Latency = !add(C864GM8Model.LoadLatency, C4GM8WriteXCHG.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteXCHGLd], (instrs XCHG8rm, XCHG16rm)>;

defm : C4GM8WriteResIntPair<WriteShift, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteShiftCL, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteRotate, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteRotateCL, [C4GM8ALU], 1>;

def C4GM8WriteRCR:SchedWriteRes<[C4GM8ALU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRCR], (instregex "RCR(8|16|32|64)rCL")>;

def C4GM8WriteRCRLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8ALU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRCR.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRCRLd], (instregex "RCR(8|16|32|64)mCL")>;

def C4GM8WriteRCL:SchedWriteRes<[C4GM8ALU]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRCL], (instregex "RCL(8|16|32|64)rCL")>;

def C4GM8WriteRCLLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8ALU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRCL.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRCLLd], (instregex "RCL(8|16|32|64)mCL")>;

defm : C4GM8WriteRes<WriteSHDrri, [C4GM8ALU], 3, [4], 6>;
defm : C4GM8WriteRes<WriteSHDrrcl, [C4GM8ALU], 3, [4], 7>;
defm : C4GM8WriteRes<WriteSHDmri, [C4GM8ALU, C4GM8AGU, C4GM8Store], !add(3, C864GM8Model.StoreLatency), [4, 1, 1], 8>;
defm : C4GM8WriteRes<WriteSHDmrcl, [C4GM8ALU, C4GM8AGU, C4GM8Store], !add(3, C864GM8Model.StoreLatency), [4, 1, 1], 8>;

defm : C4GM8WriteResIntPair<WriteJump, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteCRC32, [C4GM8ALU], 3, [12], 3>;

defm : C4GM8WriteResIntPair<WriteCMOV, [C4GM8ALU], 1>;
defm : C4GM8WriteRes<WriteSETCC, [C4GM8ALU], 1, [1], 1>;
defm : C4GM8WriteRes<WriteSETCCStore, [C4GM8ALU, C4GM8AGU, C4GM8Store], 1, [2, 1, 1], 2>;
defm : C4GM8WriteRes<WriteLAHFSAHF, [C4GM8ALU], 2, [2], 4>;

defm : C4GM8WriteRes<WriteBitTest, [C4GM8ALU], 1, [1], 1>;
defm : C4GM8WriteRes<WriteBitTestImmLd, [C4GM8AGU, C4GM8Load, C4GM8ALU], !add(C864GM8Model.LoadLatency, 1), [1,1,1], 2>;
defm : C4GM8WriteRes<WriteBitTestRegLd, [C4GM8AGU, C4GM8Load, C4GM8ALU], !add(C864GM8Model.LoadLatency, 1), [1,1,1], 6>;
defm : C4GM8WriteRes<WriteBitTestSet, [C4GM8ALU], 2, [1], 2>;
defm : C4GM8WriteRes<WriteBitTestSetImmLd, [C4GM8AGU, C4GM8Load, C4GM8ALU], !add(C864GM8Model.LoadLatency, 2), [1, 1, 1], 4>;
defm : C4GM8WriteRes<WriteBitTestSetRegLd, [C4GM8AGU, C4GM8Load, C4GM8ALU], !add(C864GM8Model.LoadLatency, 2), [1, 1, 1], 8>;

// Bit counts.
defm : C4GM8WriteResIntPair<WriteBSF, [C4GM8ALU], 3, [12], 6, -2>;
defm : C4GM8WriteResIntPair<WriteBSR, [C4GM8ALU], 4, [16], 6, -3>;
defm : C4GM8WriteResIntPair<WriteLZCNT, [C4GM8ALU], 1>;
defm : C4GM8WriteResIntPair<WriteTZCNT, [C4GM8ALU], 2, [2], 2>;
defm : C4GM8WriteResIntPair<WritePOPCNT, [C4GM8ALU], 1>;

// PLZCNT
def C4GM8WritePLZCNTXY:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePLZCNTXY], (instregex "VPLZCNT(D|Q)(Z128|Z256)rr(k|kz)")>;

def C4GM8WritePLZCNTXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePLZCNTXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePLZCNTXYLd], (instregex "VPLZCNT(D|Q)(Z128|Z256)rm(k|kz)")>;

def C4GM8WritePLZCNTZ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePLZCNTZ], (instregex "VPLZCNT(D|Q)Zrr(k|kz)")>;

def C4GM8WritePLZCNTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePLZCNTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePLZCNTZLd], (instregex "VPLZCNT(D|Q)Zrm(k|kz)")>;

// BMI1 BEXTR, BMI2 BZHI
defm : C4GM8WriteResIntPair<WriteBEXTR, [C4GM8ALU], 1, [1], 1>;
defm : C4GM8WriteResIntPair<WriteBLS, [C4GM8ALU], 2, [2], 3>;
defm : C4GM8WriteResIntPair<WriteBZHI, [C4GM8ALU], 1>;


// Floating Point Instructions
defm : C4GM8WriteRes<WriteFLD0, [C4GM8AGU, C4GM8Load, C4GM8FPU1], !add(C864GM8Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM8WriteRes<WriteFLD1, [C4GM8AGU, C4GM8Load, C4GM8FPU1], !add(C864GM8Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM8WriteRes<WriteFLDC, [C4GM8AGU, C4GM8Load, C4GM8FPU1], !add(C864GM8Model.LoadLatency, 4), [1, 1, 1], 1>;
defm : C4GM8WriteRes<WriteFLoad, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1)>;
defm : C4GM8WriteRes<WriteFLoadX, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1)>;
defm : C4GM8WriteRes<WriteFLoadY, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1)>;
defm : C4GM8WriteRes<WriteFMaskedLoad, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1)>;
defm : C4GM8WriteRes<WriteFMaskedLoadY, [C4GM8AGU, C4GM8Load, C4GM8FPU01], !add(C864GM8Model.VecLoadLatency, 1), [1, 1, 1], 2>;

defm : C4GM8WriteRes<WriteFStore, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFStoreX, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFStoreY, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFStoreNT, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteFStoreNTX, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteFStoreNTY, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteFMaskedStore32, [C4GM8FPSt, C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFMaskedStore32Y, [C4GM8FPSt, C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFMaskedStore64, [C4GM8FPSt, C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteFMaskedStore64Y, [C4GM8FPSt, C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;

defm : C4GM8WriteResFPPair<WriteFAdd, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAddX, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAddY, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAddZ, [C4GM8FPU13], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFAdd64, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAdd64X, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAdd64Y, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteFAdd64Z, [C4GM8FPU13], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFHAdd, [C4GM8FPU13], 7, [8], 4, 3>;
defm : C4GM8WriteResFPPair<WriteFHAddY, [C4GM8FPU13], 7, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WriteFHAddZ>;
defm : C4GM8WriteResFPPair<WritePHAdd, [C4GM8FPU13], 3, [8], 4, 3>;
defm : C4GM8WriteResFPPair<WritePHAddX, [C4GM8FPU13], 3, [8], 4, 3>;
defm : C4GM8WriteResFPPair<WritePHAddY, [C4GM8FPU13], 3, [8], 4, 3>;
defm : X86WriteResPairUnsupported<WritePHAddZ>;
defm : C4GM8WriteResFPPair<WriteFMul, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMulX, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMulY, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMulZ, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFMul64, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMul64X, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMul64Y, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFMul64Z, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFMA, [C4GM8FPU02], 4>;
defm : C4GM8WriteResFPPair<WriteFMAX, [C4GM8FPU02], 4>;
defm : C4GM8WriteResFPPair<WriteFMAY, [C4GM8FPU02], 4>;
defm : C4GM8WriteResFPPair<WriteFMAZ, [C4GM8FPU02], 4, [2], 2>;
defm : C4GM8WriteResFPPair<WriteFRcp, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRcpX, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRcpY, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRcpZ, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRsqrt, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRsqrtX, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRsqrtY, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFRsqrtZ, [C4GM8FPU02], 5, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFSqrt, [C4GM8FPU13], 8, [13]>;
defm : C4GM8WriteResFPPair<WriteFSqrtX, [C4GM8FPU13], 8, [13]>;
defm : C4GM8WriteResFPPair<WriteFSqrtY, [C4GM8FPU13], 8, [13]>;
defm : C4GM8WriteResFPPair<WriteFSqrtZ, [C4GM8FPU13], 16, [13]>;
defm : C4GM8WriteResFPPair<WriteFSqrt64, [C4GM8FPU13], 8, [19]>;
defm : C4GM8WriteResFPPair<WriteFSqrt64X, [C4GM8FPU13], 8, [19]>;
defm : C4GM8WriteResFPPair<WriteFSqrt64Y, [C4GM8FPU13], 8, [19]>;
defm : C4GM8WriteResFPPair<WriteFSqrt64Z, [C4GM8FPU13], 16, [19]>;
defm : C4GM8WriteResFPPair<WriteFSqrt80, [C4GM8FPU13], 22, [23]>;
defm : C4GM8WriteResFPPair<WriteFDiv, [C4GM8FPU13], 10, [9], 1>;
defm : C4GM8WriteResFPPair<WriteFDivX, [C4GM8FPU13], 10, [9], 1>;
defm : C4GM8WriteResFPPair<WriteFDivY, [C4GM8FPU13], 10, [9], 1>;
defm : C4GM8WriteResFPPair<WriteFDivZ, [C4GM8FPU13], 18, [18], 2>;
defm : C4GM8WriteResFPPair<WriteFDiv64, [C4GM8FPU13], 8, [12], 1>;
defm : C4GM8WriteResFPPair<WriteFDiv64X, [C4GM8FPU13], 8, [12], 1>;
defm : C4GM8WriteResFPPair<WriteFDiv64Y, [C4GM8FPU13], 8, [12], 1>;
defm : C4GM8WriteResFPPair<WriteFDiv64Z, [C4GM8FPU13], 16, [24], 2>;
defm : C4GM8WriteResFPPair<WriteDPPD, [C4GM8FPU13], 9, [8], 4>;
defm : C4GM8WriteResFPPair<WriteDPPS, [C4GM8FPU13], 14, [8], 8>;
defm : C4GM8WriteResFPPair<WriteDPPSY, [C4GM8FPU13], 14, [8], 8>;

// PMADD
def C4GM8WriteVPMADDXYrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDXYrrk], (instrs VPMADDWDZ128rrk, VPMADDWDZ256rrk)>;

def C4GM8WriteVPMADDXYrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADDXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDXYrmk], (instrs VPMADDWDZ128rmk, VPMADDWDZ256rmk)>;

def C4GM8WriteVPMADDZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDZrrk], (instrs VPMADDWDZrrk)>;

def C4GM8WriteVPMADDZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADDZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDZrmk], (instrs VPMADDWDZrmk)>;


// PMADD52
def C4GM8WriteVPMADD52XY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52XY], (instregex "VPMADD52(L|H)UQ(Z128|Z256)(r|rkz)$")>;

def C4GM8WriteVPMADD52XYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADD52XY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52XYLd], (instregex "VPMADD52(L|H)UQ(Z128|Z256)(m|mkz)$")>;

def C4GM8WriteVPMADD52XYrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52XYrk], (instregex "VPMADD52(L|H)UQ(Z128|Z256)rk$")>;

def C4GM8WriteVPMADD52XYmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADD52XYrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52XYmk], (instregex "VPMADD52(L|H)UQ(Z128|Z256)mk$")>;


def C4GM8WriteVPMADD52Z:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52Z], (instregex "VPMADD52(L|H)UQZ(r|rkz)$")>;

def C4GM8WriteVPMADD52ZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADD52Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52ZLd], (instregex "VPMADD52(L|H)UQZ(m|mkz)$")>;

def C4GM8WriteVPMADD52Zrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52Zrk], (instregex "VPMADD52(L|H)UQZrk$")>;

def C4GM8WriteVPMADD52Zmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADD52Zrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADD52Zmk], (instregex "VPMADD52(L|H)UQZmk$")>;


// VMULZ VPMULZ
def C4GM8WriteVMULZ_VPMULZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVMULZ_VPMULZ], (instrs VMULPSZ128rrk, VMULPDZ128rrk,
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

def C4GM8WriteVMULZ_VPMULZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVMULZ_VPMULZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVMULZ_VPMULZLd], (instrs VMULPSZ128rmk, VMULPDZ128rmk,
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

def C4GM8WriteVPMULLQZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULLQZrrk], (instrs VPMULLQZrrk, VPMULLQZ128rrk, VPMULLQZ256rrk)>;

def C4GM8WriteVPMULLQZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMULLQZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULLQZrmk], (instrs VPMULLQZrmk, VPMULLQZ128rmk, VPMULLQZ256rmk)>;


// VPMULTISHIFTQB
def C4GM8WriteVPMULTISHIFTQBZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULTISHIFTQBZ], (instrs VPMULTISHIFTQBZ128rr, VPMULTISHIFTQBZ128rrkz,
                                                  VPMULTISHIFTQBZ256rr, VPMULTISHIFTQBZ256rrkz)>;

def C4GM8WriteVPMULTISHIFTQBZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMULTISHIFTQBZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULTISHIFTQBZLd], (instrs VPMULTISHIFTQBZ128rm, VPMULTISHIFTQBZ128rmkz,
                                                    VPMULTISHIFTQBZ256rm, VPMULTISHIFTQBZ256rmkz)>;

def C4GM8WriteVPMULTISHIFTQBZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULTISHIFTQBZrrk], (instrs VPMULTISHIFTQBZ128rrk, VPMULTISHIFTQBZ256rrk,
                                                     VPMULTISHIFTQBZrr, VPMULTISHIFTQBZrrk, VPMULTISHIFTQBZrrkz)>;

def C4GM8WriteVPMULTISHIFTQBZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMULTISHIFTQBZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMULTISHIFTQBZrmk], (instrs VPMULTISHIFTQBZ128rmk, VPMULTISHIFTQBZ256rmk,
                                                     VPMULTISHIFTQBZrm, VPMULTISHIFTQBZrmk, VPMULTISHIFTQBZrmkz)>;

// VPMADDUBSWZ
def C4GM8WriteVPMADDUBSWZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDUBSWZrrk], (instrs VPMADDUBSWZrrk, VPMADDUBSWZ128rrk, VPMADDUBSWZ256rrk)>;

def C4GM8WriteVPMADDUBSWZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPMADDUBSWZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPMADDUBSWZrmk], (instrs VPMADDUBSWZrmk, VPMADDUBSWZ128rmk, VPMADDUBSWZ256rmk)>;

// FADD rrb rrbk
def C4GM8WriteFADD:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteFADD], (instregex "VADD(S|P)(S|D)(Z|Z128|Z256)rr(bk|k)$",
                                          "VSUB(S|P)(S|D)(Z|Z128|Z256)rr(bk|k)$")>;

def C4GM8WriteFADDLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteFADD.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteFADDLd], (instregex "VADD(S|P)(S|D)(Z|Z128|Z256)rm(bk|k)$",
                                            "VSUB(S|P)(S|D)(Z|Z128|Z256)rm(bk|k)$")>;

// FADD rrb rrbk Int
def C4GM8WriteFADD_Int:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteFADD_Int], (instregex "VADD(SD|SS)Zrr(bk|k)_Int",
                                              "VSUB(SD|SS)Zrr(bk|k)_Int")>;

def C4GM8WriteFADDLd_Int:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteFADD_Int.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteFADDLd_Int], (instregex "VADD(SD|SS)Zrm(bk|k)_Int",
                                                "VSUB(SD|SS)Zrm(bk|k)_Int")>;

// RCP14
def C4GM8WriteVRCP14:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14], (instrs VRCP14SSZrr, VRCP14SSZrrk, VRCP14SSZrrkz,
                                         VRCP14SDZrr, VRCP14SDZrrk, VRCP14SDZrrkz)>;

def C4GM8WriteVRCP14Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRCP14.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14Ld], (instrs VRCP14SDZrm, VRCP14SDZrmk, VRCP14SDZrmkz,
                                           VRCP14SSZrm, VRCP14SSZrmk, VRCP14SSZrmkz)>;

def C4GM8WriteVRCP14X:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14X], (instrs VRCP14PSZ128r, VRCP14PSZ128rk, VRCP14PSZ128rkz,
                                          VRCP14PDZ128r, VRCP14PDZ128rk, VRCP14PDZ128rkz)>;

def C4GM8WriteVRCP14XLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRCP14X.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14XLd], (instrs VRCP14PDZ128m, VRCP14PDZ128mb, VRCP14PDZ128mbk,
                                            VRCP14PDZ128mbkz, VRCP14PDZ128mk, VRCP14PDZ128mkz,
                                            VRCP14PSZ128m, VRCP14PSZ128mb, VRCP14PSZ128mbk,
                                            VRCP14PSZ128mbkz, VRCP14PSZ128mk, VRCP14PSZ128mkz)>;

def C4GM8WriteVRCP14Y:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14Y], (instrs VRCP14PSZ256r, VRCP14PSZ256rk, VRCP14PSZ256rkz,
                                          VRCP14PDZ256r, VRCP14PDZ256rk, VRCP14PDZ256rkz)>;

def C4GM8WriteVRCP14YLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRCP14Y.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14YLd], (instrs VRCP14PDZ256m, VRCP14PDZ256mb, VRCP14PDZ256mbk,
                                            VRCP14PDZ256mbkz, VRCP14PDZ256mk, VRCP14PDZ256mkz,
                                            VRCP14PSZ256m, VRCP14PSZ256mb, VRCP14PSZ256mbk,
                                            VRCP14PSZ256mbkz, VRCP14PSZ256mk, VRCP14PSZ256mkz)>;

def C4GM8WriteVRCP14Z:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14Z], (instrs VRCP14PDZr, VRCP14PDZrk, VRCP14PDZrkz,
                                          VRCP14PSZr, VRCP14PSZrk, VRCP14PSZrkz)>;

def C4GM8WriteVRCP14ZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRCP14Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRCP14ZLd], (instrs VRCP14PDZm, VRCP14PDZmb, VRCP14PDZmbk,
                                            VRCP14PDZmbkz, VRCP14PDZmk, VRCP14PDZmkz,
                                            VRCP14PSZm, VRCP14PSZmb, VRCP14PSZmbk,
                                            VRCP14PSZmbkz, VRCP14PSZmk, VRCP14PSZmkz)>;

// RSQRT14
def C4GM8WriteRSQRT14:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14], (instrs VRSQRT14SDZrr, VRSQRT14SDZrrk, VRSQRT14SDZrrkz,
                                          VRSQRT14SSZrr, VRSQRT14SSZrrk, VRSQRT14SSZrrkz)>;

def C4GM8WriteRSQRT14Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRSQRT14.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14Ld], (instrs VRSQRT14SDZrm, VRSQRT14SDZrmk, VRSQRT14SDZrmkz,
                                            VRSQRT14SSZrm, VRSQRT14SSZrmk, VRSQRT14SSZrmkz)>;

def C4GM8WriteRSQRT14X:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14X], (instrs VRSQRT14PDZ128r, VRSQRT14PDZ128rk, VRSQRT14PDZ128rkz,
                                           VRSQRT14PSZ128r, VRSQRT14PSZ128rk, VRSQRT14PSZ128rkz)>;

def C4GM8WriteRSQRT14XLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRSQRT14X.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14XLd], (instrs VRSQRT14PDZ128m, VRSQRT14PDZ128mb, VRSQRT14PDZ128mbk,
                                             VRSQRT14PDZ128mbkz, VRSQRT14PDZ128mk, VRSQRT14PDZ128mkz,
                                             VRSQRT14PSZ128m, VRSQRT14PSZ128mb, VRSQRT14PSZ128mbk,
                                             VRSQRT14PSZ128mbkz, VRSQRT14PSZ128mk, VRSQRT14PSZ128mkz)>;

def C4GM8WriteRSQRT14Y:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14Y], (instrs VRSQRT14PDZ256r, VRSQRT14PDZ256rk, VRSQRT14PDZ256rkz,
                                           VRSQRT14PSZ256r, VRSQRT14PSZ256rk, VRSQRT14PSZ256rkz)>;

def C4GM8WriteRSQRT14YLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRSQRT14Y.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14YLd], (instrs VRSQRT14PDZ256m, VRSQRT14PDZ256mb, VRSQRT14PDZ256mbk,
                                             VRSQRT14PDZ256mbkz, VRSQRT14PDZ256mk, VRSQRT14PDZ256mkz,
                                             VRSQRT14PSZ256m, VRSQRT14PSZ256mb, VRSQRT14PSZ256mbk,
                                             VRSQRT14PSZ256mbkz, VRSQRT14PSZ256mk, VRSQRT14PSZ256mkz)>;

def C4GM8WriteRSQRT14Z:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14Z], (instrs VRSQRT14PDZr, VRSQRT14PDZrk, VRSQRT14PDZrkz,
                                           VRSQRT14PSZr, VRSQRT14PSZrk, VRSQRT14PSZrkz)>;

def C4GM8WriteRSQRT14ZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteRSQRT14Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteRSQRT14ZLd], (instrs VRSQRT14PDZm, VRSQRT14PDZmb, VRSQRT14PDZmbk,
                                             VRSQRT14PDZmbkz, VRSQRT14PDZmk, VRSQRT14PDZmkz,
                                             VRSQRT14PSZm, VRSQRT14PSZmb, VRSQRT14PSZmbk,
                                             VRSQRT14PSZmbkz, VRSQRT14PSZmk, VRSQRT14PSZmkz)>;

defm : C4GM8WriteResFPPair<WriteFCmp, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmpX, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmpY, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmpZ, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmp64, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmp64X, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmp64Y, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCmp64Z, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFCom, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteFComX, [C4GM8FPU], 1, [2], 2>;
defm : C4GM8WriteResFPPair<WriteFBlend, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFBlendY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFBlendZ, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFVarBlend, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFVarBlendY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFVarBlendZ, [C4GM8FPU], 1>;

def C4GM8WriteFCOMI_FCOMIP:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteFCOMI_FCOMIP], (instrs COM_FIPr, COM_FIr)>;

// VMAX VMIN
def C4GM8WriteVMAXXY_VMINXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVMAXXY_VMINXY], (instrs VMAXPSZ128rrk, VMAXPDZ128rrk,
                                                VMINPSZ128rrk, VMINPDZ128rrk,
                                                VMAXCPSZ128rrk, VMAXCPDZ128rrk,
                                                VMINCPSZ128rrk, VMINCPDZ128rrk,
                                                VMAXPSZ256rrk, VMAXPDZ256rrk,
                                                VMINPSZ256rrk, VMINPDZ256rrk,
                                                VMAXCPSZ256rrk, VMAXCPDZ256rrk,
                                                VMINCPSZ256rrk, VMINCPDZ256rrk)>;

def C4GM8WriteVMAXXY_VMINXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVMAXXY_VMINXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVMAXXY_VMINXYLd], (instrs VMAXPSZ128rmk, VMAXPDZ128rmk,
                                                  VMINPSZ128rmk, VMINPDZ128rmk,
                                                  VMAXCPSZ128rmk, VMAXCPDZ128rmk,
                                                  VMINCPSZ128rmk, VMINCPDZ128rmk,
                                                  VMAXPSZ256rmk, VMAXPDZ256rmk,
                                                  VMINPSZ256rmk, VMINPDZ256rmk,
                                                  VMAXCPSZ256rmk, VMAXCPDZ256rmk,
                                                  VMINCPSZ256rmk, VMINCPDZ256rmk)>;

def C4GM8WriteVMAXZ_VMINZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVMAXZ_VMINZ], (instrs VMAXPSZrrk, VMAXPSZrrbk,
                                              VMAXPDZrrk, VMAXPDZrrbk,
                                              VMINPSZrrk, VMINPSZrrbk,
                                              VMINPDZrrk, VMINPDZrrbk,
                                              VMAXCPSZrrk, VMAXCPDZrrk,
                                              VMINCPSZrrk, VMINCPDZrrk,
                                              VMAXSSZrrk_Int, VMAXSSZrrbk_Int,
                                              VMAXSDZrrk_Int, VMAXSDZrrbk_Int,
                                              VMINSSZrrk_Int, VMINSSZrrbk_Int,
                                              VMINSDZrrk_Int, VMINSDZrrbk_Int)>;

def C4GM8WriteVMAXZ_VMINZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVMAXZ_VMINZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVMAXZ_VMINZLd], (instrs VMAXPSZrmk, VMAXPSZrmbk,
                                                VMAXPDZrmk, VMAXPDZrmbk,
                                                VMINPSZrmk, VMINPSZrmbk,
                                                VMINPDZrmk, VMINPDZrmbk,
                                                VMAXCPSZrmk, VMAXCPDZrmk,
                                                VMINCPSZrmk, VMINCPDZrmk,
                                                VMAXSSZrmk_Int, VMAXSDZrmk_Int,
                                                VMINSSZrmk_Int, VMINSDZrmk_Int)>;

// VBLEND
def C4GM8WriteVBLENDrrk:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBLENDrrk], (instregex "VPBLENDM(B|D|W|Q)Z(128|256)?rrk$",
                                               "VBLENDM(PS|PD)Zrrk$")>;

def C4GM8WriteVBLENDrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBLENDrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBLENDrmk], (instregex "VPBLENDM(B|D|W|Q)Z(128|256)?rmk$",
                                               "VBLENDM(PS|PD)Zrmk$")>;

defm : C4GM8WriteResFPPair<WriteCvtSS2I, [C4GM8FPU13], 2, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtPS2I, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPS2IY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPS2IZ, [C4GM8FPU13], 4, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtSD2I, [C4GM8FPU13], 2, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtPD2I, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPD2IY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPD2IZ, [C4GM8FPU13], 8, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtSS2SD, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPS2PD, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPS2PDY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPS2PDZ, [C4GM8FPU13], 5, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtSD2SS, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPD2PS, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPD2PSY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPD2PSZ, [C4GM8FPU13], 8, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtI2SS, [C4GM8FPU13], 4, [3], 2>;
defm : C4GM8WriteResFPPair<WriteCvtI2PS, [C4GM8FPU13], 4>;
defm : C4GM8WriteResFPPair<WriteCvtI2PSY, [C4GM8FPU13], 4>;
defm : C4GM8WriteResFPPair<WriteCvtI2PSZ, [C4GM8FPU13], 4, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtI2SD, [C4GM8FPU13], 4, [3], 2>;
defm : C4GM8WriteResFPPair<WriteCvtI2PD, [C4GM8FPU13], 4>;
defm : C4GM8WriteResFPPair<WriteCvtI2PDY, [C4GM8FPU13], 4>;
defm : C4GM8WriteResFPPair<WriteCvtI2PDZ, [C4GM8FPU13], 5, [2], 2>;
defm : C4GM8WriteResFPPair<WriteCvtPH2PS, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPH2PSY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteCvtPH2PSZ, [C4GM8FPU13], 5, [2], 2>;
defm : C4GM8WriteRes<WriteCvtPS2PH, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteRes<WriteCvtPS2PHY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteRes<WriteCvtPS2PHZ, [C4GM8FPU13], 8, [2], 2>;
defm : C4GM8WriteRes<WriteCvtPS2PHSt, [C4GM8FPU13, C4GM8FPSt, C4GM8Store], !add(4, C864GM8Model.StoreLatency), [1, 1, 1], 2>;
defm : C4GM8WriteRes<WriteCvtPS2PHYSt, [C4GM8FPU13, C4GM8FPSt, C4GM8Store], !add(4, C864GM8Model.StoreLatency), [1, 1, 1], 2>;
defm : C4GM8WriteRes<WriteCvtPS2PHZSt, [C4GM8FPU13, C4GM8FPSt, C4GM8Store], !add(8, C864GM8Model.StoreLatency), [2, 1, 1], 3>;

// CVTPD2QQ
def C4GM8WritePD2QQ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WritePD2QQ], (instregex "VCVT(T?)PD2(U?)QQ(Z|Z128|Z256)(rr$|rrkz)")>;

def C4GM8WritePD2QQLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePD2QQ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WritePD2QQLd], (instregex "VCVT(T?)PD2(U?)QQ(Z|Z128|Z256)(rm$|rmkz)")>;

// CVTSI2SS
def C4GM8WriteSI2SSrr:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSI2SSrr], (instregex "VCVT(U?)SI2SS(Z?)rr")>;

def C4GM8WriteSI2SSrm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSI2SSrr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSI2SSrm], (instregex "VCVT(U?)SI2SS(Z?)rm")>;

// SI2SD
def C4GM8WriteSI2SD:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSI2SD], (instregex "VCVT(U?)SI2SD(Z?)rr")>;

def C4GM8WriteSI2SDLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSI2SD.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSI2SDLd], (instregex "VCVT(U?)SI2SD(Z?)rm")>;

// CVT BF16
def C4GM8WriteCVTBF16:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16], (instrs VCVTNE2PS2BF16Z256rr, VCVTNE2PS2BF16Z256rrkz,
                                          VCVTNE2PS2BF16Zrrkz)>;

def C4GM8WriteCVTBF16Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteCVTBF16.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16Ld], (instrs VCVTNE2PS2BF16Z256rm, VCVTNE2PS2BF16Z256rmkz,
                                            VCVTNE2PS2BF16Zrmkz)>;

// CVT BF16
def C4GM8WriteCVTBF16rrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16rrk], (instrs VCVTNE2PS2BF16Z128rrk, VCVTNEPS2BF16Zrrk,
                                             VCVTNE2PS2BF16Zrrk)>;

def C4GM8WriteCVTBF16rmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteCVTBF16rrk.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16rmk], (instrs VCVTNE2PS2BF16Z128rmk, VCVTNEPS2BF16Zrmk)>;

// CVT BF16
def C4GM8WriteCVTBF16rr:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16rr], (instrs VCVTNE2PS2BF16Zrr)>;

def C4GM8WriteCVTBF16rm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteCVTBF16rr.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 7;
}
def : InstRW<[C4GM8WriteCVTBF16rm], (instrs VCVTNE2PS2BF16Zrm)>;

// VDPBF16
def C4GM8WriteVDPBF16:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 8;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVDPBF16], (instrs VDPBF16PSZr, VDPBF16PSZrkz,
                                          VDPBF16PSZ128r, VDPBF16PSZ128rkz,
                                          VDPBF16PSZ256r, VDPBF16PSZ256rkz)>;

def C4GM8WriteVDPBF16Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVDPBF16.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVDPBF16Ld], (instrs VDPBF16PSZm, VDPBF16PSZmkz,
                                            VDPBF16PSZ128m, VDPBF16PSZ128mkz,
                                            VDPBF16PSZ256m, VDPBF16PSZ256mkz)>;

// VDPBF16
def C4GM8WriteVDPBF16rk:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVDPBF16rk], (instrs VDPBF16PSZrk, VDPBF16PSZ128rk, VDPBF16PSZ256rk)>;

def C4GM8WriteVDPBF16mk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVDPBF16rk.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVDPBF16mk], (instrs VDPBF16PSZmk, VDPBF16PSZ128mk, VDPBF16PSZ256mk)>;

// VCVT rrk
def C4GM8WriteVCVTrrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTrrk], (instrs VCVTPD2PSZ128rrk, VCVTPD2PSZ128rrk, VCVTSD2SSZrrbk_Int,
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

def C4GM8WriteVCVTrrkLD:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCVTrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTrrkLD], (instrs VCVTPD2PSZ128rmk, VCVTPD2PSZ128rmk,
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
def C4GM8WriteVCVTrr:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTrr], (instrs VCVTPD2QQZrrb, VCVTPD2QQZrrbkz, VCVTPD2UQQZrrb,
                                         VCVTPD2UQQZrrbkz, VCVTTPD2QQZrrb, VCVTTPD2QQZrrbkz,
                                         VCVTTPD2UQQZrrb, VCVTTPD2UQQZrrbkz, VCVTQQ2PDZrr,
                                         VCVTQQ2PDZrrb, VCVTQQ2PDZrrbkz, VCVTQQ2PDZrrkz,
                                         VCVTUQQ2PDZrr, VCVTUQQ2PDZrrb, VCVTUQQ2PDZrrbkz,
                                         VCVTUQQ2PDZrrkz)>;

def C4GM8WriteVCVTrm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCVTrr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTrm], (instrs VCVTPD2QQZrmb, VCVTPD2QQZrmbkz, VCVTPD2UQQZrmb,
                                         VCVTPD2UQQZrmbkz, VCVTTPD2QQZrmb, VCVTTPD2QQZrmbkz,
                                         VCVTTPD2UQQZrmb, VCVTTPD2UQQZrmbkz, VCVTQQ2PDZrm,
                                         VCVTQQ2PDZrmb, VCVTQQ2PDZrmbkz, VCVTQQ2PDZrmkz,
                                         VCVTUQQ2PDZrm, VCVTUQQ2PDZrmb, VCVTUQQ2PDZrmbkz,
                                         VCVTUQQ2PDZrmkz)>;

// VCVTQQ2PS
def C4GM8WriteVCVTQQ2PS:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTQQ2PS], (instrs VCVTQQ2PSZrr, VCVTQQ2PSZrrb, VCVTQQ2PSZrrbkz, VCVTQQ2PSZrrkz,
                                            VCVTUQQ2PSZrr, VCVTUQQ2PSZrrb, VCVTUQQ2PSZrrbkz, VCVTUQQ2PSZrrkz)>;

def C4GM8WriteVCVTQQ2PSLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCVTQQ2PS.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCVTQQ2PSLd], (instrs VCVTQQ2PSZrm, VCVTQQ2PSZrmb, VCVTQQ2PSZrmbkz, VCVTQQ2PSZrmkz,
                                              VCVTUQQ2PSZrm, VCVTUQQ2PSZrmb, VCVTUQQ2PSZrmbkz, VCVTUQQ2PSZrmkz)>;


// VCVTNE2PS2BF16Z128
def C4GM8WriteVCVTNE2PS2BF16Z128:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 9;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVCVTNE2PS2BF16Z128], (instrs VCVTNE2PS2BF16Z128rr,
                                                     VCVTNE2PS2BF16Z128rrkz)>;

def C4GM8WriteVCVTNE2PS2BF16Z128Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCVTNE2PS2BF16Z128.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVCVTNE2PS2BF16Z128Ld], (instrs VCVTNE2PS2BF16Z128rm,
                                                       VCVTNE2PS2BF16Z128rmkz)>;

// VCVTNE2PS2BF16Z VCVTNE2PS2BF16Z256 VCVTNEPS2BF16
def C4GM8WriteVCVTNEPS2BF16:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVCVTNEPS2BF16], (instrs VCVTNE2PS2BF16Z256rrk,
                                                VCVTNEPS2BF16Z128rrk, VCVTNEPS2BF16Z256rrk)>;

def C4GM8WriteVCVTNEPS2BF16Ld:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCVTNEPS2BF16.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVCVTNEPS2BF16Ld], (instrs VCVTNE2PS2BF16Z256rmk,
                                                  VCVTNEPS2BF16Z128rmk, VCVTNEPS2BF16Z256rmk)>;


defm : C4GM8WriteResFPPair<WriteFSign, [C4GM8FPU1], 1>;
defm : C4GM8WriteResFPPair<WriteFRnd, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFRndY, [C4GM8FPU13], 4, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFRndZ, [C4GM8FPU13], 4, [2], 2>;
defm : C4GM8WriteResFPPair<WriteFLogic, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFLogicY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFLogicZ, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteFTest, [C4GM8FPU13], 1, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFTestY, [C4GM8FPU13], 1, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFTestZ, [C4GM8FPU13], 1, [2], 2>;
defm : C4GM8WriteResFPPair<WriteFShuffle, [C4GM8FPU13], 1, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFShuffleY, [C4GM8FPU13], 1, [1], 1>;
defm : C4GM8WriteResFPPair<WriteFShuffleZ, [C4GM8FPU13], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteFVarShuffle, [C4GM8FPU02], 3, [4], 1>;
defm : C4GM8WriteResFPPair<WriteFVarShuffleY,[C4GM8FPU02], 3, [4], 1>;
defm : C4GM8WriteResFPPair<WriteFVarShuffleZ,[C4GM8FPU02], 5, [8], 1>;
defm : C4GM8WriteResFPPair<WriteFShuffle256, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteFVarShuffle256, [C4GM8FPU02], 3>;

defm : C4GM8WriteRes<WriteVecLoad, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteVecLoadX, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteVecLoadY, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteVecLoadNT, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteVecLoadNTY, [C4GM8AGU, C4GM8Load], !add(C864GM8Model.VecLoadLatency, 1), [1, 1], 1>;
defm : C4GM8WriteRes<WriteVecMaskedLoad, [C4GM8AGU, C4GM8Load, C4GM8FPLd01], !add(C864GM8Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : C4GM8WriteRes<WriteVecMaskedLoadY, [C4GM8AGU, C4GM8Load, C4GM8FPLd01], !add(C864GM8Model.VecLoadLatency, 1), [1, 1, 2], 1>;
defm : C4GM8WriteRes<WriteVecStore, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteVecStoreX, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteVecStoreY, [C4GM8AGU, C4GM8Store], C864GM8Model.StoreLatency>;
defm : C4GM8WriteRes<WriteVecStoreNT, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecStoreNTY, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecMaskedStore32, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecMaskedStore32Y, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecMaskedStore64, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecMaskedStore64Y, [C4GM8FPSt, C4GM8AGU, C4GM8Store], !add(1, C864GM8Model.StoreLatency)>;
defm : C4GM8WriteRes<WriteVecMoveToGpr, [C4GM8FPLd01], 1, [2], 1>;
defm : C4GM8WriteRes<WriteVecMoveFromGpr, [C4GM8FPLd01], 1, [4], 2>;
defm : C4GM8WriteRes<WriteEMMS, [], 0, [], 1>;

defm : C4GM8WriteResFPPair<WriteVecShift, [C4GM8FPU02], 1>; // from llvm-exegesis
defm : C4GM8WriteResFPPair<WriteVecShiftX, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVecShiftY, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVecShiftZ, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVecShiftImm, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteVecShiftImmX, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteVecShiftImmY, [C4GM8FPU02], 1>;
defm : C4GM8WriteResFPPair<WriteVecShiftImmZ, [C4GM8FPU02], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVarVecShift, [C4GM8FPU02], 1, [4], 1>;
defm : C4GM8WriteResFPPair<WriteVarVecShiftY, [C4GM8FPU02], 1, [4], 1>;
defm : C4GM8WriteResFPPair<WriteVarVecShiftZ, [C4GM8FPU02], 1, [8], 1>;
defm : C4GM8WriteResFPPair<WriteVecLogic, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecLogicX, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecLogicY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecLogicZ, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecTest, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteVecTestY, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteVecTestZ, [C4GM8FPU13], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVecALU, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecALUX, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecALUY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVecALUZ, [C4GM8FPU], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVecIMul, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVecIMulX, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVecIMulY, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVecIMulZ, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteRes<WriteCLMul, [C4GM8FPU02], 3>;
defm : C4GM8WriteRes<WriteCLMulLd, [C4GM8AGU, C4GM8Load, C4GM8FPU02], !add(C864GM8Model.VecLoadLatency, 3)>;
defm : C4GM8WriteResFPPair<WritePMULLD, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WritePMULLDY, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WritePMULLDZ, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WriteShuffle, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteShuffleX, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteShuffleY, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteShuffleZ, [C4GM8FPU13], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVarShuffle, [C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteVarShuffleX,[C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteVarShuffleY,[C4GM8FPU13], 1>;
defm : C4GM8WriteResFPPair<WriteVarShuffleZ,[C4GM8FPU13], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteBlend, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteBlendY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteBlendZ, [C4GM8FPU], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteVarBlend, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVarBlendY, [C4GM8FPU], 1>;
defm : C4GM8WriteResFPPair<WriteVarBlendZ, [C4GM8FPU], 1, [2], 1>;
defm : C4GM8WriteResFPPair<WriteShuffle256, [C4GM8FPU01], 3>;
defm : C4GM8WriteResFPPair<WriteVPMOV256, [C4GM8FPU02], 3>;
defm : C4GM8WriteResFPPair<WriteVarShuffle256, [C4GM8FPU3, C4GM8FPU02], 2, [2, 2], 2>;
defm : C4GM8WriteResFPPair<WritePSADBW, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WritePSADBWX, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WritePSADBWY, [C4GM8FPU02], 3, [2], 1>;
defm : C4GM8WriteResFPPair<WritePSADBWZ, [C4GM8FPU02], 3, [4], 1>;
defm : C4GM8WriteResFPPair<WriteMPSAD, [C4GM8FPU], 4, [8], 1>;
defm : C4GM8WriteResFPPair<WriteMPSADY, [C4GM8FPU], 4, [8], 1>;
defm : C4GM8WriteResFPPair<WriteMPSADZ, [C4GM8FPU], 4, [16], 1>;
defm : C4GM8WriteResFPPair<WritePHMINPOS, [C4GM8FPU02], 3, [2], 1>;

// EXTRQ INSERTQ
def C4GM8WriteEXTRQ_INSERTQ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXTRQ_INSERTQ], (instrs EXTRQ, EXTRQI, INSERTQ, INSERTQI)>;

// VPCMP
def C4GM8WriteVPCMPXY:SchedWriteRes<[C4GM8FPU, C4GM8FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [2, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPCMPXY], (instregex "VPCMPEQ(B|D|Q|W)(Z128|Z256)rrk$",
                                             "VPCMPGT(B|D|Q|W)(Z128|Z256)rrk$",
                                             "VPCMP(U)?(B|D|Q|W)(Z128|Z256)rrik$")>;

def C4GM8WriteVPCMPXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCMPXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPCMPXYLd], (instregex "VPCMPEQ(B|D|Q|W)(Z128|Z256)rmk$",
                                               "VPCMPGT(B|D|Q|W)(Z128|Z256)rmk$",
                                               "VPCMP(U)?(B|D|Q|W)(Z128|Z256)rmik$")>;

def C4GM8WriteVPCMPZ:SchedWriteRes<[C4GM8FPU, C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [12, 6];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCMPZ], (instregex "VPCMPEQ(B|D|Q|W)Zrrk$",
                                            "VPCMPGT(B|D|Q|W)Zrrk$",
                                            "VPCMP(U)?(B|D|Q|W)Zrrik$")>;

def C4GM8WriteVPCMPZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCMPZ.Latency);
  let ReleaseAtCycles = [1, 1, 12, 6];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCMPZLd], (instregex "VPCMPEQ(B|D|Q|W)Zrmk$",
                                              "VPCMPGT(B|D|Q|W)Zrmk$",
                                              "VPCMP(U)?(B|D|Q|W)Zrmik$")>;

// VPSHUFBIT
def C4GM8WriteVPSHUFBIT:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 8;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPSHUFBIT], (instrs VPSHUFBITQMBZ128rrk, VPSHUFBITQMBZ256rrk)>;

def C4GM8WriteVPSHUFBITLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPSHUFBIT.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPSHUFBITLd], (instrs VPSHUFBITQMBZ128rmk, VPSHUFBITQMBZ256rmk)>;

def C4GM8WriteVPSHUFBITZ:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 10;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPSHUFBITZ], (instrs VPSHUFBITQMBZrrk)>;

def C4GM8WriteVPSHUFBITZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPSHUFBITZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM8WriteVPSHUFBITZLd], (instrs VPSHUFBITQMBZrmk)>;

// VSHUF VPSHUFB VPSHUFLW VPSHUFHW
def C4GM8WriteVPSHUFXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPSHUFXY], (instrs VSHUFPSZ128rrik, VSHUFPDZ128rrik,
                                           VSHUFPSZ256rrik, VSHUFPDZ256rrik,
                                           VPSHUFBZ128rrk, VPSHUFBZ128rrkz,
                                           VPSHUFHWZ128rik, VPSHUFHWZ128rikz,
                                           VPSHUFLWZ128rik, VPSHUFLWZ128rikz,
                                           VPSHUFBZ256rrk, VPSHUFBZ256rrkz,
                                           VPSHUFHWZ256rik, VPSHUFHWZ256rikz,
                                           VPSHUFLWZ256rik, VPSHUFLWZ256rikz)>;

def C4GM8WriteVPSHUFXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPSHUFXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPSHUFXYLd], (instrs VSHUFPSZ128rmik, VSHUFPDZ128rmik,
                                             VSHUFPSZ256rmik, VSHUFPDZ256rmik,
                                             VPSHUFBZ128rmk, VPSHUFBZ128rmkz,
                                             VPSHUFHWZ128mik, VPSHUFHWZ128mikz,
                                             VPSHUFLWZ128mik, VPSHUFLWZ128mikz,
                                             VPSHUFBZ256rmk, VPSHUFBZ256rmkz,
                                             VPSHUFHWZ256mik, VPSHUFHWZ256mikz,
                                             VPSHUFLWZ256mik, VPSHUFLWZ256mikz)>;

def C4GM8WriteVPSHUFZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPSHUFZ], (instrs VSHUFPSZrrik, VSHUFPDZrrik,
                                          VPSHUFBZrrk, VPSHUFBZrrkz,
                                          VPSHUFHWZrik, VPSHUFHWZrikz,
                                          VPSHUFLWZrik, VPSHUFLWZrikz)>;

def C4GM8WriteVPSHUFZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPSHUFZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPSHUFZLd], (instrs VSHUFPSZrmik, VSHUFPDZrmik,
                                            VPSHUFBZrmk, VPSHUFBZrmkz,
                                            VPSHUFHWZmik, VPSHUFHWZmikz,
                                            VPSHUFLWZmik, VPSHUFLWZmikz)>;

// VSHUFF VSHUFI
def C4GM8WriteVSHUFFY_VSHUFIY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSHUFFY_VSHUFIY], (instrs VSHUFF32X4Z256rrik, VSHUFF64X2Z256rrik,
                                                  VSHUFI32X4Z256rrik, VSHUFI64X2Z256rrik)>;

def C4GM8WriteVSHUFFY_VSHUFIYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSHUFFY_VSHUFIY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSHUFFY_VSHUFIYLd], (instrs VSHUFF32X4Z256rmik, VSHUFF64X2Z256rmik,
                                                    VSHUFI32X4Z256rmik, VSHUFI64X2Z256rmik)>;

def C4GM8WriteVSHUFFZ_VSHUFIZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVSHUFFZ_VSHUFIZ], (instrs VSHUFF32X4Zrri, VSHUFF32X4Zrrik, VSHUFF32X4Zrrikz,
                                                  VSHUFF64X2Zrri, VSHUFF64X2Zrrik, VSHUFF64X2Zrrikz,
                                                  VSHUFI32X4Zrri, VSHUFI32X4Zrrik, VSHUFI32X4Zrrikz,
                                                  VSHUFI64X2Zrri, VSHUFI64X2Zrrik, VSHUFI64X2Zrrikz)>;

def C4GM8WriteVSHUFFZ_VSHUFIZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSHUFFZ_VSHUFIZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVSHUFFZ_VSHUFIZLd], (instrs VSHUFF32X4Zrmi, VSHUFF32X4Zrmik, VSHUFF32X4Zrmikz,
                                                    VSHUFF64X2Zrmi, VSHUFF64X2Zrmik, VSHUFF64X2Zrmikz,
                                                    VSHUFI32X4Zrmi, VSHUFI32X4Zrmik, VSHUFI32X4Zrmikz,
                                                    VSHUFI64X2Zrmi, VSHUFI64X2Zrmik, VSHUFI64X2Zrmikz)>;

// UNPCK
def C4GM8WriteUNPCKXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteUNPCKXY], (instrs VUNPCKLPSZ128rrk, VUNPCKLPDZ128rrk,
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

def C4GM8WriteUNPCKXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteUNPCKXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteUNPCKXYLd], (instrs VUNPCKLPSZ128rmk, VUNPCKLPDZ128rmk,
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


def C4GM8WriteUNPCKZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteUNPCKZ], (instrs VUNPCKLPSZrrk, VUNPCKLPDZrrk,
                                         VUNPCKHPSZrrk, VUNPCKHPDZrrk,
                                         VPUNPCKHBWZrrk, VPUNPCKHBWZrrkz,
                                         VPUNPCKHWDZrrk, VPUNPCKHWDZrrkz,
                                         VPUNPCKLBWZrrk, VPUNPCKLBWZrrkz,
                                         VPUNPCKLWDZrrk, VPUNPCKLWDZrrkz)>;

def C4GM8WriteUNPCKZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteUNPCKZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteUNPCKZLd], (instrs VUNPCKLPSZrmk, VUNPCKLPDZrmk,
                                           VUNPCKHPSZrmk, VUNPCKHPDZrmk,
                                           VPUNPCKHBWZrmk, VPUNPCKHBWZrmkz,
                                           VPUNPCKHWDZrmk, VPUNPCKHWDZrmkz,
                                           VPUNPCKLBWZrmk, VPUNPCKLBWZrmkz,
                                           VPUNPCKLWDZrmk, VPUNPCKLWDZrmkz)>;

// POPCNT
def C4GM8WritePOPCNTZ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePOPCNTZ], (instrs VPOPCNTBZrrk, VPOPCNTBZrrkz,
                                          VPOPCNTWZrrk, VPOPCNTWZrrkz,
                                          VPOPCNTBZ128rrk, VPOPCNTBZ128rrkz,
                                          VPOPCNTWZ128rrk, VPOPCNTWZ128rrkz,
                                          VPOPCNTBZ256rrk, VPOPCNTBZ256rrkz,
                                          VPOPCNTWZ256rrk, VPOPCNTWZ256rrkz)>;

def C4GM8WritePOPCNTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePOPCNTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePOPCNTZLd], (instrs VPOPCNTBZrmk, VPOPCNTBZrmkz,
                                            VPOPCNTWZrmk, VPOPCNTWZrmkz,
                                            VPOPCNTBZ128rmk, VPOPCNTBZ128rmkz,
                                            VPOPCNTWZ128rmk, VPOPCNTWZ128rmkz,
                                            VPOPCNTBZ256rmk, VPOPCNTBZ256rmkz,
                                            VPOPCNTWZ256rmk, VPOPCNTWZ256rmkz)>;

// VPABS VPADDS VPADDUS VPAVG VPMAX VPMIN VPADD VPSUB VPTERNLOG
def C4GM8WriteVecALUZ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecALUZ], (instregex "VPABS(B|W)Z(128|256)?rrk$",
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

def C4GM8WriteVecALUZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVecALUZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecALUZLd], (instregex "VPABS(B|W)Z(128|256)?rmk$",
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
def C4GM8WriteVecShiftZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShiftZ], (instrs VPSHLDDZ128rrik, VPSHLDDZ128rrikz, VPSHLDDZ256rrik,
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

def C4GM8WriteVecShiftZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVecShiftZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShiftZLd], (instrs VPSHLDDZ128rmik, VPSHLDDZ128rmikz, VPSHLDDZ256rmik,
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

def C4GM8WriteVecShiftZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShiftZrrk], (instrs VPSLLDZrrk, VPSLLQZrrk, VPSRADZrrk,
                                               VPSRAQZrrk, VPSRLDZrrk, VPSRLQZrrk,
                                               VPSLLWZrrk, VPSRAWZrrk, VPSRLWZrrk,
                                               VPSLLDZ256rrk, VPSLLQZ256rrk, VPSRADZ256rrk,
                                               VPSRAQZ256rrk, VPSRLDZ256rrk, VPSRLQZ256rrk,
                                               VPSLLWZ256rrk, VPSRAWZ256rrk, VPSRLWZ256rrk)>;

def C4GM8WriteVecShiftZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVecShiftZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShiftZrmk], (instrs VPSLLDZrmk, VPSLLQZrmk, VPSRADZrmk,
                                               VPSRAQZrmk, VPSRLDZrmk, VPSRLQZrmk,
                                               VPSLLWZrmk, VPSRAWZrmk, VPSRLWZrmk,
                                               VPSLLDZ256rmk, VPSLLQZ256rmk, VPSRADZ256rmk,
                                               VPSRAQZ256rmk, VPSRLDZ256rmk, VPSRLQZ256rmk,
                                               VPSLLWZ256rmk, VPSRAWZ256rmk, VPSRLWZ256rmk)>;

def C4GM8WriteVecShift:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShift], (instrs PSLLDrr, PSLLQrr, PSLLWrr, PSRADrr,
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

def C4GM8WriteVecShiftLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVecShift.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVecShiftLd], (instrs PSLLDrm, PSLLQrm, PSLLWrm, PSRADrm,
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
def C4GM8WriteVCMPXY:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCMPXY], (instrs VCMPPDZ128rrik, VCMPPSZ128rrik,
                                         VCMPPDZ256rrik, VCMPPSZ256rrik)>;

def C4GM8WriteVCMPXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCMPXY.Latency);
  let ReleaseAtCycles = [1, 1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVCMPXYLd], (instrs VCMPPDZ128rmik, VCMPPSZ128rmik,
                                          VCMPPDZ256rmik, VCMPPSZ256rmik)>;

def C4GM8WriteVCMPZ:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [10, 10];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVCMPZ], (instrs VCMPPSZrribk, VCMPPSZrrik,
                                        VCMPPDZrribk, VCMPPDZrrik,
                                        VCMPSSZrrik_Int, VCMPSSZrribk_Int,
                                        VCMPSDZrribk_Int, VCMPSDZrrik_Int)>;

def C4GM8WriteVCMPZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVCMPZ.Latency);
  let ReleaseAtCycles = [1, 1, 10, 10];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVCMPZLd], (instrs VCMPPSZrmik, VCMPPDZrmik,
                                          VCMPSSZrmik_Int, VCMPSDZrmik_Int)>;

// VFPCLASS
def C4GM8WriteVFPCLASSXY:SchedWriteRes<[C4GM8FPU13, C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFPCLASSXY], (instrs VFPCLASSPDZ128rik, VFPCLASSPSZ128rik,
                                             VFPCLASSPDZ256rik, VFPCLASSPSZ256rik)>;

def C4GM8WriteVFPCLASSXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFPCLASSXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFPCLASSXYLd], (instrs VFPCLASSPDZ128mik, VFPCLASSPSZ128mik,
                                               VFPCLASSPDZ256mik, VFPCLASSPSZ256mik)>;

def C4GM8WriteVFPCLASSZ:SchedWriteRes<[C4GM8FPU13, C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVFPCLASSZ], (instrs VFPCLASSSDZrik, VFPCLASSSSZrik)>;

def C4GM8WriteVFPCLASSZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFPCLASSZ.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVFPCLASSZLd], (instrs VFPCLASSSDZmik, VFPCLASSSSZmik)>;

def C4GM8WriteVFPCLASSZrik:SchedWriteRes<[C4GM8FPU13, C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVFPCLASSZrik], (instrs VFPCLASSPDZrik, VFPCLASSPSZrik)>;

def C4GM8WriteVFPCLASSZrikLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFPCLASSZrik.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVFPCLASSZrikLd], (instrs VFPCLASSPDZmik, VFPCLASSPSZmik)>;

// VRNDSCALE
def C4GM8WriteVRNDSCALEXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRNDSCALEXY], (instrs VRNDSCALEPDZ128rrik, VRNDSCALEPSZ128rrik,
                                              VRNDSCALEPDZ256rrik, VRNDSCALEPSZ256rrik)>;

def C4GM8WriteVRNDSCALEXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRNDSCALEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRNDSCALEXYLd], (instrs VRNDSCALEPDZ128rmik, VRNDSCALEPSZ128rmik,
                                                VRNDSCALEPDZ256rmik, VRNDSCALEPSZ256rmik)>;

def C4GM8WriteVRNDSCALEZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRNDSCALEZ], (instrs VRNDSCALESDZrribk_Int, VRNDSCALESDZrrik_Int,
                                             VRNDSCALESSZrribk_Int, VRNDSCALESSZrrik_Int,
                                             VRNDSCALEPDZrribk, VRNDSCALEPDZrrik,
                                             VRNDSCALEPSZrribk, VRNDSCALEPSZrrik)>;

def C4GM8WriteVRNDSCALEZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRNDSCALEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRNDSCALEZLd], (instrs VRNDSCALESDZrmik_Int, VRNDSCALESSZrmik_Int,
                                               VRNDSCALEPDZrmik, VRNDSCALEPSZrmik)>;

// VREDUCE
def C4GM8WriteVREDUCEXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVREDUCEXY], (instrs VREDUCEPDZ128rrik, VREDUCEPSZ128rrik,
                                            VREDUCEPDZ256rrik, VREDUCEPSZ256rrik)>;

def C4GM8WriteVREDUCEXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVREDUCEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVREDUCEXYLd], (instrs VREDUCEPDZ128rmik, VREDUCEPSZ128rmik,
                                              VREDUCEPDZ256rmik, VREDUCEPSZ256rmik)>;

def C4GM8WriteVREDUCEZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVREDUCEZ], (instrs VREDUCESDZrribk, VREDUCESDZrrik,
                                           VREDUCESSZrribk, VREDUCESSZrrik,
                                           VREDUCEPDZrribk, VREDUCEPDZrrik,
                                           VREDUCEPSZrribk, VREDUCEPSZrrik)>;

def C4GM8WriteVREDUCEZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVREDUCEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVREDUCEZLd], (instrs VREDUCESSZrmik, VREDUCESDZrmik,
                                             VREDUCEPSZrmik, VREDUCEPDZrmik)>;

// VGETEXP VGETMANT
def C4GM8WriteVGETEXPXY_VGETMANTXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVGETEXPXY_VGETMANTXY], (instrs VGETEXPPSZ128rk, VGETEXPPDZ128rk,
                                                       VGETEXPPSZ256rk, VGETEXPPDZ256rk,
                                                       VGETMANTPSZ128rrik, VGETMANTPDZ128rrik,
                                                       VGETMANTPSZ256rrik, VGETMANTPDZ256rrik)>;

def C4GM8WriteVGETEXPXY_VGETMANTXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVGETEXPXY_VGETMANTXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVGETEXPXY_VGETMANTXYLd], (instrs VGETEXPPSZ128mk, VGETEXPPDZ128mk,
                                                         VGETEXPPSZ256mk, VGETEXPPDZ256mk,
                                                         VGETMANTPSZ128rmik, VGETMANTPDZ128rmik,
                                                         VGETMANTPSZ256rmik, VGETMANTPDZ256rmik)>;

def C4GM8WriteVGETEXPZ_VGETMANTZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVGETEXPZ_VGETMANTZ], (instrs VGETEXPSSZrbk, VGETEXPSSZrk,
                                                     VGETEXPSDZrbk, VGETEXPSDZrk,
                                                     VGETEXPPSZrbk, VGETEXPPSZrk,
                                                     VGETEXPPDZrbk, VGETEXPPDZrk,
                                                     VGETMANTSSZrribk, VGETMANTSSZrrik,
                                                     VGETMANTSDZrribk, VGETMANTSDZrrik,
                                                     VGETMANTPSZrribk, VGETMANTPSZrrik,
                                                     VGETMANTPDZrribk, VGETMANTPDZrrik)>;

def C4GM8WriteVGETEXPZ_VGETMANTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVGETEXPZ_VGETMANTZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVGETEXPZ_VGETMANTZLd], (instrs VGETEXPSSZmk, VGETEXPSDZmk,
                                                       VGETEXPPSZmbk, VGETEXPPSZmk,
                                                       VGETEXPPDZmbk, VGETEXPPDZmk,
                                                       VGETMANTSSZrmik, VGETMANTSDZrmik,
                                                       VGETMANTPSZrmik, VGETMANTPDZrmik)>;

// VSCALEF
def C4GM8WriteVSCALEFXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFXY], (instrs VSCALEFPSZ128rr, VSCALEFPSZ128rrkz,
                                            VSCALEFPSZ256rr, VSCALEFPSZ256rrkz,
                                            VSCALEFPDZ128rr, VSCALEFPDZ128rrkz,
                                            VSCALEFPDZ256rr, VSCALEFPDZ256rrkz)>;

def C4GM8WriteVSCALEFXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSCALEFXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFXYLd], (instrs VSCALEFPSZ128rm, VSCALEFPSZ128rmkz,
                                              VSCALEFPSZ256rm, VSCALEFPSZ256rmkz,
                                              VSCALEFPDZ128rm, VSCALEFPDZ128rmkz,
                                              VSCALEFPDZ256rm, VSCALEFPDZ256rmkz)>;

def C4GM8WriteVSCALEFZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFZ], (instrs VSCALEFSSZrr, VSCALEFSSZrrkz,
                                           VSCALEFSSZrrb_Int, VSCALEFSSZrrbkz_Int,
                                           VSCALEFSDZrr, VSCALEFSDZrrkz,
                                           VSCALEFSDZrrb_Int, VSCALEFSDZrrbkz_Int,
                                           VSCALEFPSZrr, VSCALEFPSZrrkz,
                                           VSCALEFPSZrrb, VSCALEFPSZrrbkz,
                                           VSCALEFPDZrr, VSCALEFPDZrrkz,
                                           VSCALEFPDZrrb, VSCALEFPDZrrbkz)>;

def C4GM8WriteVSCALEFZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSCALEFZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFZLd], (instrs VSCALEFSSZrm, VSCALEFSSZrmkz,
                                             VSCALEFSDZrm,
                                             VSCALEFPSZrm, VSCALEFPSZrmkz,
                                             VSCALEFPSZrmb, VSCALEFPSZrmbkz,
                                             VSCALEFPDZrm, VSCALEFPDZrmkz,
                                             VSCALEFPDZrmb, VSCALEFPDZrmbkz)>;

def C4GM8WriteVSCALEFXYrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFXYrrk], (instrs VSCALEFPSZ128rrk, VSCALEFPDZ128rrk,
                                               VSCALEFPSZ256rrk, VSCALEFPDZ256rrk)>;

def C4GM8WriteVSCALEFXYrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSCALEFXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFXYrmk], (instrs VSCALEFPSZ128rmk, VSCALEFPDZ128rmk,
                                               VSCALEFPSZ256rmk, VSCALEFPDZ256rmk)>;

def C4GM8WriteVSCALEFZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFZrrk], (instrs VSCALEFSSZrrk, VSCALEFSSZrrbk_Int,
                                              VSCALEFSDZrrk, VSCALEFSDZrrbk_Int,
                                              VSCALEFPSZrrk, VSCALEFPSZrrbk,
                                              VSCALEFPDZrrk, VSCALEFPDZrrbk)>;

def C4GM8WriteVSCALEFZrmkLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVSCALEFZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVSCALEFZrmkLd], (instrs VSCALEFSSZrmk, VSCALEFSDZrmk,
                                                VSCALEFPSZrmk, VSCALEFPSZrmbk,
                                                VSCALEFPDZrmk, VSCALEFPDZrmbk)>;

// VFMA
def C4GM8WriteVFMAXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVFMAXY], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)(Z128|Z256)r(bk|k)$")>;

def C4GM8WriteVFMAXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFMAXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVFMAXYLd], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)(Z128|Z256)m(bk|k)$")>;

def C4GM8WriteVFMAZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFMAZ], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)Zr(bk|k)$",
                                           "VF(MADD|MSUB|NMADD|NMSUB)(132|213|231)(SD|SS)Zr(bk|k)_Int")>;

def C4GM8WriteVFMAZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFMAZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFMAZLd], (instregex "VF(MADD|MSUB|MADDSUB|MSUBADD|NMADD|NMSUB)(132|213|231)(PD|PS)Zm(bk|k)$",
                                             "VF(MADD|MSUB|NMADD|NMSUB)(132|213|231)(SD|SS)Zm(bk|k)_Int")>;

// VPROL VPROR
def C4GM8WriteVPROLXY_VPRORXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLXY_VPRORXY], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)(rr|rrkz)$")>;

def C4GM8WriteVPROLXY_VPRORXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPROLXY_VPRORXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLXY_VPRORXYLd], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)(rm|rmkz)$")>;

def C4GM8WriteVPROLXY_VPRORXYrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLXY_VPRORXYrrk], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)rrk$")>;

def C4GM8WriteVPROLXY_VPRORXYrmkLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPROLXY_VPRORXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLXY_VPRORXYrmkLd], (instregex "VPRO(L|R)V(D|Q)(Z128|Z256)rmk$")>;

def C4GM8WriteVPROLZ_VPRORZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLZ_VPRORZ], (instregex "VPRO(L|R)V(D|Q)Z(rr|rrk|rrkz)")>;

def C4GM8WriteVPROLZ_VPRORZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPROLZ_VPRORZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLZ_VPRORZLd], (instregex "VPRO(L|R)V(D|Q)Z(rm|rmk|rmkz)")>;

def C4GM8WriteVPROLImmZ_VPRORImmZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLImmZ_VPRORImmZ], (instregex "VPRO(L|R)(D|Q)Z(ri|rik|rikz)")>;

def C4GM8WriteVPROLImmZ_VPRORImmZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPROLImmZ_VPRORImmZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPROLImmZ_VPRORImmZLd], (instregex "VPRO(L|R)(D|Q)Z(mi|mik|mikz)")>;

// VPTEST
def C4GM8WriteVPTESTXY:SchedWriteRes<[C4GM8FPU, C4GM8FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [3, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPTESTXY], (instregex "VPTEST(N)?M(B|D|Q|W)(Z128|Z256)rrk")>;

def C4GM8WriteVPTESTXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPTESTXY.Latency);
  let ReleaseAtCycles = [1, 1, 3, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVPTESTXYLd], (instregex "VPTEST(N)?M(B|D|Q|W)(Z128|Z256)rmk")>;

def C4GM8WriteVPTESTZ:SchedWriteRes<[C4GM8FPU, C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [8, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPTESTZ], (instregex "VPTEST(N)?M(B|D|Q|W)Zrrk")>;

def C4GM8WriteVPTESTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPTESTZ.Latency);
  let ReleaseAtCycles = [1, 1, 12, 6];
  let NumMicroOps = 3;
}
def : InstRW<[C4GM8WriteVPTESTZLd], (instregex "VPTEST(N)?M(B|D|Q|W)Zrmk")>;

// VPACK
def C4GM8WriteVPACKXY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPACKXY], (instrs VPACKSSDWZ128rrk, VPACKSSWBZ128rrk,
                                          VPACKUSDWZ128rrk, VPACKUSWBZ128rrk,
                                          VPACKSSDWZ256rrk, VPACKSSWBZ256rrk,
                                          VPACKUSDWZ256rrk, VPACKUSWBZ256rrk)>;

def C4GM8WriteVPACKXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPACKXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPACKXYLd], (instrs VPACKSSDWZ128rmk, VPACKSSWBZ128rmk,
                                            VPACKUSDWZ128rmk, VPACKUSWBZ128rmk,
                                            VPACKSSDWZ256rmk, VPACKSSWBZ256rmk,
                                            VPACKUSDWZ256rmk, VPACKUSWBZ256rmk)>;

def C4GM8WriteVPACKZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPACKZ], (instrs VPACKSSDWZrrk, VPACKSSWBZrrk,
                                         VPACKUSDWZrrk, VPACKUSWBZrrk)>;

def C4GM8WriteVPACKZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPACKZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPACKZLd], (instrs VPACKSSDWZrmk, VPACKSSWBZrmk,
                                           VPACKUSDWZrmk, VPACKUSWBZrmk)>;

// VRANGE
def C4GM8WriteVRANGEXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEXY], (instregex "VRANGE(PS|PD)(Z128|Z256)(rri|rrikz)$",
                                              "VRANGE(SS|SD)Z(rri|rrib|rribkz|rrikz)$")>;

def C4GM8WriteVRANGEXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRANGEXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEXYLd], (instregex "VRANGE(PS|PD)(Z128|Z256)(rmi|rmikz)$",
                                                "VRANGE(SS|SD)Z(rmi|rmib|rmibkz|rmikz)$")>;

def C4GM8WriteVRANGEZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEZ], (instregex "VRANGE(PS|PD)Z(rri|rrib|rribkz|rrikz)$")>;

def C4GM8WriteVRANGEZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRANGEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEZLd], (instregex "VRANGE(PS|PD)Z(rmi|rmib|rmibkz|rmikz)$")>;

def C4GM8WriteVRANGEXYrrik:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEXYrrik], (instregex "VRANGE(PS|PD)(Z128|Z256)rrik$",
                                                  "VRANGE(SS|SD)Z(rrik|rribk)$")>;

def C4GM8WriteVRANGEXYrmikLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRANGEXYrrik.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEXYrmikLd], (instregex "VRANGE(PS|PD)(Z128|Z256)rmik$",
                                                    "VRANGE(SS|SD)Z(rmik|rmibk)$")>;

def C4GM8WriteVRANGEZrrik:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEZrrik], (instregex "VRANGE(PS|PD)Z(rrik|rribk)$")>;

def C4GM8WriteVRANGEZrmikLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVRANGEZrrik.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVRANGEZrmikLd], (instregex "VRANGE(PS|PD)Z(rmik|rmibk)$")>;

// VFIXUPIMM
def C4GM8WriteVFIXUPIMMXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVFIXUPIMMXY], (instregex "VFIXUPIMM(PS|PD)(Z128|Z256)rrik$")>;

def C4GM8WriteVFIXUPIMMXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFIXUPIMMXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVFIXUPIMMXYLd], (instregex "VFIXUPIMM(PS|PD)(Z128|Z256)rmik$")>;

def C4GM8WriteVFIXUPIMMZ:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFIXUPIMMZ], (instregex "VFIXUPIMM(SS|SD|PS|PD)Z(rrik|rribk)$",
                                                "VFIXUPIMM(SS|SD)Z(rrik|rribk)$")>;

def C4GM8WriteVFIXUPIMMZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVFIXUPIMMZ.Latency);
  let ReleaseAtCycles = [1, 1, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFIXUPIMMZLd], (instregex "VFIXUPIMM(SS|SD|PS|PD)Z(rmik|rmibk)$",
                                                  "VFIXUPIMM(SS|SD)Z(rmik|rmibk)$")>;

def C4GM8WriteVFIXUPIMMZrribkz:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVFIXUPIMMZrribkz], (instregex "VFIXUPIMM(SS|SD)Z(rrib|rribkz)$")>;

// VCOMPRESS VPCOMPRESS
def C4GM8WriteCOMPRESSXYrrkz:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteCOMPRESSXYrrkz], (instrs VCOMPRESSPSZ128rrkz, VCOMPRESSPDZ128rrkz,
                                                 VPCOMPRESSBZ128rrkz, VPCOMPRESSWZ128rrkz,
                                                 VPCOMPRESSDZ128rrkz, VPCOMPRESSQZ128rrkz,
                                                 VCOMPRESSPSZ256rrkz, VCOMPRESSPDZ256rrkz,
                                                 VPCOMPRESSBZ256rrkz, VPCOMPRESSWZ256rrkz,
                                                 VPCOMPRESSDZ256rrkz, VPCOMPRESSQZ256rrkz)>;

def C4GM8WriteCOMPRESSXYrrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteCOMPRESSXYrrk], (instrs VCOMPRESSPSZ128rrk, VCOMPRESSPDZ128rrk,
                                                VPCOMPRESSBZ128rrk, VPCOMPRESSWZ128rrk,
                                                VPCOMPRESSDZ128rrk, VPCOMPRESSQZ128rrk,
                                                VCOMPRESSPSZ256rrk, VCOMPRESSPDZ256rrk,
                                                VPCOMPRESSBZ256rrk, VPCOMPRESSWZ256rrk,
                                                VPCOMPRESSDZ256rrk, VPCOMPRESSQZ256rrk)>;

def C4GM8WriteCOMPRESSXYmrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02, C4GM8AGU, C4GM8Store]> {
  let Latency = !add(C4GM8WriteCOMPRESSXYrrk.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [2, 4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteCOMPRESSXYmrk], (instrs VCOMPRESSPSZ128mrk, VCOMPRESSPDZ128mrk,
                                                VPCOMPRESSBZ128mrk, VPCOMPRESSWZ128mrk,
                                                VPCOMPRESSDZ128mrk, VPCOMPRESSQZ128mrk,
                                                VCOMPRESSPSZ256mrk, VCOMPRESSPDZ256mrk,
                                                VPCOMPRESSBZ256mrk, VPCOMPRESSWZ256mrk,
                                                VPCOMPRESSDZ256mrk, VPCOMPRESSQZ256mrk)>;

def C4GM8WriteCOMPRESSXY:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteCOMPRESSXY], (instrs VCOMPRESSPSZ128rr, VCOMPRESSPDZ128rr,
                                             VPCOMPRESSBZ128rr, VPCOMPRESSWZ128rr,
                                             VPCOMPRESSDZ128rr, VPCOMPRESSQZ128rr,
                                             VCOMPRESSPSZ256rr, VCOMPRESSPDZ256rr,
                                             VPCOMPRESSBZ256rr, VPCOMPRESSWZ256rr,
                                             VPCOMPRESSDZ256rr, VPCOMPRESSQZ256rr)>;

def C4GM8WriteCOMPRESSXYLd:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02, C4GM8AGU, C4GM8Store]> {
  let Latency = !add(C4GM8WriteCOMPRESSXY.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [2, 4, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteCOMPRESSXYLd], (instrs VCOMPRESSPSZ128mr, VCOMPRESSPDZ128mr,
                                               VPCOMPRESSBZ128mr, VPCOMPRESSWZ128mr,
                                               VPCOMPRESSDZ128mr, VPCOMPRESSQZ128mr,
                                               VCOMPRESSPSZ256mr, VCOMPRESSPDZ256mr,
                                               VPCOMPRESSBZ256mr, VPCOMPRESSWZ256mr,
                                               VPCOMPRESSDZ256mr, VPCOMPRESSQZ256mr)>;

def C4GM8WriteCOMPRESSZrrkz:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteCOMPRESSZrrkz], (instrs VCOMPRESSPSZrrkz, VCOMPRESSPDZrrkz,
                                                VPCOMPRESSBZrrkz, VPCOMPRESSWZrrkz,
                                                VPCOMPRESSDZrrkz, VPCOMPRESSQZrrkz)>;

def C4GM8WriteCOMPRESSZrrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 7;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteCOMPRESSZrrk], (instrs VCOMPRESSPSZrrk, VCOMPRESSPDZrrk,
                                               VPCOMPRESSBZrrk, VPCOMPRESSDZrrk,
                                               VPCOMPRESSWZrrk, VPCOMPRESSQZrrk)>;

def C4GM8WriteCOMPRESSZmrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02, C4GM8AGU, C4GM8Store]> {
  let Latency = !add(C4GM8WriteCOMPRESSZrrk.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [6, 12, 1, 1];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteCOMPRESSZmrk], (instrs VCOMPRESSPSZmrk, VCOMPRESSPDZmrk,
                                               VPCOMPRESSBZmrk, VPCOMPRESSDZmrk,
                                               VPCOMPRESSWZmrk, VPCOMPRESSQZmrk)>;

def C4GM8WriteCOMPRESSZ:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [6, 12];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteCOMPRESSZ], (instrs VCOMPRESSPSZrr, VCOMPRESSPDZrr,
                                            VPCOMPRESSBZrr, VPCOMPRESSWZrr,
                                            VPCOMPRESSDZrr, VPCOMPRESSQZrr)>;

def C4GM8WriteCOMPRESSZLd:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02, C4GM8AGU, C4GM8Store]> {
  let Latency = !add(C4GM8WriteCOMPRESSZ.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [6, 12, 1, 1];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteCOMPRESSZLd], (instrs VCOMPRESSPSZmr, VCOMPRESSPDZmr,
                                              VPCOMPRESSBZmr, VPCOMPRESSWZmr,
                                              VPCOMPRESSDZmr, VPCOMPRESSQZmr)>;

// EXPAND
def C4GM8WriteEXPANDXYrrkz:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXYrrkz], (instrs VEXPANDPSZ128rrkz, VEXPANDPDZ128rrkz,
                                               VPEXPANDBZ128rrkz, VPEXPANDWZ128rrkz,
                                               VPEXPANDDZ128rrkz, VPEXPANDQZ128rrkz,
                                               VEXPANDPSZ256rrkz, VEXPANDPDZ256rrkz,
                                               VPEXPANDBZ256rrkz, VPEXPANDWZ256rrkz,
                                               VPEXPANDDZ256rrkz, VPEXPANDQZ256rrkz)>;

def C4GM8WriteEXPANDXYrmkz:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDXYrrkz.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXYrmkz], (instrs VEXPANDPSZ128rmkz, VEXPANDPDZ128rmkz,
                                               VPEXPANDBZ128rmkz, VPEXPANDWZ128rmkz,
                                               VPEXPANDDZ128rmkz, VPEXPANDQZ128rmkz,
                                               VEXPANDPSZ256rmkz, VEXPANDPDZ256rmkz,
                                               VPEXPANDBZ256rmkz, VPEXPANDWZ256rmkz,
                                               VPEXPANDDZ256rmkz, VPEXPANDQZ256rmkz)>;

def C4GM8WriteEXPANDXYrrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXYrrk], (instrs VEXPANDPSZ128rrk, VEXPANDPDZ128rrk,
                                              VPEXPANDBZ128rrk, VPEXPANDWZ128rrk,
                                              VPEXPANDDZ128rrk, VPEXPANDQZ128rrk,
                                              VEXPANDPSZ256rrk, VEXPANDPDZ256rrk,
                                              VPEXPANDBZ256rrk, VPEXPANDWZ256rrk,
                                              VPEXPANDDZ256rrk, VPEXPANDQZ256rrk)>;

def C4GM8WriteEXPANDXYrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXYrmk], (instrs VEXPANDPSZ128rmk, VEXPANDPDZ128rmk,
                                              VPEXPANDBZ128rmk, VPEXPANDWZ128rmk,
                                              VPEXPANDDZ128rmk, VPEXPANDQZ128rmk,
                                              VEXPANDPSZ256rmk, VEXPANDPDZ256rmk,
                                              VPEXPANDBZ256rmk, VPEXPANDWZ256rmk,
                                              VPEXPANDDZ256rmk, VPEXPANDQZ256rmk)>;

def C4GM8WriteEXPANDXY:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXY], (instrs VEXPANDPSZ128rr, VEXPANDPDZ128rr,
                                           VPEXPANDBZ128rr, VPEXPANDWZ128rr,
                                           VPEXPANDDZ128rr, VPEXPANDQZ128rr,
                                           VEXPANDPSZ256rr, VEXPANDPDZ256rr,
                                           VPEXPANDBZ256rr, VPEXPANDWZ256rr,
                                           VPEXPANDDZ256rr, VPEXPANDQZ256rr)>;

def C4GM8WriteEXPANDXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDXY.Latency);
  let ReleaseAtCycles = [1, 1, 2, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteEXPANDXYLd], (instrs VEXPANDPSZ128rm, VEXPANDPDZ128rm,
                                             VPEXPANDBZ128rm, VPEXPANDWZ128rm,
                                             VPEXPANDDZ128rm, VPEXPANDQZ128rm,
                                             VEXPANDPSZ256rm, VEXPANDPDZ256rm,
                                             VPEXPANDBZ256rm, VPEXPANDWZ256rm,
                                             VPEXPANDDZ256rm, VPEXPANDQZ256rm)>;

def C4GM8WriteEXPANDZrrkz:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 10;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZrrkz], (instrs VEXPANDPSZrrkz, VEXPANDPDZrrkz,
                                              VPEXPANDBZrrkz, VPEXPANDWZrrkz,
                                              VPEXPANDDZrrkz, VPEXPANDQZrrkz)>;

def C4GM8WriteEXPANDZrmkz:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDZrrkz.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZrmkz], (instrs VEXPANDPSZrmkz, VEXPANDPDZrmkz,
                                              VPEXPANDBZrmkz, VPEXPANDWZrmkz,
                                              VPEXPANDDZrmkz, VPEXPANDQZrmkz)>;

def C4GM8WriteEXPANDZrrk:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 8;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZrrk], (instrs VEXPANDPSZrrk, VEXPANDPDZrrk,
                                             VPEXPANDBZrrk, VPEXPANDDZrrk,
                                             VPEXPANDWZrrk, VPEXPANDQZrrk)>;

def C4GM8WriteEXPANDZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZrmk], (instrs VEXPANDPSZrmk, VEXPANDPDZrmk,
                                             VPEXPANDBZrmk, VPEXPANDDZrmk,
                                             VPEXPANDWZrmk, VPEXPANDQZrmk)>;

def C4GM8WriteEXPANDZ:SchedWriteRes<[C4GM8FPU3, C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZ], (instrs VEXPANDPSZrr, VEXPANDPDZrr,
                                          VPEXPANDBZrr, VPEXPANDWZrr,
                                          VPEXPANDDZrr, VPEXPANDQZrr)>;

def C4GM8WriteEXPANDZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU3, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteEXPANDZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 16];
  let NumMicroOps = 10;
}
def : InstRW<[C4GM8WriteEXPANDZLd], (instrs VEXPANDPSZrm, VEXPANDPDZrm,
                                            VPEXPANDBZrm, VPEXPANDWZrm,
                                            VPEXPANDDZrm, VPEXPANDQZrm)>;

// VINSERT
def C4GM8WriteVINSERT256Z:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERT256Z], (instregex "VINSERT(F|I)(32X8|64X4)Zrri(k|kz)?")>;

def C4GM8WriteVINSERT256ZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVINSERT256Z.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERT256ZLd], (instregex "VINSERT(F|I)(32X8|64X4)Zrmi(k|kz)?")>;

def C4GM8WriteVINSERTY:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVINSERTY], (instregex "VINSERT(F|I)(32X4|64X2)Z256rrik")>;

def C4GM8WriteVINSERTYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVINSERTY.Latency);
  let ReleaseAtCycles = [1, 1, 4, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVINSERTYLd], (instregex "VINSERT(F|I)(32X4|64X2)Z256rmik")>;

def C4GM8WriteVINSERTZ:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERTZ], (instregex "VINSERT(F|I)(32X4|64X2)Zrrik")>;

def C4GM8WriteVINSERTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVINSERTZ.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERTZLd], (instregex "VINSERT(F|I)(32X4|64X2)Zrmik")>;

def C4GM8WriteVINSERTZrri:SchedWriteRes<[C4GM8FPU02, C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERTZrri], (instregex "VINSERT(F|I)(32X4|64X2)Zrri$")>;

def C4GM8WriteVINSERTZrmi:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVINSERTZrri.Latency);
  let ReleaseAtCycles = [1, 1, 8, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVINSERTZrmi], (instregex "VINSERT(F|I)(32X4|64X2)Zrmi$")>;

// VEXTRACT
def C4GM8WriteVEXTRACTZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVEXTRACTZ], (instregex "VEXTRACT(F|I)(32X8|64X4)Z(rri|rrikz)$")>;

def C4GM8WriteVEXTRACTZLd:SchedWriteRes<[C4GM8FPU13, C4GM8FPSt, C4GM8Store]> {
  let Latency = !add(C4GM8WriteVEXTRACTZ.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVEXTRACTZLd], (instregex "VEXTRACT(F|I)(32X8|64X4)Z(mri|mrikz)$")>;

def C4GM8WriteVEXTRACTZrrik:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVEXTRACTZrrik], (instregex "VEXTRACT(F|I)(32X8|64X4)Zrrik$")>;

def C4GM8WriteVEXTRACTZmrik:SchedWriteRes<[C4GM8FPU13, C4GM8FPSt, C4GM8Store]> {
  let Latency = !add(C4GM8WriteVEXTRACTZrrik.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVEXTRACTZmrik], (instregex "VEXTRACT(F|I)(32X8|64X4)Zmrik$")>;

def C4GM8WriteVEXTRACTY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVEXTRACTY], (instregex "VEXTRACT(F|I)(32X4|64X2)(Z|Z256)rrik$")>;

def C4GM8WriteVEXTRACTYLd:SchedWriteRes<[C4GM8FPU02, C4GM8FPSt, C4GM8Store]> {
  let Latency = !add(C4GM8WriteVEXTRACTY.Latency, C864GM8Model.StoreLatency);
  let ReleaseAtCycles = [4, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVEXTRACTYLd], (instregex "VEXTRACT(F|I)(32X4|64X2)(Z|Z256)mrik$")>;

// VPCONFLIC
def C4GM8WriteVPCONFLICX:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCONFLICX], (instregex "VPCONFLICT(D|Q)Z128(rr|rrk|rrkz)")>;

def C4GM8WriteVPCONFLICXLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCONFLICX.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCONFLICXLd], (instregex "VPCONFLICT(D|Q)Z128(rm|rmk|rmkz)")>;

def C4GM8WriteVPCONFLICY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 5;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteVPCONFLICY], (instregex "VPCONFLICT(D|Q)Z256(rr|rrkz)$")>;

def C4GM8WriteVPCONFLICYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCONFLICY.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 8;
}
def : InstRW<[C4GM8WriteVPCONFLICYLd], (instregex "VPCONFLICT(D|Q)Z256(rm|rmkz)$")>;

def C4GM8WriteVPCONFLICYrrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [6];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCONFLICYrrk], (instregex "VPCONFLICT(D|Q)Z256rrk$")>;

def C4GM8WriteVPCONFLICYrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCONFLICYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 6];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPCONFLICYrmk], (instregex "VPCONFLICT(D|Q)Z256rmk$")>;

def C4GM8WriteVPCONFLICZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 7;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPCONFLICZ], (instregex "VPCONFLICT(D|Q)Z(rr|rrkz)$")>;

def C4GM8WriteVPCONFLICZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCONFLICZ.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPCONFLICZLd], (instregex "VPCONFLICT(D|Q)Z(rm|rmkz)$")>;

def C4GM8WriteVPCONFLICZrrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 6;
  let ReleaseAtCycles = [12];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPCONFLICZrrk], (instregex "VPCONFLICT(D|Q)Zrrk$")>;

def C4GM8WriteVPCONFLICZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPCONFLICZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 12];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPCONFLICZrmk], (instregex "VPCONFLICT(D|Q)Zrmk$")>;

// VPDP
def C4GM8WriteVPDPXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPDPXY], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)(Z128|Z256)rrk$")>;

def C4GM8WriteVPDPXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPDPXY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPDPXYLd], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)(Z128|Z256)rmk$")>;

def C4GM8WriteVPDPZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPDPZ], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)Zrrk$")>;

def C4GM8WriteVPDPZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPDPZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPDPZLd], (instregex "VPDP(BUSDS|BUSD|WSSDS|WSSD)Zrmk$")>;

// VBROADCAST VPBROADCAST
def C4GM8WriteVBROADCASTX:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTX], (instregex "V(P)?BROADCAST(SS|B|W)Z128(rr|rrkz)$")>;

def C4GM8WriteVBROADCASTXLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBROADCASTX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTXLd], (instregex "V(P)?BROADCAST(SS|B|W)Z128(rm|rmkz)$")>;

def C4GM8WriteVBROADCASTXrrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTXrrk], (instregex "V(P)?BROADCAST(SS|SD|I32X2|B|W|D|Q)(r)?Z128rrk$")>;

def C4GM8WriteVBROADCASTXrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBROADCASTXrrk.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTXrmk], (instregex "V(P)?BROADCAST(SS|SD|I32X2|B|W|D|Q)(r)?Z128rmk$")>;

def C4GM8WriteVBROADCASTY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTY], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Z256rrk$")>;

def C4GM8WriteVBROADCASTYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBROADCASTY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVBROADCASTYLd], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Z256rmk$")>;

def C4GM8WriteVBROADCASTZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVBROADCASTZ], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)Z(rr|rrkz)$")>;

def C4GM8WriteVBROADCASTZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBROADCASTZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVBROADCASTZLd], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)Z(rm|rmkz)$")>;

def C4GM8WriteVBROADCASTZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVBROADCASTZrrk], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Zrrk$")>;

def C4GM8WriteVBROADCASTZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVBROADCASTZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVBROADCASTZrmk], (instregex "V(P)?BROADCAST(SS|SD|F32X2|I32X2|B|W|D|Q)(r)?Zrmk$")>;

// VPERM
def C4GM8WriteVPERMXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMXY], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)(rr|rrkz)$")>;

def C4GM8WriteVPERMXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMXYLd], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)(rm|rmkz)$")>;


def C4GM8WriteVPERMXYrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMXYrrk], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)rrk$")>;

def C4GM8WriteVPERMXYrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMXYrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMXYrmk], (instregex "VPERM(PS|PD|B|W|D|Q)(Z128|Z256)rmk$")>;


def C4GM8WriteVPERMZ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 8;
  let ReleaseAtCycles = [32];
  let NumMicroOps = 8;
}
def : InstRW<[C4GM8WriteVPERMZ], (instregex "VPERM(PS|PD|B|W|D|Q)Zrr(k|kz)?",
                                            "VPERM2(F|I)128rri")>;

def C4GM8WriteVPERMZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMZ.Latency);
  let ReleaseAtCycles = [1, 1, 32];
  let NumMicroOps = 8;
}
def : InstRW<[C4GM8WriteVPERMZLd], (instregex "VPERM(PS|PD|B|W|D|Q)Zrm(k|kz)?",
                                              "VPERM2(F|I)128rmi")>;

def C4GM8WriteVPERMImmY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmY], (instregex "VPERM(PD|Q)Z256(ri|rikz)$")>;

def C4GM8WriteVPERMImmYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMImmY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmYLd], (instregex "VPERM(PD|Q)Z256(mi|mikz)$")>;

def C4GM8WriteVPERMImmYrik:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmYrik], (instregex "VPERM(PD|Q)Z256rik$")>;

def C4GM8WriteVPERMImmYmik:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMImmYrik.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmYmik], (instregex "VPERM(PD|Q)Z256mik$")>;

def C4GM8WriteVPERMImmZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmZ], (instregex "VPERM(PD|Q)Zri(k|kz)?")>;

def C4GM8WriteVPERMImmZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMImmZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMImmZLd], (instregex "VPERM(PD|Q)Zmi(k|kz)?")>;

// VPERMI2 VPERMT2
def C4GM8WriteVPERMI2X_VPERMT2X:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMI2X_VPERMT2X], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Z128(rr|rrkz)$")>;

def C4GM8WriteVPERMI2X_VPERMT2XLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMI2X_VPERMT2X.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMI2X_VPERMT2XLd], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Z128(rm|rmkz)$")>;

def C4GM8WriteVPERMI2X_VPERMT2Xrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMI2X_VPERMT2Xrrk], (instregex "VPERM(I|T)2(PS|PD|B|D|W|Q)Z128rrk$")>;

def C4GM8WriteVPERMI2X_VPERMT2Xrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMI2X_VPERMT2Xrrk.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMI2X_VPERMT2Xrmk], (instregex "VPERM(I|T)2(PS|PD|B|D|W|Q)Z128rmk$")>;

def C4GM8WriteVPERMT2Y:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 4;
  let ReleaseAtCycles = [16];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPERMT2Y], (instregex "VPERMT2(PS|PD|B|W|D|Q)Z256rr(kz)?")>;

def C4GM8WriteVPERMT2YLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMT2Y.Latency);
  let ReleaseAtCycles = [1, 1, 16];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPERMT2YLd], (instregex "VPERMT2(PS|PD|B|W|D|Q)Z256rm(kz)?")>;

def C4GM8WriteVPERMI2Y:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 6;
  let ReleaseAtCycles = [16];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPERMI2Y], (instregex "VPERMI2(PS|PD|B|W|D|Q)Z256rr(kz)?")>;

def C4GM8WriteVPERMI2YLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMI2Y.Latency);
  let ReleaseAtCycles = [1, 1, 16];
  let NumMicroOps = 4;
}
def : InstRW<[C4GM8WriteVPERMI2YLd], (instregex "VPERMI2(PS|PD|B|W|D|Q)Z256rm(kz)?")>;

def C4GM8WriteVPERMI2Z_VPERMT2Z:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 16;
  let ReleaseAtCycles = [64];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPERMI2Z_VPERMT2Z], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Zrr(kz)?")>;

def C4GM8WriteVPERMI2Z_VPERMT2ZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMI2Z_VPERMT2Z.Latency);
  let ReleaseAtCycles = [1, 1, 64];
  let NumMicroOps = 16;
}
def : InstRW<[C4GM8WriteVPERMI2Z_VPERMT2ZLd], (instregex "VPERM(I|T)2(PS|PD|B|W|D|Q)Zrm(kz)?")>;

// VPERMIL
def C4GM8WriteVPERMILXY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILXY], (instregex "VPERMIL(PS|PD)(Z128|Z256)rrk$")>;

def C4GM8WriteVPERMILXYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMILXY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILXYLd], (instregex "VPERMIL(PS|PD)(Z128|Z256)rmk$")>;

def C4GM8WriteVPERMILZ:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZ], (instregex "VPERMIL(PS|PD)Z(rr|rrkz)$")>;

def C4GM8WriteVPERMILZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMILZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZLd], (instregex "VPERMIL(PS|PD)Z(rm|rmkz)$")>;

def C4GM8WriteVPERMILZrrk:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZrrk], (instregex "VPERMIL(PS|PD)Zrrk$")>;

def C4GM8WriteVPERMILZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMILZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZrmk], (instregex "VPERMIL(PS|PD)Zrmk$")>;

def C4GM8WriteVPERMILZrik:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZrik], (instregex "VPERMIL(PS|PD)Zri(k|kz)?")>;

def C4GM8WriteVPERMILZmik:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVPERMILZrik.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteVPERMILZmik], (instregex "VPERMIL(PS|PD)Zmi(k|kz)?")>;

// ALIGN
def C4GM8WriteALIGNX:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNX], (instrs VALIGNDZ128rrik, VALIGNDZ128rrikz,
                                         VALIGNQZ128rrik, VALIGNQZ128rrikz)>;

def C4GM8WriteALIGNXLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteALIGNX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNXLd], (instrs VALIGNDZ128rmik, VALIGNDZ128rmikz,
                                           VALIGNQZ128rmik, VALIGNQZ128rmikz)>;

def C4GM8WriteALIGNY:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 3;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNY], (instrs VALIGNDZ256rri, VALIGNDZ256rrikz,
                                         VALIGNQZ256rri, VALIGNQZ256rrikz)>;

def C4GM8WriteALIGNYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteALIGNY.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNYLd], (instrs VALIGNDZ256rmi, VALIGNDZ256rmikz,
                                           VALIGNQZ256rmi, VALIGNQZ256rmikz)>;

def C4GM8WriteALIGNYrrik:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNYrrik], (instrs VALIGNDZ256rrik, VALIGNQZ256rrik)>;

def C4GM8WriteALIGNYrmik:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteALIGNYrrik.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteALIGNYrmik], (instrs VALIGNDZ256rmik, VALIGNQZ256rmik)>;

def C4GM8WriteALIGNZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 8;
}
def : InstRW<[C4GM8WriteALIGNZ], (instrs VALIGNDZrri, VALIGNQZrri,
                                         VALIGNDZrrik, VALIGNQZrrik,
                                         VALIGNDZrrikz, VALIGNQZrrikz)>;

def C4GM8WriteALIGNZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteALIGNZ.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 8;
}
def : InstRW<[C4GM8WriteALIGNZLd], (instrs VALIGNDZrmi, VALIGNQZrmi,
                                           VALIGNDZrmik, VALIGNQZrmik,
                                           VALIGNDZrmikz, VALIGNQZrmikz)>;

// PALIGN
def C4GM8WritePALIGNX:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNX], (instrs VPALIGNRZrrik, VPALIGNRZrrikz)>;

def C4GM8WritePALIGNXLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePALIGNX.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNXLd], (instrs VPALIGNRZrmik, VPALIGNRZrmikz)>;

def C4GM8WritePALIGNY:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNY], (instrs VPALIGNRZ128rrik, VPALIGNRZ128rrikz)>;

def C4GM8WritePALIGNYLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePALIGNY.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNYLd], (instrs VPALIGNRZ128rmik, VPALIGNRZ128rmikz)>;

def C4GM8WritePALIGNZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNZ], (instrs VPALIGNRZ256rrik, VPALIGNRZ256rrikz)>;

def C4GM8WritePALIGNZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WritePALIGNZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WritePALIGNZLd], (instrs VPALIGNRZ256rmik, VPALIGNRZ256rmikz)>;

// DBPSADBWZ
def C4GM8WriteDBPSADBWZ:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteDBPSADBWZ], (instrs VDBPSADBWZrri, VDBPSADBWZrrikz,
                                            VDBPSADBWZ128rri, VDBPSADBWZ128rrikz,
                                            VDBPSADBWZ256rri, VDBPSADBWZ256rrikz)>;

def C4GM8WriteDBPSADBWZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteDBPSADBWZ.Latency);
  let ReleaseAtCycles = [1, 1, 6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteDBPSADBWZLd], (instrs VDBPSADBWZrmi, VDBPSADBWZrmikz,
                                              VDBPSADBWZ128rmi, VDBPSADBWZ128rmikz,
                                              VDBPSADBWZ256rmi, VDBPSADBWZ256rmikz)>;

def C4GM8WriteDBPSADBWZrrik:SchedWriteRes<[C4GM8FPU02, C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteDBPSADBWZrrik], (instrs VDBPSADBWZrrik, VDBPSADBWZ128rrik, VDBPSADBWZ256rrik)>;

def C4GM8WriteDBPSADBWZrmikLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteDBPSADBWZrrik.Latency);
  let ReleaseAtCycles = [1, 1, 6, 6];
  let NumMicroOps = 5;
}
def : InstRW<[C4GM8WriteDBPSADBWZrmikLd], (instrs VDBPSADBWZrmik, VDBPSADBWZ128rmik, VDBPSADBWZ256rmik)>;

// VecLogicZ rrk
def C4GM8WriteVecLogicZ:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVecLogicZ], (instregex "VPAND(D|Q)Z(128|256)?rrk$",
                                               "VPANDN(D|Q)Z(128|256)?rrk$",
                                               "VPOR(D|Q)Z(128|256)?rrk$",
                                               "VPXOR(D|Q)Z(128|256)?rrk$",
                                               "VANDNP(D|S)Z(128|256)?rrk$",
                                               "VANDP(D|S)Z(128|256)?rrk$",
                                               "VORP(D|S)Z(128|256)?rrk$",
                                               "VXORP(D|S)Z(128|256)?rrk$")>;

def C4GM8WriteVecLogicZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVecLogicZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVecLogicZLd], (instregex "VPAND(D|Q)Z(128|256)?rmk$",
                                                 "VPANDN(D|Q)Z(128|256)?rmk$",
                                                 "VPOR(D|Q)Z(128|256)?rmk$",
                                                 "VPXOR(D|Q)Z(128|256)?rmk$",
                                                 "VANDNP(D|S)Z(128|256)?rmk$",
                                                 "VANDP(D|S)Z(128|256)?rmk$",
                                                 "VORP(D|S)Z(128|256)?rmk$",
                                                 "VXORP(D|S)Z(128|256)?rmk$")>;

// GF2P8MULB
def C4GM8WriteGF2P8MULB:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteGF2P8MULB], (instrs GF2P8MULBrr, VGF2P8MULBrr, VGF2P8MULBYrr)>;

def C4GM8WriteGF2P8MULBLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteGF2P8MULB.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteGF2P8MULBLd], (instrs GF2P8MULBrm, VGF2P8MULBrm, VGF2P8MULBYrm)>;

def C4GM8WriteGF2P8MULBXYZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 3;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteGF2P8MULBXYZ], (instrs VGF2P8MULBZrr, VGF2P8MULBZrrkz,
                                               VGF2P8MULBZ128rr, VGF2P8MULBZ128rrkz,
                                               VGF2P8MULBZ256rr, VGF2P8MULBZ256rrkz)>;

def C4GM8WriteGF2P8MULBXYZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteGF2P8MULBXYZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteGF2P8MULBXYZLd], (instrs VGF2P8MULBZrm, VGF2P8MULBZrmkz,
                                                 VGF2P8MULBZ128rm, VGF2P8MULBZ128rmkz,
                                                 VGF2P8MULBZ256rm, VGF2P8MULBZ256rmkz)>;

def C4GM8WriteGF2P8MULBXYZrrk:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteGF2P8MULBXYZrrk], (instrs VGF2P8MULBZrrk, VGF2P8MULBZ128rrk, VGF2P8MULBZ256rrk)>;

def C4GM8WriteGF2P8MULBXYZrmk:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteGF2P8MULBXYZrrk.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteGF2P8MULBXYZrmk], (instrs VGF2P8MULBZrmk, VGF2P8MULBZ128rmk, VGF2P8MULBZ256rmk)>;

// VGF2P8AFFINE
def C4GM8WriteVGF2P8AFFINEZ:SchedWriteRes<[C4GM8FPU13]> {
  let Latency = 4;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVGF2P8AFFINEZ], (instrs VGF2P8AFFINEQBZrrik, VGF2P8AFFINEQBZ128rrik, VGF2P8AFFINEQBZ256rrik,
                                                VGF2P8AFFINEINVQBZrrik, VGF2P8AFFINEINVQBZ128rrik, VGF2P8AFFINEINVQBZ256rrik)>;

def C4GM8WriteVGF2P8AFFINEZLd:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU13]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteVGF2P8AFFINEZ.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteVGF2P8AFFINEZLd], (instrs VGF2P8AFFINEQBZrmik, VGF2P8AFFINEQBZ128rmik, VGF2P8AFFINEQBZ256rmik,
                                                  VGF2P8AFFINEINVQBZrmik, VGF2P8AFFINEINVQBZ128rmik, VGF2P8AFFINEINVQBZ256rmik)>;

// Vector insert/extract operations.
defm : C4GM8WriteResFPPair<WriteVecInsert, [C4GM8FPLd01], 2>;
defm : C4GM8WriteRes<WriteVecExtract, [C4GM8FPLd01], 1, [4], 1>;
defm : C4GM8WriteRes<WriteVecExtractSt, [C4GM8FPSt, C4GM8AGU, C4GM8Store], 3, [4, 1, 1], 1>;

// MOVMSK Instructions.
defm : C4GM8WriteRes<WriteFMOVMSK, [C4GM8FPU13], 1>;
defm : C4GM8WriteRes<WriteMMXMOVMSK, [C4GM8FPU13], 1>;
defm : C4GM8WriteRes<WriteVecMOVMSK, [C4GM8FPU13], 1>;
defm : C4GM8WriteRes<WriteVecMOVMSKY, [C4GM8FPU13], 1>;

// Strings instructions.
defm : C4GM8WriteResFPPair<WritePCmpIStrM, [C4GM8FPU], 7, [25], 3>;
defm : C4GM8WriteResFPPair<WritePCmpEStrM, [C4GM8FPU], 7, [25], 8>;
defm : C4GM8WriteResFPPair<WritePCmpIStrI, [C4GM8FPU], 2, [8], 2>;
defm : C4GM8WriteResFPPair<WritePCmpEStrI, [C4GM8FPU], 8, [13], 8>;

// AES Instructions.
defm : C4GM8WriteResFPPair<WriteAESDecEnc, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteAESIMC, [C4GM8FPU13], 3>;
defm : C4GM8WriteResFPPair<WriteAESKeyGen, [C4GM8FPU13], 3>;

// SHA Instructions.

def C4GM8WriteSHA1rr:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 1;
  let ReleaseAtCycles = [1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA1rr], (instrs SHA1MSG1rr, SHA1MSG2rr, SHA1NEXTErr)>;

def C4GM8WriteSHA1rm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSHA1rr.Latency);
  let ReleaseAtCycles = [1, 1, 1];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA1rm], (instrs SHA1MSG1rm, SHA1MSG2rm, SHA1NEXTErm)>;

def C4GM8WriteSHA256MSG1rr:SchedWriteRes<[C4GM8FPU]> {
  let Latency = 2;
  let ReleaseAtCycles = [2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteSHA256MSG1rr], (instrs SHA256MSG1rr)>;

def C4GM8WriteSHA256MSG1rm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSHA256MSG1rr.Latency);
  let ReleaseAtCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteSHA256MSG1rm], (instrs SHA256MSG1rm)>;

def C4GM8WriteSHA256rr:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 4;
  let ReleaseAtCycles = [4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA256rr], (instrs SHA256MSG2rr, SHA256RNDS2rr)>;

def C4GM8WriteSHA256rm:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSHA256rr.Latency);
  let ReleaseAtCycles = [1, 1, 4];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA256rm], (instrs SHA256MSG2rm, SHA256RNDS2rm)>;

def C4GM8WriteSHA1RNDS4rri:SchedWriteRes<[C4GM8FPU02]> {
  let Latency = 5;
  let ReleaseAtCycles = [8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA1RNDS4rri], (instrs SHA1RNDS4rri)>;

def C4GM8WriteSHA1RNDS4rmi:SchedWriteRes<[C4GM8AGU, C4GM8Load, C4GM8FPU02]> {
  let Latency = !add(C864GM8Model.VecLoadLatency, C4GM8WriteSHA1RNDS4rri.Latency);
  let ReleaseAtCycles = [1, 1, 8];
  let NumMicroOps = 1;
}
def : InstRW<[C4GM8WriteSHA1RNDS4rmi], (instrs SHA1RNDS4rmi)>;

defm : C4GM8WriteRes<WriteFence, [C4GM8AGU]>;
defm : C4GM8WriteRes<WriteNop, []>;

//===----------------------------------------------------------------------===//
// Zero Cycle Move
//===----------------------------------------------------------------------===//

def C4GM8WriteZeroLatency : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 1;
}

def : InstRW<[C4GM8WriteZeroLatency], (instrs MOV32rr, MOV32rr_REV,
                                              MOV64rr, MOV64rr_REV,
                                              MOVSX32rr32)>;

def C4GM8WriteSwapRenameable : SchedWriteRes<[]> {
  let Latency = 0;
  let ReleaseAtCycles = [];
  let NumMicroOps = 2;
}
def : InstRW<[C4GM8WriteSwapRenameable], (instrs XCHG32rr, XCHG32ar,
                                                 XCHG64rr, XCHG64ar)>;

defm : C4GM8WriteRes<WriteXCHG, [C4GM8ALU], 0, [8], 2>;

defm : C4GM8WriteRes<WriteFMoveX, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteFMoveY, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteFMoveZ, [], 0, [], 1>;

defm : C4GM8WriteRes<WriteVecMove, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteVecMoveX, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteVecMoveY, [], 0, [], 1>;
defm : C4GM8WriteRes<WriteVecMoveZ, [], 0, [], 1>;

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

def C4GM8WriteZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteALU]>
]>;
def : InstRW<[C4GM8WriteZeroIdiom], (instrs XOR32rr, XOR32rr_REV,
                                            XOR64rr, XOR64rr_REV,
                                            SUB32rr, SUB32rr_REV,
                                            SUB64rr, SUB64rr_REV)>;

def C4GM8WriteZeroIdiomEFLAGS : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<CheckSameRegOperand<0, 1>>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                                 [WriteALU]>
]>;
def : InstRW<[C4GM8WriteZeroIdiomEFLAGS], (instrs CMP8rr,  CMP8rr_REV,
                                                  CMP16rr, CMP16rr_REV,
                                                  CMP32rr, CMP32rr_REV,
                                                  CMP64rr, CMP64rr_REV)>;

def C4GM8WriteFZeroIdiom : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogic]>
]>;

def : InstRW<[C4GM8WriteFZeroIdiom], (instrs VXORPSrr, VXORPDrr,
                                             VXORPSZ128rr,
                                             VXORPDZ128rr,
                                             VANDNPSrr, VANDNPDrr,
                                             VANDNPSZ128rr,
                                             VANDNPDZ128rr)>;

def C4GM8WriteFZeroIdiomY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicY]>
]>;
def : InstRW<[C4GM8WriteFZeroIdiomY], (instrs VXORPSYrr, VXORPDYrr,
                                              VXORPSZ256rr,
                                              VXORPDZ256rr,
                                              VANDNPSYrr, VANDNPDYrr,
                                              VANDNPSZ256rr,
                                              VANDNPDZ256rr)>;

def C4GM8WriteFZeroIdiomZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteFLogicZ]>
]>;
def : InstRW<[C4GM8WriteFZeroIdiomZ], (instrs VXORPSZrr, VXORPDZrr,
                                              VANDNPSZrr, VANDNPDZrr)>;

def C4GM8WriteVZeroIdiomLogicX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicX]>
]>;

def : InstRW<[C4GM8WriteVZeroIdiomLogicX], (instrs VPXORrr,
                                                   VPXORDZ128rr,
                                                   VPXORQZ128rr,
                                                   VPANDNrr,
                                                   VPANDNDZ128rr,
                                                   VPANDNQZ128rr)>;

def C4GM8WriteVZeroIdiomLogicY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicY]>
]>;
def : InstRW<[C4GM8WriteVZeroIdiomLogicY], (instrs VPXORYrr,
                                                   VPXORDZ256rr,
                                                   VPXORQZ256rr,
                                                   VPANDNYrr,
                                                   VPANDNDZ256rr,
                                                   VPANDNQZ256rr)>;

def C4GM8WriteVZeroIdiomLogicZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecLogicZ]>
]>;
def : InstRW<[C4GM8WriteVZeroIdiomLogicZ], (instrs VPXORDZrr, VPXORQZrr,
                                                   VPANDNDZrr, VPANDNQZrr)>;

def C4GM8WriteVZeroIdiomALUX : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUX]>
]>;


def : InstRW<[C4GM8WriteVZeroIdiomALUX],
             (instrs VPSUBBrr, VPSUBWrr, VPSUBDrr, VPSUBQrr,
                     VPSUBBZ128rr, VPSUBWZ128rr, VPSUBDZ128rr, VPSUBQZ128rr,
                     VPCMPGTBrr, VPCMPGTWrr, VPCMPGTDrr, VPCMPGTQrr,
                     VPCMPGTBZ128rr, VPCMPGTWZ128rr,
                     VPCMPGTDZ128rr, VPCMPGTQZ128rr)>;

def C4GM8WriteVZeroIdiomALUY : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUY]>
]>;
def : InstRW<[C4GM8WriteVZeroIdiomALUY],
             (instrs VPSUBBYrr, VPSUBWYrr, VPSUBDYrr, VPSUBQYrr,
                     VPSUBBZ256rr, VPSUBWZ256rr, VPSUBDZ256rr, VPSUBQZ256rr,
                     VPCMPGTBYrr, VPCMPGTWYrr, VPCMPGTDYrr, VPCMPGTQYrr,
                     VPCMPGTBZ256rr, VPCMPGTWZ256rr,
                     VPCMPGTDZ256rr, VPCMPGTQZ256rr)>;

def C4GM8WriteVZeroIdiomALUZ : SchedWriteVariant<[
    SchedVar<MCSchedPredicate<ZeroIdiomPredicate>, [C4GM8WriteZeroLatency]>,
    SchedVar<NoSchedPred,                          [WriteVecALUZ]>
]>;
def : InstRW<[C4GM8WriteVZeroIdiomALUY],
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
