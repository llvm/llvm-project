//=- X86ScheduleZnver1.td - X86 Znver1 Scheduling -------------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Znver1 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def Znver1Model : SchedMachineModel {
  // Zen can decode 4 instructions per cycle.
  let IssueWidth = 4;
  // Based on the reorder buffer we define MicroOpBufferSize
  let MicroOpBufferSize = 192;
  let LoadLatency = 4;
  let MispredictPenalty = 17;
  let HighLatency = 25;
  let PostRAScheduler = 1;

  // FIXME: This variable is required for incomplete model.
  // We haven't catered all instructions.
  // So, we reset the value of this variable so as to
  // say that the model is incomplete.
  let CompleteModel = 0;
}

let SchedModel = Znver1Model in {

// Zen can issue micro-ops to 10 different units in one cycle.
// These are
//  * Four integer ALU units (ZALU0, ZALU1, ZALU2, ZALU3)
//  * Two AGU units (ZAGU0, ZAGU1)
//  * Four FPU units (ZFPU0, ZFPU1, ZFPU2, ZFPU3)
// AGUs feed load store queues @two loads and 1 store per cycle.

// Four ALU units are defined below
def ZnALU0 : ProcResource<1>;
def ZnALU1 : ProcResource<1>;
def ZnALU2 : ProcResource<1>;
def ZnALU3 : ProcResource<1>;

// Two AGU units are defined below
def ZnAGU0 : ProcResource<1>;
def ZnAGU1 : ProcResource<1>;

// Four FPU units are defined below
def ZnFPU0 : ProcResource<1>;
def ZnFPU1 : ProcResource<1>;
def ZnFPU2 : ProcResource<1>;
def ZnFPU3 : ProcResource<1>;

// FPU grouping
def ZnFPU     : ProcResGroup<[ZnFPU0, ZnFPU1, ZnFPU2, ZnFPU3]>;
def ZnFPU013  : ProcResGroup<[ZnFPU0, ZnFPU1, ZnFPU3]>;
def ZnFPU01   : ProcResGroup<[ZnFPU0, ZnFPU1]>;
def ZnFPU12   : ProcResGroup<[ZnFPU1, ZnFPU2]>;
def ZnFPU13   : ProcResGroup<[ZnFPU1, ZnFPU3]>;
def ZnFPU23   : ProcResGroup<[ZnFPU2, ZnFPU3]>;
def ZnFPU02   : ProcResGroup<[ZnFPU0, ZnFPU2]>;
def ZnFPU03   : ProcResGroup<[ZnFPU0, ZnFPU3]>;

// Below are the grouping of the units.
// Micro-ops to be issued to multiple units are tackled this way.

// ALU grouping
// ZnALU03 - 0,3 grouping
def ZnALU03: ProcResGroup<[ZnALU0, ZnALU3]>;

// 56 Entry (14x4 entries) Int Scheduler
def ZnALU : ProcResGroup<[ZnALU0, ZnALU1, ZnALU2, ZnALU3]> {
  let BufferSize=56;
}

// 28 Entry (14x2) AGU group. AGUs can't be used for all ALU operations
// but are relevant for some instructions
def ZnAGU : ProcResGroup<[ZnAGU0, ZnAGU1]> {
  let BufferSize=28;
}

// Integer Multiplication issued on ALU1.
def ZnMultiplier : ProcResource<1>;

// Integer division issued on ALU2.
def ZnDivider : ProcResource<1>;

// 4 Cycles load-to use Latency is captured
def : ReadAdvance<ReadAfterLd, 4>;

// (a folded load is an instruction that loads and does some operation)
// Ex: ADDPD xmm,[mem]-> This instruction has two micro-ops 
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops.
//      a. load and
//      b. addpd
// This multiclass is for folded loads for integer units.
multiclass ZnWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant takes 1-cycle on Execution Port.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on ZnAGU 
  // adds 4 cycles to the latency.
  def : WriteRes<SchedRW.Folded, [ZnAGU, ExePort]> {
     let Latency = !add(Lat, 4);
  }
}

// This multiclass is for folded loads for floating point units.
multiclass ZnWriteResFpuPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant takes 1-cycle on Execution Port.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on ZnAGU
  // adds 7 cycles to the latency.
  def : WriteRes<SchedRW.Folded, [ZnAGU, ExePort]> {
     let Latency = !add(Lat, 7);
  }
}

// WriteRMW is set for instructions with Memory write 
// operation in codegen
def : WriteRes<WriteRMW, [ZnAGU]>;

def : WriteRes<WriteStore, [ZnAGU]>;
def : WriteRes<WriteMove,  [ZnALU]>;
def : WriteRes<WriteLoad,  [ZnAGU]> { let Latency = 8; }

def : WriteRes<WriteZero,  []>;
def : WriteRes<WriteLEA, [ZnALU]>;
defm : ZnWriteResPair<WriteALU,   ZnALU, 1>;
defm : ZnWriteResPair<WriteShift, ZnALU, 1>;
defm : ZnWriteResPair<WriteJump,  ZnALU, 1>;

// IDIV
def : WriteRes<WriteIDiv, [ZnALU2, ZnDivider]> {
  let Latency = 41;
  let ResourceCycles = [1, 41];
}

def : WriteRes<WriteIDivLd, [ZnALU2, ZnAGU, ZnDivider]> {
  let Latency = 45;
  let ResourceCycles = [1, 4, 41];
}

// IMUL
def  : WriteRes<WriteIMulH, [ZnALU1, ZnMultiplier]>{
  let Latency = 4;
}
def : WriteRes<WriteIMul, [ZnALU1, ZnMultiplier]> {
  let Latency = 4;
}

def : WriteRes<WriteIMulLd,[ZnALU1, ZnMultiplier]> {
  let Latency = 8;
}

// Floating point operations
defm : ZnWriteResFpuPair<WriteFHAdd,     ZnFPU0,  3>;
defm : ZnWriteResFpuPair<WriteFAdd,      ZnFPU0,  3>;
defm : ZnWriteResFpuPair<WriteFBlend,    ZnFPU01, 1>;
defm : ZnWriteResFpuPair<WriteFVarBlend, ZnFPU01, 1>;
defm : ZnWriteResFpuPair<WriteVarBlend,  ZnFPU0,  1>;
defm : ZnWriteResFpuPair<WriteCvtI2F,    ZnFPU3,  5>;
defm : ZnWriteResFpuPair<WriteCvtF2F,    ZnFPU3,  5>;
defm : ZnWriteResFpuPair<WriteCvtF2I,    ZnFPU3,  5>;
defm : ZnWriteResFpuPair<WriteFDiv,      ZnFPU3, 15>;
defm : ZnWriteResFpuPair<WriteFShuffle,  ZnFPU12, 1>;
defm : ZnWriteResFpuPair<WriteFMul,      ZnFPU0,  5>;
defm : ZnWriteResFpuPair<WriteFRcp,      ZnFPU01, 5>;
defm : ZnWriteResFpuPair<WriteFRsqrt,    ZnFPU01, 5>;
defm : ZnWriteResFpuPair<WriteFSqrt,     ZnFPU3, 20>;

// Vector integer operations which uses FPU units
defm : ZnWriteResFpuPair<WriteVecShift,   ZnFPU,   1>;
defm : ZnWriteResFpuPair<WriteVecLogic,   ZnFPU,   1>;
defm : ZnWriteResFpuPair<WritePHAdd,      ZnFPU,   1>;
defm : ZnWriteResFpuPair<WriteVecALU,     ZnFPU,   1>;
defm : ZnWriteResFpuPair<WriteVecIMul,    ZnFPU0,  4>;
defm : ZnWriteResFpuPair<WriteShuffle,    ZnFPU,   1>;
defm : ZnWriteResFpuPair<WriteBlend,      ZnFPU01, 1>;
defm : ZnWriteResFpuPair<WriteShuffle256, ZnFPU,   2>;

// Vector Shift Operations
defm : ZnWriteResFpuPair<WriteVarVecShift, ZnFPU12, 1>;

// AES Instructions.
defm : ZnWriteResFpuPair<WriteAESDecEnc, ZnFPU01, 4>;
defm : ZnWriteResFpuPair<WriteAESIMC, ZnFPU01, 4>;
defm : ZnWriteResFpuPair<WriteAESKeyGen, ZnFPU01, 4>;

def : WriteRes<WriteFence,  [ZnAGU]>;
def : WriteRes<WriteNop, []>;

// Following instructions with latency=100 are microcoded.
// We set long latency so as to block the entire pipeline.
defm : ZnWriteResFpuPair<WriteFShuffle256, ZnFPU, 100>;

//Microcoded Instructions
let Latency = 100 in {
  def : WriteRes<WriteMicrocoded, []>;
  def : WriteRes<WriteSystem, []>;
  def : WriteRes<WriteMPSAD, []>;
  def : WriteRes<WriteMPSADLd, []>;
  def : WriteRes<WriteCLMul, []>;
  def : WriteRes<WriteCLMulLd, []>;
  def : WriteRes<WritePCmpIStrM, []>;
  def : WriteRes<WritePCmpIStrMLd, []>;
  def : WriteRes<WritePCmpEStrI, []>;
  def : WriteRes<WritePCmpEStrILd, []>;
  def : WriteRes<WritePCmpEStrM, []>;
  def : WriteRes<WritePCmpEStrMLd, []>;
  def : WriteRes<WritePCmpIStrI, []>;
  def : WriteRes<WritePCmpIStrILd, []>;
  }
}
