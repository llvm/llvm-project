//==- RISCVSchedSiFiveP800.td - SiFiveP800 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// On the SiFiveP800, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFiveP800IsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SiFiveP800IsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// 1 Micro-Op per cycle.
class SiFiveP800GetLMulCycles<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 1,
    !eq(mx, "M2") : 2,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 1,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

// Latency for segmented loads and stores are calculated as vl.
class SiFiveP800GetCyclesSegmented<string mx, int sew> {
  defvar VLEN = 128;
  int c = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
}

class SiFiveP800VSM3CCycles<string mx> {
  // c = ceil(LMUL / 2)
  int c = !cond(!eq(mx, "M2") : 1,
                !eq(mx, "M4") : 2,
                !eq(mx, "M8") : 4,
                true : 1);
}

// SiFiveP800 machine model for scheduling and other instruction cost heuristics.
def SiFiveP800Model : SchedMachineModel {
  let IssueWidth = 6;          // 6 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 288; // Max micro-ops that can be buffered.
  let LoadLatency = 4;         // Cycles for loads to access the cache.
  let MispredictPenalty = 9;   // Extra cycles for a mispredicted branch.
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVendorXSfvqmaccqoq, HasVendorXSfvqmaccdod];
  let CompleteModel = false;
}

let SchedModel = SiFiveP800Model in {

def SiFiveP800IEXQ0       : ProcResource<1>;
def SiFiveP800IEXQ1       : ProcResource<1>;
def SiFiveP800IEXQ2       : ProcResource<1>;
def SiFiveP800IEXQ3       : ProcResource<1>;
def SiFiveP800IEXQ4       : ProcResource<1>;
def SiFiveP800IEXQ5       : ProcResource<1>;
def SiFiveP800FEXQ0       : ProcResource<1>;
def SiFiveP800FEXQ1       : ProcResource<1>;

// Two Load/Store ports that can issue either two loads, two stores, or one load
// and one store.
def SiFiveP800LDST       : ProcResource<2>;
// One additional port that can only handle loads.
def SiFiveP800LD         : ProcResource<1>;
def SiFiveP800Load       : ProcResGroup<[SiFiveP800LDST, SiFiveP800LD]>;

// 6-wide pipeline with 6 ALU pipes.
def SiFiveP800IntArith    : ProcResGroup<[SiFiveP800IEXQ0, SiFiveP800IEXQ1, SiFiveP800IEXQ2, SiFiveP800IEXQ3]>;
defvar SiFiveP800SYS      = SiFiveP800IEXQ1;
defvar SiFiveP800CMOV     = SiFiveP800IEXQ3;
defvar SiFiveP800I2F      = SiFiveP800IEXQ3;
def SiFiveP800Mul         : ProcResGroup<[SiFiveP800IEXQ1, SiFiveP800IEXQ3]>;
def SiFiveP800Branch      : ProcResGroup<[SiFiveP800IEXQ4, SiFiveP800IEXQ5]>;
def SiFiveP800Div         : ProcResource<1>;

def SiFiveP800FloatArith  : ProcResGroup<[SiFiveP800FEXQ0, SiFiveP800FEXQ1]>;
defvar SiFiveP800F2I      = SiFiveP800FEXQ0;
def SiFiveP800FloatDiv    : ProcResource<1>;

// Vector pipeline
// VEXQ0 handle Mask, Simple Slide instructions,
// VEXQ1 handle Complex Slide, Permutation, Reductions, Divide instructions.
// Other vector instructions can be done in VEXQ0 and VEXQ1.
def SiFiveP800VEXQ0        : ProcResource<1>;
def SiFiveP800VEXQ1        : ProcResource<1>;
def SiFiveP800VectorArith  : ProcResGroup<[SiFiveP800VEXQ0, SiFiveP800VEXQ1]>;

def SiFiveP800VLD          : ProcResource<1>;
def SiFiveP800VST          : ProcResource<1>;
def SiFiveP800VDiv         : ProcResource<1>;
def SiFiveP800VFloatDiv    : ProcResource<1>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFiveP800IntArith]>;
def : WriteRes<WriteIALU32, [SiFiveP800IntArith]>;
def : WriteRes<WriteShiftImm, [SiFiveP800IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFiveP800IntArith]>;
def : WriteRes<WriteShiftReg, [SiFiveP800IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFiveP800IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFiveP800Branch]>;
def : WriteRes<WriteJal, [SiFiveP800Branch]>;
def : WriteRes<WriteJalr, [SiFiveP800Branch]>;

// CMOV
def P800WriteCMOV : SchedWriteRes<[SiFiveP800Branch, SiFiveP800CMOV]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[P800WriteCMOV], (instrs PseudoCCMOVGPRNoX0)>;

let Latency = 2 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFiveP800Mul]>;
def : WriteRes<WriteIMul32, [SiFiveP800Mul]>;
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFiveP800Mul]>;
def : WriteRes<WriteCPOP32, [SiFiveP800Mul]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFiveP800IEXQ2, SiFiveP800Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIDiv32,  [SiFiveP800IEXQ2, SiFiveP800Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Integer remainder
def : WriteRes<WriteIRem, [SiFiveP800IEXQ2, SiFiveP800Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIRem32,  [SiFiveP800IEXQ2, SiFiveP800Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Bitmanip
def : WriteRes<WriteRotateImm, [SiFiveP800IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFiveP800IntArith]>;
def : WriteRes<WriteRotateReg, [SiFiveP800IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFiveP800IntArith]>;

def : WriteRes<WriteCLZ, [SiFiveP800IntArith]>;
def : WriteRes<WriteCLZ32, [SiFiveP800IntArith]>;
def : WriteRes<WriteCTZ, [SiFiveP800IntArith]>;
def : WriteRes<WriteCTZ32, [SiFiveP800IntArith]>;

def : WriteRes<WriteORCB, [SiFiveP800IntArith]>;
def : WriteRes<WriteIMinMax, [SiFiveP800IntArith]>;

def : WriteRes<WriteREV8, [SiFiveP800IntArith]>;

def : WriteRes<WriteSHXADD, [SiFiveP800IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFiveP800IntArith]>;

def : WriteRes<WriteSingleBit, [SiFiveP800IntArith]>;
def : WriteRes<WriteSingleBitImm, [SiFiveP800IntArith]>;
def : WriteRes<WriteBEXT, [SiFiveP800IntArith]>;
def : WriteRes<WriteBEXTI, [SiFiveP800IntArith]>;

// Memory
def : WriteRes<WriteSTB, [SiFiveP800LDST]>;
def : WriteRes<WriteSTH, [SiFiveP800LDST]>;
def : WriteRes<WriteSTW, [SiFiveP800LDST]>;
def : WriteRes<WriteSTD, [SiFiveP800LDST]>;
def : WriteRes<WriteFST16, [SiFiveP800LDST]>;
def : WriteRes<WriteFST32, [SiFiveP800LDST]>;
def : WriteRes<WriteFST64, [SiFiveP800LDST]>;

let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFiveP800Load]>;
def : WriteRes<WriteLDH, [SiFiveP800Load]>;
def : WriteRes<WriteLDW, [SiFiveP800Load]>;
def : WriteRes<WriteLDD, [SiFiveP800Load]>;
}

let Latency = 5 in {
def : WriteRes<WriteFLD16, [SiFiveP800Load]>;
def : WriteRes<WriteFLD32, [SiFiveP800Load]>;
def : WriteRes<WriteFLD64, [SiFiveP800Load]>;
}

// Atomic memory
def : WriteRes<WriteAtomicSTW, [SiFiveP800LDST]>;
def : WriteRes<WriteAtomicSTD, [SiFiveP800LDST]>;

let Latency = 7 in {
def : WriteRes<WriteAtomicW, [SiFiveP800LDST]>;
def : WriteRes<WriteAtomicD, [SiFiveP800LDST]>;
}

let Latency = 10 in {
def : WriteRes<WriteAtomicLDW, [SiFiveP800Load]>;
def : WriteRes<WriteAtomicLDD, [SiFiveP800Load]>;
}

// Floating point
let Latency = 2 in {
def : WriteRes<WriteFAdd16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP800FloatArith]>;
}
let Latency = 3 in {
def : WriteRes<WriteFMul16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP800FloatArith]>;
}
let Latency = 4 in {
def : WriteRes<WriteFMA16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMA32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMA64, [SiFiveP800FloatArith]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFSGNJ32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFSGNJ64, [SiFiveP800FloatArith]>;

def : WriteRes<WriteFMinMax16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFiveP800FloatArith]>;
}

// Half precision.
def : WriteRes<WriteFDiv16, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 4;
  let ReleaseAtCycles = [1, 4];
}
def : WriteRes<WriteFSqrt16, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 8;
  let ReleaseAtCycles = [1, 7];
}

// Single precision.
def : WriteRes<WriteFDiv32, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 6;
  let ReleaseAtCycles = [1, 6];
}
def : WriteRes<WriteFSqrt32, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 14;
  let ReleaseAtCycles = [1, 13];
}

// Double precision
def : WriteRes<WriteFDiv64, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 11;
  let ReleaseAtCycles = [1, 11];
}
def : WriteRes<WriteFSqrt64, [SiFiveP800FEXQ1, SiFiveP800FloatDiv]> {
  let Latency = 29;
  let ReleaseAtCycles = [1, 28];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFiveP800I2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFiveP800F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFRoundF16, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFRoundF32, [SiFiveP800FloatArith]>;
def : WriteRes<WriteFRoundF64, [SiFiveP800FloatArith]>;

def : WriteRes<WriteFClass16, [SiFiveP800F2I]>;
def : WriteRes<WriteFClass32, [SiFiveP800F2I]>;
def : WriteRes<WriteFClass64, [SiFiveP800F2I]>;
def : WriteRes<WriteFCmp16, [SiFiveP800F2I]>;
def : WriteRes<WriteFCmp32, [SiFiveP800F2I]>;
def : WriteRes<WriteFCmp64, [SiFiveP800F2I]>;
def : WriteRes<WriteFMovI16ToF16, [SiFiveP800I2F]>;
def : WriteRes<WriteFMovF16ToI16, [SiFiveP800F2I]>;
def : WriteRes<WriteFMovI32ToF32, [SiFiveP800I2F]>;
def : WriteRes<WriteFMovF32ToI32, [SiFiveP800F2I]>;
def : WriteRes<WriteFMovI64ToF64, [SiFiveP800I2F]>;
def : WriteRes<WriteFMovF64ToI64, [SiFiveP800F2I]>;
def : WriteRes<WriteFLI16, [SiFiveP800I2F]>;
def : WriteRes<WriteFLI32, [SiFiveP800I2F]>;
def : WriteRes<WriteFLI64, [SiFiveP800I2F]>;
}

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [SiFiveP800SYS]>;
def : WriteRes<WriteVSETIVLI, [SiFiveP800SYS]>;
def : WriteRes<WriteVSETVL, [SiFiveP800SYS]>;

// 7. Vector Loads and Stores
// FIXME: This unit is still being improved, currently
// it is based on stage numbers. Estimates are optimistic,
// latency may be longer.
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",    [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [SiFiveP800VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDS8",   [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",  [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",  [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",  [SiFiveP800VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDUX8",  [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8",  [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFiveP800VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFiveP800VLD], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTE",    [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",    [SiFiveP800VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTS8",   [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFiveP800VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTUX8",  [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8",  [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFiveP800VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFiveP800VST], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar LMulLat = SiFiveP800GetCyclesSegmented<mx, eew>.c;
      defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
      let Latency = !add(12, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [SiFiveP800VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [SiFiveP800VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [SiFiveP800VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [SiFiveP800VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [SiFiveP800VLD], mx, IsWorstCase>;
      }
      let Latency = !add(1, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [SiFiveP800VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [SiFiveP800VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [SiFiveP800VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [SiFiveP800VST], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  let Latency = 8, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SiFiveP800VLD]>;
    def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SiFiveP800VST]>;
  }
  let Latency = 2, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [SiFiveP800VectorArith]>;
  }
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIALUV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVExtV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMX", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMI", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVShiftV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX", [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}
// Widening
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}

// Worst case needs 51/45/42/72 * lmul cycles for i8/16/32/64.
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 8): 51, !eq(sew, 16): 45, !eq(sew, 32): 42,
      /* SEW=64 */ true: 72);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SiFiveP800VEXQ1, SiFiveP800VDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SiFiveP800VEXQ1, SiFiveP800VDiv], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = 2, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF",  [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SiFiveP800VectorArith], mx, IsWorstCase>;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",  [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFClassV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SiFiveP800VectorArith], mx, IsWorstCase>;
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
    }
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SiFiveP800VectorArith], mx, sew, IsWorstCase>;
    }
  }
}

// Worst case needs around 29/25/37 * LMUL cycles for f16/32/64.
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 16): 29, !eq(sew, 32): 25, /* SEW=64 */ true: 37);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SiFiveP800VEXQ1, SiFiveP800VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SiFiveP800VEXQ1, SiFiveP800VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SiFiveP800VEXQ1, SiFiveP800VFloatDiv], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = !add(2, !mul(2, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = !add(2, !mul(2, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    let Latency = !add(6, !mul(6, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From",
                                     [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = !add(6, !mul(6, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFiveP800VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMPopV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIotaV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Simple Slide
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVISlide1X", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F", [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
}
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
}

// Complex Slide
foreach mx = ["M8", "M4", "M2"] in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !add(4, LMulLat), ReleaseAtCycles = [LMulLat] in {
    // TODO: The latencies and RThroughput for VISlideUpX and VISlideDownX are likely
    // to be different in non-trivial LMUL. Update to the correct numbers here.
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP800VEXQ1], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP800VEXQ1], mx, IsWorstCase>;
  }
}

let Latency = 2, ReleaseAtCycles = [1] in {
  def : WriteRes<WriteVMovXS, [SiFiveP800VectorArith]>;
  def : WriteRes<WriteVMovSX, [SiFiveP800VectorArith]>;
}
let Latency = 6, ReleaseAtCycles = [1] in {
  def : WriteRes<WriteVMovFS, [SiFiveP800VectorArith]>;
  def : WriteRes<WriteVMovSF, [SiFiveP800VectorArith]>;
}

// Simple Gather and Compress
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP800VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 3, ReleaseAtCycles = [1] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP800VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP800IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP800VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Vrgather.vi
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SiFiveP800VEXQ1], mx, IsWorstCase>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP800GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP800IsWorstCaseMX<mx, SchedMxList>.c;
  // Zvbb
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREVV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLZV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCPOPV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCTZV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLX",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLI",   [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  // Zvbc
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVCLMULV", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLMULX", [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  // Zvkb
  // VANDN uses WriteVIALU[V|X|I]
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREV8V",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVREV8V",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotV",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotX",    [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotI",    [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  // Zvkg
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVGHSHV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVGMULV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  // ZvknhaOrZvknhb
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSHA2CHV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSHA2CLV", [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defvar ZvknhSEWs = !listremove(SchedSEWSet<mx>.val, [8, 16]);
    // Largest SEW is the last element, assuming SchedSEWSet is sorted in ascending
    // order.
    defvar LargestZvknhSEW = !foldl(!head(ZvknhSEWs), ZvknhSEWs, last, curr, curr);
    foreach sew = ZvknhSEWs in {
      // The worst case for Zvknh[ab] is designated to the largest SEW and LMUL.
      defvar IsWorstCaseVSHA2MSV = !and(IsWorstCase, !eq(sew, LargestZvknhSEW));
      defm "" : LMULSEWWriteResMXSEW<"WriteVSHA2MSV", [SiFiveP800VEXQ0], mx, sew,
                                     IsWorstCaseVSHA2MSV>;
    }
  }
  // Zvkned
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVAESMVV",  [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF1V", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF2V", [SiFiveP800VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESZV",   [SiFiveP800VectorArith], mx, IsWorstCase>;
  }
  // Zvksed
  let Latency = 3, ReleaseAtCycles = [SiFiveP800VSM3CCycles<mx>.c] in
  defm "" : LMULWriteResMX<"WriteVSM3CV",   [SiFiveP800VEXQ0], mx, IsWorstCase>;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVSM3MEV",  [SiFiveP800VEXQ0], mx, IsWorstCase>;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSM4KV",   [SiFiveP800VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSM4RV",   [SiFiveP800VEXQ0], mx, IsWorstCase>;
  }
}

// Others
def : WriteRes<WriteCSR, [SiFiveP800SYS]>;
def : WriteRes<WriteNop, []>;
def : WriteRes<WriteRdVLENB, [SiFiveP800SYS]>;

// FIXME: This could be better modeled by looking at the regclasses of the operands.
def : InstRW<[WriteIALU, ReadIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 2, [WriteFMA16]>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 2, [WriteFMA32]>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 2, [WriteFMA64]>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFRoundF16, 0>;
def : ReadAdvance<ReadFRoundF32, 0>;
def : ReadAdvance<ReadFRoundF64, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// Bitmanip
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 0>;
def : ReadAdvance<ReadVSETVL, 0>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
// LMUL Aware
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm : LMULReadAdvance<"ReadVIALUV", 0>;
defm : LMULReadAdvance<"ReadVIALUX", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm : LMULReadAdvance<"ReadVExtV", 0>;
defm : LMULReadAdvance<"ReadVICALUV", 0>;
defm : LMULReadAdvance<"ReadVICALUX", 0>;
defm : LMULReadAdvance<"ReadVShiftV", 0>;
defm : LMULReadAdvance<"ReadVShiftX", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm : LMULReadAdvance<"ReadVICmpV", 0>;
defm : LMULReadAdvance<"ReadVICmpX", 0>;
defm : LMULReadAdvance<"ReadVIMinMaxV", 0>;
defm : LMULReadAdvance<"ReadVIMinMaxX", 0>;
defm : LMULReadAdvance<"ReadVIMulV", 0>;
defm : LMULReadAdvance<"ReadVIMulX", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm : LMULReadAdvance<"ReadVIMergeV", 0>;
defm : LMULReadAdvance<"ReadVIMergeX", 0>;
defm : LMULReadAdvance<"ReadVIMovV", 0>;
defm : LMULReadAdvance<"ReadVIMovX", 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 14. Vector Floating-Point Instructions
defm "" : LMULSEWReadAdvanceF<"ReadVFALUV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFALUF", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMulV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMulF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMulAddV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMulAddF", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULSEWReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVIotaV", 0>;

// 17. Vector Permutation Instructions
def : ReadAdvance<ReadVMovXS, 0>;
def : ReadAdvance<ReadVMovSX_V, 0>;
def : ReadAdvance<ReadVMovSX_X, 0>;
def : ReadAdvance<ReadVMovFS, 0>;
def : ReadAdvance<ReadVMovSF_V, 0>;
def : ReadAdvance<ReadVMovSF_F, 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherEI16VV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherEI16VV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// LMUL Aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Vector Crypto Extensions
// Zvbb
defm "" : LMULReadAdvance<"ReadVBREVV", 0>;
defm "" : LMULReadAdvance<"ReadVCLZV", 0>;
defm "" : LMULReadAdvance<"ReadVCPOPV", 0>;
defm "" : LMULReadAdvance<"ReadVCTZV", 0>;
defm "" : LMULReadAdvance<"ReadVWSLLV", 0>;
defm "" : LMULReadAdvance<"ReadVWSLLX", 0>;
// Zvbc
defm "" : LMULReadAdvance<"ReadVCLMULV", 0>;
defm "" : LMULReadAdvance<"ReadVCLMULX", 0>;
// Zvkb
// VANDN uses ReadVIALU[V|X|I]
defm "" : LMULReadAdvance<"ReadVBREV8V", 0>;
defm "" : LMULReadAdvance<"ReadVREV8V", 0>;
defm "" : LMULReadAdvance<"ReadVRotV", 0>;
defm "" : LMULReadAdvance<"ReadVRotX", 0>;
// Zvkg
defm "" : LMULReadAdvance<"ReadVGHSHV", 0>;
defm "" : LMULReadAdvance<"ReadVGMULV", 0>;
// Zvknha or Zvknhb
defm "" : LMULReadAdvance<"ReadVSHA2CHV", 0>;
defm "" : LMULReadAdvance<"ReadVSHA2CLV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVSHA2MSV", 0>;
// Zvkned
defm "" : LMULReadAdvance<"ReadVAESMVV", 0>;
defm "" : LMULReadAdvance<"ReadVAESKF1V", 0>;
defm "" : LMULReadAdvance<"ReadVAESKF2V", 0>;
defm "" : LMULReadAdvance<"ReadVAESZV", 0>;
// Zvksed
defm "" : LMULReadAdvance<"ReadVSM4KV", 0>;
defm "" : LMULReadAdvance<"ReadVSM4RV", 0>;
// Zbksh
defm "" : LMULReadAdvance<"ReadVSM3CV", 0>;
defm "" : LMULReadAdvance<"ReadVSM3MEV", 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;
def : ReadAdvance<ReadVPassthru_WorstCase, 0>;
foreach mx = SchedMxList in {
  def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx), 0>;
  foreach sew = SchedSEWSet<mx>.val in
    def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx  # "_E" # sew), 0>;
}

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedXsf;
}
