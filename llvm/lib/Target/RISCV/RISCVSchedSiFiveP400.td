//==- RISCVSchedSiFiveP400.td - SiFiveP400 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// On the SiFiveP400, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFiveP400IsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SiFiveP400IsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// 1 Micro-Op per cycle.
class SiFiveP400GetLMulCycles<string mx> {
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

// Latency for segmented loads and stores are calculated as vl * nf.
class SiFiveP400GetCyclesSegmented<string mx, int sew, int nf> {
  defvar VLEN = 128;
  defvar VLUpperBound = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
  int c = !mul(VLUpperBound, nf);
}

// Both variants of floating point vector reductions are based on numbers collected
// from llvm-exegesis.
class VFReduceBaseCycles<int sew> {
  // The latency for simple unordered VFReduce is `C + 6 * log2(LMUL)`,
  // and `C * LMUL` for ordered VFReduce. This helper class provides the `C`.
  int val = !cond(!eq(sew, 16): 16,
                  !eq(sew, 32): 10,
                  !eq(sew, 64): 6);
}

class AdvancedVFReduceCycles<int sew, string mx> {
  // SEW = 64 has lower latencies and RThroughputs than other SEWs.
  int latency = !cond(!eq(mx, "M1"): !if(!eq(sew, 64), 4, 6),
                      !eq(mx, "M2"): !if(!eq(sew, 64), 6, 8),
                      !eq(mx, "M4"): !if(!eq(sew, 64), 8, 10),
                      !eq(mx, "M8"): !if(!eq(sew, 64), 11, 13),
                      true: !if(!eq(sew, 64), 4, 6));
  int rthroughput = !cond(!eq(mx, "M1"): !if(!eq(sew, 64), 2, 3),
                          !eq(mx, "M2"): !if(!eq(sew, 64), 3, 4),
                          !eq(mx, "M4"): !if(!eq(sew, 64), 5, 6),
                          !eq(mx, "M8"): !if(!eq(sew, 64), 10, 12),
                          true: !if(!eq(sew, 64), 2, 3));
}

// Both variants of integer vector reductions are based on numbers collected
// from llvm-exegesis.
// TODO: Fractional LMUL's latency and rthroughput.
class SimpleVIReduceCycles<string mx> {
  defvar LMul = SiFiveP400GetLMulCycles<mx>.c;
  int latency = !mul(LMul, 2);
  int rthroughput = !cond(
                      !eq(mx, "M1"): 1,
                      !eq(mx, "M2"): 2,
                      !eq(mx, "M4"): 4,
                      !eq(mx, "M8"): 9,
                      true: 1);
}

class AdvancedVIReduceCycles<int sew, string mx> {
  // `C - 2 * log2(SEW)`, where `C` = 16.1, 18.1, 20.1, and 23.8 for
  // M1/2/4/8, respectively.
  int latency = !cond(!eq(mx, "M1"): !sub(16, !mul(2, !logtwo(sew))),
                      !eq(mx, "M2"): !sub(18, !mul(2, !logtwo(sew))),
                      !eq(mx, "M4"): !sub(20, !mul(2, !logtwo(sew))),
                      !eq(mx, "M8"): !sub(23, !mul(2, !logtwo(sew))),
                      true: 4);
  int rthroughput = !cond(
                      // `8.3 - 1.02 * log2(SEW)`
                      !eq(mx, "M1"): !sub(8, !logtwo(sew)),
                      // `10.0 - 1.16 * log2(SEW)`. Note that `9 - log2(SEW)`
                      // is closer to the floor value of the original formula.
                      !eq(mx, "M2"): !sub(9, !logtwo(sew)),
                      // `14.2 - 1.53 * log2(SEW)`
                      !eq(mx, "M4"): !div(!sub(1420, !mul(153, !logtwo(sew))), 100),
                      // `24.1 - 2.3 * log2(SEW)`
                      !eq(mx, "M8"): !div(!sub(241, !mul(23, !logtwo(sew))), 10),
                      true: 1);
}

class SiFiveP400VSM3CCycles<string mx> {
  // c = ceil(LMUL / 2)
  int c = !cond(!eq(mx, "M2") : 1,
                !eq(mx, "M4") : 2,
                !eq(mx, "M8") : 4,
                true : 1);
}

def SiFiveP400Model : SchedMachineModel {
  let IssueWidth = 3;         // 3 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 56; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr];
  let CompleteModel = false;
}

// The SiFiveP400 microarchitecure has 6 pipelines:
// Three pipelines for integer operations.
// One pipeline for FPU operations.
// One pipeline for Load operations.
// One pipeline for Store operations.
let SchedModel = SiFiveP400Model in {

def SiFiveP400IEXQ0       : ProcResource<1>;
def SiFiveP400IEXQ1       : ProcResource<1>;
def SiFiveP400IEXQ2       : ProcResource<1>;
def SiFiveP400FEXQ0       : ProcResource<1>;
def SiFiveP400Load        : ProcResource<1>;
def SiFiveP400Store       : ProcResource<1>;

def SiFiveP400IntArith    : ProcResGroup<[SiFiveP400IEXQ0, SiFiveP400IEXQ1, SiFiveP400IEXQ2]>;
defvar SiFiveP400Branch   = SiFiveP400IEXQ0;
defvar SiFiveP400SYS      = SiFiveP400IEXQ1;
defvar SiFiveP400MulDiv   = SiFiveP400IEXQ2;
defvar SiFiveP400I2F      = SiFiveP400IEXQ2;
def SiFiveP400Div         : ProcResource<1>;

defvar SiFiveP400FloatArith  = SiFiveP400FEXQ0;
defvar SiFiveP400F2I      = SiFiveP400FEXQ0;
def SiFiveP400FloatDiv    : ProcResource<1>;

// Vector pipeline
def SiFiveP400VEXQ0        : ProcResource<1>;
def SiFiveP400VLD          : ProcResource<1>;
def SiFiveP400VST          : ProcResource<1>;
def SiFiveP400VDiv         : ProcResource<1>;
def SiFiveP400VFloatDiv    : ProcResource<1>;

let Latency = 1 in {
// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFiveP400IntArith]>;
def : WriteRes<WriteIALU32, [SiFiveP400IntArith]>;
def : WriteRes<WriteShiftImm, [SiFiveP400IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFiveP400IntArith]>;
def : WriteRes<WriteShiftReg, [SiFiveP400IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFiveP400IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFiveP400Branch]>;
def : WriteRes<WriteJal, [SiFiveP400Branch]>;
def : WriteRes<WriteJalr, [SiFiveP400Branch]>;
}

// CMOV
def P400WriteCMOV : SchedWriteRes<[SiFiveP400Branch, SiFiveP400IEXQ1]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[P400WriteCMOV], (instrs PseudoCCMOVGPRNoX0)>;

let Latency = 2 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFiveP400MulDiv]>;
def : WriteRes<WriteIMul32, [SiFiveP400MulDiv]>;
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFiveP400MulDiv]>;
def : WriteRes<WriteCPOP32, [SiFiveP400MulDiv]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFiveP400MulDiv, SiFiveP400Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIDiv32, [SiFiveP400MulDiv, SiFiveP400Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Integer remainder
def : WriteRes<WriteIRem, [SiFiveP400MulDiv, SiFiveP400Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIRem32, [SiFiveP400MulDiv, SiFiveP400Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

let Latency = 1 in {
// Bitmanip
def : WriteRes<WriteRotateImm, [SiFiveP400IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFiveP400IntArith]>;
def : WriteRes<WriteRotateReg, [SiFiveP400IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFiveP400IntArith]>;

def : WriteRes<WriteCLZ, [SiFiveP400IntArith]>;
def : WriteRes<WriteCLZ32, [SiFiveP400IntArith]>;
def : WriteRes<WriteCTZ, [SiFiveP400IntArith]>;
def : WriteRes<WriteCTZ32, [SiFiveP400IntArith]>;

def : WriteRes<WriteORCB, [SiFiveP400IntArith]>;
def : WriteRes<WriteIMinMax, [SiFiveP400IntArith]>;

def : WriteRes<WriteREV8, [SiFiveP400IntArith]>;

def : WriteRes<WriteSHXADD, [SiFiveP400IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFiveP400IntArith]>;

def : WriteRes<WriteSingleBit, [SiFiveP400IntArith]>;
def : WriteRes<WriteSingleBitImm, [SiFiveP400IntArith]>;
def : WriteRes<WriteBEXT, [SiFiveP400IntArith]>;
def : WriteRes<WriteBEXTI, [SiFiveP400IntArith]>;
}

// Memory
let Latency = 1 in {
def : WriteRes<WriteSTB, [SiFiveP400Store]>;
def : WriteRes<WriteSTH, [SiFiveP400Store]>;
def : WriteRes<WriteSTW, [SiFiveP400Store]>;
def : WriteRes<WriteSTD, [SiFiveP400Store]>;
def : WriteRes<WriteFST16, [SiFiveP400Store]>;
def : WriteRes<WriteFST32, [SiFiveP400Store]>;
def : WriteRes<WriteFST64, [SiFiveP400Store]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFiveP400Load]>;
def : WriteRes<WriteLDH, [SiFiveP400Load]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDW, [SiFiveP400Load]>;
def : WriteRes<WriteLDD, [SiFiveP400Load]>;
}

let Latency = 5 in {
def : WriteRes<WriteFLD16, [SiFiveP400Load]>;
def : WriteRes<WriteFLD32, [SiFiveP400Load]>;
def : WriteRes<WriteFLD64, [SiFiveP400Load]>;
}

// Atomic memory
let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [SiFiveP400Store]>;
def : WriteRes<WriteAtomicSTD, [SiFiveP400Store]>;
def : WriteRes<WriteAtomicW, [SiFiveP400Load]>;
def : WriteRes<WriteAtomicD, [SiFiveP400Load]>;
def : WriteRes<WriteAtomicLDW, [SiFiveP400Load]>;
def : WriteRes<WriteAtomicLDD, [SiFiveP400Load]>;
}

// Floating point
let Latency = 4 in {
def : WriteRes<WriteFAdd16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP400FloatArith]>;

def : WriteRes<WriteFMul16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP400FloatArith]>;

def : WriteRes<WriteFMA16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMA32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMA64, [SiFiveP400FloatArith]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFSGNJ32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFSGNJ64, [SiFiveP400FloatArith]>;

def : WriteRes<WriteFMinMax16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFiveP400FloatArith]>;
}

// Half precision.
def : WriteRes<WriteFDiv16, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFSqrt16, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Single precision.
def : WriteRes<WriteFDiv32, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFSqrt32, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Double precision
def : WriteRes<WriteFDiv64, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}
def : WriteRes<WriteFSqrt64, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFiveP400I2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFiveP400F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFiveP400FloatArith]>;

def : WriteRes<WriteFClass16, [SiFiveP400F2I]>;
def : WriteRes<WriteFClass32, [SiFiveP400F2I]>;
def : WriteRes<WriteFClass64, [SiFiveP400F2I]>;
def : WriteRes<WriteFCmp16, [SiFiveP400F2I]>;
def : WriteRes<WriteFCmp32, [SiFiveP400F2I]>;
def : WriteRes<WriteFCmp64, [SiFiveP400F2I]>;
def : WriteRes<WriteFMovI16ToF16, [SiFiveP400I2F]>;
def : WriteRes<WriteFMovF16ToI16, [SiFiveP400F2I]>;
def : WriteRes<WriteFMovI32ToF32, [SiFiveP400I2F]>;
def : WriteRes<WriteFMovF32ToI32, [SiFiveP400F2I]>;
def : WriteRes<WriteFMovI64ToF64, [SiFiveP400I2F]>;
def : WriteRes<WriteFMovF64ToI64, [SiFiveP400F2I]>;
}

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [SiFiveP400SYS]>;
def : WriteRes<WriteVSETIVLI, [SiFiveP400SYS]>;
def : WriteRes<WriteVSETVL, [SiFiveP400SYS]>;

// 7. Vector Loads and Stores
// FIXME: This unit is still being improved, currently
// it is based on stage numbers. Estimates are optimistic,
// latency may be longer.
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",    [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [SiFiveP400VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDS8",   [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",  [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",  [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",  [SiFiveP400VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDUX8",  [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8",  [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFiveP400VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFiveP400VLD], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTE",    [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",    [SiFiveP400VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTS8",   [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFiveP400VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTUX8",  [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8",  [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFiveP400VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFiveP400VST], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
      defvar LMulLat = SiFiveP400GetCyclesSegmented<mx, eew, nf>.c;
      let Latency = !add(12, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" #eew,   [SiFiveP400VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" #eew, [SiFiveP400VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" #eew,  [SiFiveP400VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" #eew, [SiFiveP400VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" #eew, [SiFiveP400VLD], mx, IsWorstCase>;
      }
      let Latency = !add(1, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" #eew,   [SiFiveP400VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" #eew,  [SiFiveP400VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" #eew, [SiFiveP400VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" #eew, [SiFiveP400VST], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  let Latency = 8, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SiFiveP400VLD]>;
    def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SiFiveP400VST]>;
  }
  let Latency = 2, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [SiFiveP400VEXQ0]>;
  }
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIALUV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVExtV",     [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMX",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMI",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }

  let Latency = !if(!lt(LMulLat, 2), 2, LMulLat), ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVShiftV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }

  let Latency = !if(!eq(mx, "M8"), 9, 6), ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIMulV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}
// Widening
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Worst case needs 51/45/42/72 * lmul cycles for i8/16/32/64.
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 8): 51, !eq(sew, 16): 45, !eq(sew, 32): 42,
      /* SEW=64 */ true: 72);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SiFiveP400VEXQ0, SiFiveP400VDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SiFiveP400VEXQ0, SiFiveP400VDiv], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = 2, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
    let Latency = 3, ReleaseAtCycles = [LMulLat] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFClassV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  let Latency = 3, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// Worst case needs around 29/25/37 * LMUL cycles for f16/32/64.
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 16): 29, !eq(sew, 32): 25, /* SEW=64 */ true: 37);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SiFiveP400VEXQ0, SiFiveP400VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SiFiveP400VEXQ0, SiFiveP400VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SiFiveP400VEXQ0, SiFiveP400VFloatDiv], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    // Simple reduction
    defvar SimpleC = SimpleVIReduceCycles<mx>;
    let Latency = SimpleC.latency, ReleaseAtCycles = [SimpleC.rthroughput] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;

    // Advanced reduction
    defvar AdvancedC = AdvancedVIReduceCycles<sew, mx>;
    let Latency = AdvancedC.latency, ReleaseAtCycles = [AdvancedC.rthroughput] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    defvar SimpleC = SimpleVIReduceCycles<mx>;
    let Latency = SimpleC.latency, ReleaseAtCycles = [SimpleC.rthroughput] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFiveP400VEXQ0],
                                      mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Simple reduction.
    defvar BaseC = VFReduceBaseCycles<sew>.val;
    let Latency = !add(BaseC, !mul(6, !logtwo(LMulLat))), ReleaseAtCycles = [BaseC] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;

    // Advanced reduction.
    defvar AdvancedC = AdvancedVFReduceCycles<sew, mx>;
    let Latency = AdvancedC.latency, ReleaseAtCycles = [AdvancedC.rthroughput] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From",
                                   [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;

    defvar OrderedRedCycles = !mul(BaseC, LMulLat);
    let Latency = OrderedRedCycles, ReleaseAtCycles = [OrderedRedCycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SiFiveP400VEXQ0],
                                    mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = !add(6, !mul(6, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SiFiveP400VEXQ0],
                                     mx, sew, IsWorstCase>;
    }

    defvar OrderedRedCycles = !mul(VFReduceBaseCycles<sew>.val, LMulLat);
    let Latency = OrderedRedCycles, ReleaseAtCycles = [OrderedRedCycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMPopV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIotaV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Simple Slide
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVISlide1X", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Complex Slide
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;

  defvar UpLatAndCycles = !add(8, LMulLat);
  let Latency = UpLatAndCycles, ReleaseAtCycles = [UpLatAndCycles] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  defvar DownLatAndCycles = !add(8, !div(!mul(LMulLat, 3), 2));
  let Latency = DownLatAndCycles, ReleaseAtCycles = [DownLatAndCycles] in {
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

let Latency = 2, ReleaseAtCycles = [2] in {
  def : WriteRes<WriteVMovXS, [SiFiveP400VEXQ0]>;
  def : WriteRes<WriteVMovSX, [SiFiveP400VEXQ0]>;
}
let Latency = 6, ReleaseAtCycles = [2] in {
  def : WriteRes<WriteVMovFS, [SiFiveP400VEXQ0]>;
  def : WriteRes<WriteVMovSF, [SiFiveP400VEXQ0]>;
}

// Simple Gather and Compress
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 3, ReleaseAtCycles = [1] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

foreach mx = ["M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP400IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = 6, ReleaseAtCycles = [!add(!mul(LMulLat, 2), 8)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Vrgather.vi
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP400GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP400IsWorstCaseMX<mx, SchedMxList>.c;
  // Zvbb
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREVV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLZV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCPOPV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCTZV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLI",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  // Zvbc
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVCLMULV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLMULX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  // Zvkb
  // VANDN uses WriteVIALU[V|X|I]
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREV8V",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVREV8V",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotV",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotX",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotI",    [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  // Zvkg
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVGHSHV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVGMULV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  // ZvknhaOrZvknhb
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSHA2CHV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSHA2CLV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defvar ZvknhSEWs = !listremove(SchedSEWSet<mx>.val, [8, 16]);
    // Largest SEW is the last element, assuming SchedSEWSet is sorted in ascending
    // order.
    defvar LargestZvknhSEW = !foldl(!head(ZvknhSEWs), ZvknhSEWs, last, curr, curr);
    foreach sew = ZvknhSEWs in {
      // The worst case for Zvknh[ab] is designated to the largest SEW and LMUL.
      defvar IsWorstCaseVSHA2MSV = !and(IsWorstCase, !eq(sew, LargestZvknhSEW));
      defm "" : LMULSEWWriteResMXSEW<"WriteVSHA2MSV", [SiFiveP400VEXQ0], mx, sew,
                                     IsWorstCaseVSHA2MSV>;
    }
  }
  // Zvkned
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVAESMVV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF1V", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF2V", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESZV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  // Zvksed
  let Latency = 3, ReleaseAtCycles = [SiFiveP400VSM3CCycles<mx>.c] in
  defm "" : LMULWriteResMX<"WriteVSM3CV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVSM3MEV",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSM4KV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSM4RV",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Others
def : WriteRes<WriteCSR, [SiFiveP400SYS]>;
def : WriteRes<WriteNop, []>;
def : WriteRes<WriteRdVLENB, [SiFiveP400SYS]>;


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
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
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

// Others
def : ReadAdvance<ReadVMask, 0>;
def : ReadAdvance<ReadVPassthru_WorstCase, 0>;
foreach mx = SchedMxList in {
  def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx), 0>;
  foreach sew = SchedSEWSet<mx>.val in
    def : ReadAdvance<!cast<SchedRead>("ReadVPassthru_" # mx  # "_E" # sew), 0>;
}

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

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedXsfvcp;
}
