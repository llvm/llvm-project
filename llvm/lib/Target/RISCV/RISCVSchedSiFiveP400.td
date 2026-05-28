//==- RISCVSchedSiFiveP400.td - SiFiveP400 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
include "RISCVSchedSiFive8.td"

// Latency for segmented loads and stores are calculated as vl * nf.
class SiFiveP400SegmentedLdStCycles<string mx, int sew, int nf> {
  int c = !mul(SiFive8VecGetVLMAX<mx, sew>.val, nf);
}

// Both variants of floating point vector reductions are based on numbers collected
// from llvm-exegesis.
class SiFiveP400VFReduceBaseCycles<int sew> {
  // The latency for simple unordered VFReduce is `C + 6 * log2(LMUL)`,
  // and `C * LMUL` for ordered VFReduce. This helper class provides the `C`.
  int val = !cond(!eq(sew, 16): 16,
                  !eq(sew, 32): 10,
                  !eq(sew, 64): 6);
}

class SiFiveP400AdvancedVFReduceCycles<int sew, string mx> {
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
class SiFiveP400SimpleVIReduceCycles<string mx> {
  defvar LMul = SiFive8VecGetLMulCycles<mx>.c;
  int latency = !mul(LMul, 2);
  int rthroughput = !cond(
                      !eq(mx, "M1"): 1,
                      !eq(mx, "M2"): 2,
                      !eq(mx, "M4"): 4,
                      !eq(mx, "M8"): 9,
                      true: 1);
}

class SiFiveP400AdvancedVIReduceCycles<int sew, string mx> {
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
  let MicroOpBufferSize = 96; // Max micro-ops that can be buffered.
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
// Two pipelines for Load & Store operations.
let SchedModel = SiFiveP400Model in
defm SiFiveP400 : SiFive8ProcResources<num_iex = 3, num_fex = 1>;

defvar SiFiveP400FloatArith = SiFiveP400FEXQ0;

def SiFiveP400IntProcResources : SiFive8IntProcResources {
  let Arith = SiFiveP400IntArith;
  let Mul = SiFiveP400IEXQ2;
  let I2F = SiFiveP400IEXQ2;
  let Branch = SiFiveP400IEXQ0;
  let SYS = SiFiveP400IEXQ1;
  let VSet = SiFiveP400IEXQ1;
  let CMOV = SiFiveP400IEXQ1;
  let Div = [SiFiveP400IEXQ2, SiFiveP400Div];
}

def SiFiveP400FloatProcResources : SiFive8FloatProcResources {
  let Arith = SiFiveP400FEXQ0;
  let Div = [SiFiveP400FEXQ0, SiFiveP400FloatDiv];
  let F2I = SiFiveP400FEXQ0;
}

def SiFiveP400LSUProcResources : SiFive8LSUProcResources {
  let Load = SiFiveP400Load;
  let Store = SiFiveP400Store;
}

def SiFiveP400VecProcResources : SiFive8VecProcResources {
  let Arith = SiFiveP400VEXQ0;
  let Load = SiFiveP400VLD;
  let Store = SiFiveP400VST;
  let Div = [SiFiveP400VEXQ0, SiFiveP400VDiv];
  let FloatDiv = [SiFiveP400VEXQ0, SiFiveP400VFloatDiv];
}

let SchedModel = SiFiveP400Model in {

defm : SiFive8BaseWriteRes<SiFiveP400IntProcResources,
                           SiFiveP400FloatProcResources,
                           SiFiveP400LSUProcResources>;

// Floating point arithmetics
let Latency = 4 in {
def : WriteRes<WriteFAdd16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP400FloatArith]>;

def : WriteRes<WriteFMul16, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP400FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP400FloatArith]>;
}

// Floating point divisions
def : WriteRes<WriteFDiv16, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFDiv32, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 19;
  let ReleaseAtCycles = [1, 18];
}
def : WriteRes<WriteFDiv64, [SiFiveP400FEXQ0, SiFiveP400FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}

defm : SiFive8VecBaseWriteRes<SiFiveP400IntProcResources, SiFiveP400VecProcResources>;

// Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
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
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxListW>.c;
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

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
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
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
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
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
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
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    // Simple reduction
    defvar SimpleC = SiFiveP400SimpleVIReduceCycles<mx>;
    let Latency = SimpleC.latency, ReleaseAtCycles = [SimpleC.rthroughput] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From",  [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;

    // Advanced reduction
    defvar AdvancedC = SiFiveP400AdvancedVIReduceCycles<sew, mx>;
    let Latency = AdvancedC.latency, ReleaseAtCycles = [AdvancedC.rthroughput] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    defvar SimpleC = SiFiveP400SimpleVIReduceCycles<mx>;
    let Latency = SimpleC.latency, ReleaseAtCycles = [SimpleC.rthroughput] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFiveP400VEXQ0],
                                      mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Simple reduction.
    defvar BaseC = SiFiveP400VFReduceBaseCycles<sew>.val;
    let Latency = !add(BaseC, !mul(6, !logtwo(LMulLat))), ReleaseAtCycles = [BaseC] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;

    // Advanced reduction.
    defvar AdvancedC = SiFiveP400AdvancedVFReduceCycles<sew, mx>;
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
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = !add(6, !mul(6, LMulLat)), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SiFiveP400VEXQ0],
                                     mx, sew, IsWorstCase>;
    }

    defvar OrderedRedCycles = !mul(SiFiveP400VFReduceBaseCycles<sew>.val, LMulLat);
    let Latency = OrderedRedCycles, ReleaseAtCycles = [OrderedRedCycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFiveP400VEXQ0],
                                   mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMPopV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIotaV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Simple Slide
foreach mx = SchedMxList in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVISlide1X", [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP400VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Complex Slide
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;

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
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 3, ReleaseAtCycles = [1] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

foreach mx = ["M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFive8VecIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = 6, ReleaseAtCycles = [!add(!mul(LMulLat, 2), 8)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP400VEXQ0], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Vrgather.vi
foreach mx = SchedMxList in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SiFiveP400VEXQ0], mx, IsWorstCase>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar LMulLat = SiFive8VecGetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFive8VecIsWorstCaseMX<mx, SchedMxList>.c;
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

//===----------------------------------------------------------------------===//
// Bypass and advance

defm : SiFive8BaseReadAdvances;

defm : SiFive8VecBaseReadAdvances;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsf;
}
