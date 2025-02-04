//==- RISCVSchedSiFiveP600.td - SiFiveP600 Scheduling Defs ---*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// c is true if mx has the worst case behavior compared to LMULs in MxList.
/// On the SiFiveP600, the worst case LMUL is the Largest LMUL
/// and the worst case sew is the smallest SEW for that LMUL.
class SiFiveP600IsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SiFiveP600IsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// 1 Micro-Op per cycle.
class SiFiveP600GetLMulCycles<string mx> {
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
class SiFiveP600GetCyclesSegmented<string mx, int sew, int nf> {
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

class SiFiveP600VSM3CCycles<string mx> {
  // c = ceil(LMUL / 2)
  int c = !cond(!eq(mx, "M2") : 1,
                !eq(mx, "M4") : 2,
                !eq(mx, "M8") : 4,
                true : 1);
}

class SiFiveP600RVVMultiplier<string mx> {
  int c = !if(!eq(mx, "M8"), 2, 1);
}

// ======================================================================
// The latency and occupancy data in this section are primarily evaluated
// from llvm-exegesis.
// ======================================================================

class SiFiveP600VCryptoLatency<string mx> {
  int c = !cond(
    !eq(mx, "M4"): 4,
    !eq(mx, "M8"): 8,
    true:          2
  );
}

class SiFiveP600VFMinMaxReduction<string mx, int sew> {
  defvar E64Lat = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 10,
    true:           2
  );

  defvar E64Cycles = !cond(
    !eq(mx, "M1") : 3,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 5,
    !eq(mx, "M8") : 6,
    true:           2
  );

  int latency = !if(!eq(sew, 64), E64Lat, !add(E64Lat, 2));
  int cycles = !if(!eq(sew, 64), E64Cycles, !add(E64Cycles, 1));
}

class SiFiveP600VFUnorderedReduction<string mx, int sew> {
  defvar E64Lat = !cond(
    !eq(mx, "M1") : 6,
    !eq(mx, "M2") : 12,
    !eq(mx, "M4") : 18,
    !eq(mx, "M8") : 24,
    true:           2
  );

  defvar E32Cycles = !cond(
    !eq(mx, "M1") : 10,
    !eq(mx, "M2") : 10,
    !eq(mx, "M4") : 11,
    !eq(mx, "M8") : 11,
    true:           6
  );

  int latency = !if(!eq(sew, 64), E64Lat, !add(E64Lat, 4));
  int cycles = !if(!eq(sew, 64), 6, E32Cycles);
}

class SiFiveP600VFWidenUnorderedReduction<string mx> {
  int latency = !cond(
    !eq(mx, "M1") : 10,
    !eq(mx, "M2") : 18,
    !eq(mx, "M4") : 24,
    !eq(mx, "M8") : 30,
    true:           6
  );
}

class SiFiveP600VFOrderedReduction<string mx, int sew> {
  defvar Base = !if(!eq(sew, 64), 6, 10);
  int c = !cond(
    !eq(mx, "M1") : Base,
    !eq(mx, "M2") : !mul(Base, 2),
    !eq(mx, "M4") : !mul(Base, 4),
    !eq(mx, "M8") : !mul(Base, 8),
    true:           6
  );
}

class SiFiveP600VIReductionLatency<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VIMinMaxReductionLatency<string mx, int sew> {
  // +-----+-----+-----+-----+----+
  // |     | E64 | E32 | E16 | E8 |
  // +-----+-----+-----+-----+----+
  // | MF8 |  X  |  X  |  X  |  4 |
  // +-----+-----+-----+-----+----+
  // | MF4 |  X  |  X  |  4  |  6 |
  // +-----+-----+-----+-----+----+
  // | MF2 |  X  |  4  |  6  |  8 |
  // +-----+-----+-----+-----+----+
  // | M1  |  4  |  6  |  8  | 10 |
  // +-----+-----+-----+-----+----+
  // | M2  |  6  |  8  |  10 | 12 |
  // +-----+-----+-----+-----+----+
  // | M4  |  8  |  10 |  12 | 14 |
  // +-----+-----+-----+-----+----+
  // | M8  |  10 |  12 |  14 | 16 |
  // +-----+-----+-----+-----+----+
  defvar BaseIndex = !cond(
    !eq(sew, 64): 0,
    !eq(sew, 32): 1,
    !eq(sew, 16): 2,
    !eq(sew, 8):  3
  );

  defvar Latencies = [4, 6, 8, 10, 12, 14, 16];

  int c = !cond(
    !eq(mx, "M1") : Latencies[BaseIndex],
    !eq(mx, "M2") : Latencies[!add(BaseIndex, 1)],
    !eq(mx, "M4") : Latencies[!add(BaseIndex, 2)],
    !eq(mx, "M8") : Latencies[!add(BaseIndex, 3)],
    // Fractional
    !eq(mx, "MF2"): Latencies[!sub(BaseIndex, 1)],
    !eq(mx, "MF4"): Latencies[!sub(BaseIndex, 2)],
    !eq(mx, "MF8"): Latencies[!sub(BaseIndex, 3)],
  );
}

class SiFiveP600VIMinMaxReductionCycles<string mx, int sew> {
  // +-----+-----+-----+-----+----+
  // |     | E64 | E32 | E16 | E8 |
  // +-----+-----+-----+-----+----+
  // | MF8 |  X  |  X  |  X  |  3 |
  // +-----+-----+-----+-----+----+
  // | MF4 |  X  |  X  |  3  |  5 |
  // +-----+-----+-----+-----+----+
  // | MF2 |  X  |  3  |  5  |  6 |
  // +-----+-----+-----+-----+----+
  // | M1  |  3  |  4  |  6  |  8 |
  // +-----+-----+-----+-----+----+
  // | M2  |  4  |  5  |  8  |  9 |
  // +-----+-----+-----+-----+----+
  // | M4  |  5  |  6  |  10 | 11 |
  // +-----+-----+-----+-----+----+
  // | M8  |  7  |  8  |  9  | 11 |
  // +-----+-----+-----+-----+----+
  defvar Index = !cond(
    !eq(sew, 64): 0,
    !eq(sew, 32): 1,
    !eq(sew, 16): 2,
    !eq(sew, 8):  3
  );

  defvar Cycles = [
    [0, 0, 0,  3],
    [0, 0, 3,  5],
    [0, 3, 5,  6],
    [3, 4, 6,  8],
    [4, 5, 8,  9],
    [5, 6, 10, 11],
    [7, 8, 9,  11]
  ];

  int c = !cond(
    !eq(mx, "MF8"): Cycles[0][Index],
    !eq(mx, "MF4"): Cycles[1][Index],
    !eq(mx, "MF2"): Cycles[2][Index],
    !eq(mx, "M1"):  Cycles[3][Index],
    !eq(mx, "M2"):  Cycles[4][Index],
    !eq(mx, "M4"):  Cycles[5][Index],
    !eq(mx, "M8"):  Cycles[6][Index],
  );
}

class SiFiveP600VSlide1<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 3,
    !eq(mx, "M4") : 4,
    !eq(mx, "M8") : 8,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VSlideI<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 6,
    !eq(mx, "M8") : 8,
    // M1 and lower
    true:           2
  );
}

class SiFiveP600VSlideXComplex<string mx, bit isUp = false> {
  int latency = !cond(
    !eq(mx, "M2") : 11,
    !eq(mx, "M4") : 14,
    !eq(mx, "M8") : 20
  );

  int cycles = !cond(
    !eq(mx, "M2") : !if(isUp, 10, 11),
    !eq(mx, "M4") : !if(isUp, 12, 14),
    !eq(mx, "M8") : !if(isUp, 16, 20)
  );
}

class SiFiveP600VPermutationComplex<string mx> {
  int c = !cond(
    !eq(mx, "M2") : 12,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 24
  );
}

class SiFiveP600VSHA2MSCycles<string mx, int sew> {
  int c = !cond(
    !eq(mx, "M2") : !if(!eq(sew, 32), 2, 3),
    !eq(mx, "M4") : !if(!eq(sew, 32), 4, 6),
    !eq(mx, "M8") : !if(!eq(sew, 32), 8, 12),
    true: 1
  );
}

// SiFiveP600 machine model for scheduling and other instruction cost heuristics.
def SiFiveP600Model : SchedMachineModel {
  let IssueWidth = 4;         // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 160; // Max micro-ops that can be buffered.
  let LoadLatency = 4;        // Cycles for loads to access the cache.
  let MispredictPenalty = 9;  // Extra cycles for a mispredicted branch.
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr,
                             HasVendorXSfvqmaccqoq];
  let CompleteModel = false;
}

let SchedModel = SiFiveP600Model in {

def SiFiveP600IEXQ0       : ProcResource<1>;
def SiFiveP600IEXQ1       : ProcResource<1>;
def SiFiveP600IEXQ2       : ProcResource<1>;
def SiFiveP600IEXQ3       : ProcResource<1>;
def SiFiveP600FEXQ0       : ProcResource<1>;
def SiFiveP600FEXQ1       : ProcResource<1>;

// Two Load/Store ports that can issue either two loads, two stores, or one load
// and one store (P550 has one load and one separate store pipe).
def SiFiveP600LDST       : ProcResource<2>;

// 4-wide pipeline with 4 ALU pipes.
def SiFiveP600IntArith    : ProcResGroup<[SiFiveP600IEXQ0, SiFiveP600IEXQ1, SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
defvar SiFiveP600SYS      = SiFiveP600IEXQ0;
defvar SiFiveP600CMOV     = SiFiveP600IEXQ0;
defvar SiFiveP600MulI2F   = SiFiveP600IEXQ1;
def SiFiveP600Branch      : ProcResGroup<[SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
def SiFiveP600Div         : ProcResource<1>;

def SiFiveP600FloatArith  : ProcResGroup<[SiFiveP600FEXQ0, SiFiveP600FEXQ1]>;
defvar SiFiveP600F2I      = SiFiveP600FEXQ0;
def SiFiveP600FloatDiv    : ProcResource<1>;

// Vector pipeline
// VEXQ0 handle Mask, Simple Slide instructions,
// VEXQ1 handle Complex Slide, Permutation, Reductions, Divide instructions.
// Other vector instructions can be done in VEXQ0 and VEXQ1.
def SiFiveP600VEXQ0        : ProcResource<1>;
def SiFiveP600VEXQ1        : ProcResource<1>;
def SiFiveP600VectorArith  : ProcResGroup<[SiFiveP600VEXQ0, SiFiveP600VEXQ1]>;

// Only VEXQ0 has mask unit.
defvar SiFiveP600VectorMask = SiFiveP600VEXQ0;
// Only VEXQ0 has vector crypto.
defvar SiFiveP600VectorCrypto = SiFiveP600VEXQ0;

def SiFiveP600VLD          : ProcResource<1>;
def SiFiveP600VST          : ProcResource<1>;
def SiFiveP600VDiv         : ProcResource<1>;
def SiFiveP600VFloatDiv    : ProcResource<1>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU, [SiFiveP600IntArith]>;
def : WriteRes<WriteIALU32, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftImm, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftImm32, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftReg, [SiFiveP600IntArith]>;
def : WriteRes<WriteShiftReg32, [SiFiveP600IntArith]>;
// Branching
def : WriteRes<WriteJmp, [SiFiveP600Branch]>;
def : WriteRes<WriteJal, [SiFiveP600Branch]>;
def : WriteRes<WriteJalr, [SiFiveP600Branch]>;

// CMOV
def P600WriteCMOV : SchedWriteRes<[SiFiveP600Branch, SiFiveP600CMOV]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[P600WriteCMOV], (instrs PseudoCCMOVGPRNoX0)>;

let Latency = 2 in {
// Integer multiplication
def : WriteRes<WriteIMul, [SiFiveP600MulI2F]>;
def : WriteRes<WriteIMul32, [SiFiveP600MulI2F]>;
// cpop[w] look exactly like multiply.
def : WriteRes<WriteCPOP, [SiFiveP600MulI2F]>;
def : WriteRes<WriteCPOP32, [SiFiveP600MulI2F]>;
}

// Integer division
def : WriteRes<WriteIDiv, [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIDiv32,  [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Integer remainder
def : WriteRes<WriteIRem, [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 35;
  let ReleaseAtCycles = [1, 34];
}
def : WriteRes<WriteIRem32, [SiFiveP600MulI2F, SiFiveP600Div]> {
  let Latency = 20;
  let ReleaseAtCycles = [1, 19];
}

// Bitmanip
def : WriteRes<WriteRotateImm, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateImm32, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateReg, [SiFiveP600IntArith]>;
def : WriteRes<WriteRotateReg32, [SiFiveP600IntArith]>;

def : WriteRes<WriteCLZ, [SiFiveP600IntArith]>;
def : WriteRes<WriteCLZ32, [SiFiveP600IntArith]>;
def : WriteRes<WriteCTZ, [SiFiveP600IntArith]>;
def : WriteRes<WriteCTZ32, [SiFiveP600IntArith]>;

def : WriteRes<WriteORCB, [SiFiveP600IntArith]>;
def : WriteRes<WriteIMinMax, [SiFiveP600IntArith]>;

def : WriteRes<WriteREV8, [SiFiveP600IntArith]>;

def : WriteRes<WriteSHXADD, [SiFiveP600IntArith]>;
def : WriteRes<WriteSHXADD32, [SiFiveP600IntArith]>;

def : WriteRes<WriteSingleBit, [SiFiveP600IntArith]>;
def : WriteRes<WriteSingleBitImm, [SiFiveP600IntArith]>;
def : WriteRes<WriteBEXT, [SiFiveP600IntArith]>;
def : WriteRes<WriteBEXTI, [SiFiveP600IntArith]>;

// Memory
def : WriteRes<WriteSTB, [SiFiveP600LDST]>;
def : WriteRes<WriteSTH, [SiFiveP600LDST]>;
def : WriteRes<WriteSTW, [SiFiveP600LDST]>;
def : WriteRes<WriteSTD, [SiFiveP600LDST]>;
def : WriteRes<WriteFST16, [SiFiveP600LDST]>;
def : WriteRes<WriteFST32, [SiFiveP600LDST]>;
def : WriteRes<WriteFST64, [SiFiveP600LDST]>;

let Latency = 4 in {
def : WriteRes<WriteLDB, [SiFiveP600LDST]>;
def : WriteRes<WriteLDH, [SiFiveP600LDST]>;
}
let Latency = 4 in {
def : WriteRes<WriteLDW, [SiFiveP600LDST]>;
def : WriteRes<WriteLDD, [SiFiveP600LDST]>;
}

let Latency = 5 in {
def : WriteRes<WriteFLD16, [SiFiveP600LDST]>;
def : WriteRes<WriteFLD32, [SiFiveP600LDST]>;
def : WriteRes<WriteFLD64, [SiFiveP600LDST]>;
}

// Atomic memory
let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicSTD, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicD, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicLDW, [SiFiveP600LDST]>;
def : WriteRes<WriteAtomicLDD, [SiFiveP600LDST]>;
}

// Floating point
let Latency = 2 in {
def : WriteRes<WriteFAdd16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFAdd32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFAdd64, [SiFiveP600FloatArith]>;
}
let Latency = 3 in {
def : WriteRes<WriteFMul16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMul32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMul64, [SiFiveP600FloatArith]>;
}
let Latency = 4 in {
def : WriteRes<WriteFMA16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMA32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMA64, [SiFiveP600FloatArith]>;
}

let Latency = 2 in {
def : WriteRes<WriteFSGNJ16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFSGNJ32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFSGNJ64, [SiFiveP600FloatArith]>;

def : WriteRes<WriteFMinMax16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMinMax32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFMinMax64, [SiFiveP600FloatArith]>;
}

// Half precision.
def : WriteRes<WriteFDiv16, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 4;
  let ReleaseAtCycles = [1, 4];
}
def : WriteRes<WriteFSqrt16, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Single precision.
def : WriteRes<WriteFDiv32, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 6;
  let ReleaseAtCycles = [1, 6];
}
def : WriteRes<WriteFSqrt32, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 18;
  let ReleaseAtCycles = [1, 17];
}

// Double precision
def : WriteRes<WriteFDiv64, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 11;
  let ReleaseAtCycles = [1, 11];
}
def : WriteRes<WriteFSqrt64, [SiFiveP600FEXQ1, SiFiveP600FloatDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [1, 32];
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI32ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI32ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtI64ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFCvtF16ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF16ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF16ToF32, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF16ToF64, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF32ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF32ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF32ToF16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF32ToF64, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF64ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF64ToI64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCvtF64ToF16, [SiFiveP600FloatArith]>;
def : WriteRes<WriteFCvtF64ToF32, [SiFiveP600FloatArith]>;

def : WriteRes<WriteFClass16, [SiFiveP600F2I]>;
def : WriteRes<WriteFClass32, [SiFiveP600F2I]>;
def : WriteRes<WriteFClass64, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp16, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp32, [SiFiveP600F2I]>;
def : WriteRes<WriteFCmp64, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI16ToF16, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF16ToI16, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI32ToF32, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF32ToI32, [SiFiveP600F2I]>;
def : WriteRes<WriteFMovI64ToF64, [SiFiveP600MulI2F]>;
def : WriteRes<WriteFMovF64ToI64, [SiFiveP600F2I]>;
}

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [SiFiveP600SYS]>;
def : WriteRes<WriteVSETIVLI, [SiFiveP600SYS]>;
def : WriteRes<WriteVSETVL, [SiFiveP600SYS]>;

// 7. Vector Loads and Stores
// FIXME: This unit is still being improved, currently
// it is based on stage numbers. Estimates are optimistic,
// latency may be longer.
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",    [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [SiFiveP600VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDS8",   [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",  [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",  [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",  [SiFiveP600VLD], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVLDUX8",  [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX16", [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX32", [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX64", [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX8",  [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX16", [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX32", [SiFiveP600VLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX64", [SiFiveP600VLD], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 8, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTE",    [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",    [SiFiveP600VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTS8",   [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",  [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",  [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",  [SiFiveP600VST], mx, IsWorstCase>;
  }
  let Latency = 12, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSTUX8",  [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX16", [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX32", [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX64", [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX8",  [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX16", [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX32", [SiFiveP600VST], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX64", [SiFiveP600VST], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar LMulLat = SiFiveP600GetCyclesSegmented<mx, eew, nf>.c;
      defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
      let Latency = !add(12, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [SiFiveP600VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [SiFiveP600VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [SiFiveP600VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [SiFiveP600VLD], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [SiFiveP600VLD], mx, IsWorstCase>;
      }
      let Latency = !add(1, LMulLat), ReleaseAtCycles = [!add(12, LMulLat)] in {
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [SiFiveP600VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [SiFiveP600VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [SiFiveP600VST], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [SiFiveP600VST], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  let Latency = 8, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SiFiveP600VLD]>;
    def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SiFiveP600VST]>;
  }
  let Latency = 2, ReleaseAtCycles = [LMul] in {
    def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [SiFiveP600VectorArith]>;
  }
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVExtV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",   [SiFiveP600VectorMask],  mx, IsWorstCase>;
  }
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 6 in {
      defm "" : LMULWriteResMX<"WriteVIMulV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddV", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMulAddX", [SiFiveP600VectorArith], mx, IsWorstCase>;
    }

    let Latency = !mul(2, SiFiveP600RVVMultiplier<mx>.c) in {
      defm "" : LMULWriteResMX<"WriteVIALUV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIALUI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUI",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMV",  [SiFiveP600VectorMask],  mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMX",  [SiFiveP600VectorMask],  mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVICALUMI",  [SiFiveP600VectorMask],  mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeX",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMergeI",  [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftI",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVShiftX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SiFiveP600VectorArith], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVIMovV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    }
  }
}
// Widening
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SiFiveP600VectorArith], mx, IsWorstCase>;

    // Special case for variants with widen operands.
    let ReleaseAtCycles = [!mul(LMulLat, 2)] in
    def P600WriteVIWALUWidenOp_ # mx : SchedWriteRes<[SiFiveP600VectorArith]>;
  }

  defvar P600VIWALUBaseSchedRW = [!cast<SchedWrite>("P600WriteVIWALUWidenOp_" # mx),
                                  !cast<SchedRead>("ReadVPassthru_" # mx),
                                  !cast<SchedRead>("ReadVIALUV_" # mx),
                                  !cast<SchedRead>("ReadVIALUV_" # mx)];

  def : InstRW<P600VIWALUBaseSchedRW,
               (instregex "^PseudoVW(ADD|SUB)[U]?_W(V|X)_" # mx # "$")>;
  def : InstRW<P600VIWALUBaseSchedRW[0,2,3],
               (instregex "^PseudoVW(ADD|SUB)[U]?_WV_" # mx # "_TIED$")>;

  def : InstRW<!listconcat(P600VIWALUBaseSchedRW, [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVW(ADD|SUB)[U]?_W(V|X)_" # mx # "_MASK$")>;
  def : InstRW<!listconcat(P600VIWALUBaseSchedRW[0,1,3], [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVW(ADD|SUB)[U]?_WV_" # mx # "_MASK_TIED$")>;
}

// Worst case needs 51/45/42/72 * lmul cycles for i8/16/32/64.
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 8): 51, !eq(sew, 16): 45, !eq(sew, 32): 42,
      /* SEW=64 */ true: 72);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SiFiveP600VEXQ1, SiFiveP600VDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SiFiveP600VEXQ1, SiFiveP600VDiv], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulV",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = !mul(2, SiFiveP600RVVMultiplier<mx>.c), ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
    }
    let Latency = !if(!eq(mx, "M8"), 4, 3), ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = !if(!eq(mx, "M8"), 4, 3), ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in
  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SiFiveP600VectorArith], mx, IsWorstCase>;
  let Latency = 2, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",  [SiFiveP600VectorMask], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",  [SiFiveP600VectorMask], mx, IsWorstCase>;
  }
  let Latency = !mul(2, SiFiveP600RVVMultiplier<mx>.c),
      ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in {
    defm "" : LMULWriteResMX<"WriteVFClassV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",  [SiFiveP600VectorArith], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",    [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
    let Latency = 3, ReleaseAtCycles = [LMulLat] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SiFiveP600VectorArith], mx, IsWorstCase>;
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 6, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;

      // Special case for variants with widen operands.
      let ReleaseAtCycles = [!mul(LMulLat, 2)] in
      def P600WriteVFWALUWidenOp_ # mx # _E # sew : SchedWriteRes<[SiFiveP600VectorArith]>;
    }

    defvar P600VFWALUBaseSchedRW = [!cast<SchedWrite>("P600WriteVFWALUWidenOp_" # mx # "_E" # sew),
                                    !cast<SchedRead>("ReadVPassthru_" # mx # "_E" # sew),
                                    !cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew)];

    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew)]),
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "$")>;
    def : InstRW<[P600VFWALUBaseSchedRW[0], P600VFWALUBaseSchedRW[2], !cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew)],
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "_TIED$")>;

    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUF_" # mx # "_E" # sew)]),
                 (instregex "^PseudoVFW(ADD|SUB)_WFPR" # sew # "_" # mx # "_E" # sew # "$")>;

    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew), !cast<SchedRead>("ReadVMask")]),
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "_MASK$")>;
    def : InstRW<[P600VFWALUBaseSchedRW[0], P600VFWALUBaseSchedRW[1], !cast<SchedRead>("ReadVFWALUV_" # mx # "_E" # sew), !cast<SchedRead>("ReadVMask")],
                 (instregex "^PseudoVFW(ADD|SUB)_WV_" # mx # "_E" # sew # "_MASK_TIED$")>;

    def : InstRW<!listconcat(P600VFWALUBaseSchedRW, [!cast<SchedRead>("ReadVFWALUF_" # mx # "_E" # sew), !cast<SchedRead>("ReadVMask")]),
                 (instregex "^PseudoVFW(ADD|SUB)_WFPR" # sew # "_" # mx # "_E" # sew # "_MASK$")>;
  }
}
// Narrowing
foreach mx = SchedMxListW in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    let Latency = 3, ReleaseAtCycles = [!if(!eq(LMulLat, 1), 2, LMulLat)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SiFiveP600VectorArith], mx, sew, IsWorstCase>;
    }
  }
}

// Worst case needs around 29/25/37 * LMUL cycles for f16/32/64.
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    defvar DivMicroOpLat =
      !cond(!eq(sew, 16): 29, !eq(sew, 32): 25, /* SEW=64 */ true: 37);
    defvar DivLatency = !mul(DivMicroOpLat, LMulLat);
    let Latency = DivLatency, ReleaseAtCycles = [LMulLat, DivLatency] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SiFiveP600VEXQ1, SiFiveP600VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SiFiveP600VEXQ1, SiFiveP600VFloatDiv], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SiFiveP600VEXQ1, SiFiveP600VFloatDiv], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    let ReleaseAtCycles = [LMulLat] in {
      let Latency = SiFiveP600VIReductionLatency<mx>.c in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;

      let Latency = SiFiveP600VIMinMaxReductionLatency<mx, sew>.c,
          ReleaseAtCycles = [SiFiveP600VIMinMaxReductionCycles<mx, sew>.c] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = SiFiveP600VIReductionLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SiFiveP600VEXQ1],
                                     mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    let Latency = SiFiveP600VFMinMaxReduction<mx, sew>.latency,
        ReleaseAtCycles = [SiFiveP600VFMinMaxReduction<mx, sew>.cycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From",
                                   [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFUnorderedReduction<mx, sew>.latency,
        ReleaseAtCycles = [SiFiveP600VFUnorderedReduction<mx, sew>.cycles] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFOrderedReduction<mx, sew>.c,
        ReleaseAtCycles = [SiFiveP600VFOrderedReduction<mx, sew>.c] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;
    let Latency = SiFiveP600VFWidenUnorderedReduction<mx>.latency,
        ReleaseAtCycles = [6] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;

    let Latency = SiFiveP600VFOrderedReduction<mx, sew>.c,
        ReleaseAtCycles = [SiFiveP600VFOrderedReduction<mx, sew>.c] in
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SiFiveP600VEXQ1],
                                   mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2 in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SiFiveP600VectorMask], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SiFiveP600VectorMask], mx, IsWorstCase>;

    let ReleaseAtCycles = [2] in {
      defm "" : LMULWriteResMX<"WriteVMPopV", [SiFiveP600VectorMask], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVMFFSV", [SiFiveP600VectorMask], mx, IsWorstCase>;
    }
  }
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 2 in
    defm "" : LMULWriteResMX<"WriteVIotaV", [SiFiveP600VectorMask], mx, IsWorstCase>;

    // vid.v isn't executed by the mask unit.
    let Latency = !if(!eq(mx, "M8"), 4, !if(!eq(mx, "M4"), 2, 1)) in
    defm "" : LMULWriteResMX<"WriteVIdxV", [SiFiveP600VectorArith], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Simple Slide
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = SiFiveP600VSlideI<mx>.c in
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SiFiveP600VEXQ0], mx, IsWorstCase>;

    let Latency = SiFiveP600VSlide1<mx>.c in {
      defm "" : LMULWriteResMX<"WriteVISlide1X", [SiFiveP600VEXQ0], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVFSlide1F", [SiFiveP600VEXQ0], mx, IsWorstCase>;
    }
  }
}
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 2, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP600VEXQ0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP600VEXQ0], mx, IsWorstCase>;
  }
}

// Complex Slide
foreach mx = ["M8", "M4", "M2"] in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = SiFiveP600VSlideXComplex<mx>.latency in {
    let ReleaseAtCycles = [SiFiveP600VSlideXComplex<mx, /*isUp=*/true>.cycles] in
    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SiFiveP600VEXQ1], mx, IsWorstCase>;
    let ReleaseAtCycles = [SiFiveP600VSlideXComplex<mx, /*isUp=*/false>.cycles] in
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

let Latency = 2, ReleaseAtCycles = [2] in {
  def : WriteRes<WriteVMovXS, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovSX, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovFS, [SiFiveP600VectorArith]>;
  def : WriteRes<WriteVMovSF, [SiFiveP600VectorArith]>;
}

// Simple Gather and Compress
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [1] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = 3, ReleaseAtCycles = [1] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 6, ReleaseAtCycles = [SiFiveP600VPermutationComplex<mx>.c] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

foreach mx = ["M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
    defvar IsWorstCase = SiFiveP600IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = 6 in {
      let ReleaseAtCycles = [SiFiveP600VPermutationComplex<mx>.c] in {
        defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
        defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
      }

      let ReleaseAtCycles = [!add(SiFiveP600VPermutationComplex<mx>.c, 1)] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SiFiveP600VEXQ1], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Vrgather.vi
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SiFiveP600VEXQ1], mx, IsWorstCase>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar LMulLat = SiFiveP600GetLMulCycles<mx>.c;
  defvar IsWorstCase = SiFiveP600IsWorstCaseMX<mx, SchedMxList>.c;
  // Zvbb
  let ReleaseAtCycles = [LMulLat] in {
    let Latency = 2 in {
      // FIXME: Exegesis was not able to measure the latency of these instructions.
      // We probably should update them at some point.
      defm "" : LMULWriteResMX<"WriteVCPOPV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLX",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVWSLLI",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    }

    let Latency = SiFiveP600VCryptoLatency<mx>.c in {
      defm "" : LMULWriteResMX<"WriteVBREVV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVCLZV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVCTZV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;

      def P600WriteVANDN_ # mx : SchedWriteRes<[SiFiveP600VectorCrypto]>;
    }
  }

  // Special case for VANDN -- we execute it on vector crypto unit.
  defvar P600VANDNBaseSchedRW = [!cast<SchedWrite>("P600WriteVANDN_" # mx),
                                 // VANDN always merge read operand.
                                 !cast<SchedRead>("ReadVPassthru_" # mx),
                                 !cast<SchedRead>("ReadVIALUV_" # mx),
                                 !cast<SchedRead>("ReadVIALUV_" # mx)];
  def : InstRW<P600VANDNBaseSchedRW,
               (instregex "^PseudoVANDN_V(V|X)_" # mx # "$")>;
  def : InstRW<!listconcat(P600VANDNBaseSchedRW, [!cast<SchedRead>("ReadVMask")]),
               (instregex "^PseudoVANDN_V(V|X)_" # mx # "_MASK$")>;

  // Zvbc
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVCLMULV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLMULX", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvkb
  // VANDN uses WriteVIALU[V|X|I]
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVBREV8V",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVREV8V",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotV",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotX",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotI",    [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvkg
  let Latency = SiFiveP600VCryptoLatency<mx>.c, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVGHSHV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVGMULV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // ZvknhaOrZvknhb
  // FIXME: The latency is probably wrong.
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSHA2CHV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSHA2CLV", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defvar ZvknhSEWs = !listremove(SchedSEWSet<mx>.val, [8, 16]);
    // Largest SEW is the last element, assuming SchedSEWSet is sorted in ascending
    // order.
    defvar LargestZvknhSEW = !foldl(!head(ZvknhSEWs), ZvknhSEWs, last, curr, curr);
    foreach sew = ZvknhSEWs in {
      // The worst case for Zvknh[ab] is designated to the largest SEW and LMUL.
      defvar IsWorstCaseVSHA2MSV = !and(IsWorstCase, !eq(sew, LargestZvknhSEW));
      let ReleaseAtCycles = [SiFiveP600VSHA2MSCycles<mx, sew>.c] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVSHA2MSV", [SiFiveP600VectorCrypto], mx, sew,
                                     IsWorstCaseVSHA2MSV>;
    }
  }
  // Zvkned
  let Latency = 2 in {
    let ReleaseAtCycles = [LMulLat] in {
      defm "" : LMULWriteResMX<"WriteVAESMVV",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAESKF1V", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVAESKF2V", [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    }

    let ReleaseAtCycles = [!if(!lt(LMulLat, 2), LMulLat, !div(LMulLat, 2))] in
    defm "" : LMULWriteResMX<"WriteVAESZV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
  // Zvksed
  let Latency = 3, ReleaseAtCycles = [SiFiveP600VSM3CCycles<mx>.c] in
  defm "" : LMULWriteResMX<"WriteVSM3CV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  let Latency = 6, ReleaseAtCycles = [LMulLat] in
  defm "" : LMULWriteResMX<"WriteVSM3MEV",  [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  let Latency = 3, ReleaseAtCycles = [LMulLat] in {
    defm "" : LMULWriteResMX<"WriteVSM4KV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSM4RV",   [SiFiveP600VectorCrypto], mx, IsWorstCase>;
  }
}

// Others
def : WriteRes<WriteCSR, [SiFiveP600SYS]>;
def : WriteRes<WriteNop, []>;
def : WriteRes<WriteRdVLENB, [SiFiveP600SYS]>;

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
