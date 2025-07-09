//=- RISCVSchedSpacemitX60.td - Spacemit X60 Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Scheduler model for the SpacemiT-X60 processor based on documentation of the
// C908 and experiments on real hardware (bpi-f3).
//
//===----------------------------------------------------------------------===//

class SMX60IsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SMX60IsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// 1 Micro-Op per cycle.
class SMX60GetLMulCycles<string mx> {
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

// Basic scaling pattern (4,4,4,4,4,8,16): doubles at higher LMULs
// Used for: logical ops, shifts, sign ext, merge/move, FP sign/recip/convert, mask ops, slides
class SMX60GetBasicLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 16,
    !eq(mx, "MF2") : 4,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 4
  );
}

// Arithmetic scaling pattern (4,4,4,4,4,5,8): minimal increase at M4
// Used for: arithmetic (add/sub/min/max), saturating/averaging, FP add/sub/min/max
class SMX60GetArithmeticLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 4,
    !eq(mx, "M4") : 5,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 4,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 4
  );
}

// Progressive scaling pattern (4,4,4,4,6,10,18): gradual increase from M2
// Used for: mask-producing comparisons, carry ops with mask, FP comparisons
class SMX60GetProgressiveLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 10,
    !eq(mx, "M8") : 18,
    !eq(mx, "MF2") : 4,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 4
  );
}

// Widening scaling pattern (4,4,4,4,5,8,8): plateaus at higher LMULs
// Used for: widening operations
class SMX60GetWideningLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 5,
    !eq(mx, "M4") : 8,
    !eq(mx, "M8") : 8, // M8 not supported for most widening, fallback
    !eq(mx, "MF2") : 4,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 4
  );
}

// Complex FP scaling pattern (6,6,6,6,6,7,8): minimal increase
// Used for: FP FMA operations, complex FP ops
class SMX60GetComplexFPLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 6,
    !eq(mx, "M2") : 6,
    !eq(mx, "M4") : 7,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 6,
    !eq(mx, "MF4") : 6,
    !eq(mx, "MF8") : 6
  );
}

// FP reduction scaling pattern (12,12,12,15,21,33,57): progressive from M1
// Used for: FP reductions
class GetFPReductionLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 15,
    !eq(mx, "M2") : 21,
    !eq(mx, "M4") : 33,
    !eq(mx, "M8") : 57,
    !eq(mx, "MF2") : 12,
    !eq(mx, "MF4") : 12,
    !eq(mx, "MF8") : 12
  );
}

// Reduction scaling pattern (5,5,5,7,11,19,35): progressive from M1
// Used for: integer reductions (fractional=5)
class GetIntReductionLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 7,
    !eq(mx, "M2") : 11,
    !eq(mx, "M4") : 19,
    !eq(mx, "M8") : 35,
    !eq(mx, "MF2") : 5,
    !eq(mx, "MF4") : 5,
    !eq(mx, "MF8") : 5
  );
}

// Flat scaling pattern (5,5,5,5,5,5,8): constant until final jump
// Used for: e32 multiply pattern, some FP ops
class SMX60GetFlatLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 5,
    !eq(mx, "M2") : 5,
    !eq(mx, "M4") : 5,
    !eq(mx, "M8") : 8,
    !eq(mx, "MF2") : 5,
    !eq(mx, "MF4") : 5,
    !eq(mx, "MF8") : 5
  );
}

// Exponential scaling pattern (7,7,7,7,8,16,32): exponential at M4/M8
// Used for: e64 multiply pattern, complex ops
class SMX60GetExponentialLatency<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 7,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 32,
    !eq(mx, "MF2") : 7,
    !eq(mx, "MF4") : 7,
    !eq(mx, "MF8") : 7
  );
}

def SpacemitX60Model : SchedMachineModel {
  let IssueWidth        = 2; // dual-issue
  let MicroOpBufferSize = 0; // in-order
  let LoadLatency       = 3; // worse case: >= 3
  let MispredictPenalty = 9; // nine-stage

  let CompleteModel = 0;

  let UnsupportedFeatures = [HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr];
}

let SchedModel = SpacemitX60Model in {

//===----------------------------------------------------------------------===//
// Define processor resources for Spacemit-X60

// Information gathered from the C908 user manual:
let BufferSize = 0 in {
  // The LSU supports dual issue for scalar store/load instructions
  def SMX60_LS : ProcResource<2>;

  // An IEU can decode and issue two instructions at the same time
  def SMX60_IEUA : ProcResource<1>;
  def SMX60_IEUB : ProcResource<1>;
  def SMX60_IEU : ProcResGroup<[SMX60_IEUA, SMX60_IEUB]>;

  // Although the X60 does appear to support multiple issue for at least some
  // floating point instructions, this model assumes single issue as
  // increasing it reduces the gains we saw in performance
  def SMX60_FP : ProcResource<1>;

  // Vector pipeline
  // Single issue for vector store/load instructions
  def SMX60_VLS : ProcResource<1>;

  def SMX60_VIEU : ProcResource<1>;

  // The C908 user manual says: "The vector execution unit is developed by
  // extending the floating-point unit", so let's assume single issue for now
  def SMX60_VFP : ProcResource<1>;
}

//===----------------------------------------------------------------------===//

// Branching
def : WriteRes<WriteJmp, [SMX60_IEUA]>;
def : WriteRes<WriteJal, [SMX60_IEUA]>;
def : WriteRes<WriteJalr, [SMX60_IEUA]>;

// Integer arithmetic and logic
// Latency of ALU instructions is 1, but add.uw is 2
def : WriteRes<WriteIALU32, [SMX60_IEU]>;
def : WriteRes<WriteIALU, [SMX60_IEU]>;
def : WriteRes<WriteShiftImm32, [SMX60_IEU]>;
def : WriteRes<WriteShiftImm, [SMX60_IEU]>;
def : WriteRes<WriteShiftReg32, [SMX60_IEU]>;
def : WriteRes<WriteShiftReg, [SMX60_IEU]>;

// Integer multiplication
def : WriteRes<WriteIMul32, [SMX60_IEU]>  { let Latency = 3; }

// The latency of mul is 5, while in mulh, mulhsu, mulhu is 6
// Worst case latency is used
def : WriteRes<WriteIMul, [SMX60_IEU]>  { let Latency = 6; }

// Integer division/remainder
// TODO: Latency set based on C908 datasheet and hasn't been
// confirmed experimentally.
let Latency = 12, ReleaseAtCycles = [12] in {
  def : WriteRes<WriteIDiv32, [SMX60_IEUA]>;
  def : WriteRes<WriteIRem32, [SMX60_IEUA]>;
}
let Latency = 20, ReleaseAtCycles = [20] in {
  def : WriteRes<WriteIDiv, [SMX60_IEUA]>;
  def : WriteRes<WriteIRem, [SMX60_IEUA]>;
}

// Bitmanip
def : WriteRes<WriteRotateImm, [SMX60_IEU]>;
def : WriteRes<WriteRotateImm32, [SMX60_IEU]>;
def : WriteRes<WriteRotateReg, [SMX60_IEU]>;
def : WriteRes<WriteRotateReg32, [SMX60_IEU]>;

def : WriteRes<WriteCLZ, [SMX60_IEU]>;
def : WriteRes<WriteCLZ32, [SMX60_IEU]>;
def : WriteRes<WriteCTZ, [SMX60_IEU]>;
def : WriteRes<WriteCTZ32, [SMX60_IEU]>;

let Latency = 2 in {
  def : WriteRes<WriteCPOP, [SMX60_IEU]>;
  def : WriteRes<WriteCPOP32, [SMX60_IEU]>;
}

def : WriteRes<WriteORCB, [SMX60_IEU]>;
def : WriteRes<WriteIMinMax, [SMX60_IEU]>;
def : WriteRes<WriteREV8, [SMX60_IEU]>;

let Latency = 2 in {
  def : WriteRes<WriteSHXADD, [SMX60_IEU]>;
  def : WriteRes<WriteSHXADD32, [SMX60_IEU]>;
  def : WriteRes<WriteCLMUL, [SMX60_IEU]>;
}

// Single-bit instructions
def : WriteRes<WriteSingleBit, [SMX60_IEU]>;
def : WriteRes<WriteSingleBitImm, [SMX60_IEU]>;
def : WriteRes<WriteBEXT, [SMX60_IEU]>;
def : WriteRes<WriteBEXTI, [SMX60_IEU]>;

// Memory/Atomic memory
let Latency = 4 in {
  def : WriteRes<WriteSTB, [SMX60_LS]>;
  def : WriteRes<WriteSTH, [SMX60_LS]>;
  def : WriteRes<WriteSTW, [SMX60_LS]>;
  def : WriteRes<WriteSTD, [SMX60_LS]>;
  def : WriteRes<WriteFST16, [SMX60_LS]>;
  def : WriteRes<WriteFST32, [SMX60_LS]>;
  def : WriteRes<WriteFST64, [SMX60_LS]>;

  def : WriteRes<WriteLDB, [SMX60_LS]>;
  def : WriteRes<WriteLDH, [SMX60_LS]>;
  def : WriteRes<WriteLDW, [SMX60_LS]>;
  def : WriteRes<WriteLDD, [SMX60_LS]>;
  def : WriteRes<WriteFLD16, [SMX60_LS]>;
  def : WriteRes<WriteFLD32, [SMX60_LS]>;
  def : WriteRes<WriteFLD64, [SMX60_LS]>;
}

// Atomics
let Latency = 8 in {
  def : WriteRes<WriteAtomicSTW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicSTD, [SMX60_LS]>;
  def : WriteRes<WriteAtomicLDW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicLDD, [SMX60_LS]>;
}

let Latency = 12 in {
  def : WriteRes<WriteAtomicW, [SMX60_LS]>;
  def : WriteRes<WriteAtomicD, [SMX60_LS]>;
}

// Floating point units Half precision
let Latency = 4 in {
  def : WriteRes<WriteFAdd16, [SMX60_FP]>;
  def : WriteRes<WriteFMul16, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ16, [SMX60_FP]>;
  def : WriteRes<WriteFMinMax16, [SMX60_FP]>;
}
def : WriteRes<WriteFMA16, [SMX60_FP]> { let Latency = 5; }

let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv16, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt16, [SMX60_FP]>;
}

// Single precision
let Latency = 4 in {
  def : WriteRes<WriteFAdd32, [SMX60_FP]>;
  def : WriteRes<WriteFMul32, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ32, [SMX60_FP]>;
  def : WriteRes<WriteFMinMax32, [SMX60_FP]>;
}
def : WriteRes<WriteFMA32, [SMX60_FP]> { let Latency = 5; }

let Latency = 15, ReleaseAtCycles = [15] in {
  def :  WriteRes<WriteFDiv32, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt32, [SMX60_FP]>;
}

// Double precision
let Latency = 5 in {
  def : WriteRes<WriteFAdd64, [SMX60_FP]>;
  def : WriteRes<WriteFMul64, [SMX60_FP]>;
  def : WriteRes<WriteFSGNJ64, [SMX60_FP]>;
}
def : WriteRes<WriteFMinMax64, [SMX60_FP]> { let Latency = 4; }
def : WriteRes<WriteFMA64, [SMX60_FP]> { let Latency = 6; }

let Latency = 22, ReleaseAtCycles = [22] in {
  def :  WriteRes<WriteFDiv64, [SMX60_FP]>;
  def :  WriteRes<WriteFSqrt64, [SMX60_FP]>;
}

// Conversions
let Latency = 6 in {
  def : WriteRes<WriteFCvtF16ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF32ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF32ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF64ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF64ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF16ToI64, [SMX60_IEU]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFCvtI32ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI32ToF32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI32ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF32, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtI64ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFCvtF16ToF32, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF16ToF64, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF32ToF16, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF32ToF64, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF64ToF16, [SMX60_FP]>;
  def : WriteRes<WriteFCvtF64ToF32, [SMX60_FP]>;
}

let Latency = 6 in {
  def : WriteRes<WriteFClass16, [SMX60_FP]>;
  def : WriteRes<WriteFClass32, [SMX60_FP]>;
  def : WriteRes<WriteFClass64, [SMX60_FP]>;

  def : WriteRes<WriteFCmp16, [SMX60_FP]>;
  def : WriteRes<WriteFCmp32, [SMX60_FP]>;
  def : WriteRes<WriteFCmp64, [SMX60_FP]>;

  def : WriteRes<WriteFMovF32ToI32, [SMX60_IEU]>;
  def : WriteRes<WriteFMovF16ToI16, [SMX60_IEU]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFMovI16ToF16, [SMX60_IEU]>;
  def : WriteRes<WriteFMovF64ToI64, [SMX60_IEU]>;
  def : WriteRes<WriteFMovI64ToF64, [SMX60_IEU]>;
  def : WriteRes<WriteFMovI32ToF32, [SMX60_IEU]>;
}

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [SMX60_IEUA]>;
def : WriteRes<WriteVSETIVLI, [SMX60_IEUA]>;
def : WriteRes<WriteVSETVL, [SMX60_IEUA]>;

// 7. Vector Loads and Stores
// TODO: These latencies are estimations and are not confirmed experimentally
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  // Unit-stride loads and stores
  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVLDE",  [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF", [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTE",  [SMX60_VLS], mx, IsWorstCase>;
  }

  // Mask loads and stores
  let Latency = 3 in {
    defm "" : LMULWriteResMX<"WriteVLDM",  [SMX60_VLS], mx, IsWorstCase=!eq(mx, "M1")>;
    defm "" : LMULWriteResMX<"WriteVSTM",  [SMX60_VLS], mx, IsWorstCase=!eq(mx, "M1")>;
  }

  // Strided and indexed loads and stores: scale with both LMUL and EEW
  foreach eew = [8, 16, 32, 64] in {
    defvar EEWMultiplier = !div(eew, 8);
    defvar StridedLatency = !mul(SMX60GetArithmeticLatency<mx>.c, EEWMultiplier);
    let Latency = StridedLatency in {
      defm "" : LMULWriteResMX<"WriteVLDS"  # eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDUX" # eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX" # eew, [SMX60_VLS], mx, IsWorstCase>;

      defm "" : LMULWriteResMX<"WriteVSTS"  # eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTUX" # eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX" # eew, [SMX60_VLS], mx, IsWorstCase>;
    }
  }
}

// Segmented loads and stores: base latency multiplied by number of fields
// TODO: These latencies are estimations and are not confirmed experimentally
foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;
      defvar EEWMultiplier = !div(eew, 8);
      defvar StridedLatency = !mul(SMX60GetArithmeticLatency<mx>.c, EEWMultiplier);

      // Unit-stride segmented
      let Latency = !mul(SMX60GetBasicLatency<mx>.c, nf) in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" #eew,   [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" #eew,   [SMX60_VLS], mx, IsWorstCase>;
      }

      // Strided/indexed segmented
      let Latency = !mul(StridedLatency, nf) in {
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" #eew,  [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;

        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" #eew,  [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
// TODO: These latencies are estimations and are not confirmed experimentally
foreach LMul = [1, 2, 4, 8] in {
  let Latency = SMX60GetProgressiveLatency<!strconcat("M", !cast<string>(LMul))>.c in {
    def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SMX60_VLS]>;
    def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SMX60_VLS]>;
  }
}

let Latency = 4 in {
  def : WriteRes<WriteVMovSX, [SMX60_VIEU]>;
  def : WriteRes<WriteVMovFS, [SMX60_VIEU]>;
  def : WriteRes<WriteVMovSF, [SMX60_VIEU]>;
  def : WriteRes<WriteVMov1V, [SMX60_VIEU]>;
  def : WriteRes<WriteVMov2V, [SMX60_VIEU]>;
}
def : WriteRes<WriteVMovXS, [SMX60_VIEU]> { let Latency = 6; }
def : WriteRes<WriteVMov4V, [SMX60_VIEU]> { let Latency = 8; }
def : WriteRes<WriteVMov8V, [SMX60_VIEU]> { let Latency = 16; }

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = SMX60GetArithmeticLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = SMX60GetBasicLatency<mx>.c in {
    // Pattern of vadd, vsub, vrsub: 4/4/5/8
    // Pattern of vand, vor, vxor:   4/4/8/16
    // They are grouped together, so we used the worst case 4/4/5/16
    // TODO: use InstRW to override individual instructions' scheduling data
    defm "" : LMULWriteResMX<"WriteVIALUV",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",    [SMX60_VIEU], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVExtV",     [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftV",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",   [SMX60_VIEU], mx, IsWorstCase>;

    // Pattern of vadc, vsbc: 4/4/4/4/4/8/16, except for e8m8 = 9. We set e8m8 to 16
    defm "" : LMULWriteResMX<"WriteVICALUV",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",   [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = SMX60GetProgressiveLatency<mx>.c in {
    // Pattern of vmadc, vmsbc, vmseq, etc: 4/4/4/4/6/10/18, except for e8m8 = 17
    // We set e8m8 to 18
    defm "" : LMULWriteResMX<"WriteVICALUMV",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMX",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMI",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",    [SMX60_VIEU], mx, IsWorstCase>;
  }

  // Pattern of vmacc, vmadd, vmul, vmulh, etc.: e8/e16 = 4/4/5/8, e32 = 5,5,5,8,
  // e64 = 7,8,16,32. We use the worst-case until we can split the SEW.
  // TODO: change WriteVIMulV, etc to be defined with LMULSEWSchedWrites
  let Latency = SMX60GetExponentialLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVIMulV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// Widening
// Pattern of vwmul, vwmacc, etc: e8/e16 = 4/4/5/8, e32 = 5,5,5,8
// We use the worst-case for all.
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  let Latency = SMX60GetWideningLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// Division and remainder operations
// Pattern of vdivu: 11/11/11/20/40/80/160
// Pattern of vdiv: 12/12/12/22/44/88/176
// Pattern of vremu: 12/12/12/22/44/88/176
// Pattern of vrem: 13/13/13/24/48/96/192
// We use the worst-case for all: 24/24/24/24/48/96/192
// TODO: Create separate WriteVIRem to more closely match the latencies
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    // NumDLEN = 2 * LMUL (since DLEN = VLEN/2)
    defvar NumDLEN = !mul(2, SMX60GetLMulCycles<mx>.c);

    let Latency = !mul(NumDLEN, 12) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SMX60_VIEU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SMX60_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

// Simple Narrowing Shift and Clips
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// Complex Narrowing Shift and Clips
foreach mx = ["M2", "M4"] in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  let Latency = !mul(SMX60GetBasicLatency<mx>.c, 2) in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = SMX60GetArithmeticLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [SMX60_VIEU], mx, IsWorstCase>;
  }

  // Pattern of vsmul: e8/e16 = 4/4/5/8, e32 = 5,5,5,8, e64 = 7,8,16,32
  // We use the worst-case until we can split the SEW.
  // TODO: change WriteVSMulV/X to be defined with LMULSEWSchedWrites
  let Latency = SMX60GetExponentialLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVSMulV",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions

// Simple Vector Floating-Point Instructions
foreach mx = ["MF4", "MF2"] in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    let Latency = 4 in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    let Latency = 5 in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Vector Floating-Point Instructions
foreach mx = ["M1", "M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defvar ArithLat = SMX60GetArithmeticLatency<mx>.c;
    let Latency = ArithLat in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    defvar FlatLat = SMX60GetFlatLatency<mx>.c;
    let Latency = !if(!eq(sew, 64), FlatLat, ArithLat) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF",  [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    // TODO: for some reason, the following cond is not working, and always use FlatLat
    let Latency = !if(!eq(sew, 64), SMX60GetComplexFPLatency<mx>.c, FlatLat) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    let Latency = SMX60GetBasicLatency<mx>.c in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = SMX60GetProgressiveLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVFCmpV",  [SMX60_VFP], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",  [SMX60_VFP], mx, IsWorstCase>;
  }

  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVFClassV",  [SMX60_VFP], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",  [SMX60_VFP], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",    [SMX60_VFP], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
  }
}

// Widening conversion operations use 4 * LMUL cycles
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;

    let Latency = !mul(SMX60GetLMulCycles<mx>.c, 4) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListFW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListFW>.c;

  let Latency = !mul(SMX60GetLMulCycles<mx>.c, 4) in {
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    // Pattern for vfwsub/vfwadd.vv, vfwsub/vfwadd.vf: e16mf4=4, e16mf2=4, e16m1=4, e16m2=5,
    // e16m4=8, e32mf2=4, e32m1=4, e32m2=5, e32m4=8
    // Pattern for vfwsub/vfwadd.wv, vfwsub/vfwadd.wf: e16mf4=5, e16mf2=5, e16m1=5, e16m2=9,
    // e16m4=17, e32mf2=5, e32m1=5, e32m2=9, e32m4=17
    // TODO: Split .wf/.wv variants into separate scheduling classes to use 5/5/9/17
    defvar LMulLat = SMX60GetLMulCycles<mx>.c;
    let Latency = !mul(LMulLat, 4) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    // Pattern for vfwmul.vv, vfwmul.vf: e16 = 4/4/4/6/8. We use 4/4/5/8 as approximation
    // TODO: e32m4 = 8, but it's set to 5 here
    let Latency = !if(!eq(sew, 32), SMX60GetFlatLatency<mx>.c, SMX60GetWideningLatency<mx>.c) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    // Pattern for vfwmacc, vfwnmacc, etc: e16 = 5/5/5/8; e32 = 6/6/7/8
    // Use existing 6,6,7,8 as close approximation
    let Latency = SMX60GetComplexFPLatency<mx>.c in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    let Latency = !mul(LMulLat, 4) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing conversion operations use 4 * LMUL cycles
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  let Latency = !mul(SMX60GetLMulCycles<mx>.c, 4) in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    let Latency = !mul(SMX60GetLMulCycles<mx>.c, 4) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Simple floating-point division operations
foreach mx = ["MF4", "MF2"] in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    let Latency = 7 in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Complex floating-point division operations with measured latencies
foreach mx = ["M1", "M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    defvar LMulLat = SMX60GetLMulCycles<mx>.c;

    // Pattern for vfdiv.vf: e16 = 12/24/48/96; e32 = 12/24/48/96; e64 = 18/36/72/144
    // Pattern for vfrdiv.vf: e16 = 12/24/48/96; e32 = 12/24/48/96; e64 = 40/80/160/320
    // We use the worst-case, vfdiv.vf is penalized in e64
    // TODO: split vfdiv.vf and vfrdiv.vf into separate scheduling classes
    let Latency = !if(!eq(sew, 64), !mul(LMulLat, 40), !mul(LMulLat, 12)) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [SMX60_VFP], mx, sew, IsWorstCase>;
    }

    // Compute latency based on SEW and LMUL combination
    defvar SEWLatencyFactor = !cond(
      !eq(sew, 16) : 12,  // e16: 12*LMUL
      !eq(sew, 32) : 38,  // e32: 38*LMUL
      !eq(sew, 64) : 40   // e64: 40*LMUL
    );

    let Latency = !mul(LMulLat, SEWLatencyFactor) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Pattern for vfsqrt.v: e16 = 18/36/72/144; e32 = 38/76/152/304; e64 = 40/80/160/320
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Compute latency based on SEW and LMUL combination
    defvar SEWLatencyFactor = !cond(
      !eq(sew, 16) : 18,  // e16: 18*LMUL
      !eq(sew, 32) : 38,  // e32: 38*LMUL
      !eq(sew, 64) : 40   // e64: 40*LMUL
    );

    let Latency = !mul(SMX60GetLMulCycles<mx>.c, SEWLatencyFactor) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    let Latency = GetIntReductionLatency<mx>.c in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;

      // Pattern for vredsum:                  5/5/5/7/11/19/35
      // Pattern for vredand, vredor, vredxor: 4/4/4/6/10/18/34
      // They are grouped together, so we use the worst-case vredsum latency.
      // TODO: split vredand, vredor, vredxor into separate scheduling classe.
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;

    let Latency = GetIntReductionLatency<mx>.c in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;
    // Pattern for vfredmax.vs, vfredmin.vs: 12,12,15,21,33,57
    // Pattern for vfredusum.vs is slightly lower for e16/e32
    // We use the worst-case for simplificity
    let Latency = GetFPReductionLatency<mx>.c in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Simple vfredosum operations
foreach mx = ["MF4", "MF2"] in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Slightly increased latencies for e32mf2=24 (should be 12)
    defvar SimpleLatency = !cond(
      !eq(mx, "MF4") : 12,
      !eq(mx, "MF2") : 24
    );

    let Latency = SimpleLatency in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Complex vfredosum operations
foreach mx = ["M1", "M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Compute latency based on SEW and LMUL combination
    defvar SEWLatencyFactor = !cond(
      !eq(sew, 16) : 48,  // e16: 48*LMUL
      !eq(sew, 32) : 24,  // e32: 24*LMUL
      !eq(sew, 64) : 12   // e64: 12*LMUL
    );

    let Latency = !mul(SMX60GetLMulCycles<mx>.c, SEWLatencyFactor) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Simple vfwredosum.vs and vfwredusum.vs operations
foreach mx = ["MF4", "MF2"] in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;

    // Slightly increased latencies for e32mf2=32 (should be 16)
    defvar SimpleLatency = !cond(
      !eq(mx, "MF4") : 16,
      !eq(mx, "MF2") : 32
    );

    let Latency = SimpleLatency in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// Complex vfwredosum.vs and vfwredusum.vs operations (integer LMULs)
foreach mx = ["M1", "M2", "M4", "M8"] in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;

    // Compute latency based on SEW and LMUL combination
    defvar SEWLatencyFactor = !cond(
      !eq(sew, 16) : 64,  // e16: 64*LMUL
      !eq(sew, 32) : 32   // e32: 32*LMUL
    );

    let Latency = !mul(SMX60GetLMulCycles<mx>.c, SEWLatencyFactor) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [SMX60_VFP], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = 4 in  {
    defm "" : LMULWriteResMX<"WriteVMALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = 6 in  {
    defm "" : LMULWriteResMX<"WriteVMPopV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVIotaV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Slide
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  // Pattern for vslide1down.vx, vslidedown.vi/vx: 4/5/9/17
  // Pattern for vslide1up.vx: 4/4/8/16
  // We use 4/4/8/16 for simplicity
  let Latency = SMX60GetBasicLatency<mx>.c in {
    defm "" : LMULWriteResMX<"WriteVSlideI",  [SMX60_VIEU], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVISlide1X", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F", [SMX60_VFP], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVSlideUpX",   [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// Simple Gather and Compress
foreach mx = ["MF8", "MF4", "MF2", "M1"] in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = 4 in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SMX60_VIEU], mx, IsWorstCase>;

    foreach sew = SchedSEWSet<mx>.val in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SMX60_VIEU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SMX60_VIEU], mx, sew, IsWorstCase>;

      // Slightly reduced latencies for e8mf1=4 (should be 8)
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SMX60_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

// Complex Gather and Compress
foreach mx = ["M2", "M4", "M8"] in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = !mul(SMX60GetLMulCycles<mx>.c, 2) in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SMX60_VIEU], mx, IsWorstCase>;
  }

  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCaseSEW = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defvar BasicLat = SMX60GetBasicLatency<mx>.c;
    defvar ExpLat = !mul(BasicLat, BasicLat);
    let Latency = ExpLat in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SMX60_VIEU], mx, sew, IsWorstCaseSEW>;
    }

    let Latency = !if(!eq(sew, 8), !mul(ExpLat, 2), ExpLat) in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SMX60_VIEU], mx, sew, IsWorstCaseSEW>;
    }

    defvar CompressLat = !cond(
      !eq(mx, "M2") : 10,
      !eq(mx, "M4") : 36,
      true : 136  // M8
    );
    let Latency = CompressLat in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SMX60_VIEU], mx, sew, IsWorstCaseSEW>;
    }
  }
}

// Others
def : WriteRes<WriteCSR, [SMX60_IEU]>;
def : WriteRes<WriteNop, [SMX60_IEU]>;
def : WriteRes<WriteRdVLENB, [SMX60_IEUA]>;

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
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
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
def : ReadAdvance<ReadCLMUL, 0>;
// Single-bit instructions
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

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
}
