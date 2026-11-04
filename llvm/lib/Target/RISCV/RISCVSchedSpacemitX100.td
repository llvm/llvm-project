//- RISCVSchedSpacemitX100.td - Spacemit X100 Scheduling Defs -*- tablegen -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Scheduler model for the SpacemiT-X100 processor.
//
//===----------------------------------------------------------------------===//
// Latency helper classes

// Used for: arithmetic (min/max/vmerge/vmv)
class Get2348Latency<string mx> {
  int c = GetLMULValue<[/*MF8=*/2, /*MF4=*/2, /*MF2=*/2, /*M1=*/2, /*M2=*/3, /*M4=*/4, /*M8=*/8], mx>.c;
}

//===----------------------------------------------------------------------===//

class SMTX100IsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class SMTX100IsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

defvar SMTX100VLEN = 256;
defvar SMTX100DLEN = !div(SMTX100VLEN, 2);

class SMTX100GetLMulCycles<string mx> {
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

class SMTX100GetVLMAX<string mx, int sew> {
  defvar LMUL = SMTX100GetLMulCycles<mx>.c;
  int val = !cond(
    !eq(mx, "MF2") : !div(!div(SMTX100VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(SMTX100VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(SMTX100VLEN, 8), sew),
    true: !div(!mul(SMTX100VLEN, LMUL), sew)
  );
}

// Latency for segmented loads and stores are calculated as vl * nf.
class SMTX100SegmentedLdStCycles<string mx, int sew, int nf> {
  int c = !mul(SMTX100GetVLMAX<mx, sew>.val, nf);
}

class SMTX100GetVLDSLat<string mx, int sew> {
  defvar Base = [4, 5, 9, 22, 50, 114, 242];
  defvar Index = !cond(
    !eq(mx, "MF8"): 0,
    !eq(mx, "MF4"): 1,
    !eq(mx, "MF2"): 2,
    !eq(mx, "M1"):  3,
    !eq(mx, "M2"):  4,
    !eq(mx, "M4"):  5,
    !eq(mx, "M8"):  6,
  );
  defvar Shift = !cond(
    !eq(sew, 8):  0,
    !eq(sew, 16): 1,
    !eq(sew, 32): 2,
    !eq(sew, 64): 3,
  );
  int val = !if(!lt(Index, Shift), 0, Base[!sub(Index, Shift)]);
}

def SpacemitX100Model : SchedMachineModel {
  let IssueWidth = 4;          // 4 micro-ops are dispatched per cycle.
  let MicroOpBufferSize = 192; // Max micro-ops that can be buffered.
                               // 64 entry ROB. Max 3 micro-ops share one entry.
  let LoadLatency = 3;         // Cycles for loads to access the cache.
  let MispredictPenalty = 9;   // Extra cycles for a mispredicted branch.

  let CompleteModel = 0;

  let UnsupportedFeatures = [HasStdExtZknd, HasStdExtZkne, HasStdExtZknh,
                             HasStdExtZksed, HasStdExtZksh, HasStdExtZkr];
}

let SchedModel = SpacemitX100Model in {

//===----------------------------------------------------------------------===//
// Define processor resources for Spacemit-X100
let BufferSize = 6 in {
  // IQ0, IQ1, BQ: 12 entry queue, can accept 2 ops and issue 1 op per cycle
  // To model the accept bandwidth, split into 2 sub-queues of 6 entry each
  def SMTX100_IQ0 : ProcResource<2>; //Integer Queue 0
  def SMTX100_IQ1 : ProcResource<2>; //Integer Queue 1
  def SMTX100_BQ  : ProcResource<2>; //Branch Queue
}

let BufferSize = 4 in {
  // LSQ: 16 entry queue, can accept 4 ops and issue 2 op per cycle
  // To model the accept bandwidth, split into 4 sub-queues of 4 entry each
  def SMTX100_LSQ : ProcResource<4>; //Load Store Queue
  // FQ0, FQ1: 8 entry queue, can accept 2 ops and issue 1 op per cycle
  // To model the accept bandwidth, split into 2 sub-queues of 4 entry each
  def SMTX100_FQ0 : ProcResource<2>; //Float Queue 0
  def SMTX100_FQ1 : ProcResource<2>; //Float Queue 1
}

def SMTX100_IQ : ProcResGroup<[SMTX100_IQ0, SMTX100_IQ1]>;
def SMTX100_FQ : ProcResGroup<[SMTX100_FQ0, SMTX100_FQ1]>;

// all vector computre inst in VEU0/VEU1
def SMTX100_VEU0 : ProcResource<1>;
def SMTX100_VEU1 : ProcResource<1>;
// all vector memory inst in VEU2/VEU3
def SMTX100_VEU2 : ProcResource<1>;
def SMTX100_VEU3 : ProcResource<1>;

def SMTX100_VIEU : ProcResGroup<[SMTX100_VEU0, SMTX100_VEU1]>;
def SMTX100_VFPU : ProcResGroup<[SMTX100_VEU0, SMTX100_VEU1]>;
def SMTX100_VLSU : ProcResGroup<[SMTX100_VEU2, SMTX100_VEU3]>;

//===----------------------------------------------------------------------===//

// Branching
let Latency = 2 in {
  def : WriteRes<WriteJmp,  [SMTX100_BQ]>;
  def : WriteRes<WriteJal,  [SMTX100_BQ]>;
  def : WriteRes<WriteJalr, [SMTX100_BQ]>;
}

// Integer arithmetic and logic
def : WriteRes<WriteIALU32,     [SMTX100_IQ]>;
def : WriteRes<WriteIALU,       [SMTX100_IQ]>;
def : WriteRes<WriteShiftImm32, [SMTX100_IQ]>;
def : WriteRes<WriteShiftImm,   [SMTX100_IQ]>;
def : WriteRes<WriteShiftReg32, [SMTX100_IQ]>;
def : WriteRes<WriteShiftReg,   [SMTX100_IQ]>;

// Integer multiplication
def : WriteRes<WriteIMul32, [SMTX100_IQ1]> { let Latency = 2; }
def : WriteRes<WriteIMul,   [SMTX100_IQ1]> { let Latency = 3; }

// Integer division/remainder
// Latency is 4-14, Worst case latency is used
let Latency = 14, ReleaseAtCycles = [14] in {
  def : WriteRes<WriteIDiv32, [SMTX100_IQ0]>;
  def : WriteRes<WriteIRem32, [SMTX100_IQ0]>;
}
// Latency is 4-22, Worst case latency is used
let Latency = 22, ReleaseAtCycles = [22] in {
  def : WriteRes<WriteIDiv, [SMTX100_IQ0]>;
  def : WriteRes<WriteIRem, [SMTX100_IQ0]>;
}

// Bitmanip
def : WriteRes<WriteRotateImm,   [SMTX100_IQ]>;
def : WriteRes<WriteRotateImm32, [SMTX100_IQ]>;
def : WriteRes<WriteRotateReg,   [SMTX100_IQ]>;
def : WriteRes<WriteRotateReg32, [SMTX100_IQ]>;

def : WriteRes<WriteCLZ,   [SMTX100_IQ]>;
def : WriteRes<WriteCLZ32, [SMTX100_IQ]>;
def : WriteRes<WriteCTZ,   [SMTX100_IQ]>;
def : WriteRes<WriteCTZ32, [SMTX100_IQ]>;

let Latency = 2 in {
  def : WriteRes<WriteCPOP,   [SMTX100_IQ]>;
  def : WriteRes<WriteCPOP32, [SMTX100_IQ]>;
}

def : WriteRes<WriteORCB,    [SMTX100_IQ]>;
def : WriteRes<WriteIMinMax, [SMTX100_IQ]>;
def : WriteRes<WriteREV8,    [SMTX100_IQ]>;

def : WriteRes<WriteSHXADD,   [SMTX100_IQ]>;
def : WriteRes<WriteSHXADD32, [SMTX100_IQ]>;

def : WriteRes<WriteCLMUL,  [SMTX100_IQ]> { let Latency = 2; }

// Single-bit instructions
def : WriteRes<WriteSingleBit,    [SMTX100_IQ]>;
def : WriteRes<WriteSingleBitImm, [SMTX100_IQ]>;
def : WriteRes<WriteBEXT,  [SMTX100_IQ]>;
def : WriteRes<WriteBEXTI, [SMTX100_IQ]>;

// Memory/Atomic memory
def : WriteRes<WriteSTB, [SMTX100_LSQ]>;
def : WriteRes<WriteSTH, [SMTX100_LSQ]>;
def : WriteRes<WriteSTW, [SMTX100_LSQ]>;
def : WriteRes<WriteSTD, [SMTX100_LSQ]>;
def : WriteRes<WriteFST16, [SMTX100_LSQ]>;
def : WriteRes<WriteFST32, [SMTX100_LSQ]>;
def : WriteRes<WriteFST64, [SMTX100_LSQ]>;

let Latency = 3 in {
  def : WriteRes<WriteLDB, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDH, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDW, [SMTX100_LSQ]>;
  def : WriteRes<WriteLDD, [SMTX100_LSQ]>;
}

let Latency = 4 in {
  def : WriteRes<WriteFLD16, [SMTX100_LSQ]>;
  def : WriteRes<WriteFLD32, [SMTX100_LSQ]>;
  def : WriteRes<WriteFLD64, [SMTX100_LSQ]>;
}

// Atomics
// Latency is at least 7, not sure worst case latency now
let Latency = 7 in {
  def : WriteRes<WriteAtomicSTW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicSTD, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicLDW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicLDD, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicW, [SMTX100_LSQ]>;
  def : WriteRes<WriteAtomicD, [SMTX100_LSQ]>;
}

// Floating point units Half precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMul16, [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ16,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax16, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMA16, [SMTX100_FQ]> { let Latency = 5; }


// Latency is 4-12, Worst case latency is used
let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv16,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt16, [SMTX100_FQ0]>;
}

// Single precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd32,    [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ32,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax32, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMul32, [SMTX100_FQ]> { let Latency = 4; }
def : WriteRes<WriteFMA32,  [SMTX100_FQ]> { let Latency = 5; }

// Latency is 4-12, Worst case latency is used
let Latency = 12, ReleaseAtCycles = [12] in {
  def :  WriteRes<WriteFDiv32,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt32, [SMTX100_FQ0]>;
}

// Double precision
let Latency = 3 in {
  def : WriteRes<WriteFAdd64,    [SMTX100_FQ]>;
  def : WriteRes<WriteFSGNJ64,   [SMTX100_FQ]>;
  def : WriteRes<WriteFMinMax64, [SMTX100_FQ]>;
}
def : WriteRes<WriteFMul64, [SMTX100_FQ]> { let Latency = 4; }
def : WriteRes<WriteFMA64,  [SMTX100_FQ]> { let Latency = 4; }

// Latency is 4-20, Worst case latency is used
let Latency = 20, ReleaseAtCycles = [20] in {
  def :  WriteRes<WriteFDiv64,  [SMTX100_FQ0]>;
  def :  WriteRes<WriteFSqrt64, [SMTX100_FQ0]>;
}

// Zfa
let Latency = 3 in {
  def : WriteRes<WriteFRoundF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFRoundF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFRoundF64, [SMTX100_FQ1]>;

  def : WriteRes<WriteFLI16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFLI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFLI64, [SMTX100_FQ1]>;
}

// Conversions
let Latency = 3 in {
  def : WriteRes<WriteFCvtF16ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToI32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToI64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI32ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtI64ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToF32, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF16ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF32ToF64, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToF16, [SMTX100_FQ1]>;
  def : WriteRes<WriteFCvtF64ToF32, [SMTX100_FQ1]>;

  def : WriteRes<WriteFClass16, [SMTX100_FQ]>;
  def : WriteRes<WriteFClass32, [SMTX100_FQ]>;
  def : WriteRes<WriteFClass64, [SMTX100_FQ]>;

  def : WriteRes<WriteFCmp16, [SMTX100_FQ]>;
  def : WriteRes<WriteFCmp32, [SMTX100_FQ]>;
  def : WriteRes<WriteFCmp64, [SMTX100_FQ]>;

  def : WriteRes<WriteFMovF16ToI16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI16ToF16, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovF32ToI32, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI32ToF32, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovF64ToI64, [SMTX100_FQ]>;
  def : WriteRes<WriteFMovI64ToF64, [SMTX100_FQ]>;
}

// Others
def : WriteRes<WriteCSR,   [SMTX100_IQ0]>;
def : WriteRes<WriteNop,   [SMTX100_IQ]>;
def : WriteRes<WriteRdVLENB, [SMTX100_IQ0]>;

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [SMTX100_IQ0]>;
def : WriteRes<WriteVSETIVLI, [SMTX100_IQ0]>;
def : WriteRes<WriteVSETVL, [SMTX100_IQ0]>;

// 7. Vector Loads and Stores
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  // Unit-stride loads and stores
  defvar VLDELat = ConstValueUntilLMULThenDoubleBase<"M4", 4, 6, mx>.c;
  defvar VLDEOcc = ConstValueUntilLMULThenDoubleBase<"M2", 2, 3, mx>.c;
  let Latency = VLDELat, ReleaseAtCycles = [VLDEOcc] in
    defm "" : LMULWriteResMX<"WriteVLDE", [SMTX100_VLSU], mx, IsWorstCase>;

  defvar VSTELatAndOcc = GetLMULValue<[2, 2, 2, 3, 5, 10, 18], mx>.c;
  let Latency = VSTELatAndOcc, ReleaseAtCycles = [VSTELatAndOcc] in
    defm "" : LMULWriteResMX<"WriteVSTE", [SMTX100_VLSU], mx, IsWorstCase>;

  defvar VLDFFLat = ConstValueUntilLMULThenDoubleBase<"M4", 4, 6, mx>.c;
  defvar VLDFFOcc = ConstValueUntilLMULThenDoubleBase<"M2", 2, 3, mx>.c;
  let Latency = VLDFFLat, ReleaseAtCycles = [VLDFFOcc] in
    defm "" : LMULWriteResMX<"WriteVLDFF", [SMTX100_VLSU], mx, IsWorstCase>;

  // Mask loads and stores
  let Latency = 4, ReleaseAtCycles = [2] in
    defm "" : LMULWriteResMX<"WriteVLDM", [SMTX100_VLSU], mx, IsWorstCase>;

  let Latency = 2, ReleaseAtCycles = [2] in
    defm "" : LMULWriteResMX<"WriteVSTM", [SMTX100_VEU2], mx, IsWorstCase>;

  // Strided and indexed loads and stores
  // vlse
  // e8  =  [4, 5, 9, 22, 50, 114, 242]
  // e16 =  [   4, 5,  9, 22,  50, 114]
  // e32 =  [      4,  5,  9,  22,  50]
  // e64 =  [          4,  5,   9,  22]

  // Latency only decides by element
  // vsse
  // e8  = [4, 8, 16, 32, 64, 128, 256]
  // e16 = [   4,  8, 16, 32,  64, 128]
  // e32 = [       4,  8, 16,  32,  64]    
  // e64 = [           4,  8,  16,  32]
  foreach eew = [8, 16, 32, 64] in {
    defvar VLDSLatAndOcc = SMTX100GetVLDSLat<mx, eew>.val;
    let Latency = VLDSLatAndOcc, ReleaseAtCycles = [VLDSLatAndOcc] in
      defm "" : LMULWriteResMX<"WriteVLDS" # eew, [SMTX100_VLSU], mx, IsWorstCase>;

    defvar VSTSLatAndOcc = SMTX100GetVLMAX<mx, eew>.val;
    let Latency = VSTSLatAndOcc, ReleaseAtCycles = [VSTSLatAndOcc] in
      defm "" : LMULWriteResMX<"WriteVSTS"  # eew, [SMTX100_VEU2], mx, IsWorstCase>;
  }

  foreach eew = [8, 16, 32, 64] in {
    defvar IdxLdStLatAndOcc = !div(SMTX100GetVLMAX<mx, eew>.val, 2);
    let Latency = IdxLdStLatAndOcc, ReleaseAtCycles = [IdxLdStLatAndOcc] in {
      defm "" : LMULWriteResMX<"WriteVLDUX" # eew, [SMTX100_VEU2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX" # eew, [SMTX100_VEU2], mx, IsWorstCase>;

      defm "" : LMULWriteResMX<"WriteVSTUX" # eew, [SMTX100_VEU2], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX" # eew, [SMTX100_VEU2], mx, IsWorstCase>;
    }
  }
}

// Segmented loads and stores
foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

      defvar EffectiveNF= !if(!le(nf, 4), 1, nf);
      defvar SegLdStLatAndOcc = SMTX100SegmentedLdStCycles<mx, eew, EffectiveNF>.c;
      let Latency = SegLdStLatAndOcc, ReleaseAtCycles = [SegLdStLatAndOcc] in {
        // Unit-stride segmented
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" #eew, [SMTX100_VLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" #eew, [SMTX100_VLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" #eew, [SMTX100_VLSU], mx, IsWorstCase>;

        // Strided/indexed segmented
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" #eew, [SMTX100_VLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" #eew, [SMTX100_VLSU], mx, IsWorstCase>;
      }

       defvar IdxSegLdStLatAndOcc = SMTX100SegmentedLdStCycles<mx, eew, nf>.c;
       let Latency = IdxSegLdStLatAndOcc, ReleaseAtCycles = [IdxSegLdStLatAndOcc] in {
        // Indexed segmented
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" #eew, [SMTX100_VEU2], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" #eew, [SMTX100_VEU2], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" #eew, [SMTX100_VEU2], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" #eew, [SMTX100_VEU2], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  defvar WholeLdLatandOcc = !cond(
      !eq(LMul, 1) : 2,
      !eq(LMul, 2) : 3,
      !eq(LMul, 4) : 6,
      !eq(LMul, 8) : 18);
  let Latency = WholeLdLatandOcc, ReleaseAtCycles = [WholeLdLatandOcc] in
    def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SMTX100_VLSU]>;

  defvar WholeStLatandOcc = !cond(
      !eq(LMul, 1) : 3,
      !eq(LMul, 2) : 5,
      !eq(LMul, 4) : 10,
      !eq(LMul, 8) : 18);
  let Latency = WholeStLatandOcc, ReleaseAtCycles = [WholeStLatandOcc] in
    def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SMTX100_VEU2]>;

  // Whole Vector Register Move
  defvar VMovLat = !cond(
      !eq(LMul, 1) : 2,
      !eq(LMul, 2) : 3,
      !eq(LMul, 4) : 4,
      !eq(LMul, 8) : 8);
  let Latency = VMovLat, ReleaseAtCycles = [LMul] in
    def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [SMTX100_VEU0]>;
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  defvar VIALUOcc = ConstOneUntilM1ThenDouble<mx>.c;
  // Vector Integer Min/Max Instructions
  let Latency = Get2348Latency<mx>.c, ReleaseAtCycles = [VIALUOcc] in {
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // Latency of vadd, vsub, vrsub:         3/3/4/8
  // Latency of vand, vor, vxor:           2/3/4/8
  // ReleaseAtCycles : 1/2/4/8
  // They are grouped together, so we used the worst case 3/3/4/8
  // TODO: use InstRW to override individual instructions' scheduling data
  defvar VIALULat = ConstValueUntilLMULThenDoubleBase<"M4", 3, 4, mx>.c;
  let Latency = VIALULat, ReleaseAtCycles = [VIALUOcc] in {
    defm "" : LMULWriteResMX<"WriteVIALUV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // let Latency = VIALULat in
  // def SMX100WriteVCOPY_ # mx : SchedWriteRes<[]>;

  let Latency = Get2348Latency<mx>.c, ReleaseAtCycles = [VIALUOcc] in {
    // Vector Integer Merge Instructions
    defm "" : LMULWriteResMX<"WriteVIMergeV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI", [SMTX100_VIEU], mx, IsWorstCase>;
    // Vector Integer Move Instructions
    defm "" : LMULWriteResMX<"WriteVIMovV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  defvar VIShiftLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VIShiftOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VIShiftLat, ReleaseAtCycles = [VIShiftOcc] in {
    // Vector Integer Extension Instructions 
    defm "" : LMULWriteResMX<"WriteVExtV", [SMTX100_VEU0], mx, IsWorstCase>;
    // Vector Single-Width Bit Shift Instructions
    defm "" : LMULWriteResMX<"WriteVShiftV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI", [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VICALULat = ConstValueUntilLMULThenDoubleBase<"M4", 3, 4, mx>.c;
  let Latency = VICALULat, ReleaseAtCycles = [VIALUOcc] in {
    // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
    defm "" : LMULWriteResMX<"WriteVICALUV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI", [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VICmpCarryLat = GetLMULValue<[3, 3, 3, 4, 6, 10, 18], mx>.c;
  // ReleaseAtCycles: e8 = 1/2/3/5/11, e16/e32/e64 = 1/2/3/5/18
  // but we use the worse case
  defvar VICmpCarryOcc = GetLMULValue<[1, 1, 1, 2, 3, 5, 18], mx>.c;
  let Latency = VICmpCarryLat, ReleaseAtCycles = [VICmpCarryOcc] in {
    defm "" : LMULWriteResMX<"WriteVICALUMV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMI", [SMTX100_VEU0], mx, IsWorstCase>;
    // Vector Integer Comparison Instructions
    defm "" : LMULWriteResMX<"WriteVICmpV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // Latency of vmacc, vmadd, vmul, vmulh, etc.: e8/e16 = 3/3/4/8, e32 = 4/4/5/8,
  // e64 = 7/8/16/32. We use the worst-case until we can split the SEW.
  // ReleaseAtCycles: e8/e16/e32 = 1,2,4,8, e64 = 4/8/16/32 
  // TODO: change WriteVIMulV, etc to be defined with LMULSEWSchedWrites
  defvar VIMulLat = ConstValueUntilLMULThenDoubleBase<"M4", 7, 8, mx>.c;
  defvar VIMulOcc = ConstValueUntilLMULThenDouble<"M1", 4, mx>.c;
  let Latency = VIMulLat, ReleaseAtCycles = [VIMulOcc] in {
    // Vector Single-Width Integer Multiply Instructions
    defm "" : LMULWriteResMX<"WriteVIMulV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX", [SMTX100_VIEU], mx, IsWorstCase>;
    // Vector Single-Width Integer Multiply-Add Instructions
    defm "" : LMULWriteResMX<"WriteVIMulAddV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX", [SMTX100_VIEU], mx, IsWorstCase>;
  }
}

// Widening
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxListW>.c;

  // Vector Widening Integer Add/Subtract
  defvar VIWIntLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VIWideningOcc = GetLMULValue<[1, 1, 2, 2, 4, 8], mx>.c;
  let Latency = VIWIntLat, ReleaseAtCycles = [VIWideningOcc]  in {
    defm "" : LMULWriteResMX<"WriteVIWALUV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // Pattern of vwmul/vwmacc, etc: e8/e16 = 3/3/4/8, e32 = 4/4/5/8
  defvar VIWMulLat = ConstValueUntilLMULThenDoubleBase<"M2", 4, 5, mx>.c;
  let Latency = VIWMulLat, ReleaseAtCycles = [VIWideningOcc] in {
    // Vector Widening Integer Multiply Instructions
    defm "" : LMULWriteResMX<"WriteVIWMulV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX", [SMTX100_VIEU], mx, IsWorstCase>;
    // Vector Widening Integer Multiply-Add Instructions
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SMTX100_VIEU], mx, IsWorstCase>;
  }
}

// Division and remainder operations
// Pattern of vdivu: 17/17/17/25/52/102/205
// Pattern of vdiv: 17/17/17/22/44/88/176
// Pattern of vremu: 15/15/15/20/40/80/160
// Pattern of vrem: 15/15/15/22/45/89/176
// We use for all: 17/17/17/25/52/104/208
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    // Not pipelined
    defvar VIDivLatAndOcc = ConstValueUntilLMULThenDoubleBase<"M1", 17, 25, mx>.c;
    let Latency = VIDivLatAndOcc, ReleaseAtCycles = [VIDivLatAndOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SMTX100_VEU0], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxListW>.c;

  defvar VNarrowingLat = ConstValueUntilLMULThenDoubleBase<"M1", 3, 4, mx>.c;
  defvar VNarrowingOcc = ConstOneUntilMF4ThenDouble<mx>.c;
  let Latency = VNarrowingLat, ReleaseAtCycles = [VNarrowingOcc] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI", [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  defvar VFALULat = ConstValueUntilLMULThenDoubleBase<"M4", 3, 4, mx>.c;
  let Latency = VFALULat, ReleaseAtCycles = [ConstOneUntilM1ThenDouble<mx>.c] in {
    defm "" : LMULWriteResMX<"WriteVSALUV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // Latency of vsmul: e8/e16 = 3/3/4/8, e32 = 4/4/5/8, e64 = 7/8/16/32
  // We use the worst-case until we can split the SEW.
  defvar VSMulLat = ConstValueUntilLMULThenDoubleBase<"M2", 7, 8, mx>.c;
  // ReleaseAtCycles of vsmul: e8/e16/e32 = 1/2/4/8, e64 = 4/8/16/32
  // We use the worst-case until we can split the SEW.
  defvar VSMulOcc = ConstValueUntilLMULThenDoubleBase<"M1", 1, 4, mx>.c;
  // TODO: change WriteVSMulV/X to be defined with LMULSEWSchedWrites
  let Latency = VSMulLat, ReleaseAtCycles = [VSMulOcc] in {
    defm "" : LMULWriteResMX<"WriteVSMulV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  defvar VSShiftLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VSShiftOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VSShiftLat, ReleaseAtCycles = [VSShiftOcc] in {
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defvar VFALUOcc = ConstOneUntilM1ThenDouble<mx>.c;
    let Latency = Get4458Latency<mx>.c, ReleaseAtCycles = [VFALUOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }

    defvar VFMaxMinLat = ConstValueUntilLMULThenDoubleBase<"M4", 3, 4, mx>.c;
    let Latency = VFMaxMinLat, ReleaseAtCycles = [VFALUOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }

    // Latency: e32 = 4/4/5/8, e64 = 5/5/5/8
    // Slightly increased latency for sew == 64
    defvar VFMulLat = !if(!eq(sew, 64), ConstValueUntilLMULThenDoubleBase<"M8", 5, 8, mx>.c,
                          Get4458Latency<mx>.c);
    let Latency = VFMulLat, ReleaseAtCycles = [ConstOneUntilM1ThenDouble<mx>.c] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }

    defvar VFSgnjLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
    defvar VFSgnjOcc = ConstOneUntilMF2ThenDouble<mx>.c;
    let Latency = VFSgnjLat, ReleaseAtCycles = [VFSgnjOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SMTX100_VEU0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
    }

    // The following covers vfmacc, vfmsac, vfmadd and their vfn* variants in the same group, but the
    // ReleaseAtCycles takes one extra cycle for the vfn* variants on lmul == 4
    // vfmacc/vfmacc/vfmadd/vfmsub: 1/2/4/8 and vfn* variants: 1/2/5/8
    // Latency: e32 = 4/4/5/8, e64 = 5/5/5/8
    defvar VFFulAddLat = !if(!eq(sew, 64), ConstValueUntilLMULThenDoubleBase<"M8", 5, 8, mx>.c,
                             Get4458Latency<mx>.c);
    let Latency = VFFulAddLat, ReleaseAtCycles = [ConstOneUntilM1ThenDouble<mx>.c] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  defvar VFCmpLat = GetLMULValue<[3, 3, 3, 4, 6, 10, 18], mx>.c;
  defvar VFCmpOcc = GetLMULValue<[1, 1, 1, 2, 3,  5, 18], mx>.c;
  let Latency = VFCmpLat, ReleaseAtCycles = [VFCmpOcc] in {
    defm "" : LMULWriteResMX<"WriteVFCmpV", [SMTX100_VFPU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF", [SMTX100_VFPU], mx, IsWorstCase>;
  }

  defvar VFClassLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VFClassOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VFClassLat, ReleaseAtCycles = [VFClassOcc] in {
    defm "" : LMULWriteResMX<"WriteVFClassV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV", [SMTX100_VEU0], mx, IsWorstCase>;
    // TODO TEST LMUL = MF8
    defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;
    defvar VFWCvtILat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
    defvar VFWCvtIOcc = ConstOneUntilMF4ThenDouble<mx>.c;
    let Latency = VFWCvtILat, ReleaseAtCycles = [VFWCvtIOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxListFW>.c;

  defvar VFWCvtFToIVLat = ConstValueUntilLMULThenDoubleBase<"M1", 3, 4, mx>.c;
  defvar VFWCvtFToIVOcc = ConstOneUntilMF4ThenDouble<mx>.c;
  let Latency = VFWCvtFToIVLat, ReleaseAtCycles = [VFWCvtFToIVOcc] in
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SMTX100_VEU0], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    defvar VFWCvtFToFVLat = ConstValueUntilLMULThenDoubleBase<"M1", 3, 4, mx>.c;
    defvar VFWCvtFToFVOcc = ConstOneUntilMF4ThenDouble<mx>.c;
    let Latency = VFWCvtFToFVLat, ReleaseAtCycles = [VFWCvtFToFVOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SMTX100_VEU0], mx, sew, IsWorstCase>;

    // Latency for vfwsub/vfwadd.vv, vfwsub/vfwadd.vf: 
    // e16 = 3/3/3/4/8, e32 = 4/4/5/8
    // ReleaseAtCycles for vfwsub/vfwadd.vv, vfwsub/vfwadd.vf: 1/2/2/4/8
    // Latency for vfwsub/vfwadd.wv, vfwsub/vfwadd.wf:
    // e16 = 4/4/6/8/16, e32 = 5/6/8/16
    // ReleaseAtCycles for vfwsub/vfwadd.wv, vfwsub/vfwadd.wf: 2/2/4/8/16
    // We use the worst-case
    // TODO: Split .vv/.wv variants into separate scheduling classes
    defvar VFWALULat = !if(!eq(sew, 16),
      GetLMULValue<[4, 4, 4, 6, 8, 16], mx>.c,
      GetLMULValue<[5, 5, 5, 6, 8, 16], mx>.c
    );
    defvar VFWALUOcc = GetLMULValue<[2, 2, 2, 4, 8, 16], mx>.c;
    let Latency = VFWALULat, ReleaseAtCycles = [VFWALUOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }

    // Latency for vfwmul, vfwmacc, vfwnmacc, etc: e16 = 4/4/4/5/8; e32 = 5/5/5/8
    defvar VFWMulLat = !if(!eq(sew, 16),
      GetLMULValue<[4, 4, 4, 4, 5, 8], mx>.c,
      GetLMULValue<[5, 5, 5, 5, 5, 8], mx>.c
    );
    defvar VFWMulOcc = GetLMULValue<[1, 1, 2, 2, 4, 8], mx>.c;
    let Latency = VFWMulLat, ReleaseAtCycles = [VFWMulOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxListW>.c;

  defvar VFNCvtFToIVLat = GetLMULValue<[3, 3, 4, 4, 8, 16], mx>.c;
  defvar VFNCvtFToIVOcc = ConstOneUntilMF4ThenDouble<mx>.c;
  let Latency = VFNCvtFToIVLat, ReleaseAtCycles = [VFNCvtFToIVOcc] in
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SMTX100_VEU0], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    defvar VFNCvtToFVLat = GetLMULValue<[3, 3, 4, 4, 8, 16], mx>.c;
    defvar VFNCvtToFVOcc = ConstOneUntilMF4ThenDouble<mx>.c;
    let Latency = VFNCvtToFVLat, ReleaseAtCycles = [VFNCvtToFVOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
    }
  }
}

// Vector Floating-Point Division and Square Root
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Compute ReleaseAtCycles based on SEW
    // vfdiv.vf and vfrdiv.vf a slight latency difference
    // e16 = 14/28/57/116 , e32 = 24/48/82/193, e64 = 12/24/48/96
    defvar VFDivFLatAndOcc = !cond(
      !eq(sew, 16): 14,
      !eq(sew, 32): 24,
      !eq(sew, 64): 12,
    );
    let Latency = VFDivFLatAndOcc, ReleaseAtCycles = [VFDivFLatAndOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF", [SMTX100_VEU0], mx, sew, IsWorstCase>;

    // e16 = 28/56/112/224, e32 = 36/72/145/290, e64 = 66/132/263/527
    defvar VFDivVFactor = !cond(
      !eq(sew, 16): 28,
      !eq(sew, 32): 36,
      !eq(sew, 64): 66,
    );
    defvar VFDivVLatAndOcc = !mul(ConstOneUntilM1ThenDouble<mx>.c, VFDivVFactor);
    let Latency = VFDivVLatAndOcc, ReleaseAtCycles = [VFDivVLatAndOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
  }
}

// Pattern for vfsqrt.v: e16 = 26/52/104/208; e32 = 35/72/140/276; e64 = 52/99/216/412
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    defvar VFSqrtVFactor = !cond(
      !eq(sew, 16): 26,
      !eq(sew, 32): 35,
      !eq(sew, 64): 52,
    );
    defvar VFSqrtVLatAndOcc = !mul(ConstOneUntilM1ThenDouble<mx>.c, VFSqrtVFactor);
    let Latency = VFSqrtVLatAndOcc, ReleaseAtCycles = [VFSqrtVLatAndOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SMTX100_VEU0], mx, sew, IsWorstCase>;
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defvar VIRedLat = GetLMULValue<[4, 4, 4, 5, 7, 11, 19], mx>.c;
    defvar VIRedOcc = GetLMULValue<[1, 1, 1, 1, 2, 6, 19], mx>.c;
    let Latency = VIRedLat, ReleaseAtCycles = [VIRedOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SMTX100_VIEU], mx, sew, IsWorstCase>;

      // Latency:
      // Pattern for vredsum:                  4/4/4/5/7/11/19
      // Pattern for vredand, vredor, vredxor: 3/3/3/4/6/10/18
      // ReleaseAtCycles:
      // Pattern for vredsum:                  1/1/1/1/2/6/19
      // Pattern for vredand, vredor, vredxor: 1/1/1/1/2/5/18
      // They are grouped together, so we use the worst-case vredsum latency.
      // TODO: split vredand, vredor, vredxor into separate scheduling classe.
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SMTX100_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;

    defvar VIRedLat = GetLMULValue<[4, 4, 4, 5, 7, 11, 19], mx>.c;
    defvar VIRedOcc = GetLMULValue<[1, 1, 1, 1, 2, 6, 19], mx>.c;
    let Latency = VIRedLat, ReleaseAtCycles = [VIRedOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SMTX100_VIEU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Latency is slightly lower for e16/e32, We use the worst-case
    defvar VFRedLat = GetLMULValue<[12, 12, 12, 15, 21, 33, 57], mx>.c;
    defvar VFRedOcc = GetLMULValue<[8,   8,  8,  8, 14, 20, 57], mx>.c;
    let Latency = VFRedLat, ReleaseAtCycles = [VFRedOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    // Compute latency based on SEW
    defvar VFRedOV_FromLat = !cond(
      !eq(sew, 16) : ConstValueUntilLMULThenDouble<"MF4", 12, mx>.c,
      !eq(sew, 32) : ConstValueUntilLMULThenDouble<"MF2", 12, mx>.c,
      !eq(sew, 64) : ConstValueUntilLMULThenDouble<"M1", 12, mx>.c
    );
    defvar VFRedOV_FromOcc = !cond(
      !eq(sew, 16) : GetLMULValue<[8, 8, 20, 23, 47, 96, 384], mx>.c,
      !eq(sew, 32) : GetLMULValue<[8, 8,  8, 11, 23, 48, 192], mx>.c,
      !eq(sew, 64) : GetLMULValue<[6, 6,  6,  6, 12, 25,  97], mx>.c
    );
    let Latency = VFRedOV_FromLat, ReleaseAtCycles = [VFRedOV_FromOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SMTX100_VFPU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar IsWorstCase = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;

    defvar VFRedOVLat = !cond(
      !eq(sew, 16) : ConstValueUntilLMULThenDouble<"MF4", 16, mx>.c,
      !eq(sew, 32) : ConstValueUntilLMULThenDouble<"MF2", 16, mx>.c,
    );
    defvar VFRedOVOcc = !cond(
      !eq(sew, 16) : GetLMULValue<[10, 10, 26, 31, 64, 128, 512], mx>.c,
      !eq(sew, 32) : GetLMULValue<[10, 10, 10, 15, 31,  66, 256], mx>.c,
    );
    let Latency = VFRedOVLat, ReleaseAtCycles = [VFRedOVOcc] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From", [SMTX100_VFPU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SMTX100_VFPU], mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = 2, ReleaseAtCycles = [1] in
    defm "" : LMULWriteResMX<"WriteVMALUV", [SMTX100_VEU0], mx, IsWorstCase>;

  let Latency = 3, ReleaseAtCycles = [GetLMULValue<[1, 1, 1, 1, 1, 2, 3], mx>.c] in
    defm "" : LMULWriteResMX<"WriteVMSFSV", [SMTX100_VEU0], mx, IsWorstCase>;

  let Latency = 6, ReleaseAtCycles = [6] in {
    defm "" : LMULWriteResMX<"WriteVMPopV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VIotaLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VIotaOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VIotaLat, ReleaseAtCycles = [VIotaOcc] in {
    defm "" : LMULWriteResMX<"WriteVIotaV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIdxV", [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

// 16. Vector Permutation Instructions
// Slide
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  defvar VSlideUpLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar VSlideUpOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VSlideUpLat, ReleaseAtCycles =[VSlideUpOcc] in {
    defm "" : LMULWriteResMX<"WriteVSlideUpX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideUpI", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlide1Up", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1Up", [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VSlideDownLat = GetLMULValue<[3, 3, 3, 4, 5, 9, 17], mx>.c;
  defvar VSlideDownOcc = GetLMULValue<[1, 1, 1, 3, 5, 9, 17], mx>.c;
  let Latency = VSlideDownLat, ReleaseAtCycles =[VSlideDownOcc] in {
    defm "" : LMULWriteResMX<"WriteVSlideDownX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSlideDownI", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlide1Down", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1Down", [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

// ReleaseAtCycles is 2/2/2/2/2/3/6, but we can't set based on MX for now
// TODO: Split this into separate WriteRes for each MX
let Latency = 6, ReleaseAtCycles = [6] in {
  def : WriteRes<WriteVMovXS, [SMTX100_VEU0]>;
  def : WriteRes<WriteVMovFS, [SMTX100_VEU0]>;
}

// ReleaseAtCycles is 1/1/1/1/1/2/3, but we can't set based on MX for now
// TODO: Split this into separate WriteRes for each MX
let Latency = 3, ReleaseAtCycles = [3] in {
  def : WriteRes<WriteVMovSX, [SMTX100_VEU0]>;
  def : WriteRes<WriteVMovSF, [SMTX100_VEU0]>;
}

// Integer LMUL Gather and Compress
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;

  defvar VRGatherLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  let Latency = VRGatherLat, ReleaseAtCycles = [ConstOneUntilMF2ThenDouble<mx>.c] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [SMTX100_VEU0], mx, IsWorstCase>;
  }

  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCaseSEW = SMTX100IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defvar VRGatherVVLat = GetLMULValue<[3, 3, 3, 3, 8, 32, 128], mx>.c;
    defvar VRGatherVVOcc = GetLMULValue<[1, 1, 1, 2, 8, 32, 128], mx>.c;
    let Latency = VRGatherVVLat, ReleaseAtCycles = [VRGatherVVOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SMTX100_VEU0], mx, sew, IsWorstCaseSEW>;

    defvar VRGatherEI16Lat = !if(!eq(sew, 8),
      GetLMULValue<[3, 3, 4, 4, 16, 64, 128], mx>.c,
      GetLMULValue<[3, 3, 3, 3, 8,  32, 128], mx>.c);
    defvar VRGatherEI16Occ = !if(!eq(sew, 8),
      GetLMULValue<[1, 1, 2, 4, 16, 64, 128], mx>.c,
      GetLMULValue<[1, 1, 1, 2,  8, 32, 128], mx>.c);
    let Latency = VRGatherEI16Lat, ReleaseAtCycles = [VRGatherEI16Occ] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SMTX100_VEU0], mx, sew, IsWorstCaseSEW>;

    defvar VCompressVLat = GetLMULValue<[3, 3, 3, 4, 10, 36, 136], mx>.c;
    defvar VCompressVOcc = GetLMULValue<[1, 1, 1, 3, 10, 36, 136], mx>.c;
    let Latency = VCompressVLat, ReleaseAtCycles = [VCompressVOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SMTX100_VEU0], mx, sew, IsWorstCaseSEW>;
  }
}

// Vector Crypto
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxList>.c;
  defvar LMulOcc = !mul(SMTX100GetLMulCycles<mx>.c, 2);

  defvar Lat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  defvar Occ = ConstValueUntilLMULThenDouble<"MF2", 1, mx>.c;
  let Latency = Lat, ReleaseAtCycles = [Occ] in {
    // Zvbb
    defm "" : LMULWriteResMX<"WriteVBREVV",   [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLZV",    [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCTZV",    [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCPOPV",   [SMTX100_VEU0], mx, IsWorstCase>;
    // Zvkb
    // VANDN uses WriteVIALU[V|X|I]
    defm "" : LMULWriteResMX<"WriteVBREV8V",  [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVREV8V",   [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotV",    [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotX",    [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRotI",    [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VCLAndVGLat = ConstValueUntilLMULThenDoubleBase<"M2", 3, 4, mx>.c;
  let Latency = VCLAndVGLat, ReleaseAtCycles = [LMulOcc] in {
    // Zvbc
    defm "" : LMULWriteResMX<"WriteVCLMULV", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVCLMULX", [SMTX100_VEU0], mx, IsWorstCase>;

    // Zvkg
    defm "" : LMULWriteResMX<"WriteVGHSHV",   [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVGMULV",   [SMTX100_VEU0], mx, IsWorstCase>;
  }

  // ZvknhaOrZvknhb
  // vsha2ch.vv / vsha2cl.vv are modeled LMUL-only using the SEW=32 curve:
  //   M2=4, M4=8, M8=16
  //   e32: M2=4, M4=8, M8=16
  //   e64: M4=4, M8=8
  // TODO: Shold set vsha2ch.vv / vsha2cl.vv SEW-aware?
  defvar VSHALatandOcc = ConstValueUntilLMULThenDouble<"M2", 4, mx>.c;
  let Latency = VSHALatandOcc, ReleaseAtCycles = [VSHALatandOcc] in {
    defm "" : LMULWriteResMX<"WriteVSHA2CHV", [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSHA2CLV", [SMTX100_VIEU], mx, IsWorstCase>;
  }

  // vsha2ms.vv latency depends on SEW: doubling SEW shifts the doubling
  // threshold right by one LMUL step.
  //   e32: M2=4, M4=8, M8=16
  //   e64: M4=4, M8=8
  defvar ZvknhSEWs = !listremove(SchedSEWSet<mx>.val, [8, 16]);
  // Smallest SEW (=32) carries the highest latency, so the WorstCase fallback
  // is designated to the smallest SEW at the largest LMUL.
  defvar SmallestZvknhSEW = !head(ZvknhSEWs);
  foreach sew = ZvknhSEWs in {
    defvar IsWorstCaseVSHA2MSV = !and(IsWorstCase, !eq(sew, SmallestZvknhSEW));
    defvar VSHA2MSLatandOcc =
        ConstValueUntilLMULThenDouble<!if(!eq(sew, 64), "M4", "M2"), 4, mx>.c;
    let Latency = VSHA2MSLatandOcc, ReleaseAtCycles = [VSHA2MSLatandOcc] in
      defm "" : LMULSEWWriteResMXSEW<"WriteVSHA2MSV", [SMTX100_VIEU], mx, sew,
                                     IsWorstCaseVSHA2MSV>;
  }

  // Zvkned
  defvar VASELatandOcc = ConstValueUntilLMULThenDouble<"M1", 2, mx>.c;
  let Latency = VASELatandOcc, ReleaseAtCycles = [VASELatandOcc] in {
    defm "" : LMULWriteResMX<"WriteVAESMVV",  [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF1V", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESKF2V", [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAESZV",   [SMTX100_VEU0], mx, IsWorstCase>;
  }

  defvar VSM3CLat = ConstValueUntilLMULThenDouble<"M4", 4, mx>.c;
  let Latency = VSM3CLat, ReleaseAtCycles = [SMTX100GetLMulCycles<mx>.c] in
    defm "" : LMULWriteResMX<"WriteVSM3CV",   [SMTX100_VIEU], mx, IsWorstCase>;

  defvar VSM3MLat = ConstValueUntilLMULThenDoubleBase<"M4", 3, 4, mx>.c;
  let Latency = VSM3MLat, ReleaseAtCycles = [SMTX100GetLMulCycles<mx>.c] in
    defm "" : LMULWriteResMX<"WriteVSM3MEV",  [SMTX100_VIEU], mx, IsWorstCase>;

  defvar VSM4Lat = ConstValueUntilLMULThenDouble<"M2", 4, mx>.c;
  let Latency = 3, ReleaseAtCycles = [LMulOcc] in {
    defm "" : LMULWriteResMX<"WriteVSM4KV",   [SMTX100_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSM4RV",   [SMTX100_VIEU], mx, IsWorstCase>;
  }
}

foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMTX100IsWorstCaseMX<mx, SchedMxListW>.c;

  defvar VWSLat = ConstValueUntilLMULThenDoubleBase<"M1", 3, 4, mx>.c;
  defvar VWSOcc = ConstValueUntilLMULThenDouble<"MF4", 1, mx>.c;
  let Latency = VWSLat, ReleaseAtCycles = [VWSOcc] in {
    defm "" : LMULWriteResMX<"WriteVWSLLV",   [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLX",   [SMTX100_VEU0], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVWSLLI",   [SMTX100_VEU0], mx, IsWorstCase>;
  }
}

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
// Zfa
def : ReadAdvance<ReadFRoundF32, 0>;
def : ReadAdvance<ReadFRoundF64, 0>;
def : ReadAdvance<ReadFRoundF16, 0>;

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

// 11. Vector Integer Arithmetic Instructions
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

// 12. Vector Fixed-Point Arithmetic Instructions
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

// 13. Vector Floating-Point Instructions
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

// 14. Vector Reduction Operations
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

// 15. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVIotaV", 0>;

// 16. Vector Permutation Instructions
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
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfaWithQ;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsf;
}
