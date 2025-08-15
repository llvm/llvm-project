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

//===----------------------------------------------------------------------===//
// Helpers

// Maps LMUL string to corresponding value from the Values array
// LMUL values map to array indices as follows:
//   MF8 -> Values[0], MF4 -> Values[1], MF2 -> Values[2], M1 -> Values[3],
//   M2 -> Values[4], M4 -> Values[5], M8 -> Values[6]
// Shorter lists are allowed, e.g., widening instructions don't work on M8
class GetLMULValue<list<int> Values, string LMUL> {
  defvar Index = !cond(
    !eq(LMUL, "MF8"): 0,
    !eq(LMUL, "MF4"): 1,
    !eq(LMUL, "MF2"): 2,
    !eq(LMUL, "M1"):  3,
    !eq(LMUL, "M2"):  4,
    !eq(LMUL, "M4"):  5,
    !eq(LMUL, "M8"):  6,
  );

  assert !lt(Index, !size(Values)),
    "Missing LMUL value for '" # LMUL # "'. " #
    "Expected at least " # !add(Index, 1) # " elements, but got " #
    !size(Values) # ".";

  int c = Values[Index];
}

// Returns BaseValue for LMUL values before startLMUL, Value for startLMUL,
// then doubles Value for each subsequent LMUL
// Example: ConstValueUntilLMULThenDoubleBase<"M1", 2, 4, "M8"> returns:
//   MF8->2, MF4->2, MF2->2, M1->4, M2->8, M4->16, M8->32
// This is useful for modeling scheduling parameters that scale with LMUL.
class ConstValueUntilLMULThenDoubleBase<string startLMUL, int BaseValue, int Value, string currentLMUL> {
  assert !le(BaseValue, Value), "BaseValue must be less-equal to Value";
  defvar startPos = GetLMULValue<[0, 1, 2, 3, 4, 5, 6], startLMUL>.c;
  defvar currentPos = GetLMULValue<[0, 1, 2, 3, 4, 5, 6], currentLMUL>.c;

  // Calculate the difference in positions
  defvar posDiff = !sub(currentPos, startPos);

  // Calculate Value * (2^posDiff)
  int c = !cond(
    !eq(posDiff, 0) : Value,
    !eq(posDiff, 1) : !mul(Value, 2),
    !eq(posDiff, 2) : !mul(Value, 4),
    !eq(posDiff, 3) : !mul(Value, 8),
    !eq(posDiff, 4) : !mul(Value, 16),
    !eq(posDiff, 5) : !mul(Value, 32),
    !eq(posDiff, 6) : !mul(Value, 64),
    true : BaseValue
  );
}

// Same as the previous function but BaseValue == Value
class ConstValueUntilLMULThenDouble<string startLMUL, int Value, string currentLMUL> {
  int c = ConstValueUntilLMULThenDoubleBase<startLMUL, Value, Value, currentLMUL>.c;
}

// Returns MF8->1, MF4->1, MF2->2, M1->4, M2->8, M4->16, M8->32
class ConstOneUntilMF4ThenDouble<string mx> {
  int c = ConstValueUntilLMULThenDouble<"MF4", 1, mx>.c;
}

// Returns MF8->1, MF4->1, MF2->1, M1->2, M2->4, M4->8, M8->16
class ConstOneUntilMF2ThenDouble<string mx> {
  int c = ConstValueUntilLMULThenDouble<"MF2", 1, mx>.c;
}

// Returns MF8->1, MF4->1, MF2->1, M1->1, M2->2, M4->4, M8->8
class ConstOneUntilM1ThenDouble<string mx> {
  int c = ConstValueUntilLMULThenDouble<"M1", 1, mx>.c;
}

//===----------------------------------------------------------------------===//
// Latency helper classes

// Used for: arithmetic (add/sub/min/max), saturating/averaging, FP add/sub/min/max
class Get4458Latency<string mx> {
  int c = GetLMULValue<[/*MF8=*/4, /*MF4=*/4, /*MF2=*/4, /*M1=*/4, /*M2=*/4, /*M4=*/5, /*M8=*/8], mx>.c;
}

// Used for: widening operations (no M8)
class Get4588Latency<string mx> {
  int c = GetLMULValue<[/*MF8=*/4, /*MF4=*/4, /*MF2=*/4, /*M1=*/4, /*M2=*/5, /*M4=*/8], mx>.c;
}

// Used for: mask-producing comparisons, carry ops with mask, FP comparisons
class Get461018Latency<string mx> {
  int c = GetLMULValue<[/*MF8=*/4, /*MF4=*/4, /*MF2=*/4, /*M1=*/4, /*M2=*/6, /*M4=*/10, /*M8=*/18], mx>.c;
}

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

defvar SMX60VLEN = 256;
defvar SMX60DLEN = !div(SMX60VLEN, 2);

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

  // The C908 user manual says: "Vector floating-point units support vector
  // floating-point computation of different bits. In addition, vector integer
  // units are added". Developer confirmed it's a separate VIEU
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
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  // Unit-stride loads and stores
  defm "" : LMULWriteResMX<"WriteVLDE", [SMX60_VLS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVLDFF", [SMX60_VLS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSTE", [SMX60_VLS], mx, IsWorstCase>;

  // Mask loads and stores
  defm "" : LMULWriteResMX<"WriteVLDM", [SMX60_VLS], mx, IsWorstCase=!eq(mx, "M1")>;
  defm "" : LMULWriteResMX<"WriteVSTM", [SMX60_VLS], mx, IsWorstCase=!eq(mx, "M1")>;

  // Strided and indexed loads and stores
  foreach eew = [8, 16, 32, 64] in {
    defm "" : LMULWriteResMX<"WriteVLDS"  # eew, [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX" # eew, [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX" # eew, [SMX60_VLS], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVSTS"  # eew, [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX" # eew, [SMX60_VLS], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX" # eew, [SMX60_VLS], mx, IsWorstCase>;
  }
}

// Segmented loads and stores
foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

      // Unit-stride segmented
      defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;

      // Strided/indexed segmented
      defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;

      // Indexed segmented
      defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" #eew, [SMX60_VLS], mx, IsWorstCase>;
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [SMX60_VLS]>;
  def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [SMX60_VLS]>;

  def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [SMX60_VIEU]>;
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = Get4458Latency<mx>.c, ReleaseAtCycles = [4] in {
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [SMX60_VIEU], mx, IsWorstCase>;
  }

  defvar VIALULat = ConstValueUntilLMULThenDouble<"M2", 4, mx>.c;
  let Latency = VIALULat, ReleaseAtCycles = [4] in {
    // Pattern of vadd, vsub, vrsub: 4/4/5/8
    // Pattern of vand, vor, vxor:   4/4/8/16
    // They are grouped together, so we used the worst case 4/4/8/16
    // TODO: use InstRW to override individual instructions' scheduling data
    defm "" : LMULWriteResMX<"WriteVIALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI", [SMX60_VIEU], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVExtV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI", [SMX60_VIEU], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVICALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI", [SMX60_VIEU], mx, IsWorstCase>;
  }

  let Latency = Get461018Latency<mx>.c, ReleaseAtCycles = [4] in {
    defm "" : LMULWriteResMX<"WriteVICALUMV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUMI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI", [SMX60_VIEU], mx, IsWorstCase>;
  }

  // Pattern of vmacc, vmadd, vmul, vmulh, etc.: e8/e16 = 4/4/5/8, e32 = 5,5,5,8,
  // e64 = 7,8,16,32. We use the worst-case until we can split the SEW.
  // TODO: change WriteVIMulV, etc to be defined with LMULSEWSchedWrites
  let Latency = ConstValueUntilLMULThenDoubleBase<"M2", 7, 8, mx>.c, ReleaseAtCycles = [7] in {
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

  let Latency = Get4588Latency<mx>.c, ReleaseAtCycles = [4]  in {
    defm "" : LMULWriteResMX<"WriteVIWALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// Division and remainder operations
// Pattern of vdivu: 11/11/11/20/40/80/160
// Pattern of vdiv: 12/12/12/22/44/88/176
// Pattern of vremu: 12/12/12/22/44/88/176
// Pattern of vrem: 13/13/13/24/48/96/192
// We use for all: 12/12/12/24/48/96/192
// TODO: Create separate WriteVIRem to more closely match the latencies
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defvar VIDivLat = ConstValueUntilLMULThenDouble<"MF2", 12, mx>.c;
    let Latency = VIDivLat, ReleaseAtCycles = [12] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [SMX60_VIEU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [SMX60_VIEU], mx, sew, IsWorstCase>;
    }
  }
}

// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  defvar VNarrowingLat = ConstValueUntilLMULThenDouble<"M1", 4, mx>.c;
  let Latency = VNarrowingLat, ReleaseAtCycles = [4] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  let Latency = Get4458Latency<mx>.c, ReleaseAtCycles = [ConstOneUntilM1ThenDouble<mx>.c] in {
    defm "" : LMULWriteResMX<"WriteVSALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX", [SMX60_VIEU], mx, IsWorstCase>;
  }

  // Latency of vsmul: e8/e16 = 4/4/5/8, e32 = 5/5/5/8, e64 = 7/8/16/32
  // We use the worst-case until we can split the SEW.
  defvar VSMulLat = ConstValueUntilLMULThenDoubleBase<"M2", 7, 8, mx>.c;
  // Latency of vsmul: e8/e16/e32 = 1/2/4/8, e64 = 4/8/16/32
  // We use the worst-case until we can split the SEW.
  defvar VSMulOcc = ConstValueUntilLMULThenDoubleBase<"M1", 1, 4, mx>.c;
  // TODO: change WriteVSMulV/X to be defined with LMULSEWSchedWrites
  let Latency = VSMulLat, ReleaseAtCycles = [VSMulOcc] in {
    defm "" : LMULWriteResMX<"WriteVSMulV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX", [SMX60_VIEU], mx, IsWorstCase>;
  }

  defvar VSShiftLat = ConstValueUntilLMULThenDouble<"M2", 4, mx>.c;
  defvar VSShiftOcc = ConstOneUntilMF2ThenDouble<mx>.c;
  let Latency = VSShiftLat, ReleaseAtCycles = [VSShiftOcc] in {
    defm "" : LMULWriteResMX<"WriteVSShiftV", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [SMX60_VIEU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [SMX60_VIEU], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [SMX60_VFP], mx, sew, IsWorstCase>;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVFCmpV", [SMX60_VFP], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFCmpF", [SMX60_VFP], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFClassV", [SMX60_VFP], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFMergeV", [SMX60_VFP], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFMovV", [SMX60_VFP], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListFW>.c;

  defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

// Narrowing
foreach mx = SchedMxListW in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxListW>.c;

  defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [SMX60_VFP], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {

    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

// Vector Floating-Point Division and Square Root
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [SMX60_VIEU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [SMX60_VFP], mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVMALUV", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMPopV", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMFFSV", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMSFSV", [SMX60_VIEU], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVIotaV", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIdxV", [SMX60_VIEU], mx, IsWorstCase>;
}

// 16. Vector Permutation Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVSlideI", [SMX60_VIEU], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVISlide1X", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFSlide1F", [SMX60_VFP], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVSlideUpX", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSlideDownX", [SMX60_VIEU], mx, IsWorstCase>;
}

def : WriteRes<WriteVMovXS, [SMX60_VIEU]>;
def : WriteRes<WriteVMovSX, [SMX60_VIEU]>;

def : WriteRes<WriteVMovFS, [SMX60_VIEU]>;
def : WriteRes<WriteVMovSF, [SMX60_VIEU]>;

// Gather and Compress
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = SMX60IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [SMX60_VIEU], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [SMX60_VIEU], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [SMX60_VIEU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = SMX60IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVRGatherVX", [SMX60_VIEU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVRGatherVI", [SMX60_VIEU], mx, IsWorstCase>;
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
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsf;
}
