//==- RISCVSchedAndes45.td - Andes45 Scheduling Definitions --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// The worst case LMUL is the largest LMUL.
class Andes45IsWorstCaseMX<string mx, list<string> MxList> {
  defvar LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

// The worst case is the largest LMUL with the smallest SEW.
class Andes45IsWorstCaseMXSEW<string mx, int sew, list<string> MxList,
                              bit isF = 0> {
  defvar LLMUL = LargestLMUL<MxList>.r;
  defvar SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

def Andes45Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // Andes45 is in-order processor
  let IssueWidth = 2;         // 2 micro-ops dispatched per cycle
  let LoadLatency = 2;
  let MispredictPenalty = 5;
  let CompleteModel = 0;
}

let SchedModel = Andes45Model in {

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

//===----------------------------------------------------------------------===//
// Andes 45 series CPU
//   - 2 Integer Arithmetic and Logical Units (ALU)
//   - Multiply / Divide Unit (MDU)
//   - Load Store Unit (LSU)
//   - Control and Status Register Unit (CSR)
//   - Floating Point Multiply-Accumulate Unit (FMAC)
//   - Floating Point Divide / SQRT Unit (FDIV)
//   - Floating Point Move Unit (FMV)
//   - Floating Point Misc Unit (FMISC)
//
// Andes 45 series VPU
//   - Vector Arithmetic and Logical Unit (VALU)
//   - Vector Multiply Accumulate Unit (VMAC)
//   - Vector Divide Unit (VDIV)
//   - Vector Permutation Unit (VPERMUT)
//   - Vector Mask Unit (VMASK)
//   - Vector Floating-Point Miscellaneous Unit (VFMIS)
//   - Vector Floating-Point Divide Unit (VFDIV)
//===----------------------------------------------------------------------===//

let BufferSize = 0 in {
def Andes45ALU : ProcResource<2>;
def Andes45MDU : ProcResource<1>;
def Andes45LSU : ProcResource<1>;
def Andes45CSR : ProcResource<1>;

def Andes45FMAC  : ProcResource<1>;
def Andes45FDIV  : ProcResource<1>;
def Andes45FMV   : ProcResource<1>;
def Andes45FMISC : ProcResource<1>;

def Andes45VALU    : ProcResource<1>;
def Andes45VMAC    : ProcResource<1>;
def Andes45VFMIS   : ProcResource<1>;
def Andes45VPERMUT : ProcResource<1>;
def Andes45VDIV    : ProcResource<1>;
def Andes45VFDIV   : ProcResource<1>;
def Andes45VMASK   : ProcResource<1>;
def Andes45VLSU    : ProcResource<1>;
}

// Integer arithmetic and logic
def : WriteRes<WriteIALU, [Andes45ALU]>;
def : WriteRes<WriteIALU32, [Andes45ALU]>;
def : WriteRes<WriteShiftImm, [Andes45ALU]>;
def : WriteRes<WriteShiftImm32, [Andes45ALU]>;
def : WriteRes<WriteShiftReg, [Andes45ALU]>;
def : WriteRes<WriteShiftReg32, [Andes45ALU]>;

// Short forward branch
def : WriteRes<WriteSFB, [Andes45ALU]> {
  let Latency = 1;
  let NumMicroOps = 2;
}

// Branching
def : WriteRes<WriteJmp, [Andes45ALU]>;
def : WriteRes<WriteJal, [Andes45ALU]>;
def : WriteRes<WriteJalr, [Andes45ALU]>;

// Integer multiplication
let Latency = 3 in {
def : WriteRes<WriteIMul, [Andes45MDU]>;
def : WriteRes<WriteIMul32, [Andes45MDU]>;
}

// Integer division
let Latency = 39, ReleaseAtCycles = [39] in {
def : WriteRes<WriteIDiv, [Andes45MDU]>;
def : WriteRes<WriteIDiv32, [Andes45MDU]>;
}

// Integer remainder
let Latency = 39, ReleaseAtCycles = [39] in {
def : WriteRes<WriteIRem, [Andes45MDU]>;
def : WriteRes<WriteIRem32, [Andes45MDU]>;
}

// Memory
let Latency = 5 in {
def : WriteRes<WriteLDB, [Andes45LSU]>;
def : WriteRes<WriteLDH, [Andes45LSU]>;
def : WriteRes<WriteFLD16, [Andes45LSU]>;
}

let Latency = 3 in {
def : WriteRes<WriteLDW, [Andes45LSU]>;
def : WriteRes<WriteLDD, [Andes45LSU]>;
def : WriteRes<WriteFLD32, [Andes45LSU]>;
def : WriteRes<WriteFLD64, [Andes45LSU]>;
}

let Latency = 1 in {
def : WriteRes<WriteSTB, [Andes45LSU]>;
def : WriteRes<WriteSTH, [Andes45LSU]>;
def : WriteRes<WriteSTW, [Andes45LSU]>;
def : WriteRes<WriteSTD, [Andes45LSU]>;
def : WriteRes<WriteFST16, [Andes45LSU]>;
def : WriteRes<WriteFST32, [Andes45LSU]>;
def : WriteRes<WriteFST64, [Andes45LSU]>;
}

// Atomic Memory
let Latency = 9 in {
def : WriteRes<WriteAtomicW, [Andes45LSU]>;
def : WriteRes<WriteAtomicD, [Andes45LSU]>;
def : WriteRes<WriteAtomicLDW, [Andes45LSU]>;
def : WriteRes<WriteAtomicLDD, [Andes45LSU]>;
}

let Latency = 3 in {
def : WriteRes<WriteAtomicSTW, [Andes45LSU]>;
def : WriteRes<WriteAtomicSTD, [Andes45LSU]>;
}

// FMAC
let Latency = 4 in  {
def : WriteRes<WriteFAdd16, [Andes45FMAC]>;
def : WriteRes<WriteFAdd32, [Andes45FMAC]>;
def : WriteRes<WriteFAdd64, [Andes45FMAC]>;
def : WriteRes<WriteFMul16, [Andes45FMAC]>;
def : WriteRes<WriteFMul32, [Andes45FMAC]>;
def : WriteRes<WriteFMul64, [Andes45FMAC]>;
def : WriteRes<WriteFMA16, [Andes45FMAC]>;
def : WriteRes<WriteFMA32, [Andes45FMAC]>;
def : WriteRes<WriteFMA64, [Andes45FMAC]>;
}

// FDIV
let Latency = 12, ReleaseAtCycles = [12] in
def : WriteRes<WriteFDiv16, [Andes45FDIV]>;
let Latency = 11, ReleaseAtCycles = [11] in
def : WriteRes<WriteFSqrt16, [Andes45FDIV]>;

let Latency = 19, ReleaseAtCycles = [19] in
def : WriteRes<WriteFDiv32, [Andes45FDIV]>;
let Latency = 18, ReleaseAtCycles = [18] in
def : WriteRes<WriteFSqrt32, [Andes45FDIV]>;

let Latency = 33, ReleaseAtCycles = [33] in
def : WriteRes<WriteFDiv64, [Andes45FDIV]>;
let Latency = 32, ReleaseAtCycles = [32] in
def : WriteRes<WriteFSqrt64, [Andes45FDIV]>;

// FMV
def : WriteRes<WriteFSGNJ16, [Andes45FMV]>;
def : WriteRes<WriteFSGNJ32, [Andes45FMV]>;
def : WriteRes<WriteFSGNJ64, [Andes45FMV]>;
def : WriteRes<WriteFMovF16ToI16, [Andes45FMV]>;
def : WriteRes<WriteFMovI16ToF16, [Andes45FMV]>;
def : WriteRes<WriteFMovF32ToI32, [Andes45FMV]>;
def : WriteRes<WriteFMovI32ToF32, [Andes45FMV]>;
def : WriteRes<WriteFMovF64ToI64, [Andes45FMV]>;
def : WriteRes<WriteFMovI64ToF64, [Andes45FMV]>;

// FMISC
let Latency = 2 in  {
def : WriteRes<WriteFMinMax16, [Andes45FMISC]>;
def : WriteRes<WriteFMinMax32, [Andes45FMISC]>;
def : WriteRes<WriteFMinMax64, [Andes45FMISC]>;
def : WriteRes<WriteFClass16, [Andes45FMISC]>;
def : WriteRes<WriteFClass32, [Andes45FMISC]>;
def : WriteRes<WriteFClass64, [Andes45FMISC]>;
def : WriteRes<WriteFCmp16, [Andes45FMISC]>;
def : WriteRes<WriteFCmp32, [Andes45FMISC]>;
def : WriteRes<WriteFCmp64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF16ToI32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF16ToI64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF32ToI32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF32ToI64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF64ToI32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF64ToI64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI32ToF16, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI32ToF32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI32ToF64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI64ToF16, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI64ToF32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtI64ToF64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF16ToF32, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF16ToF64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF32ToF16, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF32ToF64, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF64ToF16, [Andes45FMISC]>;
def : WriteRes<WriteFCvtF64ToF32, [Andes45FMISC]>;
}

// Bitmanip
// Zba extension
def : WriteRes<WriteSHXADD, [Andes45ALU]>;
def : WriteRes<WriteSHXADD32, [Andes45ALU]>;

// Zbb extension
def : WriteRes<WriteRotateImm, [Andes45ALU]>;
def : WriteRes<WriteRotateImm32, [Andes45ALU]>;
def : WriteRes<WriteRotateReg, [Andes45ALU]>;
def : WriteRes<WriteRotateReg32, [Andes45ALU]>;
def : WriteRes<WriteREV8, [Andes45ALU]>;
def : WriteRes<WriteORCB, [Andes45ALU]>;
def : WriteRes<WriteIMinMax, [Andes45ALU]>;

let Latency = 3 in  {
def : WriteRes<WriteCLZ, [Andes45ALU]>;
def : WriteRes<WriteCLZ32, [Andes45ALU]>;
def : WriteRes<WriteCTZ, [Andes45ALU]>;
def : WriteRes<WriteCTZ32, [Andes45ALU]>;
def : WriteRes<WriteCPOP, [Andes45ALU]>;
def : WriteRes<WriteCPOP32, [Andes45ALU]>;
}

// Zbc extension
let Latency = 3 in
def : WriteRes<WriteCLMUL, [Andes45ALU]>;

// Zbs extension
def : WriteRes<WriteSingleBit, [Andes45ALU]>;
def : WriteRes<WriteSingleBitImm, [Andes45ALU]>;
def : WriteRes<WriteBEXT, [Andes45ALU]>;
def : WriteRes<WriteBEXTI, [Andes45ALU]>;

// Others
def : WriteRes<WriteCSR, [Andes45CSR]>;
def : WriteRes<WriteNop, []>;

//===----------------------------------------------------------------------===//

// Bypass and advance
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadSFBJmp, 0>;
def : ReadAdvance<ReadSFBALU, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
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
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
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
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
def : ReadAdvance<ReadRotateImm, 1>;
def : ReadAdvance<ReadRotateImm32, 1>;
def : ReadAdvance<ReadRotateReg, 1>;
def : ReadAdvance<ReadRotateReg32, 1>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
def : ReadAdvance<ReadCLMUL, 0>;
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;
def : ReadAdvance<ReadCSR, 0>;

// RVV Scheduling

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [Andes45CSR]>;
def : WriteRes<WriteVSETIVLI, [Andes45CSR]>;
def : WriteRes<WriteVSETVL, [Andes45CSR]>;

// 7. Vector Loads and Stores
foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  // Unit-stride loads and stores
  defm "" : LMULWriteResMX<"WriteVLDE", [Andes45VLSU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVLDFF", [Andes45VLSU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSTE", [Andes45VLSU], mx, IsWorstCase>;

  // Mask loads and stores
  defm "" : LMULWriteResMX<"WriteVLDM", [Andes45VLSU], mx, IsWorstCase=!eq(mx, "M1")>;
  defm "" : LMULWriteResMX<"WriteVSTM", [Andes45VLSU], mx, IsWorstCase=!eq(mx, "M1")>;

  // Strided and indexed loads and stores
  foreach eew = [8, 16, 32, 64] in {
    defm "" : LMULWriteResMX<"WriteVLDS"  # eew, [Andes45VLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDUX" # eew, [Andes45VLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDOX" # eew, [Andes45VLSU], mx, IsWorstCase>;

    defm "" : LMULWriteResMX<"WriteVSTS"  # eew, [Andes45VLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTUX" # eew, [Andes45VLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTOX" # eew, [Andes45VLSU], mx, IsWorstCase>;
  }
}

// Segmented loads and stores
foreach mx = SchedMxList in {
  foreach nf=2-8 in {
    foreach eew = [8, 16, 32, 64] in {
      defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

      // Unit-stride segmented
      defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;

      // Strided/indexed segmented
      defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;

      // Indexed segmented
      defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" #eew, [Andes45VLSU], mx, IsWorstCase>;
    }
  }
}

// Whole register move/load/store
foreach LMul = [1, 2, 4, 8] in {
  def : WriteRes<!cast<SchedWrite>("WriteVLD" # LMul # "R"), [Andes45VLSU]>;
  def : WriteRes<!cast<SchedWrite>("WriteVST" # LMul # "R"), [Andes45VLSU]>;

  def : WriteRes<!cast<SchedWrite>("WriteVMov" # LMul # "V"), [Andes45VPERMUT]>;
}

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVIALUV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIALUX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIALUI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVExtV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUMV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUMX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICALUMI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICmpV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICmpX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVICmpI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMinMaxV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMinMaxX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMergeV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMergeX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMergeI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMovV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMovX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMovI", [Andes45VALU], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVShiftV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVShiftX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVShiftI", [Andes45VALU], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVIMulV", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMulX", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMulAddV", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIMulAddX", [Andes45VMAC], mx, IsWorstCase>;
}

// Widening
foreach mx = SchedMxListW in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxListW>.c;

  defm "" : LMULWriteResMX<"WriteVIWALUV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWALUX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWALUI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWMulV", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWMulX", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWMulAddV", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIWMulAddX", [Andes45VMAC], mx, IsWorstCase>;
}

// Vector Integer Division and Remainder
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [Andes45VDIV], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [Andes45VDIV], mx, sew, IsWorstCase>;
  }
}

// Narrowing Shift
foreach mx = SchedMxListW in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxListW>.c;

  defm "" : LMULWriteResMX<"WriteVNShiftV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVNShiftX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVNShiftI", [Andes45VALU], mx, IsWorstCase>;
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVSALUV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSALUX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSALUI", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVAALUV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVAALUX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSMulV", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSMulX", [Andes45VMAC], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSShiftV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSShiftX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSShiftI", [Andes45VALU], mx, IsWorstCase>;
}

// Narrowing
foreach mx = SchedMxListW in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxListW>.c;

  defm "" : LMULWriteResMX<"WriteVNClipV", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVNClipX", [Andes45VALU], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVNClipI", [Andes45VALU], mx, IsWorstCase>;
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFALUV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFALUF", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulF", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMulAddF", [Andes45VMAC], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFRecpV", [Andes45VFDIV], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjV", [Andes45VFMIS], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSgnjF", [Andes45VFMIS], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxV", [Andes45VFMIS], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFMinMaxF", [Andes45VFMIS], mx, sew, IsWorstCase>;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFCvtIToFV", [Andes45VFMIS], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVFCmpV", [Andes45VFMIS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFCmpF", [Andes45VFMIS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFClassV", [Andes45VFMIS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFMergeV", [Andes45VFMIS], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFMovV", [Andes45VFMIS], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [Andes45VFMIS], mx, IsWorstCase>;
}

// Widening
foreach mx = SchedMxListW in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListW>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtIToFV", [Andes45VFMIS], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFW in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxListFW>.c;

  defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [Andes45VFMIS], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWALUF", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulF", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddV", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWMulAddF", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWCvtFToFV", [Andes45VFMIS], mx, sew, IsWorstCase>;
  }
}

// Narrowing
foreach mx = SchedMxListW in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxListW>.c;

  defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [Andes45VFMIS], mx, IsWorstCase>;
}

foreach mx = SchedMxListFW in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {

    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListFW, isF=1>.c;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtIToFV", [Andes45VFMIS], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFNCvtFToFV", [Andes45VFMIS], mx, sew, IsWorstCase>;
  }
}

// Vector Floating-Point Division and Square Root
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV", [Andes45VFDIV], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF", [Andes45VFDIV], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [Andes45VFDIV], mx, sew, IsWorstCase>;
  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From", [Andes45VALU], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [Andes45VALU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, 0, 1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [Andes45VALU], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, 1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListF, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [Andes45VFMIS], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, 1, 1>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, 1>.c;

    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From", [Andes45VMAC], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [Andes45VMAC], mx, sew, IsWorstCase>;
  }
}

// 15. Vector Mask Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVMALUV", [Andes45VMASK], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMPopV", [Andes45VMASK], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMFFSV", [Andes45VMASK], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVMSFSV", [Andes45VMASK], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVIotaV", [Andes45VMASK], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVIdxV", [Andes45VMASK], mx, IsWorstCase>;
}

// 16. Vector Permutation Instructions
foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVSlideI", [Andes45VPERMUT], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVISlide1X", [Andes45VPERMUT], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVFSlide1F", [Andes45VPERMUT], mx, IsWorstCase>;

  defm "" : LMULWriteResMX<"WriteVSlideUpX", [Andes45VPERMUT], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVSlideDownX", [Andes45VPERMUT], mx, IsWorstCase>;
}

def : WriteRes<WriteVMovXS, [Andes45VPERMUT]>;
def : WriteRes<WriteVMovSX, [Andes45VPERMUT]>;

def : WriteRes<WriteVMovFS, [Andes45VPERMUT]>;
def : WriteRes<WriteVMovSF, [Andes45VPERMUT]>;

// Gather and Compress
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar IsWorstCase = Andes45IsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [Andes45VPERMUT], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherEI16VV", [Andes45VPERMUT], mx, sew, IsWorstCase>;
    defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [Andes45VPERMUT], mx, sew, IsWorstCase>;
  }
}

foreach mx = SchedMxList in {
  defvar IsWorstCase = Andes45IsWorstCaseMX<mx, SchedMxList>.c;

  defm "" : LMULWriteResMX<"WriteVRGatherVX", [Andes45VPERMUT], mx, IsWorstCase>;
  defm "" : LMULWriteResMX<"WriteVRGatherVI", [Andes45VPERMUT], mx, IsWorstCase>;
}

// Others
def : WriteRes<WriteRdVLENB, [Andes45CSR]>;

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
defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFMinMaxF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
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
defm : UnsupportedSchedXsf;
}
