//===-- RISCVScheduleV.td - RISCV Scheduling Definitions V -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Define scheduler resources associated with def operands.

defvar SchedMxList = ["UpperBound", "M1", "M2", "M4", "M8", "MF2", "MF4", "MF8"];
// Used for widening and narrowing instructions as it doesn't contain M8.
defvar SchedMxListW = ["UpperBound", "MF8", "MF4", "MF2", "M1", "M2", "M4"];
defvar SchedMxListFW = ["UpperBound", "MF4", "MF2", "M1", "M2", "M4"];

// Creates SchedWrite for each (name, LMUL) pair for LMUL in SchedMxList
multiclass LMULSchedWrites<string name> {
  foreach mx = SchedMxList in {
    def name # "_" # mx : SchedWrite;
  }
}

// Creates SchedWrite for each (name, LMUL) pair for LMUL in SchedMxListW
multiclass LMULSchedWritesW<string name> {
  foreach mx = SchedMxListW in {
    def name # "_" # mx : SchedWrite;
  }
}

// Creates SchedWrite for each (name, LMUL) pair for LMUL in SchedMxListFW
multiclass LMULSchedWritesFW<string name> {
  foreach mx = SchedMxListFW in {
    def name # "_" # mx : SchedWrite;
  }
}

// Creates SchedRead for each (name, LMUL) pair for LMUL in SchedMxList
multiclass LMULSchedReads<string name> {
  foreach mx = SchedMxList in {
    def name # "_" # mx : SchedRead;
  }
}

// Creates SchedRead for each (name, LMUL) pair for LMUL in SchedMxListW
multiclass LMULSchedReadsW<string name> {
  foreach mx = SchedMxListW in {
    def name # "_" # mx : SchedRead;
  }
}

// Creates SchedRead for each (name, LMUL) pair for LMUL in SchedMxListFW
multiclass LMULSchedReadsFW<string name> {
  foreach mx = SchedMxListFW in {
    def name # "_" # mx : SchedRead;
  }
}

// Creates WriteRes for each (name, LMUL, resources) tuple for LMUL
// in SchedMxList
multiclass LMULWriteRes<string name, list<ProcResourceKind> resources> {
  foreach mx = SchedMxList in {
    def : WriteRes<!cast<SchedWrite>(name # "_" # mx), resources>;
  }
}

// Creates WriteRes for each (name, LMUL, resources) tuple for LMUL
// in SchedMxListW
multiclass LMULWriteResW<string name, list<ProcResourceKind> resources> {
  foreach mx = SchedMxListW in {
    def : WriteRes<!cast<SchedWrite>(name # "_" # mx), resources>;
  }
}

// Creates WriteRes for each (name, LMUL, resources) tuple for LMUL
// in SchedMxListFW
multiclass LMULWriteResFW<string name, list<ProcResourceKind> resources> {
  foreach mx = SchedMxListFW in {
    def : WriteRes<!cast<SchedWrite>(name # "_" # mx), resources>;
  }
}

// Creates ReadAdvance for each (name, LMUL, val) tuple for LMUL
// in SchedMxList
multiclass LMULReadAdvance<string name, int val, list<SchedWrite> writes = []> {
  foreach mx = SchedMxList in {
    def : ReadAdvance<!cast<SchedRead>(name # "_" # mx), val, writes>;
  }
}

// Creates ReadAdvance for each (name, LMUL, val) tuple for LMUL
// in SchedMxListW
multiclass LMULReadAdvanceW<string name, int val, list<SchedWrite> writes = []> {
  foreach mx = SchedMxListW in {
    def : ReadAdvance<!cast<SchedRead>(name # "_" # mx), val, writes>;
  }
}

// Creates ReadAdvance for each (name, LMUL, val) tuple for LMUL
// in SchedMxListFW
multiclass LMULReadAdvanceFW<string name, int val, list<SchedWrite> writes = []> {
  foreach mx = SchedMxListFW in {
    def : ReadAdvance<!cast<SchedRead>(name # "_" # mx), val, writes>;
  }
}

// 3.6 Vector Byte Length vlenb
def WriteRdVLENB      : SchedWrite;

// 6. Configuration-Setting Instructions
def WriteVSETVLI      : SchedWrite;
def WriteVSETIVLI     : SchedWrite;
def WriteVSETVL       : SchedWrite;

// 7. Vector Loads and Stores
// 7.4. Vector Unit-Stride Instructions
def WriteVLDE         : SchedWrite;
def WriteVSTE         : SchedWrite;
// 7.4.1. Vector Unit-Strided Mask
def WriteVLDM         : SchedWrite;
def WriteVSTM         : SchedWrite;
// 7.5. Vector Strided Instructions
def WriteVLDS8        : SchedWrite;
def WriteVLDS16       : SchedWrite;
def WriteVLDS32       : SchedWrite;
def WriteVLDS64       : SchedWrite;
def WriteVSTS8        : SchedWrite;
def WriteVSTS16       : SchedWrite;
def WriteVSTS32       : SchedWrite;
def WriteVSTS64       : SchedWrite;
// 7.6. Vector Indexed Instructions
def WriteVLDUX8       : SchedWrite;
def WriteVLDUX16      : SchedWrite;
def WriteVLDUX32      : SchedWrite;
def WriteVLDUX64      : SchedWrite;
def WriteVLDOX8       : SchedWrite;
def WriteVLDOX16      : SchedWrite;
def WriteVLDOX32      : SchedWrite;
def WriteVLDOX64      : SchedWrite;
def WriteVSTUX8       : SchedWrite;
def WriteVSTUX16      : SchedWrite;
def WriteVSTUX32      : SchedWrite;
def WriteVSTUX64      : SchedWrite;
def WriteVSTOX8       : SchedWrite;
def WriteVSTOX16      : SchedWrite;
def WriteVSTOX32      : SchedWrite;
def WriteVSTOX64      : SchedWrite;
// 7.7. Vector Unit-stride Fault-Only-First Loads
def WriteVLDFF        : SchedWrite;
// 7.8. Vector Segment Instructions
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    def WriteVLSEG # nf # e # eew : SchedWrite;
    def WriteVSSEG # nf # e # eew : SchedWrite;
    def WriteVLSEGFF # nf # e # eew : SchedWrite;
    def WriteVLSSEG # nf # e # eew : SchedWrite;
    def WriteVSSSEG # nf # e # eew : SchedWrite;
    def WriteVLUXSEG # nf # e # eew : SchedWrite;
    def WriteVLOXSEG # nf # e # eew : SchedWrite;
    def WriteVSUXSEG # nf # e # eew : SchedWrite;
    def WriteVSOXSEG # nf # e # eew : SchedWrite;
  }
}
// 7.9. Vector Whole Register Instructions
def WriteVLD1R        : SchedWrite;
def WriteVLD2R        : SchedWrite;
def WriteVLD4R        : SchedWrite;
def WriteVLD8R        : SchedWrite;
def WriteVST1R        : SchedWrite;
def WriteVST2R        : SchedWrite;
def WriteVST4R        : SchedWrite;
def WriteVST8R        : SchedWrite;

// 11. Vector Integer Arithmetic Instructions
// 11.1. Vector Single-Width Integer Add and Subtract
// 11.5. Vector Bitwise Logical Instructions
defm "" : LMULSchedWrites<"WriteVIALUV">;
defm "" : LMULSchedWrites<"WriteVIALUX">;
defm "" : LMULSchedWrites<"WriteVIALUI">;
// 11.2. Vector Widening Integer Add/Subtract
defm "" : LMULSchedWritesW<"WriteVIWALUV">;
defm "" : LMULSchedWritesW<"WriteVIWALUX">;
defm "" : LMULSchedWritesW<"WriteVIWALUI">;
// 11.3. Vector Integer Extension
defm "" : LMULSchedWrites<"WriteVExtV">;
// 11.4. Vector Integer Arithmetic with Carry or Borrow Instructions
defm "" : LMULSchedWrites<"WriteVICALUV">;
defm "" : LMULSchedWrites<"WriteVICALUX">;
defm "" : LMULSchedWrites<"WriteVICALUI">;
// 11.6. Vector Single-Width Bit Shift Instructions
defm "" : LMULSchedWrites<"WriteVShiftV">;
defm "" : LMULSchedWrites<"WriteVShiftX">;
defm "" : LMULSchedWrites<"WriteVShiftI">;
// 11.7. Vector Narrowing Integer Right Shift Instructions
defm "" : LMULSchedWritesW<"WriteVNShiftV">;
defm "" : LMULSchedWritesW<"WriteVNShiftX">;
defm "" : LMULSchedWritesW<"WriteVNShiftI">;
// 11.8. Vector Integer Comparison Instructions
// 11.9. Vector Integer Min/Max Instructions
defm "" : LMULSchedWrites<"WriteVICmpV">;
defm "" : LMULSchedWrites<"WriteVICmpX">;
defm "" : LMULSchedWrites<"WriteVICmpI">;
// 11.10. Vector Single-Width Integer Multiply Instructions
defm "" : LMULSchedWrites<"WriteVIMulV">;
defm "" : LMULSchedWrites<"WriteVIMulX">;
// 11.11. Vector Integer Divide Instructions
defm "" : LMULSchedWrites<"WriteVIDivV">;
defm "" : LMULSchedWrites<"WriteVIDivX">;
// 11.12. Vector Widening Integer Multiply Instructions
defm "" : LMULSchedWritesW<"WriteVIWMulV">;
defm "" : LMULSchedWritesW<"WriteVIWMulX">;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
defm "" : LMULSchedWrites<"WriteVIMulAddV">;
defm "" : LMULSchedWrites<"WriteVIMulAddX">;
// 11.14. Vector Widening Integer Multiply-Add Instructions
defm "" : LMULSchedWritesW<"WriteVIWMulAddV">;
defm "" : LMULSchedWritesW<"WriteVIWMulAddX">;
// 11.15. Vector Integer Merge Instructions
defm "" : LMULSchedWrites<"WriteVIMergeV">;
defm "" : LMULSchedWrites<"WriteVIMergeX">;
defm "" : LMULSchedWrites<"WriteVIMergeI">;
// 11.16. Vector Integer Move Instructions
defm "" : LMULSchedWrites<"WriteVIMovV">;
defm "" : LMULSchedWrites<"WriteVIMovX">;
defm "" : LMULSchedWrites<"WriteVIMovI">;

// 12. Vector Fixed-Point Arithmetic Instructions
// 12.1. Vector Single-Width Saturating Add and Subtract
defm "" : LMULSchedWrites<"WriteVSALUV">;
defm "" : LMULSchedWrites<"WriteVSALUX">;
defm "" : LMULSchedWrites<"WriteVSALUI">;
// 12.2. Vector Single-Width Averaging Add and Subtract
defm "" : LMULSchedWrites<"WriteVAALUV">;
defm "" : LMULSchedWrites<"WriteVAALUX">;
// 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
defm "" : LMULSchedWrites<"WriteVSMulV">;
defm "" : LMULSchedWrites<"WriteVSMulX">;
// 12.4. Vector Single-Width Scaling Shift Instructions
defm "" : LMULSchedWrites<"WriteVSShiftV">;
defm "" : LMULSchedWrites<"WriteVSShiftX">;
defm "" : LMULSchedWrites<"WriteVSShiftI">;
// 12.5. Vector Narrowing Fixed-Point Clip Instructions
defm "" : LMULSchedWritesW<"WriteVNClipV">;
defm "" : LMULSchedWritesW<"WriteVNClipX">;
defm "" : LMULSchedWritesW<"WriteVNClipI">;

// 13. Vector Floating-Point Instructions
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm "" : LMULSchedWrites<"WriteVFALUV">;
defm "" : LMULSchedWrites<"WriteVFALUF">;
// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
defm "" : LMULSchedWritesFW<"WriteVFWALUV">;
defm "" : LMULSchedWritesFW<"WriteVFWALUF">;
// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm "" : LMULSchedWrites<"WriteVFMulV">;
defm "" : LMULSchedWrites<"WriteVFMulF">;
defm "" : LMULSchedWrites<"WriteVFDivV">;
defm "" : LMULSchedWrites<"WriteVFDivF">;
// 13.5. Vector Widening Floating-Point Multiply
defm "" : LMULSchedWritesFW<"WriteVFWMulV">;
defm "" : LMULSchedWritesFW<"WriteVFWMulF">;
// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
defm "" : LMULSchedWrites<"WriteVFMulAddV">;
defm "" : LMULSchedWrites<"WriteVFMulAddF">;
// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
defm "" : LMULSchedWritesFW<"WriteVFWMulAddV">;
defm "" : LMULSchedWritesFW<"WriteVFWMulAddF">;
// 13.8. Vector Floating-Point Square-Root Instruction
defm "" : LMULSchedWrites<"WriteVFSqrtV">;
// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
defm "" : LMULSchedWrites<"WriteVFRecpV">;
// 13.11. Vector Floating-Point MIN/MAX Instructions
// 13.13. Vector Floating-Point Compare Instructions
defm "" : LMULSchedWrites<"WriteVFCmpV">;
defm "" : LMULSchedWrites<"WriteVFCmpF">;
// 13.12. Vector Floating-Point Sign-Injection Instructions
defm "" : LMULSchedWrites<"WriteVFSgnjV">;
defm "" : LMULSchedWrites<"WriteVFSgnjF">;
// 13.14. Vector Floating-Point Classify Instruction
defm "" : LMULSchedWrites<"WriteVFClassV">;
// 13.15. Vector Floating-Point Merge Instruction
defm "" : LMULSchedWrites<"WriteVFMergeV">;
// 13.16. Vector Floating-Point Move Instruction
defm "" : LMULSchedWrites<"WriteVFMovV">;
// 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedWrites<"WriteVFCvtIToFV">;
defm "" : LMULSchedWrites<"WriteVFCvtFToIV">;
// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedWritesW<"WriteVFWCvtIToFV">;
defm "" : LMULSchedWritesFW<"WriteVFWCvtFToIV">;
defm "" : LMULSchedWritesFW<"WriteVFWCvtFToFV">;
// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedWritesFW<"WriteVFNCvtIToFV">;
defm "" : LMULSchedWritesW<"WriteVFNCvtFToIV">;
defm "" : LMULSchedWritesFW<"WriteVFNCvtFToFV">;

// 14. Vector Reduction Operations
// 14.1. Vector Single-Width Integer Reduction Instructions
def WriteVIRedV       : SchedWrite;
// 14.2. Vector Widening Integer Reduction Instructions
def WriteVIWRedV      : SchedWrite;
// 14.3. Vector Single-Width Floating-Point Reduction Instructions
def WriteVFRedV       : SchedWrite;
def WriteVFRedOV      : SchedWrite;
// 14.4. Vector Widening Floating-Point Reduction Instructions
def WriteVFWRedV      : SchedWrite;
def WriteVFWRedOV     : SchedWrite;

// 15. Vector Mask Instructions
// 15.1. Vector Mask-Register Logical Instructions
def WriteVMALUV       : SchedWrite;
// 15.2. Vector Mask Population Count
def WriteVMPopV       : SchedWrite;
// 15.3. Vector Find-First-Set Mask Bit
def WriteVMFFSV       : SchedWrite;
// 15.4. Vector Set-Before-First Mask Bit
// 15.5. Vector Set-Including-First Mask Bit
// 15.6. Vector Set-only-First Mask Bit
def WriteVMSFSV       : SchedWrite;
// 15.8. Vector Iota Instruction
def WriteVMIotV       : SchedWrite;
// 15.9. Vector Element Index Instruction
def WriteVMIdxV       : SchedWrite;

// 16. Vector Permutation Instructions
// 16.1. Integer Scalar Move Instructions
def WriteVIMovVX      : SchedWrite;
def WriteVIMovXV      : SchedWrite;
// 16.2. Floating-Point Scalar Move Instructions
def WriteVFMovVF      : SchedWrite;
def WriteVFMovFV      : SchedWrite;
// 16.3. Vector Slide Instructions
def WriteVISlideX     : SchedWrite;
def WriteVISlideI     : SchedWrite;
def WriteVISlide1X    : SchedWrite;
def WriteVFSlide1F    : SchedWrite;
// 16.4. Vector Register Gather Instructions
def WriteVGatherV     : SchedWrite;
def WriteVGatherX     : SchedWrite;
def WriteVGatherI     : SchedWrite;
// 16.5. Vector Compress Instruction
def WriteVCompressV   : SchedWrite;
// 16.6. Whole Vector Register Move
def WriteVMov1V       : SchedWrite;
def WriteVMov2V       : SchedWrite;
def WriteVMov4V       : SchedWrite;
def WriteVMov8V       : SchedWrite;

//===----------------------------------------------------------------------===//
/// Define scheduler resources associated with use operands.

// 6. Configuration-Setting Instructions
def ReadVSETVLI       : SchedRead;
def ReadVSETVL        : SchedRead;

// 7. Vector Loads and Stores
def ReadVLDX          : SchedRead;
def ReadVSTX          : SchedRead;
// 7.4. Vector Unit-Stride Instructions
def ReadVSTEV        : SchedRead;
// 7.4.1. Vector Unit-Strided Mask
def ReadVSTM          : SchedRead;
// 7.5. Vector Strided Instructions
def ReadVLDSX         : SchedRead;
def ReadVSTSX         : SchedRead;
def ReadVSTS8V        : SchedRead;
def ReadVSTS16V       : SchedRead;
def ReadVSTS32V       : SchedRead;
def ReadVSTS64V       : SchedRead;
// 7.6. Vector Indexed Instructions
def ReadVLDUXV        : SchedRead;
def ReadVLDOXV        : SchedRead;
def ReadVSTUX8        : SchedRead;
def ReadVSTUX16       : SchedRead;
def ReadVSTUX32       : SchedRead;
def ReadVSTUX64       : SchedRead;
def ReadVSTUXV        : SchedRead;
def ReadVSTUX8V       : SchedRead;
def ReadVSTUX16V      : SchedRead;
def ReadVSTUX32V      : SchedRead;
def ReadVSTUX64V      : SchedRead;
def ReadVSTOX8        : SchedRead;
def ReadVSTOX16       : SchedRead;
def ReadVSTOX32       : SchedRead;
def ReadVSTOX64       : SchedRead;
def ReadVSTOXV        : SchedRead;
def ReadVSTOX8V       : SchedRead;
def ReadVSTOX16V      : SchedRead;
def ReadVSTOX32V      : SchedRead;
def ReadVSTOX64V      : SchedRead;
// 7.9. Vector Whole Register Instructions
def ReadVST1R         : SchedRead;
def ReadVST2R         : SchedRead;
def ReadVST4R         : SchedRead;
def ReadVST8R         : SchedRead;

// 11. Vector Integer Arithmetic Instructions
// 11.1. Vector Single-Width Integer Add and Subtract
// 11.5. Vector Bitwise Logical Instructions
defm "" : LMULSchedReads<"ReadVIALUV">;
defm "" : LMULSchedReads<"ReadVIALUX">;
// 11.2. Vector Widening Integer Add/Subtract
defm "" : LMULSchedReadsW<"ReadVIWALUV">;
defm "" : LMULSchedReadsW<"ReadVIWALUX">;
// 11.3. Vector Integer Extension
defm "" : LMULSchedReads<"ReadVExtV">;
// 11.4. Vector Integer Arithmetic with Carry or Borrow Instructions
defm "" : LMULSchedReads<"ReadVICALUV">;
defm "" : LMULSchedReads<"ReadVICALUX">;
// 11.6. Vector Single-Width Bit Shift Instructions
defm "" : LMULSchedReads<"ReadVShiftV">;
defm "" : LMULSchedReads<"ReadVShiftX">;
// 11.7. Vector Narrowing Integer Right Shift Instructions
defm "" : LMULSchedReadsW<"ReadVNShiftV">;
defm "" : LMULSchedReadsW<"ReadVNShiftX">;
// 11.8. Vector Integer Comparison Instructions
// 11.9. Vector Integer Min/Max Instructions
defm "" : LMULSchedReads<"ReadVICmpV">;
defm "" : LMULSchedReads<"ReadVICmpX">;
// 11.10. Vector Single-Width Integer Multiply Instructions
defm "" : LMULSchedReads<"ReadVIMulV">;
defm "" : LMULSchedReads<"ReadVIMulX">;
// 11.11. Vector Integer Divide Instructions
defm "" : LMULSchedReads<"ReadVIDivV">;
defm "" : LMULSchedReads<"ReadVIDivX">;
// 11.12. Vector Widening Integer Multiply Instructions
defm "" : LMULSchedReadsW<"ReadVIWMulV">;
defm "" : LMULSchedReadsW<"ReadVIWMulX">;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
defm "" : LMULSchedReads<"ReadVIMulAddV">;
defm "" : LMULSchedReads<"ReadVIMulAddX">;
// 11.14. Vector Widening Integer Multiply-Add Instructions
defm "" : LMULSchedReadsW<"ReadVIWMulAddV">;
defm "" : LMULSchedReadsW<"ReadVIWMulAddX">;
// 11.15. Vector Integer Merge Instructions
defm "" : LMULSchedReads<"ReadVIMergeV">;
defm "" : LMULSchedReads<"ReadVIMergeX">;
// 11.16. Vector Integer Move Instructions
defm "" : LMULSchedReads<"ReadVIMovV">;
defm "" : LMULSchedReads<"ReadVIMovX">;

// 12. Vector Fixed-Point Arithmetic Instructions
// 12.1. Vector Single-Width Saturating Add and Subtract
defm "" : LMULSchedReads<"ReadVSALUV">;
defm "" : LMULSchedReads<"ReadVSALUX">;
// 12.2. Vector Single-Width Averaging Add and Subtract
defm "" : LMULSchedReads<"ReadVAALUV">;
defm "" : LMULSchedReads<"ReadVAALUX">;
// 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
defm "" : LMULSchedReads<"ReadVSMulV">;
defm "" : LMULSchedReads<"ReadVSMulX">;
// 12.4. Vector Single-Width Scaling Shift Instructions
defm "" : LMULSchedReads<"ReadVSShiftV">;
defm "" : LMULSchedReads<"ReadVSShiftX">;
// 12.5. Vector Narrowing Fixed-Point Clip Instructions
defm "" : LMULSchedReadsW<"ReadVNClipV">;
defm "" : LMULSchedReadsW<"ReadVNClipX">;

// 13. Vector Floating-Point Instructions
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
defm "" : LMULSchedReads<"ReadVFALUV">;
defm "" : LMULSchedReads<"ReadVFALUF">;
// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
defm "" : LMULSchedReadsFW<"ReadVFWALUV">;
defm "" : LMULSchedReadsFW<"ReadVFWALUF">;
// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
defm "" : LMULSchedReads<"ReadVFMulV">;
defm "" : LMULSchedReads<"ReadVFMulF">;
defm "" : LMULSchedReads<"ReadVFDivV">;
defm "" : LMULSchedReads<"ReadVFDivF">;
// 13.5. Vector Widening Floating-Point Multiply
defm "" : LMULSchedReadsFW<"ReadVFWMulV">;
defm "" : LMULSchedReadsFW<"ReadVFWMulF">;
// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
defm "" : LMULSchedReads<"ReadVFMulAddV">;
defm "" : LMULSchedReads<"ReadVFMulAddF">;
// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
defm "" : LMULSchedReadsFW<"ReadVFWMulAddV">;
defm "" : LMULSchedReadsFW<"ReadVFWMulAddF">;
// 13.8. Vector Floating-Point Square-Root Instruction
defm "" : LMULSchedReads<"ReadVFSqrtV">;
// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
defm "" : LMULSchedReads<"ReadVFRecpV">;
// 13.11. Vector Floating-Point MIN/MAX Instructions
// 13.13. Vector Floating-Point Compare Instructions
defm "" : LMULSchedReads<"ReadVFCmpV">;
defm "" : LMULSchedReads<"ReadVFCmpF">;
// 13.12. Vector Floating-Point Sign-Injection Instructions
defm "" : LMULSchedReads<"ReadVFSgnjV">;
defm "" : LMULSchedReads<"ReadVFSgnjF">;
// 13.14. Vector Floating-Point Classify Instruction
defm "" : LMULSchedReads<"ReadVFClassV">;
// 13.15. Vector Floating-Point Merge Instruction
defm "" : LMULSchedReads<"ReadVFMergeV">;
defm "" : LMULSchedReads<"ReadVFMergeF">;
// 13.16. Vector Floating-Point Move Instruction
defm "" : LMULSchedReads<"ReadVFMovF">;
// 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedReads<"ReadVFCvtIToFV">;
defm "" : LMULSchedReads<"ReadVFCvtFToIV">;
// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedReadsW<"ReadVFWCvtIToFV">;
defm "" : LMULSchedReadsFW<"ReadVFWCvtFToIV">;
defm "" : LMULSchedReadsFW<"ReadVFWCvtFToFV">;
// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
defm "" : LMULSchedReadsFW<"ReadVFNCvtIToFV">;
defm "" : LMULSchedReadsW<"ReadVFNCvtFToIV">;
defm "" : LMULSchedReadsFW<"ReadVFNCvtFToFV">;

// 14. Vector Reduction Operations
// 14.1. Vector Single-Width Integer Reduction Instructions
def ReadVIRedV        : SchedRead;
def ReadVIRedV0       : SchedRead;
// 14.2. Vector Widening Integer Reduction Instructions
def ReadVIWRedV       : SchedRead;
def ReadVIWRedV0      : SchedRead;
// 14.3. Vector Single-Width Floating-Point Reduction Instructions
def ReadVFRedV        : SchedRead;
def ReadVFRedV0       : SchedRead;
def ReadVFRedOV       : SchedRead;
def ReadVFRedOV0      : SchedRead;
// 14.4. Vector Widening Floating-Point Reduction Instructions
def ReadVFWRedV       : SchedRead;
def ReadVFWRedV0      : SchedRead;
def ReadVFWRedOV      : SchedRead;
def ReadVFWRedOV0     : SchedRead;

// 15. Vector Mask Instructions
// 15.1. Vector Mask-Register Logical Instructions
def ReadVMALUV        : SchedRead;
// 15.2. Vector Mask Population Count
def ReadVMPopV        : SchedRead;
// 15.3. Vector Find-First-Set Mask Bit
def ReadVMFFSV        : SchedRead;
// 15.4. Vector Set-Before-First Mask Bit
// 15.5. Vector Set-Including-First Mask Bit
// 15.6. Vector Set-only-First Mask Bit
def ReadVMSFSV        : SchedRead;
// 15.8. Vector Iota Instruction
def ReadVMIotV        : SchedRead;

// 16. Vector Permutation Instructions
// 16.1. Integer Scalar Move Instructions
def ReadVIMovVX       : SchedRead;
def ReadVIMovXV       : SchedRead;
def ReadVIMovXX       : SchedRead;
// 16.2. Floating-Point Scalar Move Instructions
def ReadVFMovVF       : SchedRead;
def ReadVFMovFV       : SchedRead;
def ReadVFMovFX       : SchedRead;
// 16.3. Vector Slide Instructions
def ReadVISlideV      : SchedRead;
def ReadVISlideX      : SchedRead;
def ReadVFSlideV      : SchedRead;
def ReadVFSlideF      : SchedRead;
// 16.4. Vector Register Gather Instructions
def ReadVGatherV      : SchedRead;
def ReadVGatherX      : SchedRead;
// 16.5. Vector Compress Instruction
def ReadVCompressV    : SchedRead;
// 16.6. Whole Vector Register Move
def ReadVMov1V        : SchedRead;
def ReadVMov2V        : SchedRead;
def ReadVMov4V        : SchedRead;
def ReadVMov8V        : SchedRead;

// Others
def ReadVMask         : SchedRead;

//===----------------------------------------------------------------------===//
/// Define default scheduler resources for V.

multiclass UnsupportedSchedV {
let Unsupported = true in {

// 3.6 Vector Byte Length vlenb
def : WriteRes<WriteRdVLENB, []>;

// 6. Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, []>;
def : WriteRes<WriteVSETIVLI, []>;
def : WriteRes<WriteVSETVL, []>;

// 7. Vector Loads and Stores
def : WriteRes<WriteVLDE, []>;
def : WriteRes<WriteVSTE, []>;
def : WriteRes<WriteVLDM, []>;
def : WriteRes<WriteVSTM, []>;
def : WriteRes<WriteVLDS8, []>;
def : WriteRes<WriteVLDS16, []>;
def : WriteRes<WriteVLDS32, []>;
def : WriteRes<WriteVLDS64, []>;
def : WriteRes<WriteVSTS8, []>;
def : WriteRes<WriteVSTS16, []>;
def : WriteRes<WriteVSTS32, []>;
def : WriteRes<WriteVSTS64, []>;
def : WriteRes<WriteVLDUX8, []>;
def : WriteRes<WriteVLDUX16, []>;
def : WriteRes<WriteVLDUX32, []>;
def : WriteRes<WriteVLDUX64, []>;
def : WriteRes<WriteVLDOX8, []>;
def : WriteRes<WriteVLDOX16, []>;
def : WriteRes<WriteVLDOX32, []>;
def : WriteRes<WriteVLDOX64, []>;
def : WriteRes<WriteVSTUX8, []>;
def : WriteRes<WriteVSTUX16, []>;
def : WriteRes<WriteVSTUX32, []>;
def : WriteRes<WriteVSTUX64, []>;
def : WriteRes<WriteVSTOX8, []>;
def : WriteRes<WriteVSTOX16, []>;
def : WriteRes<WriteVSTOX32, []>;
def : WriteRes<WriteVSTOX64, []>;
def : WriteRes<WriteVLDFF, []>;
def : WriteRes<WriteVLD1R, []>;
def : WriteRes<WriteVLD2R, []>;
def : WriteRes<WriteVLD4R, []>;
def : WriteRes<WriteVLD8R, []>;
def : WriteRes<WriteVST1R, []>;
def : WriteRes<WriteVST2R, []>;
def : WriteRes<WriteVST4R, []>;
def : WriteRes<WriteVST8R, []>;
// Vector Segment Loads and Stores
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    def : WriteRes <!cast<SchedWrite>("WriteVLSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVLSEGFF" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVSSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVLSSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVSSSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVLUXSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVLOXSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVSUXSEG" # nf # "e" # eew), []>;
    def : WriteRes <!cast<SchedWrite>("WriteVSOXSEG" # nf # "e" # eew), []>;
  }
}

// 11. Vector Integer Arithmetic Instructions
defm "" : LMULWriteRes<"WriteVIALUV", []>;
defm "" : LMULWriteRes<"WriteVIALUX", []>;
defm "" : LMULWriteRes<"WriteVIALUI", []>;
defm "" : LMULWriteResW<"WriteVIWALUV", []>;
defm "" : LMULWriteResW<"WriteVIWALUX", []>;
defm "" : LMULWriteResW<"WriteVIWALUI", []>;
defm "" : LMULWriteRes<"WriteVExtV", []>;
defm "" : LMULWriteRes<"WriteVICALUV", []>;
defm "" : LMULWriteRes<"WriteVICALUX", []>;
defm "" : LMULWriteRes<"WriteVICALUI", []>;
defm "" : LMULWriteRes<"WriteVShiftV", []>;
defm "" : LMULWriteRes<"WriteVShiftX", []>;
defm "" : LMULWriteRes<"WriteVShiftI", []>;
defm "" : LMULWriteResW<"WriteVNShiftV", []>;
defm "" : LMULWriteResW<"WriteVNShiftX", []>;
defm "" : LMULWriteResW<"WriteVNShiftI", []>;
defm "" : LMULWriteRes<"WriteVICmpV", []>;
defm "" : LMULWriteRes<"WriteVICmpX", []>;
defm "" : LMULWriteRes<"WriteVICmpI", []>;
defm "" : LMULWriteRes<"WriteVIMulV", []>;
defm "" : LMULWriteRes<"WriteVIMulX", []>;
defm "" : LMULWriteRes<"WriteVIDivV", []>;
defm "" : LMULWriteRes<"WriteVIDivX", []>;
defm "" : LMULWriteResW<"WriteVIWMulV", []>;
defm "" : LMULWriteResW<"WriteVIWMulX", []>;
defm "" : LMULWriteRes<"WriteVIMulAddV", []>;
defm "" : LMULWriteRes<"WriteVIMulAddX", []>;
defm "" : LMULWriteResW<"WriteVIWMulAddV", []>;
defm "" : LMULWriteResW<"WriteVIWMulAddX", []>;
defm "" : LMULWriteRes<"WriteVIMergeV", []>;
defm "" : LMULWriteRes<"WriteVIMergeX", []>;
defm "" : LMULWriteRes<"WriteVIMergeI", []>;
defm "" : LMULWriteRes<"WriteVIMovV", []>;
defm "" : LMULWriteRes<"WriteVIMovX", []>;
defm "" : LMULWriteRes<"WriteVIMovI", []>;

// 12. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULWriteRes<"WriteVSALUV", []>;
defm "" : LMULWriteRes<"WriteVSALUX", []>;
defm "" : LMULWriteRes<"WriteVSALUI", []>;
defm "" : LMULWriteRes<"WriteVAALUV", []>;
defm "" : LMULWriteRes<"WriteVAALUX", []>;
defm "" : LMULWriteRes<"WriteVSMulV", []>;
defm "" : LMULWriteRes<"WriteVSMulX", []>;
defm "" : LMULWriteRes<"WriteVSShiftV", []>;
defm "" : LMULWriteRes<"WriteVSShiftX", []>;
defm "" : LMULWriteRes<"WriteVSShiftI", []>;
defm "" : LMULWriteResW<"WriteVNClipV", []>;
defm "" : LMULWriteResW<"WriteVNClipX", []>;
defm "" : LMULWriteResW<"WriteVNClipI", []>;

// 13. Vector Floating-Point Instructions
defm "" : LMULWriteRes<"WriteVFALUV", []>;
defm "" : LMULWriteRes<"WriteVFALUF", []>;
defm "" : LMULWriteResFW<"WriteVFWALUV", []>;
defm "" : LMULWriteResFW<"WriteVFWALUF", []>;
defm "" : LMULWriteRes<"WriteVFMulV", []>;
defm "" : LMULWriteRes<"WriteVFMulF", []>;
defm "" : LMULWriteRes<"WriteVFDivV", []>;
defm "" : LMULWriteRes<"WriteVFDivF", []>;
defm "" : LMULWriteResFW<"WriteVFWMulV", []>;
defm "" : LMULWriteResFW<"WriteVFWMulF", []>;
defm "" : LMULWriteRes<"WriteVFMulAddV", []>;
defm "" : LMULWriteRes<"WriteVFMulAddF", []>;
defm "" : LMULWriteResFW<"WriteVFWMulAddV", []>;
defm "" : LMULWriteResFW<"WriteVFWMulAddF", []>;
defm "" : LMULWriteRes<"WriteVFSqrtV", []>;
defm "" : LMULWriteRes<"WriteVFRecpV", []>;
defm "" : LMULWriteRes<"WriteVFCmpV", []>;
defm "" : LMULWriteRes<"WriteVFCmpF", []>;
defm "" : LMULWriteRes<"WriteVFSgnjV", []>;
defm "" : LMULWriteRes<"WriteVFSgnjF", []>;
defm "" : LMULWriteRes<"WriteVFClassV", []>;
defm "" : LMULWriteRes<"WriteVFMergeV", []>;
defm "" : LMULWriteRes<"WriteVFMovV", []>;
defm "" : LMULWriteRes<"WriteVFCvtIToFV", []>;
defm "" : LMULWriteRes<"WriteVFCvtFToIV", []>;
defm "" : LMULWriteResW<"WriteVFWCvtIToFV", []>;
defm "" : LMULWriteResFW<"WriteVFWCvtFToIV", []>;
defm "" : LMULWriteResFW<"WriteVFWCvtFToFV", []>;
defm "" : LMULWriteResFW<"WriteVFNCvtIToFV", []>;
defm "" : LMULWriteResW<"WriteVFNCvtFToIV", []>;
defm "" : LMULWriteResFW<"WriteVFNCvtFToFV", []>;

// 14. Vector Reduction Operations
def : WriteRes<WriteVIRedV, []>;
def : WriteRes<WriteVIWRedV, []>;
def : WriteRes<WriteVFRedV, []>;
def : WriteRes<WriteVFRedOV, []>;
def : WriteRes<WriteVFWRedV, []>;
def : WriteRes<WriteVFWRedOV, []>;

// 15. Vector Mask Instructions
def : WriteRes<WriteVMALUV, []>;
def : WriteRes<WriteVMPopV, []>;
def : WriteRes<WriteVMFFSV, []>;
def : WriteRes<WriteVMSFSV, []>;
def : WriteRes<WriteVMIotV, []>;
def : WriteRes<WriteVMIdxV, []>;

// 16. Vector Permutation Instructions
def : WriteRes<WriteVIMovVX, []>;
def : WriteRes<WriteVIMovXV, []>;
def : WriteRes<WriteVFMovVF, []>;
def : WriteRes<WriteVFMovFV, []>;
def : WriteRes<WriteVISlideX, []>;
def : WriteRes<WriteVISlideI, []>;
def : WriteRes<WriteVISlide1X, []>;
def : WriteRes<WriteVFSlide1F, []>;
def : WriteRes<WriteVGatherV, []>;
def : WriteRes<WriteVGatherX, []>;
def : WriteRes<WriteVGatherI, []>;
def : WriteRes<WriteVCompressV, []>;
def : WriteRes<WriteVMov1V, []>;
def : WriteRes<WriteVMov2V, []>;
def : WriteRes<WriteVMov4V, []>;
def : WriteRes<WriteVMov8V, []>;

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 0>;
def : ReadAdvance<ReadVSETVL, 0>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
def : ReadAdvance<ReadVSTEV, 0>;
def : ReadAdvance<ReadVSTM, 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
def : ReadAdvance<ReadVSTS8V, 0>;
def : ReadAdvance<ReadVSTS16V, 0>;
def : ReadAdvance<ReadVSTS32V, 0>;
def : ReadAdvance<ReadVSTS64V, 0>;
def : ReadAdvance<ReadVLDUXV, 0>;
def : ReadAdvance<ReadVLDOXV, 0>;
def : ReadAdvance<ReadVSTUXV, 0>;
def : ReadAdvance<ReadVSTUX8, 0>;
def : ReadAdvance<ReadVSTUX16, 0>;
def : ReadAdvance<ReadVSTUX32, 0>;
def : ReadAdvance<ReadVSTUX64, 0>;
def : ReadAdvance<ReadVSTUX8V, 0>;
def : ReadAdvance<ReadVSTUX16V, 0>;
def : ReadAdvance<ReadVSTUX32V, 0>;
def : ReadAdvance<ReadVSTUX64V, 0>;
def : ReadAdvance<ReadVSTOX8, 0>;
def : ReadAdvance<ReadVSTOX16, 0>;
def : ReadAdvance<ReadVSTOX32, 0>;
def : ReadAdvance<ReadVSTOX64, 0>;
def : ReadAdvance<ReadVSTOXV, 0>;
def : ReadAdvance<ReadVSTOX8V, 0>;
def : ReadAdvance<ReadVSTOX16V, 0>;
def : ReadAdvance<ReadVSTOX32V, 0>;
def : ReadAdvance<ReadVSTOX64V, 0>;
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 11. Vector Integer Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVIALUV", 0>;
defm "" : LMULReadAdvance<"ReadVIALUX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm "" : LMULReadAdvance<"ReadVExtV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUX", 0>;
defm "" : LMULReadAdvance<"ReadVShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm "" : LMULReadAdvance<"ReadVICmpV", 0>;
defm "" : LMULReadAdvance<"ReadVICmpX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulX", 0>;
defm "" : LMULReadAdvance<"ReadVIDivV", 0>;
defm "" : LMULReadAdvance<"ReadVIDivX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm "" : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovX", 0>;

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
defm "" : LMULReadAdvance<"ReadVFALUV", 0>;
defm "" : LMULReadAdvance<"ReadVFALUF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFDivV", 0>;
defm "" : LMULReadAdvance<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULReadAdvance<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

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
def : ReadAdvance<ReadVMALUV, 0>;
def : ReadAdvance<ReadVMPopV, 0>;
def : ReadAdvance<ReadVMFFSV, 0>;
def : ReadAdvance<ReadVMSFSV, 0>;
def : ReadAdvance<ReadVMIotV, 0>;

// 16. Vector Permutation Instructions
def : ReadAdvance<ReadVIMovVX, 0>;
def : ReadAdvance<ReadVIMovXV, 0>;
def : ReadAdvance<ReadVIMovXX, 0>;
def : ReadAdvance<ReadVFMovVF, 0>;
def : ReadAdvance<ReadVFMovFV, 0>;
def : ReadAdvance<ReadVFMovFX, 0>;
def : ReadAdvance<ReadVISlideV, 0>;
def : ReadAdvance<ReadVISlideX, 0>;
def : ReadAdvance<ReadVFSlideV, 0>;
def : ReadAdvance<ReadVFSlideF, 0>;
def : ReadAdvance<ReadVGatherV, 0>;
def : ReadAdvance<ReadVGatherX, 0>;
def : ReadAdvance<ReadVCompressV, 0>;
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

} // Unsupported
} // UnsupportedSchedV
