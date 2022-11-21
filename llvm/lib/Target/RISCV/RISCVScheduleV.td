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

// Creates SchedWrite for each (name, LMUL) pair for LMUL in lmuls argument
multiclass LMULSchedWrites<string name, list<string> lmuls = SchedMxList> {
  foreach mx = lmuls in {
    def name # "_" # mx : SchedWrite;
  }
}

// Creates SchedRead for each (name, LMUL) pair for LMUL in lmuls argument
multiclass LMULSchedReads<string name, list<string> lmuls = SchedMxList> {
  foreach mx = lmuls in {
    def name # "_" # mx : SchedRead;
  }
}

// Creates WriteRes for each (name, LMUL, resources) tuple for LMUL
// in lmuls argument
multiclass LMULWriteRes<string name, list<ProcResourceKind> resources,
                        list<string> lmuls = SchedMxList> {
  foreach mx = lmuls in {
    def : WriteRes<!cast<SchedWrite>(name # "_" # mx), resources>;
  }
}

// Creates ReadAdvance for each (name, LMUL, val) tuple for LMUL
// in lmuls argument
multiclass LMULReadAdvance<string name, int val, list<string> lmuls = SchedMxList> {
  foreach mx = lmuls in {
    def : ReadAdvance<!cast<SchedRead>(name # "_" # mx), val>;
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
defm "" : LMULSchedWrites<"WriteVIWALUV", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVIWALUX", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVIWALUI", SchedMxListW>;
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
defm "" : LMULSchedWrites<"WriteVNShiftV", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVNShiftX", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVNShiftI", SchedMxListW>;
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
defm "" : LMULSchedWrites<"WriteVIWMulV", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVIWMulX", SchedMxListW>;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
defm "" : LMULSchedWrites<"WriteVIMulAddV">;
defm "" : LMULSchedWrites<"WriteVIMulAddX">;
// 11.14. Vector Widening Integer Multiply-Add Instructions
defm "" : LMULSchedWrites<"WriteVIWMulAddV", SchedMxListW>;
defm "" : LMULSchedWrites<"WriteVIWMulAddX", SchedMxListW>;
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
def WriteVSALUV       : SchedWrite;
def WriteVSALUX       : SchedWrite;
def WriteVSALUI       : SchedWrite;
// 12.2. Vector Single-Width Averaging Add and Subtract
def WriteVAALUV       : SchedWrite;
def WriteVAALUX       : SchedWrite;
// 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
def WriteVSMulV       : SchedWrite;
def WriteVSMulX       : SchedWrite;
// 12.4. Vector Single-Width Scaling Shift Instructions
def WriteVSShiftV     : SchedWrite;
def WriteVSShiftX     : SchedWrite;
def WriteVSShiftI     : SchedWrite;
// 12.5. Vector Narrowing Fixed-Point Clip Instructions
def WriteVNClipV      : SchedWrite;
def WriteVNClipX      : SchedWrite;
def WriteVNClipI      : SchedWrite;

// 13. Vector Floating-Point Instructions
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
def WriteVFALUV       : SchedWrite;
def WriteVFALUF       : SchedWrite;
// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
def WriteVFWALUV      : SchedWrite;
def WriteVFWALUF      : SchedWrite;
// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
def WriteVFMulV       : SchedWrite;
def WriteVFMulF       : SchedWrite;
def WriteVFDivV       : SchedWrite;
def WriteVFDivF       : SchedWrite;
// 13.5. Vector Widening Floating-Point Multiply
def WriteVFWMulV      : SchedWrite;
def WriteVFWMulF      : SchedWrite;
// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
def WriteVFMulAddV    : SchedWrite;
def WriteVFMulAddF    : SchedWrite;
// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
def WriteVFWMulAddV   : SchedWrite;
def WriteVFWMulAddF   : SchedWrite;
// 13.8. Vector Floating-Point Square-Root Instruction
def WriteVFSqrtV      : SchedWrite;
// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
def WriteVFRecpV      : SchedWrite;
// 13.11. Vector Floating-Point MIN/MAX Instructions
// 13.13. Vector Floating-Point Compare Instructions
def WriteVFCmpV       : SchedWrite;
def WriteVFCmpF       : SchedWrite;
// 13.12. Vector Floating-Point Sign-Injection Instructions
def WriteVFSgnjV      : SchedWrite;
def WriteVFSgnjF      : SchedWrite;
// 13.14. Vector Floating-Point Classify Instruction
def WriteVFClassV     : SchedWrite;
// 13.15. Vector Floating-Point Merge Instruction
def WriteVFMergeV     : SchedWrite;
// 13.16. Vector Floating-Point Move Instruction
def WriteVFMovV       : SchedWrite;
// 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
def WriteVFCvtIToFV   : SchedWrite;
def WriteVFCvtFToIV   : SchedWrite;
// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
def WriteVFWCvtIToFV  : SchedWrite;
def WriteVFWCvtFToIV  : SchedWrite;
def WriteVFWCvtFToFV  : SchedWrite;
// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
def WriteVFNCvtIToFV  : SchedWrite;
def WriteVFNCvtFToIV  : SchedWrite;
def WriteVFNCvtFToFV  : SchedWrite;

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
defm "" : LMULSchedReads<"ReadVIWALUV", SchedMxListW>;
defm "" : LMULSchedReads<"ReadVIWALUX", SchedMxListW>;
// 11.3. Vector Integer Extension
defm "" : LMULSchedReads<"ReadVExtV">;
// 11.4. Vector Integer Arithmetic with Carry or Borrow Instructions
defm "" : LMULSchedReads<"ReadVICALUV">;
defm "" : LMULSchedReads<"ReadVICALUX">;
// 11.6. Vector Single-Width Bit Shift Instructions
defm "" : LMULSchedReads<"ReadVShiftV">;
defm "" : LMULSchedReads<"ReadVShiftX">;
// 11.7. Vector Narrowing Integer Right Shift Instructions
defm "" : LMULSchedReads<"ReadVNShiftV", SchedMxListW>;
defm "" : LMULSchedReads<"ReadVNShiftX", SchedMxListW>;
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
defm "" : LMULSchedReads<"ReadVIWMulV", SchedMxListW>;
defm "" : LMULSchedReads<"ReadVIWMulX", SchedMxListW>;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
defm "" : LMULSchedReads<"ReadVIMulAddV">;
defm "" : LMULSchedReads<"ReadVIMulAddX">;
// 11.14. Vector Widening Integer Multiply-Add Instructions
defm "" : LMULSchedReads<"ReadVIWMulAddV", SchedMxListW>;
defm "" : LMULSchedReads<"ReadVIWMulAddX", SchedMxListW>;
// 11.15. Vector Integer Merge Instructions
defm "" : LMULSchedReads<"ReadVIMergeV">;
defm "" : LMULSchedReads<"ReadVIMergeX">;
// 11.16. Vector Integer Move Instructions
defm "" : LMULSchedReads<"ReadVIMovV">;
defm "" : LMULSchedReads<"ReadVIMovX">;

// 12. Vector Fixed-Point Arithmetic Instructions
// 12.1. Vector Single-Width Saturating Add and Subtract
def ReadVSALUV        : SchedRead;
def ReadVSALUX        : SchedRead;
// 12.2. Vector Single-Width Averaging Add and Subtract
def ReadVAALUV        : SchedRead;
def ReadVAALUX        : SchedRead;
// 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
def ReadVSMulV        : SchedRead;
def ReadVSMulX        : SchedRead;
// 12.4. Vector Single-Width Scaling Shift Instructions
def ReadVSShiftV      : SchedRead;
def ReadVSShiftX      : SchedRead;
// 12.5. Vector Narrowing Fixed-Point Clip Instructions
def ReadVNClipV       : SchedRead;
def ReadVNClipX       : SchedRead;

// 13. Vector Floating-Point Instructions
// 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
def ReadVFALUV        : SchedRead;
def ReadVFALUF        : SchedRead;
// 13.3. Vector Widening Floating-Point Add/Subtract Instructions
def ReadVFWALUV       : SchedRead;
def ReadVFWALUF       : SchedRead;
// 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
def ReadVFMulV        : SchedRead;
def ReadVFMulF        : SchedRead;
def ReadVFDivV        : SchedRead;
def ReadVFDivF        : SchedRead;
// 13.5. Vector Widening Floating-Point Multiply
def ReadVFWMulV       : SchedRead;
def ReadVFWMulF       : SchedRead;
// 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
def ReadVFMulAddV     : SchedRead;
def ReadVFMulAddF     : SchedRead;
// 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
def ReadVFWMulAddV    : SchedRead;
def ReadVFWMulAddF    : SchedRead;
// 13.8. Vector Floating-Point Square-Root Instruction
def ReadVFSqrtV       : SchedRead;
// 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
// 13.10. Vector Floating-Point Reciprocal Estimate Instruction
def ReadVFRecpV       : SchedRead;
// 13.11. Vector Floating-Point MIN/MAX Instructions
// 13.13. Vector Floating-Point Compare Instructions
def ReadVFCmpV        : SchedRead;
def ReadVFCmpF        : SchedRead;
// 13.12. Vector Floating-Point Sign-Injection Instructions
def ReadVFSgnjV       : SchedRead;
def ReadVFSgnjF       : SchedRead;
// 13.14. Vector Floating-Point Classify Instruction
def ReadVFClassV      : SchedRead;
// 13.15. Vector Floating-Point Merge Instruction
def ReadVFMergeV      : SchedRead;
def ReadVFMergeF      : SchedRead;
// 13.16. Vector Floating-Point Move Instruction
def ReadVFMovF        : SchedRead;
// 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
def ReadVFCvtIToFV    : SchedRead;
def ReadVFCvtFToIV    : SchedRead;
// 13.18. Widening Floating-Point/Integer Type-Convert Instructions
def ReadVFWCvtIToFV   : SchedRead;
def ReadVFWCvtFToIV   : SchedRead;
def ReadVFWCvtFToFV   : SchedRead;
// 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
def ReadVFNCvtIToFV   : SchedRead;
def ReadVFNCvtFToIV   : SchedRead;
def ReadVFNCvtFToFV   : SchedRead;

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
defm "" : LMULWriteRes<"WriteVIWALUV", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIWALUX", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIWALUI", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVExtV", []>;
defm "" : LMULWriteRes<"WriteVICALUV", []>;
defm "" : LMULWriteRes<"WriteVICALUX", []>;
defm "" : LMULWriteRes<"WriteVICALUI", []>;
defm "" : LMULWriteRes<"WriteVShiftV", []>;
defm "" : LMULWriteRes<"WriteVShiftX", []>;
defm "" : LMULWriteRes<"WriteVShiftI", []>;
defm "" : LMULWriteRes<"WriteVNShiftV", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVNShiftX", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVNShiftI", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVICmpV", []>;
defm "" : LMULWriteRes<"WriteVICmpX", []>;
defm "" : LMULWriteRes<"WriteVICmpI", []>;
defm "" : LMULWriteRes<"WriteVIMulV", []>;
defm "" : LMULWriteRes<"WriteVIMulX", []>;
defm "" : LMULWriteRes<"WriteVIDivV", []>;
defm "" : LMULWriteRes<"WriteVIDivX", []>;
defm "" : LMULWriteRes<"WriteVIWMulV", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIWMulX", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIMulAddV", []>;
defm "" : LMULWriteRes<"WriteVIMulAddX", []>;
defm "" : LMULWriteRes<"WriteVIWMulAddV", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIWMulAddX", [], SchedMxListW>;
defm "" : LMULWriteRes<"WriteVIMergeV", []>;
defm "" : LMULWriteRes<"WriteVIMergeX", []>;
defm "" : LMULWriteRes<"WriteVIMergeI", []>;
defm "" : LMULWriteRes<"WriteVIMovV", []>;
defm "" : LMULWriteRes<"WriteVIMovX", []>;
defm "" : LMULWriteRes<"WriteVIMovI", []>;

// 12. Vector Fixed-Point Arithmetic Instructions
def : WriteRes<WriteVSALUV, []>;
def : WriteRes<WriteVSALUX, []>;
def : WriteRes<WriteVSALUI, []>;
def : WriteRes<WriteVAALUV, []>;
def : WriteRes<WriteVAALUX, []>;
def : WriteRes<WriteVSMulV, []>;
def : WriteRes<WriteVSMulX, []>;
def : WriteRes<WriteVSShiftV, []>;
def : WriteRes<WriteVSShiftX, []>;
def : WriteRes<WriteVSShiftI, []>;
def : WriteRes<WriteVNClipV, []>;
def : WriteRes<WriteVNClipX, []>;
def : WriteRes<WriteVNClipI, []>;

// 13. Vector Floating-Point Instructions
def : WriteRes<WriteVFALUV, []>;
def : WriteRes<WriteVFALUF, []>;
def : WriteRes<WriteVFWALUV, []>;
def : WriteRes<WriteVFWALUF, []>;
def : WriteRes<WriteVFMulV, []>;
def : WriteRes<WriteVFMulF, []>;
def : WriteRes<WriteVFDivV, []>;
def : WriteRes<WriteVFDivF, []>;
def : WriteRes<WriteVFWMulV, []>;
def : WriteRes<WriteVFWMulF, []>;
def : WriteRes<WriteVFMulAddV, []>;
def : WriteRes<WriteVFMulAddF, []>;
def : WriteRes<WriteVFWMulAddV, []>;
def : WriteRes<WriteVFWMulAddF, []>;
def : WriteRes<WriteVFSqrtV, []>;
def : WriteRes<WriteVFRecpV, []>;
def : WriteRes<WriteVFCmpV, []>;
def : WriteRes<WriteVFCmpF, []>;
def : WriteRes<WriteVFSgnjV, []>;
def : WriteRes<WriteVFSgnjF, []>;
def : WriteRes<WriteVFClassV, []>;
def : WriteRes<WriteVFMergeV, []>;
def : WriteRes<WriteVFMovV, []>;
def : WriteRes<WriteVFCvtIToFV, []>;
def : WriteRes<WriteVFCvtFToIV, []>;
def : WriteRes<WriteVFWCvtIToFV, []>;
def : WriteRes<WriteVFWCvtFToIV, []>;
def : WriteRes<WriteVFWCvtFToFV, []>;
def : WriteRes<WriteVFNCvtIToFV, []>;
def : WriteRes<WriteVFNCvtFToIV, []>;
def : WriteRes<WriteVFNCvtFToFV, []>;

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
defm "" : LMULReadAdvance<"ReadVIWALUV", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVIWALUX", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVExtV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUV", 0>;
defm "" : LMULReadAdvance<"ReadVICALUX", 0>;
defm "" : LMULReadAdvance<"ReadVShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVShiftX", 0>;
defm "" : LMULReadAdvance<"ReadVNShiftV", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVNShiftX", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVICmpV", 0>;
defm "" : LMULReadAdvance<"ReadVICmpX", 0>;
defm "" : LMULReadAdvance<"ReadVIMulV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulX", 0>;
defm "" : LMULReadAdvance<"ReadVIDivV", 0>;
defm "" : LMULReadAdvance<"ReadVIDivX", 0>;
defm "" : LMULReadAdvance<"ReadVIWMulV", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVIWMulX", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm "" : LMULReadAdvance<"ReadVIWMulAddV", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVIWMulAddX", 0, SchedMxListW>;
defm "" : LMULReadAdvance<"ReadVIMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVIMergeX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovX", 0>;

// 12. Vector Fixed-Point Arithmetic Instructions
def : ReadAdvance<ReadVSALUV, 0>;
def : ReadAdvance<ReadVSALUX, 0>;
def : ReadAdvance<ReadVAALUV, 0>;
def : ReadAdvance<ReadVAALUX, 0>;
def : ReadAdvance<ReadVSMulV, 0>;
def : ReadAdvance<ReadVSMulX, 0>;
def : ReadAdvance<ReadVSShiftV, 0>;
def : ReadAdvance<ReadVSShiftX, 0>;
def : ReadAdvance<ReadVNClipV, 0>;
def : ReadAdvance<ReadVNClipX, 0>;

// 13. Vector Floating-Point Instructions
def : ReadAdvance<ReadVFALUV, 0>;
def : ReadAdvance<ReadVFALUF, 0>;
def : ReadAdvance<ReadVFWALUV, 0>;
def : ReadAdvance<ReadVFWALUF, 0>;
def : ReadAdvance<ReadVFMulV, 0>;
def : ReadAdvance<ReadVFMulF, 0>;
def : ReadAdvance<ReadVFDivV, 0>;
def : ReadAdvance<ReadVFDivF, 0>;
def : ReadAdvance<ReadVFWMulV, 0>;
def : ReadAdvance<ReadVFWMulF, 0>;
def : ReadAdvance<ReadVFMulAddV, 0>;
def : ReadAdvance<ReadVFMulAddF, 0>;
def : ReadAdvance<ReadVFWMulAddV, 0>;
def : ReadAdvance<ReadVFWMulAddF, 0>;
def : ReadAdvance<ReadVFSqrtV, 0>;
def : ReadAdvance<ReadVFRecpV, 0>;
def : ReadAdvance<ReadVFCmpV, 0>;
def : ReadAdvance<ReadVFCmpF, 0>;
def : ReadAdvance<ReadVFSgnjV, 0>;
def : ReadAdvance<ReadVFSgnjF, 0>;
def : ReadAdvance<ReadVFClassV, 0>;
def : ReadAdvance<ReadVFMergeV, 0>;
def : ReadAdvance<ReadVFMergeF, 0>;
def : ReadAdvance<ReadVFMovF, 0>;
def : ReadAdvance<ReadVFCvtIToFV, 0>;
def : ReadAdvance<ReadVFCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtIToFV, 0>;
def : ReadAdvance<ReadVFWCvtFToIV, 0>;
def : ReadAdvance<ReadVFWCvtFToFV, 0>;
def : ReadAdvance<ReadVFNCvtIToFV, 0>;
def : ReadAdvance<ReadVFNCvtFToIV, 0>;
def : ReadAdvance<ReadVFNCvtFToFV, 0>;

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
