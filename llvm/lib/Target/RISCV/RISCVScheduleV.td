//===-- RISCVScheduleV.td - RISCV Scheduling Definitions V -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Define scheduler resources associated with def operands.

// 3.6 Vector Byte Length vlenb
def WriteRdVLENB      : SchedWrite;

// 6. Configuration-Setting Instructions
def WriteVSETVLI      : SchedWrite;
def WriteVSETIVLI     : SchedWrite;
def WriteVSETVL       : SchedWrite;

// 7. Vector Loads and Stores
// 7.4. Vector Unit-Stride Instructions
def WriteVLDE8        : SchedWrite;
def WriteVLDE16       : SchedWrite;
def WriteVLDE32       : SchedWrite;
def WriteVLDE64       : SchedWrite;
def WriteVSTE8        : SchedWrite;
def WriteVSTE16       : SchedWrite;
def WriteVSTE32       : SchedWrite;
def WriteVSTE64       : SchedWrite;
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
def WriteVLDFF8       : SchedWrite;
def WriteVLDFF16      : SchedWrite;
def WriteVLDFF32      : SchedWrite;
def WriteVLDFF64      : SchedWrite;
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
def WriteVLD1R8       : SchedWrite;
def WriteVLD1R16      : SchedWrite;
def WriteVLD1R32      : SchedWrite;
def WriteVLD1R64      : SchedWrite;
def WriteVLD2R8       : SchedWrite;
def WriteVLD2R16      : SchedWrite;
def WriteVLD2R32      : SchedWrite;
def WriteVLD2R64      : SchedWrite;
def WriteVLD4R8       : SchedWrite;
def WriteVLD4R16      : SchedWrite;
def WriteVLD4R32      : SchedWrite;
def WriteVLD4R64      : SchedWrite;
def WriteVLD8R8       : SchedWrite;
def WriteVLD8R16      : SchedWrite;
def WriteVLD8R32      : SchedWrite;
def WriteVLD8R64      : SchedWrite;
def WriteVST1R        : SchedWrite;
def WriteVST2R        : SchedWrite;
def WriteVST4R        : SchedWrite;
def WriteVST8R        : SchedWrite;

// 11. Vector Integer Arithmetic Instructions
// 11.1. Vector Single-Width Integer Add and Subtract
// 11.5. Vector Bitwise Logical Instructions
def WriteVIALUV       : SchedWrite;
def WriteVIALUX       : SchedWrite;
def WriteVIALUI       : SchedWrite;
// 11.2. Vector Widening Integer Add/Subtract
def WriteVIWALUV      : SchedWrite;
def WriteVIWALUX      : SchedWrite;
def WriteVIWALUI      : SchedWrite;
// 11.3. Vector Integer Extension
def WriteVExtV        : SchedWrite;
// 11.4. Vector Integer Arithmetic with Carry or Borrow Instructions
def WriteVICALUV      : SchedWrite;
def WriteVICALUX      : SchedWrite;
def WriteVICALUI      : SchedWrite;
// 11.6. Vector Single-Width Bit Shift Instructions
def WriteVShiftV      : SchedWrite;
def WriteVShiftX      : SchedWrite;
def WriteVShiftI      : SchedWrite;
// 11.7. Vector Narrowing Integer Right Shift Instructions
def WriteVNShiftV     : SchedWrite;
def WriteVNShiftX     : SchedWrite;
def WriteVNShiftI     : SchedWrite;
// 11.8. Vector Integer Comparison Instructions
// 11.9. Vector Integer Min/Max Instructions
def WriteVICmpV       : SchedWrite;
def WriteVICmpX       : SchedWrite;
def WriteVICmpI       : SchedWrite;
// 11.10. Vector Single-Width Integer Multiply Instructions
def WriteVIMulV       : SchedWrite;
def WriteVIMulX       : SchedWrite;
// 11.11. Vector Integer Divide Instructions
def WriteVIDivV       : SchedWrite;
def WriteVIDivX       : SchedWrite;
// 11.12. Vector Widening Integer Multiply Instructions
def WriteVIWMulV      : SchedWrite;
def WriteVIWMulX      : SchedWrite;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
def WriteVIMulAddV    : SchedWrite;
def WriteVIMulAddX    : SchedWrite;
// 11.14. Vector Widening Integer Multiply-Add Instructions
def WriteVIWMulAddV   : SchedWrite;
def WriteVIWMulAddX   : SchedWrite;
// 11.15. Vector Integer Merge Instructions
def WriteVIMergeV     : SchedWrite;
def WriteVIMergeX     : SchedWrite;
def WriteVIMergeI     : SchedWrite;
// 11.16. Vector Integer Move Instructions
def WriteVIMovV       : SchedWrite;
def WriteVIMovX       : SchedWrite;
def WriteVIMovI       : SchedWrite;

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
def WriteVFCvtFToFV   : SchedWrite;
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
def ReadVSTE8V        : SchedRead;
def ReadVSTE16V       : SchedRead;
def ReadVSTE32V       : SchedRead;
def ReadVSTE64V       : SchedRead;
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
def ReadVIALUV        : SchedRead;
def ReadVIALUX        : SchedRead;
// 11.2. Vector Widening Integer Add/Subtract
def ReadVIWALUV       : SchedRead;
def ReadVIWALUX       : SchedRead;
// 11.3. Vector Integer Extension
def ReadVExtV         : SchedRead;
// 11.4. Vector Integer Arithmetic with Carry or Borrow Instructions
def ReadVIALUCV       : SchedRead;
def ReadVIALUCX       : SchedRead;
// 11.6. Vector Single-Width Bit Shift Instructions
def ReadVShiftV       : SchedRead;
def ReadVShiftX       : SchedRead;
// 11.7. Vector Narrowing Integer Right Shift Instructions
def ReadVNShiftV      : SchedRead;
def ReadVNShiftX      : SchedRead;
// 11.8. Vector Integer Comparison Instructions
// 11.9. Vector Integer Min/Max Instructions
def ReadVICmpV        : SchedRead;
def ReadVICmpX        : SchedRead;
// 11.10. Vector Single-Width Integer Multiply Instructions
def ReadVIMulV        : SchedRead;
def ReadVIMulX        : SchedRead;
// 11.11. Vector Integer Divide Instructions
def ReadVIDivV        : SchedRead;
def ReadVIDivX        : SchedRead;
// 11.12. Vector Widening Integer Multiply Instructions
def ReadVIWMulV       : SchedRead;
def ReadVIWMulX       : SchedRead;
// 11.13. Vector Single-Width Integer Multiply-Add Instructions
def ReadVIMulAddV     : SchedRead;
def ReadVIMulAddX     : SchedRead;
// 11.14. Vector Widening Integer Multiply-Add Instructions
def ReadVIWMulAddV    : SchedRead;
def ReadVIWMulAddX    : SchedRead;
// 11.15. Vector Integer Merge Instructions
def ReadVIMergeV      : SchedRead;
def ReadVIMergeX      : SchedRead;
// 11.16. Vector Integer Move Instructions
def ReadVIMovV        : SchedRead;
def ReadVIMovX        : SchedRead;

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
def : WriteRes<WriteVLDE8, []>;
def : WriteRes<WriteVLDE16, []>;
def : WriteRes<WriteVLDE32, []>;
def : WriteRes<WriteVLDE64, []>;
def : WriteRes<WriteVSTE8, []>;
def : WriteRes<WriteVSTE16, []>;
def : WriteRes<WriteVSTE32, []>;
def : WriteRes<WriteVSTE64, []>;
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
def : WriteRes<WriteVLDFF8, []>;
def : WriteRes<WriteVLDFF16, []>;
def : WriteRes<WriteVLDFF32, []>;
def : WriteRes<WriteVLDFF64, []>;
def : WriteRes<WriteVLD1R8, []>;
def : WriteRes<WriteVLD1R16, []>;
def : WriteRes<WriteVLD1R32, []>;
def : WriteRes<WriteVLD1R64, []>;
def : WriteRes<WriteVLD2R8, []>;
def : WriteRes<WriteVLD2R16, []>;
def : WriteRes<WriteVLD2R32, []>;
def : WriteRes<WriteVLD2R64, []>;
def : WriteRes<WriteVLD4R8, []>;
def : WriteRes<WriteVLD4R16, []>;
def : WriteRes<WriteVLD4R32, []>;
def : WriteRes<WriteVLD4R64, []>;
def : WriteRes<WriteVLD8R8, []>;
def : WriteRes<WriteVLD8R16, []>;
def : WriteRes<WriteVLD8R32, []>;
def : WriteRes<WriteVLD8R64, []>;
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

// 12. Vector Integer Arithmetic Instructions
def : WriteRes<WriteVIALUV, []>;
def : WriteRes<WriteVIALUX, []>;
def : WriteRes<WriteVIALUI, []>;
def : WriteRes<WriteVIWALUV, []>;
def : WriteRes<WriteVIWALUX, []>;
def : WriteRes<WriteVIWALUI, []>;
def : WriteRes<WriteVExtV, []>;
def : WriteRes<WriteVICALUV, []>;
def : WriteRes<WriteVICALUX, []>;
def : WriteRes<WriteVICALUI, []>;
def : WriteRes<WriteVShiftV, []>;
def : WriteRes<WriteVShiftX, []>;
def : WriteRes<WriteVShiftI, []>;
def : WriteRes<WriteVNShiftV, []>;
def : WriteRes<WriteVNShiftX, []>;
def : WriteRes<WriteVNShiftI, []>;
def : WriteRes<WriteVICmpV, []>;
def : WriteRes<WriteVICmpX, []>;
def : WriteRes<WriteVICmpI, []>;
def : WriteRes<WriteVIMulV, []>;
def : WriteRes<WriteVIMulX, []>;
def : WriteRes<WriteVIDivV, []>;
def : WriteRes<WriteVIDivX, []>;
def : WriteRes<WriteVIWMulV, []>;
def : WriteRes<WriteVIWMulX, []>;
def : WriteRes<WriteVIMulAddV, []>;
def : WriteRes<WriteVIMulAddX, []>;
def : WriteRes<WriteVIWMulAddV, []>;
def : WriteRes<WriteVIWMulAddX, []>;
def : WriteRes<WriteVIMergeV, []>;
def : WriteRes<WriteVIMergeX, []>;
def : WriteRes<WriteVIMergeI, []>;
def : WriteRes<WriteVIMovV, []>;
def : WriteRes<WriteVIMovX, []>;
def : WriteRes<WriteVIMovI, []>;

// 13. Vector Fixed-Point Arithmetic Instructions
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

// 14. Vector Floating-Point Instructions
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
def : WriteRes<WriteVFCvtFToFV, []>;
def : WriteRes<WriteVFWCvtIToFV, []>;
def : WriteRes<WriteVFWCvtFToIV, []>;
def : WriteRes<WriteVFWCvtFToFV, []>;
def : WriteRes<WriteVFNCvtIToFV, []>;
def : WriteRes<WriteVFNCvtFToIV, []>;
def : WriteRes<WriteVFNCvtFToFV, []>;

// 15. Vector Reduction Operations
def : WriteRes<WriteVIRedV, []>;
def : WriteRes<WriteVIWRedV, []>;
def : WriteRes<WriteVFRedV, []>;
def : WriteRes<WriteVFRedOV, []>;
def : WriteRes<WriteVFWRedV, []>;
def : WriteRes<WriteVFWRedOV, []>;

// 16. Vector Mask Instructions
def : WriteRes<WriteVMALUV, []>;
def : WriteRes<WriteVMPopV, []>;
def : WriteRes<WriteVMFFSV, []>;
def : WriteRes<WriteVMSFSV, []>;
def : WriteRes<WriteVMIotV, []>;
def : WriteRes<WriteVMIdxV, []>;

// 17. Vector Permutation Instructions
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
def : ReadAdvance<ReadVSTE8V, 0>;
def : ReadAdvance<ReadVSTE16V, 0>;
def : ReadAdvance<ReadVSTE32V, 0>;
def : ReadAdvance<ReadVSTE64V, 0>;
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

// 12. Vector Integer Arithmetic Instructions
def : ReadAdvance<ReadVIALUV, 0>;
def : ReadAdvance<ReadVIALUX, 0>;
def : ReadAdvance<ReadVIWALUV, 0>;
def : ReadAdvance<ReadVIWALUX, 0>;
def : ReadAdvance<ReadVExtV, 0>;
def : ReadAdvance<ReadVIALUCV, 0>;
def : ReadAdvance<ReadVIALUCX, 0>;
def : ReadAdvance<ReadVShiftV, 0>;
def : ReadAdvance<ReadVShiftX, 0>;
def : ReadAdvance<ReadVNShiftV, 0>;
def : ReadAdvance<ReadVNShiftX, 0>;
def : ReadAdvance<ReadVICmpV, 0>;
def : ReadAdvance<ReadVICmpX, 0>;
def : ReadAdvance<ReadVIMulV, 0>;
def : ReadAdvance<ReadVIMulX, 0>;
def : ReadAdvance<ReadVIDivV, 0>;
def : ReadAdvance<ReadVIDivX, 0>;
def : ReadAdvance<ReadVIWMulV, 0>;
def : ReadAdvance<ReadVIWMulX, 0>;
def : ReadAdvance<ReadVIMulAddV, 0>;
def : ReadAdvance<ReadVIMulAddX, 0>;
def : ReadAdvance<ReadVIWMulAddV, 0>;
def : ReadAdvance<ReadVIWMulAddX, 0>;
def : ReadAdvance<ReadVIMergeV, 0>;
def : ReadAdvance<ReadVIMergeX, 0>;
def : ReadAdvance<ReadVIMovV, 0>;
def : ReadAdvance<ReadVIMovX, 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
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

// 14. Vector Floating-Point Instructions
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
def : ReadAdvance<ReadVMALUV, 0>;
def : ReadAdvance<ReadVMPopV, 0>;
def : ReadAdvance<ReadVMFFSV, 0>;
def : ReadAdvance<ReadVMSFSV, 0>;
def : ReadAdvance<ReadVMIotV, 0>;

// 17. Vector Permutation Instructions
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
