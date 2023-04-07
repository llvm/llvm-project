//===- RISCVScheduleV.td - RISC-V Scheduling Definitions V -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Define scheduler resources associated with def operands.

defvar SchedMxList = ["M1", "M2", "M4", "M8", "MF2", "MF4", "MF8"];
// Used for widening and narrowing instructions as it doesn't contain M8.
defvar SchedMxListW = !listremove(SchedMxList, ["M8"]);
defvar SchedMxListFW = !listremove(SchedMxList, ["M8", "MF8"]);
// Used for widening floating-point Reduction as it doesn't contain MF8.
defvar SchedMxListFWRed = !listremove(SchedMxList, ["MF8"]);

class SchedSEWSet<string mx> {
  list<int> val = !cond(!eq(mx, "M1"):  [8, 16, 32, 64],
                        !eq(mx, "M2"):  [8, 16, 32, 64],
                        !eq(mx, "M4"):  [8, 16, 32, 64],
                        !eq(mx, "M8"):  [8, 16, 32, 64],
                        !eq(mx, "MF2"): [8, 16, 32],
                        !eq(mx, "MF4"): [8, 16],
                        !eq(mx, "MF8"): [8]);
}

// Define multiclasses to define SchedWrite, SchedRead,  WriteRes, and
// ReadAdvance for each (name, LMUL) pair and for each LMUL in each of the
// SchedMxList variants above. Each multiclass is responsible for defining
// a record that represents the WorseCase behavior for name.
multiclass LMULSchedWritesImpl<string name, list<string> MxList> {
  def name # "_WorstCase" : SchedWrite;
  foreach mx = MxList in {
    def name # "_" # mx : SchedWrite;
  }
}
multiclass LMULSchedReadsImpl<string name, list<string> MxList> {
  def name # "_WorstCase" : SchedRead;
  foreach mx = MxList in {
    def name # "_" # mx : SchedRead;
  }
}
multiclass LMULWriteResImpl<string name, list<ProcResourceKind> resources> {
  if !exists<SchedWrite>(name # "_WorstCase") then
    def : WriteRes<!cast<SchedWrite>(name # "_WorstCase"), resources>;
  foreach mx = SchedMxList in {
    if !exists<SchedWrite>(name # "_" # mx) then
      def : WriteRes<!cast<SchedWrite>(name # "_" # mx), resources>;
  }
}
multiclass LMULReadAdvanceImpl<string name, int val,
                               list<SchedWrite> writes = []> {
  if !exists<SchedRead>(name # "_WorstCase") then
    def : ReadAdvance<!cast<SchedRead>(name # "_WorstCase"), val, writes>;
  foreach mx = SchedMxList in {
    if !exists<SchedRead>(name # "_" # mx) then
      def : ReadAdvance<!cast<SchedRead>(name # "_" # mx), val, writes>;
  }
}

// Define multiclasses to define SchedWrite, SchedRead,  WriteRes, and
// ReadAdvance for each (name, LMUL, SEW) tuple for each LMUL in each of the
// SchedMxList variants above. Each multiclass is responsible for defining
// a record that represents the WorseCase behavior for name.
multiclass LMULSEWSchedWritesImpl<string name, list<string> MxList> {
  def name # "_WorstCase" : SchedWrite;
  foreach mx = MxList in {
    foreach sew = SchedSEWSet<mx>.val in
      def name # "_" # mx # "_E" # sew : SchedWrite;
  }
}
multiclass LMULSEWSchedReadsImpl<string name, list<string> MxList> {
  def name # "_WorstCase" : SchedRead;
  foreach mx = MxList in {
    foreach sew = SchedSEWSet<mx>.val in
      def name # "_" # mx # "_E" # sew : SchedRead;
  }
}
multiclass LMULSEWWriteResImpl<string name, list<ProcResourceKind> resources> {
  def : WriteRes<!cast<SchedWrite>(name # "_WorstCase"), resources>;
  foreach mx = SchedMxList in {
   foreach sew = SchedSEWSet<mx>.val in
     def : WriteRes<!cast<SchedWrite>(name # "_" # mx # "_E" # sew), resources>;
  }
}
multiclass LMULSEWReadAdvanceImpl<string name, int val,
                                  list<SchedWrite> writes = []> {
  def : ReadAdvance<!cast<SchedRead>(name # "_WorstCase"), val, writes>;
  foreach mx = SchedMxList in {
    foreach sew = SchedSEWSet<mx>.val in
      def : ReadAdvance<!cast<SchedRead>(name # "_" # mx # "_E" # sew), val, writes>;
  }
}
// Define classes to define list containing all SchedWrites for each (name, LMUL)
// pair for each LMUL in each of the SchedMxList variants above and name in
// argument `names`. These classes can be used to construct a list of existing
// definitions of writes corresponding to each (name, LMUL) pair, that are needed
// by the ReadAdvance. For example:
// ```
//   defm "" : LMULReadAdvance<"ReadVIALUX", 1,
//                             LMULSchedWriteList<["WriteVIMovVX"]>.value>;
// ```
class LMULSchedWriteListImpl<list<string> names, list<string> MxList> {
  list<SchedWrite> value = !foldl([]<SchedWrite>,
                                  !foreach(name, names,
                                    !foreach(mx, MxList, !cast<SchedWrite>(name # "_" # mx))),
                                  all, writes, !listconcat(all, writes));
}

multiclass LMULSchedWrites<string name> : LMULSchedWritesImpl<name, SchedMxList>;
multiclass LMULSchedReads<string name> : LMULSchedReadsImpl<name, SchedMxList>;
multiclass LMULWriteRes<string name, list<ProcResourceKind> resources>
  : LMULWriteResImpl<name, resources>;
multiclass LMULReadAdvance<string name, int val, list<SchedWrite> writes = []>
  : LMULReadAdvanceImpl<name, val, writes>;
class LMULSchedWriteList<list<string> names> : LMULSchedWriteListImpl<names, SchedMxList>;

multiclass LMULSEWSchedWrites<string name> : LMULSEWSchedWritesImpl<name, SchedMxList>;
multiclass LMULSEWSchedReads<string name> : LMULSEWSchedReadsImpl<name, SchedMxList>;
multiclass LMULSEWWriteRes<string name, list<ProcResourceKind> resources>
  : LMULSEWWriteResImpl<name, resources>;
multiclass LMULSEWReadAdvance<string name, int val, list<SchedWrite> writes = []>
  : LMULSEWReadAdvanceImpl<name, val, writes>;

multiclass LMULSchedWritesW<string name> : LMULSchedWritesImpl<name, SchedMxListW>;
multiclass LMULSchedReadsW<string name> : LMULSchedReadsImpl<name, SchedMxListW>;
multiclass LMULWriteResW<string name, list<ProcResourceKind> resources>
  : LMULWriteResImpl<name, resources>;
multiclass LMULReadAdvanceW<string name, int val, list<SchedWrite> writes = []>
  : LMULReadAdvanceImpl<name, val, writes>;
class LMULSchedWriteListW<list<string> names> : LMULSchedWriteListImpl<names, SchedMxListW>;

multiclass LMULSchedWritesFW<string name> : LMULSchedWritesImpl<name, SchedMxListFW>;
multiclass LMULSchedReadsFW<string name> : LMULSchedReadsImpl<name, SchedMxListFW>;
multiclass LMULWriteResFW<string name, list<ProcResourceKind> resources>
  : LMULWriteResImpl<name, resources>;
multiclass LMULReadAdvanceFW<string name, int val, list<SchedWrite> writes = []>
  : LMULReadAdvanceImpl<name, val, writes>;
class LMULSchedWriteListFW<list<string> names> : LMULSchedWriteListImpl<names, SchedMxListFW>;

multiclass LMULSchedWritesFWRed<string name> : LMULSchedWritesImpl<name, SchedMxListFWRed>;
multiclass LMULWriteResFWRed<string name, list<ProcResourceKind> resources>
  : LMULWriteResImpl<name, resources>;
class LMULSchedWriteListFWRed<list<string> names> : LMULSchedWriteListImpl<names, SchedMxListFWRed>;


// 3.6 Vector Byte Length vlenb
def WriteRdVLENB      : SchedWrite;

// 6. Configuration-Setting Instructions
def WriteVSETVLI      : SchedWrite;
def WriteVSETIVLI     : SchedWrite;
def WriteVSETVL       : SchedWrite;

// 7. Vector Loads and Stores
// 7.4. Vector Unit-Stride Instructions
defm "" : LMULSchedWrites<"WriteVLDE">;
defm "" : LMULSchedWrites<"WriteVSTE">;
// 7.4.1. Vector Unit-Strided Mask
defm "" : LMULSchedWrites<"WriteVLDM">;
defm "" : LMULSchedWrites<"WriteVSTM">;
// 7.5. Vector Strided Instructions
defm "" : LMULSchedWrites<"WriteVLDS8">;
defm "" : LMULSchedWrites<"WriteVLDS16">;
defm "" : LMULSchedWrites<"WriteVLDS32">;
defm "" : LMULSchedWrites<"WriteVLDS64">;
defm "" : LMULSchedWrites<"WriteVSTS8">;
defm "" : LMULSchedWrites<"WriteVSTS16">;
defm "" : LMULSchedWrites<"WriteVSTS32">;
defm "" : LMULSchedWrites<"WriteVSTS64">;
// 7.6. Vector Indexed Instructions
defm "" : LMULSchedWrites<"WriteVLDUX8">;
defm "" : LMULSchedWrites<"WriteVLDUX16">;
defm "" : LMULSchedWrites<"WriteVLDUX32">;
defm "" : LMULSchedWrites<"WriteVLDUX64">;
defm "" : LMULSchedWrites<"WriteVLDOX8">;
defm "" : LMULSchedWrites<"WriteVLDOX16">;
defm "" : LMULSchedWrites<"WriteVLDOX32">;
defm "" : LMULSchedWrites<"WriteVLDOX64">;
defm "" : LMULSchedWrites<"WriteVSTUX8">;
defm "" : LMULSchedWrites<"WriteVSTUX16">;
defm "" : LMULSchedWrites<"WriteVSTUX32">;
defm "" : LMULSchedWrites<"WriteVSTUX64">;
defm "" : LMULSchedWrites<"WriteVSTOX8">;
defm "" : LMULSchedWrites<"WriteVSTOX16">;
defm "" : LMULSchedWrites<"WriteVSTOX32">;
defm "" : LMULSchedWrites<"WriteVSTOX64">;
// 7.7. Vector Unit-stride Fault-Only-First Loads
defm "" : LMULSchedWrites<"WriteVLDFF">;
// 7.8. Vector Segment Instructions
foreach nf=2-8 in {
  foreach eew = [8, 16, 32, 64] in {
    defm "" : LMULSchedWrites<"WriteVLSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVSSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVLSEGFF" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVLSSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVSSSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVLUXSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVLOXSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVSUXSEG" # nf # e # eew>;
    defm "" : LMULSchedWrites<"WriteVSOXSEG" # nf # e # eew>;
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
defm "" : LMULSEWSchedWrites<"WriteVIDivV">;
defm "" : LMULSEWSchedWrites<"WriteVIDivX">;
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
defm "" : LMULSEWSchedWrites<"WriteVFDivV">;
defm "" : LMULSEWSchedWrites<"WriteVFDivF">;
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
defm "" : LMULSEWSchedWrites<"WriteVFSqrtV">;
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
// The latency of reduction is determined by the size of the read resource.
// The LMUL range of read resource(VS2) for reduction operantion is between
// MF8 and M8. Use the _From suffix to indicate the number of the
// LMUL from VS2.
// 14.1. Vector Single-Width Integer Reduction Instructions
defm "" : LMULSchedWrites<"WriteVIRedV_From">;
// 14.2. Vector Widening Integer Reduction Instructions
defm "" : LMULSchedWrites<"WriteVIWRedV_From">;
// 14.3. Vector Single-Width Floating-Point Reduction Instructions
defm "" : LMULSchedWrites<"WriteVFRedV_From">;
defm "" : LMULSchedWrites<"WriteVFRedOV_From">;
// 14.4. Vector Widening Floating-Point Reduction Instructions
defm "" : LMULSchedWritesFWRed<"WriteVFWRedV_From">;
defm "" : LMULSchedWritesFWRed<"WriteVFWRedOV_From">;

// 15. Vector Mask Instructions
// 15.1. Vector Mask-Register Logical Instructions
defm "" : LMULSchedWrites<"WriteVMALUV">;
// 15.2. Vector Mask Population Count
defm "" : LMULSchedWrites<"WriteVMPopV">;
// 15.3. Vector Find-First-Set Mask Bit
defm "" : LMULSchedWrites<"WriteVMFFSV">;
// 15.4. Vector Set-Before-First Mask Bit
// 15.5. Vector Set-Including-First Mask Bit
// 15.6. Vector Set-only-First Mask Bit
defm "" : LMULSchedWrites<"WriteVMSFSV">;
// 15.8. Vector Iota Instruction
defm "" : LMULSchedWrites<"WriteVMIotV">;
// 15.9. Vector Element Index Instruction
defm "" : LMULSchedWrites<"WriteVMIdxV">;

// 16. Vector Permutation Instructions
// 16.1. Integer Scalar Move Instructions
defm "" : LMULSchedWrites<"WriteVIMovVX">;
defm "" : LMULSchedWrites<"WriteVIMovXV">;
// 16.2. Floating-Point Scalar Move Instructions
defm "" : LMULSchedWrites<"WriteVFMovVF">;
defm "" : LMULSchedWrites<"WriteVFMovFV">;
// 16.3. Vector Slide Instructions
defm "" : LMULSchedWrites<"WriteVISlideX">;
defm "" : LMULSchedWrites<"WriteVISlideI">;
defm "" : LMULSchedWrites<"WriteVISlide1X">;
defm "" : LMULSchedWrites<"WriteVFSlide1F">;
// 16.4. Vector Register Gather Instructions
defm "" : LMULSEWSchedWrites<"WriteVRGatherVV">;
defm "" : LMULSchedWrites<"WriteVRGatherVX">;
defm "" : LMULSchedWrites<"WriteVRGatherVI">;
// 16.5. Vector Compress Instruction
defm "" : LMULSEWSchedWrites<"WriteVCompressV">;
// 16.6. Whole Vector Register Move
// These are already LMUL aware
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
def ReadVLDX : SchedRead;
def ReadVSTX : SchedRead;
// 7.4. Vector Unit-Stride Instructions
defm "" : LMULSchedReads<"ReadVSTEV">;
// 7.4.1. Vector Unit-Strided Mask
defm "" : LMULSchedReads<"ReadVSTM">;
// 7.5. Vector Strided Instructions
def ReadVLDSX : SchedRead;
def ReadVSTSX : SchedRead;
defm "" : LMULSchedReads<"ReadVSTS8V">;
defm "" : LMULSchedReads<"ReadVSTS16V">;
defm "" : LMULSchedReads<"ReadVSTS32V">;
defm "" : LMULSchedReads<"ReadVSTS64V">;
// 7.6. Vector Indexed Instructions
defm "" : LMULSchedReads<"ReadVLDUXV">;
defm "" : LMULSchedReads<"ReadVLDOXV">;
defm "" : LMULSchedReads<"ReadVSTUX8">;
defm "" : LMULSchedReads<"ReadVSTUX16">;
defm "" : LMULSchedReads<"ReadVSTUX32">;
defm "" : LMULSchedReads<"ReadVSTUX64">;
defm "" : LMULSchedReads<"ReadVSTUXV">;
defm "" : LMULSchedReads<"ReadVSTUX8V">;
defm "" : LMULSchedReads<"ReadVSTUX16V">;
defm "" : LMULSchedReads<"ReadVSTUX32V">;
defm "" : LMULSchedReads<"ReadVSTUX64V">;
defm "" : LMULSchedReads<"ReadVSTOX8">;
defm "" : LMULSchedReads<"ReadVSTOX16">;
defm "" : LMULSchedReads<"ReadVSTOX32">;
defm "" : LMULSchedReads<"ReadVSTOX64">;
defm "" : LMULSchedReads<"ReadVSTOXV">;
defm "" : LMULSchedReads<"ReadVSTOX8V">;
defm "" : LMULSchedReads<"ReadVSTOX16V">;
defm "" : LMULSchedReads<"ReadVSTOX32V">;
defm "" : LMULSchedReads<"ReadVSTOX64V">;
// 7.9. Vector Whole Register Instructions
// These are already LMUL aware
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
defm "" : LMULSEWSchedReads<"ReadVIDivV">;
defm "" : LMULSEWSchedReads<"ReadVIDivX">;
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
defm "" : LMULSEWSchedReads<"ReadVFDivV">;
defm "" : LMULSEWSchedReads<"ReadVFDivF">;
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
defm "" : LMULSEWSchedReads<"ReadVFSqrtV">;
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
defm "" : LMULSchedReads<"ReadVMALUV">;
// 15.2. Vector Mask Population Count
defm "" : LMULSchedReads<"ReadVMPopV">;
// 15.3. Vector Find-First-Set Mask Bit
defm "" : LMULSchedReads<"ReadVMFFSV">;
// 15.4. Vector Set-Before-First Mask Bit
// 15.5. Vector Set-Including-First Mask Bit
// 15.6. Vector Set-only-First Mask Bit
defm "" : LMULSchedReads<"ReadVMSFSV">;
// 15.8. Vector Iota Instruction
defm "" : LMULSchedReads<"ReadVMIotV">;

// 16. Vector Permutation Instructions
// 16.1. Integer Scalar Move Instructions
defm "" : LMULSchedReads<"ReadVIMovVX">;
defm "" : LMULSchedReads<"ReadVIMovXV">;
defm "" : LMULSchedReads<"ReadVIMovXX">;
// 16.2. Floating-Point Scalar Move Instructions
defm "" : LMULSchedReads<"ReadVFMovVF">;
defm "" : LMULSchedReads<"ReadVFMovFV">;
defm "" : LMULSchedReads<"ReadVFMovFX">;
// 16.3. Vector Slide Instructions
defm "" : LMULSchedReads<"ReadVISlideV">;
defm "" : LMULSchedReads<"ReadVISlideX">;
defm "" : LMULSchedReads<"ReadVFSlideV">;
defm "" : LMULSchedReads<"ReadVFSlideF">;
// 16.4. Vector Register Gather Instructions
defm "" : LMULSEWSchedReads<"ReadVRGatherVV_data">;
defm "" : LMULSEWSchedReads<"ReadVRGatherVV_index">;
defm "" : LMULSchedReads<"ReadVRGatherVX_data">;
defm "" : LMULSchedReads<"ReadVRGatherVX_index">;
defm "" : LMULSchedReads<"ReadVRGatherVI_data">;
// 16.5. Vector Compress Instruction
defm "" : LMULSEWSchedReads<"ReadVCompressV">;
// 16.6. Whole Vector Register Move
// These are already LMUL aware
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
defm "" : LMULWriteRes<"WriteVLDE", []>;
defm "" : LMULWriteRes<"WriteVSTE", []>;
defm "" : LMULWriteRes<"WriteVLDM", []>;
defm "" : LMULWriteRes<"WriteVSTM", []>;
defm "" : LMULWriteRes<"WriteVLDS8", []>;
defm "" : LMULWriteRes<"WriteVLDS16", []>;
defm "" : LMULWriteRes<"WriteVLDS32", []>;
defm "" : LMULWriteRes<"WriteVLDS64", []>;
defm "" : LMULWriteRes<"WriteVSTS8", []>;
defm "" : LMULWriteRes<"WriteVSTS16", []>;
defm "" : LMULWriteRes<"WriteVSTS32", []>;
defm "" : LMULWriteRes<"WriteVSTS64", []>;
defm "" : LMULWriteRes<"WriteVLDUX8", []>;
defm "" : LMULWriteRes<"WriteVLDUX16", []>;
defm "" : LMULWriteRes<"WriteVLDUX32", []>;
defm "" : LMULWriteRes<"WriteVLDUX64", []>;
defm "" : LMULWriteRes<"WriteVLDOX8", []>;
defm "" : LMULWriteRes<"WriteVLDOX16", []>;
defm "" : LMULWriteRes<"WriteVLDOX32", []>;
defm "" : LMULWriteRes<"WriteVLDOX64", []>;
defm "" : LMULWriteRes<"WriteVSTUX8", []>;
defm "" : LMULWriteRes<"WriteVSTUX16", []>;
defm "" : LMULWriteRes<"WriteVSTUX32", []>;
defm "" : LMULWriteRes<"WriteVSTUX64", []>;
defm "" : LMULWriteRes<"WriteVSTOX8", []>;
defm "" : LMULWriteRes<"WriteVSTOX16", []>;
defm "" : LMULWriteRes<"WriteVSTOX32", []>;
defm "" : LMULWriteRes<"WriteVSTOX64", []>;
defm "" : LMULWriteRes<"WriteVLDFF", []>;
// These are already LMUL aware
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
    defm "" : LMULWriteRes <"WriteVLSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVLSEGFF" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVSSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVLSSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVSSSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVLUXSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVLOXSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVSUXSEG" # nf # "e" # eew, []>;
    defm "" : LMULWriteRes <"WriteVSOXSEG" # nf # "e" # eew, []>;
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
defm "" : LMULSEWWriteRes<"WriteVIDivV", []>;
defm "" : LMULSEWWriteRes<"WriteVIDivX", []>;
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
defm "" : LMULSEWWriteRes<"WriteVFDivV", []>;
defm "" : LMULSEWWriteRes<"WriteVFDivF", []>;
defm "" : LMULWriteResFW<"WriteVFWMulV", []>;
defm "" : LMULWriteResFW<"WriteVFWMulF", []>;
defm "" : LMULWriteRes<"WriteVFMulAddV", []>;
defm "" : LMULWriteRes<"WriteVFMulAddF", []>;
defm "" : LMULWriteResFW<"WriteVFWMulAddV", []>;
defm "" : LMULWriteResFW<"WriteVFWMulAddF", []>;
defm "" : LMULSEWWriteRes<"WriteVFSqrtV", []>;
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
defm "" : LMULWriteRes<"WriteVIRedV_From", []>;
defm "" : LMULWriteRes<"WriteVIWRedV_From", []>;
defm "" : LMULWriteRes<"WriteVFRedV_From", []>;
defm "" : LMULWriteRes<"WriteVFRedOV_From", []>;
defm "" : LMULWriteResFWRed<"WriteVFWRedV_From", []>;
defm "" : LMULWriteResFWRed<"WriteVFWRedOV_From", []>;

// 15. Vector Mask Instructions
defm "" : LMULWriteRes<"WriteVMALUV", []>;
defm "" : LMULWriteRes<"WriteVMPopV", []>;
defm "" : LMULWriteRes<"WriteVMFFSV", []>;
defm "" : LMULWriteRes<"WriteVMSFSV", []>;
defm "" : LMULWriteRes<"WriteVMIotV", []>;
defm "" : LMULWriteRes<"WriteVMIdxV", []>;

// 16. Vector Permutation Instructions
defm "" : LMULWriteRes<"WriteVIMovVX", []>;
defm "" : LMULWriteRes<"WriteVIMovXV", []>;
defm "" : LMULWriteRes<"WriteVFMovVF", []>;
defm "" : LMULWriteRes<"WriteVFMovFV", []>;
defm "" : LMULWriteRes<"WriteVISlideX", []>;
defm "" : LMULWriteRes<"WriteVISlideI", []>;
defm "" : LMULWriteRes<"WriteVISlide1X", []>;
defm "" : LMULWriteRes<"WriteVFSlide1F", []>;
defm "" : LMULSEWWriteRes<"WriteVRGatherVV", []>;
defm "" : LMULWriteRes<"WriteVRGatherVX", []>;
defm "" : LMULWriteRes<"WriteVRGatherVI", []>;
defm "" : LMULSEWWriteRes<"WriteVCompressV", []>;
// These are already LMUL aware
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
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
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
// These are already LMUL aware
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
defm "" : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVIDivX", 0>;
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
defm "" : LMULSEWReadAdvance<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVFSqrtV", 0>;
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
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMIotV", 0>;

// 16. Vector Permutation Instructions
defm "" : LMULReadAdvance<"ReadVIMovVX", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXV", 0>;
defm "" : LMULReadAdvance<"ReadVIMovXX", 0>;
defm "" : LMULReadAdvance<"ReadVFMovVF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFV", 0>;
defm "" : LMULReadAdvance<"ReadVFMovFX", 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULReadAdvance<"ReadVGatherV", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// These are already LMUL aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;

} // Unsupported
} // UnsupportedSchedV
