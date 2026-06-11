//===-- SISchedule.td - SI Scheduling definitions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MachineModel definitions for Southern Islands (SI)
//
//===----------------------------------------------------------------------===//

def : PredicateProlog<[{
  const SIInstrInfo *TII =
    static_cast<const SIInstrInfo*>(SchedModel->getInstrInfo());
  (void)TII;
}]>;

def WriteBranch : SchedWrite;
def WriteExport : SchedWrite;
def WriteLDS    : SchedWrite;
def WriteSALU   : SchedWrite;
def WriteSMEM   : SchedWrite;
def WriteVMEM   : SchedWrite;
def WriteBarrier : SchedWrite;

def MIVGPRRead  : SchedRead;
def MIMFMARead  : SchedRead;

// Normal 16 or 32 bit VALU instructions
def Write32Bit         : SchedWrite;
// Conversion to or from F32 (but not converting F64 to or from F32)
def WriteFloatCvt      : SchedWrite;
// F16 or F32 transcendental instructions (these are quarter rate)
def WriteTrans32       : SchedWrite;
// Other quarter rate VALU instructions
def WriteQuarterRate32 : SchedWrite;

def WriteFloatFMA   : SchedWrite;

// Slow quarter rate f64 instruction.
def WriteDouble : SchedWrite;

// half rate f64 instruction (same as v_add_f64)
def WriteDoubleAdd  : SchedWrite;

// Conversion to or from f64 instruction
def WriteDoubleCvt  : SchedWrite;

// F64 "transcendental" (actually only reciprocal and/or square root)
// instructions
def WriteTrans64    : SchedWrite;

// Half rate 64-bit instructions.
def Write64Bit : SchedWrite;

// Integer multiplications.
def WriteIntMul : SchedWrite;

// mAI multipass instructions.
def Write2PassMAI  : SchedWrite;
def Write4PassMAI  : SchedWrite;
def Write8PassMAI  : SchedWrite;
def Write16PassMAI : SchedWrite;
def Write4PassDGEMM : SchedWrite;
def Write8PassDGEMM : SchedWrite;
def Write16PassDGEMM : SchedWrite;

// WMMA/SWMMA instructions
def WriteXDL2PassWMMA : SchedWrite;
def WriteXDL4PassWMMA : SchedWrite;
def Write4PassWMMA : SchedWrite;
def Write8PassWMMA : SchedWrite;
def Write16PassWMMA : SchedWrite;

// Scalar float instructions
def WriteSFPU : SchedWrite;

// F16 or F32 pseudo scalar transcendental instructions
def WritePseudoScalarTrans : SchedWrite;

// RepeatRate = 2 variants
def Trans32Repeat2 : SchedWrite;
def Write64BitRepeat2 : SchedWrite;
def Write32BitRepeat2 : SchedWrite;
def WriteDPPRepeat2 : SchedWrite;
def WriteSwapRepeat2 : SchedWrite;

// RepeatRate = 4 variants
def IntMulRepeat4 : SchedWrite;
def Write32BitRepeat4 : SchedWrite;

// RepeatRate = 8 variants
def Write32BitRepeat8 : SchedWrite;

// RepeatRate = 16 variants
def Write32BitRepeat16 : SchedWrite;

// RepeatRate = 32 variants
def Trans64Repeat32 : SchedWrite;
def DoubleRepeat32 : SchedWrite;

// Some instructions have multiple explicit defs. If we provide actual 
// SchedWrites for each of the explicit defs, then the instruction will 
// be encoded as having multiple MicroOps. The scheduler treats instructions
// with multiple MicroOps as taking multiple cycles to issue. To avoid this
// unwanted scheduling behavior, we can associate defs with DummyWrite, which
// don't have any MicroOps
def WriteVALUDummy : SchedWrite;
def WriteSALUDummy : SchedWrite;

// FIXME: Should there be a class for instructions which are VALU
// instructions and have VALU rates, but write to the SALU (i.e. VOPC
// instructions)

class SISchedMachineModel : SchedMachineModel {
  let CompleteModel = 1;
  // MicroOpBufferSize = 1 means that instructions will always be added
  // the ready queue when they become available.  This exposes them
  // to the register pressure analysis.
  let MicroOpBufferSize = 1;
  let IssueWidth = 1;
  let PostRAScheduler = 1;

  // FIXME:Approximate 2 * branch cost.  Try to hack around bad
  // early-ifcvt heuristics. These need improvement to avoid the OOE
  // heuristics.
  int MispredictPenalty = 20;
}

def SIFullSpeedModel : SISchedMachineModel;
def SIQuarterSpeedModel : SISchedMachineModel;
def SIDPFullSpeedModel : SISchedMachineModel;
def SIDPGFX942FullSpeedModel : SISchedMachineModel;
def SIDPGFX950FullSpeedModel : SISchedMachineModel;
def GFX10SpeedModel : SISchedMachineModel;
def GFX11SpeedModel : SISchedMachineModel;
def GFX12SpeedModel : SISchedMachineModel;
def GFX1250SpeedModel : SISchedMachineModel;

// XXX: Are the resource counts correct?
def HWBranch : ProcResource<1> {
  let BufferSize = 1;
}
def HWExport : ProcResource<1> {
  let BufferSize = 1;
}
def HWLGKM   : ProcResource<1> {
  let BufferSize = 1;
}
def HWSALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWVMEM   : ProcResource<1> {
  let BufferSize = 1;
}
def HWVALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWTransVALU : ProcResource<1> { // Transcendental VALU
  let BufferSize = 1;
}
def HWRC   : ProcResource<1> { // Register destination cache
  let BufferSize = 1;
}
def HWXDL   : ProcResource<1> { // MFMA CU
  let BufferSize = 0;
}

class HWWriteRes<SchedWrite write, list<ProcResourceKind> resources,
                 int latency> : WriteRes<write, resources> {
  let Latency = latency;
}

class HWVALUWriteRes<SchedWrite write, int latency> :
  HWWriteRes<write, [HWVALU], latency>;

class UnsupportedWriteRes<SchedWrite write> : WriteRes<write, []> {
  let Unsupported = 1;
}

def PredMIReadVGPR : SchedPredicate<[{TII->hasVGPRUses(*MI)}]>;

def MIReadVGPR : SchedReadVariant<[
      SchedVar<PredMIReadVGPR, [MIVGPRRead]>,
      SchedVar<NoSchedPred, [ReadDefault]>]>;

// The latency numbers are taken from AMD Accelerated Parallel Processing
// guide. They may not be accurate.

// The latency values are 1 / (operations / cycle) / 4.
multiclass SICommonWriteRes {

  let RetireOOO = 1 in { // llvm-mca specific flag
  def : HWWriteRes<WriteBranch,  [HWBranch], 8>;
  def : HWWriteRes<WriteExport,  [HWExport], 4>;
  def : HWWriteRes<WriteLDS,     [HWLGKM],   5>; // Can be between 2 and 64
  def : HWWriteRes<WriteSALU,    [HWSALU],   1>;
  def : HWWriteRes<WriteSMEM,    [HWLGKM],   5>;
  def : HWWriteRes<WriteVMEM,    [HWVMEM],   80>;
  def : HWWriteRes<WriteBarrier, [HWBranch], 500>; // XXX: Guessed ???

  def : HWVALUWriteRes<Write32Bit,         1>;
  def : HWVALUWriteRes<WriteFloatCvt,      4>;
  def : HWVALUWriteRes<WriteTrans32,       4>;
  def : HWVALUWriteRes<WriteQuarterRate32, 4>;

  let ReleaseAtCycles = [4] in
  def : HWVALUWriteRes<Write4PassDGEMM,    4>;
  let ReleaseAtCycles = [8] in
  def : HWVALUWriteRes<Write8PassDGEMM,    8>;
  let ReleaseAtCycles = [16] in
  def : HWVALUWriteRes<Write16PassDGEMM,  16>;

  let ReleaseAtCycles = [2] in
  def : HWWriteRes<Write2PassMAI,  [HWXDL], 2>;
  let ReleaseAtCycles = [4] in
  def : HWWriteRes<Write4PassMAI,  [HWXDL], 4>;
  let ReleaseAtCycles = [8] in
  def : HWWriteRes<Write8PassMAI,  [HWXDL], 8>;
  let ReleaseAtCycles = [16] in
  def : HWWriteRes<Write16PassMAI, [HWXDL], 16>;

  let NumMicroOps = 0 in {
  def : HWWriteRes<WriteVALUDummy, [HWVALU],  1>;
  def : HWWriteRes<WriteSALUDummy, [HWSALU],  1>;
  }

  def : UnsupportedWriteRes<WriteSFPU>;
  def : UnsupportedWriteRes<WritePseudoScalarTrans>;
  } // End RetireOOO = 1

  def : ReadAdvance<MIVGPRRead, -2>;

  // Technically mfma reads can be from 0 to 4 cycles but that does not make
  // sense to model because its register setup is huge. In particular if we
  // properly model read advance as -2 for a vgpr read it will result in a
  // bad scheduling of acc writes before that mfma. To avoid it we would
  // need to consume 2 or 4 more vgprs to be initialized before the acc
  // write sequence. Just assume worst case here.
  def : ReadAdvance<MIMFMARead, -4>;
}

def PredIsVGPR32Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32}]>;
def PredIsVGPR64Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32}]>;
def WriteCopy : SchedWriteVariant<[
    SchedVar<PredIsVGPR32Copy, [Write32Bit]>,
    SchedVar<PredIsVGPR64Copy, [Write64Bit]>,
    SchedVar<NoSchedPred, [WriteSALU]>]>;

// Check if any matrix inputs are interpreted as f8 in an f8f6f4 mfma
// instruction.
def PredIsF8_MFMA_SCALE : SchedPredicate<[{
  TII->getNamedOperand(*MI, AMDGPU::OpName::cbsz)->getImm() <= AMDGPU::MFMAScaleFormats::FP8_E5M2 ||
  TII->getNamedOperand(*MI, AMDGPU::OpName::blgp)->getImm() <= AMDGPU::MFMAScaleFormats::FP8_E5M2
}]>;

let SchedModel = SIFullSpeedModel in {

defm : SICommonWriteRes;

let RetireOOO = 1 in { // llvm-mca specific flag
def : HWVALUWriteRes<Write64Bit,       2>;
def : HWVALUWriteRes<WriteIntMul,      4>;
def : HWVALUWriteRes<WriteFloatFMA,    1>;
def : HWVALUWriteRes<WriteDouble,      4>;
def : HWVALUWriteRes<WriteDoubleAdd,   2>;
def : HWVALUWriteRes<WriteDoubleCvt,   4>;
def : HWVALUWriteRes<WriteTrans64,     4>;
} // End RetireOOO = 1

def : InstRW<[WriteCopy], (instrs COPY)>;

} // End SchedModel = SIFullSpeedModel

let SchedModel = SIQuarterSpeedModel in {

defm : SICommonWriteRes;

let RetireOOO = 1 in { // llvm-mca specific flag
def : HWVALUWriteRes<Write64Bit,       2>;
def : HWVALUWriteRes<WriteIntMul,      4>;
def : HWVALUWriteRes<WriteFloatFMA,    16>;
def : HWVALUWriteRes<WriteDouble,      16>;
def : HWVALUWriteRes<WriteDoubleAdd,    8>;
def : HWVALUWriteRes<WriteDoubleCvt,    4>;
def : HWVALUWriteRes<WriteTrans64,     16>;
} // End RetireOOO = 1

def : InstRW<[WriteCopy], (instrs COPY)>;
def : InstRW<[Write64Bit, MIReadVGPR], (instregex "^V_ACCVGPR_WRITE_B32_e64$")>;
def : InstRW<[Write2PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._4X4X")>;
def : InstRW<[Write8PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._16X16X")>;
def : InstRW<[Write16PassMAI, MIMFMARead], (instregex "^V_MFMA_..._32X32X")>;

}  // End SchedModel = SIQuarterSpeedModel

let SchedModel = SIDPFullSpeedModel in {

defm : SICommonWriteRes;

let RetireOOO = 1 in { // llvm-mca specific flag
def : HWVALUWriteRes<WriteFloatFMA,    1>;
def : HWVALUWriteRes<WriteDouble,      1>;
def : HWVALUWriteRes<WriteDoubleAdd,   1>;
def : HWVALUWriteRes<WriteDoubleCvt,   1>;
def : HWVALUWriteRes<WriteTrans64,     4>;
def : HWVALUWriteRes<WriteIntMul,      1>;
def : HWVALUWriteRes<Write64Bit,       1>;
} // End RetireOOO = 1

def : InstRW<[WriteCopy], (instrs COPY)>;
def : InstRW<[Write64Bit], (instregex "^V_ACCVGPR_WRITE_B32_e64$")>;
def : InstRW<[Write2PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_4X4X")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X")>;
def : InstRW<[Write16PassMAI,  MIMFMARead], (instregex "^V_MFMA_.32_32X32X")>;
def : InstRW<[Write4PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_4X4X")>;
def : InstRW<[Write8PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_16X16X")>;

} // End SchedModel = SIDPFullSpeedModel

let SchedModel = SIDPGFX942FullSpeedModel in {

defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA,    1>;
def : HWVALUWriteRes<WriteDouble,      1>;
def : HWVALUWriteRes<WriteDoubleAdd,   1>;
def : HWVALUWriteRes<WriteDoubleCvt,   1>;
def : HWVALUWriteRes<WriteTrans64,     4>;
def : HWVALUWriteRes<WriteIntMul,      1>;
def : HWVALUWriteRes<Write64Bit,       1>;

def : InstRW<[WriteCopy], (instrs COPY)>;
def : InstRW<[Write64Bit], (instregex "^V_ACCVGPR_WRITE_B32_e64$")>;
def : InstRW<[Write2PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_4X4X")>;

def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X8X")>;
def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X16")>;
def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X32")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X[14][FBI]")>;

def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X4XF")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X8")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X16")>;
def : InstRW<[Write16PassMAI,  MIMFMARead], (instregex "^V_MFMA_.32_32X32X[124][FBI]")>;

def : InstRW<[Write4PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_4X4X")>;
def : InstRW<[Write8PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_16X16X")>;

def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_SMFMAC_.32_16X16X")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_SMFMAC_.32_32X32X")>;

} // End SchedModel = SIDPGFX942FullSpeedModel


let SchedModel = SIDPGFX950FullSpeedModel in {
defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA,    1>;
def : HWVALUWriteRes<WriteDouble,      1>;
def : HWVALUWriteRes<WriteDoubleAdd,   1>;
def : HWVALUWriteRes<WriteDoubleCvt,   1>;
def : HWVALUWriteRes<WriteTrans64,     4>;
def : HWVALUWriteRes<WriteIntMul,      1>;
def : HWVALUWriteRes<Write64Bit,       1>;

def : InstRW<[WriteCopy], (instrs COPY)>;
def : InstRW<[Write64Bit], (instregex "^V_ACCVGPR_WRITE_B32_e64$")>;
def : InstRW<[Write2PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_4X4X")>;

def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X8X")>;
def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X16")>;
def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X32")>;
def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X64")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_16X16X[14][FBI]")>;

def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X4XF")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X8")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X16")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_MFMA_.32_32X32X32_")>;
def : InstRW<[Write16PassMAI,  MIMFMARead], (instregex "^V_MFMA_.32_32X32X[124][FBI]")>;

def : InstRW<[Write4PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_4X4X")>;
def : InstRW<[Write16PassDGEMM, MIMFMARead], (instregex "^V_MFMA_.64_16X16X")>;

def : InstRW<[Write4PassMAI,   MIMFMARead], (instregex "^V_SMFMAC_.32_16X16X")>;
def : InstRW<[Write8PassMAI,   MIMFMARead], (instregex "^V_SMFMAC_.32_32X32X")>;


// If either matrix format is f8, the instruction takes 2x as many
// cycles. TODO: This isn't reflected in MCA.
def WriteMFMAScale_16X16X128_F8F6F4 : SchedWriteVariant<[
    SchedVar<PredIsF8_MFMA_SCALE, [Write8PassMAI]>,
    SchedVar<NoSchedPred, [Write4PassMAI]>]>;
def WriteMFMAScale_32X32X64_F8F6F4 : SchedWriteVariant<[
    SchedVar<PredIsF8_MFMA_SCALE, [Write16PassMAI]>,
    SchedVar<NoSchedPred, [Write8PassMAI]>]>;

def : InstRW<[WriteMFMAScale_16X16X128_F8F6F4, MIMFMARead],
       (instregex "^V_MFMA(_SCALE)?_.32_16X16X128_F8F6F4")>;
def : InstRW<[WriteMFMAScale_32X32X64_F8F6F4,  MIMFMARead],
        (instregex "^V_MFMA(_SCALE)?_.32_32X32X64_F8F6F4")>;

} // End SchedModel = SIDPGFX950FullSpeedModel


let SchedModel = GFX10SpeedModel in {

// The latency values are 1 / (operations / cycle).
// Add 1 stall cycle for VGPR read.
let RetireOOO = 1 in { // llvm-mca specific flag
def : HWWriteRes<Write32Bit,         [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteFloatCvt,      [HWVALU, HWRC],   5>;
def : HWWriteRes<Write64Bit,         [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteTrans32,       [HWTransVALU, HWRC], 10>;
def : HWWriteRes<WriteQuarterRate32, [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteFloatFMA,      [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteDouble,        [HWVALU, HWRC],   22>;
def : HWWriteRes<WriteDoubleAdd,     [HWVALU, HWRC],   22>;
def : HWWriteRes<WriteDoubleCvt,     [HWVALU, HWRC],   22>;
def : HWWriteRes<WriteIntMul,        [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteTrans64,       [HWVALU, HWTransVALU, HWRC], 24>;

def : HWWriteRes<WriteBranch,        [HWBranch],       32>;
def : HWWriteRes<WriteExport,        [HWExport, HWRC], 16>;
def : HWWriteRes<WriteLDS,           [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteSALU,          [HWSALU,   HWRC], 2>;
def : HWWriteRes<WriteSMEM,          [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteVMEM,          [HWVMEM,   HWRC], 320>;
def : HWWriteRes<WriteBarrier,       [HWBranch],       2000>;

def : UnsupportedWriteRes<WriteSFPU>;
def : UnsupportedWriteRes<WritePseudoScalarTrans>;

let NumMicroOps = 0 in {
def : HWWriteRes<WriteVALUDummy, [HWVALU],  5>;
def : HWWriteRes<WriteSALUDummy, [HWSALU],  2>;
}

} // End RetireOOO = 1

def : InstRW<[WriteCopy], (instrs COPY)>;

}  // End SchedModel = GFX10SpeedModel

let SchedModel = GFX11SpeedModel in {

// The latency values are 1 / (operations / cycle).
// Add 1 stall cycle for VGPR read.
let RetireOOO = 1 in { // llvm-mca specific flag
def : HWWriteRes<Write32Bit,         [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteFloatCvt,      [HWVALU, HWRC],   5>;
def : HWWriteRes<Write64Bit,         [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteTrans32,       [HWTransVALU, HWRC], 10>;
def : HWWriteRes<WriteQuarterRate32, [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteFloatFMA,      [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteDouble,        [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteDoubleAdd,     [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteDoubleCvt,     [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteIntMul,        [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteTrans64,       [HWVALU, HWTransVALU, HWRC], 40>;

def : HWWriteRes<WriteBranch,        [HWBranch],       32>;
def : HWWriteRes<WriteExport,        [HWExport, HWRC], 16>;
def : HWWriteRes<WriteLDS,           [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteSALU,          [HWSALU,   HWRC], 2>;
def : HWWriteRes<WriteSFPU,          [HWSALU,   HWRC], 4>;
def : HWWriteRes<WriteSMEM,          [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteVMEM,          [HWVMEM,   HWRC], 320>;
def : HWWriteRes<WriteBarrier,       [HWBranch],       2000>;
} // End RetireOOO = 1

def : UnsupportedWriteRes<WritePseudoScalarTrans>;

def : InstRW<[WriteCopy], (instrs COPY)>;

let NumMicroOps = 0 in {
def : HWWriteRes<WriteVALUDummy, [HWVALU],  5>;
def : HWWriteRes<WriteSALUDummy, [HWSALU],  2>;
}

}  // End SchedModel = GFX11SpeedModel

let SchedModel = GFX12SpeedModel in {

def : HWWriteRes<Write32Bit,             [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteFloatCvt,          [HWVALU, HWRC],   5>;
def : HWWriteRes<Write64Bit,             [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteTrans32,           [HWVALU, HWRC],   9>;
def : HWWriteRes<WriteQuarterRate32,     [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteFloatFMA,          [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteDouble,            [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteDoubleAdd,         [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteDoubleCvt,         [HWVALU, HWRC],   38>;
def : HWWriteRes<WriteIntMul,            [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteTrans64,           [HWVALU, HWRC],   40>;
def : HWWriteRes<WritePseudoScalarTrans, [HWVALU, HWRC],   8>;

def : HWWriteRes<WriteBranch,            [HWBranch],       32>;
def : HWWriteRes<WriteExport,            [HWExport, HWRC], 16>;
def : HWWriteRes<WriteLDS,               [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteSALU,              [HWSALU,   HWRC], 2>;
def : HWWriteRes<WriteSFPU,              [HWSALU,   HWRC], 4>;
def : HWWriteRes<WriteSMEM,              [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteVMEM,              [HWVMEM,   HWRC], 320>;
def : HWWriteRes<WriteBarrier,           [HWBranch],       2000>;

def : InstRW<[WriteCopy], (instrs COPY)>;

let NumMicroOps = 0 in {
def : HWWriteRes<WriteVALUDummy, [HWVALU],  5>;
def : HWWriteRes<WriteSALUDummy, [HWSALU],  2>;
}

}  // End SchedModel = GFX12SpeedModel

// Check if any matrix inputs are interpreted as f8 in an f8f6f4
// wmma instruction.
def PredIsF8_WMMA_SCALE : SchedPredicate<[{
  TII->getNamedOperand(*MI, AMDGPU::OpName::matrix_a_fmt)->getImm() <= AMDGPU::WMMA::MATRIX_FMT_BF8 ||
  TII->getNamedOperand(*MI, AMDGPU::OpName::matrix_b_fmt)->getImm() <= AMDGPU::WMMA::MATRIX_FMT_BF8
}]>;

// If either matrix format is f8, the instruction takes 2x as many
// cycles. TODO: This isn't reflected in MCA.
def WriteWMMAScale_16X16X128_F8F6F4 : SchedWriteVariant<[
    SchedVar<PredIsF8_WMMA_SCALE, [WriteXDL4PassWMMA]>,
    SchedVar<NoSchedPred, [WriteXDL2PassWMMA]>
]>;

multiclass GFX125xCommonWriteRes {

// RepeatRate = 2 variants
let RepeatRate = [2, 0] in {
def : HWWriteRes<Trans32Repeat2,    [HWTransVALU, HWRC], 8>;
def : HWWriteRes<Write64BitRepeat2, [HWVALU, HWRC],      6>;
def : HWWriteRes<Write32BitRepeat2, [HWVALU, HWRC],      5>;
def : HWWriteRes<WriteDPPRepeat2,   [HWVALU, HWRC],      5>;
def : HWWriteRes<WriteSwapRepeat2,  [HWVALU, HWRC],      5>;
}

// RepeatRate = 4 variants
let RepeatRate = [4, 0] in {
def : HWWriteRes<IntMulRepeat4,     [HWVALU, HWRC], 8>;
def : HWWriteRes<Write32BitRepeat4, [HWVALU, HWRC], 5>;
}

// RepeatRate = 8 variants
let RepeatRate = [8, 0] in {
def : HWWriteRes<Write32BitRepeat8, [HWVALU, HWRC], 5>;
}

// RepeatRate = 16 variants
let RepeatRate = [16, 0] in {
def : HWWriteRes<Write32BitRepeat16, [HWVALU, HWRC], 5>;
}

// RepeatRate = 32 variants
let RepeatRate = [32, 0, 0] in
def : HWWriteRes<Trans64Repeat32, [HWVALU, HWTransVALU, HWRC], 38>;
let RepeatRate = [32, 0] in
def : HWWriteRes<DoubleRepeat32,  [HWVALU, HWRC],              37>;

let ReleaseAtCycles = [8] in
def : HWWriteRes<WriteXDL2PassWMMA, [HWXDL], 8>;
let ReleaseAtCycles = [16] in
def : HWWriteRes<WriteXDL4PassWMMA, [HWXDL], 16>;

def : HWWriteRes<Write4PassWMMA,  [HWVALU], 16>;
def : HWWriteRes<Write8PassWMMA,  [HWVALU], 32>;
def : HWWriteRes<Write16PassWMMA, [HWVALU], 64>;

def : HWWriteRes<Write32Bit,             [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteFloatCvt,          [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteTrans32,           [HWTransVALU, HWRC],   8>;
def : HWWriteRes<WriteQuarterRate32,     [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteFloatFMA,          [HWVALU, HWRC],   5>;
def : HWWriteRes<WritePseudoScalarTrans, [HWVALU, HWRC],   8>;

def : HWWriteRes<WriteBranch,            [HWBranch],       32>;
def : HWWriteRes<WriteExport,            [HWExport, HWRC], 16>;
def : HWWriteRes<WriteLDS,               [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteSALU,              [HWSALU,   HWRC], 2>;
def : HWWriteRes<WriteSFPU,              [HWSALU,   HWRC], 4>;
def : HWWriteRes<WriteSMEM,              [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteVMEM,              [HWVMEM,   HWRC], 320>;
def : HWWriteRes<WriteBarrier,           [HWBranch],       2000>;

def : InstRW<[WriteCopy], (instrs COPY)>;

def : InstRW<[WriteXDL2PassWMMA], (instregex "^V_[S]*WMMA[C]*_.*_(FP8|BF8|BF16|F16)_w32")>;
def : InstRW<[WriteXDL4PassWMMA], (instregex "^V_[S]*WMMA[C]*_.*_(IU8|IU4)_w32")>;
def : InstRW<[WriteWMMAScale_16X16X128_F8F6F4], (instregex "^V_WMMA_.*_16X16X128_F8F6F4.*_w32")>;
def : InstRW<[Write4PassWMMA],    (instregex "^V_WMMA_F32_16X16X4_F32_w32")>;
def : InstRW<[WriteXDL2PassWMMA], (instregex "^V_WMMA.*_F32_32X16X128_F4")>;

let NumMicroOps = 0 in {
def : HWWriteRes<WriteVALUDummy, [HWVALU],  5>;
def : HWWriteRes<WriteSALUDummy, [HWSALU],  2>;
}

// VALU RepeatRate = 2 instructions
// F32 transcendentals
def : InstRW<[Trans32Repeat2], (instregex "^V_EXP_F32_e(32|64)$",
                                          "^V_LOG_F32_e(32|64)$",
                                          "^V_RCP_F32_e(32|64)$",
                                          "^V_RCP_IFLAG_F32_e(32|64)$",
                                          "^V_RSQ_F32_e(32|64)$",
                                          "^V_SQRT_F32_e(32|64)$",
                                          "^V_SIN_F32_e(32|64)$",
                                          "^V_COS_F32_e(32|64)$")>;
// BF16 transcendentals
def : InstRW<[Trans32Repeat2], (instregex "^V_EXP_BF16_e(32|64)$",
                                          "^V_LOG_BF16_e(32|64)$",
                                          "^V_RCP_BF16_e(32|64)$",
                                          "^V_RSQ_BF16_e(32|64)$",
                                          "^V_SQRT_BF16_e(32|64)$",
                                          "^V_SIN_BF16_e(32|64)$",
                                          "^V_COS_BF16_e(32|64)$")>;
// F16 transcendentals
def : InstRW<[Trans32Repeat2], (instregex "^V_EXP_F16_e(32|64)$",
                                          "^V_LOG_F16_e(32|64)$",
                                          "^V_RCP_F16_e(32|64)$",
                                          "^V_RSQ_F16_e(32|64)$",
                                          "^V_SQRT_F16_e(32|64)$",
                                          "^V_SIN_F16_e(32|64)$",
                                          "^V_COS_F16_e(32|64)$")>;
// TANH instructions
def : InstRW<[Trans32Repeat2], (instregex "^V_TANH_BF16_e(32|64)$",
                                          "^V_TANH_F16_e(32|64)$",
                                          "^V_TANH_F32_e(32|64)$")>;
// 64-bit shifts and moves
def : InstRW<[Write64BitRepeat2], (instregex "^V_LSHLREV_B64_e64$",
                                             "^V_LSHRREV_B64_e64$",
                                             "^V_ASHRREV_I64_e64$",
                                             "^V_MOV_B64_e(32|64)$",
                                             "^V_TRIG_PREOP_F64_e64$")>;

def : InstRW<[Write32BitRepeat2],  (instregex "^V_LSHL_ADD_U64_e64$")>;

// 64-bit integer comparisons
def : InstRW<[Write64BitRepeat2], (instregex "^V_CMP_EQ_I64_e(32|64)$",
                                             "^V_CMP_LE_I64_e(32|64)$",
                                             "^V_CMP_GE_I64_e(32|64)$",
                                             "^V_CMP_NE_I64_e(32|64)$",
                                             "^V_CMP_EQ_U64_e(32|64)$",
                                             "^V_CMP_LE_U64_e(32|64)$",
                                             "^V_CMP_GE_U64_e(32|64)$",
                                             "^V_CMP_NE_U64_e(32|64)$")>;
// 64-bit add
def : InstRW<[Write64BitRepeat2], (instregex "^V_ADD_U64_e(32|64)$")>;

// VALU RepeatRate = 4 instructions
// Integer multiply
def : InstRW<[IntMulRepeat4], (instregex "^V_MUL_LO_U32_e64$",
                                         "^V_MUL_HI_U32_e64$",
                                         "^V_MUL_HI_I32_e64$",
                                         "^V_MAD_U32_e64$",
                                         "^V_MAD_NC_U64_U32_e64$",
                                         "^V_MAD_NC_I64_I32_e64$")>;
def : InstRW<[IntMulRepeat4, WriteSALUDummy], (instregex "^V_MAD_CO_U64_U32_e64_gfx12$",
                                                         "^V_MAD_CO_I64_I32_e64_gfx12$")>;
// SAD instructions
def : InstRW<[IntMulRepeat4], (instregex "^V_QSAD_PK_U16_U8_e64$",
                                         "^V_MQSAD_PK_U16_U8_e64$",
                                         "^V_MQSAD_U32_U8_e64$")>;
// V_CVT_SCALE_PK8 instructions
def : InstRW<[Write32BitRepeat4], (instregex "^V_CVT_SCALE_PK8_F16_FP8_e64$",
                                             "^V_CVT_SCALE_PK8_BF16_FP8_e64$",
                                             "^V_CVT_SCALE_PK8_F16_BF8_e64$",
                                             "^V_CVT_SCALE_PK8_BF16_BF8_e64$",
                                             "^V_CVT_SCALE_PK8_F32_FP8_e64$",
                                             "^V_CVT_SCALE_PK8_F32_BF8_e64$",
                                             "^V_CVT_SCALE_PK8_F16_FP4_e64$",
                                             "^V_CVT_SCALE_PK8_BF16_FP4_e64$",
                                             "^V_CVT_SCALE_PK8_F32_FP4_e64$")>;
// V_CVT_SCALEF32_PK8 instructions
def : InstRW<[Write32BitRepeat4], (instregex "^V_CVT_SCALEF32_PK8_BF8_BF16_e64$",
                                             "^V_CVT_SCALEF32_PK8_FP8_BF16_e64$",
                                             "^V_CVT_SCALEF32_PK8_BF8_F16_e64$",
                                             "^V_CVT_SCALEF32_PK8_FP8_F16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_BF8_BF16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_FP8_BF16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_BF8_F16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_FP8_F16_e64$",
                                             "^V_CVT_SCALEF32_PK8_BF8_F32_e64$",
                                             "^V_CVT_SCALEF32_PK8_FP8_F32_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_BF8_F32_e64$",
                                             "^V_CVT_SCALEF32_SR_PK8_FP8_F32_e64$")>;
// V_PERM_PK16 instructions
def : InstRW<[Write32BitRepeat4], (instregex "^V_PERM_PK16_B4_U4_e64$",
                                             "^V_PERM_PK16_B6_U4_e64$",
                                             "^V_PERM_PK16_B8_U4_e64$")>;

// VALU RepeatRate = 8 instructions
// V_CVT_SCALEF32_PK16 instructions
def : InstRW<[Write32BitRepeat8], (instregex "^V_CVT_SCALEF32_2XPK16_BF6_F32_e64$",
                                             "^V_CVT_SCALEF32_2XPK16_FP6_F32_e64$",
                                             "^V_CVT_SCALEF32_PK16_BF6_BF16_e64$",
                                             "^V_CVT_SCALEF32_PK16_FP6_BF16_e64$",
                                             "^V_CVT_SCALEF32_PK16_BF6_F16_e64$",
                                             "^V_CVT_SCALEF32_PK16_FP6_F16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_BF6_BF16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_FP6_BF16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_BF6_F16_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_FP6_F16_e64$",
                                             "^V_CVT_SCALEF32_PK16_BF6_F32_e64$",
                                             "^V_CVT_SCALEF32_PK16_FP6_F32_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_BF6_F32_e64$",
                                             "^V_CVT_SCALEF32_SR_PK16_FP6_F32_e64$")>;
// V_CVT_SCALE_PK16 instructions
def : InstRW<[Write32BitRepeat8], (instregex "^V_CVT_SCALE_PK16_BF16_BF6_e64$",
                                             "^V_CVT_SCALE_PK16_F16_BF6_e64$",
                                             "^V_CVT_SCALE_PK16_F32_BF6_e64$",
                                             "^V_CVT_SCALE_PK16_BF16_FP6_e64$",
                                             "^V_CVT_SCALE_PK16_F16_FP6_e64$",
                                             "^V_CVT_SCALE_PK16_F32_FP6_e64$")>;

// VALU RepeatRate = 16 instructions
// V_CVT_SCALEF32_PK32 instructions
def : InstRW<[Write32BitRepeat16], (instregex "^V_CVT_SCALEF32_PK32_BF16_BF6_e64$",
                                              "^V_CVT_SCALEF32_PK32_BF6_BF16_e64$",
                                              "^V_CVT_SCALEF32_PK32_BF16_FP6_e64$",
                                              "^V_CVT_SCALEF32_PK32_FP6_BF16_e64$",
                                              "^V_CVT_SCALEF32_PK32_F16_BF6_e64$",
                                              "^V_CVT_SCALEF32_PK32_BF6_F16_e64$",
                                              "^V_CVT_SCALEF32_PK32_F16_FP6_e64$",
                                              "^V_CVT_SCALEF32_PK32_FP6_F16_e64$",
                                              "^V_CVT_SCALEF32_PK32_F32_BF6_e64$",
                                              "^V_CVT_SCALEF32_PK32_F32_FP6_e64$")>;

// VALU RepeatRate = 32 instructions
// F64 transcendentals
def : InstRW<[Trans64Repeat32], (instregex "^V_RCP_F64_e(32|64)$",
                                           "^V_RSQ_F64_e(32|64)$",
                                           "^V_SQRT_F64_e(32|64)$")>;
// 64-bit integer operations
def : InstRW<[DoubleRepeat32], (instregex "^V_MUL_U64_e32$",
                                          "^V_MAX_U64_e64$",
                                          "^V_MAX_I64_e64$",
                                          "^V_MIN_U64_e64$",
                                          "^V_MIN_I64_e64$")>;
// F64 comparisons
def : InstRW<[DoubleRepeat32], (instregex "^V_CMP_EQ_F64_e(32|64)$",
                                          "^V_CMP_LE_F64_e(32|64)$",
                                          "^V_CMP_GE_F64_e(32|64)$")>;
// F64 conversions
def : InstRW<[DoubleRepeat32], (instregex "^V_CVT_I32_F64_e(32|64)$",
                                          "^V_CVT_F64_I32_e(32|64)$",
                                          "^V_CVT_F32_F64_e(32|64)$",
                                          "^V_CVT_F64_F32_e(32|64)$",
                                          "^V_CVT_U32_F64_e(32|64)$",
                                          "^V_CVT_F64_U32_e(32|64)$")>;
// F64 rounding and frexp
def : InstRW<[DoubleRepeat32], (instregex "^V_TRUNC_F64_e(32|64)$",
                                          "^V_CEIL_F64_e(32|64)$",
                                          "^V_RNDNE_F64_e(32|64)$",
                                          "^V_FLOOR_F64_e(32|64)$",
                                          "^V_FREXP_EXP_I32_F64_e(32|64)$",
                                          "^V_FREXP_MANT_F64_e(32|64)$")>;
// F64 arithmetic
def : InstRW<[DoubleRepeat32], (instregex "^V_FRACT_F64_e(32|64)$",
                                          "^V_FMA_F64_e64$",
                                          "^V_DIV_FIXUP_F64_e64$",
                                          "^V_DIV_FMAS_F64_e64$",
                                          "^V_ADD_F64_e64$",
                                          "^V_MUL_F64_e64$",
                                          "^V_MIN_NUM_F64_e64$",
                                          "^V_MAX_NUM_F64_e64$",
                                          "^V_LDEXP_F64_e64$",
                                          "^V_MINIMUM_F64_e64$",
                                          "^V_MAXIMUM_F64_e64$")>;
} // End GFX125xCommonWriteRes

let SchedModel = GFX1250SpeedModel in {
defm : GFX125xCommonWriteRes;

def : HWWriteRes<Write64Bit,             [HWVALU, HWRC],   6>;
def : HWWriteRes<WriteIntMul,            [HWVALU, HWRC],   8>;
def : HWWriteRes<WriteDouble,            [HWVALU, HWRC],   37>;
def : HWWriteRes<WriteDoubleAdd,         [HWVALU, HWRC],   37>;
def : HWWriteRes<WriteDoubleCvt,         [HWVALU, HWRC],   37>;
def : HWWriteRes<WriteTrans64,           [HWVALU, HWTransVALU, HWRC], 38>;


} // SchedModel = GFX1250SpeedModel
