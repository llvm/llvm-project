//===-- SISchedule.td - SI Scheduling definitons -------------------------===//
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

// Vector ALU instructions
def Write32Bit         : SchedWrite;
def WriteFloatCvt      : SchedWrite;
def WriteQuarterRate32 : SchedWrite;

def WriteFloatFMA   : SchedWrite;

// Slow quarter rate f64 instruction.
def WriteDouble : SchedWrite;

// half rate f64 instruction (same as v_add_f64)
def WriteDoubleAdd  : SchedWrite;

// Conversion to or from f64 instruction
def WriteDoubleCvt  : SchedWrite;

// Half rate 64-bit instructions.
def Write64Bit : SchedWrite;

// mAI multipass instructions.
def Write2PassMAI  : SchedWrite;
def Write8PassMAI  : SchedWrite;
def Write16PassMAI : SchedWrite;

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
def GFX10SpeedModel : SISchedMachineModel;

// XXX: Are the resource counts correct?
def HWBranch : ProcResource<1> {
  let BufferSize = 1;
}
def HWExport : ProcResource<1> {
  let BufferSize = 7; // Taken from S_WAITCNT
}
def HWLGKM   : ProcResource<1> {
  let BufferSize = 31;  // Taken from S_WAITCNT
}
def HWSALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWVMEM   : ProcResource<1> {
  let BufferSize = 15;  // Taken from S_WAITCNT
}
def HWVALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWRC   : ProcResource<1> { // Register destination cache
  let BufferSize = 1;
}

class HWWriteRes<SchedWrite write, list<ProcResourceKind> resources,
                 int latency> : WriteRes<write, resources> {
  let Latency = latency;
}

class HWVALUWriteRes<SchedWrite write, int latency> :
  HWWriteRes<write, [HWVALU], latency>;

def PredMIReadVGPR : SchedPredicate<[{TII->hasVGPRUses(*MI)}]>;

def MIReadVGPR : SchedReadVariant<[
      SchedVar<PredMIReadVGPR, [MIVGPRRead]>,
      SchedVar<NoSchedPred, [ReadDefault]>]>;

// The latency numbers are taken from AMD Accelerated Parallel Processing
// guide. They may not be accurate.

// The latency values are 1 / (operations / cycle) / 4.
multiclass SICommonWriteRes {

  def : HWWriteRes<WriteBranch,  [HWBranch], 8>;
  def : HWWriteRes<WriteExport,  [HWExport], 4>;
  def : HWWriteRes<WriteLDS,     [HWLGKM],   5>; // Can be between 2 and 64
  def : HWWriteRes<WriteSALU,    [HWSALU],   1>;
  def : HWWriteRes<WriteSMEM,    [HWLGKM],   5>;
  def : HWWriteRes<WriteVMEM,    [HWVMEM],   80>;
  def : HWWriteRes<WriteBarrier, [HWBranch], 500>; // XXX: Guessed ???

  def : HWVALUWriteRes<Write32Bit,         1>;
  def : HWVALUWriteRes<Write64Bit,         2>;
  def : HWVALUWriteRes<WriteFloatCvt,      4>;
  def : HWVALUWriteRes<WriteQuarterRate32, 4>;
  def : HWVALUWriteRes<Write2PassMAI,      2>;
  def : HWVALUWriteRes<Write8PassMAI,      8>;
  def : HWVALUWriteRes<Write16PassMAI,    16>;

  def : ReadAdvance<MIVGPRRead, -2>;
  def : InstRW<[Write64Bit, MIReadVGPR], (instregex "^V_ACCVGPR_WRITE_B32$")>;

  // Technicaly mfma reads can be from 0 to 4 cycles but that does not make
  // sense to model because its register setup is huge. In particular if we
  // properly model read advanice as -2 for a vgpr read it will result in a
  // bad scheduling of acc writes before that mfma. To avoid it we would
  // need to consume 2 or 4 more vgprs to be initialized before the acc
  // write sequence. Just assume worst case here.
  def : ReadAdvance<MIMFMARead, -4>;

  def : InstRW<[Write2PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._4X4X")>;
  def : InstRW<[Write8PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._16X16X")>;
  def : InstRW<[Write16PassMAI, MIMFMARead], (instregex "^V_MFMA_..._32X32X")>;
}

def PredIsVGPR32Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32}]>;
def PredIsVGPR64Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32}]>;
def WriteCopy : SchedWriteVariant<[
    SchedVar<PredIsVGPR32Copy, [Write32Bit]>,
    SchedVar<PredIsVGPR64Copy, [Write64Bit]>,
    SchedVar<NoSchedPred, [WriteSALU]>]>;

let SchedModel = SIFullSpeedModel in {

defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA,   1>;
def : HWVALUWriteRes<WriteDouble,     4>;
def : HWVALUWriteRes<WriteDoubleAdd,  2>;
def : HWVALUWriteRes<WriteDoubleCvt,  4>;

def : InstRW<[WriteCopy], (instrs COPY)>;

} // End SchedModel = SIFullSpeedModel

let SchedModel = SIQuarterSpeedModel in {

defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA, 16>;
def : HWVALUWriteRes<WriteDouble,   16>;
def : HWVALUWriteRes<WriteDoubleAdd, 8>;
def : HWVALUWriteRes<WriteDoubleCvt, 4>;

def : InstRW<[WriteCopy], (instrs COPY)>;

}  // End SchedModel = SIQuarterSpeedModel

let SchedModel = GFX10SpeedModel in {

// The latency values are 1 / (operations / cycle).
// Add 1 stall cycle for VGPR read.
def : HWWriteRes<Write32Bit,         [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteFloatCvt,      [HWVALU, HWRC],   5>;
def : HWWriteRes<Write64Bit,         [HWVALU, HWRC],   9>;
def : HWWriteRes<WriteQuarterRate32, [HWVALU, HWRC],   17>;
def : HWWriteRes<WriteFloatFMA,      [HWVALU, HWRC],   5>;
def : HWWriteRes<WriteDouble,        [HWVALU, HWRC],   17>;
def : HWWriteRes<WriteDoubleAdd,     [HWVALU, HWRC],   17>;
def : HWWriteRes<WriteDoubleCvt,     [HWVALU, HWRC],   17>;

def : HWWriteRes<WriteBranch,        [HWBranch],       32>;
def : HWWriteRes<WriteExport,        [HWExport, HWRC], 16>;
def : HWWriteRes<WriteLDS,           [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteSALU,          [HWSALU,   HWRC], 5>;
def : HWWriteRes<WriteSMEM,          [HWLGKM,   HWRC], 20>;
def : HWWriteRes<WriteVMEM,          [HWVMEM,   HWRC], 320>;
def : HWWriteRes<WriteBarrier,       [HWBranch],       2000>;

def : InstRW<[WriteCopy], (instrs COPY)>;

}  // End SchedModel = GFX10SpeedModel
