//===-- HexagonSubtarget.cpp - Hexagon Subtarget Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>

using namespace llvm;

#define DEBUG_TYPE "hexagon-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool> EnableMemOps("enable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(true),
  cl::desc("Generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> DisableMemOps("disable-hexagon-memops",
  cl::Hidden, cl::ZeroOrMore, cl::ValueDisallowed, cl::init(false),
  cl::desc("Do not generate V4 MEMOP in code generation for Hexagon target"));

static cl::opt<bool> EnableIEEERndNear("enable-hexagon-ieee-rnd-near",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Generate non-chopped conversion from fp to int."));

static cl::opt<bool> EnableBSBSched("enable-bsb-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(true));

static cl::opt<bool> EnableHexagonHVXDouble("enable-hexagon-hvx-double",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Double Vector eXtensions"));

static cl::opt<bool> EnableHexagonHVX("enable-hexagon-hvx",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Enable Hexagon Vector eXtensions"));

static cl::opt<bool> EnableTCLatencySched("enable-tc-latency-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(false));

static cl::opt<bool> EnableDotCurSched("enable-cur-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(true),
  cl::desc("Enable the scheduler to generate .cur"));

static cl::opt<bool> EnableVecFrwdSched("enable-evec-frwd-sched",
  cl::Hidden, cl::ZeroOrMore, cl::init(true));

static cl::opt<bool> DisableHexagonMISched("disable-hexagon-misched",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Disable Hexagon MI Scheduling"));

static cl::opt<bool> EnableSubregLiveness("hexagon-subreg-liveness",
  cl::Hidden, cl::ZeroOrMore, cl::init(true),
  cl::desc("Enable subregister liveness tracking for Hexagon"));

static cl::opt<bool> OverrideLongCalls("hexagon-long-calls",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("If present, forces/disables the use of long calls"));

static cl::opt<bool> EnablePredicatedCalls("hexagon-pred-calls",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Consider calls to be predicable"));

void HexagonSubtarget::initializeEnvironment() {
  UseMemOps = false;
  ModeIEEERndNear = false;
  UseBSBScheduling = false;
}

HexagonSubtarget &
HexagonSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  CPUString = Hexagon_MC::selectHexagonCPU(getTargetTriple(), CPU);

  static std::map<StringRef, HexagonArchEnum> CpuTable {
    { "hexagonv4", V4 },
    { "hexagonv5", V5 },
    { "hexagonv55", V55 },
    { "hexagonv60", V60 },
    { "hexagonv62", V62 },
  };

  auto foundIt = CpuTable.find(CPUString);
  if (foundIt != CpuTable.end())
    HexagonArchVersion = foundIt->second;
  else
    llvm_unreachable("Unrecognized Hexagon processor version");

  UseHVXOps = false;
  UseHVXDblOps = false;
  UseLongCalls = false;
  ParseSubtargetFeatures(CPUString, FS);

  if (EnableHexagonHVX.getPosition())
    UseHVXOps = EnableHexagonHVX;
  if (EnableHexagonHVXDouble.getPosition())
    UseHVXDblOps = EnableHexagonHVXDouble;
  if (OverrideLongCalls.getPosition())
    UseLongCalls = OverrideLongCalls;

  return *this;
}

HexagonSubtarget::HexagonSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    : HexagonGenSubtargetInfo(TT, CPU, FS), CPUString(CPU),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering() {

  initializeEnvironment();

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // UseMemOps on by default unless disabled explicitly
  if (DisableMemOps)
    UseMemOps = false;
  else if (EnableMemOps)
    UseMemOps = true;
  else
    UseMemOps = false;

  if (EnableIEEERndNear)
    ModeIEEERndNear = true;
  else
    ModeIEEERndNear = false;

  UseBSBScheduling = hasV60TOps() && EnableBSBSched;
}

/// \brief Perform target specific adjustments to the latency of a schedule
/// dependency.
void HexagonSubtarget::adjustSchedDependency(SUnit *Src, SUnit *Dst,
                                             SDep &Dep) const {
  MachineInstr *SrcInst = Src->getInstr();
  MachineInstr *DstInst = Dst->getInstr();
  if (!Src->isInstr() || !Dst->isInstr())
    return;

  const HexagonInstrInfo *QII = getInstrInfo();

  // Instructions with .new operands have zero latency.
  SmallSet<SUnit *, 4> ExclSrc;
  SmallSet<SUnit *, 4> ExclDst;
  if (QII->canExecuteInBundle(*SrcInst, *DstInst) &&
      isBestZeroLatency(Src, Dst, QII, ExclSrc, ExclDst)) {
    Dep.setLatency(0);
    return;
  }

  if (!hasV60TOps())
    return;

  // If it's a REG_SEQUENCE, use its destination instruction to determine
  // the correct latency.
  if (DstInst->isRegSequence() && Dst->NumSuccs == 1) {
    unsigned RSeqReg = DstInst->getOperand(0).getReg();
    MachineInstr *RSeqDst = Dst->Succs[0].getSUnit()->getInstr();
    unsigned UseIdx = -1;
    for (unsigned OpNum = 0; OpNum < RSeqDst->getNumOperands(); OpNum++) {
      const MachineOperand &MO = RSeqDst->getOperand(OpNum);
      if (MO.isReg() && MO.getReg() && MO.isUse() && MO.getReg() == RSeqReg) {
        UseIdx = OpNum;
        break;
      }
    }
    unsigned RSeqLatency = (InstrInfo.getOperandLatency(&InstrItins, *SrcInst,
                                                        0, *RSeqDst, UseIdx));
    Dep.setLatency(RSeqLatency);
  }

  // Try to schedule uses near definitions to generate .cur.
  ExclSrc.clear();
  ExclDst.clear();
  if (EnableDotCurSched && QII->isToBeScheduledASAP(*SrcInst, *DstInst) &&
      isBestZeroLatency(Src, Dst, QII, ExclSrc, ExclDst)) {
    Dep.setLatency(0);
    return;
  }

  updateLatency(*SrcInst, *DstInst, Dep);
}


void HexagonSubtarget::HexagonDAGMutation::apply(ScheduleDAGInstrs *DAG) {
  for (auto &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;
    SmallVector<SDep, 4> Erase;
    for (auto &D : SU.Preds)
      if (D.getKind() == SDep::Output && D.getReg() == Hexagon::USR_OVF)
        Erase.push_back(D);
    for (auto &E : Erase)
      SU.removePred(E);
  }

  for (auto &SU : DAG->SUnits) {
    // Update the latency of chain edges between v60 vector load or store
    // instructions to be 1. These instruction cannot be scheduled in the
    // same packet.
    MachineInstr &MI1 = *SU.getInstr();
    auto *QII = static_cast<const HexagonInstrInfo*>(DAG->TII);
    bool IsStoreMI1 = MI1.mayStore();
    bool IsLoadMI1 = MI1.mayLoad();
    if (!QII->isHVXVec(MI1) || !(IsStoreMI1 || IsLoadMI1))
      continue;
    for (auto &SI : SU.Succs) {
      if (SI.getKind() != SDep::Order || SI.getLatency() != 0)
        continue;
      MachineInstr &MI2 = *SI.getSUnit()->getInstr();
      if (!QII->isHVXVec(MI2))
        continue;
      if ((IsStoreMI1 && MI2.mayStore()) || (IsLoadMI1 && MI2.mayLoad())) {
        SI.setLatency(1);
        SU.setHeightDirty();
        // Change the dependence in the opposite direction too.
        for (auto &PI : SI.getSUnit()->Preds) {
          if (PI.getSUnit() != &SU || PI.getKind() != SDep::Order)
            continue;
          PI.setLatency(1);
          SI.getSUnit()->setDepthDirty();
        }
      }
    }
  }
}


void HexagonSubtarget::getPostRAMutations(
      std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(make_unique<HexagonSubtarget::HexagonDAGMutation>());
}

void HexagonSubtarget::getSMSMutations(
      std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(make_unique<HexagonSubtarget::HexagonDAGMutation>());
}


// Pin the vtable to this file.
void HexagonSubtarget::anchor() {}

bool HexagonSubtarget::enableMachineScheduler() const {
  if (DisableHexagonMISched.getNumOccurrences())
    return !DisableHexagonMISched;
  return true;
}

bool HexagonSubtarget::usePredicatedCalls() const {
  return EnablePredicatedCalls;
}

void HexagonSubtarget::updateLatency(MachineInstr &SrcInst,
      MachineInstr &DstInst, SDep &Dep) const {
  if (Dep.isArtificial()) {
    Dep.setLatency(1);
    return;
  }

  if (!hasV60TOps())
    return;

  auto &QII = static_cast<const HexagonInstrInfo&>(*getInstrInfo());

  // BSB scheduling.
  if (QII.isHVXVec(SrcInst) || useBSBScheduling())
    Dep.setLatency((Dep.getLatency() + 1) >> 1);
}

void HexagonSubtarget::restoreLatency(SUnit *Src, SUnit *Dst) const {
  MachineInstr *SrcI = Src->getInstr();
  for (auto &I : Src->Succs) {
    if (!I.isAssignedRegDep() || I.getSUnit() != Dst)
      continue;
    unsigned DepR = I.getReg();
    int DefIdx = -1;
    for (unsigned OpNum = 0; OpNum < SrcI->getNumOperands(); OpNum++) {
      const MachineOperand &MO = SrcI->getOperand(OpNum);
      if (MO.isReg() && MO.isDef() && MO.getReg() == DepR)
        DefIdx = OpNum;
    }
    assert(DefIdx >= 0 && "Def Reg not found in Src MI");
    MachineInstr *DstI = Dst->getInstr();
    for (unsigned OpNum = 0; OpNum < DstI->getNumOperands(); OpNum++) {
      const MachineOperand &MO = DstI->getOperand(OpNum);
      if (MO.isReg() && MO.isUse() && MO.getReg() == DepR) {
        int Latency = (InstrInfo.getOperandLatency(&InstrItins, *SrcI,
                                                   DefIdx, *DstI, OpNum));

        // For some instructions (ex: COPY), we might end up with < 0 latency
        // as they don't have any Itinerary class associated with them.
        if (Latency <= 0)
          Latency = 1;

        I.setLatency(Latency);
        updateLatency(*SrcI, *DstI, I);
      }
    }

    // Update the latency of opposite edge too.
    for (auto &J : Dst->Preds) {
      if (J.getSUnit() != Src)
        continue;
      J.setLatency(I.getLatency());
    }
  }
}

/// Change the latency between the two SUnits.
void HexagonSubtarget::changeLatency(SUnit *Src, SUnit *Dst, unsigned Lat)
      const {
  for (auto &I : Src->Succs) {
    if (I.getSUnit() != Dst)
      continue;
    SDep T = I;
    I.setLatency(Lat);

    // Update the latency of opposite edge too.
    T.setSUnit(Src);
    auto F = std::find(Dst->Preds.begin(), Dst->Preds.end(), T);
    assert(F != Dst->Preds.end());
    F->setLatency(I.getLatency());
  }
}

/// If the SUnit has a zero latency edge, return the other SUnit.
static SUnit *getZeroLatency(SUnit *N, SmallVector<SDep, 4> &Deps) {
  for (auto &I : Deps)
    if (I.isAssignedRegDep() && I.getLatency() == 0 &&
        !I.getSUnit()->getInstr()->isPseudo())
      return I.getSUnit();
  return nullptr;
}

// Return true if these are the best two instructions to schedule
// together with a zero latency. Only one dependence should have a zero
// latency. If there are multiple choices, choose the best, and change
// the others, if needed.
bool HexagonSubtarget::isBestZeroLatency(SUnit *Src, SUnit *Dst,
      const HexagonInstrInfo *TII, SmallSet<SUnit*, 4> &ExclSrc,
      SmallSet<SUnit*, 4> &ExclDst) const {
  MachineInstr &SrcInst = *Src->getInstr();
  MachineInstr &DstInst = *Dst->getInstr();

  // Ignore Boundary SU nodes as these have null instructions.
  if (Dst->isBoundaryNode())
    return false;

  if (SrcInst.isPHI() || DstInst.isPHI())
    return false;

  if (!TII->isToBeScheduledASAP(SrcInst, DstInst) &&
      !TII->canExecuteInBundle(SrcInst, DstInst))
    return false;

  // The architecture doesn't allow three dependent instructions in the same
  // packet. So, if the destination has a zero latency successor, then it's
  // not a candidate for a zero latency predecessor.
  if (getZeroLatency(Dst, Dst->Succs) != nullptr)
    return false;

  // Check if the Dst instruction is the best candidate first.
  SUnit *Best = nullptr;
  SUnit *DstBest = nullptr;
  SUnit *SrcBest = getZeroLatency(Dst, Dst->Preds);
  if (SrcBest == nullptr || Src->NodeNum >= SrcBest->NodeNum) {
    // Check that Src doesn't have a better candidate.
    DstBest = getZeroLatency(Src, Src->Succs);
    if (DstBest == nullptr || Dst->NodeNum <= DstBest->NodeNum)
      Best = Dst;
  }
  if (Best != Dst)
    return false;

  // The caller frequently adds the same dependence twice. If so, then
  // return true for this case too.
  if ((Src == SrcBest && Dst == DstBest ) ||
      (SrcBest == nullptr && Dst == DstBest) ||
      (Src == SrcBest && Dst == nullptr))
    return true;

  // Reassign the latency for the previous bests, which requires setting
  // the dependence edge in both directions.
  if (SrcBest != nullptr) {
    if (!hasV60TOps())
      changeLatency(SrcBest, Dst, 1);
    else
      restoreLatency(SrcBest, Dst);
  }
  if (DstBest != nullptr) {
    if (!hasV60TOps())
      changeLatency(Src, DstBest, 1);
    else
      restoreLatency(Src, DstBest);
  }

  // Attempt to find another opprotunity for zero latency in a different
  // dependence.
  if (SrcBest && DstBest)
    // If there is an edge from SrcBest to DstBst, then try to change that
    // to 0 now.
    changeLatency(SrcBest, DstBest, 0);
  else if (DstBest) {
    // Check if the previous best destination instruction has a new zero
    // latency dependence opportunity.
    ExclSrc.insert(Src);
    for (auto &I : DstBest->Preds)
      if (ExclSrc.count(I.getSUnit()) == 0 &&
          isBestZeroLatency(I.getSUnit(), DstBest, TII, ExclSrc, ExclDst))
        changeLatency(I.getSUnit(), DstBest, 0);
  } else if (SrcBest) {
    // Check if previous best source instruction has a new zero latency
    // dependence opportunity.
    ExclDst.insert(Dst);
    for (auto &I : SrcBest->Succs)
      if (ExclDst.count(I.getSUnit()) == 0 &&
          isBestZeroLatency(SrcBest, I.getSUnit(), TII, ExclSrc, ExclDst))
        changeLatency(SrcBest, I.getSUnit(), 0);
  }

  return true;
}

unsigned HexagonSubtarget::getL1CacheLineSize() const {
  return 32;
}

unsigned HexagonSubtarget::getL1PrefetchDistance() const {
  return 32;
}

bool HexagonSubtarget::enableSubRegLiveness() const {
  return EnableSubregLiveness;
}

