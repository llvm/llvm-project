//===- HexagonSubtarget.cpp - Hexagon Subtarget Information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "HexagonSubtarget.h"
#include "HexagonInstrInfo.h"
#include "HexagonRegisterInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IntrinsicsHexagon.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "hexagon-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "HexagonGenSubtargetInfo.inc"

static cl::opt<bool> EnableBSBSched("enable-bsb-sched", cl::Hidden,
                                    cl::init(true));

static cl::opt<bool> EnableTCLatencySched("enable-tc-latency-sched", cl::Hidden,
                                          cl::init(false));

static cl::opt<bool>
    EnableDotCurSched("enable-cur-sched", cl::Hidden, cl::init(true),
                      cl::desc("Enable the scheduler to generate .cur"));

static cl::opt<bool>
    DisableHexagonMISched("disable-hexagon-misched", cl::Hidden,
                          cl::desc("Disable Hexagon MI Scheduling"));

static cl::opt<bool> OverrideLongCalls(
    "hexagon-long-calls", cl::Hidden,
    cl::desc("If present, forces/disables the use of long calls"));

static cl::opt<bool>
    EnablePredicatedCalls("hexagon-pred-calls", cl::Hidden,
                          cl::desc("Consider calls to be predicable"));

static cl::opt<bool> SchedPredsCloser("sched-preds-closer", cl::Hidden,
                                      cl::init(true));

static cl::opt<bool> SchedRetvalOptimization("sched-retval-optimization",
                                             cl::Hidden, cl::init(true));

static cl::opt<bool> EnableCheckBankConflict(
    "hexagon-check-bank-conflict", cl::Hidden, cl::init(true),
    cl::desc("Enable checking for cache bank conflicts"));

HexagonSubtarget::HexagonSubtarget(const Triple &TT, StringRef CPU,
                                   StringRef FS, const TargetMachine &TM)
    : HexagonGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      OptLevel(TM.getOptLevel()),
      CPUString(std::string(Hexagon_MC::selectHexagonCPU(CPU))),
      TargetTriple(TT), InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo(TM, *this), InstrItins(getInstrItineraryForCPU(CPUString)) {
  Hexagon_MC::addArchSubtarget(this, FS);
  // Beware of the default constructor of InstrItineraryData: it will
  // reset all members to 0.
  assert(InstrItins.Itineraries != nullptr && "InstrItins not initialized");
}

HexagonSubtarget &
HexagonSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  std::optional<Hexagon::ArchEnum> ArchVer = Hexagon::getCpu(CPUString);
  if (ArchVer)
    HexagonArchVersion = *ArchVer;
  else
    llvm_unreachable("Unrecognized Hexagon processor version");

  UseHVX128BOps = false;
  UseHVX64BOps = false;
  UseAudioOps = false;
  UseLongCalls = false;

  SubtargetFeatures Features(FS);

  // Turn on QFloat if the HVX version is v68+.
  // The function ParseSubtargetFeatures will set feature bits and initialize
  // subtarget's variables all in one, so there isn't a good way to preprocess
  // the feature string, other than by tinkering with it directly.
  auto IsQFloatFS = [](StringRef F) {
    return F == "+hvx-qfloat" || F == "-hvx-qfloat";
  };
  if (!llvm::count_if(Features.getFeatures(), IsQFloatFS)) {
    auto getHvxVersion = [&Features](StringRef FS) -> StringRef {
      for (StringRef F : llvm::reverse(Features.getFeatures())) {
        if (F.starts_with("+hvxv"))
          return F;
      }
      for (StringRef F : llvm::reverse(Features.getFeatures())) {
        if (F == "-hvx")
          return StringRef();
        if (F.starts_with("+hvx") || F == "-hvx")
          return F.take_front(4);  // Return "+hvx" or "-hvx".
      }
      return StringRef();
    };

    bool AddQFloat = false;
    StringRef HvxVer = getHvxVersion(FS);
    if (HvxVer.starts_with("+hvxv")) {
      int Ver = 0;
      if (!HvxVer.drop_front(5).consumeInteger(10, Ver) && Ver >= 68)
        AddQFloat = true;
    } else if (HvxVer == "+hvx") {
      if (hasV68Ops())
        AddQFloat = true;
    }

    if (AddQFloat)
      Features.AddFeature("+hvx-qfloat");
  }

  std::string FeatureString = Features.getString();
  ParseSubtargetFeatures(CPUString, /*TuneCPU*/ CPUString, FeatureString);

  // Resolve the shadow call stack pointer register.  At most one "scs-reg-rN"
  // feature may be given; R18 is the default.  R18 is chosen because it is the
  // lowest callee-saved register that neither the Hexagon Linux kernel (which
  // reserves R19 for the thread-info pointer) nor code that reserves the upper
  // callee-saved range already claims.
  static_assert(Hexagon::R27 - Hexagon::R16 == 11,
                "Callee-saved R16-R27 are assumed to be consecutive");
  SCSPReg = Hexagon::R18;
  bool SCSRegSelected = false;
  for (unsigned Reg = Hexagon::R16; Reg <= Hexagon::R27; ++Reg) {
    if (!SCSPointerRegister[Reg])
      continue;
    if (SCSRegSelected)
      report_fatal_error(
          "Only one shadow call stack pointer register may be selected");
    SCSPReg = Reg;
    SCSRegSelected = true;
  }

  if (useHVXV68Ops())
    UseHVXFloatingPoint = UseHVXIEEEFPOps || UseHVXQFloatOps;

  if (UseHVXQFloatOps && UseHVXIEEEFPOps && UseHVXFloatingPoint)
    LLVM_DEBUG(
        dbgs() << "Behavior is undefined for simultaneous qfloat and ieee hvx codegen...");

  if (OverrideLongCalls.getPosition())
    UseLongCalls = OverrideLongCalls;

  UseBSBScheduling = hasV60Ops() && EnableBSBSched;

  if (isTinyCore()) {
    // Tiny core has a single thread, so back-to-back scheduling is enabled by
    // default.
    if (!EnableBSBSched.getPosition())
      UseBSBScheduling = false;
  }

  FeatureBitset FeatureBits = getFeatureBits();
  if (HexagonDisableDuplex)
    setFeatureBits(FeatureBits.reset(Hexagon::FeatureDuplex));
  setFeatureBits(Hexagon_MC::completeHVXFeatures(FeatureBits));

  return *this;
}

void HexagonSubtarget::initLibcallLoweringInfo(
    LibcallLoweringInfo &Info) const {
  // The generic arithmetic/division helper routines (__adddf3, __divsi3, ...)
  // exist in Hexagon's compiler-rt alongside the preferred __hexagon_*
  // variants, so both are available. The __hexagon_* variant is the one that
  // must be used; select it explicitly here.
  static const struct {
    const RTLIB::Libcall Op;
    const RTLIB::LibcallImpl Impl;
  } LibraryCalls[] = {
      {RTLIB::SDIV_I32, RTLIB::impl___hexagon_divsi3},
      {RTLIB::SDIV_I64, RTLIB::impl___hexagon_divdi3},
      {RTLIB::UDIV_I32, RTLIB::impl___hexagon_udivsi3},
      {RTLIB::UDIV_I64, RTLIB::impl___hexagon_udivdi3},
      {RTLIB::SREM_I32, RTLIB::impl___hexagon_modsi3},
      {RTLIB::SREM_I64, RTLIB::impl___hexagon_moddi3},
      {RTLIB::UREM_I32, RTLIB::impl___hexagon_umodsi3},
      {RTLIB::UREM_I64, RTLIB::impl___hexagon_umoddi3},
      {RTLIB::ADD_F64, RTLIB::impl___hexagon_adddf3},
      {RTLIB::SUB_F64, RTLIB::impl___hexagon_subdf3},
      {RTLIB::MUL_F64, RTLIB::impl___hexagon_muldf3},
      {RTLIB::DIV_F64, RTLIB::impl___hexagon_divdf3},
      {RTLIB::DIV_F32, RTLIB::impl___hexagon_divsf3},
  };
  for (const auto &LC : LibraryCalls)
    Info.setLibcallImpl(LC.Op, LC.Impl);
}

bool HexagonSubtarget::isHVXElementType(MVT Ty, bool IncludeBool) const {
  if (!useHVXOps())
    return false;
  if (Ty.isVector())
    Ty = Ty.getVectorElementType();
  if (IncludeBool && Ty == MVT::i1)
    return true;
  ArrayRef<MVT> ElemTypes = getHVXElementTypes();
  return llvm::is_contained(ElemTypes, Ty);
}

bool HexagonSubtarget::isHVXVectorType(EVT VecTy, bool IncludeBool) const {
  if (!VecTy.isSimple())
    return false;
  if (!VecTy.isVector() || !useHVXOps() || VecTy.isScalableVector())
    return false;
  MVT ElemTy = VecTy.getSimpleVT().getVectorElementType();
  if (!IncludeBool && ElemTy == MVT::i1)
    return false;

  unsigned HwLen = getVectorLength();
  unsigned NumElems = VecTy.getVectorNumElements();
  ArrayRef<MVT> ElemTypes = getHVXElementTypes();

  if (IncludeBool && ElemTy == MVT::i1) {
    // Boolean HVX vector types are formed from regular HVX vector types
    // by replacing the element type with i1.
    for (MVT T : ElemTypes)
      if (NumElems * T.getSizeInBits() == 8 * HwLen)
        return true;
    return false;
  }

  unsigned VecWidth = VecTy.getSizeInBits();
  if (VecWidth != 8 * HwLen && VecWidth != 16 * HwLen)
    return false;
  return llvm::is_contained(ElemTypes, ElemTy);
}

bool HexagonSubtarget::isTypeForHVX(Type *VecTy, bool IncludeBool) const {
  if (!VecTy->isVectorTy() || isa<ScalableVectorType>(VecTy))
    return false;
  // HLX types have their own legalization path and never go through HVX.
  if (useHLXOps() && isTypeForHLX(VecTy))
    return false;
  // Avoid types like <2 x i32*>.
  Type *ScalTy = VecTy->getScalarType();
  if (!ScalTy->isIntegerTy() &&
      !(ScalTy->isFloatingPointTy() && useHVXFloatingPoint()))
    return false;
  // The given type may be something like <17 x i32>, which is not MVT,
  // but can be represented as (non-simple) EVT.
  EVT Ty = EVT::getEVT(VecTy, /*HandleUnknown*/false);
  if (!Ty.getVectorElementType().isSimple())
    return false;

  auto isHvxTy = [this, IncludeBool](MVT SimpleTy) {
    if (isHVXVectorType(SimpleTy, IncludeBool))
      return true;
    auto Action = getTargetLowering()->getPreferredVectorAction(SimpleTy);
    return Action == TargetLoweringBase::TypeWidenVector;
  };

  // Round up EVT to have power-of-2 elements, and keep checking if it
  // qualifies for HVX, dividing it in half after each step.
  MVT ElemTy = Ty.getVectorElementType().getSimpleVT();
  unsigned VecLen = PowerOf2Ceil(Ty.getVectorNumElements());
  while (VecLen > 1) {
    MVT SimpleTy = MVT::getVectorVT(ElemTy, VecLen);
    if (SimpleTy.isValid() && isHvxTy(SimpleTy))
      return true;
    VecLen /= 2;
  }

  return false;
}

bool HexagonSubtarget::isHLXElementType(MVT Ty) const {
  if (!useHLXOps())
    return false;
  if (Ty.isVector())
    Ty = Ty.getVectorElementType();
  ArrayRef<MVT> ElemTypes = getHLXElementTypes();
  return llvm::is_contained(ElemTypes, Ty);
}

bool HexagonSubtarget::isHLXVectorType(EVT VecTy) const {
  if (!VecTy.isSimple())
    return false;
  if (!VecTy.isVector() || !useHLXOps() || VecTy.isScalableVector())
    return false;
  MVT ElemTy = VecTy.getSimpleVT().getVectorElementType();

  unsigned HwLen = getHLXVectorLength();
  ArrayRef<MVT> ElemTypes = getHLXElementTypes();

  unsigned VecWidth = VecTy.getSizeInBits();
  if (VecWidth != 8 * HwLen && VecWidth != 16 * HwLen)
    return false;
  return llvm::is_contained(ElemTypes, ElemTy);
}

bool HexagonSubtarget::isTypeForHLX(Type *VecTy) const {
  if (!VecTy->isVectorTy() || isa<ScalableVectorType>(VecTy))
    return false;
  // Avoid types like <2 x i32*>.
  Type *ScalTy = VecTy->getScalarType();
  if (!ScalTy->isIntegerTy() && !ScalTy->isFloatingPointTy())
    return false;
  // The given type may be something like <17 x i32>, which is not MVT,
  // but can be represented as (non-simple) EVT.
  EVT Ty = EVT::getEVT(VecTy, /*HandleUnknown*/ false);
  if (!Ty.getVectorElementType().isSimple())
    return false;

  auto isHLXTy = [this](MVT SimpleTy) {
    if (isHLXVectorType(SimpleTy))
      return true;
    // No widening for now
    return false;
  };

  // Round up EVT to have power-of-2 elements, and keep checking if it
  // qualifies for HLX, dividing it in half after each step.
  MVT ElemTy = Ty.getVectorElementType().getSimpleVT();
  unsigned VecLen = PowerOf2Ceil(Ty.getVectorNumElements());
  while (VecLen > 1) {
    MVT SimpleTy = MVT::getVectorVT(ElemTy, VecLen);
    if (SimpleTy.isValid() && isHLXTy(SimpleTy))
      return true;
    VecLen /= 2;
  }

  return false;
}

void HexagonSubtarget::UsrOverflowMutation::apply(ScheduleDAGInstrs *DAG) {
  for (SUnit &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;
    SmallVector<SDep, 4> Erase;
    for (auto &D : SU.Preds)
      if (D.getKind() == SDep::Output && D.getReg() == Hexagon::USR_OVF)
        Erase.push_back(D);
    for (auto &E : Erase)
      SU.removePred(E);
  }
}

void HexagonSubtarget::HVXMemLatencyMutation::apply(ScheduleDAGInstrs *DAG) {
  for (SUnit &SU : DAG->SUnits) {
    // Update the latency of chain edges between v60 vector load or store
    // instructions to be 1. These instruction cannot be scheduled in the
    // same packet.
    MachineInstr &MI1 = *SU.getInstr();
    auto *QII = static_cast<const HexagonInstrInfo*>(DAG->TII);
    bool IsStoreMI1 = MI1.mayStore();
    bool IsLoadMI1 = MI1.mayLoad();
    if (!QII->isHVXVec(MI1) || !(IsStoreMI1 || IsLoadMI1))
      continue;
    for (SDep &SI : SU.Succs) {
      if (SI.getKind() != SDep::Order || SI.getLatency() != 0)
        continue;
      MachineInstr &MI2 = *SI.getSUnit()->getInstr();
      if (!QII->isHVXVec(MI2))
        continue;
      if ((IsStoreMI1 && MI2.mayStore()) || (IsLoadMI1 && MI2.mayLoad())) {
        SI.setLatency(1);
        SU.setHeightDirty();
        // Change the dependence in the opposite direction too.
        for (SDep &PI : SI.getSUnit()->Preds) {
          if (PI.getSUnit() != &SU || PI.getKind() != SDep::Order)
            continue;
          PI.setLatency(1);
          SI.getSUnit()->setDepthDirty();
        }
      }
    }
  }
}

// Ensure that HMX activate and weight instructions are scheduled together.
// The main idea is to:
// 1. Reduce any latency between corresponding A/W to 0.
// 2. Make the depths of corresponding A/W instructions be the same. (The
//    post-RA scheduler is a top-down list scheduler.)
void HexagonSubtarget::HMXActivateWeightMutation::apply(
    ScheduleDAGInstrs *DAG) {
  if (DAG->SUnits.empty())
    return;

  auto &HII = static_cast<const HexagonInstrInfo &>(*DAG->TII);
  DenseMap<SUnit *, SUnit *> AWMap; // Map: Activation <-> Weight.
  DenseSet<SUnit *> First;          // Which of the two came first.
  // The SUnits in the DAG follow the order of instructions. The code below
  // depends on it to pair up activation and weight instructions.
  SUnit *A = nullptr, *W = nullptr;
  for (SUnit &SU : DAG->SUnits) {
    MachineInstr *MI = SU.getInstr();
    if (HII.isHMXActivateInstr(*MI)) {
      assert(A == nullptr && "Unexpected activation instruction");
      A = &SU;
      if (!W)
        First.insert(A);
    } else if (HII.isHMXWeightInstr(*MI)) {
      assert(W == nullptr && "Unexpected weight instruction");
      W = &SU;
      if (!A)
        First.insert(W);
    }
    if (A && W) {
      AWMap.insert({A, W});
      AWMap.insert({W, A});
      A = W = nullptr;
    }
  }
  assert(A == nullptr && "Unmatched activation instruction");
  assert(W == nullptr && "Unmatched weight instruction");

  // Reset the latency between matching instructions to 0.
  for (SUnit *FirstSU : First) {
    SUnit *SecondSU = AWMap.lookup(FirstSU);
    // Set the successor latency to 0.
    for (SDep &D : FirstSU->Succs) {
      if (D.getSUnit() == SecondSU) {
        D.setLatency(0);
        FirstSU->setHeightDirty();
      }
    }
    // Set the predecessor latency to 0.
    for (SDep &D : SecondSU->Preds) {
      if (D.getSUnit() == FirstSU) {
        D.setLatency(0);
        SecondSU->setDepthDirty();
      }
    }
  }

  SUnit *Entry = &DAG->SUnits[0];
  assert(Entry->getDepth() == 0);

  // For each pair, reset the depth (max distance from the entry) to be the
  // max of the two. We can't just set the depth directly, because the post-RA
  // scheduler recalculates it during scheduling. There are also assertions
  // in it that can fire when the depth on the node is inconsistent with the
  // calculations.
  // To avoid this issue, make the depth "real" by adding a fake edge from
  // the Entry node (which is assumed to have a depth of 0).
  // Adding a new edge will mark the depths of successors (recursively)
  // as dirty, potentially including subsequent pairs. The "getDepth" call
  // will recompute a dirty depth, so it's important to iterate over the nodes
  // in such a way that predecessors are visited first.
  //
  // Skip the first SUnit in case it's an A/W instruction, so that we don't
  // try to create an edge from that node to itself.
  for (SUnit &SU : MutableArrayRef<SUnit>(DAG->SUnits).drop_front()) {
    if (!First.count(&SU))
      continue;
    SUnit *FirstSU = &SU;
    SUnit *SecondSU = AWMap.lookup(FirstSU);
    unsigned MaxDepth = std::max(FirstSU->getDepth(), SecondSU->getDepth());
    SDep DepthEdge(Entry, SDep::Artificial);
    DepthEdge.setLatency(MaxDepth);
    FirstSU->addPred(DepthEdge);
    SecondSU->addPred(DepthEdge);
  }
}

// Check if a call and subsequent A2_tfrpi instructions should maintain
// scheduling affinity. We are looking for the TFRI to be consumed in
// the next instruction. This should help reduce the instances of
// double register pairs being allocated and scheduled before a call
// when not used until after the call. This situation is exacerbated
// by the fact that we allocate the pair from the callee saves list,
// leading to excess spills and restores.
bool HexagonSubtarget::CallMutation::shouldTFRICallBind(
      const HexagonInstrInfo &HII, const SUnit &Inst1,
      const SUnit &Inst2) const {
  if (Inst1.getInstr()->getOpcode() != Hexagon::A2_tfrpi)
    return false;

  // TypeXTYPE are 64 bit operations.
  unsigned Type = HII.getType(*Inst2.getInstr());
  return Type == HexagonII::TypeS_2op || Type == HexagonII::TypeS_3op ||
         Type == HexagonII::TypeALU64 || Type == HexagonII::TypeM;
}

void HexagonSubtarget::CallMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  ScheduleDAGMI *DAG = static_cast<ScheduleDAGMI*>(DAGInstrs);
  SUnit* LastSequentialCall = nullptr;
  // Map from virtual register to physical register from the copy.
  DenseMap<unsigned, unsigned> VRegHoldingReg;
  // Map from the physical register to the instruction that uses virtual
  // register. This is used to create the barrier edge.
  DenseMap<unsigned, SUnit *> LastVRegUse;
  auto &TRI = *DAG->MF.getSubtarget().getRegisterInfo();
  auto &HII = *DAG->MF.getSubtarget<HexagonSubtarget>().getInstrInfo();

  // Currently we only catch the situation when compare gets scheduled
  // before preceding call.
  for (unsigned su = 0, e = DAG->SUnits.size(); su != e; ++su) {
    // Remember the call.
    if (DAG->SUnits[su].getInstr()->isCall())
      LastSequentialCall = &DAG->SUnits[su];
    // Look for a compare that defines a predicate.
    else if (DAG->SUnits[su].getInstr()->isCompare() && LastSequentialCall)
      DAG->addEdge(&DAG->SUnits[su], SDep(LastSequentialCall, SDep::Barrier));
    // Look for call and tfri* instructions.
    else if (SchedPredsCloser && LastSequentialCall && su > 1 && su < e-1 &&
             shouldTFRICallBind(HII, DAG->SUnits[su], DAG->SUnits[su+1]))
      DAG->addEdge(&DAG->SUnits[su], SDep(&DAG->SUnits[su-1], SDep::Barrier));
    // Prevent redundant register copies due to reads and writes of physical
    // registers. The original motivation for this was the code generated
    // between two calls, which are caused both the return value and the
    // argument for the next call being in %r0.
    // Example:
    //   1: <call1>
    //   2: %vreg = COPY %r0
    //   3: <use of %vreg>
    //   4: %r0 = ...
    //   5: <call2>
    // The scheduler would often swap 3 and 4, so an additional register is
    // needed. This code inserts a Barrier dependence between 3 & 4 to prevent
    // this.
    // The code below checks for all the physical registers, not just R0/D0/V0.
    else if (SchedRetvalOptimization) {
      const MachineInstr *MI = DAG->SUnits[su].getInstr();
      if (MI->isCopy() && MI->getOperand(1).getReg().isPhysical()) {
        // %vregX = COPY %r0
        VRegHoldingReg[MI->getOperand(0).getReg()] = MI->getOperand(1).getReg();
        LastVRegUse.erase(MI->getOperand(1).getReg());
      } else {
        for (const MachineOperand &MO : MI->operands()) {
          if (!MO.isReg())
            continue;
          if (MO.isUse() && !MI->isCopy() &&
              VRegHoldingReg.count(MO.getReg())) {
            // <use of %vregX>
            LastVRegUse[VRegHoldingReg[MO.getReg()]] = &DAG->SUnits[su];
          } else if (MO.isDef() && MO.getReg().isPhysical()) {
            for (MCRegAliasIterator AI(MO.getReg(), &TRI, true); AI.isValid();
                 ++AI) {
              if (auto It = LastVRegUse.find(*AI); It != LastVRegUse.end()) {
                if (It->second != &DAG->SUnits[su])
                  // %r0 = ...
                  DAG->addEdge(&DAG->SUnits[su],
                               SDep(It->second, SDep::Barrier));
                LastVRegUse.erase(It);
              }
            }
          }
        }
      }
    }
  }
}

namespace {
// Address of a load in the form used by the bank conflict check:
//   (Base << Shift) + Sym + Offset
// Sym is null when the offset is a plain immediate.
struct BankAddr {
  Register Base;
  unsigned Shift = 0;
  const GlobalValue *Sym = nullptr;
  int64_t Offset = 0;
  uint64_t Size = 0;
};
} // end anonymous namespace

// Decompose a scalar load into a BankAddr. Handles the base + offset form,
// with either an immediate (memw(Rs+#s11)) or a global address
// (memw(Rs+##sym)) as the offset, and the shifted-register + global form
// (memw(Rt<<#u2+##sym)).
static std::optional<BankAddr> getBankAddr(const MachineInstr &MI,
                                           const HexagonInstrInfo &HII) {
  if (!MI.mayLoad() || MI.mayStore() || HII.isMemOp(MI) ||
      HII.isPostIncrement(MI) || HII.isHLXVec(MI))
    return std::nullopt;
  unsigned AddrMode = HII.getAddrMode(MI);
  if (AddrMode != HexagonII::BaseImmOffset &&
      AddrMode != HexagonII::BaseLongOffset)
    return std::nullopt;

  // Operand 0 is the loaded value; the address operands follow.
  unsigned BasePos = 1;
  if (HII.isPredicated(MI))
    ++BasePos;

  BankAddr A;
  unsigned OffsetPos = BasePos + 1;
  if (AddrMode == HexagonII::BaseLongOffset) {
    // Rt << #u2 + ##u32
    const MachineOperand &ShiftOp = MI.getOperand(BasePos + 1);
    if (!ShiftOp.isImm())
      return std::nullopt;
    A.Shift = ShiftOp.getImm();
    ++OffsetPos;
  }

  const MachineOperand &BaseOp = MI.getOperand(BasePos);
  if (!BaseOp.isReg() || BaseOp.getSubReg() != 0)
    return std::nullopt;
  A.Base = BaseOp.getReg();

  const MachineOperand &OffsetOp = MI.getOperand(OffsetPos);
  if (OffsetOp.isImm()) {
    A.Offset = OffsetOp.getImm();
  } else if (OffsetOp.isGlobal()) {
    // Only a direct reference to the symbol tells where it is relative to the
    // bank boundaries; GOT, TLS and other relocations do not.
    if ((OffsetOp.getTargetFlags() & ~HexagonII::HMOTF_ConstExtended) !=
        HexagonII::MO_NO_FLAG)
      return std::nullopt;
    A.Sym = OffsetOp.getGlobal();
    A.Offset = OffsetOp.getOffset();
  } else {
    return std::nullopt;
  }

  A.Size = HII.getMemAccessSize(MI);
  return A;
}

// Return true if the offsets of the two loads can be compared to decide
// whether they map to the same bank, i.e. whether the symbols (if any)
// contribute the same value to the bank bits of the address.
static bool haveComparableSyms(const BankAddr &A0, const BankAddr &A1,
                               const DataLayout &DL) {
  if (A0.Sym == A1.Sym)
    return true;
  // Different symbols: the linker places each one independently, so their
  // low address bits are known only from their alignment. A symbol aligned to
  // the bank period (32 bytes) does not affect the bank bits, the same as a
  // plain immediate offset (null Sym).
  const Align BankPeriod(32);
  auto HasKnownBankBits = [&](const GlobalValue *Sym) {
    return !Sym || Sym->getPointerAlignment(DL) >= BankPeriod;
  };
  return HasKnownBankBits(A0.Sym) && HasKnownBankBits(A1.Sym);
}

// Return true if loads MI0 and MI1 likely map to the same cache bank and
// should not be issued in the same packet. If IgnoreSameLine is true, loads
// that likely read the same cache line, which cannot conflict, are not
// reported.
bool HexagonSubtarget::mayHaveBankConflict(const MachineInstr &MI0,
                                           const MachineInstr &MI1,
                                           const HexagonInstrInfo &HII,
                                           bool IgnoreSameLine) {
  if (!EnableCheckBankConflict)
    return false;
  std::optional<BankAddr> A0 = getBankAddr(MI0, HII);
  // Skip accesses as large as the L1 cache line.
  if (!A0 || A0->Size == 0 || A0->Size >= 32)
    return false;
  std::optional<BankAddr> A1 = getBankAddr(MI1, HII);
  if (!A1 || A1->Size == 0 || A1->Size >= 32 || A0->Base != A1->Base ||
      A0->Shift != A1->Shift ||
      !haveComparableSyms(*A0, *A1, MI0.getMF()->getDataLayout()))
    return false;
  // A bank conflict needs two different cache lines. As the bank check
  // below does, assume the base is 32-byte aligned: offsets within the same
  // 32-byte block then read the same line and cannot conflict.
  if (IgnoreSameLine && A0->Sym == A1->Sym &&
      (A0->Offset & ~int64_t(31)) == (A1->Offset & ~int64_t(31)))
    return false;
  // Compare the bank bits of the two offsets. If any of them differ, the
  // loads go to different banks and can share a packet.
  //
  // Which address bits select the bank depends on the access sizes:
  //
  //   Load pair                                  Bank bits   Mask
  //   -----------------------------------------  ---------   ----
  //   double word + any load                     4:3         0x18
  //   word/half-word/byte + word/half-word/byte  4:2         0x1C
  //
  // Bit 2 only tells two loads apart when neither of them is a double word.
  // A double word covers both words of its 8-byte slot, so once one is
  // involved, offsets that differ only in bit 2 still count as a conflict.
  // Example (same base):
  //   memw(Rs+#0) and memw(Rs+#4) differ in bit 2  -> no conflict
  //   memw(Rs+#0) and memh(Rs+#4) differ in bit 2  -> no conflict
  //   memd(Rs+#0) and memw(Rs+#4) match on 4:3     -> conflict
  //   memd(Rs+#0) and memd(Rs+#8) differ in bit 3  -> no conflict
  bool HasDoubleWord = A0->Size == 8 || A1->Size == 8;
  int64_t BankMask = HasDoubleWord ? 0x18 : 0x1C;
  return ((A0->Offset ^ A1->Offset) & BankMask) == 0;
}

void HexagonSubtarget::BankConflictMutation::apply(ScheduleDAGInstrs *DAG) {
  if (!EnableCheckBankConflict)
    return;

  const auto &HII = static_cast<const HexagonInstrInfo&>(*DAG->TII);

  // Create artificial edges between loads that could likely cause a bank
  // conflict. Since such loads would normally not have any dependency
  // between them, we cannot rely on existing edges.
  for (unsigned i = 0, e = DAG->SUnits.size(); i != e; ++i) {
    SUnit &S0 = DAG->SUnits[i];
    if (!getBankAddr(*S0.getInstr(), HII))
      continue;
    // Scan only up to 32 instructions ahead (to avoid n^2 complexity).
    for (unsigned j = i+1, m = std::min(i+32, e); j != m; ++j) {
      SUnit &S1 = DAG->SUnits[j];
      // Use the conservative check here: the edge only guides the scheduler,
      // and the packetizer decides whether the loads can share a packet.
      if (!mayHaveBankConflict(*S0.getInstr(), *S1.getInstr(), HII,
                               /*IgnoreSameLine=*/false))
        continue;
      // The loads likely conflict: add an artificial edge and set extra
      // latency.
      SDep A(&S0, SDep::Artificial);
      A.setLatency(1);
      S1.addPred(A, true);
    }
  }
}

void HexagonSubtarget::HLXPseudoMutation::apply(ScheduleDAGInstrs *DAG) {
  const auto &HII = static_cast<const HexagonInstrInfo &>(*DAG->TII);
  const auto &HST = DAG->MF.getSubtarget<HexagonSubtarget>();
  if (!HST.useHLXOps())
    return;

  auto setLatency = [](SUnit &From, SUnit &To) {
    for (SDep &Dep : From.Succs)
      if (Dep.getSUnit() == &To && Dep.getKind() == SDep::Data)
        Dep.setLatency(0);
    for (SDep &Dep : To.Preds)
      if (Dep.getSUnit() == &From && Dep.getKind() == SDep::Data)
        Dep.setLatency(0);
  };

  for (SUnit &SU : DAG->SUnits) {
    if (!SU.isInstr())
      continue;

    MachineInstr &MI = *SU.getInstr();
    if (HII.isTmpLoadPseudo(MI)) {
      for (SDep &Dep : SU.Succs)
        if (Dep.getKind() == SDep::Data)
          setLatency(SU, *Dep.getSUnit());
    } else if (HII.isNewStorePseudo(MI)) {
      for (SDep &Dep : SU.Preds)
        if (Dep.getKind() == SDep::Data)
          setLatency(*Dep.getSUnit(), SU);
    }
  }
}

/// Enable use of alias analysis during code generation (during MI
/// scheduling, DAGCombine, etc.).
bool HexagonSubtarget::useAA() const {
  if (OptLevel != CodeGenOptLevel::None)
    return true;
  return false;
}

/// Perform target specific adjustments to the latency of a schedule
/// dependency.
void HexagonSubtarget::adjustSchedDependency(
    SUnit *Src, int SrcOpIdx, SUnit *Dst, int DstOpIdx, SDep &Dep,
    const TargetSchedModel *SchedModel) const {
  if (!Src->isInstr() || !Dst->isInstr())
    return;

  MachineInstr *SrcInst = Src->getInstr();
  MachineInstr *DstInst = Dst->getInstr();
  const HexagonInstrInfo *QII = getInstrInfo();

  // Instructions with .new operands have zero latency.
  SmallPtrSet<SUnit *, 4> ExclSrc;
  SmallPtrSet<SUnit *, 4> ExclDst;
  if (QII->canExecuteInBundle(*SrcInst, *DstInst) &&
      isBestZeroLatency(Src, Dst, QII, ExclSrc, ExclDst)) {
    Dep.setLatency(0);
    return;
  }

  // Set the latency for a copy to zero since we hope that is will get
  // removed.
  if (DstInst->isCopy())
    Dep.setLatency(0);

  // If it's a REG_SEQUENCE/COPY, use its destination instruction to determine
  // the correct latency.
  // If there are multiple uses of the def of COPY/REG_SEQUENCE, set the latency
  // only if the latencies on all the uses are equal, otherwise set it to
  // default.
  if ((DstInst->isRegSequence() || DstInst->isCopy())) {
    Register DReg = DstInst->getOperand(0).getReg();
    std::optional<unsigned> DLatency;
    for (const auto &DDep : Dst->Succs) {
      MachineInstr *DDst = DDep.getSUnit()->getInstr();
      int UseIdx = -1;
      for (unsigned OpNum = 0; OpNum < DDst->getNumOperands(); OpNum++) {
        const MachineOperand &MO = DDst->getOperand(OpNum);
        if (MO.isReg() && MO.getReg() && MO.isUse() && MO.getReg() == DReg) {
          UseIdx = OpNum;
          break;
        }
      }

      if (UseIdx == -1)
        continue;

      std::optional<unsigned> Latency =
          InstrInfo.getOperandLatency(&InstrItins, *SrcInst, 0, *DDst, UseIdx);

      // Set DLatency for the first time.
      if (!DLatency)
        DLatency = Latency;

      // For multiple uses, if the Latency is different across uses, reset
      // DLatency.
      if (DLatency != Latency) {
        DLatency = std::nullopt;
        break;
      }
    }
    Dep.setLatency(DLatency.value_or(0));
  }

  // Try to schedule uses near definitions to generate .cur.
  ExclSrc.clear();
  ExclDst.clear();
  if (EnableDotCurSched && QII->isToBeScheduledASAP(*SrcInst, *DstInst) &&
      isBestZeroLatency(Src, Dst, QII, ExclSrc, ExclDst)) {
    Dep.setLatency(0);
    return;
  }
  int Latency = Dep.getLatency();
  bool IsArtificial = Dep.isArtificial();
  Latency = updateLatency(*SrcInst, *DstInst, IsArtificial, Latency);
  Dep.setLatency(Latency);
}

void HexagonSubtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(std::make_unique<UsrOverflowMutation>());
  Mutations.push_back(std::make_unique<HVXMemLatencyMutation>());
  Mutations.push_back(std::make_unique<HMXActivateWeightMutation>());
  Mutations.push_back(std::make_unique<BankConflictMutation>());
  Mutations.push_back(std::make_unique<HLXPseudoMutation>());
}

void HexagonSubtarget::getSMSMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(std::make_unique<UsrOverflowMutation>());
  Mutations.push_back(std::make_unique<HVXMemLatencyMutation>());
  Mutations.push_back(std::make_unique<HMXActivateWeightMutation>());
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

int HexagonSubtarget::updateLatency(MachineInstr &SrcInst,
                                    MachineInstr &DstInst, bool IsArtificial,
                                    int Latency) const {
  if (IsArtificial)
    return 1;
  if (!hasV60Ops())
    return Latency;

  const HexagonInstrInfo &QII = *getInstrInfo();
  // BSB scheduling.
  if (QII.isHVXVec(SrcInst) || useBSBScheduling())
    Latency = (Latency + 1) >> 1;
  return Latency;
}

void HexagonSubtarget::restoreLatency(SUnit *Src, SUnit *Dst) const {
  MachineInstr *SrcI = Src->getInstr();
  for (auto &I : Src->Succs) {
    if (!I.isAssignedRegDep() || I.getSUnit() != Dst)
      continue;
    Register DepR = I.getReg();
    int DefIdx = -1;
    for (unsigned OpNum = 0; OpNum < SrcI->getNumOperands(); OpNum++) {
      const MachineOperand &MO = SrcI->getOperand(OpNum);
      bool IsSameOrSubReg = false;
      if (MO.isReg()) {
        Register MOReg = MO.getReg();
        if (DepR.isVirtual()) {
          IsSameOrSubReg = (MOReg == DepR);
        } else {
          IsSameOrSubReg = getRegisterInfo()->isSubRegisterEq(DepR, MOReg);
        }
        if (MO.isDef() && IsSameOrSubReg)
          DefIdx = OpNum;
      }
    }
    assert(DefIdx >= 0 && "Def Reg not found in Src MI");
    MachineInstr *DstI = Dst->getInstr();
    SDep T = I;
    for (unsigned OpNum = 0; OpNum < DstI->getNumOperands(); OpNum++) {
      const MachineOperand &MO = DstI->getOperand(OpNum);
      if (MO.isReg() && MO.isUse() && MO.getReg() == DepR) {
        std::optional<unsigned> Latency = InstrInfo.getOperandLatency(
            &InstrItins, *SrcI, DefIdx, *DstI, OpNum);

        // For some instructions (ex: COPY), we might end up with < 0 latency
        // as they don't have any Itinerary class associated with them.
        if (!Latency)
          Latency = 0;
        bool IsArtificial = I.isArtificial();
        Latency = updateLatency(*SrcI, *DstI, IsArtificial, *Latency);
        I.setLatency(*Latency);
      }
    }

    // Update the latency of opposite edge too.
    T.setSUnit(Src);
    auto F = find(Dst->Preds, T);
    assert(F != Dst->Preds.end());
    F->setLatency(I.getLatency());
  }
}

/// Change the latency between the two SUnits.
void HexagonSubtarget::changeLatency(SUnit *Src, SUnit *Dst, unsigned Lat)
      const {
  for (auto &I : Src->Succs) {
    if (!I.isAssignedRegDep() || I.getSUnit() != Dst)
      continue;
    SDep T = I;
    I.setLatency(Lat);

    // Update the latency of opposite edge too.
    T.setSUnit(Src);
    auto F = find(Dst->Preds, T);
    assert(F != Dst->Preds.end());
    F->setLatency(Lat);
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
bool HexagonSubtarget::isBestZeroLatency(
    SUnit *Src, SUnit *Dst, const HexagonInstrInfo *TII,
    SmallPtrSet<SUnit *, 4> &ExclSrc, SmallPtrSet<SUnit *, 4> &ExclDst) const {
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
    if (!hasV60Ops())
      changeLatency(SrcBest, Dst, 1);
    else
      restoreLatency(SrcBest, Dst);
  }
  if (DstBest != nullptr) {
    if (!hasV60Ops())
      changeLatency(Src, DstBest, 1);
    else
      restoreLatency(Src, DstBest);
  }

  // Attempt to find another opportunity for zero latency in a different
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

bool HexagonSubtarget::enableSubRegLiveness() const { return true; }

Intrinsic::ID HexagonSubtarget::getIntrinsicId(unsigned Opc) const {
  struct Scalar {
    unsigned Opcode;
    Intrinsic::ID IntId;
  };
  struct Hvx {
    unsigned Opcode;
    Intrinsic::ID Int64Id, Int128Id;
  };

  static Scalar ScalarInts[] = {
#define GET_SCALAR_INTRINSICS
#include "HexagonDepInstrIntrinsics.inc"
#undef GET_SCALAR_INTRINSICS
  };

  static Hvx HvxInts[] = {
#define GET_HVX_INTRINSICS
#include "HexagonDepInstrIntrinsics.inc"
#undef GET_HVX_INTRINSICS
  };

  const auto CmpOpcode = [](auto A, auto B) { return A.Opcode < B.Opcode; };
  [[maybe_unused]] static bool SortedScalar =
      (llvm::sort(ScalarInts, CmpOpcode), true);
  [[maybe_unused]] static bool SortedHvx =
      (llvm::sort(HvxInts, CmpOpcode), true);

  auto [BS, ES] = std::make_pair(std::begin(ScalarInts), std::end(ScalarInts));
  auto [BH, EH] = std::make_pair(std::begin(HvxInts), std::end(HvxInts));

  auto FoundScalar = std::lower_bound(BS, ES, Scalar{Opc, 0}, CmpOpcode);
  if (FoundScalar != ES && FoundScalar->Opcode == Opc)
    return FoundScalar->IntId;

  auto FoundHvx = std::lower_bound(BH, EH, Hvx{Opc, 0, 0}, CmpOpcode);
  if (FoundHvx != EH && FoundHvx->Opcode == Opc) {
    unsigned HwLen = getVectorLength();
    if (HwLen == 64)
      return FoundHvx->Int64Id;
    if (HwLen == 128)
      return FoundHvx->Int128Id;
  }

  std::string error = "Invalid opcode (" + std::to_string(Opc) + ")";
  llvm_unreachable(error.c_str());
  return 0;
}
