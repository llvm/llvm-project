//===-- GCNPreRAAntiHints.cpp - MFMA register anti-hints ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert register allocation anti-hints.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAAntiHints.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetSchedule.h"

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "amdgpu-anti-hints"

namespace HC = llvm::AMDGPU::HazardClass;

enum class AntiHintRule {
  None,
  MFMAWAW,
  MFMAWAR,
  WMMAWARAB,
  SWMMACWARIndex,
  WMMAWAW,
  TransWAR,
  MemAddrWAR,
  VAVdstWAR,
  All,
};

static cl::bits<AntiHintRule> AntiHintRuleSelection(
    "amdgpu-anti-hints-rules", cl::Hidden, cl::CommaSeparated,
    cl::desc("Anti-hints rules to select."),
    cl::values(clEnumValN(AntiHintRule::None, "none", "Select no rules"),
               clEnumValN(AntiHintRule::MFMAWAW, "mfma-waw",
                          "MFMA destination write-after-write"),
               clEnumValN(AntiHintRule::MFMAWAR, "mfma-war",
                          "XDL MFMA src2 write-after-read"),
               clEnumValN(AntiHintRule::WMMAWARAB, "wmma-war-ab",
                          "XDL WMMA A/B source write-after-read"),
               clEnumValN(AntiHintRule::SWMMACWARIndex, "swmmac-war-index",
                          "XDL SWMMAC sparse index write-after-read"),
               clEnumValN(AntiHintRule::WMMAWAW, "wmma-waw",
                          "XDL WMMA destination write-after-write"),
               clEnumValN(AntiHintRule::TransWAR, "trans-war",
                          "TRANS source write-after-read"),
               clEnumValN(AntiHintRule::MemAddrWAR, "mem-addr-war",
                          "Memory address/data write-after-read (s_wait_xcnt)"),
               clEnumValN(AntiHintRule::VAVdstWAR, "va-vdst-war",
                          "VALU source write-after-read by a load "
                          "(s_wait_alu va_vdst)"),
               clEnumValN(AntiHintRule::All, "all",
                          "Select all rules (default); per-rule "
                          "-amdgpu-anti-hints-for-* still apply")));

static cl::opt<bool>
    EnableAntiHintsForAddr("amdgpu-anti-hints-for-addr", cl::Hidden,
                           cl::desc("Enable Anti-Hints for memory address "
                                    "operands and subsequent VGPR writes to "
                                    "avoid wait xcnt."),
                           cl::init(true));

static cl::opt<bool>
    EnableAntiHintsForVAVdst("amdgpu-anti-hints-for-va-vdst", cl::Hidden,
                             cl::desc("Enable Anti-Hints for VA-VDST."),
                             cl::init(false));

static cl::opt<unsigned>
    VAVDSTLookbackWindow("amdgpu-va-vdst-lookback-window", cl::Hidden,
                         cl::desc("Lookback window for VA_VDST anti-hints."),
                         cl::init(32));

static cl::opt<unsigned>
    AddrAntiHintWindow("amdgpu-addr-anti-hint-window", cl::Hidden,
                       cl::desc("Number of later memory instructions an "
                                "address anti-hint stays open."),
                       cl::init(16));

namespace {

// Classify the MI into a HazardClassMask.
HazardClassMask getInstHazardClass(const MachineInstr &MI,
                                   const HazardContext &Ctx) {
  const SIInstrInfo &TII = *Ctx.TII;
  HazardClassMask Mask = HC::None;

  if (TII.isLDSDMA(MI))
    Mask = HC::VALU | HC::VMEM | HC::DS | HC::LDSDMA;
  else if (TII.isWMMA(MI) || SIInstrInfo::isSWMMAC(MI))
    Mask = HC::WMMA;
  else if (TII.isMFMA(MI))
    Mask = HC::MFMA;
  else if (SIInstrInfo::isTRANS(MI))
    Mask = HC::TRANS;
  else if (SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/true))
    Mask = HC::VALU;
  else if (TII.isDS(MI))
    Mask = HC::DS;
  else if (TII.isVMEM(MI))
    Mask = HC::VMEM;
  else if (TII.isSMRD(MI))
    Mask = HC::SMEM;
  else if (TII.isEXP(MI))
    Mask = HC::EXP;
  else if (SIInstrInfo::isSALU(MI))
    Mask = HC::SALU;
  else if (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
           Ctx.TRI->hasVGPRs(Ctx.MRI->getRegClass(MI.getOperand(0).getReg())))
    Mask = HC::VALU;
  if (SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/true))
    Mask |= HC::RawVALU;
  if (MI.mayLoad())
    Mask |= HC::Load;
  for (const MachineOperand &MO : MI.defs()) {
    if (MO.isReg() && MO.getReg().isVirtual() &&
        Ctx.TRI->hasVGPRs(Ctx.MRI->getRegClass(MO.getReg()))) {
      Mask |= HC::WritesVGPR;
      break;
    }
  }

  return Mask;
}

void collectOperandRegs(const MachineInstr &MI, HazardOperand Op,
                        const HazardContext &Ctx,
                        SmallVectorImpl<Register> &Out) {
  const SIInstrInfo &TII = *Ctx.TII;
  auto Add = [&](const MachineOperand *MO) {
    if (MO && MO->isReg() && MO->getReg().isVirtual() &&
        Ctx.TRI->hasVGPRs(Ctx.MRI->getRegClass(MO->getReg())))
      Out.push_back(MO->getReg());
  };
  auto Named = [&](AMDGPU::OpName N) { Add(TII.getNamedOperand(MI, N)); };
  switch (Op) {
  case HazardOperand::None:
    break;
  case HazardOperand::Def:
    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isDef())
        Add(&MO);
    break;
  case HazardOperand::Src0:
    Named(AMDGPU::OpName::src0);
    break;
  case HazardOperand::Src1:
    Named(AMDGPU::OpName::src1);
    break;
  case HazardOperand::Src2:
    Named(AMDGPU::OpName::src2);
    break;
  case HazardOperand::Src0Src1:
    Named(AMDGPU::OpName::src0);
    Named(AMDGPU::OpName::src1);
    break;
  case HazardOperand::Idx:
    Named(AMDGPU::OpName::idx);
    break;
  case HazardOperand::Vaddr:
    Named(AMDGPU::OpName::vaddr);
    break;
  case HazardOperand::AnySrc:
    Named(AMDGPU::OpName::src0);
    Named(AMDGPU::OpName::src1);
    Named(AMDGPU::OpName::src2);
    break;
  case HazardOperand::AnyUse:
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && MO.isUse())
        Add(&MO);
    break;
  }
}

enum class MFMAHazardKind { RAW, WAW, WAR };

// MFMA anti-hint wait-state window, mirroring GCNHazardRecognizer.cpp wait
// states.
unsigned mfmaWaitStates(const MachineInstr &MFMA, MFMAHazardKind Kind,
                        HazardClassMask ReaderClass, const HazardContext &Ctx) {
  const SIInstrInfo &TII = *Ctx.TII;
  const GCNSubtarget &ST = *Ctx.ST;
  const int NumPasses = Ctx.SchedModel->computeInstrLatency(&MFMA);
  const bool IsDGEMM = SIInstrInfo::isDGEMM(MFMA.getOpcode());
  const bool Mem = ReaderClass & (HC::VMEM | HC::DS | HC::EXP);

  auto GFX940NPass = [&]() -> unsigned {
    return TII.isXDL(MFMA)
               ? NumPasses + 3 + (NumPasses != 2 && ST.hasGFX950Insts())
               : NumPasses + 2;
  };
  auto SMFMANPass = [&]() -> unsigned {
    switch (NumPasses) {
    case 2:
      return 5;
    case 8:
      return 11;
    case 16:
      return 19;
    default:
      return 0;
    }
  };

  switch (Kind) {
  case MFMAHazardKind::RAW:
    if (IsDGEMM) {
      switch (NumPasses) {
      case 4:
        return Mem ? 9 : 6;
      case 8:
      case 16:
        return Mem ? 18 : (ST.hasGFX950Insts() ? 19 : 11);
      default:
        return 0;
      }
    }
    return ST.hasGFX940Insts() ? GFX940NPass() : SMFMANPass();

  case MFMAHazardKind::WAW:
    if (IsDGEMM) {
      switch (NumPasses) {
      case 4:
        return 6;
      case 8:
      case 16:
        return 11;
      default:
        return 0;
      }
    }
    return ST.hasGFX940Insts() ? GFX940NPass() : SMFMANPass();

  case MFMAHazardKind::WAR:
    switch (NumPasses) {
    case 2:
      return 1;
    case 4:
      return 3;
    case 8:
      return 7;
    case 16:
      return 15;
    default:
      return 15;
    }
  }
  return 0;
}

unsigned mfmaWawWindow(const MachineInstr &P, HazardClassMask,
                       const HazardContext &Ctx) {
  return mfmaWaitStates(P, MFMAHazardKind::WAW, HC::None, Ctx);
}
unsigned mfmaWarWindow(const MachineInstr &P, HazardClassMask,
                       const HazardContext &Ctx) {
  return mfmaWaitStates(P, MFMAHazardKind::WAR, HC::None, Ctx);
}

constexpr unsigned NumWMMAHazardCategories = 7;
unsigned wmmaHazardCategory(const MachineInstr &Producer,
                            const HazardContext &Ctx) {
  const bool IsSWMMAC = SIInstrInfo::isSWMMAC(Producer);
  const bool LowestRate = Ctx.ST->hasGFX125xLowestRateWMMA();
  switch (Ctx.SchedModel->computeInstrLatency(&Producer)) {
  case 4:
    return 6;
  case 8:
    return IsSWMMAC ? 2 : 0;
  case 16:
    return LowestRate ? 4 : (IsSWMMAC ? 3 : 1);
  case 32:
    return 5;
  default:
    return NumWMMAHazardCategories;
  }
}

// WMMA/SWMMAC co-exec window.
unsigned wmmaCoexecWindow(const MachineInstr &Producer,
                          HazardClassMask ConsumerClass,
                          const HazardContext &Ctx) {
  const unsigned Category = wmmaHazardCategory(Producer, Ctx);
  if (Category >= NumWMMAHazardCategories)
    return 0;

  static constexpr unsigned WMMAWaitStates[] = {5, 9, 3, 5, 9, 17, 2};
  static constexpr unsigned VALUWaitStates[] = {4, 8, 2, 4, 8, 16, 1};
  return (ConsumerClass & HC::WMMA) ? WMMAWaitStates[Category]
                                    : VALUWaitStates[Category];
}

unsigned mfmaReaderRawWindow(const MachineInstr &Producer,
                             HazardClassMask ReaderClass,
                             const HazardContext &Ctx) {
  return mfmaWaitStates(Producer, MFMAHazardKind::RAW, ReaderClass, Ctx);
}

bool hasMFMAHazard(const HazardContext &Ctx) {
  return Ctx.ST->hasGFX90AInsts();
}

bool hasWMMACoexecHazard(const HazardContext &Ctx) {
  return Ctx.ST->hasWMMACoexecutionHazards();
}

bool hasTransCoexecHazard(const HazardContext &Ctx) {
  return Ctx.ST->hasTransCoexecutionHazard();
}

bool hasGFX1250Insts(const HazardContext &Ctx) {
  return Ctx.ST->hasGFX1250Insts();
}

bool ruleSelected(AntiHintRule Rule) {
  // No -amdgpu-anti-hints-rules on the command line selects every rule.
  if (!AntiHintRuleSelection.getNumOccurrences() ||
      AntiHintRuleSelection.isSet(AntiHintRule::All))
    return true;
  return AntiHintRuleSelection.isSet(Rule);
}

bool isMFMAWAWRuleEnabled(const HazardContext &Ctx) {
  return hasMFMAHazard(Ctx) && ruleSelected(AntiHintRule::MFMAWAW);
}

bool isMFMAWARRuleEnabled(const HazardContext &Ctx) {
  return hasMFMAHazard(Ctx) && ruleSelected(AntiHintRule::MFMAWAR);
}

bool isWMMAWARABRuleEnabled(const HazardContext &Ctx) {
  return hasWMMACoexecHazard(Ctx) && ruleSelected(AntiHintRule::WMMAWARAB);
}

bool isSWMMACWARIndexRuleEnabled(const HazardContext &Ctx) {
  return hasWMMACoexecHazard(Ctx) && ruleSelected(AntiHintRule::SWMMACWARIndex);
}

bool isWMMAWAWRuleEnabled(const HazardContext &Ctx) {
  return hasWMMACoexecHazard(Ctx) && ruleSelected(AntiHintRule::WMMAWAW);
}

bool isTransWARRuleEnabled(const HazardContext &Ctx) {
  return hasTransCoexecHazard(Ctx) && ruleSelected(AntiHintRule::TransWAR);
}

bool isAddrWARRuleEnabled(const HazardContext &Ctx) {
  return EnableAntiHintsForAddr && hasGFX1250Insts(Ctx) &&
         ruleSelected(AntiHintRule::MemAddrWAR);
}

bool isVAVdstWARRuleEnabled(const HazardContext &Ctx) {
  return EnableAntiHintsForVAVdst && hasGFX1250Insts(Ctx) &&
         ruleSelected(AntiHintRule::VAVdstWAR);
}

bool isXDLMFMA(const MachineInstr &MI, const HazardContext &Ctx) {
  return Ctx.TII->isXDL(MI);
}

bool isXDLWMMA(const MachineInstr &MI, const HazardContext &Ctx) {
  return Ctx.TII->isXDLWMMA(MI);
}

bool isXDLSWMMAC(const MachineInstr &MI, const HazardContext &Ctx) {
  return Ctx.TII->isXDLWMMA(MI) && SIInstrInfo::isSWMMAC(MI);
}

bool isDSorVMEMLoad(const MachineInstr &MI, const HazardContext &Ctx) {
  return MI.mayLoad() && (Ctx.TII->isDS(MI) || Ctx.TII->isVMEM(MI));
}

unsigned resolveWindow(const ConsumerTarget &CT, const MachineInstr &MI,
                       const HazardContext &Ctx) {
  // Explicity given window length overrides the computed one.
  if (CT.Window.OptWindowLength)
    return *CT.Window.OptWindowLength;
  if (CT.Window.Fn)
    return CT.Window.Fn(MI, CT.Side.Match.AnyOf, Ctx);
  return CT.Window.WindowLength;
}

// Build the anti-hints rules.
class HazardRuleSet {
  SmallVector<HazardAntiHintRule, 0> Rules;

public:
  class RuleBuilder {

    HazardRuleSet &S;
    unsigned Idx;
    HazardAntiHintRule &rule() const { return S.Rules[Idx]; }

  public:
    RuleBuilder(HazardRuleSet &S, unsigned Idx) : S(S), Idx(Idx) {}

    RuleBuilder &producer(ClassMatch M, HazardOperand Op,
                          InstPredicate Predicate = nullptr) {
      rule().Producer = {M, Op, Predicate};
      return *this;
    }

    RuleBuilder &rawCredit(AdvanceForRawWindowFn AdvanceForRawWindow) {
      rule().AdvanceForRawWindow = AdvanceForRawWindow;
      return *this;
    }

    RuleBuilder &
    consumer(ClassMatch M, HazardOperand Op, WindowSpec Window,
             HazardClassMask CountMask = 0,
             AntiHintDirection Direction = AntiHintDirection::OneDirectional,
             InstPredicate Predicate = nullptr) {
      rule().Consumers.push_back(
          {{M, Op, Predicate}, Window, CountMask, Direction});
      return *this;
    }

    RuleBuilder &enabledIf(RulePredicate Predicate) {
      rule().Predicate = Predicate;
      return *this;
    }

    RuleBuilder &lifetime(Lifetime Life) {
      rule().Life = Life;
      return *this;
    }
  };

  RuleBuilder addRule() {
    Rules.emplace_back();
    return RuleBuilder(*this, Rules.size() - 1);
  }

  SmallVector<HazardAntiHintRule, 0> buildRules() { return std::move(Rules); }
};

// Here the anti-hints rules are inserted.
SmallVector<HazardAntiHintRule, 0> buildAntiHintsRules() {
  using HO = HazardOperand;
  HazardRuleSet S;

  // MFMA relevent windows and consumers
  const WindowSpec MfmaWawWindow{0, nullptr, mfmaWawWindow};
  const WindowSpec MfmaWarWindow{0, nullptr, mfmaWarWindow};
  const ClassMatch MfmaConsumers = {HC::DS | HC::VALU | HC::VMEM | HC::TRANS |
                                    HC::EXP};

  // WMMA relevent windows and consumers
  const WindowSpec WmmaCoexecWindow{0, nullptr, wmmaCoexecWindow};
  const ClassMatch WmmaCoexecConsumers = {/*AnyOf=*/HC::VALU | HC::TRANS,
                                          /*AllOf=*/HC::None,
                                          /*NoneOf=*/HC::LDSDMA};
  const HazardClassMask CoexecCounters = HC::VALU | HC::TRANS | HC::WMMA;

  // MFMA WAW rules
  S.addRule()
      .enabledIf(isMFMAWAWRuleEnabled)
      .producer({HC::MFMA}, HO::Def)
      .rawCredit(mfmaReaderRawWindow)
      .consumer(MfmaConsumers, HO::Def, MfmaWawWindow, HC::None,
                AntiHintDirection::OneDirectional);

  // MFMA WAR rules
  S.addRule()
      .enabledIf(isMFMAWARRuleEnabled)
      .producer({HC::MFMA}, HO::Src2, isXDLMFMA)
      .rawCredit(mfmaReaderRawWindow)
      .consumer(MfmaConsumers, HO::Def, MfmaWarWindow, HC::None,
                AntiHintDirection::OneDirectional);

  // WMMA WAR.
  S.addRule()
      .enabledIf(isWMMAWARABRuleEnabled)
      .producer({HC::WMMA}, HO::Src0Src1, isXDLWMMA)
      .consumer(WmmaCoexecConsumers, HO::Def, WmmaCoexecWindow, CoexecCounters,
                AntiHintDirection::OneDirectional)
      .consumer({HC::VMEM | HC::DS}, HO::Def, WmmaCoexecWindow, CoexecCounters,
                AntiHintDirection::OneDirectional);

  // SWMMAC WAR.
  S.addRule()
      .enabledIf(isSWMMACWARIndexRuleEnabled)
      .producer({HC::WMMA}, HO::Src2, isXDLSWMMAC)
      .consumer(WmmaCoexecConsumers, HO::Def, WmmaCoexecWindow, CoexecCounters,
                AntiHintDirection::OneDirectional);

  // WMMA WAW.
  S.addRule()
      .enabledIf(isWMMAWAWRuleEnabled)
      .producer({HC::WMMA}, HO::Def, isXDLWMMA)
      .consumer(WmmaCoexecConsumers, HO::Def, WmmaCoexecWindow, CoexecCounters,
                AntiHintDirection::OneDirectional);

  // TRANS WAR.
  S.addRule()
      .enabledIf(isTransWARRuleEnabled)
      .producer({HC::TRANS}, HO::AnyUse)
      .consumer({HC::VALU | HC::WMMA}, HO::Def, {/*WindowLength=*/1},
                CoexecCounters, AntiHintDirection::OneDirectional);

  // Address WAR.
  S.addRule()
      .enabledIf(isAddrWARRuleEnabled)
      .producer({HC::VMEM}, HO::AnyUse)
      .consumer({HC::WritesVGPR}, HO::Def, {0, &AddrAntiHintWindow}, HC::VMEM,
                AntiHintDirection::Symmetric);

  // VA_VDST WAR.
  S.addRule()
      .enabledIf(isVAVdstWARRuleEnabled)
      .lifetime(Lifetime::RegisterCount)
      .producer({HC::RawVALU}, HO::AnyUse)
      .consumer({HC::Load}, HO::Def, {0, &VAVDSTLookbackWindow}, HC::None,
                AntiHintDirection::Symmetric, isDSorVMEMLoad);

  return S.buildRules();
}

ArrayRef<HazardAntiHintRule> getAntiHintsRules() {
  static const SmallVector<HazardAntiHintRule, 0> Rules = buildAntiHintsRules();
  return Rules;
}

struct AntiHintWindow {
  SmallVector<Register, 4> Regs;
  const MachineInstr *Producer = nullptr;
  unsigned Len = 0;
  unsigned Elapsed = 0;
};

using ConsumerTracking = SmallVector<AntiHintWindow, 3>;
// One per consumer target of a rule.
using RuleTracking = SmallVector<ConsumerTracking, 3>;

class AntiHintEngine {
  const HazardContext &Ctx;
  ArrayRef<HazardAntiHintRule> Rules;

  SmallVector<bool, 8> RuleApplies;
  bool AnyEnabled = false;

public:
  AntiHintEngine(const HazardContext &Ctx)
      : Ctx(Ctx), Rules(getAntiHintsRules()), RuleApplies(Rules.size()) {
    for (unsigned R = 0; R < Rules.size(); ++R) {
      const HazardAntiHintRule &Rule = Rules[R];
      RuleApplies[R] = !Rule.Predicate || Rule.Predicate(Ctx);
      AnyEnabled |= RuleApplies[R];
    }
  }

  void run(MachineFunction &MF) {
    if (!AnyEnabled)
      return;

    SmallVector<RuleTracking, 8> Tracking(Rules.size());
    for (unsigned R = 0; R < Rules.size(); ++R)
      Tracking[R].resize(Rules[R].Consumers.size());

    for (const MachineBasicBlock &MBB : MF) {
      for (RuleTracking &RT : Tracking)
        for (ConsumerTracking &Track : RT)
          Track.clear();
      for (const MachineInstr &MI : MBB) {
        if (MI.isMetaInstruction())
          continue;
        const HazardClassMask C = getInstHazardClass(MI, Ctx);
        // Wait states this instruction contributes to an open window.
        unsigned WaitStates = SIInstrInfo::getNumWaitStates(MI);
        addAntiHintsAndExpire(MI, C, WaitStates, Tracking);
        addWindows(MI, C, Tracking);
      }
    }
  }

private:
  // Check if the right class and predicate matches.
  bool sideMatches(const HazardSide &Side, const MachineInstr &MI,
                   HazardClassMask C) {
    return Side.Match.matches(C) &&
           (!Side.Predicate || Side.Predicate(MI, Ctx));
  }

  // Producer phase: open a window for each consumer target.
  void addWindows(const MachineInstr &MI, HazardClassMask C,
                  MutableArrayRef<RuleTracking> Tracking) {
    for (unsigned R = 0; R < Rules.size(); ++R) {
      const HazardAntiHintRule &Rule = Rules[R];
      if (!RuleApplies[R] || !sideMatches(Rule.Producer, MI, C))
        continue;

      SmallVector<Register, 4> Regs;
      // Collect the producer regs.
      collectOperandRegs(MI, Rule.Producer.Op, Ctx, Regs);
      if (Regs.empty())
        continue;
      for (unsigned ConsumerIdx = 0, E = Rule.Consumers.size();
           ConsumerIdx != E; ++ConsumerIdx)
        seedWindow(Rule, Rule.Consumers[ConsumerIdx], MI, Regs,
                   Tracking[R][ConsumerIdx]);
    }
  }

  void seedWindow(const HazardAntiHintRule &Rule, const ConsumerTarget &CT,
                  const MachineInstr &MI, ArrayRef<Register> Regs,
                  ConsumerTracking &Track) {
    unsigned Window = resolveWindow(CT, MI, Ctx);
    if (!Window)
      return;

    if (Rule.Life == Lifetime::RegisterCount) {
      // Window length caps the register count.
      if (Track.empty())
        Track.push_back({{}, &MI, Window, 0});
      SmallVectorImpl<Register> &Recent = Track.front().Regs;
      for (Register Reg : Regs) {
        if (llvm::is_contained(Recent, Reg))
          continue;
        Recent.push_back(Reg);
        if (Recent.size() > Window)
          Recent.erase(Recent.begin());
      }
      return;
    }

    Track.push_back({SmallVector<Register, 4>(Regs), &MI, Window, 0});
  }

  void advanceByRawWindow(const MachineInstr &MI, HazardClassMask C,
                          const HazardAntiHintRule &Rule,
                          const ConsumerTarget &CT, ConsumerTracking &Track) {

    if (!Rule.AdvanceForRawWindow || !CT.Side.Match.matches(C))
      return;
    for (AntiHintWindow &Window : Track) {
      // Determine if def generated by producer is read by the MI consumer.
      bool ReadsProducerDef = llvm::any_of(
          Window.Producer->all_defs(), [&](const MachineOperand &MO) {
            return MO.getReg().isVirtual() &&
                   MI.readsVirtualRegister(MO.getReg());
          });
      // Advance by RAW window from the producer if that is larger than current
      // Window.Elapsed.
      if (ReadsProducerDef)
        Window.Elapsed = std::max(
            Window.Elapsed, Rule.AdvanceForRawWindow(*Window.Producer, C, Ctx));
    }
  }

  // Consumer phase: add anti-hints, then charge and expire open windows.
  void addAntiHintsAndExpire(const MachineInstr &MI, HazardClassMask C,
                             unsigned WaitStates,
                             MutableArrayRef<RuleTracking> Tracking) {
    for (unsigned R = 0; R < Rules.size(); ++R) {
      const HazardAntiHintRule &Rule = Rules[R];
      if (!RuleApplies[R])
        continue;
      for (unsigned ConsumerIdx = 0, E = Rule.Consumers.size();
           ConsumerIdx != E; ++ConsumerIdx) {
        const ConsumerTarget &CT = Rule.Consumers[ConsumerIdx];
        ConsumerTracking &Track = Tracking[R][ConsumerIdx];
        if (Track.empty())
          continue;

        // Before adding the anti-hints, see if advancing by RAW window will
        // help remove the window.
        const bool Counts = Rule.Life == Lifetime::WindowBudget;
        if (Counts) {
          advanceByRawWindow(MI, C, Rule, CT, Track);
          llvm::erase_if(Track, [](const AntiHintWindow &Window) {
            return Window.Elapsed >= Window.Len;
          });
        }

        // Add anti-hints if the consumer matches the instruction.
        if (sideMatches(CT.Side, MI, C))
          addAntiHints(CT, MI, Track);

        // This instruction's own wait states count toward the next one.
        if (Counts && (!CT.CounterMask || (C & CT.CounterMask)))
          for (AntiHintWindow &Window : Track)
            Window.Elapsed += WaitStates;
      }
    }
  }

  bool isCopyOf(Register Cand, Register HazardReg) const {
    const MachineInstr *Def = Ctx.MRI->getUniqueVRegDef(Cand);
    return Def && Def->isCopy() && Def->getOperand(1).getReg() == HazardReg;
  }

  void addAntiHints(const ConsumerTarget &CT, const MachineInstr &MI,
                    const ConsumerTracking &Track) {
    if (Track.empty())
      return;
    SmallVector<Register, 4> ConsumerRegs;
    // Collect the consumer regs.
    collectOperandRegs(MI, CT.Side.Op, Ctx, ConsumerRegs);
    if (ConsumerRegs.empty())
      return;
    SlotIndex Slot = Ctx.LIS->getInstructionIndex(MI).getRegSlot();
    auto AntiHint = [&](Register ProducerReg) {
      if (!Ctx.LIS->hasInterval(ProducerReg))
        return;
      const LiveInterval &ProducerLI = Ctx.LIS->getInterval(ProducerReg);
      // Skip a live producer reg.
      if (ProducerLI.liveAt(Slot))
        return;
      for (Register ConsumerReg : ConsumerRegs) {
        if (ConsumerReg == ProducerReg || isCopyOf(ConsumerReg, ProducerReg))
          continue;

        Ctx.MRI->addRegAllocationAntiHints(ConsumerReg, ProducerReg);
        if (CT.Direction == AntiHintDirection::Symmetric)
          Ctx.MRI->addRegAllocationAntiHints(ProducerReg, ConsumerReg);
        LLVM_DEBUG(
            dbgs() << "anti-hint: keep " << printReg(ProducerReg, Ctx.TRI)
                   << (CT.Direction == AntiHintDirection::Symmetric ? " <-> "
                                                                    : " <- ")
                   << printReg(ConsumerReg, Ctx.TRI) << " (consumer "
                   << Ctx.TII->getName(MI.getOpcode()) << ")\n");
      }
    };
    for (const AntiHintWindow &Window : Track)
      for (Register ProducerReg : Window.Regs)
        AntiHint(ProducerReg);
  }
};

} // namespace

void AMDGPU::applyAntiHintRules(MachineFunction &MF, const HazardContext &Ctx) {
  AntiHintEngine(Ctx).run(MF);
}
