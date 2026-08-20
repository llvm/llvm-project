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

static cl::opt<std::string> AntiHintRuleSelection(
    "amdgpu-anti-hints-rules", cl::Hidden,
    cl::desc("Comma-separated anti-hints rules (waw, war), or all or none."),
    cl::init("all"));

namespace {

// Classify the MI into a HazardClassMask.
HazardClassMask getInstHazardClass(const MachineInstr &MI,
                                   const HazardContext &Ctx) {
  const SIInstrInfo &TII = *Ctx.TII;
  HazardClassMask Mask = HC::None;

  if (TII.isLDSDMA(MI))
    Mask = HC::VALU | HC::VMEM | HC::DS;
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

unsigned mfmaWawWindow(const MachineInstr &P, const HazardContext &Ctx) {
  return mfmaWaitStates(P, MFMAHazardKind::WAW, HC::None, Ctx);
}
unsigned mfmaWarWindow(const MachineInstr &P, const HazardContext &Ctx) {
  return mfmaWaitStates(P, MFMAHazardKind::WAR, HC::None, Ctx);
}

unsigned mfmaReaderRawWindow(const MachineInstr &Producer,
                             HazardClassMask ReaderClass,
                             const HazardContext &Ctx) {
  return mfmaWaitStates(Producer, MFMAHazardKind::RAW, ReaderClass, Ctx);
}

bool hasMFMAHazard(const HazardContext &Ctx) {
  return Ctx.ST->hasGFX90AInsts();
}

bool ruleSelected(StringRef Name) {
  StringRef Selection(AntiHintRuleSelection);
  if (Selection.equals_insensitive("all"))
    return true;
  if (Selection.equals_insensitive("none"))
    return false;
  SmallVector<StringRef, 3> Selected;
  Selection.split(Selected, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return llvm::any_of(Selected, [Name](StringRef S) {
    return S.trim().equals_insensitive(Name);
  });
}

bool isMFMAWAWRuleEnabled(const HazardContext &Ctx) {
  return hasMFMAHazard(Ctx) && ruleSelected("waw");
}

bool isMFMAWARRuleEnabled(const HazardContext &Ctx) {
  return hasMFMAHazard(Ctx) && ruleSelected("war");
}

bool isXDLMFMA(const MachineInstr &MI, const HazardContext &Ctx) {
  return Ctx.TII->isXDL(MI);
}

unsigned resolveWindow(const ConsumerTarget &CT, const MachineInstr &MI,
                       const HazardContext &Ctx) {
  // Explicity given window length overrides the computed one.
  if (CT.Window.OptWindowLength &&
      CT.Window.OptWindowLength->getNumOccurrences())
    return *CT.Window.OptWindowLength;
  if (CT.Window.Fn)
    return CT.Window.Fn(MI, Ctx);
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

    RuleBuilder &consumer(ClassMatch M, HazardOperand Op, WindowSpec Window,
                          HazardClassMask CountMask = 0,
                          ConsumerHint Hint = ConsumerHint::OneDirectional,
                          InstPredicate Predicate = nullptr) {
      rule().Consumers.push_back({{M, Op, Predicate}, Window, CountMask, Hint});
      return *this;
    }

    RuleBuilder &enabledIf(RulePredicate Predicate) {
      rule().Predicate = Predicate;
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

  const WindowSpec MfmaWawWindow{0, nullptr, mfmaWawWindow};
  const WindowSpec MfmaWarWindow{0, nullptr, mfmaWarWindow};
  const ClassMatch MfmaConsumers = {HC::DS | HC::VALU | HC::VMEM | HC::TRANS |
                                    HC::EXP};

  // MFMA WAW rules
  S.addRule()
      .enabledIf(isMFMAWAWRuleEnabled)
      .producer({HC::MFMA}, HO::Def)
      .rawCredit(mfmaReaderRawWindow)
      .consumer(MfmaConsumers, HO::Def, MfmaWawWindow, HC::None,
                ConsumerHint::OneDirectional);

  // MFMA WAR rules
  S.addRule()
      .enabledIf(isMFMAWARRuleEnabled)
      .producer({HC::MFMA}, HO::Src2, isXDLMFMA)
      .rawCredit(mfmaReaderRawWindow)
      .consumer(MfmaConsumers, HO::Def, MfmaWarWindow, HC::None,
                ConsumerHint::OneDirectional);

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
           ConsumerIdx != E; ++ConsumerIdx) {
        unsigned Window = resolveWindow(Rule.Consumers[ConsumerIdx], MI, Ctx);
        if (!Window)
          continue;
        Tracking[R][ConsumerIdx].push_back({Regs, &MI, Window, 0});
      }
    }
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
        advanceByRawWindow(MI, C, Rule, CT, Track);
        llvm::erase_if(Track, [](const AntiHintWindow &Window) {
          return Window.Elapsed >= Window.Len;
        });

        // Add anti-hints if the consumer matches the instruction.
        if (sideMatches(CT.Side, MI, C))
          addAntiHints(CT, MI, Track);

        // This instruction's own wait states count toward the next one.
        if (!CT.CounterMask || (C & CT.CounterMask))
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
      if (ProducerLI.liveAt(Slot))
        return;
      for (Register ConsumerReg : ConsumerRegs) {
        if (ConsumerReg == ProducerReg || isCopyOf(ConsumerReg, ProducerReg))
          continue;
        // Only allow anti-hints when producer and consumer reg live ranges do
        // not overlap
        if (Ctx.LIS->hasInterval(ConsumerReg) &&
            ProducerLI.overlaps(Ctx.LIS->getInterval(ConsumerReg)))
          continue;
        Ctx.MRI->addRegAllocationAntiHints(ConsumerReg, ProducerReg);
        if (CT.Hint == ConsumerHint::Symmetric)
          Ctx.MRI->addRegAllocationAntiHints(ProducerReg, ConsumerReg);
        LLVM_DEBUG(
            dbgs() << "anti-hint: keep " << printReg(ProducerReg, Ctx.TRI)
                   << (CT.Hint == ConsumerHint::Symmetric ? " <-> " : " <- ")
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
