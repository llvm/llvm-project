//===-- AMDGPURBLegalizeRules.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Definitions of RBLegalize Rules for all opcodes.
/// Implementation of container for all the Rules and search.
/// Fast search for most common case when Rule.Predicate checks LLT and
/// uniformity of register in operand 0.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURBLegalizeRules.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;
using namespace AMDGPU;

RegBankLLTMapping::RegBankLLTMapping(
    std::initializer_list<RegBankLLTMapingApplyID> DstOpMappingList,
    std::initializer_list<RegBankLLTMapingApplyID> SrcOpMappingList,
    LoweringMethodID LoweringMethod)
    : DstOpMapping(DstOpMappingList), SrcOpMapping(SrcOpMappingList),
      LoweringMethod(LoweringMethod) {}

PredicateMapping::PredicateMapping(
    std::initializer_list<UniformityLLTOpPredicateID> OpList,
    std::function<bool(const MachineInstr &)> TestFunc)
    : OpUniformityAndTypes(OpList), TestFunc(TestFunc) {}

bool matchUniformityAndLLT(Register Reg, UniformityLLTOpPredicateID UniID,
                           const MachineUniformityInfo &MUI,
                           const MachineRegisterInfo &MRI) {
  switch (UniID) {
  case S1:
    return MRI.getType(Reg) == LLT::scalar(1);
  case S16:
    return MRI.getType(Reg) == LLT::scalar(16);
  case S32:
    return MRI.getType(Reg) == LLT::scalar(32);
  case S64:
    return MRI.getType(Reg) == LLT::scalar(64);
  case P1:
    return MRI.getType(Reg) == LLT::pointer(1, 64);

  case UniS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isUniform(Reg);
  case UniS16:
    return MRI.getType(Reg) == LLT::scalar(16) && MUI.isUniform(Reg);
  case UniS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isUniform(Reg);
  case UniS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isUniform(Reg);

  case DivS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isDivergent(Reg);
  case DivS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isDivergent(Reg);
  case DivS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isDivergent(Reg);
  case DivP1:
    return MRI.getType(Reg) == LLT::pointer(1, 64) && MUI.isDivergent(Reg);

  case _:
    return true;
  default:
    llvm_unreachable("missing matchUniformityAndLLT\n");
  }
}

bool PredicateMapping::match(const MachineInstr &MI,
                             const MachineUniformityInfo &MUI,
                             const MachineRegisterInfo &MRI) const {
  // Check LLT signature.
  for (unsigned i = 0; i < OpUniformityAndTypes.size(); ++i) {
    if (OpUniformityAndTypes[i] == _) {
      if (MI.getOperand(i).isReg() &&
          MI.getOperand(i).getReg() != AMDGPU::NoRegister)
        return false;
      continue;
    }

    // Remaining IDs check registers.
    if (!MI.getOperand(i).isReg())
      return false;

    if (!matchUniformityAndLLT(MI.getOperand(i).getReg(),
                               OpUniformityAndTypes[i], MUI, MRI))
      return false;
  }

  // More complex check.
  if (TestFunc)
    return TestFunc(MI);

  return true;
}

SetOfRulesForOpcode::SetOfRulesForOpcode() {}

SetOfRulesForOpcode::SetOfRulesForOpcode(FastRulesTypes FastTypes)
    : FastTypes(FastTypes) {}

UniformityLLTOpPredicateID LLTToId(LLT Ty) {
  if (Ty == LLT::scalar(16))
    return S16;
  if (Ty == LLT::scalar(32))
    return S32;
  if (Ty == LLT::scalar(64))
    return S64;
  if (Ty == LLT::fixed_vector(2, 16))
    return V2S16;
  if (Ty == LLT::fixed_vector(2, 32))
    return V2S32;
  if (Ty == LLT::fixed_vector(3, 32))
    return V3S32;
  if (Ty == LLT::fixed_vector(4, 32))
    return V4S32;
  return _;
}

const RegBankLLTMapping &
SetOfRulesForOpcode::findMappingForMI(const MachineInstr &MI,
                                      const MachineRegisterInfo &MRI,
                                      const MachineUniformityInfo &MUI) const {
  // Search in "Fast Rules".
  // Note: if fast rules are enabled, RegBankLLTMapping must be added in each
  // slot that could "match fast Predicate". If not, Invalid Mapping is
  // returned which results in failure, does not search "Slow Rules".
  if (FastTypes != No) {
    Register Reg = MI.getOperand(0).getReg();
    int Slot = getFastPredicateSlot(LLTToId(MRI.getType(Reg)));
    if (Slot != -1) {
      if (MUI.isUniform(Reg))
        return Uni[Slot];
      else
        return Div[Slot];
    }
  }

  // Slow search for more complex rules.
  for (const RBSRule &Rule : Rules) {
    if (Rule.Predicate.match(MI, MUI, MRI))
      return Rule.OperandMapping;
  }
  MI.dump();
  llvm_unreachable("no rules found for MI");
}

void SetOfRulesForOpcode::addRule(RBSRule Rule) { Rules.push_back(Rule); }

void SetOfRulesForOpcode::addFastRuleDivergent(UniformityLLTOpPredicateID Ty,
                                               RegBankLLTMapping RuleApplyIDs) {
  int Slot = getFastPredicateSlot(Ty);
  assert(Slot != -1 && "Ty unsupported in this FastRulesTypes");
  Div[Slot] = RuleApplyIDs;
}

void SetOfRulesForOpcode::addFastRuleUniform(UniformityLLTOpPredicateID Ty,
                                             RegBankLLTMapping RuleApplyIDs) {
  int Slot = getFastPredicateSlot(Ty);
  assert(Slot != -1 && "Ty unsupported in this FastRulesTypes");
  Uni[Slot] = RuleApplyIDs;
}

int SetOfRulesForOpcode::getFastPredicateSlot(
    UniformityLLTOpPredicateID Ty) const {
  switch (FastTypes) {
  case Standard:
    switch (Ty) {
    case S32:
      return 0;
    case S16:
      return 1;
    case S64:
      return 2;
    case V2S16:
      return 3;
    default:
      return -1;
    }
  case Vector:
    switch (Ty) {
    case S32:
      return 0;
    case V2S32:
      return 1;
    case V3S32:
      return 2;
    case V4S32:
      return 3;
    default:
      return -1;
    }
  default:
    return -1;
  }
}

RegBankLegalizeRules::RuleSetInitializer
RegBankLegalizeRules::addRulesForGOpcs(std::initializer_list<unsigned> OpcList,
                                       FastRulesTypes FastTypes) {
  return RuleSetInitializer(OpcList, GRulesAlias, GRules, FastTypes);
}

RegBankLegalizeRules::RuleSetInitializer
RegBankLegalizeRules::addRulesForIOpcs(std::initializer_list<unsigned> OpcList,
                                       FastRulesTypes FastTypes) {
  return RuleSetInitializer(OpcList, IRulesAlias, IRules, FastTypes);
}

const SetOfRulesForOpcode &
RegBankLegalizeRules::getRulesForOpc(MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::G_INTRINSIC || Opc == AMDGPU::G_INTRINSIC_CONVERGENT ||
      Opc == AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS ||
      Opc == AMDGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS) {
    unsigned IntrID = cast<GIntrinsic>(MI).getIntrinsicID();
    // assert(IRules.contains(IntrID));
    if (!IRulesAlias.contains(IntrID)) {
      MI.dump();
      llvm_unreachable("no rules for opc");
    }
    return IRules.at(IRulesAlias.at(IntrID));
  }
  // assert(GRules.contains(Opc));
  if (!GRulesAlias.contains(Opc)) {
    MI.dump();
    llvm_unreachable("no rules for opc");
  }
  return GRules.at(GRulesAlias.at(Opc));
}

// Initialize rules
RegBankLegalizeRules::RegBankLegalizeRules(const GCNSubtarget &_ST,
                                           MachineRegisterInfo &_MRI)
    : ST(&_ST), MRI(&_MRI) {

  addRulesForGOpcs({G_ADD}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_MUL}, Standard).Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_XOR, G_OR, G_AND}, Standard)
      .Any({{UniS1}, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32AExt}}})
      .Any({{DivS1}, {{Vcc}, {Vcc, Vcc}}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}, SplitTo32});

  addRulesForGOpcs({G_SHL}, Standard)
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr32}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32}});

  // Note: we only write S1 rules for G_IMPLICIT_DEF, G_CONSTANT, G_FCONSTANT
  // and G_FREEZE here, rest is trivially regbankselected earlier
  addRulesForGOpcs({G_CONSTANT})
      .Any({{UniS1, _}, {{Sgpr32Trunc}, {None}, UniCstExt}});

  addRulesForGOpcs({G_ICMP})
      .Any({{UniS1, _, S32}, {{Sgpr32Trunc}, {None, Sgpr32, Sgpr32}}})
      .Any({{DivS1, _, S32}, {{Vcc}, {None, Vgpr32, Vgpr32}}});

  addRulesForGOpcs({G_FCMP})
      .Any({{UniS1, _, S32}, {{UniInVcc}, {None, Vgpr32, Vgpr32}}})
      .Any({{DivS1, _, S32}, {{Vcc}, {None, Vgpr32, Vgpr32}}});

  addRulesForGOpcs({G_BRCOND})
      .Any({{UniS1}, {{}, {Sgpr32AExtBoolInReg}}})
      .Any({{DivS1}, {{}, {Vcc}}});

  addRulesForGOpcs({G_BR}).Any({{_}, {{}, {None}}});

  addRulesForGOpcs({G_SELECT}, Standard)
      .Div(S32, {{Vgpr32}, {Vcc, Vgpr32, Vgpr32}})
      .Uni(S32, {{Sgpr32}, {Sgpr32AExtBoolInReg, Sgpr32, Sgpr32}});

  addRulesForGOpcs({G_ANYEXT}).Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}});

  // In global-isel G_TRUNC in-reg is treated as no-op, inst selected into COPY.
  // It is up to user to deal with truncated bits.
  addRulesForGOpcs({G_TRUNC})
      .Any({{UniS16, S32}, {{Sgpr16}, {Sgpr32}}})
      // This is non-trivial. VgprToVccCopy is done using compare instruction.
      .Any({{DivS1, DivS32}, {{Vcc}, {Vgpr32}, VgprToVccCopy}});

  addRulesForGOpcs({G_ZEXT, G_SEXT})
      .Any({{UniS32, S1}, {{Sgpr32}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{UniS64, S32}, {{Sgpr64}, {Sgpr32}, Ext32To64}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}, Ext32To64}});

  addRulesForGOpcs({G_LOAD}).Any({{DivS32, DivP1}, {{Vgpr32}, {VgprP1}}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_LOAD}, Vector)
      .Div(V4S32, {{VgprV4S32}, {SgprV4S32, Vgpr32, Vgpr32, Sgpr32}})
      .Uni(V4S32, {{UniInVgprV4S32}, {SgprV4S32, Vgpr32, Vgpr32, Sgpr32}});

  addRulesForGOpcs({G_STORE})
      .Any({{S32, P1}, {{}, {Vgpr32, VgprP1}}})
      .Any({{S64, P1}, {{}, {Vgpr64, VgprP1}}})
      .Any({{V4S32, P1}, {{}, {VgprV4S32, VgprP1}}});

  addRulesForGOpcs({G_PTR_ADD}).Any({{DivP1}, {{VgprP1}, {VgprP1, Vgpr64}}});

  addRulesForGOpcs({G_ABS}, Standard).Uni(S16, {{Sgpr32Trunc}, {Sgpr32SExt}});

  bool hasSALUFloat = ST->hasSALUFloatInsts();

  addRulesForGOpcs({G_FADD}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasSALUFloat)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasSALUFloat)
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_FPTOUI})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS32, S32}, {{UniInVgprS32}, {Vgpr32}}}, !hasSALUFloat);

  addRulesForGOpcs({G_UITOFP})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS32, S32}, {{UniInVgprS32}, {Vgpr32}}}, !hasSALUFloat);

  using namespace Intrinsic;

  // This is "intrinsic lane mask" it was set to i32/i64 in llvm-ir.
  addRulesForIOpcs({amdgcn_end_cf}).Any({{_, S32}, {{}, {None, Sgpr32}}});

  addRulesForIOpcs({amdgcn_if_break}, Standard)
      .Uni(S32, {{Sgpr32}, {IntrId, Vcc, Sgpr32}});

} // end initialize rulles
