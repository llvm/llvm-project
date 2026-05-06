//===-- AMDGPURegBankLegalizeRules.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Definitions of RegBankLegalize Rules for all opcodes.
/// Implementation of container for all the Rules and search.
/// Fast search for most common case when Rule.Predicate checks LLT and
/// uniformity of register in operand 0.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURegBankLegalizeRules.h"
#include "AMDGPUInstrInfo.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

#define DEBUG_TYPE "amdgpu-regbanklegalize"

using namespace llvm;
using namespace AMDGPU;

bool AMDGPU::isAnyPtr(LLT Ty, unsigned Width) {
  return Ty.isPointer() && Ty.getSizeInBits() == Width;
}

RegBankLLTMapping::RegBankLLTMapping(
    std::initializer_list<RegBankLLTMappingApplyID> DstOpMappingList,
    std::initializer_list<RegBankLLTMappingApplyID> SrcOpMappingList,
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
  case S128:
    return MRI.getType(Reg) == LLT::scalar(128);
  case P0:
    return MRI.getType(Reg) == LLT::pointer(0, 64);
  case P1:
    return MRI.getType(Reg) == LLT::pointer(1, 64);
  case P2:
    return MRI.getType(Reg) == LLT::pointer(2, 32);
  case P3:
    return MRI.getType(Reg) == LLT::pointer(3, 32);
  case P4:
    return MRI.getType(Reg) == LLT::pointer(4, 64);
  case P5:
    return MRI.getType(Reg) == LLT::pointer(5, 32);
  case P8:
    return MRI.getType(Reg) == LLT::pointer(8, 128);
  case Ptr32:
    return isAnyPtr(MRI.getType(Reg), 32);
  case Ptr64:
    return isAnyPtr(MRI.getType(Reg), 64);
  case Ptr128:
    return isAnyPtr(MRI.getType(Reg), 128);
  case V2S16:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 16);
  case V2S32:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 32);
  case V3S32:
    return MRI.getType(Reg) == LLT::fixed_vector(3, 32);
  case V4S32:
    return MRI.getType(Reg) == LLT::fixed_vector(4, 32);
  case B32:
    return MRI.getType(Reg).getSizeInBits() == 32;
  case B64:
    return MRI.getType(Reg).getSizeInBits() == 64;
  case B96:
    return MRI.getType(Reg).getSizeInBits() == 96;
  case B128:
    return MRI.getType(Reg).getSizeInBits() == 128;
  case B160:
    return MRI.getType(Reg).getSizeInBits() == 160;
  case B256:
    return MRI.getType(Reg).getSizeInBits() == 256;
  case B512:
    return MRI.getType(Reg).getSizeInBits() == 512;
  case DivAnyTy:
    return MUI.isDivergent(Reg);
  case UniS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isUniform(Reg);
  case UniS16:
    return MRI.getType(Reg) == LLT::scalar(16) && MUI.isUniform(Reg);
  case UniS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isUniform(Reg);
  case UniS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isUniform(Reg);
  case UniS128:
    return MRI.getType(Reg) == LLT::scalar(128) && MUI.isUniform(Reg);
  case UniP0:
    return MRI.getType(Reg) == LLT::pointer(0, 64) && MUI.isUniform(Reg);
  case UniP1:
    return MRI.getType(Reg) == LLT::pointer(1, 64) && MUI.isUniform(Reg);
  case UniP2:
    return MRI.getType(Reg) == LLT::pointer(2, 32) && MUI.isUniform(Reg);
  case UniP3:
    return MRI.getType(Reg) == LLT::pointer(3, 32) && MUI.isUniform(Reg);
  case UniP4:
    return MRI.getType(Reg) == LLT::pointer(4, 64) && MUI.isUniform(Reg);
  case UniP5:
    return MRI.getType(Reg) == LLT::pointer(5, 32) && MUI.isUniform(Reg);
  case UniP8:
    return MRI.getType(Reg) == LLT::pointer(8, 128) && MUI.isUniform(Reg);
  case UniPtr32:
    return isAnyPtr(MRI.getType(Reg), 32) && MUI.isUniform(Reg);
  case UniPtr64:
    return isAnyPtr(MRI.getType(Reg), 64) && MUI.isUniform(Reg);
  case UniPtr128:
    return isAnyPtr(MRI.getType(Reg), 128) && MUI.isUniform(Reg);
  case UniV2S16:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 16) && MUI.isUniform(Reg);
  case UniV2S32:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 32) && MUI.isUniform(Reg);
  case UniB32:
    return MRI.getType(Reg).getSizeInBits() == 32 && MUI.isUniform(Reg);
  case UniB64:
    return MRI.getType(Reg).getSizeInBits() == 64 && MUI.isUniform(Reg);
  case UniB96:
    return MRI.getType(Reg).getSizeInBits() == 96 && MUI.isUniform(Reg);
  case UniB128:
    return MRI.getType(Reg).getSizeInBits() == 128 && MUI.isUniform(Reg);
  case UniB160:
    return MRI.getType(Reg).getSizeInBits() == 160 && MUI.isUniform(Reg);
  case UniB256:
    return MRI.getType(Reg).getSizeInBits() == 256 && MUI.isUniform(Reg);
  case UniB512:
    return MRI.getType(Reg).getSizeInBits() == 512 && MUI.isUniform(Reg);
  case UniBRC: {
    if (!MUI.isUniform(Reg))
      return false;
    // Check if there is SGPR register class of same size as the LLT.
    const SIRegisterInfo *TRI =
        static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
    // There is no 16 bit SGPR register class. Extra size check is required
    // since getSGPRClassForBitWidth returns SReg_32RegClass for Size 16.
    unsigned LLTSize = MRI.getType(Reg).getSizeInBits();
    return LLTSize >= 32 && TRI->getSGPRClassForBitWidth(LLTSize);
  }
  case DivS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isDivergent(Reg);
  case DivS16:
    return MRI.getType(Reg) == LLT::scalar(16) && MUI.isDivergent(Reg);
  case DivS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isDivergent(Reg);
  case DivS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isDivergent(Reg);
  case DivS128:
    return MRI.getType(Reg) == LLT::scalar(128) && MUI.isDivergent(Reg);
  case DivP0:
    return MRI.getType(Reg) == LLT::pointer(0, 64) && MUI.isDivergent(Reg);
  case DivP1:
    return MRI.getType(Reg) == LLT::pointer(1, 64) && MUI.isDivergent(Reg);
  case DivP2:
    return MRI.getType(Reg) == LLT::pointer(2, 32) && MUI.isDivergent(Reg);
  case DivP3:
    return MRI.getType(Reg) == LLT::pointer(3, 32) && MUI.isDivergent(Reg);
  case DivP4:
    return MRI.getType(Reg) == LLT::pointer(4, 64) && MUI.isDivergent(Reg);
  case DivP5:
    return MRI.getType(Reg) == LLT::pointer(5, 32) && MUI.isDivergent(Reg);
  case DivPtr32:
    return isAnyPtr(MRI.getType(Reg), 32) && MUI.isDivergent(Reg);
  case DivPtr64:
    return isAnyPtr(MRI.getType(Reg), 64) && MUI.isDivergent(Reg);
  case DivPtr128:
    return isAnyPtr(MRI.getType(Reg), 128) && MUI.isDivergent(Reg);
  case DivV2S16:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 16) && MUI.isDivergent(Reg);
  case DivV2S32:
    return MRI.getType(Reg) == LLT::fixed_vector(2, 32) && MUI.isDivergent(Reg);
  case DivV3S32:
    return MRI.getType(Reg) == LLT::fixed_vector(3, 32) && MUI.isDivergent(Reg);
  case DivV4S16:
    return MRI.getType(Reg) == LLT::fixed_vector(4, 16) && MUI.isDivergent(Reg);
  case DivV6S32:
    return MRI.getType(Reg) == LLT::fixed_vector(6, 32) && MUI.isDivergent(Reg);
  case DivB32:
    return MRI.getType(Reg).getSizeInBits() == 32 && MUI.isDivergent(Reg);
  case DivB64:
    return MRI.getType(Reg).getSizeInBits() == 64 && MUI.isDivergent(Reg);
  case DivB96:
    return MRI.getType(Reg).getSizeInBits() == 96 && MUI.isDivergent(Reg);
  case DivB128:
    return MRI.getType(Reg).getSizeInBits() == 128 && MUI.isDivergent(Reg);
  case DivB160:
    return MRI.getType(Reg).getSizeInBits() == 160 && MUI.isDivergent(Reg);
  case DivB256:
    return MRI.getType(Reg).getSizeInBits() == 256 && MUI.isDivergent(Reg);
  case DivB512:
    return MRI.getType(Reg).getSizeInBits() == 512 && MUI.isDivergent(Reg);
  case DivBRC: {
    if (!MUI.isDivergent(Reg))
      return false;
    // Check if there is VGPR register class of same size as the LLT.
    const SIRegisterInfo *TRI =
        static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
    return TRI->getSGPRClassForBitWidth(MRI.getType(Reg).getSizeInBits());
  }
  case BRC: {
    // Check if there is SGPR and VGPR register class of same size as the LLT.
    const SIRegisterInfo *TRI =
        static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
    unsigned LLTSize = MRI.getType(Reg).getSizeInBits();
    return LLTSize >= 32 && TRI->getSGPRClassForBitWidth(LLTSize) &&
           TRI->getVGPRClassForBitWidth(LLTSize);
  }
  case _:
    return true;
  default:
    llvm_unreachable("missing matchUniformityAndLLT");
  }
}

bool PredicateMapping::match(const MachineInstr &MI,
                             const MachineUniformityInfo &MUI,
                             const MachineRegisterInfo &MRI) const {
  // Check LLT signature.
  for (unsigned i = 0; i < OpUniformityAndTypes.size(); ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (OpUniformityAndTypes[i] == _) {
      assert((!MI.getOperand(i).isReg() ||
              !MI.getOperand(i).getReg().isVirtual()) &&
             "_ is for non-register and physical register operands only");
      continue;
    }

    // Remaining IDs check registers.
    if (!MO.isReg())
      return false;

    if (!matchUniformityAndLLT(MO.getReg(), OpUniformityAndTypes[i], MUI, MRI))
      return false;
  }

  // More complex check.
  if (TestFunc)
    return TestFunc(MI);

  return true;
}

SetOfRulesForOpcode::SetOfRulesForOpcode() = default;

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

UniformityLLTOpPredicateID LLTToBId(LLT Ty) {
  if (Ty == LLT::scalar(32) || Ty == LLT::fixed_vector(2, 16) ||
      isAnyPtr(Ty, 32))
    return B32;
  if (Ty == LLT::scalar(64) || Ty == LLT::fixed_vector(2, 32) ||
      Ty == LLT::fixed_vector(4, 16) || isAnyPtr(Ty, 64))
    return B64;
  if (Ty == LLT::fixed_vector(3, 32))
    return B96;
  if (Ty == LLT::fixed_vector(4, 32) || Ty == LLT::fixed_vector(2, 64) ||
      Ty == LLT::fixed_vector(8, 16) || isAnyPtr(Ty, 128))
    return B128;
  return _;
}

const RegBankLLTMapping *
SetOfRulesForOpcode::findMappingForMI(const MachineInstr &MI,
                                      const MachineRegisterInfo &MRI,
                                      const MachineUniformityInfo &MUI) const {
  // Search in "Fast Rules".
  // Note: if fast rules are enabled, RegBankLLTMapping must be added in each
  // slot that could "match fast Predicate". If not, InvalidMapping is
  // returned which results in failure, does not search "Slow Rules".
  if (FastTypes != NoFastRules) {
    Register Reg = MI.getOperand(0).getReg();
    int Slot;
    if (FastTypes == StandardB)
      Slot = getFastPredicateSlot(LLTToBId(MRI.getType(Reg)));
    else
      Slot = getFastPredicateSlot(LLTToId(MRI.getType(Reg)));

    if (Slot != -1)
      return MUI.isUniform(Reg) ? &Uni[Slot] : &Div[Slot];
  }

  // Slow search for more complex rules.
  for (const RegBankLegalizeRule &Rule : Rules) {
    if (Rule.Predicate.match(MI, MUI, MRI))
      return &Rule.OperandMapping;
  }

  return nullptr;
}

void SetOfRulesForOpcode::addRule(RegBankLegalizeRule Rule) {
  Rules.push_back(Rule);
}

void SetOfRulesForOpcode::addFastRuleDivergent(UniformityLLTOpPredicateID Ty,
                                               RegBankLLTMapping RuleApplyIDs) {
  int Slot = getFastPredicateSlot(Ty);
  assert(Slot != -1 && "Ty unsupported in this FastRulesTypes");
  Div[Slot] = std::move(RuleApplyIDs);
}

void SetOfRulesForOpcode::addFastRuleUniform(UniformityLLTOpPredicateID Ty,
                                             RegBankLLTMapping RuleApplyIDs) {
  int Slot = getFastPredicateSlot(Ty);
  assert(Slot != -1 && "Ty unsupported in this FastRulesTypes");
  Uni[Slot] = std::move(RuleApplyIDs);
}

int SetOfRulesForOpcode::getFastPredicateSlot(
    UniformityLLTOpPredicateID Ty) const {
  switch (FastTypes) {
  case Standard: {
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
  }
  case StandardB: {
    switch (Ty) {
    case B32:
      return 0;
    case B64:
      return 1;
    case B96:
      return 2;
    case B128:
      return 3;
    default:
      return -1;
    }
  }
  case Vector: {
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

const SetOfRulesForOpcode *
RegBankLegalizeRules::getRulesForOpc(MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::G_INTRINSIC || Opc == AMDGPU::G_INTRINSIC_CONVERGENT ||
      Opc == AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS ||
      Opc == AMDGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS) {
    unsigned IntrID = cast<GIntrinsic>(MI).getIntrinsicID();
    auto IRAIt = IRulesAlias.find(IntrID);
    if (IRAIt == IRulesAlias.end())
      return nullptr;
    return &IRules.at(IRAIt->second);
  }

  auto GRAIt = GRulesAlias.find(Opc);
  if (GRAIt == GRulesAlias.end())
    return nullptr;
  return &GRules.at(GRAIt->second);
}

// Syntactic sugar wrapper for predicate lambda that enables '&&', '||' and '!'.
class Predicate {
private:
  struct Elt {
    // Save formula composed of Pred, '&&', '||' and '!' as a jump table.
    // Sink ! to Pred. For example !((A && !B) || C) -> (!A || B) && !C
    // Sequences of && and || will be represented by jumps, for example:
    // (A && B && ... X) or (A && B && ... X) || Y
    //   A == true jump to B
    //   A == false jump to end or Y, result is A(false) or Y
    // (A || B || ... X) or (A || B || ... X) && Y
    //   A == true jump to end or Y, result is A(true) or Y
    //   A == false jump to B
    // Notice that when negating expression, we simply flip Neg on each Pred
    // and swap TJumpOffset and FJumpOffset (&& becomes ||, || becomes &&).
    std::function<bool(const MachineInstr &)> Pred;
    bool Neg; // Neg of Pred is calculated before jump
    unsigned TJumpOffset;
    unsigned FJumpOffset;
  };

  SmallVector<Elt, 8> Expression;

  Predicate(SmallVectorImpl<Elt> &&Expr) { Expression.swap(Expr); };

public:
  Predicate(std::function<bool(const MachineInstr &)> Pred) {
    Expression.push_back({Pred, false, 1, 1});
  };

  bool operator()(const MachineInstr &MI) const {
    unsigned Idx = 0;
    unsigned ResultIdx = Expression.size();
    bool Result;
    do {
      Result = Expression[Idx].Pred(MI);
      Result = Expression[Idx].Neg ? !Result : Result;
      if (Result) {
        Idx += Expression[Idx].TJumpOffset;
      } else {
        Idx += Expression[Idx].FJumpOffset;
      }
    } while ((Idx != ResultIdx));

    return Result;
  };

  Predicate operator!() const {
    SmallVector<Elt, 8> NegExpression;
    for (const Elt &ExprElt : Expression) {
      NegExpression.push_back({ExprElt.Pred, !ExprElt.Neg, ExprElt.FJumpOffset,
                               ExprElt.TJumpOffset});
    }
    return Predicate(std::move(NegExpression));
  };

  Predicate operator&&(const Predicate &RHS) const {
    SmallVector<Elt, 8> AndExpression = Expression;

    unsigned RHSSize = RHS.Expression.size();
    unsigned ResultIdx = Expression.size();
    for (unsigned i = 0; i < ResultIdx; ++i) {
      // LHS results in false, whole expression results in false.
      if (i + AndExpression[i].FJumpOffset == ResultIdx)
        AndExpression[i].FJumpOffset += RHSSize;
    }

    AndExpression.append(RHS.Expression);

    return Predicate(std::move(AndExpression));
  }

  Predicate operator||(const Predicate &RHS) const {
    SmallVector<Elt, 8> OrExpression = Expression;

    unsigned RHSSize = RHS.Expression.size();
    unsigned ResultIdx = Expression.size();
    for (unsigned i = 0; i < ResultIdx; ++i) {
      // LHS results in true, whole expression results in true.
      if (i + OrExpression[i].TJumpOffset == ResultIdx)
        OrExpression[i].TJumpOffset += RHSSize;
    }

    OrExpression.append(RHS.Expression);

    return Predicate(std::move(OrExpression));
  }
};

// Initialize rules
RegBankLegalizeRules::RegBankLegalizeRules(const GCNSubtarget &_ST,
                                           MachineRegisterInfo &_MRI)
    : ST(&_ST), MRI(&_MRI) {

  addRulesForGOpcs({G_ADD, G_SUB}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32AExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackAExt})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}});

  addRulesForGOpcs({G_UADDO, G_USUBO}, Standard)
      .Uni(S32, {{Sgpr32, Sgpr32Trunc}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32, Vcc}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_UADDE, G_USUBE, G_SADDE, G_SSUBE}, Standard)
      .Uni(S32, {{Sgpr32, Sgpr32Trunc}, {Sgpr32, Sgpr32, Sgpr32AExtBoolInReg}})
      .Div(S32, {{Vgpr32, Vcc}, {Vgpr32, Vgpr32, Vcc}});

  addRulesForGOpcs({G_UADDSAT, G_SADDSAT, G_USUBSAT, G_SSUBSAT}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}});

  bool HasVecMulU64 = ST->hasVectorMulU64();
  addRulesForGOpcs({G_MUL}, Standard)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S64, {{SgprB64}, {SgprB64, SgprB64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32AExt}})
      .Div(S64, {{VgprB64}, {VgprB64, VgprB64}}, HasVecMulU64)
      .Div(S64, {{VgprB64}, {VgprB64, VgprB64}, SplitTo32Mul}, !HasVecMulU64);

  bool hasMulHi = ST->hasScalarMulHiInsts();
  addRulesForGOpcs({G_UMULH, G_SMULH}, Standard)
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasMulHi)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasMulHi);

  addRulesForGOpcs({G_AMDGPU_MAD_U64_U32}, Standard)
      .Div(S64, {{Vgpr64, Vcc}, {Vgpr32, Vgpr32, Vgpr64}})
      .Uni(S64, {{Sgpr64, Sgpr32Trunc}, {Sgpr32, Sgpr32, Sgpr64}, UniMAD64});

  bool HasScalarSMulU64 = ST->hasScalarSMulU64();
  addRulesForGOpcs({G_AMDGPU_S_MUL_U64_U32, G_AMDGPU_S_MUL_I64_I32}, Standard)
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr64}, UniMul64}, HasScalarSMulU64)
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}, DivSMulToMAD});

  addRulesForGOpcs({G_XOR, G_OR, G_AND}, StandardB)
      .Any({{UniS1}, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32AExt}}})
      .Any({{DivS1}, {{Vcc}, {Vcc, Vcc}}})
      .Any({{UniS16}, {{Sgpr16}, {Sgpr16, Sgpr16}}})
      .Any({{DivS16}, {{Vgpr16}, {Vgpr16, Vgpr16}}})
      .Uni(B32, {{SgprB32}, {SgprB32, SgprB32}})
      .Div(B32, {{VgprB32}, {VgprB32, VgprB32}})
      .Uni(B64, {{SgprB64}, {SgprB64, SgprB64}})
      .Div(B64, {{VgprB64}, {VgprB64, VgprB64}, SplitTo32});

  addRulesForGOpcs({G_SHL}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32ZExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackBitShift})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32}});

  addRulesForGOpcs({G_LSHR}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32ZExt, Sgpr32ZExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackBitShift})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32}});

  addRulesForGOpcs({G_ASHR}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32SExt, Sgpr32ZExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackBitShift})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32}});

  addRulesForGOpcs({G_FSHR}, Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_BSWAP}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16}});

  addRulesForGOpcs({G_AMDGPU_CVT_F32_UBYTE0, G_AMDGPU_CVT_F32_UBYTE1,
                    G_AMDGPU_CVT_F32_UBYTE2, G_AMDGPU_CVT_F32_UBYTE3,
                    G_AMDGPU_RCP_IFLAG},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}});

  addRulesForGOpcs({G_FRAME_INDEX}).Any({{UniP5, _}, {{SgprP5}, {None}}});

  addRulesForGOpcs({G_UBFX, G_SBFX}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32, Sgpr32}, S_BFE})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Sgpr32, Sgpr32}, S_BFE})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32, Vgpr32}, V_BFE});

  addRulesForGOpcs({G_SMIN, G_SMAX}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32SExt, Sgpr32SExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackMinMax})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}});

  addRulesForGOpcs({G_UMIN, G_UMAX}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32ZExt, Sgpr32ZExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, UnpackMinMax})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}});

  addRulesForGOpcs({G_IMPLICIT_DEF})
      .Any({{UniS1}, {{Sgpr32Trunc}, {}}})
      .Any({{UniS16}, {{Sgpr16}, {}}})
      .Any({{UniBRC}, {{SgprBRC}, {}}});

  addRulesForGOpcs({G_CONSTANT}, Standard)
      .Any({{UniS1, _}, {{Sgpr32Trunc}, {}, UniCstExt}})
      .Uni(S16, {{Sgpr16}, {}})
      .Uni(S32, {{Sgpr32}, {}})
      .Uni(S64, {{Sgpr64}, {}})
      .Any({{UniPtr32, _}, {{SgprPtr32}, {}}})
      .Any({{UniPtr64, _}, {{SgprPtr64}, {}}});

  addRulesForGOpcs({G_FCONSTANT}, Standard)
      .Uni(S16, {{Sgpr16}, {}})
      .Uni(S32, {{Sgpr32}, {}})
      .Uni(S64, {{Sgpr64}, {}});

  addRulesForGOpcs({G_FREEZE})
      .Any({{UniS1}, {{Sgpr32Trunc}, {Sgpr32AExt}}})
      .Any({{DivS1}, {{Vcc}, {Vcc}}})
      .Any({{UniS16}, {{Sgpr16}, {Sgpr16}}})
      .Any({{UniBRC}, {{SgprBRC}, {SgprBRC}}})
      .Any({{DivBRC}, {{VgprBRC}, {VgprBRC}}});

  addRulesForGOpcs({G_BITCAST})
      .Any({{UniBRC}, {{SgprBRC}, {SgprBRC}}})
      .Any({{DivBRC}, {{VgprBRC}, {VgprBRC}}});

  addRulesForGOpcs({G_UNMERGE_VALUES})
      .Any({{UniS16}, {{}, {}, UnmergeToShiftTrunc}})
      .Any({{UniBRC}, {{}, {}, VerifyAllSgpr}})
      .Any({{DivBRC}, {{}, {}, ApplyAllVgpr}});

  addRulesForGOpcs({G_BUILD_VECTOR})
      .Any({{UniBRC, S16}, {{}, {}, VerifyAllSgpr}})
      .Any({{UniBRC, BRC}, {{}, {}, VerifyAllSgpr}})
      .Any({{DivBRC, S16}, {{}, {}, ApplyAllVgpr}})
      .Any({{DivBRC, BRC}, {{}, {}, ApplyAllVgpr}});

  addRulesForGOpcs({G_MERGE_VALUES, G_CONCAT_VECTORS})
      .Any({{UniBRC, BRC}, {{}, {}, VerifyAllSgpr}})
      .Any({{DivBRC, BRC}, {{}, {}, ApplyAllVgpr}});

  addRulesForGOpcs({G_PHI})
      .Any({{UniS1}, {{}, {}, AextToS32InIncomingBlockGPHI}})
      .Any({{UniS16}, {{}, {}, VerifyAllSgprGPHI}})
      .Any({{UniBRC}, {{}, {}, VerifyAllSgprGPHI}})
      .Any({{DivBRC}, {{}, {}, VerifyAllSgprOrVgprGPHI}});

  addRulesForGOpcs({G_EXTRACT_VECTOR_ELT})
      .Any({{UniB32, UniBRC, UniS32}, {{SgprB32}, {SgprBRC, Sgpr32}}})
      .Any({{DivB32, DivBRC, UniS32}, {{VgprB32}, {VgprBRC, Sgpr32}}})
      .Any({{DivB32, BRC, DivS32},
            {{VgprB32}, {VgprBRC, Vgpr32}, ExtrVecEltToSel}})
      .Any({{UniB64, UniBRC, UniS32}, {{SgprB64}, {SgprBRC, Sgpr32}}})
      .Any({{DivB64, DivBRC, UniS32},
            {{VgprB64}, {VgprBRC, Sgpr32}, ExtrVecEltTo32}})
      .Any({{DivB64, BRC, DivS32},
            {{VgprB64}, {VgprBRC, Vgpr32}, ExtrVecEltToSel}});

  addRulesForGOpcs({G_INSERT_VECTOR_ELT})
      .Any({{UniBRC, UniBRC, UniB32, UniS32},
            {{SgprBRC}, {SgprBRC, SgprB32, Sgpr32}}})
      .Any(
          {{DivBRC, BRC, B32, UniS32}, {{VgprBRC}, {VgprBRC, VgprB32, Sgpr32}}})
      .Any({{DivBRC, BRC, B32, DivS32},
            {{VgprBRC}, {VgprBRC, VgprB32, Vgpr32}, InsVecEltToSel}})
      .Any({{UniBRC, UniBRC, UniB64, UniS32},
            {{SgprBRC}, {SgprBRC, SgprB64, Sgpr32}, InsVecEltToSel}})
      .Any({{DivBRC, BRC, B64, UniS32},
            {{VgprBRC}, {VgprBRC, VgprB64, Sgpr32}, InsVecEltTo32}})
      .Any({{DivBRC, BRC, B64, DivS32},
            {{VgprBRC}, {VgprBRC, VgprB64, Vgpr32}, InsVecEltToSel}});

  // INTERSECT_RAY {Div}, {{VgprDst...}, {VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  // INTERSECT_RAY {Uni}, {{UniInVgprDst...}, {VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  addRulesForGOpcs({G_AMDGPU_BVH_INTERSECT_RAY, G_AMDGPU_BVH_DUAL_INTERSECT_RAY,
                    G_AMDGPU_BVH8_INTERSECT_RAY})
      .Any({{}, {{}, {}, ApplyBVH_INTERSECT_RAY}});

  // LOAD       {Div}, {{VgprDst...}, {VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  // LOAD       {Uni}, {{UniInVgprDst...}, {VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  // LOAD_NORET {}, {{}, {Imm, VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  // STORE      {}, {{}, {VgprSrc, ..., Sgpr_WF_RsrcIdx}}
  addRulesForGOpcs({G_AMDGPU_INTRIN_IMAGE_LOAD, G_AMDGPU_INTRIN_IMAGE_LOAD_D16,
                    G_AMDGPU_INTRIN_IMAGE_LOAD_NORET,
                    G_AMDGPU_INTRIN_IMAGE_STORE,
                    G_AMDGPU_INTRIN_IMAGE_STORE_D16})
      .Any({{}, {{}, {}, ApplyINTRIN_IMAGE}});

  Predicate isSignedICmp([](const MachineInstr &MI) -> bool {
    auto Pred =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
    return CmpInst::isSigned(Pred);
  });

  Predicate isEqualityICmp([](const MachineInstr &MI) -> bool {
    auto Pred =
        static_cast<CmpInst::Predicate>(MI.getOperand(1).getPredicate());
    return ICmpInst::isEquality(Pred);
  });

  bool HasScalarCompareEq64 = ST->hasScalarCompareEq64();
  // clang-format off
  addRulesForGOpcs({G_ICMP})
      .Any({{{UniS1, _, S16}, isEqualityICmp}, {{Sgpr32Trunc}, {None, Sgpr32ZExt, Sgpr32ZExt}}})
      .Any({{{UniS1, _, S16}, !isEqualityICmp && isSignedICmp}, {{Sgpr32Trunc}, {None, Sgpr32SExt, Sgpr32SExt}}})
      .Any({{{UniS1, _, S16}, !isEqualityICmp && !isSignedICmp}, {{Sgpr32Trunc}, {None, Sgpr32ZExt, Sgpr32ZExt}}})
      .Any({{{DivS1, _, S16}}, {{Vcc}, {None, Vgpr16, Vgpr16}}})
      .Any({{{UniS1, _, S32}}, {{Sgpr32Trunc}, {None, Sgpr32, Sgpr32}}})
      .Any({{{DivS1, _, S32}}, {{Vcc}, {None, Vgpr32, Vgpr32}}})
      .Any({{{UniS1, _, S64}, isEqualityICmp}, {{Sgpr32Trunc}, {None, Sgpr64, Sgpr64}}}, HasScalarCompareEq64)
      .Any({{{UniS1, _, S64}, isEqualityICmp}, {{UniInVcc}, {None, Vgpr64, Vgpr64}}}, !HasScalarCompareEq64)
      .Any({{{UniS1, _, S64}, !isEqualityICmp}, {{UniInVcc}, {None, Vgpr64, Vgpr64}}})
      .Any({{{DivS1, _, S64}}, {{Vcc}, {None, Vgpr64, Vgpr64}}})
      .Any({{{UniS1, _, Ptr32}}, {{Sgpr32Trunc}, {None, SgprPtr32, SgprPtr32}}})
      .Any({{{DivS1, _, Ptr32}}, {{Vcc}, {None, VgprPtr32, VgprPtr32}}})
      .Any({{{UniS1, _, Ptr64}, isEqualityICmp}, {{Sgpr32Trunc}, {None, SgprPtr64, SgprPtr64}}}, HasScalarCompareEq64)
      .Any({{{UniS1, _, Ptr64}, isEqualityICmp}, {{UniInVcc}, {None, VgprPtr64, VgprPtr64}}}, !HasScalarCompareEq64)
      .Any({{{UniS1, _, Ptr64}, !isEqualityICmp}, {{UniInVcc}, {None, VgprPtr64, VgprPtr64}}})
      .Any({{{DivS1, _, Ptr64}}, {{Vcc}, {None, VgprPtr64, VgprPtr64}}});
  // clang-format on

  addRulesForGOpcs({G_BRCOND})
      .Any({{UniS1}, {{}, {Sgpr32AExtBoolInReg}}})
      .Any({{DivS1}, {{}, {Vcc}}});

  addRulesForGOpcs({G_BR}).Any({{_}, {{}, {None}}});

  addRulesForGOpcs({G_SELECT}, StandardB)
      .Any({{DivS16}, {{Vgpr16}, {Vcc, Vgpr16, Vgpr16}}})
      .Any({{UniS16}, {{Sgpr16}, {Sgpr32AExtBoolInReg, Sgpr16, Sgpr16}}})
      .Div(B32, {{VgprB32}, {Vcc, VgprB32, VgprB32}})
      .Uni(B32, {{SgprB32}, {Sgpr32AExtBoolInReg, SgprB32, SgprB32}})
      .Div(B64, {{VgprB64}, {Vcc, VgprB64, VgprB64}, SplitTo32Select})
      .Uni(B64, {{SgprB64}, {Sgpr32AExtBoolInReg, SgprB64, SgprB64}});

  addRulesForGOpcs({G_ANYEXT})
      .Any({{UniS16, S1}, {{None}, {None}}}) // should be combined away
      .Any({{UniS32, S1}, {{None}, {None}}}) // should be combined away
      .Any({{UniS64, S1}, {{None}, {None}}}) // should be combined away
      .Any({{DivS16, S1}, {{Vgpr16}, {Vcc}, VccExtToSel}})
      .Any({{DivS32, S1}, {{Vgpr32}, {Vcc}, VccExtToSel}})
      .Any({{DivS64, S1}, {{Vgpr64}, {Vcc}, VccExtToSel}})
      .Any({{UniS64, S32}, {{Sgpr64}, {Sgpr32}, Ext32To64}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}, Ext32To64}})
      .Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}})
      .Any({{DivS32, S16}, {{Vgpr32}, {Vgpr16}}});

  bool Has16bitCmp = ST->has16BitInsts();

  // In global-isel G_TRUNC in-reg is treated as no-op, inst selected into COPY.
  // It is up to user to deal with truncated bits.
  // S1, S16, S32 and S64 results are handled with specific rules. Remaining
  // (result, source) pairs with valid register classes are covered by the
  // generic UniBRC/DivBRC wildcard rules.
  addRulesForGOpcs({G_TRUNC})
      .Any({{UniS1, UniS16}, {{None}, {None}}}) // should be combined away
      .Any({{UniS1, UniS32}, {{None}, {None}}}) // should be combined away
      .Any({{UniS1, UniS64}, {{None}, {None}}}) // should be combined away
      .Any({{UniS16, S32}, {{Sgpr16}, {Sgpr32}}})
      .Any({{UniBRC, UniBRC}, {{SgprBRC}, {SgprBRC}}})
      .Any({{DivBRC, DivBRC}, {{VgprBRC}, {VgprBRC}}})
      .Any({{UniV2S16, V2S32}, {{SgprV2S16}, {SgprV2S32}}})
      .Any({{DivV2S16, V2S32}, {{VgprV2S16}, {VgprV2S32}}})
      // This is non-trivial. VgprToVccCopy is done using compare instruction.
      .Any({{DivS1, DivS16}, {{Vcc}, {Vgpr16}, VgprToVccCopy}}, Has16bitCmp)
      .Any({{DivS1, DivS16}, {{Vcc}, {Vgpr32AExt}, VgprToVccCopy}},
           !Has16bitCmp)
      .Any({{DivS1, DivS32}, {{Vcc}, {Vgpr32}, VgprToVccCopy}})
      .Any({{DivS1, DivS64}, {{Vcc}, {Vgpr64}, VgprToVccCopy}});

  addRulesForGOpcs({G_ZEXT})
      .Any({{UniS16, S1}, {{Sgpr32Trunc}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{UniS32, S1}, {{Sgpr32}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{UniS64, S1}, {{Sgpr64}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{DivS16, S1}, {{Vgpr16}, {Vcc}, VccExtToSel}})
      .Any({{DivS32, S1}, {{Vgpr32}, {Vcc}, VccExtToSel}})
      .Any({{DivS64, S1}, {{Vgpr64}, {Vcc}, VccExtToSel}})
      .Any({{UniS64, S32}, {{Sgpr64}, {Sgpr32}, Ext32To64}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}, Ext32To64}})
      // not extending S16 to S32 is questionable.
      .Any({{UniS64, S16}, {{Sgpr64}, {Sgpr32ZExt}, Ext32To64}})
      .Any({{DivS64, S16}, {{Vgpr64}, {Vgpr32ZExt}, Ext32To64}})
      .Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}})
      .Any({{DivS32, S16}, {{Vgpr32}, {Vgpr16}}});

  addRulesForGOpcs({G_SEXT})
      .Any({{UniS16, S1}, {{Sgpr32Trunc}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{UniS32, S1}, {{Sgpr32}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{UniS64, S1}, {{Sgpr64}, {Sgpr32AExtBoolInReg}, UniExtToSel}})
      .Any({{DivS16, S1}, {{Vgpr16}, {Vcc}, VccExtToSel}})
      .Any({{DivS32, S1}, {{Vgpr32}, {Vcc}, VccExtToSel}})
      .Any({{DivS64, S1}, {{Vgpr64}, {Vcc}, VccExtToSel}})
      .Any({{UniS64, S32}, {{Sgpr64}, {Sgpr32}, Ext32To64}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}, Ext32To64}})
      // not extending S16 to S32 is questionable.
      .Any({{UniS64, S16}, {{Sgpr64}, {Sgpr32SExt}, Ext32To64}})
      .Any({{DivS64, S16}, {{Vgpr64}, {Vgpr32SExt}, Ext32To64}})
      .Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}})
      .Any({{DivS32, S16}, {{Vgpr32}, {Vgpr16}}});

  addRulesForGOpcs({G_SEXT_INREG})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}})
      .Any({{DivS32, S32}, {{Vgpr32}, {Vgpr32}}})
      .Any({{UniS64, S64}, {{Sgpr64}, {Sgpr64}}})
      .Any({{DivS64, S64}, {{Vgpr64}, {Vgpr64}, SplitTo32SExtInReg}});

  addRulesForGOpcs({G_ASSERT_ZEXT, G_ASSERT_SEXT}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Imm}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Imm}})
      .Uni(S64, {{Sgpr64}, {Sgpr64, Imm}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Imm}});

  addRulesForGOpcs({G_ASSERT_ALIGN}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64}})
      .Any({{UniPtr32}, {{SgprPtr32}, {SgprPtr32}}})
      .Any({{DivPtr32}, {{VgprPtr32}, {VgprPtr32}}})
      .Any({{UniPtr64}, {{SgprPtr64}, {SgprPtr64}}})
      .Any({{DivPtr64}, {{VgprPtr64}, {VgprPtr64}}});

  // Atomic read-modify-write operations: result and value are always VGPR,
  // pointer varies by address space.
  addRulesForGOpcs({G_ATOMICRMW_ADD, G_ATOMICRMW_SUB, G_ATOMICRMW_XCHG,
                    G_ATOMICRMW_AND, G_ATOMICRMW_OR, G_ATOMICRMW_XOR,
                    G_ATOMICRMW_MIN, G_ATOMICRMW_MAX, G_ATOMICRMW_UMIN,
                    G_ATOMICRMW_UMAX, G_ATOMICRMW_UINC_WRAP,
                    G_ATOMICRMW_UDEC_WRAP, G_ATOMICRMW_FMIN, G_ATOMICRMW_FMAX})
      .Any({{DivS32, P0, S32}, {{Vgpr32}, {VgprP0, Vgpr32}}})
      .Any({{DivS64, P0, S64}, {{Vgpr64}, {VgprP0, Vgpr64}}})
      .Any({{DivS32, P1, S32}, {{Vgpr32}, {VgprP1, Vgpr32}}})
      .Any({{DivS64, P1, S64}, {{Vgpr64}, {VgprP1, Vgpr64}}})
      .Any({{DivS32, P3, S32}, {{Vgpr32}, {VgprP3, Vgpr32}}})
      .Any({{DivS64, P3, S64}, {{Vgpr64}, {VgprP3, Vgpr64}}});

  bool HasAtomicFlatPkAdd16Insts = ST->hasAtomicFlatPkAdd16Insts();
  bool HasAtomicBufferGlobalPkAddF16Insts =
      ST->hasAtomicBufferGlobalPkAddF16NoRtnInsts() ||
      ST->hasAtomicBufferGlobalPkAddF16Insts();
  bool HasAtomicDsPkAdd16Insts = ST->hasAtomicDsPkAdd16Insts();
  addRulesForGOpcs({G_ATOMICRMW_FADD})
      .Any({{DivS32, P0, S32}, {{Vgpr32}, {VgprP0, Vgpr32}}})
      .Any({{DivS64, P0, S64}, {{Vgpr64}, {VgprP0, Vgpr64}}})
      .Any({{DivS32, P1, S32}, {{Vgpr32}, {VgprP1, Vgpr32}}})
      .Any({{DivS64, P1, S64}, {{Vgpr64}, {VgprP1, Vgpr64}}})
      .Any({{DivS32, P3, S32}, {{Vgpr32}, {VgprP3, Vgpr32}}})
      .Any({{DivS64, P3, S64}, {{Vgpr64}, {VgprP3, Vgpr64}}})
      .Any({{DivV2S16, P0, V2S16}, {{VgprV2S16}, {VgprP0, VgprV2S16}}},
           HasAtomicFlatPkAdd16Insts)
      .Any({{DivV2S16, P1, V2S16}, {{VgprV2S16}, {VgprP1, VgprV2S16}}},
           HasAtomicBufferGlobalPkAddF16Insts)
      .Any({{DivV2S16, P3, V2S16}, {{VgprV2S16}, {VgprP3, VgprV2S16}}},
           HasAtomicDsPkAdd16Insts);

  addRulesForGOpcs({G_ATOMIC_CMPXCHG})
      .Any({{DivS32, P2}, {{Vgpr32}, {VgprP2, Vgpr32, Vgpr32}}})
      .Any({{DivS64, P2}, {{Vgpr64}, {VgprP2, Vgpr64, Vgpr64}}})
      .Any({{DivS32, P3}, {{Vgpr32}, {VgprP3, Vgpr32, Vgpr32}}})
      .Any({{DivS64, P3}, {{Vgpr64}, {VgprP3, Vgpr64, Vgpr64}}});

  addRulesForGOpcs({G_AMDGPU_ATOMIC_CMPXCHG})
      .Any({{DivS32, P0}, {{Vgpr32}, {VgprP0, VgprV2S32}}})
      .Any({{DivS32, P1}, {{Vgpr32}, {VgprP1, VgprV2S32}}})
      .Any({{DivS64, P0}, {{Vgpr64}, {VgprP0, VgprV2S64}}})
      .Any({{DivS64, P1}, {{Vgpr64}, {VgprP1, VgprV2S64}}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_ATOMIC_CMPSWAP}, Standard)
      .Div(S32, {{Vgpr32},
                 {Vgpr32, Vgpr32, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(S64, {{Vgpr64},
                 {Vgpr64, Vgpr64, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_ATOMIC_ADD, G_AMDGPU_BUFFER_ATOMIC_AND,
                    G_AMDGPU_BUFFER_ATOMIC_DEC, G_AMDGPU_BUFFER_ATOMIC_FMAX,
                    G_AMDGPU_BUFFER_ATOMIC_FMIN, G_AMDGPU_BUFFER_ATOMIC_INC,
                    G_AMDGPU_BUFFER_ATOMIC_OR, G_AMDGPU_BUFFER_ATOMIC_SMAX,
                    G_AMDGPU_BUFFER_ATOMIC_SMIN, G_AMDGPU_BUFFER_ATOMIC_SUB,
                    G_AMDGPU_BUFFER_ATOMIC_SWAP, G_AMDGPU_BUFFER_ATOMIC_UMAX,
                    G_AMDGPU_BUFFER_ATOMIC_UMIN, G_AMDGPU_BUFFER_ATOMIC_XOR},
                   Standard)
      .Div(S32, {{Vgpr32}, {Vgpr32, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(S64, {{Vgpr64}, {Vgpr64, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  bool hasSMRDx3 = ST->hasScalarDwordx3Loads();
  bool hasSMRDSmall = ST->hasScalarSubwordLoads();
  bool usesTrue16 = ST->useRealTrue16Insts();

  Predicate isAlign16([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->getAlign() >= Align(16);
  });

  Predicate isAlign4([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->getAlign() >= Align(4);
  });

  Predicate isAtomicMMO([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->isAtomic();
  });

  Predicate isUniMMO([](const MachineInstr &MI) -> bool {
    return AMDGPU::isUniformMMO(*MI.memoperands_begin());
  });

  Predicate isConst([](const MachineInstr &MI) -> bool {
    // Address space in MMO be different then address space on pointer.
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    const unsigned AS = MMO->getAddrSpace();
    return AS == AMDGPUAS::CONSTANT_ADDRESS ||
           AS == AMDGPUAS::CONSTANT_ADDRESS_32BIT;
  });

  Predicate isVolatileMMO([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->isVolatile();
  });

  Predicate isInvMMO([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->isInvariant();
  });

  Predicate isNoClobberMMO([](const MachineInstr &MI) -> bool {
    return (*MI.memoperands_begin())->getFlags() & MONoClobber;
  });

  Predicate isNaturalAligned([](const MachineInstr &MI) -> bool {
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    return MMO->getAlign() >= Align(MMO->getSize().getValue());
  });

  Predicate is8Or16BitMMO([](const MachineInstr &MI) -> bool {
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    const unsigned MemSize = 8 * MMO->getSize().getValue();
    return MemSize == 16 || MemSize == 8;
  });

  Predicate is32BitMMO([](const MachineInstr &MI) -> bool {
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    return 8 * MMO->getSize().getValue() == 32;
  });

  auto isUL = !isAtomicMMO && isUniMMO && (isConst || !isVolatileMMO) &&
              (isConst || isInvMMO || isNoClobberMMO);

  // clang-format off
  // TODO: S32Dst, 16-bit any-extending load should not appear on True16 targets
  addRulesForGOpcs({G_LOAD})
      // flat, addrspace(0), never uniform - flat_load
      .Any({{DivS16, P0}, {{Vgpr16}, {VgprP0}}}, usesTrue16)
      .Any({{DivB32, P0}, {{VgprB32}, {VgprP0}}}) // 32-bit load, 8-bit and 16-bit any-extending load
      .Any({{DivB64, P0}, {{VgprB64}, {VgprP0}}})
      .Any({{DivB96, P0}, {{VgprB96}, {VgprP0}}})
      .Any({{DivB128, P0}, {{VgprB128}, {VgprP0}}})

       // global, addrspace(1)
       // divergent - global_load
      .Any({{DivS16, P1}, {{Vgpr16}, {VgprP1}}}, usesTrue16)
      .Any({{DivB32, P1}, {{VgprB32}, {VgprP1}}}) //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{DivB64, P1}, {{VgprB64}, {VgprP1}}})
      .Any({{DivB96, P1}, {{VgprB96}, {VgprP1}}})
      .Any({{DivB128, P1}, {{VgprB128}, {VgprP1}}})
      .Any({{DivB256, P1}, {{VgprB256}, {VgprP1}, SplitLoad}})
      .Any({{DivB512, P1}, {{VgprB512}, {VgprP1}, SplitLoad}})

       // uniform - s_load
      .Any({{{UniS16, P1}, isNaturalAligned && isUL}, {{Sgpr32Trunc}, {SgprP1}}}, usesTrue16 && hasSMRDSmall) // s16 load
      .Any({{{UniS16, P1}, isAlign4 && isUL}, {{Sgpr32Trunc}, {SgprP1}, WidenMMOToS32}}, usesTrue16 && !hasSMRDSmall) // s16 load to 32-bit load
      .Any({{{UniB32, P1}, isNaturalAligned && isUL}, {{SgprB32}, {SgprP1}}}, hasSMRDSmall) //32-bit load, 8-bit and 16-bit any-extending load
       // TODO: SplitLoad when !isNaturalAligned && isUL and target hasSMRDSmall
      .Any({{{UniB32, P1}, is8Or16BitMMO && isAlign4 && isUL}, {{SgprB32}, {SgprP1}, WidenMMOToS32}}, !hasSMRDSmall)  //8-bit and 16-bit any-extending load to 32-bit load
      .Any({{{UniB32, P1}, is32BitMMO && isAlign4 && isUL}, {{SgprB32}, {SgprP1}}}) //32-bit load
      .Any({{{UniB64, P1}, isAlign4 && isUL}, {{SgprB64}, {SgprP1}}})
      .Any({{{UniB96, P1}, isAlign16 && isUL}, {{SgprB96}, {SgprP1}, WidenLoad}}, !hasSMRDx3)
      .Any({{{UniB96, P1}, isAlign4 && !isAlign16 && isUL}, {{SgprB96}, {SgprP1}, SplitLoad}}, !hasSMRDx3)
      .Any({{{UniB96, P1}, isAlign4 && isUL}, {{SgprB96}, {SgprP1}}}, hasSMRDx3)
      .Any({{{UniB128, P1}, isAlign4 && isUL}, {{SgprB128}, {SgprP1}}})
      .Any({{{UniB256, P1}, isAlign4 && isUL}, {{SgprB256}, {SgprP1}}})
      .Any({{{UniB512, P1}, isAlign4 && isUL}, {{SgprB512}, {SgprP1}}})

      // Uniform via global or buffer load, for example volatile or non-aligned
      // uniform load. Not using standard {{UniInVgprTy}, {VgprP1}} since it is
      // selected as global_load, use SgprP1 for pointer instead to match
      // patterns without flat-for-global, default for GFX7 and older.
      // -> +flat-for-global + {{UniInVgprTy}, {SgprP1}} - global_load
      // -> -flat-for-global + {{UniInVgprTy}, {SgprP1}} - buffer_load
      .Any({{{UniS16, P1}, !isNaturalAligned || !isUL}, {{UniInVgprS16}, {SgprP1}}}, usesTrue16 && hasSMRDSmall) // s16 load
      .Any({{{UniS16, P1}, !isAlign4 || !isUL}, {{UniInVgprS16}, {SgprP1}}}, usesTrue16 && !hasSMRDSmall) // s16 load
      .Any({{{UniB32, P1}, !isNaturalAligned || !isUL}, {{UniInVgprB32}, {SgprP1}}}, hasSMRDSmall) //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{{UniB32, P1}, !isAlign4 || !isUL}, {{UniInVgprB32}, {SgprP1}}}, !hasSMRDSmall)  //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{{UniB64, P1}, !isAlign4 || !isUL}, {{UniInVgprB64}, {SgprP1}}})
      .Any({{{UniB96, P1}, !isAlign4 || !isUL}, {{UniInVgprB96}, {SgprP1}}})
      .Any({{{UniB128, P1}, !isAlign4 || !isUL}, {{UniInVgprB128}, {SgprP1}}})
      .Any({{{UniB256, P1}, !isAlign4 || !isUL}, {{UniInVgprB256}, {SgprP1}, SplitLoad}})
      .Any({{{UniB512, P1}, !isAlign4 || !isUL}, {{UniInVgprB512}, {SgprP1}, SplitLoad}})

      // local, addrspace(3) - ds_load
      .Any({{DivS16, P3}, {{Vgpr16}, {VgprP3}}}, usesTrue16)
      .Any({{DivB32, P3}, {{VgprB32}, {VgprP3}}}) // 32-bit load, 8-bit and 16-bit any-extending load
      .Any({{DivB64, P3}, {{VgprB64}, {VgprP3}}})
      .Any({{DivB96, P3}, {{VgprB96}, {VgprP3}}})
      .Any({{DivB128, P3}, {{VgprB128}, {VgprP3}}})

      .Any({{UniS16, P3}, {{UniInVgprS16}, {SgprP3}}}, usesTrue16) // 16-bit load
      .Any({{UniB32, P3}, {{UniInVgprB32}, {VgprP3}}}) // 32-bit load, 8-bit and 16-bit any-extending load
      .Any({{UniB64, P3}, {{UniInVgprB64}, {VgprP3}}})
      .Any({{UniB96, P3}, {{UniInVgprB96}, {VgprP3}}})
      .Any({{UniB128, P3}, {{UniInVgprB128}, {VgprP3}}})

      // constant, addrspace(4)
      // divergent - global_load
      .Any({{DivS16, P4}, {{Vgpr16}, {VgprP4}}}, usesTrue16)
      .Any({{DivB32, P4}, {{VgprB32}, {VgprP4}}}) //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{DivB64, P4}, {{VgprB64}, {VgprP4}}})
      .Any({{DivB96, P4}, {{VgprB96}, {VgprP4}}})
      .Any({{DivB128, P4}, {{VgprB128}, {VgprP4}}})
      .Any({{DivB256, P4}, {{VgprB256}, {VgprP4}, SplitLoad}})
      .Any({{DivB512, P4}, {{VgprB512}, {VgprP4}, SplitLoad}})

       // uniform - s_load
      .Any({{{UniS16, P4}, isNaturalAligned && isUL}, {{Sgpr32Trunc}, {SgprP4}}}, usesTrue16 && hasSMRDSmall) // s16 load
      .Any({{{UniS16, P4}, isAlign4 && isUL}, {{Sgpr32Trunc}, {SgprP4}, WidenMMOToS32}}, usesTrue16 && !hasSMRDSmall) // s16 load to 32-bit load
      .Any({{{UniB32, P4}, isNaturalAligned && isUL}, {{SgprB32}, {SgprP4}}}, hasSMRDSmall) //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{{UniB32, P4}, is8Or16BitMMO && isAlign4 && isUL}, {{SgprB32}, {SgprP4}, WidenMMOToS32}}, !hasSMRDSmall)  //8-bit and 16-bit any-extending load to 32-bit load
      .Any({{{UniB32, P4}, is32BitMMO && isAlign4 && isUL}, {{SgprB32}, {SgprP4}}}) //32-bit load
      .Any({{{UniB64, P4}, isAlign4 && isUL}, {{SgprB64}, {SgprP4}}})
      .Any({{{UniB96, P4}, isAlign16 && isUL}, {{SgprB96}, {SgprP4}, WidenLoad}}, !hasSMRDx3)
      .Any({{{UniB96, P4}, isAlign4 && !isAlign16 && isUL}, {{SgprB96}, {SgprP4}, SplitLoad}}, !hasSMRDx3)
      .Any({{{UniB96, P4}, isAlign4 && isUL}, {{SgprB96}, {SgprP4}}}, hasSMRDx3)
      .Any({{{UniB128, P4}, isAlign4 && isUL}, {{SgprB128}, {SgprP4}}})
      .Any({{{UniB256, P4}, isAlign4 && isUL}, {{SgprB256}, {SgprP4}}})
      .Any({{{UniB512, P4}, isAlign4 && isUL}, {{SgprB512}, {SgprP4}}})

      // uniform in vgpr - global_load or buffer_load
      .Any({{{UniS16, P4}, !isNaturalAligned || !isUL}, {{UniInVgprS16}, {SgprP4}}}, usesTrue16 && hasSMRDSmall) // s16 load
      .Any({{{UniS16, P4}, !isAlign4 || !isUL}, {{UniInVgprS16}, {SgprP4}}}, usesTrue16 && !hasSMRDSmall) // s16 load
      .Any({{{UniB32, P4}, !isNaturalAligned || !isUL}, {{UniInVgprB32}, {SgprP4}}}, hasSMRDSmall) //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{{UniB32, P4}, !isAlign4 || !isUL}, {{UniInVgprB32}, {SgprP4}}}, !hasSMRDSmall)  //32-bit load, 8-bit and 16-bit any-extending load
      .Any({{{UniB64, P4}, !isAlign4 || !isUL}, {{UniInVgprB64}, {SgprP4}}})
      .Any({{{UniB96, P4}, !isAlign4 || !isUL}, {{UniInVgprB96}, {SgprP4}}})
      .Any({{{UniB128, P4}, !isAlign4 || !isUL}, {{UniInVgprB128}, {SgprP4}}})
      .Any({{{UniB256, P4}, !isAlign4 || !isUL}, {{UniInVgprB256}, {SgprP4}, SplitLoad}})
      .Any({{{UniB512, P4}, !isAlign4 || !isUL}, {{UniInVgprB512}, {SgprP4}, SplitLoad}})

      // private, addrspace(5), never uniform - scratch_load
      .Any({{DivS16, P5}, {{Vgpr16}, {VgprP5}}}, usesTrue16)
      .Any({{DivB32, P5}, {{VgprB32}, {VgprP5}}}) // 32-bit load, 8-bit and 16-bit any-extending load
      .Any({{DivB64, P5}, {{VgprB64}, {VgprP5}}})
      .Any({{DivB96, P5}, {{VgprB96}, {VgprP5}}})
      .Any({{DivB128, P5}, {{VgprB128}, {VgprP5}}})
      
      .Any({{DivS32, Ptr128}, {{Vgpr32}, {VgprPtr128}}});


  addRulesForGOpcs({G_ZEXTLOAD, G_SEXTLOAD}) // i8 and i16 zeroextending loads
      .Any({{DivS32, P0}, {{Vgpr32}, {VgprP0}}})

      .Any({{DivS32, P1}, {{Vgpr32}, {VgprP1}}})
      .Any({{{UniS32, P1}, isAlign4 && isUL}, {{Sgpr32}, {SgprP1}, WidenMMOToS32}}, !hasSMRDSmall)
      .Any({{{UniS32, P1}, isNaturalAligned && isUL}, {{Sgpr32}, {SgprP1}}}, hasSMRDSmall)
      .Any({{{UniS32, P1}, !isAlign4 || !isUL}, {{UniInVgprS32}, {SgprP1}}}, !hasSMRDSmall)
      .Any({{{UniS32, P1}, !isNaturalAligned || !isUL}, {{UniInVgprS32}, {SgprP1}}}, hasSMRDSmall)

      .Any({{DivS32, P3}, {{Vgpr32}, {VgprP3}}})
      .Any({{UniS32, P3}, {{UniInVgprS32}, {VgprP3}}})

      .Any({{DivS32, P4}, {{Vgpr32}, {VgprP4}}})
      .Any({{{UniS32, P4}, isAlign4 && isUL}, {{Sgpr32}, {SgprP4}, WidenMMOToS32}}, !hasSMRDSmall)
      .Any({{{UniS32, P4}, isNaturalAligned && isUL}, {{Sgpr32}, {SgprP4}}}, hasSMRDSmall)
      .Any({{{UniS32, P4}, !isAlign4 || !isUL}, {{UniInVgprS32}, {SgprP4}}}, !hasSMRDSmall)
      .Any({{{UniS32, P4}, !isNaturalAligned || !isUL}, {{UniInVgprS32}, {SgprP4}}}, hasSMRDSmall)

      .Any({{DivS32, P5}, {{Vgpr32}, {VgprP5}}});

  addRulesForGOpcs({G_STORE})
      // addrspace(0)
      .Any({{S16, P0}, {{}, {Vgpr16, VgprP0}}}, usesTrue16) // 16-bit store
      .Any({{B32, P0}, {{}, {VgprB32, VgprP0}}}) // 32-bit store, 8-bit and 16-bit truncating store
      .Any({{B64, P0}, {{}, {VgprB64, VgprP0}}})
      .Any({{B96, P0}, {{}, {VgprB96, VgprP0}}})
      .Any({{B128, P0}, {{}, {VgprB128, VgprP0}}})

       // addrspace(1), there are no stores to addrspace(4)
       // For targets:
       // - with "+flat-for-global" - global_store
       // - without(-flat-for-global) - buffer_store addr64
      .Any({{S16, DivP1}, {{}, {Vgpr16, VgprP1}}}, usesTrue16) // 16-bit store
      .Any({{B32, DivP1}, {{}, {VgprB32, VgprP1}}}) // 32-bit store, 8-bit and 16-bit truncating store
      .Any({{B64, DivP1}, {{}, {VgprB64, VgprP1}}})
      .Any({{B96, DivP1}, {{}, {VgprB96, VgprP1}}})
      .Any({{B128, DivP1}, {{}, {VgprB128, VgprP1}}})

       // For UniP1, use sgpr ptr to match flat-for-global patterns. Targets:
       // - with "+flat-for-global" - global_store for both sgpr and vgpr ptr
       // - without(-flat-for-global) - need sgpr ptr to select buffer_store
      .Any({{S16, UniP1}, {{}, {Vgpr16, SgprP1}}}, usesTrue16) // 16-bit store
      .Any({{B32, UniP1}, {{}, {VgprB32, SgprP1}}}) // 32-bit store, 8-bit and 16-bit truncating store
      .Any({{B64, UniP1}, {{}, {VgprB64, SgprP1}}})
      .Any({{B96, UniP1}, {{}, {VgprB96, SgprP1}}})
      .Any({{B128, UniP1}, {{}, {VgprB128, SgprP1}}})

      // addrspace(3) and addrspace(5)
      .Any({{S16, Ptr32}, {{}, {Vgpr16, VgprPtr32}}}, usesTrue16) // 16-bit store
      .Any({{B32, Ptr32}, {{}, {VgprB32, VgprPtr32}}}) // 32-bit store, 8-bit and 16-bit truncating store
      .Any({{B64, Ptr32}, {{}, {VgprB64, VgprPtr32}}})
      .Any({{B96, Ptr32}, {{}, {VgprB96, VgprPtr32}}})
      .Any({{B128, Ptr32}, {{}, {VgprB128, VgprPtr32}}});

  // clang-format on

  addRulesForGOpcs({G_AMDGPU_BUFFER_LOAD, G_AMDGPU_BUFFER_LOAD_FORMAT,
                    G_AMDGPU_TBUFFER_LOAD_FORMAT},
                   StandardB)
      .Div(B32, {{VgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B32, {{UniInVgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B64, {{VgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B64, {{UniInVgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B96, {{VgprB96}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B96, {{UniInVgprB96}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B128, {{VgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B128, {{UniInVgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_LOAD_USHORT, G_AMDGPU_BUFFER_LOAD_UBYTE,
                    G_AMDGPU_BUFFER_LOAD_SSHORT, G_AMDGPU_BUFFER_LOAD_SBYTE},
                   StandardB)
      .Div(B32, {{VgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B32, {{UniInVgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  addRulesForGOpcs(
      {G_AMDGPU_BUFFER_LOAD_UBYTE_TFE, G_AMDGPU_BUFFER_LOAD_USHORT_TFE},
      StandardB)
      .Div(B64, {{VgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B64, {{UniInVgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_LOAD_TFE, G_AMDGPU_BUFFER_LOAD_FORMAT_TFE},
                   StandardB)
      .Div(B64, {{VgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B64, {{UniInVgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B96, {{VgprB96}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B96, {{UniInVgprB96}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B128, {{VgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B128, {{UniInVgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Any({{DivB160}, {{VgprB160}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{UniB160},
            {{UniInVgprB160}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}});

  addRulesForGOpcs(
      {G_AMDGPU_BUFFER_LOAD_FORMAT_D16, G_AMDGPU_TBUFFER_LOAD_FORMAT_D16},
      StandardB)
      .Div(B32, {{VgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B32, {{UniInVgprB32}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B64, {{VgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B64, {{UniInVgprB64}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Div(B128, {{VgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}})
      .Uni(B128, {{UniInVgprB128}, {SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}});

  addRulesForGOpcs({G_AMDGPU_BUFFER_STORE, G_AMDGPU_BUFFER_STORE_BYTE,
                    G_AMDGPU_BUFFER_STORE_SHORT, G_AMDGPU_BUFFER_STORE_FORMAT,
                    G_AMDGPU_BUFFER_STORE_FORMAT_D16,
                    G_AMDGPU_TBUFFER_STORE_FORMAT,
                    G_AMDGPU_TBUFFER_STORE_FORMAT_D16})
      .Any({{B32}, {{}, {VgprB32, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{B64}, {{}, {VgprB64, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{B96}, {{}, {VgprB96, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{B128}, {{}, {VgprB128, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}});

  // Buffer atomics: resource descriptor + scalar offset are SGPR, data and
  // address components are VGPR.
  //
  // Operand order (SIInstructions.td BufferAtomicGenericInstruction):
  //   dst = op vdata, rsrc, vindex, voffset, soffset, offset_imm, cachepolicy,
  //        idxen_imm
  addRulesForGOpcs({G_AMDGPU_BUFFER_ATOMIC_FADD})
      .Any({{S32, S32, V4S32, S32, S32, S32},
            {{Vgpr32}, {Vgpr32, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{S64, S64, V4S32, S32, S32, S32},
            {{Vgpr64}, {Vgpr64, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}})
      .Any({{V2S16, V2S16, V4S32, S32, S32, S32},
            {{VgprV2S16},
             {VgprV2S16, SgprV4S32_WF, Vgpr32, Vgpr32, Sgpr32_WF}}});

  addRulesForGOpcs({G_PTR_ADD})
      .Any({{UniPtr32}, {{SgprPtr32}, {SgprPtr32, Sgpr32}}})
      .Any({{DivPtr32}, {{VgprPtr32}, {VgprPtr32, Vgpr32}}})
      .Any({{UniPtr64}, {{SgprPtr64}, {SgprPtr64, Sgpr64}}})
      .Any({{DivPtr64}, {{VgprPtr64}, {VgprPtr64, Vgpr64}}});

  addRulesForGOpcs({G_INTTOPTR})
      .Any({{UniPtr32}, {{SgprPtr32}, {Sgpr32}}})
      .Any({{DivPtr32}, {{VgprPtr32}, {Vgpr32}}})
      .Any({{UniPtr64}, {{SgprPtr64}, {Sgpr64}}})
      .Any({{DivPtr64}, {{VgprPtr64}, {Vgpr64}}})
      .Any({{UniPtr128}, {{SgprPtr128}, {Sgpr128}}})
      .Any({{DivPtr128}, {{VgprPtr128}, {Vgpr128}}});

  addRulesForGOpcs({G_PTRTOINT})
      .Any({{UniS32}, {{Sgpr32}, {SgprPtr32}}})
      .Any({{DivS32}, {{Vgpr32}, {VgprPtr32}}})
      .Any({{UniS64}, {{Sgpr64}, {SgprPtr64}}})
      .Any({{DivS64}, {{Vgpr64}, {VgprPtr64}}})
      .Any({{UniS128}, {{Sgpr128}, {SgprPtr128}}})
      .Any({{DivS128}, {{Vgpr128}, {VgprPtr128}}});

  // FIXME: Update llvm/test/CodeGen/AMDGPU/ptrmask.ll to use GlobalISel.
  // Currently crashes on P8 (buffer resource) tests due to legalizer issue.
  addRulesForGOpcs({G_PTRMASK})
      .Any({{UniP1}, {{SgprP1}, {SgprP1, Sgpr64}}})
      .Any({{DivP1}, {{VgprP1}, {VgprP1, Vgpr64}}})
      .Any({{UniP3}, {{SgprP3}, {SgprP3, Sgpr32}}})
      .Any({{DivP3}, {{VgprP3}, {VgprP3, Vgpr32}}});

  addRulesForGOpcs({G_ABS}, Standard)
      .Uni(S16, {{Sgpr32Trunc}, {Sgpr32SExt}})
      .Div(S16, {{Vgpr16}, {Vgpr16}, AbsToNegMax})
      .Uni(S32, {{Sgpr32}, {Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}, AbsToNegMax})
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16}, AbsToS32})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16}, AbsToNegMax});

  addRulesForGOpcs({G_BITREVERSE}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(S64, {{Sgpr64}, {Sgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64}});

  addRulesForGOpcs({G_AMDGPU_FFBH_U32, G_AMDGPU_FFBL_B32, G_CTLZ_ZERO_UNDEF,
                    G_CTTZ_ZERO_UNDEF})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}})
      .Any({{DivS32, S32}, {{Vgpr32}, {Vgpr32}}})
      .Any({{UniS32, S64}, {{Sgpr32}, {Sgpr64}}})
      .Any({{DivS32, S64}, {{Vgpr32}, {Vgpr64}, SplitBitCount64To32}});

  addRulesForGOpcs({G_FENCE}).Any({{{}}, {{}, {}}});

  addRulesForGOpcs({G_READSTEADYCOUNTER, G_READCYCLECOUNTER}, Standard)
      .Uni(S64, {{Sgpr64}, {}});

  addRulesForGOpcs({G_BLOCK_ADDR}).Any({{UniP0}, {{SgprP0}, {}}});

  addRulesForGOpcs({G_GLOBAL_VALUE})
      .Any({{UniP0}, {{SgprP0}, {}}})
      .Any({{UniP1}, {{SgprP1}, {}}})
      .Any({{UniP3}, {{SgprP3}, {}}})
      .Any({{UniP4}, {{SgprP4}, {}}})
      .Any({{UniP8}, {{SgprP8}, {}}});

  addRulesForGOpcs({G_AMDGPU_WAVE_ADDRESS}).Any({{UniP5}, {{SgprP5}, {}}});

  addRulesForGOpcs({G_SI_CALL})
      .Any({{_, UniP0}, {{None}, {SgprP0}}})
      .Any({{_, DivP0}, {{None}, {SgprP0Call_WF}}})
      .Any({{_, UniP4}, {{None}, {SgprP4}}})
      .Any({{_, DivP4}, {{None}, {SgprP4Call_WF}}});

  bool hasSALUFloat = ST->hasSALUFloatInsts();

  addRulesForGOpcs({G_FADD, G_FMUL, G_STRICT_FADD, G_STRICT_FMUL}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}}, !hasSALUFloat)
      .Uni(S16, {{Sgpr16}, {Sgpr16, Sgpr16}}, hasSALUFloat)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasSALUFloat)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasSALUFloat)
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64, Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16}}, !hasSALUFloat)
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16, SgprV2S16}, ScalarizeToS16},
           hasSALUFloat)
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}});

  addRulesForGOpcs({G_FSUB, G_STRICT_FSUB}, Standard)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S16, {{Sgpr16}, {Sgpr16, Sgpr16}}, hasSALUFloat)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}}, !hasSALUFloat)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasSALUFloat)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasSALUFloat);

  addRulesForGOpcs({G_FMAD}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_FLDEXP, G_STRICT_FLDEXP}, Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64, Vgpr32}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr32}});

  addRulesForGOpcs({G_FMA, G_STRICT_FMA}, Standard)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16, Vgpr16}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64, Vgpr64, Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64, Vgpr64}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16, VgprV2S16}})
      .Any({{UniV2S32}, {{UniInVgprV2S32}, {VgprV2S32, VgprV2S32, VgprV2S32}}})
      .Any({{DivV2S32}, {{VgprV2S32}, {VgprV2S32, VgprV2S32, VgprV2S32}}})
      .Uni(S16, {{Sgpr16}, {Sgpr16, Sgpr16, Sgpr16}}, hasSALUFloat)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16, Vgpr16}}, !hasSALUFloat)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32, Sgpr32}}, hasSALUFloat)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32, Vgpr32}}, !hasSALUFloat)
      .Uni(V2S16,
           {{SgprV2S16}, {SgprV2S16, SgprV2S16, SgprV2S16}, ScalarizeToS16},
           hasSALUFloat)
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16, VgprV2S16}},
           !hasSALUFloat);

  addRulesForGOpcs({G_AMDGPU_FMED3}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}});

  // TODO: This opcode is generated from the i64->i16 signed clamped pattern in
  // the PreLegalizerCombiner. Move the combine to RegBankCombiner to keep more
  // instructions on SALU.
  addRulesForGOpcs({G_AMDGPU_SMED3}, Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32, Vgpr32}});

  // FNEG and FABS are either folded as source modifiers or can be selected as
  // bitwise XOR and AND with Mask. XOR and AND are available on SALU but for
  // targets without SALU float we still select them as VGPR since there would
  // be no real sgpr use.
  addRulesForGOpcs({G_FNEG, G_FABS}, Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16}}, !hasSALUFloat)
      .Uni(S16, {{Sgpr16}, {Sgpr16}}, hasSALUFloat)
      .Div(S16, {{Vgpr16}, {Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32}}, !hasSALUFloat)
      .Uni(S32, {{Sgpr32}, {Sgpr32}}, hasSALUFloat)
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16}}, !hasSALUFloat)
      .Uni(V2S16, {{SgprV2S16}, {SgprV2S16}, ScalarizeToS16}, hasSALUFloat)
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16}})
      .Any({{UniV2S32}, {{UniInVgprV2S32}, {VgprV2S32}}})
      .Any({{DivV2S32}, {{VgprV2S32}, {VgprV2S32}}});

  addRulesForGOpcs({G_FCANONICALIZE}, Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(S16, {{UniInVgprS16}, {Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16}})
      .Any({{UniV2S32}, {{UniInVgprV2S32}, {VgprV2S32}}})
      .Any({{DivV2S32}, {{VgprV2S32}, {VgprV2S32}}});

  bool hasPST = ST->hasPseudoScalarTrans();
  addRulesForGOpcs({G_FSQRT}, Standard)
      .Div(S16, {{Vgpr16}, {Vgpr16}})
      .Uni(S16, {{Sgpr16}, {Sgpr16}}, hasPST)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16}}, !hasPST);

  addRulesForGOpcs({G_FPTOUI, G_FPTOSI})
      .Any({{UniS16, S16}, {{UniInVgprS16}, {Vgpr16}}})
      .Any({{DivS16, S16}, {{Vgpr16}, {Vgpr16}}})
      .Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}}, hasSALUFloat)
      .Any({{UniS32, S16}, {{UniInVgprS32}, {Vgpr16}}}, !hasSALUFloat)
      .Any({{DivS32, S16}, {{Vgpr32}, {Vgpr16}}})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS32, S32}, {{UniInVgprS32}, {Vgpr32}}}, !hasSALUFloat)
      .Any({{DivS32, S32}, {{Vgpr32}, {Vgpr32}}})
      .Any({{UniS32, S64}, {{UniInVgprS32}, {Vgpr64}}})
      .Any({{DivS32, S64}, {{Vgpr32}, {Vgpr64}}});

  addRulesForGOpcs({G_UITOFP, G_SITOFP})
      .Any({{UniS16, S16}, {{UniInVgprS16}, {Vgpr16}}})
      .Any({{DivS16, S16}, {{Vgpr16}, {Vgpr16}}})
      .Any({{UniS16, S32}, {{Sgpr16}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS16, S32}, {{UniInVgprS16}, {Vgpr32}}}, !hasSALUFloat)
      .Any({{DivS16, S32}, {{Vgpr16}, {Vgpr32}}})
      .Any({{UniS32, S32}, {{Sgpr32}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS32, S32}, {{UniInVgprS32}, {Vgpr32}}}, !hasSALUFloat)
      .Any({{DivS32, S32}, {{Vgpr32}, {Vgpr32}}})
      .Any({{UniS64, S32}, {{UniInVgprS64}, {Vgpr32}}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}}});

  addRulesForGOpcs({G_AMDGPU_S_BUFFER_PREFETCH})
      .Any({{}, {{}, {SgprV4S32_ReadFirstLane, Imm, SgprB32_ReadFirstLane}}});

  addRulesForGOpcs({G_FPEXT})
      .Any({{DivS32, S16}, {{Vgpr32}, {Vgpr16}}})
      .Any({{UniS64, S32}, {{UniInVgprS64}, {Vgpr32}}})
      .Any({{DivS64, S32}, {{Vgpr64}, {Vgpr32}}})
      .Any({{UniS32, S16}, {{Sgpr32}, {Sgpr16}}}, hasSALUFloat)
      .Any({{UniS32, S16}, {{UniInVgprS32}, {Vgpr16}}}, !hasSALUFloat);

  addRulesForGOpcs({G_AMDGPU_CVT_PK_I16_I32}, Standard)
      .Uni(V2S16, {{UniInVgprV2S16}, {Vgpr32, Vgpr32}})
      .Div(V2S16, {{VgprV2S16}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_AMDGPU_FMIN_LEGACY, G_AMDGPU_FMAX_LEGACY}, Standard)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  bool hasSALUMinimumMaximumInsts = ST->hasSALUMinimumMaximumInsts();

  addRulesForGOpcs({G_FMINIMUM, G_FMAXIMUM}, Standard)
      .Uni(S16, {{Sgpr16}, {Sgpr16, Sgpr16}}, hasSALUMinimumMaximumInsts)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}}, !hasSALUMinimumMaximumInsts)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasSALUMinimumMaximumInsts)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasSALUMinimumMaximumInsts)
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64, Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}});

  addRulesForGOpcs({G_FMINNUM_IEEE, G_FMAXNUM_IEEE, G_FMINNUM, G_FMAXNUM,
                    G_FMINIMUMNUM, G_FMAXIMUMNUM},
                   Standard)
      .Div(S16, {{Vgpr16}, {Vgpr16, Vgpr16}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64, Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64, Vgpr64}})
      .Uni(V2S16, {{UniInVgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {VgprV2S16, VgprV2S16}})
      .Uni(S16, {{Sgpr16}, {Sgpr16, Sgpr16}}, hasSALUFloat)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16, Vgpr16}}, !hasSALUFloat)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}}, hasSALUFloat)
      .Uni(S32, {{UniInVgprS32}, {Vgpr32, Vgpr32}}, !hasSALUFloat);

  addRulesForGOpcs({G_FPTRUNC})
      .Any({{DivS16, S32}, {{Vgpr16}, {Vgpr32}}})
      .Any({{UniS32, S64}, {{UniInVgprS32}, {Vgpr64}}})
      .Any({{DivS32, S64}, {{Vgpr32}, {Vgpr64}}})
      .Any({{UniV2S16, V2S32}, {{UniInVgprV2S16}, {VgprV2S32}}})
      .Any({{DivV2S16, V2S32}, {{VgprV2S16}, {VgprV2S32}}})
      .Any({{UniS16, S32}, {{Sgpr16}, {Sgpr32}}}, hasSALUFloat)
      .Any({{UniS16, S32}, {{UniInVgprS16}, {Vgpr32}}}, !hasSALUFloat);

  addRulesForGOpcs({G_IS_FPCLASS})
      .Any({{DivS1, S16}, {{Vcc}, {Vgpr16}}})
      .Any({{UniS1, S16}, {{UniInVcc}, {Vgpr16}}})
      .Any({{DivS1, S32}, {{Vcc}, {Vgpr32}}})
      .Any({{UniS1, S32}, {{UniInVcc}, {Vgpr32}}})
      .Any({{DivS1, S64}, {{Vcc}, {Vgpr64}}})
      .Any({{UniS1, S64}, {{UniInVcc}, {Vgpr64}}});

  addRulesForGOpcs({G_FCMP}, Standard)
      .Any({{UniS1, _, S16}, {{Sgpr32Trunc}, {None, Sgpr16, Sgpr16}}},
           hasSALUFloat)
      .Any({{UniS1, _, S16}, {{UniInVcc}, {None, Vgpr16, Vgpr16}}},
           !hasSALUFloat)
      .Any({{DivS1, _, S16}, {{Vcc}, {None, Vgpr16, Vgpr16}}})
      .Any({{UniS1, _, S32}, {{Sgpr32Trunc}, {None, Sgpr32, Sgpr32}}},
           hasSALUFloat)
      .Any({{UniS1, _, S32}, {{UniInVcc}, {None, Vgpr32, Vgpr32}}},
           !hasSALUFloat)
      .Any({{DivS1, _, S32}, {{Vcc}, {None, Vgpr32, Vgpr32}}})
      .Any({{UniS1, _, S64}, {{UniInVcc}, {None, Vgpr64, Vgpr64}}})
      .Any({{DivS1, _, S64}, {{Vcc}, {None, Vgpr64, Vgpr64}}});

  addRulesForGOpcs({G_INTRINSIC_TRUNC, G_INTRINSIC_ROUNDEVEN, G_FFLOOR, G_FCEIL,
                    G_FEXP2, G_FLOG2},
                   Standard)
      .Uni(S16, {{UniInVgprS16}, {Vgpr16}})
      .Div(S16, {{Vgpr16}, {Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {Vgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {Vgpr64}})
      .Div(S64, {{Vgpr64}, {Vgpr64}});

  using namespace Intrinsic;

  addRulesForIOpcs({returnaddress}).Any({{UniP0}, {{SgprP0}, {}}});

  addRulesForIOpcs({amdgcn_s_getpc}).Any({{UniS64, _}, {{Sgpr64}, {None}}});

  addRulesForIOpcs({amdgcn_s_getreg}).Any({{}, {{Sgpr32}, {IntrId, Imm}}});

  addRulesForIOpcs({amdgcn_s_setreg})
      .Any({{_, _, S32}, {{}, {IntrId, Imm, SgprB32_ReadFirstLane}}});

  addRulesForIOpcs({amdgcn_s_sendmsg, amdgcn_s_sendmsghalt})
      .Any({{}, {{}, {IntrId, Imm, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_s_sendmsg_rtn})
      .Any({{S32}, {{Sgpr32}, {}}})
      .Any({{S64}, {{Sgpr64}, {}}});

  addRulesForIOpcs({amdgcn_s_memrealtime, amdgcn_s_memtime}, Standard)
      .Uni(S64, {{Sgpr64}, {IntrId}});

  addRulesForIOpcs({amdgcn_groupstaticsize, amdgcn_pops_exiting_wave_id,
                    amdgcn_reloc_constant, amdgcn_s_get_waveid_in_workgroup},
                   Standard)
      .Uni(S32, {{Sgpr32}, {IntrId}});

  // Intrinsics with no register operands.
  addRulesForIOpcs({amdgcn_asyncmark,
                    amdgcn_endpgm,
                    amdgcn_init_exec,
                    amdgcn_s_barrier,
                    amdgcn_s_barrier_leave,
                    amdgcn_s_barrier_signal,
                    amdgcn_s_barrier_wait,
                    amdgcn_s_monitor_sleep,
                    amdgcn_s_nop,
                    amdgcn_s_sethalt,
                    amdgcn_s_setprio,
                    amdgcn_s_setprio_inc_wg,
                    amdgcn_s_sleep,
                    amdgcn_s_ttracedata_imm,
                    amdgcn_s_wait_asynccnt,
                    amdgcn_s_wait_bvhcnt,
                    amdgcn_s_wait_dscnt,
                    amdgcn_s_wait_event,
                    amdgcn_s_wait_event_export_ready,
                    amdgcn_s_wait_expcnt,
                    amdgcn_s_wait_kmcnt,
                    amdgcn_s_wait_loadcnt,
                    amdgcn_s_wait_samplecnt,
                    amdgcn_s_wait_storecnt,
                    amdgcn_s_wait_tensorcnt,
                    amdgcn_s_waitcnt,
                    amdgcn_unreachable,
                    amdgcn_wait_asyncmark,
                    amdgcn_wave_barrier})
      .Any({{}, {{}, {}}});

  addRulesForIOpcs({amdgcn_init_exec_from_input})
      .Any({{}, {{}, {IntrId, Sgpr32, Imm}}});

  addRulesForIOpcs({amdgcn_s_ttracedata}).Any({{}, {{}, {IntrId, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_s_sleep_var})
      .Any({{}, {{}, {IntrId, SgprB32_ReadFirstLane}}});

  addRulesForIOpcs({amdgcn_s_barrier_join, amdgcn_s_wakeup_barrier})
      .Any({{}, {{}, {IntrId, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_s_barrier_signal_var, amdgcn_s_barrier_init})
      .Any({{}, {{}, {IntrId, SgprB32_M0, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_s_barrier_signal_isfirst})
      .Any({{UniS1}, {{Sgpr32Trunc}, {}}});

  addRulesForIOpcs(
      {amdgcn_s_get_named_barrier_state, amdgcn_s_get_barrier_state}, Standard)
      .Uni(S32, {{Sgpr32}, {IntrId, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_flat_prefetch}).Any({{}, {{}, {IntrId, VgprP0}}});

  addRulesForIOpcs({amdgcn_global_prefetch}).Any({{}, {{}, {IntrId, VgprP1}}});

  addRulesForIOpcs({amdgcn_s_prefetch_data})
      .Any({{}, {{}, {IntrId, SgprB64_ReadFirstLane, SgprB32_ReadFirstLane}}});

  addRulesForIOpcs({amdgcn_class})
      .Any({{UniS1, _, S16}, {{UniInVcc}, {IntrId, Vgpr16, Vgpr32}}})
      .Any({{DivS1, _, S16}, {{Vcc}, {IntrId, Vgpr16, Vgpr32}}})
      .Any({{UniS1, _, S32}, {{UniInVcc}, {IntrId, Vgpr32, Vgpr32}}})
      .Any({{DivS1, _, S32}, {{Vcc}, {IntrId, Vgpr32, Vgpr32}}})
      .Any({{UniS1, _, S64}, {{UniInVcc}, {IntrId, Vgpr64, Vgpr32}}})
      .Any({{DivS1, _, S64}, {{Vcc}, {IntrId, Vgpr64, Vgpr32}}});

  // This is "intrinsic lane mask" it was set to i32/i64 in llvm-ir.
  addRulesForIOpcs({amdgcn_end_cf})
      .Any({{_, UniS32}, {{}, {IntrId, Sgpr32}}})
      .Any({{_, UniS64}, {{}, {IntrId, Sgpr64}}});

  addRulesForIOpcs({amdgcn_if_break}, Standard)
      .Uni(S64, {{Sgpr64}, {IntrId, Vcc, Sgpr64}})
      .Uni(S32, {{Sgpr32}, {IntrId, Vcc, Sgpr32}});

  addRulesForIOpcs({amdgcn_exp})
      .Any({{_, _, _, S32, S32, S32, S32},
            {{}, {IntrId, Imm, Imm, Vgpr32, Vgpr32, Vgpr32, Vgpr32}}});

  addRulesForIOpcs({amdgcn_exp_compr})
      .Any({{_, _, _, V2S16}, {{}, {IntrId, Imm, Imm, VgprV2S16, VgprV2S16}}});

  addRulesForIOpcs({amdgcn_exp_row})
      .Any({{_, _, _, S32, S32, S32, S32, _, S32},
            {{},
             {IntrId, Imm, Imm, Vgpr32, Vgpr32, Vgpr32, Vgpr32, Imm,
              SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_lds_direct_load}, StandardB)
      .Div(B32, {{VgprB32}, {IntrId, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_lds_param_load}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_mbcnt_lo, amdgcn_mbcnt_hi}, Standard)
      .Div(S32, {{}, {Vgpr32, None, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_readfirstlane})
      .Any({{UniB32, _, DivB32}, {{}, {SgprB32, None, VgprB32}}})
      // this should not exist in the first place, it is from call lowering
      // readfirstlaning just in case register is not in sgpr.
      .Any({{UniS32, _, UniS32}, {{}, {Sgpr32, None, Vgpr32}}});

  addRulesForIOpcs({amdgcn_readlane}, StandardB)
      .Uni(B32, {{SgprB32}, {IntrId, VgprB32, SgprB32_ReadFirstLane}});

  addRulesForIOpcs({amdgcn_writelane}, StandardB)
      .Div(B32,
           {{VgprB32},
            {IntrId, SgprB32_ReadFirstLane, SgprB32_ReadFirstLane, VgprB32}});

  addRulesForIOpcs({amdgcn_add_max_i32, amdgcn_add_max_u32, amdgcn_add_min_i32,
                    amdgcn_add_min_u32},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_pk_add_max_i16, amdgcn_pk_add_max_u16,
                    amdgcn_pk_add_min_i16, amdgcn_pk_add_min_u16},
                   Standard)
      .Uni(V2S16, {{UniInVgprV2S16}, {IntrId, VgprV2S16, VgprV2S16, VgprV2S16}})
      .Div(V2S16, {{VgprV2S16}, {IntrId, VgprV2S16, VgprV2S16, VgprV2S16}});

  addRulesForIOpcs({amdgcn_permlane16, amdgcn_permlanex16}, Standard)
      .Div(S32, {{Vgpr32},
                 {IntrId, Vgpr32, Vgpr32, SgprB32_ReadFirstLane,
                  SgprB32_ReadFirstLane, Imm, Imm}});

  addRulesForIOpcs({amdgcn_permlane_bcast, amdgcn_permlane_up,
                    amdgcn_permlane_down, amdgcn_permlane_xor},
                   Standard)
      .Div(S32,
           {{Vgpr32},
            {IntrId, Vgpr32, SgprB32_ReadFirstLane, SgprB32_ReadFirstLane}});

  addRulesForIOpcs({amdgcn_permlane_idx_gen}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, SgprB32_ReadFirstLane}});

  addRulesForIOpcs({amdgcn_perm}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs(
      {amdgcn_wave_reduce_add, amdgcn_wave_reduce_and, amdgcn_wave_reduce_fadd,
       amdgcn_wave_reduce_fmax, amdgcn_wave_reduce_fmin,
       amdgcn_wave_reduce_fsub, amdgcn_wave_reduce_max, amdgcn_wave_reduce_min,
       amdgcn_wave_reduce_or, amdgcn_wave_reduce_sub, amdgcn_wave_reduce_umax,
       amdgcn_wave_reduce_umin, amdgcn_wave_reduce_xor},
      Standard)
      .Uni(S32, {{Sgpr32}, {IntrId, Sgpr32}})
      .Div(S32, {{Sgpr32ToVgprDst}, {IntrId, VgprB32}})
      .Uni(S64, {{Sgpr64}, {IntrId, Sgpr64}})
      .Div(S64, {{Sgpr64ToVgprDst}, {IntrId, VgprB64}});

  addRulesForIOpcs({amdgcn_bitop3, amdgcn_fmad_ftz}, Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16, Vgpr16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16, Vgpr16, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_udot4, amdgcn_sdot4, amdgcn_udot8, amdgcn_sdot8,
                    amdgcn_dot4_f32_bf8_bf8, amdgcn_dot4_f32_bf8_fp8,
                    amdgcn_dot4_f32_fp8_fp8, amdgcn_dot4_f32_fp8_bf8},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_rsq, amdgcn_rsq_clamp}, Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16}})
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64}});

  addRulesForIOpcs({amdgcn_mul_u24, amdgcn_mul_i24}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_ds_bpermute, amdgcn_ds_bpermute_fi_b32,
                    amdgcn_ds_permute, amdgcn_fmul_legacy, amdgcn_mulhi_i24,
                    amdgcn_mulhi_u24},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_cvt_sr_bf8_f32, amdgcn_cvt_sr_fp8_f32,
                    amdgcn_cvt_pk_bf8_f32, amdgcn_cvt_pk_fp8_f32},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_cvt_f32_bf8, amdgcn_cvt_f32_fp8}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}});

  addRulesForIOpcs({amdgcn_cvt_pk_f32_bf8, amdgcn_cvt_pk_f32_fp8})
      .Any({{UniV2S32}, {{UniInVgprV2S32}, {IntrId, Vgpr32}}})
      .Any({{DivV2S32}, {{VgprV2S32}, {IntrId, Vgpr32}}});

  addRulesForIOpcs({amdgcn_cubesc, amdgcn_cubetc, amdgcn_cubema, amdgcn_cubeid,
                    amdgcn_fma_legacy},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_frexp_mant, amdgcn_fract}, Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16}})
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64}});

  addRulesForIOpcs({amdgcn_prng_b32})
      .Any({{UniS32}, {{UniInVgprS32}, {IntrId, Vgpr32}}})
      .Any({{DivS32}, {{Vgpr32}, {IntrId, Vgpr32}}});

  addRulesForIOpcs({amdgcn_sffbh}, Standard)
      .Uni(S32, {{Sgpr32}, {IntrId, Sgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}});

  addRulesForIOpcs({amdgcn_ubfe, amdgcn_sbfe}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Uni(S32, {{Sgpr32}, {IntrId, Sgpr32, Sgpr32, Sgpr32}, S_BFE})
      .Uni(S64, {{Sgpr64}, {IntrId, Sgpr64, Sgpr32, Sgpr32}, S_BFE})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64, Vgpr32, Vgpr32}, V_BFE});

  addRulesForIOpcs({amdgcn_cvt_pk_i16, amdgcn_cvt_pk_u16, amdgcn_cvt_pknorm_i16,
                    amdgcn_cvt_pknorm_u16, amdgcn_cvt_pkrtz},
                   Standard)
      .Uni(V2S16, {{UniInVgprV2S16}, {IntrId, Vgpr32, Vgpr32}})
      .Div(V2S16, {{VgprV2S16}, {IntrId, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_cvt_scalef32_sr_pk32_bf6_f16,
                    amdgcn_cvt_scalef32_sr_pk32_fp6_f16,
                    amdgcn_cvt_scalef32_sr_pk32_bf6_bf16,
                    amdgcn_cvt_scalef32_sr_pk32_fp6_bf16},
                   Standard)
      .Any({{DivV6S32}, {{VgprV6S32}, {IntrId, VgprV32S16, Vgpr32, Vgpr32}}});

  addRulesForIOpcs({amdgcn_cvt_scalef32_sr_pk32_bf6_f32,
                    amdgcn_cvt_scalef32_sr_pk32_fp6_f32},
                   Standard)
      .Any({{DivV6S32}, {{VgprV6S32}, {IntrId, VgprV32S32, Vgpr32, Vgpr32}}});

  addRulesForIOpcs({amdgcn_global_load_tr_b64})
      .Any({{DivB64, _, UniP1}, {{VgprB64}, {IntrId, SgprP1}}})
      .Any({{DivB64, _, DivP1}, {{VgprB64}, {IntrId, VgprP1}}})
      .Any({{DivB32, _, UniP1}, {{VgprB32}, {IntrId, SgprP1}}})
      .Any({{DivB32, _, DivP1}, {{VgprB32}, {IntrId, VgprP1}}});

  addRulesForIOpcs({amdgcn_global_load_tr_b128})
      .Any({{DivB64, _, UniP1}, {{VgprB64}, {IntrId, SgprP1}}})
      .Any({{DivB64, _, DivP1}, {{VgprB64}, {IntrId, VgprP1}}})
      .Any({{DivB128, _, UniP1}, {{VgprB128}, {IntrId, SgprP1}}})
      .Any({{DivB128, _, DivP1}, {{VgprB128}, {IntrId, VgprP1}}});

  addRulesForIOpcs({amdgcn_global_load_tr4_b64})
      .Any({{DivV2S32, _, UniP1}, {{VgprV2S32}, {IntrId, SgprP1}}})
      .Any({{DivV2S32, _, DivP1}, {{VgprV2S32}, {IntrId, VgprP1}}});

  addRulesForIOpcs({amdgcn_global_load_tr6_b96})
      .Any({{DivV3S32, _, UniP1}, {{VgprV3S32}, {IntrId, SgprP1}}})
      .Any({{DivV3S32, _, DivP1}, {{VgprV3S32}, {IntrId, VgprP1}}});

  addRulesForIOpcs({amdgcn_ds_load_tr4_b64, amdgcn_ds_load_tr8_b64})
      .Any({{DivV2S32}, {{VgprV2S32}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_ds_load_tr6_b96})
      .Any({{DivV3S32}, {{VgprV3S32}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_ds_load_tr16_b128})
      .Any({{DivB128}, {{VgprB128}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_global_atomic_ordered_add_b64})
      .Any({{DivS64}, {{Vgpr64}, {IntrId, VgprP1, Vgpr64}}});

  addRulesForIOpcs(
      {amdgcn_global_atomic_fmin_num, amdgcn_global_atomic_fmax_num}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, VgprP1, Vgpr32}});

  addRulesForIOpcs({amdgcn_flat_atomic_fmin_num, amdgcn_flat_atomic_fmax_num},
                   Standard)
      .Div(S32, {{Vgpr32}, {IntrId, VgprP0, Vgpr32}});

  addRulesForIOpcs({amdgcn_raw_buffer_load_lds})
      .Any({{_}, {{}, {IntrId, SgprV4S32, SgprP3, Imm, Vgpr32, Sgpr32}}});

  addRulesForIOpcs({amdgcn_struct_buffer_load_lds})
      .Any({{_},
            {{}, {IntrId, SgprV4S32, SgprP3, Imm, Vgpr32, Vgpr32, Sgpr32}}});

  addRulesForIOpcs({amdgcn_raw_ptr_buffer_load_lds})
      .Any({{_}, {{}, {IntrId, SgprP8, SgprP3, Imm, Vgpr32, Sgpr32}}});

  addRulesForIOpcs({amdgcn_struct_ptr_buffer_load_lds})
      .Any({{_}, {{}, {IntrId, SgprP8, SgprP3, Imm, Vgpr32, Vgpr32, Sgpr32}}});

  addRulesForIOpcs({amdgcn_global_load_lds})
      .Any({{}, {{}, {IntrId, VgprP1, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_global_load_async_to_lds_b8,
                    amdgcn_global_load_async_to_lds_b32,
                    amdgcn_global_load_async_to_lds_b64,
                    amdgcn_global_load_async_to_lds_b128,
                    amdgcn_global_store_async_from_lds_b8,
                    amdgcn_global_store_async_from_lds_b32,
                    amdgcn_global_store_async_from_lds_b64,
                    amdgcn_global_store_async_from_lds_b128})
      .Any({{}, {{}, {IntrId, VgprP1, VgprP3}}});

  addRulesForIOpcs({amdgcn_perm_pk16_b4_u4}, StandardB)
      .Uni(B64, {{UniInVgprB64}, {IntrId, Vgpr32, Vgpr32, VgprV2S32}})
      .Div(B64, {{VgprB64}, {IntrId, Vgpr32, Vgpr32, VgprV2S32}});

  addRulesForIOpcs({amdgcn_perm_pk16_b6_u4}, StandardB)
      .Uni(B96, {{UniInVgprB96}, {IntrId, Vgpr32, VgprB64, VgprV2S32}})
      .Div(B96, {{VgprB96}, {IntrId, Vgpr32, VgprB64, VgprV2S32}});

  addRulesForIOpcs({amdgcn_perm_pk16_b8_u4}, StandardB)
      .Uni(B128, {{UniInVgprB128}, {IntrId, VgprB64, VgprB64, VgprV2S32}})
      .Div(B128, {{VgprB128}, {IntrId, VgprB64, VgprB64, VgprV2S32}});

  addRulesForIOpcs({amdgcn_wwm, amdgcn_strict_wwm, amdgcn_wqm, amdgcn_softwqm,
                    amdgcn_strict_wqm},
                   StandardB)
      .Div(B32, {{VgprB32}, {IntrId, VgprB32}})
      .Uni(B32, {{SgprB32}, {IntrId, SgprB32}})
      .Div(B64, {{VgprB64}, {IntrId, VgprB64}})
      .Uni(B64, {{SgprB64}, {IntrId, SgprB64}})
      .Div(B96, {{VgprB96}, {IntrId, VgprB96}})
      .Uni(B96, {{SgprB96}, {IntrId, SgprB96}})
      .Div(B128, {{VgprB128}, {IntrId, VgprB128}})
      .Uni(B128, {{SgprB128}, {IntrId, SgprB128}})
      .Any({{UniB256}, {{SgprB256}, {IntrId, SgprB256}}})
      .Any({{DivB256}, {{VgprB256}, {IntrId, VgprB256}}})
      .Any({{UniB512}, {{SgprB512}, {IntrId, SgprB512}}})
      .Any({{DivB512}, {{VgprB512}, {IntrId, VgprB512}}});

  addRulesForIOpcs({amdgcn_wqm_demote}).Any({{}, {{}, {IntrId, Vcc}}});

  addRulesForIOpcs({amdgcn_ballot}, Standard)
      .Uni(S64, {{Sgpr64}, {IntrId, Vcc}})
      .Uni(S32, {{Sgpr32}, {IntrId, Vcc}});

  addRulesForIOpcs({amdgcn_inverse_ballot})
      .Any({{DivS1, _, S32}, {{Vcc}, {IntrId, SgprB32_ReadFirstLane}}})
      .Any({{DivS1, _, S64}, {{Vcc}, {IntrId, SgprB64_ReadFirstLane}}});

  addRulesForIOpcs({amdgcn_live_mask, amdgcn_ps_live})
      .Any({{DivS1}, {{Vcc}, {}}});

  addRulesForIOpcs({amdgcn_mov_dpp, amdgcn_mov_dpp8}, StandardB)
      .Div(B32, {{VgprB32}, {IntrId, VgprB32}})
      .Div(B64, {{VgprB64}, {IntrId, VgprB64}});

  addRulesForIOpcs({amdgcn_update_dpp}, StandardB)
      .Div(B32, {{VgprB32}, {IntrId, VgprB32, VgprB32}})
      .Div(B64, {{VgprB64}, {IntrId, VgprB64, VgprB64}});

  addRulesForIOpcs({amdgcn_sin, amdgcn_cos}, Standard)
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16}})
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}});

  addRulesForIOpcs({amdgcn_trig_preop}, Standard)
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64, Vgpr32}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64, Vgpr32}});

  addRulesForIOpcs({amdgcn_exp2}, Standard)
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16}})
      .Uni(S16, {{Sgpr16}, {IntrId, Sgpr16}}, hasPST)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16}}, !hasPST)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Uni(S32, {{Sgpr32}, {IntrId, Sgpr32}}, hasPST)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}}, !hasPST);

  addRulesForIOpcs({amdgcn_sqrt}, Standard)
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16}})
      .Uni(S16, {{Sgpr16}, {IntrId, Sgpr16}}, hasPST)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16}}, !hasPST)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Uni(S32, {{Sgpr32}, {IntrId, Sgpr32}}, hasPST)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}}, !hasPST)
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64}});

  addRulesForIOpcs({amdgcn_ds_atomic_async_barrier_arrive_b64})
      .Any({{}, {{}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_ds_atomic_barrier_arrive_rtn_b64}, Standard)
      .Div(S64, {{Vgpr64}, {IntrId, VgprP3, Vgpr64}});

  addRulesForIOpcs({amdgcn_ds_add_gs_reg_rtn, amdgcn_ds_sub_gs_reg_rtn},
                   Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr32}});

  addRulesForIOpcs({amdgcn_ds_append, amdgcn_ds_consume}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, SgprB32_M0}})
      .Div(S32, {{Vgpr32}, {IntrId, SgprB32_M0}});

  addRulesForIOpcs(
      {amdgcn_ds_bvh_stack_rtn, amdgcn_ds_bvh_stack_push4_pop1_rtn}, Standard)
      .Div(S32, {{Vgpr32, Vgpr32}, {IntrId, Vgpr32, Vgpr32, VgprV4S32}});

  addRulesForIOpcs({amdgcn_ds_bvh_stack_push8_pop1_rtn}, Standard)
      .Div(S32, {{Vgpr32, Vgpr32}, {IntrId, Vgpr32, Vgpr32, VgprV8S32}});

  addRulesForIOpcs({amdgcn_ds_bvh_stack_push8_pop2_rtn}, Standard)
      .Div(S64, {{Vgpr64, Vgpr32}, {IntrId, Vgpr32, Vgpr32, VgprV8S32}});

  addRulesForIOpcs({amdgcn_ds_gws_sema_p, amdgcn_ds_gws_sema_v,
                    amdgcn_ds_gws_sema_release_all})
      .Any({{}, {{}, {IntrId, SgprB32_M0}}});

  addRulesForIOpcs(
      {amdgcn_ds_gws_barrier, amdgcn_ds_gws_init, amdgcn_ds_gws_sema_br})
      .Any({{}, {{}, {IntrId, Vgpr32, SgprB32_M0}}});

  addRulesForIOpcs({amdgcn_ds_ordered_add, amdgcn_ds_ordered_swap}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, SgprB32_M0, Vgpr32}});

  addRulesForIOpcs({amdgcn_ds_swizzle}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32}});

  addRulesForIOpcs({amdgcn_permlane16_var, amdgcn_permlanex16_var}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_permlane16_swap, amdgcn_permlane32_swap}, Standard)
      .Div(S32, {{Vgpr32, Vgpr32}, {IntrId, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_permlane64}, StandardB)
      .Div(B32, {{VgprB32}, {IntrId, VgprB32}});

  addRulesForIOpcs({amdgcn_ds_read_tr4_b64, amdgcn_ds_read_tr8_b64})
      .Any({{DivV2S32}, {{VgprV2S32}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_ds_read_tr6_b96})
      .Any({{DivV3S32}, {{VgprV3S32}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_ds_read_tr16_b64})
      .Any({{DivV4S16}, {{VgprV4S16}, {IntrId, VgprP3}}});

  addRulesForIOpcs({amdgcn_interp_p1}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_interp_p1_f16}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Imm, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_interp_p2}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_interp_p2_f16}, Standard)
      .Div(S16,
           {{Vgpr16}, {IntrId, Vgpr32, Vgpr32, Imm, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_interp_mov}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Imm, Imm, Imm, SgprB32_M0}});

  addRulesForIOpcs({amdgcn_interp_inreg_p10, amdgcn_interp_inreg_p2,
                    amdgcn_interp_inreg_p10_f16, amdgcn_interp_p10_rtz_f16},
                   Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_interp_inreg_p2_f16, amdgcn_interp_p2_rtz_f16},
                   Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr32, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_div_fmas}, Standard)
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32, Vcc}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32, Vcc}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64, Vgpr64, Vgpr64, Vcc}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64, Vgpr64, Vgpr64, Vcc}});

  addRulesForIOpcs({amdgcn_div_fixup}, Standard)
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr16, Vgpr16, Vgpr16}})
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr16, Vgpr16, Vgpr16}})
      .Div(S32, {{Vgpr32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Uni(S32, {{UniInVgprS32}, {IntrId, Vgpr32, Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64}, {IntrId, Vgpr64, Vgpr64, Vgpr64}})
      .Uni(S64, {{UniInVgprS64}, {IntrId, Vgpr64, Vgpr64, Vgpr64}});

  addRulesForIOpcs({amdgcn_div_scale}, Standard)
      .Div(S32, {{Vgpr32, Vcc}, {IntrId, Vgpr32, Vgpr32}})
      .Uni(S32, {{UniInVgprS32, UniInVcc}, {IntrId, Vgpr32, Vgpr32}})
      .Div(S64, {{Vgpr64, Vcc}, {IntrId, Vgpr64, Vgpr64}})
      .Uni(S64, {{UniInVgprS64, UniInVcc}, {IntrId, Vgpr64, Vgpr64}});

  addRulesForIOpcs({amdgcn_fdot2, amdgcn_sdot2, amdgcn_udot2}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, VgprV2S16, VgprV2S16, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, VgprV2S16, VgprV2S16, Vgpr32}});

  addRulesForIOpcs({amdgcn_fdot2_f16_f16}, Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, VgprV2S16, VgprV2S16, Vgpr16}})
      .Div(S16, {{Vgpr16}, {IntrId, VgprV2S16, VgprV2S16, Vgpr16}});

  addRulesForIOpcs({amdgcn_sudot4, amdgcn_sudot8}, Standard)
      .Uni(S32, {{UniInVgprS32}, {IntrId, Imm, Vgpr32, Imm, Vgpr32, Vgpr32}})
      .Div(S32, {{Vgpr32}, {IntrId, Imm, Vgpr32, Imm, Vgpr32, Vgpr32}});

  addRulesForIOpcs({amdgcn_s_alloc_vgpr})
      .Any({{UniS1}, {{Sgpr32Trunc}, {IntrId, SgprB32_ReadFirstLane}}});

  addRulesForIOpcs({amdgcn_sat_pk4_i4_i8, amdgcn_sat_pk4_u4_u8}, Standard)
      .Uni(S16, {{UniInVgprS16}, {IntrId, Vgpr32}})
      .Div(S16, {{Vgpr16}, {IntrId, Vgpr32}});

  // TODO: Add handling for GFX90A+ which should use VGPRs instead of AGPRs.
  bool HasGFX90AInsts = ST->hasGFX90AInsts();
  addRulesForIOpcs({amdgcn_mfma_f32_32x32x1f32,  amdgcn_mfma_f32_16x16x1f32,
                    amdgcn_mfma_f32_4x4x1f32,    amdgcn_mfma_f32_32x32x2f32,
                    amdgcn_mfma_f32_16x16x4f32,  amdgcn_mfma_f32_32x32x4f16,
                    amdgcn_mfma_f32_16x16x4f16,  amdgcn_mfma_f32_4x4x4f16,
                    amdgcn_mfma_f32_32x32x8f16,  amdgcn_mfma_f32_16x16x16f16,
                    amdgcn_mfma_i32_32x32x4i8,   amdgcn_mfma_i32_16x16x4i8,
                    amdgcn_mfma_i32_4x4x4i8,     amdgcn_mfma_i32_32x32x8i8,
                    amdgcn_mfma_i32_16x16x16i8,  amdgcn_mfma_f32_32x32x2bf16,
                    amdgcn_mfma_f32_16x16x2bf16, amdgcn_mfma_f32_4x4x2bf16,
                    amdgcn_mfma_f32_32x32x4bf16, amdgcn_mfma_f32_16x16x8bf16})
      .Any({{DivAnyTy},
            {{AgprAnyTy}, {IntrId, VgprAnyTy, VgprAnyTy, AgprAnyTy}}},
           !HasGFX90AInsts);

  // WMMA/SWMMAC intrinsics: all register operands map to VGPR.
  addRulesForIOpcs(
      {// WMMA GFX11+
       amdgcn_wmma_f32_16x16x16_f16, amdgcn_wmma_f32_16x16x16_bf16,
       amdgcn_wmma_f16_16x16x16_f16, amdgcn_wmma_bf16_16x16x16_bf16,
       amdgcn_wmma_f16_16x16x16_f16_tied, amdgcn_wmma_bf16_16x16x16_bf16_tied,
       amdgcn_wmma_i32_16x16x16_iu8, amdgcn_wmma_i32_16x16x16_iu4,
       // WMMA GFX12
       amdgcn_wmma_f32_16x16x16_fp8_fp8, amdgcn_wmma_f32_16x16x16_fp8_bf8,
       amdgcn_wmma_f32_16x16x16_bf8_fp8, amdgcn_wmma_f32_16x16x16_bf8_bf8,
       amdgcn_wmma_i32_16x16x32_iu4,
       // WMMA GFX1250
       amdgcn_wmma_f32_16x16x4_f32, amdgcn_wmma_f32_16x16x32_bf16,
       amdgcn_wmma_f32_16x16x32_f16, amdgcn_wmma_f16_16x16x32_f16,
       amdgcn_wmma_bf16_16x16x32_bf16, amdgcn_wmma_bf16f32_16x16x32_bf16,
       amdgcn_wmma_f32_16x16x64_fp8_fp8, amdgcn_wmma_f32_16x16x64_fp8_bf8,
       amdgcn_wmma_f32_16x16x64_bf8_fp8, amdgcn_wmma_f32_16x16x64_bf8_bf8,
       amdgcn_wmma_f16_16x16x64_fp8_fp8, amdgcn_wmma_f16_16x16x64_fp8_bf8,
       amdgcn_wmma_f16_16x16x64_bf8_fp8, amdgcn_wmma_f16_16x16x64_bf8_bf8,
       amdgcn_wmma_f16_16x16x128_fp8_fp8, amdgcn_wmma_f16_16x16x128_fp8_bf8,
       amdgcn_wmma_f16_16x16x128_bf8_fp8, amdgcn_wmma_f16_16x16x128_bf8_bf8,
       amdgcn_wmma_f32_16x16x128_fp8_fp8, amdgcn_wmma_f32_16x16x128_fp8_bf8,
       amdgcn_wmma_f32_16x16x128_bf8_fp8, amdgcn_wmma_f32_16x16x128_bf8_bf8,
       amdgcn_wmma_i32_16x16x64_iu8, amdgcn_wmma_f32_16x16x128_f8f6f4,
       amdgcn_wmma_scale_f32_16x16x128_f8f6f4,
       amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, amdgcn_wmma_f32_32x16x128_f4,
       amdgcn_wmma_scale_f32_32x16x128_f4, amdgcn_wmma_scale16_f32_32x16x128_f4,
       // SWMMAC GFX12
       amdgcn_swmmac_f32_16x16x32_f16, amdgcn_swmmac_f32_16x16x32_bf16,
       amdgcn_swmmac_f16_16x16x32_f16, amdgcn_swmmac_bf16_16x16x32_bf16,
       amdgcn_swmmac_i32_16x16x32_iu8, amdgcn_swmmac_i32_16x16x32_iu4,
       amdgcn_swmmac_i32_16x16x64_iu4, amdgcn_swmmac_f32_16x16x32_fp8_fp8,
       amdgcn_swmmac_f32_16x16x32_fp8_bf8, amdgcn_swmmac_f32_16x16x32_bf8_fp8,
       amdgcn_swmmac_f32_16x16x32_bf8_bf8,
       // SWMMAC GFX1250
       amdgcn_swmmac_f32_16x16x64_f16, amdgcn_swmmac_f32_16x16x64_bf16,
       amdgcn_swmmac_f16_16x16x64_f16, amdgcn_swmmac_bf16_16x16x64_bf16,
       amdgcn_swmmac_bf16f32_16x16x64_bf16, amdgcn_swmmac_f32_16x16x128_fp8_fp8,
       amdgcn_swmmac_f32_16x16x128_fp8_bf8, amdgcn_swmmac_f32_16x16x128_bf8_fp8,
       amdgcn_swmmac_f32_16x16x128_bf8_bf8, amdgcn_swmmac_f16_16x16x128_fp8_fp8,
       amdgcn_swmmac_f16_16x16x128_fp8_bf8, amdgcn_swmmac_f16_16x16x128_bf8_fp8,
       amdgcn_swmmac_f16_16x16x128_bf8_bf8, amdgcn_swmmac_i32_16x16x128_iu8})
      .Any({{}, {{}, {}, ApplyAllVgpr}});

} // end initialize rules
