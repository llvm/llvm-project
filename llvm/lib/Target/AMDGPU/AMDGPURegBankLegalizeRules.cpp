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
  case P1:
    return MRI.getType(Reg) == LLT::pointer(1, 64);
  case P3:
    return MRI.getType(Reg) == LLT::pointer(3, 32);
  case P4:
    return MRI.getType(Reg) == LLT::pointer(4, 64);
  case P5:
    return MRI.getType(Reg) == LLT::pointer(5, 32);
  case B32:
    return MRI.getType(Reg).getSizeInBits() == 32;
  case B64:
    return MRI.getType(Reg).getSizeInBits() == 64;
  case B96:
    return MRI.getType(Reg).getSizeInBits() == 96;
  case B128:
    return MRI.getType(Reg).getSizeInBits() == 128;
  case B256:
    return MRI.getType(Reg).getSizeInBits() == 256;
  case B512:
    return MRI.getType(Reg).getSizeInBits() == 512;
  case UniS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isUniform(Reg);
  case UniS16:
    return MRI.getType(Reg) == LLT::scalar(16) && MUI.isUniform(Reg);
  case UniS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isUniform(Reg);
  case UniS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isUniform(Reg);
  case UniP1:
    return MRI.getType(Reg) == LLT::pointer(1, 64) && MUI.isUniform(Reg);
  case UniP3:
    return MRI.getType(Reg) == LLT::pointer(3, 32) && MUI.isUniform(Reg);
  case UniP4:
    return MRI.getType(Reg) == LLT::pointer(4, 64) && MUI.isUniform(Reg);
  case UniP5:
    return MRI.getType(Reg) == LLT::pointer(5, 32) && MUI.isUniform(Reg);
  case UniB32:
    return MRI.getType(Reg).getSizeInBits() == 32 && MUI.isUniform(Reg);
  case UniB64:
    return MRI.getType(Reg).getSizeInBits() == 64 && MUI.isUniform(Reg);
  case UniB96:
    return MRI.getType(Reg).getSizeInBits() == 96 && MUI.isUniform(Reg);
  case UniB128:
    return MRI.getType(Reg).getSizeInBits() == 128 && MUI.isUniform(Reg);
  case UniB256:
    return MRI.getType(Reg).getSizeInBits() == 256 && MUI.isUniform(Reg);
  case UniB512:
    return MRI.getType(Reg).getSizeInBits() == 512 && MUI.isUniform(Reg);
  case DivS1:
    return MRI.getType(Reg) == LLT::scalar(1) && MUI.isDivergent(Reg);
  case DivS32:
    return MRI.getType(Reg) == LLT::scalar(32) && MUI.isDivergent(Reg);
  case DivS64:
    return MRI.getType(Reg) == LLT::scalar(64) && MUI.isDivergent(Reg);
  case DivP1:
    return MRI.getType(Reg) == LLT::pointer(1, 64) && MUI.isDivergent(Reg);
  case DivP3:
    return MRI.getType(Reg) == LLT::pointer(3, 32) && MUI.isDivergent(Reg);
  case DivP4:
    return MRI.getType(Reg) == LLT::pointer(4, 64) && MUI.isDivergent(Reg);
  case DivP5:
    return MRI.getType(Reg) == LLT::pointer(5, 32) && MUI.isDivergent(Reg);
  case DivB32:
    return MRI.getType(Reg).getSizeInBits() == 32 && MUI.isDivergent(Reg);
  case DivB64:
    return MRI.getType(Reg).getSizeInBits() == 64 && MUI.isDivergent(Reg);
  case DivB96:
    return MRI.getType(Reg).getSizeInBits() == 96 && MUI.isDivergent(Reg);
  case DivB128:
    return MRI.getType(Reg).getSizeInBits() == 128 && MUI.isDivergent(Reg);
  case DivB256:
    return MRI.getType(Reg).getSizeInBits() == 256 && MUI.isDivergent(Reg);
  case DivB512:
    return MRI.getType(Reg).getSizeInBits() == 512 && MUI.isDivergent(Reg);
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
    if (OpUniformityAndTypes[i] == _) {
      if (MI.getOperand(i).isReg())
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

UniformityLLTOpPredicateID LLTToBId(LLT Ty) {
  if (Ty == LLT::scalar(32) || Ty == LLT::fixed_vector(2, 16) ||
      Ty == LLT::pointer(3, 32) || Ty == LLT::pointer(5, 32) ||
      Ty == LLT::pointer(6, 32))
    return B32;
  if (Ty == LLT::scalar(64) || Ty == LLT::fixed_vector(2, 32) ||
      Ty == LLT::fixed_vector(4, 16) || Ty == LLT::pointer(1, 64) ||
      Ty == LLT::pointer(4, 64))
    return B64;
  if (Ty == LLT::fixed_vector(3, 32))
    return B96;
  if (Ty == LLT::fixed_vector(4, 32))
    return B128;
  return _;
}

const RegBankLLTMapping &
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
      return MUI.isUniform(Reg) ? Uni[Slot] : Div[Slot];
  }

  // Slow search for more complex rules.
  for (const RegBankLegalizeRule &Rule : Rules) {
    if (Rule.Predicate.match(MI, MUI, MRI))
      return Rule.OperandMapping;
  }

  LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
  llvm_unreachable("None of the rules defined for MI's opcode matched MI");
}

void SetOfRulesForOpcode::addRule(RegBankLegalizeRule Rule) {
  Rules.push_back(Rule);
}

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

const SetOfRulesForOpcode &
RegBankLegalizeRules::getRulesForOpc(MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  if (Opc == AMDGPU::G_INTRINSIC || Opc == AMDGPU::G_INTRINSIC_CONVERGENT ||
      Opc == AMDGPU::G_INTRINSIC_W_SIDE_EFFECTS ||
      Opc == AMDGPU::G_INTRINSIC_CONVERGENT_W_SIDE_EFFECTS) {
    unsigned IntrID = cast<GIntrinsic>(MI).getIntrinsicID();
    if (!IRulesAlias.contains(IntrID)) {
      LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
      llvm_unreachable("No rules defined for intrinsic opcode");
    }
    return IRules.at(IRulesAlias.at(IntrID));
  }

  if (!GRulesAlias.contains(Opc)) {
    LLVM_DEBUG(dbgs() << "MI: "; MI.dump(););
    llvm_unreachable("No rules defined for generic opcode");
  }
  return GRules.at(GRulesAlias.at(Opc));
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

  addRulesForGOpcs({G_ADD}, Standard)
      .Uni(S32, {{Sgpr32}, {Sgpr32, Sgpr32}})
      .Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_MUL}, Standard).Div(S32, {{Vgpr32}, {Vgpr32, Vgpr32}});

  addRulesForGOpcs({G_XOR, G_OR, G_AND}, StandardB)
      .Any({{UniS1}, {{Sgpr32Trunc}, {Sgpr32AExt, Sgpr32AExt}}})
      .Any({{DivS1}, {{Vcc}, {Vcc, Vcc}}})
      .Div(B64, {{VgprB64}, {VgprB64, VgprB64}, SplitTo32});

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

  addRulesForGOpcs({G_SELECT}, StandardB)
      .Div(B32, {{VgprB32}, {Vcc, VgprB32, VgprB32}})
      .Uni(B32, {{SgprB32}, {Sgpr32AExtBoolInReg, SgprB32, SgprB32}});

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

  bool hasUnalignedLoads = ST->getGeneration() >= AMDGPUSubtarget::GFX12;
  bool hasSMRDSmall = ST->hasScalarSubwordLoads();

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
    return AMDGPUInstrInfo::isUniformMMO(*MI.memoperands_begin());
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

  Predicate isNaturalAlignedSmall([](const MachineInstr &MI) -> bool {
    const MachineMemOperand *MMO = *MI.memoperands_begin();
    const unsigned MemSize = 8 * MMO->getSize().getValue();
    return (MemSize == 16 && MMO->getAlign() >= Align(2)) ||
           (MemSize == 8 && MMO->getAlign() >= Align(1));
  });

  auto isUL = !isAtomicMMO && isUniMMO && (isConst || !isVolatileMMO) &&
              (isConst || isInvMMO || isNoClobberMMO);

  // clang-format off
  addRulesForGOpcs({G_LOAD})
      .Any({{DivB32, DivP1}, {{VgprB32}, {VgprP1}}})
      .Any({{{UniB256, UniP1}, isAlign4 && isUL}, {{SgprB256}, {SgprP1}}})
      .Any({{{UniB512, UniP1}, isAlign4 && isUL}, {{SgprB512}, {SgprP1}}})
      .Any({{{UniB256, UniP1}, !isAlign4 || !isUL}, {{UniInVgprB256}, {VgprP1}, SplitLoad}})
      .Any({{{UniB512, UniP1}, !isAlign4 || !isUL}, {{UniInVgprB512}, {VgprP1}, SplitLoad}})

      .Any({{DivB32, UniP3}, {{VgprB32}, {VgprP3}}})
      .Any({{{UniB32, UniP3}, isAlign4 && isUL}, {{SgprB32}, {SgprP3}}})
      .Any({{{UniB32, UniP3}, !isAlign4 || !isUL}, {{UniInVgprB32}, {VgprP3}}})

      .Any({{{DivB256, DivP4}}, {{VgprB256}, {VgprP4}, SplitLoad}})
      .Any({{{UniB32, UniP4}, isNaturalAlignedSmall && isUL}, {{SgprB32}, {SgprP4}}}, hasSMRDSmall) // i8 and i16 load
      .Any({{{UniB32, UniP4}, isAlign4 && isUL}, {{SgprB32}, {SgprP4}}})
      .Any({{{UniB96, UniP4}, isAlign16 && isUL}, {{SgprB96}, {SgprP4}, WidenLoad}}, !hasUnalignedLoads)
      .Any({{{UniB96, UniP4}, isAlign4 && !isAlign16 && isUL}, {{SgprB96}, {SgprP4}, SplitLoad}}, !hasUnalignedLoads)
      .Any({{{UniB96, UniP4}, isAlign4 && isUL}, {{SgprB96}, {SgprP4}}}, hasUnalignedLoads)
      .Any({{{UniB256, UniP4}, isAlign4 && isUL}, {{SgprB256}, {SgprP4}}})
      .Any({{{UniB512, UniP4}, isAlign4 && isUL}, {{SgprB512}, {SgprP4}}})
      .Any({{{UniB32, UniP4}, !isNaturalAlignedSmall || !isUL}, {{UniInVgprB32}, {VgprP4}}}, hasSMRDSmall) // i8 and i16 load
      .Any({{{UniB32, UniP4}, !isAlign4 || !isUL}, {{UniInVgprB32}, {VgprP4}}})
      .Any({{{UniB256, UniP4}, !isAlign4 || !isUL}, {{UniInVgprB256}, {VgprP4}, SplitLoad}})
      .Any({{{UniB512, UniP4}, !isAlign4 || !isUL}, {{UniInVgprB512}, {VgprP4}, SplitLoad}})

      .Any({{DivB32, P5}, {{VgprB32}, {VgprP5}}});

  addRulesForGOpcs({G_ZEXTLOAD}) // i8 and i16 zero-extending loads
      .Any({{{UniB32, UniP3}, !isAlign4 || !isUL}, {{UniInVgprB32}, {VgprP3}}})
      .Any({{{UniB32, UniP4}, !isAlign4 || !isUL}, {{UniInVgprB32}, {VgprP4}}});
  // clang-format on

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

} // end initialize rules
