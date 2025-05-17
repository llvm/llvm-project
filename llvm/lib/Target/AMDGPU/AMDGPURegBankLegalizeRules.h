//===- AMDGPURegBankLegalizeRules --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGBANKLEGALIZERULES_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGBANKLEGALIZERULES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace llvm {

class MachineRegisterInfo;
class MachineInstr;
class GCNSubtarget;
class MachineFunction;
template <typename T> class GenericUniformityInfo;
template <typename T> class GenericSSAContext;
using MachineSSAContext = GenericSSAContext<MachineFunction>;
using MachineUniformityInfo = GenericUniformityInfo<MachineSSAContext>;

namespace AMDGPU {

// IDs used to build predicate for RegBankLegalizeRule. Predicate can have one
// or more IDs and each represents a check for 'uniform or divergent' + LLT or
// just LLT on register operand.
// Most often checking one operand is enough to decide which RegBankLLTMapping
// to apply (see Fast Rules), IDs are useful when two or more operands need to
// be checked.
enum UniformityLLTOpPredicateID {
  _,
  // scalars
  S1,
  S16,
  S32,
  S64,

  UniS1,
  UniS16,
  UniS32,
  UniS64,

  DivS1,
  DivS32,
  DivS64,

  // pointers
  P0,
  P1,
  P3,
  P4,
  P5,

  UniP0,
  UniP1,
  UniP3,
  UniP4,
  UniP5,

  DivP0,
  DivP1,
  DivP3,
  DivP4,
  DivP5,

  // vectors
  V2S16,
  V2S32,
  V3S32,
  V4S32,

  // B types
  B32,
  B64,
  B96,
  B128,
  B256,
  B512,

  UniB32,
  UniB64,
  UniB96,
  UniB128,
  UniB256,
  UniB512,

  DivB32,
  DivB64,
  DivB96,
  DivB128,
  DivB256,
  DivB512,
};

// How to apply register bank on register operand.
// In most cases, this serves as a LLT and register bank assert.
// Can change operands and insert copies, extends, truncs, and read-any-lanes.
// Anything more complicated requires LoweringMethod.
enum RegBankLLTMappingApplyID {
  InvalidMapping,
  None,
  IntrId,
  Imm,
  Vcc,

  // sgpr scalars, pointers, vectors and B-types
  Sgpr16,
  Sgpr32,
  Sgpr64,
  SgprP1,
  SgprP3,
  SgprP4,
  SgprP5,
  SgprV4S32,
  SgprB32,
  SgprB64,
  SgprB96,
  SgprB128,
  SgprB256,
  SgprB512,

  // vgpr scalars, pointers, vectors and B-types
  Vgpr32,
  Vgpr64,
  VgprP0,
  VgprP1,
  VgprP3,
  VgprP4,
  VgprP5,
  VgprB32,
  VgprB64,
  VgprB96,
  VgprB128,
  VgprB256,
  VgprB512,
  VgprV4S32,

  // Dst only modifiers: read-any-lane and truncs
  UniInVcc,
  UniInVgprS32,
  UniInVgprV4S32,
  UniInVgprB32,
  UniInVgprB64,
  UniInVgprB96,
  UniInVgprB128,
  UniInVgprB256,
  UniInVgprB512,

  Sgpr32Trunc,

  // Src only modifiers: waterfalls, extends
  Sgpr32AExt,
  Sgpr32AExtBoolInReg,
  Sgpr32SExt,
};

// Instruction needs to be replaced with sequence of instructions. Lowering was
// not done by legalizer since instructions is available in either sgpr or vgpr.
// For example S64 AND is available on sgpr, for that reason S64 AND is legal in
// context of Legalizer that only checks LLT. But S64 AND is not available on
// vgpr. Lower it to two S32 vgpr ANDs.
enum LoweringMethodID {
  DoNotLower,
  VccExtToSel,
  UniExtToSel,
  S_BFE,
  V_BFE,
  VgprToVccCopy,
  SplitTo32,
  Ext32To64,
  UniCstExt,
  SplitLoad,
  WidenLoad,
};

enum FastRulesTypes {
  NoFastRules,
  Standard,  // S16, S32, S64, V2S16
  StandardB, // B32, B64, B96, B128
  Vector,    // S32, V2S32, V3S32, V4S32
};

struct RegBankLLTMapping {
  SmallVector<RegBankLLTMappingApplyID, 2> DstOpMapping;
  SmallVector<RegBankLLTMappingApplyID, 4> SrcOpMapping;
  LoweringMethodID LoweringMethod;
  RegBankLLTMapping(
      std::initializer_list<RegBankLLTMappingApplyID> DstOpMappingList,
      std::initializer_list<RegBankLLTMappingApplyID> SrcOpMappingList,
      LoweringMethodID LoweringMethod = DoNotLower);
};

struct PredicateMapping {
  SmallVector<UniformityLLTOpPredicateID, 4> OpUniformityAndTypes;
  std::function<bool(const MachineInstr &)> TestFunc;
  PredicateMapping(
      std::initializer_list<UniformityLLTOpPredicateID> OpList,
      std::function<bool(const MachineInstr &)> TestFunc = nullptr);

  bool match(const MachineInstr &MI, const MachineUniformityInfo &MUI,
             const MachineRegisterInfo &MRI) const;
};

struct RegBankLegalizeRule {
  PredicateMapping Predicate;
  RegBankLLTMapping OperandMapping;
};

class SetOfRulesForOpcode {
  // "Slow Rules". More complex 'Rules[i].Predicate', check them one by one.
  SmallVector<RegBankLegalizeRule, 4> Rules;

  // "Fast Rules"
  // Instead of testing each 'Rules[i].Predicate' we do direct access to
  // RegBankLLTMapping using getFastPredicateSlot. For example if:
  // - FastTypes == Standard Uni[0] holds Mapping in case Op 0 is uniform S32
  // - FastTypes == Vector Div[3] holds Mapping in case Op 0 is divergent V4S32
  FastRulesTypes FastTypes = NoFastRules;
#define InvMapping RegBankLLTMapping({InvalidMapping}, {InvalidMapping})
  RegBankLLTMapping Uni[4] = {InvMapping, InvMapping, InvMapping, InvMapping};
  RegBankLLTMapping Div[4] = {InvMapping, InvMapping, InvMapping, InvMapping};

public:
  SetOfRulesForOpcode();
  SetOfRulesForOpcode(FastRulesTypes FastTypes);

  const RegBankLLTMapping &
  findMappingForMI(const MachineInstr &MI, const MachineRegisterInfo &MRI,
                   const MachineUniformityInfo &MUI) const;

  void addRule(RegBankLegalizeRule Rule);

  void addFastRuleDivergent(UniformityLLTOpPredicateID Ty,
                            RegBankLLTMapping RuleApplyIDs);
  void addFastRuleUniform(UniformityLLTOpPredicateID Ty,
                          RegBankLLTMapping RuleApplyIDs);

private:
  int getFastPredicateSlot(UniformityLLTOpPredicateID Ty) const;
};

// Essentially 'map<Opcode(or intrinsic_opcode), SetOfRulesForOpcode>' but a
// little more efficient.
class RegBankLegalizeRules {
  const GCNSubtarget *ST;
  MachineRegisterInfo *MRI;
  // Separate maps for G-opcodes and instrinsics since they are in different
  // enums. Multiple opcodes can share same set of rules.
  // RulesAlias = map<Opcode, KeyOpcode>
  // Rules = map<KeyOpcode, SetOfRulesForOpcode>
  SmallDenseMap<unsigned, unsigned, 256> GRulesAlias;
  SmallDenseMap<unsigned, SetOfRulesForOpcode, 128> GRules;
  SmallDenseMap<unsigned, unsigned, 128> IRulesAlias;
  SmallDenseMap<unsigned, SetOfRulesForOpcode, 64> IRules;
  class RuleSetInitializer {
    SetOfRulesForOpcode *RuleSet;

  public:
    // Used for clang-format line breaks and to force  writing all rules for
    // opcode in same place.
    template <class AliasMap, class RulesMap>
    RuleSetInitializer(std::initializer_list<unsigned> OpcList,
                       AliasMap &RulesAlias, RulesMap &Rules,
                       FastRulesTypes FastTypes = NoFastRules) {
      unsigned KeyOpcode = *OpcList.begin();
      for (unsigned Opc : OpcList) {
        [[maybe_unused]] auto [_, NewInput] =
            RulesAlias.try_emplace(Opc, KeyOpcode);
        assert(NewInput && "Can't redefine existing Rules");
      }

      auto [DenseMapIter, NewInput] = Rules.try_emplace(KeyOpcode, FastTypes);
      assert(NewInput && "Can't redefine existing Rules");

      RuleSet = &DenseMapIter->second;
    }

    RuleSetInitializer(const RuleSetInitializer &) = delete;
    RuleSetInitializer &operator=(const RuleSetInitializer &) = delete;
    RuleSetInitializer(RuleSetInitializer &&) = delete;
    RuleSetInitializer &operator=(RuleSetInitializer &&) = delete;
    ~RuleSetInitializer() = default;

    RuleSetInitializer &Div(UniformityLLTOpPredicateID Ty,
                            RegBankLLTMapping RuleApplyIDs,
                            bool STPred = true) {
      if (STPred)
        RuleSet->addFastRuleDivergent(Ty, RuleApplyIDs);
      return *this;
    }

    RuleSetInitializer &Uni(UniformityLLTOpPredicateID Ty,
                            RegBankLLTMapping RuleApplyIDs,
                            bool STPred = true) {
      if (STPred)
        RuleSet->addFastRuleUniform(Ty, RuleApplyIDs);
      return *this;
    }

    RuleSetInitializer &Any(RegBankLegalizeRule Init, bool STPred = true) {
      if (STPred)
        RuleSet->addRule(Init);
      return *this;
    }
  };

  RuleSetInitializer addRulesForGOpcs(std::initializer_list<unsigned> OpcList,
                                      FastRulesTypes FastTypes = NoFastRules);

  RuleSetInitializer addRulesForIOpcs(std::initializer_list<unsigned> OpcList,
                                      FastRulesTypes FastTypes = NoFastRules);

public:
  // Initialize rules for all opcodes.
  RegBankLegalizeRules(const GCNSubtarget &ST, MachineRegisterInfo &MRI);

  // In case we don't want to regenerate same rules, we can use already
  // generated rules but need to refresh references to objects that are
  // created for this run.
  void refreshRefs(const GCNSubtarget &_ST, MachineRegisterInfo &_MRI) {
    ST = &_ST;
    MRI = &_MRI;
  };

  const SetOfRulesForOpcode &getRulesForOpc(MachineInstr &MI) const;
};

} // end namespace AMDGPU
} // end namespace llvm

#endif
