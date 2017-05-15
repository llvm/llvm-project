//==-- llvm/CodeGen/GlobalISel/LegalizerInfo.h -------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// Interface for Targets to specify which operations they can successfully
/// select and how the others should be expanded most efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_MACHINELEGALIZER_H
#define LLVM_CODEGEN_GLOBALISEL_MACHINELEGALIZER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/Target/TargetOpcodes.h"

#include <cstdint>
#include <functional>

namespace llvm {
class LLVMContext;
class MachineInstr;
class MachineIRBuilder;
class MachineRegisterInfo;
class Type;
class VectorType;

/// Legalization is decided based on an instruction's opcode, which type slot
/// we're considering, and what the existing type is. These aspects are gathered
/// together for convenience in the InstrAspect class.
struct InstrAspect {
  unsigned Opcode;
  unsigned Idx;
  LLT Type;

  InstrAspect(unsigned Opcode, LLT Type) : Opcode(Opcode), Idx(0), Type(Type) {}
  InstrAspect(unsigned Opcode, unsigned Idx, LLT Type)
      : Opcode(Opcode), Idx(Idx), Type(Type) {}

  bool operator==(const InstrAspect &RHS) const {
    return Opcode == RHS.Opcode && Idx == RHS.Idx && Type == RHS.Type;
  }
};

class LegalizerInfo {
public:
  enum LegalizeAction : std::uint8_t {
    /// The operation is expected to be selectable directly by the target, and
    /// no transformation is necessary.
    Legal,

    /// The operation should be synthesized from multiple instructions acting on
    /// a narrower scalar base-type. For example a 64-bit add might be
    /// implemented in terms of 32-bit add-with-carry.
    NarrowScalar,

    /// The operation should be implemented in terms of a wider scalar
    /// base-type. For example a <2 x s8> add could be implemented as a <2
    /// x s32> add (ignoring the high bits).
    WidenScalar,

    /// The (vector) operation should be implemented by splitting it into
    /// sub-vectors where the operation is legal. For example a <8 x s64> add
    /// might be implemented as 4 separate <2 x s64> adds.
    FewerElements,

    /// The (vector) operation should be implemented by widening the input
    /// vector and ignoring the lanes added by doing so. For example <2 x i8> is
    /// rarely legal, but you might perform an <8 x i8> and then only look at
    /// the first two results.
    MoreElements,

    /// The operation itself must be expressed in terms of simpler actions on
    /// this target. E.g. a SREM replaced by an SDIV and subtraction.
    Lower,

    /// The operation should be implemented as a call to some kind of runtime
    /// support library. For example this usually happens on machines that don't
    /// support floating-point operations natively.
    Libcall,

    /// The target wants to do something special with this combination of
    /// operand and type. A callback will be issued when it is needed.
    Custom,

    /// This operation is completely unsupported on the target. A programming
    /// error has occurred.
    Unsupported,

    /// Sentinel value for when no action was found in the specified table.
    NotFound,
  };

  LegalizerInfo();
  virtual ~LegalizerInfo() = default;

  /// Compute any ancillary tables needed to quickly decide how an operation
  /// should be handled. This must be called after all "set*Action"methods but
  /// before any query is made or incorrect results may be returned.
  void computeTables();

  /// More friendly way to set an action for common types that have an LLT
  /// representation.
  void setAction(const InstrAspect &Aspect, LegalizeAction Action) {
    TablesInitialized = false;
    unsigned Opcode = Aspect.Opcode - FirstOp;
    if (Actions[Opcode].size() <= Aspect.Idx)
      Actions[Opcode].resize(Aspect.Idx + 1);
    Actions[Aspect.Opcode - FirstOp][Aspect.Idx][Aspect.Type] = Action;
  }

  /// If an operation on a given vector type (say <M x iN>) isn't explicitly
  /// specified, we proceed in 2 stages. First we legalize the underlying scalar
  /// (so that there's at least one legal vector with that scalar), then we
  /// adjust the number of elements in the vector so that it is legal. The
  /// desired action in the first step is controlled by this function.
  void setScalarInVectorAction(unsigned Opcode, LLT ScalarTy,
                               LegalizeAction Action) {
    assert(!ScalarTy.isVector());
    ScalarInVectorActions[std::make_pair(Opcode, ScalarTy)] = Action;
  }


  /// Determine what action should be taken to legalize the given generic
  /// instruction opcode, type-index and type. Requires computeTables to have
  /// been called.
  ///
  /// \returns a pair consisting of the kind of legalization that should be
  /// performed and the destination type.
  std::pair<LegalizeAction, LLT> getAction(const InstrAspect &Aspect) const;

  /// Determine what action should be taken to legalize the given generic
  /// instruction.
  ///
  /// \returns a tuple consisting of the LegalizeAction that should be
  /// performed, the type-index it should be performed on and the destination
  /// type.
  std::tuple<LegalizeAction, unsigned, LLT>
  getAction(const MachineInstr &MI, const MachineRegisterInfo &MRI) const;

  /// Iterate the given function (typically something like doubling the width)
  /// on Ty until we find a legal type for this operation.
  Optional<LLT> findLegalType(const InstrAspect &Aspect,
                    function_ref<LLT(LLT)> NextType) const {
    LegalizeAction Action;
    const TypeMap &Map = Actions[Aspect.Opcode - FirstOp][Aspect.Idx];
    LLT Ty = Aspect.Type;
    do {
      Ty = NextType(Ty);
      auto ActionIt = Map.find(Ty);
      if (ActionIt == Map.end()) {
        auto DefaultIt = DefaultActions.find(Aspect.Opcode);
        if (DefaultIt == DefaultActions.end())
          return None;
        Action = DefaultIt->second;
      }
      else
        Action = ActionIt->second;
    } while(Action != Legal);
    return Ty;
  }

  /// Find what type it's actually OK to perform the given operation on, given
  /// the general approach we've decided to take.
  Optional<LLT> findLegalType(const InstrAspect &Aspect, LegalizeAction Action) const;

  std::pair<LegalizeAction, LLT> findLegalAction(const InstrAspect &Aspect,
                                                 LegalizeAction Action) const {
    auto LegalType = findLegalType(Aspect, Action);
    if (!LegalType)
      return std::make_pair(LegalizeAction::Unsupported, LLT());
    return std::make_pair(Action, *LegalType);
  }

  /// Find the specified \p Aspect in the primary (explicitly set) Actions
  /// table. Returns either the action the target requested or NotFound if there
  /// was no setAction call.
  LegalizeAction findInActions(const InstrAspect &Aspect) const {
    if (Aspect.Opcode < FirstOp || Aspect.Opcode > LastOp)
      return NotFound;
    if (Aspect.Idx >= Actions[Aspect.Opcode - FirstOp].size())
      return NotFound;
    const TypeMap &Map = Actions[Aspect.Opcode - FirstOp][Aspect.Idx];
    auto ActionIt =  Map.find(Aspect.Type);
    if (ActionIt == Map.end())
      return NotFound;

    return ActionIt->second;
  }

  bool isLegal(const MachineInstr &MI, const MachineRegisterInfo &MRI) const;

  virtual bool legalizeCustom(MachineInstr &MI,
                              MachineRegisterInfo &MRI,
                              MachineIRBuilder &MIRBuilder) const;

private:
  static const int FirstOp = TargetOpcode::PRE_ISEL_GENERIC_OPCODE_START;
  static const int LastOp = TargetOpcode::PRE_ISEL_GENERIC_OPCODE_END;

  typedef DenseMap<LLT, LegalizeAction> TypeMap;
  typedef DenseMap<std::pair<unsigned, LLT>, LegalizeAction> SIVActionMap;

  SmallVector<TypeMap, 1> Actions[LastOp - FirstOp + 1];
  SIVActionMap ScalarInVectorActions;
  DenseMap<std::pair<unsigned, LLT>, uint16_t> MaxLegalVectorElts;
  DenseMap<unsigned, LegalizeAction> DefaultActions;

  bool TablesInitialized;
};


} // End namespace llvm.

#endif
