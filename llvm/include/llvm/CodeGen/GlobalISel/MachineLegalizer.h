//==-- llvm/CodeGen/GlobalISel/MachineLegalizer.h ----------------*- C++ -*-==//
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

#include <cstdint>
#include <functional>

namespace llvm {
class LLVMContext;
class MachineInstr;
class Type;
class VectorType;

class MachineLegalizer {
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
  };

  MachineLegalizer();

  /// Compute any ancillary tables needed to quickly decide how an operation
  /// should be handled. This must be called after all "set*Action"methods but
  /// before any query is made or incorrect results may be returned.
  void computeTables();

  /// More friendly way to set an action for common types that have an LLT
  /// representation.
  void setAction(unsigned Opcode, LLT Ty, LegalizeAction Action) {
    TablesInitialized = false;
    Actions[std::make_pair(Opcode, Ty)] = Action;
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
  /// instruction and type. Requires computeTables to have been called.
  ///
  /// \returns a pair consisting of the kind of legalization that should be
  /// performed and the destination type.
  std::pair<LegalizeAction, LLT> getAction(unsigned Opcode, LLT) const;
  std::pair<LegalizeAction, LLT> getAction(const MachineInstr &MI) const;

  /// Iterate the given function (typically something like doubling the width)
  /// on Ty until we find a legal type for this operation.
  LLT findLegalType(unsigned Opcode, LLT Ty,
                      std::function<LLT(LLT)> NextType) const {
    LegalizeAction Action;
    do {
      Ty = NextType(Ty);
      auto ActionIt = Actions.find(std::make_pair(Opcode, Ty));
      if (ActionIt == Actions.end())
        Action = DefaultActions.find(Opcode)->second;
      else
        Action = ActionIt->second;
    } while(Action != Legal);
    return Ty;
  }

  /// Find what type it's actually OK to perform the given operation on, given
  /// the general approach we've decided to take.
  LLT findLegalType(unsigned Opcode, LLT Ty, LegalizeAction Action) const;

  std::pair<LegalizeAction, LLT> findLegalAction(unsigned Opcode, LLT Ty,
                                                 LegalizeAction Action) const {
    return std::make_pair(Action, findLegalType(Opcode, Ty, Action));
  }

  bool isLegal(const MachineInstr &MI) const;

private:
  typedef DenseMap<std::pair<unsigned, LLT>, LegalizeAction> ActionMap;

  ActionMap Actions;
  ActionMap ScalarInVectorActions;
  DenseMap<std::pair<unsigned, LLT>, uint16_t> MaxLegalVectorElts;
  DenseMap<unsigned, LegalizeAction> DefaultActions;

  bool TablesInitialized;
};

} // End namespace llvm.

#endif
