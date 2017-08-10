//===------ VirtualInstruction.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tools for determining which instructions are within a statement and the
// nature of their operands.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_VIRTUALINSTRUCTION_H
#define POLLY_SUPPORT_VIRTUALINSTRUCTION_H

#include "polly/ScopInfo.h"

namespace polly {

/// Determine the nature of a value's use within a statement.
///
/// These are not always representable by llvm::Use. For instance, scalar write
/// MemoryAccesses do use a value, but are not associated with an instruction's
/// argument.
///
/// Despite its name it is not tied to virtual instructions (although it works
/// fine with them), but to promote consistent handling of values used in
/// statements.
class VirtualUse {
public:
  /// The different types of uses. Handling usually differentiates a lot between
  /// these; one can use a switch to handle each case (and get warned by the
  /// compiler if one is not handled).
  enum UseKind {
    // An llvm::Constant.
    Constant,

    // An llvm::BasicBlock.
    Block,

    // A value that can be generated using ScopExpander.
    Synthesizable,

    // A load that always reads the same value throughout the SCoP (address and
    // the value located there a SCoP-invariant) and has been hoisted in front
    // of the SCoP.
    Hoisted,

    // Definition before the SCoP and not synthesizable. Can be an instruction
    // outside the SCoP, a function argument or a global value. Whether there is
    // a scalar MemoryAccess in this statement for reading it depends on the
    // -polly-analyze-read-only-scalars switch.
    ReadOnly,

    // A definition within the same statement. No MemoryAccess between
    // definition and use are necessary.
    Intra,

    // Definition in another statement. There is a scalar MemoryAccess that
    // makes it available in this statement.
    Inter
  };

private:
  /// The statement where a value is used.
  ScopStmt *User;

  /// The value that is used.
  Value *Val;

  /// The type of value use.
  UseKind Kind;

  /// The value represented as llvm::SCEV expression.
  const SCEV *ScevExpr;

  /// If this is an inter-statement (or read-only) use, contains the
  /// MemoryAccess that makes the value available in this statement. In case of
  /// intra-statement uses, can contain a MemoryKind::Array access. In all other
  /// cases, it is a nullptr.
  MemoryAccess *InputMA;

  VirtualUse(ScopStmt *User, Value *Val, UseKind Kind, const SCEV *ScevExpr,
             MemoryAccess *InputMA)
      : User(User), Val(Val), Kind(Kind), ScevExpr(ScevExpr), InputMA(InputMA) {
  }

public:
  /// Get a VirtualUse for an llvm::Use.
  ///
  /// @param S       The Scop object.
  /// @param U       The llvm::Use the get information for.
  /// @param LI      The LoopInfo analysis. Needed to determine whether the
  ///                value is synthesizable.
  /// @param Virtual Whether to ignore existing MemoryAcccess.
  ///
  /// @return The VirtualUse representing the same use as @p U.
  static VirtualUse create(Scop *S, Use &U, LoopInfo *LI, bool Virtual);

  /// Get a VirtualUse for any kind of use of a value within a statement.
  ///
  /// @param S         The Scop object.
  /// @param UserStmt  The statement in which @p Val is used. Can be nullptr, in
  ///                  which case it assumed that the statement has been
  ///                  removed, which is only possible if no instruction in it
  ///                  had side-effects or computes a value used by another
  ///                  statement.
  /// @param UserScope Loop scope in which the value is used. Needed to
  ///                  determine whether the value is synthesizable.
  /// @param Val       The value being used.
  /// @param Virtual   Whether to use (and prioritize over instruction location)
  ///                  information about MemoryAccesses.
  ///
  /// @return A VirtualUse object that gives information about @p Val's use in
  ///         @p UserStmt.
  static VirtualUse create(Scop *S, ScopStmt *UserStmt, Loop *UserScope,
                           Value *Val, bool Virtual);

  static VirtualUse create(ScopStmt *UserStmt, Loop *UserScope, Value *Val,
                           bool Virtual) {
    return create(UserStmt->getParent(), UserStmt, UserScope, Val, Virtual);
  }

  bool isConstant() const { return Kind == Constant; }
  bool isBlock() const { return Kind == Block; }
  bool isSynthesizable() const { return Kind == Synthesizable; }
  bool isHoisted() const { return Kind == Hoisted; }
  bool isReadOnly() const { return Kind == ReadOnly; }
  bool isIntra() const { return Kind == Intra; }
  bool isInter() const { return Kind == Inter; }

  /// Return user statement.
  ScopStmt *getUser() const { return User; }

  /// Return the used value.
  llvm::Value *getValue() const { return Val; }

  /// Return the type of use.
  UseKind getKind() const { return Kind; }

  /// Return the ScalarEvolution representation of @p Val.
  const SCEV *getScevExpr() const { return ScevExpr; }

  /// Return the MemoryAccess that makes the value available in this statement,
  /// if any.
  MemoryAccess *getMemoryAccess() const { return InputMA; }

  /// Print a description of this object.
  ///
  /// @param OS           Stream to print to.
  /// @param Reproducible If true, ensures that the output is stable between
  /// runs and is suitable to check in regression tests. This excludes printing
  /// e.g. pointer values.
  ///                     If false, the output should not be used for regression
  ///                     tests, but may contain more information useful in
  ///                     debugger sessions.
  void print(raw_ostream &OS, bool Reproducible = true) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const;
#endif
};

} // namespace polly

#endif /* POLLY_SUPPORT_VIRTUALINSTRUCTION_H */
