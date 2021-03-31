//===------ VirtualInstruction.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
using llvm::User;

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
  static VirtualUse create(Scop *S, const Use &U, LoopInfo *LI, bool Virtual);

  /// Get a VirtualUse for uses within statements.
  ///
  /// It is assumed that the user is not a PHINode. Such uses are always
  /// VirtualUse::Inter unless in a regions statement.
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
  ///                     runs and is suitable to check in regression tests.
  ///                     This excludes printing e.g. pointer values. If false,
  ///                     the output should not be used for regression tests,
  ///                     but may contain more information useful in debugger
  ///                     sessions.
  void print(raw_ostream &OS, bool Reproducible = true) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const;
#endif
};

/// An iterator for virtual operands.
class VirtualOperandIterator
    : public std::iterator<std::forward_iterator_tag, VirtualUse> {
  friend class VirtualInstruction;
  friend class VirtualUse;

  using super = std::iterator<std::forward_iterator_tag, VirtualUse>;
  using Self = VirtualOperandIterator;

  ScopStmt *User;
  User::op_iterator U;

  VirtualOperandIterator(ScopStmt *User, User::op_iterator U)
      : User(User), U(U) {}

public:
  using pointer = typename super::pointer;
  using reference = typename super::reference;

  inline bool operator==(const Self &that) const {
    assert(this->User == that.User);
    return this->U == that.U;
  }

  inline bool operator!=(const Self &that) const {
    assert(this->User == that.User);
    return this->U != that.U;
  }

  VirtualUse operator*() const {
    return VirtualUse::create(User, User->getSurroundingLoop(), U->get(), true);
  }

  Use *operator->() const { return U; }

  Self &operator++() {
    U++;
    return *this;
  }

  Self operator++(int) {
    Self tmp = *this;
    ++*this;
    return tmp;
  }
};

/// This class represents a "virtual instruction", an instruction in a ScopStmt,
/// effectively a ScopStmt/Instruction-pair.
///
/// An instructions can be moved between statements (e.g. to avoid a scalar
/// dependency) and even can be contained in multiple statements (for instance,
/// to recompute a value instead of transferring it), hence 'virtual'. This
/// class is required to represent such instructions that are not in their
/// 'physical' location anymore.
///
/// A statement can currently not contain the same instructions multiple times
/// (that is, from different loop iterations). Therefore, a
/// ScopStmt/Instruction-pair uniquely identifies a virtual instructions.
/// ScopStmt::getInstruction() can contain the same instruction multiple times,
/// but they necessarily compute the same value.
class VirtualInstruction {
  friend class VirtualOperandIterator;
  friend struct llvm::DenseMapInfo<VirtualInstruction>;

private:
  /// The statement this virtual instruction is in.
  ScopStmt *Stmt = nullptr;

  /// The instruction of a statement.
  Instruction *Inst = nullptr;

public:
  VirtualInstruction() {}

  /// Create a new virtual instruction of an instruction @p Inst in @p Stmt.
  VirtualInstruction(ScopStmt *Stmt, Instruction *Inst)
      : Stmt(Stmt), Inst(Inst) {
    assert(Stmt && Inst);
  }

  VirtualOperandIterator operand_begin() const {
    return VirtualOperandIterator(Stmt, Inst->op_begin());
  }

  VirtualOperandIterator operand_end() const {
    return VirtualOperandIterator(Stmt, Inst->op_end());
  }

  /// Returns a list of virtual operands.
  ///
  /// Virtual operands, like virtual instructions, need to encode the ScopStmt
  /// they are in.
  llvm::iterator_range<VirtualOperandIterator> operands() const {
    return {operand_begin(), operand_end()};
  }

  /// Return the SCoP everything is contained in.
  Scop *getScop() const { return Stmt->getParent(); }

  /// Return the ScopStmt this virtual instruction is in.
  ScopStmt *getStmt() const { return Stmt; }

  /// Return the instruction in the statement.
  Instruction *getInstruction() const { return Inst; }

  /// Print a description of this object.
  ///
  /// @param OS           Stream to print to.
  /// @param Reproducible If true, ensures that the output is stable between
  ///                     runs and is suitable for checks in regression tests.
  ///                     This excludes printing e.g., pointer values. If false,
  ///                     the output should not be used for regression tests,
  ///                     but may contain more information useful in debugger
  ///                     sessions.
  void print(raw_ostream &OS, bool Reproducible = true) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump() const;
#endif
};

static inline bool operator==(VirtualInstruction LHS, VirtualInstruction RHS) {
  return LHS.getStmt() == RHS.getStmt() &&
         LHS.getInstruction() == RHS.getInstruction();
}

/// Find all reachable instructions and accesses.
///
/// @param S              The SCoP to find everything reachable in.
/// @param LI             LoopInfo required for analysis.
/// @param UsedInsts[out] Receives all reachable instructions.
/// @param UsedAccs[out]  Receives all reachable accesses.
/// @param OnlyLocal      If non-nullptr, activates local mode: The SCoP is
///                       assumed to consist only of this statement and is
///                       conservatively correct. Does not require walking the
///                       whole SCoP.
void markReachable(Scop *S, LoopInfo *LI,
                   DenseSet<VirtualInstruction> &UsedInsts,
                   DenseSet<MemoryAccess *> &UsedAccs,
                   ScopStmt *OnlyLocal = nullptr);
} // namespace polly

namespace llvm {
/// Support VirtualInstructions in llvm::DenseMaps.
template <> struct DenseMapInfo<polly::VirtualInstruction> {
public:
  static bool isEqual(polly::VirtualInstruction LHS,
                      polly::VirtualInstruction RHS) {
    return DenseMapInfo<polly::ScopStmt *>::isEqual(LHS.getStmt(),
                                                    RHS.getStmt()) &&
           DenseMapInfo<Instruction *>::isEqual(LHS.getInstruction(),
                                                RHS.getInstruction());
  }

  static polly::VirtualInstruction getTombstoneKey() {
    polly::VirtualInstruction TombstoneKey;
    TombstoneKey.Stmt = DenseMapInfo<polly::ScopStmt *>::getTombstoneKey();
    TombstoneKey.Inst = DenseMapInfo<Instruction *>::getTombstoneKey();
    return TombstoneKey;
  }

  static polly::VirtualInstruction getEmptyKey() {
    polly::VirtualInstruction EmptyKey;
    EmptyKey.Stmt = DenseMapInfo<polly::ScopStmt *>::getEmptyKey();
    EmptyKey.Inst = DenseMapInfo<Instruction *>::getEmptyKey();
    return EmptyKey;
  }

  static unsigned getHashValue(polly::VirtualInstruction Val) {
    return DenseMapInfo<std::pair<polly::ScopStmt *, Instruction *>>::
        getHashValue(std::make_pair(Val.getStmt(), Val.getInstruction()));
  }
};
} // namespace llvm

#endif /* POLLY_SUPPORT_VIRTUALINSTRUCTION_H */
