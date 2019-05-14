//===- ScopDetectionDiagnostic.h - Diagnostic for ScopDetection -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small set of diagnostic helper classes to encapsulate any errors occurred
// during the detection of Scops.
//
// The ScopDetection defines a set of error classes (via Statistic variables)
// that groups a number of individual errors into a group, e.g. non-affinity
// related errors.
// On error we generate an object that carries enough additional information
// to diagnose the error and generate a helpful error message.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOPDETECTIONDIAGNOSTIC_H
#define POLLY_SCOPDETECTIONDIAGNOSTIC_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Instruction.h"
#include <cstddef>

using namespace llvm;

namespace llvm {

class AliasSet;
class BasicBlock;
class OptimizationRemarkEmitter;
class Region;
class SCEV;
} // namespace llvm

namespace polly {

/// Type to hold region delimiters (entry & exit block).
using BBPair = std::pair<BasicBlock *, BasicBlock *>;

/// Return the region delimiters (entry & exit block) of @p R.
BBPair getBBPairForRegion(const Region *R);

/// Set the begin and end source location for the region limited by @p P.
void getDebugLocations(const BBPair &P, DebugLoc &Begin, DebugLoc &End);

class RejectLog;

/// Emit optimization remarks about the rejected regions to the user.
///
/// This emits the content of the reject log as optimization remarks.
/// Remember to at least track failures (-polly-detect-track-failures).
/// @param P The region delimiters (entry & exit) we emit remarks for.
/// @param Log The error log containing all messages being emitted as remark.
void emitRejectionRemarks(const BBPair &P, const RejectLog &Log,
                          OptimizationRemarkEmitter &ORE);

// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
enum class RejectReasonKind {
  // CFG Category
  CFG,
  InvalidTerminator,
  IrreducibleRegion,
  UnreachableInExit,
  LastCFG,

  // Non-Affinity
  AffFunc,
  UndefCond,
  InvalidCond,
  UndefOperand,
  NonAffBranch,
  NoBasePtr,
  UndefBasePtr,
  VariantBasePtr,
  NonAffineAccess,
  DifferentElementSize,
  LastAffFunc,

  LoopBound,
  LoopHasNoExit,
  LoopHasMultipleExits,
  LoopOnlySomeLatches,

  FuncCall,
  NonSimpleMemoryAccess,

  Alias,

  // Other
  Other,
  IntToPtr,
  Alloca,
  UnknownInst,
  Entry,
  Unprofitable,
  LastOther
};

//===----------------------------------------------------------------------===//
/// Base class of all reject reasons found during Scop detection.
///
/// Subclasses of RejectReason should provide means to capture enough
/// diagnostic information to help clients figure out what and where something
/// went wrong in the Scop detection.
class RejectReason {
private:
  const RejectReasonKind Kind;

protected:
  static const DebugLoc Unknown;

public:
  RejectReason(RejectReasonKind K);

  virtual ~RejectReason() = default;

  RejectReasonKind getKind() const { return Kind; }

  /// Generate the remark name to identify this remark.
  ///
  /// @return A short string that identifies the error.
  virtual std::string getRemarkName() const = 0;

  /// Get the Basic Block containing this remark.
  ///
  /// @return The Basic Block containing this remark.
  virtual const Value *getRemarkBB() const = 0;

  /// Generate a reasonable diagnostic message describing this error.
  ///
  /// @return A debug message representing this error.
  virtual std::string getMessage() const = 0;

  /// Generate a message for the end-user describing this error.
  ///
  /// The message provided has to be suitable for the end-user. So it should
  /// not reference any LLVM internal data structures or terminology.
  /// Ideally, the message helps the end-user to increase the size of the
  /// regions amenable to Polly.
  ///
  /// @return A short message representing this error.
  virtual std::string getEndUserMessage() const { return "Unspecified error."; }

  /// Get the source location of this error.
  ///
  /// @return The debug location for this error.
  virtual const DebugLoc &getDebugLoc() const;
};

using RejectReasonPtr = std::shared_ptr<RejectReason>;

/// Stores all errors that occurred during the detection.
class RejectLog {
  Region *R;
  SmallVector<RejectReasonPtr, 1> ErrorReports;

public:
  explicit RejectLog(Region *R) : R(R) {}

  using iterator = SmallVector<RejectReasonPtr, 1>::const_iterator;

  iterator begin() const { return ErrorReports.begin(); }
  iterator end() const { return ErrorReports.end(); }
  size_t size() const { return ErrorReports.size(); }

  /// Returns true, if we store at least one error.
  ///
  /// @return true, if we store at least one error.
  bool hasErrors() const { return size() > 0; }

  void print(raw_ostream &OS, int level = 0) const;

  const Region *region() const { return R; }
  void report(RejectReasonPtr Reject) { ErrorReports.push_back(Reject); }
};

//===----------------------------------------------------------------------===//
/// Base class for CFG related reject reasons.
///
/// Scop candidates that violate structural restrictions can be grouped under
/// this reject reason class.
class ReportCFG : public RejectReason {
public:
  ReportCFG(const RejectReasonKind K);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures bad terminator within a Scop candidate.
class ReportInvalidTerminator : public ReportCFG {
  BasicBlock *BB;

public:
  ReportInvalidTerminator(BasicBlock *BB)
      : ReportCFG(RejectReasonKind::InvalidTerminator), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures irreducible regions in CFG.
class ReportIrreducibleRegion : public ReportCFG {
  Region *R;
  DebugLoc DbgLoc;

public:
  ReportIrreducibleRegion(Region *R, DebugLoc DbgLoc)
      : ReportCFG(RejectReasonKind::IrreducibleRegion), R(R), DbgLoc(DbgLoc) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures regions with an unreachable in the exit block.
class ReportUnreachableInExit : public ReportCFG {
  BasicBlock *BB;
  DebugLoc DbgLoc;

public:
  ReportUnreachableInExit(BasicBlock *BB, DebugLoc DbgLoc)
      : ReportCFG(RejectReasonKind::UnreachableInExit), BB(BB), DbgLoc(DbgLoc) {
  }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Base class for non-affine reject reasons.
///
/// Scop candidates that violate restrictions to affinity are reported under
/// this class.
class ReportAffFunc : public RejectReason {
protected:
  // The instruction that caused non-affinity to occur.
  const Instruction *Inst;

public:
  ReportAffFunc(const RejectReasonKind K, const Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  const DebugLoc &getDebugLoc() const override { return Inst->getDebugLoc(); }
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a condition that is based on an 'undef' value.
class ReportUndefCond : public ReportAffFunc {
  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportUndefCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(RejectReasonKind::UndefCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an invalid condition
///
/// Conditions have to be either constants or icmp instructions.
class ReportInvalidCond : public ReportAffFunc {
  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportInvalidCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(RejectReasonKind::InvalidCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an undefined operand.
class ReportUndefOperand : public ReportAffFunc {
  // The BasicBlock we found the undefined operand in.
  BasicBlock *BB;

public:
  ReportUndefOperand(BasicBlock *BB, const Instruction *Inst)
      : ReportAffFunc(RejectReasonKind::UndefOperand, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a non-affine branch.
class ReportNonAffBranch : public ReportAffFunc {
  // The BasicBlock we found the non-affine branch in.
  BasicBlock *BB;

  /// LHS & RHS of the failed condition.
  //@{
  const SCEV *LHS;
  const SCEV *RHS;
  //@}

public:
  ReportNonAffBranch(BasicBlock *BB, const SCEV *LHS, const SCEV *RHS,
                     const Instruction *Inst)
      : ReportAffFunc(RejectReasonKind::NonAffBranch, Inst), BB(BB), LHS(LHS),
        RHS(RHS) {}

  const SCEV *lhs() { return LHS; }
  const SCEV *rhs() { return RHS; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a missing base pointer.
class ReportNoBasePtr : public ReportAffFunc {
public:
  ReportNoBasePtr(const Instruction *Inst)
      : ReportAffFunc(RejectReasonKind::NoBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an undefined base pointer.
class ReportUndefBasePtr : public ReportAffFunc {
public:
  ReportUndefBasePtr(const Instruction *Inst)
      : ReportAffFunc(RejectReasonKind::UndefBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a base pointer that is not invariant in the region.
class ReportVariantBasePtr : public ReportAffFunc {
  // The variant base pointer.
  Value *BaseValue;

public:
  ReportVariantBasePtr(Value *BaseValue, const Instruction *Inst)
      : ReportAffFunc(RejectReasonKind::VariantBasePtr, Inst),
        BaseValue(BaseValue) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a non-affine access function.
class ReportNonAffineAccess : public ReportAffFunc {
  // The non-affine access function.
  const SCEV *AccessFunction;

  // The base pointer of the memory access.
  const Value *BaseValue;

public:
  ReportNonAffineAccess(const SCEV *AccessFunction, const Instruction *Inst,
                        const Value *V)
      : ReportAffFunc(RejectReasonKind::NonAffineAccess, Inst),
        AccessFunction(AccessFunction), BaseValue(V) {}

  const SCEV *get() { return AccessFunction; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Report array accesses with differing element size.
class ReportDifferentArrayElementSize : public ReportAffFunc {
  // The base pointer of the memory access.
  const Value *BaseValue;

public:
  ReportDifferentArrayElementSize(const Instruction *Inst, const Value *V)
      : ReportAffFunc(RejectReasonKind::DifferentElementSize, Inst),
        BaseValue(V) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non affine loop bounds.
class ReportLoopBound : public RejectReason {
  // The offending loop.
  Loop *L;

  // The non-affine loop bound.
  const SCEV *LoopCount;

  // A copy of the offending loop's debug location.
  const DebugLoc Loc;

public:
  ReportLoopBound(Loop *L, const SCEV *LoopCount);

  const SCEV *loopCount() { return LoopCount; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors when loop has no exit.
class ReportLoopHasNoExit : public RejectReason {
  /// The loop that has no exit.
  Loop *L;

  const DebugLoc Loc;

public:
  ReportLoopHasNoExit(Loop *L)
      : RejectReason(RejectReasonKind::LoopHasNoExit), L(L),
        Loc(L->getStartLoc()) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors when a loop has multiple exists.
class ReportLoopHasMultipleExits : public RejectReason {
  /// The loop that has multiple exits.
  Loop *L;

  const DebugLoc Loc;

public:
  ReportLoopHasMultipleExits(Loop *L)
      : RejectReason(RejectReasonKind::LoopHasMultipleExits), L(L),
        Loc(L->getStartLoc()) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors when not all loop latches are part of the scop.
class ReportLoopOnlySomeLatches : public RejectReason {
  /// The loop for which not all loop latches are part of the scop.
  Loop *L;

  const DebugLoc Loc;

public:
  ReportLoopOnlySomeLatches(Loop *L)
      : RejectReason(RejectReasonKind::LoopOnlySomeLatches), L(L),
        Loc(L->getStartLoc()) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non-side-effect-known function calls.
class ReportFuncCall : public RejectReason {
  // The offending call instruction.
  Instruction *Inst;

public:
  ReportFuncCall(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with aliasing.
class ReportAlias : public RejectReason {
public:
  using PointerSnapshotTy = std::vector<const Value *>;

private:
  /// Format an invalid alias set.
  ///
  //  @param Prefix A prefix string to put before the list of aliasing pointers.
  //  @param Suffix A suffix string to put after the list of aliasing pointers.
  std::string formatInvalidAlias(std::string Prefix = "",
                                 std::string Suffix = "") const;

  Instruction *Inst;

  // A snapshot of the llvm values that took part in the aliasing error.
  mutable PointerSnapshotTy Pointers;

public:
  ReportAlias(Instruction *Inst, AliasSet &AS);

  const PointerSnapshotTy &getPointers() const { return Pointers; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Base class for otherwise ungrouped reject reasons.
class ReportOther : public RejectReason {
public:
  ReportOther(const RejectReasonKind K);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with bad IntToPtr instructions.
class ReportIntToPtr : public ReportOther {
  // The offending base value.
  Instruction *BaseValue;

public:
  ReportIntToPtr(Instruction *BaseValue);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with alloca instructions.
class ReportAlloca : public ReportOther {
  Instruction *Inst;

public:
  ReportAlloca(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with unknown instructions.
class ReportUnknownInst : public ReportOther {
  Instruction *Inst;

public:
  ReportUnknownInst(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with regions containing the function entry block.
class ReportEntry : public ReportOther {
  BasicBlock *BB;

public:
  ReportEntry(BasicBlock *BB);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Report regions that seem not profitable to be optimized.
class ReportUnprofitable : public ReportOther {
  Region *R;

public:
  ReportUnprofitable(Region *R);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  std::string getEndUserMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non-simple memory accesses.
class ReportNonSimpleMemoryAccess : public ReportOther {
  // The offending call instruction.
  Instruction *Inst;

public:
  ReportNonSimpleMemoryAccess(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  std::string getRemarkName() const override;
  const Value *getRemarkBB() const override;
  std::string getMessage() const override;
  const DebugLoc &getDebugLoc() const override;
  std::string getEndUserMessage() const override;
  //@}
};
} // namespace polly

#endif // POLLY_SCOPDETECTIONDIAGNOSTIC_H
