//=== ScopDetectionDiagnostic.h -- Diagnostic for ScopDetection -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#ifndef POLLY_SCOP_DETECTION_DIAGNOSTIC_H
#define POLLY_SCOP_DETECTION_DIAGNOSTIC_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <string>

using namespace llvm;

namespace llvm {
class SCEV;
class BasicBlock;
class Value;
class Region;
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
void emitRejectionRemarks(const BBPair &P, const RejectLog &Log);

// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
enum RejectReasonKind {
  // CFG Category
  rrkCFG,
  rrkInvalidTerminator,
  rrkCondition,
  rrkLastCFG,
  rrkIrreducibleRegion,

  // Non-Affinity
  rrkAffFunc,
  rrkUndefCond,
  rrkInvalidCond,
  rrkUndefOperand,
  rrkNonAffBranch,
  rrkNoBasePtr,
  rrkUndefBasePtr,
  rrkVariantBasePtr,
  rrkNonAffineAccess,
  rrkDifferentElementSize,
  rrkLastAffFunc,

  rrkLoopBound,
  rrkLoopHasNoExit,

  rrkFuncCall,
  rrkNonSimpleMemoryAccess,

  rrkAlias,

  // Other
  rrkOther,
  rrkIntToPtr,
  rrkAlloca,
  rrkUnknownInst,
  rrkEntry,
  rrkUnprofitable,
  rrkLastOther
};

//===----------------------------------------------------------------------===//
/// Base class of all reject reasons found during Scop detection.
///
/// Subclasses of RejectReason should provide means to capture enough
/// diagnostic information to help clients figure out what and where something
/// went wrong in the Scop detection.
class RejectReason {
  //===--------------------------------------------------------------------===//
private:
  const RejectReasonKind Kind;

protected:
  static const DebugLoc Unknown;

public:
  RejectReasonKind getKind() const { return Kind; }

  RejectReason(RejectReasonKind K) : Kind(K) {}

  virtual ~RejectReason() {}

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
  virtual const llvm::DebugLoc &getDebugLoc() const;
};

typedef std::shared_ptr<RejectReason> RejectReasonPtr;

/// Stores all errors that ocurred during the detection.
class RejectLog {
  Region *R;
  llvm::SmallVector<RejectReasonPtr, 1> ErrorReports;

public:
  explicit RejectLog(Region *R) : R(R) {}

  typedef llvm::SmallVector<RejectReasonPtr, 1>::const_iterator iterator;

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
  //===--------------------------------------------------------------------===//
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
      : ReportCFG(rrkInvalidTerminator), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures irreducible regions in CFG.
class ReportIrreducibleRegion : public ReportCFG {
  Region *R;
  DebugLoc DbgLoc;

public:
  ReportIrreducibleRegion(Region *R, DebugLoc DbgLoc)
      : ReportCFG(rrkIrreducibleRegion), R(R), DbgLoc(DbgLoc) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Base class for non-affine reject reasons.
///
/// Scop candidates that violate restrictions to affinity are reported under
/// this class.
class ReportAffFunc : public RejectReason {
  //===--------------------------------------------------------------------===//

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
  virtual const DebugLoc &getDebugLoc() const override {
    return Inst->getDebugLoc();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a condition that is based on an 'undef' value.
class ReportUndefCond : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportUndefCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(rrkUndefCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an invalid condition
///
/// Conditions have to be either constants or icmp instructions.
class ReportInvalidCond : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportInvalidCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(rrkInvalidCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an undefined operand.
class ReportUndefOperand : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the undefined operand in.
  BasicBlock *BB;

public:
  ReportUndefOperand(BasicBlock *BB, const Instruction *Inst)
      : ReportAffFunc(rrkUndefOperand, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a non-affine branch.
class ReportNonAffBranch : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

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
      : ReportAffFunc(rrkNonAffBranch, Inst), BB(BB), LHS(LHS), RHS(RHS) {}

  const SCEV *lhs() { return LHS; }
  const SCEV *rhs() { return RHS; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a missing base pointer.
class ReportNoBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  ReportNoBasePtr(const Instruction *Inst)
      : ReportAffFunc(rrkNoBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures an undefined base pointer.
class ReportUndefBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  ReportUndefBasePtr(const Instruction *Inst)
      : ReportAffFunc(rrkUndefBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a base pointer that is not invariant in the region.
class ReportVariantBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The variant base pointer.
  Value *BaseValue;

public:
  ReportVariantBasePtr(Value *BaseValue, const Instruction *Inst)
      : ReportAffFunc(rrkVariantBasePtr, Inst), BaseValue(BaseValue) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures a non-affine access function.
class ReportNonAffineAccess : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The non-affine access function.
  const SCEV *AccessFunction;

  // The base pointer of the memory access.
  const Value *BaseValue;

public:
  ReportNonAffineAccess(const SCEV *AccessFunction, const Instruction *Inst,
                        const Value *V)
      : ReportAffFunc(rrkNonAffineAccess, Inst), AccessFunction(AccessFunction),
        BaseValue(V) {}

  const SCEV *get() { return AccessFunction; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Report array accesses with differing element size.
class ReportDifferentArrayElementSize : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The base pointer of the memory access.
  const Value *BaseValue;

public:
  ReportDifferentArrayElementSize(const Instruction *Inst, const Value *V)
      : ReportAffFunc(rrkDifferentElementSize, Inst), BaseValue(V) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non affine loop bounds.
class ReportLoopBound : public RejectReason {
  //===--------------------------------------------------------------------===//

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
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors when loop has no exit.
class ReportLoopHasNoExit : public RejectReason {
  //===--------------------------------------------------------------------===//

  /// The loop that has no exit.
  Loop *L;

  const DebugLoc Loc;

public:
  ReportLoopHasNoExit(Loop *L)
      : RejectReason(rrkLoopHasNoExit), L(L), Loc(L->getStartLoc()) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non-side-effect-known function calls.
class ReportFuncCall : public RejectReason {
  //===--------------------------------------------------------------------===//

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
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with aliasing.
class ReportAlias : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  typedef std::vector<const llvm::Value *> PointerSnapshotTy;

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
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Base class for otherwise ungrouped reject reasons.
class ReportOther : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportOther(const RejectReasonKind K);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with bad IntToPtr instructions.
class ReportIntToPtr : public ReportOther {
  //===--------------------------------------------------------------------===//

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
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with alloca instructions.
class ReportAlloca : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportAlloca(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with unknown instructions.
class ReportUnknownInst : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportUnknownInst(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with regions containing the function entry block.
class ReportEntry : public ReportOther {
  //===--------------------------------------------------------------------===//
  BasicBlock *BB;

public:
  ReportEntry(BasicBlock *BB);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Report regions that seem not profitable to be optimized.
class ReportUnprofitable : public ReportOther {
  //===--------------------------------------------------------------------===//
  Region *R;

public:
  ReportUnprofitable(Region *R);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// Captures errors with non-simple memory accesses.
class ReportNonSimpleMemoryAccess : public ReportOther {
  //===--------------------------------------------------------------------===//

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
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

} // namespace polly

#endif // POLLY_SCOP_DETECTION_DIAGNOSTIC_H
