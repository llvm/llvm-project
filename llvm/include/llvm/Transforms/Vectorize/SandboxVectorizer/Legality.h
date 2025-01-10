//===- Legality.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Legality checks for the Sandbox Vectorizer.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"

namespace llvm::sandboxir {

class LegalityAnalysis;
class Value;

enum class LegalityResultID {
  Pack,  ///> Collect scalar values.
  Widen, ///> Vectorize by combining scalars to a vector.
};

/// The reason for vectorizing or not vectorizing.
enum class ResultReason {
  NotInstructions,
  DiffOpcodes,
  DiffTypes,
  DiffMathFlags,
  DiffWrapFlags,
  NotConsecutive,
  CantSchedule,
  Unimplemented,
  Infeasible,
};

#ifndef NDEBUG
struct ToStr {
  static const char *getLegalityResultID(LegalityResultID ID) {
    switch (ID) {
    case LegalityResultID::Pack:
      return "Pack";
    case LegalityResultID::Widen:
      return "Widen";
    }
    llvm_unreachable("Unknown LegalityResultID enum");
  }

  static const char *getVecReason(ResultReason Reason) {
    switch (Reason) {
    case ResultReason::NotInstructions:
      return "NotInstructions";
    case ResultReason::DiffOpcodes:
      return "DiffOpcodes";
    case ResultReason::DiffTypes:
      return "DiffTypes";
    case ResultReason::DiffMathFlags:
      return "DiffMathFlags";
    case ResultReason::DiffWrapFlags:
      return "DiffWrapFlags";
    case ResultReason::NotConsecutive:
      return "NotConsecutive";
    case ResultReason::CantSchedule:
      return "CantSchedule";
    case ResultReason::Unimplemented:
      return "Unimplemented";
    case ResultReason::Infeasible:
      return "Infeasible";
    }
    llvm_unreachable("Unknown ResultReason enum");
  }
};
#endif // NDEBUG

/// The legality outcome is represented by a class rather than an enum class
/// because in some cases the legality checks are expensive and look for a
/// particular instruction that can be passed along to the vectorizer to avoid
/// repeating the same expensive computation.
class LegalityResult {
protected:
  LegalityResultID ID;
  /// Only Legality can create LegalityResults.
  LegalityResult(LegalityResultID ID) : ID(ID) {}
  friend class LegalityAnalysis;

  /// We shouldn't need copies.
  LegalityResult(const LegalityResult &) = delete;
  LegalityResult &operator=(const LegalityResult &) = delete;

public:
  virtual ~LegalityResult() {}
  LegalityResultID getSubclassID() const { return ID; }
#ifndef NDEBUG
  virtual void print(raw_ostream &OS) const {
    OS << ToStr::getLegalityResultID(ID);
  }
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const LegalityResult &LR) {
    LR.print(OS);
    return OS;
  }
#endif // NDEBUG
};

/// Base class for results with reason.
class LegalityResultWithReason : public LegalityResult {
  [[maybe_unused]] ResultReason Reason;
  LegalityResultWithReason(LegalityResultID ID, ResultReason Reason)
      : LegalityResult(ID), Reason(Reason) {}
  friend class Pack; // For constructor.

public:
  ResultReason getReason() const { return Reason; }
#ifndef NDEBUG
  void print(raw_ostream &OS) const override {
    LegalityResult::print(OS);
    OS << " Reason: " << ToStr::getVecReason(Reason);
  }
#endif
};

class Widen final : public LegalityResult {
  friend class LegalityAnalysis;
  Widen() : LegalityResult(LegalityResultID::Widen) {}

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::Widen;
  }
};

class Pack final : public LegalityResultWithReason {
  Pack(ResultReason Reason)
      : LegalityResultWithReason(LegalityResultID::Pack, Reason) {}
  friend class LegalityAnalysis; // For constructor.

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::Pack;
  }
};

/// Performs the legality analysis and returns a LegalityResult object.
class LegalityAnalysis {
  Scheduler Sched;
  /// Owns the legality result objects created by createLegalityResult().
  SmallVector<std::unique_ptr<LegalityResult>> ResultPool;
  /// Checks opcodes, types and other IR-specifics and returns a ResultReason
  /// object if not vectorizable, or nullptr otherwise.
  std::optional<ResultReason>
  notVectorizableBasedOnOpcodesAndTypes(ArrayRef<Value *> Bndl);

  ScalarEvolution &SE;
  const DataLayout &DL;

public:
  LegalityAnalysis(AAResults &AA, ScalarEvolution &SE, const DataLayout &DL,
                   Context &Ctx)
      : Sched(AA, Ctx), SE(SE), DL(DL) {}
  /// A LegalityResult factory.
  template <typename ResultT, typename... ArgsT>
  ResultT &createLegalityResult(ArgsT... Args) {
    ResultPool.push_back(std::unique_ptr<ResultT>(new ResultT(Args...)));
    return cast<ResultT>(*ResultPool.back());
  }
  /// Checks if it's legal to vectorize the instructions in \p Bndl.
  /// \Returns a LegalityResult object owned by LegalityAnalysis.
  /// \p SkipScheduling skips the scheduler check and is only meant for testing.
  // TODO: Try to remove the SkipScheduling argument by refactoring the tests.
  const LegalityResult &canVectorize(ArrayRef<Value *> Bndl,
                                     bool SkipScheduling = false);
  void clear() { Sched.clear(); }
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H
