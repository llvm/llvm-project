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
class InstrMaps;

class ShuffleMask {
public:
  using IndicesVecT = SmallVector<int, 8>;

private:
  IndicesVecT Indices;

public:
  ShuffleMask(SmallVectorImpl<int> &&Indices) : Indices(std::move(Indices)) {}
  ShuffleMask(std::initializer_list<int> Indices) : Indices(Indices) {}
  explicit ShuffleMask(ArrayRef<int> Indices) : Indices(Indices) {}
  operator ArrayRef<int>() const { return Indices; }
  /// Creates and returns an identity shuffle mask of size \p Sz.
  /// For example if Sz == 4 the returned mask is {0, 1, 2, 3}.
  static ShuffleMask getIdentity(unsigned Sz) {
    IndicesVecT Indices;
    Indices.reserve(Sz);
    for (auto Idx : seq<int>(0, (int)Sz))
      Indices.push_back(Idx);
    return ShuffleMask(std::move(Indices));
  }
  /// \Returns true if the mask is a perfect identity mask with consecutive
  /// indices, i.e., performs no lane shuffling, like 0,1,2,3...
  bool isIdentity() const {
    for (auto [Idx, Elm] : enumerate(Indices)) {
      if ((int)Idx != Elm)
        return false;
    }
    return true;
  }
  bool operator==(const ShuffleMask &Other) const {
    return Indices == Other.Indices;
  }
  bool operator!=(const ShuffleMask &Other) const { return !(*this == Other); }
  size_t size() const { return Indices.size(); }
  int operator[](int Idx) const { return Indices[Idx]; }
  using const_iterator = IndicesVecT::const_iterator;
  const_iterator begin() const { return Indices.begin(); }
  const_iterator end() const { return Indices.end(); }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const ShuffleMask &Mask) {
    Mask.print(OS);
    return OS;
  }
  void print(raw_ostream &OS) const {
    interleave(Indices, OS, [&OS](auto Elm) { OS << Elm; }, ",");
  }
  LLVM_DUMP_METHOD void dump() const;
#endif
};

enum class LegalityResultID {
  Pack,                    ///> Collect scalar values.
  Widen,                   ///> Vectorize by combining scalars to a vector.
  DiamondReuse,            ///> Don't generate new code, reuse existing vector.
  DiamondReuseWithShuffle, ///> Reuse the existing vector but add a shuffle.
  DiamondReuseMultiInput,  ///> Reuse more than one vector and/or scalars.
};

/// The reason for vectorizing or not vectorizing.
enum class ResultReason {
  NotInstructions,
  DiffOpcodes,
  DiffTypes,
  DiffMathFlags,
  DiffWrapFlags,
  DiffBBs,
  RepeatedInstrs,
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
    case LegalityResultID::DiamondReuse:
      return "DiamondReuse";
    case LegalityResultID::DiamondReuseWithShuffle:
      return "DiamondReuseWithShuffle";
    case LegalityResultID::DiamondReuseMultiInput:
      return "DiamondReuseMultiInput";
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
    case ResultReason::DiffBBs:
      return "DiffBBs";
    case ResultReason::RepeatedInstrs:
      return "RepeatedInstrs";
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

class DiamondReuse final : public LegalityResult {
  friend class LegalityAnalysis;
  Value *Vec;
  DiamondReuse(Value *Vec)
      : LegalityResult(LegalityResultID::DiamondReuse), Vec(Vec) {}

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::DiamondReuse;
  }
  Value *getVector() const { return Vec; }
};

class DiamondReuseWithShuffle final : public LegalityResult {
  friend class LegalityAnalysis;
  Value *Vec;
  ShuffleMask Mask;
  DiamondReuseWithShuffle(Value *Vec, const ShuffleMask &Mask)
      : LegalityResult(LegalityResultID::DiamondReuseWithShuffle), Vec(Vec),
        Mask(Mask) {}

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::DiamondReuseWithShuffle;
  }
  Value *getVector() const { return Vec; }
  const ShuffleMask &getMask() const { return Mask; }
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

/// Describes how to collect the values needed by each lane.
class CollectDescr {
public:
  /// Describes how to get a value element. If the value is a vector then it
  /// also provides the index to extract it from.
  class ExtractElementDescr {
    Value *V;
    /// The index in `V` that the value can be extracted from.
    /// This is nullopt if we need to use `V` as a whole.
    std::optional<int> ExtractIdx;

  public:
    ExtractElementDescr(Value *V, int ExtractIdx)
        : V(V), ExtractIdx(ExtractIdx) {}
    ExtractElementDescr(Value *V) : V(V), ExtractIdx(std::nullopt) {}
    Value *getValue() const { return V; }
    bool needsExtract() const { return ExtractIdx.has_value(); }
    int getExtractIdx() const { return *ExtractIdx; }
  };

  using DescrVecT = SmallVector<ExtractElementDescr, 4>;
  DescrVecT Descrs;

public:
  CollectDescr(SmallVectorImpl<ExtractElementDescr> &&Descrs)
      : Descrs(std::move(Descrs)) {}
  /// If all elements come from a single vector input, then return that vector
  /// and also the shuffle mask required to get them in order.
  std::optional<std::pair<Value *, ShuffleMask>> getSingleInput() const {
    const auto &Descr0 = *Descrs.begin();
    Value *V0 = Descr0.getValue();
    if (!Descr0.needsExtract())
      return std::nullopt;
    ShuffleMask::IndicesVecT MaskIndices;
    MaskIndices.push_back(Descr0.getExtractIdx());
    for (const auto &Descr : drop_begin(Descrs)) {
      if (!Descr.needsExtract())
        return std::nullopt;
      if (Descr.getValue() != V0)
        return std::nullopt;
      MaskIndices.push_back(Descr.getExtractIdx());
    }
    return std::make_pair(V0, ShuffleMask(std::move(MaskIndices)));
  }
  bool hasVectorInputs() const {
    return any_of(Descrs, [](const auto &D) { return D.needsExtract(); });
  }
  const SmallVector<ExtractElementDescr, 4> &getDescrs() const {
    return Descrs;
  }
};

class DiamondReuseMultiInput final : public LegalityResult {
  friend class LegalityAnalysis;
  CollectDescr Descr;
  DiamondReuseMultiInput(CollectDescr &&Descr)
      : LegalityResult(LegalityResultID::DiamondReuseMultiInput),
        Descr(std::move(Descr)) {}

public:
  static bool classof(const LegalityResult *From) {
    return From->getSubclassID() == LegalityResultID::DiamondReuseMultiInput;
  }
  const CollectDescr &getCollectDescr() const { return Descr; }
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
  InstrMaps &IMaps;

  /// Finds how we can collect the values in \p Bndl from the vectorized or
  /// non-vectorized code. It returns a map of the value we should extract from
  /// and the corresponding shuffle mask we need to use.
  CollectDescr getHowToCollectValues(ArrayRef<Value *> Bndl) const;

public:
  LegalityAnalysis(AAResults &AA, ScalarEvolution &SE, const DataLayout &DL,
                   Context &Ctx, InstrMaps &IMaps)
      : Sched(AA, Ctx), SE(SE), DL(DL), IMaps(IMaps) {}
  /// A LegalityResult factory.
  template <typename ResultT, typename... ArgsT>
  ResultT &createLegalityResult(ArgsT &&...Args) {
    ResultPool.push_back(
        std::unique_ptr<ResultT>(new ResultT(std::move(Args)...)));
    return cast<ResultT>(*ResultPool.back());
  }
  /// Checks if it's legal to vectorize the instructions in \p Bndl.
  /// \Returns a LegalityResult object owned by LegalityAnalysis.
  /// \p SkipScheduling skips the scheduler check and is only meant for testing.
  // TODO: Try to remove the SkipScheduling argument by refactoring the tests.
  const LegalityResult &canVectorize(ArrayRef<Value *> Bndl,
                                     bool SkipScheduling = false);
  void clear();
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_LEGALITY_H
