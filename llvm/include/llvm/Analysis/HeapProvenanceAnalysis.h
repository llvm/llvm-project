#ifndef LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H
#define LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include <string>

namespace llvm {

struct HeapProvenanceLattice {
  enum class StateKind {
    Uninit = 0,
    HeapChunkHead,
    HeapChunkInterim,
    Unknown
  } State = StateKind::Uninit;

  enum Direction {
    None = 0,
    Forward = 1,
    Backward = 2,
    Both = 3
  } Dir = None;

  struct Payload {
    enum class Kind { None, Ref, Select, Phi } K = Kind::None;
    const Value *Val = nullptr;

    bool operator==(const Payload &RHS) const {
      return K == RHS.K && Val == RHS.Val;
    }
    bool operator!=(const Payload &RHS) const { return !(*this == RHS); }
  } HeadPayload;

  bool isUninit() const { return State == StateKind::Uninit; }
  bool isValid() const {
    return State == StateKind::HeapChunkHead || State == StateKind::HeapChunkInterim;
  }
  bool operator==(const HeapProvenanceLattice &RHS) const {
    return State == RHS.State && Dir == RHS.Dir && HeadPayload == RHS.HeadPayload;
  }
  bool operator!=(const HeapProvenanceLattice &RHS) const {
    return !(*this == RHS);
  }

  const Value *getHead() const {
    if (isValid())
      return HeadPayload.Val;
    return nullptr;
  }

  std::string getDirectionStr() const {
    if (Dir == Forward)
      return " [forward: from alloc]";
    if (Dir == Backward)
      return " [backward: into dealloc]";
    if (Dir == Both)
      return " [both: alloc & dealloc]";
    return "";
  }

  std::string getExpr() const {
    if (State == StateKind::Uninit)
      return "Uninit";
    if (State == StateKind::Unknown)
      return "Unknown";
    if (HeadPayload.K == Payload::Kind::Ref && HeadPayload.Val) {
      std::string Name = HeadPayload.Val->getName().str();
      return Name.empty() ? "Ref" : ("Ref(" + Name + ")");
    }
    if (HeadPayload.K == Payload::Kind::Select)
      return "Select";
    if (HeadPayload.K == Payload::Kind::Phi)
      return "Phi";
    return "Head";
  }
};

class ForwardHeapProvenanceAnalysis;
class BackwardHeapProvenanceAnalysis;
class HeapProvenanceAnalysis;

class ForwardHeapProvenanceAnalysisResult {
  DenseMap<const Value *, HeapProvenanceLattice> ValueMap;
public:
  void setInfo(const Value *V, const HeapProvenanceLattice &Info) {
    ValueMap[V] = Info;
  }
  const HeapProvenanceLattice &getInfo(const Value *V) const {
    static HeapProvenanceLattice Empty;
    auto It = ValueMap.find(V);
    return It != ValueMap.end() ? It->second : Empty;
  }
  DenseMap<const Value *, HeapProvenanceLattice> &getMap() { return ValueMap; }
  const DenseMap<const Value *, HeapProvenanceLattice> &getMap() const {
    return ValueMap;
  }
  bool invalidate(Function &, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);
};

class ForwardHeapProvenanceAnalysis
    : public AnalysisInfoMixin<ForwardHeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<ForwardHeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = ForwardHeapProvenanceAnalysisResult;
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

class BackwardHeapProvenanceAnalysisResult {
  DenseMap<const Value *, HeapProvenanceLattice> ValueMap;
public:
  void setInfo(const Value *V, const HeapProvenanceLattice &Info) {
    ValueMap[V] = Info;
  }
  const HeapProvenanceLattice &getInfo(const Value *V) const {
    static HeapProvenanceLattice Empty;
    auto It = ValueMap.find(V);
    return It != ValueMap.end() ? It->second : Empty;
  }
  DenseMap<const Value *, HeapProvenanceLattice> &getMap() { return ValueMap; }
  const DenseMap<const Value *, HeapProvenanceLattice> &getMap() const {
    return ValueMap;
  }
  bool invalidate(Function &, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);
};

class BackwardHeapProvenanceAnalysis
    : public AnalysisInfoMixin<BackwardHeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<BackwardHeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = BackwardHeapProvenanceAnalysisResult;
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

// Combined wrapper for backward compatibility with ObjectSizeOffsetEvaluator
class HeapProvenanceAnalysisResult {
public:
  using ProvenanceInfo = HeapProvenanceLattice;

private:
  ForwardHeapProvenanceAnalysisResult ForwardRes;
  BackwardHeapProvenanceAnalysisResult BackwardRes;

public:
  HeapProvenanceAnalysisResult() = default;
  HeapProvenanceAnalysisResult(ForwardHeapProvenanceAnalysisResult F,
                               BackwardHeapProvenanceAnalysisResult B)
      : ForwardRes(std::move(F)), BackwardRes(std::move(B)) {}

  const ProvenanceInfo &getInfo(const Value *V) const {
    const auto &F = ForwardRes.getInfo(V);
    if (F.isValid())
      return F;
    return BackwardRes.getInfo(V);
  }

  const ForwardHeapProvenanceAnalysisResult &getForwardResult() const { return ForwardRes; }
  const BackwardHeapProvenanceAnalysisResult &getBackwardResult() const { return BackwardRes; }

  bool invalidate(Function &, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &);
};

class HeapProvenanceAnalysis
    : public AnalysisInfoMixin<HeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<HeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = HeapProvenanceAnalysisResult;
  static Result analyzeFunction(Function &F);
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

class HeapProvenancePrinterPass
    : public PassInfoMixin<HeapProvenancePrinterPass> {
  raw_ostream &OS;
public:
  explicit HeapProvenancePrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H
