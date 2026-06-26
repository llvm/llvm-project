#ifndef LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H
#define LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include <string>

namespace llvm {

/// ============================================================================
/// Heap Provenance Analysis (HPA) Theory & Dominance Properties
/// ============================================================================
///
/// 1. Forward vs. Backward Propagation & Dominance Relations:
///    Let A -> B denote that value A donates its heap provenance to value B.
///
///    - Forward Propagation (A --hpa_fwd--> B):
///      For unary SSA derivations (e.g., B = GEP(A), BitCast(A), AddrSpaceCast(A)),
///      B is computed directly from operand A along def-use chains, so A strictly
///      dominates B (A dom B).
///      Note on Control Flow Merges (PHINode / SelectInst): When B selects between
///      multiple incoming heads A1, A2, ..., no single antecedent Ai dominates B across
///      all CFG paths. Therefore, HPA transitions the HeadPayload at PHI/Select nodes
///      to B itself ({Kind::Phi/Select, B}). Because B trivially dominates all its
///      own uses, runtime instrumentation safely queries malloc_usable_size(B).
///
///    - Backward Propagation (A --hpa_bwd--> B):
///      A is a deallocation site (e.g., call to free operand B), and B is the
///      pointer definition being deallocated. In SSA form, because deallocation A
///      consumes operand B, definition B strictly dominates A (B dom A).
///      Note: A does NOT post-dominate B in the CFG, because B may return, escape,
///      or pass to other calls without being freed locally on all paths. Backward
///      HPA infers a flow-insensitive property: if B is freed anywhere, it must be
///      a valid heap pointer across its entire live range dominated by B.
///
/// 2. GEP & Pointer Arithmetic Variations:
///    For B = GEP(A, offset), forward propagation maintains A dom B. When bounds
///    checking instruments a memory access at instruction I using derived pointer
///    B, HPA traces B back to root allocation head A. Because A dom B and B dom I
///    (or B == I operand), the root head A strictly dominates the access I.
///
/// 3. Hoisting & SSA Insertion Point Criteria:
///    When BoundsCheckingPass instruments an access on pointer V with root head
///    HeadVal, the runtime size query (malloc_usable_size(HeadVal)) is positioned
///    safely in SSA form using InsertAfter on the latest definition between V
///    and HeadVal. This hoists the metadata computation above the access while
///    guaranteeing strict SSA dominance across PHIs and Select merges without
///    breaking basic block terminators.
/// ============================================================================
struct HeapProvenanceLattice {
  enum class StateKind {
    Uninit = 0,
    HeapChunkHead,
    HeapChunkInterior,
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
    return State == StateKind::HeapChunkHead || State == StateKind::HeapChunkInterior;
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
    if (It != ValueMap.end())
      return It->second;
    if (auto *GEP = dyn_cast<GEPOperator>(V))
      return getInfo(GEP->getPointerOperand());
    if (auto *BC = dyn_cast<BitCastOperator>(V))
      return getInfo(BC->getOperand(0));
    if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
      return getInfo(ASC->getOperand(0));
    return Empty;
  }
  DenseMap<const Value *, HeapProvenanceLattice> &getMap() { return ValueMap; }
  const DenseMap<const Value *, HeapProvenanceLattice> &getMap() const {
    return ValueMap;
  }
  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &);
};

class ForwardHeapProvenanceAnalysis
    : public AnalysisInfoMixin<ForwardHeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<ForwardHeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = ForwardHeapProvenanceAnalysisResult;
  Result run(Module &M, ModuleAnalysisManager &MAM);
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
    if (It != ValueMap.end())
      return It->second;
    if (auto *GEP = dyn_cast<GEPOperator>(V))
      return getInfo(GEP->getPointerOperand());
    if (auto *BC = dyn_cast<BitCastOperator>(V))
      return getInfo(BC->getOperand(0));
    if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
      return getInfo(ASC->getOperand(0));
    return Empty;
  }
  DenseMap<const Value *, HeapProvenanceLattice> &getMap() { return ValueMap; }
  const DenseMap<const Value *, HeapProvenanceLattice> &getMap() const {
    return ValueMap;
  }
  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &);
};

class BackwardHeapProvenanceAnalysis
    : public AnalysisInfoMixin<BackwardHeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<BackwardHeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = BackwardHeapProvenanceAnalysisResult;
  Result run(Module &M, ModuleAnalysisManager &MAM);
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

  bool invalidate(Module &, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &);
};

class HeapProvenanceAnalysis
    : public AnalysisInfoMixin<HeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<HeapProvenanceAnalysis>;
  static AnalysisKey Key;
public:
  using Result = HeapProvenanceAnalysisResult;
  static Result analyzeModule(Module &M);
  Result run(Module &M, ModuleAnalysisManager &MAM);
};

class HeapProvenancePrinterPass
    : public PassInfoMixin<HeapProvenancePrinterPass> {
  raw_ostream &OS;
public:
  explicit HeapProvenancePrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H
