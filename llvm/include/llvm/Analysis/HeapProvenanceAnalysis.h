#ifndef LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H
#define LLVM_ANALYSIS_HEAPPROVENANCEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include <string>
#include <vector>

namespace llvm {

class HeapProvenanceAnalysisResult {
public:
  struct ProvenanceInfo {
    enum Kind {
      Uninit = 0,
      HeapChunkPtr,
      RecoverableHeapChunkPtr,
      Unknown
    } State = Uninit;

    enum Direction {
      None = 0,
      Forward = 1,
      Backward = 2,
      Both = 3
    } Dir = None;

    int64_t ConstOffset = 0;
    std::vector<std::string> SymOffsets;
    std::string CustomExpr;

    bool isUninit() const { return State == Uninit; }
    bool isValid() const {
      return State == HeapChunkPtr || State == RecoverableHeapChunkPtr;
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
      if (State == Uninit)
        return "Uninit";
      if (State == Unknown)
        return "Unknown";
      if (!CustomExpr.empty())
        return CustomExpr;
      if (State == HeapChunkPtr && ConstOffset == 0 && SymOffsets.empty())
        return "head";

      std::string Res = "head";
      if (ConstOffset > 0)
        Res += " + " + std::to_string(ConstOffset);
      else if (ConstOffset < 0)
        Res += " - " + std::to_string(-ConstOffset);

      for (const auto &S : SymOffsets) {
        if (!S.empty() && S[0] == '-')
          Res += " " + S;
        else
          Res += " + " + S;
      }
      return Res;
    }
  };

private:
  DenseMap<const Value *, ProvenanceInfo> ValueMap;

public:
  void setInfo(const Value *V, const ProvenanceInfo &Info) {
    ValueMap[V] = Info;
  }
  const ProvenanceInfo &getInfo(const Value *V) const {
    static ProvenanceInfo Empty;
    auto It = ValueMap.find(V);
    return It != ValueMap.end() ? It->second : Empty;
  }
  DenseMap<const Value *, ProvenanceInfo> &getMap() { return ValueMap; }
  const DenseMap<const Value *, ProvenanceInfo> &getMap() const {
    return ValueMap;
  }
};

class HeapProvenanceAnalysis
    : public AnalysisInfoMixin<HeapProvenanceAnalysis> {
  friend AnalysisInfoMixin<HeapProvenanceAnalysis>;
  static AnalysisKey Key;

public:
  using Result = HeapProvenanceAnalysisResult;
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
