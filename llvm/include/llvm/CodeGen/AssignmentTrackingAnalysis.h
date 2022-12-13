#ifndef LLVM_CODEGEN_ASSIGNMENTTRACKINGANALYSIS_H
#define LLVM_CODEGEN_ASSIGNMENTTRACKINGANALYSIS_H

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"

namespace llvm {
class Function;
class Instruction;
class Value;
class raw_ostream;
} // namespace llvm
class FunctionVarLocsBuilder;

namespace llvm {
/// Type wrapper for integer ID for Variables. 0 is reserved.
enum class VariableID : unsigned { Reserved = 0 };
/// Variable location definition used by FunctionVarLocs.
struct VarLocInfo {
  llvm::VariableID VariableID;
  DIExpression *Expr = nullptr;
  DebugLoc DL;
  Value *V = nullptr; // TODO: Needs to be value_s_ for variadic expressions.
};

/// Data structure describing the variable locations in a function. Used as the
/// result of the AssignmentTrackingAnalysis pass. Essentially read-only
/// outside of AssignmentTrackingAnalysis where it is built.
class FunctionVarLocs {
  /// Maps VarLocInfo.VariableID to a DebugVariable for VarLocRecords.
  SmallVector<DebugVariable> Variables;
  /// List of variable location changes grouped by the instruction the
  /// change occurs before (see VarLocsBeforeInst). The elements from
  /// zero to SingleVarLocEnd represent variables with a single location.
  SmallVector<VarLocInfo> VarLocRecords;
  /// End of range of VarLocRecords that represent variables with a single
  /// location that is valid for the entire scope. Range starts at 0.
  unsigned SingleVarLocEnd = 0;
  /// Maps an instruction to a range of VarLocs that start just before it.
  DenseMap<const Instruction *, std::pair<unsigned, unsigned>>
      VarLocsBeforeInst;

public:
  /// Return the DILocalVariable for the location definition represented by \p
  /// ID.
  DILocalVariable *getDILocalVariable(const VarLocInfo *Loc) const {
    VariableID VarID = Loc->VariableID;
    return getDILocalVariable(VarID);
  }
  /// Return the DILocalVariable of the variable represented by \p ID.
  DILocalVariable *getDILocalVariable(VariableID ID) const {
    return const_cast<DILocalVariable *>(getVariable(ID).getVariable());
  }
  /// Return the DebugVariable represented by \p ID.
  const DebugVariable &getVariable(VariableID ID) const {
    return Variables[static_cast<unsigned>(ID)];
  }

  ///@name iterators
  ///@{
  /// First single-location variable location definition.
  const VarLocInfo *single_locs_begin() const { return VarLocRecords.begin(); }
  /// One past the last single-location variable location definition.
  const VarLocInfo *single_locs_end() const {
    const auto *It = VarLocRecords.begin();
    std::advance(It, SingleVarLocEnd);
    return It;
  }
  /// First variable location definition that comes before \p Before.
  const VarLocInfo *locs_begin(const Instruction *Before) const {
    auto Span = VarLocsBeforeInst.lookup(Before);
    const auto *It = VarLocRecords.begin();
    std::advance(It, Span.first);
    return It;
  }
  /// One past the last variable location definition that comes before \p
  /// Before.
  const VarLocInfo *locs_end(const Instruction *Before) const {
    auto Span = VarLocsBeforeInst.lookup(Before);
    const auto *It = VarLocRecords.begin();
    std::advance(It, Span.second);
    return It;
  }
  ///@}

  void print(raw_ostream &OS, const Function &Fn) const;

  ///@{
  /// Non-const methods used by AssignmentTrackingAnalysis (which invalidate
  /// analysis results if called incorrectly).
  void init(FunctionVarLocsBuilder &Builder);
  void clear();
  ///@}
};

class AssignmentTrackingAnalysis : public FunctionPass {
  std::unique_ptr<FunctionVarLocs> Results;

public:
  static char ID;

  AssignmentTrackingAnalysis();

  bool runOnFunction(Function &F) override;

  static bool isRequired() { return true; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  const FunctionVarLocs *getResults() { return Results.get(); }
};

} // end namespace llvm
#endif // LLVM_CODEGEN_ASSIGNMENTTRACKINGANALYSIS_H
