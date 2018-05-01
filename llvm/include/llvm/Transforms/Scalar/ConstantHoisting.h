//==- ConstantHoisting.h - Prepare code for expensive constants --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass identifies expensive constants to hoist and coalesces them to
// better prepare it for SelectionDAG-based code generation. This works around
// the limitations of the basic-block-at-a-time approach.
//
// First it scans all instructions for integer constants and calculates its
// cost. If the constant can be folded into the instruction (the cost is
// TCC_Free) or the cost is just a simple operation (TCC_BASIC), then we don't
// consider it expensive and leave it alone. This is the default behavior and
// the default implementation of getIntImmCost will always return TCC_Free.
//
// If the cost is more than TCC_BASIC, then the integer constant can't be folded
// into the instruction and it might be beneficial to hoist the constant.
// Similar constants are coalesced to reduce register pressure and
// materialization code.
//
// When a constant is hoisted, it is also hidden behind a bitcast to force it to
// be live-out of the basic block. Otherwise the constant would be just
// duplicated and each basic block would have its own copy in the SelectionDAG.
// The SelectionDAG recognizes such constants as opaque and doesn't perform
// certain transformations on them, which would create a new expensive constant.
//
// This optimization is only applied to integer constants in instructions and
// simple (this means not nested) constant cast expressions. For example:
// %0 = load i64* inttoptr (i64 big_constant to i64*)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_CONSTANTHOISTING_H
#define LLVM_TRANSFORMS_SCALAR_CONSTANTHOISTING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include <algorithm>
#include <vector>

namespace llvm {

class BasicBlock;
class BlockFrequencyInfo;
class Constant;
class ConstantInt;
class DominatorTree;
class Function;
class Instruction;
class TargetTransformInfo;

/// A private "module" namespace for types and utilities used by
/// ConstantHoisting. These are implementation details and should not be used by
/// clients.
namespace consthoist {

/// Keeps track of the user of a constant and the operand index where the
/// constant is used.
struct ConstantUser {
  Instruction *Inst;
  unsigned OpndIdx;

  ConstantUser(Instruction *Inst, unsigned Idx) : Inst(Inst), OpndIdx(Idx) {}
};

using ConstantUseListType = SmallVector<ConstantUser, 8>;

/// Keeps track of a constant candidate and its uses.
struct ConstantCandidate {
  ConstantUseListType Uses;
  ConstantInt *ConstInt;
  unsigned CumulativeCost = 0;

  ConstantCandidate(ConstantInt *ConstInt) : ConstInt(ConstInt) {}

  /// Add the user to the use list and update the cost.
  void addUser(Instruction *Inst, unsigned Idx, unsigned Cost) {
    CumulativeCost += Cost;
    Uses.push_back(ConstantUser(Inst, Idx));
  }
};

/// This represents a constant that has been rebased with respect to a
/// base constant. The difference to the base constant is recorded in Offset.
struct RebasedConstantInfo {
  ConstantUseListType Uses;
  Constant *Offset;

  RebasedConstantInfo(ConstantUseListType &&Uses, Constant *Offset)
    : Uses(std::move(Uses)), Offset(Offset) {}
};

using RebasedConstantListType = SmallVector<RebasedConstantInfo, 4>;

/// A base constant and all its rebased constants.
struct ConstantInfo {
  ConstantInt *BaseConstant;
  RebasedConstantListType RebasedConstants;
};

} // end namespace consthoist

class ConstantHoistingPass : public PassInfoMixin<ConstantHoistingPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  // Glue for old PM.
  bool runImpl(Function &F, TargetTransformInfo &TTI, DominatorTree &DT,
               BlockFrequencyInfo *BFI, BasicBlock &Entry);

  void releaseMemory() {
    ConstantVec.clear();
    ClonedCastMap.clear();
    ConstCandVec.clear();
  }

private:
  using ConstCandMapType = DenseMap<ConstantInt *, unsigned>;
  using ConstCandVecType = std::vector<consthoist::ConstantCandidate>;

  const TargetTransformInfo *TTI;
  DominatorTree *DT;
  BlockFrequencyInfo *BFI;
  BasicBlock *Entry;

  /// Keeps track of constant candidates found in the function.
  ConstCandVecType ConstCandVec;

  /// Keep track of cast instructions we already cloned.
  SmallDenseMap<Instruction *, Instruction *> ClonedCastMap;

  /// These are the final constants we decided to hoist.
  SmallVector<consthoist::ConstantInfo, 8> ConstantVec;

  Instruction *findMatInsertPt(Instruction *Inst, unsigned Idx = ~0U) const;
  SmallPtrSet<Instruction *, 8>
  findConstantInsertionPoint(const consthoist::ConstantInfo &ConstInfo) const;
  void collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                 Instruction *Inst, unsigned Idx,
                                 ConstantInt *ConstInt);
  void collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                 Instruction *Inst, unsigned Idx);
  void collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                 Instruction *Inst);
  void collectConstantCandidates(Function &Fn);
  void findAndMakeBaseConstant(ConstCandVecType::iterator S,
                               ConstCandVecType::iterator E);
  unsigned maximizeConstantsInRange(ConstCandVecType::iterator S,
                                    ConstCandVecType::iterator E,
                                    ConstCandVecType::iterator &MaxCostItr);
  void findBaseConstants();
  void emitBaseConstants(Instruction *Base, Constant *Offset,
                         const consthoist::ConstantUser &ConstUser);
  bool emitBaseConstants();
  void deleteDeadCastInst() const;
  bool optimizeConstants(Function &Fn);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_CONSTANTHOISTING_H
