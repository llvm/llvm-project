//===- DebugSSAUpdater.h - Debug SSA Update Tool ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the DebugSSAUpdater class, which is used to evaluate the
// live values of debug variables in IR. This uses SSA construction, treating
// debug value records as definitions, to determine at each point in the program
// which definition(s) are live at a given point. This is useful for analysis of
// the state of debug variables, such as measuring the change in values of a
// variable over time, or calculating coverage stats.
//
// NB: This is an expensive analysis that is generally not suitable for use in
// LLVM passes, but may be useful for standalone tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DEBUGSSAUPDATER_H
#define LLVM_TRANSFORMS_UTILS_DEBUGSSAUPDATER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include <cstdint>

namespace llvm {

////////////////////////////////////////
// SSAUpdater specialization classes

class DbgSSAPhi;
template <typename T> class SSAUpdaterTraits;

/// A definition of a variable; can represent either a debug value, no
/// definition (the variable has not yet been defined), or a phi value*.
/// *Meaning multiple definitions that are live-in to a block from different
/// predecessors, not a debug value that uses an IR PHINode.
struct DbgValueDef {
  DbgSSAPhi *Phi;
  bool IsUndef;
  bool IsMemory;
  Metadata *Locations;
  DIExpression *Expression;

  DbgValueDef()
      : Phi(nullptr), IsUndef(true), IsMemory(false), Locations(nullptr),
        Expression(nullptr) {}
  DbgValueDef(int)
      : Phi(nullptr), IsUndef(true), IsMemory(false), Locations(nullptr),
        Expression(nullptr) {}
  DbgValueDef(bool IsMemory, Metadata *Locations, DIExpression *Expression)
      : Phi(nullptr), IsUndef(false), IsMemory(IsMemory), Locations(Locations),
        Expression(Expression) {}
  DbgValueDef(DbgVariableRecord *DVR) : Phi(nullptr) {
    assert(!DVR->isDbgAssign() && "#dbg_assign not yet supported");
    IsUndef = DVR->isKillLocation();
    IsMemory = DVR->isAddressOfVariable();
    Locations = DVR->getRawLocation();
    Expression = DVR->getExpression();
  }
  DbgValueDef(DbgSSAPhi *Phi)
      : Phi(Phi), IsUndef(false), IsMemory(false), Locations(nullptr),
        Expression(nullptr) {}

  bool agreesWith(DbgValueDef Other) const {
    if (IsUndef && Other.IsUndef)
      return true;
    return std::tie(Phi, IsUndef, IsMemory, Locations, Expression) ==
           std::tie(Other.Phi, Other.IsUndef, Other.IsMemory, Other.Locations,
                    Other.Expression);
  }

  operator bool() const { return !IsUndef; }
  bool operator==(DbgValueDef Other) const { return agreesWith(Other); }
  bool operator!=(DbgValueDef Other) const { return !agreesWith(Other); }

  void print(raw_ostream &OS) const;
};

class DbgSSABlock;
class DebugSSAUpdater;

/// Represents the live-in definitions of a variable to a block with multiple
/// predecessors.
class DbgSSAPhi {
public:
  SmallVector<std::pair<DbgSSABlock *, DbgValueDef>, 4> IncomingValues;
  DbgSSABlock *ParentBlock;
  DbgSSAPhi(DbgSSABlock *ParentBlock) : ParentBlock(ParentBlock) {}

  DbgSSABlock *getParent() { return ParentBlock; }
  unsigned getNumIncomingValues() const { return IncomingValues.size(); }
  DbgSSABlock *getIncomingBlock(size_t Idx) {
    return IncomingValues[Idx].first;
  }
  DbgValueDef getIncomingValue(size_t Idx) {
    return IncomingValues[Idx].second;
  }
  void addIncoming(DbgSSABlock *BB, DbgValueDef DV) {
    IncomingValues.push_back({BB, DV});
  }

  void print(raw_ostream &OS) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const DbgValueDef &DV) {
  DV.print(OS);
  return OS;
}
inline raw_ostream &operator<<(raw_ostream &OS, const DbgSSAPhi &PHI) {
  PHI.print(OS);
  return OS;
}

/// Thin wrapper around a block successor iterator.
class DbgSSABlockSuccIterator {
public:
  succ_iterator SuccIt;
  DebugSSAUpdater &Updater;

  DbgSSABlockSuccIterator(succ_iterator SuccIt, DebugSSAUpdater &Updater)
      : SuccIt(SuccIt), Updater(Updater) {}

  bool operator!=(const DbgSSABlockSuccIterator &OtherIt) const {
    return OtherIt.SuccIt != SuccIt;
  }

  DbgSSABlockSuccIterator &operator++() {
    ++SuccIt;
    return *this;
  }

  DbgSSABlock *operator*();
};

/// Thin wrapper around a block successor iterator.
class DbgSSABlockPredIterator {
public:
  pred_iterator PredIt;
  DebugSSAUpdater &Updater;

  DbgSSABlockPredIterator(pred_iterator PredIt, DebugSSAUpdater &Updater)
      : PredIt(PredIt), Updater(Updater) {}

  bool operator!=(const DbgSSABlockPredIterator &OtherIt) const {
    return OtherIt.PredIt != PredIt;
  }

  DbgSSABlockPredIterator &operator++() {
    ++PredIt;
    return *this;
  }

  DbgSSABlock *operator*();
};

class DbgSSABlock {
public:
  BasicBlock &BB;
  DebugSSAUpdater &Updater;
  using PHIListT = SmallVector<DbgSSAPhi, 1>;
  /// List of PHIs in this block. There should only ever be one, but this needs
  /// to be a list for the SSAUpdater.
  PHIListT PHIList;

  DbgSSABlock(BasicBlock &BB, DebugSSAUpdater &Updater)
      : BB(BB), Updater(Updater) {}

  DbgSSABlockPredIterator pred_begin() {
    return DbgSSABlockPredIterator(llvm::pred_begin(&BB), Updater);
  }

  DbgSSABlockPredIterator pred_end() {
    return DbgSSABlockPredIterator(llvm::pred_end(&BB), Updater);
  }

  iterator_range<DbgSSABlockPredIterator> predecessors() {
    return iterator_range(pred_begin(), pred_end());
  }

  DbgSSABlockSuccIterator succ_begin() {
    return DbgSSABlockSuccIterator(llvm::succ_begin(&BB), Updater);
  }

  DbgSSABlockSuccIterator succ_end() {
    return DbgSSABlockSuccIterator(llvm::succ_end(&BB), Updater);
  }

  iterator_range<DbgSSABlockSuccIterator> successors() {
    return iterator_range(succ_begin(), succ_end());
  }

  /// SSAUpdater has requested a PHI: create that within this block record.
  DbgSSAPhi *newPHI() {
    assert(PHIList.empty() &&
           "Only one PHI should exist per-block per-variable");
    PHIList.emplace_back(this);
    return &PHIList.back();
  }

  /// SSAUpdater wishes to know what PHIs already exist in this block.
  PHIListT &phis() { return PHIList; }
};

/// Class used to determine the live ranges of debug variables in IR using
/// SSA construction (via the SSAUpdaterImpl class), used for analysis purposes.
class DebugSSAUpdater {
  friend class SSAUpdaterTraits<DebugSSAUpdater>;
  using AvailableValsTy = DenseMap<DbgSSABlock *, DbgValueDef>;

private:
  /// This keeps track of which value to use on a per-block basis. When we
  /// insert PHI nodes, we keep track of them here.
  AvailableValsTy AV;

  /// Pointer to an optionally-passed vector into which, if it is non-null,
  /// the PHIs that describe ambiguous variable locations will be inserted.
  SmallVectorImpl<DbgSSAPhi *> *InsertedPHIs;

  DenseMap<BasicBlock *, DbgSSABlock *> BlockMap;

public:
  /// If InsertedPHIs is specified, it will be filled
  /// in with all PHI Nodes created by rewriting.
  explicit DebugSSAUpdater(
      SmallVectorImpl<DbgSSAPhi *> *InsertedPHIs = nullptr);
  DebugSSAUpdater(const DebugSSAUpdater &) = delete;
  DebugSSAUpdater &operator=(const DebugSSAUpdater &) = delete;

  ~DebugSSAUpdater() {
    for (auto &Block : BlockMap)
      delete Block.second;
  }

  void reset() {
    for (auto &Block : BlockMap)
      delete Block.second;

    if (InsertedPHIs)
      InsertedPHIs->clear();
    BlockMap.clear();
  }

  void initialize();

  /// For a given BB, create a wrapper block for it. Stores it in the
  /// DebugSSAUpdater block map.
  DbgSSABlock *getDbgSSABlock(BasicBlock *BB) {
    auto it = BlockMap.find(BB);
    if (it == BlockMap.end()) {
      BlockMap[BB] = new DbgSSABlock(*BB, *this);
      it = BlockMap.find(BB);
    }
    return it->second;
  }

  /// Indicate that a rewritten value is available in the specified block
  /// with the specified value.
  void addAvailableValue(DbgSSABlock *BB, DbgValueDef DV);

  /// Return true if the DebugSSAUpdater already has a value for the specified
  /// block.
  bool hasValueForBlock(DbgSSABlock *BB) const;

  /// Return the value for the specified block if the DebugSSAUpdater has one,
  /// otherwise return nullptr.
  DbgValueDef findValueForBlock(DbgSSABlock *BB) const;

  /// Construct SSA form, materializing a value that is live at the end
  /// of the specified block.
  DbgValueDef getValueAtEndOfBlock(DbgSSABlock *BB);

  /// Construct SSA form, materializing a value that is live in the
  /// middle of the specified block.
  ///
  /// \c getValueInMiddleOfBlock is the same as \c GetValueAtEndOfBlock except
  /// in one important case: if there is a definition of the rewritten value
  /// after the 'use' in BB.  Consider code like this:
  ///
  /// \code
  ///      X1 = ...
  ///   SomeBB:
  ///      use(X)
  ///      X2 = ...
  ///      br Cond, SomeBB, OutBB
  /// \endcode
  ///
  /// In this case, there are two values (X1 and X2) added to the AvailableVals
  /// set by the client of the rewriter, and those values are both live out of
  /// their respective blocks.  However, the use of X happens in the *middle* of
  /// a block.  Because of this, we need to insert a new PHI node in SomeBB to
  /// merge the appropriate values, and this value isn't live out of the block.
  DbgValueDef getValueInMiddleOfBlock(DbgSSABlock *BB);

private:
  DbgValueDef getValueAtEndOfBlockInternal(DbgSSABlock *BB);
};

struct DbgRangeEntry {
  BasicBlock::iterator Start;
  BasicBlock::iterator End;
  // Should be non-PHI.
  DbgValueDef Value;
};

/// Utility class used to store the names of SSA values after their owning
/// modules have been destroyed. Values are added via \c addValue to receive a
/// corresponding ID, which can then be used to retrieve the name of the SSA
/// value via \c getName at any point. Adding the same value multiple times
/// returns the same ID, making \c addValue idempotent.
class SSAValueNameMap {
  struct Config : ValueMapConfig<Value *> {
    enum { FollowRAUW = false };
  };

public:
  using ValueID = uint64_t;
  ValueID addValue(Value *V);
  std::string getName(ValueID ID) { return ValueIDToNameMap[ID]; }

private:
  DenseMap<ValueID, std::string> ValueIDToNameMap;
  ValueMap<Value *, ValueID, Config> ValueToIDMap;
  ValueID NextID = 0;
};

/// Utility class used to find and store the live debug ranges for variables in
/// a module. This class uses the DebugSSAUpdater for each variable added with
/// \c addVariable to find either a single-location value, e.g. #dbg_declare, or
/// a set of live value ranges corresponding to the set of #dbg_value records.
class DbgValueRangeTable {
  DenseMap<DebugVariableAggregate, SmallVector<DbgRangeEntry>>
      OrigVariableValueRangeTable;
  DenseMap<DebugVariableAggregate, DbgValueDef> OrigSingleLocVariableValueTable;

public:
  void addVariable(Function *F, DebugVariableAggregate DVA);
  bool hasVariableEntry(DebugVariableAggregate DVA) const {
    return OrigVariableValueRangeTable.contains(DVA) ||
           OrigSingleLocVariableValueTable.contains(DVA);
  }
  bool hasSingleLocEntry(DebugVariableAggregate DVA) const {
    return OrigSingleLocVariableValueTable.contains(DVA);
  }
  ArrayRef<DbgRangeEntry> getVariableRanges(DebugVariableAggregate DVA) {
    return OrigVariableValueRangeTable[DVA];
  }
  DbgValueDef getSingleLoc(DebugVariableAggregate DVA) {
    return OrigSingleLocVariableValueTable[DVA];
  }

  void printValues(DebugVariableAggregate DVA, raw_ostream &OS);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEBUGSSAUPDATER_H
