//===- DebugSSAUpdater.cpp - Debug Variable SSA Update Tool ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DebugSSAUpdater class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DebugSSAUpdater.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Transforms/Utils/SSAUpdaterImpl.h"

using namespace llvm;

#define DEBUG_TYPE "debug-ssa-updater"

void DbgValueDef::print(raw_ostream &OS) const {
  OS << "DbgVal{ ";
  if (IsUndef) {
    OS << "undef }";
    return;
  }
  if (Phi) {
    OS << *Phi << "}";
    return;
  }
  OS << (IsMemory ? "Mem: " : "Def: ") << *Locations << " - " << *Expression
     << " }";
}

void DbgSSAPhi::print(raw_ostream &OS) const {
  OS << "DbgPhi ";
  for (auto &[BB, DV] : IncomingValues)
    OS << "[" << BB->BB.getName() << ", " << DV << "] ";
}

using AvailableValsTy = DenseMap<DbgSSABlock *, DbgValueDef>;

DebugSSAUpdater::DebugSSAUpdater(SmallVectorImpl<DbgSSAPhi *> *NewPHI)
    : InsertedPHIs(NewPHI) {}

void DebugSSAUpdater::initialize() { AV.clear(); }

bool DebugSSAUpdater::hasValueForBlock(DbgSSABlock *BB) const {
  return AV.count(BB);
}

DbgValueDef DebugSSAUpdater::findValueForBlock(DbgSSABlock *BB) const {
  return AV.lookup(BB);
}

void DebugSSAUpdater::addAvailableValue(DbgSSABlock *BB, DbgValueDef DV) {
  AV[BB] = DV;
}

DbgValueDef DebugSSAUpdater::getValueAtEndOfBlock(DbgSSABlock *BB) {
  DbgValueDef Res = getValueAtEndOfBlockInternal(BB);
  return Res;
}

DbgValueDef DebugSSAUpdater::getValueInMiddleOfBlock(DbgSSABlock *BB) {
  // If there is no definition of the renamed variable in this block, just use
  // 'getValueAtEndOfBlock' to do our work.
  if (!hasValueForBlock(BB))
    return getValueAtEndOfBlock(BB);

  // Otherwise, we have the hard case. Get the live-in values for each
  // predecessor.
  SmallVector<std::pair<DbgSSABlock *, DbgValueDef>, 8> PredValues;
  DbgValueDef SingularValue;

  bool IsFirstPred = true;
  for (DbgSSABlock *PredBB : BB->predecessors()) {
    DbgValueDef PredVal = getValueAtEndOfBlock(PredBB);
    PredValues.push_back(std::make_pair(PredBB, PredVal));

    // Compute SingularValue.
    if (IsFirstPred) {
      SingularValue = PredVal;
      IsFirstPred = false;
    } else if (!PredVal.agreesWith(SingularValue))
      SingularValue = DbgValueDef();
  }

  // If there are no predecessors, just return undef.
  if (PredValues.empty())
    return DbgValueDef();

  // Otherwise, if all the merged values are the same, just use it.
  if (!SingularValue.IsUndef)
    return SingularValue;

  // Ok, we have no way out, insert a new one now.
  DbgSSAPhi *InsertedPHI = BB->newPHI();

  // Fill in all the predecessors of the PHI.
  for (const auto &PredValue : PredValues)
    InsertedPHI->addIncoming(PredValue.first, PredValue.second);

  // See if the PHI node can be merged to a single value. This can happen in
  // loop cases when we get a PHI of itself and one other value.

  // If the client wants to know about all new instructions, tell it.
  if (InsertedPHIs)
    InsertedPHIs->push_back(InsertedPHI);

  LLVM_DEBUG(dbgs() << "  Inserted PHI: " << *InsertedPHI << "\n");
  return InsertedPHI;
}

DbgSSABlock *DbgSSABlockSuccIterator::operator*() {
  return Updater.getDbgSSABlock(*SuccIt);
}
DbgSSABlock *DbgSSABlockPredIterator::operator*() {
  return Updater.getDbgSSABlock(*PredIt);
}

namespace llvm {

template <> class SSAUpdaterTraits<DebugSSAUpdater> {
public:
  using BlkT = DbgSSABlock;
  using ValT = DbgValueDef;
  using PhiT = DbgSSAPhi;
  using BlkSucc_iterator = DbgSSABlockSuccIterator;

  static BlkSucc_iterator BlkSucc_begin(BlkT *BB) { return BB->succ_begin(); }
  static BlkSucc_iterator BlkSucc_end(BlkT *BB) { return BB->succ_end(); }

  class PHI_iterator {
  private:
    DbgSSAPhi *PHI;
    unsigned Idx;

  public:
    explicit PHI_iterator(DbgSSAPhi *P) // begin iterator
        : PHI(P), Idx(0) {}
    PHI_iterator(DbgSSAPhi *P, bool) // end iterator
        : PHI(P), Idx(PHI->getNumIncomingValues()) {}

    PHI_iterator &operator++() {
      ++Idx;
      return *this;
    }
    bool operator==(const PHI_iterator &X) const { return Idx == X.Idx; }
    bool operator!=(const PHI_iterator &X) const { return !operator==(X); }

    DbgValueDef getIncomingValue() { return PHI->getIncomingValue(Idx); }
    DbgSSABlock *getIncomingBlock() { return PHI->getIncomingBlock(Idx); }
  };

  static PHI_iterator PHI_begin(PhiT *PHI) { return PHI_iterator(PHI); }
  static PHI_iterator PHI_end(PhiT *PHI) { return PHI_iterator(PHI, true); }

  /// FindPredecessorBlocks - Put the predecessors of BB into the Preds
  /// vector.
  static void FindPredecessorBlocks(DbgSSABlock *BB,
                                    SmallVectorImpl<DbgSSABlock *> *Preds) {
    for (auto PredIt = BB->pred_begin(); PredIt != BB->pred_end(); ++PredIt)
      Preds->push_back(*PredIt);
  }

  /// GetPoisonVal - Get an undefined value of the same type as the value
  /// being handled.
  static DbgValueDef GetPoisonVal(DbgSSABlock *BB, DebugSSAUpdater *Updater) {
    return DbgValueDef();
  }

  /// CreateEmptyPHI - Create a new debug PHI entry for the specified block.
  static DbgSSAPhi *CreateEmptyPHI(DbgSSABlock *BB, unsigned NumPreds,
                                   DebugSSAUpdater *Updater) {
    DbgSSAPhi *PHI = BB->newPHI();
    return PHI;
  }

  /// AddPHIOperand - Add the specified value as an operand of the PHI for
  /// the specified predecessor block.
  static void AddPHIOperand(DbgSSAPhi *PHI, DbgValueDef Val,
                            DbgSSABlock *Pred) {
    PHI->addIncoming(Pred, Val);
  }

  /// ValueIsPHI - Check if a value is a PHI.
  static DbgSSAPhi *ValueIsPHI(DbgValueDef Val, DebugSSAUpdater *Updater) {
    return Val.Phi;
  }

  /// ValueIsNewPHI - Like ValueIsPHI but also check if the PHI has no source
  /// operands, i.e., it was just added.
  static DbgSSAPhi *ValueIsNewPHI(DbgValueDef Val, DebugSSAUpdater *Updater) {
    DbgSSAPhi *PHI = ValueIsPHI(Val, Updater);
    if (PHI && PHI->getNumIncomingValues() == 0)
      return PHI;
    return nullptr;
  }

  /// GetPHIValue - For the specified PHI instruction, return the value
  /// that it defines.
  static DbgValueDef GetPHIValue(DbgSSAPhi *PHI) { return PHI; }
};

} // end namespace llvm

/// Check to see if AvailableVals has an entry for the specified BB and if so,
/// return it. If not, construct SSA form by first calculating the required
/// placement of PHIs and then inserting new PHIs where needed.
DbgValueDef DebugSSAUpdater::getValueAtEndOfBlockInternal(DbgSSABlock *BB) {
  if (AV.contains(BB))
    return AV[BB];

  SSAUpdaterImpl<DebugSSAUpdater> Impl(this, &AV, InsertedPHIs);
  return Impl.GetValue(BB);
}

bool isContained(DIScope *Inner, DIScope *Outer) {
  if (Inner == Outer)
    return true;
  if (!Inner->getScope())
    return false;
  return isContained(Inner->getScope(), Outer);
}

void DbgValueRangeTable::addVariable(Function *F, DebugVariableAggregate DVA) {
  const DILocalVariable *Var = DVA.getVariable();
  const DILocation *InlinedAt = DVA.getInlinedAt();

  DenseMap<BasicBlock *, SmallVector<DbgVariableRecord *>> BlockDbgRecordValues;
  DenseSet<BasicBlock *> HasAnyInstructionsInScope;
  int NumRecordsFound = 0;
  DbgVariableRecord *LastRecordFound = nullptr;
  bool DeclareRecordFound = false;

  LLVM_DEBUG(dbgs() << "Finding variable info for " << *Var << " at "
                    << InlinedAt << "\n");

  for (auto &BB : *F) {
    auto &DbgRecordValues = BlockDbgRecordValues[&BB];
    bool FoundInstructionInScope = false;
    for (auto &I : BB) {
      LLVM_DEBUG(dbgs() << "Instruction: '" << I << "'\n");

      for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
        if (DVR.getVariable() == Var &&
            DVR.getDebugLoc().getInlinedAt() == InlinedAt) {
          assert(!DVR.isDbgAssign() && "No support for #dbg_assign yet.");
          if (DVR.isDbgDeclare())
            DeclareRecordFound = true;
          ++NumRecordsFound;
          LastRecordFound = &DVR;
          DbgRecordValues.push_back(&DVR);
        }
      }
      if (!FoundInstructionInScope && I.getDebugLoc()) {
        if (I.getDebugLoc().getInlinedAt() == InlinedAt &&
            isContained(cast<DILocalScope>(I.getDebugLoc().getScope()),
                        Var->getScope())) {
          FoundInstructionInScope = true;
          HasAnyInstructionsInScope.insert(&BB);
        }
      }
    }
    LLVM_DEBUG(dbgs() << "DbgRecordValues found in '" << BB.getName() << "':\n";
               for_each(DbgRecordValues, [](auto *DV) { DV->dump(); }));
  }

  if (!NumRecordsFound) {
    LLVM_DEBUG(dbgs() << "No dbg_records found for variable!\n");
    return;
  }

  // Now that we have all the DbgValues, we can start defining available values
  // for each block. The end goal is to have, for every block with any
  // instructions in scope, a LiveIn value.
  // Currently we anticipate that either a variable has a set of #dbg_values, in
  // which case we need a complete SSA liveness analysis to determine live-in
  // values per-block, or a variable has a single #dbg_declare.
  if (DeclareRecordFound) {
    // FIXME: This should be changed for fragments!
    LLVM_DEBUG(dbgs() << "Single location found for variable!\n");
    assert(NumRecordsFound == 1 &&
           "Found multiple records for a #dbg_declare variable!");
    OrigSingleLocVariableValueTable[DVA] = DbgValueDef(LastRecordFound);
    return;
  }

  // We don't have a single location for the variable's entire scope, so instead
  // we must now perform a liveness analysis to create a location list.
  DenseMap<BasicBlock *, DbgValueDef> LiveInMap;
  SmallVector<DbgSSAPhi *> HypotheticalPHIs;
  DebugSSAUpdater SSAUpdater(&HypotheticalPHIs);
  SSAUpdater.initialize();
  for (auto &[BB, DVs] : BlockDbgRecordValues) {
    auto *DbgBB = SSAUpdater.getDbgSSABlock(BB);
    if (DVs.empty())
      continue;
    auto *LastValueInBlock = DVs.back();
    LLVM_DEBUG(dbgs() << "Last value in " << BB->getName() << ": "
                      << *LastValueInBlock << "\n");
    SSAUpdater.addAvailableValue(DbgBB, DbgValueDef(LastValueInBlock));
  }

  for (BasicBlock &BB : *F) {
    if (!HasAnyInstructionsInScope.contains(&BB)) {
      LLVM_DEBUG(dbgs() << "Skipping finding debug ranges for '" << BB.getName()
                        << "' due to no in-scope instructions.\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "Finding live-in value for '" << BB.getName()
                      << "'...\n");
    DbgValueDef LiveValue =
        SSAUpdater.getValueInMiddleOfBlock(SSAUpdater.getDbgSSABlock(&BB));
    LLVM_DEBUG(dbgs() << "Found live-in: " << LiveValue << "\n");
    auto HasValidValue = [](DbgValueDef DV) {
      return !DV.IsUndef && DV.Phi == nullptr;
    };

    SmallVector<DbgRangeEntry> BlockDbgRanges;
    BasicBlock::iterator LastIt = BB.begin();
    for (auto *DVR : BlockDbgRecordValues[&BB]) {
      // Create a range that ends as of DVR.
      BasicBlock::iterator DVRStartIt =
          const_cast<Instruction *>(DVR->getInstruction())->getIterator();
      if (HasValidValue(LiveValue))
        BlockDbgRanges.push_back({LastIt, DVRStartIt, LiveValue});
      LiveValue = DbgValueDef(DVR);
      LastIt = DVRStartIt;
    }

    // After considering all in-block debug values, if any, create a range
    // covering the remainder of the block.
    if (HasValidValue(LiveValue))
      BlockDbgRanges.push_back({LastIt, BB.end(), LiveValue});
    LLVM_DEBUG(dbgs() << "Create set of ranges with " << BlockDbgRanges.size()
                      << " entries!\n");
    if (!BlockDbgRanges.empty())
      OrigVariableValueRangeTable[DVA].append(BlockDbgRanges);
  }
}

void DbgValueRangeTable::printValues(DebugVariableAggregate DVA,
                                     raw_ostream &OS) {
  OS << "Variable Table for '" << DVA.getVariable()->getName() << "' (at "
     << DVA.getInlinedAt() << "):\n";
  if (!hasVariableEntry(DVA)) {
    OS << "  Empty!\n";
    return;
  }
  if (hasSingleLocEntry(DVA)) {
    OS << "  SingleLoc: " << OrigSingleLocVariableValueTable[DVA] << "\n";
    return;
  }
  OS << "  LocRange:\n";
  for (DbgRangeEntry RangeEntry : OrigVariableValueRangeTable[DVA]) {
    OS << "    (";
    if (RangeEntry.Start == RangeEntry.Start->getParent()->begin() &&
        RangeEntry.End == RangeEntry.Start->getParent()->end()) {
      OS << RangeEntry.Start->getParent()->getName();
    } else {
      OS << RangeEntry.Start->getParent()->getName() << ": "
         << *RangeEntry.Start << ", ";
      if (RangeEntry.End == RangeEntry.Start->getParent()->end())
        OS << "..";
      else
        OS << *RangeEntry.End;
    }
    OS << ") [" << RangeEntry.Value << "]\n";
  }
}

SSAValueNameMap::ValueID SSAValueNameMap::addValue(Value *V) {
  auto ExistingID = ValueToIDMap.find(V);
  if (ExistingID != ValueToIDMap.end())
    return ExistingID->second;
  // First, get a new ID and Map V to it.
  ValueID NewID = NextID++;
  ValueToIDMap.insert({V, NewID});
  // Then, get the name string for V and map NewID to it.
  assert(!ValueIDToNameMap.contains(NewID) &&
         "New value ID already maps to a name?");
  std::string &ValueText = ValueIDToNameMap[NewID];
  raw_string_ostream Stream(ValueText);
  V->printAsOperand(Stream, true);
  return NewID;
}
