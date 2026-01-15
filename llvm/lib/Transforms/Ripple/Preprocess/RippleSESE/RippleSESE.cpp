//===----- RippleSESE.cpp: Ensure CFG satisfy Ripple's SESE criterion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Ripple/Preprocess/RippleSESE.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/HeatUtils.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/User.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include "llvm/Transforms/Ripple/SubgraphCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <queue>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace llvm {
class Value;
}

using namespace llvm;
using namespace llvm::subgraphcfg;

#define DEBUG_TYPE "ripple-sese"

////////////////////////////////////////////////////////////////////////////////
///                               RippleSESEPass                             ///
////////////////////////////////////////////////////////////////////////////////

namespace {

[[maybe_unused]]
void writeCFGToDotFile(Function &F, FunctionAnalysisManager &AM,
                       std::string FilePrefix = "cfg.", bool CFGOnly = false) {
  std::string Filename = (FilePrefix + F.getName() + ".dot").str();
  auto *BFI = &AM.getResult<BlockFrequencyAnalysis>(F);
  auto *BPI = &AM.getResult<BranchProbabilityAnalysis>(F);
  auto MaxFreq = getMaxFreq(F, BFI);

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);

  DOTFuncInfo CFGInfo(&F, BFI, BPI, MaxFreq);
  CFGInfo.setHeatColors(true);
  CFGInfo.setEdgeWeights(false);
  CFGInfo.setRawEdgeWeights(false);

  if (!EC)
    WriteGraph(File, &CFGInfo, CFGOnly);
  else
    errs() << "  error opening file for writing!";
}

/// @brief Returns a set "S" of basic blocks. A basic block "B" is in "S" iff
/// "B" lies on a path from \p From to \p To. \p From and \p To are not elements
/// of "S".
/// @param From Begin point of the path search.
/// @param To End point of the path search.
DenseSet<BasicBlock *> allBasicBlocksFromTo(BasicBlock *From, BasicBlock *To) {
  std::queue<BasicBlock *> ToProcess;
  DenseSet<BasicBlock *> Visited;
  ToProcess.push(From);
  while (!ToProcess.empty()) {
    BasicBlock *BB = ToProcess.front();
    ToProcess.pop();
    Visited.insert(BB);
    for (auto *SI : successors(BB)) {
      if (!Visited.contains(SI) && SI != To)
        ToProcess.push(SI);
    }
  }
  Visited.erase(From);
  return Visited;
}

/// @brief Records a mapping from intructions in \p F to their ripple shapes.
/// @return Error, if any, encountered during the shape propagation while
/// computing the intruction to ripple shape mapping.
Expected<std::map<AssertingVH<const Instruction>, TensorShape>>
getInstructionToRippleShape(
    TargetMachine *TM, Function &F, FunctionAnalysisManager &AM,
    Ripple::ProcessingStatus &PS,
    DenseSet<AssertingVH<Function>> &SpecializationsPending,
    DenseSet<AssertingVH<Function>> &SpecializationsAvailable) {
  // WIP machine model w/ 1 vector dimension
  std::vector<std::pair<Ripple::PEIdentifier, Ripple::DimType>> DimTypes = {
      {0, Ripple::VectorDimension}};

  Ripple R(TM, F, AM, DimTypes, PS, SpecializationsPending,
           SpecializationsAvailable);
  if (R.tensorRank() == 0) {
    // No ripple usage.
    return std::map<AssertingVH<const Instruction>, TensorShape>();
  }

  R.initFuncRPOT();

  R.loadRippleLibDeclarations();

  bool WaitingForSpecialization = false;
  if (Error E = R.propagateShapes(WaitingForSpecialization))
    return std::move(E);

  if (WaitingForSpecialization)
    return std::map<AssertingVH<const Instruction>, TensorShape>();

  return R.getInstructionToRippleShape();
}

/// @brief Updates \p BBsWithVectorBranch with all basic blocks in \p F that
/// have a branch with a non-scalar ripple shape.
Expected<std::vector<BasicBlock *>> getNonScalarRippleShapedBranches(
    TargetMachine *TM, Function &F, FunctionAnalysisManager &AM,
    DominatorTreeAnalysis::Result &DT, PostDominatorTreeAnalysis::Result &PDT,
    Ripple::ProcessingStatus &PS,
    DenseSet<AssertingVH<Function>> &SpecializationsPending,
    DenseSet<AssertingVH<Function>> &SpecializationsAvailable) {
  std::vector<BasicBlock *> BBsWithVectorBranch;

  auto ExpectedInstToRippleShape = getInstructionToRippleShape(
      TM, F, AM, PS, SpecializationsPending, SpecializationsAvailable);
  if (!ExpectedInstToRippleShape) {
    return ExpectedInstToRippleShape.takeError();
  }
  auto &InstructionToRippleShape = *ExpectedInstToRippleShape;

  if (InstructionToRippleShape.empty())
    // No ripple-ids were found in the function => no SESE-fication needed.
    return BBsWithVectorBranch;

  ReversePostOrderTraversal<Function *> *FuncRPOT =
      new ReversePostOrderTraversal<Function *>(&F);

  for (auto *BB : *FuncRPOT) {
    if (BB->back().isSpecialTerminator())
      continue;
    Value *Last = BB->getTerminator();
    CondBrInst *Branch = dyn_cast<CondBrInst>(Last);
    SwitchInst *Switch = dyn_cast<SwitchInst>(Last);
    if (Branch) {
      if (InstructionToRippleShape.at(Branch).isVector())
        if (!hasTrivialLoopLikeBackEdge(
                BB, PDT.getNode(BB)->getIDom()->getBlock(), DT))
          BBsWithVectorBranch.push_back(BB);
    } else if (Switch) {
      if (InstructionToRippleShape.at(Switch).isVector())
        BBsWithVectorBranch.push_back(BB);
    }
  }

  delete FuncRPOT;
  return BBsWithVectorBranch;
}

template <class C>
std::vector<BasicBlock *>
postDomOrderBasicBlocks(C BBContainer,
                        PostDominatorTreeAnalysis::Result &PDomTree) {
  std::vector<BasicBlock *> CVec(BBContainer.begin(), BBContainer.end());
  std::sort(CVec.begin(), CVec.end(),
            [&](const BasicBlock *BB1, const BasicBlock *BB2) {
              return (PDomTree.getNode(BB1)->getLevel() <
                      PDomTree.getNode(BB2)->getLevel());
            });
  return CVec;
}

/// @brief Returns *True* iff the SESEfication cloning process in \ref
/// fixCFGForSESE will terminate.
/// @param BranchBB The basic block with the ripple-id dependent conditional.
/// @param BranchBBPdom Immediate post-dominator of BranchBB.
/// @param BBWithExternalPred The basic block on the path from BranchBB to
/// BranchBBPdom that has an external predecessor i.e. the basic block from
/// which the cloning must commence.
bool willSESEficationCloningTerminate(const BasicBlock *BranchBB,
                                      const BasicBlock *BBWithExternalPred,
                                      BasicBlock *BranchBBPdom,
                                      DominatorTreeAnalysis::Result &DT) {
  const SmallPtrSet<BasicBlock *, 1> ExclusionSet({BranchBBPdom});

  // If there is a cycle between a BB from which the BBs are to be cloned, we
  // require that within the SESE-region it should not be a part of a cycle as
  // it would lead the cloning process to run indefinitely.
  return !isPotentiallyReachable(BBWithExternalPred, BranchBB, &ExclusionSet,
                                 &DT);
}

/// @brief Returns *True* iff the basic blocks to be cloned in \ref
/// fixCFGForSESE contain IndirectBrs.
/// @param BBWithExternalPred The basic block on the path from BranchBB to
/// BranchBBPdom that has an external predecessor i.e. the basic block from
/// which the cloning must commence.
/// @param BranchBBPdom Immediate post-dominator of SESE region in if-convert.
bool anyBBToBeClonedContainsIndirectBr(BasicBlock *BBWithExternalPred,
                                       BasicBlock *BranchBBPdom,
                                       PostDominatorTreeAnalysis::Result &PDT) {
  assert(PDT.dominates(BranchBBPdom, BBWithExternalPred));
  auto BBsToClone = allBasicBlocksFromTo(BBWithExternalPred, BranchBBPdom);
  auto DoesBBTerminateWithIndirectBr = [](BasicBlock *BBToCheck) {
    return isa<IndirectBrInst>(BBToCheck->getTerminator());
  };
  for (auto *PredBB : predecessors(BBWithExternalPred)) {
    if (DoesBBTerminateWithIndirectBr(PredBB))
      return true;
  }
  for (auto *BBToClone : BBsToClone) {
    if (DoesBBTerminateWithIndirectBr(BBToClone))
      return true;
  }
  return false;
}

/// @brief Updates the CFG in "F", so that all branches that have non-scalar
/// ripple shapes comprise an SESE sub-CFG.
/// For example, consider the following CFG with the basic block "A" having a
/// non-scalar shape ripple shape.
///
///         entry
///           ▼
///        ┌──A──┐
///        ▼     ▼   G
///        B     C ◄─┘
///        ▼     ▼
///        D     E
///        └► F ◄┘
///           ▼
///           H
///
/// We observe that the sub-CFG associated with the ripple vector branch "A" is
/// non-SESE since "C" has an incoming edge from "G" which is not on any paths
/// from "A" to its immediate post-dominator, "F". This violates one of the
/// requirements for "-ripple" pass.  To ensure that "-ripple" can be applied to
/// the CFGs of above kind, we clone certain basic blocks to comply with
/// ripple's SESE criterion.
///
/// The key transformation in fixCFGForSESE comprises of appropriately cloning
/// the basic block paths starting within the sub-CFG that have predecessors
/// outside the sub-CFG. After applying \ref fixCFGForSESE to our example CFG,
/// we will obtain:
///         entry
///           ▼
///        ┌──A──┐   G
///        ▼     ▼   ▼
///        B    C'   C
///        ▼     ▼   ▼
///        D    E'   E
///        └► F ◄┘───┘
///           ▼
///           H
/// where, C' and E' are the basic blocks cloned from C and E respectively.
/// Further, the phi-nodes in the immediate post-dominator, F, are updated to
/// ensure that the transformed CFG is consistent with the untransformed one.
Error fixCFGForSESE(TargetMachine *TM, Function &F, FunctionAnalysisManager &AM,
                    Ripple::ProcessingStatus &PS,
                    DenseSet<AssertingVH<Function>> &SpecializationsPending,
                    DenseSet<AssertingVH<Function>> &SpecializationsAvailable) {

  DominatorTreeAnalysis::Result &DomTree =
      AM.getResult<DominatorTreeAnalysis>(F);
  PostDominatorTreeAnalysis::Result &PDomTree =
      AM.getResult<PostDominatorTreeAnalysis>(F);
  DomTreeUpdater DTU(DomTree, PDomTree, DomTreeUpdater::UpdateStrategy::Lazy);
  bool AreDomTreesValid =
      true; // TODO: Maintain std::vector<DominatorTree::Updates> instead.

  auto ExpectedBBsWithVectorBranch = getNonScalarRippleShapedBranches(
      TM, F, AM, DomTree, PDomTree, PS, SpecializationsPending,
      SpecializationsAvailable);
  if (!ExpectedBBsWithVectorBranch) {
    return ExpectedBBsWithVectorBranch.takeError();
  }
  std::vector<BasicBlock *> &BBsWithVectorBranch = *ExpectedBBsWithVectorBranch;

  for (auto It = BBsWithVectorBranch.rbegin(); It != BBsWithVectorBranch.rend();
       It++) {

    if (!AreDomTreesValid) {
      DTU.recalculate(F);
      AreDomTreesValid = true;
    }

    auto *BranchBB = *It;
    auto *Node = PDomTree.getNode(BranchBB)->getIDom();
    if (PDomTree.isVirtualRoot(Node)) {
      LLVM_DEBUG(dbgs() << "Pdom is virtual\n");
      // Fix unreachable paths, if possible
      SubgraphCFG CFGNoUnreachable(F, getAllBBsLeadingTo<UnreachableInst>(F));
      auto *BranchBBInSubgraph = CFGNoUnreachable.get(BranchBB);
      // TODO: we can probably support this case too but it will require a
      // rework of the if-convert pass
      if (!BranchBBInSubgraph) {
        DiagnosticInfoOptimizationFailure Diag(
            F, BranchBB->getTerminator()->getDebugLoc(),
            "this vector conditional is part of a program path that always "
            "leads to the function's exit through non-return paths (e.g., "
            "assert); please make sure that at least one path leads to the "
            "function exit");
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(),
                                 "Cannot fix the CFG for unreachable");
      }
      PostDomTreeBase<SubgraphBB> FilteredPdomTree;
      FilteredPdomTree.recalculate(CFGNoUnreachable);
      LLVM_DEBUG(dbgs() << "PdomTree without unreachable paths:\n";
                 FilteredPdomTree.print(dbgs()));
      auto *BranchBBNodeIDom =
          FilteredPdomTree.getNode(BranchBBInSubgraph)->getIDom();
      if (FilteredPdomTree.isVirtualRoot(BranchBBNodeIDom)) {
        DiagnosticInfoOptimizationFailure Diag(
            F, BranchBB->getTerminator()->getDebugLoc(),
            "this vector conditional has paths that will terminate the "
            "function for some lanes and not for others; please make sure not "
            "to use exceptions without catching before the function return "
            "paths");
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(),
                                 "Cannot fix the CFG for unreachable");
      }
      BasicBlock *NoUnreachPdom = BranchBBNodeIDom->getBlock()->BB;

      auto BBsInBetweenSet = allBasicBlocksFromTo(BranchBB, NoUnreachPdom);
      SmallPtrSet<BasicBlock *, 16> UnreachablePaths;
      UnreachablePaths.insert_range(
          llvm::make_filter_range(BBsInBetweenSet, [&](BasicBlock *BB) {
            return !CFGNoUnreachable.get(BB);
          }));
      SmallPtrSet<BasicBlock *, 16> CloneForSubgraph;
      for (auto *UnreachableBB : UnreachablePaths) {
        LLVM_DEBUG(dbgs() << "BB part of unreachable path: " << *UnreachableBB
                          << "\n");
        // We need to clone the sub-graph if there is a path coming from outside
        // this SESE region
        for (auto *Pred : predecessors(UnreachableBB))
          if (!BBsInBetweenSet.contains(Pred) && Pred != BranchBB)
            CloneForSubgraph.insert(UnreachableBB);
      }
      bool Changed = !CloneForSubgraph.empty();
      while (Changed) {
        Changed = false;
        for (auto *ToClone : CloneForSubgraph) {
          for (auto *Succ : successors(ToClone)) {
            assert(BBsInBetweenSet.contains(Succ));
            CloneForSubgraph.insert(Succ);
            Changed = true;
          }
          if (Changed)
            break;
        }
      }

      for ([[maybe_unused]] auto *ToClone : CloneForSubgraph)
        LLVM_DEBUG(dbgs() << "BB needing clone of unreachable path: "
                          << *ToClone << "\n");

      if (!CloneForSubgraph.empty()) {
        ValueToValueMapTy VMap;
        // Clone
        for (auto *BB : CloneForSubgraph)
          VMap[BB] = CloneBasicBlock(BB, VMap, ".cloned", &F);
        // Remap
        for (auto *BB : CloneForSubgraph)
          remapInstructionsInBlocks(cast<BasicBlock>(VMap[BB]), VMap);
        // Updates to the Dom/Pdom trees for newly created nodes
        for (auto *BB : CloneForSubgraph)
          for (auto *Succ : successors(BB)) {
            assert(llvm::find(CloneForSubgraph, Succ) !=
                       CloneForSubgraph.end() &&
                   "Did not clone the whole sub-graph");
            DTU.applyUpdates({{DominatorTree::Insert, BB, Succ}});
          }

        // Fix the Phis
        for (auto *BB : CloneForSubgraph) {
          // Remove predes of the original BB for edges coming from this
          // sub-graph and from the vector branch
          auto comesFromRegion = [&](const BasicBlock *BB) -> bool {
            return BBsInBetweenSet.contains(BB) || BB == BranchBB;
          };
          for (PHINode &Phi : BB->phis())
            Phi.removeIncomingValueIf(
                [&](unsigned Idx) {
                  return comesFromRegion(Phi.getIncomingBlock(Idx));
                },
                /*RemovePhiIfEmpty*/ false);
          // Remove values coming from outside this sub-graph
          auto *ClonedBB = cast<BasicBlock>(VMap[BB]);
          for (PHINode &Phi : ClonedBB->phis())
            Phi.removeIncomingValueIf(
                [&](unsigned Idx) {
                  return not comesFromRegion(Phi.getIncomingBlock(Idx));
                },
                /*RemovePhiIfEmpty*/ false);
        }

        auto makeSubgraphPointToClones = [&](BasicBlock &BB) {
          SmallPtrSet<BasicBlock *, 8> SuccsBefore;
          SuccsBefore.insert_range(successors(&BB));

          Instruction &Inst = *BB.getTerminator();
          RemapDbgRecordRange(Inst.getModule(), Inst.getDbgRecordRange(), VMap,
                              RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
          RemapInstruction(&Inst, VMap,
                           RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

          SmallPtrSet<BasicBlock *, 8> SuccsAfter;
          SuccsAfter.insert_range(successors(&BB));
          for (auto *Before : SuccsBefore)
            if (!SuccsAfter.contains(Before))
              DTU.applyUpdates({{DominatorTree::Delete, &BB, Before}});
          for (auto *After : SuccsAfter)
            if (!SuccsBefore.contains(After))
              DTU.applyUpdates({{DominatorTree::Insert, &BB, After}});
        };
        // Fix the branches in BBsInBetweenSet to point to the cloned BBs
        for (auto BB : make_filter_range(BBsInBetweenSet, [&](auto *Basic) {
               return !CloneForSubgraph.contains(Basic);
             })) {
          makeSubgraphPointToClones(*BB);
        }
        makeSubgraphPointToClones(*BranchBB);
        for (auto *BB : CloneForSubgraph) {
          UnreachablePaths.erase(BB);
          UnreachablePaths.insert(cast<BasicBlock>(VMap[BB]));
        }
        // Now we have unreachable paths that only enter through the branch
      }

      for (auto *BB : UnreachablePaths) {
        if (successors(BB).empty()) {
          LLVM_DEBUG(dbgs() << "Fixing BB\n");
          auto &Unreach = BB->back();
          assert(isa<UnreachableInst>(Unreach));
          IRBuilder<> IRB(BB);
          IRB.CreateBr(NoUnreachPdom);
          Unreach.eraseFromParent();
          LLVM_DEBUG(dbgs() << "Fixed BB: " << *BB << "\n");
          // This branch will never be taken but we have to fix the PHI to have
          // a valid CFG for the if-conversion
          for (auto &Phi : NoUnreachPdom->phis())
            Phi.addIncoming(UndefValue::get(Phi.getType()), BB);
          DTU.applyUpdates({{DominatorTree::Insert, BB, NoUnreachPdom}});
        }
      }
      LLVM_DEBUG(dbgs() << "Function after unreachable fix:\n";
                 F.print(dbgs()));
      DTU.flush();
      LLVM_DEBUG(assert(DomTree.verify()); DomTree.print(dbgs()));
      LLVM_DEBUG(assert(PDomTree.verify()); PDomTree.print(dbgs()));
      assert(!verifyFunction(F, &errs()));
      Node = PDomTree.getNode(BranchBB)->getIDom();
      if (PDomTree.isVirtualRoot(Node))
        llvm_unreachable("It did not work haha");
    }
    BasicBlock *BranchPDom = Node->getBlock();
    assert(PDomTree.dominates(BranchPDom, BranchBB));
    auto BBsInBetweenSet = allBasicBlocksFromTo(BranchBB, BranchPDom);
    auto BBsInBetween = postDomOrderBasicBlocks(BBsInBetweenSet, PDomTree);

    for (auto *BB : BBsInBetween) {
      bool HasIncomingEdgeFromOutside =
          std::any_of(pred_begin(BB), pred_end(BB), [&](BasicBlock *PredBB) {
            return !(BBsInBetweenSet.contains(PredBB) || PredBB == BranchBB);
          });
      if (!HasIncomingEdgeFromOutside)
        continue;

      if (!AreDomTreesValid) {
        DTU.recalculate(F);
        AreDomTreesValid = true;
      }

      LLVM_DEBUG(dbgs() << "'" << BB->getName()
                        << "' has an outside pred in the sub-CFG from '"
                        << BranchBB->getName() << "' to '"
                        << BranchPDom->getName() << "'.\n";);

      if (!willSESEficationCloningTerminate(BranchBB, BB, BranchPDom,
                                            DomTree)) {
        StringRef ErrMsg =
            "SESE-fication via cloning will not terminate due to back-edges."
            " This violates the SESE requirement of Ripple.";
        DiagnosticInfoOptimizationFailure Diag(
            F, BranchBB->getTerminator()->getDebugLoc(), ErrMsg);
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(), ErrMsg);
      }
      if (anyBBToBeClonedContainsIndirectBr(BB, BranchPDom, PDomTree)) {
        StringRef ErrMsg =
            "SESE-fication via cloning is not possible due to IndirectBrs "
            " on a cloning path. This violates the SESE requirement of Ripple.";
        DiagnosticInfoOptimizationFailure Diag(
            F, BranchBB->getTerminator()->getDebugLoc(), ErrMsg);
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(), ErrMsg);
      }

      // Clone the basic blocks on the path from BB to its PostDom.
      assert(PDomTree.dominates(BranchPDom, BB));
      auto BBsToClone = allBasicBlocksFromTo(BB, BranchPDom);
      BBsToClone.insert(BB);
      ValueToValueMapTy VMap;
      DenseMap<BasicBlock *, BasicBlock *> Clones;
      for (auto *BBToClone : BBsToClone) {
        auto *ClonedBB = CloneBasicBlock(BBToClone, VMap, ".cloned", &F);
        Clones.insert({BBToClone, ClonedBB});
        VMap.insert({BBToClone, ClonedBB});
      }

      // The cloned BBs must become exclusively part of the SESE, and the
      // un-cloned BBs lie outside the SESE. Update the predecessor of the basic
      // block with incoming edge from outside to make that happen.
      SmallVector<BasicBlock *> BBPredsInBetween;
      for (auto *BBToClone : BBsToClone) {
        for (auto *PredBB : predecessors(BBToClone)) {
          if (BBsInBetweenSet.contains(PredBB) || PredBB == BranchBB) {
            if (!BBsToClone.contains(PredBB)) {
              BBPredsInBetween.push_back(PredBB);
              removeIncomingBlockFromPhis(PredBB, BBToClone);
            }
          }
        }
      }
      remapInstructionsInBlocks(BBPredsInBetween, VMap);

      // Use the values in VMap to map the values in the cloned basic blocks.
      remapInstructionsInBlocks(
          to_vector_of<BasicBlock *>(make_second_range(Clones)), VMap);

      // In the clone of the BB with external predecessor, remove external preds
      // from it.
      for (auto *PredBB : predecessors(BB))
        if (!(BBsInBetweenSet.contains(PredBB) || PredBB == BranchBB))
          removeIncomingBlockFromPhis(PredBB, Clones[BB]);

      // We only need to update the phis in PostDom, because no basic blocks
      // (expect already in clones) would be dominated by the cloned basic
      // blocks.
      for (auto &Phi : BranchPDom->phis()) {
        auto NOperands = Phi.getNumOperands();
        for (unsigned IOperand = 0; IOperand < NOperands; IOperand++) {
          auto *IthValue = Phi.getIncomingValue(IOperand);
          auto *IthBlock = Phi.getIncomingBlock(IOperand);
          if (Clones.contains(IthBlock)) {
            if (auto ClonedIthVal = VMap.find(IthValue);
                ClonedIthVal != VMap.end()) {
              Phi.addIncoming(ClonedIthVal->second, Clones.at(IthBlock));
            } else {
              // Value defined in a dominator of IthBlock -> has not been
              // changed during SESE-fication.
              Phi.addIncoming(IthValue, Clones.at(IthBlock));
            }
          }
        }
      }
      AreDomTreesValid = false;
    }
  }
  return Error::success();
}

} // namespace

// Entrypoint for this pass.
PreservedAnalyses RippleSESEPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<TargetLibraryAnalysis>();
  // Print the CFG before.
  LLVM_DEBUG(writeCFGToDotFile(F, AM, "ripplesese.before."););

  if (Error E = fixCFGForSESE(TM, F, AM, PS, SpecializationsPending,
                              SpecializationsAvailable)) {
    DiagnosticInfoOptimizationFailure Diag(
        F, {}, "ripple-sese failed for " + F.getName() + ".");
    F.getContext().diagnose(Diag);
    llvm::consumeError(std::move(E));
    PS = Ripple::ProcessingStatus::SemanticsCheckFailure;
    return PA;
  }

  // Print the CFG after.
  LLVM_DEBUG(writeCFGToDotFile(F, AM, "ripplesese.after."););

  LLVM_DEBUG({
    if (verifyFunction(F, &errs())) {
      dbgs() << "Function verification failed after ripple-sese pass: "
             << F.getName() << "\n";
    } else {
      dbgs() << "Function verified successfully: " << F.getName() << "\n";
    }
  });
  PS = Ripple::ProcessingStatus::Success;

  return PA;
}
