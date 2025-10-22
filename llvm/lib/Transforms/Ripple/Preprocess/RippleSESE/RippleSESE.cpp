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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Ripple/Ripple.h"
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

namespace llvm { class Value; }

using namespace llvm;

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
/// @param PDomTree Post-dominator tree of the CFG for whom "S" is to be
/// calculated.
/// @param From Begin point of the path search.
/// @param To End point of the path search.
DenseSet<BasicBlock *>
allBasicBlocksFromTo(PostDominatorTreeAnalysis::Result &PDomTree,
                     BasicBlock *From, BasicBlock *To) {
  assert(PDomTree.dominates(To, From));
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
    BranchInst *Branch = dyn_cast<BranchInst>(Last);
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
  auto BBsToClone = allBasicBlocksFromTo(PDT, BBWithExternalPred, BranchBBPdom);
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
    BasicBlock *BranchPDom = PDomTree.getNode(BranchBB)->getIDom()->getBlock();
    auto BBsInBetweenSet = allBasicBlocksFromTo(PDomTree, BranchBB, BranchPDom);
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
        DiagnosticInfoOptimizationFailure Diag(F, {}, ErrMsg);
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(), ErrMsg);
      }
      if (anyBBToBeClonedContainsIndirectBr(BB, BranchPDom, PDomTree)) {
        StringRef ErrMsg =
            "SESE-fication via cloning is not possible due to IndirectBrs "
            " on a cloning path. This violates the SESE requirement of Ripple.";
        DiagnosticInfoOptimizationFailure Diag(F, {}, ErrMsg);
        F.getContext().diagnose(Diag);
        return createStringError(inconvertibleErrorCode(), ErrMsg);
      }

      // Clone the basic blocks on the path from BB to its PostDom.
      auto BBsToClone = allBasicBlocksFromTo(PDomTree, BB, BranchPDom);
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
      for (auto *PredBB : predecessors(BB)) {
        if (BBsInBetweenSet.contains(PredBB) || PredBB == BranchBB) {
          BBPredsInBetween.push_back(PredBB);
          removeIncomingBlockFromPhis(PredBB, BB);
        } else {
          removeIncomingBlockFromPhis(PredBB, Clones[BB]);
        }
      }
      remapInstructionsInBlocks(BBPredsInBetween, VMap);

      // Use the values in VMap to map the values in the cloned basic blocks.
      remapInstructionsInBlocks(
          to_vector_of<BasicBlock *>(make_second_range(Clones)), VMap);

      // We only need to update the phis in PostDom, because no basic blocks
      // (expect already in clones) would be dominated by the cloned basic
      // blocks.
      for (auto &Phi : BranchPDom->phis()) {
        auto NOperands = Phi.getNumOperands();
        for (unsigned IOperand = 0; IOperand < NOperands; IOperand++) {
          auto *IthValue = Phi.getIncomingValue(IOperand);
          auto *IthBlock = Phi.getIncomingBlock(IOperand);
          if (Clones.contains(IthBlock)) {
            if (auto ClonedIthVal = VMap.find(IthValue); ClonedIthVal != VMap.end()) {
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
    return PA;
  }

  // Print the CFG after.
  LLVM_DEBUG(writeCFGToDotFile(F, AM, "ripplesese.after."););
  return PA;
}
