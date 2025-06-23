//===- FunctionPropertiesAnalysis.cpp - Function Properties Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <deque>

using namespace llvm;

namespace llvm {
LLVM_ABI cl::opt<bool> EnableDetailedFunctionProperties(
    "enable-detailed-function-properties", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to compute detailed function properties."));

static cl::opt<unsigned> BigBasicBlockInstructionThreshold(
    "big-basic-block-instruction-threshold", cl::Hidden, cl::init(500),
    cl::desc("The minimum number of instructions a basic block should contain "
             "before being considered big."));

static cl::opt<unsigned> MediumBasicBlockInstructionThreshold(
    "medium-basic-block-instruction-threshold", cl::Hidden, cl::init(15),
    cl::desc("The minimum number of instructions a basic block should contain "
             "before being considered medium-sized."));
} // namespace llvm

static cl::opt<unsigned> CallWithManyArgumentsThreshold(
    "call-with-many-arguments-threshold", cl::Hidden, cl::init(4),
    cl::desc("The minimum number of arguments a function call must have before "
             "it is considered having many arguments."));

namespace {
int64_t getNumBlocksFromCond(const BasicBlock &BB) {
  int64_t Ret = 0;
  if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
    if (BI->isConditional())
      Ret += BI->getNumSuccessors();
  } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
    Ret += (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
  }
  return Ret;
}

int64_t getUses(const Function &F) {
  return ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();
}
} // namespace

void FunctionPropertiesInfo::reIncludeBB(const BasicBlock &BB) {
  updateForBB(BB, +1);
}

void FunctionPropertiesInfo::updateForBB(const BasicBlock &BB,
                                         int64_t Direction) {
  assert(Direction == 1 || Direction == -1);
  BasicBlockCount += Direction;
  BlocksReachedFromConditionalInstruction +=
      (Direction * getNumBlocksFromCond(BB));
  for (const auto &I : BB) {
    if (auto *CS = dyn_cast<CallBase>(&I)) {
      const auto *Callee = CS->getCalledFunction();
      if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
        DirectCallsToDefinedFunctions += Direction;
    }
    if (I.getOpcode() == Instruction::Load) {
      LoadInstCount += Direction;
    } else if (I.getOpcode() == Instruction::Store) {
      StoreInstCount += Direction;
    }
  }
  TotalInstructionCount += Direction * BB.sizeWithoutDebug();

  if (EnableDetailedFunctionProperties) {
    unsigned SuccessorCount = succ_size(&BB);
    if (SuccessorCount == 1)
      BasicBlocksWithSingleSuccessor += Direction;
    else if (SuccessorCount == 2)
      BasicBlocksWithTwoSuccessors += Direction;
    else if (SuccessorCount > 2)
      BasicBlocksWithMoreThanTwoSuccessors += Direction;

    unsigned PredecessorCount = pred_size(&BB);
    if (PredecessorCount == 1)
      BasicBlocksWithSinglePredecessor += Direction;
    else if (PredecessorCount == 2)
      BasicBlocksWithTwoPredecessors += Direction;
    else if (PredecessorCount > 2)
      BasicBlocksWithMoreThanTwoPredecessors += Direction;

    if (TotalInstructionCount > BigBasicBlockInstructionThreshold)
      BigBasicBlocks += Direction;
    else if (TotalInstructionCount > MediumBasicBlockInstructionThreshold)
      MediumBasicBlocks += Direction;
    else
      SmallBasicBlocks += Direction;

    // Calculate critical edges by looking through all successors of a basic
    // block that has multiple successors and finding ones that have multiple
    // predecessors, which represent critical edges.
    if (SuccessorCount > 1) {
      for (const auto *Successor : successors(&BB)) {
        if (pred_size(Successor) > 1)
          CriticalEdgeCount += Direction;
      }
    }

    ControlFlowEdgeCount += Direction * SuccessorCount;

    if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (!BI->isConditional())
        UnconditionalBranchCount += Direction;
    }

    for (const Instruction &I : BB.instructionsWithoutDebug()) {
      if (I.isCast())
        CastInstructionCount += Direction;

      if (I.getType()->isFloatTy())
        FloatingPointInstructionCount += Direction;
      else if (I.getType()->isIntegerTy())
        IntegerInstructionCount += Direction;

      if (isa<IntrinsicInst>(I))
        ++IntrinsicCount;

      if (const auto *Call = dyn_cast<CallInst>(&I)) {
        if (Call->isIndirectCall())
          IndirectCallCount += Direction;
        else
          DirectCallCount += Direction;

        if (Call->getType()->isIntegerTy())
          CallReturnsIntegerCount += Direction;
        else if (Call->getType()->isFloatingPointTy())
          CallReturnsFloatCount += Direction;
        else if (Call->getType()->isPointerTy())
          CallReturnsPointerCount += Direction;
        else if (Call->getType()->isVectorTy()) {
          if (Call->getType()->getScalarType()->isIntegerTy())
            CallReturnsVectorIntCount += Direction;
          else if (Call->getType()->getScalarType()->isFloatingPointTy())
            CallReturnsVectorFloatCount += Direction;
          else if (Call->getType()->getScalarType()->isPointerTy())
            CallReturnsVectorPointerCount += Direction;
        }

        if (Call->arg_size() > CallWithManyArgumentsThreshold)
          CallWithManyArgumentsCount += Direction;

        for (const auto &Arg : Call->args()) {
          if (Arg->getType()->isPointerTy()) {
            CallWithPointerArgumentCount += Direction;
            break;
          }
        }
      }

#define COUNT_OPERAND(OPTYPE)                                                  \
  if (isa<OPTYPE>(Operand)) {                                                  \
    OPTYPE##OperandCount += Direction;                                         \
    continue;                                                                  \
  }

      for (unsigned int OperandIndex = 0; OperandIndex < I.getNumOperands();
           ++OperandIndex) {
        Value *Operand = I.getOperand(OperandIndex);
        COUNT_OPERAND(GlobalValue)
        COUNT_OPERAND(ConstantInt)
        COUNT_OPERAND(ConstantFP)
        COUNT_OPERAND(Constant)
        COUNT_OPERAND(Instruction)
        COUNT_OPERAND(BasicBlock)
        COUNT_OPERAND(InlineAsm)
        COUNT_OPERAND(Argument)

        // We only get to this point if we haven't matched any of the other
        // operand types.
        UnknownOperandCount += Direction;
      }

#undef CHECK_OPERAND
    }
  }

  if (IR2VecVocab) {
    // We instantiate the IR2Vec embedder each time, as having an unique
    // pointer to the embedder as member of the class would make it
    // non-copyable. Instantiating the embedder in itself is not costly.
    auto EmbOrErr = ir2vec::Embedder::create(IR2VecKind::Symbolic,
                                             *BB.getParent(), *IR2VecVocab);
    if (Error Err = EmbOrErr.takeError()) {
      handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
        BB.getContext().emitError("Error creating IR2Vec embeddings: " +
                                  EI.message());
      });
      return;
    }
    auto Embedder = std::move(*EmbOrErr);
    const auto &BBEmbedding = Embedder->getBBVector(BB);
    // Subtract BBEmbedding from Function embedding if the direction is -1,
    // and add it if the direction is +1.
    if (Direction == -1)
      FunctionEmbedding -= BBEmbedding;
    else
      FunctionEmbedding += BBEmbedding;
  }
}

void FunctionPropertiesInfo::updateAggregateStats(const Function &F,
                                                  const LoopInfo &LI) {

  Uses = getUses(F);
  TopLevelLoopCount = llvm::size(LI);
  MaxLoopDepth = 0;
  std::deque<const Loop *> Worklist;
  llvm::append_range(Worklist, LI);
  while (!Worklist.empty()) {
    const auto *L = Worklist.front();
    MaxLoopDepth =
        std::max(MaxLoopDepth, static_cast<int64_t>(L->getLoopDepth()));
    Worklist.pop_front();
    llvm::append_range(Worklist, L->getSubLoops());
  }
}

FunctionPropertiesInfo FunctionPropertiesInfo::getFunctionPropertiesInfo(
    Function &F, FunctionAnalysisManager &FAM) {
  // We use the cached result of the IR2VecVocabAnalysis run by
  // InlineAdvisorAnalysis. If the IR2VecVocabAnalysis is not run, we don't
  // use IR2Vec embeddings.
  auto VocabResult = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F)
                         .getCachedResult<IR2VecVocabAnalysis>(*F.getParent());
  return getFunctionPropertiesInfo(F, FAM.getResult<DominatorTreeAnalysis>(F),
                                   FAM.getResult<LoopAnalysis>(F), VocabResult);
}

FunctionPropertiesInfo FunctionPropertiesInfo::getFunctionPropertiesInfo(
    const Function &F, const DominatorTree &DT, const LoopInfo &LI,
    const IR2VecVocabResult *VocabResult) {

  FunctionPropertiesInfo FPI;
  if (VocabResult && VocabResult->isValid()) {
    FPI.IR2VecVocab = VocabResult->getVocabulary();
    FPI.FunctionEmbedding = ir2vec::Embedding(VocabResult->getDimension(), 0.0);
  }
  for (const auto &BB : F)
    if (DT.isReachableFromEntry(&BB))
      FPI.reIncludeBB(BB);
  FPI.updateAggregateStats(F, LI);
  return FPI;
}

bool FunctionPropertiesInfo::operator==(
    const FunctionPropertiesInfo &FPI) const {
  if (BasicBlockCount != FPI.BasicBlockCount ||
      BlocksReachedFromConditionalInstruction !=
          FPI.BlocksReachedFromConditionalInstruction ||
      Uses != FPI.Uses ||
      DirectCallsToDefinedFunctions != FPI.DirectCallsToDefinedFunctions ||
      LoadInstCount != FPI.LoadInstCount ||
      StoreInstCount != FPI.StoreInstCount ||
      MaxLoopDepth != FPI.MaxLoopDepth ||
      TopLevelLoopCount != FPI.TopLevelLoopCount ||
      TotalInstructionCount != FPI.TotalInstructionCount ||
      BasicBlocksWithSingleSuccessor != FPI.BasicBlocksWithSingleSuccessor ||
      BasicBlocksWithTwoSuccessors != FPI.BasicBlocksWithTwoSuccessors ||
      BasicBlocksWithMoreThanTwoSuccessors !=
          FPI.BasicBlocksWithMoreThanTwoSuccessors ||
      BasicBlocksWithSinglePredecessor !=
          FPI.BasicBlocksWithSinglePredecessor ||
      BasicBlocksWithTwoPredecessors != FPI.BasicBlocksWithTwoPredecessors ||
      BasicBlocksWithMoreThanTwoPredecessors !=
          FPI.BasicBlocksWithMoreThanTwoPredecessors ||
      BigBasicBlocks != FPI.BigBasicBlocks ||
      MediumBasicBlocks != FPI.MediumBasicBlocks ||
      SmallBasicBlocks != FPI.SmallBasicBlocks ||
      CastInstructionCount != FPI.CastInstructionCount ||
      FloatingPointInstructionCount != FPI.FloatingPointInstructionCount ||
      IntegerInstructionCount != FPI.IntegerInstructionCount ||
      ConstantIntOperandCount != FPI.ConstantIntOperandCount ||
      ConstantFPOperandCount != FPI.ConstantFPOperandCount ||
      ConstantOperandCount != FPI.ConstantOperandCount ||
      InstructionOperandCount != FPI.InstructionOperandCount ||
      BasicBlockOperandCount != FPI.BasicBlockOperandCount ||
      GlobalValueOperandCount != FPI.GlobalValueOperandCount ||
      InlineAsmOperandCount != FPI.InlineAsmOperandCount ||
      ArgumentOperandCount != FPI.ArgumentOperandCount ||
      UnknownOperandCount != FPI.UnknownOperandCount ||
      CriticalEdgeCount != FPI.CriticalEdgeCount ||
      ControlFlowEdgeCount != FPI.ControlFlowEdgeCount ||
      UnconditionalBranchCount != FPI.UnconditionalBranchCount ||
      IntrinsicCount != FPI.IntrinsicCount ||
      DirectCallCount != FPI.DirectCallCount ||
      IndirectCallCount != FPI.IndirectCallCount ||
      CallReturnsIntegerCount != FPI.CallReturnsIntegerCount ||
      CallReturnsFloatCount != FPI.CallReturnsFloatCount ||
      CallReturnsPointerCount != FPI.CallReturnsPointerCount ||
      CallReturnsVectorIntCount != FPI.CallReturnsVectorIntCount ||
      CallReturnsVectorFloatCount != FPI.CallReturnsVectorFloatCount ||
      CallReturnsVectorPointerCount != FPI.CallReturnsVectorPointerCount ||
      CallWithManyArgumentsCount != FPI.CallWithManyArgumentsCount ||
      CallWithPointerArgumentCount != FPI.CallWithPointerArgumentCount) {
    return false;
  }
  // Check the equality of the function embeddings. We don't check the equality
  // of Vocabulary as it remains the same.
  if (!FunctionEmbedding.approximatelyEquals(FPI.FunctionEmbedding))
    return false;

  return true;
}

void FunctionPropertiesInfo::print(raw_ostream &OS) const {
#define PRINT_PROPERTY(PROP_NAME) OS << #PROP_NAME ": " << PROP_NAME << "\n";

  PRINT_PROPERTY(BasicBlockCount)
  PRINT_PROPERTY(BlocksReachedFromConditionalInstruction)
  PRINT_PROPERTY(Uses)
  PRINT_PROPERTY(DirectCallsToDefinedFunctions)
  PRINT_PROPERTY(LoadInstCount)
  PRINT_PROPERTY(StoreInstCount)
  PRINT_PROPERTY(MaxLoopDepth)
  PRINT_PROPERTY(TopLevelLoopCount)
  PRINT_PROPERTY(TotalInstructionCount)

  if (EnableDetailedFunctionProperties) {
    PRINT_PROPERTY(BasicBlocksWithSingleSuccessor)
    PRINT_PROPERTY(BasicBlocksWithTwoSuccessors)
    PRINT_PROPERTY(BasicBlocksWithMoreThanTwoSuccessors)
    PRINT_PROPERTY(BasicBlocksWithSinglePredecessor)
    PRINT_PROPERTY(BasicBlocksWithTwoPredecessors)
    PRINT_PROPERTY(BasicBlocksWithMoreThanTwoPredecessors)
    PRINT_PROPERTY(BigBasicBlocks)
    PRINT_PROPERTY(MediumBasicBlocks)
    PRINT_PROPERTY(SmallBasicBlocks)
    PRINT_PROPERTY(CastInstructionCount)
    PRINT_PROPERTY(FloatingPointInstructionCount)
    PRINT_PROPERTY(IntegerInstructionCount)
    PRINT_PROPERTY(ConstantIntOperandCount)
    PRINT_PROPERTY(ConstantFPOperandCount)
    PRINT_PROPERTY(ConstantOperandCount)
    PRINT_PROPERTY(InstructionOperandCount)
    PRINT_PROPERTY(BasicBlockOperandCount)
    PRINT_PROPERTY(GlobalValueOperandCount)
    PRINT_PROPERTY(InlineAsmOperandCount)
    PRINT_PROPERTY(ArgumentOperandCount)
    PRINT_PROPERTY(UnknownOperandCount)
    PRINT_PROPERTY(CriticalEdgeCount)
    PRINT_PROPERTY(ControlFlowEdgeCount)
    PRINT_PROPERTY(UnconditionalBranchCount)
    PRINT_PROPERTY(IntrinsicCount)
    PRINT_PROPERTY(DirectCallCount)
    PRINT_PROPERTY(IndirectCallCount)
    PRINT_PROPERTY(CallReturnsIntegerCount)
    PRINT_PROPERTY(CallReturnsFloatCount)
    PRINT_PROPERTY(CallReturnsPointerCount)
    PRINT_PROPERTY(CallReturnsVectorIntCount)
    PRINT_PROPERTY(CallReturnsVectorFloatCount)
    PRINT_PROPERTY(CallReturnsVectorPointerCount)
    PRINT_PROPERTY(CallWithManyArgumentsCount)
    PRINT_PROPERTY(CallWithPointerArgumentCount)
  }

#undef PRINT_PROPERTY

  OS << "\n";
}

AnalysisKey FunctionPropertiesAnalysis::Key;

FunctionPropertiesInfo
FunctionPropertiesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPropertiesInfo::getFunctionPropertiesInfo(F, FAM);
}

PreservedAnalyses
FunctionPropertiesPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of CFA for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<FunctionPropertiesAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

FunctionPropertiesUpdater::FunctionPropertiesUpdater(
    FunctionPropertiesInfo &FPI, CallBase &CB)
    : FPI(FPI), CallSiteBB(*CB.getParent()), Caller(*CallSiteBB.getParent()) {
  assert(isa<CallInst>(CB) || isa<InvokeInst>(CB));
  // For BBs that are likely to change, we subtract from feature totals their
  // contribution. Some features, like max loop counts or depths, are left
  // invalid, as they will be updated post-inlining.
  SmallPtrSet<const BasicBlock *, 4> LikelyToChangeBBs;
  // The CB BB will change - it'll either be split or the callee's body (single
  // BB) will be pasted in.
  LikelyToChangeBBs.insert(&CallSiteBB);

  // The caller's entry BB may change due to new alloca instructions.
  LikelyToChangeBBs.insert(&*Caller.begin());

  // The users of the value returned by call instruction can change
  // leading to the change in embeddings being computed, when used.
  // We conservatively add the BBs with such uses to LikelyToChangeBBs.
  for (const auto *User : CB.users())
    CallUsers.insert(dyn_cast<Instruction>(User)->getParent());
  // CallSiteBB can be removed from CallUsers if present, it's taken care
  // separately.
  CallUsers.erase(&CallSiteBB);
  LikelyToChangeBBs.insert_range(CallUsers);

  // The successors may become unreachable in the case of `invoke` inlining.
  // We track successors separately, too, because they form a boundary, together
  // with the CB BB ('Entry') between which the inlined callee will be pasted.
  Successors.insert_range(successors(&CallSiteBB));

  // the outcome of the inlining may be that some edges get lost (DCEd BBs
  // because inlining brought some constant, for example). We don't know which
  // edges will be removed, so we list all of them as potentially removable.
  // Some BBs have (at this point) duplicate edges. Remove duplicates, otherwise
  // the DT updater will not apply changes correctly.
  DenseSet<const BasicBlock *> Inserted;
  for (auto *Succ : successors(&CallSiteBB))
    if (Inserted.insert(Succ).second)
      DomTreeUpdates.emplace_back(DominatorTree::UpdateKind::Delete,
                                  const_cast<BasicBlock *>(&CallSiteBB),
                                  const_cast<BasicBlock *>(Succ));
  // Reuse Inserted (which has some allocated capacity at this point) below, if
  // we have an invoke.
  Inserted.clear();
  // Inlining only handles invoke and calls. If this is an invoke, and inlining
  // it pulls another invoke, the original landing pad may get split, so as to
  // share its content with other potential users. So the edge up to which we
  // need to invalidate and then re-account BB data is the successors of the
  // current landing pad. We can leave the current lp, too - if it doesn't get
  // split, then it will be the place traversal stops. Either way, the
  // discounted BBs will be checked if reachable and re-added.
  if (const auto *II = dyn_cast<InvokeInst>(&CB)) {
    const auto *UnwindDest = II->getUnwindDest();
    Successors.insert_range(successors(UnwindDest));
    // Same idea as above, we pretend we lose all these edges.
    for (auto *Succ : successors(UnwindDest))
      if (Inserted.insert(Succ).second)
        DomTreeUpdates.emplace_back(DominatorTree::UpdateKind::Delete,
                                    const_cast<BasicBlock *>(UnwindDest),
                                    const_cast<BasicBlock *>(Succ));
  }

  // Exclude the CallSiteBB, if it happens to be its own successor (1-BB loop).
  // We are only interested in BBs the graph moves past the callsite BB to
  // define the frontier past which we don't want to re-process BBs. Including
  // the callsite BB in this case would prematurely stop the traversal in
  // finish().
  Successors.erase(&CallSiteBB);

  LikelyToChangeBBs.insert_range(Successors);

  // Commit the change. While some of the BBs accounted for above may play dual
  // role - e.g. caller's entry BB may be the same as the callsite BB - set
  // insertion semantics make sure we account them once. This needs to be
  // followed in `finish`, too.
  for (const auto *BB : LikelyToChangeBBs)
    FPI.updateForBB(*BB, -1);
}

DominatorTree &FunctionPropertiesUpdater::getUpdatedDominatorTree(
    FunctionAnalysisManager &FAM) const {
  auto &DT =
      FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(Caller));

  SmallVector<DominatorTree::UpdateType, 2> FinalDomTreeUpdates;

  DenseSet<const BasicBlock *> Inserted;
  for (auto *Succ : successors(&CallSiteBB))
    if (Inserted.insert(Succ).second)
      FinalDomTreeUpdates.push_back({DominatorTree::UpdateKind::Insert,
                                     const_cast<BasicBlock *>(&CallSiteBB),
                                     const_cast<BasicBlock *>(Succ)});

  // Perform the deletes last, so that any new nodes connected to nodes
  // participating in the edge deletion are known to the DT.
  for (auto &Upd : DomTreeUpdates)
    if (!llvm::is_contained(successors(Upd.getFrom()), Upd.getTo()))
      FinalDomTreeUpdates.push_back(Upd);

  DT.applyUpdates(FinalDomTreeUpdates);
#ifdef EXPENSIVE_CHECKS
  assert(DT.verify(DominatorTree::VerificationLevel::Full));
#endif
  return DT;
}

void FunctionPropertiesUpdater::finish(FunctionAnalysisManager &FAM) const {
  // Update feature values from the BBs that were copied from the callee, or
  // might have been modified because of inlining. The latter have been
  // subtracted in the FunctionPropertiesUpdater ctor.
  // There could be successors that were reached before but now are only
  // reachable from elsewhere in the CFG.
  // One example is the following diamond CFG (lines are arrows pointing down):
  //    A
  //  /   \
  // B     C
  // |     |
  // |     D
  // |     |
  // |     E
  //  \   /
  //    F
  // There's a call site in C that is inlined. Upon doing that, it turns out
  // it expands to
  //   call void @llvm.trap()
  //   unreachable
  // F isn't reachable from C anymore, but we did discount it when we set up
  // FunctionPropertiesUpdater, so we need to re-include it here.
  // At the same time, D and E were reachable before, but now are not anymore,
  // so we need to leave D out (we discounted it at setup), and explicitly
  // remove E.
  SetVector<const BasicBlock *> Reinclude;
  SetVector<const BasicBlock *> Unreachable;
  auto &DT = getUpdatedDominatorTree(FAM);

  if (&CallSiteBB != &*Caller.begin())
    Reinclude.insert(&*Caller.begin());

  // Reinclude the BBs which use the values returned by call instruction
  Reinclude.insert_range(CallUsers);

  // Distribute the successors to the 2 buckets.
  for (const auto *Succ : Successors)
    if (DT.isReachableFromEntry(Succ))
      Reinclude.insert(Succ);
    else
      Unreachable.insert(Succ);

  // For reinclusion, we want to stop at the reachable successors, who are at
  // the beginning of the worklist; but, starting from the callsite bb and
  // ending at those successors, we also want to perform a traversal.
  // IncludeSuccessorsMark is the index after which we include successors.
  const auto IncludeSuccessorsMark = Reinclude.size();
  bool CSInsertion = Reinclude.insert(&CallSiteBB);
  (void)CSInsertion;
  assert(CSInsertion);
  for (size_t I = 0; I < Reinclude.size(); ++I) {
    const auto *BB = Reinclude[I];
    FPI.reIncludeBB(*BB);
    if (I >= IncludeSuccessorsMark)
      Reinclude.insert_range(successors(BB));
  }

  // For exclusion, we don't need to exclude the set of BBs that were successors
  // before and are now unreachable, because we already did that at setup. For
  // the rest, as long as a successor is unreachable, we want to explicitly
  // exclude it.
  const auto AlreadyExcludedMark = Unreachable.size();
  for (size_t I = 0; I < Unreachable.size(); ++I) {
    const auto *U = Unreachable[I];
    if (I >= AlreadyExcludedMark)
      FPI.updateForBB(*U, -1);
    for (const auto *Succ : successors(U))
      if (!DT.isReachableFromEntry(Succ))
        Unreachable.insert(Succ);
  }

  const auto &LI = FAM.getResult<LoopAnalysis>(const_cast<Function &>(Caller));
  FPI.updateAggregateStats(Caller, LI);
#ifdef EXPENSIVE_CHECKS
  assert(isUpdateValid(Caller, FPI, FAM));
#endif
}

bool FunctionPropertiesUpdater::isUpdateValid(Function &F,
                                              const FunctionPropertiesInfo &FPI,
                                              FunctionAnalysisManager &FAM) {
  if (!FAM.getResult<DominatorTreeAnalysis>(F).verify(
          DominatorTree::VerificationLevel::Full))
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  auto VocabResult = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F)
                         .getCachedResult<IR2VecVocabAnalysis>(*F.getParent());
  auto Fresh =
      FunctionPropertiesInfo::getFunctionPropertiesInfo(F, DT, LI, VocabResult);
  return FPI == Fresh;
}
