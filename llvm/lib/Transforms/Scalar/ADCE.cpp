//===- ADCE.cpp - Code to perform dead code elimination -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/ADCE.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "adce"

STATISTIC(NumRemoved, "Number of instructions removed");

// This is a tempoary option until we change the interface
// to this pass based on optimization level.
static cl::opt<bool> RemoveControlFlowFlag("adce-remove-control-flow",
                                           cl::init(false), cl::Hidden);

namespace {
/// Information about Instructions
struct InstInfoType {
  /// True if the associated instruction is live.
  bool Live = false;
  /// Quick access to information for block containing associated Instruction.
  struct BlockInfoType *Block = nullptr;
};

/// Information about basic blocks relevant to dead code elimination.
struct BlockInfoType {
  /// True when this block contains a live instructions.
  bool Live = false;
  /// True when this block ends in an unconditional branch.
  bool UnconditionalBranch = false;
  /// True when this block is known to have live PHI nodes.
  bool HasLivePhiNodes = false;
  /// Control dependence sources need to be live for this block.
  bool CFLive = false;

  /// Quick access to the LiveInfo for the terminator,
  /// holds the value &InstInfo[Terminator]
  InstInfoType *TerminatorLiveInfo = nullptr;

  bool terminatorIsLive() const { return TerminatorLiveInfo->Live; }

  /// Corresponding BasicBlock.
  BasicBlock *BB = nullptr;

  /// Cache of BB->getTerminator()
  TerminatorInst *Terminator = nullptr;
};

class AggressiveDeadCodeElimination {
  Function &F;
  PostDominatorTree &PDT;

  /// Mapping of blocks to associated information, an element in BlockInfoVec.
  DenseMap<BasicBlock *, BlockInfoType> BlockInfo;
  bool isLive(BasicBlock *BB) { return BlockInfo[BB].Live; }

  /// Mapping of instructions to associated information.
  DenseMap<Instruction *, InstInfoType> InstInfo;
  bool isLive(Instruction *I) { return InstInfo[I].Live; }

  /// Instructions known to be live where we need to mark
  /// reaching definitions as live.
  SmallVector<Instruction *, 128> Worklist;
  /// Debug info scopes around a live instruction.
  SmallPtrSet<const Metadata *, 32> AliveScopes;

  /// Set of blocks with not known to have live terminators.
  SmallPtrSet<BasicBlock *, 16> BlocksWithDeadTerminators;

  /// The set of blocks which we have determined are live in the
  /// most recent iteration of propagating liveness.
  SmallPtrSet<BasicBlock *, 16> NewLiveBlocks;

  /// Set up auxiliary data structures for Instructions and BasicBlocks and
  /// initialize the Worklist to the set of must-be-live Instruscions.
  void initialize();
  /// Return true for operations which are always treated as live.
  bool isAlwaysLive(Instruction &I);
  /// Return true for instrumentation instructions for value profiling.
  bool isInstrumentsConstant(Instruction &I);

  /// Propagate liveness to reaching definitions.
  void markLiveInstructions();
  /// Mark an instruction as live.
  void markLive(Instruction *I);
  
  /// Mark terminators of control predecessors of a PHI node live.
  void markPhiLive(PHINode *PN);

  /// Record the Debug Scopes which surround live debug information.
  void collectLiveScopes(const DILocalScope &LS);
  void collectLiveScopes(const DILocation &DL);

  /// Analyze dead branches to find those whose branches are the sources
  /// of control dependences impacting a live block. Those branches are
  /// marked live.
  void markLiveBranchesFromControlDependences();

  /// Remove instructions not marked live, return if any any instruction
  /// was removed.
  bool removeDeadInstructions();

public:
  AggressiveDeadCodeElimination(Function &F, PostDominatorTree &PDT)
      : F(F), PDT(PDT) {}
  bool performDeadCodeElimination();
};
}

bool AggressiveDeadCodeElimination::performDeadCodeElimination() {
  initialize();
  markLiveInstructions();
  return removeDeadInstructions();
}

static bool isUnconditionalBranch(TerminatorInst *Term) {
  auto BR = dyn_cast<BranchInst>(Term);
  return BR && BR->isUnconditional();
}

void AggressiveDeadCodeElimination::initialize() {

  auto NumBlocks = F.size();

  // We will have an entry in the map for each block so we grow the
  // structure to twice that size to keep the load factor low in the hash table.
  BlockInfo.reserve(NumBlocks);
  size_t NumInsts = 0;

  // Iterate over blocks and initialize BlockInfoVec entries, count
  // instructions to size the InstInfo hash table.
  for (auto &BB : F) {
    NumInsts += BB.size();
    auto &Info = BlockInfo[&BB];
    Info.BB = &BB;
    Info.Terminator = BB.getTerminator();
    Info.UnconditionalBranch = isUnconditionalBranch(Info.Terminator);
  }

  // Initialize instruction map and set pointers to block info.
  InstInfo.reserve(NumInsts);
  for (auto &BBInfo : BlockInfo)
    for (Instruction &I : *BBInfo.second.BB)
      InstInfo[&I].Block = &BBInfo.second;

  // Since BlockInfoVec holds pointers into InstInfo and vice-versa, we may not
  // add any more elements to either after this point.
  for (auto &BBInfo : BlockInfo)
    BBInfo.second.TerminatorLiveInfo = &InstInfo[BBInfo.second.Terminator];

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : instructions(F))
    if (isAlwaysLive(I))
      markLive(&I);

  if (!RemoveControlFlowFlag)
    return;

  // This is temporary: will update with post order traveral to
  // find loop bottoms
  SmallPtrSet<BasicBlock *, 16> Seen;
  for (auto &BB : F) {
    Seen.insert(&BB);
    TerminatorInst *Term = BB.getTerminator();
    if (isLive(Term))
      continue;

    for (auto Succ : successors(&BB))
      if (Seen.count(Succ)) {
        // back edge....
        markLive(Term);
        break;
      }
  }
  // End temporary handling of loops.

  // Mark blocks live if there is no path from the block to the
  // return of the function or a successor for which this is true.
  // This protects IDFCalculator which cannot handle such blocks.
  for (auto &BBInfoPair : BlockInfo) {
    auto &BBInfo = BBInfoPair.second;
    if (BBInfo.terminatorIsLive())
      continue;
    auto *BB = BBInfo.BB;
    if (!PDT.getNode(BB)) {
      DEBUG(dbgs() << "Not post-dominated by return: " << BB->getName()
                   << '\n';);
      markLive(BBInfo.Terminator);
      continue;
    }
    for (auto Succ : successors(BB))
      if (!PDT.getNode(Succ)) {
        DEBUG(dbgs() << "Successor not post-dominated by return: "
                     << BB->getName() << '\n';);
        markLive(BBInfo.Terminator);
        break;
      }
  }

  // Treat the entry block as always live
  auto *BB = &F.getEntryBlock();
  auto &EntryInfo = BlockInfo[BB];
  EntryInfo.Live = true;
  if (EntryInfo.UnconditionalBranch)
    markLive(EntryInfo.Terminator);

  // Build initial collection of blocks with dead terminators
  for (auto &BBInfo : BlockInfo)
    if (!BBInfo.second.terminatorIsLive())
      BlocksWithDeadTerminators.insert(BBInfo.second.BB);
}

bool AggressiveDeadCodeElimination::isAlwaysLive(Instruction &I) {
  // TODO -- use llvm::isInstructionTriviallyDead
  if (I.isEHPad() || I.mayHaveSideEffects()) {
    // Skip any value profile instrumentation calls if they are
    // instrumenting constants.
    if (isInstrumentsConstant(I))
      return false;
    return true;
  }
  if (!isa<TerminatorInst>(I))
    return false;
  if (RemoveControlFlowFlag && (isa<BranchInst>(I) || isa<SwitchInst>(I)))
    return false;
  return true;
}

// Check if this instruction is a runtime call for value profiling and
// if it's instrumenting a constant.
bool AggressiveDeadCodeElimination::isInstrumentsConstant(Instruction &I) {
  // TODO -- move this test into llvm::isInstructionTriviallyDead
  if (CallInst *CI = dyn_cast<CallInst>(&I))
    if (Function *Callee = CI->getCalledFunction())
      if (Callee->getName().equals(getInstrProfValueProfFuncName()))
        if (isa<Constant>(CI->getArgOperand(0)))
          return true;
  return false;
}

void AggressiveDeadCodeElimination::markLiveInstructions() {

  // Propagate liveness backwards to operands.
  do {
    // Worklist holds newly discovered live instructions
    // where we need to mark the inputs as live.
    while (!Worklist.empty()) {
      Instruction *LiveInst = Worklist.pop_back_val();
      DEBUG(dbgs() << "work live: "; LiveInst->dump(););

      // Collect the live debug info scopes attached to this instruction.
      if (const DILocation *DL = LiveInst->getDebugLoc())
        collectLiveScopes(*DL);

      for (Use &OI : LiveInst->operands())
        if (Instruction *Inst = dyn_cast<Instruction>(OI))
          markLive(Inst);
      
      if (auto *PN = dyn_cast<PHINode>(LiveInst))
        markPhiLive(PN);
    }
    markLiveBranchesFromControlDependences();

    if (Worklist.empty()) {
      // Temporary until we can actually delete branches.
      SmallVector<TerminatorInst *, 16> DeadTerminators;
      for (auto *BB : BlocksWithDeadTerminators)
        DeadTerminators.push_back(BB->getTerminator());
      for (auto *I : DeadTerminators)
        markLive(I);
      assert(BlocksWithDeadTerminators.empty());
      // End temporary.
    }
  } while (!Worklist.empty());

  assert(BlocksWithDeadTerminators.empty());
}

void AggressiveDeadCodeElimination::markLive(Instruction *I) {

  auto &Info = InstInfo[I];
  if (Info.Live)
    return;

  DEBUG(dbgs() << "mark live: "; I->dump());
  Info.Live = true;
  Worklist.push_back(I);

  // Mark the containing block live
  auto &BBInfo = *Info.Block;
  if (BBInfo.Terminator == I)
    BlocksWithDeadTerminators.erase(BBInfo.BB);
  if (BBInfo.Live)
    return;

  DEBUG(dbgs() << "mark block live: " << BBInfo.BB->getName() << '\n');
  BBInfo.Live = true;
  if (!BBInfo.CFLive) {
    BBInfo.CFLive = true;
    NewLiveBlocks.insert(BBInfo.BB);
  }

  // Mark unconditional branches at the end of live
  // blocks as live since there is no work to do for them later
  if (BBInfo.UnconditionalBranch && I != BBInfo.Terminator)
    markLive(BBInfo.Terminator);
}

void AggressiveDeadCodeElimination::collectLiveScopes(const DILocalScope &LS) {
  if (!AliveScopes.insert(&LS).second)
    return;

  if (isa<DISubprogram>(LS))
    return;

  // Tail-recurse through the scope chain.
  collectLiveScopes(cast<DILocalScope>(*LS.getScope()));
}

void AggressiveDeadCodeElimination::collectLiveScopes(const DILocation &DL) {
  // Even though DILocations are not scopes, shove them into AliveScopes so we
  // don't revisit them.
  if (!AliveScopes.insert(&DL).second)
    return;

  // Collect live scopes from the scope chain.
  collectLiveScopes(*DL.getScope());

  // Tail-recurse through the inlined-at chain.
  if (const DILocation *IA = DL.getInlinedAt())
    collectLiveScopes(*IA);
}

void AggressiveDeadCodeElimination::markPhiLive(PHINode *PN) {
  auto &Info = BlockInfo[PN->getParent()];
  // Only need to check this once per block.
  if (Info.HasLivePhiNodes)
    return;
  Info.HasLivePhiNodes = true;

  // If a predecessor block is not live, mark it as control-flow live
  // which will trigger marking live branches upon which
  // that block is control dependent.
  for (auto *PredBB : predecessors(Info.BB)) {
    auto &Info = BlockInfo[PredBB];
    if (!Info.CFLive) {
      Info.CFLive = true;
      NewLiveBlocks.insert(PredBB);
    }
  }
}

void AggressiveDeadCodeElimination::markLiveBranchesFromControlDependences() {

  if (BlocksWithDeadTerminators.empty())
    return;

  DEBUG({
    dbgs() << "new live blocks:\n";
    for (auto *BB : NewLiveBlocks)
      dbgs() << "\t" << BB->getName() << '\n';
    dbgs() << "dead terminator blocks:\n";
    for (auto *BB : BlocksWithDeadTerminators)
      dbgs() << "\t" << BB->getName() << '\n';
  });

  // The dominance frontier of a live block X in the reverse
  // control graph is the set of blocks upon which X is control
  // dependent. The following sequence computes the set of blocks
  // which currently have dead terminators that are control
  // dependence sources of a block which is in NewLiveBlocks.

  SmallVector<BasicBlock *, 32> IDFBlocks;
  ReverseIDFCalculator IDFs(PDT);
  IDFs.setDefiningBlocks(NewLiveBlocks);
  IDFs.setLiveInBlocks(BlocksWithDeadTerminators);
  IDFs.calculate(IDFBlocks);
  NewLiveBlocks.clear();

  // Dead terminators which control live blocks are now marked live.
  for (auto BB : IDFBlocks) {
    DEBUG(dbgs() << "live control in: " << BB->getName() << '\n');
    markLive(BB->getTerminator());
  }
}

//===----------------------------------------------------------------------===//
//
//  Routines to update the CFG and SSA information before removing dead code.
//
//===----------------------------------------------------------------------===//
bool AggressiveDeadCodeElimination::removeDeadInstructions() {

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : instructions(F)) {
    // Check if the instruction is alive.
    if (isLive(&I))
      continue;

    assert(!I.isTerminator() && "NYI: Removing Control Flow");

    if (auto *DII = dyn_cast<DbgInfoIntrinsic>(&I)) {
      // Check if the scope of this variable location is alive.
      if (AliveScopes.count(DII->getDebugLoc()->getScope()))
        continue;

      // Fallthrough and drop the intrinsic.
      DEBUG({
        // If intrinsic is pointing at a live SSA value, there may be an
        // earlier optimization bug: if we know the location of the variable,
        // why isn't the scope of the location alive?
        if (Value *V = DII->getVariableLocation())
          if (Instruction *II = dyn_cast<Instruction>(V))
            if (isLive(II))
              dbgs() << "Dropping debug info for " << *DII << "\n";
      });
    }

    // Prepare to delete.
    Worklist.push_back(&I);
    I.dropAllReferences();
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }

  return !Worklist.empty();
}

//===----------------------------------------------------------------------===//
//
// Pass Manager integration code
//
//===----------------------------------------------------------------------===//
PreservedAnalyses ADCEPass::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
  if (!AggressiveDeadCodeElimination(F, PDT).performDeadCodeElimination())
    return PreservedAnalyses::all();

  // FIXME: This should also 'preserve the CFG'.
  auto PA = PreservedAnalyses();
  PA.preserve<GlobalsAA>();
  return PA;
}

namespace {
struct ADCELegacyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ADCELegacyPass() : FunctionPass(ID) {
    initializeADCELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    auto &PDT = getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    return AggressiveDeadCodeElimination(F, PDT).performDeadCodeElimination();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.setPreservesCFG(); // TODO -- will remove when we start removing branches
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char ADCELegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(ADCELegacyPass, "adce",
                      "Aggressive Dead Code Elimination", false, false)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(ADCELegacyPass, "adce", "Aggressive Dead Code Elimination",
                    false, false)

FunctionPass *llvm::createAggressiveDCEPass() { return new ADCELegacyPass(); }
