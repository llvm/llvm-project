//===----------- LoopVersioningLICM.cpp - LICM Loop Versioning ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// When alias analysis is uncertain about the aliasing between any two accesses,
// it will return MayAlias. This uncertainty from alias analysis restricts LICM
// from proceeding further. In cases where alias analysis is uncertain we might
// use loop versioning as an alternative.
//
// Loop Versioning will create a version of the loop with aggressive aliasing
// assumptions in addition to the original with conservative (default) aliasing
// assumptions. The version of the loop making aggressive aliasing assumptions
// will have all the memory accesses marked as no-alias. These two versions of
// loop will be preceded by a memory runtime check. This runtime check consists
// of bound checks for all unique memory accessed in loop, and it ensures the
// lack of memory aliasing. The result of the runtime check determines which of
// the loop versions is executed: If the runtime check detects any memory
// aliasing, then the original loop is executed. Otherwise, the version with
// aggressive aliasing assumptions is used.
//
// Following are the top level steps:
//
// a) Perform LoopVersioningLICM's feasibility check.
// b) If loop is a candidate for versioning then create a memory bound check,
//    by considering all the memory accesses in loop body.
// c) Clone original loop and set all memory accesses as no-alias in new loop.
// d) Set original loop & versioned loop as a branch target of the runtime check
//    result.
//
// It transforms loop as shown below:
//
//                         +----------------+
//                         |Runtime Memcheck|
//                         +----------------+
//                                 |
//              +----------+----------------+----------+
//              |                                      |
//    +---------+----------+               +-----------+----------+
//    |Orig Loop Preheader |               |Cloned Loop Preheader |
//    +--------------------+               +----------------------+
//              |                                      |
//    +--------------------+               +----------------------+
//    |Orig Loop Body      |               |Cloned Loop Body      |
//    +--------------------+               +----------------------+
//              |                                      |
//    +--------------------+               +----------------------+
//    |Orig Loop Exit Block|               |Cloned Loop Exit Block|
//    +--------------------+               +-----------+----------+
//              |                                      |
//              +----------+--------------+-----------+
//                                 |
//                           +-----+----+
//                           |Join Block|
//                           +----------+
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "loop-versioning-licm"
static const char* LICMVersioningMetaData =
    "llvm.loop.licm_versioning.disable";

using namespace llvm;

/// Threshold minimum allowed percentage for possible
/// invariant instructions in a loop.
static cl::opt<float>
    LVInvarThreshold("licm-versioning-invariant-threshold",
                     cl::desc("LoopVersioningLICM's minimum allowed percentage"
                              "of possible invariant instructions per loop"),
                     cl::init(25), cl::Hidden);

/// Threshold for maximum allowed loop nest/depth
static cl::opt<unsigned> LVLoopDepthThreshold(
    "licm-versioning-max-depth-threshold",
    cl::desc(
        "LoopVersioningLICM's threshold for maximum allowed loop nest/depth"),
    cl::init(2), cl::Hidden);

/// \brief Create MDNode for input string.
static MDNode *createStringMetadata(Loop *TheLoop, StringRef Name, unsigned V) {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  Metadata *MDs[] = {
      MDString::get(Context, Name),
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), V))};
  return MDNode::get(Context, MDs);
}

/// \brief Check string metadata in loop, if it exist return true,
/// else return false.
bool llvm::checkStringMetadataIntoLoop(Loop *TheLoop, StringRef Name) {
  MDNode *LoopID = TheLoop->getLoopID();
  // Return false if LoopID is false.
  if (!LoopID)
    return false;
  // Iterate over LoopID operands and look for MDString Metadata
  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
    MDNode *MD = dyn_cast<MDNode>(LoopID->getOperand(i));
    if (!MD)
      continue;
    MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;
    // Return true if MDString holds expected MetaData.
    if (Name.equals(S->getString()))
      return true;
  }
  return false;
}

/// \brief Set input string into loop metadata by keeping other values intact.
void llvm::addStringMetadataToLoop(Loop *TheLoop, const char *MDString,
                                   unsigned V) {
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, retain it.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      MDs.push_back(Node);
    }
  }
  // Add new metadata.
  MDs.push_back(createStringMetadata(TheLoop, MDString, V));
  // Replace current metadata node with new one.
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  TheLoop->setLoopID(NewLoopID);
}

namespace {
struct LoopVersioningLICM : public LoopPass {
  static char ID;

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequiredID(LCSSAID);
    AU.addRequired<LoopAccessAnalysis>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  using llvm::Pass::doFinalization;

  bool doFinalization() override { return false; }

  LoopVersioningLICM()
      : LoopPass(ID), AA(nullptr), SE(nullptr), LI(nullptr), DT(nullptr),
        TLI(nullptr), LAA(nullptr), LAI(nullptr), Changed(false),
        Preheader(nullptr), CurLoop(nullptr), CurAST(nullptr),
        LoopDepthThreshold(LVLoopDepthThreshold),
        InvariantThreshold(LVInvarThreshold), LoadAndStoreCounter(0),
        InvariantCounter(0), IsReadOnlyLoop(true) {
    initializeLoopVersioningLICMPass(*PassRegistry::getPassRegistry());
  }

  AliasAnalysis *AA;         // Current AliasAnalysis information
  ScalarEvolution *SE;       // Current ScalarEvolution
  LoopInfo *LI;              // Current LoopInfo
  DominatorTree *DT;         // Dominator Tree for the current Loop.
  TargetLibraryInfo *TLI;    // TargetLibraryInfo for constant folding.
  LoopAccessAnalysis *LAA;   // Current LoopAccessAnalysis
  const LoopAccessInfo *LAI; // Current Loop's LoopAccessInfo

  bool Changed;            // Set to true when we change anything.
  BasicBlock *Preheader;   // The preheader block of the current loop.
  Loop *CurLoop;           // The current loop we are working on.
  AliasSetTracker *CurAST; // AliasSet information for the current loop.
  ValueToValueMap Strides;

  unsigned LoopDepthThreshold;  // Maximum loop nest threshold
  float InvariantThreshold;     // Minimum invariant threshold
  unsigned LoadAndStoreCounter; // Counter to track num of load & store
  unsigned InvariantCounter;    // Counter to track num of invariant
  bool IsReadOnlyLoop;          // Read only loop marker.

  bool isLegalForVersioning();
  bool legalLoopStructure();
  bool legalLoopInstructions();
  bool legalLoopMemoryAccesses();
  void collectStridedAccess(Value *LoadOrStoreInst);
  bool isLoopAlreadyVisited();
  void setNoAliasToLoop(Loop *);
  bool instructionSafeForVersioning(Instruction *);
  const char *getPassName() const override { return "Loop Versioning"; }
};
}

/// \brief Collects stride access from a given value.
void LoopVersioningLICM::collectStridedAccess(Value *MemAccess) {
  Value *Ptr = nullptr;
  if (LoadInst *LI = dyn_cast<LoadInst>(MemAccess))
    Ptr = LI->getPointerOperand();
  else if (StoreInst *SI = dyn_cast<StoreInst>(MemAccess))
    Ptr = SI->getPointerOperand();
  else
    return;

  Value *Stride = getStrideFromPointer(Ptr, SE, CurLoop);
  if (!Stride)
    return;

  DEBUG(dbgs() << "Found a strided access that we can version");
  DEBUG(dbgs() << "  Ptr: " << *Ptr << " Stride: " << *Stride << "\n");
  Strides[Ptr] = Stride;
}

/// \brief Check loop structure and confirms it's good for LoopVersioningLICM.
bool LoopVersioningLICM::legalLoopStructure() {
  // Loop must have a preheader, if not return false.
  if (!CurLoop->getLoopPreheader()) {
    DEBUG(dbgs() << "    loop preheader is missing\n");
    return false;
  }
  // Loop should be innermost loop, if not return false.
  if (CurLoop->getSubLoops().size()) {
    DEBUG(dbgs() << "    loop is not innermost\n");
    return false;
  }
  // Loop should have a single backedge, if not return false.
  if (CurLoop->getNumBackEdges() != 1) {
    DEBUG(dbgs() << "    loop has multiple backedges\n");
    return false;
  }
  // Loop must have a single exiting block, if not return false.
  if (!CurLoop->getExitingBlock()) {
    DEBUG(dbgs() << "    loop has multiple exiting block\n");
    return false;
  }
  // We only handle bottom-tested loop, i.e. loop in which the condition is
  // checked at the end of each iteration. With that we can assume that all
  // instructions in the loop are executed the same number of times.
  if (CurLoop->getExitingBlock() != CurLoop->getLoopLatch()) {
    DEBUG(dbgs() << "    loop is not bottom tested\n");
    return false;
  }
  // Parallel loops must not have aliasing loop-invariant memory accesses.
  // Hence we don't need to version anything in this case.
  if (CurLoop->isAnnotatedParallel()) {
    DEBUG(dbgs() << "    Parallel loop is not worth versioning\n");
    return false;
  }
  // Loop depth more then LoopDepthThreshold are not allowed
  if (CurLoop->getLoopDepth() > LoopDepthThreshold) {
    DEBUG(dbgs() << "    loop depth is more then threshold\n");
    return false;
  }
  // Loop should have a dedicated exit block, if not return false.
  if (!CurLoop->hasDedicatedExits()) {
    DEBUG(dbgs() << "    loop does not has dedicated exit blocks\n");
    return false;
  }
  // We need to be able to compute the loop trip count in order
  // to generate the bound checks.
  const SCEV *ExitCount = SE->getBackedgeTakenCount(CurLoop);
  if (ExitCount == SE->getCouldNotCompute()) {
    DEBUG(dbgs() << "    loop does not has trip count\n");
    return false;
  }
  return true;
}

/// \brief Check memory accesses in loop and confirms it's good for
/// LoopVersioningLICM.
bool LoopVersioningLICM::legalLoopMemoryAccesses() {
  bool HasMayAlias = false;
  bool TypeSafety = false;
  bool HasMod = false;
  // Memory check:
  // Transform phase will generate a versioned loop and also a runtime check to
  // ensure the pointers are independent and they don’t alias.
  // In version variant of loop, alias meta data asserts that all access are
  // mutually independent.
  //
  // Pointers aliasing in alias domain are avoided because with multiple
  // aliasing domains we may not be able to hoist potential loop invariant
  // access out of the loop.
  //
  // Iterate over alias tracker sets, and confirm AliasSets doesn't have any
  // must alias set.
  for (const auto &I : *CurAST) {
    const AliasSet &AS = I;
    // Skip Forward Alias Sets, as this should be ignored as part of
    // the AliasSetTracker object.
    if (AS.isForwardingAliasSet())
      continue;
    // With MustAlias its not worth adding runtime bound check.
    if (AS.isMustAlias())
      return false;
    Value *SomePtr = AS.begin()->getValue();
    bool TypeCheck = true;
    // Check for Mod & MayAlias
    HasMayAlias |= AS.isMayAlias();
    HasMod |= AS.isMod();
    for (const auto &A : AS) {
      Value *Ptr = A.getValue();
      // Alias tracker should have pointers of same data type.
      TypeCheck = (TypeCheck && (SomePtr->getType() == Ptr->getType()));
    }
    // At least one alias tracker should have pointers of same data type.
    TypeSafety |= TypeCheck;
  }
  // Ensure types should be of same type.
  if (!TypeSafety) {
    DEBUG(dbgs() << "    Alias tracker type safety failed!\n");
    return false;
  }
  // Ensure loop body shouldn't be read only.
  if (!HasMod) {
    DEBUG(dbgs() << "    No memory modified in loop body\n");
    return false;
  }
  // Make sure alias set has may alias case.
  // If there no alias memory ambiguity, return false.
  if (!HasMayAlias) {
    DEBUG(dbgs() << "    No ambiguity in memory access.\n");
    return false;
  }
  return true;
}

/// \brief Check loop instructions safe for Loop versioning.
/// It returns true if it's safe else returns false.
/// Consider following:
/// 1) Check all load store in loop body are non atomic & non volatile.
/// 2) Check function call safety, by ensuring its not accessing memory.
/// 3) Loop body shouldn't have any may throw instruction.
bool LoopVersioningLICM::instructionSafeForVersioning(Instruction *I) {
  assert(I != nullptr && "Null instruction found!");
  // Check function call safety
  if (isa<CallInst>(I) && !AA->doesNotAccessMemory(CallSite(I))) {
    DEBUG(dbgs() << "    Unsafe call site found.\n");
    return false;
  }
  // Avoid loops with possiblity of throw
  if (I->mayThrow()) {
    DEBUG(dbgs() << "    May throw instruction found in loop body\n");
    return false;
  }
  // If current instruction is load instructions
  // make sure it's a simple load (non atomic & non volatile)
  if (I->mayReadFromMemory()) {
    LoadInst *Ld = dyn_cast<LoadInst>(I);
    if (!Ld || !Ld->isSimple()) {
      DEBUG(dbgs() << "    Found a non-simple load.\n");
      return false;
    }
    LoadAndStoreCounter++;
    collectStridedAccess(Ld);
    Value *Ptr = Ld->getPointerOperand();
    // Check loop invariant.
    if (SE->isLoopInvariant(SE->getSCEV(Ptr), CurLoop))
      InvariantCounter++;
  }
  // If current instruction is store instruction
  // make sure it's a simple store (non atomic & non volatile)
  else if (I->mayWriteToMemory()) {
    StoreInst *St = dyn_cast<StoreInst>(I);
    if (!St || !St->isSimple()) {
      DEBUG(dbgs() << "    Found a non-simple store.\n");
      return false;
    }
    LoadAndStoreCounter++;
    collectStridedAccess(St);
    Value *Ptr = St->getPointerOperand();
    // Check loop invariant.
    if (SE->isLoopInvariant(SE->getSCEV(Ptr), CurLoop))
      InvariantCounter++;

    IsReadOnlyLoop = false;
  }
  return true;
}

/// \brief Check loop instructions and confirms it's good for
/// LoopVersioningLICM.
bool LoopVersioningLICM::legalLoopInstructions() {
  // Resetting counters.
  LoadAndStoreCounter = 0;
  InvariantCounter = 0;
  IsReadOnlyLoop = true;
  // Iterate over loop blocks and instructions of each block and check
  // instruction safety.
  for (auto *Block : CurLoop->getBlocks())
    for (auto &Inst : *Block) {
      // If instruction is unsafe just return false.
      if (!instructionSafeForVersioning(&Inst))
        return false;
    }
  // Get LoopAccessInfo from current loop.
  LAI = &LAA->getInfo(CurLoop, Strides);
  // Check LoopAccessInfo for need of runtime check.
  if (LAI->getRuntimePointerChecking()->getChecks().empty()) {
    DEBUG(dbgs() << "    LAA: Runtime check not found !!\n");
    return false;
  }
  // Number of runtime-checks should be less then RuntimeMemoryCheckThreshold
  if (LAI->getNumRuntimePointerChecks() >
      VectorizerParams::RuntimeMemoryCheckThreshold) {
    DEBUG(dbgs() << "    LAA: Runtime checks are more than threshold !!\n");
    return false;
  }
  // Loop should have at least one invariant load or store instruction.
  if (!InvariantCounter) {
    DEBUG(dbgs() << "    Invariant not found !!\n");
    return false;
  }
  // Read only loop not allowed.
  if (IsReadOnlyLoop) {
    DEBUG(dbgs() << "    Found a read-only loop!\n");
    return false;
  }
  // Profitablity check:
  // Check invariant threshold, should be in limit.
  if (InvariantCounter * 100 < InvariantThreshold * LoadAndStoreCounter) {
    DEBUG(dbgs()
          << "    Invariant load & store are less then defined threshold\n");
    DEBUG(dbgs() << "    Invariant loads & stores: "
                 << ((InvariantCounter * 100) / LoadAndStoreCounter) << "%\n");
    DEBUG(dbgs() << "    Invariant loads & store threshold: "
                 << InvariantThreshold << "%\n");
    return false;
  }
  return true;
}

/// \brief It checks loop is already visited or not.
/// check loop meta data, if loop revisited return true
/// else false.
bool LoopVersioningLICM::isLoopAlreadyVisited() {
  // Check LoopVersioningLICM metadata into loop
  if (checkStringMetadataIntoLoop(CurLoop, LICMVersioningMetaData)) {
    return true;
  }
  return false;
}

/// \brief Checks legality for LoopVersioningLICM by considering following:
/// a) loop structure legality   b) loop instruction legality
/// c) loop memory access legality.
/// Return true if legal else returns false.
bool LoopVersioningLICM::isLegalForVersioning() {
  DEBUG(dbgs() << "Loop: " << *CurLoop);
  // Make sure not re-visiting same loop again.
  if (isLoopAlreadyVisited()) {
    DEBUG(
        dbgs() << "    Revisiting loop in LoopVersioningLICM not allowed.\n\n");
    return false;
  }
  // Check loop structure leagality.
  if (!legalLoopStructure()) {
    DEBUG(
        dbgs() << "    Loop structure not suitable for LoopVersioningLICM\n\n");
    return false;
  }
  // Check loop instruction leagality.
  if (!legalLoopInstructions()) {
    DEBUG(dbgs()
          << "    Loop instructions not suitable for LoopVersioningLICM\n\n");
    return false;
  }
  // Check loop memory access leagality.
  if (!legalLoopMemoryAccesses()) {
    DEBUG(dbgs()
          << "    Loop memory access not suitable for LoopVersioningLICM\n\n");
    return false;
  }
  // Loop versioning is feasible, return true.
  DEBUG(dbgs() << "    Loop Versioning found to be beneficial\n\n");
  return true;
}

/// \brief Update loop with aggressive aliasing assumptions.
/// It marks no-alias to any pairs of memory operations by assuming
/// loop should not have any must-alias memory accesses pairs.
/// During LoopVersioningLICM legality we ignore loops having must
/// aliasing memory accesses.
void LoopVersioningLICM::setNoAliasToLoop(Loop *VerLoop) {
  // Get latch terminator instruction.
  Instruction *I = VerLoop->getLoopLatch()->getTerminator();
  // Create alias scope domain.
  MDBuilder MDB(I->getContext());
  MDNode *NewDomain = MDB.createAnonymousAliasScopeDomain("LVDomain");
  StringRef Name = "LVAliasScope";
  SmallVector<Metadata *, 4> Scopes, NoAliases;
  MDNode *NewScope = MDB.createAnonymousAliasScope(NewDomain, Name);
  // Iterate over each instruction of loop.
  // set no-alias for all load & store instructions.
  for (auto *Block : CurLoop->getBlocks()) {
    for (auto &Inst : *Block) {
      // Only interested in instruction that may modify or read memory.
      if (!Inst.mayReadFromMemory() && !Inst.mayWriteToMemory())
        continue;
      Scopes.push_back(NewScope);
      NoAliases.push_back(NewScope);
      // Set no-alias for current instruction.
      Inst.setMetadata(
          LLVMContext::MD_noalias,
          MDNode::concatenate(Inst.getMetadata(LLVMContext::MD_noalias),
                              MDNode::get(Inst.getContext(), NoAliases)));
      // set alias-scope for current instruction.
      Inst.setMetadata(
          LLVMContext::MD_alias_scope,
          MDNode::concatenate(Inst.getMetadata(LLVMContext::MD_alias_scope),
                              MDNode::get(Inst.getContext(), Scopes)));
    }
  }
}

bool LoopVersioningLICM::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;
  Changed = false;
  // Get Analysis information.
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  LAA = &getAnalysis<LoopAccessAnalysis>();
  LAI = nullptr;
  // Set Current Loop
  CurLoop = L;
  // Get the preheader block.
  Preheader = L->getLoopPreheader();
  // Initial allocation
  CurAST = new AliasSetTracker(*AA);

  // Loop over the body of this loop, construct AST.
  for (auto *Block : L->getBlocks()) {
    if (LI->getLoopFor(Block) == L) // Ignore blocks in subloop.
      CurAST->add(*Block);          // Incorporate the specified basic block
  }
  // Check feasiblity of LoopVersioningLICM.
  // If versioning found to be feasible and beneficial then proceed
  // else simply return, by cleaning up memory.
  if (isLegalForVersioning()) {
    // Do loop versioning.
    // Create memcheck for memory accessed inside loop.
    // Clone original loop, and set blocks properly.
    LoopVersioning LVer(*LAI, CurLoop, LI, DT, SE, true);
    LVer.versionLoop();
    // Set Loop Versioning metaData for original loop.
    addStringMetadataToLoop(LVer.getNonVersionedLoop(), LICMVersioningMetaData);
    // Set Loop Versioning metaData for version loop.
    addStringMetadataToLoop(LVer.getVersionedLoop(), LICMVersioningMetaData);
    // Set "llvm.mem.parallel_loop_access" metaData to versioned loop.
    addStringMetadataToLoop(LVer.getVersionedLoop(),
                            "llvm.mem.parallel_loop_access");
    // Update version loop with aggressive aliasing assumption.
    setNoAliasToLoop(LVer.getVersionedLoop());
    Changed = true;
  }
  // Delete allocated memory.
  delete CurAST;
  return Changed;
}

char LoopVersioningLICM::ID = 0;
INITIALIZE_PASS_BEGIN(LoopVersioningLICM, "loop-versioning-licm",
                      "Loop Versioning For LICM", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_DEPENDENCY(LoopAccessAnalysis)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(LoopVersioningLICM, "loop-versioning-licm",
                    "Loop Versioning For LICM", false, false)

Pass *llvm::createLoopVersioningLICMPass() { return new LoopVersioningLICM(); }
