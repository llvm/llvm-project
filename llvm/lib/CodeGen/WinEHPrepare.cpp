//===-- WinEHPrepare - Prepare exception handling for code generation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers LLVM IR exception handling into something closer to what the
// backend wants for functions using a personality function from a runtime
// provided by MSVC. Functions with other personality functions are left alone
// and may be prepared by other passes. In particular, all supported MSVC
// personality functions require cleanup code to be outlined, and the C++
// personality requires catch handler code to be outlined.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LibCallSemantics.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "winehprepare"

static cl::opt<bool> DisableDemotion(
    "disable-demotion", cl::Hidden,
    cl::desc(
        "Clone multicolor basic blocks but do not demote cross funclet values"),
    cl::init(false));

static cl::opt<bool> DisableCleanups(
    "disable-cleanups", cl::Hidden,
    cl::desc("Do not remove implausible terminators or other similar cleanups"),
    cl::init(false));

namespace {
  
class WinEHPrepare : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid.
  WinEHPrepare(const TargetMachine *TM = nullptr) : FunctionPass(ID) {}

  bool runOnFunction(Function &Fn) override;

  bool doFinalization(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const char *getPassName() const override {
    return "Windows exception handling preparation";
  }

private:
  void insertPHIStores(PHINode *OriginalPHI, AllocaInst *SpillSlot);
  void
  insertPHIStore(BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
                 SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist);
  AllocaInst *insertPHILoads(PHINode *PN, Function &F);
  void replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                          DenseMap<BasicBlock *, Value *> &Loads, Function &F);
  void demoteNonlocalUses(Value *V, SetVector<BasicBlock *> &ColorsForBB,
                          Function &F);
  bool prepareExplicitEH(Function &F,
                         SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void replaceTerminatePadWithCleanup(Function &F);
  void colorFunclets(Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void resolveFuncletAncestry(Function &F,
                              SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void resolveFuncletAncestryForPath(
      Function &F, SmallVectorImpl<BasicBlock *> &FuncletPath,
      std::map<BasicBlock *, BasicBlock *> &IdentityMap);
  void makeFuncletEdgeUnreachable(BasicBlock *Parent, BasicBlock *Child);
  BasicBlock *cloneFuncletForParent(Function &F, BasicBlock *FuncletEntry,
                                    BasicBlock *Parent);
  void updateTerminatorsAfterFuncletClone(
      Function &F, BasicBlock *OrigFunclet, BasicBlock *CloneFunclet,
      BasicBlock *OrigBlock, BasicBlock *CloneBlock, BasicBlock *CloneParent,
      ValueToValueMapTy &VMap,
      std::map<BasicBlock *, BasicBlock *> &Orig2Clone);

  void demotePHIsOnFunclets(Function &F);
  void demoteUsesBetweenFunclets(Function &F);
  void demoteArgumentUses(Function &F);
  void cloneCommonBlocks(Function &F,
                         SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void removeImplausibleTerminators(Function &F);
  void cleanupPreparedFunclets(Function &F);
  void verifyPreparedFunclets(Function &F);

  // All fields are reset by runOnFunction.
  EHPersonality Personality = EHPersonality::Unknown;

  std::map<BasicBlock *, SetVector<BasicBlock *>> BlockColors;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletBlocks;
  std::map<BasicBlock *, std::vector<BasicBlock *>> FuncletChildren;
  std::map<BasicBlock *, std::vector<BasicBlock *>> FuncletParents;

  // This is a flag that indicates an uncommon situation where we need to
  // clone funclets has been detected.
  bool FuncletCloningRequired = false;
  // When a funclet with multiple parents contains a catchret, the block to
  // which it returns will be cloned so that there is a copy in each parent
  // but one of the copies will not be properly linked to the catchret and
  // in most cases will have no predecessors.  This double map allows us
  // to find these cloned blocks when we clone the child funclet.
  std::map<BasicBlock *, std::map<BasicBlock *, BasicBlock*>> EstrangedBlocks;
};

} // end anonymous namespace

char WinEHPrepare::ID = 0;
INITIALIZE_TM_PASS(WinEHPrepare, "winehprepare", "Prepare Windows exceptions",
                   false, false)

FunctionPass *llvm::createWinEHPass(const TargetMachine *TM) {
  return new WinEHPrepare(TM);
}

static void findFuncletEntryPoints(Function &Fn,
                                   SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  EntryBlocks.push_back(&Fn.getEntryBlock());
  for (BasicBlock &BB : Fn) {
    Instruction *First = BB.getFirstNonPHI();
    if (!First->isEHPad())
      continue;
    assert(!isa<LandingPadInst>(First) &&
           "landingpad cannot be used with funclet EH personality");
    // Find EH pad blocks that represent funclet start points.
    if (!isa<CatchEndPadInst>(First) && !isa<CleanupEndPadInst>(First))
      EntryBlocks.push_back(&BB);
  }
}

bool WinEHPrepare::runOnFunction(Function &Fn) {
  if (!Fn.hasPersonalityFn())
    return false;

  // Classify the personality to see what kind of preparation we need.
  Personality = classifyEHPersonality(Fn.getPersonalityFn());

  // Do nothing if this is not a funclet-based personality.
  if (!isFuncletEHPersonality(Personality))
    return false;

  // Remove unreachable blocks.  It is not valuable to assign them a color and
  // their existence can trick us into thinking values are alive when they are
  // not.
  removeUnreachableBlocks(Fn);

  SmallVector<BasicBlock *, 4> EntryBlocks;
  findFuncletEntryPoints(Fn, EntryBlocks);
  return prepareExplicitEH(Fn, EntryBlocks);
}

bool WinEHPrepare::doFinalization(Module &M) { return false; }

void WinEHPrepare::getAnalysisUsage(AnalysisUsage &AU) const {}

static int addUnwindMapEntry(WinEHFuncInfo &FuncInfo, int ToState,
                             const BasicBlock *BB) {
  CxxUnwindMapEntry UME;
  UME.ToState = ToState;
  UME.Cleanup = BB;
  FuncInfo.CxxUnwindMap.push_back(UME);
  return FuncInfo.getLastStateNumber();
}

static void addTryBlockMapEntry(WinEHFuncInfo &FuncInfo, int TryLow,
                                int TryHigh, int CatchHigh,
                                ArrayRef<const CatchPadInst *> Handlers) {
  WinEHTryBlockMapEntry TBME;
  TBME.TryLow = TryLow;
  TBME.TryHigh = TryHigh;
  TBME.CatchHigh = CatchHigh;
  assert(TBME.TryLow <= TBME.TryHigh);
  for (const CatchPadInst *CPI : Handlers) {
    WinEHHandlerType HT;
    Constant *TypeInfo = cast<Constant>(CPI->getArgOperand(0));
    if (TypeInfo->isNullValue())
      HT.TypeDescriptor = nullptr;
    else
      HT.TypeDescriptor = cast<GlobalVariable>(TypeInfo->stripPointerCasts());
    HT.Adjectives = cast<ConstantInt>(CPI->getArgOperand(1))->getZExtValue();
    HT.Handler = CPI->getParent();
    if (isa<ConstantPointerNull>(CPI->getArgOperand(2)))
      HT.CatchObj.Alloca = nullptr;
    else
      HT.CatchObj.Alloca = cast<AllocaInst>(CPI->getArgOperand(2));
    TBME.HandlerArray.push_back(HT);
  }
  FuncInfo.TryBlockMap.push_back(TBME);
}

static const CatchPadInst *getSingleCatchPadPredecessor(const BasicBlock *BB) {
  for (const BasicBlock *PredBlock : predecessors(BB))
    if (auto *CPI = dyn_cast<CatchPadInst>(PredBlock->getFirstNonPHI()))
      return CPI;
  return nullptr;
}

/// Find all the catchpads that feed directly into the catchendpad. Frontends
/// using this personality should ensure that each catchendpad and catchpad has
/// one or zero catchpad predecessors.
///
/// The following C++ generates the IR after it:
///   try {
///   } catch (A) {
///   } catch (B) {
///   }
///
/// IR:
///   %catchpad.A
///     catchpad [i8* A typeinfo]
///         to label %catch.A unwind label %catchpad.B
///   %catchpad.B
///     catchpad [i8* B typeinfo]
///         to label %catch.B unwind label %endcatches
///   %endcatches
///     catchendblock unwind to caller
static void
findCatchPadsForCatchEndPad(const BasicBlock *CatchEndBB,
                            SmallVectorImpl<const CatchPadInst *> &Handlers) {
  const CatchPadInst *CPI = getSingleCatchPadPredecessor(CatchEndBB);
  while (CPI) {
    Handlers.push_back(CPI);
    CPI = getSingleCatchPadPredecessor(CPI->getParent());
  }
  // We've pushed these back into reverse source order.  Reverse them to get
  // the list back into source order.
  std::reverse(Handlers.begin(), Handlers.end());
}

// Given BB which ends in an unwind edge, return the EHPad that this BB belongs
// to. If the unwind edge came from an invoke, return null.
static const BasicBlock *getEHPadFromPredecessor(const BasicBlock *BB) {
  const TerminatorInst *TI = BB->getTerminator();
  if (isa<InvokeInst>(TI))
    return nullptr;
  if (TI->isEHPad())
    return BB;
  return cast<CleanupReturnInst>(TI)->getCleanupPad()->getParent();
}

static void calculateExplicitCXXStateNumbers(WinEHFuncInfo &FuncInfo,
                                             const BasicBlock &BB,
                                             int ParentState) {
  assert(BB.isEHPad());
  const Instruction *FirstNonPHI = BB.getFirstNonPHI();
  // All catchpad instructions will be handled when we process their
  // respective catchendpad instruction.
  if (isa<CatchPadInst>(FirstNonPHI))
    return;

  if (isa<CatchEndPadInst>(FirstNonPHI)) {
    SmallVector<const CatchPadInst *, 2> Handlers;
    findCatchPadsForCatchEndPad(&BB, Handlers);
    const BasicBlock *FirstTryPad = Handlers.front()->getParent();
    int TryLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);
    FuncInfo.EHPadStateMap[Handlers.front()] = TryLow;
    for (const BasicBlock *PredBlock : predecessors(FirstTryPad))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, TryLow);
    int CatchLow = addUnwindMapEntry(FuncInfo, ParentState, nullptr);

    // catchpads are separate funclets in C++ EH due to the way rethrow works.
    // In SEH, they aren't, so no invokes will unwind to the catchendpad.
    FuncInfo.EHPadStateMap[FirstNonPHI] = CatchLow;
    int TryHigh = CatchLow - 1;
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, CatchLow);
    int CatchHigh = FuncInfo.getLastStateNumber();
    addTryBlockMapEntry(FuncInfo, TryLow, TryHigh, CatchHigh, Handlers);
    DEBUG(dbgs() << "TryLow[" << FirstTryPad->getName() << "]: " << TryLow
                 << '\n');
    DEBUG(dbgs() << "TryHigh[" << FirstTryPad->getName() << "]: " << TryHigh
                 << '\n');
    DEBUG(dbgs() << "CatchHigh[" << FirstTryPad->getName() << "]: " << CatchHigh
                 << '\n');
  } else if (isa<CleanupPadInst>(FirstNonPHI)) {
    // A cleanup can have multiple exits; don't re-process after the first.
    if (FuncInfo.EHPadStateMap.count(FirstNonPHI))
      return;
    int CleanupState = addUnwindMapEntry(FuncInfo, ParentState, &BB);
    FuncInfo.EHPadStateMap[FirstNonPHI] = CleanupState;
    DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, CleanupState);
  } else if (auto *CEPI = dyn_cast<CleanupEndPadInst>(FirstNonPHI)) {
    // Propagate ParentState to the cleanuppad in case it doesn't have
    // any cleanuprets.
    BasicBlock *CleanupBlock = CEPI->getCleanupPad()->getParent();
    calculateExplicitCXXStateNumbers(FuncInfo, *CleanupBlock, ParentState);
    // Anything unwinding through CleanupEndPadInst is in ParentState.
    FuncInfo.EHPadStateMap[FirstNonPHI] = ParentState;
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitCXXStateNumbers(FuncInfo, *PredBlock, ParentState);
  } else if (isa<TerminatePadInst>(FirstNonPHI)) {
    report_fatal_error("Not yet implemented!");
  } else {
    llvm_unreachable("unexpected EH Pad!");
  }
}

static int addSEHExcept(WinEHFuncInfo &FuncInfo, int ParentState,
                        const Function *Filter, const BasicBlock *Handler) {
  SEHUnwindMapEntry Entry;
  Entry.ToState = ParentState;
  Entry.IsFinally = false;
  Entry.Filter = Filter;
  Entry.Handler = Handler;
  FuncInfo.SEHUnwindMap.push_back(Entry);
  return FuncInfo.SEHUnwindMap.size() - 1;
}

static int addSEHFinally(WinEHFuncInfo &FuncInfo, int ParentState,
                         const BasicBlock *Handler) {
  SEHUnwindMapEntry Entry;
  Entry.ToState = ParentState;
  Entry.IsFinally = true;
  Entry.Filter = nullptr;
  Entry.Handler = Handler;
  FuncInfo.SEHUnwindMap.push_back(Entry);
  return FuncInfo.SEHUnwindMap.size() - 1;
}

static void calculateExplicitSEHStateNumbers(WinEHFuncInfo &FuncInfo,
                                             const BasicBlock &BB,
                                             int ParentState) {
  assert(BB.isEHPad());
  const Instruction *FirstNonPHI = BB.getFirstNonPHI();
  // All catchpad instructions will be handled when we process their
  // respective catchendpad instruction.
  if (isa<CatchPadInst>(FirstNonPHI))
    return;

  if (isa<CatchEndPadInst>(FirstNonPHI)) {
    // Extract the filter function and the __except basic block and create a
    // state for them.
    SmallVector<const CatchPadInst *, 1> Handlers;
    findCatchPadsForCatchEndPad(&BB, Handlers);
    assert(Handlers.size() == 1 &&
           "SEH doesn't have multiple handlers per __try");
    const CatchPadInst *CPI = Handlers.front();
    const BasicBlock *CatchPadBB = CPI->getParent();
    const Constant *FilterOrNull =
        cast<Constant>(CPI->getArgOperand(0)->stripPointerCasts());
    const Function *Filter = dyn_cast<Function>(FilterOrNull);
    assert((Filter || FilterOrNull->isNullValue()) &&
           "unexpected filter value");
    int TryState = addSEHExcept(FuncInfo, ParentState, Filter, CatchPadBB);

    // Everything in the __try block uses TryState as its parent state.
    FuncInfo.EHPadStateMap[CPI] = TryState;
    DEBUG(dbgs() << "Assigning state #" << TryState << " to BB "
                 << CatchPadBB->getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(CatchPadBB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, TryState);

    // Everything in the __except block unwinds to ParentState, just like code
    // outside the __try.
    FuncInfo.EHPadStateMap[FirstNonPHI] = ParentState;
    DEBUG(dbgs() << "Assigning state #" << ParentState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, ParentState);
  } else if (isa<CleanupPadInst>(FirstNonPHI)) {
    // A cleanup can have multiple exits; don't re-process after the first.
    if (FuncInfo.EHPadStateMap.count(FirstNonPHI))
      return;
    int CleanupState = addSEHFinally(FuncInfo, ParentState, &BB);
    FuncInfo.EHPadStateMap[FirstNonPHI] = CleanupState;
    DEBUG(dbgs() << "Assigning state #" << CleanupState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, CleanupState);
  } else if (auto *CEPI = dyn_cast<CleanupEndPadInst>(FirstNonPHI)) {
    // Propagate ParentState to the cleanuppad in case it doesn't have
    // any cleanuprets.
    BasicBlock *CleanupBlock = CEPI->getCleanupPad()->getParent();
    calculateExplicitSEHStateNumbers(FuncInfo, *CleanupBlock, ParentState);
    // Anything unwinding through CleanupEndPadInst is in ParentState.
    FuncInfo.EHPadStateMap[FirstNonPHI] = ParentState;
    DEBUG(dbgs() << "Assigning state #" << ParentState << " to BB "
                 << BB.getName() << '\n');
    for (const BasicBlock *PredBlock : predecessors(&BB))
      if ((PredBlock = getEHPadFromPredecessor(PredBlock)))
        calculateExplicitSEHStateNumbers(FuncInfo, *PredBlock, ParentState);
  } else if (isa<TerminatePadInst>(FirstNonPHI)) {
    report_fatal_error("Not yet implemented!");
  } else {
    llvm_unreachable("unexpected EH Pad!");
  }
}

/// Check if the EH Pad unwinds to caller.  Cleanups are a little bit of a
/// special case because we have to look at the cleanupret instruction that uses
/// the cleanuppad.
static bool doesEHPadUnwindToCaller(const Instruction *EHPad) {
  auto *CPI = dyn_cast<CleanupPadInst>(EHPad);
  if (!CPI)
    return EHPad->mayThrow();

  // This cleanup does not return or unwind, so we say it unwinds to caller.
  if (CPI->use_empty())
    return true;

  const Instruction *User = CPI->user_back();
  if (auto *CRI = dyn_cast<CleanupReturnInst>(User))
    return CRI->unwindsToCaller();
  return cast<CleanupEndPadInst>(User)->unwindsToCaller();
}

void llvm::calculateSEHStateNumbers(const Function *Fn,
                                    WinEHFuncInfo &FuncInfo) {
  // Don't compute state numbers twice.
  if (!FuncInfo.SEHUnwindMap.empty())
    return;

  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad() || !doesEHPadUnwindToCaller(BB.getFirstNonPHI()))
      continue;
    calculateExplicitSEHStateNumbers(FuncInfo, BB, -1);
  }
}

void llvm::calculateWinCXXEHStateNumbers(const Function *Fn,
                                         WinEHFuncInfo &FuncInfo) {
  // Return if it's already been done.
  if (!FuncInfo.EHPadStateMap.empty())
    return;

  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad())
      continue;
    if (BB.isLandingPad())
      report_fatal_error("MSVC C++ EH cannot use landingpads");
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    if (!doesEHPadUnwindToCaller(FirstNonPHI))
      continue;
    calculateExplicitCXXStateNumbers(FuncInfo, BB, -1);
  }
}

static int addClrEHHandler(WinEHFuncInfo &FuncInfo, int ParentState,
                           ClrHandlerType HandlerType, uint32_t TypeToken,
                           const BasicBlock *Handler) {
  ClrEHUnwindMapEntry Entry;
  Entry.Parent = ParentState;
  Entry.Handler = Handler;
  Entry.HandlerType = HandlerType;
  Entry.TypeToken = TypeToken;
  FuncInfo.ClrEHUnwindMap.push_back(Entry);
  return FuncInfo.ClrEHUnwindMap.size() - 1;
}

void llvm::calculateClrEHStateNumbers(const Function *Fn,
                                      WinEHFuncInfo &FuncInfo) {
  // Return if it's already been done.
  if (!FuncInfo.EHPadStateMap.empty())
    return;

  SmallVector<std::pair<const Instruction *, int>, 8> Worklist;

  // Each pad needs to be able to refer to its parent, so scan the function
  // looking for top-level handlers and seed the worklist with them.
  for (const BasicBlock &BB : *Fn) {
    if (!BB.isEHPad())
      continue;
    if (BB.isLandingPad())
      report_fatal_error("CoreCLR EH cannot use landingpads");
    const Instruction *FirstNonPHI = BB.getFirstNonPHI();
    if (!doesEHPadUnwindToCaller(FirstNonPHI))
      continue;
    // queue this with sentinel parent state -1 to mean unwind to caller.
    Worklist.emplace_back(FirstNonPHI, -1);
  }

  while (!Worklist.empty()) {
    const Instruction *Pad;
    int ParentState;
    std::tie(Pad, ParentState) = Worklist.pop_back_val();

    int PredState;
    if (const CleanupEndPadInst *EndPad = dyn_cast<CleanupEndPadInst>(Pad)) {
      FuncInfo.EHPadStateMap[EndPad] = ParentState;
      // Queue the cleanuppad, in case it doesn't have a cleanupret.
      Worklist.emplace_back(EndPad->getCleanupPad(), ParentState);
      // Preds of the endpad should get the parent state.
      PredState = ParentState;
    } else if (const CleanupPadInst *Cleanup = dyn_cast<CleanupPadInst>(Pad)) {
      // A cleanup can have multiple exits; don't re-process after the first.
      if (FuncInfo.EHPadStateMap.count(Pad))
        continue;
      // CoreCLR personality uses arity to distinguish faults from finallies.
      const BasicBlock *PadBlock = Cleanup->getParent();
      ClrHandlerType HandlerType =
          (Cleanup->getNumOperands() ? ClrHandlerType::Fault
                                     : ClrHandlerType::Finally);
      int NewState =
          addClrEHHandler(FuncInfo, ParentState, HandlerType, 0, PadBlock);
      FuncInfo.EHPadStateMap[Cleanup] = NewState;
      // Propagate the new state to all preds of the cleanup
      PredState = NewState;
    } else if (const CatchEndPadInst *EndPad = dyn_cast<CatchEndPadInst>(Pad)) {
      FuncInfo.EHPadStateMap[EndPad] = ParentState;
      // Preds of the endpad should get the parent state.
      PredState = ParentState;
    } else if (const CatchPadInst *Catch = dyn_cast<CatchPadInst>(Pad)) {
      const BasicBlock *PadBlock = Catch->getParent();
      uint32_t TypeToken = static_cast<uint32_t>(
          cast<ConstantInt>(Catch->getArgOperand(0))->getZExtValue());
      int NewState = addClrEHHandler(FuncInfo, ParentState,
                                     ClrHandlerType::Catch, TypeToken, PadBlock);
      FuncInfo.EHPadStateMap[Catch] = NewState;
      // Preds of the catch get its state
      PredState = NewState;
    } else {
      llvm_unreachable("Unexpected EH pad");
    }

    // Queue all predecessors with the given state
    for (const BasicBlock *Pred : predecessors(Pad->getParent())) {
      if ((Pred = getEHPadFromPredecessor(Pred)))
        Worklist.emplace_back(Pred->getFirstNonPHI(), PredState);
    }
  }
}

void WinEHPrepare::replaceTerminatePadWithCleanup(Function &F) {
  if (Personality != EHPersonality::MSVC_CXX)
    return;
  for (BasicBlock &BB : F) {
    Instruction *First = BB.getFirstNonPHI();
    auto *TPI = dyn_cast<TerminatePadInst>(First);
    if (!TPI)
      continue;

    if (TPI->getNumArgOperands() != 1)
      report_fatal_error(
          "Expected a unary terminatepad for MSVC C++ personalities!");

    auto *TerminateFn = dyn_cast<Function>(TPI->getArgOperand(0));
    if (!TerminateFn)
      report_fatal_error("Function operand expected in terminatepad for MSVC "
                         "C++ personalities!");

    // Insert the cleanuppad instruction.
    auto *CPI = CleanupPadInst::Create(
        BB.getContext(), {}, Twine("terminatepad.for.", BB.getName()), &BB);

    // Insert the call to the terminate instruction.
    auto *CallTerminate = CallInst::Create(TerminateFn, {}, &BB);
    CallTerminate->setDoesNotThrow();
    CallTerminate->setDoesNotReturn();
    CallTerminate->setCallingConv(TerminateFn->getCallingConv());

    // Insert a new terminator for the cleanuppad using the same successor as
    // the terminatepad.
    CleanupReturnInst::Create(CPI, TPI->getUnwindDest(), &BB);

    // Let's remove the terminatepad now that we've inserted the new
    // instructions.
    TPI->eraseFromParent();
  }
}

static void
colorFunclets(Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks,
              std::map<BasicBlock *, SetVector<BasicBlock *>> &BlockColors,
              std::map<BasicBlock *, std::set<BasicBlock *>> &FuncletBlocks) {
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 16> Worklist;
  BasicBlock *EntryBlock = &F.getEntryBlock();

  // Build up the color map, which maps each block to its set of 'colors'.
  // For any block B, the "colors" of B are the set of funclets F (possibly
  // including a root "funclet" representing the main function), such that
  // F will need to directly contain B or a copy of B (where the term "directly
  // contain" is used to distinguish from being "transitively contained" in
  // a nested funclet).
  // Use a CFG walk driven by a worklist of (block, color) pairs.  The "color"
  // sets attached during this processing to a block which is the entry of some
  // funclet F is actually the set of F's parents -- i.e. the union of colors
  // of all predecessors of F's entry.  For all other blocks, the color sets
  // are as defined above.  A post-pass fixes up the block color map to reflect
  // the same sense of "color" for funclet entries as for other blocks.

  DEBUG_WITH_TYPE("winehprepare-coloring", dbgs() << "\nColoring funclets for "
                                                  << F.getName() << "\n");

  Worklist.push_back({EntryBlock, EntryBlock});

  while (!Worklist.empty()) {
    BasicBlock *Visiting;
    BasicBlock *Color;
    std::tie(Visiting, Color) = Worklist.pop_back_val();
    DEBUG_WITH_TYPE("winehprepare-coloring",
                    dbgs() << "Visiting " << Visiting->getName() << ", "
                           << Color->getName() << "\n");
    Instruction *VisitingHead = Visiting->getFirstNonPHI();
    if (VisitingHead->isEHPad() && !isa<CatchEndPadInst>(VisitingHead) &&
        !isa<CleanupEndPadInst>(VisitingHead)) {
      // Mark this as a funclet head as a member of itself.
      FuncletBlocks[Visiting].insert(Visiting);
      // Queue exits (i.e. successors of rets/endpads) with the parent color.
      // Skip any exits that are catchendpads, since the parent color must then
      // represent one of the catches chained to that catchendpad, but the
      // catchendpad should get the color of the common parent of all its
      // chained catches (i.e. the grandparent color of the current pad).
      // We don't need to worry abou catchendpads going unvisited, since the
      // catches chained to them must have unwind edges to them by which we will
      // visit them.
      for (User *U : VisitingHead->users()) {
        if (auto *Exit = dyn_cast<TerminatorInst>(U)) {
          for (BasicBlock *Succ : successors(Exit->getParent()))
            if (!isa<CatchEndPadInst>(*Succ->getFirstNonPHI()))
              if (BlockColors[Succ].insert(Color)) {
                DEBUG_WITH_TYPE("winehprepare-coloring",
                                dbgs() << "  Assigned color \'"
                                       << Color->getName() << "\' to block \'"
                                       << Succ->getName() << "\'.\n");
                Worklist.push_back({Succ, Color});
              }
        }
      }
      // Handle CatchPad specially since its successors need different colors.
      if (CatchPadInst *CatchPad = dyn_cast<CatchPadInst>(VisitingHead)) {
        // Visit the normal successor with the color of the new EH pad, and
        // visit the unwind successor with the color of the parent.
        BasicBlock *NormalSucc = CatchPad->getNormalDest();
        if (BlockColors[NormalSucc].insert(Visiting)) {
          DEBUG_WITH_TYPE("winehprepare-coloring",
                          dbgs() << "  Assigned color \'" << Visiting->getName()
                                 << "\' to block \'" << NormalSucc->getName()
                                 << "\'.\n");
          Worklist.push_back({NormalSucc, Visiting});
        }
        BasicBlock *UnwindSucc = CatchPad->getUnwindDest();
        if (BlockColors[UnwindSucc].insert(Color)) {
          DEBUG_WITH_TYPE("winehprepare-coloring",
                          dbgs() << "  Assigned color \'" << Color->getName()
                                 << "\' to block \'" << UnwindSucc->getName()
                                 << "\'.\n");
          Worklist.push_back({UnwindSucc, Color});
        }
        continue;
      }
      // Switch color to the current node, except for terminate pads which
      // have no bodies and only unwind successors and so need their successors
      // visited with the color of the parent.
      if (!isa<TerminatePadInst>(VisitingHead))
        Color = Visiting;
    } else {
      // Note that this is a member of the given color.
      FuncletBlocks[Color].insert(Visiting);
    }

    TerminatorInst *Terminator = Visiting->getTerminator();
    if (isa<CleanupReturnInst>(Terminator) ||
        isa<CatchReturnInst>(Terminator) ||
        isa<CleanupEndPadInst>(Terminator)) {
      // These blocks' successors have already been queued with the parent
      // color.
      continue;
    }
    for (BasicBlock *Succ : successors(Visiting)) {
      if (isa<CatchEndPadInst>(Succ->getFirstNonPHI())) {
        // The catchendpad needs to be visited with the parent's color, not
        // the current color.  This will happen in the code above that visits
        // any catchpad unwind successor with the parent color, so we can
        // safely skip this successor here.
        continue;
      }
      if (BlockColors[Succ].insert(Color)) {
        DEBUG_WITH_TYPE("winehprepare-coloring",
                        dbgs() << "  Assigned color \'" << Color->getName()
                               << "\' to block \'" << Succ->getName()
                               << "\'.\n");
        Worklist.push_back({Succ, Color});
      }
    }
  }
}

static BasicBlock *getEndPadForCatch(CatchPadInst *Catch) {
  // The catch may have sibling catches.  Follow the unwind chain until we get
  // to the catchendpad.
  BasicBlock *NextUnwindDest = Catch->getUnwindDest();
  auto *UnwindTerminator = NextUnwindDest->getTerminator();
  while (auto *NextCatch = dyn_cast<CatchPadInst>(UnwindTerminator)) {
    NextUnwindDest = NextCatch->getUnwindDest();
    UnwindTerminator = NextUnwindDest->getTerminator();
  }
  // The last catch in the chain must unwind to a catchendpad.
  assert(isa<CatchEndPadInst>(UnwindTerminator));
  return NextUnwindDest;
}

static void updateClonedEHPadUnwindToParent(
    BasicBlock *UnwindDest, BasicBlock *OrigBlock, BasicBlock *CloneBlock,
    std::vector<BasicBlock *> &OrigParents, BasicBlock *CloneParent) {
  auto updateUnwindTerminator = [](BasicBlock *BB) {
    auto *Terminator = BB->getTerminator();
    if (isa<CatchEndPadInst>(Terminator) ||
        isa<CleanupEndPadInst>(Terminator)) {
      removeUnwindEdge(BB);
    } else {
      // If the block we're updating has a cleanupendpad or cleanupret
      // terminator, we just want to replace that terminator with an
      // unreachable instruction.
      assert(isa<CleanupEndPadInst>(Terminator) ||
             isa<CleanupReturnInst>(Terminator));
      // Loop over all of the successors, removing the block's entry from any
      // PHI nodes.
      for (succ_iterator SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI)
        (*SI)->removePredecessor(BB);
      // Remove the terminator and replace it with an unreachable instruction.
      BB->getTerminator()->eraseFromParent();
      new UnreachableInst(BB->getContext(), BB);
    }
  };

  assert(UnwindDest->isEHPad());
  // There are many places to which this EH terminator can unwind and each has
  // slightly different rules for whether or not it fits with the given
  // location.
  auto *EHPadInst = UnwindDest->getFirstNonPHI();
  if (isa<CatchEndPadInst>(EHPadInst)) {
    auto *CloneParentCatch =
        dyn_cast<CatchPadInst>(CloneParent->getFirstNonPHI());
    if (!CloneParentCatch ||
        getEndPadForCatch(CloneParentCatch) != UnwindDest) {
      DEBUG_WITH_TYPE(
          "winehprepare-coloring",
          dbgs() << "      removing unwind destination of clone block \'"
                 << CloneBlock->getName() << "\'.\n");
      updateUnwindTerminator(CloneBlock);
    }
    // It's possible that the catch end pad is a legal match for both the clone
    // and the original, so they must be checked separately.  If the original
    // funclet will still have multiple parents after the current clone parent
    // is removed, we'll leave its unwind terminator until later.
    assert(OrigParents.size() >= 2);
    if (OrigParents.size() != 2)
      return;

    // If the original funclet will have a single parent after the clone parent
    // is removed, check that parent's unwind destination.
    assert(OrigParents.front() == CloneParent ||
           OrigParents.back() == CloneParent);
    BasicBlock *OrigParent;
    if (OrigParents.front() == CloneParent)
      OrigParent = OrigParents.back();
    else
      OrigParent = OrigParents.front();

    auto *OrigParentCatch =
        dyn_cast<CatchPadInst>(OrigParent->getFirstNonPHI());
    if (!OrigParentCatch || getEndPadForCatch(OrigParentCatch) != UnwindDest) {
      DEBUG_WITH_TYPE(
          "winehprepare-coloring",
          dbgs() << "      removing unwind destination of original block \'"
                 << OrigBlock << "\'.\n");
      updateUnwindTerminator(OrigBlock);
    }
  } else if (auto *CleanupEnd = dyn_cast<CleanupEndPadInst>(EHPadInst)) {
    // If the EH terminator unwinds to a cleanupendpad, that cleanupendpad
    // must be ending a cleanuppad of either our clone parent or one
    // one of the parents of the original funclet.
    auto *CloneParentCP =
        dyn_cast<CleanupPadInst>(CloneParent->getFirstNonPHI());
    auto *EndedCP = CleanupEnd->getCleanupPad();
    if (EndedCP == CloneParentCP) {
      // If it is ending the cleanuppad of our cloned parent, then we
      // want to remove the unwind destination of the EH terminator that
      // we associated with the original funclet.
      assert(isa<CatchEndPadInst>(OrigBlock->getFirstNonPHI()));
      DEBUG_WITH_TYPE(
          "winehprepare-coloring",
          dbgs() << "      removing unwind destination of original block \'"
                 << OrigBlock->getName() << "\'.\n");
      updateUnwindTerminator(OrigBlock);
    } else {
      // If it isn't ending the cleanuppad of our clone parent, then we
      // want to remove the unwind destination of the EH terminator that
      // associated with our cloned funclet.
      assert(isa<CatchEndPadInst>(CloneBlock->getFirstNonPHI()));
      DEBUG_WITH_TYPE(
          "winehprepare-coloring",
          dbgs() << "      removing unwind destination of clone block \'"
                 << CloneBlock->getName() << "\'.\n");
      updateUnwindTerminator(CloneBlock);
    }
  } else {
    // If the EH terminator unwinds to a catchpad, cleanuppad or
    // terminatepad that EH pad must be a sibling of the funclet we're
    // cloning.  We'll clone it later and update one of the catchendpad
    // instrunctions that unwinds to it at that time.
    assert(isa<CatchPadInst>(EHPadInst) || isa<CleanupPadInst>(EHPadInst) ||
           isa<TerminatePadInst>(EHPadInst));
  }
}

// If the terminator is a catchpad, we must also clone the catchendpad to which
// it unwinds and add this to the clone parent's block list.  The catchendpad
// unwinds to either its caller, a sibling EH pad, a cleanup end pad in its
// parent funclet or a catch end pad in its grandparent funclet (which must be
// coupled with the parent funclet).  If it has no unwind destination
// (i.e. unwind to caller), there is nothing to be done. If the unwind
// destination is a sibling EH pad, we will update the terminators later (in
// resolveFuncletAncestryForPath).  If it unwinds to a cleanup end pad or a
// catch end pad and this end pad corresponds to the clone parent, we will
// remove the unwind destination in the original catchendpad. If it unwinds to
// a cleanup end pad or a catch end pad that does not correspond to the clone
// parent, we will remove the unwind destination in the cloned catchendpad.
static void updateCatchTerminators(
    Function &F, CatchPadInst *OrigCatch, CatchPadInst *CloneCatch,
    std::vector<BasicBlock *> &OrigParents, BasicBlock *CloneParent,
    ValueToValueMapTy &VMap,
    std::map<BasicBlock *, SetVector<BasicBlock *>> &BlockColors,
    std::map<BasicBlock *, std::set<BasicBlock *>> &FuncletBlocks) {
  // If we're cloning a catch pad that unwinds to a catchendpad, we also
  // need to clone the catchendpad.  The coloring algorithm associates
  // the catchendpad block with the funclet's parent, so we have some work
  // to do here to figure out whether the original belongs to the clone
  // parent or one of the original funclets other parents (it might have
  // more than one at this point).  In either case, we might also need to
  // remove the unwind edge if the catchendpad doesn't unwind to a block
  // in the right grandparent funclet.
  Instruction *I = CloneCatch->getUnwindDest()->getFirstNonPHI();
  if (auto *CEP = dyn_cast<CatchEndPadInst>(I)) {
    assert(BlockColors[CEP->getParent()].size() == 1);
    BasicBlock *CEPFunclet = *(BlockColors[CEP->getParent()].begin());
    BasicBlock *CEPCloneParent = nullptr;
    CatchPadInst *PredCatch = nullptr;
    if (CEPFunclet == CloneParent) {
      // The catchendpad is in the clone parent, so we need to clone it
      // and associate the clone with the original funclet's parent.  If
      // the original funclet had multiple parents, we'll add it to the
      // first parent that isn't the clone parent.  The logic in
      // updateClonedEHPadUnwindToParent() will only remove the unwind edge
      // if there is only one parent other than the clone parent, so we don't
      // need to verify the ancestry.  The catchendpad will eventually be
      // cloned into the correct parent and all invalid unwind edges will be
      // removed.
      for (auto *Parent : OrigParents) {
        if (Parent != CloneParent) {
          CEPCloneParent = Parent;
          break;
        }
      }
      PredCatch = OrigCatch;
    } else {
      CEPCloneParent = CloneParent;
      PredCatch = CloneCatch;
    }
    assert(CEPCloneParent && PredCatch);
    DEBUG_WITH_TYPE("winehprepare-coloring",
                    dbgs() << "  Cloning catchendpad \'"
                           << CEP->getParent()->getName() << "\' for funclet \'"
                           << CEPCloneParent->getName() << "\'.\n");
    BasicBlock *ClonedCEP = CloneBasicBlock(
        CEP->getParent(), VMap, Twine(".from.", CEPCloneParent->getName()));
    // Insert the clone immediately after the original to ensure determinism
    // and to keep the same relative ordering of any funclet's blocks.
    ClonedCEP->insertInto(&F, CEP->getParent()->getNextNode());
    PredCatch->setUnwindDest(ClonedCEP);
    FuncletBlocks[CEPCloneParent].insert(ClonedCEP);
    BlockColors[ClonedCEP].insert(CEPCloneParent);
    DEBUG_WITH_TYPE("winehprepare-coloring",
                    dbgs() << "    Assigning color \'"
                           << CEPCloneParent->getName() << "\' to block \'"
                           << ClonedCEP->getName() << "\'.\n");
    auto *ClonedCEPInst = cast<CatchEndPadInst>(ClonedCEP->getTerminator());
    if (auto *Dest = ClonedCEPInst->getUnwindDest())
      updateClonedEHPadUnwindToParent(Dest, OrigCatch->getUnwindDest(),
                                      CloneCatch->getUnwindDest(), OrigParents,
                                      CloneParent);
  }
}

// While we are cloning a funclet because it has multiple parents, we will call
// this routine to update the terminators for the original and cloned copies
// of each basic block.  All blocks in the funclet have been clone by this time.
// OrigBlock and CloneBlock will be identical except for their block label.
//
// If the terminator is a catchpad, we must also clone the catchendpad to which
// it unwinds and in most cases update either the original catchendpad or the
// clone.  See the updateCatchTerminators() helper routine for details.
//
// If the terminator is a catchret its successor is a block in its parent
// funclet.  If the instruction returns to a block in the parent for which the
// cloned funclet was created, the terminator in the original block must be
// replaced by an unreachable instruction.  Otherwise the terminator in the
// clone block must be replaced by an unreachable instruction.
//
// If the terminator is a cleanupret or cleanupendpad it either unwinds to
// caller or unwinds to a sibling EH pad, a cleanup end pad in its parent
// funclet or a catch end pad in its grandparent funclet (which must be
// coupled with the parent funclet).  If it unwinds to caller there is
// nothing to be done. If the unwind destination is a sibling EH pad, we will
// update the terminators later (in resolveFuncletAncestryForPath).  If it
// unwinds to a cleanup end pad or a catch end pad and this end pad corresponds
// to the clone parent, we will replace the terminator in the original block
// with an unreachable instruction. If it unwinds to a cleanup end pad or a
// catch end pad that does not correspond to the clone parent, we will replace
// the terminator in the clone block with an unreachable instruction.
//
// If the terminator is an invoke instruction, we will handle it after all
// siblings of the current funclet have been cloned.
void WinEHPrepare::updateTerminatorsAfterFuncletClone(
    Function &F, BasicBlock *OrigFunclet, BasicBlock *CloneFunclet,
    BasicBlock *OrigBlock, BasicBlock *CloneBlock, BasicBlock *CloneParent,
    ValueToValueMapTy &VMap, std::map<BasicBlock *, BasicBlock *> &Orig2Clone) {
  // If the cloned block doesn't have an exceptional terminator, there is
  // nothing to be done here.
  TerminatorInst *CloneTerminator = CloneBlock->getTerminator();
  if (!CloneTerminator->isExceptional())
    return;

  if (auto *CloneCatch = dyn_cast<CatchPadInst>(CloneTerminator)) {
    // A cloned catch pad has a lot of wrinkles, so we'll call a helper function
    // to update this case.
    auto *OrigCatch = cast<CatchPadInst>(OrigBlock->getTerminator());
    updateCatchTerminators(F, OrigCatch, CloneCatch,
                           FuncletParents[OrigFunclet], CloneParent, VMap,
                           BlockColors, FuncletBlocks);
  } else if (auto *CRI = dyn_cast<CatchReturnInst>(CloneTerminator)) {
    if (FuncletBlocks[CloneParent].count(CRI->getSuccessor())) {
      BasicBlock *OrigParent;
      // The original funclet may have more than two parents, but that's OK.
      // We just need to remap the original catchret to any of the parents.
      // All of the parents should have an entry in the EstrangedBlocks map
      // if any of them do.
      if (FuncletParents[OrigFunclet].front() == CloneParent)
        OrigParent = FuncletParents[OrigFunclet].back();
      else
        OrigParent = FuncletParents[OrigFunclet].front();
      for (succ_iterator SI = succ_begin(OrigBlock), SE = succ_end(OrigBlock);
           SI != SE; ++SI)
        (*SI)->removePredecessor(OrigBlock);
      BasicBlock *LostBlock = EstrangedBlocks[OrigParent][CRI->getSuccessor()];
      auto *OrigCatchRet = cast<CatchReturnInst>(OrigBlock->getTerminator());
      if (LostBlock) {
        OrigCatchRet->setSuccessor(LostBlock);
      } else {
        OrigCatchRet->eraseFromParent();
        new UnreachableInst(OrigBlock->getContext(), OrigBlock);
      }
    } else {
      for (succ_iterator SI = succ_begin(CloneBlock), SE = succ_end(CloneBlock);
           SI != SE; ++SI)
        (*SI)->removePredecessor(CloneBlock);
      BasicBlock *LostBlock = EstrangedBlocks[CloneParent][CRI->getSuccessor()];
      if (LostBlock) {
        CRI->setSuccessor(LostBlock);
      } else {
        CRI->eraseFromParent();
        new UnreachableInst(CloneBlock->getContext(), CloneBlock);
      }
    }
  } else if (isa<CleanupReturnInst>(CloneTerminator) ||
             isa<CleanupEndPadInst>(CloneTerminator)) {
    BasicBlock *UnwindDest = nullptr;

    // A cleanup pad can unwind through either a cleanupret or a cleanupendpad
    // but both are handled the same way.
    if (auto *CRI = dyn_cast<CleanupReturnInst>(CloneTerminator))
      UnwindDest = CRI->getUnwindDest();
    else if (auto *CEI = dyn_cast<CleanupEndPadInst>(CloneTerminator))
      UnwindDest = CEI->getUnwindDest();

    // If the instruction has no local unwind destination, there is nothing
    // to be done.
    if (!UnwindDest)
      return;

    // The unwind destination may be a sibling EH pad, a catchendpad in
    // a grandparent funclet (ending a catchpad in the parent) or a cleanup
    // cleanupendpad in the parent.  Call a helper routine to diagnose this
    // and remove either the clone or original terminator as needed.
    updateClonedEHPadUnwindToParent(UnwindDest, OrigBlock, CloneBlock,
                                    FuncletParents[OrigFunclet], CloneParent);
  }
}

// Clones all blocks used by the specified funclet to avoid the funclet having
// multiple parent funclets.  All terminators in the parent that unwind to the
// original funclet are remapped to unwind to the clone.  Any terminator in the
// original funclet which returned to this parent is converted to an unreachable
// instruction. Likewise, any terminator in the cloned funclet which returns to
// a parent funclet other than the specified parent is converted to an
// unreachable instruction.
BasicBlock *WinEHPrepare::cloneFuncletForParent(Function &F,
                                                BasicBlock *FuncletEntry,
                                                BasicBlock *Parent) {
  std::set<BasicBlock *> &BlocksInFunclet = FuncletBlocks[FuncletEntry];

  DEBUG_WITH_TYPE("winehprepare-coloring",
                  dbgs() << "Cloning funclet \'" << FuncletEntry->getName()
                         << "\' for parent \'" << Parent->getName() << "\'.\n");

  std::map<BasicBlock *, BasicBlock *> Orig2Clone;
  ValueToValueMapTy VMap;
  for (BasicBlock *BB : BlocksInFunclet) {
    // Create a new basic block and copy instructions into it.
    BasicBlock *CBB =
        CloneBasicBlock(BB, VMap, Twine(".from.", Parent->getName()));

    // Insert the clone immediately after the original to ensure determinism
    // and to keep the same relative ordering of any funclet's blocks.
    CBB->insertInto(&F, BB->getNextNode());

    // Add basic block mapping.
    VMap[BB] = CBB;

    // Record delta operations that we need to perform to our color mappings.
    Orig2Clone[BB] = CBB;
  } // end for (BasicBlock *BB : BlocksInFunclet)

  BasicBlock *ClonedFunclet = Orig2Clone[FuncletEntry];
  assert(ClonedFunclet);

  // Set the coloring for the blocks we just cloned.
  std::set<BasicBlock *> &ClonedBlocks = FuncletBlocks[ClonedFunclet];
  for (auto &BBMapping : Orig2Clone) {
    BasicBlock *NewBlock = BBMapping.second;
    ClonedBlocks.insert(NewBlock);
    BlockColors[NewBlock].insert(ClonedFunclet);

    DEBUG_WITH_TYPE("winehprepare-coloring",
                    dbgs() << "  Assigning color \'" << ClonedFunclet->getName()
                           << "\' to block \'" << NewBlock->getName()
                           << "\'.\n");

    // Use the VMap to remap the instructions in this cloned block.
    for (Instruction &I : *NewBlock)
      RemapInstruction(&I, VMap, RF_IgnoreMissingEntries);
  }

  // All the cloned blocks have to be colored in the loop above before we can
  // update the terminators because doing so can require checking the color of
  // other blocks in the cloned funclet.
  for (auto &BBMapping : Orig2Clone) {
    BasicBlock *OldBlock = BBMapping.first;
    BasicBlock *NewBlock = BBMapping.second;

    // Update the terminator, if necessary, in both the original block and the
    // cloned so that the original funclet never returns to a block in the
    // clone parent and the clone funclet never returns to a block in any other
    // of the original funclet's parents.
    updateTerminatorsAfterFuncletClone(F, FuncletEntry, ClonedFunclet, OldBlock,
                                       NewBlock, Parent, VMap, Orig2Clone);

    // Check to see if the cloned block successor has PHI nodes. If so, we need
    // to add entries to the PHI nodes for the cloned block now.
    for (BasicBlock *SuccBB : successors(NewBlock)) {
      for (Instruction &SuccI : *SuccBB) {
        auto *SuccPN = dyn_cast<PHINode>(&SuccI);
        if (!SuccPN)
          break;

        // Ok, we have a PHI node.  Figure out what the incoming value was for
        // the OldBlock.
        int OldBlockIdx = SuccPN->getBasicBlockIndex(OldBlock);
        if (OldBlockIdx == -1)
          break;
        Value *IV = SuccPN->getIncomingValue(OldBlockIdx);

        // Remap the value if necessary.
        if (auto *Inst = dyn_cast<Instruction>(IV)) {
          ValueToValueMapTy::iterator I = VMap.find(Inst);
          if (I != VMap.end())
            IV = I->second;
        }

        SuccPN->addIncoming(IV, NewBlock);
      }
    }
  }

  // Erase the clone's parent from the original funclet's parent list.
  std::vector<BasicBlock *> &Parents = FuncletParents[FuncletEntry];
  Parents.erase(std::remove(Parents.begin(), Parents.end(), Parent),
                Parents.end());

  // Store the cloned funclet's parent.
  assert(std::find(FuncletParents[ClonedFunclet].begin(),
                   FuncletParents[ClonedFunclet].end(),
                   Parent) == std::end(FuncletParents[ClonedFunclet]));
  FuncletParents[ClonedFunclet].push_back(Parent);

  // Copy any children of the original funclet to the clone.  We'll either
  // clone them too or make that path unreachable when we take the next step
  // in resolveFuncletAncestryForPath().
  for (auto *Child : FuncletChildren[FuncletEntry]) {
    assert(std::find(FuncletChildren[ClonedFunclet].begin(),
                     FuncletChildren[ClonedFunclet].end(),
                     Child) == std::end(FuncletChildren[ClonedFunclet]));
    FuncletChildren[ClonedFunclet].push_back(Child);
    assert(std::find(FuncletParents[Child].begin(), FuncletParents[Child].end(),
                     ClonedFunclet) == std::end(FuncletParents[Child]));
    FuncletParents[Child].push_back(ClonedFunclet);
  }

  // Find any blocks that unwound to the original funclet entry from the
  // clone parent block and remap them to the clone.
  for (auto *U : FuncletEntry->users()) {
    auto *UT = dyn_cast<TerminatorInst>(U);
    if (!UT)
      continue;
    BasicBlock *UBB = UT->getParent();
    assert(BlockColors[UBB].size() == 1);
    BasicBlock *UFunclet = *(BlockColors[UBB].begin());
    // Funclets shouldn't be able to loop back on themselves.
    assert(UFunclet != FuncletEntry);
    // If this instruction unwinds to the original funclet from the clone
    // parent, remap the terminator so that it unwinds to the clone instead.
    // We will perform a similar transformation for siblings after all
    // the siblings have been cloned.
    if (UFunclet == Parent) {
      // We're about to break the path from this block to the uncloned funclet
      // entry, so remove it as a predeccessor to clean up the PHIs.
      FuncletEntry->removePredecessor(UBB);
      TerminatorInst *Terminator = UBB->getTerminator();
      RemapInstruction(Terminator, VMap, RF_IgnoreMissingEntries);
    }
  }

  // This asserts a condition that is relied upon inside the loop below,
  // namely that no predecessors of the original funclet entry block
  // are also predecessors of the cloned funclet entry block.
  assert(std::all_of(pred_begin(FuncletEntry), pred_end(FuncletEntry),
                     [&ClonedFunclet](BasicBlock *Pred) {
                       return std::find(pred_begin(ClonedFunclet),
                                        pred_end(ClonedFunclet),
                                        Pred) == pred_end(ClonedFunclet);
                     }));

  // Remove any invalid PHI node entries in the cloned funclet.cl
  std::vector<PHINode *> PHIsToErase;
  for (Instruction &I : *ClonedFunclet) {
    auto *PN = dyn_cast<PHINode>(&I);
    if (!PN)
      break;

    // Predecessors of the original funclet do not reach the cloned funclet,
    // but the cloning process assumes they will.  Remove them now.
    for (auto *Pred : predecessors(FuncletEntry))
      PN->removeIncomingValue(Pred, false);
  }
  for (auto *PN : PHIsToErase)
    PN->eraseFromParent();

  // Replace the original funclet in the parent's children vector with the
  // cloned funclet.
  for (auto &It : FuncletChildren[Parent]) {
    if (It == FuncletEntry) {
      It = ClonedFunclet;
      break;
    }
  }

  return ClonedFunclet;
}

// Removes the unwind edge for any exceptional terminators within the specified
// parent funclet that previously unwound to the specified child funclet.
void WinEHPrepare::makeFuncletEdgeUnreachable(BasicBlock *Parent,
                                              BasicBlock *Child) {
  for (BasicBlock *BB : FuncletBlocks[Parent]) {
    TerminatorInst *Terminator = BB->getTerminator();
    if (!Terminator->isExceptional())
      continue;

    // Look for terninators that unwind to the child funclet.
    BasicBlock *UnwindDest = nullptr;
    if (auto *I = dyn_cast<InvokeInst>(Terminator))
      UnwindDest = I->getUnwindDest();
    else if (auto *I = dyn_cast<CatchEndPadInst>(Terminator))
      UnwindDest = I->getUnwindDest();
    else if (auto *I = dyn_cast<TerminatePadInst>(Terminator))
      UnwindDest = I->getUnwindDest();
    // cleanupendpad, catchret and cleanupret don't represent a parent-to-child
    // funclet transition, so we don't need to consider them here.

    // If the child funclet is the unwind destination, replace the terminator
    // with an unreachable instruction.
    if (UnwindDest == Child)
      removeUnwindEdge(BB);
  }
  // The specified parent is no longer a parent of the specified child.
  std::vector<BasicBlock *> &Children = FuncletChildren[Parent];
  Children.erase(std::remove(Children.begin(), Children.end(), Child),
                 Children.end());
}

// This routine is called after funclets with multiple parents are cloned for
// a specific parent.  Here we look for children of the specified funclet that
// unwind to other children of that funclet and update the unwind destinations
// to ensure that each sibling is connected to the correct clone of the sibling
// to which it unwinds.
//
// If the terminator is an invoke instruction, it unwinds either to a child
// EH pad, a cleanup end pad in the current funclet, or a catch end pad in a
// parent funclet (which ends either the current catch pad or a sibling
// catch pad).  If it unwinds to a child EH pad, the child will have multiple
// parents after this funclet is cloned and this case will be handled later in
// the resolveFuncletAncestryForPath processing.  If it unwinds to a
// cleanup end pad in the current funclet, the instruction remapping during
// the cloning process should have already mapped the unwind destination to
// the cloned copy of the cleanup end pad.  If it unwinds to a catch end pad
// there are two possibilities: either the catch end pad is the unwind
// destination for the catch pad we are currently cloning or it is the unwind
// destination for a sibling catch pad.  If it it the unwind destination of the
// catch pad we are cloning, we need to update the cloned invoke instruction
// to unwind to the cloned catch end pad.  Otherwise, we will handle this
// later (in resolveFuncletAncestryForPath).
static void updateSiblingToSiblingUnwind(
    BasicBlock *CurFunclet,
    std::map<BasicBlock *, SetVector<BasicBlock *>> &BlockColors,
    std::map<BasicBlock *, std::set<BasicBlock *>> &FuncletBlocks,
    std::map<BasicBlock *, std::vector<BasicBlock *>> &FuncletParents,
    std::map<BasicBlock *, std::vector<BasicBlock *>> &FuncletChildren,
    std::map<BasicBlock *, BasicBlock *> &Funclet2Orig) {
  // Remap any bad sibling-to-sibling transitions for funclets that
  // we just cloned.
  for (BasicBlock *ChildFunclet : FuncletChildren[CurFunclet]) {
    for (auto *BB : FuncletBlocks[ChildFunclet]) {
      TerminatorInst *Terminator = BB->getTerminator();
      if (!Terminator->isExceptional())
        continue;

      // See if this terminator has an unwind destination.
      // Note that catchendpads are handled when the associated catchpad
      // is cloned.  They don't fit the pattern we're looking for here.
      BasicBlock *UnwindDest = nullptr;
      if (auto *I = dyn_cast<CatchPadInst>(Terminator)) {
        UnwindDest = I->getUnwindDest();
        // The catchendpad is not a sibling destination.  This case should
        // have been handled in cloneFuncletForParent().
        if (isa<CatchEndPadInst>(Terminator)) {
          assert(BlockColors[UnwindDest].size() == 1 &&
                 "Cloned catchpad unwinds to an pad with multiple parents.");
          assert(FuncletParents[UnwindDest].front() == CurFunclet &&
                 "Cloned catchpad unwinds to the wrong parent.");
          continue;
        }
      } else {
        if (auto *I = dyn_cast<CleanupReturnInst>(Terminator))
          UnwindDest = I->getUnwindDest();
        else if (auto *I = dyn_cast<CleanupEndPadInst>(Terminator))
          UnwindDest = I->getUnwindDest();

        // If the cleanup unwinds to caller, there is nothing to be done.
        if (!UnwindDest)
          continue;
      }

      // If the destination is not a cleanup pad, catch pad or terminate pad
      // we don't need to handle it here.
      Instruction *EHPad = UnwindDest->getFirstNonPHI();
      if (!isa<CleanupPadInst>(EHPad) && !isa<CatchPadInst>(EHPad) &&
          !isa<TerminatePadInst>(EHPad))
        continue;

      // If it is one of these, then it is either a sibling of the current
      // child funclet or a clone of one of those siblings.
      // If it is a sibling, no action is needed.
      if (FuncletParents[UnwindDest].size() == 1 &&
          FuncletParents[UnwindDest].front() == CurFunclet)
        continue;

      // If the unwind destination is a clone of a sibling, we need to figure
      // out which sibling it is a clone of and use that sibling as the
      // unwind destination.
      BasicBlock *DestOrig = Funclet2Orig[UnwindDest];
      BasicBlock *TargetSibling = nullptr;
      for (auto &Mapping : Funclet2Orig) {
        if (Mapping.second != DestOrig)
          continue;
        BasicBlock *MappedFunclet = Mapping.first;
        if (FuncletParents[MappedFunclet].size() == 1 &&
            FuncletParents[MappedFunclet].front() == CurFunclet) {
          TargetSibling = MappedFunclet;
        }
      }
      // If we didn't find the sibling we were looking for then the
      // unwind destination is not a clone of one of child's siblings.
      // That's unexpected.
      assert(TargetSibling && "Funclet unwinds to unexpected destination.");

      // Update the terminator instruction to unwind to the correct sibling.
      if (auto *I = dyn_cast<CatchPadInst>(Terminator))
        I->setUnwindDest(TargetSibling);
      else if (auto *I = dyn_cast<CleanupReturnInst>(Terminator))
        I->setUnwindDest(TargetSibling);
      else if (auto *I = dyn_cast<CleanupEndPadInst>(Terminator))
        I->setUnwindDest(TargetSibling);
    }
  }
  
  // Invoke remapping can't be done correctly until after all of their
  // other sibling-to-sibling unwinds have been remapped.
  for (BasicBlock *ChildFunclet : FuncletChildren[CurFunclet]) {
    bool NeedOrigInvokeRemapping = false;
    for (auto *BB : FuncletBlocks[ChildFunclet]) {
      TerminatorInst *Terminator = BB->getTerminator();
      auto *II = dyn_cast<InvokeInst>(Terminator);
      if (!II)
        continue;

      BasicBlock *UnwindDest = II->getUnwindDest();
      assert(UnwindDest && "Invoke unwinds to a null destination.");
      assert(UnwindDest->isEHPad() && "Invoke does not unwind to an EH pad.");
      auto *EHPadInst = UnwindDest->getFirstNonPHI();
      if (isa<CleanupEndPadInst>(EHPadInst)) {
        // An invoke that unwinds to a cleanup end pad must be in a cleanup pad.
        assert(isa<CleanupPadInst>(ChildFunclet->getFirstNonPHI()) &&
               "Unwinding to cleanup end pad from a non cleanup pad funclet.");
        // The funclet cloning should have remapped the destination to the cloned
        // cleanup end pad.
        assert(FuncletBlocks[ChildFunclet].count(UnwindDest) &&
               "Unwind destination for invoke was not updated during cloning.");
      } else if (isa<CatchEndPadInst>(EHPadInst)) {
        // If the invoke unwind destination is the unwind destination for
        // the current child catch pad funclet, there is nothing to be done.
        BasicBlock *OrigFunclet = Funclet2Orig[ChildFunclet];
        auto *CurCatch = cast<CatchPadInst>(ChildFunclet->getFirstNonPHI());
        auto *OrigCatch = cast<CatchPadInst>(OrigFunclet->getFirstNonPHI());
        if (OrigCatch->getUnwindDest() == UnwindDest) {
          // If the invoke unwinds to a catch end pad that is the unwind
          // destination for the original catch pad, the cloned invoke should
          // unwind to the cloned catch end pad.
          II->setUnwindDest(CurCatch->getUnwindDest());
        } else if (CurCatch->getUnwindDest() == UnwindDest) {
          // If the invoke unwinds to a catch end pad that is the unwind
          // destination for the clone catch pad, the original invoke should
          // unwind to the unwind destination of the original catch pad.
          // This happens when the catch end pad is matched to the clone
          // parent when the catchpad instruction is cloned and the original
          // invoke instruction unwinds to the original catch end pad (which
          // is now the unwind destination of the cloned catch pad).
          NeedOrigInvokeRemapping = true;
        } else {
          // Otherwise, the invoke unwinds to a catch end pad that is the unwind
          // destination another catch pad in the unwind chain from either the
          // current catch pad or one of its clones.  If it is already the
          // catch end pad at the end unwind chain from the current catch pad,
          // we'll need to check the invoke instructions in the original funclet
          // later.  Otherwise, we need to remap this invoke now.
          assert((getEndPadForCatch(OrigCatch) == UnwindDest ||
                  getEndPadForCatch(CurCatch) == UnwindDest) &&
                "Invoke within catch pad unwinds to an invalid catch end pad.");
          BasicBlock *CurCatchEnd = getEndPadForCatch(CurCatch);
          if (CurCatchEnd == UnwindDest)
            NeedOrigInvokeRemapping = true;
          else
            II->setUnwindDest(CurCatchEnd);
        }
      }
    }
    if (NeedOrigInvokeRemapping) {
      // To properly remap invoke instructions that unwind to catch end pads
      // that are not the unwind destination of the catch pad funclet in which
      // the invoke appears, we must also look at the uncloned invoke in the
      // original funclet.  If we saw an invoke that was already properly
      // unwinding to a sibling's catch end pad, we need to check the invokes
      // in the original funclet.
      BasicBlock *OrigFunclet = Funclet2Orig[ChildFunclet];
      for (auto *BB : FuncletBlocks[OrigFunclet]) {
        auto *II = dyn_cast<InvokeInst>(BB->getTerminator());
        if (!II)
          continue;

        BasicBlock *UnwindDest = II->getUnwindDest();
        assert(UnwindDest && "Invoke unwinds to a null destination.");
        assert(UnwindDest->isEHPad() && "Invoke does not unwind to an EH pad.");
        auto *CEP = dyn_cast<CatchEndPadInst>(UnwindDest->getFirstNonPHI());
        if (!CEP)
          continue;

        // If the invoke unwind destination is the unwind destination for
        // the original catch pad funclet, there is nothing to be done.
        auto *OrigCatch = cast<CatchPadInst>(OrigFunclet->getFirstNonPHI());

        // If the invoke unwinds to a catch end pad that is neither the unwind
        // destination of OrigCatch or the destination another catch pad in the
        // unwind chain from current catch pad, we need to remap the invoke.
        BasicBlock *OrigCatchEnd = getEndPadForCatch(OrigCatch);
        if (OrigCatchEnd != UnwindDest)
          II->setUnwindDest(OrigCatchEnd);
      }
    }
  }
}

void WinEHPrepare::resolveFuncletAncestry(
    Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  // Most of the time this will be unnecessary.  If the conditions arise that
  // require this work, this flag will be set.
  if (!FuncletCloningRequired)
    return;
  
  // Funclet2Orig is used to map any cloned funclets back to the original
  // funclet from which they were cloned.  The map is seeded with the
  // original funclets mapping to themselves.
  std::map<BasicBlock *, BasicBlock *> Funclet2Orig;
  for (auto *Funclet : EntryBlocks)
    Funclet2Orig[Funclet] = Funclet;

  // Start with the entry funclet and walk the funclet parent-child tree.
  SmallVector<BasicBlock *, 4> FuncletPath;
  FuncletPath.push_back(&(F.getEntryBlock()));
  resolveFuncletAncestryForPath(F, FuncletPath, Funclet2Orig);
}

// Walks the funclet control flow, cloning any funclets that have more than one
// parent funclet and breaking any cyclic unwind chains so that the path becomes
// unreachable at the point where a funclet would have unwound to a funclet that
// was already in the chain.
void WinEHPrepare::resolveFuncletAncestryForPath(
    Function &F, SmallVectorImpl<BasicBlock *> &FuncletPath,
    std::map<BasicBlock *, BasicBlock *> &Funclet2Orig) {
  bool ClonedAnyChildren = false;
  BasicBlock *CurFunclet = FuncletPath.back();
  // Copy the children vector because we might changing it.
  std::vector<BasicBlock *> Children(FuncletChildren[CurFunclet]);
  for (BasicBlock *ChildFunclet : Children) {
    // Don't allow the funclet chain to unwind back on itself.
    // If this funclet is already in the current funclet chain, make the
    // path to it through the current funclet unreachable.
    bool IsCyclic = false;
    BasicBlock *ChildIdentity = Funclet2Orig[ChildFunclet];
    for (BasicBlock *Ancestor : FuncletPath) {
      BasicBlock *AncestorIdentity = Funclet2Orig[Ancestor];
      if (AncestorIdentity == ChildIdentity) {
        IsCyclic = true;
        break;
      }
    }
    // If the unwind chain wraps back on itself, break the chain.
    if (IsCyclic) {
      makeFuncletEdgeUnreachable(CurFunclet, ChildFunclet);
      continue;
    }
    // If this child funclet has other parents, clone the entire funclet.
    if (FuncletParents[ChildFunclet].size() > 1) {
      ChildFunclet = cloneFuncletForParent(F, ChildFunclet, CurFunclet);
      Funclet2Orig[ChildFunclet] = ChildIdentity;
      ClonedAnyChildren = true;
    }
    FuncletPath.push_back(ChildFunclet);
    resolveFuncletAncestryForPath(F, FuncletPath, Funclet2Orig);
    FuncletPath.pop_back();
  }
  // If we didn't clone any children, we can return now.
  if (!ClonedAnyChildren)
    return;

  updateSiblingToSiblingUnwind(CurFunclet, BlockColors, FuncletBlocks,
                               FuncletParents, FuncletChildren, Funclet2Orig);
}

void WinEHPrepare::colorFunclets(Function &F,
                                 SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  ::colorFunclets(F, EntryBlocks, BlockColors, FuncletBlocks);

  // The processing above actually accumulated the parent set for this
  // funclet into the color set for its entry; use the parent set to
  // populate the children map, and reset the color set to include just
  // the funclet itself (no instruction can target a funclet entry except on
  // that transitions to the child funclet).
  for (BasicBlock *FuncletEntry : EntryBlocks) {
    SetVector<BasicBlock *> &ColorMapItem = BlockColors[FuncletEntry];
    // It will be rare for funclets to have multiple parents, but if any
    // do we need to clone the funclet later to address that.  Here we
    // set a flag indicating that this case has arisen so that we don't
    // have to do a lot of checking later to handle the more common case.
    if (ColorMapItem.size() > 1)
      FuncletCloningRequired = true;
    for (BasicBlock *Parent : ColorMapItem) {
      assert(std::find(FuncletChildren[Parent].begin(),
                       FuncletChildren[Parent].end(),
                       FuncletEntry) == std::end(FuncletChildren[Parent]));
      FuncletChildren[Parent].push_back(FuncletEntry);
      assert(std::find(FuncletParents[FuncletEntry].begin(),
                       FuncletParents[FuncletEntry].end(),
                       Parent) == std::end(FuncletParents[FuncletEntry]));
      FuncletParents[FuncletEntry].push_back(Parent);
    }
    ColorMapItem.clear();
    ColorMapItem.insert(FuncletEntry);
  }
}

void llvm::calculateCatchReturnSuccessorColors(const Function *Fn,
                                               WinEHFuncInfo &FuncInfo) {
  SmallVector<BasicBlock *, 4> EntryBlocks;
  // colorFunclets needs the set of EntryBlocks, get them using
  // findFuncletEntryPoints.
  findFuncletEntryPoints(const_cast<Function &>(*Fn), EntryBlocks);

  std::map<BasicBlock *, SetVector<BasicBlock *>> BlockColors;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletBlocks;
  // Figure out which basic blocks belong to which funclets.
  colorFunclets(const_cast<Function &>(*Fn), EntryBlocks, BlockColors,
                FuncletBlocks);

  // The static colorFunclets routine assigns multiple colors to funclet entries
  // because that information is needed to calculate funclets' parent-child
  // relationship, but we don't need those relationship here and ultimately the
  // entry blocks should have the color of the funclet they begin.
  for (BasicBlock *FuncletEntry : EntryBlocks) {
    BlockColors[FuncletEntry].clear();
    BlockColors[FuncletEntry].insert(FuncletEntry);
  }

  // We need to find the catchret successors.  To do this, we must first find
  // all the catchpad funclets.
  for (auto &Funclet : FuncletBlocks) {
    // Figure out what kind of funclet we are looking at; We only care about
    // catchpads.
    BasicBlock *FuncletPadBB = Funclet.first;
    Instruction *FirstNonPHI = FuncletPadBB->getFirstNonPHI();
    auto *CatchPad = dyn_cast<CatchPadInst>(FirstNonPHI);
    if (!CatchPad)
      continue;

    // The users of a catchpad are always catchrets.
    for (User *Exit : CatchPad->users()) {
      auto *CatchReturn = dyn_cast<CatchReturnInst>(Exit);
      if (!CatchReturn)
        continue;
      BasicBlock *CatchRetSuccessor = CatchReturn->getSuccessor();
      SetVector<BasicBlock *> &SuccessorColors = BlockColors[CatchRetSuccessor];
      assert(SuccessorColors.size() == 1 && "Expected BB to be monochrome!");
      BasicBlock *Color = *SuccessorColors.begin();
      // Record the catchret successor's funclet membership.
      FuncInfo.CatchRetSuccessorColorMap[CatchReturn] = Color;
    }
  }
}

void WinEHPrepare::demotePHIsOnFunclets(Function &F) {
  // Strip PHI nodes off of EH pads.
  SmallVector<PHINode *, 16> PHINodes;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = &*FI++;
    if (!BB->isEHPad())
      continue;
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
      Instruction *I = &*BI++;
      auto *PN = dyn_cast<PHINode>(I);
      // Stop at the first non-PHI.
      if (!PN)
        break;

      AllocaInst *SpillSlot = insertPHILoads(PN, F);
      if (SpillSlot)
        insertPHIStores(PN, SpillSlot);

      PHINodes.push_back(PN);
    }
  }

  for (auto *PN : PHINodes) {
    // There may be lingering uses on other EH PHIs being removed
    PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
    PN->eraseFromParent();
  }
}

void WinEHPrepare::demoteUsesBetweenFunclets(Function &F) {
  // Turn all inter-funclet uses of a Value into loads and stores.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = &*FI++;
    SetVector<BasicBlock *> &ColorsForBB = BlockColors[BB];
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
      Instruction *I = &*BI++;
      // Funclets are permitted to use static allocas.
      if (auto *AI = dyn_cast<AllocaInst>(I))
        if (AI->isStaticAlloca())
          continue;

      demoteNonlocalUses(I, ColorsForBB, F);
    }
  }
}

void WinEHPrepare::demoteArgumentUses(Function &F) {
  // Also demote function parameters used in funclets.
  SetVector<BasicBlock *> &ColorsForEntry = BlockColors[&F.getEntryBlock()];
  for (Argument &Arg : F.args())
    demoteNonlocalUses(&Arg, ColorsForEntry, F);
}

void WinEHPrepare::cloneCommonBlocks(
    Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  // We need to clone all blocks which belong to multiple funclets.  Values are
  // remapped throughout the funclet to propogate both the new instructions
  // *and* the new basic blocks themselves.
  for (BasicBlock *FuncletPadBB : EntryBlocks) {
    std::set<BasicBlock *> &BlocksInFunclet = FuncletBlocks[FuncletPadBB];

    std::map<BasicBlock *, BasicBlock *> Orig2Clone;
    ValueToValueMapTy VMap;
    for (auto BlockIt = BlocksInFunclet.begin(),
              BlockEnd = BlocksInFunclet.end();
         BlockIt != BlockEnd;) {
      // Increment the iterator inside the loop because we might be removing
      // blocks from the set.
      BasicBlock *BB = *BlockIt++;
      SetVector<BasicBlock *> &ColorsForBB = BlockColors[BB];
      // We don't need to do anything if the block is monochromatic.
      size_t NumColorsForBB = ColorsForBB.size();
      if (NumColorsForBB == 1)
        continue;

      // If this block is a catchendpad, it shouldn't be cloned.
      // We will only see a catchendpad with multiple colors in the case where
      // some funclet has multiple parents.  In that case, the color will be
      // resolved during the resolveFuncletAncestry processing.
      // For now, find the catchpad that unwinds to this block and assign
      // that catchpad's first parent to be the color for this block.
      if (isa<CatchEndPadInst>(BB->getFirstNonPHI())) {
        assert(
            FuncletCloningRequired &&
            "Found multi-colored catchendpad with no multi-parent funclets.");
        BasicBlock *CatchParent = nullptr;
        // There can only be one catchpad predecessor for a catchendpad.
        for (BasicBlock *PredBB : predecessors(BB)) {
          if (isa<CatchPadInst>(PredBB->getTerminator())) {
            CatchParent = PredBB;
            break;
          }
        }
        // There must be one catchpad predecessor for a catchendpad.
        assert(CatchParent && "No catchpad found for catchendpad.");

        // If the catchpad has multiple parents, we'll clone the catchendpad
        // when we clone the catchpad funclet and insert it into the correct
        // funclet.  For now, we just select the first parent of the catchpad
        // and give the catchendpad that color.
        BasicBlock *CorrectColor = FuncletParents[CatchParent].front();
        assert(FuncletBlocks[CorrectColor].count(BB));
        assert(BlockColors[BB].count(CorrectColor));

        // Remove this block from the FuncletBlocks set of any funclet that
        // isn't the funclet whose color we just selected.
        for (auto It = BlockColors[BB].begin(), End = BlockColors[BB].end();
             It != End; ) {
          // The iterator must be incremented here because we are removing
          // elements from the set we're walking.
          auto Temp = It++;
          BasicBlock *ContainingFunclet = *Temp;
          if (ContainingFunclet != CorrectColor) {
            FuncletBlocks[ContainingFunclet].erase(BB);
            BlockColors[BB].remove(ContainingFunclet);
          }
        }

        // This should leave just one color for BB.
        assert(BlockColors[BB].size() == 1);
        continue;
      }

      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Cloning block \'" << BB->getName()
                              << "\' for funclet \'" << FuncletPadBB->getName()
                              << "\'.\n");

      // Create a new basic block and copy instructions into it!
      BasicBlock *CBB =
          CloneBasicBlock(BB, VMap, Twine(".for.", FuncletPadBB->getName()));
      // Insert the clone immediately after the original to ensure determinism
      // and to keep the same relative ordering of any funclet's blocks.
      CBB->insertInto(&F, BB->getNextNode());

      // Add basic block mapping.
      VMap[BB] = CBB;

      // Record delta operations that we need to perform to our color mappings.
      Orig2Clone[BB] = CBB;
    }

    // If nothing was cloned, we're done cloning in this funclet.
    if (Orig2Clone.empty())
      continue;

    // Update our color mappings to reflect that one block has lost a color and
    // another has gained a color.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;

      BlocksInFunclet.insert(NewBlock);
      BlockColors[NewBlock].insert(FuncletPadBB);

      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Assigned color \'" << FuncletPadBB->getName()
                              << "\' to block \'" << NewBlock->getName()
                              << "\'.\n");

      BlocksInFunclet.erase(OldBlock);
      BlockColors[OldBlock].remove(FuncletPadBB);

      DEBUG_WITH_TYPE("winehprepare-coloring",
                      dbgs() << "  Removed color \'" << FuncletPadBB->getName()
                              << "\' from block \'" << OldBlock->getName()
                              << "\'.\n");

      // If we are cloning a funclet that might share a child funclet with
      // another funclet, look to see if the cloned block is reached from a
      // catchret instruction.  If so, save this association so we can retrieve
      // the possibly orphaned clone when we clone the child funclet.
      if (FuncletCloningRequired) {
        for (auto *Pred : predecessors(OldBlock)) {
          auto *Terminator = Pred->getTerminator();
          if (!isa<CatchReturnInst>(Terminator))
            continue;
          // If this block is reached from a catchret instruction in a funclet
          // that has multiple parents, it will have a color for each of those
          // parents.  We just removed the color of one of the parents, but
          // the cloned block will be unreachable until we clone the child
          // funclet that contains the catchret instruction.  In that case we
          // need to create a mapping that will let us find the cloned block
          // later and associate it with the cloned child funclet.
          bool BlockWillBeEstranged = false;
          for (auto *Color : BlockColors[Pred]) {
            if (FuncletParents[Color].size() > 1) {
              BlockWillBeEstranged = true;
              break; // Breaks out of the color loop
            }
          }
          if (BlockWillBeEstranged) {
            EstrangedBlocks[FuncletPadBB][OldBlock] = NewBlock;
            DEBUG_WITH_TYPE("winehprepare-coloring",
                            dbgs() << "  Saved mapping of estranged block \'"
                                  << NewBlock->getName() << "\' for \'"
                                  << FuncletPadBB->getName() << "\'.\n");
            break; // Breaks out of the predecessor loop
          }
        }
      }
    }

    // Loop over all of the instructions in this funclet, fixing up operand
    // references as we go.  This uses VMap to do all the hard work.
    for (BasicBlock *BB : BlocksInFunclet)
      // Loop over all instructions, fixing each one as we find it...
      for (Instruction &I : *BB)
        RemapInstruction(&I, VMap,
                         RF_IgnoreMissingEntries | RF_NoModuleLevelChanges);

    // Check to see if SuccBB has PHI nodes. If so, we need to add entries to
    // the PHI nodes for NewBB now.
    for (auto &BBMapping : Orig2Clone) {
      BasicBlock *OldBlock = BBMapping.first;
      BasicBlock *NewBlock = BBMapping.second;
      for (BasicBlock *SuccBB : successors(NewBlock)) {
        for (Instruction &SuccI : *SuccBB) {
          auto *SuccPN = dyn_cast<PHINode>(&SuccI);
          if (!SuccPN)
            break;

          // Ok, we have a PHI node.  Figure out what the incoming value was for
          // the OldBlock.
          int OldBlockIdx = SuccPN->getBasicBlockIndex(OldBlock);
          if (OldBlockIdx == -1)
            break;
          Value *IV = SuccPN->getIncomingValue(OldBlockIdx);

          // Remap the value if necessary.
          if (auto *Inst = dyn_cast<Instruction>(IV)) {
            ValueToValueMapTy::iterator I = VMap.find(Inst);
            if (I != VMap.end())
              IV = I->second;
          }

          SuccPN->addIncoming(IV, NewBlock);
        }
      }
    }

    for (ValueToValueMapTy::value_type VT : VMap) {
      // If there were values defined in BB that are used outside the funclet,
      // then we now have to update all uses of the value to use either the
      // original value, the cloned value, or some PHI derived value.  This can
      // require arbitrary PHI insertion, of which we are prepared to do, clean
      // these up now.
      SmallVector<Use *, 16> UsesToRename;

      auto *OldI = dyn_cast<Instruction>(const_cast<Value *>(VT.first));
      if (!OldI)
        continue;
      auto *NewI = cast<Instruction>(VT.second);
      // Scan all uses of this instruction to see if it is used outside of its
      // funclet, and if so, record them in UsesToRename.
      for (Use &U : OldI->uses()) {
        Instruction *UserI = cast<Instruction>(U.getUser());
        BasicBlock *UserBB = UserI->getParent();
        SetVector<BasicBlock *> &ColorsForUserBB = BlockColors[UserBB];
        assert(!ColorsForUserBB.empty());
        if (ColorsForUserBB.size() > 1 ||
            *ColorsForUserBB.begin() != FuncletPadBB)
          UsesToRename.push_back(&U);
      }

      // If there are no uses outside the block, we're done with this
      // instruction.
      if (UsesToRename.empty())
        continue;

      // We found a use of OldI outside of the funclet.  Rename all uses of OldI
      // that are outside its funclet to be uses of the appropriate PHI node
      // etc.
      SSAUpdater SSAUpdate;
      SSAUpdate.Initialize(OldI->getType(), OldI->getName());
      SSAUpdate.AddAvailableValue(OldI->getParent(), OldI);
      SSAUpdate.AddAvailableValue(NewI->getParent(), NewI);

      while (!UsesToRename.empty())
        SSAUpdate.RewriteUseAfterInsertions(*UsesToRename.pop_back_val());
    }
  }
}

void WinEHPrepare::removeImplausibleTerminators(Function &F) {
  // Remove implausible terminators and replace them with UnreachableInst.
  for (auto &Funclet : FuncletBlocks) {
    BasicBlock *FuncletPadBB = Funclet.first;
    std::set<BasicBlock *> &BlocksInFunclet = Funclet.second;
    Instruction *FirstNonPHI = FuncletPadBB->getFirstNonPHI();
    auto *CatchPad = dyn_cast<CatchPadInst>(FirstNonPHI);
    auto *CleanupPad = dyn_cast<CleanupPadInst>(FirstNonPHI);

    for (BasicBlock *BB : BlocksInFunclet) {
      TerminatorInst *TI = BB->getTerminator();
      // CatchPadInst and CleanupPadInst can't transfer control to a ReturnInst.
      bool IsUnreachableRet = isa<ReturnInst>(TI) && (CatchPad || CleanupPad);
      // The token consumed by a CatchReturnInst must match the funclet token.
      bool IsUnreachableCatchret = false;
      if (auto *CRI = dyn_cast<CatchReturnInst>(TI))
        IsUnreachableCatchret = CRI->getCatchPad() != CatchPad;
      // The token consumed by a CleanupReturnInst must match the funclet token.
      bool IsUnreachableCleanupret = false;
      if (auto *CRI = dyn_cast<CleanupReturnInst>(TI))
        IsUnreachableCleanupret = CRI->getCleanupPad() != CleanupPad;
      // The token consumed by a CleanupEndPadInst must match the funclet token.
      bool IsUnreachableCleanupendpad = false;
      if (auto *CEPI = dyn_cast<CleanupEndPadInst>(TI))
        IsUnreachableCleanupendpad = CEPI->getCleanupPad() != CleanupPad;
      if (IsUnreachableRet || IsUnreachableCatchret ||
          IsUnreachableCleanupret || IsUnreachableCleanupendpad) {
        // Loop through all of our successors and make sure they know that one
        // of their predecessors is going away.
        for (BasicBlock *SuccBB : TI->successors())
          SuccBB->removePredecessor(BB);

        if (IsUnreachableCleanupendpad) {
          // We can't simply replace a cleanupendpad with unreachable, because
          // its predecessor edges are EH edges and unreachable is not an EH
          // pad.  Change all predecessors to the "unwind to caller" form.
          for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB);
               PI != PE;) {
            BasicBlock *Pred = *PI++;
            removeUnwindEdge(Pred);
          }
        }

        new UnreachableInst(BB->getContext(), TI);
        TI->eraseFromParent();
      }
      // FIXME: Check for invokes/cleanuprets/cleanupendpads which unwind to
      // implausible catchendpads (i.e. catchendpad not in immediate parent
      // funclet).
    }
  }
}

void WinEHPrepare::cleanupPreparedFunclets(Function &F) {
  // Clean-up some of the mess we made by removing useles PHI nodes, trivial
  // branches, etc.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = &*FI++;
    SimplifyInstructionsInBlock(BB);
    ConstantFoldTerminator(BB, /*DeleteDeadConditions=*/true);
    MergeBlockIntoPredecessor(BB);
  }

  // We might have some unreachable blocks after cleaning up some impossible
  // control flow.
  removeUnreachableBlocks(F);
}

void WinEHPrepare::verifyPreparedFunclets(Function &F) {
  // Recolor the CFG to verify that all is well.
  for (BasicBlock &BB : F) {
    size_t NumColors = BlockColors[&BB].size();
    assert(NumColors == 1 && "Expected monochromatic BB!");
    if (NumColors == 0)
      report_fatal_error("Uncolored BB!");
    if (NumColors > 1)
      report_fatal_error("Multicolor BB!");
    if (!DisableDemotion) {
      bool EHPadHasPHI = BB.isEHPad() && isa<PHINode>(BB.begin());
      assert(!EHPadHasPHI && "EH Pad still has a PHI!");
      if (EHPadHasPHI)
        report_fatal_error("EH Pad still has a PHI!");
    }
  }
}

bool WinEHPrepare::prepareExplicitEH(
    Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  replaceTerminatePadWithCleanup(F);

  // Determine which blocks are reachable from which funclet entries.
  colorFunclets(F, EntryBlocks);

  if (!DisableDemotion) {
    demotePHIsOnFunclets(F);

    demoteUsesBetweenFunclets(F);

    demoteArgumentUses(F);
  }

  cloneCommonBlocks(F, EntryBlocks);

  resolveFuncletAncestry(F, EntryBlocks);

  if (!DisableCleanups) {
    removeImplausibleTerminators(F);

    cleanupPreparedFunclets(F);
  }

  verifyPreparedFunclets(F);

  BlockColors.clear();
  FuncletBlocks.clear();
  FuncletChildren.clear();
  FuncletParents.clear();
  EstrangedBlocks.clear();
  FuncletCloningRequired = false;

  return true;
}

// TODO: Share loads when one use dominates another, or when a catchpad exit
// dominates uses (needs dominators).
AllocaInst *WinEHPrepare::insertPHILoads(PHINode *PN, Function &F) {
  BasicBlock *PHIBlock = PN->getParent();
  AllocaInst *SpillSlot = nullptr;

  if (isa<CleanupPadInst>(PHIBlock->getFirstNonPHI())) {
    // Insert a load in place of the PHI and replace all uses.
    SpillSlot = new AllocaInst(PN->getType(), nullptr,
                               Twine(PN->getName(), ".wineh.spillslot"),
                               &F.getEntryBlock().front());
    Value *V = new LoadInst(SpillSlot, Twine(PN->getName(), ".wineh.reload"),
                            &*PHIBlock->getFirstInsertionPt());
    PN->replaceAllUsesWith(V);
    return SpillSlot;
  }

  DenseMap<BasicBlock *, Value *> Loads;
  for (Value::use_iterator UI = PN->use_begin(), UE = PN->use_end();
       UI != UE;) {
    Use &U = *UI++;
    auto *UsingInst = cast<Instruction>(U.getUser());
    BasicBlock *UsingBB = UsingInst->getParent();
    if (UsingBB->isEHPad()) {
      // Use is on an EH pad phi.  Leave it alone; we'll insert loads and
      // stores for it separately.
      assert(isa<PHINode>(UsingInst));
      continue;
    }
    replaceUseWithLoad(PN, U, SpillSlot, Loads, F);
  }
  return SpillSlot;
}

// TODO: improve store placement.  Inserting at def is probably good, but need
// to be careful not to introduce interfering stores (needs liveness analysis).
// TODO: identify related phi nodes that can share spill slots, and share them
// (also needs liveness).
void WinEHPrepare::insertPHIStores(PHINode *OriginalPHI,
                                   AllocaInst *SpillSlot) {
  // Use a worklist of (Block, Value) pairs -- the given Value needs to be
  // stored to the spill slot by the end of the given Block.
  SmallVector<std::pair<BasicBlock *, Value *>, 4> Worklist;

  Worklist.push_back({OriginalPHI->getParent(), OriginalPHI});

  while (!Worklist.empty()) {
    BasicBlock *EHBlock;
    Value *InVal;
    std::tie(EHBlock, InVal) = Worklist.pop_back_val();

    PHINode *PN = dyn_cast<PHINode>(InVal);
    if (PN && PN->getParent() == EHBlock) {
      // The value is defined by another PHI we need to remove, with no room to
      // insert a store after the PHI, so each predecessor needs to store its
      // incoming value.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
        Value *PredVal = PN->getIncomingValue(i);

        // Undef can safely be skipped.
        if (isa<UndefValue>(PredVal))
          continue;

        insertPHIStore(PN->getIncomingBlock(i), PredVal, SpillSlot, Worklist);
      }
    } else {
      // We need to store InVal, which dominates EHBlock, but can't put a store
      // in EHBlock, so need to put stores in each predecessor.
      for (BasicBlock *PredBlock : predecessors(EHBlock)) {
        insertPHIStore(PredBlock, InVal, SpillSlot, Worklist);
      }
    }
  }
}

void WinEHPrepare::insertPHIStore(
    BasicBlock *PredBlock, Value *PredVal, AllocaInst *SpillSlot,
    SmallVectorImpl<std::pair<BasicBlock *, Value *>> &Worklist) {

  if (PredBlock->isEHPad() &&
      !isa<CleanupPadInst>(PredBlock->getFirstNonPHI())) {
    // Pred is unsplittable, so we need to queue it on the worklist.
    Worklist.push_back({PredBlock, PredVal});
    return;
  }

  // Otherwise, insert the store at the end of the basic block.
  new StoreInst(PredVal, SpillSlot, PredBlock->getTerminator());
}

// The SetVector == operator uses the std::vector == operator, so it doesn't
// actually tell us whether or not the two sets contain the same colors. This
// function does that.
// FIXME: Would it be better to add a isSetEquivalent() method to SetVector?
static bool isBlockColorSetEquivalent(SetVector<BasicBlock *> &SetA,
                                      SetVector<BasicBlock *> &SetB) {
  if (SetA.size() != SetB.size())
    return false;
  for (auto *Color : SetA)
    if (!SetB.count(Color))
      return false;
  return true;
}

// TODO: Share loads for same-funclet uses (requires dominators if funclets
// aren't properly nested).
void WinEHPrepare::demoteNonlocalUses(Value *V,
                                      SetVector<BasicBlock *> &ColorsForBB,
                                      Function &F) {
  // Tokens can only be used non-locally due to control flow involving
  // unreachable edges.  Don't try to demote the token usage, we'll simply
  // delete the cloned user later.
  if (isa<CatchPadInst>(V) || isa<CleanupPadInst>(V))
    return;

  DenseMap<BasicBlock *, Value *> Loads;
  AllocaInst *SpillSlot = nullptr;
  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end(); UI != UE;) {
    Use &U = *UI++;
    auto *UsingInst = cast<Instruction>(U.getUser());
    BasicBlock *UsingBB = UsingInst->getParent();

    // Is the Use inside a block which is colored the same as the Def?
    // If so, we don't need to escape the Def because we will clone
    // ourselves our own private copy.
    SetVector<BasicBlock *> &ColorsForUsingBB = BlockColors[UsingBB];
    if (isBlockColorSetEquivalent(ColorsForUsingBB, ColorsForBB))
      continue;

    replaceUseWithLoad(V, U, SpillSlot, Loads, F);
  }
  if (SpillSlot) {
    // Insert stores of the computed value into the stack slot.
    // We have to be careful if I is an invoke instruction,
    // because we can't insert the store AFTER the terminator instruction.
    BasicBlock::iterator InsertPt;
    if (isa<Argument>(V)) {
      InsertPt = F.getEntryBlock().getTerminator()->getIterator();
    } else if (isa<TerminatorInst>(V)) {
      auto *II = cast<InvokeInst>(V);
      // We cannot demote invoke instructions to the stack if their normal
      // edge is critical. Therefore, split the critical edge and create a
      // basic block into which the store can be inserted.
      if (!II->getNormalDest()->getSinglePredecessor()) {
        unsigned SuccNum =
            GetSuccessorNumber(II->getParent(), II->getNormalDest());
        assert(isCriticalEdge(II, SuccNum) && "Expected a critical edge!");
        BasicBlock *NewBlock = SplitCriticalEdge(II, SuccNum);
        assert(NewBlock && "Unable to split critical edge.");
        // Update the color mapping for the newly split edge.
        SetVector<BasicBlock *> &ColorsForUsingBB = BlockColors[II->getParent()];
        BlockColors[NewBlock] = ColorsForUsingBB;
        for (BasicBlock *FuncletPad : ColorsForUsingBB)
          FuncletBlocks[FuncletPad].insert(NewBlock);
      }
      InsertPt = II->getNormalDest()->getFirstInsertionPt();
    } else {
      InsertPt = cast<Instruction>(V)->getIterator();
      ++InsertPt;
      // Don't insert before PHI nodes or EH pad instrs.
      for (; isa<PHINode>(InsertPt) || InsertPt->isEHPad(); ++InsertPt)
        ;
    }
    new StoreInst(V, SpillSlot, &*InsertPt);
  }
}

void WinEHPrepare::replaceUseWithLoad(Value *V, Use &U, AllocaInst *&SpillSlot,
                                      DenseMap<BasicBlock *, Value *> &Loads,
                                      Function &F) {
  // Lazilly create the spill slot.
  if (!SpillSlot)
    SpillSlot = new AllocaInst(V->getType(), nullptr,
                               Twine(V->getName(), ".wineh.spillslot"),
                               &F.getEntryBlock().front());

  auto *UsingInst = cast<Instruction>(U.getUser());
  if (auto *UsingPHI = dyn_cast<PHINode>(UsingInst)) {
    // If this is a PHI node, we can't insert a load of the value before
    // the use.  Instead insert the load in the predecessor block
    // corresponding to the incoming value.
    //
    // Note that if there are multiple edges from a basic block to this
    // PHI node that we cannot have multiple loads.  The problem is that
    // the resulting PHI node will have multiple values (from each load)
    // coming in from the same block, which is illegal SSA form.
    // For this reason, we keep track of and reuse loads we insert.
    BasicBlock *IncomingBlock = UsingPHI->getIncomingBlock(U);
    if (auto *CatchRet =
            dyn_cast<CatchReturnInst>(IncomingBlock->getTerminator())) {
      // Putting a load above a catchret and use on the phi would still leave
      // a cross-funclet def/use.  We need to split the edge, change the
      // catchret to target the new block, and put the load there.
      BasicBlock *PHIBlock = UsingInst->getParent();
      BasicBlock *NewBlock = SplitEdge(IncomingBlock, PHIBlock);
      // SplitEdge gives us:
      //   IncomingBlock:
      //     ...
      //     br label %NewBlock
      //   NewBlock:
      //     catchret label %PHIBlock
      // But we need:
      //   IncomingBlock:
      //     ...
      //     catchret label %NewBlock
      //   NewBlock:
      //     br label %PHIBlock
      // So move the terminators to each others' blocks and swap their
      // successors.
      BranchInst *Goto = cast<BranchInst>(IncomingBlock->getTerminator());
      Goto->removeFromParent();
      CatchRet->removeFromParent();
      IncomingBlock->getInstList().push_back(CatchRet);
      NewBlock->getInstList().push_back(Goto);
      Goto->setSuccessor(0, PHIBlock);
      CatchRet->setSuccessor(NewBlock);
      // Update the color mapping for the newly split edge.
      SetVector<BasicBlock *> &ColorsForPHIBlock = BlockColors[PHIBlock];
      BlockColors[NewBlock] = ColorsForPHIBlock;
      for (BasicBlock *FuncletPad : ColorsForPHIBlock)
        FuncletBlocks[FuncletPad].insert(NewBlock);
      // Treat the new block as incoming for load insertion.
      IncomingBlock = NewBlock;
    }
    Value *&Load = Loads[IncomingBlock];
    // Insert the load into the predecessor block
    if (!Load)
      Load = new LoadInst(SpillSlot, Twine(V->getName(), ".wineh.reload"),
                          /*Volatile=*/false, IncomingBlock->getTerminator());

    U.set(Load);
  } else {
    // Reload right before the old use.
    auto *Load = new LoadInst(SpillSlot, Twine(V->getName(), ".wineh.reload"),
                              /*Volatile=*/false, UsingInst);
    U.set(Load);
  }
}

void WinEHFuncInfo::addIPToStateRange(const BasicBlock *PadBB,
                                      MCSymbol *InvokeBegin,
                                      MCSymbol *InvokeEnd) {
  assert(PadBB->isEHPad() && EHPadStateMap.count(PadBB->getFirstNonPHI()) &&
         "should get EH pad BB with precomputed state");
  InvokeToStateMap[InvokeBegin] =
      std::make_pair(EHPadStateMap[PadBB->getFirstNonPHI()], InvokeEnd);
}
