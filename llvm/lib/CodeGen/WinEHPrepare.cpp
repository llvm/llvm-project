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
  void demoteNonlocalUses(Value *V, std::set<BasicBlock *> &ColorsForBB,
                          Function &F);
  bool prepareExplicitEH(Function &F,
                         SmallVectorImpl<BasicBlock *> &EntryBlocks);
  void replaceTerminatePadWithCleanup(Function &F);
  void colorFunclets(Function &F, SmallVectorImpl<BasicBlock *> &EntryBlocks);
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

  std::map<BasicBlock *, std::set<BasicBlock *>> BlockColors;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletBlocks;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletChildren;
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
              std::map<BasicBlock *, std::set<BasicBlock *>> &BlockColors,
              std::map<BasicBlock *, std::set<BasicBlock *>> &FuncletBlocks,
              std::map<BasicBlock *, std::set<BasicBlock *>> &FuncletChildren) {
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

  Worklist.push_back({EntryBlock, EntryBlock});

  while (!Worklist.empty()) {
    BasicBlock *Visiting;
    BasicBlock *Color;
    std::tie(Visiting, Color) = Worklist.pop_back_val();
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
              if (BlockColors[Succ].insert(Color).second)
                Worklist.push_back({Succ, Color});
        }
      }
      // Handle CatchPad specially since its successors need different colors.
      if (CatchPadInst *CatchPad = dyn_cast<CatchPadInst>(VisitingHead)) {
        // Visit the normal successor with the color of the new EH pad, and
        // visit the unwind successor with the color of the parent.
        BasicBlock *NormalSucc = CatchPad->getNormalDest();
        if (BlockColors[NormalSucc].insert(Visiting).second) {
          Worklist.push_back({NormalSucc, Visiting});
        }
        BasicBlock *UnwindSucc = CatchPad->getUnwindDest();
        if (BlockColors[UnwindSucc].insert(Color).second) {
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
      if (BlockColors[Succ].insert(Color).second) {
        Worklist.push_back({Succ, Color});
      }
    }
  }

  // The processing above actually accumulated the parent set for this
  // funclet into the color set for its entry; use the parent set to
  // populate the children map, and reset the color set to include just
  // the funclet itself (no instruction can target a funclet entry except on
  // that transitions to the child funclet).
  for (BasicBlock *FuncletEntry : EntryBlocks) {
    std::set<BasicBlock *> &ColorMapItem = BlockColors[FuncletEntry];
    for (BasicBlock *Parent : ColorMapItem)
      FuncletChildren[Parent].insert(FuncletEntry);
    ColorMapItem.clear();
    ColorMapItem.insert(FuncletEntry);
  }
}

void WinEHPrepare::colorFunclets(Function &F,
                                 SmallVectorImpl<BasicBlock *> &EntryBlocks) {
  ::colorFunclets(F, EntryBlocks, BlockColors, FuncletBlocks, FuncletChildren);
}

void llvm::calculateCatchReturnSuccessorColors(const Function *Fn,
                                               WinEHFuncInfo &FuncInfo) {
  SmallVector<BasicBlock *, 4> EntryBlocks;
  // colorFunclets needs the set of EntryBlocks, get them using
  // findFuncletEntryPoints.
  findFuncletEntryPoints(const_cast<Function &>(*Fn), EntryBlocks);

  std::map<BasicBlock *, std::set<BasicBlock *>> BlockColors;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletBlocks;
  std::map<BasicBlock *, std::set<BasicBlock *>> FuncletChildren;
  // Figure out which basic blocks belong to which funclets.
  colorFunclets(const_cast<Function &>(*Fn), EntryBlocks, BlockColors,
                FuncletBlocks, FuncletChildren);

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
      std::set<BasicBlock *> &SuccessorColors = BlockColors[CatchRetSuccessor];
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
    std::set<BasicBlock *> &ColorsForBB = BlockColors[BB];
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
  std::set<BasicBlock *> &ColorsForEntry = BlockColors[&F.getEntryBlock()];
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
    for (BasicBlock *BB : BlocksInFunclet) {
      std::set<BasicBlock *> &ColorsForBB = BlockColors[BB];
      // We don't need to do anything if the block is monochromatic.
      size_t NumColorsForBB = ColorsForBB.size();
      if (NumColorsForBB == 1)
        continue;

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

      BlocksInFunclet.erase(OldBlock);
      BlockColors[OldBlock].erase(FuncletPadBB);
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
        std::set<BasicBlock *> &ColorsForUserBB = BlockColors[UserBB];
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

  if (!DisableCleanups) {
    removeImplausibleTerminators(F);

    cleanupPreparedFunclets(F);
  }

  verifyPreparedFunclets(F);

  BlockColors.clear();
  FuncletBlocks.clear();
  FuncletChildren.clear();

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

// TODO: Share loads for same-funclet uses (requires dominators if funclets
// aren't properly nested).
void WinEHPrepare::demoteNonlocalUses(Value *V,
                                      std::set<BasicBlock *> &ColorsForBB,
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
    std::set<BasicBlock *> &ColorsForUsingBB = BlockColors[UsingBB];
    if (ColorsForUsingBB == ColorsForBB)
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
        std::set<BasicBlock *> &ColorsForUsingBB = BlockColors[II->getParent()];
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
      std::set<BasicBlock *> &ColorsForPHIBlock = BlockColors[PHIBlock];
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
