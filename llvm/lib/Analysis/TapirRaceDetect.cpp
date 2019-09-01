//===-- lib/Analysis/TapirRaceDetect.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TapirRaceDetect is an LLVM pass that analyses Tapir tasks and dependences
// between memory accesses to find accesses that might race.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TapirRaceDetect.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "tapir-race-detect"

// Statistics
STATISTIC(AssumeSyncBeforeReturn, "Number of calls where the existence of a "
          "race depends on the sync-before-return property.");

static cl::opt<bool>
    AssumeSafeMalloc(
        "assume-safe-malloc", cl::init(true), cl::Hidden,
        cl::desc("Assume that calls to allocation functions are safe."));

static cl::opt<bool>
    IgnoreTerminationCalls(
        "ignore-termination-calls", cl::init(true), cl::Hidden,
        cl::desc("Ignore calls in program-terminating exit blocks."));

static cl::opt<unsigned>
    MaxUsesToExploreCapture(
        "max-uses-to-explore-capture", cl::init(20), cl::Hidden,
        cl::desc("Maximum number of uses to explore for a capture query."));

// Boilerplate for legacy and new pass managers

TapirRaceDetect::Result
TapirRaceDetect::run(Function &F, FunctionAnalysisManager &FAM) {
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &TI = FAM.getResult<TaskAnalysis>(F);
  auto &DI = FAM.getResult<DependenceAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto *TLI = &FAM.getResult<TargetLibraryAnalysis>(F);
  return RaceInfo(&F, DT, LI, TI, DI, SE, TLI);
}

AnalysisKey TapirRaceDetect::Key;

INITIALIZE_PASS_BEGIN(TapirRaceDetectWrapperPass, "tapir-race-detect",
                      "Tapir Race Detection", true, true)
INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TaskInfoWrapperPass)
INITIALIZE_PASS_END(TapirRaceDetectWrapperPass, "tapir-race-detect",
                    "Tapir Race Detection", true, true)

char TapirRaceDetectWrapperPass::ID = 0;

bool TapirRaceDetectWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &TI = getAnalysis<TaskInfoWrapperPass>().getTaskInfo();
  auto &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto *TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  Info.reset(new RaceInfo(&F, DT, LI, TI, DI, SE, TLI));
  return false;
}

RaceInfo &TapirRaceDetectWrapperPass::getRaceInfo() const { return *Info; }

void TapirRaceDetectWrapperPass::releaseMemory() { Info.reset(); }

void TapirRaceDetectWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DependenceAnalysisWrapperPass>();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<LoopInfoWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequiredTransitive<TaskInfoWrapperPass>();
}

FunctionPass *llvm::createTapirRaceDetectWrapperPass() {
  return new TapirRaceDetectWrapperPass();
}

void TapirRaceDetectWrapperPass::print(raw_ostream &OS,
                                       const Module *) const {
  Info->print(OS);
}

PreservedAnalyses
TapirRaceDetectPrinterPass::run(Function &F, FunctionAnalysisManager &FAM) {
  OS << "'Tapir race detection' for function '" << F.getName() << "':\n";
  FAM.getResult<TapirRaceDetect>(F).print(OS);
  return PreservedAnalyses::all();
}

namespace {

// Structure to record the set of child tasks that might be in parallel with
// this spindle, ignoring back edges of loops.
//
// TODO: Improve this analysis to track the loop back edges responsible for
// specific maybe-parallel tasks.  Use these back-edge tags to refine the
// dependence-analysis component of static race detection.  Possible test case:
// intel/BlackScholes.
struct MaybeParallelTasksInLoopBody : public MaybeParallelTasks {
  MPTaskListTy TaskList;
  LoopInfo &LI;

  MaybeParallelTasksInLoopBody(LoopInfo &LI) : LI(LI) {}

  // This method performs the data-flow update computation on a given spindle.
  bool evaluate(const Spindle *S, unsigned EvalNum) {
    LLVM_DEBUG(dbgs() << "MPTInLoop::evaluate @ " << S->getEntry()->getName()
               << "\n");
    if (!TaskList.count(S))
      TaskList.try_emplace(S);

    bool Complete = true;
    for (const Spindle::SpindleEdge &PredEdge : S->in_edges()) {
      const Spindle *Pred = PredEdge.first;
      const BasicBlock *Inc = PredEdge.second;

      // If the incoming edge is a sync edge, get the associated sync region.
      const Value *SyncRegSynced = nullptr;
      if (const SyncInst *SI = dyn_cast<SyncInst>(Inc->getTerminator()))
        SyncRegSynced = SI->getSyncRegion();

      // Skip back edges for this task list.
      if (Loop *L = LI.getLoopFor(S->getEntry()))
        if ((L->getHeader() == S->getEntry()) && L->contains(Inc))
          continue;

      // Iterate through the tasks in the task list for Pred.
      for (const Task *MP : TaskList[Pred]) {
        // Filter out any tasks that are synced by the sync region.
        if (const DetachInst *DI = MP->getDetach())
          if (SyncRegSynced == DI->getSyncRegion())
            continue;
        // Insert the task into this spindle's task list.  If this task is a new
        // addition, then we haven't yet reached the fixed point of this
        // analysis.
        if (TaskList[S].insert(MP).second)
          Complete = false;
      }
    }
    LLVM_DEBUG({
        dbgs() << "  New MPT list for " << S->getEntry()->getName()
               << (Complete ? " (complete)\n" : " (not complete)\n");
        for (const Task *MP : TaskList[S])
          dbgs() << "    " << MP->getEntry()->getName() << "\n";
      });
    return Complete;
  }
};

class AccessPtrAnalysis {
public:
  /// Read or write access location.
  // using MemAccessInfo = PointerIntPair<const Value *, 1, bool>;
  using MemAccessInfo = RaceInfo::MemAccessInfo;
  // using MemAccessInfoList = SmallVector<MemAccessInfo, 8>;
  // using AccessToUnderlyingObjMap =
  //   DenseMap<MemAccessInfo, SmallPtrSet<Value *, 1>>;
  using AccessToUnderlyingObjMap = RaceInfo::AccessToUnderlyingObjMap;

  AccessPtrAnalysis(const DataLayout &DL, DominatorTree &DT,
                    TaskInfo &TI, LoopInfo &LI, DependenceInfo &DI,
                    ScalarEvolution &SE, const TargetLibraryInfo *TLI,
                    AccessToUnderlyingObjMap &AccessToObjs)
      : DL(DL), DT(DT), TI(TI), LI(LI), DI(DI), AA(DI.getAA()), SE(SE),
        TLI(TLI), AccessToObjs(AccessToObjs), MPTasksInLoop(LI) {
    TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);
  }

  void addFunctionArgument(Value *Arg);
  void addAccess(Instruction *I);

  void processAccessPtrs(RaceInfo::ResultTy &Result,
                         RaceInfo::ObjectMRTy &ObjectMRForRace,
                         RaceInfo::PtrChecksTy &AllPtrRtChecks);

private:
  using PtrAccessSet = SetVector<MemAccessInfo>;

  void checkForRacesHelper(const Task *T,
                           // SmallVectorImpl<const Task *> &AncestorMPTasks,
                           RaceInfo::ResultTy &Result,
                           RaceInfo::ObjectMRTy &ObjectMRForRace);
  bool checkOpaqueAccesses(GeneralAccess &GA1, GeneralAccess &GA2);
  void evaluateMaybeParallelAccesses(GeneralAccess &GA1, GeneralAccess &GA2,
                                     RaceInfo::ResultTy &Result,
                                     RaceInfo::ObjectMRTy &ObjectMRForRace);
  bool checkDependence(std::unique_ptr<Dependence> D, GeneralAccess &GA1,
                       GeneralAccess &GA2);
  void getRTPtrChecks(Loop *L, RaceInfo::ResultTy &Result,
                      RaceInfo::PtrChecksTy &AllPtrRtChecks);

  bool PointerCapturedBefore(const Value *Ptr, const Instruction *I,
                             unsigned MaxUsesToExplore) const;

  void recordLocalRace(const GeneralAccess &GA, RaceInfo::ResultTy &Result,
                       RaceInfo::ObjectMRTy &ObjectMRForRace);
  const DataLayout &DL;
  DominatorTree &DT;
  TaskInfo &TI;
  LoopInfo &LI;
  DependenceInfo &DI;
  AliasAnalysis *AA;
  ScalarEvolution &SE;

  const TargetLibraryInfo *TLI;
  // // An alias set tracker to partition the access set by underlying object and
  // // intrinsic property (such as TBAA metadata).
  // AliasSetTracker AST;
  SmallPtrSet<Value *, 4> ArgumentPtrs;
  AccessToUnderlyingObjMap &AccessToObjs;

  MaybeParallelTasks MPTasks;
  MaybeParallelTasksInLoopBody MPTasksInLoop;

  // A mapping of tasks to instructions in that task that might participate in a
  // determinacy race.
  using TaskAccessMapTy = DenseMap<const Task *, SmallVector<GeneralAccess, 4>>;
  TaskAccessMapTy TaskAccessMap;

  // A mapping of spindles to instructions in that spindle that might
  // participate in a determinacy race.
  using SpindleAccessMapTy =
    DenseMap<const Spindle *, SmallVector<GeneralAccess, 4>>;
  SpindleAccessMapTy SpindleAccessMap;

  // A mapping of loops to instructions in that loop that might
  // participate in a determinacy race.
  using LoopAccessMapTy = DenseMap<const Loop *, SmallVector<GeneralAccess, 4>>;
  LoopAccessMapTy LoopAccessMap;

  mutable DenseMap<std::pair<const Value *, const Instruction *>, bool>
  MayBeCapturedCache;

  // /// We need to check that all of the pointers in this list are disjoint
  // /// at runtime. Using std::unique_ptr to make using move ctor simpler.
  // DenseMap<const Loop *, RuntimePointerChecking *> AllPtrRtChecking;
};

} // end anonymous namespace

static bool checkInstructionForRace(const Instruction *I,
                                    const TargetLibraryInfo *TLI) {
  if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<VAArgInst>(I) ||
      isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
      isa<AnyMemSetInst>(I) || isa<AnyMemTransferInst>(I))
    return true;

  if (isa<CallBase>(I)) {
    // Ignore debug info intrinsics
    if (isa<DbgInfoIntrinsic>(I))
      return false;

    // Ignore other intrinsics.
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
      // TODO: Exclude all intrinsics for which
      // TTI::getIntrinsicCost() == TCC_Free?
      switch (II->getIntrinsicID()) {
      default: return true;
      case Intrinsic::annotation:
      case Intrinsic::assume:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::launder_invariant_group:
      case Intrinsic::strip_invariant_group:
      case Intrinsic::is_constant:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::objectsize:
      case Intrinsic::ptr_annotation:
      case Intrinsic::var_annotation:
      case Intrinsic::syncregion_start:
        return false;
      }

    // We can assume allocation functions are safe.
    if (AssumeSafeMalloc && isAllocationFn(I, TLI))
      return false;

    // If this call occurs in a termination block of the program, ignore it.
    if (IgnoreTerminationCalls &&
        isa<UnreachableInst>(I->getParent()->getTerminator())) {
      const Function *CF = cast<CallBase>(I)->getCalledFunction();
      // If this is an ordinary function call in a terminating block, ignore it.
      if (!CF->hasFnAttribute(Attribute::NoReturn))
        return false;
      // If this is a call to a terminating function, such as "exit" or "abort",
      // ignore it.
      if (CF->hasName() &&
          ((CF->getName() == "exit") || (CF->getName() == "abort") ||
           (CF->getName() == "__clang_call_terminate") ||
           (CF->getName() == "__assert_fail")))
        return false;
    }

    // We want to instrument calls in general.
    return true;
  }
  return false;
}

// Get the general memory accesses for the instruction \p I, and stores those
// accesses into \p AccI.  Returns true if general memory accesses could be
// derived for I, false otherwise.
static void GetGeneralAccesses(
    Instruction *I, SmallVectorImpl<GeneralAccess> &AccI, AliasAnalysis *AA,
    const TargetLibraryInfo *TLI) {
  // Handle common memory instructions
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    MemoryLocation Loc = MemoryLocation::get(LI);
    if (!AA->pointsToConstantMemory(Loc))
      AccI.push_back(GeneralAccess(LI, Loc, ModRefInfo::Ref));
    return;
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    AccI.push_back(GeneralAccess(SI, MemoryLocation::get(SI), ModRefInfo::Mod));
    return;
  }
  // Handle atomic instructions
  if (AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I)) {
    AccI.push_back(GeneralAccess(CXI, MemoryLocation::get(CXI),
                                 ModRefInfo::Mod));
    return;
  }
  if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    AccI.push_back(GeneralAccess(RMWI, MemoryLocation::get(RMWI),
                                 ModRefInfo::Mod));
    return;
  }

  // Handle VAArgs.
  if (VAArgInst *VAAI = dyn_cast<VAArgInst>(I)) {
    MemoryLocation Loc = MemoryLocation::get(VAAI);
    if (!AA->pointsToConstantMemory(Loc))
      AccI.push_back(GeneralAccess(VAAI, Loc, ModRefInfo::ModRef));
    return;
  }

  // Handle memory intrinsics.
  if (AnyMemSetInst *MSI = dyn_cast<AnyMemSetInst>(I)) {
    AccI.push_back(GeneralAccess(MSI, MemoryLocation::getForDest(MSI),
                                 ModRefInfo::Mod));
    return;
  }
  if (AnyMemTransferInst *MTI = dyn_cast<AnyMemTransferInst>(I)) {
    AccI.push_back(GeneralAccess(MTI, MemoryLocation::getForDest(MTI),
                                 0, ModRefInfo::Mod));
    MemoryLocation Loc = MemoryLocation::getForSource(MTI);
    if (!AA->pointsToConstantMemory(Loc))
      AccI.push_back(GeneralAccess(MTI, Loc, 1, ModRefInfo::Ref));
    return;
  }

  // Handle arbitrary call sites by examining pointee arguments.
  //
  // This logic is based on that in AliasSetTracker.cpp.
  if (const CallBase *Call = dyn_cast<CallBase>(I)) {
    ModRefInfo CallMask = createModRefInfo(AA->getModRefBehavior(Call));

    // Some intrinsics are marked as modifying memory for control flow modelling
    // purposes, but don't actually modify any specific memory location.
    using namespace PatternMatch;
    if (Call->use_empty() &&
        match(Call, m_Intrinsic<Intrinsic::invariant_start>()))
      CallMask = clearMod(CallMask);
    // TODO: See if we need to exclude additional intrinsics.

    for (auto IdxArgPair : enumerate(Call->args())) {
      int ArgIdx = IdxArgPair.index();
      const Value *Arg = IdxArgPair.value();
      if (!Arg->getType()->isPointerTy())
        continue;
      MemoryLocation ArgLoc =
        MemoryLocation::getForArgument(Call, ArgIdx, TLI);
      if (AA->pointsToConstantMemory(ArgLoc))
        continue;
      ModRefInfo ArgMask = AA->getArgModRefInfo(Call, ArgIdx);
      ArgMask = intersectModRef(CallMask, ArgMask);
      if (!isNoModRef(ArgMask)) {
        // dbgs() << "New GA for " << *I << "\n  arg " << *Arg << "\n";
        // if (ArgLoc.Size != LocationSize::unknown())
        //   dbgs() << "  size " << ArgLoc.Size.getValue() << "\n";
        AccI.push_back(GeneralAccess(I, ArgLoc, ArgIdx, ArgMask));
      }
    }

    // If we find a free call and we assume malloc is safe, don't worry about
    // opaque accesses by that free call.
    if (AssumeSafeMalloc && isFreeCall(Call, TLI))
      return;

    if (!Call->onlyAccessesArgMemory())
      // Add a generic GeneralAccess for this call to represent the fact that it
      // might access arbitrary global memory.
      AccI.push_back(GeneralAccess(I, None, CallMask));
    return;
  }
}

void AccessPtrAnalysis::addFunctionArgument(Value *Arg) {
  ArgumentPtrs.insert(Arg);
}

void AccessPtrAnalysis::addAccess(Instruction *I) {
  // The AST can handle LoadInst, StoreInst, VAArgInst, AnyMemSetInst,
  // AnyMemTransferInst, and function calls.
  if (checkInstructionForRace(I, TLI)) {
    // AST.add(I);
    SmallVector<GeneralAccess, 1> GA;
    GetGeneralAccesses(I, GA, DI.getAA(), TLI);
    TaskAccessMap[TI.getTaskFor(I->getParent())].append(GA.begin(), GA.end());
    SpindleAccessMap[TI.getSpindleFor(I->getParent())].append(GA.begin(),
                                                              GA.end());
    if (Loop *L = LI.getLoopFor(I->getParent()))
      LoopAccessMap[L].append(GA.begin(), GA.end());

    for (GeneralAccess Acc : GA) {
      // Skip this access if it does not have a valid pointer.
      if (!Acc.getPtr())
        continue;

      MemAccessInfo Access(Acc.getPtr(), Acc.isMod());
      // DepCands.insert(Access);

      SmallVector<Value *, 1> Objects;
      GetUnderlyingObjects(const_cast<Value *>(Acc.getPtr()), Objects, DL, &LI,
                           0);
      for (Value *Obj : Objects) {
        // nullptr never alias, don't join sets for pointer that have "null" in
        // their UnderlyingObjects list.
        if (isa<ConstantPointerNull>(Obj) &&
            !NullPointerIsDefined(I->getFunction(),
                                  Obj->getType()->getPointerAddressSpace()))
          continue;

        // Is this value a constant that cannot be derived from any pointer
        // value (we need to exclude constant expressions, for example, that
        // are formed from arithmetic on global symbols).
        if (const Constant *C = dyn_cast<Constant>(Obj)) {
          // This check is derived from Transforms/Utils/InlineFunction.cpp
          bool IsNonPtrConst = isa<BlockAddress>(C) || isa<ConstantInt>(C) ||
            isa<ConstantFP>(C) || isa<ConstantPointerNull>(C) ||
            isa<ConstantDataSequential>(C) || isa<UndefValue>(C) ||
            isa<ConstantTokenNone>(C);
          if (IsNonPtrConst)
            continue;
        }

        if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Obj))
          // Constant and thread-local variables cannot race.
          if (GV->isConstant() || GV->isThreadLocal())
            continue;

        if (isa<Function>(Obj))
          // Assume that functions are read-only
          continue;

        AccessToObjs[Access].insert(Obj);

        // UnderlyingObjToAccessMap::iterator Prev = ObjToLastAccess.find(Obj);
        // if (Prev != ObjToLastAccess.end())
        //   DepCands.unionSets(Access, Prev->second);

        // ObjToLastAccess[Obj] = Access;
      }
    }
  }
}

static Loop *getCommonLoop(const BasicBlock *B1, const BasicBlock *B2,
                           LoopInfo &LI) {
  unsigned B1Level = LI.getLoopDepth(B1);
  unsigned B2Level = LI.getLoopDepth(B2);
  Loop *L1 = LI.getLoopFor(B1);
  Loop *L2 = LI.getLoopFor(B2);
  while (B1Level > B2Level) {
    L1 = L1->getParentLoop();
    B1Level--;
  }
  while (B2Level > B1Level) {
    L2 = L2->getParentLoop();
    B2Level--;
  }
  while (L1 != L2) {
    L1 = L1->getParentLoop();
    L2 = L2->getParentLoop();
  }
  return L1;
}

static const Spindle *GetRepSpindleInTask(const Spindle *S, const Task *T,
                                          const TaskInfo &TI) {
  const Task *Encl = T->getSubTaskEnclosing(S->getEntry());
  if (Encl->isRootTask())
    return S;
  return TI.getSpindleFor(Encl->getDetach()->getContinue());
}

bool AccessPtrAnalysis::checkDependence(std::unique_ptr<Dependence> D,
                                        GeneralAccess &GA1,
                                        GeneralAccess &GA2) {
  if (!D) {
    LLVM_DEBUG(dbgs() << "No dependence\n");
    return false;
  }

  LLVM_DEBUG({
      D->dump(dbgs());
      StringRef DepType =
        D->isFlow() ? "flow" : D->isAnti() ? "anti" : "output";
      dbgs() << "Found " << DepType
             << " dependency between Src and Dst\n";
      unsigned Levels = D->getLevels();
      for (unsigned II = 1; II <= Levels; ++II) {
        const SCEV *Distance = D->getDistance(II);
        if (Distance)
          dbgs() << "Level " << II << " distance " << *Distance << "\n";
      }
    });

  Instruction *I1 = GA1.I;
  Instruction *I2 = GA2.I;
  BasicBlock *B1 = I1->getParent();
  BasicBlock *B2 = I2->getParent();

  // Only dependencies that cross tasks can produce determinacy races.
  // Dependencies that cross loop iterations within the same task don't matter.
  // To search the relevant loops, start at the spindle entry that most closely
  // dominates both instructions, and check outwards, up to the topmost root for
  // which Dst is in a maybe-parallel task.  Dominating blocks within the
  // spindle are all guaranteed to execute in series with each other, so
  // dependencies between those instructions matter.

  // Use the base objects for the addresses to try to further refine the checks.

  // TODO: Use lifetime_begin intrinsics to further refine checks.
  unsigned MinObjDepth = static_cast<unsigned>(-1);
  SmallPtrSet<Value *, 1> BaseObjs;
  MemAccessInfo MA1(GA1.getPtr(), GA1.isMod());
  MemAccessInfo MA2(GA2.getPtr(), GA2.isMod());
  for (Value *Obj : AccessToObjs[MA1]) {
    if (AccessToObjs[MA2].count(Obj))
      BaseObjs.insert(Obj);
    else {
      MinObjDepth = 0;
      break;
    }
  }
  for (Value *Obj : AccessToObjs[MA2]) {
    if (AccessToObjs[MA1].count(Obj))
      BaseObjs.insert(Obj);
    else {
      MinObjDepth = 0;
      break;
    }
  }
  if (MinObjDepth != 0) {
    for (Value *Obj : BaseObjs) {
      LLVM_DEBUG(dbgs() << "Checking base object " << *Obj << "\n");
      assert(!(isa<ConstantPointerNull>(Obj) &&
               !NullPointerIsDefined(B1->getParent(),
                                     Obj->getType()->getPointerAddressSpace()))
             && "nullptr in list of base objects");

      if (!isa<Instruction>(Obj)) {
        MinObjDepth = 0;
        break;
      }
      unsigned ObjDepth = LI.getLoopDepth(cast<Instruction>(Obj)->getParent());
      if (ObjDepth < MinObjDepth)
        MinObjDepth = ObjDepth;
    }
  }
  LLVM_DEBUG(dbgs() << "Min loop depth " << MinObjDepth <<
             " for underlying object.\n");

  // Find the basic block that dominates both instructions.
  // BasicBlock *Dom = DT.findNearestCommonDominator(B1, B1);
  // Find the deepest loop that contains both I1 and I2.
  // Loop *CommonLoop = LI.getLoopFor(Dom);
  // unsigned MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;
  // while (MaxLoopDepthToCheck && (//!CommonLoop->contains(B1) ||
  //                                !CommonLoop->contains(B2))) {
  //   CommonLoop = CommonLoop->getParentLoop();
  //   MaxLoopDepthToCheck--;
  // }

  // Find the deepest loop that contains both B1 and B2.
  Loop *CommonLoop = getCommonLoop(B1, B2, LI);
  unsigned MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;

  // Check if dependence does not depend on looping.
  if (0 == MaxLoopDepthToCheck)
    // If there's no loop to worry about, then the existence of the dependence
    // implies the potential for a race.
    return true;

  LLVM_DEBUG({
      if (MinObjDepth > MaxLoopDepthToCheck) {
        dbgs() << "\tI1 " << *I1 << "\n\tI2 " << *I2;
        dbgs() << "\n\tPtr1 " << *GA1.getPtr()
               << " (null? " << (isa<ConstantPointerNull>(GA1.getPtr())) << ")";
        dbgs() << "\n\tPtr2 " << *GA2.getPtr()
               << " (null? " << (isa<ConstantPointerNull>(GA2.getPtr())) << ")";
        dbgs() << "\n\tAddrspace "
               << GA1.getPtr()->getType()->getPointerAddressSpace();
        dbgs() << "\n\tnullptr is defined? "
               << (NullPointerIsDefined(B1->getParent()));
        dbgs() << "\n\tMaxLoopDepthToCheck " << MaxLoopDepthToCheck;
        dbgs() << "\n\tMinObjDepthToCheck " << MinObjDepth << "\n";
      }
    });
  assert(MinObjDepth <= MaxLoopDepthToCheck &&
         "Minimum loop depth of underlying object cannot be greater "
         "than maximum loop depth of dependence.");

  // Get the task that encloses both B1 and B2.
  Task *CommonTask = TI.getEnclosingTask(B1, B2);
  // Get the representative spindles for both B1 and B2 in this common task.
  const Spindle *I1Spindle = GetRepSpindleInTask(TI.getSpindleFor(B1),
                                                 CommonTask, TI);
  const Spindle *I2Spindle = GetRepSpindleInTask(TI.getSpindleFor(B2),
                                                 CommonTask, TI);
  // If this common loop does not contain the common task, then dependencies at
  // the level of this common loop do not constitute a potential race.  Find the
  // loop that contains the enclosing task.
  //
  // Skip this step if either representative spindle is a shared-eh spindle,
  // because those are more complicated.
  if (!I1Spindle->isSharedEH() && !I2Spindle->isSharedEH()) {
    if (!CommonLoop->contains(CommonTask->getEntry())) {
      Loop *CommonTaskLoop = LI.getLoopFor(CommonTask->getEntry());
      assert(!CommonTaskLoop || CommonTaskLoop->contains(CommonLoop) &&
             "Loop for common task does not contain common loop.");
      CommonLoop = CommonTaskLoop;
    }
    // Update MaxLoopDepthToCheck
    MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;

    // Check if dependence does not depend on looping.
    if (0 == MaxLoopDepthToCheck)
      // // If there's no loop to worry about, then the existence of the dependence
      // // implies the potential for a race.
      // return true;
      MaxLoopDepthToCheck = MinObjDepth;
  }

  if (MaxLoopDepthToCheck == MinObjDepth) {
    LLVM_DEBUG(dbgs() << "Minimum object depth matches maximum loop depth.\n");
    if (TI.getTaskFor(B1) == TI.getTaskFor(B2))
      return false;

    // Check if dependence does not depend on looping.
    if (0 == MaxLoopDepthToCheck)
      // If there's no loop to worry about, then the existence of the dependence
      // implies the potential for a race.
      return true;

    if (!(D->getDirection(MaxLoopDepthToCheck) & Dependence::DVEntry::EQ))
      // Apparent dependence does not occur within the same iteration.
      return false;

    // Check if the instructions are parallel when the loop backedge is excluded
    // from dataflow.
    for (const Task *MPT : MPTasksInLoop.TaskList[I1Spindle])
      if (TI.encloses(MPT, B2))
        return true;
    for (const Task *MPT : MPTasksInLoop.TaskList[I2Spindle])
      if (TI.encloses(MPT, B1))
        return true;

    return false;
  }

  // Get the whole loop stack to check above the common loop.
  SmallVector<Loop *, 4> LoopsToCheck;
  Loop *CurrLoop = CommonLoop;
  while (CurrLoop) {
    LoopsToCheck.push_back(CurrLoop);
    CurrLoop = CurrLoop->getParentLoop();
  }

  // Check the loop stack from the top down until a loop is found where the
  // dependence might cross parallel tasks.
  unsigned MinLoopDepthToCheck = 1;
  while (!LoopsToCheck.empty()) {
    Loop *CurrLoop = LoopsToCheck.pop_back_val();
    // If we're not yet at the minimum loop depth of the underlying object, go
    // deeper.
    if (MinLoopDepthToCheck < MinObjDepth) {
      ++MinLoopDepthToCheck;
      continue;
    }

    // Check the maybe-parallel tasks for the spindle containing the loop
    // header.
    Spindle *CurrSpindle = TI.getSpindleFor(CurrLoop->getHeader());
    bool MPTEnclosesDst = false;
    for (const Task *MPT : MPTasks.TaskList[CurrSpindle]) {
      if (TI.encloses(MPT, B2)) {
        MPTEnclosesDst = true;
        break;
      }
    }

    // If Dst is found in a maybe-parallel task, then the minimum loop depth has
    // been found.
    if (MPTEnclosesDst)
      break;
    // Otherwise go deeper.
    ++MinLoopDepthToCheck;
  }

  // Scan the loop nests in common from inside out.
  for (unsigned II = MaxLoopDepthToCheck; II >= MinLoopDepthToCheck; --II) {
    LLVM_DEBUG(dbgs() << "Checking loop level " << II << "\n");
    if (D->isScalar(II))
      return true;
    if (D->getDirection(II) & unsigned(~Dependence::DVEntry::EQ))
      return true;
  }

  LLVM_DEBUG(dbgs() << "Dependence does not cross parallel tasks.\n");
  return false;
}

bool AccessPtrAnalysis::PointerCapturedBefore(const Value *Ptr,
                                              const Instruction *I,
                                              unsigned MaxUsesToExplore =
                                              MaxUsesToExploreCapture) const {
  auto CaptureQuery = std::make_pair(Ptr, I);
  if (MayBeCapturedCache.count(CaptureQuery))
    return MayBeCapturedCache[CaptureQuery];

  bool Result = false;
  if (isa<GlobalValue>(Ptr))
    // We assume that globals are captured.
    //
    // TODO: Possibly refine this check for private or internal globals.
    Result = true;
  else
    Result = PointerMayBeCapturedBefore(Ptr, false, false, I, &DT, true,
                                        nullptr, MaxUsesToExplore);
  MayBeCapturedCache[CaptureQuery] = Result;
  return Result;
}

bool AccessPtrAnalysis::checkOpaqueAccesses(GeneralAccess &GA1,
                                            GeneralAccess &GA2) {
  // If neither instruction may write to memory, then no race is possible.
  if (!GA1.I->mayWriteToMemory() && !GA2.I->mayWriteToMemory())
    return false;

  if (!GA1.Loc && !GA2.Loc) {
    const CallBase *Call1 = cast<CallBase>(GA1.I);
    const CallBase *Call2 = cast<CallBase>(GA2.I);

    assert(!AA->doesNotAccessMemory(Call1) && !AA->doesNotAccessMemory(Call2) &&
           "Opaque call does not access memory.");
    assert(!AA->onlyAccessesArgPointees(AA->getModRefBehavior(Call1)) &&
           !AA->onlyAccessesArgPointees(AA->getModRefBehavior(Call2)) &&
           "Opaque call only accesses arg pointees.");

    // // If both calls only read memory, then there's no dependence.
    // if (AA->onlyReadsMemory(Call1) && AA->onlyReadsMemory(Call2))
    //   return false;

    // We have two logically-parallel calls that opaquely access memory, and at
    // least one call modifies memory.  Hence we have a dependnece and potential
    // race.
    return true;
  }

  BasicBlock *B1 = GA1.I->getParent();
  BasicBlock *B2 = GA2.I->getParent();

  // Get information about the non-opaque access.
  const Value *Ptr;
  Instruction *NonOpaque;
  bool IsWrite;
  if (GA1.Loc) {
    Ptr = GA1.getPtr();
    NonOpaque = GA1.I;
    IsWrite = GA1.isMod();
  } else { // GA2.Loc
    Ptr = GA2.getPtr();
    NonOpaque = GA2.I;
    IsWrite = GA2.isMod();
  }

  // One access is opaque, while the other has a pointer.  For the opaque access
  // to race, the pointer must escape before the non-opaque instruction.
  if (!PointerCapturedBefore(Ptr, NonOpaque))
    return false;

  // TODO: Use the instruction that performs the capture to further bound the
  // subsequent loop checks.

  // Otherwise we check the logical parallelism of the access.  Because one of
  // the pointers is null, we assume that the "minimum object depth" is 0.
  unsigned MinObjDepth = 0;
  LLVM_DEBUG(dbgs() << "Min loop depth " << MinObjDepth <<
             " for underlying object.\n");

  // Find the deepest loop that contains both B1 and B2.
  Loop *CommonLoop = getCommonLoop(B1, B2, LI);
  unsigned MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;

  // Check if dependence does not depend on looping.
  if (0 == MaxLoopDepthToCheck)
    // If there's no loop to worry about, then the existence of the dependence
    // implies the potential for a race.
    return true;

  LLVM_DEBUG(
      if (MinObjDepth > MaxLoopDepthToCheck) {
        dbgs() << "\tI1 " << *GA1.I << "\n\tI2 " << *GA2.I;
        dbgs() << "\n\tMaxLoopDepthToCheck " << MaxLoopDepthToCheck;
        dbgs() << "\n\tMinObjDepthToCheck " << MinObjDepth << "\n";
        dbgs() << *GA1.I->getFunction();
      });
  assert(MinObjDepth <= MaxLoopDepthToCheck &&
         "Minimum loop depth of underlying object cannot be greater "
         "than maximum loop depth of dependence.");

  // Get the task that encloses both B1 and B2.
  Task *CommonTask = TI.getEnclosingTask(B1, B2);
  // Get the representative spindles for both B1 and B2 in this common task.
  const Spindle *I1Spindle = GetRepSpindleInTask(TI.getSpindleFor(B1),
                                                 CommonTask, TI);
  const Spindle *I2Spindle = GetRepSpindleInTask(TI.getSpindleFor(B2),
                                                 CommonTask, TI);
  // If this common loop does not contain the common task, then dependencies at
  // the level of this common loop do not constitute a potential race.  Find the
  // loop that contains the enclosing task.
  //
  // Skip this step if either representative spindle is a shared-eh spindle,
  // because those are more complicated.
  if (!I1Spindle->isSharedEH() && !I2Spindle->isSharedEH()) {
    if (!CommonLoop->contains(CommonTask->getEntry())) {
      Loop *CommonTaskLoop = LI.getLoopFor(CommonTask->getEntry());
      assert(!CommonTaskLoop || CommonTaskLoop->contains(CommonLoop) &&
             "Loop for common task does not contain common loop.");
      CommonLoop = CommonTaskLoop;
    }
    // Update MaxLoopDepthToCheck
    MaxLoopDepthToCheck = CommonLoop ? CommonLoop->getLoopDepth() : 0;

    // Check if dependence does not depend on looping.
    if (0 == MaxLoopDepthToCheck)
      // // If there's no loop to worry about, then the existence of the dependence
      // // implies the potential for a race.
      // return true;
      MaxLoopDepthToCheck = MinObjDepth;
  }

  if (MaxLoopDepthToCheck == MinObjDepth) {
    LLVM_DEBUG(dbgs() << "Minimum object depth matches maximum loop depth.\n");
    if (TI.getTaskFor(B1) == TI.getTaskFor(B2))
      return false;

    // Check if dependence does not depend on looping.
    if (0 == MaxLoopDepthToCheck)
      // If there's no loop to worry about, then the existence of the dependence
      // implies the potential for a race.
      return true;

    // Check if the instructions are parallel when the loop backedge is excluded
    // from dataflow.
    for (const Task *MPT : MPTasksInLoop.TaskList[I1Spindle])
      if (TI.encloses(MPT, B2))
        return true;
    for (const Task *MPT : MPTasksInLoop.TaskList[I2Spindle])
      if (TI.encloses(MPT, B1))
        return true;

    return false;
  }

  // Get the whole loop stack to check above the common loop.
  SmallVector<Loop *, 4> LoopsToCheck;
  Loop *CurrLoop = CommonLoop;
  while (CurrLoop) {
    LoopsToCheck.push_back(CurrLoop);
    CurrLoop = CurrLoop->getParentLoop();
  }

  // Check the loop stack from the top down until a loop is found where the
  // dependence might cross parallel tasks.
  unsigned MinLoopDepthToCheck = 1;
  while (!LoopsToCheck.empty()) {
    Loop *CurrLoop = LoopsToCheck.pop_back_val();
    // If we're not yet at the minimum loop depth of the underlying object, go
    // deeper.
    if (MinLoopDepthToCheck < MinObjDepth) {
      ++MinLoopDepthToCheck;
      continue;
    }

    // Check the maybe-parallel tasks for the spindle containing the loop
    // header.
    Spindle *CurrSpindle = TI.getSpindleFor(CurrLoop->getHeader());
    bool MPTEnclosesDst = false;
    for (const Task *MPT : MPTasks.TaskList[CurrSpindle]) {
      if (TI.encloses(MPT, B2)) {
        MPTEnclosesDst = true;
        break;
      }
    }

    // If Dst is found in a maybe-parallel task, then the minimum loop depth has
    // been found.
    if (MPTEnclosesDst)
      break;
    // Otherwise go deeper.
    ++MinLoopDepthToCheck;
  }

  // The opaque access acts like a dependence across all iterations of any loops
  // containing the accesses.
  return true;

  // if (const CallBase *Call1 = dyn_cast<CallBase>(GA1.I))
  //   if (const CallBase *Call2 = dyn_cast<CallBase>(GA2.I))
  //     return isModSet(AA->getModRefInfo(Call1, Call2));

  // assert((GA1.Loc || GA2.Loc) &&
  //        "Non-call general accesses lack memory locations.");

  // // TODO: Fix this.  When one instruction is an ordinary load or store, it may
  // // appear to opaquely access a GA with no pointer, when in fact its accessing
  // // the pointer argument in a non-racing manner.
  // if (GA1.Loc)
  //   return isModSet(AA->getModRefInfo(GA2.I, GA1.Loc));
  // return isModSet(AA->getModRefInfo(GA1.I, GA2.Loc));
}

static void setObjectMRForRace(RaceInfo::ObjectMRTy &ObjectMRForRace,
                               const Value *Ptr, ModRefInfo MRI) {
  if (!ObjectMRForRace.count(Ptr))
    ObjectMRForRace[Ptr] = ModRefInfo::NoModRef;
  ObjectMRForRace[Ptr] = unionModRef(ObjectMRForRace[Ptr], MRI);
}

void AccessPtrAnalysis::recordLocalRace(const GeneralAccess &GA,
                                        RaceInfo::ResultTy &Result,
                                        RaceInfo::ObjectMRTy &ObjectMRForRace) {
  Result.recordLocalRace(GA);

  if (!GA.getPtr())
    return;

  for (Value *Obj : AccessToObjs[MemAccessInfo(GA.getPtr(), GA.isMod())]) {
    if (GA.isMod())
      setObjectMRForRace(ObjectMRForRace, Obj, ModRefInfo::Ref);
    setObjectMRForRace(ObjectMRForRace, Obj, ModRefInfo::Mod);
  }
}

static void recordAncestorRace(const GeneralAccess &GA, const Value *Ptr,
                               RaceInfo::ResultTy &Result,
                               RaceInfo::ObjectMRTy &ObjectMRForRace) {
  if (GA.isMod()) {
    Result.recordRaceViaAncestorRef(GA);
    setObjectMRForRace(ObjectMRForRace, Ptr, ModRefInfo::Ref);
  }
  Result.recordRaceViaAncestorMod(GA);
  setObjectMRForRace(ObjectMRForRace, Ptr, ModRefInfo::Mod);
}

void AccessPtrAnalysis::evaluateMaybeParallelAccesses(
    GeneralAccess &GA1, GeneralAccess &GA2, RaceInfo::ResultTy &Result,
    RaceInfo::ObjectMRTy &ObjectMRForRace) {
  // No race is possible if no access modifies.
  if (!GA1.isMod() && !GA2.isMod())
    return;

  bool LocalRace = false;
  if (!GA1.getPtr() || !GA2.getPtr()) {
    LLVM_DEBUG({
        dbgs() << "Checking for race involving opaque access:\n"
               << "  GA1 =\n";
        if (GA1.getPtr())
          dbgs() << "    Ptr:" << *GA1.getPtr() << "\n";
        else
          dbgs() << "    Ptr: null\n";
        dbgs() << "    I:" << *GA1.I << "\n"
               << "  GA2 =\n";
        if (GA2.getPtr())
          dbgs() << "    Ptr:" << *GA2.getPtr() << "\n";
        else
          dbgs() << "    Ptr: null\n";
        dbgs() << "    I:" << *GA2.I << "\n";});
    if (checkOpaqueAccesses(GA1, GA2))
      LocalRace = true;
  } else {
    // If either GA has a nullptr, then skip the check, since nullptr's cannot
    // alias.
    Function *F = GA1.I->getFunction();
    if (isa<ConstantPointerNull>(GA1.getPtr()) &&
        !NullPointerIsDefined(
            F, GA1.getPtr()->getType()->getPointerAddressSpace()))
      return;
    if (isa<ConstantPointerNull>(GA2.getPtr()) &&
        !NullPointerIsDefined(
            F, GA2.getPtr()->getType()->getPointerAddressSpace()))
      return;

    LLVM_DEBUG(dbgs() << "Checking for race from dependence:\n"
               << "  GA1 =\n"
               << "    Ptr:" << *GA1.getPtr() << "\n    I:" << *GA1.I << "\n"
               << "  GA2 =\n"
               << "    Ptr:" << *GA2.getPtr() << "\n    I:" << *GA2.I << "\n");
    if (checkDependence(DI.depends(&GA1, &GA2, true), GA1, GA2))
      LocalRace = true;
  }

  if (LocalRace) {
    LLVM_DEBUG(dbgs() << "Local race found:\n"
               << "  I1 =" << *GA1.I << "\n  I2 =" << *GA2.I << "\n");
    recordLocalRace(GA1, Result, ObjectMRForRace);
    recordLocalRace(GA2, Result, ObjectMRForRace);
  }
}

void AccessPtrAnalysis::checkForRacesHelper(
    const Task *T, /*SmallVectorImpl<const Task *> &AncestorMPTasks,*/
    RaceInfo::ResultTy &Result, RaceInfo::ObjectMRTy &ObjectMRForRace) {
  // // AncestorMPTasks contains instructions that might race with anything in this
  // // task.
  // for (const Task *MPT : AncestorMPTasks) {
  //   LLVM_DEBUG(dbgs() << "Testing Task@" << T->getEntry()->getName()
  //              << " against Task@" << MPT->getEntry()->getName()
  //              << "\n");
  //   for (GeneralAccess GA1 : TaskAccessMap[T])
  //     for (const Task *SubMPT : depth_first(MPT))
  //       for (GeneralAccess GA2 : TaskAccessMap[SubMPT])
  //         evaluateMaybeParallelAccesses(GA1, GA2, Result);
  // }

  // size_t OrigSize = AncestorMPTasks.size();
  SmallPtrSet<const Spindle *, 4> Visited;

  // Now handle each spindle in this task.
  for (const Spindle *S :
         depth_first<InTask<Spindle *>>(T->getEntrySpindle())) {
    LLVM_DEBUG(dbgs() << "Testing Spindle@" << S->getEntry()->getName()
               << "\n");
    for (GeneralAccess GA1 : SpindleAccessMap[S]) {
      if (GA1.getPtr()) {
        LLVM_DEBUG({
            dbgs() << "GA1 Underlying objects:\n";
            for (Value *Obj : AccessToObjs[MemAccessInfo(GA1.getPtr(),
                                                         GA1.isMod())])
              dbgs() << "    " << *Obj << "\n";
          });
        for (Value *Obj : AccessToObjs[MemAccessInfo(GA1.getPtr(),
                                                     GA1.isMod())]) {
          if (isa<AllocaInst>(Obj))
            // Races on alloca'd objects are checked locally.
            continue;

          if (isAllocationFn(Obj, TLI) && AssumeSafeMalloc)
            // Races on malloc'd objects are checked locally.
            continue;

          if (const Argument *A = dyn_cast<Argument>(Obj)) {
            // Check if the attributes on the argument preclude a race with the
            // caller.
            if (A->hasByValAttr() || A->hasNoAliasAttr() ||
                A->hasStructRetAttr() || A->hasInAllocaAttr())
              continue;

            // Otherwise record the possible race with an ancestor.
            LLVM_DEBUG(dbgs() << "Setting race via ancestor:\n"
                       << "  GA.I: " << *GA1.I << "\n"
                       << "  Arg: " << *A << "\n");
            recordAncestorRace(GA1, A, Result, ObjectMRForRace);
            continue;
          }

          if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Obj)) {
            // Constant and thread-local variables cannot race.
            assert(!GV->isConstant() && !GV->isThreadLocal() &&
                   "Constant or thread-local GV should be excluded.");
            if (!GV->hasPrivateLinkage() && !GV->hasInternalLinkage()) {
              // Races are only possible with ancestor functions in this module.
              LLVM_DEBUG(dbgs() << "Setting race via private/internal global:\n"
                         << "  GA.I: " << *GA1.I << "\n"
                         << "  GV: " << *GV << "\n");
              // TODO: Add suppressions for private and internal global
              // variables, then record this as a potential ancestor race
              // instead of an opaque race
              //
              // recordAncestorRace(GA1, GV, Result, ObjectMRForRace);
              Result.recordOpaqueRace(GA1);
            } else {
              // Record the possible opaque race.
              LLVM_DEBUG(dbgs() << "Setting opaque race:\n"
                         << "  GA.I: " << *GA1.I << "\n"
                         << "  GV: " << *GV << "\n");
              Result.recordOpaqueRace(GA1);
            }
            continue;
          }

          if (isa<ConstantExpr>(Obj)) {
            // Record the possible opaque race.
            LLVM_DEBUG(dbgs() << "Setting opaque race:\n"
                       << "  GA.I: " << *GA1.I << "\n"
                       << "  Obj: " << *Obj << "\n");
            Result.recordOpaqueRace(GA1);
            continue;
          }

          if (!isa<Instruction>(Obj)) {
            dbgs() << "Unexpected underlying object: " << *Obj << "\n";
          }

          // Record the possible opaque race.
          LLVM_DEBUG(dbgs() << "Setting opaque race:\n"
                     << "  GA.I: " << *GA1.I << "\n"
                     << "  Obj: " << *Obj << "\n");
          Result.recordOpaqueRace(GA1);
        }
      }
    }
    for (const Task *MPT : MPTasks.TaskList[S]) {
      LLVM_DEBUG(dbgs() << "Testing against Task@"
                 << MPT->getEntry()->getName() << "\n");
      for (const Task *SubMPT : depth_first(MPT))
        for (GeneralAccess GA1 : SpindleAccessMap[S])
          for (GeneralAccess GA2 : TaskAccessMap[SubMPT])
            evaluateMaybeParallelAccesses(GA1, GA2, Result, ObjectMRForRace);
    }
    // If a successor of this spindle belongs to a subtask, recursively process
    // that subtask.
    for (const Spindle *Succ : successors(S)) {
      if (S->succInSubTask(Succ)) {
        // Skip successor spindles we've seen before.
        if (!Visited.insert(Succ).second)
          continue;
        // AncestorMPTasks.append(MPTasks.TaskList[S].begin(),
        //                        MPTasks.TaskList[S].end());
        checkForRacesHelper(Succ->getParentTask(), /*AncestorMPTasks,*/ Result,
                            ObjectMRForRace);
        // AncestorMPTasks.erase(AncestorMPTasks.begin() + OrigSize,
        //                       AncestorMPTasks.end());
      }
    }
  }
}

/// Check whether a pointer can participate in a runtime bounds check.
/// If \p Assume, try harder to prove that we can compute the bounds of \p Ptr
/// by adding run-time checks (overflow checks) if necessary.
static bool hasComputableBounds(PredicatedScalarEvolution &PSE,
                                const ValueToValueMap &Strides, Value *Ptr,
                                Loop *L, bool Assume) {
  const SCEV *PtrScev = replaceSymbolicStrideSCEV(PSE, Strides, Ptr);

  // The bounds for loop-invariant pointer is trivial.
  if (PSE.getSE()->isLoopInvariant(PtrScev, L))
    return true;

  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);

  if (!AR && Assume)
    AR = PSE.getAsAddRec(Ptr);

  if (!AR)
    return false;

  return AR->isAffine();
}

/// Check whether a pointer address cannot wrap.
static bool isNoWrap(PredicatedScalarEvolution &PSE,
                     const ValueToValueMap &Strides, Value *Ptr, Loop *L) {
  const SCEV *PtrScev = PSE.getSCEV(Ptr);
  if (PSE.getSE()->isLoopInvariant(PtrScev, L))
    return true;

  int64_t Stride = getPtrStride(PSE, Ptr, L, Strides);
  if (Stride == 1 || PSE.hasNoOverflow(Ptr, SCEVWrapPredicate::IncrementNUSW))
    return true;

  return false;
}

namespace {
// This class is based on LoopAccessAnalysis, but is not focused on
// vectorization.
class RTPtrCheckAnalysis {
public:
  using MemAccessInfo = PointerIntPair<Value *, 1, bool>;
  using MemAccessInfoList = SmallVector<MemAccessInfo, 8>;
  using DepCandidates = EquivalenceClasses<MemAccessInfo>;
  using UnderlyingObjToAccessMap = DenseMap<Value *, MemAccessInfo>;

  RTPtrCheckAnalysis(Loop *L, RuntimePointerChecking &RtCheck,
                     AliasAnalysis *AA, ScalarEvolution &SE)
      : TheLoop(L), RtCheck(RtCheck), PSE(SE, *L), AST(*AA) {}

  void addAccess(GeneralAccess GA, bool IsReadOnlyPtr = false) {
    if (GA.getPtr()) {
      LLVM_DEBUG(dbgs() << "Adding access for RT pointer checking:\n"
                 << "  GA.I: " << *GA.I << "\n"
                 << "  GA.Ptr: " << *GA.getPtr() << "\n");
      AST.add(GA.I);
      Value *Ptr = const_cast<Value *>(GA.getPtr());
      Accesses.insert(MemAccessInfo(Ptr, GA.isMod()));
      if (IsReadOnlyPtr)
        ReadOnlyPtr.insert(Ptr);
      collectStridedAccess(GA.I);
    }
  }
  void processAccesses(
      AccessPtrAnalysis::AccessToUnderlyingObjMap &AccessToObjs);
  bool canCheckPtrAtRT(bool ShouldCheckWrap = false);

  /// Initial processing of memory accesses determined that we need to
  /// perform dependency checking.
  ///
  /// Note that this can later be cleared if we retry memcheck analysis without
  /// dependency checking (i.e. FoundNonConstantDistanceDependence).
  bool isDependencyCheckNeeded() { return !CheckDeps.empty(); }

private:
  void collectStridedAccess(Value *MemAccess);
  bool createCheckForAccess(MemAccessInfo Access,
                            DenseMap<Value *, unsigned> &DepSetId,
                            unsigned &RunningDepId, unsigned ASId,
                            bool ShouldCheckWrap, bool Assume);

  /// The loop being checked.
  Loop *TheLoop;

  /// The resulting RT check.
  RuntimePointerChecking &RtCheck;

  SetVector<MemAccessInfo> Accesses;

  /// List of accesses that need a further dependence check.
  MemAccessInfoList CheckDeps;

  /// Set of pointers that are read only.
  SmallPtrSet<Value*, 16> ReadOnlyPtr;

  // Sets of potentially dependent accesses - members of one set share an
  // underlying pointer. The set "CheckDeps" identfies which sets really need a
  // dependence check.
  DepCandidates DepCands;

  /// The SCEV predicate containing all the SCEV-related assumptions.
  PredicatedScalarEvolution PSE;

  /// An alias set tracker to partition the access set by underlying object and
  /// intrinsic property (such as TBAA metadata).
  AliasSetTracker AST;

  /// Initial processing of memory accesses determined that we may need
  /// to add memchecks.  Perform the analysis to determine the necessary checks.
  ///
  /// Note that, this is different from isDependencyCheckNeeded.  When we retry
  /// memcheck analysis without dependency checking
  /// (i.e. FoundNonConstantDistanceDependence), isDependencyCheckNeeded is
  /// cleared while this remains set if we have potentially dependent accesses.
  bool IsRTCheckAnalysisNeeded = false;

  /// If an access has a symbolic strides, this maps the pointer value to
  /// the stride symbol.
  ValueToValueMap SymbolicStrides;

  /// Set of symbolic strides values.
  SmallPtrSet<Value *, 8> StrideSet;
};
} // end anonymous namespace

// This code is borrowed from LoopAccessAnalysis.cpp
void RTPtrCheckAnalysis::collectStridedAccess(Value *MemAccess) {
  Value *Ptr = nullptr;
  if (LoadInst *LI = dyn_cast<LoadInst>(MemAccess))
    Ptr = LI->getPointerOperand();
  else if (StoreInst *SI = dyn_cast<StoreInst>(MemAccess))
    Ptr = SI->getPointerOperand();
  else
    return;

  Value *Stride = getStrideFromPointer(Ptr, PSE.getSE(), TheLoop);
  if (!Stride)
    return;

  LLVM_DEBUG(dbgs() << "TapirRD: Found a strided access that is a candidate "
                       "for versioning:");
  LLVM_DEBUG(dbgs() << "  Ptr: " << *Ptr << " Stride: " << *Stride << "\n");

  // Avoid adding the "Stride == 1" predicate when we know that
  // Stride >= Trip-Count. Such a predicate will effectively optimize a single
  // or zero iteration loop, as Trip-Count <= Stride == 1.
  //
  // TODO: We are currently not making a very informed decision on when it is
  // beneficial to apply stride versioning. It might make more sense that the
  // users of this analysis (such as the vectorizer) will trigger it, based on
  // their specific cost considerations; For example, in cases where stride
  // versioning does  not help resolving memory accesses/dependences, the
  // vectorizer should evaluate the cost of the runtime test, and the benefit
  // of various possible stride specializations, considering the alternatives
  // of using gather/scatters (if available).

  const SCEV *StrideExpr = PSE.getSCEV(Stride);
  const SCEV *BETakenCount = PSE.getBackedgeTakenCount();

  // Match the types so we can compare the stride and the BETakenCount.
  // The Stride can be positive/negative, so we sign extend Stride;
  // The backdgeTakenCount is non-negative, so we zero extend BETakenCount.
  const DataLayout &DL = TheLoop->getHeader()->getModule()->getDataLayout();
  uint64_t StrideTypeSize = DL.getTypeAllocSize(StrideExpr->getType());
  uint64_t BETypeSize = DL.getTypeAllocSize(BETakenCount->getType());
  const SCEV *CastedStride = StrideExpr;
  const SCEV *CastedBECount = BETakenCount;
  ScalarEvolution *SE = PSE.getSE();
  if (BETypeSize >= StrideTypeSize)
    CastedStride = SE->getNoopOrSignExtend(StrideExpr, BETakenCount->getType());
  else
    CastedBECount = SE->getZeroExtendExpr(BETakenCount, StrideExpr->getType());
  const SCEV *StrideMinusBETaken = SE->getMinusSCEV(CastedStride, CastedBECount);
  // Since TripCount == BackEdgeTakenCount + 1, checking:
  // "Stride >= TripCount" is equivalent to checking:
  // Stride - BETakenCount > 0
  if (SE->isKnownPositive(StrideMinusBETaken)) {
    LLVM_DEBUG(
        dbgs() << "TapirRD: Stride>=TripCount; No point in versioning as the "
                  "Stride==1 predicate will imply that the loop executes "
                  "at most once.\n");
    return;
  }
  LLVM_DEBUG(dbgs() << "TapirRDA: Found a strided access that we can version.");

  SymbolicStrides[Ptr] = Stride;
  StrideSet.insert(Stride);
}

// This code is based on AccessAnalysis::processMemAccesses() in
// LoopAccessAnalysis.cpp.
void RTPtrCheckAnalysis::processAccesses(
    AccessPtrAnalysis::AccessToUnderlyingObjMap &AccessToObjs) {
  // The AliasSetTracker has nicely partitioned our pointers by metadata
  // compatibility and potential for underlying-object overlap. As a result, we
  // only need to check for potential pointer dependencies within each alias
  // set.
  for (auto &AS : AST) {
    // Note that both the alias-set tracker and the alias sets themselves used
    // linked lists internally and so the iteration order here is deterministic
    // (matching the original instruction order within each set).

    bool SetHasWrite = false;

    // Map of pointers to last access encountered.
    UnderlyingObjToAccessMap ObjToLastAccess;

    // Set of access to check after all writes have been processed.
    SetVector<MemAccessInfo> DeferredAccesses;

    // Iterate over each alias set twice, once to process read/write pointers,
    // and then to process read-only pointers.
    for (int SetIteration = 0; SetIteration < 2; ++SetIteration) {
      bool UseDeferred = SetIteration > 0;
      SetVector<MemAccessInfo> &S = UseDeferred ? DeferredAccesses : Accesses;

      for (auto AV : AS) {
        Value *Ptr = AV.getValue();
        LLVM_DEBUG(dbgs() << "Found pointer is alias set: " << *Ptr << "\n");

        // For a single memory access in AliasSetTracker, Accesses may contain
        // both read and write, and they both need to be handled for CheckDeps.
        for (auto AC : S) {
          LLVM_DEBUG(dbgs() << "  Access pointer: " << *AC.getPointer() << "\n");
          if (AC.getPointer() != Ptr)
            continue;

          bool IsWrite = AC.getInt();

          // If we're using the deferred access set, then it contains only
          // reads.
          bool IsReadOnlyPtr = ReadOnlyPtr.count(Ptr) && !IsWrite;
          if (UseDeferred && !IsReadOnlyPtr)
            continue;
          // Otherwise, the pointer must be in the PtrAccessSet, either as a
          // read or a write.
          assert(((IsReadOnlyPtr && UseDeferred) || IsWrite ||
                  S.count(MemAccessInfo(Ptr, false))) &&
                 "Alias-set pointer not in the access set?");

          MemAccessInfo Access(Ptr, IsWrite);
          DepCands.insert(Access);

          // Memorize read-only pointers for later processing and skip them in
          // the first round (they need to be checked after we have seen all
          // write pointers). Note: we also mark pointer that are not
          // consecutive as "read-only" pointers (so that we check
          // "a[b[i]] +="). Hence, we need the second check for "!IsWrite".
          if (!UseDeferred && IsReadOnlyPtr) {
            DeferredAccesses.insert(Access);
            continue;
          }

          // If this is a write - check other reads and writes for conflicts. If
          // this is a read only check other writes for conflicts (but only if
          // there is no other write to the ptr - this is an optimization to
          // catch "a[i] = a[i] + " without having to do a dependence check).
          if ((IsWrite || IsReadOnlyPtr) && SetHasWrite) {
            CheckDeps.push_back(Access);
            IsRTCheckAnalysisNeeded = true;
          }

          if (IsWrite)
            SetHasWrite = true;

          for (Value *Obj : AccessToObjs[
                   AccessPtrAnalysis::MemAccessInfo(Ptr, IsWrite)]) {
            UnderlyingObjToAccessMap::iterator Prev =
              ObjToLastAccess.find(Obj);
            if (Prev != ObjToLastAccess.end())
              DepCands.unionSets(Access, Prev->second);

            ObjToLastAccess[Obj] = Access;
          }
        }
      }
    }
  }
}

// This code is borrowed from LoopAccessAnalysis.cpp
bool RTPtrCheckAnalysis::createCheckForAccess(
    MemAccessInfo Access,  DenseMap<Value *, unsigned> &DepSetId,
    unsigned &RunningDepId, unsigned ASId, bool ShouldCheckWrap, bool Assume) {
  Value *Ptr = Access.getPointer();

  if (!hasComputableBounds(PSE, SymbolicStrides, Ptr, TheLoop, Assume))
    return false;

  // When we run after a failing dependency check we have to make sure
  // we don't have wrapping pointers.
  if (ShouldCheckWrap && !isNoWrap(PSE, SymbolicStrides, Ptr, TheLoop)) {
    auto *Expr = PSE.getSCEV(Ptr);
    if (!Assume || !isa<SCEVAddRecExpr>(Expr))
      return false;
    PSE.setNoOverflow(Ptr, SCEVWrapPredicate::IncrementNUSW);
  }

  // The id of the dependence set.
  unsigned DepId;

  if (isDependencyCheckNeeded()) {
    Value *Leader = DepCands.getLeaderValue(Access).getPointer();
    unsigned &LeaderId = DepSetId[Leader];
    if (!LeaderId)
      LeaderId = RunningDepId++;
    DepId = LeaderId;
  } else
    // Each access has its own dependence set.
    DepId = RunningDepId++;

  bool IsWrite = Access.getInt();
  RtCheck.insert(TheLoop, Ptr, IsWrite, DepId, ASId, SymbolicStrides, PSE);
  LLVM_DEBUG(dbgs() << "TapirRD: Found a runtime check ptr:" << *Ptr << '\n');

  return true;
}

// This code is borrowed from LoopAccessAnalysis.cpp
bool RTPtrCheckAnalysis::canCheckPtrAtRT(bool ShouldCheckWrap) {
  // Find pointers with computable bounds. We are going to use this information
  // to place a runtime bound check.
  bool CanDoRT = true;

  bool NeedRTCheck = false;
  if (!IsRTCheckAnalysisNeeded) return true;

  bool IsDepCheckNeeded = isDependencyCheckNeeded();

  // We assign a consecutive id to access from different alias sets.
  // Accesses between different groups doesn't need to be checked.
  unsigned ASId = 1;
  for (auto &AS : AST) {
    int NumReadPtrChecks = 0;
    int NumWritePtrChecks = 0;
    bool CanDoAliasSetRT = true;

    // We assign consecutive id to access from different dependence sets.
    // Accesses within the same set don't need a runtime check.
    unsigned RunningDepId = 1;
    DenseMap<Value *, unsigned> DepSetId;

    SmallVector<MemAccessInfo, 4> Retries;

    for (auto A : AS) {
      Value *Ptr = A.getValue();
      bool IsWrite = Accesses.count(MemAccessInfo(Ptr, true));
      MemAccessInfo Access(Ptr, IsWrite);

      if (IsWrite)
        ++NumWritePtrChecks;
      else
        ++NumReadPtrChecks;

      if (!createCheckForAccess(Access, DepSetId, RunningDepId, ASId,
                                ShouldCheckWrap, false)) {
        LLVM_DEBUG(dbgs() << "TapirRD: Can't find bounds for ptr:" << *Ptr << '\n');
        Retries.push_back(Access);
        CanDoAliasSetRT = false;
      }
    }

    // If we have at least two writes or one write and a read then we need to
    // check them.  But there is no need to checks if there is only one
    // dependence set for this alias set.
    //
    // Note that this function computes CanDoRT and NeedRTCheck independently.
    // For example CanDoRT=false, NeedRTCheck=false means that we have a pointer
    // for which we couldn't find the bounds but we don't actually need to emit
    // any checks so it does not matter.
    bool NeedsAliasSetRTCheck = false;
    if (!(IsDepCheckNeeded && CanDoAliasSetRT && RunningDepId == 2))
      NeedsAliasSetRTCheck = (NumWritePtrChecks >= 2 ||
                             (NumReadPtrChecks >= 1 && NumWritePtrChecks >= 1));

    // We need to perform run-time alias checks, but some pointers had bounds
    // that couldn't be checked.
    if (NeedsAliasSetRTCheck && !CanDoAliasSetRT) {
      // Reset the CanDoSetRt flag and retry all accesses that have failed.
      // We know that we need these checks, so we can now be more aggressive
      // and add further checks if required (overflow checks).
      CanDoAliasSetRT = true;
      for (auto Access : Retries)
        if (!createCheckForAccess(Access, DepSetId, RunningDepId, ASId,
                                  ShouldCheckWrap, /*Assume=*/true)) {
          CanDoAliasSetRT = false;
          break;
        }
    }

    CanDoRT &= CanDoAliasSetRT;
    NeedRTCheck |= NeedsAliasSetRTCheck;
    ++ASId;
  }

  // If the pointers that we would use for the bounds comparison have different
  // address spaces, assume the values aren't directly comparable, so we can't
  // use them for the runtime check. We also have to assume they could
  // overlap. In the future there should be metadata for whether address spaces
  // are disjoint.
  unsigned NumPointers = RtCheck.Pointers.size();
  for (unsigned i = 0; i < NumPointers; ++i) {
    for (unsigned j = i + 1; j < NumPointers; ++j) {
      // Only need to check pointers between two different dependency sets.
      if (RtCheck.Pointers[i].DependencySetId ==
          RtCheck.Pointers[j].DependencySetId)
       continue;
      // Only need to check pointers in the same alias set.
      if (RtCheck.Pointers[i].AliasSetId != RtCheck.Pointers[j].AliasSetId)
        continue;

      Value *PtrI = RtCheck.Pointers[i].PointerValue;
      Value *PtrJ = RtCheck.Pointers[j].PointerValue;

      unsigned ASi = PtrI->getType()->getPointerAddressSpace();
      unsigned ASj = PtrJ->getType()->getPointerAddressSpace();
      if (ASi != ASj) {
        LLVM_DEBUG(
            dbgs() << "TapirRD: Runtime check would require comparison between"
                      " different address spaces\n");
        return false;
      }
    }
  }

  if (NeedRTCheck && CanDoRT)
    RtCheck.generateChecks(DepCands, IsDepCheckNeeded);

  LLVM_DEBUG(dbgs() << "TapirRD: We need to do " << RtCheck.getNumberOfChecks()
                    << " pointer comparisons.\n");

  RtCheck.Need = NeedRTCheck;

  bool CanDoRTIfNeeded = !NeedRTCheck || CanDoRT;
  if (!CanDoRTIfNeeded)
    RtCheck.reset();
  return CanDoRTIfNeeded;
}

// This code is borrowed from LoopAccessAnalysis.cpp
static bool canAnalyzeLoop(Loop *TheLoop, PredicatedScalarEvolution &PSE) {
  // We need to have a loop header.
  LLVM_DEBUG(dbgs() << "TapirRD: Found a loop in "
                    << TheLoop->getHeader()->getParent()->getName() << ": "
                    << TheLoop->getHeader()->getName() << '\n');

  // We can only analyze innermost loops.
  if (!TheLoop->empty()) {
    LLVM_DEBUG(dbgs() << "TapirRD: loop is not the innermost loop\n");
    // recordAnalysis("NotInnerMostLoop") << "loop is not the innermost loop";
    return false;
  }

  // We must have a single backedge.
  if (TheLoop->getNumBackEdges() != 1) {
    LLVM_DEBUG(
        dbgs() << "TapirRD: loop control flow is not understood by analyzer\n");
    // recordAnalysis("CFGNotUnderstood")
    //     << "loop control flow is not understood by analyzer";
    return false;
  }

  // We must have a single exiting block.
  if (!TheLoop->getExitingBlock()) {
    LLVM_DEBUG(
        dbgs() << "TapirRD: loop control flow is not understood by analyzer\n");
    // recordAnalysis("CFGNotUnderstood")
    //     << "loop control flow is not understood by analyzer";
    return false;
  }

  // We only handle bottom-tested loops, i.e. loop in which the condition is
  // checked at the end of each iteration. With that we can assume that all
  // instructions in the loop are executed the same number of times.
  if (TheLoop->getExitingBlock() != TheLoop->getLoopLatch()) {
    LLVM_DEBUG(
        dbgs() << "TapirRD: loop control flow is not understood by analyzer\n");
    // recordAnalysis("CFGNotUnderstood")
    //     << "loop control flow is not understood by analyzer";
    return false;
  }

  // ScalarEvolution needs to be able to find the exit count.
  const SCEV *ExitCount = PSE.getBackedgeTakenCount();
  if (ExitCount == PSE.getSE()->getCouldNotCompute()) {
    // recordAnalysis("CantComputeNumberOfIterations")
    //     << "could not determine number of loop iterations";
    LLVM_DEBUG(dbgs() << "TapirRD: SCEV could not compute the loop exit count.\n");
    return false;
  }

  return true;
}

void AccessPtrAnalysis::getRTPtrChecks(Loop *L, RaceInfo::ResultTy &Result,
                                       RaceInfo::PtrChecksTy &AllPtrRtChecks) {
  LLVM_DEBUG(dbgs() << "getRTPtrChecks: " << *L << "\n");

  AllPtrRtChecks[L] =
    llvm::make_unique<RuntimePointerChecking>(&SE);

  RTPtrCheckAnalysis RPCA(L, *AllPtrRtChecks[L].get(), AA, SE);
  SmallPtrSet<const Value *, 16> Seen;
  // First handle all stores
  for (GeneralAccess GA : LoopAccessMap[L]) {
    // Exclude accesses not involved in a local race
    if (!Result.count(GA.I) ||
        !RaceInfo::isLocalRace(Result.getRaceType(GA.I)))
      continue;

    if (GA.isMod()) {
      RPCA.addAccess(GA);
      if (GA.getPtr())
        Seen.insert(GA.getPtr());
    }
  }
  // Now handle loads, checking if any pointers are only read from
  for (GeneralAccess GA : LoopAccessMap[L]) {
    // Exclude accesses not involved in a local race
    if (!Result.count(GA.I) ||
        !RaceInfo::isLocalRace(Result.getRaceType(GA.I)))
      continue;

    if (!GA.isMod()) {
      if (!GA.getPtr())
        RPCA.addAccess(GA);

      RPCA.addAccess(GA, !Seen.count(GA.getPtr()));
    }
  }

  RPCA.processAccesses(AccessToObjs);

  bool CanDoRTIfNeeded = RPCA.canCheckPtrAtRT();

  // TODO: Do something with CanDoRTIfNeeded
}

void AccessPtrAnalysis::processAccessPtrs(
    RaceInfo::ResultTy &Result, RaceInfo::ObjectMRTy &ObjectMRForRace,
    RaceInfo::PtrChecksTy &AllPtrRtChecks) {
  TI.evaluateParallelState<MaybeParallelTasks>(MPTasks);
  TI.evaluateParallelState<MaybeParallelTasksInLoopBody>(MPTasksInLoop);

  // using InstPtrPair = std::pair<const Instruction *, const Value *>;
  // SmallPtrSet<InstPtrPair, 32> Visited;
  for (const Spindle *S :
         depth_first<Spindle *>(TI.getRootTask()->getEntrySpindle())) {
    for (GeneralAccess GA : SpindleAccessMap[S]) {
      // InstPtrPair Visit =
      //   std::make_pair<const Instruction *, const Value *>(GA.I, GA.getPtr());
      // // Skip instructions we've already visited.
      // if (!Visited.insert(Visit).second)
      //   continue;

      if (!GA.getPtr()) {
        if (const CallBase *Call = dyn_cast<CallBase>(GA.I)) {
          if (!Call->onlyAccessesArgMemory() &&
              !(AssumeSafeMalloc && (isAllocationFn(Call, TLI) ||
                                     isFreeCall(Call, TLI)))) {
            LLVM_DEBUG(dbgs() << "Setting opaque race:\n" << "  GA.I: "
                       << *GA.I << "\n");
            Result.recordOpaqueRace(GA);
          }
        }
      }

      // Check for aliasing against the function arguments.
      for (Value *ArgPtr : ArgumentPtrs) {
        LLVM_DEBUG({
            dbgs() << "Checking instruction against arg pointer:\n"
                   << "  GA.I: " << *GA.I << "\n"
                   << "  Arg: " << *ArgPtr << "\n";
          });
        if (!GA.getPtr()) {
          ModRefInfo MRI = AA->getModRefInfo(GA.I, MemoryLocation(ArgPtr));
          if (isModSet(MRI)) {
            LLVM_DEBUG(dbgs() << "  Mod is set.\n");
            // recordAncestorRace(GA, ArgPtr, Result, ObjectMRForRace);
            Result.recordRaceViaAncestorRef(GA);
            Result.recordRaceViaAncestorMod(GA);
            setObjectMRForRace(ObjectMRForRace, ArgPtr, ModRefInfo::ModRef);
          }
          if (isRefSet(MRI)) {
            LLVM_DEBUG(dbgs() << "  Ref is set.\n");
            // recordAncestorRace(GA, ArgPtr, Result, ObjectMRForRace);
            Result.recordRaceViaAncestorMod(GA);
            setObjectMRForRace(ObjectMRForRace, ArgPtr, ModRefInfo::Mod);
          }
        } else {
          MemoryLocation GALoc = *GA.Loc;
          if (AA->alias(GALoc, MemoryLocation(ArgPtr))) {
            Argument *Arg = cast<Argument>(ArgPtr);
            if (GA.isMod() && !Arg->onlyReadsMemory()) {
              LLVM_DEBUG(dbgs() << "  Mod is set.\n");
              // recordAncestorRace(GA, ArgPtr, Result, ObjectMRForRace);
              Result.recordRaceViaAncestorRef(GA);
              Result.recordRaceViaAncestorMod(GA);
              setObjectMRForRace(ObjectMRForRace, ArgPtr, ModRefInfo::ModRef);
            }
            if (GA.isRef()) {
              LLVM_DEBUG(dbgs() << "  Ref is set.\n");
              // recordAncestorRace(GA, ArgPtr, Result, ObjectMRForRace);
              Result.recordRaceViaAncestorMod(GA);
              setObjectMRForRace(ObjectMRForRace, ArgPtr, ModRefInfo::Mod);
            }
          }
        }
      }
    }
  }
  // SmallVector<const Task *, 4> RootMPTasks;
  checkForRacesHelper(TI.getRootTask(), /*RootMPTasks,*/ Result,
                      ObjectMRForRace);

  // Based on preliminary experiments, it doesn't appear that getRTPtrChecks,
  // which is adapted from LoopAccessAnalysis, comes up with enough runtime
  // pointer checks often enough to be worthwhile.  It might be worth revisiting
  // this code later.

  // for (Loop *TopLevelLoop : LI) {
  //   for (Loop *L : depth_first(TopLevelLoop)) {
  //     PredicatedScalarEvolution PSE(SE, *L);
  //     if (canAnalyzeLoop(L, PSE))
  //       getRTPtrChecks(L, Result, AllPtrRtChecks);
  //   }
  // }
}

RaceInfo::RaceInfo(Function *F, DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
                   DependenceInfo &DI, ScalarEvolution &SE,
                   const TargetLibraryInfo *TLI)
    : F(F), DT(DT), LI(LI), TI(TI), DI(DI), SE(SE), TLI(TLI) {
  analyzeFunction();
}

void RaceInfo::getObjectsFor(Instruction *I,
                             SmallPtrSetImpl<Value *> &Objects) {
  SmallVector<GeneralAccess, 1> GA;
  GetGeneralAccesses(I, GA, DI.getAA(), TLI);
  for (GeneralAccess Acc : GA) {
    // Skip this access if it does not have a valid pointer.
    if (!Acc.getPtr())
      continue;

    getObjectsFor(MemAccessInfo(Acc.getPtr(), Acc.isMod()), Objects);
  }
}

void RaceInfo::getObjectsFor(MemAccessInfo Access,
                             SmallPtrSetImpl<Value *> &Objects) {
  for (Value *Obj : AccessToObjs[Access])
    Objects.insert(Obj);
}

void RaceInfo::print(raw_ostream &OS) const {
  if (Result.empty()) {
    OS << "No possible races\n";
    return;
  }
  RaceType OverallRT = getOverallRaceType();
  OS << "Overall race type:";
  printRaceType(OverallRT, OS);
  OS << "\n";
  for (auto Res : Result) {
    OS << *Res.first << "\n";
    for (auto &RD : Res.second) {
      if (RD.getPtr())
        OS << "    " << *RD.getPtr();
      else
        OS << "    null";
      OS << "\n    ";
      printRaceType(RD.Type, OS);
      OS << "\n";
    }
  }
  OS << "Underlying objects of races:\n";
  for (auto Res : ObjectMRForRace) {
    OS << *Res.first << "\n   ";
    if (isModSet(Res.second))
      OS << " Mod";
    if (isRefSet(Res.second))
      OS << " Ref";
    OS << "\n";
  }
  OS << "RT pointer checks:\n";
  for (auto &RtChecks : AllPtrRtChecks) {
    OS << RtChecks.first->getHeader()->getName() << ":\n";
    RtChecks.second->print(OS);
  }
  // for (Loop *TopLevelLoop : LI)
  //   for (Loop *L : depth_first(TopLevelLoop))
  //     if (AllPtrRtChecks.count(L)) {
  //       auto &RtChecks = AllPtrRtChecks.find(L);
  //       OS.indent(2) << L->getHeader()->getName() << ":\n";
  //       (*RtChecks)->print(OS);
  //     }
}

// The main analysis routine.
void RaceInfo::analyzeFunction() {
  LLVM_DEBUG(dbgs() << "Analyzing function '" << F->getName() << "'\n");

  const DataLayout &DL = F->getParent()->getDataLayout();

  // At a high level, we need to identify pairs of instructions that might
  // execute in parallel and alias.

  AccessPtrAnalysis APA(DL, DT, TI, LI, DI, SE, TLI, AccessToObjs);
  // Record pointer arguments to this function
  for (Argument &Arg : F->args())
    if (Arg.getType()->isPtrOrPtrVectorTy())
      APA.addFunctionArgument(&Arg);
  // TODO: Add global variables to APA.

  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.mayReadFromMemory() || I.mayWriteToMemory()) {
        if (checkInstructionForRace(&I, TLI))
          APA.addAccess(&I);
      }
    }
  }

  APA.processAccessPtrs(Result, ObjectMRForRace, AllPtrRtChecks);
}
