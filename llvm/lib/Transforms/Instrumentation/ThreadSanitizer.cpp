//===-- ThreadSanitizer.cpp - race detector -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "tsan"

static cl::opt<bool> ClInstrumentMemoryAccesses(
    "tsan-instrument-memory-accesses", cl::init(true),
    cl::desc("Instrument memory accesses"), cl::Hidden);
static cl::opt<bool>
    ClInstrumentFuncEntryExit("tsan-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool> ClHandleCxxExceptions(
    "tsan-handle-cxx-exceptions", cl::init(true),
    cl::desc("Handle C++ exceptions (insert cleanup blocks for unwinding)"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentAtomics("tsan-instrument-atomics",
                                         cl::init(true),
                                         cl::desc("Instrument atomics"),
                                         cl::Hidden);
static cl::opt<bool> ClInstrumentMemIntrinsics(
    "tsan-instrument-memintrinsics", cl::init(true),
    cl::desc("Instrument memintrinsics (memset/memcpy/memmove)"), cl::Hidden);
static cl::opt<bool> ClDistinguishVolatile(
    "tsan-distinguish-volatile", cl::init(false),
    cl::desc("Emit special instrumentation for accesses to volatiles"),
    cl::Hidden);
static cl::opt<bool> ClInstrumentReadBeforeWrite(
    "tsan-instrument-read-before-write", cl::init(false),
    cl::desc("Do not eliminate read instrumentation for read-before-writes"),
    cl::Hidden);
static cl::opt<bool> ClCompoundReadBeforeWrite(
    "tsan-compound-read-before-write", cl::init(false),
    cl::desc("Emit special compound instrumentation for reads-before-writes"),
    cl::Hidden);
static cl::opt<bool>
    ClOmitNonCaptured("tsan-omit-by-pointer-capturing", cl::init(true),
                      cl::desc("Omit accesses due to pointer capturing"),
                      cl::Hidden);
// Removes redundant TSan instrumentation using CFG dominance analysis.
// If access A dominates access B, both target the same memory location
// (confirmed via MustAlias), A covers at least as many bytes as B, and no
// synchronization (atomic, fence, or call without nosync) exists on any CFG
// path from A to B, then B's instrumentation is redundant: any race
// detectable at B would already be detected at A.
// Coverage rule: a dominating write subsumes a subsequent read or write; a
// dominating read subsumes only a subsequent read.
// Post-dominance elimination (removing A when B post-dominates A) will be
// added in a follow-up patch. See DominanceBasedElimination for the full
// correctness argument.
static cl::opt<bool>
    ClUseDominanceAnalysis("tsan-use-dominance-analysis", cl::init(true),
                           cl::desc("Eliminate redundant instrumentation using "
                                    "dominance analysis"),
                           cl::Hidden);

STATISTIC(NumInstrumentedReads, "Number of instrumented reads");
STATISTIC(NumInstrumentedWrites, "Number of instrumented writes");
STATISTIC(NumOmittedReadsBeforeWrite,
          "Number of reads ignored due to following writes");
STATISTIC(NumAccessesWithBadSize, "Number of accesses with bad size");
STATISTIC(NumInstrumentedVtableWrites, "Number of vtable ptr writes");
STATISTIC(NumInstrumentedVtableReads, "Number of vtable ptr reads");
STATISTIC(NumOmittedReadsFromConstantGlobals,
          "Number of reads from constant globals");
STATISTIC(NumOmittedReadsFromVtable, "Number of vtable reads");
STATISTIC(NumOmittedNonCaptured, "Number of accesses ignored due to capturing");
STATISTIC(NumOmittedByDominance, "Number of accesses ignored due to dominance");

const char kTsanModuleCtorName[] = "tsan.module_ctor";
const char kTsanInitName[] = "__tsan_init";

namespace {
// Internal Instruction wrapper that contains more information about the
// Instruction from prior analysis.
struct InstructionInfo {
  // Instrumentation emitted for this instruction is for a compounded set of
  // read and write operations in the same basic block.
  static constexpr unsigned kCompoundRW = (1U << 0);

  explicit InstructionInfo(Instruction *Inst) : Inst(Inst) {}

  bool isWriteOperation() const {
    return isa<StoreInst>(Inst) || (Flags & kCompoundRW);
  }

  Instruction *Inst;
  unsigned Flags = 0;
};

/// Eliminates redundant TSan instrumentation using dominance analysis.
///
/// An instrumented access B is redundant if there exists a prior access A such
/// that all four conditions hold:
///
///   (1) A dominates B — A executes on every path that reaches B, so any
///       execution that sees a race at B also sees A beforehand.
///   (2) A and B MustAlias — they access the same concrete memory location,
///       so a race partner of B would also race with A.
///   (3) Coverage: if B is a write, A must also be a write. A write races with
///       both reads and writes, so a dominating write covers either type. A
///       read races only with writes, so a dominating read cannot cover a
///       subsequent write (the write could introduce a new race with a third
///       thread's read).
///   (4) The path from A to B is synchronization-free — no atomic, fence, or
///       call without `nosync` appears on any CFG path between them. If such an
///       instruction existed, a second thread could synchronize with the
///       current thread between A and B (establishing a new happens-before
///       edge), making it possible for B to race with that thread while A does
///       not.
///
/// When all four conditions hold, any race detectable at B is already
/// detectable at A; removing B's instrumentation is sound.
class DominanceBasedElimination {
public:
  DominanceBasedElimination(SmallVectorImpl<InstructionInfo> &AllInstr,
                            DominatorTree &DT, AAResults &AA)
      : AllInstr(AllInstr), DT(DT), AA(AA) {}

  void run() { eliminate(); }

private:
  /// Returns true if the instruction cannot cause inter-thread synchronization.
  static bool isInstrSafe(const Instruction *Inst);

  /// Returns true if no synchronization exists on any CFG path from DomInst
  /// to CurrInst. CanReachEnd must be the reverse-reachability set of
  /// CurrInst's block, pre-computed once per CurrInst by the caller.
  static bool
  isPathClear(Instruction *DomInst, Instruction *CurrInst,
              const SmallPtrSetImpl<const BasicBlock *> &CanReachEnd);

  DenseMap<Instruction *, size_t> createInstrToIndexMap() const;

  /// Searches AllInstr for a dominating instruction that covers AllInstr[i].
  /// Marks AllInstr[i] for removal and returns true if one is found.
  bool findAndMarkDominatingInstr(
      size_t i, const DenseMap<Instruction *, size_t> &InstrToIndexMap,
      const SmallPtrSetImpl<const BasicBlock *> &CanReachEnd,
      SmallVectorImpl<bool> &ToRemove);

  void eliminate();

  SmallVectorImpl<InstructionInfo> &AllInstr;
  DominatorTree &DT;
  AAResults &AA;
};

/// ThreadSanitizer: instrument the code in module to find races.
///
/// Instantiating ThreadSanitizer inserts the tsan runtime library API function
/// declarations into the module if they don't exist already. Instantiating
/// ensures the __tsan_init function is in the list of global constructors for
/// the module.
struct ThreadSanitizer {
  ThreadSanitizer(DominatorTree *DT = nullptr, AAResults *AA = nullptr)
      : DT(DT), AA(AA) {
    // Check options and warn user.
    if (ClInstrumentReadBeforeWrite && ClCompoundReadBeforeWrite) {
      errs()
          << "warning: Option -tsan-compound-read-before-write has no effect "
             "when -tsan-instrument-read-before-write is set.\n";
    }
  }

  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);

private:
  void initialize(Module &M, const TargetLibraryInfo &TLI);
  bool instrumentLoadOrStore(const InstructionInfo &II, const DataLayout &DL);
  bool instrumentAtomic(Instruction *I, const DataLayout &DL);
  bool instrumentMemIntrinsic(Instruction *I);
  void chooseInstructionsToInstrument(SmallVectorImpl<Instruction *> &Local,
                                      SmallVectorImpl<InstructionInfo> &All,
                                      const DataLayout &DL);
  bool addrPointsToConstantData(Value *Addr);
  int getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr, const DataLayout &DL);
  void InsertRuntimeIgnores(Function &F);

  DominatorTree *DT = nullptr;
  AAResults *AA = nullptr;

  Type *IntptrTy;
  FunctionCallee TsanFuncEntry;
  FunctionCallee TsanFuncExit;
  FunctionCallee TsanIgnoreBegin;
  FunctionCallee TsanIgnoreEnd;
  // Accesses sizes are powers of two: 1, 2, 4, 8, 16.
  static const size_t kNumberOfAccessSizes = 5;
  FunctionCallee TsanRead[kNumberOfAccessSizes];
  FunctionCallee TsanWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedWrite[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileRead[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedVolatileWrite[kNumberOfAccessSizes];
  FunctionCallee TsanCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanUnalignedCompoundRW[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicLoad[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicStore[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicRMW[AtomicRMWInst::LAST_BINOP + 1]
                              [kNumberOfAccessSizes];
  FunctionCallee TsanAtomicCAS[kNumberOfAccessSizes];
  FunctionCallee TsanAtomicThreadFence;
  FunctionCallee TsanAtomicSignalFence;
  FunctionCallee TsanVptrUpdate;
  FunctionCallee TsanVptrLoad;
  FunctionCallee MemmoveFn, MemcpyFn, MemsetFn;
};

//-----------------------------------------------------------------------------
// DominanceBasedElimination Implementation
//-----------------------------------------------------------------------------

static bool isTsanAtomic(const Instruction *I) {
  // TODO: Ask TTI whether synchronization scope is between threads.
  auto SSID = getAtomicSyncScopeID(I);
  if (!SSID)
    return false;
  if (isa<LoadInst>(I) || isa<StoreInst>(I))
    return *SSID != SyncScope::SingleThread;
  return true;
}

bool DominanceBasedElimination::isInstrSafe(const Instruction *Inst) {
  // Atomics with inter-thread scope establish happens-before between threads
  // (e.g., a release store followed by an acquire load synchronizes the two
  // threads). Any such operation on the path from A to B would allow another
  // thread to "see" the state after A but before B, potentially racing with B
  // while A does not observe the race.
  if (isTsanAtomic(Inst))
    return false;

  // A function call may contain mutex unlocks, thread spawns, or other
  // release-like operations that create new happens-before edges. A call is
  // safe only if it is guaranteed not to synchronize with other threads in
  // any way. hasFnAttr checks both the call-site attribute and the callee's
  // function attributes, so it handles direct calls, indirect calls with a
  // nosync call-site annotation, and intrinsics uniformly.
  if (const auto *CB = dyn_cast<CallBase>(Inst))
    return CB->hasFnAttr(Attribute::NoSync);

  // Fences are handled by isTsanAtomic above. All remaining instructions
  // (arithmetic, branches, GEPs, etc.) cannot establish inter-thread
  // happens-before and are safe to cross.
  return true;
}

DenseMap<Instruction *, size_t>
DominanceBasedElimination::createInstrToIndexMap() const {
  DenseMap<Instruction *, size_t> InstrToIndexMap;
  InstrToIndexMap.reserve(AllInstr.size());
  for (size_t i = 0; i < AllInstr.size(); ++i)
    InstrToIndexMap[AllInstr[i].Inst] = i;
  return InstrToIndexMap;
}

bool DominanceBasedElimination::isPathClear(
    Instruction *DomInst, Instruction *CurrInst,
    const SmallPtrSetImpl<const BasicBlock *> &CanReachEnd) {
  BasicBlock *DomBB = DomInst->getParent();
  BasicBlock *CurrBB = CurrInst->getParent();

  // Intra-BB: only the instructions strictly between DomInst and CurrInst
  // can introduce synchronization. Scan the open interval (DomInst, CurrInst).
  // A loop around CurrBB is irrelevant here: DomInst and CurrInst share a
  // block with DomInst first, so every dynamic execution of CurrInst is
  // immediately preceded by DomInst within the same iteration.
  if (DomBB == CurrBB) {
    for (auto It = std::next(DomInst->getIterator());
         It != CurrInst->getIterator(); ++It)
      if (!isInstrSafe(&*It))
        return false;
    return true;
  }

  // Forward BFS that scans, in full, every block reachable from the given seed
  // successors while staying inside the cone (CanReachEnd) and never passing
  // back through CurrBB. Returns false if any scanned block synchronizes.
  // Restricting traversal to CanReachEnd (the reverse-reachability set of
  // CurrBB computed by the caller) ensures we only inspect blocks that lie on
  // a path to CurrBB; blocks on diverging paths that never reach CurrBB must
  // not veto the elimination.
  auto ConeBlocksClear = [&](auto Seeds) {
    SmallPtrSet<const BasicBlock *, 32> Visited;
    SmallVector<const BasicBlock *, 16> Worklist;
    auto Enqueue = [&](const BasicBlock *BB) {
      if (CanReachEnd.count(BB) && BB != CurrBB && Visited.insert(BB).second)
        Worklist.push_back(BB);
    };
    for (const BasicBlock *Succ : Seeds)
      Enqueue(Succ);
    while (!Worklist.empty()) {
      const BasicBlock *BB = Worklist.pop_back_val();
      for (const Instruction &I : *BB)
        if (!isInstrSafe(&I))
          return false;
      for (const BasicBlock *Succ : successors(BB))
        Enqueue(Succ);
    }
    return true;
  };

  // Inter-BB: every path from DomInst to CurrInst is covered by four regions.
  //
  // Region 1 — suffix of DomBB (instructions after DomInst in DomInst's block).
  // These execute on every path from DomInst before leaving the block.
  for (auto It = std::next(DomInst->getIterator()); It != DomBB->end(); ++It)
    if (!isInstrSafe(&*It))
      return false;

  // Region 2 — prefix of CurrBB (instructions before CurrInst in CurrInst's
  // block). These execute on every path that arrives at CurrInst.
  for (auto It = CurrBB->begin(); It != CurrInst->getIterator(); ++It)
    if (!isInstrSafe(&*It))
      return false;

  // Region 3 — intermediate blocks on a loop-free path from DomBB to CurrBB.
  if (!ConeBlocksClear(successors(DomBB)))
    return false;

  // Region 4 — loop back-edge. If CurrInst can be reached again from itself,
  // then between two consecutive dynamic executions of CurrInst the program
  // runs CurrInst -> (suffix of CurrBB) -> ... -> (prefix of CurrBB) ->
  // CurrInst. DomInst dominates CurrInst but, lying outside the loop, may
  // execute only once and therefore cannot cover the later executions of
  // CurrInst unless this back-edge path is itself synchronization-free. (When
  // DomInst is inside the loop it re-executes every iteration and covers
  // CurrInst directly; this check may then reject soundly-eliminable cases,
  // but it is never unsound.) The cycle exists iff some successor of CurrBB can
  // still reach CurrBB, i.e. is in CanReachEnd.
  bool CurrInCycle = false;
  for (const BasicBlock *Succ : successors(CurrBB))
    if (CanReachEnd.count(Succ)) {
      CurrInCycle = true;
      break;
    }
  if (CurrInCycle) {
    // Suffix of CurrBB (instructions after CurrInst), which executes before the
    // back-edge re-enters CurrBB. CurrBB's prefix is already covered by
    // Region 2, so the two scans together cover the whole block.
    for (auto It = std::next(CurrInst->getIterator()); It != CurrBB->end();
         ++It)
      if (!isInstrSafe(&*It))
        return false;
    // Blocks on the CurrBB -> CurrBB back-edge path.
    if (!ConeBlocksClear(successors(CurrBB)))
      return false;
  }

  return true;
}

bool DominanceBasedElimination::findAndMarkDominatingInstr(
    size_t i, const DenseMap<Instruction *, size_t> &InstrToIndexMap,
    const SmallPtrSetImpl<const BasicBlock *> &CanReachEnd,
    SmallVectorImpl<bool> &ToRemove) {
  LLVM_DEBUG(dbgs() << "\nAnalyzing: " << *(AllInstr[i].Inst) << "\n");
  const InstructionInfo &CurrII = AllInstr[i];
  Instruction *CurrInst = CurrII.Inst;
  const BasicBlock *CurrBB = CurrInst->getParent();

  const DomTreeNode *CurrDTNode = DT.getNode(CurrBB);
  if (!CurrDTNode)
    return false;

  // A dominating access A must be in a block that is an ancestor of CurrBB in
  // the dominator tree (condition 1). Walking up from CurrBB's node to the
  // tree root visits exactly those blocks, so we only scan instrumented
  // accesses inside them. This is more efficient than scanning all of AllInstr
  // and filtering by domination after the fact.
  for (const DomTreeNode *IDomNode = CurrDTNode; IDomNode;
       IDomNode = IDomNode->getIDom()) {
    const BasicBlock *DomBB = IDomNode->getBlock();
    if (!DomBB)
      break;

    // When DomBB == CurrBB, the dominating instruction must appear before
    // CurrInst in program order; instructions after it do not dominate it.
    auto EndIt = (DomBB == CurrBB) ? CurrInst->getIterator() : DomBB->end();

    for (auto InstIt = DomBB->begin(); InstIt != EndIt; ++InstIt) {
      LLVM_DEBUG(dbgs() << "Candidate: " << *InstIt << "\n");

      const auto It = InstrToIndexMap.find(&*InstIt);
      if (It == InstrToIndexMap.end() || ToRemove[It->second])
        continue; // not an instrumented access, or already eliminated

      const size_t DomIndex = It->second;
      const InstructionInfo &DomII = AllInstr[DomIndex];
      Instruction *DomInst = DomII.Inst;

      auto IsVolatile = [](const Instruction *I) {
        if (const auto *L = dyn_cast<LoadInst>(I))
          return L->isVolatile();
        if (const auto *S = dyn_cast<StoreInst>(I))
          return S->isVolatile();
        return false;
      };
      // With -tsan-distinguish-volatile, volatile and non-volatile accesses
      // emit different runtime calls and must not be merged.
      if (ClDistinguishVolatile &&
          (IsVolatile(DomInst) || IsVolatile(CurrInst)))
        continue;

      // Condition (2): same memory location.
      // isMustAlias checks that the base pointers are identical but does NOT
      // compare access sizes — two MemoryLocations with the same pointer but
      // different sizes both return MustAlias. We therefore check sizes
      // separately: DomInst must cover at least as many bytes as CurrInst,
      // otherwise races on the extra bytes in CurrInst's range would not be
      // detected by DomInst's instrumentation call.
      const MemoryLocation CurrLoc = MemoryLocation::get(CurrInst);
      const MemoryLocation DomLoc = MemoryLocation::get(DomInst);
      if (!AA.isMustAlias(CurrLoc, DomLoc))
        continue;
      // Require both sizes to be known, non-scalable, and DomSize >= CurrSize.
      if (!DomLoc.Size.hasValue() || !CurrLoc.Size.hasValue() ||
          DomLoc.Size.isScalable() || CurrLoc.Size.isScalable() ||
          DomLoc.Size.getValue().getFixedValue() <
              CurrLoc.Size.getValue().getFixedValue())
        continue;

      // Condition (3): write coverage rule.
      // A write races with both reads and writes on other threads, so a
      // dominating write makes CurrInst's check redundant regardless of
      // whether CurrInst is a read or a write.
      // A read races only with writes, so a dominating read cannot subsume
      // a subsequent write: the write might race with a third thread's read
      // that the dominating read would not catch.
      if (!DomII.isWriteOperation() && CurrII.isWriteOperation())
        continue;

      // Condition (4): synchronization-free path (checked last because it
      // involves CFG traversal and is the most expensive test).
      if (isPathClear(DomInst, CurrInst, CanReachEnd)) {
        LLVM_DEBUG(dbgs() << "TSAN: Omitting " << *CurrInst << " (dominated by "
                          << *DomInst << ")\n");
        ToRemove[i] = true;
        return true;
      }
    }
  }
  return false;
}

void DominanceBasedElimination::eliminate() {
  LLVM_DEBUG(dbgs() << "Starting dominance-based elimination\n");
  if (AllInstr.empty())
    return;

  SmallVector<bool, 16> ToRemove(AllInstr.size(), false);
  unsigned RemovedCount = 0;
  const DenseMap<Instruction *, size_t> InstrToIndexMap =
      createInstrToIndexMap();

  for (size_t i = 0; i < AllInstr.size(); ++i) {
    if (ToRemove[i])
      continue;

    // Build the reverse-reachability set of CurrBB: the set of all blocks
    // from which CurrBB is reachable. isPathClear uses this to restrict its
    // forward BFS to blocks that actually lie on a path from DomBB to CurrBB.
    //
    // We compute it here — once per CurrInst — rather than inside isPathClear,
    // because findAndMarkDominatingInstr may call isPathClear several times
    // for different DomInst candidates that all share the same CurrBB.
    // Pre-computing avoids repeating the reverse BFS for each candidate.
    const BasicBlock *CurrBB = AllInstr[i].Inst->getParent();
    SmallPtrSet<const BasicBlock *, 32> CanReachEnd;
    SmallVector<const BasicBlock *, 16> RBFSWorklist;
    CanReachEnd.insert(CurrBB);
    RBFSWorklist.push_back(CurrBB);
    while (!RBFSWorklist.empty()) {
      const BasicBlock *BB = RBFSWorklist.pop_back_val();
      for (const BasicBlock *Pred : predecessors(BB))
        if (CanReachEnd.insert(Pred).second)
          RBFSWorklist.push_back(Pred);
    }

    if (findAndMarkDominatingInstr(i, InstrToIndexMap, CanReachEnd, ToRemove))
      RemovedCount++;
  }

  LLVM_DEBUG(dbgs() << "\nFinal instruction status:\n";
             for (size_t i = 0; i < AllInstr.size(); ++i) dbgs()
             << "[" << (ToRemove[i] ? "REMOVED" : "KEPT") << "]\t"
             << *AllInstr[i].Inst << "\n");

  if (RemovedCount > 0) {
    auto It = ToRemove.begin();
    erase_if(AllInstr, [&](const InstructionInfo &) { return *It++; });
    NumOmittedByDominance += RemovedCount;
  }
  LLVM_DEBUG(dbgs() << "Dominance elimination complete\n");
}

//-----------------------------------------------------------------------------
// ThreadSanitizer Implementation
//-----------------------------------------------------------------------------

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kTsanModuleCtorName, kTsanInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}
}  // namespace

PreservedAnalyses ThreadSanitizerPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  DominatorTree *DT = nullptr;
  AAResults *AA = nullptr;

  if (ClUseDominanceAnalysis) {
    DT = &FAM.getResult<DominatorTreeAnalysis>(F);
    AA = &FAM.getResult<AAManager>(F);
  }

  ThreadSanitizer TSan(DT, AA);
  if (TSan.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleThreadSanitizerPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  // Return early if nosanitize_thread module flag is present for the module.
  if (checkIfAlreadyInstrumented(M, "nosanitize_thread"))
    return PreservedAnalyses::all();
  insertModuleCtor(M);
  return PreservedAnalyses::none();
}
void ThreadSanitizer::initialize(Module &M, const TargetLibraryInfo &TLI) {
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &Ctx = M.getContext();
  IntptrTy = DL.getIntPtrType(Ctx);

  IRBuilder<> IRB(Ctx);
  AttributeList Attr;
  Attr = Attr.addFnAttribute(Ctx, Attribute::NoUnwind);
  // Initialize the callbacks.
  TsanFuncEntry = M.getOrInsertFunction("__tsan_func_entry", Attr,
                                        IRB.getVoidTy(), IRB.getPtrTy());
  TsanFuncExit =
      M.getOrInsertFunction("__tsan_func_exit", Attr, IRB.getVoidTy());
  TsanIgnoreBegin = M.getOrInsertFunction("__tsan_ignore_thread_begin", Attr,
                                          IRB.getVoidTy());
  TsanIgnoreEnd =
      M.getOrInsertFunction("__tsan_ignore_thread_end", Attr, IRB.getVoidTy());
  IntegerType *OrdTy = IRB.getInt32Ty();
  for (size_t i = 0; i < kNumberOfAccessSizes; ++i) {
    const unsigned ByteSize = 1U << i;
    const unsigned BitSize = ByteSize * 8;
    std::string ByteSizeStr = utostr(ByteSize);
    std::string BitSizeStr = utostr(BitSize);
    SmallString<32> ReadName("__tsan_read" + ByteSizeStr);
    TsanRead[i] = M.getOrInsertFunction(ReadName, Attr, IRB.getVoidTy(),
                                        IRB.getPtrTy());

    SmallString<32> WriteName("__tsan_write" + ByteSizeStr);
    TsanWrite[i] = M.getOrInsertFunction(WriteName, Attr, IRB.getVoidTy(),
                                         IRB.getPtrTy());

    SmallString<64> UnalignedReadName("__tsan_unaligned_read" + ByteSizeStr);
    TsanUnalignedRead[i] = M.getOrInsertFunction(
        UnalignedReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedWriteName("__tsan_unaligned_write" + ByteSizeStr);
    TsanUnalignedWrite[i] = M.getOrInsertFunction(
        UnalignedWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileReadName("__tsan_volatile_read" + ByteSizeStr);
    TsanVolatileRead[i] = M.getOrInsertFunction(
        VolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> VolatileWriteName("__tsan_volatile_write" + ByteSizeStr);
    TsanVolatileWrite[i] = M.getOrInsertFunction(
        VolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileReadName("__tsan_unaligned_volatile_read" +
                                              ByteSizeStr);
    TsanUnalignedVolatileRead[i] = M.getOrInsertFunction(
        UnalignedVolatileReadName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedVolatileWriteName(
        "__tsan_unaligned_volatile_write" + ByteSizeStr);
    TsanUnalignedVolatileWrite[i] = M.getOrInsertFunction(
        UnalignedVolatileWriteName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> CompoundRWName("__tsan_read_write" + ByteSizeStr);
    TsanCompoundRW[i] = M.getOrInsertFunction(
        CompoundRWName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    SmallString<64> UnalignedCompoundRWName("__tsan_unaligned_read_write" +
                                            ByteSizeStr);
    TsanUnalignedCompoundRW[i] = M.getOrInsertFunction(
        UnalignedCompoundRWName, Attr, IRB.getVoidTy(), IRB.getPtrTy());

    Type *Ty = Type::getIntNTy(Ctx, BitSize);
    Type *PtrTy = PointerType::get(Ctx, 0);
    SmallString<32> AtomicLoadName("__tsan_atomic" + BitSizeStr + "_load");
    TsanAtomicLoad[i] =
        M.getOrInsertFunction(AtomicLoadName,
                              TLI.getAttrList(&Ctx, {1}, /*Signed=*/true,
                                              /*Ret=*/BitSize <= 32, Attr),
                              Ty, PtrTy, OrdTy);

    // Args of type Ty need extension only when BitSize is 32 or less.
    using Idxs = std::vector<unsigned>;
    Idxs Idxs2Or12   ((BitSize <= 32) ? Idxs({1, 2})       : Idxs({2}));
    Idxs Idxs34Or1234((BitSize <= 32) ? Idxs({1, 2, 3, 4}) : Idxs({3, 4}));
    SmallString<32> AtomicStoreName("__tsan_atomic" + BitSizeStr + "_store");
    TsanAtomicStore[i] = M.getOrInsertFunction(
        AtomicStoreName,
        TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true, /*Ret=*/false, Attr),
        IRB.getVoidTy(), PtrTy, Ty, OrdTy);

    for (unsigned Op = AtomicRMWInst::FIRST_BINOP;
         Op <= AtomicRMWInst::LAST_BINOP; ++Op) {
      TsanAtomicRMW[Op][i] = nullptr;
      const char *NamePart = nullptr;
      if (Op == AtomicRMWInst::Xchg)
        NamePart = "_exchange";
      else if (Op == AtomicRMWInst::Add)
        NamePart = "_fetch_add";
      else if (Op == AtomicRMWInst::Sub)
        NamePart = "_fetch_sub";
      else if (Op == AtomicRMWInst::And)
        NamePart = "_fetch_and";
      else if (Op == AtomicRMWInst::Or)
        NamePart = "_fetch_or";
      else if (Op == AtomicRMWInst::Xor)
        NamePart = "_fetch_xor";
      else if (Op == AtomicRMWInst::Nand)
        NamePart = "_fetch_nand";
      else
        continue;
      SmallString<32> RMWName("__tsan_atomic" + itostr(BitSize) + NamePart);
      TsanAtomicRMW[Op][i] = M.getOrInsertFunction(
          RMWName,
          TLI.getAttrList(&Ctx, Idxs2Or12, /*Signed=*/true,
                          /*Ret=*/BitSize <= 32, Attr),
          Ty, PtrTy, Ty, OrdTy);
    }

    SmallString<32> AtomicCASName("__tsan_atomic" + BitSizeStr +
                                  "_compare_exchange_val");
    TsanAtomicCAS[i] = M.getOrInsertFunction(
        AtomicCASName,
        TLI.getAttrList(&Ctx, Idxs34Or1234, /*Signed=*/true,
                        /*Ret=*/BitSize <= 32, Attr),
        Ty, PtrTy, Ty, Ty, OrdTy, OrdTy);
  }
  TsanVptrUpdate =
      M.getOrInsertFunction("__tsan_vptr_update", Attr, IRB.getVoidTy(),
                            IRB.getPtrTy(), IRB.getPtrTy());
  TsanVptrLoad = M.getOrInsertFunction("__tsan_vptr_read", Attr,
                                       IRB.getVoidTy(), IRB.getPtrTy());
  TsanAtomicThreadFence = M.getOrInsertFunction(
      "__tsan_atomic_thread_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  TsanAtomicSignalFence = M.getOrInsertFunction(
      "__tsan_atomic_signal_fence",
      TLI.getAttrList(&Ctx, {0}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getVoidTy(), OrdTy);

  MemmoveFn =
      M.getOrInsertFunction("__tsan_memmove", Attr, IRB.getPtrTy(),
                            IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemcpyFn =
      M.getOrInsertFunction("__tsan_memcpy", Attr, IRB.getPtrTy(),
                            IRB.getPtrTy(), IRB.getPtrTy(), IntptrTy);
  MemsetFn = M.getOrInsertFunction(
      "__tsan_memset",
      TLI.getAttrList(&Ctx, {1}, /*Signed=*/true, /*Ret=*/false, Attr),
      IRB.getPtrTy(), IRB.getPtrTy(), IRB.getInt32Ty(), IntptrTy);
}

static bool isVtableAccess(Instruction *I) {
  if (MDNode *Tag = I->getMetadata(LLVMContext::MD_tbaa))
    return Tag->isTBAAVtableAccess();
  return false;
}

// Do not instrument known races/"benign races" that come from compiler
// instrumentation. The user has no way of suppressing them.
static bool shouldInstrumentReadWriteFromAddress(const Module *M, Value *Addr) {
  // Peel off GEPs and BitCasts.
  // Note: This also peels AddrspaceCasts, so this should not be used when
  // checking the address space below.
  Value *PeeledAddr = Addr->stripInBoundsOffsets();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(PeeledAddr)) {
    if (GV->hasSection()) {
      StringRef SectionName = GV->getSection();
      // Check if the global is in the PGO counters section.
      auto OF = M->getTargetTriple().getObjectFormat();
      if (SectionName.ends_with(
              getInstrProfSectionName(IPSK_cnts, OF, /*AddSegmentInfo=*/false)))
        return false;
    }
  }

  // Do not instrument accesses from different address spaces; we cannot deal
  // with them.
  Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() != 0)
    return false;

  return true;
}

bool ThreadSanitizer::addrPointsToConstantData(Value *Addr) {
  // If this is a GEP, just analyze its pointer operand.
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr))
    Addr = GEP->getPointerOperand();

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Addr)) {
    if (GV->isConstant()) {
      // Reads from constant globals can not race with any writes.
      NumOmittedReadsFromConstantGlobals++;
      return true;
    }
  } else if (LoadInst *L = dyn_cast<LoadInst>(Addr)) {
    if (isVtableAccess(L)) {
      // Reads from a vtable pointer can not race with any writes.
      NumOmittedReadsFromVtable++;
      return true;
    }
  }
  return false;
}

// Instrumenting some of the accesses may be proven redundant.
// Currently handled:
//  - read-before-write (within same BB, no calls between)
//  - not captured variables
//
// We do not handle some of the patterns that should not survive
// after the classic compiler optimizations.
// E.g. two reads from the same temp should be eliminated by CSE,
// two writes should be eliminated by DSE, etc.
//
// 'Local' is a vector of insns within the same BB (no calls between).
// 'All' is a vector of insns that will be instrumented.
void ThreadSanitizer::chooseInstructionsToInstrument(
    SmallVectorImpl<Instruction *> &Local,
    SmallVectorImpl<InstructionInfo> &All, const DataLayout &DL) {
  DenseMap<Value *, size_t> WriteTargets; // Map of addresses to index in All
  // Iterate from the end.
  for (Instruction *I : reverse(Local)) {
    const bool IsWrite = isa<StoreInst>(*I);
    Value *Addr = IsWrite ? cast<StoreInst>(I)->getPointerOperand()
                          : cast<LoadInst>(I)->getPointerOperand();

    if (!shouldInstrumentReadWriteFromAddress(I->getModule(), Addr))
      continue;

    if (!IsWrite) {
      const auto WriteEntry = WriteTargets.find(Addr);
      if (!ClInstrumentReadBeforeWrite && WriteEntry != WriteTargets.end()) {
        auto &WI = All[WriteEntry->second];
        // If we distinguish volatile accesses and if either the read or write
        // is volatile, do not omit any instrumentation.
        const bool AnyVolatile =
            ClDistinguishVolatile && (cast<LoadInst>(I)->isVolatile() ||
                                      cast<StoreInst>(WI.Inst)->isVolatile());
        if (!AnyVolatile) {
          // We will write to this temp, so no reason to analyze the read.
          // Mark the write instruction as compound.
          WI.Flags |= InstructionInfo::kCompoundRW;
          NumOmittedReadsBeforeWrite++;
          continue;
        }
      }

      if (addrPointsToConstantData(Addr)) {
        // Addr points to some constant data -- it can not race with any writes.
        continue;
      }
    }

    const AllocaInst *AI = findAllocaForValue(Addr);
    // Instead of Addr, we should check whether its base pointer is captured.
    if (AI && !PointerMayBeCaptured(AI, /*ReturnCaptures=*/true) &&
        ClOmitNonCaptured) {
      // The variable is addressable but not captured, so it cannot be
      // referenced from a different thread and participate in a data race
      // (see llvm/Analysis/CaptureTracking.h for details).
      NumOmittedNonCaptured++;
      continue;
    }

    // Instrument this instruction.
    All.emplace_back(I);
    if (IsWrite) {
      // For read-before-write and compound instrumentation we only need one
      // write target, and we can override any previous entry if it exists.
      WriteTargets[Addr] = All.size() - 1;
    }
  }
  Local.clear();
}

void ThreadSanitizer::InsertRuntimeIgnores(Function &F) {
  InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                               F.getEntryBlock().getFirstNonPHIIt());
  IRB.CreateCall(TsanIgnoreBegin);
  EscapeEnumerator EE(F, "tsan_ignore_cleanup", ClHandleCxxExceptions);
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(TsanIgnoreEnd);
  }
}

bool ThreadSanitizer::sanitizeFunction(Function &F,
                                       const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __tsan_init from within
  // the module constructor.
  if (F.getName() == kTsanModuleCtorName)
    return false;
  // Naked functions can not have prologue/epilogue
  // (__tsan_func_entry/__tsan_func_exit) generated, so don't instrument them at
  // all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  // __attribute__(disable_sanitizer_instrumentation) prevents all kinds of
  // instrumentation.
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return false;

  initialize(*F.getParent(), TLI);
  SmallVector<InstructionInfo, 8> AllLoadsAndStores;
  SmallVector<Instruction*, 8> LocalLoadsAndStores;
  SmallVector<Instruction*, 8> AtomicAccesses;
  SmallVector<Instruction*, 8> MemIntrinCalls;
  bool Res = false;
  bool HasCalls = false;
  bool SanitizeFunction = F.hasFnAttribute(Attribute::SanitizeThread);
  const DataLayout &DL = F.getDataLayout();

  // Traverse all instructions, collect loads/stores/returns, check for calls.
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      // Skip instructions inserted by another instrumentation.
      if (Inst.hasMetadata(LLVMContext::MD_nosanitize))
        continue;
      if (isTsanAtomic(&Inst))
        AtomicAccesses.push_back(&Inst);
      else if (isa<LoadInst>(Inst) || isa<StoreInst>(Inst))
        LocalLoadsAndStores.push_back(&Inst);
      else if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst)) {
        if (CallInst *CI = dyn_cast<CallInst>(&Inst))
          maybeMarkSanitizerLibraryCallNoBuiltin(CI, &TLI);
        if (isa<MemIntrinsic>(Inst))
          MemIntrinCalls.push_back(&Inst);
        HasCalls = true;
        chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores,
                                       DL);
      }
    }
    chooseInstructionsToInstrument(LocalLoadsAndStores, AllLoadsAndStores, DL);
  }

  if (ClUseDominanceAnalysis && DT && AA) {
    DominanceBasedElimination DBE(AllLoadsAndStores, *DT, *AA);
    DBE.run();
  }

  // We have collected all loads and stores.
  // FIXME: many of these accesses do not need to be checked for races
  // (e.g. variables that do not escape, etc).

  // Instrument memory accesses only if we want to report bugs in the function.
  if (ClInstrumentMemoryAccesses && SanitizeFunction)
    for (const auto &II : AllLoadsAndStores) {
      Res |= instrumentLoadOrStore(II, DL);
    }

  // Instrument atomic memory accesses in any case (they can be used to
  // implement synchronization).
  if (ClInstrumentAtomics)
    for (auto *Inst : AtomicAccesses) {
      Res |= instrumentAtomic(Inst, DL);
    }

  if (ClInstrumentMemIntrinsics && SanitizeFunction)
    for (auto *Inst : MemIntrinCalls) {
      Res |= instrumentMemIntrinsic(Inst);
    }

  if (F.hasFnAttribute("sanitize_thread_no_checking_at_run_time")) {
    assert(!F.hasFnAttribute(Attribute::SanitizeThread));
    if (HasCalls)
      InsertRuntimeIgnores(F);
  }

  // Instrument function entry/exit points if there were instrumented accesses.
  if ((Res || HasCalls) && ClInstrumentFuncEntryExit) {
    InstrumentationIRBuilder IRB(&F.getEntryBlock(),
                                 F.getEntryBlock().getFirstNonPHIIt());
    auto ProgramAsPtrTy = PointerType::get(F.getParent()->getContext(),
                                           DL.getProgramAddressSpace());
    Value *ReturnAddress = IRB.CreateIntrinsic(
        Intrinsic::returnaddress, {ProgramAsPtrTy}, IRB.getInt32(0));
    IRB.CreateCall(TsanFuncEntry, ReturnAddress);

    EscapeEnumerator EE(F, "tsan_cleanup", ClHandleCxxExceptions);
    while (IRBuilder<> *AtExit = EE.Next()) {
      InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
      AtExit->CreateCall(TsanFuncExit, {});
    }
    Res = true;
  }
  return Res;
}

bool ThreadSanitizer::instrumentLoadOrStore(const InstructionInfo &II,
                                            const DataLayout &DL) {
  InstrumentationIRBuilder IRB(II.Inst);
  const bool IsWrite = isa<StoreInst>(*II.Inst);
  Value *Addr = IsWrite ? cast<StoreInst>(II.Inst)->getPointerOperand()
                        : cast<LoadInst>(II.Inst)->getPointerOperand();
  Type *OrigTy = getLoadStoreType(II.Inst);

  // swifterror memory addresses are mem2reg promoted by instruction selection.
  // As such they cannot have regular uses like an instrumentation function and
  // it makes no sense to track them as memory.
  if (Addr->isSwiftError())
    return false;

  int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
  if (Idx < 0)
    return false;
  if (IsWrite && isVtableAccess(II.Inst)) {
    LLVM_DEBUG(dbgs() << "  VPTR : " << *II.Inst << "\n");
    Value *StoredValue = cast<StoreInst>(II.Inst)->getValueOperand();
    // StoredValue may be a vector type if we are storing several vptrs at once.
    // In this case, just take the first element of the vector since this is
    // enough to find vptr races.
    if (isa<VectorType>(StoredValue->getType()))
      StoredValue = IRB.CreateExtractElement(
          StoredValue, ConstantInt::get(IRB.getInt32Ty(), 0));
    if (StoredValue->getType()->isIntegerTy())
      StoredValue = IRB.CreateIntToPtr(StoredValue, IRB.getPtrTy());
    // Call TsanVptrUpdate.
    IRB.CreateCall(TsanVptrUpdate, {Addr, StoredValue});
    NumInstrumentedVtableWrites++;
    return true;
  }
  if (!IsWrite && isVtableAccess(II.Inst)) {
    IRB.CreateCall(TsanVptrLoad, Addr);
    NumInstrumentedVtableReads++;
    return true;
  }

  const Align Alignment = IsWrite ? cast<StoreInst>(II.Inst)->getAlign()
                                  : cast<LoadInst>(II.Inst)->getAlign();
  const bool IsCompoundRW =
      ClCompoundReadBeforeWrite && (II.Flags & InstructionInfo::kCompoundRW);
  const bool IsVolatile = ClDistinguishVolatile &&
                          (IsWrite ? cast<StoreInst>(II.Inst)->isVolatile()
                                   : cast<LoadInst>(II.Inst)->isVolatile());
  assert((!IsVolatile || !IsCompoundRW) && "Compound volatile invalid!");

  const uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  FunctionCallee OnAccessFunc = nullptr;
  if (Alignment >= Align(8) || (Alignment.value() % (TypeSize / 8)) == 0) {
    if (IsCompoundRW)
      OnAccessFunc = TsanCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanVolatileWrite[Idx] : TsanVolatileRead[Idx];
    else
      OnAccessFunc = IsWrite ? TsanWrite[Idx] : TsanRead[Idx];
  } else {
    if (IsCompoundRW)
      OnAccessFunc = TsanUnalignedCompoundRW[Idx];
    else if (IsVolatile)
      OnAccessFunc = IsWrite ? TsanUnalignedVolatileWrite[Idx]
                             : TsanUnalignedVolatileRead[Idx];
    else
      OnAccessFunc = IsWrite ? TsanUnalignedWrite[Idx] : TsanUnalignedRead[Idx];
  }
  IRB.CreateCall(OnAccessFunc, Addr);
  if (IsCompoundRW || IsWrite)
    NumInstrumentedWrites++;
  if (IsCompoundRW || !IsWrite)
    NumInstrumentedReads++;
  return true;
}

static ConstantInt *createOrdering(IRBuilder<> *IRB, AtomicOrdering ord) {
  uint32_t v = 0;
  switch (ord) {
    case AtomicOrdering::NotAtomic:
      llvm_unreachable("unexpected atomic ordering!");
    case AtomicOrdering::Unordered:              [[fallthrough]];
    case AtomicOrdering::Monotonic:              v = 0; break;
    // Not specified yet:
    // case AtomicOrdering::Consume:                v = 1; break;
    case AtomicOrdering::Acquire:                v = 2; break;
    case AtomicOrdering::Release:                v = 3; break;
    case AtomicOrdering::AcquireRelease:         v = 4; break;
    case AtomicOrdering::SequentiallyConsistent: v = 5; break;
  }
  return IRB->getInt32(v);
}

// If a memset intrinsic gets inlined by the code gen, we will miss races on it.
// So, we either need to ensure the intrinsic is not inlined, or instrument it.
// We do not instrument memset/memmove/memcpy intrinsics (too complicated),
// instead we simply replace them with regular function calls, which are then
// intercepted by the run-time.
// Since tsan is running after everyone else, the calls should not be
// replaced back with intrinsics. If that becomes wrong at some point,
// we will need to call e.g. __tsan_memset to avoid the intrinsics.
bool ThreadSanitizer::instrumentMemIntrinsic(Instruction *I) {
  InstrumentationIRBuilder IRB(I);
  if (MemSetInst *M = dyn_cast<MemSetInst>(I)) {
    Value *Cast1 = IRB.CreateIntCast(M->getArgOperand(1), IRB.getInt32Ty(), false);
    Value *Cast2 = IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false);
    IRB.CreateCall(
        MemsetFn,
        {M->getArgOperand(0),
         Cast1,
         Cast2});
    I->eraseFromParent();
  } else if (MemTransferInst *M = dyn_cast<MemTransferInst>(I)) {
    IRB.CreateCall(
        isa<MemCpyInst>(M) ? MemcpyFn : MemmoveFn,
        {M->getArgOperand(0),
         M->getArgOperand(1),
         IRB.CreateIntCast(M->getArgOperand(2), IntptrTy, false)});
    I->eraseFromParent();
  }
  return false;
}

// Both llvm and ThreadSanitizer atomic operations are based on C++11/C1x
// standards.  For background see C++11 standard.  A slightly older, publicly
// available draft of the standard (not entirely up-to-date, but close enough
// for casual browsing) is available here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3242.pdf
// The following page contains more background information:
// http://www.hpl.hp.com/personal/Hans_Boehm/c++mm/

bool ThreadSanitizer::instrumentAtomic(Instruction *I, const DataLayout &DL) {
  InstrumentationIRBuilder IRB(I);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Addr = LI->getPointerOperand();
    Type *OrigTy = LI->getType();
    int Idx = getMemoryAccessFuncIndex(OrigTy, Addr, DL);
    if (Idx < 0)
      return false;
    Value *Args[] = {Addr,
                     createOrdering(&IRB, LI->getOrdering())};
    Value *C = IRB.CreateCall(TsanAtomicLoad[Idx], Args);
    Value *Cast = IRB.CreateBitOrPointerCast(C, OrigTy);
    I->replaceAllUsesWith(Cast);
    I->eraseFromParent();
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Value *Addr = SI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(SI->getValueOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Args[] = {Addr,
                     IRB.CreateBitOrPointerCast(SI->getValueOperand(), Ty),
                     createOrdering(&IRB, SI->getOrdering())};
    IRB.CreateCall(TsanAtomicStore[Idx], Args);
    SI->eraseFromParent();
  } else if (AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    Value *Addr = RMWI->getPointerOperand();
    int Idx =
        getMemoryAccessFuncIndex(RMWI->getValOperand()->getType(), Addr, DL);
    if (Idx < 0)
      return false;
    FunctionCallee F = TsanAtomicRMW[RMWI->getOperation()][Idx];
    if (!F)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *Val = RMWI->getValOperand();
    Value *Args[] = {Addr, IRB.CreateBitOrPointerCast(Val, Ty),
                     createOrdering(&IRB, RMWI->getOrdering())};
    Value *C = IRB.CreateCall(F, Args);
    I->replaceAllUsesWith(IRB.CreateBitOrPointerCast(C, Val->getType()));
    I->eraseFromParent();
  } else if (AtomicCmpXchgInst *CASI = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Addr = CASI->getPointerOperand();
    Type *OrigOldValTy = CASI->getNewValOperand()->getType();
    int Idx = getMemoryAccessFuncIndex(OrigOldValTy, Addr, DL);
    if (Idx < 0)
      return false;
    const unsigned ByteSize = 1U << Idx;
    const unsigned BitSize = ByteSize * 8;
    Type *Ty = Type::getIntNTy(IRB.getContext(), BitSize);
    Value *CmpOperand =
      IRB.CreateBitOrPointerCast(CASI->getCompareOperand(), Ty);
    Value *NewOperand =
      IRB.CreateBitOrPointerCast(CASI->getNewValOperand(), Ty);
    Value *Args[] = {Addr,
                     CmpOperand,
                     NewOperand,
                     createOrdering(&IRB, CASI->getSuccessOrdering()),
                     createOrdering(&IRB, CASI->getFailureOrdering())};
    CallInst *C = IRB.CreateCall(TsanAtomicCAS[Idx], Args);
    Value *Success = IRB.CreateICmpEQ(C, CmpOperand);
    Value *OldVal = C;
    if (Ty != OrigOldValTy) {
      // The value is a pointer, so we need to cast the return value.
      OldVal = IRB.CreateIntToPtr(C, OrigOldValTy);
    }

    Value *Res =
      IRB.CreateInsertValue(PoisonValue::get(CASI->getType()), OldVal, 0);
    Res = IRB.CreateInsertValue(Res, Success, 1);

    I->replaceAllUsesWith(Res);
    I->eraseFromParent();
  } else if (FenceInst *FI = dyn_cast<FenceInst>(I)) {
    Value *Args[] = {createOrdering(&IRB, FI->getOrdering())};
    FunctionCallee F = FI->getSyncScopeID() == SyncScope::SingleThread
                           ? TsanAtomicSignalFence
                           : TsanAtomicThreadFence;
    IRB.CreateCall(F, Args);
    FI->eraseFromParent();
  }
  return true;
}

int ThreadSanitizer::getMemoryAccessFuncIndex(Type *OrigTy, Value *Addr,
                                              const DataLayout &DL) {
  assert(OrigTy->isSized());
  if (OrigTy->isScalableTy()) {
    // FIXME: support vscale.
    return -1;
  }
  uint32_t TypeSize = DL.getTypeStoreSizeInBits(OrigTy);
  if (TypeSize != 8  && TypeSize != 16 &&
      TypeSize != 32 && TypeSize != 64 && TypeSize != 128) {
    NumAccessesWithBadSize++;
    // Ignore all unusual sizes.
    return -1;
  }
  size_t Idx = llvm::countr_zero(TypeSize / 8);
  assert(Idx < kNumberOfAccessSizes);
  return Idx;
}
