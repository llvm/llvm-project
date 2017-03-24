//===-- SanitizerCoverage.cpp - coverage instrumentation for sanitizers ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Coverage instrumentation that works with AddressSanitizer
// and potentially with other Sanitizers.
//
// We create a Guard variable with the same linkage
// as the function and inject this code into the entry block (SCK_Function)
// or all blocks (SCK_BB):
// if (Guard < 0) {
//    __sanitizer_cov(&Guard);
// }
// The accesses to Guard are atomic. The rest of the logic is
// in __sanitizer_cov (it's fine to call it more than once).
//
// With SCK_Edge we also split critical edges this effectively
// instrumenting all edges.
//
// This coverage implementation provides very limited data:
// it only tells if a given function (block) was ever executed. No counters.
// But for many use cases this is what we need and the added slowdown small.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "sancov"

static const char *const SanCovModuleInitName = "__sanitizer_cov_module_init";
static const char *const SanCovName = "__sanitizer_cov";
static const char *const SanCovWithCheckName = "__sanitizer_cov_with_check";
static const char *const SanCovIndirCallName = "__sanitizer_cov_indir_call16";
static const char *const SanCovTracePCIndirName =
    "__sanitizer_cov_trace_pc_indir";
static const char *const SanCovTraceEnterName =
    "__sanitizer_cov_trace_func_enter";
static const char *const SanCovTraceBBName =
    "__sanitizer_cov_trace_basic_block";
static const char *const SanCovTracePCName = "__sanitizer_cov_trace_pc";
static const char *const SanCovTraceCmp1 = "__sanitizer_cov_trace_cmp1";
static const char *const SanCovTraceCmp2 = "__sanitizer_cov_trace_cmp2";
static const char *const SanCovTraceCmp4 = "__sanitizer_cov_trace_cmp4";
static const char *const SanCovTraceCmp8 = "__sanitizer_cov_trace_cmp8";
static const char *const SanCovTraceDiv4 = "__sanitizer_cov_trace_div4";
static const char *const SanCovTraceDiv8 = "__sanitizer_cov_trace_div8";
static const char *const SanCovTraceGep = "__sanitizer_cov_trace_gep";
static const char *const SanCovTraceSwitchName = "__sanitizer_cov_trace_switch";
static const char *const SanCovModuleCtorName = "sancov.module_ctor";
static const uint64_t SanCtorAndDtorPriority = 2;

static const char *const SanCovTracePCGuardName =
    "__sanitizer_cov_trace_pc_guard";
static const char *const SanCovTracePCGuardInitName =
    "__sanitizer_cov_trace_pc_guard_init";

static cl::opt<int> ClCoverageLevel(
    "sanitizer-coverage-level",
    cl::desc("Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, "
             "3: all blocks and critical edges, "
             "4: above plus indirect calls"),
    cl::Hidden, cl::init(0));

static cl::opt<unsigned> ClCoverageBlockThreshold(
    "sanitizer-coverage-block-threshold",
    cl::desc("Use a callback with a guard check inside it if there are"
             " more than this number of blocks."),
    cl::Hidden, cl::init(0));

static cl::opt<bool>
    ClExperimentalTracing("sanitizer-coverage-experimental-tracing",
                          cl::desc("Experimental basic-block tracing: insert "
                                   "callbacks at every basic block"),
                          cl::Hidden, cl::init(false));

static cl::opt<bool> ClExperimentalTracePC("sanitizer-coverage-trace-pc",
                                           cl::desc("Experimental pc tracing"),
                                           cl::Hidden, cl::init(false));

static cl::opt<bool> ClTracePCGuard("sanitizer-coverage-trace-pc-guard",
                                    cl::desc("pc tracing with a guard"),
                                    cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClCMPTracing("sanitizer-coverage-trace-compares",
                 cl::desc("Tracing of CMP and similar instructions"),
                 cl::Hidden, cl::init(false));

static cl::opt<bool> ClDIVTracing("sanitizer-coverage-trace-divs",
                                  cl::desc("Tracing of DIV instructions"),
                                  cl::Hidden, cl::init(false));

static cl::opt<bool> ClGEPTracing("sanitizer-coverage-trace-geps",
                                  cl::desc("Tracing of GEP instructions"),
                                  cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClPruneBlocks("sanitizer-coverage-prune-blocks",
                  cl::desc("Reduce the number of instrumented blocks"),
                  cl::Hidden, cl::init(true));

// Experimental 8-bit counters used as an additional search heuristic during
// coverage-guided fuzzing.
// The counters are not thread-friendly:
//   - contention on these counters may cause significant slowdown;
//   - the counter updates are racy and the results may be inaccurate.
// They are also inaccurate due to 8-bit integer overflow.
static cl::opt<bool> ClUse8bitCounters("sanitizer-coverage-8bit-counters",
                                       cl::desc("Experimental 8-bit counters"),
                                       cl::Hidden, cl::init(false));

namespace {

SanitizerCoverageOptions getOptions(int LegacyCoverageLevel) {
  SanitizerCoverageOptions Res;
  switch (LegacyCoverageLevel) {
  case 0:
    Res.CoverageType = SanitizerCoverageOptions::SCK_None;
    break;
  case 1:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Function;
    break;
  case 2:
    Res.CoverageType = SanitizerCoverageOptions::SCK_BB;
    break;
  case 3:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Edge;
    break;
  case 4:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Edge;
    Res.IndirectCalls = true;
    break;
  }
  return Res;
}

SanitizerCoverageOptions OverrideFromCL(SanitizerCoverageOptions Options) {
  // Sets CoverageType and IndirectCalls.
  SanitizerCoverageOptions CLOpts = getOptions(ClCoverageLevel);
  Options.CoverageType = std::max(Options.CoverageType, CLOpts.CoverageType);
  Options.IndirectCalls |= CLOpts.IndirectCalls;
  Options.TraceBB |= ClExperimentalTracing;
  Options.TraceCmp |= ClCMPTracing;
  Options.TraceDiv |= ClDIVTracing;
  Options.TraceGep |= ClGEPTracing;
  Options.Use8bitCounters |= ClUse8bitCounters;
  Options.TracePC |= ClExperimentalTracePC;
  Options.TracePCGuard |= ClTracePCGuard;
  return Options;
}

class SanitizerCoverageModule : public ModulePass {
public:
  SanitizerCoverageModule(
      const SanitizerCoverageOptions &Options = SanitizerCoverageOptions())
      : ModulePass(ID), Options(OverrideFromCL(Options)) {
    initializeSanitizerCoverageModulePass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);
  static char ID; // Pass identification, replacement for typeid
  StringRef getPassName() const override { return "SanitizerCoverageModule"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
  }

private:
  void InjectCoverageForIndirectCalls(Function &F,
                                      ArrayRef<Instruction *> IndirCalls);
  void InjectTraceForCmp(Function &F, ArrayRef<Instruction *> CmpTraceTargets);
  void InjectTraceForDiv(Function &F,
                         ArrayRef<BinaryOperator *> DivTraceTargets);
  void InjectTraceForGep(Function &F,
                         ArrayRef<GetElementPtrInst *> GepTraceTargets);
  void InjectTraceForSwitch(Function &F,
                            ArrayRef<Instruction *> SwitchTraceTargets);
  bool InjectCoverage(Function &F, ArrayRef<BasicBlock *> AllBlocks);
  void CreateFunctionGuardArray(size_t NumGuards, Function &F);
  void SetNoSanitizeMetadata(Instruction *I);
  void InjectCoverageAtBlock(Function &F, BasicBlock &BB, size_t Idx,
                             bool UseCalls);
  unsigned NumberOfInstrumentedBlocks() {
    return SanCovFunction->getNumUses() +
           SanCovWithCheckFunction->getNumUses() + SanCovTraceBB->getNumUses() +
           SanCovTraceEnter->getNumUses();
  }
  StringRef getSanCovTracePCGuardSection() const;
  StringRef getSanCovTracePCGuardSectionStart() const;
  StringRef getSanCovTracePCGuardSectionEnd() const;
  Function *SanCovFunction;
  Function *SanCovWithCheckFunction;
  Function *SanCovIndirCallFunction, *SanCovTracePCIndir;
  Function *SanCovTraceEnter, *SanCovTraceBB, *SanCovTracePC, *SanCovTracePCGuard;
  Function *SanCovTraceCmpFunction[4];
  Function *SanCovTraceDivFunction[2];
  Function *SanCovTraceGepFunction;
  Function *SanCovTraceSwitchFunction;
  InlineAsm *EmptyAsm;
  Type *IntptrTy, *IntptrPtrTy, *Int64Ty, *Int64PtrTy, *Int32Ty, *Int32PtrTy;
  Module *CurModule;
  Triple TargetTriple;
  LLVMContext *C;
  const DataLayout *DL;

  GlobalVariable *GuardArray;
  GlobalVariable *FunctionGuardArray;  // for trace-pc-guard.
  GlobalVariable *EightBitCounterArray;
  bool HasSancovGuardsSection;

  SanitizerCoverageOptions Options;
};

} // namespace

bool SanitizerCoverageModule::runOnModule(Module &M) {
  if (Options.CoverageType == SanitizerCoverageOptions::SCK_None)
    return false;
  C = &(M.getContext());
  DL = &M.getDataLayout();
  CurModule = &M;
  TargetTriple = Triple(M.getTargetTriple());
  HasSancovGuardsSection = false;
  IntptrTy = Type::getIntNTy(*C, DL->getPointerSizeInBits());
  IntptrPtrTy = PointerType::getUnqual(IntptrTy);
  Type *VoidTy = Type::getVoidTy(*C);
  IRBuilder<> IRB(*C);
  Type *Int8PtrTy = PointerType::getUnqual(IRB.getInt8Ty());
  Int64PtrTy = PointerType::getUnqual(IRB.getInt64Ty());
  Int32PtrTy = PointerType::getUnqual(IRB.getInt32Ty());
  Int64Ty = IRB.getInt64Ty();
  Int32Ty = IRB.getInt32Ty();

  SanCovFunction = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovName, VoidTy, Int32PtrTy, nullptr));
  SanCovWithCheckFunction = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovWithCheckName, VoidTy, Int32PtrTy, nullptr));
  SanCovTracePCIndir = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTracePCIndirName, VoidTy, IntptrTy, nullptr));
  SanCovIndirCallFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovIndirCallName, VoidTy, IntptrTy, IntptrTy, nullptr));
  SanCovTraceCmpFunction[0] =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceCmp1, VoidTy, IRB.getInt8Ty(), IRB.getInt8Ty(), nullptr));
  SanCovTraceCmpFunction[1] = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTraceCmp2, VoidTy, IRB.getInt16Ty(),
                            IRB.getInt16Ty(), nullptr));
  SanCovTraceCmpFunction[2] = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTraceCmp4, VoidTy, IRB.getInt32Ty(),
                            IRB.getInt32Ty(), nullptr));
  SanCovTraceCmpFunction[3] =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceCmp8, VoidTy, Int64Ty, Int64Ty, nullptr));

  SanCovTraceDivFunction[0] =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceDiv4, VoidTy, IRB.getInt32Ty(), nullptr));
  SanCovTraceDivFunction[1] =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceDiv8, VoidTy, Int64Ty, nullptr));
  SanCovTraceGepFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceGep, VoidTy, IntptrTy, nullptr));
  SanCovTraceSwitchFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          SanCovTraceSwitchName, VoidTy, Int64Ty, Int64PtrTy, nullptr));

  // We insert an empty inline asm after cov callbacks to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);

  SanCovTracePC = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTracePCName, VoidTy, nullptr));
  SanCovTracePCGuard = checkSanitizerInterfaceFunction(M.getOrInsertFunction(
      SanCovTracePCGuardName, VoidTy, Int32PtrTy, nullptr));
  SanCovTraceEnter = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTraceEnterName, VoidTy, Int32PtrTy, nullptr));
  SanCovTraceBB = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(SanCovTraceBBName, VoidTy, Int32PtrTy, nullptr));

  // At this point we create a dummy array of guards because we don't
  // know how many elements we will need.
  Type *Int32Ty = IRB.getInt32Ty();
  Type *Int8Ty = IRB.getInt8Ty();

  if (!Options.TracePCGuard)
    GuardArray =
        new GlobalVariable(M, Int32Ty, false, GlobalValue::ExternalLinkage,
                           nullptr, "__sancov_gen_cov_tmp");
  if (Options.Use8bitCounters)
    EightBitCounterArray =
        new GlobalVariable(M, Int8Ty, false, GlobalVariable::ExternalLinkage,
                           nullptr, "__sancov_gen_cov_tmp");

  for (auto &F : M)
    runOnFunction(F);

  auto N = NumberOfInstrumentedBlocks();

  GlobalVariable *RealGuardArray = nullptr;
  if (!Options.TracePCGuard) {
    // Now we know how many elements we need. Create an array of guards
    // with one extra element at the beginning for the size.
    Type *Int32ArrayNTy = ArrayType::get(Int32Ty, N + 1);
    RealGuardArray = new GlobalVariable(
        M, Int32ArrayNTy, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Int32ArrayNTy), "__sancov_gen_cov");

    // Replace the dummy array with the real one.
    GuardArray->replaceAllUsesWith(
        IRB.CreatePointerCast(RealGuardArray, Int32PtrTy));
    GuardArray->eraseFromParent();
  }

  GlobalVariable *RealEightBitCounterArray;
  if (Options.Use8bitCounters) {
    // Make sure the array is 16-aligned.
    static const int CounterAlignment = 16;
    Type *Int8ArrayNTy = ArrayType::get(Int8Ty, alignTo(N, CounterAlignment));
    RealEightBitCounterArray = new GlobalVariable(
        M, Int8ArrayNTy, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Int8ArrayNTy), "__sancov_gen_cov_counter");
    RealEightBitCounterArray->setAlignment(CounterAlignment);
    EightBitCounterArray->replaceAllUsesWith(
        IRB.CreatePointerCast(RealEightBitCounterArray, Int8PtrTy));
    EightBitCounterArray->eraseFromParent();
  }

  // Create variable for module (compilation unit) name
  Constant *ModNameStrConst =
      ConstantDataArray::getString(M.getContext(), M.getName(), true);
  GlobalVariable *ModuleName = new GlobalVariable(
      M, ModNameStrConst->getType(), true, GlobalValue::PrivateLinkage,
      ModNameStrConst, "__sancov_gen_modname");
  if (Options.TracePCGuard) {
    if (HasSancovGuardsSection) {
      Function *CtorFunc;
      GlobalVariable *SecStart = new GlobalVariable(
          M, Int32PtrTy, false, GlobalVariable::ExternalLinkage, nullptr,
          getSanCovTracePCGuardSectionStart());
      SecStart->setVisibility(GlobalValue::HiddenVisibility);
      GlobalVariable *SecEnd = new GlobalVariable(
          M, Int32PtrTy, false, GlobalVariable::ExternalLinkage, nullptr,
          getSanCovTracePCGuardSectionEnd());
      SecEnd->setVisibility(GlobalValue::HiddenVisibility);

      std::tie(CtorFunc, std::ignore) = createSanitizerCtorAndInitFunctions(
          M, SanCovModuleCtorName, SanCovTracePCGuardInitName,
          {Int32PtrTy, Int32PtrTy},
          {IRB.CreatePointerCast(SecStart, Int32PtrTy),
            IRB.CreatePointerCast(SecEnd, Int32PtrTy)});

      if (TargetTriple.supportsCOMDAT()) {
        // Use comdat to dedup CtorFunc.
        CtorFunc->setComdat(M.getOrInsertComdat(SanCovModuleCtorName));
        appendToGlobalCtors(M, CtorFunc, SanCtorAndDtorPriority, CtorFunc);
      } else {
        appendToGlobalCtors(M, CtorFunc, SanCtorAndDtorPriority);
      }
    }
  } else if (!Options.TracePC) {
    Function *CtorFunc;
    std::tie(CtorFunc, std::ignore) = createSanitizerCtorAndInitFunctions(
        M, SanCovModuleCtorName, SanCovModuleInitName,
        {Int32PtrTy, IntptrTy, Int8PtrTy, Int8PtrTy},
        {IRB.CreatePointerCast(RealGuardArray, Int32PtrTy),
         ConstantInt::get(IntptrTy, N),
         Options.Use8bitCounters
             ? IRB.CreatePointerCast(RealEightBitCounterArray, Int8PtrTy)
             : Constant::getNullValue(Int8PtrTy),
         IRB.CreatePointerCast(ModuleName, Int8PtrTy)});

    appendToGlobalCtors(M, CtorFunc, SanCtorAndDtorPriority);
  }

  return true;
}

// True if block has successors and it dominates all of them.
static bool isFullDominator(const BasicBlock *BB, const DominatorTree *DT) {
  if (succ_begin(BB) == succ_end(BB))
    return false;

  for (const BasicBlock *SUCC : make_range(succ_begin(BB), succ_end(BB))) {
    if (!DT->dominates(BB, SUCC))
      return false;
  }

  return true;
}

// True if block has predecessors and it postdominates all of them.
static bool isFullPostDominator(const BasicBlock *BB,
                                const PostDominatorTree *PDT) {
  if (pred_begin(BB) == pred_end(BB))
    return false;

  for (const BasicBlock *PRED : make_range(pred_begin(BB), pred_end(BB))) {
    if (!PDT->dominates(BB, PRED))
      return false;
  }

  return true;
}

static bool shouldInstrumentBlock(const Function& F, const BasicBlock *BB, const DominatorTree *DT,
                                  const PostDominatorTree *PDT) {
  // Don't insert coverage for unreachable blocks: we will never call
  // __sanitizer_cov() for them, so counting them in
  // NumberOfInstrumentedBlocks() might complicate calculation of code coverage
  // percentage. Also, unreachable instructions frequently have no debug
  // locations.
  if (isa<UnreachableInst>(BB->getTerminator()))
    return false;

  // Don't insert coverage into blocks without a valid insertion point
  // (catchswitch blocks).
  if (BB->getFirstInsertionPt() == BB->end())
    return false;

  if (!ClPruneBlocks || &F.getEntryBlock() == BB)
    return true;

  return !(isFullDominator(BB, DT) || isFullPostDominator(BB, PDT));
}

bool SanitizerCoverageModule::runOnFunction(Function &F) {
  if (F.empty())
    return false;
  if (F.getName().find(".module_ctor") != std::string::npos)
    return false; // Should not instrument sanitizer init functions.
  if (F.getName().startswith("__sanitizer_"))
    return false;  // Don't instrument __sanitizer_* callbacks.
  // Don't instrument MSVC CRT configuration helpers. They may run before normal
  // initialization.
  if (F.getName() == "__local_stdio_printf_options" ||
      F.getName() == "__local_stdio_scanf_options")
    return false;
  // Don't instrument functions using SEH for now. Splitting basic blocks like
  // we do for coverage breaks WinEHPrepare.
  // FIXME: Remove this when SEH no longer uses landingpad pattern matching.
  if (F.hasPersonalityFn() &&
      isAsynchronousEHPersonality(classifyEHPersonality(F.getPersonalityFn())))
    return false;
  if (Options.CoverageType >= SanitizerCoverageOptions::SCK_Edge)
    SplitAllCriticalEdges(F);
  SmallVector<Instruction *, 8> IndirCalls;
  SmallVector<BasicBlock *, 16> BlocksToInstrument;
  SmallVector<Instruction *, 8> CmpTraceTargets;
  SmallVector<Instruction *, 8> SwitchTraceTargets;
  SmallVector<BinaryOperator *, 8> DivTraceTargets;
  SmallVector<GetElementPtrInst *, 8> GepTraceTargets;

  const DominatorTree *DT =
      &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  const PostDominatorTree *PDT =
      &getAnalysis<PostDominatorTreeWrapperPass>(F).getPostDomTree();

  for (auto &BB : F) {
    if (shouldInstrumentBlock(F, &BB, DT, PDT))
      BlocksToInstrument.push_back(&BB);
    for (auto &Inst : BB) {
      if (Options.IndirectCalls) {
        CallSite CS(&Inst);
        if (CS && !CS.getCalledFunction())
          IndirCalls.push_back(&Inst);
      }
      if (Options.TraceCmp) {
        if (isa<ICmpInst>(&Inst))
          CmpTraceTargets.push_back(&Inst);
        if (isa<SwitchInst>(&Inst))
          SwitchTraceTargets.push_back(&Inst);
      }
      if (Options.TraceDiv)
        if (BinaryOperator *BO = dyn_cast<BinaryOperator>(&Inst))
          if (BO->getOpcode() == Instruction::SDiv ||
              BO->getOpcode() == Instruction::UDiv)
            DivTraceTargets.push_back(BO);
      if (Options.TraceGep)
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&Inst))
          GepTraceTargets.push_back(GEP);
   }
  }

  InjectCoverage(F, BlocksToInstrument);
  InjectCoverageForIndirectCalls(F, IndirCalls);
  InjectTraceForCmp(F, CmpTraceTargets);
  InjectTraceForSwitch(F, SwitchTraceTargets);
  InjectTraceForDiv(F, DivTraceTargets);
  InjectTraceForGep(F, GepTraceTargets);
  return true;
}
void SanitizerCoverageModule::CreateFunctionGuardArray(size_t NumGuards,
                                                       Function &F) {
  if (!Options.TracePCGuard) return;
  HasSancovGuardsSection = true;
  ArrayType *ArrayOfInt32Ty = ArrayType::get(Int32Ty, NumGuards);
  FunctionGuardArray = new GlobalVariable(
      *CurModule, ArrayOfInt32Ty, false, GlobalVariable::PrivateLinkage,
      Constant::getNullValue(ArrayOfInt32Ty), "__sancov_gen_");
  if (auto Comdat = F.getComdat())
    FunctionGuardArray->setComdat(Comdat);
  FunctionGuardArray->setSection(getSanCovTracePCGuardSection());
}

bool SanitizerCoverageModule::InjectCoverage(Function &F,
                                             ArrayRef<BasicBlock *> AllBlocks) {
  if (AllBlocks.empty()) return false;
  switch (Options.CoverageType) {
  case SanitizerCoverageOptions::SCK_None:
    return false;
  case SanitizerCoverageOptions::SCK_Function:
    CreateFunctionGuardArray(1, F);
    InjectCoverageAtBlock(F, F.getEntryBlock(), 0, false);
    return true;
  default: {
    bool UseCalls = ClCoverageBlockThreshold < AllBlocks.size();
    CreateFunctionGuardArray(AllBlocks.size(), F);
    for (size_t i = 0, N = AllBlocks.size(); i < N; i++)
      InjectCoverageAtBlock(F, *AllBlocks[i], i, UseCalls);
    return true;
  }
  }
}

// On every indirect call we call a run-time function
// __sanitizer_cov_indir_call* with two parameters:
//   - callee address,
//   - global cache array that contains CacheSize pointers (zero-initialized).
//     The cache is used to speed up recording the caller-callee pairs.
// The address of the caller is passed implicitly via caller PC.
// CacheSize is encoded in the name of the run-time function.
void SanitizerCoverageModule::InjectCoverageForIndirectCalls(
    Function &F, ArrayRef<Instruction *> IndirCalls) {
  if (IndirCalls.empty())
    return;
  const int CacheSize = 16;
  const int CacheAlignment = 64; // Align for better performance.
  Type *Ty = ArrayType::get(IntptrTy, CacheSize);
  for (auto I : IndirCalls) {
    IRBuilder<> IRB(I);
    CallSite CS(I);
    Value *Callee = CS.getCalledValue();
    if (isa<InlineAsm>(Callee))
      continue;
    GlobalVariable *CalleeCache = new GlobalVariable(
        *F.getParent(), Ty, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Ty), "__sancov_gen_callee_cache");
    CalleeCache->setAlignment(CacheAlignment);
    if (Options.TracePC || Options.TracePCGuard)
      IRB.CreateCall(SanCovTracePCIndir,
                     IRB.CreatePointerCast(Callee, IntptrTy));
    else
      IRB.CreateCall(SanCovIndirCallFunction,
                     {IRB.CreatePointerCast(Callee, IntptrTy),
                      IRB.CreatePointerCast(CalleeCache, IntptrTy)});
  }
}

// For every switch statement we insert a call:
// __sanitizer_cov_trace_switch(CondValue,
//      {NumCases, ValueSizeInBits, Case0Value, Case1Value, Case2Value, ... })

void SanitizerCoverageModule::InjectTraceForSwitch(
    Function &, ArrayRef<Instruction *> SwitchTraceTargets) {
  for (auto I : SwitchTraceTargets) {
    if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
      IRBuilder<> IRB(I);
      SmallVector<Constant *, 16> Initializers;
      Value *Cond = SI->getCondition();
      if (Cond->getType()->getScalarSizeInBits() >
          Int64Ty->getScalarSizeInBits())
        continue;
      Initializers.push_back(ConstantInt::get(Int64Ty, SI->getNumCases()));
      Initializers.push_back(
          ConstantInt::get(Int64Ty, Cond->getType()->getScalarSizeInBits()));
      if (Cond->getType()->getScalarSizeInBits() <
          Int64Ty->getScalarSizeInBits())
        Cond = IRB.CreateIntCast(Cond, Int64Ty, false);
      for (auto It : SI->cases()) {
        Constant *C = It.getCaseValue();
        if (C->getType()->getScalarSizeInBits() <
            Int64Ty->getScalarSizeInBits())
          C = ConstantExpr::getCast(CastInst::ZExt, It.getCaseValue(), Int64Ty);
        Initializers.push_back(C);
      }
      std::sort(Initializers.begin() + 2, Initializers.end(),
                [](const Constant *A, const Constant *B) {
                  return cast<ConstantInt>(A)->getLimitedValue() <
                         cast<ConstantInt>(B)->getLimitedValue();
                });
      ArrayType *ArrayOfInt64Ty = ArrayType::get(Int64Ty, Initializers.size());
      GlobalVariable *GV = new GlobalVariable(
          *CurModule, ArrayOfInt64Ty, false, GlobalVariable::InternalLinkage,
          ConstantArray::get(ArrayOfInt64Ty, Initializers),
          "__sancov_gen_cov_switch_values");
      IRB.CreateCall(SanCovTraceSwitchFunction,
                     {Cond, IRB.CreatePointerCast(GV, Int64PtrTy)});
    }
  }
}

void SanitizerCoverageModule::InjectTraceForDiv(
    Function &, ArrayRef<BinaryOperator *> DivTraceTargets) {
  for (auto BO : DivTraceTargets) {
    IRBuilder<> IRB(BO);
    Value *A1 = BO->getOperand(1);
    if (isa<ConstantInt>(A1)) continue;
    if (!A1->getType()->isIntegerTy())
      continue;
    uint64_t TypeSize = DL->getTypeStoreSizeInBits(A1->getType());
    int CallbackIdx = TypeSize == 32 ? 0 :
        TypeSize == 64 ? 1 : -1;
    if (CallbackIdx < 0) continue;
    auto Ty = Type::getIntNTy(*C, TypeSize);
    IRB.CreateCall(SanCovTraceDivFunction[CallbackIdx],
                   {IRB.CreateIntCast(A1, Ty, true)});
  }
}

void SanitizerCoverageModule::InjectTraceForGep(
    Function &, ArrayRef<GetElementPtrInst *> GepTraceTargets) {
  for (auto GEP : GepTraceTargets) {
    IRBuilder<> IRB(GEP);
    for (auto I = GEP->idx_begin(); I != GEP->idx_end(); ++I)
      if (!isa<ConstantInt>(*I) && (*I)->getType()->isIntegerTy())
        IRB.CreateCall(SanCovTraceGepFunction,
                       {IRB.CreateIntCast(*I, IntptrTy, true)});
  }
}

void SanitizerCoverageModule::InjectTraceForCmp(
    Function &, ArrayRef<Instruction *> CmpTraceTargets) {
  for (auto I : CmpTraceTargets) {
    if (ICmpInst *ICMP = dyn_cast<ICmpInst>(I)) {
      IRBuilder<> IRB(ICMP);
      Value *A0 = ICMP->getOperand(0);
      Value *A1 = ICMP->getOperand(1);
      if (!A0->getType()->isIntegerTy())
        continue;
      uint64_t TypeSize = DL->getTypeStoreSizeInBits(A0->getType());
      int CallbackIdx = TypeSize == 8 ? 0 :
                        TypeSize == 16 ? 1 :
                        TypeSize == 32 ? 2 :
                        TypeSize == 64 ? 3 : -1;
      if (CallbackIdx < 0) continue;
      // __sanitizer_cov_trace_cmp((type_size << 32) | predicate, A0, A1);
      auto Ty = Type::getIntNTy(*C, TypeSize);
      IRB.CreateCall(
          SanCovTraceCmpFunction[CallbackIdx],
          {IRB.CreateIntCast(A0, Ty, true), IRB.CreateIntCast(A1, Ty, true)});
    }
  }
}

void SanitizerCoverageModule::SetNoSanitizeMetadata(Instruction *I) {
  I->setMetadata(I->getModule()->getMDKindID("nosanitize"),
                 MDNode::get(*C, None));
}

void SanitizerCoverageModule::InjectCoverageAtBlock(Function &F, BasicBlock &BB,
                                                    size_t Idx, bool UseCalls) {
  BasicBlock::iterator IP = BB.getFirstInsertionPt();
  bool IsEntryBB = &BB == &F.getEntryBlock();
  DebugLoc EntryLoc;
  if (IsEntryBB) {
    if (auto SP = F.getSubprogram())
      EntryLoc = DebugLoc::get(SP->getScopeLine(), 0, SP);
    // Keep static allocas and llvm.localescape calls in the entry block.  Even
    // if we aren't splitting the block, it's nice for allocas to be before
    // calls.
    IP = PrepareToSplitEntryBlock(BB, IP);
  } else {
    EntryLoc = IP->getDebugLoc();
  }

  IRBuilder<> IRB(&*IP);
  IRB.SetCurrentDebugLocation(EntryLoc);
  if (Options.TracePC) {
    IRB.CreateCall(SanCovTracePC); // gets the PC using GET_CALLER_PC.
    IRB.CreateCall(EmptyAsm, {}); // Avoids callback merge.
  } else if (Options.TracePCGuard) {
    auto GuardPtr = IRB.CreateIntToPtr(
        IRB.CreateAdd(IRB.CreatePointerCast(FunctionGuardArray, IntptrTy),
                      ConstantInt::get(IntptrTy, Idx * 4)),
        Int32PtrTy);
    if (!UseCalls) {
      auto GuardLoad = IRB.CreateLoad(GuardPtr);
      GuardLoad->setAtomic(AtomicOrdering::Monotonic);
      GuardLoad->setAlignment(8);
      SetNoSanitizeMetadata(GuardLoad);  // Don't instrument with e.g. asan.
      auto Cmp = IRB.CreateICmpNE(
          GuardLoad, Constant::getNullValue(GuardLoad->getType()));
      auto Ins = SplitBlockAndInsertIfThen(
          Cmp, &*IP, false, MDBuilder(*C).createBranchWeights(1, 100000));
      IRB.SetInsertPoint(Ins);
      IRB.SetCurrentDebugLocation(EntryLoc);
    }
    IRB.CreateCall(SanCovTracePCGuard, GuardPtr);
    IRB.CreateCall(EmptyAsm, {}); // Avoids callback merge.
  } else {
    Value *GuardP = IRB.CreateAdd(
        IRB.CreatePointerCast(GuardArray, IntptrTy),
        ConstantInt::get(IntptrTy, (1 + NumberOfInstrumentedBlocks()) * 4));
    GuardP = IRB.CreateIntToPtr(GuardP, Int32PtrTy);
    if (Options.TraceBB) {
      IRB.CreateCall(IsEntryBB ? SanCovTraceEnter : SanCovTraceBB, GuardP);
    } else if (UseCalls) {
      IRB.CreateCall(SanCovWithCheckFunction, GuardP);
    } else {
      LoadInst *Load = IRB.CreateLoad(GuardP);
      Load->setAtomic(AtomicOrdering::Monotonic);
      Load->setAlignment(4);
      SetNoSanitizeMetadata(Load);
      Value *Cmp =
          IRB.CreateICmpSGE(Constant::getNullValue(Load->getType()), Load);
      Instruction *Ins = SplitBlockAndInsertIfThen(
          Cmp, &*IP, false, MDBuilder(*C).createBranchWeights(1, 100000));
      IRB.SetInsertPoint(Ins);
      IRB.SetCurrentDebugLocation(EntryLoc);
      // __sanitizer_cov gets the PC of the instruction using GET_CALLER_PC.
      IRB.CreateCall(SanCovFunction, GuardP);
      IRB.CreateCall(EmptyAsm, {}); // Avoids callback merge.
    }
  }

  if (Options.Use8bitCounters) {
    IRB.SetInsertPoint(&*IP);
    Value *P = IRB.CreateAdd(
        IRB.CreatePointerCast(EightBitCounterArray, IntptrTy),
        ConstantInt::get(IntptrTy, NumberOfInstrumentedBlocks() - 1));
    P = IRB.CreateIntToPtr(P, IRB.getInt8PtrTy());
    LoadInst *LI = IRB.CreateLoad(P);
    Value *Inc = IRB.CreateAdd(LI, ConstantInt::get(IRB.getInt8Ty(), 1));
    StoreInst *SI = IRB.CreateStore(Inc, P);
    SetNoSanitizeMetadata(LI);
    SetNoSanitizeMetadata(SI);
  }
}

StringRef SanitizerCoverageModule::getSanCovTracePCGuardSection() const {
  if (TargetTriple.getObjectFormat() == Triple::COFF)
    return ".SCOV$M";
  if (TargetTriple.isOSBinFormatMachO())
    return "__DATA,__sancov_guards";
  return "__sancov_guards";
}

StringRef SanitizerCoverageModule::getSanCovTracePCGuardSectionStart() const {
  if (TargetTriple.isOSBinFormatMachO())
    return "\1section$start$__DATA$__sancov_guards";
  return "__start___sancov_guards";
}

StringRef SanitizerCoverageModule::getSanCovTracePCGuardSectionEnd() const {
  if (TargetTriple.isOSBinFormatMachO())
    return "\1section$end$__DATA$__sancov_guards";
  return "__stop___sancov_guards";
}


char SanitizerCoverageModule::ID = 0;
INITIALIZE_PASS_BEGIN(SanitizerCoverageModule, "sancov",
                      "SanitizerCoverage: TODO."
                      "ModulePass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SanitizerCoverageModule, "sancov",
                    "SanitizerCoverage: TODO."
                    "ModulePass",
                    false, false)
ModulePass *llvm::createSanitizerCoverageModulePass(
    const SanitizerCoverageOptions &Options) {
  return new SanitizerCoverageModule(Options);
}
