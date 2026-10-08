//===-- SanitizerCoverage.cpp - coverage instrumentation for sanitizers ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coverage instrumentation done on LLVM IR level, works with Sanitizers.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "sancov"

const char SanCovTracePCIndirName[] = "__sanitizer_cov_trace_pc_indir";
const char SanCovTracePCName[] = "__sanitizer_cov_trace_pc";
const char SanCovTracePCEntryName[] = "__sanitizer_cov_trace_pc_entry";
const char SanCovTracePCExitName[] = "__sanitizer_cov_trace_pc_exit";
const char SanCovTraceArgsName[] = "__sanitizer_cov_trace_args";
const char SanCovTraceRetName[] = "__sanitizer_cov_trace_ret";
const char SanCovTraceCmp1[] = "__sanitizer_cov_trace_cmp1";
const char SanCovTraceCmp2[] = "__sanitizer_cov_trace_cmp2";
const char SanCovTraceCmp4[] = "__sanitizer_cov_trace_cmp4";
const char SanCovTraceCmp8[] = "__sanitizer_cov_trace_cmp8";
const char SanCovTraceConstCmp1[] = "__sanitizer_cov_trace_const_cmp1";
const char SanCovTraceConstCmp2[] = "__sanitizer_cov_trace_const_cmp2";
const char SanCovTraceConstCmp4[] = "__sanitizer_cov_trace_const_cmp4";
const char SanCovTraceConstCmp8[] = "__sanitizer_cov_trace_const_cmp8";
const char SanCovLoad1[] = "__sanitizer_cov_load1";
const char SanCovLoad2[] = "__sanitizer_cov_load2";
const char SanCovLoad4[] = "__sanitizer_cov_load4";
const char SanCovLoad8[] = "__sanitizer_cov_load8";
const char SanCovLoad16[] = "__sanitizer_cov_load16";
const char SanCovStore1[] = "__sanitizer_cov_store1";
const char SanCovStore2[] = "__sanitizer_cov_store2";
const char SanCovStore4[] = "__sanitizer_cov_store4";
const char SanCovStore8[] = "__sanitizer_cov_store8";
const char SanCovStore16[] = "__sanitizer_cov_store16";
const char SanCovTraceDiv4[] = "__sanitizer_cov_trace_div4";
const char SanCovTraceDiv8[] = "__sanitizer_cov_trace_div8";
const char SanCovTraceGep[] = "__sanitizer_cov_trace_gep";
const char SanCovTraceSwitchName[] = "__sanitizer_cov_trace_switch";
const char SanCovModuleCtorTracePcGuardName[] =
    "sancov.module_ctor_trace_pc_guard";
const char SanCovModuleCtor8bitCountersName[] =
    "sancov.module_ctor_8bit_counters";
const char SanCovModuleCtorBoolFlagName[] = "sancov.module_ctor_bool_flag";
static const uint64_t SanCtorAndDtorPriority = 2;

const char SanCovTracePCGuardName[] = "__sanitizer_cov_trace_pc_guard";
const char SanCovTracePCGuardInitName[] = "__sanitizer_cov_trace_pc_guard_init";
const char SanCov8bitCountersInitName[] = "__sanitizer_cov_8bit_counters_init";
const char SanCovBoolFlagInitName[] = "__sanitizer_cov_bool_flag_init";
const char SanCovPCsInitName[] = "__sanitizer_cov_pcs_init";
const char SanCovCFsInitName[] = "__sanitizer_cov_cfs_init";

const char SanCovGuardsSectionName[] = "sancov_guards";
const char SanCovCountersSectionName[] = "sancov_cntrs";
const char SanCovBoolFlagSectionName[] = "sancov_bools";
const char SanCovPCsSectionName[] = "sancov_pcs";
const char SanCovCFsSectionName[] = "sancov_cfs";
const char SanCovCallbackGateSectionName[] = "sancov_gate";

const char SanCovStackDepthCallbackName[] = "__sanitizer_cov_stack_depth";
const char SanCovLowestStackName[] = "__sancov_lowest_stack";
const char SanCovCallbackGateName[] = "__sancov_should_track";

static cl::opt<int> ClCoverageLevel(
    "sanitizer-coverage-level",
    cl::desc("Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, "
             "3: all blocks and critical edges"),
    cl::Hidden);

static cl::opt<bool> ClTracePC("sanitizer-coverage-trace-pc",
                               cl::desc("Experimental pc tracing"), cl::Hidden);

static cl::opt<bool> ClTracePCEntryExit(
    "sanitizer-coverage-trace-pc-entry-exit",
    cl::desc("pc tracing with separate entry/exit callbacks"), cl::Hidden);

static cl::opt<bool> ClTracePCGuard("sanitizer-coverage-trace-pc-guard",
                                    cl::desc("pc tracing with a guard"),
                                    cl::Hidden);

// If true, we create a global variable that contains PCs of all instrumented
// BBs, put this global into a named section, and pass this section's bounds
// to __sanitizer_cov_pcs_init.
// This way the coverage instrumentation does not need to acquire the PCs
// at run-time. Works with trace-pc-guard, inline-8bit-counters, and
// inline-bool-flag.
static cl::opt<bool> ClCreatePCTable("sanitizer-coverage-pc-table",
                                     cl::desc("create a static PC table"),
                                     cl::Hidden);

static cl::opt<bool>
    ClInline8bitCounters("sanitizer-coverage-inline-8bit-counters",
                         cl::desc("increments 8-bit counter for every edge"),
                         cl::Hidden);

static cl::opt<bool>
    ClSancovDropCtors("sanitizer-coverage-drop-ctors",
                      cl::desc("do not emit module ctors for global counters"),
                      cl::Hidden);

static cl::opt<bool>
    ClInlineBoolFlag("sanitizer-coverage-inline-bool-flag",
                     cl::desc("sets a boolean flag for every edge"),
                     cl::Hidden);

static cl::opt<bool>
    ClCMPTracing("sanitizer-coverage-trace-compares",
                 cl::desc("Tracing of CMP and similar instructions"),
                 cl::Hidden);

static cl::opt<bool> ClDIVTracing("sanitizer-coverage-trace-divs",
                                  cl::desc("Tracing of DIV instructions"),
                                  cl::Hidden);

static cl::opt<bool> ClLoadTracing("sanitizer-coverage-trace-loads",
                                   cl::desc("Tracing of load instructions"),
                                   cl::Hidden);

static cl::opt<bool> ClStoreTracing("sanitizer-coverage-trace-stores",
                                    cl::desc("Tracing of store instructions"),
                                    cl::Hidden);

static cl::opt<bool> ClGEPTracing("sanitizer-coverage-trace-geps",
                                  cl::desc("Tracing of GEP instructions"),
                                  cl::Hidden);

static cl::opt<bool> ClTraceArgs("sanitizer-coverage-trace-args",
                                 cl::desc("Tracing of function arguments"),
                                 cl::Hidden);

static cl::opt<bool> ClTraceRet("sanitizer-coverage-trace-ret",
                                cl::desc("Tracing of return values"),
                                cl::Hidden);

static cl::opt<bool>
    ClPruneBlocks("sanitizer-coverage-prune-blocks",
                  cl::desc("Reduce the number of instrumented blocks"),
                  cl::Hidden, cl::init(true));

static cl::opt<bool> ClStackDepth("sanitizer-coverage-stack-depth",
                                  cl::desc("max stack depth tracing"),
                                  cl::Hidden);

static cl::opt<int> ClStackDepthCallbackMin(
    "sanitizer-coverage-stack-depth-callback-min",
    cl::desc("max stack depth tracing should use callback and only when "
             "stack depth more than specified"),
    cl::Hidden);

static cl::opt<bool>
    ClCollectCF("sanitizer-coverage-control-flow",
                cl::desc("collect control flow for each function"), cl::Hidden);

static cl::opt<bool> ClGatedCallbacks(
    "sanitizer-coverage-gated-trace-callbacks",
    cl::desc("Gate the invocation of the tracing callbacks on a global variable"
             ". Currently only supported for trace-pc-guard and trace-cmp."),
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
  Options.TraceCmp |= ClCMPTracing;
  Options.TraceDiv |= ClDIVTracing;
  Options.TraceGep |= ClGEPTracing;
  Options.TracePC |= ClTracePC;
  Options.TracePCEntryExit |= ClTracePCEntryExit;
  Options.TracePCGuard |= ClTracePCGuard;
  Options.Inline8bitCounters |= ClInline8bitCounters;
  Options.InlineBoolFlag |= ClInlineBoolFlag;
  Options.PCTable |= ClCreatePCTable;
  Options.NoPrune |= !ClPruneBlocks;
  Options.StackDepth |= ClStackDepth;
  Options.StackDepthCallbackMin = std::max(Options.StackDepthCallbackMin,
                                           ClStackDepthCallbackMin.getValue());
  Options.TraceLoads |= ClLoadTracing;
  Options.TraceStores |= ClStoreTracing;
  Options.TraceArgs |= ClTraceArgs;
  Options.TraceRet |= ClTraceRet;
  Options.GatedCallbacks |= ClGatedCallbacks;
  if (!Options.TracePCGuard && !Options.TracePC && !Options.TracePCEntryExit &&
      !Options.Inline8bitCounters && !Options.StackDepth &&
      !Options.InlineBoolFlag && !Options.TraceLoads && !Options.TraceStores &&
      !Options.TraceArgs && !Options.TraceRet)
    Options.TracePCGuard = true; // TracePCGuard is default.
  Options.CollectControlFlow |= ClCollectCF;
  return Options;
}

class ModuleSanitizerCoverage {
public:
  using DomTreeCallback = function_ref<const DominatorTree &(Function &F)>;
  using PostDomTreeCallback =
      function_ref<const PostDominatorTree &(Function &F)>;

  ModuleSanitizerCoverage(Module &M, DomTreeCallback DTCallback,
                          PostDomTreeCallback PDTCallback,
                          const SanitizerCoverageOptions &Options,
                          const SpecialCaseList *Allowlist,
                          const SpecialCaseList *Blocklist)
      : M(M), DTCallback(DTCallback), PDTCallback(PDTCallback),
        Options(Options), Allowlist(Allowlist), Blocklist(Blocklist) {}

  bool instrumentModule();

private:
  void createFunctionControlFlow(Function &F);
  void instrumentFunction(Function &F);
  void InjectCoverageForIndirectCalls(Function &F,
                                      ArrayRef<Instruction *> IndirCalls);
  void InjectTraceForCmp(Function &F, ArrayRef<Instruction *> CmpTraceTargets,
                         Value *&FunctionGateCmp);
  void InjectTraceForDiv(Function &F,
                         ArrayRef<BinaryOperator *> DivTraceTargets);
  void InjectTraceForGep(Function &F,
                         ArrayRef<GetElementPtrInst *> GepTraceTargets);
  void InjectTraceForLoadsAndStores(Function &F, ArrayRef<LoadInst *> Loads,
                                    ArrayRef<StoreInst *> Stores);
  void InjectTraceForExits(Function &F);
  void InjectTraceForArgs(Function &F);
  void InjectTraceForRet(Function &F);
  void InjectTraceForSwitch(Function &F,
                            ArrayRef<Instruction *> SwitchTraceTargets,
                            Value *&FunctionGateCmp);
  bool InjectCoverage(Function &F, ArrayRef<BasicBlock *> AllBlocks,
                      Value *&FunctionGateCmp, bool IsLeafFunc);
  GlobalVariable *CreateFunctionLocalArrayInSection(size_t NumElements,
                                                    Function &F, Type *Ty,
                                                    const char *Section);
  GlobalVariable *CreatePCArray(Function &F, ArrayRef<BasicBlock *> AllBlocks);
  void CreateFunctionLocalArrays(Function &F, ArrayRef<BasicBlock *> AllBlocks);
  Instruction *CreateGateBranch(Function &F, Value *&FunctionGateCmp,
                                Instruction *I);
  Value *CreateFunctionLocalGateCmp(IRBuilder<> &IRB);
  void InjectCoverageAtBlock(Function &F, BasicBlock &BB, size_t Idx,
                             Value *&FunctionGateCmp, bool IsLeafFunc);
  Function *CreateInitCallsForSections(Module &M, const char *CtorName,
                                       const char *InitFunctionName, Type *Ty,
                                       const char *Section);
  std::pair<Value *, Value *> CreateSecStartEnd(Module &M, const char *Section,
                                                Type *Ty);

  std::string getSectionName(const std::string &Section) const;
  std::string getSectionStart(const std::string &Section) const;
  std::string getSectionEnd(const std::string &Section) const;

  /// The `offsets` and `num_fields` arguments of the trace-args/trace-ret
  /// callbacks: a constant table holding one {byte offset, byte size} pair per
  /// field of a struct, and the size of that struct. Table is null when the
  /// field layout is unknown, in which case NumFields and ObjectSize are 0.
  struct FieldOffsets {
    Constant *Table = nullptr;
    unsigned NumFields = 0;
    uint64_t ObjectSize = 0;
  };
  FieldOffsets getFieldOffsets(DIType *Ty);

  Module &M;
  DomTreeCallback DTCallback;
  PostDomTreeCallback PDTCallback;

  FunctionCallee SanCovStackDepthCallback;
  FunctionCallee SanCovTracePCIndir;
  FunctionCallee SanCovTracePC, SanCovTracePCGuard;
  FunctionCallee SanCovTracePCEntry, SanCovTracePCExit;
  FunctionCallee SanCovTraceArgsFunc, SanCovTraceRetFunc;
  std::array<FunctionCallee, 4> SanCovTraceCmpFunction;
  std::array<FunctionCallee, 4> SanCovTraceConstCmpFunction;
  std::array<FunctionCallee, 5> SanCovLoadFunction;
  std::array<FunctionCallee, 5> SanCovStoreFunction;
  std::array<FunctionCallee, 2> SanCovTraceDivFunction;
  FunctionCallee SanCovTraceGepFunction;
  FunctionCallee SanCovTraceSwitchFunction;
  GlobalVariable *SanCovLowestStack;
  GlobalVariable *SanCovCallbackGate;
  Type *PtrTy, *IntptrTy, *Int64Ty, *Int32Ty, *Int16Ty, *Int8Ty, *Int1Ty;
  Module *CurModule;
  Triple TargetTriple;
  LLVMContext *C;
  const DataLayout *DL;

  GlobalVariable *FunctionGuardArray;       // for trace-pc-guard.
  GlobalVariable *Function8bitCounterArray; // for inline-8bit-counters.
  GlobalVariable *FunctionBoolArray;        // for inline-bool-flag.
  GlobalVariable *FunctionPCsArray;         // for pc-table.
  GlobalVariable *FunctionCFsArray;         // for control flow table
  SmallVector<GlobalValue *, 20> GlobalsToAppendToUsed;
  SmallVector<GlobalValue *, 20> GlobalsToAppendToCompilerUsed;

  /// Field offset tables are shared by every value of the same struct type: a
  /// type used by hundreds of functions must not emit hundreds of identical
  /// tables.
  DenseMap<const DICompositeType *, FieldOffsets> FieldOffsetsCache;

  SanitizerCoverageOptions Options;

  const SpecialCaseList *Allowlist;
  const SpecialCaseList *Blocklist;
};
} // namespace

SanitizerCoveragePass::SanitizerCoveragePass(
    SanitizerCoverageOptions Options, IntrusiveRefCntPtr<vfs::FileSystem> VFS,
    const std::vector<std::string> &AllowlistFiles,
    const std::vector<std::string> &BlocklistFiles)
    : Options(std::move(Options)),
      VFS(VFS ? std::move(VFS) : vfs::getRealFileSystem()) {
  if (AllowlistFiles.size() > 0)
    Allowlist = SpecialCaseList::createOrDie(AllowlistFiles, *this->VFS);
  if (BlocklistFiles.size() > 0)
    Blocklist = SpecialCaseList::createOrDie(BlocklistFiles, *this->VFS);
}

PreservedAnalyses SanitizerCoveragePass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto DTCallback = [&FAM](Function &F) -> const DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto PDTCallback = [&FAM](Function &F) -> const PostDominatorTree & {
    return FAM.getResult<PostDominatorTreeAnalysis>(F);
  };
  ModuleSanitizerCoverage ModuleSancov(M, DTCallback, PDTCallback,
                                       OverrideFromCL(Options), Allowlist.get(),
                                       Blocklist.get());
  if (!ModuleSancov.instrumentModule())
    return PreservedAnalyses::all();

  PreservedAnalyses PA = PreservedAnalyses::none();
  // GlobalsAA is considered stateless and does not get invalidated unless
  // explicitly invalidated; PreservedAnalyses::none() is not enough. Sanitizers
  // make changes that require GlobalsAA to be invalidated.
  PA.abandon<GlobalsAA>();
  return PA;
}

std::pair<Value *, Value *>
ModuleSanitizerCoverage::CreateSecStartEnd(Module &M, const char *Section,
                                           Type *Ty) {
  // Use ExternalWeak so that if all sections are discarded due to section
  // garbage collection, the linker will not report undefined symbol errors.
  // Windows defines the start/stop symbols in compiler-rt so no need for
  // ExternalWeak.
  GlobalValue::LinkageTypes Linkage = TargetTriple.isOSBinFormatCOFF()
                                          ? GlobalVariable::ExternalLinkage
                                          : GlobalVariable::ExternalWeakLinkage;
  GlobalVariable *SecStart = new GlobalVariable(M, Ty, false, Linkage, nullptr,
                                                getSectionStart(Section));
  SecStart->setVisibility(GlobalValue::HiddenVisibility);
  GlobalVariable *SecEnd = new GlobalVariable(M, Ty, false, Linkage, nullptr,
                                              getSectionEnd(Section));
  SecEnd->setVisibility(GlobalValue::HiddenVisibility);
  if (!TargetTriple.isOSBinFormatCOFF())
    return std::make_pair(SecStart, SecEnd);

  // Account for the fact that on windows-msvc __start_* symbols actually
  // point to a uint64_t before the start of the array.
  auto *GEP = ConstantExpr::getPtrAdd(
      SecStart, ConstantInt::get(IntptrTy, sizeof(uint64_t)));
  return std::make_pair(GEP, SecEnd);
}

Function *ModuleSanitizerCoverage::CreateInitCallsForSections(
    Module &M, const char *CtorName, const char *InitFunctionName, Type *Ty,
    const char *Section) {
  if (ClSancovDropCtors)
    return nullptr;
  auto SecStartEnd = CreateSecStartEnd(M, Section, Ty);
  auto SecStart = SecStartEnd.first;
  auto SecEnd = SecStartEnd.second;
  Function *CtorFunc;
  std::tie(CtorFunc, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, CtorName, InitFunctionName, {PtrTy, PtrTy}, {SecStart, SecEnd});
  assert(CtorFunc->getName() == CtorName);

  if (TargetTriple.supportsCOMDAT()) {
    // Use comdat to dedup CtorFunc.
    CtorFunc->setComdat(M.getOrInsertComdat(CtorName));
    appendToGlobalCtors(M, CtorFunc, SanCtorAndDtorPriority, CtorFunc);
  } else {
    appendToGlobalCtors(M, CtorFunc, SanCtorAndDtorPriority);
  }

  if (TargetTriple.isOSBinFormatCOFF()) {
    // In COFF files, if the contructors are set as COMDAT (they are because
    // COFF supports COMDAT) and the linker flag /OPT:REF (strip unreferenced
    // functions and data) is used, the constructors get stripped. To prevent
    // this, give the constructors weak ODR linkage and ensure the linker knows
    // to include the sancov constructor. This way the linker can deduplicate
    // the constructors but always leave one copy.
    CtorFunc->setLinkage(GlobalValue::WeakODRLinkage);
  }
  return CtorFunc;
}

bool ModuleSanitizerCoverage::instrumentModule() {
  if (Options.CoverageType == SanitizerCoverageOptions::SCK_None)
    return false;
  if (Allowlist &&
      !Allowlist->inSection("coverage", "src", M.getSourceFileName()))
    return false;
  if (Blocklist &&
      Blocklist->inSection("coverage", "src", M.getSourceFileName()))
    return false;
  C = &(M.getContext());
  DL = &M.getDataLayout();
  CurModule = &M;
  TargetTriple = M.getTargetTriple();
  FunctionGuardArray = nullptr;
  Function8bitCounterArray = nullptr;
  FunctionBoolArray = nullptr;
  FunctionPCsArray = nullptr;
  FunctionCFsArray = nullptr;
  IntptrTy = Type::getIntNTy(*C, DL->getPointerSizeInBits());
  PtrTy = PointerType::getUnqual(*C);
  Type *VoidTy = Type::getVoidTy(*C);
  IRBuilder<> IRB(M);
  Int64Ty = IRB.getInt64Ty();
  Int32Ty = IRB.getInt32Ty();
  Int16Ty = IRB.getInt16Ty();
  Int8Ty = IRB.getInt8Ty();
  Int1Ty = IRB.getInt1Ty();

  SanCovTracePCIndir =
      M.getOrInsertFunction(SanCovTracePCIndirName, VoidTy, IntptrTy);
  // Make sure smaller parameters are zero-extended to i64 if required by the
  // target ABI.
  AttributeList SanCovTraceCmpZeroExtAL;
  SanCovTraceCmpZeroExtAL =
      SanCovTraceCmpZeroExtAL.addParamAttribute(*C, 0, Attribute::ZExt);
  SanCovTraceCmpZeroExtAL =
      SanCovTraceCmpZeroExtAL.addParamAttribute(*C, 1, Attribute::ZExt);

  SanCovTraceCmpFunction[0] =
      M.getOrInsertFunction(SanCovTraceCmp1, SanCovTraceCmpZeroExtAL, VoidTy,
                            IRB.getInt8Ty(), IRB.getInt8Ty());
  SanCovTraceCmpFunction[1] =
      M.getOrInsertFunction(SanCovTraceCmp2, SanCovTraceCmpZeroExtAL, VoidTy,
                            IRB.getInt16Ty(), IRB.getInt16Ty());
  SanCovTraceCmpFunction[2] =
      M.getOrInsertFunction(SanCovTraceCmp4, SanCovTraceCmpZeroExtAL, VoidTy,
                            IRB.getInt32Ty(), IRB.getInt32Ty());
  SanCovTraceCmpFunction[3] =
      M.getOrInsertFunction(SanCovTraceCmp8, VoidTy, Int64Ty, Int64Ty);

  SanCovTraceConstCmpFunction[0] = M.getOrInsertFunction(
      SanCovTraceConstCmp1, SanCovTraceCmpZeroExtAL, VoidTy, Int8Ty, Int8Ty);
  SanCovTraceConstCmpFunction[1] = M.getOrInsertFunction(
      SanCovTraceConstCmp2, SanCovTraceCmpZeroExtAL, VoidTy, Int16Ty, Int16Ty);
  SanCovTraceConstCmpFunction[2] = M.getOrInsertFunction(
      SanCovTraceConstCmp4, SanCovTraceCmpZeroExtAL, VoidTy, Int32Ty, Int32Ty);
  SanCovTraceConstCmpFunction[3] =
      M.getOrInsertFunction(SanCovTraceConstCmp8, VoidTy, Int64Ty, Int64Ty);

  // Loads.
  SanCovLoadFunction[0] = M.getOrInsertFunction(SanCovLoad1, VoidTy, PtrTy);
  SanCovLoadFunction[1] = M.getOrInsertFunction(SanCovLoad2, VoidTy, PtrTy);
  SanCovLoadFunction[2] = M.getOrInsertFunction(SanCovLoad4, VoidTy, PtrTy);
  SanCovLoadFunction[3] = M.getOrInsertFunction(SanCovLoad8, VoidTy, PtrTy);
  SanCovLoadFunction[4] = M.getOrInsertFunction(SanCovLoad16, VoidTy, PtrTy);
  // Stores.
  SanCovStoreFunction[0] = M.getOrInsertFunction(SanCovStore1, VoidTy, PtrTy);
  SanCovStoreFunction[1] = M.getOrInsertFunction(SanCovStore2, VoidTy, PtrTy);
  SanCovStoreFunction[2] = M.getOrInsertFunction(SanCovStore4, VoidTy, PtrTy);
  SanCovStoreFunction[3] = M.getOrInsertFunction(SanCovStore8, VoidTy, PtrTy);
  SanCovStoreFunction[4] = M.getOrInsertFunction(SanCovStore16, VoidTy, PtrTy);

  {
    AttributeList AL;
    AL = AL.addParamAttribute(*C, 0, Attribute::ZExt);
    SanCovTraceDivFunction[0] =
        M.getOrInsertFunction(SanCovTraceDiv4, AL, VoidTy, IRB.getInt32Ty());
  }
  SanCovTraceDivFunction[1] =
      M.getOrInsertFunction(SanCovTraceDiv8, VoidTy, Int64Ty);
  SanCovTraceGepFunction =
      M.getOrInsertFunction(SanCovTraceGep, VoidTy, IntptrTy);
  SanCovTraceSwitchFunction =
      M.getOrInsertFunction(SanCovTraceSwitchName, VoidTy, Int64Ty, PtrTy);

  SanCovLowestStack = M.getOrInsertGlobal(SanCovLowestStackName, IntptrTy);
  if (SanCovLowestStack->getValueType() != IntptrTy) {
    C->emitError(StringRef("'") + SanCovLowestStackName +
                 "' should not be declared by the user");
    return true;
  }
  SanCovLowestStack->setThreadLocalMode(
      GlobalValue::ThreadLocalMode::InitialExecTLSModel);
  if (Options.StackDepth && !SanCovLowestStack->isDeclaration())
    SanCovLowestStack->setInitializer(Constant::getAllOnesValue(IntptrTy));

  if (Options.GatedCallbacks) {
    if (!Options.TracePCGuard && !Options.TraceCmp) {
      C->emitError(StringRef("'") + ClGatedCallbacks.ArgStr +
                   "' is only supported with trace-pc-guard or trace-cmp");
      return true;
    }

    SanCovCallbackGate = cast<GlobalVariable>(
        M.getOrInsertGlobal(SanCovCallbackGateName, Int64Ty));
    SanCovCallbackGate->setSection(
        getSectionName(SanCovCallbackGateSectionName));
    SanCovCallbackGate->setInitializer(Constant::getNullValue(Int64Ty));
    SanCovCallbackGate->setLinkage(GlobalVariable::LinkOnceAnyLinkage);
    SanCovCallbackGate->setVisibility(GlobalVariable::HiddenVisibility);
    appendToCompilerUsed(M, SanCovCallbackGate);
  }

  SanCovTracePC = M.getOrInsertFunction(SanCovTracePCName, VoidTy);
  SanCovTracePCEntry = M.getOrInsertFunction(SanCovTracePCEntryName, VoidTy);
  SanCovTracePCExit = M.getOrInsertFunction(SanCovTracePCExitName, VoidTy);
  SanCovTracePCGuard =
      M.getOrInsertFunction(SanCovTracePCGuardName, VoidTy, PtrTy);

  // See the "Argument and return value tracing" section below for the meaning
  // of the arguments.
  // void __sanitizer_cov_trace_args(u64 pc, u32 arg_idx, u32 size, u64 val,
  //                                 u64 *offsets, u32 num_fields)
  SanCovTraceArgsFunc =
      M.getOrInsertFunction(SanCovTraceArgsName, VoidTy, Int64Ty, Int32Ty,
                            Int32Ty, Int64Ty, PtrTy, Int32Ty);
  // void __sanitizer_cov_trace_ret(u64 pc, u32 size, u64 val,
  //                                u64 *offsets, u32 num_fields)
  SanCovTraceRetFunc = M.getOrInsertFunction(
      SanCovTraceRetName, VoidTy, Int64Ty, Int32Ty, Int64Ty, PtrTy, Int32Ty);

  SanCovStackDepthCallback =
      M.getOrInsertFunction(SanCovStackDepthCallbackName, VoidTy);

  for (auto &F : M)
    instrumentFunction(F);

  Function *Ctor = nullptr;

  if (FunctionGuardArray)
    Ctor = CreateInitCallsForSections(M, SanCovModuleCtorTracePcGuardName,
                                      SanCovTracePCGuardInitName, Int32Ty,
                                      SanCovGuardsSectionName);
  if (Function8bitCounterArray)
    Ctor = CreateInitCallsForSections(M, SanCovModuleCtor8bitCountersName,
                                      SanCov8bitCountersInitName, Int8Ty,
                                      SanCovCountersSectionName);
  if (FunctionBoolArray) {
    Ctor = CreateInitCallsForSections(M, SanCovModuleCtorBoolFlagName,
                                      SanCovBoolFlagInitName, Int1Ty,
                                      SanCovBoolFlagSectionName);
  }
  if (Ctor && Options.PCTable) {
    auto SecStartEnd = CreateSecStartEnd(M, SanCovPCsSectionName, IntptrTy);
    FunctionCallee InitFunction =
        declareSanitizerInitFunction(M, SanCovPCsInitName, {PtrTy, PtrTy});
    IRBuilder<> IRBCtor(Ctor->getEntryBlock().getTerminator());
    IRBCtor.CreateCall(InitFunction, {SecStartEnd.first, SecStartEnd.second});
  }

  if (Ctor && Options.CollectControlFlow) {
    auto SecStartEnd = CreateSecStartEnd(M, SanCovCFsSectionName, IntptrTy);
    FunctionCallee InitFunction =
        declareSanitizerInitFunction(M, SanCovCFsInitName, {PtrTy, PtrTy});
    IRBuilder<> IRBCtor(Ctor->getEntryBlock().getTerminator());
    IRBCtor.CreateCall(InitFunction, {SecStartEnd.first, SecStartEnd.second});
  }

  appendToUsed(M, GlobalsToAppendToUsed);
  appendToCompilerUsed(M, GlobalsToAppendToCompilerUsed);
  return true;
}

// True if block has successors and it dominates all of them.
static bool isFullDominator(const BasicBlock *BB, const DominatorTree &DT) {
  if (succ_empty(BB))
    return false;

  return llvm::all_of(successors(BB), [&](const BasicBlock *SUCC) {
    return DT.dominates(BB, SUCC);
  });
}

// True if block has predecessors and it postdominates all of them.
static bool isFullPostDominator(const BasicBlock *BB,
                                const PostDominatorTree &PDT) {
  if (pred_empty(BB))
    return false;

  return llvm::all_of(predecessors(BB), [&](const BasicBlock *PRED) {
    return PDT.dominates(BB, PRED);
  });
}

static bool shouldInstrumentBlock(const Function &F, const BasicBlock *BB,
                                  const DominatorTree &DT,
                                  const PostDominatorTree &PDT,
                                  const SanitizerCoverageOptions &Options) {
  // Don't insert coverage for blocks containing nothing but unreachable: we
  // will never call __sanitizer_cov() for them, so counting them in
  // NumberOfInstrumentedBlocks() might complicate calculation of code coverage
  // percentage. Also, unreachable instructions frequently have no debug
  // locations.
  if (isa<UnreachableInst>(BB->getFirstNonPHIOrDbgOrLifetime()))
    return false;

  // Don't insert coverage into blocks without a valid insertion point
  // (catchswitch blocks).
  if (BB->getFirstInsertionPt() == BB->end())
    return false;

  if (Options.NoPrune || &F.getEntryBlock() == BB)
    return true;

  if (Options.CoverageType == SanitizerCoverageOptions::SCK_Function &&
      &F.getEntryBlock() != BB)
    return false;

  // Do not instrument full dominators, or full post-dominators with multiple
  // predecessors.
  return !isFullDominator(BB, DT) &&
         !(isFullPostDominator(BB, PDT) && !BB->getSinglePredecessor());
}

// Returns true iff From->To is a backedge.
// A twist here is that we treat From->To as a backedge if
//   * To dominates From or
//   * To->UniqueSuccessor dominates From
static bool IsBackEdge(BasicBlock *From, BasicBlock *To,
                       const DominatorTree &DT) {
  if (DT.dominates(To, From))
    return true;
  if (auto Next = To->getUniqueSuccessor())
    if (DT.dominates(Next, From))
      return true;
  return false;
}

// Prunes uninteresting Cmp instrumentation:
//   * CMP instructions that feed into loop backedge branch.
//
// Note that Cmp pruning is controlled by the same flag as the
// BB pruning.
static bool IsInterestingCmp(ICmpInst *CMP, const DominatorTree &DT,
                             const SanitizerCoverageOptions &Options) {
  if (!Options.NoPrune)
    if (CMP->hasOneUse())
      if (auto BR = dyn_cast<CondBrInst>(CMP->user_back()))
        for (BasicBlock *B : BR->successors())
          if (IsBackEdge(BR->getParent(), B, DT))
            return false;
  return true;
}

void ModuleSanitizerCoverage::instrumentFunction(Function &F) {
  if (F.empty())
    return;
  if (F.getName().contains(".module_ctor"))
    return; // Should not instrument sanitizer init functions.
  if (F.getName().starts_with("__sanitizer_"))
    return; // Don't instrument __sanitizer_* callbacks.
  // Don't touch available_externally functions, their actual body is elewhere.
  if (F.getLinkage() == GlobalValue::AvailableExternallyLinkage)
    return;
  // Don't instrument MSVC CRT configuration helpers. They may run before normal
  // initialization.
  if (F.getName() == "__local_stdio_printf_options" ||
      F.getName() == "__local_stdio_scanf_options")
    return;
  if (isa<UnreachableInst>(F.getEntryBlock().getTerminator()))
    return;
  // Don't instrument functions using SEH for now. Splitting basic blocks like
  // we do for coverage breaks WinEHPrepare.
  // FIXME: Remove this when SEH no longer uses landingpad pattern matching.
  if (F.hasPersonalityFn() &&
      isAsynchronousEHPersonality(classifyEHPersonality(F.getPersonalityFn())))
    return;
  if (Allowlist && !Allowlist->inSection("coverage", "fun", F.getName()))
    return;
  if (Blocklist && Blocklist->inSection("coverage", "fun", F.getName()))
    return;
  // Do not apply any instrumentation for naked functions.
  if (F.hasFnAttribute(Attribute::Naked))
    return;
  if (F.hasFnAttribute(Attribute::NoSanitizeCoverage))
    return;
  if (F.hasFnAttribute(Attribute::DisableSanitizerInstrumentation))
    return;
  if (Options.CoverageType >= SanitizerCoverageOptions::SCK_Edge) {
    SplitAllCriticalEdges(
        F, CriticalEdgeSplittingOptions().setIgnoreUnreachableDests());
  }
  SmallVector<Instruction *, 8> IndirCalls;
  SmallVector<BasicBlock *, 16> BlocksToInstrument;
  SmallVector<Instruction *, 8> CmpTraceTargets;
  SmallVector<Instruction *, 8> SwitchTraceTargets;
  SmallVector<BinaryOperator *, 8> DivTraceTargets;
  SmallVector<GetElementPtrInst *, 8> GepTraceTargets;
  SmallVector<LoadInst *, 8> Loads;
  SmallVector<StoreInst *, 8> Stores;

  const DominatorTree &DT = DTCallback(F);
  const PostDominatorTree &PDT = PDTCallback(F);
  bool IsLeafFunc = true;

  for (auto &BB : F) {
    if (shouldInstrumentBlock(F, &BB, DT, PDT, Options))
      BlocksToInstrument.push_back(&BB);
    for (auto &Inst : BB) {
      if (Options.IndirectCalls) {
        CallBase *CB = dyn_cast<CallBase>(&Inst);
        if (CB && CB->isIndirectCall())
          IndirCalls.push_back(&Inst);
      }
      if (Options.TraceCmp) {
        if (ICmpInst *CMP = dyn_cast<ICmpInst>(&Inst))
          if (IsInterestingCmp(CMP, DT, Options))
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
      if (Options.TraceLoads)
        if (LoadInst *LI = dyn_cast<LoadInst>(&Inst))
          Loads.push_back(LI);
      if (Options.TraceStores)
        if (StoreInst *SI = dyn_cast<StoreInst>(&Inst))
          Stores.push_back(SI);
      if (Options.StackDepth)
        if (isa<InvokeInst>(Inst) ||
            (isa<CallInst>(Inst) && !isa<IntrinsicInst>(Inst)))
          IsLeafFunc = false;
    }
  }

  if (Options.CollectControlFlow)
    createFunctionControlFlow(F);

  Value *FunctionGateCmp = nullptr;
  InjectCoverage(F, BlocksToInstrument, FunctionGateCmp, IsLeafFunc);
  InjectCoverageForIndirectCalls(F, IndirCalls);
  InjectTraceForCmp(F, CmpTraceTargets, FunctionGateCmp);
  InjectTraceForSwitch(F, SwitchTraceTargets, FunctionGateCmp);
  InjectTraceForDiv(F, DivTraceTargets);
  InjectTraceForGep(F, GepTraceTargets);
  InjectTraceForLoadsAndStores(F, Loads, Stores);

  if (Options.TracePCEntryExit)
    InjectTraceForExits(F);

  if (Options.TraceArgs)
    InjectTraceForArgs(F);

  if (Options.TraceRet)
    InjectTraceForRet(F);
}

GlobalVariable *ModuleSanitizerCoverage::CreateFunctionLocalArrayInSection(
    size_t NumElements, Function &F, Type *Ty, const char *Section) {
  ArrayType *ArrayTy = ArrayType::get(Ty, NumElements);
  auto Array = new GlobalVariable(
      *CurModule, ArrayTy, false, GlobalVariable::PrivateLinkage,
      Constant::getNullValue(ArrayTy), "__sancov_gen_");

  // sancov_pcs parallels the other arrays, so they must be retained or
  // discarded together. Put them in F's comdat to tie them to F for linker GC
  // benefit. Outside ELF (nodeduplicate), a new comdat for an interposable F
  // could prevail over a strong definition (COFF: the weak external becomes a
  // COMDAT definition; Wasm: comdats deduplicate first). noipa doesn't affect
  // symbol resolution.
  if (TargetTriple.supportsCOMDAT() &&
      (F.hasComdat() || TargetTriple.isOSBinFormatELF() ||
       !F.isInterposable(/*CheckNoIPA=*/false)))
    if (auto Comdat = getOrCreateFunctionComdat(F, TargetTriple))
      Array->setComdat(Comdat);
  Array->setSection(getSectionName(Section));
  Array->setAlignment(Align(DL->getTypeStoreSize(Ty).getFixedValue()));

  // Optimizers (e.g. GlobalOpt/ConstantMerge) may not discard the arrays as a
  // unit, so retain them in the compiler; without a comdat, in the linker too.
  if (Array->hasComdat())
    GlobalsToAppendToCompilerUsed.push_back(Array);
  else
    GlobalsToAppendToUsed.push_back(Array);

  return Array;
}

GlobalVariable *
ModuleSanitizerCoverage::CreatePCArray(Function &F,
                                       ArrayRef<BasicBlock *> AllBlocks) {
  size_t N = AllBlocks.size();
  assert(N);
  SmallVector<Constant *, 32> PCs;
  IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
  for (size_t i = 0; i < N; i++) {
    if (&F.getEntryBlock() == AllBlocks[i]) {
      PCs.push_back((Constant *)IRB.CreatePointerCast(&F, PtrTy));
      PCs.push_back(
          (Constant *)IRB.CreateIntToPtr(ConstantInt::get(IntptrTy, 1), PtrTy));
    } else {
      PCs.push_back((Constant *)IRB.CreatePointerCast(
          BlockAddress::get(AllBlocks[i]), PtrTy));
      PCs.push_back(Constant::getNullValue(PtrTy));
    }
  }
  auto *PCArray =
      CreateFunctionLocalArrayInSection(N * 2, F, PtrTy, SanCovPCsSectionName);
  PCArray->setInitializer(
      ConstantArray::get(ArrayType::get(PtrTy, N * 2), PCs));
  PCArray->setConstant(true);

  return PCArray;
}

void ModuleSanitizerCoverage::CreateFunctionLocalArrays(
    Function &F, ArrayRef<BasicBlock *> AllBlocks) {
  if (Options.TracePCGuard)
    FunctionGuardArray = CreateFunctionLocalArrayInSection(
        AllBlocks.size(), F, Int32Ty, SanCovGuardsSectionName);

  if (Options.Inline8bitCounters)
    Function8bitCounterArray = CreateFunctionLocalArrayInSection(
        AllBlocks.size(), F, Int8Ty, SanCovCountersSectionName);
  if (Options.InlineBoolFlag)
    FunctionBoolArray = CreateFunctionLocalArrayInSection(
        AllBlocks.size(), F, Int1Ty, SanCovBoolFlagSectionName);

  if (Options.PCTable)
    FunctionPCsArray = CreatePCArray(F, AllBlocks);
}

Value *ModuleSanitizerCoverage::CreateFunctionLocalGateCmp(IRBuilder<> &IRB) {
  auto Load = IRB.CreateLoad(Int64Ty, SanCovCallbackGate);
  Load->setNoSanitizeMetadata();
  auto Cmp = IRB.CreateIsNotNull(Load);
  Cmp->setName("sancov gate cmp");
  return Cmp;
}

Instruction *ModuleSanitizerCoverage::CreateGateBranch(Function &F,
                                                       Value *&FunctionGateCmp,
                                                       Instruction *IP) {
  if (!FunctionGateCmp) {
    // Create this in the entry block
    BasicBlock &BB = F.getEntryBlock();
    BasicBlock::iterator IP = BB.getFirstInsertionPt();
    IP = PrepareToSplitEntryBlock(BB, IP);
    IRBuilder<> EntryIRB(&*IP);
    FunctionGateCmp = CreateFunctionLocalGateCmp(EntryIRB);
  }
  // Set the branch weights in order to minimize the price paid when the
  // gate is turned off, allowing the default enablement of this
  // instrumentation with as little of a performance cost as possible
  auto Weights = MDBuilder(*C).createBranchWeights(1, 100000);
  return SplitBlockAndInsertIfThen(FunctionGateCmp, IP, false, Weights);
}

bool ModuleSanitizerCoverage::InjectCoverage(Function &F,
                                             ArrayRef<BasicBlock *> AllBlocks,
                                             Value *&FunctionGateCmp,
                                             bool IsLeafFunc) {
  if (AllBlocks.empty())
    return false;
  CreateFunctionLocalArrays(F, AllBlocks);
  for (size_t i = 0, N = AllBlocks.size(); i < N; i++)
    InjectCoverageAtBlock(F, *AllBlocks[i], i, FunctionGateCmp, IsLeafFunc);

  return true;
}

// On every indirect call we call a run-time function
// __sanitizer_cov_indir_call* with two parameters:
//   - callee address,
//   - global cache array that contains CacheSize pointers (zero-initialized).
//     The cache is used to speed up recording the caller-callee pairs.
// The address of the caller is passed implicitly via caller PC.
// CacheSize is encoded in the name of the run-time function.
void ModuleSanitizerCoverage::InjectCoverageForIndirectCalls(
    Function &F, ArrayRef<Instruction *> IndirCalls) {
  if (IndirCalls.empty())
    return;
  assert(Options.TracePC || Options.TracePCEntryExit || Options.TracePCGuard ||
         Options.Inline8bitCounters || Options.InlineBoolFlag);
  for (auto *I : IndirCalls) {
    InstrumentationIRBuilder IRB(I);
    CallBase &CB = cast<CallBase>(*I);
    Value *Callee = CB.getCalledOperand();
    if (isa<InlineAsm>(Callee))
      continue;
    IRB.CreateCall(SanCovTracePCIndir, IRB.CreatePointerCast(Callee, IntptrTy));
  }
}

// For every switch statement we insert a call:
// __sanitizer_cov_trace_switch(CondValue,
//      {NumCases, ValueSizeInBits, Case0Value, Case1Value, Case2Value, ... })

void ModuleSanitizerCoverage::InjectTraceForSwitch(
    Function &F, ArrayRef<Instruction *> SwitchTraceTargets,
    Value *&FunctionGateCmp) {
  for (auto *I : SwitchTraceTargets) {
    if (SwitchInst *SI = dyn_cast<SwitchInst>(I)) {
      InstrumentationIRBuilder IRB(I);
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
        ConstantInt *C = It.getCaseValue();
        if (C->getType()->getScalarSizeInBits() < 64)
          C = ConstantInt::get(C->getContext(), C->getValue().zext(64));
        Initializers.push_back(C);
      }
      llvm::sort(drop_begin(Initializers, 2),
                 [](const Constant *A, const Constant *B) {
                   return cast<ConstantInt>(A)->getLimitedValue() <
                          cast<ConstantInt>(B)->getLimitedValue();
                 });
      ArrayType *ArrayOfInt64Ty = ArrayType::get(Int64Ty, Initializers.size());
      GlobalVariable *GV = new GlobalVariable(
          *CurModule, ArrayOfInt64Ty, false, GlobalVariable::InternalLinkage,
          ConstantArray::get(ArrayOfInt64Ty, Initializers),
          "__sancov_gen_cov_switch_values");
      if (Options.GatedCallbacks) {
        auto GateBranch = CreateGateBranch(F, FunctionGateCmp, I);
        IRBuilder<> GateIRB(GateBranch);
        GateIRB.CreateCall(SanCovTraceSwitchFunction, {Cond, GV});
      } else {
        IRB.CreateCall(SanCovTraceSwitchFunction, {Cond, GV});
      }
    }
  }
}

void ModuleSanitizerCoverage::InjectTraceForDiv(
    Function &, ArrayRef<BinaryOperator *> DivTraceTargets) {
  for (auto *BO : DivTraceTargets) {
    InstrumentationIRBuilder IRB(BO);
    Value *A1 = BO->getOperand(1);
    if (isa<ConstantInt>(A1))
      continue;
    if (!A1->getType()->isIntegerTy())
      continue;
    uint64_t TypeSize = DL->getTypeStoreSizeInBits(A1->getType());
    int CallbackIdx = TypeSize == 32 ? 0 : TypeSize == 64 ? 1 : -1;
    if (CallbackIdx < 0)
      continue;
    auto Ty = Type::getIntNTy(*C, TypeSize);
    IRB.CreateCall(SanCovTraceDivFunction[CallbackIdx],
                   {IRB.CreateIntCast(A1, Ty, true)});
  }
}

void ModuleSanitizerCoverage::InjectTraceForGep(
    Function &, ArrayRef<GetElementPtrInst *> GepTraceTargets) {
  for (auto *GEP : GepTraceTargets) {
    InstrumentationIRBuilder IRB(GEP);
    for (Use &Idx : GEP->indices())
      if (!isa<ConstantInt>(Idx) && Idx->getType()->isIntegerTy())
        IRB.CreateCall(SanCovTraceGepFunction,
                       {IRB.CreateIntCast(Idx, IntptrTy, true)});
  }
}

void ModuleSanitizerCoverage::InjectTraceForLoadsAndStores(
    Function &, ArrayRef<LoadInst *> Loads, ArrayRef<StoreInst *> Stores) {
  auto CallbackIdx = [&](Type *ElementTy) -> int {
    uint64_t TypeSize = DL->getTypeStoreSizeInBits(ElementTy);
    return TypeSize == 8     ? 0
           : TypeSize == 16  ? 1
           : TypeSize == 32  ? 2
           : TypeSize == 64  ? 3
           : TypeSize == 128 ? 4
                             : -1;
  };
  for (auto *LI : Loads) {
    InstrumentationIRBuilder IRB(LI);
    auto Ptr = LI->getPointerOperand();
    int Idx = CallbackIdx(LI->getType());
    if (Idx < 0)
      continue;
    IRB.CreateCall(SanCovLoadFunction[Idx], Ptr);
  }
  for (auto *SI : Stores) {
    InstrumentationIRBuilder IRB(SI);
    auto Ptr = SI->getPointerOperand();
    int Idx = CallbackIdx(SI->getValueOperand()->getType());
    if (Idx < 0)
      continue;
    IRB.CreateCall(SanCovStoreFunction[Idx], Ptr);
  }
}

void ModuleSanitizerCoverage::InjectTraceForExits(Function &F) {
  EscapeEnumerator EE(F, "sancov_exit");
  while (IRBuilder<> *AtExit = EE.Next()) {
    InstrumentationIRBuilder::ensureDebugInfo(*AtExit, F);
    AtExit->CreateCall(SanCovTracePCExit, {})
        ->setTailCallKind(CallInst::TCK_NoTail);
  }
}

void ModuleSanitizerCoverage::InjectTraceForCmp(
    Function &F, ArrayRef<Instruction *> CmpTraceTargets,
    Value *&FunctionGateCmp) {
  for (auto *I : CmpTraceTargets) {
    if (ICmpInst *ICMP = dyn_cast<ICmpInst>(I)) {
      InstrumentationIRBuilder IRB(ICMP);
      Value *A0 = ICMP->getOperand(0);
      Value *A1 = ICMP->getOperand(1);
      if (!A0->getType()->isIntegerTy())
        continue;
      uint64_t TypeSize = DL->getTypeStoreSizeInBits(A0->getType());
      int CallbackIdx = TypeSize == 8    ? 0
                        : TypeSize == 16 ? 1
                        : TypeSize == 32 ? 2
                        : TypeSize == 64 ? 3
                                         : -1;
      if (CallbackIdx < 0)
        continue;
      // __sanitizer_cov_trace_cmp((type_size << 32) | predicate, A0, A1);
      auto CallbackFunc = SanCovTraceCmpFunction[CallbackIdx];
      bool FirstIsConst = isa<ConstantInt>(A0);
      bool SecondIsConst = isa<ConstantInt>(A1);
      // If both are const, then we don't need such a comparison.
      if (FirstIsConst && SecondIsConst)
        continue;
      // If only one is const, then make it the first callback argument.
      if (FirstIsConst || SecondIsConst) {
        CallbackFunc = SanCovTraceConstCmpFunction[CallbackIdx];
        if (SecondIsConst)
          std::swap(A0, A1);
      }

      auto Ty = Type::getIntNTy(*C, TypeSize);
      if (Options.GatedCallbacks) {
        auto GateBranch = CreateGateBranch(F, FunctionGateCmp, I);
        IRBuilder<> GateIRB(GateBranch);
        GateIRB.CreateCall(CallbackFunc, {GateIRB.CreateIntCast(A0, Ty, true),
                                          GateIRB.CreateIntCast(A1, Ty, true)});
      } else {
        IRB.CreateCall(CallbackFunc, {IRB.CreateIntCast(A0, Ty, true),
                                      IRB.CreateIntCast(A1, Ty, true)});
      }
    }
  }
}

void ModuleSanitizerCoverage::InjectCoverageAtBlock(Function &F, BasicBlock &BB,
                                                    size_t Idx,
                                                    Value *&FunctionGateCmp,
                                                    bool IsLeafFunc) {
  BasicBlock::iterator IP = BB.getFirstInsertionPt();
  bool IsEntryBB = &BB == &F.getEntryBlock();
  DebugLoc EntryLoc;
  if (IsEntryBB) {
    if (auto SP = F.getSubprogram())
      EntryLoc = DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP);
    // Keep static allocas and llvm.localescape calls in the entry block.  Even
    // if we aren't splitting the block, it's nice for allocas to be before
    // calls.
    IP = PrepareToSplitEntryBlock(BB, IP);
  }

  InstrumentationIRBuilder IRB(&*IP);
  if (EntryLoc)
    IRB.SetCurrentDebugLocation(EntryLoc);
  if (Options.TracePC || (IsEntryBB && Options.TracePCEntryExit)) {
    FunctionCallee Callee = IsEntryBB && Options.TracePCEntryExit
                                ? SanCovTracePCEntry
                                : SanCovTracePC;
    IRB.CreateCall(Callee)
        ->setCannotMerge(); // gets the PC using GET_CALLER_PC.
  }
  if (Options.TracePCGuard) {
    auto GuardPtr = IRB.CreateConstInBoundsGEP2_64(
        FunctionGuardArray->getValueType(), FunctionGuardArray, 0, Idx);
    if (Options.GatedCallbacks) {
      Instruction *I = &*IP;
      auto GateBranch = CreateGateBranch(F, FunctionGateCmp, I);
      IRBuilder<> GateIRB(GateBranch);
      GateIRB.CreateCall(SanCovTracePCGuard, GuardPtr)->setCannotMerge();
    } else {
      IRB.CreateCall(SanCovTracePCGuard, GuardPtr)->setCannotMerge();
    }
  }
  if (Options.Inline8bitCounters) {
    auto CounterPtr = IRB.CreateGEP(
        Function8bitCounterArray->getValueType(), Function8bitCounterArray,
        {ConstantInt::get(IntptrTy, 0), ConstantInt::get(IntptrTy, Idx)});
    auto Load = IRB.CreateLoad(Int8Ty, CounterPtr);
    auto Inc = IRB.CreateAdd(Load, ConstantInt::get(Int8Ty, 1));
    auto Store = IRB.CreateStore(Inc, CounterPtr);
    Load->setNoSanitizeMetadata();
    Store->setNoSanitizeMetadata();
  }
  if (Options.InlineBoolFlag) {
    auto FlagPtr = IRB.CreateGEP(
        FunctionBoolArray->getValueType(), FunctionBoolArray,
        {ConstantInt::get(IntptrTy, 0), ConstantInt::get(IntptrTy, Idx)});
    auto Load = IRB.CreateLoad(Int1Ty, FlagPtr);
    auto ThenTerm = SplitBlockAndInsertIfThen(
        IRB.CreateIsNull(Load), &*IP, false,
        MDBuilder(IRB.getContext()).createUnlikelyBranchWeights());
    InstrumentationIRBuilder ThenIRB(ThenTerm);
    auto Store = ThenIRB.CreateStore(ConstantInt::getTrue(Int1Ty), FlagPtr);
    if (EntryLoc)
      Store->setDebugLoc(EntryLoc);
    Load->setNoSanitizeMetadata();
    Store->setNoSanitizeMetadata();
  }
  if (Options.StackDepth && IsEntryBB && !IsLeafFunc) {
    Module *M = F.getParent();
    const DataLayout &DL = M->getDataLayout();

    if (Options.StackDepthCallbackMin) {
      // In callback mode, only add call when stack depth reaches minimum.
      int EstimatedStackSize = 0;
      // If dynamic alloca found, always add call.
      bool HasDynamicAlloc = false;
      // Find an insertion point after last "alloca".
      llvm::Instruction *InsertBefore = nullptr;

      // Examine all allocas in the basic block. since we're too early
      // to have results from Intrinsic::frameaddress, we have to manually
      // estimate the stack size.
      for (auto &I : BB) {
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          // Move potential insertion point past the "alloca".
          InsertBefore = AI->getNextNode();

          // Make an estimate on the stack usage.
          if (auto AllocaSize = AI->getAllocationSize(DL)) {
            if (AllocaSize->isFixed())
              EstimatedStackSize += AllocaSize->getFixedValue();
            else
              HasDynamicAlloc = true;
          } else {
            HasDynamicAlloc = true;
          }
        }
      }

      if (HasDynamicAlloc ||
          EstimatedStackSize >= Options.StackDepthCallbackMin) {
        if (InsertBefore)
          IRB.SetInsertPoint(InsertBefore);
        auto Call = IRB.CreateCall(SanCovStackDepthCallback);
        if (EntryLoc)
          Call->setDebugLoc(EntryLoc);
        Call->setCannotMerge();
      }
    } else {
      // Check stack depth.  If it's the deepest so far, record it.
      auto FrameAddrPtr = IRB.CreateIntrinsic(
          Intrinsic::frameaddress, IRB.getPtrTy(DL.getAllocaAddrSpace()),
          {Constant::getNullValue(Int32Ty)});
      auto FrameAddrInt = IRB.CreatePtrToInt(FrameAddrPtr, IntptrTy);
      auto LowestStack = IRB.CreateLoad(IntptrTy, SanCovLowestStack);
      auto IsStackLower = IRB.CreateICmpULT(FrameAddrInt, LowestStack);
      auto ThenTerm = SplitBlockAndInsertIfThen(
          IsStackLower, &*IP, false,
          MDBuilder(IRB.getContext()).createUnlikelyBranchWeights());
      InstrumentationIRBuilder ThenIRB(ThenTerm);
      auto Store = ThenIRB.CreateStore(FrameAddrInt, SanCovLowestStack);
      if (EntryLoc)
        Store->setDebugLoc(EntryLoc);
      LowestStack->setNoSanitizeMetadata();
      Store->setNoSanitizeMetadata();
    }
  }
}

std::string
ModuleSanitizerCoverage::getSectionName(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatCOFF()) {
    if (Section == SanCovCountersSectionName)
      return ".SCOV$CM";
    if (Section == SanCovBoolFlagSectionName)
      return ".SCOV$BM";
    if (Section == SanCovPCsSectionName)
      return ".SCOVP$M";
    return ".SCOV$GM"; // For SanCovGuardsSectionName.
  }
  if (TargetTriple.isOSBinFormatMachO())
    return "__DATA,__" + Section;
  return "__" + Section;
}

std::string
ModuleSanitizerCoverage::getSectionStart(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatMachO())
    return "\1section$start$__DATA$__" + Section;
  return "__start___" + Section;
}

std::string
ModuleSanitizerCoverage::getSectionEnd(const std::string &Section) const {
  if (TargetTriple.isOSBinFormatMachO())
    return "\1section$end$__DATA$__" + Section;
  return "__stop___" + Section;
}

void ModuleSanitizerCoverage::createFunctionControlFlow(Function &F) {
  SmallVector<Constant *, 32> CFs;
  IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());

  for (auto &BB : F) {
    // blockaddress can not be used on function's entry block.
    if (&BB == &F.getEntryBlock())
      CFs.push_back((Constant *)IRB.CreatePointerCast(&F, PtrTy));
    else
      CFs.push_back(
          (Constant *)IRB.CreatePointerCast(BlockAddress::get(&BB), PtrTy));

    for (auto SuccBB : successors(&BB)) {
      assert(SuccBB != &F.getEntryBlock());
      CFs.push_back(
          (Constant *)IRB.CreatePointerCast(BlockAddress::get(SuccBB), PtrTy));
    }

    CFs.push_back((Constant *)Constant::getNullValue(PtrTy));

    for (auto &Inst : BB) {
      if (CallBase *CB = dyn_cast<CallBase>(&Inst)) {
        if (CB->isIndirectCall()) {
          // TODO(navidem): handle indirect calls, for now mark its existence.
          CFs.push_back((Constant *)IRB.CreateIntToPtr(
              ConstantInt::getAllOnesValue(IntptrTy), PtrTy));
        } else {
          auto CalledF = CB->getCalledFunction();
          if (CalledF && !CalledF->isIntrinsic())
            CFs.push_back((Constant *)IRB.CreatePointerCast(CalledF, PtrTy));
        }
      }
    }

    CFs.push_back((Constant *)Constant::getNullValue(PtrTy));
  }

  FunctionCFsArray = CreateFunctionLocalArrayInSection(CFs.size(), F, PtrTy,
                                                       SanCovCFsSectionName);
  FunctionCFsArray->setInitializer(
      ConstantArray::get(ArrayType::get(PtrTy, CFs.size()), CFs));
  FunctionCFsArray->setConstant(true);
}

//===----------------------------------------------------------------------===//
// Argument and return value tracing.
//===----------------------------------------------------------------------===//
//
// -sanitizer-coverage-trace-args reports every source-level parameter of an
// instrumented function on entry, -sanitizer-coverage-trace-ret reports the
// value of every return:
//
//   void __sanitizer_cov_trace_args(u64 pc, u32 arg_idx, u32 size, u64 val,
//                                   u64 *offsets, u32 num_fields);
//   void __sanitizer_cov_trace_ret (u64 pc, u32 size, u64 val,
//                                   u64 *offsets, u32 num_fields);
//
// `pc` is the address of the instrumented function. What `val` holds depends on
// `num_fields`:
//
// - `num_fields == 0`: `val` is the value itself, the low `size` bytes of it,
//   zero-extended. Integers, pointers and floating-point values are reported
//   this way, which is nearly everything, and needs no memory of its own.
//
// - `num_fields != 0`: `val` is the address of a `size` byte object and
//   `offsets` points at a constant table of `num_fields` {byte offset, byte
//   size} pairs describing its fields, so that a consumer can read them out of
//   it. Only an object that already lives in memory is reported this way: a
//   pointer to a struct, a by-value struct the ABI passes indirectly, or the
//   buffer of an indirect struct return. The pass never creates memory to
//   report a value from, so no stack slot escapes and no frame is realigned on
//   its account.
//
// `size == 0` means the pass had nothing to report - a parameter the optimizer
// removed, a value of a type it cannot widen, a void return.
//
// `arg_idx` is the zero-based *source-level* parameter index. Argument values
// are recovered from debug records rather than from the IR argument list,
// because the two disagree as soon as the ABI rewrites the signature; see
// collectSourceParams(). A function without debug records falls back to the IR
// argument list, which keeps both modes usable on code built without -g, at
// the price of exposing the ABI's view of the arguments.

/// Peel the typedefs and qualifiers off \p Ty to reach the type they decorate.
static DIType *stripTypedefsAndQualifiers(DIType *Ty) {
  while (auto *Derived = dyn_cast_or_null<DIDerivedType>(Ty)) {
    switch (Derived->getTag()) {
    case dwarf::DW_TAG_typedef:
    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_volatile_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_atomic_type:
    case dwarf::DW_TAG_immutable_type:
      Ty = Derived->getBaseType();
      continue;
    default:
      return Ty;
    }
  }
  return Ty;
}

/// Return the aggregate whose fields describe \p Ty: either \p Ty itself, for
/// a by-value struct, or its pointee, for a pointer to one. Returns null for
/// any other type.
static DICompositeType *getTracedStructType(DIType *Ty) {
  Ty = stripTypedefsAndQualifiers(Ty);
  if (auto *Derived = dyn_cast_or_null<DIDerivedType>(Ty))
    if (Derived->getTag() == dwarf::DW_TAG_pointer_type)
      Ty = stripTypedefsAndQualifiers(Derived->getBaseType());

  auto *Composite = dyn_cast_or_null<DICompositeType>(Ty);
  if (!Composite)
    return nullptr;
  switch (Composite->getTag()) {
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
    return Composite;
  default:
    return nullptr;
  }
}

/// Number of parameters \p SP declares in source, or 0 if that is unknown.
static unsigned getNumDeclaredParams(DISubprogram *SP) {
  if (!SP || !SP->getType())
    return 0;
  // The type array is {return type, parameter types...}.
  unsigned Size = SP->getType()->getTypeArray().size();
  return Size ? Size - 1 : 0;
}

/// Declared type of \p SP's \p Idx-th parameter, numbered from 1 as DWARF
/// numbers parameters, or null if it is unknown.
static DIType *getDeclaredParamType(DISubprogram *SP, unsigned Idx) {
  if (Idx == 0 || Idx > getNumDeclaredParams(SP))
    return nullptr;
  return SP->getType()->getTypeArray()[Idx];
}

/// Declared return type of \p SP, or null if it is unknown or void.
static DIType *getDeclaredReturnType(DISubprogram *SP) {
  if (!SP || !SP->getType() || SP->getType()->getTypeArray().empty())
    return nullptr;
  return SP->getType()->getTypeArray()[0];
}

ModuleSanitizerCoverage::FieldOffsets
ModuleSanitizerCoverage::getFieldOffsets(DIType *Ty) {
  DICompositeType *Composite = getTracedStructType(Ty);
  if (!Composite)
    return {};
  if (auto It = FieldOffsetsCache.find(Composite);
      It != FieldOffsetsCache.end())
    return It->second;

  SmallVector<Constant *, 16> Pairs;
  for (DINode *Element : Composite->getElements()) {
    auto *Member = dyn_cast<DIDerivedType>(Element);
    if (!Member || Member->getTag() != dwarf::DW_TAG_member ||
        Member->isStaticMember())
      continue;
    uint64_t SizeInBits = Member->getSizeInBits();
    if (!SizeInBits)
      continue; // A flexible array member or an empty base class.
    // The callbacks describe fields in bytes, so widen a bitfield to the bytes
    // that hold it instead of reporting a zero-sized field.
    uint64_t FirstByte = Member->getOffsetInBits() / 8;
    uint64_t EndByte = divideCeil(Member->getOffsetInBits() + SizeInBits, 8);
    Pairs.push_back(ConstantInt::get(Int64Ty, FirstByte));
    Pairs.push_back(ConstantInt::get(Int64Ty, EndByte - FirstByte));
  }

  FieldOffsets Offsets;
  if (!Pairs.empty()) {
    ArrayType *TableTy = ArrayType::get(Int64Ty, Pairs.size());
    auto *Table = new GlobalVariable(
        M, TableTy, /*isConstant=*/true, GlobalVariable::PrivateLinkage,
        ConstantArray::get(TableTy, Pairs), "__sancov_offsets_");
    Table->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    Offsets = {Table, static_cast<unsigned>(Pairs.size() / 2),
               divideCeil(Composite->getSizeInBits(), 8)};
  }
  FieldOffsetsCache[Composite] = Offsets;
  return Offsets;
}

namespace {

/// One of the IR values that hold a source-level object, together with its bit
/// offset in that object. A scalar has a single piece at offset 0.
using ValuePiece = std::pair<Value *, uint64_t>;

/// A source-level parameter of a function and the entry-block values that hold
/// it. The ABI may coerce one parameter into several values, each described by
/// a DW_OP_LLVM_fragment debug record.
struct SourceParam {
  DIType *Ty = nullptr;
  SmallVector<ValuePiece, 2> Pieces;
  /// Whether Pieces holds fragments of the parameter rather than the whole of
  /// it. A parameter can lose a fragment (see collectSourceParams), so the
  /// number of pieces alone does not answer this.
  bool Fragmented = false;
  /// Whether Pieces holds incoming Arguments rather than derived values.
  bool FromArguments = false;

  /// Record that \p V holds this parameter's bits from \p BitOffset on.
  ///
  /// A record naming an incoming Argument describes the parameter as it was
  /// passed, so it wins over records naming values derived from it. Exact
  /// duplicates are dropped: a repeated record would otherwise make a scalar
  /// look like a multi-piece aggregate and be needlessly reassembled.
  void addPiece(Value *V, uint64_t BitOffset, bool IsFragment) {
    bool IsArgument = isa<Argument>(V);
    if (IsArgument && !FromArguments) {
      Pieces.clear();
      Fragmented = false;
      FromArguments = true;
    } else if (!IsArgument && FromArguments) {
      return;
    }
    if (is_contained(Pieces, ValuePiece(V, BitOffset)))
      return;
    Pieces.emplace_back(V, BitOffset);
    Fragmented |= IsFragment;
  }
};

using SourceParamMap = SmallDenseMap<unsigned, SourceParam, 8>;

} // namespace

/// Map the source-level parameters of \p F to the values that hold them on
/// entry, keyed by the DWARF parameter number (counted from 1).
///
/// The frontend numbers parameters, in DILocalVariable::getArg(), before ABI
/// lowering, so the number keeps naming the same source parameter even when
/// the ABI inserts a hidden argument or splits an aggregate across several.
/// Debug records are the only link back to that numbering, which is why they,
/// and not the IR argument list, drive trace-args.
///
/// A parameter is left out of \p Params when it has no location the trace call
/// can use; the caller reports those with a null value pointer.
static void collectSourceParams(Function &F, SourceParamMap &Params) {
  BasicBlock &EntryBB = F.getEntryBlock();
  DISubprogram *SP = F.getSubprogram();

  for (Instruction &I : instructions(F)) {
    for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
      DILocalVariable *Var = DVR.getVariable();
      if (!Var || !Var->getArg())
        continue;
      // Only this function's own parameters are numbered for us. An inlined
      // callee's parameters are numbered too, and its records may name our
      // Arguments: kmalloc(size, flags) inlined into f(ptr, size) leaves
      // #dbg_value(%size, "size", arg: 1) behind, which would otherwise be
      // taken as a description of f's first parameter.
      if (DVR.getDebugLoc() && DVR.getDebugLoc().getInlinedAt())
        continue;
      if (SP && Var->getScope() && Var->getScope()->getSubprogram() != SP)
        continue;
      // A #dbg_declare names the parameter's storage rather than its incoming
      // value. That storage is written by the prologue, which follows the trace
      // call, so reading it here would report whatever the stack held.
      if (DVR.isDbgDeclare())
        continue;

      // Only a location that is the value itself can be reported. An
      // expression that computes the value - a field shifted out of a wider
      // register, an offset applied to a pointer - cannot be replayed into the
      // spill slot, and storing the value it starts from would write the wrong
      // bytes and overrun the piece.
      DIExpression *Expr = DVR.getExpression();
      std::optional<DIExpression::FragmentInfo> Fragment =
          Expr->getFragmentInfo();
      if (Expr->getNumElements() != (Fragment ? 3u : 0u))
        continue;

      Value *V = DVR.getValue();
      if (!V)
        continue;
      // A parameter that interprocedural optimization proved dead is described
      // as poison, and callers pass poison for it: there is nothing to report.
      if (isa<UndefValue>(V))
        continue;
      // Debug records are exempt from SSA dominance, so a record may name a
      // value that is not available where the trace call goes. Reporting it
      // would produce IR failing the verifier with "Instruction does not
      // dominate all uses".
      if (auto *Def = dyn_cast<Instruction>(V))
        if (Def->getParent() != &EntryBB || Def->isTerminator())
          continue;

      SourceParam &Param = Params[Var->getArg()];
      Param.Ty = Var->getType();
      Param.addPiece(V, Fragment ? Fragment->OffsetInBits : 0,
                     Fragment.has_value());
    }
  }
}

/// Widen \p V to the 64-bit value the callbacks take: a pointer is reported as
/// the address it holds, a floating-point value as its bit pattern, and a value
/// wider than 64 bits by its low half. Returns the value together with the
/// number of bytes of it that are meaningful, or {nullptr, 0} for a type that
/// cannot be widened - a vector, or an aggregate the ABI passes in registers.
static std::pair<Value *, uint64_t> getReportedValue(IRBuilderBase &IRB,
                                                     Value *V) {
  const DataLayout &DL = IRB.GetInsertBlock()->getDataLayout();
  Type *Ty = V->getType();
  IntegerType *Int64Ty = IRB.getInt64Ty();

  // ptrtoint widens or narrows to the requested type, so this also holds for a
  // target whose pointers are narrower than 64 bits.
  if (Ty->isPointerTy())
    return {IRB.CreatePtrToInt(V, Int64Ty), DL.getPointerSize()};

  if (Ty->isFloatingPointTy()) {
    unsigned Bits = Ty->getPrimitiveSizeInBits();
    V = IRB.CreateBitCast(V, IRB.getIntNTy(Bits));
    Ty = V->getType();
  }
  if (!Ty->isIntegerTy())
    return {nullptr, 0};

  uint64_t Size = divideCeil(Ty->getIntegerBitWidth(), 8);
  if (Size > sizeof(uint64_t)) {
    // A value that does not fit is reported by its low bytes rather than not at
    // all: they are what a comparison against it looks at first.
    V = IRB.CreateTrunc(V, Int64Ty);
    Size = sizeof(uint64_t);
  } else {
    V = IRB.CreateZExt(V, Int64Ty);
  }
  return {V, Size};
}

/// Widen \p V for the callbacks, splitting a value that the ABI passed as a
/// first-class aggregate into its elements: those live in separate registers
/// and share no address, so each is reported on its own. Appends at least one
/// entry, {nullptr, 0} for a value that cannot be reported at all.
static void
getReportedValues(IRBuilderBase &IRB, Value *V,
                  SmallVectorImpl<std::pair<Value *, uint64_t>> &Out) {
  size_t First = Out.size();
  Type *Ty = V->getType();
  unsigned NumElements = Ty->isStructTy()  ? Ty->getStructNumElements()
                         : Ty->isArrayTy() ? Ty->getArrayNumElements()
                                           : 0;
  if (NumElements)
    for (unsigned I = 0; I != NumElements; ++I)
      Out.push_back(getReportedValue(IRB, IRB.CreateExtractValue(V, I)));
  else
    Out.push_back(getReportedValue(IRB, V));

  if (Out.size() == First)
    Out.emplace_back(nullptr, 0);
}

void ModuleSanitizerCoverage::InjectTraceForArgs(Function &F) {
  DISubprogram *SP = F.getSubprogram();
  BasicBlock &EntryBB = F.getEntryBlock();

  SourceParamMap Params;
  collectSourceParams(F, Params);

  // Trace as early as the reported values allow: at the first insertion point
  // of the entry block, pushed down past the definition of every value that is
  // reported so that it dominates the call.
  BasicBlock::iterator IP = EntryBB.getFirstInsertionPt();
  for (const auto &[Idx, Param] : Params)
    for (const auto &[V, BitOffset] : Param.Pieces)
      if (auto *Def = dyn_cast<Instruction>(V); Def && !Def->comesBefore(&*IP))
        IP = std::next(Def->getIterator());

  // InstrumentationIRBuilder gives the calls a synthetic !dbg location. A
  // plain IRBuilder would leave them without one, which fails the verifier
  // ("inlinable function call ... requires a !dbg location") once a function
  // built with -g is inlined.
  InstrumentationIRBuilder IRB(&*IP);
  Value *PC = IRB.CreatePtrToInt(&F, Int64Ty);

  auto trace = [&](unsigned Idx, Value *Val, uint64_t Size,
                   FieldOffsets Offsets) {
    IRB.CreateCall(
        SanCovTraceArgsFunc,
        {PC, ConstantInt::get(Int32Ty, Idx - 1),
         ConstantInt::get(Int32Ty, Size),
         Val ? Val : ConstantInt::get(Int64Ty, 0),
         Offsets.Table ? Offsets.Table : ConstantPointerNull::get(PtrTy),
         ConstantInt::get(Int32Ty, Offsets.NumFields)});
  };

  // Report \p V as the value of the \p Idx-th parameter, declared as \p Ty.
  // A pointer to a struct whose fields are known is reported as the address of
  // that struct, so that a consumer can read the fields out of it; every other
  // value is reported as the value itself. \p WholeObject says whether \p V is
  // the entire parameter, which a fragment of a struct the ABI split across
  // registers is not - such a fragment may well be a pointer, but it is not the
  // address of the struct its type describes.
  auto traceValue = [&](unsigned Idx, Value *V, DIType *Ty, bool WholeObject) {
    // Ask for the field table only on the path that uses it: building one for a
    // struct that ends up reported by value would leave an unreferenced global
    // behind in every object file.
    if (WholeObject && V->getType()->isPointerTy())
      if (FieldOffsets Offsets = getFieldOffsets(Ty); Offsets.Table) {
        trace(Idx, IRB.CreatePtrToInt(V, Int64Ty), Offsets.ObjectSize, Offsets);
        return;
      }
    SmallVector<std::pair<Value *, uint64_t>, 2> Values;
    getReportedValues(IRB, V, Values);
    for (auto [Val, Size] : Values)
      trace(Idx, Val, Size, {});
  };

  // Without debug records there is nothing to map the IR arguments back to, so
  // report them positionally and let the declared parameter types, if there
  // are any, describe their fields. ABI lowering is visible to the consumer in
  // this mode: a coerced aggregate is reported as the values it was coerced
  // into, and the indices of the parameters after it shift accordingly.
  if (Params.empty()) {
    unsigned Idx = 1;
    for (Argument &Arg : F.args()) {
      // A struct-return pointer has no source-level counterpart.
      if (Arg.hasStructRetAttr())
        continue;
      traceValue(Idx, &Arg, getDeclaredParamType(SP, Idx),
                 /*WholeObject=*/true);
      ++Idx;
    }
    return;
  }

  // One call per source-level parameter, in source order, except that a
  // parameter the ABI split across registers is reported once per piece: the
  // pieces have no common address to report them from. A parameter with no
  // usable location is still reported, with size 0, so that a consumer sees
  // every parameter the function declares.
  unsigned NumParams = getNumDeclaredParams(SP);
  for (const auto &[Idx, Param] : Params)
    NumParams = std::max(NumParams, Idx);

  for (unsigned Idx = 1; Idx <= NumParams; ++Idx) {
    auto It = Params.find(Idx);
    if (It == Params.end() || It->second.Pieces.empty()) {
      trace(Idx, nullptr, 0, {});
      continue;
    }
    const SourceParam &Param = It->second;
    bool WholeObject = Param.Pieces.size() == 1 && !Param.Fragmented;
    for (const auto &[V, BitOffset] : Param.Pieces)
      traceValue(Idx, V, Param.Ty, WholeObject);
  }
}

void ModuleSanitizerCoverage::InjectTraceForRet(Function &F) {
  DIType *RetTy = getDeclaredReturnType(F.getSubprogram());

  // A struct returned by value may be lowered to an indirect return: the IR
  // function returns void and writes the result through a hidden struct-return
  // pointer. Report that buffer, so an indirect return is not dropped. This
  // mirrors the argument side, which skips the same pointer.
  Argument *SRetArg = nullptr;
  for (Argument &Arg : F.args())
    if (Arg.hasStructRetAttr()) {
      SRetArg = &Arg;
      break;
    }

  for (BasicBlock &BB : F) {
    // Only a return carries a value to report; unwinding leaves the function
    // without one.
    auto *RI = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!RI)
      continue;
    // A musttail call has to stay adjacent to the return that forwards it, so
    // there is nowhere to put the call.
    if (auto *CI = dyn_cast_or_null<CallInst>(RI->getPrevNode());
        CI && CI->isMustTailCall())
      continue;

    InstrumentationIRBuilder IRB(RI);
    Value *PC = IRB.CreatePtrToInt(&F, Int64Ty);

    auto trace = [&](Value *Val, uint64_t Size, FieldOffsets Offsets) {
      IRB.CreateCall(
          SanCovTraceRetFunc,
          {PC, ConstantInt::get(Int32Ty, Size),
           Val ? Val : ConstantInt::get(Int64Ty, 0),
           Offsets.Table ? Offsets.Table : ConstantPointerNull::get(PtrTy),
           ConstantInt::get(Int32Ty, Offsets.NumFields)});
    };

    Value *RetVal = RI->getReturnValue();
    // The value is reported by address when it is one: a pointer whose pointee
    // fields are known, or the buffer of an indirect struct return. A struct
    // returned in registers has no address, so it is reported as its elements.
    if (RetVal ? RetVal->getType()->isPointerTy() : SRetArg != nullptr) {
      Value *Object = RetVal ? RetVal : SRetArg;
      if (FieldOffsets Offsets = getFieldOffsets(RetTy); Offsets.Table) {
        trace(IRB.CreatePtrToInt(Object, Int64Ty), Offsets.ObjectSize, Offsets);
        continue;
      }
      // An indirect return whose field layout is unknown is a struct in memory,
      // not a value that fits in a register, so there is nothing to report.
      if (!RetVal) {
        trace(nullptr, 0, {});
        continue;
      }
    }

    if (!RetVal) {
      trace(nullptr, 0, {}); // A void return.
      continue;
    }
    SmallVector<std::pair<Value *, uint64_t>, 2> Values;
    getReportedValues(IRB, RetVal, Values);
    for (auto [Val, Size] : Values)
      trace(Val, Size, {});
  }
}
