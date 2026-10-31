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
#include "InstrumentationOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
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

SanitizerCoverageOptions overrideFromCL(const InstrumentationOptions &Opts,
                                        SanitizerCoverageOptions Options) {
  // Sets CoverageType and IndirectCalls.
  SanitizerCoverageOptions CLOpts = getOptions(Opts.sanitizer_coverage_level);
  Options.CoverageType = std::max(Options.CoverageType, CLOpts.CoverageType);
  Options.IndirectCalls |= CLOpts.IndirectCalls;
  Options.TraceCmp |= Opts.sanitizer_coverage_trace_compares;
  Options.TraceDiv |= Opts.sanitizer_coverage_trace_divs;
  Options.TraceGep |= Opts.sanitizer_coverage_trace_geps;
  Options.TracePC |= Opts.sanitizer_coverage_trace_pc;
  Options.TracePCEntryExit |= Opts.sanitizer_coverage_trace_pc_entry_exit;
  Options.TracePCGuard |= Opts.sanitizer_coverage_trace_pc_guard;
  Options.Inline8bitCounters |= Opts.sanitizer_coverage_inline_8bit_counters;
  Options.InlineBoolFlag |= Opts.sanitizer_coverage_inline_bool_flag;
  Options.PCTable |= Opts.sanitizer_coverage_pc_table;
  Options.NoPrune |= !Opts.sanitizer_coverage_prune_blocks;
  Options.StackDepth |= Opts.sanitizer_coverage_stack_depth;
  Options.StackDepthCallbackMin =
      std::max(Options.StackDepthCallbackMin,
               Opts.sanitizer_coverage_stack_depth_callback_min);
  Options.TraceLoads |= Opts.sanitizer_coverage_trace_loads;
  Options.TraceStores |= Opts.sanitizer_coverage_trace_stores;
  Options.TraceArgs |= Opts.sanitizer_coverage_trace_args;
  Options.TraceRet |= Opts.sanitizer_coverage_trace_ret;
  Options.GatedCallbacks |= Opts.sanitizer_coverage_gated_trace_callbacks;
  if (!Options.TracePCGuard && !Options.TracePC && !Options.TracePCEntryExit &&
      !Options.Inline8bitCounters && !Options.StackDepth &&
      !Options.InlineBoolFlag && !Options.TraceLoads && !Options.TraceStores &&
      !Options.TraceArgs && !Options.TraceRet)
    Options.TracePCGuard = true; // TracePCGuard is default.
  Options.CollectControlFlow |= Opts.sanitizer_coverage_control_flow;
  return Options;
}

class ModuleSanitizerCoverage {
public:
  using DomTreeCallback = function_ref<const DominatorTree &(Function &F)>;
  using PostDomTreeCallback =
      function_ref<const PostDominatorTree &(Function &F)>;

  ModuleSanitizerCoverage(const InstrumentationOptions &Opts, Module &M,
                          DomTreeCallback DTCallback,
                          PostDomTreeCallback PDTCallback,
                          const SanitizerCoverageOptions &Options,
                          const SpecialCaseList *Allowlist,
                          const SpecialCaseList *Blocklist)
      : Opts(Opts), M(M), DTCallback(DTCallback), PDTCallback(PDTCallback),
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

  /// The `offsets` argument of the trace-args/trace-ret callbacks for an
  /// object in memory: a constant table whose first element is the number of
  /// fields, followed by [byte offset, byte size] pairs covering the elements
  /// of its IR type, and the store size of that type. Table is null for an
  /// object of no fixed size, in which case ObjectSize is 0.
  struct FieldOffsets {
    Constant *Table = nullptr;
    uint64_t ObjectSize = 0;
  };
  FieldOffsets getObjectOffsets(Type *Ty);
  /// Create one trace-args/trace-ret call, which is never a tail call and
  /// cannot be merged.
  void emitTraceCall(IRBuilderBase &IRB, FunctionCallee Callee,
                     std::optional<uint32_t> ArgIdx, uint64_t Size, Value *Val,
                     Value *Table = nullptr);
  /// Emit the trace-args/trace-ret call for \p V through \p IRB: one call per
  /// element of a first-class aggregate, and one for anything else.
  void traceValue(IRBuilderBase &IRB, FunctionCallee Callee,
                  std::optional<uint32_t> ArgIdx, Value *V);
  void traceObject(IRBuilderBase &IRB, FunctionCallee Callee,
                   std::optional<uint32_t> ArgIdx, Value *Addr, Type *Ty);
  /// Report pointer \p Ptr together with the first \p Bytes bytes it points
  /// to, which the IR says are dereferenceable.
  void tracePointee(IRBuilderBase &IRB, FunctionCallee Callee,
                    std::optional<uint32_t> ArgIdx, Value *Ptr, uint64_t Bytes);
  /// How many bytes behind \p Ptr to report, given that the IR says
  /// \p Dereferenceable of them may be read: none if \p Ptr is not a pointer
  /// in the default address space.
  uint64_t getPointeeBytes(Value *Ptr, uint64_t Dereferenceable) const;

  const InstrumentationOptions &Opts;
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

  /// Object tables are shared by every object of the same IR type: a type
  /// passed by hundreds of functions must not emit hundreds of identical
  /// tables.
  DenseMap<Type *, FieldOffsets> ObjectOffsetsCache;

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
  const InstrumentationOptions &Opts = getInstrumentationOptions(M);
  ModuleSanitizerCoverage ModuleSancov(Opts, M, DTCallback, PDTCallback,
                                       overrideFromCL(Opts, Options),
                                       Allowlist.get(), Blocklist.get());
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
  if (Opts.sanitizer_coverage_drop_ctors)
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
      C->emitError("'sanitizer-coverage-gated-trace-callbacks' is only "
                   "supported with trace-pc-guard or trace-cmp");
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
  // void __sanitizer_cov_trace_args(u64 tag, u64 val, const u64 *offsets)
  SanCovTraceArgsFunc = M.getOrInsertFunction(SanCovTraceArgsName, VoidTy,
                                              Int64Ty, Int64Ty, PtrTy);
  // void __sanitizer_cov_trace_ret(u64 size, u64 val, const u64 *offsets)
  SanCovTraceRetFunc = M.getOrInsertFunction(SanCovTraceRetName, VoidTy,
                                             Int64Ty, Int64Ty, PtrTy);

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
  if (isa<UnreachableInst>(BB->getFirstNonPHIOrLifetime()))
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

  // Before InjectCoverage, which then puts its entry-block instrumentation
  // ahead of the argument calls: allocas; pc_entry; trace_args.
  if (Options.TraceArgs)
    InjectTraceForArgs(F);

  Value *FunctionGateCmp = nullptr;
  InjectCoverage(F, BlocksToInstrument, FunctionGateCmp, IsLeafFunc);
  InjectCoverageForIndirectCalls(F, IndirCalls);
  InjectTraceForCmp(F, CmpTraceTargets, FunctionGateCmp);
  InjectTraceForSwitch(F, SwitchTraceTargets, FunctionGateCmp);
  InjectTraceForDiv(F, DivTraceTargets);
  InjectTraceForGep(F, GepTraceTargets);
  InjectTraceForLoadsAndStores(F, Loads, Stores);

  // Before InjectTraceForExits, so that a return reads trace_ret; pc_exit; ret.
  if (Options.TraceRet)
    InjectTraceForRet(F);

  if (Options.TracePCEntryExit)
    InjectTraceForExits(F);
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
// See clang/docs/SanitizerCoverage.md for the full design and callback
// contract:
//   void __sanitizer_cov_trace_args(u64 tag, u64 val, const u64 *offsets);
//   void __sanitizer_cov_trace_ret (u64 size, u64 val, const u64 *offsets);
//===----------------------------------------------------------------------===//

ModuleSanitizerCoverage::FieldOffsets
ModuleSanitizerCoverage::getObjectOffsets(Type *Ty) {
  // trace-args packs the size into the low 32 bits of its tag.
  if (!Ty || !Ty->isSized() || DL->getTypeStoreSize(Ty).isScalable() ||
      DL->getTypeStoreSize(Ty).getFixedValue() > UINT32_MAX)
    return {};
  auto [It, Inserted] = ObjectOffsetsCache.try_emplace(Ty);
  if (!Inserted)
    return It->second;

  // A struct is described by its elements, recursively; anything else as one
  // field, or in 8-byte pieces if it is wider, so that a runtime reading every
  // field into a 64-bit word reads all of the object. Element 0 is the number
  // of pairs, filled in below.
  SmallVector<Constant *, 17> Pairs = {nullptr};
  auto AddField = [&](uint64_t Offset, uint64_t Bytes) {
    for (uint64_t Piece = 0; Piece < Bytes; Piece += sizeof(uint64_t)) {
      Pairs.push_back(ConstantInt::get(Int64Ty, Offset + Piece));
      Pairs.push_back(ConstantInt::get(
          Int64Ty, std::min<uint64_t>(Bytes - Piece, sizeof(uint64_t))));
    }
  };
  auto AddType = [&](auto &Self, Type *T, uint64_t Offset) -> void {
    if (auto *STy = dyn_cast<StructType>(T)) {
      const StructLayout *Layout = DL->getStructLayout(STy);
      for (unsigned I = 0, E = STy->getNumElements(); I != E; ++I)
        Self(Self, STy->getElementType(I),
             Offset + Layout->getElementOffset(I).getFixedValue());
      return;
    }
    AddField(Offset, DL->getTypeStoreSize(T).getFixedValue());
  };
  uint64_t Size = DL->getTypeStoreSize(Ty).getFixedValue();
  AddType(AddType, Ty, 0);

  if (Pairs.size() > 1) {
    Pairs[0] = ConstantInt::get(Int64Ty, Pairs.size() / 2);
    ArrayType *TableTy = ArrayType::get(Int64Ty, Pairs.size());
    auto *Table = new GlobalVariable(
        M, TableTy, /*isConstant=*/true, GlobalVariable::PrivateLinkage,
        ConstantArray::get(TableTy, Pairs), "__sancov_offsets_");
    Table->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    It->second = {Table, Size};
  }
  return It->second;
}

/// Widen \p V to the 64-bit value the callbacks take: a pointer is reported as
/// the address it holds, a floating-point value or a vector as its bit pattern,
/// and a value wider than 64 bits by its low 64 bits. Returns the value
/// together with the number of bytes of it that are meaningful, or
/// {nullptr, 0} for a type that has no fixed-size bit pattern, such as a
/// scalable vector.
static std::pair<Value *, uint64_t> getReportedValue(IRBuilderBase &IRB,
                                                     Value *V) {
  const DataLayout &DL = IRB.GetInsertBlock()->getDataLayout();
  Type *Ty = V->getType();
  IntegerType *Int64Ty = IRB.getInt64Ty();

  // The callback receives bits, not poison: an argument callers pass nothing
  // for (dead argument elimination passes poison), or the padding of a struct
  // coerced into an integer, is whatever it happens to be. This also keeps
  // MemorySanitizer from reporting the instrumentation's own use of it.
  if (!isGuaranteedNotToBeUndefOrPoison(V))
    V = IRB.CreateFreeze(V);

  // ptrtoint widens or narrows to the requested type, so this also holds for a
  // target, or an address space, whose pointers are narrower than 64 bits.
  if (Ty->isPointerTy())
    return {IRB.CreatePtrToInt(V, Int64Ty),
            std::min<uint64_t>(DL.getPointerSize(Ty->getPointerAddressSpace()),
                               sizeof(uint64_t))};

  // A vector of pointers has no bit pattern of its own; its addresses do.
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty);
      VTy && VTy->getElementType()->isPointerTy()) {
    V = IRB.CreatePtrToInt(
        V, VectorType::get(DL.getIntPtrType(VTy->getElementType()),
                           VTy->getElementCount()));
    Ty = V->getType();
  }
  if (Ty->isFloatingPointTy() || isa<FixedVectorType>(Ty)) {
    TypeSize Bits = Ty->getPrimitiveSizeInBits();
    if (Bits.isScalable() || !Bits.getFixedValue())
      return {nullptr, 0};
    V = IRB.CreateBitCast(V, IRB.getIntNTy(Bits.getFixedValue()));
    Ty = V->getType();
  }
  if (!Ty->isIntegerTy())
    return {nullptr, 0};

  uint64_t Size = divideCeil(Ty->getIntegerBitWidth(), 8);
  if (Size > sizeof(uint64_t))
    return {IRB.CreateTrunc(V, Int64Ty), sizeof(uint64_t)};
  return {IRB.CreateZExt(V, Int64Ty), Size};
}

void ModuleSanitizerCoverage::emitTraceCall(IRBuilderBase &IRB,
                                            FunctionCallee Callee,
                                            std::optional<uint32_t> ArgIdx,
                                            uint64_t Size, Value *Val,
                                            Value *Table) {
  uint64_t Tag = ArgIdx ? ((uint64_t)*ArgIdx << 32) | Size : Size;
  Value *ValArg =
      Val ? (Val->getType()->isPointerTy() ? IRB.CreatePtrToInt(Val, Int64Ty)
                                           : Val)
          : ConstantInt::get(Int64Ty, 0);
  Value *TableArg = Table ? Table : ConstantPointerNull::get(PtrTy);

  CallInst *CI = IRB.CreateCall(
      Callee, {ConstantInt::get(Int64Ty, Tag), ValArg, TableArg});
  // An observation is identified by its call site, which the runtime reads as
  // its own return address. A tail call would hand the callback the caller's
  // return address instead, and a trace-ret call of a void function is in tail
  // position. Merging identical calls would conflate distinct return sites.
  CI->setTailCallKind(CallInst::TCK_NoTail);
  CI->setCannotMerge();
}

void ModuleSanitizerCoverage::traceValue(IRBuilderBase &IRB,
                                         FunctionCallee Callee,
                                         std::optional<uint32_t> ArgIdx,
                                         Value *V) {
  // A first-class aggregate lives in registers and has no address, so each of
  // its elements is reported on its own.
  Type *Ty = V->getType();
  unsigned NumElements = Ty->isStructTy()  ? Ty->getStructNumElements()
                         : Ty->isArrayTy() ? Ty->getArrayNumElements()
                                           : 0;
  if (NumElements) {
    for (unsigned I = 0; I != NumElements; ++I)
      traceValue(IRB, Callee, ArgIdx, IRB.CreateExtractValue(V, I));
    return;
  }

  std::pair<Value *, uint64_t> Reported = getReportedValue(IRB, V);
  emitTraceCall(IRB, Callee, ArgIdx, Reported.second, Reported.first);
}

void ModuleSanitizerCoverage::traceObject(IRBuilderBase &IRB,
                                          FunctionCallee Callee,
                                          std::optional<uint32_t> ArgIdx,
                                          Value *Addr, Type *Ty) {
  FieldOffsets Offsets = getObjectOffsets(Ty);
  emitTraceCall(IRB, Callee, ArgIdx, Offsets.ObjectSize,
                Offsets.Table ? Addr : nullptr, Offsets.Table);
}

uint64_t
ModuleSanitizerCoverage::getPointeeBytes(Value *Ptr,
                                         uint64_t Dereferenceable) const {
  // A pointer of another address space may not be readable through the
  // generic address the runtime gets.
  if (!Ptr->getType()->isPointerTy() ||
      Ptr->getType()->getPointerAddressSpace() != 0)
    return 0;
  return std::min<uint64_t>(Dereferenceable,
                            Opts.sanitizer_coverage_trace_pointee_bytes);
}

void ModuleSanitizerCoverage::tracePointee(IRBuilderBase &IRB,
                                           FunctionCallee Callee,
                                           std::optional<uint32_t> ArgIdx,
                                           Value *Ptr, uint64_t Bytes) {
  // The IR says nothing about the pointee but how many bytes of it may be
  // read, so they are read as bytes: the table of a byte array that long.
  FieldOffsets Offsets =
      getObjectOffsets(ArrayType::get(IRB.getInt8Ty(), Bytes));
  emitTraceCall(IRB, Callee, ArgIdx, Offsets.ObjectSize, Ptr, Offsets.Table);
}

void ModuleSanitizerCoverage::InjectTraceForArgs(Function &F) {
  // Static allocas stay first in the entry block. This runs before
  // InjectCoverage, whose entry-block instrumentation (pc_entry, trace_pc,
  // the guard or counter) is then inserted ahead of these calls.
  BasicBlock &EntryBB = F.getEntryBlock();
  BasicBlock::iterator IP =
      PrepareToSplitEntryBlock(EntryBB, EntryBB.getFirstInsertionPt());

  // InstrumentationIRBuilder gives the calls a !dbg location, which the
  // verifier requires of a call in a function with debug information once it
  // can be inlined. The location is the function's own opening line rather
  // than the one of the instruction the calls precede, which may belong to an
  // inlined callee: the arguments are the function's own. Only the location
  // depends on debug information, never the calls.
  InstrumentationIRBuilder IRB(&*IP);
  if (DISubprogram *SP = F.getSubprogram())
    IRB.SetCurrentDebugLocation(
        DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP));

  for (Argument &Arg : F.args()) {
    // A struct-return pointer is where the return value goes, which trace-ret
    // reports; the error slot of a Swift throwing function may only be loaded,
    // stored, or passed on as a swifterror argument. Neither is observed, and
    // neither renumbers the arguments that are: arg_idx stays the IR position.
    if (Arg.hasStructRetAttr() || Arg.hasSwiftErrorAttr())
      continue;
    // Callers may pass anything for an argument the function does not use: a
    // value is still reported, but an object is not read.
    bool Unused = Arg.use_empty();
    uint32_t ArgIdx = Arg.getArgNo();
    // An argument the ABI says points at a copy of the parameter in memory is
    // that object, not a value.
    // An object the function does not use may be passed as a poison pointer,
    // by dead argument elimination in its callers: it is not handed to the
    // runtime to read.
    if (Type *MemTy = Arg.getPointeeInMemoryValueType()) {
      traceObject(IRB, SanCovTraceArgsFunc, ArgIdx, Unused ? nullptr : &Arg,
                  Unused ? nullptr : MemTy);
    } else if (uint64_t Bytes =
                   // Argument::getDereferenceableBytes() asserts on a
                   // non-pointer. Without noundef the pointer may be poison,
                   // which no freeze makes safe to read through.
               Unused || !Arg.getType()->isPointerTy() ||
                       !Arg.hasAttribute(Attribute::NoUndef)
                   ? 0
                   : getPointeeBytes(
                         &Arg, std::max(Arg.getDereferenceableBytes(),
                                        Arg.getDereferenceableOrNullBytes()))) {
      // What the pointer points to, as far as the IR says it may be read: at
      // the top of the entry block nothing can have freed it yet.
      tracePointee(IRB, SanCovTraceArgsFunc, ArgIdx, &Arg, Bytes);
    } else {
      traceValue(IRB, SanCovTraceArgsFunc, ArgIdx, &Arg);
    }
  }
}

void ModuleSanitizerCoverage::InjectTraceForRet(Function &F) {
  // An indirect return leaves an IR function returning void and writes the
  // returned object through a hidden struct-return pointer; that buffer is
  // what is reported, so an indirect return is not dropped.
  Argument *SRetArg = nullptr;
  for (Argument &Arg : F.args())
    if (Arg.hasStructRetAttr()) {
      SRetArg = &Arg;
      break;
    }

  // A function that never writes its sret buffer may be passed a poison one.
  bool SRetUnused = SRetArg && SRetArg->use_empty();
  // A returned pointer the IR says may be read from is reported with what it
  // points to: the return attribute holds where the function returns.
  uint64_t RetPointeeBytes =
      F.hasRetAttribute(Attribute::NoUndef)
          ? std::max(F.getAttributes().getRetDereferenceableBytes(),
                     F.getAttributes().getRetDereferenceableOrNullBytes())
          : 0;
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
    if (SRetArg) {
      traceObject(IRB, SanCovTraceRetFunc, std::nullopt,
                  SRetUnused ? nullptr : SRetArg,
                  SRetUnused ? nullptr : SRetArg->getParamStructRetType());
    } else if (Value *RetVal = RI->getReturnValue()) {
      if (uint64_t Bytes = getPointeeBytes(RetVal, RetPointeeBytes))
        tracePointee(IRB, SanCovTraceRetFunc, std::nullopt, RetVal, Bytes);
      else
        traceValue(IRB, SanCovTraceRetFunc, std::nullopt, RetVal);
    } else {
      emitTraceCall(IRB, SanCovTraceRetFunc, std::nullopt, 0, nullptr); // void
    }
  }
}
