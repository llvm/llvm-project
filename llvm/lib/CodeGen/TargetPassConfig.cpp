//===- TargetPassConfig.cpp - Target independent code generation passes ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetPassConfig.h"
#include "CodeGenOptions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/CSEConfigBase.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Discriminator.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include <cassert>
#include <optional>
#include <string>

using namespace llvm;

namespace {
class TriggerCrashFunctionLegacyPass : public FunctionPass {
public:
  static char ID;
  TriggerCrashFunctionLegacyPass() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override {
    abort();
    return false;
  }
  StringRef getPassName() const override { return "TriggerCrashFunctionPass"; }
};
} // namespace

char TriggerCrashFunctionLegacyPass::ID = 0;

// Enable or disable the MachineOutliner.
static cl::opt<RunOutliner> EnableMachineOutliner(
    "enable-machine-outliner", cl::desc("Enable the machine outliner"),
    cl::Hidden, cl::ValueOptional, cl::init(RunOutliner::TargetDefault),
    cl::values(
        clEnumValN(RunOutliner::AlwaysOutline, "always",
                   "Run on all functions guaranteed to be beneficial"),
        clEnumValN(RunOutliner::OptimisticPGO, "optimistic-pgo",
                   "Outline cold code only. If a code block does not have "
                   "profile data, optimistically assume it is cold."),
        clEnumValN(RunOutliner::ConservativePGO, "conservative-pgo",
                   "Outline cold code only. If a code block does not have "
                   "profile, data, conservatively assume it is hot."),
        clEnumValN(RunOutliner::NeverOutline, "never", "Disable all outlining"),
        // Sentinel value for unspecified option.
        clEnumValN(RunOutliner::AlwaysOutline, "", "")));

/// Option names for limiting the codegen pipeline.
/// Those are used in error reporting and we didn't want
/// to duplicate their names all over the place.
static const char StartAfterOptName[] = "start-after";
static const char StartBeforeOptName[] = "start-before";
static const char StopAfterOptName[] = "stop-after";
static const char StopBeforeOptName[] = "stop-before";

/// Allow standard passes to be disabled by command line options. This supports
/// simple binary flags that either suppress the pass or do nothing.
/// i.e. -disable-mypass=false has no effect.
/// These should be converted to boolOrDefault in order to use applyOverride.
static IdentifyingPassPtr applyDisable(IdentifyingPassPtr PassID,
                                       bool Override) {
  if (Override)
    return IdentifyingPassPtr();
  return PassID;
}

/// Allow standard passes to be disabled by the command line, regardless of who
/// is adding the pass.
///
/// StandardID is the pass identified in the standard pass pipeline and provided
/// to addPass(). It may be a target-specific ID in the case that the target
/// directly adds its own pass, but in that case we harmlessly fall through.
///
/// TargetID is the pass that the target has configured to override StandardID.
///
/// StandardID may be a pseudo ID. In that case TargetID is the name of the real
/// pass to run. This allows multiple options to control a single pass depending
/// on where in the pipeline that pass is added.
static IdentifyingPassPtr overridePass(AnalysisID StandardID,
                                       IdentifyingPassPtr TargetID) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (StandardID == &PostRASchedulerID)
    return applyDisable(TargetID, Opts.disable_post_ra);

  if (StandardID == &BranchFolderPassID)
    return applyDisable(TargetID, Opts.disable_branch_fold);

  if (StandardID == &TailDuplicateLegacyID)
    return applyDisable(TargetID, Opts.disable_tail_duplicate);

  if (StandardID == &EarlyTailDuplicateLegacyID)
    return applyDisable(TargetID, Opts.disable_early_taildup);

  if (StandardID == &MachineBlockPlacementID)
    return applyDisable(TargetID, Opts.disable_block_placement);

  if (StandardID == &StackSlotColoringID)
    return applyDisable(TargetID, Opts.disable_ssc);

  if (StandardID == &DeadMachineInstructionElimID)
    return applyDisable(TargetID, Opts.disable_machine_dce);

  if (StandardID == &EarlyIfConverterLegacyID)
    return applyDisable(TargetID, Opts.disable_early_ifcvt);

  if (StandardID == &EarlyMachineLICMID)
    return applyDisable(TargetID, Opts.disable_machine_licm);

  if (StandardID == &MachineCSELegacyID)
    return applyDisable(TargetID, Opts.disable_machine_cse);

  if (StandardID == &MachineLICMID)
    return applyDisable(TargetID, Opts.disable_postra_machine_licm);

  if (StandardID == &MachineSinkingLegacyID)
    return applyDisable(TargetID, Opts.disable_machine_sink);

  if (StandardID == &PostRAMachineSinkingID)
    return applyDisable(TargetID, Opts.disable_postra_machine_sink);

  if (StandardID == &MachineCopyPropagationID)
    return applyDisable(TargetID, Opts.disable_copyprop);

  return TargetID;
}

// Find the FSProfile file name. The internal option takes the precedence
// before getting from TargetMachine.
static std::string getFSProfileFile(const TargetMachine *TM) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (!Opts.fs_profile_file.empty())
    return Opts.fs_profile_file.str();
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileFile;
}

// Find the Profile remapping file name. The internal option takes the
// precedence before getting from TargetMachine.
static std::string getFSRemappingFile(const TargetMachine *TM) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (!Opts.fs_remapping_file.empty())
    return Opts.fs_remapping_file.str();
  const std::optional<PGOOptions> &PGOOpt = TM->getPGOOption();
  if (PGOOpt == std::nullopt || PGOOpt->Action != PGOOptions::SampleUse)
    return std::string();
  return PGOOpt->ProfileRemappingFile;
}

//===---------------------------------------------------------------------===//
/// TargetPassConfig
//===---------------------------------------------------------------------===//

INITIALIZE_PASS(TargetPassConfig, "targetpassconfig",
                "Target Pass Configuration", false, false)
char TargetPassConfig::ID = 0;

namespace {

struct InsertedPass {
  AnalysisID TargetPassID;
  IdentifyingPassPtr InsertedPassID;

  InsertedPass(AnalysisID TargetPassID, IdentifyingPassPtr InsertedPassID)
      : TargetPassID(TargetPassID), InsertedPassID(InsertedPassID) {}

  Pass *getInsertedPass() const {
    assert(InsertedPassID.isValid() && "Illegal Pass ID!");
    if (InsertedPassID.isInstance())
      return InsertedPassID.getInstance();
    Pass *NP = Pass::createPass(InsertedPassID.getID());
    assert(NP && "Pass ID not registered");
    return NP;
  }
};

} // end anonymous namespace

namespace llvm {

class PassConfigImpl {
public:
  // List of passes explicitly substituted by this target. Normally this is
  // empty, but it is a convenient way to suppress or replace specific passes
  // that are part of a standard pass pipeline without overridding the entire
  // pipeline. This mechanism allows target options to inherit a standard pass's
  // user interface. For example, a target may disable a standard pass by
  // default by substituting a pass ID of zero, and the user may still enable
  // that standard pass with an explicit command line option.
  DenseMap<AnalysisID,IdentifyingPassPtr> TargetPasses;

  /// Store the pairs of <AnalysisID, AnalysisID> of which the second pass
  /// is inserted after each instance of the first one.
  SmallVector<InsertedPass, 4> InsertedPasses;
};

} // end namespace llvm

// Out of line virtual method.
TargetPassConfig::~TargetPassConfig() {
  delete Impl;
}

static const PassInfo *getPassInfo(StringRef PassName) {
  if (PassName.empty())
    return nullptr;

  const PassRegistry &PR = *PassRegistry::getPassRegistry();
  const PassInfo *PI = PR.getPassInfo(PassName);
  if (!PI)
    reportFatalUsageError(Twine('\"') + Twine(PassName) +
                          Twine("\" pass is not registered."));
  return PI;
}

static AnalysisID getPassIDFromName(StringRef PassName) {
  const PassInfo *PI = getPassInfo(PassName);
  return PI ? PI->getTypeInfo() : nullptr;
}

static std::pair<StringRef, unsigned>
getPassNameAndInstanceNum(StringRef PassName) {
  StringRef Name, InstanceNumStr;
  std::tie(Name, InstanceNumStr) = PassName.split(',');

  unsigned InstanceNum = 0;
  if (!InstanceNumStr.empty() && InstanceNumStr.getAsInteger(10, InstanceNum))
    reportFatalUsageError("invalid pass instance specifier " + PassName);

  return std::make_pair(Name, InstanceNum);
}

void TargetPassConfig::setStartStopPasses() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  StringRef StartBeforeName;
  std::tie(StartBeforeName, StartBeforeInstanceNum) =
      getPassNameAndInstanceNum(Opts.start_before);

  StringRef StartAfterName;
  std::tie(StartAfterName, StartAfterInstanceNum) =
      getPassNameAndInstanceNum(Opts.start_after);

  StringRef StopBeforeName;
  std::tie(StopBeforeName, StopBeforeInstanceNum) =
      getPassNameAndInstanceNum(Opts.stop_before);

  StringRef StopAfterName;
  std::tie(StopAfterName, StopAfterInstanceNum) =
      getPassNameAndInstanceNum(Opts.stop_after);

  StartBefore = getPassIDFromName(StartBeforeName);
  StartAfter = getPassIDFromName(StartAfterName);
  StopBefore = getPassIDFromName(StopBeforeName);
  StopAfter = getPassIDFromName(StopAfterName);
  if (StartBefore && StartAfter)
    reportFatalUsageError(Twine(StartBeforeOptName) + Twine(" and ") +
                          Twine(StartAfterOptName) + Twine(" specified!"));
  if (StopBefore && StopAfter)
    reportFatalUsageError(Twine(StopBeforeOptName) + Twine(" and ") +
                          Twine(StopAfterOptName) + Twine(" specified!"));
  Started = (StartAfter == nullptr) && (StartBefore == nullptr);
}

static cl::boolOrDefault toBoolOrDefault(BoolOrDefault B) {
  if (B == BoolOrDefault::Default)
    return cl::boolOrDefault::BOU_UNSET;
  return B == BoolOrDefault::True ? cl::boolOrDefault::BOU_TRUE
                                  : cl::boolOrDefault::BOU_FALSE;
}

CGPassBuilderOption llvm::getCGPassBuilderOption() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  CGPassBuilderOption Opt;
  Opt.EnableGlobalISelAbort = Opts.global_isel_abort;
  if (Opts.enable_ipra != BoolOrDefault::Default)
    Opt.EnableIPRA = Opts.enable_ipra == BoolOrDefault::True;
  Opt.EnableFastISelOption = toBoolOrDefault(Opts.fast_isel);
  Opt.EnableRegAllocFastTied = toBoolOrDefault(Opts.regalloc_fast_tied);
  Opt.EnableGlobalISelOption = toBoolOrDefault(Opts.global_isel);
  Opt.VerifyMachineCode = toBoolOrDefault(Opts.verify_machineinstrs);
  Opt.DisableAtExitBasedGlobalDtorLowering =
      Opts.disable_atexit_based_global_dtor_lowering;
  Opt.DisableExpandReductions = Opts.disable_expand_reductions;
  Opt.PrintAfterISel = Opts.print_after_isel;
  Opt.FSProfileFile = Opts.fs_profile_file.str();
  Opt.EnableGCEmptyBlocks = Opts.enable_gc_empty_basic_blocks;
  Opt.EnableBlockPlacementStats = Opts.enable_block_placement_stats;
  Opt.EnableGlobalMergeFunc = Opts.enable_global_merge_func;
  Opt.EnableImplicitNullChecks = Opts.enable_implicit_null_checks;
  Opt.EnableMachineOutliner = EnableMachineOutliner;
  Opt.MISchedPostRA = Opts.misched_postra;
  Opt.DisableLSR = Opts.disable_lsr;
  Opt.DisableConstantHoisting = Opts.disable_constant_hoisting;
  Opt.DisableCGP = Opts.disable_cgp;
  Opt.DisablePartialLibcallInlining = Opts.disable_partial_libcall_inlining;
  Opt.DisableSelectOptimize = Opts.disable_select_optimize;
  Opt.PrintISelInput = Opts.print_isel_input;
  Opt.PrintRegUsage = Opts.print_regusage;
  Opt.DebugifyAndStripAll = toBoolOrDefault(Opts.debugify_and_strip_all_safe);
  Opt.DebugifyCheckAndStripAll =
      toBoolOrDefault(Opts.debugify_check_and_strip_all_safe);
  Opt.DisableRAFSProfileLoader = Opts.disable_ra_fsprofile_loader;
  Opt.DisableCFIFixup = Opts.disable_cfi_fixup;
  return Opt;
}

void llvm::registerCodeGenCallback(PassInstrumentationCallbacks &PIC,
                                   TargetMachine &TM) {

  // Register a callback for disabling passes.
  PIC.registerShouldRunOptionalPassCallback([](StringRef P, IRUnitRef) {
    const CodeGenOptions &Opts = CodeGenOptions::Global;

#define DISABLE_PASS(Option, Name)                                             \
  if (Option && P.contains(#Name))                                             \
    return false;
    DISABLE_PASS(Opts.disable_block_placement, MachineBlockPlacementPass)
    DISABLE_PASS(Opts.disable_branch_fold, BranchFolderPass)
    DISABLE_PASS(Opts.disable_copyprop, MachineCopyPropagationPass)
    DISABLE_PASS(Opts.disable_early_ifcvt, EarlyIfConverterLegacyPass)
    DISABLE_PASS(Opts.disable_early_taildup, EarlyTailDuplicatePass)
    DISABLE_PASS(Opts.disable_machine_cse, MachineCSELegacyPass)
    DISABLE_PASS(Opts.disable_machine_dce, DeadMachineInstructionElimPass)
    DISABLE_PASS(Opts.disable_machine_licm, EarlyMachineLICMPass)
    DISABLE_PASS(Opts.disable_machine_sink, MachineSinkingPass)
    DISABLE_PASS(Opts.disable_postra_machine_licm, MachineLICMPass)
    DISABLE_PASS(Opts.disable_postra_machine_sink, PostRAMachineSinkingPass)
    DISABLE_PASS(Opts.disable_post_ra, PostRASchedulerPass)
    DISABLE_PASS(Opts.disable_ssc, StackSlotColoringPass)
    DISABLE_PASS(Opts.disable_tail_duplicate, TailDuplicatePass)

    return true;
  });
}

Expected<TargetPassConfig::StartStopInfo>
TargetPassConfig::getStartStopInfo(PassInstrumentationCallbacks &PIC) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  auto [StartBefore, StartBeforeInstanceNum] =
      getPassNameAndInstanceNum(Opts.start_before);
  auto [StartAfter, StartAfterInstanceNum] =
      getPassNameAndInstanceNum(Opts.start_after);
  auto [StopBefore, StopBeforeInstanceNum] =
      getPassNameAndInstanceNum(Opts.stop_before);
  auto [StopAfter, StopAfterInstanceNum] =
      getPassNameAndInstanceNum(Opts.stop_after);

  if (!StartBefore.empty() && !StartAfter.empty())
    return make_error<StringError>(
        Twine(StartBeforeOptName) + " and " + StartAfterOptName + " specified!",
        std::make_error_code(std::errc::invalid_argument));
  if (!StopBefore.empty() && !StopAfter.empty())
    return make_error<StringError>(
        Twine(StopBeforeOptName) + " and " + StopAfterOptName + " specified!",
        std::make_error_code(std::errc::invalid_argument));

  StartStopInfo Result;
  Result.StartPass = StartBefore.empty() ? StartAfter : StartBefore;
  Result.StopPass = StopBefore.empty() ? StopAfter : StopBefore;
  Result.StartInstanceNum =
      StartBefore.empty() ? StartAfterInstanceNum : StartBeforeInstanceNum;
  Result.StopInstanceNum =
      StopBefore.empty() ? StopAfterInstanceNum : StopBeforeInstanceNum;
  Result.StartAfter = !StartAfter.empty();
  Result.StopAfter = !StopAfter.empty();
  Result.StartInstanceNum += Result.StartInstanceNum == 0;
  Result.StopInstanceNum += Result.StopInstanceNum == 0;
  return Result;
}

// Out of line constructor provides default values for pass options and
// registers all common codegen passes.
TargetPassConfig::TargetPassConfig(TargetMachine &TM, PassManagerBase &PM)
    : ImmutablePass(ID), PM(&PM), TM(&TM) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  Impl = new PassConfigImpl();

  PassRegistry &PR = *PassRegistry::getPassRegistry();
  // Register all target independent codegen passes to activate their PassIDs,
  // including this pass itself.
  initializeCodeGen(PR);

  initializeLibcallLoweringInfoWrapperPass(PR);

  // Also register alias analysis passes required by codegen passes.
  initializeBasicAAWrapperPassPass(PR);
  initializeAAResultsWrapperPassPass(PR);

  if (Opts.enable_ipra != BoolOrDefault::Default) {
    TM.Options.EnableIPRA = Opts.enable_ipra == BoolOrDefault::True;
  } else {
    // If not explicitly specified, use target default.
    TM.Options.EnableIPRA |= TM.useIPRA();
  }

  if (TM.Options.EnableIPRA)
    setRequiresCodeGenSCCOrder();

  if (Opts.regalloc_fast_tied != BoolOrDefault::Default)
    TM.setEnableTiedFastRegAlloc(Opts.regalloc_fast_tied ==
                                 BoolOrDefault::True);

  if (Opts.global_isel_abort)
    TM.Options.GlobalISelAbort = *Opts.global_isel_abort;

  if (Opts.function_splitting)
    TM.Options.FunctionSplitting = *Opts.function_splitting;

  setStartStopPasses();
}

CodeGenOptLevel TargetPassConfig::getOptLevel() const {
  return TM->getOptLevel();
}

/// Insert InsertedPassID pass after TargetPassID.
void TargetPassConfig::insertPass(AnalysisID TargetPassID,
                                  IdentifyingPassPtr InsertedPassID) {
  assert(((!InsertedPassID.isInstance() &&
           TargetPassID != InsertedPassID.getID()) ||
          (InsertedPassID.isInstance() &&
           TargetPassID != InsertedPassID.getInstance()->getPassID())) &&
         "Insert a pass after itself!");
  Impl->InsertedPasses.emplace_back(TargetPassID, InsertedPassID);
}

/// createPassConfig - Create a pass configuration object to be used by
/// addPassToEmitX methods for generating a pipeline of CodeGen passes.
///
/// Targets may override this to extend TargetPassConfig.
TargetPassConfig *
CodeGenTargetMachineImpl::createPassConfig(PassManagerBase &PM) {
  return new TargetPassConfig(*this, PM);
}

TargetPassConfig::TargetPassConfig()
  : ImmutablePass(ID) {
  reportFatalUsageError("trying to construct TargetPassConfig without a target "
                        "machine. Scheduling a CodeGen pass without a target "
                        "triple set?");
}

bool TargetPassConfig::willCompleteCodeGenPipeline() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  return Opts.stop_before.empty() && Opts.stop_after.empty();
}

bool TargetPassConfig::hasLimitedCodeGenPipeline() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  return !Opts.start_before.empty() || !Opts.start_after.empty() ||
         !willCompleteCodeGenPipeline();
}

std::string TargetPassConfig::getLimitedCodeGenPipelineReason() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (!hasLimitedCodeGenPipeline())
    return std::string();
  std::string Res;
  StringRef PassNames[] = {Opts.start_after, Opts.start_before, Opts.stop_after,
                           Opts.stop_before};
  static const char *OptNames[] = {StartAfterOptName, StartBeforeOptName,
                                   StopAfterOptName, StopBeforeOptName};
  bool IsFirst = true;
  for (int Idx = 0; Idx < 4; ++Idx)
    if (!PassNames[Idx].empty()) {
      if (!IsFirst)
        Res += " and ";
      IsFirst = false;
      Res += OptNames[Idx];
    }
  return Res;
}

// Helper to verify the analysis is really immutable.
void TargetPassConfig::setOpt(bool &Opt, bool Val) {
  assert(!Initialized && "PassConfig is immutable");
  Opt = Val;
}

void TargetPassConfig::substitutePass(AnalysisID StandardID,
                                      IdentifyingPassPtr TargetID) {
  Impl->TargetPasses[StandardID] = TargetID;
}

IdentifyingPassPtr TargetPassConfig::getPassSubstitution(AnalysisID ID) const {
  DenseMap<AnalysisID, IdentifyingPassPtr>::const_iterator
    I = Impl->TargetPasses.find(ID);
  if (I == Impl->TargetPasses.end())
    return ID;
  return I->second;
}

bool TargetPassConfig::isPassSubstitutedOrOverridden(AnalysisID ID) const {
  IdentifyingPassPtr TargetID = getPassSubstitution(ID);
  IdentifyingPassPtr FinalPtr = overridePass(ID, TargetID);
  return !FinalPtr.isValid() || FinalPtr.isInstance() ||
      FinalPtr.getID() != ID;
}

/// Add a pass to the PassManager if that pass is supposed to be run.  If the
/// Started/Stopped flags indicate either that the compilation should start at
/// a later pass or that it should stop after an earlier pass, then do not add
/// the pass.  Finally, compare the current pass against the StartAfter
/// and StopAfter options and change the Started/Stopped flags accordingly.
void TargetPassConfig::addPass(Pass *P) {
  assert(!Initialized && "PassConfig is immutable");

  // Cache the Pass ID here in case the pass manager finds this pass is
  // redundant with ones already scheduled / available, and deletes it.
  // Fundamentally, once we add the pass to the manager, we no longer own it
  // and shouldn't reference it.
  AnalysisID PassID = P->getPassID();

  if (StartBefore == PassID && StartBeforeCount++ == StartBeforeInstanceNum)
    Started = true;
  if (StopBefore == PassID && StopBeforeCount++ == StopBeforeInstanceNum)
    Stopped = true;
  if (Started && !Stopped) {
    if (AddingMachinePasses) {
      // Construct banner message before PM->add() as that may delete the pass.
      std::string Banner =
          std::string("After ") + std::string(P->getPassName());
      addMachinePrePasses();
      PM->add(P);
      addMachinePostPasses(Banner);
    } else {
      PM->add(P);
    }

    // Add the passes after the pass P if there is any.
    for (const auto &IP : Impl->InsertedPasses)
      if (IP.TargetPassID == PassID)
        addPass(IP.getInsertedPass());
  } else {
    delete P;
  }

  if (StopAfter == PassID && StopAfterCount++ == StopAfterInstanceNum)
    Stopped = true;

  if (StartAfter == PassID && StartAfterCount++ == StartAfterInstanceNum)
    Started = true;
  if (Stopped && !Started)
    reportFatalUsageError("Cannot stop compilation after pass that is not run");
}

/// Add a CodeGen pass at this point in the pipeline after checking for target
/// and command line overrides.
///
/// addPass cannot return a pointer to the pass instance because is internal the
/// PassManager and the instance we create here may already be freed.
AnalysisID TargetPassConfig::addPass(AnalysisID PassID) {
  IdentifyingPassPtr TargetID = getPassSubstitution(PassID);
  IdentifyingPassPtr FinalPtr = overridePass(PassID, TargetID);
  if (!FinalPtr.isValid())
    return nullptr;

  Pass *P;
  if (FinalPtr.isInstance())
    P = FinalPtr.getInstance();
  else {
    P = Pass::createPass(FinalPtr.getID());
    if (!P)
      llvm_unreachable("Pass ID not registered");
  }
  AnalysisID FinalID = P->getPassID();
  addPass(P); // Ends the lifetime of P.

  return FinalID;
}

void TargetPassConfig::printAndVerify(const std::string &Banner) {
  addPrintPass(Banner);
  addVerifyPass(Banner);
}

void TargetPassConfig::addPrintPass(const std::string &Banner) {
  if (CodeGenOptions::Global.print_after_isel)
    PM->add(createMachineFunctionPrinterPass(dbgs(), Banner));
}

void TargetPassConfig::addVerifyPass(const std::string &Banner) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  bool Verify = valueOr(Opts.verify_machineinstrs, false);
#ifdef EXPENSIVE_CHECKS
  if (Opts.verify_machineinstrs == BoolOrDefault::Default)
    Verify = TM->isMachineVerifierClean();
#endif
  if (Verify)
    PM->add(createMachineVerifierPass(Banner));
}

void TargetPassConfig::addDebugifyPass() {
  PM->add(createDebugifyMachineModulePass());
}

void TargetPassConfig::addStripDebugPass() {
  PM->add(createStripDebugMachineModuleLegacyPass(/*OnlyDebugified=*/true));
}

void TargetPassConfig::addCheckDebugPass() {
  PM->add(createCheckDebugMachineModuleLegacyPass());
}

void TargetPassConfig::addMachinePrePasses(bool AllowDebugify) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (AllowDebugify && DebugifyIsSafe &&
      (valueOr(Opts.debugify_and_strip_all_safe, false) ||
       valueOr(Opts.debugify_check_and_strip_all_safe, false)))
    addDebugifyPass();
}

void TargetPassConfig::addMachinePostPasses(const std::string &Banner) {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (DebugifyIsSafe) {
    if (valueOr(Opts.debugify_check_and_strip_all_safe, false)) {
      addCheckDebugPass();
      addStripDebugPass();
    } else if (valueOr(Opts.debugify_and_strip_all_safe, false))
      addStripDebugPass();
  }
  addVerifyPass(Banner);
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
void TargetPassConfig::addIRPasses() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!DisableVerify)
    addPass(createVerifierPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    // Basic AliasAnalysis support.
    // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
    // BasicAliasAnalysis wins if they disagree. This is intended to help
    // support "obvious" type-punning idioms.
    addPass(createTypeBasedAAWrapperPass());
    addPass(createScopedNoAliasAAWrapperPass());
    addPass(createBasicAAWrapperPass());

    // Run loop strength reduction before anything else.
    if (!Opts.disable_lsr) {
      addPass(createCanonicalizeFreezeInLoopsPass());
      addPass(createLoopStrengthReducePass());
      if (EnableLoopTermFold)
        addPass(createLoopTermFoldPass());
    }
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addPass(&GCLoweringID);
  addPass(&ShadowStackGCLoweringID);

  // For MachO, lower @llvm.global_dtors into @llvm.global_ctors with
  // __cxa_atexit() calls to avoid emitting the deprecated __mod_term_func.
  if (TM->getTargetTriple().isOSBinFormatMachO() &&
      !Opts.disable_atexit_based_global_dtor_lowering)
    addPass(createLowerGlobalDtorsLegacyPass());

  // Make sure that no unreachable blocks are instruction selected.
  addPass(createUnreachableBlockEliminationPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOptLevel::None && !Opts.disable_constant_hoisting)
    addPass(createConstantHoistingPass());

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opts.disable_replace_with_vec_lib)
    addPass(createReplaceWithVeclibLegacyPass());

  if (getOptLevel() != CodeGenOptLevel::None &&
      !Opts.disable_partial_libcall_inlining)
    addPass(createPartiallyInlineLibCallsPass());

  // Instrument function entry after all inlining.
  addPass(createPostInlineEntryExitInstrumenterPass());

  // Add scalarization of target's unsupported masked memory intrinsics pass.
  // the unsupported intrinsic will be replaced with a chain of basic blocks,
  // that stores/loads element one-by-one if the appropriate mask bit is set.
  addPass(createScalarizeMaskedMemIntrinLegacyPass());

  // Expand reduction intrinsics into shuffle sequences if the target wants to.
  // Allow disabling it for testing purposes.
  if (!Opts.disable_expand_reductions)
    addPass(createExpandReductionsPass());

  // Convert conditional moves to conditional jumps when profitable.
  if (getOptLevel() != CodeGenOptLevel::None && !Opts.disable_select_optimize)
    addPass(createSelectOptimizePass());

  if (Opts.enable_global_merge_func)
    addPass(createGlobalMergeFuncPass());

  if (TM->getTargetTriple().isOSWindows())
    addPass(createWindowsSecureHotPatchingPass());
}

/// Turn exception handling constructs into something the code generators can
/// handle.
void TargetPassConfig::addPassesToHandleExceptions() {
  const MCAsmInfo &MCAI = TM->getMCAsmInfo();
  switch (MCAI.getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    addPass(createSjLjEHPreparePass(TM));
    [[fallthrough]];
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::AIX:
  case ExceptionHandling::ZOS:
    addPass(createDwarfEHPass(getOptLevel()));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addPass(createWinEHPass());
    addPass(createDwarfEHPass(getOptLevel()));
    break;
  case ExceptionHandling::Wasm:
    // Wasm EH uses Windows EH instructions, but it does not need to demote PHIs
    // on catchpads and cleanuppads because it does not outline them into
    // funclets. Catchswitch blocks are not lowered in SelectionDAG, so we
    // should remove PHIs there. WinEHPrepare derives this from the Wasm
    // personality, so no explicit flag is needed here.
    addPass(createWinEHPass());
    break;
  case ExceptionHandling::Default:
  case ExceptionHandling::None:
  case ExceptionHandling::Emscripten:
    // Emscripten EH is lowered earlier by WebAssemblyLowerEmscriptenEHSjLj, so
    // by this point it needs no generic EH preparation, like the None case.
    addPass(createLowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    addPass(createUnreachableBlockEliminationPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
void TargetPassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None &&
      !CodeGenOptions::Global.disable_cgp)
    addPass(createCodeGenPrepareLegacyPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
void TargetPassConfig::addISelPrepare() {
  addPreISel();

  // Force codegen to run according to the callgraph.
  if (requiresCodeGenSCCOrder())
    addPass(new DummyCGSCCPass);

  addPass(createInlineAsmPreparePass());

  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addPass(createSafeStackPass());
  addPass(createStackProtectorPass());

  if (CodeGenOptions::Global.print_isel_input)
    addPass(createPrintFunctionPass(
        dbgs(), "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    addPass(createVerifierPass());
}

bool TargetPassConfig::addCoreISelPasses() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  // Enable FastISel with -fast-isel, but allow that to be overridden.
  TM->setO0WantsFastISel(valueOr(Opts.fast_isel, true));

  // Determine an instruction selector.
  enum class SelectorType { SelectionDAG, FastISel, GlobalISel };
  SelectorType Selector;

  if (valueOr(Opts.fast_isel, false))
    Selector = SelectorType::FastISel;
  else if (valueOr(Opts.global_isel, false) ||
           (TM->Options.EnableGlobalISel && valueOr(Opts.global_isel, true)))
    Selector = SelectorType::GlobalISel;
  else if (TM->getOptLevel() == CodeGenOptLevel::None &&
           TM->getO0WantsFastISel())
    Selector = SelectorType::FastISel;
  else
    Selector = SelectorType::SelectionDAG;

  // Set consistently TM->Options.EnableFastISel and EnableGlobalISel.
  if (Selector == SelectorType::FastISel) {
    TM->setFastISel(true);
    TM->setGlobalISel(false);
  } else if (Selector == SelectorType::GlobalISel) {
    TM->setFastISel(false);
    TM->setGlobalISel(true);
  }

  // FIXME: Injecting into the DAGISel pipeline seems to cause issues with
  //        analyses needing to be re-run. This can result in being unable to
  //        schedule passes (particularly with 'Function Alias Analysis
  //        Results'). It's not entirely clear why but AFAICT this seems to be
  //        due to one FunctionPassManager not being able to use analyses from a
  //        previous one. As we're injecting a ModulePass we break the usual
  //        pass manager into two. GlobalISel with the fallback path disabled
  //        and -run-pass seem to be unaffected. The majority of GlobalISel
  //        testing uses -run-pass so this probably isn't too bad.
  SaveAndRestore SavedDebugifyIsSafe(DebugifyIsSafe);
  if (Selector != SelectorType::GlobalISel || !isGlobalISelAbortEnabled())
    DebugifyIsSafe = false;

  // Add instruction selector passes for global isel if enabled.
  if (Selector == SelectorType::GlobalISel) {
    SaveAndRestore SavedAddingMachinePasses(AddingMachinePasses, true);
    if (addIRTranslator())
      return true;

    addPreLegalizeMachineIR();

    if (addLegalizeMachineIR())
      return true;

    // Before running the register bank selector, ask the target if it
    // wants to run some passes.
    addPreRegBankSelect();

    if (addRegBankSelect())
      return true;

    addPreGlobalInstructionSelect();

    if (addGlobalInstructionSelect())
      return true;
  }

  // Pass to reset the MachineFunction if the ISel failed. Outside of the above
  // if so that the verifier is not added to it.
  if (Selector == SelectorType::GlobalISel)
    addPass(createResetMachineFunctionLegacyPass(
        reportDiagnosticWhenGlobalISelFallback(), isGlobalISelAbortEnabled()));

  // Run the SDAG InstSelector, providing a fallback path when we do not want to
  // abort on not-yet-supported input.
  if (Selector != SelectorType::GlobalISel || !isGlobalISelAbortEnabled())
    if (addInstSelector())
      return true;

  // Expand pseudo-instructions emitted by ISel. Don't run the verifier before
  // FinalizeISel.
  addPass(&FinalizeISelID);

  // Print the instruction selected machine code...
  printAndVerify("After Instruction Selection");

  return false;
}

bool TargetPassConfig::addISelPasses() {
  if (TM->useEmulatedTLS())
    addPass(createLowerEmuTLSPass());

  PM->add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));
  // ObjCARCContract operates on ObjC intrinsics and must run before
  // PreISelIntrinsicLowering.
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createObjCARCContractPass());
  addPass(createPreISelIntrinsicLoweringPass());
  addPass(createExpandIRInstsPass(getOptLevel()));
  addIRPasses();

  if (CodeGenOptions::Global.codegen_pipeline_trigger_crash)
    addPass(new TriggerCrashFunctionLegacyPass());

  addCodeGenPrepare();
  addPassesToHandleExceptions();
  addISelPrepare();

  return addCoreISelPasses();
}

/// -regalloc=... command line option.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc>>
    RegAlloc("regalloc", cl::Hidden, cl::init(&useDefaultRegisterAllocator),
             cl::desc("Register allocator to use"));

/// Add the complete set of target-independent postISel code generator passes.
///
/// This can be read as the standard order of major LLVM CodeGen stages. Stages
/// with nontrivial configuration or multiple passes are broken out below in
/// add%Stage routines.
///
/// Any TargetPassConfig::addXX routine may be overriden by the Target. The
/// addPre/Post methods with empty header implementations allow injecting
/// target-specific fixups just before or after major stages. Additionally,
/// targets have the flexibility to change pass order within a stage by
/// overriding default implementation of add%Stage routines below. Each
/// technique has maintainability tradeoffs because alternate pass orders are
/// not well supported. addPre/Post works better if the target pass is easily
/// tied to a common pass. But if it has subtle dependencies on multiple passes,
/// the target should override the stage instead.
///
/// TODO: We could use a single addPre/Post(ID) hook to allow pass injection
/// before/after any target-independent pass. But it's currently overkill.
void TargetPassConfig::addMachinePasses() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  AddingMachinePasses = true;

  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOptLevel::None) {
    addMachineSSAOptimization();
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(&LocalStackSlotAllocationID);
  }

  if (TM->Options.EnableIPRA)
    addPass(createRegUsageInfoPropPass());

  // Run pre-ra passes.
  addPreRegAlloc();

  // Debugifying the register allocator passes seems to provoke some
  // non-determinism that affects CodeGen and there doesn't seem to be a point
  // where it becomes safe again so stop debugifying here.
  DebugifyIsSafe = false;

  // Add a FSDiscriminator pass right before RA, so that we could get
  // more precise SampleFDO profile for RA.
  if (EnableFSDiscriminator) {
    addPass(createMIRAddFSDiscriminatorsPass(
        sampleprof::FSDiscriminatorPass::Pass1));
    const std::string ProfileFile = getFSProfileFile(TM);
    if (!ProfileFile.empty() && !Opts.disable_ra_fsprofile_loader)
      addPass(createMIRProfileLoaderPass(ProfileFile, getFSRemappingFile(TM),
                                         sampleprof::FSDiscriminatorPass::Pass1,
                                         nullptr));
  }

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (getOptimizeRegAlloc())
    addOptimizedRegAlloc();
  else
    addFastRegAlloc();

  // Run post-ra passes.
  addPostRegAlloc();

  addPass(&RemoveRedundantDebugValuesID);

  addPass(&FixupStatepointCallerSavedID);

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(&PostRAMachineSinkingID);
    addPass(&ShrinkWrapID);
  }

  // Prolog/Epilog inserter needs a TargetMachine to instantiate. But only
  // do so if it hasn't been disabled, substituted, or overridden.
  if (!isPassSubstitutedOrOverridden(&PrologEpilogCodeInserterID))
      addPass(createPrologEpilogInserterPass());

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOptLevel::None)
      addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  addPass(&ExpandPostRAPseudosID);

  // Run pre-sched2 passes.
  addPreSched2();

  if (Opts.enable_implicit_null_checks)
    addPass(&ImplicitNullChecksID);

  // Second pass scheduler.
  // Let Target optionally insert this pass by itself at some other
  // point.
  if (getOptLevel() != CodeGenOptLevel::None &&
      !TM->targetSchedulesPostRAScheduling()) {
    if (Opts.misched_postra)
      addPass(&PostMachineSchedulerID);
    else
      addPass(&PostRASchedulerID);
  }

  // GC
  addGCPasses();

  // Basic block placement.
  if (getOptLevel() != CodeGenOptLevel::None)
    addBlockPlacement();

  // Insert before XRay Instrumentation.
  addPass(&FEntryInserterID);

  addPass(&XRayInstrumentationID);
  addPass(&PatchableFunctionID);

  addPreEmitPass();

  if (TM->Options.EnableIPRA)
    // Collect register usage information and produce a register mask of
    // clobbered registers, to be used to optimize call sites.
    addPass(createRegUsageInfoCollector());

  // FIXME: Some backends are incompatible with running the verifier after
  // addPreEmitPass.  Maybe only pass "false" here for those targets?
  addPass(&FuncletLayoutID);

  addPass(&RemoveLoadsIntoFakeUsesID);
  addPass(&StackMapLivenessID);
  addPass(&LiveDebugValuesID);
  addPass(&MachineSanitizerBinaryMetadataID);

  if (TM->Options.EnableMachineOutliner &&
      getOptLevel() != CodeGenOptLevel::None &&
      EnableMachineOutliner != RunOutliner::NeverOutline) {
    if (EnableMachineOutliner != RunOutliner::TargetDefault ||
        TM->supportsDefaultOutlining())
      addPass(createMachineOutlinerPass(EnableMachineOutliner));
  }

  if (Opts.enable_gc_empty_basic_blocks)
    addPass(llvm::createGCEmptyBasicBlocksLegacyPass());

  if (EnableFSDiscriminator)
    addPass(createMIRAddFSDiscriminatorsPass(
        sampleprof::FSDiscriminatorPass::PassLast));

  const bool SplitFunctionsWithPGO =
      TM->Options.FunctionSplitting == FunctionSplittingMode::All;

  if (SplitFunctionsWithPGO || Opts.split_static_data ||
      TM->Options.EnableStaticDataPartitioning) {
    const std::string ProfileFile = getFSProfileFile(TM);
    if (!ProfileFile.empty()) {
      if (EnableFSDiscriminator) {
        addPass(createMIRProfileLoaderPass(
            ProfileFile, getFSRemappingFile(TM),
            sampleprof::FSDiscriminatorPass::PassLast, nullptr));
      } else {
        // Sample profile is given, but FSDiscriminator is not
        // enabled, this may result in performance regression.
        WithColor::warning()
            << "Using AutoFDO without FSDiscriminator for MFS may regress "
               "performance.\n";
      }
    }
  }

  if (Opts.split_static_data || TM->Options.EnableStaticDataPartitioning) {
    // The static data splitter pass is a machine function pass. and
    // static data annotator pass is a module-wide pass. See the file comment
    // in StaticDataAnnotator.cpp for the motivation.
    addPass(createStaticDataSplitterLegacyPass());
    addPass(createStaticDataAnnotatorLegacyPass());
  }
  // We run the BasicBlockSections pass if we need BB sections, BB address map,
  // or function splitting. With function splitting, functions which are
  // eligible for basic-block-sections optimizations (`=all`, or `=list=` with
  // function included in the list profile) get that optimization, and the other
  // functions are split using their PGO/AutoFDO profile.
  if (TM->getBBSectionsType() != llvm::BasicBlockSection::None ||
      TM->Options.BBAddrMap || SplitFunctionsWithPGO) {
    if (shouldEmitBBHash() || Opts.basic_block_section_match_infer)
      addPass(llvm::createMachineBlockHashInfoPass());
    if (TM->getBBSectionsType() == llvm::BasicBlockSection::List) {
      addPass(llvm::createBasicBlockSectionsProfileReaderWrapperPass(
          TM->getBBSectionsFuncListBuf()));
      if (Opts.basic_block_section_match_infer)
        addPass(llvm::createBasicBlockMatchingAndInferencePass());
      else {
        addPass(llvm::createBasicBlockPathCloningPass());
        addPass(llvm::createInsertCodePrefetchPass());
      }
    }
    addPass(llvm::createBasicBlockSectionsPass(SplitFunctionsWithPGO));
  }

  addPostBBSections();

  if (!Opts.disable_cfi_fixup && TM->Options.EnableCFIFixup)
    addPass(createCFIFixupLegacy());

  PM->add(createStackFrameLayoutAnalysisPass());

  // Add passes that directly emit MI after all other MI passes.
  addPreEmitPass2();

  AddingMachinePasses = false;
}

/// Add passes that optimize machine instructions in SSA form.
void TargetPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  addPass(&EarlyTailDuplicateLegacyID);

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(&OptimizePHIsLegacyID);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(&StackColoringLegacyID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(&LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(&DeadMachineInstructionElimID);

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  addILPOpts();

  addPass(&EarlyMachineLICMID);
  addPass(&MachineCSELegacyID);

  addPass(&MachineSinkingLegacyID);

  addPass(&PeepholeOptimizerLegacyID);
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addPass(&DeadMachineInstructionElimID);
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static llvm::once_flag InitializeDefaultRegisterAllocatorFlag;

static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                useDefaultRegisterAllocator);

static void initializeDefaultRegisterAllocatorOnce() {
  if (!RegisterRegAlloc::getDefault())
    RegisterRegAlloc::setDefault(RegAlloc);
}

bool TargetPassConfig::getOptimizeRegAlloc() const {
  // An explicit -regalloc choice implies its pipeline: only the fast
  // allocator uses the unoptimized one.
  llvm::call_once(InitializeDefaultRegisterAllocatorFlag,
                  initializeDefaultRegisterAllocatorOnce);
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();
  if (Ctor != (RegisterRegAlloc::FunctionPassCtor)&useDefaultRegisterAllocator)
    return Ctor !=
           (RegisterRegAlloc::FunctionPassCtor)&createFastRegisterAllocator;
  return getOptLevel() != CodeGenOptLevel::None;
}

/// Instantiate the default register allocator pass for this target for either
/// the optimized or unoptimized allocation path. This will be added to the pass
/// manager by addFastRegAlloc in the unoptimized case or addOptimizedRegAlloc
/// in the optimized case.
///
/// A target that uses the standard regalloc pass order for fast or optimized
/// allocation may still override this for per-target regalloc
/// selection. But -regalloc=... always takes precedence.
FunctionPass *TargetPassConfig::createTargetRegisterAllocator(bool Optimized) {
  if (Optimized)
    return createGreedyRegisterAllocator();
  else
    return createFastRegisterAllocator();
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
///
/// This helper ensures that the regalloc= option is always available,
/// even for targets that override the default allocator.
///
/// FIXME: When MachinePassRegistry register pass IDs instead of function ptrs,
/// this can be folded into addPass.
FunctionPass *TargetPassConfig::createRegAllocPass(bool Optimized) {
  // getOptimizeRegAlloc, called before the pipeline branches, has initialized
  // the global default.
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  // With no -regalloc= override, ask the target for a regalloc pass.
  return createTargetRegisterAllocator(Optimized);
}

bool TargetPassConfig::isCustomizedRegAlloc() {
  return RegAlloc !=
         (RegisterRegAlloc::FunctionPassCtor)&useDefaultRegisterAllocator;
}

bool TargetPassConfig::addRegAssignAndRewriteFast() {
  if (RegAlloc != (RegisterRegAlloc::FunctionPassCtor)&useDefaultRegisterAllocator &&
      RegAlloc != (RegisterRegAlloc::FunctionPassCtor)&createFastRegisterAllocator)
    reportFatalUsageError(
        "Must use fast (default) register allocator for unoptimized regalloc.");

  addPass(createRegAllocPass(false));

  // Allow targets to change the register assignments after
  // fast register allocation.
  addPostFastRegAllocRewrite();
  return true;
}

bool TargetPassConfig::addRegAssignAndRewriteOptimized() {
  // Add the selected register allocation pass.
  addPass(createRegAllocPass(true));

  // Allow targets to change the register assignments before rewriting.
  addPreRewrite();

  // Finally rewrite virtual registers.
  addPass(&VirtRegRewriterID);

  // Regalloc scoring for ML-driven eviction - noop except when learning a new
  // eviction policy.
  addPass(createRegAllocScoringPass());
  return true;
}

/// Return true if the default global register allocator is in use and
/// has not be overriden on the command line with '-regalloc=...'
bool TargetPassConfig::usingDefaultRegAlloc() const {
  return RegAlloc.getNumOccurrences() == 0;
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
void TargetPassConfig::addFastRegAlloc() {
  addPass(&PHIEliminationID);
  if (!TM->enableTiedFastRegAlloc())
    addPass(&TwoAddressInstructionPassID);

  addRegAssignAndRewriteFast();
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
void TargetPassConfig::addOptimizedRegAlloc() {
  addPass(&DetectDeadLanesID);

  addPass(&InitUndefID);

  addPass(&ProcessImplicitDefsID);

  // LiveVariables currently requires pure SSA form.
  //
  // FIXME: Once TwoAddressInstruction pass no longer uses kill flags,
  // LiveVariables can be removed completely, and LiveIntervals can be directly
  // computed. (We still either need to regenerate kill flags after regalloc, or
  // preferably fix the scavenger to not depend on them).
  // FIXME: UnreachableMachineBlockElim is a dependant pass of LiveVariables.
  // When LiveVariables is removed this has to be removed/moved either.
  // Explicit addition of UnreachableMachineBlockElim allows stopping before or
  // after it with -stop-before/-stop-after.
  addPass(&UnreachableMachineBlockElimID);
  addPass(&LiveVariablesID);

  // Run SSA machine scheduler runs just before PHI elimination.
  if (EnableSSAMachineScheduler) {
    addPass(&LiveIntervalsID);
    addPass(&SSAMachineSchedulerID);
  }

  // LiveIntervals is computed unconditionally before TwoAddressInstruction so
  // that pass can rely on it instead of LiveVariables. This is a step toward
  // removing LiveVariables entirely.
  addPass(&LiveIntervalsID);

  // Edge splitting is smarter with machine loop info.
  addPass(&MachineLoopInfoID);
  addPass(&PHIEliminationID);

  addPass(&TwoAddressInstructionPassID);
  addPass(&RegisterCoalescerID);

  // The machine scheduler may accidentally create disconnected components
  // when moving subregister definitions around, avoid this by splitting them to
  // separate vregs before. Splitting can also improve reg. allocation quality.
  addPass(&RenameIndependentSubregsID);

  // PreRA instruction scheduling.
  addPass(&MachineSchedulerID);

  if (addRegAssignAndRewriteOptimized()) {
    // Perform stack slot coloring and post-ra machine LICM.
    addPass(&StackSlotColoringID);

    // Allow targets to expand pseudo instructions depending on the choice of
    // registers before MachineCopyPropagation.
    addPostRewrite();

    // Copy propagate to forward register uses and try to eliminate COPYs that
    // were not coalesced.
    addPass(&MachineCopyPropagationID);

    // Run post-ra machine LICM to hoist reloads / remats.
    //
    // FIXME: can this move into MachineLateOptimization?
    addPass(&MachineLICMID);
  }
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
void TargetPassConfig::addMachineLateOptimization() {
  // Cleanup of redundant immediate/address loads.
  addPass(&MachineLateInstrsCleanupID);

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addPass(&BranchFolderPassID);

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM->requiresStructuredCFG())
    addPass(&TailDuplicateLegacyID);

  // Copy propagation.
  addPass(&MachineCopyPropagationID);
}

/// Add standard GC passes.
bool TargetPassConfig::addGCPasses() {
  addPass(&GCMachineCodeAnalysisID);
  return true;
}

/// Add standard basic block placement passes.
void TargetPassConfig::addBlockPlacement() {
  const CodeGenOptions &Opts = CodeGenOptions::Global;
  if (EnableFSDiscriminator) {
    addPass(createMIRAddFSDiscriminatorsPass(
        sampleprof::FSDiscriminatorPass::Pass2));
    const std::string ProfileFile = getFSProfileFile(TM);
    if (!ProfileFile.empty() && !Opts.disable_layout_fsprofile_loader)
      addPass(createMIRProfileLoaderPass(ProfileFile, getFSRemappingFile(TM),
                                         sampleprof::FSDiscriminatorPass::Pass2,
                                         nullptr));
  }
  if (addPass(&MachineBlockPlacementID)) {
    // Run a separate pass to collect block placement statistics.
    if (Opts.enable_block_placement_stats)
      addPass(&MachineBlockPlacementStatsID);
  }
}

//===---------------------------------------------------------------------===//
/// GlobalISel Configuration
//===---------------------------------------------------------------------===//
bool TargetPassConfig::isGlobalISelAbortEnabled() const {
  return TM->Options.GlobalISelAbort == GlobalISelAbortMode::Enable;
}

bool TargetPassConfig::reportDiagnosticWhenGlobalISelFallback() const {
  return TM->Options.GlobalISelAbort == GlobalISelAbortMode::DisableWithDiag;
}

std::unique_ptr<CSEConfigBase> TargetPassConfig::getCSEConfig() const {
  return std::make_unique<CSEConfigBase>();
}
