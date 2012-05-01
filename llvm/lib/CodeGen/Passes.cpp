//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static cl::opt<bool> DisablePostRA("disable-post-ra", cl::Hidden,
    cl::desc("Disable Post Regalloc"));
static cl::opt<bool> DisableBranchFold("disable-branch-fold", cl::Hidden,
    cl::desc("Disable branch folding"));
static cl::opt<bool> DisableTailDuplicate("disable-tail-duplicate", cl::Hidden,
    cl::desc("Disable tail duplication"));
static cl::opt<bool> DisableEarlyTailDup("disable-early-taildup", cl::Hidden,
    cl::desc("Disable pre-register allocation tail duplication"));
static cl::opt<bool> DisableBlockPlacement("disable-block-placement",
    cl::Hidden, cl::desc("Disable the probability-driven block placement, and "
                         "re-enable the old code placement pass"));
static cl::opt<bool> EnableBlockPlacementStats("enable-block-placement-stats",
    cl::Hidden, cl::desc("Collect probability-driven block placement stats"));
static cl::opt<bool> DisableCodePlace("disable-code-place", cl::Hidden,
    cl::desc("Disable code placement"));
static cl::opt<bool> DisableSSC("disable-ssc", cl::Hidden,
    cl::desc("Disable Stack Slot Coloring"));
static cl::opt<bool> DisableMachineDCE("disable-machine-dce", cl::Hidden,
    cl::desc("Disable Machine Dead Code Elimination"));
static cl::opt<bool> DisableMachineLICM("disable-machine-licm", cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineCSE("disable-machine-cse", cl::Hidden,
    cl::desc("Disable Machine Common Subexpression Elimination"));
static cl::opt<cl::boolOrDefault>
OptimizeRegAlloc("optimize-regalloc", cl::Hidden,
    cl::desc("Enable optimized register allocation compilation path."));
static cl::opt<cl::boolOrDefault>
EnableMachineSched("enable-misched", cl::Hidden,
    cl::desc("Enable the machine instruction scheduling pass."));
static cl::opt<bool> EnableStrongPHIElim("strong-phi-elim", cl::Hidden,
    cl::desc("Use strong PHI elimination."));
static cl::opt<bool> DisablePostRAMachineLICM("disable-postra-machine-licm",
    cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineSink("disable-machine-sink", cl::Hidden,
    cl::desc("Disable Machine Sinking"));
static cl::opt<bool> DisableLSR("disable-lsr", cl::Hidden,
    cl::desc("Disable Loop Strength Reduction Pass"));
static cl::opt<bool> DisableCGP("disable-cgp", cl::Hidden,
    cl::desc("Disable Codegen Prepare"));
static cl::opt<bool> DisableCopyProp("disable-copyprop", cl::Hidden,
    cl::desc("Disable Copy Propagation pass"));
static cl::opt<bool> PrintLSR("print-lsr-output", cl::Hidden,
    cl::desc("Print LLVM IR produced by the loop-reduce pass"));
static cl::opt<bool> PrintISelInput("print-isel-input", cl::Hidden,
    cl::desc("Print LLVM IR input to isel pass"));
static cl::opt<bool> PrintGCInfo("print-gc", cl::Hidden,
    cl::desc("Dump garbage collector data"));
static cl::opt<bool> VerifyMachineCode("verify-machineinstrs", cl::Hidden,
    cl::desc("Verify generated machine code"),
    cl::init(getenv("LLVM_VERIFY_MACHINEINSTRS")!=NULL));

/// Allow standard passes to be disabled by command line options. This supports
/// simple binary flags that either suppress the pass or do nothing.
/// i.e. -disable-mypass=false has no effect.
/// These should be converted to boolOrDefault in order to use applyOverride.
static AnalysisID applyDisable(AnalysisID ID, bool Override) {
  if (Override)
    return &NoPassID;
  return ID;
}

/// Allow Pass selection to be overriden by command line options. This supports
/// flags with ternary conditions. TargetID is passed through by default. The
/// pass is suppressed when the option is false. When the option is true, the
/// StandardID is selected if the target provides no default.
static AnalysisID applyOverride(AnalysisID TargetID, cl::boolOrDefault Override,
                                AnalysisID StandardID) {
  switch (Override) {
  case cl::BOU_UNSET:
    return TargetID;
  case cl::BOU_TRUE:
    if (TargetID != &NoPassID)
      return TargetID;
    if (StandardID == &NoPassID)
      report_fatal_error("Target cannot enable pass");
    return StandardID;
  case cl::BOU_FALSE:
    return &NoPassID;
  }
  llvm_unreachable("Invalid command line option state");
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
static AnalysisID overridePass(AnalysisID StandardID, AnalysisID TargetID) {
  if (StandardID == &PostRASchedulerID)
    return applyDisable(TargetID, DisablePostRA);

  if (StandardID == &BranchFolderPassID)
    return applyDisable(TargetID, DisableBranchFold);

  if (StandardID == &TailDuplicateID)
    return applyDisable(TargetID, DisableTailDuplicate);

  if (StandardID == &TargetPassConfig::EarlyTailDuplicateID)
    return applyDisable(TargetID, DisableEarlyTailDup);

  if (StandardID == &MachineBlockPlacementID)
    return applyDisable(TargetID, DisableCodePlace);

  if (StandardID == &CodePlacementOptID)
    return applyDisable(TargetID, DisableCodePlace);

  if (StandardID == &StackSlotColoringID)
    return applyDisable(TargetID, DisableSSC);

  if (StandardID == &DeadMachineInstructionElimID)
    return applyDisable(TargetID, DisableMachineDCE);

  if (StandardID == &MachineLICMID)
    return applyDisable(TargetID, DisableMachineLICM);

  if (StandardID == &MachineCSEID)
    return applyDisable(TargetID, DisableMachineCSE);

  if (StandardID == &MachineSchedulerID)
    return applyOverride(TargetID, EnableMachineSched, StandardID);

  if (StandardID == &TargetPassConfig::PostRAMachineLICMID)
    return applyDisable(TargetID, DisablePostRAMachineLICM);

  if (StandardID == &MachineSinkingID)
    return applyDisable(TargetID, DisableMachineSink);

  if (StandardID == &MachineCopyPropagationID)
    return applyDisable(TargetID, DisableCopyProp);

  return TargetID;
}

//===---------------------------------------------------------------------===//
/// TargetPassConfig
//===---------------------------------------------------------------------===//

INITIALIZE_PASS(TargetPassConfig, "targetpassconfig",
                "Target Pass Configuration", false, false)
char TargetPassConfig::ID = 0;

static char NoPassIDAnchor = 0;
char &llvm::NoPassID = NoPassIDAnchor;

// Pseudo Pass IDs.
char TargetPassConfig::EarlyTailDuplicateID = 0;
char TargetPassConfig::PostRAMachineLICMID = 0;

namespace llvm {
class PassConfigImpl {
public:
  // List of passes explicitly substituted by this target. Normally this is
  // empty, but it is a convenient way to suppress or replace specific passes
  // that are part of a standard pass pipeline without overridding the entire
  // pipeline. This mechanism allows target options to inherit a standard pass's
  // user interface. For example, a target may disable a standard pass by
  // default by substituting NoPass, and the user may still enable that standard
  // pass with an explicit command line option.
  DenseMap<AnalysisID,AnalysisID> TargetPasses;
};
} // namespace llvm

// Out of line virtual method.
TargetPassConfig::~TargetPassConfig() {
  delete Impl;
}

// Out of line constructor provides default values for pass options and
// registers all common codegen passes.
TargetPassConfig::TargetPassConfig(TargetMachine *tm, PassManagerBase &pm)
  : ImmutablePass(ID), TM(tm), PM(&pm), Impl(0), Initialized(false),
    DisableVerify(false),
    EnableTailMerge(true) {

  Impl = new PassConfigImpl();

  // Register all target independent codegen passes to activate their PassIDs,
  // including this pass itself.
  initializeCodeGen(*PassRegistry::getPassRegistry());

  // Substitute Pseudo Pass IDs for real ones.
  substitutePass(EarlyTailDuplicateID, TailDuplicateID);
  substitutePass(PostRAMachineLICMID, MachineLICMID);

  // Temporarily disable experimental passes.
  substitutePass(MachineSchedulerID, NoPassID);
}

/// createPassConfig - Create a pass configuration object to be used by
/// addPassToEmitX methods for generating a pipeline of CodeGen passes.
///
/// Targets may override this to extend TargetPassConfig.
TargetPassConfig *LLVMTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new TargetPassConfig(this, PM);
}

TargetPassConfig::TargetPassConfig()
  : ImmutablePass(ID), PM(0) {
  llvm_unreachable("TargetPassConfig should not be constructed on-the-fly");
}

// Helper to verify the analysis is really immutable.
void TargetPassConfig::setOpt(bool &Opt, bool Val) {
  assert(!Initialized && "PassConfig is immutable");
  Opt = Val;
}

void TargetPassConfig::substitutePass(char &StandardID, char &TargetID) {
  Impl->TargetPasses[&StandardID] = &TargetID;
}

AnalysisID TargetPassConfig::getPassSubstitution(AnalysisID ID) const {
  DenseMap<AnalysisID, AnalysisID>::const_iterator
    I = Impl->TargetPasses.find(ID);
  if (I == Impl->TargetPasses.end())
    return ID;
  return I->second;
}

/// Add a CodeGen pass at this point in the pipeline after checking for target
/// and command line overrides.
AnalysisID TargetPassConfig::addPass(char &ID) {
  assert(!Initialized && "PassConfig is immutable");

  AnalysisID TargetID = getPassSubstitution(&ID);
  AnalysisID FinalID = overridePass(&ID, TargetID);
  if (FinalID == &NoPassID)
    return FinalID;

  Pass *P = Pass::createPass(FinalID);
  if (!P)
    llvm_unreachable("Pass ID not registered");
  PM->add(P);
  return FinalID;
}

void TargetPassConfig::printAndVerify(const char *Banner) const {
  if (TM->shouldPrintMachineCode())
    PM->add(createMachineFunctionPrinterPass(dbgs(), Banner));

  if (VerifyMachineCode)
    PM->add(createMachineVerifierPass(Banner));
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
void TargetPassConfig::addIRPasses() {
  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM->add(createTypeBasedAliasAnalysisPass());
  PM->add(createBasicAliasAnalysisPass());

  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!DisableVerify)
    PM->add(createVerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOpt::None && !DisableLSR) {
    PM->add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM->add(createPrintFunctionPass("\n\n*** Code after LSR ***\n", &dbgs()));
  }

  PM->add(createGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM->add(createUnreachableBlockEliminationPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
void TargetPassConfig::addISelPrepare() {
  if (getOptLevel() != CodeGenOpt::None && !DisableCGP)
    PM->add(createCodeGenPreparePass(getTargetLowering()));

  PM->add(createStackProtectorPass(getTargetLowering()));

  addPreISel();

  if (PrintISelInput)
    PM->add(createPrintFunctionPass("\n\n"
                                    "*** Final LLVM Code input to ISel ***\n",
                                    &dbgs()));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    PM->add(createVerifierPass());
}

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
  // Print the instruction selected machine code...
  printAndVerify("After Instruction Selection");

  // Expand pseudo-instructions emitted by ISel.
  addPass(ExpandISelPseudosID);

  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOpt::None) {
    addMachineSSAOptimization();
  }
  else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(LocalStackSlotAllocationID);
  }

  // Run pre-ra passes.
  if (addPreRegAlloc())
    printAndVerify("After PreRegAlloc passes");

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (getOptimizeRegAlloc())
    addOptimizedRegAlloc(createRegAllocPass(true));
  else
    addFastRegAlloc(createRegAllocPass(false));

  // Run post-ra passes.
  if (addPostRegAlloc())
    printAndVerify("After PostRegAlloc passes");

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  addPass(PrologEpilogCodeInserterID);
  printAndVerify("After PrologEpilogCodeInserter");

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOpt::None)
    addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  addPass(ExpandPostRAPseudosID);
  printAndVerify("After ExpandPostRAPseudos");

  // Run pre-sched2 passes.
  if (addPreSched2())
    printAndVerify("After PreSched2 passes");

  // Second pass scheduler.
  if (getOptLevel() != CodeGenOpt::None) {
    addPass(PostRASchedulerID);
    printAndVerify("After PostRAScheduler");
  }

  // GC
  addPass(GCMachineCodeAnalysisID);
  if (PrintGCInfo)
    PM->add(createGCInfoPrinter(dbgs()));

  // Basic block placement.
  if (getOptLevel() != CodeGenOpt::None)
    addBlockPlacement();

  if (addPreEmitPass())
    printAndVerify("After PreEmit passes");
}

/// Add passes that optimize machine instructions in SSA form.
void TargetPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  if (addPass(EarlyTailDuplicateID) != &NoPassID)
    printAndVerify("After Pre-RegAlloc TailDuplicate");

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(OptimizePHIsID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(DeadMachineInstructionElimID);
  printAndVerify("After codegen DCE pass");

  addPass(MachineLICMID);
  addPass(MachineCSEID);
  addPass(MachineSinkingID);
  printAndVerify("After Machine LICM, CSE and Sinking passes");

  addPass(PeepholeOptimizerID);
  printAndVerify("After codegen peephole optimization pass");
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

bool TargetPassConfig::getOptimizeRegAlloc() const {
  switch (OptimizeRegAlloc) {
  case cl::BOU_UNSET: return getOptLevel() != CodeGenOpt::None;
  case cl::BOU_TRUE:  return true;
  case cl::BOU_FALSE: return false;
  }
  llvm_unreachable("Invalid optimize-regalloc state");
}

/// RegisterRegAlloc's global Registry tracks allocator registration.
MachinePassRegistry RegisterRegAlloc::Registry;

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static FunctionPass *useDefaultRegisterAllocator() { return 0; }
static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                useDefaultRegisterAllocator);

/// -regalloc=... command line option.
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc> >
RegAlloc("regalloc",
         cl::init(&useDefaultRegisterAllocator),
         cl::desc("Register allocator to use"));


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
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();

  // Initialize the global default.
  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  // With no -regalloc= override, ask the target for a regalloc pass.
  return createTargetRegisterAllocator(Optimized);
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
void TargetPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  addPass(PHIEliminationID);
  addPass(TwoAddressInstructionPassID);

  PM->add(RegAllocPass);
  printAndVerify("After Register Allocation");
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
void TargetPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  // LiveVariables currently requires pure SSA form.
  //
  // FIXME: Once TwoAddressInstruction pass no longer uses kill flags,
  // LiveVariables can be removed completely, and LiveIntervals can be directly
  // computed. (We still either need to regenerate kill flags after regalloc, or
  // preferably fix the scavenger to not depend on them).
  addPass(LiveVariablesID);

  // Add passes that move from transformed SSA into conventional SSA. This is a
  // "copy coalescing" problem.
  //
  if (!EnableStrongPHIElim) {
    // Edge splitting is smarter with machine loop info.
    addPass(MachineLoopInfoID);
    addPass(PHIEliminationID);
  }
  addPass(TwoAddressInstructionPassID);

  // FIXME: Either remove this pass completely, or fix it so that it works on
  // SSA form. We could modify LiveIntervals to be independent of this pass, But
  // it would be even better to simply eliminate *all* IMPLICIT_DEFs before
  // leaving SSA.
  addPass(ProcessImplicitDefsID);

  if (EnableStrongPHIElim)
    addPass(StrongPHIEliminationID);

  addPass(RegisterCoalescerID);

  // PreRA instruction scheduling.
  if (addPass(MachineSchedulerID) != &NoPassID)
    printAndVerify("After Machine Scheduling");

  // Add the selected register allocation pass.
  PM->add(RegAllocPass);
  printAndVerify("After Register Allocation");

  // FinalizeRegAlloc is convenient until MachineInstrBundles is more mature,
  // but eventually, all users of it should probably be moved to addPostRA and
  // it can go away.  Currently, it's the intended place for targets to run
  // FinalizeMachineBundles, because passes other than MachineScheduling an
  // RegAlloc itself may not be aware of bundles.
  if (addFinalizeRegAlloc())
    printAndVerify("After RegAlloc finalization");

  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addPass(StackSlotColoringID);

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  addPass(PostRAMachineLICMID);

  printAndVerify("After StackSlotColoring and postra Machine LICM");
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
void TargetPassConfig::addMachineLateOptimization() {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (addPass(BranchFolderPassID) != &NoPassID)
    printAndVerify("After BranchFolding");

  // Tail duplication.
  if (addPass(TailDuplicateID) != &NoPassID)
    printAndVerify("After TailDuplicate");

  // Copy propagation.
  if (addPass(MachineCopyPropagationID) != &NoPassID)
    printAndVerify("After copy propagation pass");
}

/// Add standard basic block placement passes.
void TargetPassConfig::addBlockPlacement() {
  AnalysisID ID = &NoPassID;
  if (!DisableBlockPlacement) {
    // MachineBlockPlacement is a new pass which subsumes the functionality of
    // CodPlacementOpt. The old code placement pass can be restored by
    // disabling block placement, but eventually it will be removed.
    ID = addPass(MachineBlockPlacementID);
  } else {
    ID = addPass(CodePlacementOptID);
  }
  if (ID != &NoPassID) {
    // Run a separate pass to collect block placement statistics.
    if (EnableBlockPlacementStats)
      addPass(MachineBlockPlacementStatsID);

    printAndVerify("After machine block placement.");
  }
}
