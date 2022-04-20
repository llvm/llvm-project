//===- PassManagerBuilder.cpp - Build Standard Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

namespace llvm {
cl::opt<bool> RunPartialInlining("enable-partial-inlining", cl::init(false),
                                 cl::Hidden, cl::ZeroOrMore,
                                 cl::desc("Run Partial inlinining pass"));

static cl::opt<bool>
UseGVNAfterVectorization("use-gvn-after-vectorization",
  cl::init(false), cl::Hidden,
  cl::desc("Run GVN instead of Early CSE after vectorization passes"));

cl::opt<bool> ExtraVectorizerPasses(
    "extra-vectorizer-passes", cl::init(false), cl::Hidden,
    cl::desc("Run cleanup optimization passes after vectorization."));

static cl::opt<bool>
RunLoopRerolling("reroll-loops", cl::Hidden,
                 cl::desc("Run the loop rerolling pass"));

cl::opt<bool> RunNewGVN("enable-newgvn", cl::init(false), cl::Hidden,
                        cl::desc("Run the NewGVN pass"));

// Experimental option to use CFL-AA
static cl::opt<::CFLAAType>
    UseCFLAA("use-cfl-aa", cl::init(::CFLAAType::None), cl::Hidden,
             cl::desc("Enable the new, experimental CFL alias analysis"),
             cl::values(clEnumValN(::CFLAAType::None, "none", "Disable CFL-AA"),
                        clEnumValN(::CFLAAType::Steensgaard, "steens",
                                   "Enable unification-based CFL-AA"),
                        clEnumValN(::CFLAAType::Andersen, "anders",
                                   "Enable inclusion-based CFL-AA"),
                        clEnumValN(::CFLAAType::Both, "both",
                                   "Enable both variants of CFL-AA")));

cl::opt<bool> EnableLoopInterchange(
    "enable-loopinterchange", cl::init(false), cl::Hidden,
    cl::desc("Enable the experimental LoopInterchange Pass"));

cl::opt<bool> EnableUnrollAndJam("enable-unroll-and-jam", cl::init(false),
                                 cl::Hidden,
                                 cl::desc("Enable Unroll And Jam Pass"));

cl::opt<bool> EnableLoopFlatten("enable-loop-flatten", cl::init(false),
                                cl::Hidden,
                                cl::desc("Enable the LoopFlatten Pass"));

cl::opt<bool> EnableDFAJumpThreading("enable-dfa-jump-thread",
                                     cl::desc("Enable DFA jump threading."),
                                     cl::init(false), cl::Hidden);

static cl::opt<bool>
    EnablePrepareForThinLTO("prepare-for-thinlto", cl::init(false), cl::Hidden,
                            cl::desc("Enable preparation for ThinLTO."));

static cl::opt<bool>
    EnablePerformThinLTO("perform-thinlto", cl::init(false), cl::Hidden,
                         cl::desc("Enable performing ThinLTO."));

cl::opt<bool> EnableHotColdSplit("hot-cold-split", cl::init(false),
    cl::ZeroOrMore, cl::desc("Enable hot-cold splitting pass"));

cl::opt<bool> EnableIROutliner("ir-outliner", cl::init(false), cl::Hidden,
    cl::desc("Enable ir outliner pass"));

static cl::opt<bool> UseLoopVersioningLICM(
    "enable-loop-versioning-licm", cl::init(false), cl::Hidden,
    cl::desc("Enable the experimental Loop Versioning LICM pass"));

cl::opt<bool>
    DisablePreInliner("disable-preinline", cl::init(false), cl::Hidden,
                      cl::desc("Disable pre-instrumentation inliner"));

cl::opt<int> PreInlineThreshold(
    "preinline-threshold", cl::Hidden, cl::init(75), cl::ZeroOrMore,
    cl::desc("Control the amount of inlining in pre-instrumentation inliner "
             "(default = 75)"));

cl::opt<bool>
    EnableGVNHoist("enable-gvn-hoist", cl::init(false), cl::ZeroOrMore,
                   cl::desc("Enable the GVN hoisting pass (default = off)"));

static cl::opt<bool>
    DisableLibCallsShrinkWrap("disable-libcalls-shrinkwrap", cl::init(false),
                              cl::Hidden,
                              cl::desc("Disable shrink-wrap library calls"));

static cl::opt<bool> EnableSimpleLoopUnswitch(
    "enable-simple-loop-unswitch", cl::init(false), cl::Hidden,
    cl::desc("Enable the simple loop unswitch pass. Also enables independent "
             "cleanup passes integrated into the loop pass manager pipeline."));

cl::opt<bool>
    EnableGVNSink("enable-gvn-sink", cl::init(false), cl::ZeroOrMore,
                  cl::desc("Enable the GVN sinking pass (default = off)"));

// This option is used in simplifying testing SampleFDO optimizations for
// profile loading.
cl::opt<bool>
    EnableCHR("enable-chr", cl::init(true), cl::Hidden,
              cl::desc("Enable control height reduction optimization (CHR)"));

cl::opt<bool> FlattenedProfileUsed(
    "flattened-profile-used", cl::init(false), cl::Hidden,
    cl::desc("Indicate the sample profile being used is flattened, i.e., "
             "no inline hierachy exists in the profile. "));

cl::opt<bool> EnableOrderFileInstrumentation(
    "enable-order-file-instrumentation", cl::init(false), cl::Hidden,
    cl::desc("Enable order file instrumentation (default = off)"));

cl::opt<bool> EnableMatrix(
    "enable-matrix", cl::init(false), cl::Hidden,
    cl::desc("Enable lowering of the matrix intrinsics"));

cl::opt<bool> EnableConstraintElimination(
    "enable-constraint-elimination", cl::init(false), cl::Hidden,
    cl::desc(
        "Enable pass to eliminate conditions based on linear constraints."));

cl::opt<bool> EnableFunctionSpecialization(
    "enable-function-specialization", cl::init(false), cl::Hidden,
    cl::desc("Enable Function Specialization pass"));

cl::opt<AttributorRunOption> AttributorRun(
    "attributor-enable", cl::Hidden, cl::init(AttributorRunOption::NONE),
    cl::desc("Enable the attributor inter-procedural deduction pass."),
    cl::values(clEnumValN(AttributorRunOption::ALL, "all",
                          "enable all attributor runs"),
               clEnumValN(AttributorRunOption::MODULE, "module",
                          "enable module-wide attributor runs"),
               clEnumValN(AttributorRunOption::CGSCC, "cgscc",
                          "enable call graph SCC attributor runs"),
               clEnumValN(AttributorRunOption::NONE, "none",
                          "disable attributor runs")));

extern cl::opt<bool> EnableKnowledgeRetention;
} // namespace llvm

PassManagerBuilder::PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = nullptr;
    Inliner = nullptr;
    DisableUnrollLoops = false;
    SLPVectorize = false;
    LoopVectorize = true;
    LoopsInterleaved = true;
    RerollLoops = RunLoopRerolling;
    NewGVN = RunNewGVN;
    LicmMssaOptCap = SetLicmMssaOptCap;
    LicmMssaNoAccForPromotionCap = SetLicmMssaNoAccForPromotionCap;
    DisableGVNLoadPRE = false;
    ForgetAllSCEVInLoopUnroll = ForgetSCEVInLoopUnroll;
    VerifyInput = false;
    VerifyOutput = false;
    MergeFunctions = false;
    PrepareForLTO = false;
    EnablePGOInstrGen = false;
    EnablePGOCSInstrGen = false;
    EnablePGOCSInstrUse = false;
    PGOInstrGen = "";
    PGOInstrUse = "";
    PGOSampleUse = "";
    PrepareForThinLTO = EnablePrepareForThinLTO;
    PerformThinLTO = EnablePerformThinLTO;
    DivergentTarget = false;
    CallGraphProfile = true;
}

PassManagerBuilder::~PassManagerBuilder() {
  delete LibraryInfo;
  delete Inliner;
}

/// Set of global extensions, automatically added as part of the standard set.
static ManagedStatic<
    SmallVector<std::tuple<PassManagerBuilder::ExtensionPointTy,
                           PassManagerBuilder::ExtensionFn,
                           PassManagerBuilder::GlobalExtensionID>,
                8>>
    GlobalExtensions;
static PassManagerBuilder::GlobalExtensionID GlobalExtensionsCounter;

/// Check if GlobalExtensions is constructed and not empty.
/// Since GlobalExtensions is a managed static, calling 'empty()' will trigger
/// the construction of the object.
static bool GlobalExtensionsNotEmpty() {
  return GlobalExtensions.isConstructed() && !GlobalExtensions->empty();
}

PassManagerBuilder::GlobalExtensionID
PassManagerBuilder::addGlobalExtension(PassManagerBuilder::ExtensionPointTy Ty,
                                       PassManagerBuilder::ExtensionFn Fn) {
  auto ExtensionID = GlobalExtensionsCounter++;
  GlobalExtensions->push_back(std::make_tuple(Ty, std::move(Fn), ExtensionID));
  return ExtensionID;
}

void PassManagerBuilder::removeGlobalExtension(
    PassManagerBuilder::GlobalExtensionID ExtensionID) {
  // RegisterStandardPasses may try to call this function after GlobalExtensions
  // has already been destroyed; doing so should not generate an error.
  if (!GlobalExtensions.isConstructed())
    return;

  auto GlobalExtension =
      llvm::find_if(*GlobalExtensions, [ExtensionID](const auto &elem) {
        return std::get<2>(elem) == ExtensionID;
      });
  assert(GlobalExtension != GlobalExtensions->end() &&
         "The extension ID to be removed should always be valid.");

  GlobalExtensions->erase(GlobalExtension);
}

void PassManagerBuilder::addExtension(ExtensionPointTy Ty, ExtensionFn Fn) {
  Extensions.push_back(std::make_pair(Ty, std::move(Fn)));
}

void PassManagerBuilder::addExtensionsToPM(ExtensionPointTy ETy,
                                           legacy::PassManagerBase &PM) const {
  if (GlobalExtensionsNotEmpty()) {
    for (auto &Ext : *GlobalExtensions) {
      if (std::get<0>(Ext) == ETy)
        std::get<1>(Ext)(*this, PM);
    }
  }
  for (unsigned i = 0, e = Extensions.size(); i != e; ++i)
    if (Extensions[i].first == ETy)
      Extensions[i].second(*this, PM);
}

void PassManagerBuilder::addInitialAliasAnalysisPasses(
    legacy::PassManagerBase &PM) const {
  switch (UseCFLAA) {
  case ::CFLAAType::Steensgaard:
    PM.add(createCFLSteensAAWrapperPass());
    break;
  case ::CFLAAType::Andersen:
    PM.add(createCFLAndersAAWrapperPass());
    break;
  case ::CFLAAType::Both:
    PM.add(createCFLSteensAAWrapperPass());
    PM.add(createCFLAndersAAWrapperPass());
    break;
  default:
    break;
  }

  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAAWrapperPass());
  PM.add(createScopedNoAliasAAWrapperPass());
}

void PassManagerBuilder::populateFunctionPassManager(
    legacy::FunctionPassManager &FPM) {
  addExtensionsToPM(EP_EarlyAsPossible, FPM);

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    FPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  // The backends do not handle matrix intrinsics currently.
  // Make sure they are also lowered in O0.
  // FIXME: A lightweight version of the pass should run in the backend
  //        pipeline on demand.
  if (EnableMatrix && OptLevel == 0)
    FPM.add(createLowerMatrixIntrinsicsMinimalPass());

  if (OptLevel == 0) return;

  addInitialAliasAnalysisPasses(FPM);

  // Lower llvm.expect to metadata before attempting transforms.
  // Compare/branch metadata may alter the behavior of passes like SimplifyCFG.
  FPM.add(createLowerExpectIntrinsicPass());
  FPM.add(createCFGSimplificationPass());
  FPM.add(createSROAPass());
  FPM.add(createEarlyCSEPass());
}

void PassManagerBuilder::addFunctionSimplificationPasses(
    legacy::PassManagerBase &MPM) {
  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  assert(OptLevel >= 1 && "Calling function optimizer with no optimization level!");
  MPM.add(createSROAPass());
  MPM.add(createEarlyCSEPass(true /* Enable mem-ssa. */)); // Catch trivial redundancies
  if (EnableKnowledgeRetention)
    MPM.add(createAssumeSimplifyPass());

  if (OptLevel > 1) {
    if (EnableGVNHoist)
      MPM.add(createGVNHoistPass());
    if (EnableGVNSink) {
      MPM.add(createGVNSinkPass());
      MPM.add(createCFGSimplificationPass(
          SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
    }
  }

  if (EnableConstraintElimination)
    MPM.add(createConstraintEliminationPass());

  if (OptLevel > 1) {
    // Speculative execution if the target has divergent branches; otherwise nop.
    MPM.add(createSpeculativeExecutionIfHasBranchDivergencePass());

    MPM.add(createJumpThreadingPass());         // Thread jumps.
    MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  }
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Merge & remove BBs
  // Combine silly seq's
  if (OptLevel > 2)
    MPM.add(createAggressiveInstCombinerPass());
  MPM.add(createInstructionCombiningPass());
  if (SizeLevel == 0 && !DisableLibCallsShrinkWrap)
    MPM.add(createLibCallsShrinkWrapPass());
  addExtensionsToPM(EP_Peephole, MPM);

  // TODO: Investigate the cost/benefit of tail call elimination on debugging.
  if (OptLevel > 1)
    MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true)));                            // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions

  // The matrix extension can introduce large vector operations early, which can
  // benefit from running vector-combine early on.
  if (EnableMatrix)
    MPM.add(createVectorCombinePass());

  // Begin the loop pass pipeline.
  if (EnableSimpleLoopUnswitch) {
    // The simple loop unswitch pass relies on separate cleanup passes. Schedule
    // them first so when we re-process a loop they run before other loop
    // passes.
    MPM.add(createLoopInstSimplifyPass());
    MPM.add(createLoopSimplifyCFGPass());
  }
  // Try to remove as much code from the loop header as possible,
  // to reduce amount of IR that will have to be duplicated. However,
  // do not perform speculative hoisting the first time as LICM
  // will destroy metadata that may not need to be destroyed if run
  // after loop rotation.
  // TODO: Investigate promotion cap for O1.
  MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                         /*AllowSpeculation=*/false));
  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1, PrepareForLTO));
  // TODO: Investigate promotion cap for O1.
  MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                         /*AllowSpeculation=*/true));
  if (EnableSimpleLoopUnswitch)
    MPM.add(createSimpleLoopUnswitchLegacyPass());
  else
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3, DivergentTarget));
  // FIXME: We break the loop pass pipeline here in order to do full
  // simplifycfg. Eventually loop-simplifycfg should be enhanced to replace the
  // need for this.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
  MPM.add(createInstructionCombiningPass());
  // We resume loop passes creating a second loop pipeline here.
  if (EnableLoopFlatten) {
    MPM.add(createLoopFlattenPass()); // Flatten loops
    MPM.add(createLoopSimplifyCFGPass());
  }
  MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  addExtensionsToPM(EP_LateLoopOptimizations, MPM);
  MPM.add(createLoopDeletionPass());          // Delete dead loops

  if (EnableLoopInterchange)
    MPM.add(createLoopInterchangePass()); // Interchange loops

  // Unroll small loops and perform peeling.
  MPM.add(createSimpleLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                     ForgetAllSCEVInLoopUnroll));
  addExtensionsToPM(EP_LoopOptimizerEnd, MPM);
  // This ends the loop pass pipelines.

  // Break up allocas that may now be splittable after loop unrolling.
  MPM.add(createSROAPass());

  if (OptLevel > 1) {
    MPM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    MPM.add(NewGVN ? createNewGVNPass()
                   : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
  }
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  if (EnableConstraintElimination)
    MPM.add(createConstraintEliminationPass());

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass());        // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, MPM);
  if (OptLevel > 1) {
    if (EnableDFAJumpThreading && SizeLevel == 0)
      MPM.add(createDFAJumpThreadingPass());

    MPM.add(createJumpThreadingPass());         // Thread jumps
    MPM.add(createCorrelatedValuePropagationPass());
  }
  MPM.add(createAggressiveDCEPass()); // Delete dead instructions

  MPM.add(createMemCpyOptPass());               // Remove memcpy / form memset
  // TODO: Investigate if this is too expensive at O1.
  if (OptLevel > 1) {
    MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
    MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                           /*AllowSpeculation=*/true));
  }

  addExtensionsToPM(EP_ScalarOptimizerLate, MPM);

  if (RerollLoops)
    MPM.add(createLoopRerollPass());

  // Merge & remove BBs and sink & hoist common instructions.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().hoistCommonInsts(true).sinkCommonInsts(true)));
  // Clean up after everything.
  MPM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, MPM);

  if (EnableCHR && OptLevel >= 3 &&
      (!PGOInstrUse.empty() || !PGOSampleUse.empty() || EnablePGOCSInstrGen))
    MPM.add(createControlHeightReductionLegacyPass());
}

/// FIXME: Should LTO cause any differences to this set of passes?
void PassManagerBuilder::addVectorPasses(legacy::PassManagerBase &PM,
                                         bool IsFullLTO) {
  PM.add(createLoopVectorizePass(!LoopsInterleaved, !LoopVectorize));

  if (IsFullLTO) {
    // The vectorizer may have significantly shortened a loop body; unroll
    // again. Unroll small loops to hide loop backedge latency and saturate any
    // parallel execution resources of an out-of-order processor. We also then
    // need to clean up redundancies and loop invariant code.
    // FIXME: It would be really good to use a loop-integrated instruction
    // combiner for cleanup here so that the unrolling and LICM can be pipelined
    // across the loop nests.
    // We do UnrollAndJam in a separate LPM to ensure it happens before unroll
    if (EnableUnrollAndJam && !DisableUnrollLoops)
      PM.add(createLoopUnrollAndJamPass(OptLevel));
    PM.add(createLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                ForgetAllSCEVInLoopUnroll));
    PM.add(createWarnMissedTransformationsPass());
  }

  if (!IsFullLTO) {
    // Eliminate loads by forwarding stores from the previous iteration to loads
    // of the current iteration.
    PM.add(createLoopLoadEliminationPass());
  }
  // Cleanup after the loop optimization passes.
  PM.add(createInstructionCombiningPass());

  if (OptLevel > 1 && ExtraVectorizerPasses) {
    // At higher optimization levels, try to clean up any runtime overlap and
    // alignment checks inserted by the vectorizer. We want to track correlated
    // runtime checks for two inner loops in the same outer loop, fold any
    // common computations, hoist loop-invariant aspects out of any outer loop,
    // and unswitch the runtime checks if possible. Once hoisted, we may have
    // dead (or speculatable) control flows or more combining opportunities.
    PM.add(createEarlyCSEPass());
    PM.add(createCorrelatedValuePropagationPass());
    PM.add(createInstructionCombiningPass());
    PM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                          /*AllowSpeculation=*/true));
    PM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3, DivergentTarget));
    PM.add(createCFGSimplificationPass(
        SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
    PM.add(createInstructionCombiningPass());
  }

  // Now that we've formed fast to execute loop structures, we do further
  // optimizations. These are run afterward as they might block doing complex
  // analyses and transforms such as what are needed for loop vectorization.

  // Cleanup after loop vectorization, etc. Simplification passes like CVP and
  // GVN, loop transforms, and others have already run, so it's now better to
  // convert to more optimized IR using more aggressive simplify CFG options.
  // The extra sinking transform can create larger basic blocks, so do this
  // before SLP vectorization.
  PM.add(createCFGSimplificationPass(SimplifyCFGOptions()
                                         .forwardSwitchCondToPhi(true)
                                         .convertSwitchRangeToICmp(true)
                                         .convertSwitchToLookupTable(true)
                                         .needCanonicalLoops(false)
                                         .hoistCommonInsts(true)
                                         .sinkCommonInsts(true)));

  if (IsFullLTO) {
    PM.add(createSCCPPass());                 // Propagate exposed constants
    PM.add(createInstructionCombiningPass()); // Clean up again
    PM.add(createBitTrackingDCEPass());
  }

  // Optimize parallel scalar instruction chains into SIMD instructions.
  if (SLPVectorize) {
    PM.add(createSLPVectorizerPass());
    if (OptLevel > 1 && ExtraVectorizerPasses)
      PM.add(createEarlyCSEPass());
  }

  // Enhance/cleanup vector code.
  PM.add(createVectorCombinePass());

  if (!IsFullLTO) {
    addExtensionsToPM(EP_Peephole, PM);
    PM.add(createInstructionCombiningPass());

    if (EnableUnrollAndJam && !DisableUnrollLoops) {
      // Unroll and Jam. We do this before unroll but need to be in a separate
      // loop pass manager in order for the outer loop to be processed by
      // unroll and jam before the inner loop is unrolled.
      PM.add(createLoopUnrollAndJamPass(OptLevel));
    }

    // Unroll small loops
    PM.add(createLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                ForgetAllSCEVInLoopUnroll));

    if (!DisableUnrollLoops) {
      // LoopUnroll may generate some redundency to cleanup.
      PM.add(createInstructionCombiningPass());

      // Runtime unrolling will introduce runtime check in loop prologue. If the
      // unrolled loop is a inner loop, then the prologue will be inside the
      // outer loop. LICM pass can help to promote the runtime check out if the
      // checked value is loop invariant.
      PM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                            /*AllowSpeculation=*/true));
    }

    PM.add(createWarnMissedTransformationsPass());
  }

  // After vectorization and unrolling, assume intrinsics may tell us more
  // about pointer alignments.
  PM.add(createAlignmentFromAssumptionsPass());

  if (IsFullLTO)
    PM.add(createInstructionCombiningPass());
}

void PassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
  MPM.add(createAnnotation2MetadataLegacyPass());

  if (!PGOSampleUse.empty()) {
    MPM.add(createPruneEHPass());
    // In ThinLTO mode, when flattened profile is used, all the available
    // profile information will be annotated in PreLink phase so there is
    // no need to load the profile again in PostLink.
    if (!(FlattenedProfileUsed && PerformThinLTO))
      MPM.add(createSampleProfileLoaderPass(PGOSampleUse));
  }

  // Allow forcing function attributes as a debugging and tuning aid.
  MPM.add(createForceFunctionAttrsLegacyPass());

  // If all optimizations are disabled, just run the always-inline pass and,
  // if enabled, the function merging pass.
  if (OptLevel == 0) {
    if (Inliner) {
      MPM.add(Inliner);
      Inliner = nullptr;
    }

    // FIXME: The BarrierNoopPass is a HACK! The inliner pass above implicitly
    // creates a CGSCC pass manager, but we don't want to add extensions into
    // that pass manager. To prevent this we insert a no-op module pass to reset
    // the pass manager to get the same behavior as EP_OptimizerLast in non-O0
    // builds. The function merging pass is
    if (MergeFunctions)
      MPM.add(createMergeFunctionsPass());
    else if (GlobalExtensionsNotEmpty() || !Extensions.empty())
      MPM.add(createBarrierNoopPass());

    if (PerformThinLTO) {
      MPM.add(createLowerTypeTestsPass(nullptr, nullptr, true));
      // Drop available_externally and unreferenced globals. This is necessary
      // with ThinLTO in order to avoid leaving undefined references to dead
      // globals in the object file.
      MPM.add(createEliminateAvailableExternallyPass());
      MPM.add(createGlobalDCEPass());
    }

    addExtensionsToPM(EP_EnabledOnOptLevel0, MPM);

    if (PrepareForLTO || PrepareForThinLTO) {
      MPM.add(createCanonicalizeAliasesPass());
      // Rename anon globals to be able to export them in the summary.
      // This has to be done after we add the extensions to the pass manager
      // as there could be passes (e.g. Adddress sanitizer) which introduce
      // new unnamed globals.
      MPM.add(createNameAnonGlobalPass());
    }

    MPM.add(createAnnotationRemarksLegacyPass());
    return;
  }

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    MPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  addInitialAliasAnalysisPasses(MPM);

  // For ThinLTO there are two passes of indirect call promotion. The
  // first is during the compile phase when PerformThinLTO=false and
  // intra-module indirect call targets are promoted. The second is during
  // the ThinLTO backend when PerformThinLTO=true, when we promote imported
  // inter-module indirect calls. For that we perform indirect call promotion
  // earlier in the pass pipeline, here before globalopt. Otherwise imported
  // available_externally functions look unreferenced and are removed.
  if (PerformThinLTO) {
    MPM.add(createLowerTypeTestsPass(nullptr, nullptr, true));
  }

  // For SamplePGO in ThinLTO compile phase, we do not want to unroll loops
  // as it will change the CFG too much to make the 2nd profile annotation
  // in backend more difficult.
  bool PrepareForThinLTOUsingPGOSampleProfile =
      PrepareForThinLTO && !PGOSampleUse.empty();
  if (PrepareForThinLTOUsingPGOSampleProfile)
    DisableUnrollLoops = true;

  // Infer attributes about declarations if possible.
  MPM.add(createInferFunctionAttrsLegacyPass());

  // Infer attributes on declarations, call sites, arguments, etc.
  if (AttributorRun & AttributorRunOption::MODULE)
    MPM.add(createAttributorLegacyPass());

  addExtensionsToPM(EP_ModuleOptimizerEarly, MPM);

  if (OptLevel > 2)
    MPM.add(createCallSiteSplittingPass());

  // Propage constant function arguments by specializing the functions.
  if (OptLevel > 2 && EnableFunctionSpecialization)
    MPM.add(createFunctionSpecializationPass());

  MPM.add(createIPSCCPPass());          // IP SCCP
  MPM.add(createCalledValuePropagationPass());

  MPM.add(createGlobalOptimizerPass()); // Optimize out global vars
  // Promote any localized global vars.
  MPM.add(createPromoteMemoryToRegisterPass());

  MPM.add(createDeadArgEliminationPass()); // Dead argument elimination

  MPM.add(createInstructionCombiningPass()); // Clean up after IPCP & DAE
  addExtensionsToPM(EP_Peephole, MPM);
  MPM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().convertSwitchRangeToICmp(
          true))); // Clean up after IPCP & DAE

  // We add a module alias analysis pass here. In part due to bugs in the
  // analysis infrastructure this "works" in that the analysis stays alive
  // for the entire SCC pass run below.
  MPM.add(createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  MPM.add(createPruneEHPass()); // Remove dead EH info
  bool RunInliner = false;
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = nullptr;
    RunInliner = true;
  }

  // Infer attributes on declarations, call sites, arguments, etc. for an SCC.
  if (AttributorRun & AttributorRunOption::CGSCC)
    MPM.add(createAttributorCGSCCLegacyPass());

  // Try to perform OpenMP specific optimizations. This is a (quick!) no-op if
  // there are no OpenMP runtime calls present in the module.
  if (OptLevel > 1)
    MPM.add(createOpenMPOptCGSCCLegacyPass());

  MPM.add(createPostOrderFunctionAttrsLegacyPass());
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args

  addExtensionsToPM(EP_CGSCCOptimizerLate, MPM);
  addFunctionSimplificationPasses(MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  MPM.add(createBarrierNoopPass());

  if (RunPartialInlining)
    MPM.add(createPartialInliningPass());

  if (OptLevel > 1 && !PrepareForLTO && !PrepareForThinLTO)
    // Remove avail extern fns and globals definitions if we aren't
    // compiling an object file for later LTO. For LTO we want to preserve
    // these so they are eligible for inlining at link-time. Note if they
    // are unreferenced they will be removed by GlobalDCE later, so
    // this only impacts referenced available externally globals.
    // Eventually they will be suppressed during codegen, but eliminating
    // here enables more opportunity for GlobalDCE as it may make
    // globals referenced by available external functions dead
    // and saves running remaining passes on the eliminated functions.
    MPM.add(createEliminateAvailableExternallyPass());

  if (EnableOrderFileInstrumentation)
    MPM.add(createInstrOrderFilePass());

  MPM.add(createReversePostOrderFunctionAttrsPass());

  // The inliner performs some kind of dead code elimination as it goes,
  // but there are cases that are not really caught by it. We might
  // at some point consider teaching the inliner about them, but it
  // is OK for now to run GlobalOpt + GlobalDCE in tandem as their
  // benefits generally outweight the cost, making the whole pipeline
  // faster.
  if (RunInliner) {
    MPM.add(createGlobalOptimizerPass());
    MPM.add(createGlobalDCEPass());
  }

  // If we are planning to perform ThinLTO later, let's not bloat the code with
  // unrolling/vectorization/... now. We'll first run the inliner + CGSCC passes
  // during ThinLTO and perform the rest of the optimizations afterward.
  if (PrepareForThinLTO) {
    // Ensure we perform any last passes, but do so before renaming anonymous
    // globals in case the passes add any.
    addExtensionsToPM(EP_OptimizerLast, MPM);
    MPM.add(createCanonicalizeAliasesPass());
    // Rename anon globals to be able to export them in the summary.
    MPM.add(createNameAnonGlobalPass());
    return;
  }

  if (PerformThinLTO)
    // Optimize globals now when performing ThinLTO, this enables more
    // optimizations later.
    MPM.add(createGlobalOptimizerPass());

  // Scheduling LoopVersioningLICM when inlining is over, because after that
  // we may see more accurate aliasing. Reason to run this late is that too
  // early versioning may prevent further inlining due to increase of code
  // size. By placing it just after inlining other optimizations which runs
  // later might get benefit of no-alias assumption in clone loop.
  if (UseLoopVersioningLICM) {
    MPM.add(createLoopVersioningLICMPass());    // Do LoopVersioningLICM
    MPM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                           /*AllowSpeculation=*/true));
  }

  // We add a fresh GlobalsModRef run at this point. This is particularly
  // useful as the above will have inlined, DCE'ed, and function-attr
  // propagated everything. We should at this point have a reasonably minimal
  // and richly annotated call graph. By computing aliasing and mod/ref
  // information for all local globals here, the late loop passes and notably
  // the vectorizer will be able to use them to help recognize vectorizable
  // memory operations.
  //
  // Note that this relies on a bug in the pass manager which preserves
  // a module analysis into a function pass pipeline (and throughout it) so
  // long as the first function pass doesn't invalidate the module analysis.
  // Thus both Float2Int and LoopRotate have to preserve AliasAnalysis for
  // this to work. Fortunately, it is trivial to preserve AliasAnalysis
  // (doing nothing preserves it as it is required to be conservatively
  // correct in the face of IR changes).
  MPM.add(createGlobalsAAWrapperPass());

  MPM.add(createFloat2IntPass());
  MPM.add(createLowerConstantIntrinsicsPass());

  if (EnableMatrix) {
    MPM.add(createLowerMatrixIntrinsicsPass());
    // CSE the pointer arithmetic of the column vectors.  This allows alias
    // analysis to establish no-aliasing between loads and stores of different
    // columns of the same matrix.
    MPM.add(createEarlyCSEPass(false));
  }

  addExtensionsToPM(EP_VectorizerStart, MPM);

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1, PrepareForLTO));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  MPM.add(createLoopDistributePass());

  addVectorPasses(MPM, /* IsFullLTO */ false);

  // FIXME: We shouldn't bother with this anymore.
  MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes

  // GlobalOpt already deletes dead functions and globals, at -O2 try a
  // late pass of GlobalDCE.  It is capable of deleting dead cycles.
  if (OptLevel > 1) {
    MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
    MPM.add(createConstantMergePass());     // Merge dup global constants
  }

  // See comment in the new PM for justification of scheduling splitting at
  // this stage (\ref buildModuleSimplificationPipeline).
  if (EnableHotColdSplit && !(PrepareForLTO || PrepareForThinLTO))
    MPM.add(createHotColdSplittingPass());

  if (EnableIROutliner)
    MPM.add(createIROutlinerPass());

  if (MergeFunctions)
    MPM.add(createMergeFunctionsPass());

  // Add Module flag "CG Profile" based on Branch Frequency Information.
  if (CallGraphProfile)
    MPM.add(createCGProfileLegacyPass());

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  MPM.add(createLoopSinkPass());
  // Get rid of LCSSA nodes.
  MPM.add(createInstSimplifyLegacyPass());

  // This hoists/decomposes div/rem ops. It should run after other sink/hoist
  // passes to avoid re-sinking, but before SimplifyCFG because it can allow
  // flattening of blocks.
  MPM.add(createDivRemPairsPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.
  MPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions().convertSwitchRangeToICmp(true)));

  addExtensionsToPM(EP_OptimizerLast, MPM);

  if (PrepareForLTO) {
    MPM.add(createCanonicalizeAliasesPass());
    // Rename anon globals to be able to handle them in the summary
    MPM.add(createNameAnonGlobalPass());
  }

  MPM.add(createAnnotationRemarksLegacyPass());
}

void PassManagerBuilder::addLTOOptimizationPasses(legacy::PassManagerBase &PM) {
  // Load sample profile before running the LTO optimization pipeline.
  if (!PGOSampleUse.empty()) {
    PM.add(createPruneEHPass());
    PM.add(createSampleProfileLoaderPass(PGOSampleUse));
  }

  // Remove unused virtual tables to improve the quality of code generated by
  // whole-program devirtualization and bitset lowering.
  PM.add(createGlobalDCEPass());

  // Provide AliasAnalysis services for optimizations.
  addInitialAliasAnalysisPasses(PM);

  // Allow forcing function attributes as a debugging and tuning aid.
  PM.add(createForceFunctionAttrsLegacyPass());

  // Infer attributes about declarations if possible.
  PM.add(createInferFunctionAttrsLegacyPass());

  if (OptLevel > 1) {
    // Split call-site with more constrained arguments.
    PM.add(createCallSiteSplittingPass());

    // Propage constant function arguments by specializing the functions.
    if (EnableFunctionSpecialization && OptLevel > 2)
      PM.add(createFunctionSpecializationPass());

    // Propagate constants at call sites into the functions they call.  This
    // opens opportunities for globalopt (and inlining) by substituting function
    // pointers passed as arguments to direct uses of functions.
    PM.add(createIPSCCPPass());

    // Attach metadata to indirect call sites indicating the set of functions
    // they may target at run-time. This should follow IPSCCP.
    PM.add(createCalledValuePropagationPass());

    // Infer attributes on declarations, call sites, arguments, etc.
    if (AttributorRun & AttributorRunOption::MODULE)
      PM.add(createAttributorLegacyPass());
  }

  // Infer attributes about definitions. The readnone attribute in particular is
  // required for virtual constant propagation.
  PM.add(createPostOrderFunctionAttrsLegacyPass());
  PM.add(createReversePostOrderFunctionAttrsPass());

  // Split globals using inrange annotations on GEP indices. This can help
  // improve the quality of generated code when virtual constant propagation or
  // control flow integrity are enabled.
  PM.add(createGlobalSplitPass());

  // Apply whole-program devirtualization and virtual constant propagation.
  PM.add(createWholeProgramDevirtPass(ExportSummary, nullptr));

  // That's all we need at opt level 1.
  if (OptLevel == 1)
    return;

  // Now that we internalized some globals, see if we can hack on them!
  PM.add(createGlobalOptimizerPass());
  // Promote any localized global vars.
  PM.add(createPromoteMemoryToRegisterPass());

  // Linking modules together can lead to duplicated global constants, only
  // keep one copy of each constant.
  PM.add(createConstantMergePass());

  // Remove unused arguments from functions.
  PM.add(createDeadArgEliminationPass());

  // Reduce the code after globalopt and ipsccp.  Both can open up significant
  // simplification opportunities, and both can propagate functions through
  // function pointers.  When this happens, we often have to resolve varargs
  // calls, etc, so let instcombine do this.
  if (OptLevel > 2)
    PM.add(createAggressiveInstCombinerPass());
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);

  // Inline small functions
  bool RunInliner = Inliner;
  if (RunInliner) {
    PM.add(Inliner);
    Inliner = nullptr;
  }

  PM.add(createPruneEHPass());   // Remove dead EH info.

  // Infer attributes on declarations, call sites, arguments, etc. for an SCC.
  if (AttributorRun & AttributorRunOption::CGSCC)
    PM.add(createAttributorCGSCCLegacyPass());

  // Try to perform OpenMP specific optimizations. This is a (quick!) no-op if
  // there are no OpenMP runtime calls present in the module.
  if (OptLevel > 1)
    PM.add(createOpenMPOptCGSCCLegacyPass());

  // Optimize globals again if we ran the inliner.
  if (RunInliner)
    PM.add(createGlobalOptimizerPass());
  PM.add(createGlobalDCEPass()); // Remove dead functions.

  // If we didn't decide to inline a function, check to see if we can
  // transform it to pass arguments by value instead of by reference.
  PM.add(createArgumentPromotionPass());

  // The IPO passes may leave cruft around.  Clean up after them.
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);
  PM.add(createJumpThreadingPass(/*FreezeSelectCond*/ true));

  // Break up allocas
  PM.add(createSROAPass());

  // LTO provides additional opportunities for tailcall elimination due to
  // link-time inlining, and visibility of nocapture attribute.
  if (OptLevel > 1)
    PM.add(createTailCallEliminationPass());

  // Infer attributes on declarations, call sites, arguments, etc.
  PM.add(createPostOrderFunctionAttrsLegacyPass()); // Add nocapture.
  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createGlobalsAAWrapperPass()); // IP alias analysis.

  PM.add(createLICMPass(LicmMssaOptCap, LicmMssaNoAccForPromotionCap,
                        /*AllowSpeculation=*/true));
  PM.add(NewGVN ? createNewGVNPass()
                : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.

  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass());
  PM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds.

  // More loops are countable; try to optimize them.
  if (EnableLoopFlatten)
    PM.add(createLoopFlattenPass());
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopDeletionPass());
  if (EnableLoopInterchange)
    PM.add(createLoopInterchangePass());

  if (EnableConstraintElimination)
    PM.add(createConstraintEliminationPass());

  // Unroll small loops and perform peeling.
  PM.add(createSimpleLoopUnrollPass(OptLevel, DisableUnrollLoops,
                                    ForgetAllSCEVInLoopUnroll));
  PM.add(createLoopDistributePass());

  addVectorPasses(PM, /* IsFullLTO */ true);

  addExtensionsToPM(EP_Peephole, PM);

  PM.add(createJumpThreadingPass(/*FreezeSelectCond*/ true));
}

void PassManagerBuilder::addLateLTOOptimizationPasses(
    legacy::PassManagerBase &PM) {
  // See comment in the new PM for justification of scheduling splitting at
  // this stage (\ref buildLTODefaultPipeline).
  if (EnableHotColdSplit)
    PM.add(createHotColdSplittingPass());

  // Delete basic blocks, which optimization passes may have killed.
  PM.add(
      createCFGSimplificationPass(SimplifyCFGOptions().hoistCommonInsts(true)));

  // Drop bodies of available externally objects to improve GlobalDCE.
  PM.add(createEliminateAvailableExternallyPass());

  // Now that we have optimized the program, discard unreachable functions.
  PM.add(createGlobalDCEPass());

  // FIXME: this is profitable (for compiler time) to do at -O0 too, but
  // currently it damages debug info.
  if (MergeFunctions)
    PM.add(createMergeFunctionsPass());
}

LLVMPassManagerBuilderRef LLVMPassManagerBuilderCreate() {
  PassManagerBuilder *PMB = new PassManagerBuilder();
  return wrap(PMB);
}

void LLVMPassManagerBuilderDispose(LLVMPassManagerBuilderRef PMB) {
  PassManagerBuilder *Builder = unwrap(PMB);
  delete Builder;
}

void
LLVMPassManagerBuilderSetOptLevel(LLVMPassManagerBuilderRef PMB,
                                  unsigned OptLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->OptLevel = OptLevel;
}

void
LLVMPassManagerBuilderSetSizeLevel(LLVMPassManagerBuilderRef PMB,
                                   unsigned SizeLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->SizeLevel = SizeLevel;
}

void
LLVMPassManagerBuilderSetDisableUnitAtATime(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  // NOTE: The DisableUnitAtATime switch has been removed.
}

void
LLVMPassManagerBuilderSetDisableUnrollLoops(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableUnrollLoops = Value;
}

void
LLVMPassManagerBuilderSetDisableSimplifyLibCalls(LLVMPassManagerBuilderRef PMB,
                                                 LLVMBool Value) {
  // NOTE: The simplify-libcalls pass has been removed.
}

void
LLVMPassManagerBuilderUseInlinerWithThreshold(LLVMPassManagerBuilderRef PMB,
                                              unsigned Threshold) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->Inliner = createFunctionInliningPass(Threshold);
}

void
LLVMPassManagerBuilderPopulateFunctionPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::FunctionPassManager *FPM = unwrap<legacy::FunctionPassManager>(PM);
  Builder->populateFunctionPassManager(*FPM);
}

void
LLVMPassManagerBuilderPopulateModulePassManager(LLVMPassManagerBuilderRef PMB,
                                                LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::PassManagerBase *MPM = unwrap(PM);
  Builder->populateModulePassManager(*MPM);
}
