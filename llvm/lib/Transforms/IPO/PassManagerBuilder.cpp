//===- PassManagerBuilder.cpp - Build Standard Pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

static cl::opt<bool>
RunLoopVectorization("vectorize-loops", cl::Hidden,
                     cl::desc("Run the Loop vectorization passes"));

static cl::opt<bool>
RunSLPVectorization("vectorize-slp", cl::Hidden,
                    cl::desc("Run the SLP vectorization passes"));

static cl::opt<bool>
RunBBVectorization("vectorize-slp-aggressive", cl::Hidden,
                    cl::desc("Run the BB vectorization passes"));

static cl::opt<bool>
UseGVNAfterVectorization("use-gvn-after-vectorization",
  cl::init(false), cl::Hidden,
  cl::desc("Run GVN instead of Early CSE after vectorization passes"));

static cl::opt<bool> ExtraVectorizerPasses(
    "extra-vectorizer-passes", cl::init(false), cl::Hidden,
    cl::desc("Run cleanup optimization passes after vectorization."));

static cl::opt<bool>
RunLoopRerolling("reroll-loops", cl::Hidden,
                 cl::desc("Run the loop rerolling pass"));

static cl::opt<bool> RunLoadCombine("combine-loads", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Run the load combining pass"));

static cl::opt<bool> RunNewGVN("enable-newgvn", cl::init(false), cl::Hidden,
                               cl::desc("Run the NewGVN pass"));

static cl::opt<bool>
RunSLPAfterLoopVectorization("run-slp-after-loop-vectorization",
  cl::init(true), cl::Hidden,
  cl::desc("Run the SLP vectorizer (and BB vectorizer) after the Loop "
           "vectorizer instead of before"));

// Experimental option to use CFL-AA
enum class CFLAAType { None, Steensgaard, Andersen, Both };
static cl::opt<CFLAAType>
    UseCFLAA("use-cfl-aa", cl::init(CFLAAType::None), cl::Hidden,
             cl::desc("Enable the new, experimental CFL alias analysis"),
             cl::values(clEnumValN(CFLAAType::None, "none", "Disable CFL-AA"),
                        clEnumValN(CFLAAType::Steensgaard, "steens",
                                   "Enable unification-based CFL-AA"),
                        clEnumValN(CFLAAType::Andersen, "anders",
                                   "Enable inclusion-based CFL-AA"),
                        clEnumValN(CFLAAType::Both, "both",
                                   "Enable both variants of CFL-AA")));

static cl::opt<bool> EnableLoopInterchange(
    "enable-loopinterchange", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopInterchange Pass"));

static cl::opt<bool> EnableNonLTOGlobalsModRef(
    "enable-non-lto-gmr", cl::init(true), cl::Hidden,
    cl::desc(
        "Enable the GlobalsModRef AliasAnalysis outside of the LTO pipeline."));

static cl::opt<bool> EnableLoopLoadElim(
    "enable-loop-load-elim", cl::init(true), cl::Hidden,
    cl::desc("Enable the LoopLoadElimination Pass"));

static cl::opt<bool>
    EnablePrepareForThinLTO("prepare-for-thinlto", cl::init(false), cl::Hidden,
                            cl::desc("Enable preparation for ThinLTO."));

static cl::opt<bool> RunPGOInstrGen(
    "profile-generate", cl::init(false), cl::Hidden,
    cl::desc("Enable PGO instrumentation."));

static cl::opt<std::string>
    PGOOutputFile("profile-generate-file", cl::init(""), cl::Hidden,
                      cl::desc("Specify the path of profile data file."));

static cl::opt<std::string> RunPGOInstrUse(
    "profile-use", cl::init(""), cl::Hidden, cl::value_desc("filename"),
    cl::desc("Enable use phase of PGO instrumentation and specify the path "
             "of profile data file"));

static cl::opt<bool> UseLoopVersioningLICM(
    "enable-loop-versioning-licm", cl::init(false), cl::Hidden,
    cl::desc("Enable the experimental Loop Versioning LICM pass"));

static cl::opt<bool>
    DisablePreInliner("disable-preinline", cl::init(false), cl::Hidden,
                      cl::desc("Disable pre-instrumentation inliner"));

static cl::opt<int> PreInlineThreshold(
    "preinline-threshold", cl::Hidden, cl::init(75), cl::ZeroOrMore,
    cl::desc("Control the amount of inlining in pre-instrumentation inliner "
             "(default = 75)"));

static cl::opt<bool> EnableGVNHoist(
    "enable-gvn-hoist", cl::init(false), cl::Hidden,
    cl::desc("Enable the GVN hoisting pass (default = off)"));

static cl::opt<bool>
    DisableLibCallsShrinkWrap("disable-libcalls-shrinkwrap", cl::init(false),
                              cl::Hidden,
                              cl::desc("Disable shrink-wrap library calls"));

static cl::opt<bool>
    EnableSimpleLoopUnswitch("enable-simple-loop-unswitch", cl::init(false),
                             cl::Hidden,
                             cl::desc("Enable the simple loop unswitch pass."));

PassManagerBuilder::PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = nullptr;
    Inliner = nullptr;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
    BBVectorize = RunBBVectorization;
    SLPVectorize = RunSLPVectorization;
    LoopVectorize = RunLoopVectorization;
    RerollLoops = RunLoopRerolling;
    LoadCombine = RunLoadCombine;
    NewGVN = RunNewGVN;
    DisableGVNLoadPRE = false;
    VerifyInput = false;
    VerifyOutput = false;
    MergeFunctions = false;
    PrepareForLTO = false;
    EnablePGOInstrGen = RunPGOInstrGen;
    PGOInstrGen = PGOOutputFile;
    PGOInstrUse = RunPGOInstrUse;
    PrepareForThinLTO = EnablePrepareForThinLTO;
    PerformThinLTO = false;
    DivergentTarget = false;
}

PassManagerBuilder::~PassManagerBuilder() {
  delete LibraryInfo;
  delete Inliner;
}

/// Set of global extensions, automatically added as part of the standard set.
static ManagedStatic<SmallVector<std::pair<PassManagerBuilder::ExtensionPointTy,
   PassManagerBuilder::ExtensionFn>, 8> > GlobalExtensions;

void PassManagerBuilder::addGlobalExtension(
    PassManagerBuilder::ExtensionPointTy Ty,
    PassManagerBuilder::ExtensionFn Fn) {
  GlobalExtensions->push_back(std::make_pair(Ty, std::move(Fn)));
}

void PassManagerBuilder::addExtension(ExtensionPointTy Ty, ExtensionFn Fn) {
  Extensions.push_back(std::make_pair(Ty, std::move(Fn)));
}

void PassManagerBuilder::addExtensionsToPM(ExtensionPointTy ETy,
                                           legacy::PassManagerBase &PM) const {
  for (unsigned i = 0, e = GlobalExtensions->size(); i != e; ++i)
    if ((*GlobalExtensions)[i].first == ETy)
      (*GlobalExtensions)[i].second(*this, PM);
  for (unsigned i = 0, e = Extensions.size(); i != e; ++i)
    if (Extensions[i].first == ETy)
      Extensions[i].second(*this, PM);
}

void PassManagerBuilder::addInitialAliasAnalysisPasses(
    legacy::PassManagerBase &PM) const {
  switch (UseCFLAA) {
  case CFLAAType::Steensgaard:
    PM.add(createCFLSteensAAWrapperPass());
    break;
  case CFLAAType::Andersen:
    PM.add(createCFLAndersAAWrapperPass());
    break;
  case CFLAAType::Both:
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

void PassManagerBuilder::addInstructionCombiningPass(
    legacy::PassManagerBase &PM) const {
  bool ExpensiveCombines = OptLevel > 2;
  PM.add(createInstructionCombiningPass(ExpensiveCombines));
}

void PassManagerBuilder::populateFunctionPassManager(
    legacy::FunctionPassManager &FPM) {
  addExtensionsToPM(EP_EarlyAsPossible, FPM);

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    FPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (OptLevel == 0) return;

  addInitialAliasAnalysisPasses(FPM);

  FPM.add(createCFGSimplificationPass());
  FPM.add(createSROAPass());
  FPM.add(createEarlyCSEPass());
  FPM.add(createLowerExpectIntrinsicPass());
}

// Do PGO instrumentation generation or use pass as the option specified.
void PassManagerBuilder::addPGOInstrPasses(legacy::PassManagerBase &MPM) {
  if (!EnablePGOInstrGen && PGOInstrUse.empty())
    return;
  // Perform the preinline and cleanup passes for O1 and above.
  // And avoid doing them if optimizing for size.
  if (OptLevel > 0 && SizeLevel == 0 && !DisablePreInliner) {
    // Create preinline pass. We construct an InlineParams object and specify
    // the threshold here to avoid the command line options of the regular
    // inliner to influence pre-inlining. The only fields of InlineParams we
    // care about are DefaultThreshold and HintThreshold.
    InlineParams IP;
    IP.DefaultThreshold = PreInlineThreshold;
    // FIXME: The hint threshold has the same value used by the regular inliner.
    // This should probably be lowered after performance testing.
    IP.HintThreshold = 325;

    MPM.add(createFunctionInliningPass(IP));
    MPM.add(createSROAPass());
    MPM.add(createEarlyCSEPass());             // Catch trivial redundancies
    MPM.add(createCFGSimplificationPass());    // Merge & remove BBs
    MPM.add(createInstructionCombiningPass()); // Combine silly seq's
    addExtensionsToPM(EP_Peephole, MPM);
  }
  if (EnablePGOInstrGen) {
    MPM.add(createPGOInstrumentationGenLegacyPass());
    // Add the profile lowering pass.
    InstrProfOptions Options;
    if (!PGOInstrGen.empty())
      Options.InstrProfileOutput = PGOInstrGen;
    MPM.add(createInstrProfilingLegacyPass(Options));
  }
  if (!PGOInstrUse.empty())
    MPM.add(createPGOInstrumentationUseLegacyPass(PGOInstrUse));
  // Indirect call promotion that promotes intra-module targets only.
  // For ThinLTO this is done earlier due to interactions with globalopt
  // for imported functions. We don't run this at -O0.
  if (OptLevel > 0)
    MPM.add(
        createPGOIndirectCallPromotionLegacyPass(false, !PGOSampleUse.empty()));
}
void PassManagerBuilder::addFunctionSimplificationPasses(
    legacy::PassManagerBase &MPM) {
  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  MPM.add(createSROAPass());
  MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
  if (EnableGVNHoist)
    MPM.add(createGVNHoistPass());
  // Speculative execution if the target has divergent branches; otherwise nop.
  MPM.add(createSpeculativeExecutionIfHasBranchDivergencePass());
  MPM.add(createJumpThreadingPass());         // Thread jumps.
  MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  // Combine silly seq's
  addInstructionCombiningPass(MPM);
  if (SizeLevel == 0 && !DisableLibCallsShrinkWrap)
    MPM.add(createLibCallsShrinkWrapPass());
  addExtensionsToPM(EP_Peephole, MPM);

  // Optimize memory intrinsic calls based on the profiled size information.
  if (SizeLevel == 0)
    MPM.add(createPGOMemOPSizeOptLegacyPass());

  MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions
  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));
  MPM.add(createLICMPass());                  // Hoist loop invariants
  if (EnableSimpleLoopUnswitch)
    MPM.add(createSimpleLoopUnswitchLegacyPass());
  else
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3, DivergentTarget));
  MPM.add(createCFGSimplificationPass());
  addInstructionCombiningPass(MPM);
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  addExtensionsToPM(EP_LateLoopOptimizations, MPM);
  MPM.add(createLoopDeletionPass());          // Delete dead loops

  if (EnableLoopInterchange) {
    MPM.add(createLoopInterchangePass()); // Interchange loops
    MPM.add(createCFGSimplificationPass());
  }
  if (!DisableUnrollLoops)
    MPM.add(createSimpleLoopUnrollPass(OptLevel));    // Unroll small loops
  addExtensionsToPM(EP_LoopOptimizerEnd, MPM);

  if (OptLevel > 1) {
    MPM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    MPM.add(NewGVN ? createNewGVNPass()
                   : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
  }
  MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass());        // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  addInstructionCombiningPass(MPM);
  addExtensionsToPM(EP_Peephole, MPM);
  MPM.add(createJumpThreadingPass());         // Thread jumps
  MPM.add(createCorrelatedValuePropagationPass());
  MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
  MPM.add(createLICMPass());

  addExtensionsToPM(EP_ScalarOptimizerLate, MPM);

  if (RerollLoops)
    MPM.add(createLoopRerollPass());
  if (!RunSLPAfterLoopVectorization) {
    if (SLPVectorize)
      MPM.add(createSLPVectorizerPass());   // Vectorize parallel scalar chains.

    if (BBVectorize) {
      MPM.add(createBBVectorizePass());
      addInstructionCombiningPass(MPM);
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(NewGVN
                    ? createNewGVNPass()
                    : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass(OptLevel));
    }
  }

  if (LoadCombine)
    MPM.add(createLoadCombinePass());

  MPM.add(createAggressiveDCEPass());         // Delete dead instructions
  MPM.add(createCFGSimplificationPass()); // Merge & remove BBs
  // Clean up after everything.
  addInstructionCombiningPass(MPM);
  addExtensionsToPM(EP_Peephole, MPM);
}

void PassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
  if (!PGOSampleUse.empty()) {
    MPM.add(createPruneEHPass());
    MPM.add(createSampleProfileLoaderPass(PGOSampleUse));
  }

  // Allow forcing function attributes as a debugging and tuning aid.
  MPM.add(createForceFunctionAttrsLegacyPass());

  // If all optimizations are disabled, just run the always-inline pass and,
  // if enabled, the function merging pass.
  if (OptLevel == 0) {
    addPGOInstrPasses(MPM);
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
    else if (!GlobalExtensions->empty() || !Extensions.empty())
      MPM.add(createBarrierNoopPass());

    addExtensionsToPM(EP_EnabledOnOptLevel0, MPM);

    // Rename anon globals to be able to export them in the summary.
    // This has to be done after we add the extensions to the pass manager
    // as there could be passes (e.g. Adddress sanitizer) which introduce
    // new unnamed globals.
    if (PrepareForThinLTO)
      MPM.add(createNameAnonGlobalPass());
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
  if (PerformThinLTO)
    MPM.add(createPGOIndirectCallPromotionLegacyPass(/*InLTO = */ true,
                                                     !PGOSampleUse.empty()));

  // For SamplePGO in ThinLTO compile phase, we do not want to unroll loops
  // as it will change the CFG too much to make the 2nd profile annotation
  // in backend more difficult.
  bool PrepareForThinLTOUsingPGOSampleProfile =
      PrepareForThinLTO && !PGOSampleUse.empty();
  if (PrepareForThinLTOUsingPGOSampleProfile)
    DisableUnrollLoops = true;

  if (!DisableUnitAtATime) {
    // Infer attributes about declarations if possible.
    MPM.add(createInferFunctionAttrsLegacyPass());

    addExtensionsToPM(EP_ModuleOptimizerEarly, MPM);

    MPM.add(createIPSCCPPass());          // IP SCCP
    MPM.add(createGlobalOptimizerPass()); // Optimize out global vars
    // Promote any localized global vars.
    MPM.add(createPromoteMemoryToRegisterPass());

    MPM.add(createDeadArgEliminationPass()); // Dead argument elimination

    addInstructionCombiningPass(MPM); // Clean up after IPCP & DAE
    addExtensionsToPM(EP_Peephole, MPM);
    MPM.add(createCFGSimplificationPass()); // Clean up after IPCP & DAE
  }

  // For SamplePGO in ThinLTO compile phase, we do not want to do indirect
  // call promotion as it will change the CFG too much to make the 2nd
  // profile annotation in backend more difficult.
  // PGO instrumentation is added during the compile phase for ThinLTO, do
  // not run it a second time
  if (!PerformThinLTO && !PrepareForThinLTOUsingPGOSampleProfile)
    addPGOInstrPasses(MPM);

  if (EnableNonLTOGlobalsModRef)
    // We add a module alias analysis pass here. In part due to bugs in the
    // analysis infrastructure this "works" in that the analysis stays alive
    // for the entire SCC pass run below.
    MPM.add(createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  if (!DisableUnitAtATime)
    MPM.add(createPruneEHPass()); // Remove dead EH info
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = nullptr;
  }
  if (!DisableUnitAtATime)
    MPM.add(createPostOrderFunctionAttrsLegacyPass());
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass()); // Scalarize uninlined fn args

  addExtensionsToPM(EP_CGSCCOptimizerLate, MPM);
  addFunctionSimplificationPasses(MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  MPM.add(createBarrierNoopPass());

  if (!DisableUnitAtATime && OptLevel > 1 && !PrepareForLTO &&
      !PrepareForThinLTO)
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

  if (!DisableUnitAtATime)
    MPM.add(createReversePostOrderFunctionAttrsPass());

  // If we are planning to perform ThinLTO later, let's not bloat the code with
  // unrolling/vectorization/... now. We'll first run the inliner + CGSCC passes
  // during ThinLTO and perform the rest of the optimizations afterward.
  if (PrepareForThinLTO) {
    // Reduce the size of the IR as much as possible.
    MPM.add(createGlobalOptimizerPass());
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
    MPM.add(createLICMPass());                  // Hoist loop invariants
  }

  if (EnableNonLTOGlobalsModRef)
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

  addExtensionsToPM(EP_VectorizerStart, MPM);

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.  This is
  // currently only performed for loops marked with the metadata
  // llvm.loop.distribute=true or when -enable-loop-distribute is specified.
  MPM.add(createLoopDistributePass());

  MPM.add(createLoopVectorizePass(DisableUnrollLoops, LoopVectorize));

  // Eliminate loads by forwarding stores from the previous iteration to loads
  // of the current iteration.
  if (EnableLoopLoadElim)
    MPM.add(createLoopLoadEliminationPass());

  // FIXME: Because of #pragma vectorize enable, the passes below are always
  // inserted in the pipeline, even when the vectorizer doesn't run (ex. when
  // on -O1 and no #pragma is found). Would be good to have these two passes
  // as function calls, so that we can only pass them when the vectorizer
  // changed the code.
  addInstructionCombiningPass(MPM);
  if (OptLevel > 1 && ExtraVectorizerPasses) {
    // At higher optimization levels, try to clean up any runtime overlap and
    // alignment checks inserted by the vectorizer. We want to track correllated
    // runtime checks for two inner loops in the same outer loop, fold any
    // common computations, hoist loop-invariant aspects out of any outer loop,
    // and unswitch the runtime checks if possible. Once hoisted, we may have
    // dead (or speculatable) control flows or more combining opportunities.
    MPM.add(createEarlyCSEPass());
    MPM.add(createCorrelatedValuePropagationPass());
    addInstructionCombiningPass(MPM);
    MPM.add(createLICMPass());
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3, DivergentTarget));
    MPM.add(createCFGSimplificationPass());
    addInstructionCombiningPass(MPM);
  }

  if (RunSLPAfterLoopVectorization) {
    if (SLPVectorize) {
      MPM.add(createSLPVectorizerPass());   // Vectorize parallel scalar chains.
      if (OptLevel > 1 && ExtraVectorizerPasses) {
        MPM.add(createEarlyCSEPass());
      }
    }

    if (BBVectorize) {
      MPM.add(createBBVectorizePass());
      addInstructionCombiningPass(MPM);
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(NewGVN
                    ? createNewGVNPass()
                    : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass(OptLevel));
    }
  }

  addExtensionsToPM(EP_Peephole, MPM);
  MPM.add(createLateCFGSimplificationPass()); // Switches to lookup tables
  addInstructionCombiningPass(MPM);

  if (!DisableUnrollLoops) {
    MPM.add(createLoopUnrollPass(OptLevel));    // Unroll small loops

    // LoopUnroll may generate some redundency to cleanup.
    addInstructionCombiningPass(MPM);

    // Runtime unrolling will introduce runtime check in loop prologue. If the
    // unrolled loop is a inner loop, then the prologue will be inside the
    // outer loop. LICM pass can help to promote the runtime check out if the
    // checked value is loop invariant.
    MPM.add(createLICMPass());
 }

  // After vectorization and unrolling, assume intrinsics may tell us more
  // about pointer alignments.
  MPM.add(createAlignmentFromAssumptionsPass());

  if (!DisableUnitAtATime) {
    // FIXME: We shouldn't bother with this anymore.
    MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes

    // GlobalOpt already deletes dead functions and globals, at -O2 try a
    // late pass of GlobalDCE.  It is capable of deleting dead cycles.
    if (OptLevel > 1) {
      MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
      MPM.add(createConstantMergePass());     // Merge dup global constants
    }
  }

  if (MergeFunctions)
    MPM.add(createMergeFunctionsPass());

  // LoopSink pass sinks instructions hoisted by LICM, which serves as a
  // canonicalization pass that enables other optimizations. As a result,
  // LoopSink pass needs to be a very late IR pass to avoid undoing LICM
  // result too early.
  MPM.add(createLoopSinkPass());
  // Get rid of LCSSA nodes.
  MPM.add(createInstructionSimplifierPass());

  // LoopSink (and other loop passes since the last simplifyCFG) might have
  // resulted in single-entry-single-exit or empty blocks. Clean up the CFG.
  MPM.add(createCFGSimplificationPass());

  addExtensionsToPM(EP_OptimizerLast, MPM);
}

void PassManagerBuilder::addLTOOptimizationPasses(legacy::PassManagerBase &PM) {
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
    // Indirect call promotion. This should promote all the targets that are
    // left by the earlier promotion pass that promotes intra-module targets.
    // This two-step promotion is to save the compile time. For LTO, it should
    // produce the same result as if we only do promotion here.
    PM.add(
        createPGOIndirectCallPromotionLegacyPass(true, !PGOSampleUse.empty()));

    // Propagate constants at call sites into the functions they call.  This
    // opens opportunities for globalopt (and inlining) by substituting function
    // pointers passed as arguments to direct uses of functions.
    PM.add(createIPSCCPPass());
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
  addInstructionCombiningPass(PM);
  addExtensionsToPM(EP_Peephole, PM);

  // Inline small functions
  bool RunInliner = Inliner;
  if (RunInliner) {
    PM.add(Inliner);
    Inliner = nullptr;
  }

  PM.add(createPruneEHPass());   // Remove dead EH info.

  // Optimize globals again if we ran the inliner.
  if (RunInliner)
    PM.add(createGlobalOptimizerPass());
  PM.add(createGlobalDCEPass()); // Remove dead functions.

  // If we didn't decide to inline a function, check to see if we can
  // transform it to pass arguments by value instead of by reference.
  PM.add(createArgumentPromotionPass());

  // The IPO passes may leave cruft around.  Clean up after them.
  addInstructionCombiningPass(PM);
  addExtensionsToPM(EP_Peephole, PM);
  PM.add(createJumpThreadingPass());

  // Break up allocas
  PM.add(createSROAPass());

  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createPostOrderFunctionAttrsLegacyPass()); // Add nocapture.
  PM.add(createGlobalsAAWrapperPass()); // IP alias analysis.

  PM.add(createLICMPass());                 // Hoist loop invariants.
  PM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds.
  PM.add(NewGVN ? createNewGVNPass()
                : createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.

  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass());

  // More loops are countable; try to optimize them.
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopDeletionPass());
  if (EnableLoopInterchange)
    PM.add(createLoopInterchangePass());

  if (!DisableUnrollLoops)
    PM.add(createSimpleLoopUnrollPass(OptLevel));   // Unroll small loops
  PM.add(createLoopVectorizePass(true, LoopVectorize));
  // The vectorizer may have significantly shortened a loop body; unroll again.
  if (!DisableUnrollLoops)
    PM.add(createLoopUnrollPass(OptLevel));

  // Now that we've optimized loops (in particular loop induction variables),
  // we may have exposed more scalar opportunities. Run parts of the scalar
  // optimizer again at this point.
  addInstructionCombiningPass(PM); // Initial cleanup
  PM.add(createCFGSimplificationPass()); // if-convert
  PM.add(createSCCPPass()); // Propagate exposed constants
  addInstructionCombiningPass(PM); // Clean up again
  PM.add(createBitTrackingDCEPass());

  // More scalar chains could be vectorized due to more alias information
  if (RunSLPAfterLoopVectorization)
    if (SLPVectorize)
      PM.add(createSLPVectorizerPass()); // Vectorize parallel scalar chains.

  // After vectorization, assume intrinsics may tell us more about pointer
  // alignments.
  PM.add(createAlignmentFromAssumptionsPass());

  if (LoadCombine)
    PM.add(createLoadCombinePass());

  // Cleanup and simplify the code after the scalar optimizations.
  addInstructionCombiningPass(PM);
  addExtensionsToPM(EP_Peephole, PM);

  PM.add(createJumpThreadingPass());
}

void PassManagerBuilder::addLateLTOOptimizationPasses(
    legacy::PassManagerBase &PM) {
  // Delete basic blocks, which optimization passes may have killed.
  PM.add(createCFGSimplificationPass());

  // Drop bodies of available externally objects to improve GlobalDCE.
  PM.add(createEliminateAvailableExternallyPass());

  // Now that we have optimized the program, discard unreachable functions.
  PM.add(createGlobalDCEPass());

  // FIXME: this is profitable (for compiler time) to do at -O0 too, but
  // currently it damages debug info.
  if (MergeFunctions)
    PM.add(createMergeFunctionsPass());
}

void PassManagerBuilder::populateThinLTOPassManager(
    legacy::PassManagerBase &PM) {
  PerformThinLTO = true;

  if (VerifyInput)
    PM.add(createVerifierPass());

  if (ImportSummary) {
    // These passes import type identifier resolutions for whole-program
    // devirtualization and CFI. They must run early because other passes may
    // disturb the specific instruction patterns that these passes look for,
    // creating dependencies on resolutions that may not appear in the summary.
    //
    // For example, GVN may transform the pattern assume(type.test) appearing in
    // two basic blocks into assume(phi(type.test, type.test)), which would
    // transform a dependency on a WPD resolution into a dependency on a type
    // identifier resolution for CFI.
    //
    // Also, WPD has access to more precise information than ICP and can
    // devirtualize more effectively, so it should operate on the IR first.
    PM.add(createWholeProgramDevirtPass(nullptr, ImportSummary));
    PM.add(createLowerTypeTestsPass(nullptr, ImportSummary));
  }

  populateModulePassManager(PM);

  if (VerifyOutput)
    PM.add(createVerifierPass());
  PerformThinLTO = false;
}

void PassManagerBuilder::populateLTOPassManager(legacy::PassManagerBase &PM) {
  if (LibraryInfo)
    PM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (VerifyInput)
    PM.add(createVerifierPass());

  if (OptLevel != 0)
    addLTOOptimizationPasses(PM);

  // Create a function that performs CFI checks for cross-DSO calls with targets
  // in the current module.
  PM.add(createCrossDSOCFIPass());

  // Lower type metadata and the type.test intrinsic. This pass supports Clang's
  // control flow integrity mechanisms (-fsanitize=cfi*) and needs to run at
  // link time if CFI is enabled. The pass does nothing if CFI is disabled.
  PM.add(createLowerTypeTestsPass(ExportSummary, nullptr));

  if (OptLevel != 0)
    addLateLTOOptimizationPasses(PM);

  if (VerifyOutput)
    PM.add(createVerifierPass());
}

inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
}

inline LLVMPassManagerBuilderRef wrap(PassManagerBuilder *P) {
  return reinterpret_cast<LLVMPassManagerBuilderRef>(P);
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
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableUnitAtATime = Value;
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

void LLVMPassManagerBuilderPopulateLTOPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM,
                                                  LLVMBool Internalize,
                                                  LLVMBool RunInliner) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::PassManagerBase *LPM = unwrap(PM);

  // A small backwards compatibility hack. populateLTOPassManager used to take
  // an RunInliner option.
  if (RunInliner && !Builder->Inliner)
    Builder->Inliner = createFunctionInliningPass();

  Builder->populateLTOPassManager(*LPM);
}
