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
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/FunctionInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
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

static cl::opt<bool> UseNewSROA("use-new-sroa",
  cl::init(true), cl::Hidden,
  cl::desc("Enable the new, experimental SROA pass"));

static cl::opt<bool>
RunLoopRerolling("reroll-loops", cl::Hidden,
                 cl::desc("Run the loop rerolling pass"));

static cl::opt<bool>
RunFloat2Int("float-to-int", cl::Hidden, cl::init(true),
             cl::desc("Run the float2int (float demotion) pass"));

static cl::opt<bool> RunLoadCombine("combine-loads", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Run the load combining pass"));

static cl::opt<bool>
RunSLPAfterLoopVectorization("run-slp-after-loop-vectorization",
  cl::init(true), cl::Hidden,
  cl::desc("Run the SLP vectorizer (and BB vectorizer) after the Loop "
           "vectorizer instead of before"));

static cl::opt<bool> UseCFLAA("use-cfl-aa",
  cl::init(false), cl::Hidden,
  cl::desc("Enable the new, experimental CFL alias analysis"));

static cl::opt<bool>
EnableMLSM("mlsm", cl::init(true), cl::Hidden,
           cl::desc("Enable motion of merged load and store"));

static cl::opt<bool> EnableLoopInterchange(
    "enable-loopinterchange", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopInterchange Pass"));

static cl::opt<bool> EnableLoopDistribute(
    "enable-loop-distribute", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopDistribution Pass"));

static cl::opt<bool> EnableNonLTOGlobalsModRef(
    "enable-non-lto-gmr", cl::init(true), cl::Hidden,
    cl::desc(
        "Enable the GlobalsModRef AliasAnalysis outside of the LTO pipeline."));

static cl::opt<bool> EnableLoopLoadElim(
    "enable-loop-load-elim", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopLoadElimination Pass"));

PassManagerBuilder::PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = nullptr;
    Inliner = nullptr;
    FunctionIndex = nullptr;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
    BBVectorize = RunBBVectorization;
    SLPVectorize = RunSLPVectorization;
    LoopVectorize = RunLoopVectorization;
    RerollLoops = RunLoopRerolling;
    LoadCombine = RunLoadCombine;
    DisableGVNLoadPRE = false;
    VerifyInput = false;
    VerifyOutput = false;
    MergeFunctions = false;
    PrepareForLTO = false;
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
  GlobalExtensions->push_back(std::make_pair(Ty, Fn));
}

void PassManagerBuilder::addExtension(ExtensionPointTy Ty, ExtensionFn Fn) {
  Extensions.push_back(std::make_pair(Ty, Fn));
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
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  if (UseCFLAA)
    PM.add(createCFLAAWrapperPass());
  PM.add(createTypeBasedAAWrapperPass());
  PM.add(createScopedNoAliasAAWrapperPass());
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
  if (UseNewSROA)
    FPM.add(createSROAPass());
  else
    FPM.add(createScalarReplAggregatesPass());
  FPM.add(createEarlyCSEPass());
  FPM.add(createLowerExpectIntrinsicPass());
}

void PassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
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
    else if (!GlobalExtensions->empty() || !Extensions.empty())
      MPM.add(createBarrierNoopPass());

    addExtensionsToPM(EP_EnabledOnOptLevel0, MPM);
    return;
  }

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    MPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  addInitialAliasAnalysisPasses(MPM);

  if (!DisableUnitAtATime) {
    addExtensionsToPM(EP_ModuleOptimizerEarly, MPM);

    MPM.add(createIPSCCPPass());              // IP SCCP
    MPM.add(createGlobalOptimizerPass());     // Optimize out global vars
    // Promote any localized global vars
    MPM.add(createPromoteMemoryToRegisterPass());

    MPM.add(createDeadArgEliminationPass());  // Dead argument elimination

    MPM.add(createInstructionCombiningPass());// Clean up after IPCP & DAE
    addExtensionsToPM(EP_Peephole, MPM);
    MPM.add(createCFGSimplificationPass());   // Clean up after IPCP & DAE
  }

  if (EnableNonLTOGlobalsModRef)
    // We add a module alias analysis pass here. In part due to bugs in the
    // analysis infrastructure this "works" in that the analysis stays alive
    // for the entire SCC pass run below.
    MPM.add(createGlobalsAAWrapperPass());

  // Start of CallGraph SCC passes.
  if (!DisableUnitAtATime)
    MPM.add(createPruneEHPass());             // Remove dead EH info
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = nullptr;
  }
  if (!DisableUnitAtATime)
    MPM.add(createFunctionAttrsPass());       // Set readonly/readnone attrs
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass());   // Scalarize uninlined fn args

  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  if (UseNewSROA)
    MPM.add(createSROAPass());
  else
    MPM.add(createScalarReplAggregatesPass(-1, false));
  MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
  MPM.add(createJumpThreadingPass());         // Thread jumps.
  MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Combine silly seq's
  addExtensionsToPM(EP_Peephole, MPM);

  MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions
  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));
  MPM.add(createLICMPass());                  // Hoist loop invariants
  MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3));
  MPM.add(createCFGSimplificationPass());
  MPM.add(createInstructionCombiningPass());
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  MPM.add(createLoopDeletionPass());          // Delete dead loops
  if (EnableLoopInterchange) {
    MPM.add(createLoopInterchangePass()); // Interchange loops
    MPM.add(createCFGSimplificationPass());
  }
  if (!DisableUnrollLoops)
    MPM.add(createSimpleLoopUnrollPass());    // Unroll small loops
  addExtensionsToPM(EP_LoopOptimizerEnd, MPM);

  if (OptLevel > 1) {
    if (EnableMLSM)
      MPM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    MPM.add(createGVNPass(DisableGVNLoadPRE));  // Remove redundancies
  }
  MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass());        // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
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
      MPM.add(createInstructionCombiningPass());
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass());
    }
  }

  if (LoadCombine)
    MPM.add(createLoadCombinePass());

  MPM.add(createAggressiveDCEPass());         // Delete dead instructions
  MPM.add(createCFGSimplificationPass()); // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Clean up after everything.
  addExtensionsToPM(EP_Peephole, MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  MPM.add(createBarrierNoopPass());

  if (!DisableUnitAtATime && OptLevel > 1 && !PrepareForLTO) {
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

  if (RunFloat2Int)
    MPM.add(createFloat2IntPass());

  addExtensionsToPM(EP_VectorizerStart, MPM);

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.
  if (EnableLoopDistribute)
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
  MPM.add(createInstructionCombiningPass());
  if (OptLevel > 1 && ExtraVectorizerPasses) {
    // At higher optimization levels, try to clean up any runtime overlap and
    // alignment checks inserted by the vectorizer. We want to track correllated
    // runtime checks for two inner loops in the same outer loop, fold any
    // common computations, hoist loop-invariant aspects out of any outer loop,
    // and unswitch the runtime checks if possible. Once hoisted, we may have
    // dead (or speculatable) control flows or more combining opportunities.
    MPM.add(createEarlyCSEPass());
    MPM.add(createCorrelatedValuePropagationPass());
    MPM.add(createInstructionCombiningPass());
    MPM.add(createLICMPass());
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3));
    MPM.add(createCFGSimplificationPass());
    MPM.add(createInstructionCombiningPass());
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
      MPM.add(createInstructionCombiningPass());
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass());
    }
  }

  addExtensionsToPM(EP_Peephole, MPM);
  MPM.add(createCFGSimplificationPass());
  MPM.add(createInstructionCombiningPass());

  if (!DisableUnrollLoops) {
    MPM.add(createLoopUnrollPass());    // Unroll small loops

    // LoopUnroll may generate some redundency to cleanup.
    MPM.add(createInstructionCombiningPass());

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

  addExtensionsToPM(EP_OptimizerLast, MPM);
}

void PassManagerBuilder::addLTOOptimizationPasses(legacy::PassManagerBase &PM) {
  // Provide AliasAnalysis services for optimizations.
  addInitialAliasAnalysisPasses(PM);

  if (FunctionIndex)
    PM.add(createFunctionImportPass(FunctionIndex));

  // Propagate constants at call sites into the functions they call.  This
  // opens opportunities for globalopt (and inlining) by substituting function
  // pointers passed as arguments to direct uses of functions.
  PM.add(createIPSCCPPass());

  // Now that we internalized some globals, see if we can hack on them!
  PM.add(createFunctionAttrsPass()); // Add norecurse if possible.
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
  PM.add(createInstructionCombiningPass());
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
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);
  PM.add(createJumpThreadingPass());

  // Break up allocas
  if (UseNewSROA)
    PM.add(createSROAPass());
  else
    PM.add(createScalarReplAggregatesPass());

  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createFunctionAttrsPass()); // Add nocapture.
  PM.add(createGlobalsAAWrapperPass()); // IP alias analysis.

  PM.add(createLICMPass());                 // Hoist loop invariants.
  if (EnableMLSM)
    PM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds.
  PM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.

  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass());

  // More loops are countable; try to optimize them.
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopDeletionPass());
  if (EnableLoopInterchange)
    PM.add(createLoopInterchangePass());

  PM.add(createLoopVectorizePass(true, LoopVectorize));

  // Now that we've optimized loops (in particular loop induction variables),
  // we may have exposed more scalar opportunities. Run parts of the scalar
  // optimizer again at this point.
  PM.add(createInstructionCombiningPass()); // Initial cleanup
  PM.add(createCFGSimplificationPass()); // if-convert
  PM.add(createSCCPPass()); // Propagate exposed constants
  PM.add(createInstructionCombiningPass()); // Clean up again
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
  PM.add(createInstructionCombiningPass());
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

void PassManagerBuilder::populateLTOPassManager(legacy::PassManagerBase &PM) {
  if (LibraryInfo)
    PM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (VerifyInput)
    PM.add(createVerifierPass());

  if (OptLevel > 1)
    addLTOOptimizationPasses(PM);

  // Create a function that performs CFI checks for cross-DSO calls with targets
  // in the current module.
  PM.add(createCrossDSOCFIPass());

  // Lower bit sets to globals. This pass supports Clang's control flow
  // integrity mechanisms (-fsanitize=cfi*) and needs to run at link time if CFI
  // is enabled. The pass does nothing if CFI is disabled.
  PM.add(createLowerBitSetsPass());

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
