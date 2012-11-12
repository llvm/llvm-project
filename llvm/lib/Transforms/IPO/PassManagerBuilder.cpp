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

#include "llvm/PassManager.h"
#include "llvm/DefaultPasses.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

static cl::opt<bool>
RunLoopVectorization("vectorize-loops",
                     cl::desc("Run the Loop vectorization passes"));

static cl::opt<bool>
RunBBVectorization("vectorize", cl::desc("Run the BB vectorization passes"));

static cl::opt<bool>
UseGVNAfterVectorization("use-gvn-after-vectorization",
  cl::init(false), cl::Hidden,
  cl::desc("Run GVN instead of Early CSE after vectorization passes"));

static cl::opt<bool> UseNewSROA("use-new-sroa",
  cl::init(true), cl::Hidden,
  cl::desc("Enable the new, experimental SROA pass"));

PassManagerBuilder::PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = 0;
    Inliner = 0;
    DisableSimplifyLibCalls = false;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
    Vectorize = RunBBVectorization;
    LoopVectorize = RunLoopVectorization;
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
                                           PassManagerBase &PM) const {
  for (unsigned i = 0, e = GlobalExtensions->size(); i != e; ++i)
    if ((*GlobalExtensions)[i].first == ETy)
      (*GlobalExtensions)[i].second(*this, PM);
  for (unsigned i = 0, e = Extensions.size(); i != e; ++i)
    if (Extensions[i].first == ETy)
      Extensions[i].second(*this, PM);
}

void
PassManagerBuilder::addInitialAliasAnalysisPasses(PassManagerBase &PM) const {
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAliasAnalysisPass());
  PM.add(createBasicAliasAnalysisPass());
}

void PassManagerBuilder::populateFunctionPassManager(FunctionPassManager &FPM) {
  addExtensionsToPM(EP_EarlyAsPossible, FPM);

  // Add LibraryInfo if we have some.
  if (LibraryInfo) FPM.add(new TargetLibraryInfo(*LibraryInfo));

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

void PassManagerBuilder::populateModulePassManager(PassManagerBase &MPM) {
  // If all optimizations are disabled, just run the always-inline pass.
  if (OptLevel == 0) {
    if (Inliner) {
      MPM.add(Inliner);
      Inliner = 0;
    }

    // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
    // pass manager, but we don't want to add extensions into that pass manager.
    // To prevent this we must insert a no-op module pass to reset the pass
    // manager to get the same behavior as EP_OptimizerLast in non-O0 builds.
    if (!GlobalExtensions->empty() || !Extensions.empty())
      MPM.add(createBarrierNoopPass());

    addExtensionsToPM(EP_EnabledOnOptLevel0, MPM);
    return;
  }

  // Add LibraryInfo if we have some.
  if (LibraryInfo) MPM.add(new TargetLibraryInfo(*LibraryInfo));

  addInitialAliasAnalysisPasses(MPM);

  if (!DisableUnitAtATime) {
    addExtensionsToPM(EP_ModuleOptimizerEarly, MPM);

    MPM.add(createGlobalOptimizerPass());     // Optimize out global vars

    MPM.add(createIPSCCPPass());              // IP SCCP
    MPM.add(createDeadArgEliminationPass());  // Dead argument elimination

    MPM.add(createInstructionCombiningPass());// Clean up after IPCP & DAE
    MPM.add(createCFGSimplificationPass());   // Clean up after IPCP & DAE
  }

  // Start of CallGraph SCC passes.
  if (!DisableUnitAtATime)
    MPM.add(createPruneEHPass());             // Remove dead EH info
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = 0;
  }
  if (!DisableUnitAtATime)
    MPM.add(createFunctionAttrsPass());       // Set readonly/readnone attrs
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass());   // Scalarize uninlined fn args

  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  if (UseNewSROA)
    MPM.add(createSROAPass(/*RequiresDomTree*/ false));
  else
    MPM.add(createScalarReplAggregatesPass(-1, false));
  MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
  if (!DisableSimplifyLibCalls)
    MPM.add(createSimplifyLibCallsPass());    // Library Call Optimizations
  MPM.add(createJumpThreadingPass());         // Thread jumps.
  MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Combine silly seq's

  MPM.add(createTailCallEliminationPass());   // Eliminate tail calls
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions
  MPM.add(createLoopRotatePass());            // Rotate Loop
  MPM.add(createLICMPass());                  // Hoist loop invariants
  MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3));
  MPM.add(createInstructionCombiningPass());
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  MPM.add(createLoopDeletionPass());          // Delete dead loops

  if (LoopVectorize) {
    MPM.add(createLoopVectorizePass());
    MPM.add(createLICMPass());
  }

  if (!DisableUnrollLoops)
    MPM.add(createLoopUnrollPass());          // Unroll small loops
  addExtensionsToPM(EP_LoopOptimizerEnd, MPM);

  if (OptLevel > 1)
    MPM.add(createGVNPass());                 // Remove redundancies
  MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
  MPM.add(createJumpThreadingPass());         // Thread jumps
  MPM.add(createCorrelatedValuePropagationPass());
  MPM.add(createDeadStoreEliminationPass());  // Delete dead stores

  addExtensionsToPM(EP_ScalarOptimizerLate, MPM);

  if (Vectorize) {
    MPM.add(createBBVectorizePass());
    MPM.add(createInstructionCombiningPass());
    if (OptLevel > 1 && UseGVNAfterVectorization)
      MPM.add(createGVNPass());                   // Remove redundancies
    else
      MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
  }

  MPM.add(createAggressiveDCEPass());         // Delete dead instructions
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Clean up after everything.

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
  addExtensionsToPM(EP_OptimizerLast, MPM);
}

void PassManagerBuilder::populateLTOPassManager(PassManagerBase &PM,
                                                bool Internalize,
                                                bool RunInliner,
                                                bool DisableGVNLoadPRE) {
  // Provide AliasAnalysis services for optimizations.
  addInitialAliasAnalysisPasses(PM);

  // Now that composite has been compiled, scan through the module, looking
  // for a main function.  If main is defined, mark all other functions
  // internal.
  if (Internalize) {
    std::vector<const char*> E;
    E.push_back("main");
    PM.add(createInternalizePass(E));
  }

  // Propagate constants at call sites into the functions they call.  This
  // opens opportunities for globalopt (and inlining) by substituting function
  // pointers passed as arguments to direct uses of functions.
  PM.add(createIPSCCPPass());

  // Now that we internalized some globals, see if we can hack on them!
  PM.add(createGlobalOptimizerPass());

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

  // Inline small functions
  if (RunInliner)
    PM.add(createFunctionInliningPass());

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
  PM.add(createJumpThreadingPass());
  // Break up allocas
  if (UseNewSROA)
    PM.add(createSROAPass());
  else
    PM.add(createScalarReplAggregatesPass());

  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createFunctionAttrsPass()); // Add nocapture.
  PM.add(createGlobalsModRefPass()); // IP alias analysis.

  PM.add(createLICMPass());                 // Hoist loop invariants.
  PM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.
  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass());

  // Cleanup and simplify the code after the scalar optimizations.
  PM.add(createInstructionCombiningPass());

  PM.add(createJumpThreadingPass());

  // Delete basic blocks, which optimization passes may have killed.
  PM.add(createCFGSimplificationPass());

  // Now that we have optimized the program, discard unreachable functions.
  PM.add(createGlobalDCEPass());
}

LLVMPassManagerBuilderRef LLVMPassManagerBuilderCreate(void) {
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
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableSimplifyLibCalls = Value;
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
  FunctionPassManager *FPM = unwrap<FunctionPassManager>(PM);
  Builder->populateFunctionPassManager(*FPM);
}

void
LLVMPassManagerBuilderPopulateModulePassManager(LLVMPassManagerBuilderRef PMB,
                                                LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  PassManagerBase *MPM = unwrap(PM);
  Builder->populateModulePassManager(*MPM);
}

void LLVMPassManagerBuilderPopulateLTOPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM,
                                                  bool Internalize,
                                                  bool RunInliner) {
  PassManagerBuilder *Builder = unwrap(PMB);
  PassManagerBase *LPM = unwrap(PM);
  Builder->populateLTOPassManager(*LPM, Internalize, RunInliner);
}
