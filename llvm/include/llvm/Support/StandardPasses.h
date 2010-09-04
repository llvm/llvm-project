//===-- llvm/Support/StandardPasses.h - Standard pass lists -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for creating a "standard" set of
// optimization passes, so that compilers and tools which use optimization
// passes use the same set of standard passes.
//
// These are implemented as inline functions so that we do not have to worry
// about link issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STANDARDPASSES_H
#define LLVM_SUPPORT_STANDARDPASSES_H

#include "llvm/PassManager.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/IPO.h"

namespace llvm {
  /// createStandardFunctionPasses - Add the standard list of function passes to
  /// the provided pass manager.
  ///
  /// \arg OptimizationLevel - The optimization level, corresponding to -O0,
  /// -O1, etc.
  static inline void createStandardFunctionPasses(PassManagerBase *PM,
                                                  unsigned OptimizationLevel);

  /// createStandardModulePasses - Add the standard list of module passes to the
  /// provided pass manager.
  ///
  /// \arg OptimizationLevel - The optimization level, corresponding to -O0,
  /// -O1, etc.
  /// \arg OptimizeSize - Whether the transformations should optimize for size.
  /// \arg UnitAtATime - Allow passes which may make global module changes.
  /// \arg UnrollLoops - Allow loop unrolling.
  /// \arg SimplifyLibCalls - Allow library calls to be simplified.
  /// \arg HaveExceptions - Whether the module may have code using exceptions.
  /// \arg InliningPass - The inlining pass to use, if any, or null. This will
  /// always be added, even at -O0.a
  static inline void createStandardModulePasses(PassManagerBase *PM,
                                                unsigned OptimizationLevel,
                                                bool OptimizeSize,
                                                bool UnitAtATime,
                                                bool UnrollLoops,
                                                bool SimplifyLibCalls,
                                                bool HaveExceptions,
                                                Pass *InliningPass);

  /// createStandardLTOPasses - Add the standard list of module passes suitable
  /// for link time optimization.
  ///
  /// Internalize - Run the internalize pass.
  /// RunInliner - Use a function inlining pass.
  /// VerifyEach - Run the verifier after each pass.
  static inline void createStandardLTOPasses(PassManagerBase *PM,
                                             bool Internalize,
                                             bool RunInliner,
                                             bool VerifyEach);

  // Implementations

  static inline void createStandardFunctionPasses(PassManagerBase *PM,
                                                  unsigned OptimizationLevel) {
    if (OptimizationLevel > 0) {
      PM->add(createCFGSimplificationPass());
      if (OptimizationLevel == 1)
        PM->add(createPromoteMemoryToRegisterPass());
      else
        PM->add(createScalarReplAggregatesPass());
      PM->add(createInstructionCombiningPass());
    }
  }

  /// createStandardModulePasses - Add the standard module passes.  This is
  /// expected to be run after the standard function passes.
  static inline void createStandardModulePasses(PassManagerBase *PM,
                                                unsigned OptimizationLevel,
                                                bool OptimizeSize,
                                                bool UnitAtATime,
                                                bool UnrollLoops,
                                                bool SimplifyLibCalls,
                                                bool HaveExceptions,
                                                Pass *InliningPass) {
    if (OptimizationLevel == 0) {
      if (InliningPass)
        PM->add(InliningPass);
      return;
    }
    
    if (UnitAtATime) {
      PM->add(createGlobalOptimizerPass());     // Optimize out global vars
      
      PM->add(createIPSCCPPass());              // IP SCCP
      PM->add(createDeadArgEliminationPass());  // Dead argument elimination
    }
    PM->add(createInstructionCombiningPass());  // Clean up after IPCP & DAE
    PM->add(createCFGSimplificationPass());     // Clean up after IPCP & DAE
    
    // Start of CallGraph SCC passes.
    if (UnitAtATime && HaveExceptions)
      PM->add(createPruneEHPass());           // Remove dead EH info
    if (InliningPass)
      PM->add(InliningPass);
    if (UnitAtATime)
      PM->add(createFunctionAttrsPass());       // Set readonly/readnone attrs
    if (OptimizationLevel > 2)
      PM->add(createArgumentPromotionPass());   // Scalarize uninlined fn args
    
    // Start of function pass.
    PM->add(createScalarReplAggregatesPass());  // Break up aggregate allocas
    if (SimplifyLibCalls)
      PM->add(createSimplifyLibCallsPass());    // Library Call Optimizations
    PM->add(createInstructionCombiningPass());  // Cleanup for scalarrepl.
    PM->add(createJumpThreadingPass());         // Thread jumps.
    PM->add(createCFGSimplificationPass());     // Merge & remove BBs
    PM->add(createInstructionCombiningPass());  // Combine silly seq's
    
    PM->add(createTailCallEliminationPass());   // Eliminate tail calls
    PM->add(createCFGSimplificationPass());     // Merge & remove BBs
    PM->add(createReassociatePass());           // Reassociate expressions
    PM->add(createLoopRotatePass());            // Rotate Loop
    PM->add(createLICMPass());                  // Hoist loop invariants
    PM->add(createLoopUnswitchPass(OptimizeSize || OptimizationLevel < 3));
    PM->add(createInstructionCombiningPass());  
    PM->add(createIndVarSimplifyPass());        // Canonicalize indvars
    PM->add(createLoopDeletionPass());          // Delete dead loops
    if (UnrollLoops)
      PM->add(createLoopUnrollPass());          // Unroll small loops
    PM->add(createInstructionCombiningPass());  // Clean up after the unroller
    if (OptimizationLevel > 1)
      PM->add(createGVNPass());                 // Remove redundancies
    PM->add(createMemCpyOptPass());             // Remove memcpy / form memset
    PM->add(createSCCPPass());                  // Constant prop with SCCP
  
    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    PM->add(createInstructionCombiningPass());
    PM->add(createJumpThreadingPass());         // Thread jumps
    PM->add(createCorrelatedValuePropagationPass());
    PM->add(createDeadStoreEliminationPass());  // Delete dead stores
    PM->add(createAggressiveDCEPass());         // Delete dead instructions
    PM->add(createCFGSimplificationPass());     // Merge & remove BBs

    if (UnitAtATime) {
      PM->add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
      PM->add(createDeadTypeEliminationPass()); // Eliminate dead types

      // GlobalOpt already deletes dead functions and globals, at -O3 try a
      // late pass of GlobalDCE.  It is capable of deleting dead cycles.
      if (OptimizationLevel > 2)
        PM->add(createGlobalDCEPass());         // Remove dead fns and globals.
    
      if (OptimizationLevel > 1)
        PM->add(createConstantMergePass());       // Merge dup global constants
    }
  }

  static inline void addOnePass(PassManagerBase *PM, Pass *P, bool AndVerify) {
    PM->add(P);

    if (AndVerify)
      PM->add(createVerifierPass());
  }

  static inline void createStandardLTOPasses(PassManagerBase *PM,
                                             bool Internalize,
                                             bool RunInliner,
                                             bool VerifyEach) {
    // Now that composite has been compiled, scan through the module, looking
    // for a main function.  If main is defined, mark all other functions
    // internal.
    if (Internalize)
      addOnePass(PM, createInternalizePass(true), VerifyEach);

    // Propagate constants at call sites into the functions they call.  This
    // opens opportunities for globalopt (and inlining) by substituting function
    // pointers passed as arguments to direct uses of functions.  
    addOnePass(PM, createIPSCCPPass(), VerifyEach);

    // Now that we internalized some globals, see if we can hack on them!
    addOnePass(PM, createGlobalOptimizerPass(), VerifyEach);
    
    // Linking modules together can lead to duplicated global constants, only
    // keep one copy of each constant...
    addOnePass(PM, createConstantMergePass(), VerifyEach);
    
    // Remove unused arguments from functions...
    addOnePass(PM, createDeadArgEliminationPass(), VerifyEach);

    // Reduce the code after globalopt and ipsccp.  Both can open up significant
    // simplification opportunities, and both can propagate functions through
    // function pointers.  When this happens, we often have to resolve varargs
    // calls, etc, so let instcombine do this.
    addOnePass(PM, createInstructionCombiningPass(), VerifyEach);

    // Inline small functions
    if (RunInliner)
      addOnePass(PM, createFunctionInliningPass(), VerifyEach);

    addOnePass(PM, createPruneEHPass(), VerifyEach);   // Remove dead EH info.
    // Optimize globals again if we ran the inliner.
    if (RunInliner)
      addOnePass(PM, createGlobalOptimizerPass(), VerifyEach);
    addOnePass(PM, createGlobalDCEPass(), VerifyEach); // Remove dead functions.

    // If we didn't decide to inline a function, check to see if we can
    // transform it to pass arguments by value instead of by reference.
    addOnePass(PM, createArgumentPromotionPass(), VerifyEach);

    // The IPO passes may leave cruft around.  Clean up after them.
    addOnePass(PM, createInstructionCombiningPass(), VerifyEach);
    addOnePass(PM, createJumpThreadingPass(), VerifyEach);
    // Break up allocas
    addOnePass(PM, createScalarReplAggregatesPass(), VerifyEach);

    // Run a few AA driven optimizations here and now, to cleanup the code.
    addOnePass(PM, createFunctionAttrsPass(), VerifyEach); // Add nocapture.
    addOnePass(PM, createGlobalsModRefPass(), VerifyEach); // IP alias analysis.

    addOnePass(PM, createLICMPass(), VerifyEach);      // Hoist loop invariants.
    addOnePass(PM, createGVNPass(), VerifyEach);       // Remove redundancies.
    addOnePass(PM, createMemCpyOptPass(), VerifyEach); // Remove dead memcpys.
    // Nuke dead stores.
    addOnePass(PM, createDeadStoreEliminationPass(), VerifyEach);

    // Cleanup and simplify the code after the scalar optimizations.
    addOnePass(PM, createInstructionCombiningPass(), VerifyEach);

    addOnePass(PM, createJumpThreadingPass(), VerifyEach);
    
    // Delete basic blocks, which optimization passes may have killed.
    addOnePass(PM, createCFGSimplificationPass(), VerifyEach);

    // Now that we have optimized the program, discard unreachable functions.
    addOnePass(PM, createGlobalDCEPass(), VerifyEach);
  }
}

#endif
