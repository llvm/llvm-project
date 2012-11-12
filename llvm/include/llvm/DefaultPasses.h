//===- llvm/DefaultPasses.h - Default Pass Support code --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file defines the infrastructure for registering the standard pass list.
// This defines sets of standard optimizations that plugins can modify and
// front ends can use.
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEFAULT_PASS_SUPPORT_H
#define LLVM_DEFAULT_PASS_SUPPORT_H

#include "llvm/PassSupport.h"

namespace llvm {

class PassManagerBase;

/// Unique identifiers for the default standard passes.  The addresses of
/// these symbols are used to uniquely identify passes from the default list.
namespace DefaultStandardPasses {
extern unsigned char AggressiveDCEID;
extern unsigned char ArgumentPromotionID;
extern unsigned char BasicAliasAnalysisID;
extern unsigned char CFGSimplificationID;
extern unsigned char ConstantMergeID;
extern unsigned char CorrelatedValuePropagationID;
extern unsigned char DeadArgEliminationID;
extern unsigned char DeadStoreEliminationID;
extern unsigned char EarlyCSEID;
extern unsigned char FunctionAttrsID;
extern unsigned char FunctionInliningID;
extern unsigned char GVNID;
extern unsigned char GlobalDCEID;
extern unsigned char GlobalOptimizerID;
extern unsigned char GlobalsModRefID;
extern unsigned char IPSCCPID;
extern unsigned char IndVarSimplifyID;
extern unsigned char InlinerPlaceholderID;
extern unsigned char InstructionCombiningID;
extern unsigned char JumpThreadingID;
extern unsigned char LICMID;
extern unsigned char LoopDeletionID;
extern unsigned char LoopIdiomID;
extern unsigned char LoopRotateID;
extern unsigned char LoopUnrollID;
extern unsigned char LoopUnswitchID;
extern unsigned char MemCpyOptID;
extern unsigned char PruneEHID;
extern unsigned char ReassociateID;
extern unsigned char SCCPID;
extern unsigned char ScalarReplAggregatesID;
extern unsigned char SimplifyLibCallsID;
extern unsigned char StripDeadPrototypesID;
extern unsigned char TailCallEliminationID;
extern unsigned char TypeBasedAliasAnalysisID;
}

/// StandardPass - The class responsible for maintaining the lists of standard 
class StandardPass {
  friend class RegisterStandardPassLists;
  public:
  /// Predefined standard sets of passes
  enum StandardSet {
    AliasAnalysis,
    Function,
    Module,
    LTO
  };
  /// Flags to specify whether a pass should be enabled.  Passes registered
  /// with the standard sets may specify a minimum optimization level and one
  /// or more flags that must be set when constructing the set for the pass to
  /// be used.
  enum OptimizationFlags {
    /// Optimize for size was requested.
    OptimizeSize = 1<<0,
    /// Allow passes which may make global module changes.
    UnitAtATime = 1<<1,
    /// UnrollLoops - Allow loop unrolling.
    UnrollLoops = 1<<2,
    /// Allow library calls to be simplified.
    SimplifyLibCalls = 1<<3,
    /// Whether the module may have code using exceptions.
    HaveExceptions = 1<<4,
    // Run an inliner pass as part of this set.
    RunInliner = 1<<5
  };
  enum OptimizationFlagComponents {
    /// The low bits are used to store the optimization level.  When requesting
    /// passes, this should store the requested optimisation level.  When
    /// setting passes, this should set the minimum optimization level at which
    /// the pass will run.
    OptimizationLevelMask=0xf,
    /// The maximum optimisation level at which the pass is run.
    MaxOptimizationLevelMask=0xf0,
    // Flags that must be set
    RequiredFlagMask=0xff00,
    // Flags that may not be set.
    DisallowedFlagMask=0xff0000,
    MaxOptimizationLevelShift=4,
    RequiredFlagShift=8,
    DisallowedFlagShift=16
  };
  /// Returns the optimisation level from a set of flags.
  static unsigned OptimizationLevel(unsigned flags) {
      return flags & OptimizationLevelMask;
  }
  /// Returns the maximum optimization level for this set of flags
  static unsigned MaxOptimizationLevel(unsigned flags) {
      return (flags & MaxOptimizationLevelMask) >> 4;
  }
  /// Constructs a set of flags from the specified minimum and maximum
  /// optimisation level
  static unsigned OptimzationFlags(unsigned minLevel=0, unsigned maxLevel=0xf,
      unsigned requiredFlags=0, unsigned disallowedFlags=0) {
    return ((minLevel & OptimizationLevelMask) |
            ((maxLevel<<MaxOptimizationLevelShift) & MaxOptimizationLevelMask)
            | ((requiredFlags<<RequiredFlagShift) & RequiredFlagMask)
            | ((disallowedFlags<<DisallowedFlagShift) & DisallowedFlagMask));
  }
  /// Returns the flags that must be set for this to match
  static unsigned RequiredFlags(unsigned flags) {
      return (flags & RequiredFlagMask) >> RequiredFlagShift;
  }
  /// Returns the flags that must not be set for this to match
  static unsigned DisallowedFlags(unsigned flags) {
      return (flags & DisallowedFlagMask) >> DisallowedFlagShift;
  }
  /// Register a standard pass in the specified set.  If flags is non-zero,
  /// then the pass will only be returned when the specified flags are set.
  template<typename passName>
  class RegisterStandardPass {
    public:
    RegisterStandardPass(StandardSet set, unsigned char *runBefore=0,
        unsigned flags=0, unsigned char *ID=0) {
      // Use the pass's ID if one is not specified
      RegisterDefaultPass(PassInfo::NormalCtor_t(callDefaultCtor<passName>),
               ID ? ID : (unsigned char*)&passName::ID, runBefore, set, flags);
    }
  };
  /// Adds the passes from the specified set to the provided pass manager
  static void AddPassesFromSet(PassManagerBase *PM,
                               StandardSet set,
                               unsigned flags=0,
                               bool VerifyEach=false,
                               Pass *inliner=0);
  private:
  /// Registers the default passes.  This is set by RegisterStandardPassLists
  /// and is called lazily.
  static void (*RegisterDefaultPasses)(void);
  /// Creates the verifier pass that is inserted when a VerifyEach is passed to
  /// AddPassesFromSet()
  static Pass* (*CreateVerifierPass)(void);
  /// Registers the pass
  static void RegisterDefaultPass(PassInfo::NormalCtor_t constructor,
                                  unsigned char *newPass,
                                  unsigned char *oldPass,
                                  StandardSet set,
                                  unsigned flags=0);
};

} // namespace llvm

#endif
