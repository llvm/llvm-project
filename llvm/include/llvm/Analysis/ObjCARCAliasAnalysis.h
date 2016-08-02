//===- ObjCARCAliasAnalysis.h - ObjC ARC Alias Analysis ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares a simple ARC-aware AliasAnalysis using special knowledge
/// of Objective C to enhance other optimization passes which rely on the Alias
/// Analysis infrastructure.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_OBJCARCALIASANALYSIS_H
#define LLVM_ANALYSIS_OBJCARCALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Pass.h"

namespace llvm {
namespace objcarc {

/// \brief This is a simple alias analysis implementation that uses knowledge
/// of ARC constructs to answer queries.
///
/// TODO: This class could be generalized to know about other ObjC-specific
/// tricks. Such as knowing that ivars in the non-fragile ABI are non-aliasing
/// even though their offsets are dynamic.
class ObjCARCAAResult : public AAResultBase<ObjCARCAAResult> {
  friend AAResultBase<ObjCARCAAResult>;

  const DataLayout &DL;

public:
  explicit ObjCARCAAResult(const DataLayout &DL, const TargetLibraryInfo &TLI)
      : AAResultBase(TLI), DL(DL) {}
  ObjCARCAAResult(ObjCARCAAResult &&Arg)
      : AAResultBase(std::move(Arg)), DL(Arg.DL) {}

  /// Handle invalidation events from the new pass manager.
  ///
  /// By definition, this result is stateless and so remains valid.
  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB);
  bool pointsToConstantMemory(const MemoryLocation &Loc, bool OrLocal);

  using AAResultBase::getModRefBehavior;
  FunctionModRefBehavior getModRefBehavior(const Function *F);

  using AAResultBase::getModRefInfo;
  ModRefInfo getModRefInfo(ImmutableCallSite CS, const MemoryLocation &Loc);
};

/// Analysis pass providing a never-invalidated alias analysis result.
class ObjCARCAA {
public:
  typedef ObjCARCAAResult Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  ObjCARCAAResult run(Function &F, AnalysisManager<Function> *AM);

  /// \brief Provide access to a name for this pass for debugging purposes.
  static StringRef name() { return "ObjCARCAA"; }

private:
  static char PassID;
};

/// Legacy wrapper pass to provide the ObjCARCAAResult object.
class ObjCARCAAWrapperPass : public ImmutablePass {
  std::unique_ptr<ObjCARCAAResult> Result;

public:
  static char ID;

  ObjCARCAAWrapperPass();

  ObjCARCAAResult &getResult() { return *Result; }
  const ObjCARCAAResult &getResult() const { return *Result; }

  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace objcarc
} // namespace llvm

#endif
