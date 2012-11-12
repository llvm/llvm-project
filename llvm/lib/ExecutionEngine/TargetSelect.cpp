//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This just asks the TargetRegistry for the appropriate target to use, and
// allows the user to specify a specific one on the commandline with -march=x,
// -mcpu=y, and -mattr=a,-b,+c. Clients should initialize targets prior to
// calling selectTarget().
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Module.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

TargetMachine *EngineBuilder::selectTarget() {
  Triple TT;

  // MCJIT can generate code for remote targets, but the old JIT and Interpreter
  // must use the host architecture.
  if (UseMCJIT && WhichEngine != EngineKind::Interpreter && M)
    TT.setTriple(M->getTargetTriple());
  else
    TT.setTriple(LLVM_HOSTTRIPLE);
  return selectTarget(TT, MArch, MCPU, MAttrs);
}

/// selectTarget - Pick a target either via -march or by guessing the native
/// arch.  Add any CPU features specified via -mcpu or -mattr.
TargetMachine *EngineBuilder::selectTarget(const Triple &TargetTriple,
                              StringRef MArch,
                              StringRef MCPU,
                              const SmallVectorImpl<std::string>& MAttrs) {
  Triple TheTriple(TargetTriple);
  if (TheTriple.getTriple().empty())
    TheTriple.setTriple(sys::getDefaultTargetTriple());

  // Adjust the triple to match what the user requested.
  const Target *TheTarget = 0;
  if (!MArch.empty()) {
    for (TargetRegistry::iterator it = TargetRegistry::begin(),
           ie = TargetRegistry::end(); it != ie; ++it) {
      if (MArch == it->getName()) {
        TheTarget = &*it;
        break;
      }
    }

    if (!TheTarget) {
      if (ErrorStr)
        *ErrorStr = "No available targets are compatible with this -march, "
                    "see -version for the available targets.\n";
      return 0;
    }

    // Adjust the triple to match (if known), otherwise stick with the
    // requested/host triple.
    Triple::ArchType Type = Triple::getArchTypeForLLVMName(MArch);
    if (Type != Triple::UnknownArch)
      TheTriple.setArch(Type);
  } else {
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TheTriple.getTriple(), Error);
    if (TheTarget == 0) {
      if (ErrorStr)
        *ErrorStr = Error;
      return 0;
    }
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (!MAttrs.empty()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Allocate a target...
  TargetMachine *Target = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                                         MCPU, FeaturesStr,
                                                         Options,
                                                         RelocModel, CMModel,
                                                         OptLevel);
  assert(Target && "Could not allocate target machine!");
  return Target;
}
