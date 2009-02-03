//===-- TargetSelect.cpp - Target Chooser Code ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This just asks the TargetMachineRegistry for the appropriate JIT to use, and
// allows the user to specify a specific one on the commandline with -march=x.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Support/RegistryParser.h"
#include "llvm/Support/Streams.h"
#include "llvm/Target/SubtargetFeature.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

static cl::opt<const TargetMachineRegistry::entry*, false,
               RegistryParser<TargetMachine> >
MArch("march", cl::desc("Architecture to generate assembly for:"));

static cl::opt<std::string>
MCPU("mcpu",
  cl::desc("Target a specific cpu type (-mcpu=help for details)"),
  cl::value_desc("cpu-name"),
  cl::init(""));

static cl::list<std::string>
MAttrs("mattr",
  cl::CommaSeparated,
  cl::desc("Target specific attributes (-mattr=help for details)"),
  cl::value_desc("a1,+a2,-a3,..."));

/// createInternal - Create an return a new JIT compiler if there is one
/// available for the current target.  Otherwise, return null.
///
ExecutionEngine *JIT::createJIT(ModuleProvider *MP, std::string *ErrorStr,
                                JITMemoryManager *JMM, bool Fast) {
  const TargetMachineRegistry::entry *TheArch = MArch;
  if (TheArch == 0) {
    std::string Error;
    TheArch = TargetMachineRegistry::getClosestTargetForJIT(Error);
    if (TheArch == 0) {
      if (ErrorStr)
        *ErrorStr = Error;
      return 0;
    }
  } else if (TheArch->JITMatchQualityFn() == 0) {
    cerr << "WARNING: This target JIT is not designed for the host you are"
         << " running.  If bad things happen, please choose a different "
         << "-march switch.\n";
  }

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (!MCPU.empty() || !MAttrs.empty()) {
    SubtargetFeatures Features;
    Features.setCPU(MCPU);
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  // Allocate a target...
  TargetMachine *Target = TheArch->CtorFn(*MP->getModule(), FeaturesStr);
  assert(Target && "Could not allocate target machine!");

  // If the target supports JIT code generation, return a new JIT now.
  if (TargetJITInfo *TJ = Target->getJITInfo())
    return new JIT(MP, *Target, *TJ, JMM, Fast);

  if (ErrorStr)
    *ErrorStr = "target does not support JIT code generation";
  return 0;
}
