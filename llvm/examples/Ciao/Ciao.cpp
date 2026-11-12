//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/RunCodeGen.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"

using namespace llvm;

namespace llvm {
void initializeLegacyCiaoPass(PassRegistry &);
} // namespace llvm

namespace {
struct LegacyCiao : public MachineFunctionPass {
  static char ID;
  LegacyCiao() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    outs() << "Ciao: " << MF.getName() << "\n";
    return false;
  }
};
} // namespace

char LegacyCiao::ID = 0;
INITIALIZE_PASS(LegacyCiao, "ciao", "Ciao World Pass", false, false)

// Insert the pass into every legacy codegen pipeline that the host builds
static RegisterTargetPassConfigCallback RegisterLegacyCiao{
    [](auto &TM, auto &PM, auto *TPC) {
      initializeLegacyCiaoPass(*PassRegistry::getPassRegistry());
      TPC->insertPass(&GCLoweringID, &LegacyCiao::ID);
    }};

namespace {
bool ForwardCodeGen = false;

// Take over code generation and forward it to the default pipeline. Plugins
// that run an LLVM-based pipeline of their own work this way.
bool preCodeGenCallback(TargetMachine &TM, Module &M, raw_pwrite_stream &OS,
                        raw_pwrite_stream *DwoOS, CodeGenFileType CGFT,
                        bool DisableVerify, bool DisableSimplifyLibCalls,
                        const ModuleSummaryIndex *MSI,
                        IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  if (!ForwardCodeGen)
    return false;
  // Leave a trace in the output that tests can check for.
  for (Function &F : M)
    if (!F.isDeclaration())
      F.setName("ciao_" + F.getName());
  if (Error Err =
          runCodeGenPipeline(TM, M, OS, DwoOS, CGFT, DisableVerify,
                             DisableSimplifyLibCalls, MSI, std::move(VFS)))
    M.getContext().emitError(toString(std::move(Err)));
  return true;
}

Error parseArguments(ArrayRef<const char *> Args) {
  for (StringRef Arg : Args)
    if (Arg == "forward-codegen")
      ForwardCodeGen = true;
  return Error::success();
}
} // namespace

// We use the pass-plugin for preCodeGenCallback and argument parsing. The
// legacy codegen pipeline relies on the static initialization.
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Ciao", LLVM_VERSION_STRING, nullptr,
          preCodeGenCallback, parseArguments};
}
