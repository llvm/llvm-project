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

// We don't use the pass-plugin interface (yet), but -load-pass-plugin requires
// the entry point.
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Ciao", LLVM_VERSION_STRING};
}
