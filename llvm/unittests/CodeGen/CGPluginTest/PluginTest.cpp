//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugin/CodeGenTestPass.h"

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Config/config.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"
#include "llvm/Target/TargetMachine.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
class CGPluginTests : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargetMCs();
    InitializeAllTargetInfos();
    InitializeAllTargets();
  }
};
} // namespace llvm

TEST_F(CGPluginTests, LoadPlugin) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  auto PluginPath{std::string{"CGTestPlugin"} + LLVM_PLUGIN_EXT};

  std::string Error;
  auto Library = sys::DynamicLibrary::getLibrary(PluginPath.c_str(), &Error);
  ASSERT_TRUE(Library.isValid()) << Error;
  sys::DynamicLibrary::closeLibrary(Library);
}

TEST_F(CGPluginTests, ExecuteCallback) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  volatile bool CallbackExecuted = false;
  volatile bool MPassExecuted = false;

  RegisterTargetPassConfigCallback X{[&](auto &TM, auto &PM, auto *TPC) {
    CallbackExecuted = true;
    TPC->insertPass(&GCLoweringID, &CodeGenTest::ID);
  }};

  CodeGenTest::RunCallback = [&] { MPassExecuted = true; };

  TargetOptions Options;
  std::unique_ptr<MCContext> MCC;
  for (auto T : TargetRegistry::targets()) {
    if (!T.hasTargetMachine())
      continue;
    Triple TT{T.getName(), "", ""};
    auto *TM = T.createTargetMachine(TT, "", "", Options, std::nullopt,
                                     std::nullopt, CodeGenOptLevel::Default);
    ASSERT_TRUE(TM);

    legacy::PassManager PM;
    MCC.reset(new MCContext(TT, TM->getMCAsmInfo(), TM->getMCRegisterInfo(),
                            TM->getMCSubtargetInfo()));
    auto *PtrMCC = MCC.get();
    CallbackExecuted = false;
    MPassExecuted = false;
    if (TM->addPassesToEmitMC(PM, PtrMCC, outs()) == true)
      continue;
    ASSERT_TRUE(CallbackExecuted) << T.getName() << " callback failed";
    ASSERT_TRUE(MPassExecuted) << T.getName() << " MachinePass failed";
  }
}
