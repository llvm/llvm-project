//===- unittests/Passes/Plugins/PluginsTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Config/config.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "gtest/gtest.h"

#include "TestPlugin.h"

#include <cstdint>

using namespace llvm;

void anchor() {}

static std::string LibPath(const std::string Name = "TestPlugin") {
  const auto &Argvs = testing::internal::GetArgvs();
  const char *Argv0 = Argvs.size() > 0 ? Argvs[0].c_str() : "PluginsTests";
  void *Ptr = (void *)(intptr_t)anchor;
  std::string Path = sys::fs::getMainExecutable(Argv0, Ptr);
  llvm::SmallString<256> Buf{sys::path::parent_path(Path)};
  sys::path::append(Buf, (Name + LLVM_PLUGIN_EXT).c_str());
  return std::string(Buf.str());
}

TEST(PluginsTests, LoadPlugin) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  auto PluginPath = LibPath();
  ASSERT_NE("", PluginPath);

  Expected<PassPlugin> Plugin = PassPlugin::load(PluginPath);
  ASSERT_TRUE(!!Plugin) << "Plugin path: " << PluginPath;

  ASSERT_EQ(TEST_PLUGIN_NAME, Plugin->getPluginName());
  ASSERT_EQ(TEST_PLUGIN_VERSION, Plugin->getPluginVersion());

  LLVMContext C;
  PassBuilder PB(C);
  ModulePassManager PM;
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, "plugin-pass"), Failed());

  Plugin->registerPassBuilderCallbacks(PB);
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, "plugin-pass"), Succeeded());
}

// Test that llvmGetPassPluginInfo from DoublerPlugin is called twice with
// -fpass-plugin=DoublerPlugin -fpass-plugin=TestPlugin
// -fpass-plugin=DoublerPlugin.
TEST(PluginsTests, LoadMultiplePlugins) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  auto DoublerPluginPath = LibPath("DoublerPlugin");
  auto TestPluginPath = LibPath("TestPlugin");
  ASSERT_NE("", DoublerPluginPath);
  ASSERT_NE("", TestPluginPath);

  Expected<PassPlugin> DoublerPlugin1 = PassPlugin::load(DoublerPluginPath);
  ASSERT_TRUE(!!DoublerPlugin1)
      << "Plugin path: " << DoublerPlugin1->getFilename();

  Expected<PassPlugin> TestPlugin = PassPlugin::load(TestPluginPath);
  ASSERT_TRUE(!!TestPlugin) << "Plugin path: " << TestPlugin->getFilename();

  // If llvmGetPassPluginInfo is resolved as a weak symbol taking into account
  // all loaded symbols, the second call to PassPlugin::load will actually
  // return the llvmGetPassPluginInfo from the most recently loaded plugin, in
  // this case TestPlugin.
  Expected<PassPlugin> DoublerPlugin2 = PassPlugin::load(DoublerPluginPath);
  ASSERT_TRUE(!!DoublerPlugin2)
      << "Plugin path: " << DoublerPlugin2->getFilename();

  ASSERT_EQ("DoublerPlugin", DoublerPlugin1->getPluginName());
  ASSERT_EQ("2.2-unit", DoublerPlugin1->getPluginVersion());
  ASSERT_EQ(TEST_PLUGIN_NAME, TestPlugin->getPluginName());
  ASSERT_EQ(TEST_PLUGIN_VERSION, TestPlugin->getPluginVersion());
  // Check that the plugin name/version is set correctly when loaded a second
  // time
  ASSERT_EQ("DoublerPlugin", DoublerPlugin2->getPluginName());
  ASSERT_EQ("2.2-unit", DoublerPlugin2->getPluginVersion());

  LLVMContext C;
  PassBuilder PB(C);
  ModulePassManager PM;
  const char *PipelineText = "module(doubler-pass,plugin-pass,doubler-pass)";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Failed());
  TestPlugin->registerPassBuilderCallbacks(PB);
  DoublerPlugin1->registerPassBuilderCallbacks(PB);
  DoublerPlugin2->registerPassBuilderCallbacks(PB);
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded());

  SMDiagnostic Err;
  std::unique_ptr<Module> M =
      parseAssemblyString(R"IR(@doubleme = constant i32 7)IR", Err, C);

  // Check that the initial value is 7
  {
    auto *GV = M->getNamedValue("doubleme");
    auto *Init = cast<GlobalVariable>(GV)->getInitializer();
    auto *CI = cast<ConstantInt>(Init);
    ASSERT_EQ(CI->getSExtValue(), 7);
  }

  ModuleAnalysisManager MAM;
  // Register required pass instrumentation analysis.
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  PM.run(*M, MAM);

  // Check that the final value is 28 because DoublerPlugin::run was called
  // twice, indicating that the llvmGetPassPluginInfo and registerCallbacks
  // were correctly called.
  {
    // Check the value was doubled twice
    auto *GV = M->getNamedValue("doubleme");
    auto *Init = cast<GlobalVariable>(GV)->getInitializer();
    auto *CI = cast<ConstantInt>(Init);
    ASSERT_EQ(CI->getSExtValue(), 28);
  }
}

// Parses a module with the initializer 7, runs \p Pipeline on it and returns
// the new initializer.
static int64_t runOnSeven(const PassPlugin &Plugin, LLVMContext &C,
                          StringRef Pipeline) {
  PassBuilder PB(C);
  Plugin.registerPassBuilderCallbacks(PB);
  ModulePassManager PM;
  if (Error E = PB.parsePassPipeline(PM, Pipeline)) {
    consumeError(std::move(E));
    return -1;
  }
  SMDiagnostic Err;
  std::unique_ptr<Module> M =
      parseAssemblyString("@doubleme = constant i32 7", Err, C);
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  PM.run(*M, MAM);
  auto *GV = cast<GlobalVariable>(M->getNamedValue("doubleme"));
  return cast<ConstantInt>(GV->getInitializer())->getSExtValue();
}

// The options of a plugin live in the LLVMContext of a compilation and not in
// the plugin library. One loaded plugin acts as an independent instance in
// each context.
TEST(PluginsTests, ConfigurePerContext) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  Expected<PassPlugin> Plugin = PassPlugin::load(LibPath("DoublerPlugin"));
  ASSERT_THAT_EXPECTED(Plugin, Succeeded());

  LLVMContext C1, C2, Unconfigured;
  ASSERT_THAT_ERROR(passPluginArguments({Plugin->getInfo()},
                                        {"DoublerPlugin,-doubler-factor=3"},
                                        C1.getLibraryOptions()),
                    Succeeded());
  ASSERT_THAT_ERROR(
      passPluginArguments({Plugin->getInfo()},
                          {"DoublerPlugin,-doubler-factor=5",
                           "DoublerPlugin,-doubler-pass-name=quintupler"},
                          C2.getLibraryOptions()),
      Succeeded());

  // The option that the pass reads when it runs
  EXPECT_EQ(runOnSeven(*Plugin, C1, "doubler-pass"), 21);
  EXPECT_EQ(runOnSeven(*Plugin, C2, "quintupler"), 35);
  EXPECT_EQ(runOnSeven(*Plugin, Unconfigured, "doubler-pass"), 14);

  // The option that the plugin reads when it registers its callbacks
  EXPECT_EQ(runOnSeven(*Plugin, C1, "quintupler"), -1);
  EXPECT_EQ(runOnSeven(*Plugin, C2, "doubler-pass"), -1);

  // Hosts copy their library options into each context they compile in, e.g.
  // for the ThinLTO backends.
  LLVMContext Backend;
  Backend.getLibraryOptions().assign(C2.getLibraryOptions());
  EXPECT_EQ(runOnSeven(*Plugin, Backend, "quintupler"), 35);
}

// Within one context there is a single set of options per plugin. Loading the
// plugin twice doesn't give two configurations.
TEST(PluginsTests, OneConfigurationPerContext) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif

  Expected<PassPlugin> Plugin1 = PassPlugin::load(LibPath("DoublerPlugin"));
  ASSERT_THAT_EXPECTED(Plugin1, Succeeded());
  Expected<PassPlugin> Plugin2 = PassPlugin::load(LibPath("DoublerPlugin"));
  ASSERT_THAT_EXPECTED(Plugin2, Succeeded());

  // Arguments are addressed by plugin name, which is ambiguous here
  LLVMContext C;
  EXPECT_THAT_ERROR(
      passPluginArguments({Plugin1->getInfo(), Plugin2->getInfo()},
                          {"DoublerPlugin,-doubler-factor=3"},
                          C.getLibraryOptions()),
      Failed());

  // Configuring them one by one sets the same options, so the last one wins
  ASSERT_THAT_ERROR(passPluginArguments({Plugin1->getInfo()},
                                        {"DoublerPlugin,-doubler-factor=3"},
                                        C.getLibraryOptions()),
                    Succeeded());
  ASSERT_THAT_ERROR(passPluginArguments({Plugin2->getInfo()},
                                        {"DoublerPlugin,-doubler-factor=5"},
                                        C.getLibraryOptions()),
                    Succeeded());
  EXPECT_EQ(runOnSeven(*Plugin1, C, "doubler-pass"), 35);
  EXPECT_EQ(runOnSeven(*Plugin2, C, "doubler-pass"), 35);
}
