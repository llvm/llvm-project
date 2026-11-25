//===-- ModulesTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "Handler/RequestHandler.h"
#include "Protocol/ProtocolRequests.h"
#include "TestBase.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/Host/HostInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <limits>

using namespace llvm;
using namespace lldb;
using namespace lldb_dap;
using namespace lldb_dap_tests;
using namespace lldb_dap::protocol;

class ModulesRequestHandlerTest : public DAPTestBase {
protected:
  void SetUp() override {
    DAPTestBase::SetUp();
    CreateDebugger();
    dap->debugger.SetScriptLanguage(eScriptLanguageNone);
    SBError error = SBDebugger::SetInternalVariable(
        "symbols.enable-external-lookup", "false",
        dap->debugger.GetInstanceName());
    ASSERT_TRUE(error.Success()) << error.GetCString();

    std::string program =
        lldb_private::HostInfo::GetProgramFileSpec().GetPath();

    // Use the linux binary as an extra module.
    Expected<lldb_private::TestFile> linux_yaml =
        lldb_private::TestFile::fromYamlFile(k_linux_binary);
    ASSERT_THAT_EXPECTED(linux_yaml, Succeeded());
    Expected<sys::fs::TempFile> linux_file = linux_yaml->writeToTemporaryFile();
    ASSERT_THAT_EXPECTED(linux_file, Succeeded());
    binary = std::move(*linux_file);

    dap->target = dap->debugger.CreateTarget(
        program.c_str(), /*target_triple=*/nullptr, /*platform_name=*/nullptr,
        /*add_dependent_modules=*/false, error);
    ASSERT_TRUE(dap->target) << error.GetCString();
    ASSERT_TRUE(error.Success()) << error.GetCString();

    SBModuleSpec spec;
    spec.SetTriple("x86_64-unknown-linux-gnu");
    spec.SetFileSpec(SBFileSpec(binary->TmpName.c_str(), /*resolve=*/true));
    ASSERT_TRUE(dap->target.AddModule(spec));
    ASSERT_EQ(dap->target.GetNumModules(), 2u);
  }
};

TEST_F(ModulesRequestHandlerTest, ModuleCountPastTheEnd) {
  ModulesRequestHandler handler(*dap);
  Expected<ModulesResponseBody> all_modules = handler.Run(ModulesArguments{});
  ASSERT_THAT_EXPECTED(all_modules, Succeeded());
  const uint32_t total_modules = all_modules->totalModules;
  ASSERT_GE(total_modules, 2u);

  // The modules after the first one, with an exact count.
  ModulesArguments rest_args;
  rest_args.startModule = 1;
  rest_args.moduleCount = total_modules - 1;
  Expected<ModulesResponseBody> rest = handler.Run(rest_args);
  ASSERT_THAT_EXPECTED(rest, Succeeded());

  // startModule + moduleCount does not fit in a uint32_t.
  ModulesArguments page_args;
  page_args.startModule = 1;
  page_args.moduleCount = std::numeric_limits<uint32_t>::max();

  Expected<ModulesResponseBody> page = handler.Run(page_args);
  ASSERT_THAT_EXPECTED(page, Succeeded());
  EXPECT_EQ(page->totalModules, total_modules);
  ASSERT_EQ(page->modules.size(), rest->modules.size());

  for (const auto &[page_module, rest_module] :
       llvm::zip(page->modules, rest->modules)) {

    EXPECT_EQ(page_module.id, rest_module.id);
  }
}
