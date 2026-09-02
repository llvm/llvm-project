//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/WebAssembly/PlatformWasm.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/Environment.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using ::testing::ElementsAre;

static std::vector<std::string> GetArgStrings(const Args &args) {
  std::vector<std::string> result;
  for (const Args::ArgEntry &entry : args)
    result.push_back(entry.c_str());
  return result;
}

TEST(PlatformWasmTest, MakeRuntimeCommand) {
  Args args = PlatformWasm::MakeRuntimeCommand(
      "/bin/runtime", Args(), "-g=127.0.0.1:", 1234, /*env_arg=*/"",
      Environment(), "/tmp/module.wasm", Args());

  EXPECT_THAT(
      GetArgStrings(args),
      ElementsAre("/bin/runtime", "-g=127.0.0.1:1234", "/tmp/module.wasm"));
}

TEST(PlatformWasmTest, MakeRuntimeCommandRuntimeArgsPrecedePort) {
  // A runtime dispatching on a leading subcommand names it through
  // runtime-args, which is only usable if those come before the port.
  Args runtime_args;
  runtime_args.AppendArgument("run");

  Args args = PlatformWasm::MakeRuntimeCommand(
      "/bin/runtime", runtime_args, "--debugger-port=", 1234, /*env_arg=*/"",
      Environment(), "/tmp/module.wasm", Args());

  EXPECT_THAT(GetArgStrings(args),
              ElementsAre("/bin/runtime", "run", "--debugger-port=1234",
                          "/tmp/module.wasm"));
}

TEST(PlatformWasmTest, MakeRuntimeCommandForwardsEnvironment) {
  Environment env;
  env["KEY"] = "value";

  Args args = PlatformWasm::MakeRuntimeCommand("/bin/runtime", Args(),
                                               "-g=", 1234, "--env=", env,
                                               "/tmp/module.wasm", Args());

  EXPECT_THAT(GetArgStrings(args),
              ElementsAre("/bin/runtime", "-g=1234", "--env=KEY=value",
                          "/tmp/module.wasm"));
}

TEST(PlatformWasmTest, MakeRuntimeCommandWithoutEnvArgDropsEnvironment) {
  Environment env;
  env["KEY"] = "value";

  Args args = PlatformWasm::MakeRuntimeCommand(
      "/bin/runtime", Args(), "-g=", 1234,
      /*env_arg=*/"", env, "/tmp/module.wasm", Args());

  EXPECT_THAT(GetArgStrings(args),
              ElementsAre("/bin/runtime", "-g=1234", "/tmp/module.wasm"));
}

TEST(PlatformWasmTest, MakeRuntimeCommandModulePathReplacesArgZero) {
  Args inferior_args;
  inferior_args.AppendArgument("module.wasm");
  inferior_args.AppendArgument("--flag");

  Args args = PlatformWasm::MakeRuntimeCommand(
      "/bin/runtime", Args(), "-g=", 1234, /*env_arg=*/"", Environment(),
      "/tmp/module.wasm", inferior_args);

  EXPECT_THAT(GetArgStrings(args), ElementsAre("/bin/runtime", "-g=1234",
                                               "/tmp/module.wasm", "--flag"));
}

TEST(PlatformWasmTest, MakeRuntimeCommandWithoutModulePath) {
  Args inferior_args;
  inferior_args.AppendArgument("module.wasm");

  Args args = PlatformWasm::MakeRuntimeCommand(
      "/bin/runtime", Args(), "-g=", 1234, /*env_arg=*/"", Environment(),
      /*module_path=*/"", inferior_args);

  EXPECT_THAT(GetArgStrings(args),
              ElementsAre("/bin/runtime", "-g=1234", "module.wasm"));
}
