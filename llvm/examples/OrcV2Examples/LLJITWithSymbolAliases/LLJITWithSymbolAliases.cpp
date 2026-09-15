//===-- LLJITWithSymbolAliases.cpp - Symbol aliases with LLJIT ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This example demonstrates how to use the symbolAliases utility to define
// alternate names for symbols already present in a JITDylib. We define two
// aliases:
//
//   - "aliased_foo" as an alias for "foo", a function defined in a JIT'd IR
//     module.
//   - "aliased_bar" as an alias for "bar", a precompiled function added to
//     the JITDylib via absoluteSymbols.
//
// We then look up both aliases and call them to confirm that they resolve to
// the original definitions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "../ExampleModules.h"

using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

// IR module containing the simplest possible function: foo returns 42.
const llvm::StringRef FooMod =
    R"(
  define i32 @foo() {
  entry:
    ret i32 42
  }
)";

// Precompiled function that we will expose to the JIT via absoluteSymbols.
static int bar() { return 7; }

int main(int argc, char *argv[]) {
  // Initialize LLVM.
  InitLLVM X(argc, argv);

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  cl::ParseCommandLineOptions(argc, argv, "LLJITWithSymbolAliases");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  // Create an LLJIT instance and add the IR module containing 'foo'.
  auto J = ExitOnErr(LLJITBuilder().create());
  ExitOnErr(J->addIRModule(ExitOnErr(parseExampleModule(FooMod, "foo-mod"))));

  // Add the precompiled 'bar' function as an absolute symbol.
  auto &JD = J->getMainJITDylib();
  ExitOnErr(JD.define(absoluteSymbols(
      {{J->mangleAndIntern("bar"),
        {ExecutorAddr::fromPtr(&bar),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));

  // Define aliases: 'aliased_foo' -> 'foo' and 'aliased_bar' -> 'bar'.
  ExitOnErr(JD.define(symbolAliases(
      {{J->mangleAndIntern("aliased_foo"),
        {J->mangleAndIntern("foo"),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}},
       {J->mangleAndIntern("aliased_bar"),
        {J->mangleAndIntern("bar"),
         JITSymbolFlags::Exported | JITSymbolFlags::Callable}}})));

  // Look up the aliases and call them.
  auto AliasedFoo = ExitOnErr(J->lookup("aliased_foo")).toPtr<int()>();
  auto AliasedBar = ExitOnErr(J->lookup("aliased_bar")).toPtr<int()>();

  outs() << "aliased_foo() = " << AliasedFoo() << "\n"
         << "aliased_bar() = " << AliasedBar() << "\n";

  return 0;
}
