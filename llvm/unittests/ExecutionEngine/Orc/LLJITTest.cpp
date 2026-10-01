//===--------------- LLJITTest.cpp - Unit tests for LLJIT -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "OrcTestCommon.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <vector>

using namespace llvm;
using namespace llvm::orc;

namespace {

std::vector<int> InitOrder;
void recordInit(int I) { InitOrder.push_back(I); }
void fromGenerator() {}

ThreadSafeModule parseModule(StringRef Source, StringRef Name) {
  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Err;
  auto M = parseAssemblyString(Source, Err, *Ctx);
  if (!M) {
    Err.print("LLJITTest", errs());
    return ThreadSafeModule();
  }
  M->setModuleIdentifier(Name);
  return ThreadSafeModule(std::move(M), std::move(Ctx));
}

// A module whose initializer calls fromGenerator(), then records 1.
const char *OuterModule = R"(
  declare void @recordInit(i32)
  declare void @fromGenerator()
  define internal void @init() {
    call void @fromGenerator()
    call void @recordInit(i32 1)
    ret void
  }
  @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]
)";

// A module whose initializer records 2.
const char *NestedModule = R"(
  declare void @recordInit(i32)
  define internal void @init() {
    call void @recordInit(i32 2)
    ret void
  }
  @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
    [{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]
)";

// Defines fromGenerator() like a generator that loads a library whose static
// initializers add and initialize more JIT'd code.
class InitializingGenerator : public DefinitionGenerator {
public:
  InitializingGenerator(LLJIT &J) : J(J) {}

  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override {
    SymbolStringPtr Name = J.mangleAndIntern("fromGenerator");
    if (llvm::none_of(LookupSet,
                      [&](const auto &KV) { return KV.first == Name; }))
      return Error::success();
    if (Error Err = J.addIRModule(JD, parseModule(NestedModule, "nested")))
      return Err;
    if (Error Err = J.initialize(JD))
      return Err;
    return JD.define(absoluteSymbols(
        {{Name,
          {ExecutorAddr::fromPtr(&fromGenerator), JITSymbolFlags::Exported}}}));
  }

private:
  LLJIT &J;
};

TEST(LLJITTest, InitializeWhileLinkingInitializers) {
  // Linking the outer module runs the generator, which initializes the
  // nested module. That must run only the nested module's initializer: the
  // outer one is not ready until the generator returns.
  OrcNativeTarget::initialize();
  auto J = LLJITBuilder().create();
  if (!J) {
    consumeError(J.takeError());
    GTEST_SKIP();
  }
  JITDylib &JD = (*J)->getMainJITDylib();
  cantFail(JD.define(absoluteSymbols(
      {{(*J)->mangleAndIntern("recordInit"),
        {ExecutorAddr::fromPtr(&recordInit), JITSymbolFlags::Exported}}})));
  JD.addGenerator(std::make_unique<InitializingGenerator>(**J));

  ASSERT_THAT_ERROR((*J)->addIRModule(parseModule(OuterModule, "outer")),
                    Succeeded());
  InitOrder.clear();
  ASSERT_THAT_ERROR((*J)->initialize(JD), Succeeded());
  EXPECT_EQ(InitOrder, (std::vector<int>{2, 1}));
}

} // end anonymous namespace
