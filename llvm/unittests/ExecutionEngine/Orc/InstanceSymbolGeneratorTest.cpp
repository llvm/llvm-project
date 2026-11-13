//===- InstanceSymbolGeneratorTest.cpp - Test instance-symbol generation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that ExecutionSession makes the executor's instance symbols available
// via the bootstrap JITDylib when the executor provides the ORC runtime's
// instance-symbol lookup function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/Mangler.h"
#include "llvm/ExecutionEngine/Orc/Shared/SPSCI/SessionSPSCI.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RemoteSymbolLookupSet.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <atomic>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

int Foo = 0;
int Bar = 0;
std::atomic<size_t> LookupCalls = 0;

using LookupSPSSig = rt::sps_ci::SessionLookupInstanceSymbols::SPSSig;

// Stand-in for the runtime's instance-symbol lookup, with instance symbols
// orc_rt_test_Foo, orc_rt_test_Bar, and orc_rt_test_Null (at a null address).
// Missing symbols are reported as the runtime reports them: a null address if
// weakly referenced, otherwise no value.
CWrapperFunctionBuffer lookupInstanceSymbols(const char *ArgData,
                                             size_t ArgSize) {
  ++LookupCalls;
  return WrapperFunction<LookupSPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<RemoteSymbolLookupSetElement> Symbols) {
               RemoteSymbolLookupResult Result;
               for (auto &Sym : Symbols) {
                 if (Sym.Name == "orc_rt_test_Foo")
                   Result.push_back(ExecutorAddr::fromPtr(&Foo));
                 else if (Sym.Name == "orc_rt_test_Bar")
                   Result.push_back(ExecutorAddr::fromPtr(&Bar));
                 else if (Sym.Name == "orc_rt_test_Null")
                   Result.push_back(ExecutorAddr());
                 else if (Sym.Required)
                   Result.push_back(std::nullopt);
                 else
                   Result.push_back(ExecutorAddr());
               }
               return Result;
             })
      .release();
}

// A broken lookup that returns no results, whatever it's asked for.
CWrapperFunctionBuffer lookupInstanceSymbolsMalformed(const char *ArgData,
                                                      size_t ArgSize) {
  return WrapperFunction<LookupSPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<RemoteSymbolLookupSetElement> Symbols) {
               return RemoteSymbolLookupResult();
             })
      .release();
}

// A SelfExecutorProcessControl whose bootstrap symbols include the given
// instance-symbol lookup function, if any.
class InstanceSymbolsEPC : public SelfExecutorProcessControl {
public:
  InstanceSymbolsEPC(ExecutorAddr LookupFn)
      : SelfExecutorProcessControl(std::make_shared<SymbolStringPool>(),
                                   std::make_unique<InPlaceTaskDispatcher>(),
                                   Triple(sys::getProcessTriple()),
                                   sys::Process::getPageSizeEstimate()) {
    if (LookupFn)
      BootstrapSymbols
          [Mangler(getTargetTriple())
               .mangledCopy(rt::sps_ci::SessionLookupInstanceSymbols::Name)] =
              LookupFn;
  }
};

class InstanceSymbolGeneratorTest : public testing::Test {
protected:
  void SetUp() override { LookupCalls = 0; }

  void TearDown() override {
    if (ES)
      cantFail(ES->endSession());
  }

  void createSession(ExecutorAddr LookupFn) {
    ES = std::make_unique<ExecutionSession>(
        std::make_unique<InstanceSymbolsEPC>(LookupFn));
  }

  Expected<ExecutorSymbolDef> lookup(StringRef Name) {
    return ES->lookup({&ES->getBootstrapJITDylib()}, ES->intern(Name));
  }

  std::unique_ptr<ExecutionSession> ES;
};

TEST_F(InstanceSymbolGeneratorTest, LookupInstanceSymbol) {
  createSession(ExecutorAddr::fromPtr(lookupInstanceSymbols));

  auto Sym = lookup("orc_rt_test_Foo");
  ASSERT_THAT_EXPECTED(Sym, Succeeded());
  EXPECT_EQ(Sym->getAddress(), ExecutorAddr::fromPtr(&Foo));

  // The first lookup defines the symbol in the bootstrap JITDylib, so a second
  // lookup shouldn't call the executor again.
  auto Sym2 = lookup("orc_rt_test_Foo");
  ASSERT_THAT_EXPECTED(Sym2, Succeeded());
  EXPECT_EQ(Sym2->getAddress(), ExecutorAddr::fromPtr(&Foo));
  EXPECT_EQ(LookupCalls, 1U);
}

TEST_F(InstanceSymbolGeneratorTest, LookupNullInstanceSymbol) {
  // An instance symbol at a null address is still an instance symbol, and
  // should be distinguished from a missing one.
  createSession(ExecutorAddr::fromPtr(lookupInstanceSymbols));

  auto Sym = lookup("orc_rt_test_Null");
  ASSERT_THAT_EXPECTED(Sym, Succeeded());
  EXPECT_EQ(Sym->getAddress(), ExecutorAddr());
}

TEST_F(InstanceSymbolGeneratorTest, LookupMissingSymbol) {
  createSession(ExecutorAddr::fromPtr(lookupInstanceSymbols));
  EXPECT_THAT_EXPECTED(lookup("orc_rt_test_Missing"),
                       Failed<SymbolsNotFound>());
}

TEST_F(InstanceSymbolGeneratorTest, LookupMixedSet) {
  createSession(ExecutorAddr::fromPtr(lookupInstanceSymbols));

  SymbolLookupSet LS;
  LS.add(ES->intern("orc_rt_test_Foo"));
  LS.add(ES->intern("orc_rt_test_Bar"));
  LS.add(ES->intern("orc_rt_test_Missing"),
         SymbolLookupFlags::WeaklyReferencedSymbol);

  auto Syms = ES->lookup(makeJITDylibSearchOrder(&ES->getBootstrapJITDylib()),
                         std::move(LS));
  ASSERT_THAT_EXPECTED(Syms, Succeeded());
  EXPECT_EQ(Syms->size(), 2U);
  EXPECT_EQ((*Syms)[ES->intern("orc_rt_test_Foo")].getAddress(),
            ExecutorAddr::fromPtr(&Foo));
  EXPECT_EQ((*Syms)[ES->intern("orc_rt_test_Bar")].getAddress(),
            ExecutorAddr::fromPtr(&Bar));
  EXPECT_FALSE(Syms->count(ES->intern("orc_rt_test_Missing")));
}

TEST_F(InstanceSymbolGeneratorTest, NoGeneratorWithoutLookupFunction) {
  createSession(ExecutorAddr());
  EXPECT_THAT_EXPECTED(lookup("orc_rt_test_Foo"), Failed<SymbolsNotFound>());
  EXPECT_EQ(LookupCalls, 0U);
}

TEST_F(InstanceSymbolGeneratorTest, MalformedLookupResult) {
  createSession(ExecutorAddr::fromPtr(lookupInstanceSymbolsMalformed));
  EXPECT_THAT_EXPECTED(
      lookup("orc_rt_test_Foo"),
      FailedWithMessage("Instance symbol lookup returned 0 results for 1 "
                        "symbols"));
}

} // namespace
