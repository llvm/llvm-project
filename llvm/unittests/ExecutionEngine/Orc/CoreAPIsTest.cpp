//===----------- CoreAPIsTest.cpp - Unit tests for Core ORC APIs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "OrcTestCommon.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(CoreAPIsTest, AsynchronousSymbolQuerySuccessfulResolutionOnly) {
  SymbolStringPool SP;
  auto Foo = SP.intern("foo");
  constexpr JITTargetAddress FakeAddr = 0xdeadbeef;
  SymbolNameSet Names({Foo});

  bool OnResolutionRun = false;
  bool OnReadyRun = false;
  auto OnResolution =
    [&](Expected<SymbolMap> Result) {
      EXPECT_TRUE(!!Result) << "Resolution unexpectedly returned error";
      auto I = Result->find(Foo);
      EXPECT_NE(I, Result->end()) << "Could not find symbol definition";
      EXPECT_EQ(cantFail(I->second.getAddress()), FakeAddr)
        << "Resolution returned incorrect result";
      OnResolutionRun = true;
    };
  auto OnReady = 
    [&](Error Err) {
      cantFail(std::move(Err));
      OnReadyRun = true;
    };

  AsynchronousSymbolQuery Q(Names, OnResolution, OnReady);

  Q.setDefinition(Foo, JITSymbol(FakeAddr, JITSymbolFlags::Exported));

  EXPECT_TRUE(OnResolutionRun) << "OnResolutionCallback was not run";
  EXPECT_FALSE(OnReadyRun) << "OnReady unexpectedly run";
  errs() << "Exiting test\n";
}

}
