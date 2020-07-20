//===----- CompileOnDemandLayerTest.cpp - Unit tests for the COD layer ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "OrcTestCommon.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class DummyTrampolinePool : public orc::TrampolinePool {
protected:
  Error grow() override { llvm_unreachable("Unimplemented"); }
};

class DummyCallbackManager : public JITCompileCallbackManager {
public:
  DummyCallbackManager(ExecutionSession &ES)
      : JITCompileCallbackManager(std::make_unique<DummyTrampolinePool>(), ES,
                                  0) {}
};

class DummyStubsManager : public orc::IndirectStubsManager {
public:
  Error createStub(StringRef StubName, JITTargetAddress InitAddr,
                   JITSymbolFlags Flags) override {
    llvm_unreachable("Not implemented");
  }

  Error createStubs(const StubInitsMap &StubInits) override {
    llvm_unreachable("Not implemented");
  }

  JITEvaluatedSymbol findStub(StringRef Name, bool ExportedStubsOnly) override {
    llvm_unreachable("Not implemented");
  }

  JITEvaluatedSymbol findPointer(StringRef Name) override {
    llvm_unreachable("Not implemented");
  }

  Error updatePointer(StringRef Name, JITTargetAddress NewAddr) override {
    llvm_unreachable("Not implemented");
  }
};

TEST(LegacyCompileOnDemandLayerTest, FindSymbol) {
  MockBaseLayer<int, std::shared_ptr<Module>> TestBaseLayer;
  TestBaseLayer.findSymbolImpl =
    [](const std::string &Name, bool) {
      if (Name == "foo")
        return JITSymbol(1, JITSymbolFlags::Exported);
      return JITSymbol(nullptr);
    };


  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  DummyCallbackManager CallbackMgr(ES);

  auto GetResolver =
      [](orc::VModuleKey) -> std::shared_ptr<llvm::orc::SymbolResolver> {
    llvm_unreachable("Should never be called");
  };

  auto SetResolver = [](orc::VModuleKey, std::shared_ptr<orc::SymbolResolver>) {
    llvm_unreachable("Should never be called");
  };

  llvm::orc::LegacyCompileOnDemandLayer<decltype(TestBaseLayer)> COD(
      AcknowledgeORCv1Deprecation, ES, TestBaseLayer, GetResolver, SetResolver,
      [](Function &F) { return std::set<Function *>{&F}; }, CallbackMgr,
      [] { return std::make_unique<DummyStubsManager>(); }, true);

  auto Sym = COD.findSymbol("foo", true);

  EXPECT_TRUE(!!Sym) << "CompileOnDemand::findSymbol should call findSymbol in "
                        "the base layer.";
}
}
