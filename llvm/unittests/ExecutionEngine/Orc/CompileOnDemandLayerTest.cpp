//===----- CompileOnDemandLayerTest.cpp - Unit tests for the COD layer ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "OrcTestCommon.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class DummyCallbackManager : public orc::JITCompileCallbackManager {
public:
  DummyCallbackManager() : JITCompileCallbackManager(0) {}

public:
  Error grow() override { llvm_unreachable("not implemented"); }
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

  JITSymbol findStub(StringRef Name, bool ExportedStubsOnly) override {
    llvm_unreachable("Not implemented");
  }

  JITSymbol findPointer(StringRef Name) override {
    llvm_unreachable("Not implemented");
  }

  Error updatePointer(StringRef Name, JITTargetAddress NewAddr) override {
    llvm_unreachable("Not implemented");
  }
};

TEST(CompileOnDemandLayerTest, FindSymbol) {
  MockBaseLayer<int, std::shared_ptr<Module>> TestBaseLayer;
  TestBaseLayer.findSymbolImpl =
    [](const std::string &Name, bool) {
      if (Name == "foo")
        return JITSymbol(1, JITSymbolFlags::Exported);
      return JITSymbol(nullptr);
    };

  DummyCallbackManager CallbackMgr;

  llvm::orc::CompileOnDemandLayer<decltype(TestBaseLayer)> COD(
      TestBaseLayer, [](Function &F) { return std::set<Function *>{&F}; },
      CallbackMgr, [] { return llvm::make_unique<DummyStubsManager>(); }, true);

  auto Sym = COD.findSymbol("foo", true);

  EXPECT_TRUE(!!Sym) << "CompileOnDemand::findSymbol should call findSymbol in "
                        "the base layer.";
}
}
