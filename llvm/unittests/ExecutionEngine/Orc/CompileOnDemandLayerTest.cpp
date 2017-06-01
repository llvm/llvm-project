//===----- CompileOnDemandLayerTest.cpp - Unit tests for the COD layer ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class DummyCallbackManager : public orc::JITCompileCallbackManager {
public:
  DummyCallbackManager() : JITCompileCallbackManager(0) {}

public:
  void grow() override { llvm_unreachable("not implemented"); }
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
  auto MockBaseLayer = createMockBaseLayer<int>(
      DoNothingAndReturn<int>(0), DoNothingAndReturn<void>(),
      [](const std::string &Name, bool) {
        if (Name == "foo")
          return JITSymbol(1, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      },
      DoNothingAndReturn<JITSymbol>(nullptr));

  typedef decltype(MockBaseLayer) MockBaseLayerT;
  DummyCallbackManager CallbackMgr;

  llvm::orc::CompileOnDemandLayer<MockBaseLayerT> COD(
      MockBaseLayer, [](Function &F) { return std::set<Function *>{&F}; },
      CallbackMgr, [] { return llvm::make_unique<DummyStubsManager>(); }, true);

  auto Sym = COD.findSymbol("foo", true);

  EXPECT_TRUE(!!Sym) << "CompileOnDemand::findSymbol should call findSymbol in "
                        "the base layer.";
}
}
