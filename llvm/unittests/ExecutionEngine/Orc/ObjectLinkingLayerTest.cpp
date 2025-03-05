//===-------- ObjectLinkingLayerTest.cpp - ObjectLinkingLayer tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

namespace {

const char BlockContentBytes[] = {0x01, 0x02, 0x03, 0x04,
                                  0x05, 0x06, 0x07, 0x08};

ArrayRef<char> BlockContent(BlockContentBytes);

class ObjectLinkingLayerTest : public testing::Test {
public:
  ~ObjectLinkingLayerTest() {
    if (auto Err = ES.endSession())
      ES.reportError(std::move(Err));
  }

protected:
  ExecutionSession ES{std::make_unique<UnsupportedExecutorProcessControl>()};
  JITDylib &JD = ES.createBareJITDylib("main");
  ObjectLinkingLayer ObjLinkingLayer{
      ES, std::make_unique<InProcessMemoryManager>(4096)};
};

TEST_F(ObjectLinkingLayerTest, AddLinkGraph) {
  auto G = std::make_unique<LinkGraph>(
      "foo", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
      SubtargetFeatures(), x86_64::getEdgeKindName);

  auto &Sec1 = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &B1 = G->createContentBlock(Sec1, BlockContent,
                                   orc::ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B1, 4, "_X", 4, Linkage::Strong, Scope::Default, false,
                      false);
  G->addDefinedSymbol(B1, 4, "_Y", 4, Linkage::Weak, Scope::Default, false,
                      false);
  G->addDefinedSymbol(B1, 4, "_Z", 4, Linkage::Strong, Scope::Hidden, false,
                      false);
  G->addDefinedSymbol(B1, 4, "_W", 4, Linkage::Strong, Scope::Default, true,
                      false);

  EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G)), Succeeded());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_X"), Succeeded());
}

TEST_F(ObjectLinkingLayerTest, ResourceTracker) {
  // This test transfers allocations to previously unknown ResourceTrackers,
  // while increasing the number of trackers in the ObjectLinkingLayer, which
  // may invalidate some iterators internally.
  std::vector<ResourceTrackerSP> Trackers;
  for (unsigned I = 0; I < 64; I++) {
    auto G = std::make_unique<LinkGraph>(
        "foo", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
        SubtargetFeatures(), x86_64::getEdgeKindName);

    auto &Sec1 = G->createSection("__data", MemProt::Read | MemProt::Write);
    auto &B1 = G->createContentBlock(Sec1, BlockContent,
                                     orc::ExecutorAddr(0x1000), 8, 0);
    llvm::SmallString<0> SymbolName;
    SymbolName += "_X";
    SymbolName += std::to_string(I);
    G->addDefinedSymbol(B1, 4, SymbolName, 4, Linkage::Strong, Scope::Default,
                        false, false);

    auto RT1 = JD.createResourceTracker();
    EXPECT_THAT_ERROR(ObjLinkingLayer.add(RT1, std::move(G)), Succeeded());
    EXPECT_THAT_EXPECTED(ES.lookup(&JD, SymbolName), Succeeded());

    auto RT2 = JD.createResourceTracker();
    RT1->transferTo(*RT2);

    Trackers.push_back(RT2);
  }
}

TEST_F(ObjectLinkingLayerTest, ClaimLateDefinedWeakSymbols) {
  // Check that claiming weak symbols works as expected.
  //
  // To do this we'll need a custom plugin to inject some new symbols during
  // the link.
  class TestPlugin : public ObjectLinkingLayer::Plugin {
  public:
    void modifyPassConfig(MaterializationResponsibility &MR,
                          jitlink::LinkGraph &G,
                          jitlink::PassConfiguration &Config) override {
      Config.PrePrunePasses.insert(
          Config.PrePrunePasses.begin(), [](LinkGraph &G) {
            auto *DataSec = G.findSectionByName("__data");
            auto &DataBlock = G.createContentBlock(
                *DataSec, BlockContent, orc::ExecutorAddr(0x2000), 8, 0);
            G.addDefinedSymbol(DataBlock, 4, "_x", 4, Linkage::Weak,
                               Scope::Default, false, false);

            auto &TextSec =
                G.createSection("__text", MemProt::Read | MemProt::Write);
            auto &FuncBlock = G.createContentBlock(
                TextSec, BlockContent, orc::ExecutorAddr(0x3000), 8, 0);
            G.addDefinedSymbol(FuncBlock, 4, "_f", 4, Linkage::Weak,
                               Scope::Default, true, false);

            return Error::success();
          });
    }

    Error notifyFailed(MaterializationResponsibility &MR) override {
      llvm_unreachable("unexpected error");
    }

    Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
      return Error::success();
    }
    void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                     ResourceKey SrcKey) override {
      llvm_unreachable("unexpected resource transfer");
    }
  };

  ObjLinkingLayer.addPlugin(std::make_unique<TestPlugin>());
  auto G = std::make_unique<LinkGraph>(
      "foo", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
      SubtargetFeatures(), getGenericEdgeKindName);

  auto &DataSec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &DataBlock = G->createContentBlock(DataSec, BlockContent,
                                          orc::ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(DataBlock, 4, "_anchor", 4, Linkage::Weak, Scope::Default,
                      false, true);

  EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G)), Succeeded());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_anchor"), Succeeded());
}

TEST_F(ObjectLinkingLayerTest, HandleErrorDuringPostAllocationPass) {
  // We want to confirm that Errors in post allocation passes correctly
  // abandon the in-flight allocation and report an error.
  class TestPlugin : public ObjectLinkingLayer::Plugin {
  public:
    ~TestPlugin() { EXPECT_TRUE(ErrorReported); }

    void modifyPassConfig(MaterializationResponsibility &MR,
                          jitlink::LinkGraph &G,
                          jitlink::PassConfiguration &Config) override {
      Config.PostAllocationPasses.insert(
          Config.PostAllocationPasses.begin(), [](LinkGraph &G) {
            return make_error<StringError>("Kaboom", inconvertibleErrorCode());
          });
    }

    Error notifyFailed(MaterializationResponsibility &MR) override {
      ErrorReported = true;
      return Error::success();
    }

    Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
      return Error::success();
    }
    void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                     ResourceKey SrcKey) override {
      llvm_unreachable("unexpected resource transfer");
    }

  private:
    bool ErrorReported = false;
  };

  // We expect this test to generate errors. Consume them so that we don't
  // add noise to the test logs.
  ES.setErrorReporter(consumeError);

  ObjLinkingLayer.addPlugin(std::make_unique<TestPlugin>());
  auto G = std::make_unique<LinkGraph>(
      "foo", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
      SubtargetFeatures(), getGenericEdgeKindName);

  auto &DataSec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &DataBlock = G->createContentBlock(DataSec, BlockContent,
                                          orc::ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(DataBlock, 4, "_anchor", 4, Linkage::Weak, Scope::Default,
                      false, true);

  EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G)), Succeeded());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_anchor"), Failed());
}

TEST_F(ObjectLinkingLayerTest, AddAndRemovePlugins) {
  class TestPlugin : public ObjectLinkingLayer::Plugin {
  public:
    TestPlugin(size_t &ActivationCount, bool &PluginDestroyed)
        : ActivationCount(ActivationCount), PluginDestroyed(PluginDestroyed) {}

    ~TestPlugin() { PluginDestroyed = true; }

    void modifyPassConfig(MaterializationResponsibility &MR,
                          jitlink::LinkGraph &G,
                          jitlink::PassConfiguration &Config) override {
      ++ActivationCount;
    }

    Error notifyFailed(MaterializationResponsibility &MR) override {
      ADD_FAILURE() << "TestPlugin::notifyFailed called unexpectedly";
      return Error::success();
    }

    Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
      return Error::success();
    }

    void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                     ResourceKey SrcKey) override {}

  private:
    size_t &ActivationCount;
    bool &PluginDestroyed;
  };

  size_t ActivationCount = 0;
  bool PluginDestroyed = false;

  auto P = std::make_shared<TestPlugin>(ActivationCount, PluginDestroyed);

  ObjLinkingLayer.addPlugin(P);

  {
    auto G1 = std::make_unique<LinkGraph>(
        "G1", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
        SubtargetFeatures(), x86_64::getEdgeKindName);

    auto &DataSec = G1->createSection("__data", MemProt::Read | MemProt::Write);
    auto &DataBlock = G1->createContentBlock(DataSec, BlockContent,
                                             orc::ExecutorAddr(0x1000), 8, 0);
    G1->addDefinedSymbol(DataBlock, 4, "_anchor1", 4, Linkage::Weak,
                         Scope::Default, false, true);

    EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G1)), Succeeded());
    EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_anchor1"), Succeeded());
    EXPECT_EQ(ActivationCount, 1U);
  }

  ObjLinkingLayer.removePlugin(*P);

  {
    auto G2 = std::make_unique<LinkGraph>(
        "G2", ES.getSymbolStringPool(), Triple("x86_64-apple-darwin"),
        SubtargetFeatures(), x86_64::getEdgeKindName);

    auto &DataSec = G2->createSection("__data", MemProt::Read | MemProt::Write);
    auto &DataBlock = G2->createContentBlock(DataSec, BlockContent,
                                             orc::ExecutorAddr(0x1000), 8, 0);
    G2->addDefinedSymbol(DataBlock, 4, "_anchor2", 4, Linkage::Weak,
                         Scope::Default, false, true);

    EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G2)), Succeeded());
    EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_anchor2"), Succeeded());
    EXPECT_EQ(ActivationCount, 1U);
  }

  P.reset();
  EXPECT_TRUE(PluginDestroyed);
}

TEST(ObjectLinkingLayerSearchGeneratorTest, AbsoluteSymbolsObjectLayer) {
  class TestEPC : public UnsupportedExecutorProcessControl,
                  public DylibManager {
  public:
    TestEPC()
        : UnsupportedExecutorProcessControl(nullptr, nullptr,
                                            "x86_64-apple-darwin") {
      this->DylibMgr = this;
    }

    Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
      return ExecutorAddr::fromPtr((void *)nullptr);
    }

    void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                            SymbolLookupCompleteFn Complete) override {
      std::vector<ExecutorSymbolDef> Result;
      EXPECT_EQ(Request.size(), 1u);
      for (auto &LR : Request) {
        EXPECT_EQ(LR.Symbols.size(), 1u);
        for (auto &Sym : LR.Symbols) {
          if (*Sym.first == "_testFunc") {
            ExecutorSymbolDef Def{ExecutorAddr::fromPtr((void *)0x1000),
                                  JITSymbolFlags::Exported};
            Result.push_back(Def);
          } else {
            ADD_FAILURE() << "unexpected symbol request " << *Sym.first;
          }
        }
      }
      Complete(std::vector<tpctypes::LookupResult>{1, Result});
    }
  };

  ExecutionSession ES{std::make_unique<TestEPC>()};
  JITDylib &JD = ES.createBareJITDylib("main");
  ObjectLinkingLayer ObjLinkingLayer{
      ES, std::make_unique<InProcessMemoryManager>(4096)};

  auto G = EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
      ES, {}, [&](JITDylib &JD, SymbolMap Syms) {
        auto G =
            absoluteSymbolsLinkGraph(Triple("x86_64-apple-darwin"),
                                     ES.getSymbolStringPool(), std::move(Syms));
        return ObjLinkingLayer.add(JD, std::move(G));
      });
  ASSERT_THAT_EXPECTED(G, Succeeded());
  JD.addGenerator(std::move(*G));

  class CheckDefs : public ObjectLinkingLayer::Plugin {
  public:
    ~CheckDefs() { EXPECT_TRUE(SawSymbolDef); }

    void modifyPassConfig(MaterializationResponsibility &MR,
                          jitlink::LinkGraph &G,
                          jitlink::PassConfiguration &Config) override {
      Config.PostAllocationPasses.push_back([this](LinkGraph &G) {
        unsigned SymCount = 0;
        for (Symbol *Sym : G.absolute_symbols()) {
          SymCount += 1;
          if (!Sym->hasName()) {
            ADD_FAILURE() << "unexpected unnamed symbol";
            continue;
          }
          if (*Sym->getName() == "_testFunc")
            SawSymbolDef = true;
          else
            ADD_FAILURE() << "unexpected symbol " << Sym->getName();
        }
        EXPECT_EQ(SymCount, 1u);
        return Error::success();
      });
    }

    Error notifyFailed(MaterializationResponsibility &MR) override {
      return Error::success();
    }

    Error notifyRemovingResources(JITDylib &JD, ResourceKey K) override {
      return Error::success();
    }
    void notifyTransferringResources(JITDylib &JD, ResourceKey DstKey,
                                     ResourceKey SrcKey) override {
      llvm_unreachable("unexpected resource transfer");
    }

  private:
    bool SawSymbolDef = false;
  };

  ObjLinkingLayer.addPlugin(std::make_unique<CheckDefs>());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_testFunc"), Succeeded());
  EXPECT_THAT_ERROR(ES.endSession(), Succeeded());
}

} // end anonymous namespace
