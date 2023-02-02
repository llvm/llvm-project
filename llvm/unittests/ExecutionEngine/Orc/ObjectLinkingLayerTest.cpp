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
  auto G =
      std::make_unique<LinkGraph>("foo", Triple("x86_64-apple-darwin"), 8,
                                  support::little, x86_64::getEdgeKindName);

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

  auto G =
      std::make_unique<LinkGraph>("foo", Triple("x86_64-apple-darwin"), 8,
                                  support::little, x86_64::getEdgeKindName);

  auto &DataSec = G->createSection("__data", MemProt::Read | MemProt::Write);
  auto &DataBlock = G->createContentBlock(DataSec, BlockContent,
                                          orc::ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(DataBlock, 4, "_anchor", 4, Linkage::Weak, Scope::Default,
                      false, true);

  EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G)), Succeeded());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_anchor"), Succeeded());
}

} // end anonymous namespace
