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

  EXPECT_THAT_ERROR(ObjLinkingLayer.add(JD, std::move(G)), Succeeded());

  EXPECT_THAT_EXPECTED(ES.lookup(&JD, "_X"), Succeeded());
}

} // end anonymous namespace
