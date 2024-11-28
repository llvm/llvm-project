//===---- MemoryManagerErrorTests.cpp - Test memory manager error paths ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestUtils.h"
#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::jitlink;

TEST(MemoryManagerErrorTest, ErrorOnFirstAllocate) {
  // Check that we can get addresses for blocks, symbols, and edges.
  auto G = std::make_unique<LinkGraph>(
      "foo", std::make_shared<orc::SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), 8, llvm::endianness::little,
      getGenericEdgeKindName);

  ArrayRef<char> Content = "hello, world!";
  auto &Sec =
      G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B = G->createContentBlock(Sec, Content, B1Addr, 8, 0);
  G->addDefinedSymbol(B, 4, "S", 4, Linkage::Strong, Scope::Default, false,
                      false);

  Error Err = Error::success();
  auto Ctx = makeMockContext(
      JoinErrorsInto(Err),
      [](MockJITLinkMemoryManager &MemMgr) {
        MemMgr.Allocate = [](const JITLinkDylib *JD, LinkGraph &G) {
          return make_error<StringError>("Failed to allocate",
                                         inconvertibleErrorCode());
        };
      },
      defaultCtxSetup);

  link_MachO_x86_64(std::move(G), std::move(Ctx));

  EXPECT_THAT_ERROR(std::move(Err), Failed());
}
