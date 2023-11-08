//===--------- JITLinkMocks.cpp - Mock APIs for JITLink unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITLinkMocks.h"
#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::jitlink;

void lookupResolveEverythingToNull(
    const llvm::jitlink::JITLinkContext::LookupMap &Symbols,
    std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation> LC) {
  llvm::orc::ExecutorAddr Null;
  llvm::jitlink::AsyncLookupResult Result;
  for (auto &KV : Symbols)
    Result[KV.first] = {Null, llvm::JITSymbolFlags::Exported};
  LC->run(std::move(Result));
}

void lookupErrorOut(
    const llvm::jitlink::JITLinkContext::LookupMap &Symbols,
    std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation> LC) {
  LC->run(llvm::make_error<llvm::StringError>("Lookup failed",
                                              llvm::inconvertibleErrorCode()));
}

std::unique_ptr<MockJITLinkContext> makeMockContext(
    llvm::unique_function<void(llvm::Error)> HandleFailed,
    llvm::unique_function<void(MockJITLinkMemoryManager &)> SetupMemMgr,
    llvm::unique_function<void(MockJITLinkContext &)> SetupContext) {
  auto MemMgr = std::make_unique<MockJITLinkMemoryManager>();
  SetupMemMgr(*MemMgr);
  auto Ctx = std::make_unique<MockJITLinkContext>(std::move(MemMgr),
                                                  std::move(HandleFailed));
  SetupContext(*Ctx);
  return Ctx;
}

void defaultMemMgrSetup(MockJITLinkMemoryManager &) {}
void defaultCtxSetup(MockJITLinkContext &) {}

TEST(JITLinkMocks, SmokeTest) {
  // Check that the testing infrastructure defaults can "link" a graph
  // successfully.
  auto G = std::make_unique<LinkGraph>("foo", Triple("x86_64-apple-darwin"), 8,
                                       llvm::endianness::little,
                                       getGenericEdgeKindName);

  ArrayRef<char> Content = "hello, world!";
  auto &Sec =
      G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  orc::ExecutorAddr B1Addr(0x1000);
  auto &B = G->createContentBlock(Sec, Content, B1Addr, 8, 0);
  G->addDefinedSymbol(B, 4, "S", 4, Linkage::Strong, Scope::Default, false,
                      false);

  Error Err = Error::success();
  auto Ctx =
      makeMockContext(JoinErrorsInto(Err), defaultMemMgrSetup, defaultCtxSetup);

  link_MachO_x86_64(std::move(G), std::move(Ctx));

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
}
