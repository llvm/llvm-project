//===--- JITLinkGenericTests.cpp - Tests for the generic JIT linker -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the target-independent link pipeline implemented by JITLinkerBase
// (see lib/ExecutionEngine/JITLink/JITLinkGeneric.h): the sequencing of link
// phases and pass lists, error reporting from each phase, and the lifetimes
// of the linker, LinkGraph, and SymbolStringPool across continuations,
// including continuations that complete synchronously.
//
// These tests use the mock JITLinkContext and JITLinkMemoryManager from
// JITLinkTestUtils.h to control lookup, allocation, and finalization results
// and timing. Tests of the LinkGraph API belong in LinkGraphTests.cpp, and
// tests of target-specific fixups, stubs, or object-file parsing belong in
// the corresponding target- or format-specific test files.
//
//===----------------------------------------------------------------------===//

#include "JITLinkTestUtils.h"
#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"
#include "llvm/ExecutionEngine/JITLink/x86_64.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::jitlink;

namespace {

/// Build a graph containing a single data block that points to an external
/// symbol "E", so that linking it will issue a lookup. The graph holds the
/// only reference to its SymbolStringPool.
std::unique_ptr<LinkGraph> makeGraphWithOneExternal() {
  auto G = std::make_unique<LinkGraph>(
      "foo", std::make_shared<orc::SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), 8, SubtargetFeatures(),
      getGenericEdgeKindName);

  auto &Sec =
      G->createSection("__data", orc::MemProt::Read | orc::MemProt::Write);
  auto &B = G->createContentBlock(Sec, BlockContent.slice(0, 8),
                                  orc::ExecutorAddr(0x1000), 8, 0);
  G->addDefinedSymbol(B, 0, "S", 8, Linkage::Strong, Scope::Default, false,
                      false);
  // Keep the external live so that it reaches lookup.
  auto &E = G->addExternalSymbol("E", 0, false);
  B.addEdge(x86_64::Pointer64, 0, E, 0);

  return G;
}

} // end anonymous namespace

TEST(JITLinkGenericTest, FailedLookupWithGraphOwnedPool) {
  // The graph holds the only reference to its SymbolStringPool, so the pool is
  // destroyed along with the graph when the link fails inside lookup. Check
  // that no SymbolStringPtrs outlive the pool.
  Error Err = Error::success();
  auto Ctx = makeMockContext(
      JoinErrorsInto(Err), defaultMemMgrSetup,
      [](MockJITLinkContext &Ctx) { Ctx.Lookup = lookupErrorOut; });

  link_MachO_x86_64(makeGraphWithOneExternal(), std::move(Ctx));

  EXPECT_THAT_ERROR(std::move(Err), FailedWithMessage("Lookup failed"));
}

TEST(JITLinkGenericTest, SuccessfulLookupWithGraphOwnedPool) {
  // The graph holds the only reference to its SymbolStringPool. Lookup and
  // finalization both complete synchronously, so the link completes (and the
  // graph and pool are destroyed) before linkPhase3 returns. Check that no
  // SymbolStringPtrs from the lookup result outlive the pool.
  Error Err = Error::success();
  auto Ctx = makeMockContext(JoinErrorsInto(Err), defaultMemMgrSetup,
                             [](MockJITLinkContext &Ctx) {
                               Ctx.Lookup = lookupResolveEverythingToNull;
                             });

  link_MachO_x86_64(makeGraphWithOneExternal(), std::move(Ctx));

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
}
