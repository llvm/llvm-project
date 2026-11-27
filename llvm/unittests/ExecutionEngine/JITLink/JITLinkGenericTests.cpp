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

/// Build an empty graph. The graph holds the only reference to its
/// SymbolStringPool.
std::unique_ptr<LinkGraph> makeEmptyGraph() {
  return std::make_unique<LinkGraph>(
      "foo", std::make_shared<orc::SymbolStringPool>(),
      Triple("x86_64-apple-darwin"), 8, SubtargetFeatures(),
      getGenericEdgeKindName);
}

/// Build a graph containing a single data block that points to an external
/// symbol "E", so that linking it will issue a lookup. The graph holds the
/// only reference to its SymbolStringPool.
std::unique_ptr<LinkGraph> makeGraphWithOneExternal() {
  auto G = makeEmptyGraph();

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

/// A stage of the link pipeline at which passes can be run.
struct PassStage {
  const char *Name;
  LinkGraphPassList PassConfiguration::*Passes;
  bool AfterAllocation;
};

const PassStage PassStages[] = {
    {"PrePrune", &PassConfiguration::PrePrunePasses, false},
    {"PostPrune", &PassConfiguration::PostPrunePasses, false},
    {"PostAllocation", &PassConfiguration::PostAllocationPasses, true},
    {"PreFixup", &PassConfiguration::PreFixupPasses, true},
    {"PostFixup", &PassConfiguration::PostFixupPasses, true},
};

enum class GraphKind { Empty, NonEmpty };

class JITLinkGenericPassErrorTest
    : public testing::TestWithParam<std::tuple<PassStage, GraphKind>> {};

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

TEST(JITLinkGenericTest, EmptySectionsAreNotAllocated) {
  // Sections with no blocks need no memory.
  auto G = makeEmptyGraph();
  G->createSection("__text", orc::MemProt::Read | orc::MemProt::Exec);

  bool AllocateCalled = false;
  Error Err = Error::success();
  auto Ctx = makeMockContext(
      JoinErrorsInto(Err),
      [&](MockJITLinkMemoryManager &MemMgr) {
        MemMgr.Allocate = [&](const JITLinkDylib *JD, LinkGraph &LG) {
          AllocateCalled = true;
          return MemMgr.defaultAllocate(JD, LG);
        };
      },
      defaultCtxSetup);

  link_MachO_x86_64(std::move(G), std::move(Ctx));

  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_FALSE(AllocateCalled);
}

TEST_P(JITLinkGenericPassErrorTest, FailingPassIsReported) {
  // Check that an error returned by a pass at any stage of the pipeline is
  // reported exactly once via notifyFailed, that the link stops, and that any
  // allocation is abandoned. Empty graphs skip allocation, so they exercise
  // the failure paths without an allocation to abandon.
  const PassStage &Stage = std::get<0>(GetParam());
  GraphKind Kind = std::get<1>(GetParam());
  std::string ExpectedMsg = std::string(Stage.Name) + " pass failed";

  bool AllocateCalled = false;
  bool AbandonCalled = false;
  bool FinalizeCalled = false;
  bool NotifyFinalizedCalled = false;
  unsigned FailureCount = 0;
  Error Err = Error::success();

  auto Ctx = makeMockContext(
      [&](Error E) {
        ++FailureCount;
        Err = joinErrors(std::move(Err), std::move(E));
      },
      [&](MockJITLinkMemoryManager &MemMgr) {
        MemMgr.Allocate = [&](const JITLinkDylib *JD, LinkGraph &G) {
          AllocateCalled = true;
          return MemMgr.defaultAllocate(JD, G);
        };
        MemMgr.Abandon =
            [&](std::unique_ptr<MockJITLinkMemoryManager::Alloc> A) {
              AbandonCalled = true;
              return MemMgr.defaultAbandon(std::move(A));
            };
        MemMgr.Finalize =
            [&](std::unique_ptr<MockJITLinkMemoryManager::Alloc> A) {
              FinalizeCalled = true;
              return MemMgr.defaultFinalize(std::move(A));
            };
      },
      [&](MockJITLinkContext &Ctx) {
        Ctx.Lookup = lookupResolveEverythingToNull;
        Ctx.NotifyFinalized = [&](JITLinkMemoryManager::FinalizedAlloc) {
          NotifyFinalizedCalled = true;
        };
        Ctx.ModifyPassConfig = [Passes = Stage.Passes, ExpectedMsg](
                                   LinkGraph &, PassConfiguration &Config) {
          (Config.*Passes).push_back([ExpectedMsg](LinkGraph &) {
            return make_error<StringError>(ExpectedMsg,
                                           inconvertibleErrorCode());
          });
          return Error::success();
        };
      });

  link_MachO_x86_64(Kind == GraphKind::Empty ? makeEmptyGraph()
                                             : makeGraphWithOneExternal(),
                    std::move(Ctx));

  EXPECT_EQ(FailureCount, 1U) << "Failure should be reported exactly once";
  EXPECT_THAT_ERROR(std::move(Err), FailedWithMessage(ExpectedMsg));
  EXPECT_FALSE(FinalizeCalled) << "Allocation should not be finalized";
  EXPECT_FALSE(NotifyFinalizedCalled) << "Link should not complete";

  // Non-empty graphs allocate before the post-allocation passes run, and must
  // abandon that allocation on failure.
  bool ExpectAlloc = Kind == GraphKind::NonEmpty && Stage.AfterAllocation;
  EXPECT_EQ(AllocateCalled, ExpectAlloc);
  EXPECT_EQ(AbandonCalled, ExpectAlloc);
}

INSTANTIATE_TEST_SUITE_P(
    AllStages, JITLinkGenericPassErrorTest,
    testing::Combine(testing::ValuesIn(PassStages),
                     testing::Values(GraphKind::Empty, GraphKind::NonEmpty)),
    [](const testing::TestParamInfo<JITLinkGenericPassErrorTest::ParamType>
           &Info) {
      const PassStage &Stage = std::get<0>(Info.param);
      GraphKind Kind = std::get<1>(Info.param);
      return std::string(Stage.Name) +
             (Kind == GraphKind::Empty ? "_EmptyGraph" : "_NonEmptyGraph");
    });
