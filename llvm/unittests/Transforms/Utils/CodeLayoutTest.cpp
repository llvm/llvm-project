#include "llvm/Transforms/Utils/CodeLayout.h"
#include "llvm/Support/CommandLine.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace llvm::codelayout;
using testing::ElementsAreArray;

namespace llvm::codelayout {
extern cl::opt<unsigned> CDMaxChainSize;
}

namespace {
TEST(CodeLayout, ThreeFunctions) {
  // Place the most likely successor (2) first.
  {
    const uint64_t Counts[3] = {140, 40, 140};
    const std::vector<uint64_t> Sizes(std::size(Counts), 9);
    const EdgeCount Edges[] = {{0, 1, 40}, {0, 2, 100}, {1, 2, 40}};
    const std::vector<uint64_t> CallOffsets(std::size(Edges), 5);
    auto Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
    EXPECT_THAT(Order, ElementsAreArray({0, 2, 1}));
  }

  // Prefer fallthroughs even in the presence of a heavy successor.
  {
    const uint64_t Counts[3] = {180, 80, 180};
    const std::vector<uint64_t> Sizes(std::size(Counts), 9);
    const EdgeCount Edges[] = {{0, 1, 80}, {0, 2, 100}, {1, 2, 80}};
    const uint64_t CallOffsets[] = {9, 5, 9};
    auto Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
    EXPECT_THAT(Order, ElementsAreArray({0, 1, 2}));
  }
}

TEST(CodeLayout, HotChain) {
  // Place the hot chain (0,3,4,2) continuously.
  {
    const uint64_t Counts[5] = {22, 7, 22, 15, 46};
    const std::vector<uint64_t> Sizes(std::size(Counts), 9);
    const EdgeCount Edges[] = {{0, 1, 7},  {1, 2, 7},  {0, 3, 15},
                               {3, 4, 15}, {4, 4, 31}, {4, 2, 15}};
    const std::vector<uint64_t> CallOffsets(std::size(Edges), 5);
    auto Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
    EXPECT_THAT(Order, ElementsAreArray({0, 3, 4, 2, 1}));

    // -cdsort-max-chain-size disables forming a larger chain and therefore may
    // change the result.
    unsigned Saved = CDMaxChainSize;
    CDMaxChainSize.setValue(3);
    Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
    EXPECT_THAT(Order, ElementsAreArray({0, 3, 4, 1, 2}));
    CDMaxChainSize.setValue(Saved);
  }
}

TEST(CodeLayout, BreakLoop) {
  // There are two loops (1,2,3) and (1,2,4). It is beneficial to place 4
  // elsewhere.
  const uint64_t Counts[6] = {177, 371, 196, 124, 70, 177};
  std::vector<uint64_t> Sizes(std::size(Counts), 9);
  const EdgeCount Edges[] = {{0, 1, 177}, {1, 2, 196}, {2, 3, 124}, {3, 1, 124},
                             {1, 5, 177}, {2, 4, 79},  {4, 1, 70}};
  const std::vector<uint64_t> CallOffsets(std::size(Edges), 5);
  auto Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
  EXPECT_THAT(Order, ElementsAreArray({4, 0, 1, 2, 3, 5}));

  // When node 0 is larger, it is beneficial to move node 4 closer to the
  // (1,2,3) loop.
  Sizes[0] = 18;
  Order = computeCacheDirectedLayout(Sizes, Counts, Edges, CallOffsets);
  EXPECT_THAT(Order, ElementsAreArray({0, 4, 1, 2, 3, 5}));
}
} // namespace
