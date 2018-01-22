//===- llvm/unittests/IR/DominatorTreeBatchUpdatesTest.cpp ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <random>
#include "CFGBuilder.h"
#include "gtest/gtest.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#define DEBUG_TYPE "batch-update-tests"

using namespace llvm;

namespace {
const auto CFGInsert = CFGBuilder::ActionKind::Insert;
const auto CFGDelete = CFGBuilder::ActionKind::Delete;

struct PostDomTree : PostDomTreeBase<BasicBlock> {
  PostDomTree(Function &F) { recalculate(F); }
};

using DomUpdate = DominatorTree::UpdateType;
static_assert(
    std::is_same<DomUpdate, PostDomTree::UpdateType>::value,
    "Trees differing only in IsPostDom should have the same update types");
using DomSNCA = DomTreeBuilder::SemiNCAInfo<DomTreeBuilder::BBDomTree>;
using PostDomSNCA = DomTreeBuilder::SemiNCAInfo<DomTreeBuilder::BBPostDomTree>;
const auto Insert = DominatorTree::Insert;
const auto Delete = DominatorTree::Delete;

std::vector<DomUpdate> ToDomUpdates(CFGBuilder &B,
                                    std::vector<CFGBuilder::Update> &In) {
  std::vector<DomUpdate> Res;
  Res.reserve(In.size());

  for (const auto &CFGU : In)
    Res.push_back({CFGU.Action == CFGInsert ? Insert : Delete,
                   B.getOrAddBlock(CFGU.Edge.From),
                   B.getOrAddBlock(CFGU.Edge.To)});
  return Res;
}
}  // namespace

TEST(DominatorTreeBatchUpdates, LegalizeDomUpdates) {
  CFGHolder Holder;
  CFGBuilder Builder(Holder.F, {{"A", "B"}}, {});

  BasicBlock *A = Builder.getOrAddBlock("A");
  BasicBlock *B = Builder.getOrAddBlock("B");
  BasicBlock *C = Builder.getOrAddBlock("C");
  BasicBlock *D = Builder.getOrAddBlock("D");

  std::vector<DomUpdate> Updates = {
      {Insert, B, C}, {Insert, C, D}, {Delete, B, C}, {Insert, B, C},
      {Insert, B, D}, {Delete, C, D}, {Delete, A, B}};
  SmallVector<DomUpdate, 4> Legalized;
  DomSNCA::LegalizeUpdates(Updates, Legalized);
  DEBUG(dbgs() << "Legalized updates:\t");
  DEBUG(for (auto &U : Legalized) dbgs() << U << ", ");
  DEBUG(dbgs() << "\n");
  EXPECT_EQ(Legalized.size(), 3UL);
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Insert, B, C}), Legalized.end());
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Insert, B, D}), Legalized.end());
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Delete, A, B}), Legalized.end());
}

TEST(DominatorTreeBatchUpdates, LegalizePostDomUpdates) {
  CFGHolder Holder;
  CFGBuilder Builder(Holder.F, {{"A", "B"}}, {});

  BasicBlock *A = Builder.getOrAddBlock("A");
  BasicBlock *B = Builder.getOrAddBlock("B");
  BasicBlock *C = Builder.getOrAddBlock("C");
  BasicBlock *D = Builder.getOrAddBlock("D");

  std::vector<DomUpdate> Updates = {
      {Insert, B, C}, {Insert, C, D}, {Delete, B, C}, {Insert, B, C},
      {Insert, B, D}, {Delete, C, D}, {Delete, A, B}};
  SmallVector<DomUpdate, 4> Legalized;
  PostDomSNCA::LegalizeUpdates(Updates, Legalized);
  DEBUG(dbgs() << "Legalized postdom updates:\t");
  DEBUG(for (auto &U : Legalized) dbgs() << U << ", ");
  DEBUG(dbgs() << "\n");
  EXPECT_EQ(Legalized.size(), 3UL);
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Insert, C, B}), Legalized.end());
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Insert, D, B}), Legalized.end());
  EXPECT_NE(llvm::find(Legalized, DomUpdate{Delete, B, A}), Legalized.end());
}

TEST(DominatorTreeBatchUpdates, SingleInsertion) {
  CFGHolder Holder;
  CFGBuilder Builder(Holder.F, {{"A", "B"}}, {{CFGInsert, {"B", "C"}}});

  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(DT.verify());

  BasicBlock *B = Builder.getOrAddBlock("B");
  BasicBlock *C = Builder.getOrAddBlock("C");
  std::vector<DomUpdate> Updates = {{Insert, B, C}};

  ASSERT_TRUE(Builder.applyUpdate());

  DT.applyUpdates(Updates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(Updates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, SingleDeletion) {
  CFGHolder Holder;
  CFGBuilder Builder(Holder.F, {{"A", "B"}, {"B", "C"}},
                     {{CFGDelete, {"B", "C"}}});

  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(DT.verify());

  BasicBlock *B = Builder.getOrAddBlock("B");
  BasicBlock *C = Builder.getOrAddBlock("C");
  std::vector<DomUpdate> Updates = {{Delete, B, C}};

  ASSERT_TRUE(Builder.applyUpdate());

  DT.applyUpdates(Updates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(Updates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, FewInsertion) {
  std::vector<CFGBuilder::Update> CFGUpdates = {{CFGInsert, {"B", "C"}},
                                                {CFGInsert, {"C", "B"}},
                                                {CFGInsert, {"C", "D"}},
                                                {CFGInsert, {"D", "E"}}};

  CFGHolder Holder;
  CFGBuilder Builder(Holder.F, {{"A", "B"}}, CFGUpdates);

  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  BasicBlock *B = Builder.getOrAddBlock("B");
  BasicBlock *C = Builder.getOrAddBlock("C");
  BasicBlock *D = Builder.getOrAddBlock("D");
  BasicBlock *E = Builder.getOrAddBlock("E");

  std::vector<DomUpdate> Updates = {
      {Insert, B, C}, {Insert, C, B}, {Insert, C, D}, {Insert, D, E}};

  while (Builder.applyUpdate())
    ;

  DT.applyUpdates(Updates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(Updates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, FewDeletions) {
  std::vector<CFGBuilder::Update> CFGUpdates = {{CFGDelete, {"B", "C"}},
                                                {CFGDelete, {"C", "B"}},
                                                {CFGDelete, {"B", "D"}},
                                                {CFGDelete, {"D", "E"}}};

  CFGHolder Holder;
  CFGBuilder Builder(
      Holder.F, {{"A", "B"}, {"B", "C"}, {"B", "D"}, {"D", "E"}, {"C", "B"}},
      CFGUpdates);

  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  auto Updates = ToDomUpdates(Builder, CFGUpdates);

  while (Builder.applyUpdate())
    ;

  DT.applyUpdates(Updates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(Updates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, InsertDelete) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"3", "8"}, {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {
      {CFGInsert, {"2", "4"}},  {CFGInsert, {"12", "10"}},
      {CFGInsert, {"10", "9"}}, {CFGInsert, {"7", "6"}},
      {CFGInsert, {"7", "5"}},  {CFGDelete, {"3", "8"}},
      {CFGInsert, {"10", "7"}}, {CFGInsert, {"2", "8"}},
      {CFGDelete, {"3", "4"}},  {CFGDelete, {"8", "9"}},
      {CFGDelete, {"11", "12"}}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  while (B.applyUpdate())
    ;

  auto DomUpdates = ToDomUpdates(B, Updates);
  DT.applyUpdates(DomUpdates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(DomUpdates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, InsertDeleteExhaustive) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"}, {"2", "3"}, {"3", "4"},  {"4", "5"},  {"5", "6"},  {"5", "7"},
      {"3", "8"}, {"8", "9"}, {"9", "10"}, {"8", "11"}, {"11", "12"}};

  std::vector<CFGBuilder::Update> Updates = {
      {CFGInsert, {"2", "4"}},  {CFGInsert, {"12", "10"}},
      {CFGInsert, {"10", "9"}}, {CFGInsert, {"7", "6"}},
      {CFGInsert, {"7", "5"}},  {CFGDelete, {"3", "8"}},
      {CFGInsert, {"10", "7"}}, {CFGInsert, {"2", "8"}},
      {CFGDelete, {"3", "4"}},  {CFGDelete, {"8", "9"}},
      {CFGDelete, {"11", "12"}}};

  std::mt19937 Generator(0);
  for (unsigned i = 0; i < 16; ++i) {
    std::shuffle(Updates.begin(), Updates.end(), Generator);
    CFGHolder Holder;
    CFGBuilder B(Holder.F, Arcs, Updates);
    DominatorTree DT(*Holder.F);
    EXPECT_TRUE(DT.verify());
    PostDomTree PDT(*Holder.F);
    EXPECT_TRUE(PDT.verify());

    while (B.applyUpdate())
      ;

    auto DomUpdates = ToDomUpdates(B, Updates);
    DT.applyUpdates(DomUpdates);
    EXPECT_TRUE(DT.verify());
    PDT.applyUpdates(DomUpdates);
    EXPECT_TRUE(PDT.verify());
  }
}

// These are some odd flowgraphs, usually generated from csmith cases,
// which are difficult on post dom trees.
TEST(DominatorTreeBatchUpdates, InfiniteLoop) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"},
      {"2", "3"},
      {"3", "6"}, {"3", "5"},
      {"4", "5"},
      {"5", "2"},
      {"6", "3"}, {"6", "4"}};

  // SplitBlock on 3 -> 5
  std::vector<CFGBuilder::Update> Updates = {
      {CFGInsert, {"N", "5"}},  {CFGInsert, {"3", "N"}}, {CFGDelete, {"3", "5"}}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  while (B.applyUpdate())
    ;

  auto DomUpdates = ToDomUpdates(B, Updates);
  DT.applyUpdates(DomUpdates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(DomUpdates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, DeadBlocks) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"},
      {"2", "3"},
      {"3", "4"}, {"3", "7"},
      {"4", "4"},
      {"5", "6"}, {"5", "7"},
      {"6", "7"},
      {"7", "2"}, {"7", "8"}};

  // Remove dead 5 and 7,
  // plus SplitBlock on 7 -> 8
  std::vector<CFGBuilder::Update> Updates = {
      {CFGDelete, {"6", "7"}},  {CFGDelete, {"5", "7"}}, {CFGDelete, {"5", "6"}},
      {CFGInsert, {"N", "8"}},  {CFGInsert, {"7", "N"}}, {CFGDelete, {"7", "8"}}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  while (B.applyUpdate())
    ;

  auto DomUpdates = ToDomUpdates(B, Updates);
  DT.applyUpdates(DomUpdates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(DomUpdates);
  EXPECT_TRUE(PDT.verify());
}

TEST(DominatorTreeBatchUpdates, InfiniteLoop2) {
  std::vector<CFGBuilder::Arc> Arcs = {
      {"1", "2"},
      {"2", "6"}, {"2", "3"},
      {"3", "4"},
      {"4", "5"}, {"4", "6"},
      {"5", "4"},
      {"6", "2"}};

  // SplitBlock on 4 -> 6
  std::vector<CFGBuilder::Update> Updates = {
      {CFGInsert, {"N", "6"}},  {CFGInsert, {"4", "N"}}, {CFGDelete, {"4", "6"}}};

  CFGHolder Holder;
  CFGBuilder B(Holder.F, Arcs, Updates);
  DominatorTree DT(*Holder.F);
  EXPECT_TRUE(DT.verify());
  PostDomTree PDT(*Holder.F);
  EXPECT_TRUE(PDT.verify());

  while (B.applyUpdate())
    ;

  auto DomUpdates = ToDomUpdates(B, Updates);
  DT.applyUpdates(DomUpdates);
  EXPECT_TRUE(DT.verify());
  PDT.applyUpdates(DomUpdates);
  EXPECT_TRUE(PDT.verify());
}
