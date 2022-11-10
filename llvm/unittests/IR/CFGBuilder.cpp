//===- llvm/Testing/Support/CFGBuilder.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CFGBuilder.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "cfg-builder"

using namespace llvm;

CFGHolder::CFGHolder(StringRef ModuleName, StringRef FunctionName)
    : Context(std::make_unique<LLVMContext>()),
      M(std::make_unique<Module>(ModuleName, *Context)) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Context), {}, false);
  F = Function::Create(FTy, Function::ExternalLinkage, FunctionName, M.get());
}
CFGHolder::~CFGHolder() = default;

CFGBuilder::CFGBuilder(Function *F, const std::vector<Arc> &InitialArcs,
                       std::vector<Update> Updates)
    : F(F), Updates(std::move(Updates)) {
  assert(F);
  buildCFG(InitialArcs);
}

static void ConnectBlocks(BasicBlock *From, BasicBlock *To) {
  LLVM_DEBUG(dbgs() << "Creating BB arc " << From->getName() << " -> "
                    << To->getName() << "\n";
             dbgs().flush());
  auto *IntTy = IntegerType::get(From->getContext(), 32);

  if (isa<UnreachableInst>(From->getTerminator()))
    From->getTerminator()->eraseFromParent();
  if (!From->getTerminator()) {
    IRBuilder<> IRB(From);
    IRB.CreateSwitch(ConstantInt::get(IntTy, 0), To);
    return;
  }

  SwitchInst *SI = cast<SwitchInst>(From->getTerminator());
  const auto Last = SI->getNumCases();

  auto *IntVal = ConstantInt::get(IntTy, Last);
  SI->addCase(IntVal, To);
}

static void DisconnectBlocks(BasicBlock *From, BasicBlock *To) {
  LLVM_DEBUG(dbgs() << "Deleting BB arc " << From->getName() << " -> "
                    << To->getName() << "\n";
             dbgs().flush());
  SwitchInst *SI = cast<SwitchInst>(From->getTerminator());

  if (SI->getNumCases() == 0) {
    SI->eraseFromParent();
    IRBuilder<> IRB(From);
    IRB.CreateUnreachable();
    return;
  }

  if (SI->getDefaultDest() == To) {
    auto FirstC = SI->case_begin();
    SI->setDefaultDest(FirstC->getCaseSuccessor());
    SI->removeCase(FirstC);
    return;
  }

  for (auto CIt = SI->case_begin(); CIt != SI->case_end(); ++CIt)
    if (CIt->getCaseSuccessor() == To) {
      SI->removeCase(CIt);
      return;
    }
}

BasicBlock *CFGBuilder::getOrAddBlock(StringRef BlockName) {
  auto BIt = NameToBlock.find(BlockName);
  if (BIt != NameToBlock.end())
    return BIt->second;

  auto *BB = BasicBlock::Create(F->getParent()->getContext(), BlockName, F);
  IRBuilder<> IRB(BB);
  IRB.CreateUnreachable();
  NameToBlock[BlockName] = BB;
  return BB;
}

bool CFGBuilder::connect(const Arc &A) {
  BasicBlock *From = getOrAddBlock(A.From);
  BasicBlock *To = getOrAddBlock(A.To);
  if (Arcs.count(A) != 0)
    return false;

  Arcs.insert(A);
  ConnectBlocks(From, To);
  return true;
}

bool CFGBuilder::disconnect(const Arc &A) {
  assert(NameToBlock.count(A.From) != 0 && "No block to disconnect (From)");
  assert(NameToBlock.count(A.To) != 0 && "No block to disconnect (To)");
  if (Arcs.count(A) == 0)
    return false;

  BasicBlock *From = getOrAddBlock(A.From);
  BasicBlock *To = getOrAddBlock(A.To);
  Arcs.erase(A);
  DisconnectBlocks(From, To);
  return true;
}

void CFGBuilder::buildCFG(const std::vector<Arc> &NewArcs) {
  for (const auto &A : NewArcs) {
    const bool Connected = connect(A);
    (void)Connected;
    assert(Connected);
  }
}

Optional<CFGBuilder::Update> CFGBuilder::getNextUpdate() const {
  if (UpdateIdx == Updates.size())
    return None;
  return Updates[UpdateIdx];
}

Optional<CFGBuilder::Update> CFGBuilder::applyUpdate() {
  if (UpdateIdx == Updates.size())
    return None;
  Update NextUpdate = Updates[UpdateIdx++];
  if (NextUpdate.Action == ActionKind::Insert)
    connect(NextUpdate.Edge);
  else
    disconnect(NextUpdate.Edge);

  return NextUpdate;
}

void CFGBuilder::dump(raw_ostream &OS) const {
  OS << "Arcs:\n";
  size_t i = 0;
  for (const auto &A : Arcs)
    OS << "  " << i++ << ":\t" << A.From << " -> " << A.To << "\n";

  OS << "Updates:\n";
  i = 0;
  for (const auto &U : Updates) {
    OS << (i + 1 == UpdateIdx ? "->" : "  ") << i
       << ((U.Action == ActionKind::Insert) ? "\tIns " : "\tDel ")
       << U.Edge.From << " -> " << U.Edge.To << "\n";
    ++i;
  }
}

//---- CFGBuilder tests ---------------------------------------------------===//

TEST(CFGBuilder, Construction) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {{"entry", "a"}, {"a", "b"}, {"a", "c"},
                                       {"c", "d"},     {"d", "b"}, {"d", "e"},
                                       {"d", "f"},     {"e", "f"}};
  CFGBuilder B(Holder.F, Arcs, {});

  EXPECT_TRUE(B.getOrAddBlock("entry") == &Holder.F->getEntryBlock());
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("entry")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("a")->getTerminator()));
  EXPECT_TRUE(isa<UnreachableInst>(B.getOrAddBlock("b")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("d")->getTerminator()));

  auto *DSwitch = cast<SwitchInst>(B.getOrAddBlock("d")->getTerminator());
  // d has 3 successors, but one of them if going to be a default case
  EXPECT_EQ(DSwitch->getNumCases(), 2U);
  EXPECT_FALSE(B.getNextUpdate()); // No updates to apply.
}

TEST(CFGBuilder, Insertions) {
  CFGHolder Holder;
  const auto Insert = CFGBuilder::ActionKind::Insert;
  std::vector<CFGBuilder::Update> Updates = {
      {Insert, {"entry", "a"}}, {Insert, {"a", "b"}}, {Insert, {"a", "c"}},
      {Insert, {"c", "d"}},     {Insert, {"d", "b"}}, {Insert, {"d", "e"}},
      {Insert, {"d", "f"}},     {Insert, {"e", "f"}}};
  const size_t NumUpdates = Updates.size();

  CFGBuilder B(Holder.F, {}, Updates);

  size_t i = 0;
  while (B.applyUpdate())
    ++i;
  EXPECT_EQ(i, NumUpdates);

  EXPECT_TRUE(B.getOrAddBlock("entry") == &Holder.F->getEntryBlock());
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("entry")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("a")->getTerminator()));
  EXPECT_TRUE(isa<UnreachableInst>(B.getOrAddBlock("b")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("d")->getTerminator()));

  auto *DSwitch = cast<SwitchInst>(B.getOrAddBlock("d")->getTerminator());
  // d has 3 successors, but one of them if going to be a default case
  EXPECT_EQ(DSwitch->getNumCases(), 2U);
  EXPECT_FALSE(B.getNextUpdate()); // No updates to apply.
}

TEST(CFGBuilder, Deletions) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"entry", "a"}, {"a", "b"}, {"a", "c"}, {"c", "d"}, {"d", "b"}};
  const auto Delete = CFGBuilder::ActionKind::Delete;
  std::vector<CFGBuilder::Update> Updates = {
      {Delete, {"c", "d"}}, {Delete, {"a", "c"}}, {Delete, {"entry", "a"}},
  };
  const size_t NumUpdates = Updates.size();

  CFGBuilder B(Holder.F, Arcs, Updates);

  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("entry")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("a")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("c")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("d")->getTerminator()));

  auto UpdateC = B.applyUpdate();

  EXPECT_TRUE(UpdateC);
  EXPECT_EQ(UpdateC->Action, CFGBuilder::ActionKind::Delete);
  EXPECT_EQ(UpdateC->Edge.From, "c");
  EXPECT_EQ(UpdateC->Edge.To, "d");
  EXPECT_TRUE(isa<UnreachableInst>(B.getOrAddBlock("c")->getTerminator()));

  size_t i = 1;
  while (B.applyUpdate())
    ++i;
  EXPECT_EQ(i, NumUpdates);

  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("a")->getTerminator()));
  EXPECT_TRUE(isa<UnreachableInst>(B.getOrAddBlock("entry")->getTerminator()));
}

TEST(CFGBuilder, Rebuild) {
  CFGHolder Holder;
  std::vector<CFGBuilder::Arc> Arcs = {
      {"entry", "a"}, {"a", "b"}, {"a", "c"}, {"c", "d"}, {"d", "b"}};
  const auto Insert = CFGBuilder::ActionKind::Insert;
  const auto Delete = CFGBuilder::ActionKind::Delete;
  std::vector<CFGBuilder::Update> Updates = {
      {Delete, {"c", "d"}}, {Delete, {"a", "c"}}, {Delete, {"entry", "a"}},
      {Insert, {"c", "d"}}, {Insert, {"a", "c"}}, {Insert, {"entry", "a"}},
  };
  const size_t NumUpdates = Updates.size();

  CFGBuilder B(Holder.F, Arcs, Updates);
  size_t i = 0;
  while (B.applyUpdate())
    ++i;
  EXPECT_EQ(i, NumUpdates);

  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("entry")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("a")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("c")->getTerminator()));
  EXPECT_TRUE(isa<SwitchInst>(B.getOrAddBlock("d")->getTerminator()));
}

static_assert(std::is_trivially_copyable_v<succ_iterator>,
              "trivially copyable");
static_assert(std::is_trivially_copyable_v<const_succ_iterator>,
              "trivially copyable");
static_assert(std::is_trivially_copyable_v<succ_range>, "trivially copyable");
static_assert(std::is_trivially_copyable_v<const_succ_range>,
              "trivially copyable");
