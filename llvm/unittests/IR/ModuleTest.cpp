//===- unittests/IR/ModuleTest.cpp - Module unit tests --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "gtest/gtest.h"

#include <random>

using namespace llvm;

namespace {

bool sortByName(const GlobalVariable &L, const GlobalVariable &R) {
  return L.getName() < R.getName();
}

bool sortByNameReverse(const GlobalVariable &L, const GlobalVariable &R) {
  return sortByName(R, L);
}

TEST(ModuleTest, sortGlobalsByName) {
  LLVMContext Context;
  for (auto compare : {&sortByName, &sortByNameReverse}) {
    Module M("M", Context);
    Type *T = Type::getInt8Ty(Context);
    GlobalValue::LinkageTypes L = GlobalValue::ExternalLinkage;
    (void)new GlobalVariable(M, T, false, L, nullptr, "A");
    (void)new GlobalVariable(M, T, false, L, nullptr, "F");
    (void)new GlobalVariable(M, T, false, L, nullptr, "G");
    (void)new GlobalVariable(M, T, false, L, nullptr, "E");
    (void)new GlobalVariable(M, T, false, L, nullptr, "B");
    (void)new GlobalVariable(M, T, false, L, nullptr, "H");
    (void)new GlobalVariable(M, T, false, L, nullptr, "C");
    (void)new GlobalVariable(M, T, false, L, nullptr, "D");

    // Sort the globals by name.
    EXPECT_FALSE(std::is_sorted(M.global_begin(), M.global_end(), compare));
    M.getGlobalList().sort(compare);
    EXPECT_TRUE(std::is_sorted(M.global_begin(), M.global_end(), compare));
  }
}

TEST(ModuleTest, randomNumberGenerator) {
  LLVMContext Context;
  static char ID;
  struct DummyPass : ModulePass {
    DummyPass() : ModulePass(ID) {}
    bool runOnModule(Module &) { return true; }
  } DP;

  Module M("R", Context);

  std::uniform_int_distribution<int> dist;
  const size_t NBCheck = 10;

  std::array<int, NBCheck> RandomStreams[2];
  for (auto &RandomStream : RandomStreams) {
    std::unique_ptr<RandomNumberGenerator> RNG{M.createRNG(&DP)};
    std::generate(RandomStream.begin(), RandomStream.end(),
                  [&]() { return dist(*RNG); });
  }

  EXPECT_TRUE(std::equal(RandomStreams[0].begin(), RandomStreams[0].end(),
                         RandomStreams[1].begin()));
}

} // end namespace
