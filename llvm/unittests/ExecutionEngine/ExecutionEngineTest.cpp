//===- ExecutionEngineTest.cpp - Unit tests for ExecutionEngine -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ExecutionEngineTest : public testing::Test {
protected:
  ExecutionEngineTest()
    : M(new Module("<main>", getGlobalContext())), Error(""),
      Engine(EngineBuilder(M).setErrorStr(&Error).create()) {
  }

  virtual void SetUp() {
    ASSERT_TRUE(Engine.get() != NULL) << "EngineBuilder returned error: '"
      << Error << "'";
  }

  GlobalVariable *NewExtGlobal(Type *T, const Twine &Name) {
    return new GlobalVariable(*M, T, false,  // Not constant.
                              GlobalValue::ExternalLinkage, NULL, Name);
  }

  Module *const M;
  std::string Error;
  const OwningPtr<ExecutionEngine> Engine;
};

TEST_F(ExecutionEngineTest, ForwardGlobalMapping) {
  GlobalVariable *G1 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global1");
  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(&Mem1, Engine->getPointerToGlobalIfAvailable(G1));
  int32_t Mem2 = 4;
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1));
  Engine->updateGlobalMapping(G1, NULL);
  EXPECT_EQ(NULL, Engine->getPointerToGlobalIfAvailable(G1));
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1));

  GlobalVariable *G2 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global1");
  EXPECT_EQ(NULL, Engine->getPointerToGlobalIfAvailable(G2))
    << "The NULL return shouldn't depend on having called"
    << " updateGlobalMapping(..., NULL)";
  // Check that update...() can be called before add...().
  Engine->updateGlobalMapping(G2, &Mem1);
  EXPECT_EQ(&Mem1, Engine->getPointerToGlobalIfAvailable(G2));
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1))
    << "A second mapping shouldn't affect the first.";
}

TEST_F(ExecutionEngineTest, ReverseGlobalMapping) {
  GlobalVariable *G1 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global1");

  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));
  int32_t Mem2 = 4;
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(NULL, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem2));

  GlobalVariable *G2 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global2");
  Engine->updateGlobalMapping(G2, &Mem1);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem2));
  Engine->updateGlobalMapping(G1, NULL);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1))
    << "Removing one mapping doesn't affect a different one.";
  EXPECT_EQ(NULL, Engine->getGlobalValueAtAddress(&Mem2));
  Engine->updateGlobalMapping(G2, &Mem2);
  EXPECT_EQ(NULL, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem2))
    << "Once a mapping is removed, we can point another GV at the"
    << " now-free address.";
}

TEST_F(ExecutionEngineTest, ClearModuleMappings) {
  GlobalVariable *G1 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global1");

  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));

  Engine->clearGlobalMappingsFromModule(M);

  EXPECT_EQ(NULL, Engine->getGlobalValueAtAddress(&Mem1));

  GlobalVariable *G2 =
      NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global2");
  // After clearing the module mappings, we can assign a new GV to the
  // same address.
  Engine->addGlobalMapping(G2, &Mem1);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1));
}

TEST_F(ExecutionEngineTest, DestructionRemovesGlobalMapping) {
  GlobalVariable *G1 =
    NewExtGlobal(Type::getInt32Ty(getGlobalContext()), "Global1");
  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  // Make sure the reverse mapping is enabled.
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));
  // When the GV goes away, the ExecutionEngine should remove any
  // mappings that refer to it.
  G1->eraseFromParent();
  EXPECT_EQ(NULL, Engine->getGlobalValueAtAddress(&Mem1));
}

}
