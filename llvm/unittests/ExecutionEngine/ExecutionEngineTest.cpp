//===- ExecutionEngineTest.cpp - Unit tests for ExecutionEngine -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ManagedStatic.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ExecutionEngineTest : public testing::Test {
private:
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

protected:
  ExecutionEngineTest() {
    auto Owner = make_unique<Module>("<main>", Context);
    M = Owner.get();
    Engine.reset(EngineBuilder(std::move(Owner)).setErrorStr(&Error).create());
  }

  void SetUp() override {
    ASSERT_TRUE(Engine.get() != nullptr) << "EngineBuilder returned error: '"
      << Error << "'";
  }

  GlobalVariable *NewExtGlobal(Type *T, const Twine &Name) {
    return new GlobalVariable(*M, T, false,  // Not constant.
                              GlobalValue::ExternalLinkage, nullptr, Name);
  }

  std::string Error;
  LLVMContext Context;
  Module *M;  // Owned by ExecutionEngine.
  std::unique_ptr<ExecutionEngine> Engine;
};

TEST_F(ExecutionEngineTest, ForwardGlobalMapping) {
  GlobalVariable *G1 = NewExtGlobal(Type::getInt32Ty(Context), "Global1");
  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(&Mem1, Engine->getPointerToGlobalIfAvailable(G1));
  EXPECT_EQ(&Mem1, Engine->getPointerToGlobalIfAvailable("Global1"));
  int32_t Mem2 = 4;
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1));
  Engine->updateGlobalMapping(G1, nullptr);
  EXPECT_EQ(nullptr, Engine->getPointerToGlobalIfAvailable(G1));
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1));

  GlobalVariable *G2 = NewExtGlobal(Type::getInt32Ty(Context), "Global1");
  EXPECT_EQ(nullptr, Engine->getPointerToGlobalIfAvailable(G2))
    << "The NULL return shouldn't depend on having called"
    << " updateGlobalMapping(..., NULL)";
  // Check that update...() can be called before add...().
  Engine->updateGlobalMapping(G2, &Mem1);
  EXPECT_EQ(&Mem1, Engine->getPointerToGlobalIfAvailable(G2));
  EXPECT_EQ(&Mem2, Engine->getPointerToGlobalIfAvailable(G1))
    << "A second mapping shouldn't affect the first.";
}

TEST_F(ExecutionEngineTest, ReverseGlobalMapping) {
  GlobalVariable *G1 = NewExtGlobal(Type::getInt32Ty(Context), "Global1");

  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));
  int32_t Mem2 = 4;
  Engine->updateGlobalMapping(G1, &Mem2);
  EXPECT_EQ(nullptr, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem2));

  GlobalVariable *G2 = NewExtGlobal(Type::getInt32Ty(Context), "Global2");
  Engine->updateGlobalMapping(G2, &Mem1);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem2));
  Engine->updateGlobalMapping(G1, nullptr);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1))
    << "Removing one mapping doesn't affect a different one.";
  EXPECT_EQ(nullptr, Engine->getGlobalValueAtAddress(&Mem2));
  Engine->updateGlobalMapping(G2, &Mem2);
  EXPECT_EQ(nullptr, Engine->getGlobalValueAtAddress(&Mem1));
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem2))
    << "Once a mapping is removed, we can point another GV at the"
    << " now-free address.";
}

TEST_F(ExecutionEngineTest, ClearModuleMappings) {
  GlobalVariable *G1 = NewExtGlobal(Type::getInt32Ty(Context), "Global1");

  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));

  Engine->clearGlobalMappingsFromModule(M);

  EXPECT_EQ(nullptr, Engine->getGlobalValueAtAddress(&Mem1));

  GlobalVariable *G2 = NewExtGlobal(Type::getInt32Ty(Context), "Global2");
  // After clearing the module mappings, we can assign a new GV to the
  // same address.
  Engine->addGlobalMapping(G2, &Mem1);
  EXPECT_EQ(G2, Engine->getGlobalValueAtAddress(&Mem1));
}

TEST_F(ExecutionEngineTest, DestructionRemovesGlobalMapping) {
  GlobalVariable *G1 = NewExtGlobal(Type::getInt32Ty(Context), "Global1");
  int32_t Mem1 = 3;
  Engine->addGlobalMapping(G1, &Mem1);
  // Make sure the reverse mapping is enabled.
  EXPECT_EQ(G1, Engine->getGlobalValueAtAddress(&Mem1));
  // When the GV goes away, the ExecutionEngine should remove any
  // mappings that refer to it.
  G1->eraseFromParent();
  EXPECT_EQ(nullptr, Engine->getGlobalValueAtAddress(&Mem1));
}

TEST_F(ExecutionEngineTest, LookupWithMangledAndDemangledSymbol) {
  int x;
  int _x;
  llvm::sys::DynamicLibrary::AddSymbol("x", &x);
  llvm::sys::DynamicLibrary::AddSymbol("_x", &_x);

  // RTDyldMemoryManager::getSymbolAddressInProcess expects a mangled symbol,
  // but DynamicLibrary is a wrapper for dlsym, which expects the unmangled C
  // symbol name. This test verifies that getSymbolAddressInProcess strips the
  // leading '_' on Darwin, but not on other platforms.
#ifdef __APPLE__
  EXPECT_EQ(reinterpret_cast<uint64_t>(&x),
            RTDyldMemoryManager::getSymbolAddressInProcess("_x"));
#else
  EXPECT_EQ(reinterpret_cast<uint64_t>(&_x),
            RTDyldMemoryManager::getSymbolAddressInProcess("_x"));
#endif
}

}
