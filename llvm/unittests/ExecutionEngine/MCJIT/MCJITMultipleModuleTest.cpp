//===- MCJITMultipeModuleTest.cpp - Unit tests for the MCJIT ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test suite verifies MCJIT for handling multiple modules in a single
// ExecutionEngine by building multiple modules, making function calls across
// modules, accessing global variables, etc.
//===----------------------------------------------------------------------===//

#include "MCJITTestBase.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MCJITMultipleModuleTest : public testing::Test, public MCJITTestBase {};

// FIXME: ExecutionEngine has no support empty modules
/*
TEST_F(MCJITMultipleModuleTest, multiple_empty_modules) {
  SKIP_UNSUPPORTED_PLATFORM;

  createJIT(M.take());
  // JIT-compile
  EXPECT_NE(0, TheJIT->getObjectImage())
    << "Unable to generate executable loaded object image";

  TheJIT->addModule(createEmptyModule("<other module>"));
  TheJIT->addModule(createEmptyModule("<other other module>"));

  // JIT again
  EXPECT_NE(0, TheJIT->getObjectImage())
    << "Unable to generate executable loaded object image";
}
*/

// Helper Function to test add operation
void checkAdd(uint64_t ptr) {
  ASSERT_TRUE(ptr != 0) << "Unable to get pointer to function.";
  int (*AddPtr)(int, int) = (int (*)(int, int))ptr;
  EXPECT_EQ(0, AddPtr(0, 0));
  EXPECT_EQ(1, AddPtr(1, 0));
  EXPECT_EQ(3, AddPtr(1, 2));
  EXPECT_EQ(-5, AddPtr(-2, -3));
  EXPECT_EQ(30, AddPtr(10, 20));
  EXPECT_EQ(-30, AddPtr(-10, -20));
  EXPECT_EQ(-40, AddPtr(-10, -30));
}

void checkAccumulate(uint64_t ptr) {
  ASSERT_TRUE(ptr != 0) << "Unable to get pointer to function.";
  int32_t (*FPtr)(int32_t) = (int32_t (*)(int32_t))(intptr_t)ptr;
  EXPECT_EQ(0, FPtr(0));
  EXPECT_EQ(1, FPtr(1));
  EXPECT_EQ(3, FPtr(2));
  EXPECT_EQ(6, FPtr(3));
  EXPECT_EQ(10, FPtr(4));
  EXPECT_EQ(15, FPtr(5));
}

// FIXME: ExecutionEngine has no support empty modules
/*
TEST_F(MCJITMultipleModuleTest, multiple_empty_modules) {
  SKIP_UNSUPPORTED_PLATFORM;

  createJIT(M.take());
  // JIT-compile
  EXPECT_NE(0, TheJIT->getObjectImage())
    << "Unable to generate executable loaded object image";

  TheJIT->addModule(createEmptyModule("<other module>"));
  TheJIT->addModule(createEmptyModule("<other other module>"));

  // JIT again
  EXPECT_NE(0, TheJIT->getObjectImage())
    << "Unable to generate executable loaded object image";
}
*/

// Module A { Function FA },
// Module B { Function FB },
// execute FA then FB
TEST_F(MCJITMultipleModuleTest, two_module_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB;
  createTwoModuleCase(A, FA, B, FB);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Function FB },
// execute FB then FA
TEST_F(MCJITMultipleModuleTest, two_module_reverse_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB;
  createTwoModuleCase(A, FA, B, FB);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FB->getName().str());
  TheJIT->finalizeObject();
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// execute FB then FA
TEST_F(MCJITMultipleModuleTest, two_module_extern_reverse_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB;
  createTwoModuleExternCase(A, FA, B, FB);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FB->getName().str());
  TheJIT->finalizeObject();
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// execute FA then FB
TEST_F(MCJITMultipleModuleTest, two_module_extern_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB;
  createTwoModuleExternCase(A, FA, B, FB);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA1, Function FA2 which calls FA1 },
// Module B { Extern FA1, Function FB which calls FA1 },
// execute FB then FA2
TEST_F(MCJITMultipleModuleTest, two_module_consecutive_call_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA1, *FA2, *FB;
  createTwoModuleExternCase(A, FA1, B, FB);
  FA2 = insertSimpleCallFunction(A.get(), FA1);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FB->getName().str());
  TheJIT->finalizeObject();
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FA2->getName().str());
  checkAdd(ptr);
}

// TODO:
// Module A { Extern Global GVB, Global Variable GVA, Function FA loads GVB },
// Module B { Extern Global GVA, Global Variable GVB, Function FB loads GVA },


// Module A { Global Variable GVA, Function FA loads GVA },
// Module B { Global Variable GVB, Internal Global GVC, Function FB loads GVB },
// execute FB then FA, also check that the global variables are properly accesible
// through the ExecutionEngine APIs
TEST_F(MCJITMultipleModuleTest, two_module_global_variables_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB;
  GlobalVariable *GVA, *GVB, *GVC;

  A.reset(createEmptyModule("A"));
  B.reset(createEmptyModule("B"));

  int32_t initialNum = 7;
  GVA = insertGlobalInt32(A.get(), "GVA", initialNum);
  GVB = insertGlobalInt32(B.get(), "GVB", initialNum);
  FA = startFunction(A.get(),
                     FunctionType::get(Builder.getInt32Ty(), {}, false), "FA");
  endFunctionWithRet(FA, Builder.CreateLoad(GVA));
  FB = startFunction(B.get(),
                     FunctionType::get(Builder.getInt32Ty(), {}, false), "FB");
  endFunctionWithRet(FB, Builder.CreateLoad(GVB));

  GVC = insertGlobalInt32(B.get(), "GVC", initialNum);
  GVC->setLinkage(GlobalValue::InternalLinkage);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  EXPECT_EQ(GVA, TheJIT->FindGlobalVariableNamed("GVA"));
  EXPECT_EQ(GVB, TheJIT->FindGlobalVariableNamed("GVB"));
  EXPECT_EQ(GVC, TheJIT->FindGlobalVariableNamed("GVC",true));
  EXPECT_EQ(nullptr, TheJIT->FindGlobalVariableNamed("GVC"));

  uint64_t FBPtr = TheJIT->getFunctionAddress(FB->getName().str());
  TheJIT->finalizeObject();
  EXPECT_TRUE(0 != FBPtr);
  int32_t(*FuncPtr)() = (int32_t(*)())FBPtr;
  EXPECT_EQ(initialNum, FuncPtr())
    << "Invalid value for global returned from JITted function in module B";

  uint64_t FAPtr = TheJIT->getFunctionAddress(FA->getName().str());
  EXPECT_TRUE(0 != FAPtr);
  FuncPtr = (int32_t(*)())FAPtr;
  EXPECT_EQ(initialNum, FuncPtr())
    << "Invalid value for global returned from JITted function in module A";
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// Module C { Extern FA, Function FC which calls FA },
// execute FC, FB, FA
TEST_F(MCJITMultipleModuleTest, three_module_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B, C;
  Function *FA, *FB, *FC;
  createThreeModuleCase(A, FA, B, FB, C, FC);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));
  TheJIT->addModule(std::move(C));

  uint64_t ptr = TheJIT->getFunctionAddress(FC->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// Module C { Extern FA, Function FC which calls FA },
// execute FA, FB, FC
TEST_F(MCJITMultipleModuleTest, three_module_case_reverse_order) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B, C;
  Function *FA, *FB, *FC;
  createThreeModuleCase(A, FA, B, FB, C, FC);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));
  TheJIT->addModule(std::move(C));

  uint64_t ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FC->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// Module C { Extern FB, Function FC which calls FB },
// execute FC, FB, FA
TEST_F(MCJITMultipleModuleTest, three_module_chain_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B, C;
  Function *FA, *FB, *FC;
  createThreeModuleChainedCallsCase(A, FA, B, FB, C, FC);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));
  TheJIT->addModule(std::move(C));

  uint64_t ptr = TheJIT->getFunctionAddress(FC->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);
}

// Module A { Function FA },
// Module B { Extern FA, Function FB which calls FA },
// Module C { Extern FB, Function FC which calls FB },
// execute FA, FB, FC
TEST_F(MCJITMultipleModuleTest, three_modules_chain_case_reverse_order) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B, C;
  Function *FA, *FB, *FC;
  createThreeModuleChainedCallsCase(A, FA, B, FB, C, FC);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));
  TheJIT->addModule(std::move(C));

  uint64_t ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FB->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(FC->getName().str());
  checkAdd(ptr);
}

// Module A { Extern FB, Function FA which calls FB1 },
// Module B { Extern FA, Function FB1, Function FB2 which calls FA },
// execute FA, then FB1
// FIXME: this test case is not supported by MCJIT
TEST_F(MCJITMultipleModuleTest, cross_module_dependency_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB1, *FB2;
  createCrossModuleRecursiveCase(A, FA, B, FB1, FB2);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAccumulate(ptr);

  ptr = TheJIT->getFunctionAddress(FB1->getName().str());
  checkAccumulate(ptr);
}

// Module A { Extern FB, Function FA which calls FB1 },
// Module B { Extern FA, Function FB1, Function FB2 which calls FA },
// execute FB1 then FA
// FIXME: this test case is not supported by MCJIT
TEST_F(MCJITMultipleModuleTest, cross_module_dependency_case_reverse_order) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB1, *FB2;
  createCrossModuleRecursiveCase(A, FA, B, FB1, FB2);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FB1->getName().str());
  checkAccumulate(ptr);

  ptr = TheJIT->getFunctionAddress(FA->getName().str());
  checkAccumulate(ptr);
}

// Module A { Extern FB1, Function FA which calls FB1 },
// Module B { Extern FA, Function FB1, Function FB2 which calls FA },
// execute FB1 then FB2
// FIXME: this test case is not supported by MCJIT
TEST_F(MCJITMultipleModuleTest, cross_module_dependency_case3) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB1, *FB2;
  createCrossModuleRecursiveCase(A, FA, B, FB1, FB2);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  uint64_t ptr = TheJIT->getFunctionAddress(FB1->getName().str());
  checkAccumulate(ptr);

  ptr = TheJIT->getFunctionAddress(FB2->getName().str());
  checkAccumulate(ptr);
}

// Test that FindFunctionNamed finds the definition of
// a function in the correct module. We check two functions
// in two different modules, to make sure that for at least
// one of them MCJIT had to ignore the extern declaration.
TEST_F(MCJITMultipleModuleTest, FindFunctionNamed_test) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A, B;
  Function *FA, *FB1, *FB2;
  createCrossModuleRecursiveCase(A, FA, B, FB1, FB2);

  createJIT(std::move(A));
  TheJIT->addModule(std::move(B));

  EXPECT_EQ(FA, TheJIT->FindFunctionNamed(FA->getName().data()));
  EXPECT_EQ(FB1, TheJIT->FindFunctionNamed(FB1->getName().data()));
}

} // end anonymous namespace
