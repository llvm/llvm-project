//===- MCJITTest.cpp - Unit tests for the MCJIT ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This test suite verifies basic MCJIT functionality such as making function
// calls, using global variables, and compiling multpile modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/MCJIT.h"
#include "MCJITTestBase.h"
#include "SectionMemoryManager.h"
#include "gtest/gtest.h"

using namespace llvm;

class MCJITTest : public testing::Test, public MCJITTestBase {
protected:

  virtual void SetUp() {
    M.reset(createEmptyModule("<main>"));
  }
};

namespace {

// FIXME: In order to JIT an empty module, there needs to be
// an interface to ExecutionEngine that forces compilation but
// does require retrieval of a pointer to a function/global.
/*
TEST_F(MCJITTest, empty_module) {
  createJIT(M.take());
  //EXPECT_NE(0, TheJIT->getObjectImage())
  //  << "Unable to generate executable loaded object image";
}
*/

TEST_F(MCJITTest, global_variable) {
  SKIP_UNSUPPORTED_PLATFORM;

  int initialValue = 5;
  GlobalValue *Global = insertGlobalInt32(M.get(), "test_global", initialValue);
  createJIT(M.take());
  void *globalPtr =  TheJIT->getPointerToGlobal(Global);
  static_cast<SectionMemoryManager*>(MM)->invalidateInstructionCache();
  EXPECT_TRUE(0 != globalPtr)
    << "Unable to get pointer to global value from JIT";

  EXPECT_EQ(initialValue, *(int32_t*)globalPtr)
    << "Unexpected initial value of global";
}

TEST_F(MCJITTest, add_function) {
  SKIP_UNSUPPORTED_PLATFORM;

  Function *F = insertAddFunction(M.get());
  createJIT(M.take());
  void *addPtr = TheJIT->getPointerToFunction(F);
  static_cast<SectionMemoryManager*>(MM)->invalidateInstructionCache();
  EXPECT_TRUE(0 != addPtr)
    << "Unable to get pointer to function from JIT";

  int (*AddPtrTy)(int, int) = (int(*)(int, int))(intptr_t)addPtr;
  EXPECT_EQ(0, AddPtrTy(0, 0));
  EXPECT_EQ(3, AddPtrTy(1, 2));
  EXPECT_EQ(-5, AddPtrTy(-2, -3));
}

TEST_F(MCJITTest, run_main) {
  SKIP_UNSUPPORTED_PLATFORM;

  int rc = 6;
  Function *Main = insertMainFunction(M.get(), 6);
  createJIT(M.take());
  void *vPtr = TheJIT->getPointerToFunction(Main);
  static_cast<SectionMemoryManager*>(MM)->invalidateInstructionCache();
  EXPECT_TRUE(0 != vPtr)
    << "Unable to get pointer to main() from JIT";

  int (*FuncPtr)(void) = (int(*)(void))(intptr_t)vPtr;
  int returnCode = FuncPtr();
  EXPECT_EQ(returnCode, rc);
}

TEST_F(MCJITTest, return_global) {
  SKIP_UNSUPPORTED_PLATFORM;

  int32_t initialNum = 7;
  GlobalVariable *GV = insertGlobalInt32(M.get(), "myglob", initialNum);

  Function *ReturnGlobal = startFunction<int32_t(void)>(M.get(),
                                                        "ReturnGlobal");
  Value *ReadGlobal = Builder.CreateLoad(GV);
  endFunctionWithRet(ReturnGlobal, ReadGlobal);

  createJIT(M.take());
  void *rgvPtr = TheJIT->getPointerToFunction(ReturnGlobal);
  static_cast<SectionMemoryManager*>(MM)->invalidateInstructionCache();
  EXPECT_TRUE(0 != rgvPtr);

  int32_t(*FuncPtr)(void) = (int32_t(*)(void))(intptr_t)rgvPtr;
  EXPECT_EQ(initialNum, FuncPtr())
    << "Invalid value for global returned from JITted function";
}

// FIXME: This case fails due to a bug with getPointerToGlobal().
// The bug is due to MCJIT not having an implementation of getPointerToGlobal()
// which results in falling back on the ExecutionEngine implementation that
// allocates a new memory block for the global instead of using the same
// global variable that is emitted by MCJIT. Hence, the pointer (gvPtr below)
// has the correct initial value, but updates to the real global (accessed by
// JITted code) are not propagated. Instead, getPointerToGlobal() should return
// a pointer into the loaded ObjectImage to reference the emitted global.
/*
TEST_F(MCJITTest, increment_global) {
  SKIP_UNSUPPORTED_PLATFORM;

  int32_t initialNum = 5;
  Function *IncrementGlobal = startFunction<int32_t(void)>(M.get(), "IncrementGlobal");
  GlobalVariable *GV = insertGlobalInt32(M.get(), "my_global", initialNum);
  Value *DerefGV = Builder.CreateLoad(GV);
  Value *AddResult = Builder.CreateAdd(DerefGV,
                                       ConstantInt::get(Context, APInt(32, 1)));
  Builder.CreateStore(AddResult, GV);
  endFunctionWithRet(IncrementGlobal, AddResult);

  createJIT(M.take());
  void *gvPtr = TheJIT->getPointerToGlobal(GV);
  EXPECT_EQ(initialNum, *(int32_t*)gvPtr);

  void *vPtr = TheJIT->getPointerToFunction(IncrementGlobal);
  EXPECT_TRUE(0 != vPtr)
    << "Unable to get pointer to main() from JIT";

  int32_t(*FuncPtr)(void) = (int32_t(*)(void))(intptr_t)vPtr;

  for(int i = 1; i < 3; ++i) {
    int32_t result = FuncPtr();
    EXPECT_EQ(initialNum + i, result);            // OK
    EXPECT_EQ(initialNum + i, *(int32_t*)gvPtr);  // FAILS
  }
}
*/

TEST_F(MCJITTest, multiple_functions) {
  SKIP_UNSUPPORTED_PLATFORM;

  unsigned int numLevels = 23;
  int32_t innerRetVal= 5;

  Function *Inner = startFunction<int32_t(void)>(M.get(), "Inner");
  endFunctionWithRet(Inner, ConstantInt::get(Context, APInt(32, innerRetVal)));

  Function *Outer;
  for (unsigned int i = 0; i < numLevels; ++i) {
    std::stringstream funcName;
    funcName << "level_" << i;
    Outer = startFunction<int32_t(void)>(M.get(), funcName.str());
    Value *innerResult = Builder.CreateCall(Inner);
    endFunctionWithRet(Outer, innerResult);

    Inner = Outer;
  }

  createJIT(M.take());
  void *vPtr = TheJIT->getPointerToFunction(Outer);
  static_cast<SectionMemoryManager*>(MM)->invalidateInstructionCache();
  EXPECT_TRUE(0 != vPtr)
    << "Unable to get pointer to outer function from JIT";

  int32_t(*FuncPtr)(void) = (int32_t(*)(void))(intptr_t)vPtr;
  EXPECT_EQ(innerRetVal, FuncPtr())
    << "Incorrect result returned from function";
}

// FIXME: ExecutionEngine has no support empty modules
/*
TEST_F(MCJITTest, multiple_empty_modules) {
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

// FIXME: MCJIT must support multiple modules
/*
TEST_F(MCJITTest, multiple_modules) {
  SKIP_UNSUPPORTED_PLATFORM;

  Function *Callee = insertAddFunction(M.get());
  createJIT(M.take());

  // caller function is defined in a different module
  M.reset(createEmptyModule("<caller module>"));

  Function *CalleeRef = insertExternalReferenceToFunction(M.get(), Callee);
  Function *Caller = insertSimpleCallFunction(M.get(), CalleeRef);

  TheJIT->addModule(M.take());

  // get a function pointer in a module that was not used in EE construction
  void *vPtr = TheJIT->getPointerToFunction(Caller);
  EXPECT_NE(0, vPtr)
    << "Unable to get pointer to caller function from JIT";

  int(*FuncPtr)(int, int) = (int(*)(int, int))(intptr_t)vPtr;
  EXPECT_EQ(0, FuncPtr(0, 0));
  EXPECT_EQ(30, FuncPtr(10, 20));
  EXPECT_EQ(-30, FuncPtr(-10, -20));

  // ensure caller is destroyed before callee (free use before def)
  M.reset();
}
*/

}
