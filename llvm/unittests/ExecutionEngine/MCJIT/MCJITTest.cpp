//===- MCJITTest.cpp - Unit tests for the MCJIT -----------------*- C++ -*-===//
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
#include "llvm/Support/DynamicLibrary.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MCJITTest : public testing::Test, public MCJITTestBase {
protected:
  void SetUp() override { M.reset(createEmptyModule("<main>")); }
};

// FIXME: Ensure creating an execution engine does not crash when constructed
//        with a null module.
/*
TEST_F(MCJITTest, null_module) {
  createJIT(0);
}
*/

// FIXME: In order to JIT an empty module, there needs to be
// an interface to ExecutionEngine that forces compilation but
// does not require retrieval of a pointer to a function/global.
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
  createJIT(std::move(M));
  void *globalPtr =  TheJIT->getPointerToGlobal(Global);
  EXPECT_TRUE(nullptr != globalPtr)
    << "Unable to get pointer to global value from JIT";

  EXPECT_EQ(initialValue, *(int32_t*)globalPtr)
    << "Unexpected initial value of global";
}

TEST_F(MCJITTest, add_function) {
  SKIP_UNSUPPORTED_PLATFORM;

  Function *F = insertAddFunction(M.get());
  createJIT(std::move(M));
  uint64_t addPtr = TheJIT->getFunctionAddress(F->getName().str());
  EXPECT_TRUE(0 != addPtr)
    << "Unable to get pointer to function from JIT";

  ASSERT_TRUE(addPtr != 0) << "Unable to get pointer to function .";
  int (*AddPtr)(int, int) = (int(*)(int, int))addPtr ;
  EXPECT_EQ(0,   AddPtr(0, 0));
  EXPECT_EQ(1,   AddPtr(1, 0));
  EXPECT_EQ(3,   AddPtr(1, 2));
  EXPECT_EQ(-5,  AddPtr(-2, -3));
  EXPECT_EQ(30,  AddPtr(10, 20));
  EXPECT_EQ(-30, AddPtr(-10, -20));
  EXPECT_EQ(-40, AddPtr(-10, -30));
}

TEST_F(MCJITTest, run_main) {
  SKIP_UNSUPPORTED_PLATFORM;

  int rc = 6;
  Function *Main = insertMainFunction(M.get(), 6);
  createJIT(std::move(M));
  uint64_t ptr = TheJIT->getFunctionAddress(Main->getName().str());
  EXPECT_TRUE(0 != ptr)
    << "Unable to get pointer to main() from JIT";

  int (*FuncPtr)() = (int(*)())ptr;
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

  createJIT(std::move(M));
  uint64_t rgvPtr = TheJIT->getFunctionAddress(ReturnGlobal->getName().str());
  EXPECT_TRUE(0 != rgvPtr);

  int32_t(*FuncPtr)() = (int32_t(*)())rgvPtr;
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

  void *vPtr = TheJIT->getFunctionAddress(IncrementGlobal->getName().str());
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

// PR16013: XFAIL this test on ARM, which currently can't handle multiple relocations.
#if !defined(__arm__)

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
    Value *innerResult = Builder.CreateCall(Inner, {});
    endFunctionWithRet(Outer, innerResult);

    Inner = Outer;
  }

  createJIT(std::move(M));
  uint64_t ptr = TheJIT->getFunctionAddress(Outer->getName().str());
  EXPECT_TRUE(0 != ptr)
    << "Unable to get pointer to outer function from JIT";

  int32_t(*FuncPtr)() = (int32_t(*)())ptr;
  EXPECT_EQ(innerRetVal, FuncPtr())
    << "Incorrect result returned from function";
}

#endif /*!defined(__arm__)*/

TEST_F(MCJITTest, multiple_decl_lookups) {
  SKIP_UNSUPPORTED_PLATFORM;

  Function *Foo = insertExternalReferenceToFunction<void(void)>(M.get(), "_exit");
  createJIT(std::move(M));
  void *A = TheJIT->getPointerToFunction(Foo);
  void *B = TheJIT->getPointerToFunction(Foo);

  EXPECT_TRUE(A != nullptr) << "Failed lookup - test not correctly configured.";
  EXPECT_EQ(A, B) << "Repeat calls to getPointerToFunction fail.";
}

typedef void * (*FunctionHandlerPtr)(const std::string &str);

TEST_F(MCJITTest, lazy_function_creator_pointer) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  Function *Foo = insertExternalReferenceToFunction<int32_t(void)>(M.get(),
                                                                   "\1Foo");
  startFunction<int32_t(void)>(M.get(), "Parent");
  CallInst *Call = Builder.CreateCall(Foo, {});
  Builder.CreateRet(Call);
  
  createJIT(std::move(M));
  
  // Set up the lazy function creator that records the name of the last
  // unresolved external function found in the module. Using a function pointer
  // prevents us from capturing local variables, which is why this is static.
  static std::string UnresolvedExternal;
  FunctionHandlerPtr UnresolvedHandler = [] (const std::string &str) {
    // Try to resolve the function in the current process before marking it as
    // unresolved. This solves an issue on ARM where '__aeabi_*' function names
    // are passed to this handler.
    void *symbol =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(str.c_str());
    if (symbol) {
      return symbol;
    }
    
    UnresolvedExternal = str;
    return (void *)(uintptr_t)-1;
  };
  TheJIT->InstallLazyFunctionCreator(UnresolvedHandler);
  
  // JIT the module.
  TheJIT->finalizeObject();
  
  // Verify that our handler was called.
  EXPECT_EQ(UnresolvedExternal, "Foo");
}

TEST_F(MCJITTest, lazy_function_creator_lambda) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  Function *Foo1 = insertExternalReferenceToFunction<int32_t(void)>(M.get(),
                                                                   "\1Foo1");
  Function *Foo2 = insertExternalReferenceToFunction<int32_t(void)>(M.get(),
                                                                   "\1Foo2");
  startFunction<int32_t(void)>(M.get(), "Parent");
  CallInst *Call1 = Builder.CreateCall(Foo1, {});
  CallInst *Call2 = Builder.CreateCall(Foo2, {});
  Value *Result = Builder.CreateAdd(Call1, Call2);
  Builder.CreateRet(Result);
  
  createJIT(std::move(M));
  
  // Set up the lazy function creator that records the name of unresolved
  // external functions in the module.
  std::vector<std::string> UnresolvedExternals;
  auto UnresolvedHandler = [&UnresolvedExternals] (const std::string &str) {
    // Try to resolve the function in the current process before marking it as
    // unresolved. This solves an issue on ARM where '__aeabi_*' function names
    // are passed to this handler.
    void *symbol =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(str.c_str());
    if (symbol) {
      return symbol;
    }
    UnresolvedExternals.push_back(str);
    return (void *)(uintptr_t)-1;
  };
  TheJIT->InstallLazyFunctionCreator(UnresolvedHandler);
  
  // JIT the module.
  TheJIT->finalizeObject();
  
  // Verify that our handler was called for each unresolved function.
  auto I = UnresolvedExternals.begin(), E = UnresolvedExternals.end();
  EXPECT_EQ(UnresolvedExternals.size(), 2u);
  EXPECT_FALSE(std::find(I, E, "Foo1") == E);
  EXPECT_FALSE(std::find(I, E, "Foo2") == E);
}

} // end anonymous namespace
