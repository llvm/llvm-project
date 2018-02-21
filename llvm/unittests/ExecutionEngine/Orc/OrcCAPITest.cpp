//===--------------- OrcCAPITest.cpp - Unit tests Orc C API ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm-c/Core.h"
#include "llvm-c/OrcBindings.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"
#include "gtest/gtest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace llvm {

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

class OrcCAPIExecutionTest : public testing::Test, public OrcExecutionTest {
protected:
  std::unique_ptr<Module> createTestModule(const Triple &TT) {
    ModuleBuilder MB(Context, TT.str(), "");
    Function *TestFunc = MB.createFunctionDecl<int()>("testFunc");
    Function *Main = MB.createFunctionDecl<int(int, char*[])>("main");

    Main->getBasicBlockList().push_back(BasicBlock::Create(Context));
    IRBuilder<> B(&Main->back());
    Value* Result = B.CreateCall(TestFunc);
    B.CreateRet(Result);

    return MB.takeModule();
  }

  std::unique_ptr<MemoryBuffer> createTestObject() {
    orc::SimpleCompiler IRCompiler(*TM);
    auto M = createTestModule(TM->getTargetTriple());
    M->setDataLayout(TM->createDataLayout());
    return IRCompiler(*M);
  }

  typedef int (*MainFnTy)();

  static int myTestFuncImpl() {
    return 42;
  }

  static char *testFuncName;

  static uint64_t myResolver(const char *Name, void *Ctx) {
    if (!strncmp(Name, testFuncName, 8))
      return (uint64_t)&myTestFuncImpl;
    return 0;
  }

  struct CompileContext {
    CompileContext() : Compiled(false) { }

    OrcCAPIExecutionTest* APIExecTest;
    std::unique_ptr<Module> M;
    LLVMOrcModuleHandle H;
    bool Compiled;
  };

  static LLVMOrcTargetAddress myCompileCallback(LLVMOrcJITStackRef JITStack,
                                                void *Ctx) {
    CompileContext *CCtx = static_cast<CompileContext*>(Ctx);
    auto *ET = CCtx->APIExecTest;
    CCtx->M = ET->createTestModule(ET->TM->getTargetTriple());
    LLVMSharedModuleRef SM = LLVMOrcMakeSharedModule(wrap(CCtx->M.release()));
    LLVMOrcAddEagerlyCompiledIR(JITStack, &CCtx->H, SM, myResolver, nullptr);
    LLVMOrcDisposeSharedModuleRef(SM);
    CCtx->Compiled = true;
    LLVMOrcTargetAddress MainAddr;
    LLVMOrcGetSymbolAddress(JITStack, &MainAddr, "main");
    LLVMOrcSetIndirectStubPointer(JITStack, "foo", MainAddr);
    return MainAddr;
  }
};

char *OrcCAPIExecutionTest::testFuncName = nullptr;

TEST_F(OrcCAPIExecutionTest, TestEagerIRCompilation) {
  if (!TM)
    return;

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()));

  std::unique_ptr<Module> M = createTestModule(TM->getTargetTriple());

  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");

  LLVMSharedModuleRef SM = LLVMOrcMakeSharedModule(wrap(M.release()));
  LLVMOrcModuleHandle H;
  LLVMOrcAddEagerlyCompiledIR(JIT, &H, SM, myResolver, nullptr);
  LLVMOrcDisposeSharedModuleRef(SM);
  LLVMOrcTargetAddress MainAddr;
  LLVMOrcGetSymbolAddress(JIT, &MainAddr, "main");
  MainFnTy MainFn = (MainFnTy)MainAddr;
  int Result = MainFn();
  EXPECT_EQ(Result, 42)
    << "Eagerly JIT'd code did not return expected result";

  LLVMOrcRemoveModule(JIT, H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

TEST_F(OrcCAPIExecutionTest, TestLazyIRCompilation) {
  if (!TM)
    return;

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()));

  std::unique_ptr<Module> M = createTestModule(TM->getTargetTriple());

  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");

  LLVMSharedModuleRef SM = LLVMOrcMakeSharedModule(wrap(M.release()));
  LLVMOrcModuleHandle H;
  LLVMOrcAddLazilyCompiledIR(JIT, &H, SM, myResolver, nullptr);
  LLVMOrcDisposeSharedModuleRef(SM);
  LLVMOrcTargetAddress MainAddr;
  LLVMOrcGetSymbolAddress(JIT, &MainAddr, "main");
  MainFnTy MainFn = (MainFnTy)MainAddr;
  int Result = MainFn();
  EXPECT_EQ(Result, 42)
    << "Lazily JIT'd code did not return expected result";

  LLVMOrcRemoveModule(JIT, H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

TEST_F(OrcCAPIExecutionTest, TestAddObjectFile) {
  if (!TM)
    return;

  auto ObjBuffer = createTestObject();

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()));
  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");

  LLVMOrcModuleHandle H;
  LLVMOrcAddObjectFile(JIT, &H, wrap(ObjBuffer.release()), myResolver, nullptr);
  LLVMOrcTargetAddress MainAddr;
  LLVMOrcGetSymbolAddress(JIT, &MainAddr, "main");
  MainFnTy MainFn = (MainFnTy)MainAddr;
  int Result = MainFn();
  EXPECT_EQ(Result, 42)
    << "Lazily JIT'd code did not return expected result";

  LLVMOrcRemoveModule(JIT, H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

TEST_F(OrcCAPIExecutionTest, TestDirectCallbacksAPI) {
  if (!TM)
    return;

  LLVMOrcJITStackRef JIT =
    LLVMOrcCreateInstance(wrap(TM.get()));

  LLVMOrcGetMangledSymbol(JIT, &testFuncName, "testFunc");

  CompileContext C;
  C.APIExecTest = this;
  LLVMOrcTargetAddress CCAddr;
  LLVMOrcCreateLazyCompileCallback(JIT, &CCAddr, myCompileCallback, &C);
  LLVMOrcCreateIndirectStub(JIT, "foo", CCAddr);
  LLVMOrcTargetAddress MainAddr;
  LLVMOrcGetSymbolAddress(JIT, &MainAddr, "foo");
  MainFnTy FooFn = (MainFnTy)MainAddr;
  int Result = FooFn();
  EXPECT_TRUE(C.Compiled)
    << "Function wasn't lazily compiled";
  EXPECT_EQ(Result, 42)
    << "Direct-callback JIT'd code did not return expected result";

  C.Compiled = false;
  FooFn();
  EXPECT_FALSE(C.Compiled)
    << "Direct-callback JIT'd code was JIT'd twice";

  LLVMOrcRemoveModule(JIT, C.H);

  LLVMOrcDisposeMangledSymbol(testFuncName);
  LLVMOrcDisposeInstance(JIT);
}

} // namespace llvm
