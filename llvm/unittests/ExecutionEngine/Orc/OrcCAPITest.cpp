//===--- OrcCAPITest.cpp - Unit tests for the OrcJIT v2 C API ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/LLJIT.h"
#include "llvm-c/Orc.h"
#include "gtest/gtest.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// OrcCAPITestBase contains several helper methods and pointers for unit tests
// written for the LLVM-C API. It provides the following helpers:
//
// 1. Jit: an LLVMOrcLLJIT instance which is freed upon test exit
// 2. ExecutionSession: the LLVMOrcExecutionSession for the JIT
// 3. MainDylib: the main JITDylib for the LLJIT instance
// 4. materializationUnitFn: function pointer to an empty function, used for
//                           materialization unit testing
// 5. definitionGeneratorFn: function pointer for a basic
//                           LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction
// 6. createTestModule: helper method for creating a basic thread-safe-module
class OrcCAPITestBase : public testing::Test {
protected:
  LLVMOrcLLJITRef Jit = nullptr;
  LLVMOrcExecutionSessionRef ExecutionSession = nullptr;
  LLVMOrcJITDylibRef MainDylib = nullptr;

public:
  static void SetUpTestCase() {
    LLVMInitializeNativeTarget();

    // Attempt to set up a JIT instance once to verify that we can.
    LLVMOrcJITTargetMachineBuilderRef JTMB = nullptr;
    if (LLVMErrorRef E = LLVMOrcJITTargetMachineBuilderDetectHost(&JTMB)) {
      // If setup fails then disable these tests.
      LLVMConsumeError(E);
      TargetSupported = false;
      return;
    }

    char *Triple = LLVMOrcJITTargetMachineBuilderGetTargetTriple(JTMB);
    if (!isSupported(Triple)) {
      // If this triple isn't supported then bail out.
      TargetSupported = false;
      LLVMOrcJITTargetMachineBuilderDisposeTargetTriple(JTMB, Triple);
      LLVMOrcDisposeJITTargetMachineBuilder(JTMB);
      return;
    }

    LLVMOrcJITTargetMachineBuilderDisposeTargetTriple(JTMB, Triple);
    LLVMOrcLLJITBuilderRef Builder = LLVMOrcCreateLLJITBuilder();
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(Builder, JTMB);
    LLVMOrcLLJITRef J;
    if (LLVMErrorRef E = LLVMOrcCreateLLJIT(&J, Builder)) {
      // If setup fails then disable these tests.
      TargetSupported = false;
      LLVMConsumeError(E);
      return;
    }

    LLVMOrcDisposeLLJIT(J);
    TargetSupported = true;
  }

  void SetUp() override {
    if (!TargetSupported)
      return;

    LLVMOrcJITTargetMachineBuilderRef JTMB = nullptr;
    LLVMErrorRef E1 = LLVMOrcJITTargetMachineBuilderDetectHost(&JTMB);
    assert(E1 == LLVMErrorSuccess && "Expected call to detect host to succeed");

    LLVMOrcLLJITBuilderRef Builder = LLVMOrcCreateLLJITBuilder();
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(Builder, JTMB);
    LLVMErrorRef E2 = LLVMOrcCreateLLJIT(&Jit, Builder);
    assert(E2 == LLVMErrorSuccess &&
           "Expected call to create LLJIT to succeed");
    ExecutionSession = LLVMOrcLLJITGetExecutionSession(Jit);
    MainDylib = LLVMOrcLLJITGetMainJITDylib(Jit);
  }
  void TearDown() override {
    LLVMOrcDisposeLLJIT(Jit);
    Jit = nullptr;
  }

protected:
  static bool isSupported(StringRef Triple) {
    if (Triple.startswith("armv7"))
      return false;
    return true;
  }

  static void materializationUnitFn() {}
  // Stub definition generator, where all Names are materialized from the
  // materializationUnitFn() test function and defined into the JIT Dylib
  static LLVMErrorRef
  definitionGeneratorFn(LLVMOrcDefinitionGeneratorRef G, void *Ctx,
                        LLVMOrcLookupStateRef *LS, LLVMOrcLookupKind K,
                        LLVMOrcJITDylibRef JD, LLVMOrcJITDylibLookupFlags F,
                        LLVMOrcCLookupSet Names, size_t NamesCount) {
    for (size_t I = 0; I < NamesCount; I++) {
      LLVMOrcCLookupSetElement Element = Names[I];
      LLVMOrcJITTargetAddress Addr =
          (LLVMOrcJITTargetAddress)(&materializationUnitFn);
      LLVMJITSymbolFlags Flags = {LLVMJITSymbolGenericFlagsWeak, 0};
      LLVMJITEvaluatedSymbol Sym = {Addr, Flags};
      LLVMOrcRetainSymbolStringPoolEntry(Element.Name);
      LLVMJITCSymbolMapPair Pair = {Element.Name, Sym};
      LLVMJITCSymbolMapPair Pairs[] = {Pair};
      LLVMOrcMaterializationUnitRef MU = LLVMOrcAbsoluteSymbols(Pairs, 1);
      LLVMErrorRef Err = LLVMOrcJITDylibDefine(JD, MU);
      if (Err)
        return Err;
    }
    return LLVMErrorSuccess;
  }
  // create a test LLVM IR module containing a function named "sum" which has
  // returns the sum of its two parameters
  static LLVMOrcThreadSafeModuleRef createTestModule() {
    LLVMOrcThreadSafeContextRef TSC = LLVMOrcCreateNewThreadSafeContext();
    LLVMContextRef Ctx = LLVMOrcThreadSafeContextGetContext(TSC);
    LLVMModuleRef Mod = LLVMModuleCreateWithNameInContext("test", Ctx);
    {
      LLVMTypeRef Int32Ty = LLVMInt32TypeInContext(Ctx);
      LLVMTypeRef ParamTys[] = {Int32Ty, Int32Ty};
      LLVMTypeRef TestFnTy = LLVMFunctionType(Int32Ty, ParamTys, 2, 0);
      LLVMValueRef TestFn = LLVMAddFunction(Mod, "sum", TestFnTy);
      LLVMBuilderRef IRBuilder = LLVMCreateBuilderInContext(Ctx);
      LLVMBasicBlockRef EntryBB = LLVMAppendBasicBlock(TestFn, "entry");
      LLVMPositionBuilderAtEnd(IRBuilder, EntryBB);
      LLVMValueRef Arg1 = LLVMGetParam(TestFn, 0);
      LLVMValueRef Arg2 = LLVMGetParam(TestFn, 1);
      LLVMValueRef Sum = LLVMBuildAdd(IRBuilder, Arg1, Arg2, "");
      LLVMBuildRet(IRBuilder, Sum);
      LLVMDisposeBuilder(IRBuilder);
    }
    LLVMOrcThreadSafeModuleRef TSM = LLVMOrcCreateNewThreadSafeModule(Mod, TSC);
    LLVMOrcDisposeThreadSafeContext(TSC);
    return TSM;
  }

  static bool TargetSupported;
};

bool OrcCAPITestBase::TargetSupported = false;

TEST_F(OrcCAPITestBase, SymbolStringPoolUniquing) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  LLVMOrcSymbolStringPoolEntryRef E1 =
      LLVMOrcExecutionSessionIntern(ExecutionSession, "aaa");
  LLVMOrcSymbolStringPoolEntryRef E2 =
      LLVMOrcExecutionSessionIntern(ExecutionSession, "aaa");
  LLVMOrcSymbolStringPoolEntryRef E3 =
      LLVMOrcExecutionSessionIntern(ExecutionSession, "bbb");
  const char *SymbolName = LLVMOrcSymbolStringPoolEntryStr(E1);
  ASSERT_EQ(E1, E2) << "String pool entries are not unique";
  ASSERT_NE(E1, E3) << "Unique symbol pool entries are equal";
  ASSERT_STREQ("aaa", SymbolName) << "String value of symbol is not equal";
  LLVMOrcReleaseSymbolStringPoolEntry(E1);
  LLVMOrcReleaseSymbolStringPoolEntry(E2);
  LLVMOrcReleaseSymbolStringPoolEntry(E3);
}

TEST_F(OrcCAPITestBase, JITDylibLookup) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }
  LLVMOrcJITDylibRef DoesNotExist =
      LLVMOrcExecutionSessionGetJITDylibByName(ExecutionSession, "test");
  ASSERT_FALSE(!!DoesNotExist);
  LLVMOrcJITDylibRef L1 =
      LLVMOrcExecutionSessionCreateBareJITDylib(ExecutionSession, "test");
  LLVMOrcJITDylibRef L2 =
      LLVMOrcExecutionSessionGetJITDylibByName(ExecutionSession, "test");
  ASSERT_EQ(L1, L2) << "Located JIT Dylib is not equal to original";
}

TEST_F(OrcCAPITestBase, MaterializationUnitCreation) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  LLVMOrcSymbolStringPoolEntryRef Name =
      LLVMOrcLLJITMangleAndIntern(Jit, "test");
  LLVMJITSymbolFlags Flags = {LLVMJITSymbolGenericFlagsWeak, 0};
  LLVMOrcJITTargetAddress Addr =
      (LLVMOrcJITTargetAddress)(&materializationUnitFn);
  LLVMJITEvaluatedSymbol Sym = {Addr, Flags};
  LLVMJITCSymbolMapPair Pair = {Name, Sym};
  LLVMJITCSymbolMapPair Pairs[] = {Pair};
  LLVMOrcMaterializationUnitRef MU = LLVMOrcAbsoluteSymbols(Pairs, 1);
  LLVMOrcJITDylibDefine(MainDylib, MU);
  LLVMOrcJITTargetAddress OutAddr;
  if (LLVMOrcLLJITLookup(Jit, &OutAddr, "test")) {
    FAIL() << "Failed to look up \"test\" symbol";
  }
  ASSERT_EQ(Addr, OutAddr);
}

TEST_F(OrcCAPITestBase, DefinitionGenerators) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  LLVMOrcDefinitionGeneratorRef Gen =
      LLVMOrcCreateCustomCAPIDefinitionGenerator(&definitionGeneratorFn,
                                                 nullptr);
  LLVMOrcJITDylibAddGenerator(MainDylib, Gen);
  LLVMOrcJITTargetAddress OutAddr;
  if (LLVMOrcLLJITLookup(Jit, &OutAddr, "test")) {
    FAIL() << "The DefinitionGenerator did not create symbol \"test\"";
  }
  LLVMOrcJITTargetAddress ExpectedAddr =
      (LLVMOrcJITTargetAddress)(&materializationUnitFn);
  ASSERT_EQ(ExpectedAddr, OutAddr);
}

TEST_F(OrcCAPITestBase, ResourceTrackerDefinitionLifetime) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  // This test case ensures that all symbols loaded into a JITDylib with a
  // ResourceTracker attached are cleared from the JITDylib once the RT is
  // removed.
  LLVMOrcResourceTrackerRef RT =
      LLVMOrcJITDylibCreateResourceTracker(MainDylib);
  LLVMOrcThreadSafeModuleRef TSM = createTestModule();
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModuleWithRT(Jit, RT, TSM)) {
    FAIL() << "Failed to add LLVM IR module to LLJIT";
  }
  LLVMOrcJITTargetAddress TestFnAddr;
  if (LLVMOrcLLJITLookup(Jit, &TestFnAddr, "sum")) {
    FAIL() << "Symbol \"sum\" was not added into JIT";
  }
  ASSERT_TRUE(!!TestFnAddr);
  LLVMOrcResourceTrackerRemove(RT);
  LLVMOrcJITTargetAddress OutAddr;
  LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &OutAddr, "sum");
  ASSERT_TRUE(Err);
  ASSERT_FALSE(OutAddr);
  LLVMOrcReleaseResourceTracker(RT);
  LLVMConsumeError(Err);
}

TEST_F(OrcCAPITestBase, ResourceTrackerTransfer) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  LLVMOrcResourceTrackerRef DefaultRT =
      LLVMOrcJITDylibGetDefaultResourceTracker(MainDylib);
  LLVMOrcResourceTrackerRef RT2 =
      LLVMOrcJITDylibCreateResourceTracker(MainDylib);
  LLVMOrcThreadSafeModuleRef TSM = createTestModule();
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModuleWithRT(Jit, DefaultRT, TSM)) {
    FAIL() << "Failed to add LLVM IR module to LLJIT";
  }
  LLVMOrcJITTargetAddress Addr;
  if (LLVMOrcLLJITLookup(Jit, &Addr, "sum")) {
    FAIL() << "Symbol \"sum\" was not added into JIT";
  }
  LLVMOrcResourceTrackerTransferTo(DefaultRT, RT2);
  LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &Addr, "sum");
  ASSERT_FALSE(Err);
  LLVMOrcReleaseResourceTracker(RT2);
}

TEST_F(OrcCAPITestBase, ExecutionTest) {
  if (!Jit) {
    // TODO: Use GTEST_SKIP() when GTest is updated to version 1.10.0
    return;
  }

  using SumFunctionType = int32_t (*)(int32_t, int32_t);

  // This test performs OrcJIT compilation of a simple sum module
  LLVMInitializeNativeAsmPrinter();
  LLVMOrcThreadSafeModuleRef TSM = createTestModule();
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModule(Jit, MainDylib, TSM)) {
    FAIL() << "Failed to add LLVM IR module to LLJIT";
  }
  LLVMOrcJITTargetAddress TestFnAddr;
  if (LLVMOrcLLJITLookup(Jit, &TestFnAddr, "sum")) {
    FAIL() << "Symbol \"sum\" was not added into JIT";
  }
  auto *SumFn = (SumFunctionType)(TestFnAddr);
  int32_t Result = SumFn(1, 1);
  ASSERT_EQ(2, Result);
}
