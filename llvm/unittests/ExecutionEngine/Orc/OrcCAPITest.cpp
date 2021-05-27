//===--- OrcCAPITest.cpp - Unit tests for the OrcJIT v2 C API ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/Error.h"
#include "llvm-c/LLJIT.h"
#include "llvm-c/Orc.h"
#include "gtest/gtest.h"

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

using namespace llvm;
using namespace llvm::orc;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeModule, LLVMOrcThreadSafeModuleRef)

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
    LLVMInitializeNativeAsmParser();
    LLVMInitializeNativeAsmPrinter();

    // Attempt to set up a JIT instance once to verify that we can.
    LLVMOrcJITTargetMachineBuilderRef JTMB = nullptr;
    if (LLVMErrorRef E = LLVMOrcJITTargetMachineBuilderDetectHost(&JTMB)) {
      // If setup fails then disable these tests.
      LLVMConsumeError(E);
      TargetSupported = false;
      return;
    }

    // Capture the target triple. We'll use it for both verification that
    // this target is *supposed* to be supported, and error messages in
    // the case that it fails anyway.
    char *TT = LLVMOrcJITTargetMachineBuilderGetTargetTriple(JTMB);
    TargetTriple = TT;
    LLVMDisposeMessage(TT);

    if (!isSupported(TargetTriple)) {
      // If this triple isn't supported then bail out.
      TargetSupported = false;
      LLVMOrcDisposeJITTargetMachineBuilder(JTMB);
      return;
    }

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
      GTEST_SKIP();

    LLVMOrcJITTargetMachineBuilderRef JTMB = nullptr;
    LLVMErrorRef E1 = LLVMOrcJITTargetMachineBuilderDetectHost(&JTMB);
    assert(E1 == LLVMErrorSuccess && "Expected call to detect host to succeed");
    (void)E1;

    LLVMOrcLLJITBuilderRef Builder = LLVMOrcCreateLLJITBuilder();
    LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(Builder, JTMB);
    LLVMErrorRef E2 = LLVMOrcCreateLLJIT(&Jit, Builder);
    assert(E2 == LLVMErrorSuccess &&
           "Expected call to create LLJIT to succeed");
    (void)E2;
    ExecutionSession = LLVMOrcLLJITGetExecutionSession(Jit);
    MainDylib = LLVMOrcLLJITGetMainJITDylib(Jit);
  }
  void TearDown() override {
    LLVMOrcDisposeLLJIT(Jit);
    Jit = nullptr;
  }

protected:
  static bool isSupported(StringRef Triple) {
    // TODO: Print error messages in failure logs, use them to audit this list.
    // Some architectures may be unsupportable or missing key components, but
    // some may just be failing due to bugs in this testcase.
    if (Triple.startswith("armv7") || Triple.startswith("armv8l"))
      return false;
    llvm::Triple T(Triple);
    if (T.isOSAIX() && T.isPPC64())
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

  static Error createSMDiagnosticError(llvm::SMDiagnostic &Diag) {
    std::string Msg;
    {
      raw_string_ostream OS(Msg);
      Diag.print("", OS);
    }
    return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
  }

  // Create an LLVM IR module from the given StringRef.
  static Expected<std::unique_ptr<Module>>
  parseTestModule(LLVMContext &Ctx, StringRef Source, StringRef Name) {
    assert(TargetSupported &&
           "Attempted to create module for unsupported target");
    SMDiagnostic Err;
    if (auto M = parseIR(MemoryBufferRef(Source, Name), Err, Ctx))
      return std::move(M);
    return createSMDiagnosticError(Err);
  }

  // returns the sum of its two parameters
  static LLVMOrcThreadSafeModuleRef createTestModule(StringRef Source,
                                                     StringRef Name) {
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = cantFail(parseTestModule(*Ctx, Source, Name));
    return wrap(new ThreadSafeModule(std::move(M), std::move(Ctx)));
  }

  static LLVMMemoryBufferRef createTestObject(StringRef Source,
                                              StringRef Name) {
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = cantFail(parseTestModule(*Ctx, Source, Name));

    auto JTMB = cantFail(JITTargetMachineBuilder::detectHost());
    M->setDataLayout(cantFail(JTMB.getDefaultDataLayoutForTarget()));
    auto TM = cantFail(JTMB.createTargetMachine());

    SimpleCompiler SC(*TM);
    auto ObjBuffer = cantFail(SC(*M));
    return wrap(ObjBuffer.release());
  }

  static std::string TargetTriple;
  static bool TargetSupported;
};

std::string OrcCAPITestBase::TargetTriple;
bool OrcCAPITestBase::TargetSupported = false;

namespace {

constexpr StringRef SumExample =
    R"(
    define i32 @sum(i32 %x, i32 %y) {
    entry:
      %r = add nsw i32 %x, %y
      ret i32 %r
    }
  )";

} // end anonymous namespace.

// Consumes the given error ref and returns the string error message.
static std::string toString(LLVMErrorRef E) {
  char *ErrMsg = LLVMGetErrorMessage(E);
  std::string Result(ErrMsg);
  LLVMDisposeErrorMessage(ErrMsg);
  return Result;
}

TEST_F(OrcCAPITestBase, SymbolStringPoolUniquing) {
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
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &OutAddr, "test"))
    FAIL() << "Failed to look up \"test\" symbol (triple = " << TargetTriple
           << "): " << toString(E);
  ASSERT_EQ(Addr, OutAddr);
}

TEST_F(OrcCAPITestBase, DefinitionGenerators) {
  LLVMOrcDefinitionGeneratorRef Gen =
      LLVMOrcCreateCustomCAPIDefinitionGenerator(&definitionGeneratorFn,
                                                 nullptr);
  LLVMOrcJITDylibAddGenerator(MainDylib, Gen);
  LLVMOrcJITTargetAddress OutAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &OutAddr, "test"))
    FAIL() << "The DefinitionGenerator did not create symbol \"test\" "
           << "(triple = " << TargetTriple << "): " << toString(E);
  LLVMOrcJITTargetAddress ExpectedAddr =
      (LLVMOrcJITTargetAddress)(&materializationUnitFn);
  ASSERT_EQ(ExpectedAddr, OutAddr);
}

TEST_F(OrcCAPITestBase, ResourceTrackerDefinitionLifetime) {
  // This test case ensures that all symbols loaded into a JITDylib with a
  // ResourceTracker attached are cleared from the JITDylib once the RT is
  // removed.
  LLVMOrcResourceTrackerRef RT =
      LLVMOrcJITDylibCreateResourceTracker(MainDylib);
  LLVMOrcThreadSafeModuleRef TSM = createTestModule(SumExample, "sum.ll");
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModuleWithRT(Jit, RT, TSM))
    FAIL() << "Failed to add LLVM IR module to LLJIT (triple = " << TargetTriple
           << "): " << toString(E);
  LLVMOrcJITTargetAddress TestFnAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &TestFnAddr, "sum"))
    FAIL() << "Symbol \"sum\" was not added into JIT (triple = " << TargetTriple
           << "): " << toString(E);
  ASSERT_TRUE(!!TestFnAddr);
  LLVMOrcResourceTrackerRemove(RT);
  LLVMOrcJITTargetAddress OutAddr;
  LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &OutAddr, "sum");
  ASSERT_TRUE(Err);
  LLVMConsumeError(Err);

  ASSERT_FALSE(OutAddr);
  LLVMOrcReleaseResourceTracker(RT);
}

TEST_F(OrcCAPITestBase, ResourceTrackerTransfer) {
  LLVMOrcResourceTrackerRef DefaultRT =
      LLVMOrcJITDylibGetDefaultResourceTracker(MainDylib);
  LLVMOrcResourceTrackerRef RT2 =
      LLVMOrcJITDylibCreateResourceTracker(MainDylib);
  LLVMOrcThreadSafeModuleRef TSM = createTestModule(SumExample, "sum.ll");
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModuleWithRT(Jit, DefaultRT, TSM))
    FAIL() << "Failed to add LLVM IR module to LLJIT (triple = " << TargetTriple
           << "): " << toString(E);
  LLVMOrcJITTargetAddress Addr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &Addr, "sum"))
    FAIL() << "Symbol \"sum\" was not added into JIT (triple = " << TargetTriple
           << "): " << toString(E);
  LLVMOrcResourceTrackerTransferTo(DefaultRT, RT2);
  LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &Addr, "sum");
  ASSERT_FALSE(Err);
  LLVMOrcReleaseResourceTracker(RT2);
}

TEST_F(OrcCAPITestBase, AddObjectBuffer) {
  LLVMOrcObjectLayerRef ObjLinkingLayer = LLVMOrcLLJITGetObjLinkingLayer(Jit);
  LLVMMemoryBufferRef ObjBuffer = createTestObject(SumExample, "sum.ll");

  if (LLVMErrorRef E = LLVMOrcObjectLayerAddObjectFile(ObjLinkingLayer,
                                                       MainDylib, ObjBuffer))
    FAIL() << "Failed to add object file to ObjLinkingLayer (triple = "
           << TargetTriple << "): " << toString(E);

  LLVMOrcJITTargetAddress SumAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &SumAddr, "sum"))
    FAIL() << "Symbol \"sum\" was not added into JIT (triple = " << TargetTriple
           << "): " << toString(E);
  ASSERT_TRUE(!!SumAddr);
}

TEST_F(OrcCAPITestBase, ExecutionTest) {
  using SumFunctionType = int32_t (*)(int32_t, int32_t);

  // This test performs OrcJIT compilation of a simple sum module
  LLVMInitializeNativeAsmPrinter();
  LLVMOrcThreadSafeModuleRef TSM = createTestModule(SumExample, "sum.ll");
  if (LLVMErrorRef E = LLVMOrcLLJITAddLLVMIRModule(Jit, MainDylib, TSM))
    FAIL() << "Failed to add LLVM IR module to LLJIT (triple = " << TargetTriple
           << ")" << toString(E);
  LLVMOrcJITTargetAddress TestFnAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &TestFnAddr, "sum"))
    FAIL() << "Symbol \"sum\" was not added into JIT (triple = " << TargetTriple
           << "): " << toString(E);
  auto *SumFn = (SumFunctionType)(TestFnAddr);
  int32_t Result = SumFn(1, 1);
  ASSERT_EQ(2, Result);
}
