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
#include "llvm-c/LLJITUtils.h"
#include "llvm-c/Orc.h"
#include "gtest/gtest.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include <string>

using namespace llvm;
using namespace llvm::orc;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ObjectLayer, LLVMOrcObjectLayerRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ThreadSafeModule, LLVMOrcThreadSafeModuleRef)

// A class that sets strings for extension attributes by querying
// TargetLibraryInfo.
struct TargetI32ArgExtensions {
  std::string Ret;
  std::string Arg;
  TargetI32ArgExtensions(std::string TargetTriple, bool Signed = true) {
    Triple T(TargetTriple);
    if (auto AK = TargetLibraryInfo::getExtAttrForI32Return(T, Signed))
      Ret = Attribute::getNameFromAttrKind(AK).str() + " ";
    if (auto AK = TargetLibraryInfo::getExtAttrForI32Param(T, Signed))
      Arg = Attribute::getNameFromAttrKind(AK).str() + " ";
  }
};

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

    // Create test functions in text format, with the proper extension
    // attributes.
    if (SumExample.empty()) {
      TargetI32ArgExtensions ArgExt(TargetTriple);
      std::ostringstream OS;
      OS << "define " << ArgExt.Ret << "i32 "
         << "@sum(i32 " << ArgExt.Arg << "%x, i32 " << ArgExt.Arg << "%y)"
         << R"( {
          entry:
          %r = add nsw i32 %x, %y
          ret i32 %r
          }
        )";
      SumExample = OS.str();

      OS << R"(
          !llvm.module.flags = !{!0}
          !llvm.dbg.cu = !{!1}
          !0 = !{i32 2, !"Debug Info Version", i32 3}
          !1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, emissionKind: FullDebug)
          !2 = !DIFile(filename: "sum.c", directory: "/tmp")
        )";
      SumDebugExample = OS.str();
    }
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
    // Check whether Jit has already been torn down -- we allow clients to do
    // this manually to check teardown behavior.
    if (Jit) {
      LLVMOrcDisposeLLJIT(Jit);
      Jit = nullptr;
    }
  }

protected:
  static bool isSupported(StringRef Triple) {
    // TODO: Print error messages in failure logs, use them to audit this list.
    // Some architectures may be unsupportable or missing key components, but
    // some may just be failing due to bugs in this testcase.
    if (Triple.starts_with("armv7") || Triple.starts_with("armv8l"))
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
      LLVMOrcCSymbolMapPair Pair = {Element.Name, Sym};
      LLVMOrcCSymbolMapPair Pairs[] = {Pair};
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

  static std::string SumExample;
  static std::string SumDebugExample;
};

std::string OrcCAPITestBase::TargetTriple;
bool OrcCAPITestBase::TargetSupported = false;

std::string OrcCAPITestBase::SumExample;
std::string OrcCAPITestBase::SumDebugExample;

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
  LLVMOrcCSymbolMapPair Pair = {Name, Sym};
  LLVMOrcCSymbolMapPair Pairs[] = {Pair};
  LLVMOrcMaterializationUnitRef MU = LLVMOrcAbsoluteSymbols(Pairs, 1);
  if (LLVMErrorRef E = LLVMOrcJITDylibDefine(MainDylib, MU))
    FAIL() << "Unexpected error while adding \"test\" symbol (triple = "
           << TargetTriple << "): " << toString(E);
  LLVMOrcJITTargetAddress OutAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &OutAddr, "test"))
    FAIL() << "Failed to look up \"test\" symbol (triple = " << TargetTriple
           << "): " << toString(E);
  ASSERT_EQ(Addr, OutAddr);
}

struct ExecutionSessionLookupHelper {
  bool ExpectSuccess = true;
  bool CallbackReceived = false;
  size_t NumExpectedPairs;
  LLVMOrcCSymbolMapPair *ExpectedMapping;
};

static void executionSessionLookupHandlerCallback(LLVMErrorRef Err,
                                                  LLVMOrcCSymbolMapPairs Result,
                                                  size_t NumPairs,
                                                  void *RawCtx) {
  auto *Ctx = static_cast<ExecutionSessionLookupHelper *>(RawCtx);
  Ctx->CallbackReceived = true;
  if (Ctx->ExpectSuccess) {
    EXPECT_THAT_ERROR(unwrap(Err), Succeeded());
    EXPECT_EQ(NumPairs, Ctx->NumExpectedPairs)
        << "Expected " << Ctx->NumExpectedPairs << " entries in result, got "
        << NumPairs;
    auto ExpectedMappingEnd = Ctx->ExpectedMapping + Ctx->NumExpectedPairs;
    for (unsigned I = 0; I != NumPairs; ++I) {
      auto J =
          std::find_if(Ctx->ExpectedMapping, ExpectedMappingEnd,
                       [N = Result[I].Name](const LLVMOrcCSymbolMapPair &Val) {
                         return Val.Name == N;
                       });
      EXPECT_NE(J, ExpectedMappingEnd)
          << "Missing symbol \""
          << LLVMOrcSymbolStringPoolEntryStr(Result[I].Name) << "\"";
      if (J != ExpectedMappingEnd) {
        EXPECT_EQ(Result[I].Sym.Address, J->Sym.Address)
            << "Result map for \"" << Result[I].Name
            << "\" differs from expected value: "
            << formatv("{0:x} vs {1:x}", Result[I].Sym.Address, J->Sym.Address);
      }
    }
  } else
    EXPECT_THAT_ERROR(unwrap(Err), Failed());
}

TEST_F(OrcCAPITestBase, ExecutionSessionLookup_Success) {
  // Test a successful generic lookup. We will look up three symbols over two
  // JITDylibs: { "Foo" (Required), "Bar" (Weakly-ref), "Baz" (Required) } over
  // { MainJITDylib (Exported-only), ExtraJD (All symbols) }.
  //
  // Foo will be defined as exported in MainJD.
  // Bar will be defined as non-exported in MainJD.
  // Baz will be defined as non-exported in ExtraJD.
  //
  // This will require (1) that we find the regular exported symbol Foo in
  // MainJD, (2) that we *don't* find the non-exported symbol Bar in MainJD
  // but also don't error (since it's weakly referenced), and (3) that we
  // find the non-exported symbol Baz in ExtraJD (since we're searching all
  // symbols in ExtraJD).

  ExecutionSessionLookupHelper H;
  LLVMOrcSymbolStringPoolEntryRef Foo = LLVMOrcLLJITMangleAndIntern(Jit, "Foo");
  LLVMOrcSymbolStringPoolEntryRef Bar = LLVMOrcLLJITMangleAndIntern(Jit, "Bar");
  LLVMOrcSymbolStringPoolEntryRef Baz = LLVMOrcLLJITMangleAndIntern(Jit, "Baz");

  // Create ExtraJD.
  LLVMOrcJITDylibRef ExtraJD = nullptr;
  if (auto E = LLVMOrcExecutionSessionCreateJITDylib(ExecutionSession, &ExtraJD,
                                                     "ExtraJD")) {
    FAIL() << "Unexpected error while creating JITDylib \"ExtraJD\" (triple = "
           << TargetTriple << "): " << toString(E);
    return;
  }

  // Add exported symbols "Foo" and "Bar" to Main JITDylib.
  LLVMOrcRetainSymbolStringPoolEntry(Foo);
  LLVMOrcRetainSymbolStringPoolEntry(Bar);
  LLVMOrcCSymbolMapPair MainJDPairs[] = {
      {Foo, {0x1, {LLVMJITSymbolGenericFlagsExported, 0}}},
      {Bar, {0x2, {LLVMJITSymbolGenericFlagsNone, 0}}}};
  LLVMOrcMaterializationUnitRef MainJDMU =
      LLVMOrcAbsoluteSymbols(MainJDPairs, 2);
  if (LLVMErrorRef E = LLVMOrcJITDylibDefine(MainDylib, MainJDMU))
    FAIL() << "Unexpected error while adding MainDylib symbols (triple = "
           << TargetTriple << "): " << toString(E);

  // Add non-exported symbol "Baz" to ExtraJD.
  LLVMOrcRetainSymbolStringPoolEntry(Baz);
  LLVMOrcCSymbolMapPair ExtraJDPairs[] = {
      {Baz, {0x3, {LLVMJITSymbolGenericFlagsNone, 0}}}};
  LLVMOrcMaterializationUnitRef ExtraJDMU =
      LLVMOrcAbsoluteSymbols(ExtraJDPairs, 1);
  if (LLVMErrorRef E = LLVMOrcJITDylibDefine(ExtraJD, ExtraJDMU))
    FAIL() << "Unexpected error while adding ExtraJD symbols (triple = "
           << TargetTriple << "): " << toString(E);

  // Create expected mapping for result:
  LLVMOrcCSymbolMapPair ExpectedMapping[] = {
      {Foo, {0x1, {LLVMJITSymbolGenericFlagsExported, 0}}},
      {Baz, {0x3, {LLVMJITSymbolGenericFlagsNone, 0}}}};
  H.ExpectedMapping = ExpectedMapping;
  H.NumExpectedPairs = 2;

  // Issue the lookup. We're using the default same-thread dispatch, so the
  // handler should have run by the time we return from this call.
  LLVMOrcCJITDylibSearchOrderElement SO[] = {
      {MainDylib, LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly},
      {ExtraJD, LLVMOrcJITDylibLookupFlagsMatchAllSymbols}};

  LLVMOrcRetainSymbolStringPoolEntry(Foo);
  LLVMOrcRetainSymbolStringPoolEntry(Bar);
  LLVMOrcRetainSymbolStringPoolEntry(Baz);
  LLVMOrcCLookupSetElement LS[] = {
      {Foo, LLVMOrcSymbolLookupFlagsRequiredSymbol},
      {Bar, LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol},
      {Baz, LLVMOrcSymbolLookupFlagsRequiredSymbol}};
  LLVMOrcExecutionSessionLookup(ExecutionSession, LLVMOrcLookupKindStatic, SO,
                                2, LS, 3, executionSessionLookupHandlerCallback,
                                &H);

  EXPECT_TRUE(H.CallbackReceived) << "Lookup callback never received";

  // Release our local string ptrs.
  LLVMOrcReleaseSymbolStringPoolEntry(Baz);
  LLVMOrcReleaseSymbolStringPoolEntry(Bar);
  LLVMOrcReleaseSymbolStringPoolEntry(Foo);
}

TEST_F(OrcCAPITestBase, ExecutionSessionLookup_Failure) {
  // Test generic lookup failure case. We will look up a symbol in MainDylib
  // without defining it. We expect this to result in a symbol-not-found error.

  ExecutionSessionLookupHelper H;
  H.ExpectSuccess = false;

  LLVMOrcCJITDylibSearchOrderElement SO[] = {
      {MainDylib, LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly}};
  LLVMOrcCLookupSetElement LS[] = {{LLVMOrcLLJITMangleAndIntern(Jit, "Foo"),
                                    LLVMOrcSymbolLookupFlagsRequiredSymbol}};
  LLVMOrcExecutionSessionLookup(ExecutionSession, LLVMOrcLookupKindStatic, SO,
                                1, LS, 1, executionSessionLookupHandlerCallback,
                                &H);

  EXPECT_TRUE(H.CallbackReceived) << "Lookup callback never received";
}

TEST_F(OrcCAPITestBase, DefinitionGenerators) {
  LLVMOrcDefinitionGeneratorRef Gen =
      LLVMOrcCreateCustomCAPIDefinitionGenerator(&definitionGeneratorFn,
                                                 nullptr, nullptr);
  LLVMOrcJITDylibAddGenerator(MainDylib, Gen);
  LLVMOrcJITTargetAddress OutAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &OutAddr, "test"))
    FAIL() << "The DefinitionGenerator did not create symbol \"test\" "
           << "(triple = " << TargetTriple << "): " << toString(E);
  LLVMOrcJITTargetAddress ExpectedAddr =
      (LLVMOrcJITTargetAddress)(&materializationUnitFn);
  ASSERT_EQ(ExpectedAddr, OutAddr);
}

#if defined(_AIX)
TEST_F(OrcCAPITestBase, DISABLED_ResourceTrackerDefinitionLifetime) {
#else
TEST_F(OrcCAPITestBase, ResourceTrackerDefinitionLifetime) {
#endif
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

#if defined(_AIX)
TEST_F(OrcCAPITestBase, DISABLED_ResourceTrackerTransfer) {
#else
TEST_F(OrcCAPITestBase, ResourceTrackerTransfer) {
#endif
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

#if defined(_AIX)
TEST_F(OrcCAPITestBase, DISABLED_AddObjectBuffer) {
#else
TEST_F(OrcCAPITestBase, AddObjectBuffer) {
#endif
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

// JITLink debug support plugins put information about JITed code in this GDB
// JIT Interface global from OrcTargetProcess.
extern "C" struct jit_descriptor __jit_debug_descriptor;

static void *findLastDebugDescriptorEntryPtr() {
  struct jit_code_entry *Last = __jit_debug_descriptor.first_entry;
  while (Last && Last->next_entry)
    Last = Last->next_entry;
  return Last;
}

#if defined(_AIX) or not(defined(__ELF__) or defined(__MACH__))
TEST_F(OrcCAPITestBase, DISABLED_EnableDebugSupport) {
#else
static LLVM_ATTRIBUTE_USED void linkComponents() {
  errs() << "Linking in runtime functions\n"
         << (void *)&llvm_orc_registerJITLoaderGDBWrapper << '\n'
         << (void *)&llvm_orc_registerJITLoaderGDBAllocAction << '\n';
}
TEST_F(OrcCAPITestBase, EnableDebugSupport) {
#endif
  void *Before = findLastDebugDescriptorEntryPtr();
  LLVMMemoryBufferRef ObjBuffer = createTestObject(SumDebugExample, "sum.ll");
  LLVMOrcObjectLayerRef ObjLayer = LLVMOrcLLJITGetObjLinkingLayer(Jit);

  if (LLVMErrorRef E = LLVMOrcLLJITEnableDebugSupport(Jit)) {
    EXPECT_FALSE(isa<ObjectLinkingLayer>(unwrap(ObjLayer)))
        << "Error testing LLJIT debug support "
        << "(triple = " << TargetTriple << "): " << toString(E);
    GTEST_SKIP() << "LLJIT C bindings provide debug support only for JITLink";
  }

  if (LLVMErrorRef E =
          LLVMOrcObjectLayerAddObjectFile(ObjLayer, MainDylib, ObjBuffer))
    FAIL() << "Failed to add object file to ObjLinkingLayer (triple = "
           << TargetTriple << "): " << toString(E);

  LLVMOrcJITTargetAddress SumAddr;
  if (LLVMErrorRef E = LLVMOrcLLJITLookup(Jit, &SumAddr, "sum"))
    FAIL() << "Symbol \"sum\" was not added into JIT (triple = " << TargetTriple
           << "): " << toString(E);

  void *After = findLastDebugDescriptorEntryPtr();
  ASSERT_NE(Before, After);
}

#if defined(_AIX)
TEST_F(OrcCAPITestBase, DISABLED_ExecutionTest) {
#else
TEST_F(OrcCAPITestBase, ExecutionTest) {
#endif
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

void Destroy(void *Ctx) {}

void TargetFn() {}

void Materialize(void *Ctx, LLVMOrcMaterializationResponsibilityRef MR) {
  LLVMOrcJITDylibRef JD =
      LLVMOrcMaterializationResponsibilityGetTargetDylib(MR);
  ASSERT_TRUE(!!JD);

  LLVMOrcExecutionSessionRef ES =
      LLVMOrcMaterializationResponsibilityGetExecutionSession(MR);
  ASSERT_TRUE(!!ES);

  LLVMOrcSymbolStringPoolEntryRef InitSym =
      LLVMOrcMaterializationResponsibilityGetInitializerSymbol(MR);
  ASSERT_TRUE(!InitSym);

  size_t NumSymbols;
  LLVMOrcCSymbolFlagsMapPairs Symbols =
      LLVMOrcMaterializationResponsibilityGetSymbols(MR, &NumSymbols);

  ASSERT_TRUE(!!Symbols);
  ASSERT_EQ(NumSymbols, (size_t)1);

  LLVMOrcSymbolStringPoolEntryRef *RequestedSymbols =
      LLVMOrcMaterializationResponsibilityGetRequestedSymbols(MR, &NumSymbols);

  ASSERT_TRUE(!!RequestedSymbols);
  ASSERT_EQ(NumSymbols, (size_t)1);

  LLVMOrcCSymbolFlagsMapPair TargetSym = Symbols[0];

  ASSERT_EQ(RequestedSymbols[0], TargetSym.Name);
  LLVMOrcRetainSymbolStringPoolEntry(TargetSym.Name);

  LLVMOrcDisposeCSymbolFlagsMap(Symbols);
  LLVMOrcDisposeSymbols(RequestedSymbols);

  LLVMOrcJITTargetAddress Addr = (LLVMOrcJITTargetAddress)(&TargetFn);

  LLVMJITSymbolFlags Flags = {
      LLVMJITSymbolGenericFlagsExported | LLVMJITSymbolGenericFlagsCallable, 0};
  ASSERT_EQ(TargetSym.Flags.GenericFlags, Flags.GenericFlags);
  ASSERT_EQ(TargetSym.Flags.TargetFlags, Flags.TargetFlags);

  LLVMJITEvaluatedSymbol Sym = {Addr, Flags};

  LLVMOrcLLJITRef J = (LLVMOrcLLJITRef)Ctx;

  LLVMOrcSymbolStringPoolEntryRef OtherSymbol =
      LLVMOrcLLJITMangleAndIntern(J, "other");
  LLVMOrcSymbolStringPoolEntryRef DependencySymbol =
      LLVMOrcLLJITMangleAndIntern(J, "dependency");

  LLVMOrcRetainSymbolStringPoolEntry(OtherSymbol);
  LLVMOrcRetainSymbolStringPoolEntry(DependencySymbol);
  LLVMOrcCSymbolFlagsMapPair NewSymbols[] = {
      {OtherSymbol, Flags},
      {DependencySymbol, Flags},
  };
  LLVMOrcMaterializationResponsibilityDefineMaterializing(MR, NewSymbols, 2);

  LLVMOrcRetainSymbolStringPoolEntry(OtherSymbol);
  LLVMOrcMaterializationResponsibilityRef OtherMR = NULL;
  {
    LLVMErrorRef Err = LLVMOrcMaterializationResponsibilityDelegate(
        MR, &OtherSymbol, 1, &OtherMR);
    if (Err) {
      char *ErrMsg = LLVMGetErrorMessage(Err);
      fprintf(stderr, "Error: %s\n", ErrMsg);
      LLVMDisposeErrorMessage(ErrMsg);
      LLVMOrcMaterializationResponsibilityFailMaterialization(MR);
      LLVMOrcDisposeMaterializationResponsibility(MR);
      return;
    }
  }
  assert(OtherMR);

  LLVMOrcCSymbolMapPair OtherPair = {OtherSymbol, Sym};
  LLVMOrcMaterializationUnitRef OtherMU = LLVMOrcAbsoluteSymbols(&OtherPair, 1);
  // OtherSymbol is no longer owned by us
  {
    LLVMErrorRef Err =
        LLVMOrcMaterializationResponsibilityReplace(OtherMR, OtherMU);
    if (Err) {
      char *ErrMsg = LLVMGetErrorMessage(Err);
      fprintf(stderr, "Error: %s\n", ErrMsg);
      LLVMDisposeErrorMessage(ErrMsg);

      LLVMOrcMaterializationResponsibilityFailMaterialization(OtherMR);
      LLVMOrcMaterializationResponsibilityFailMaterialization(MR);

      LLVMOrcDisposeMaterializationResponsibility(OtherMR);
      LLVMOrcDisposeMaterializationResponsibility(MR);
      LLVMOrcDisposeMaterializationUnit(OtherMU);
      return;
    }
  }
  LLVMOrcDisposeMaterializationResponsibility(OtherMR);

  // FIXME: Implement async lookup
  LLVMOrcRetainSymbolStringPoolEntry(DependencySymbol);
  LLVMOrcCSymbolMapPair Pair = {DependencySymbol, Sym};
  LLVMOrcMaterializationResponsibilityNotifyResolved(MR, &Pair, 1);
  // DependencySymbol no longer owned by us

  Pair = {TargetSym.Name, Sym};
  LLVMOrcMaterializationResponsibilityNotifyResolved(MR, &Pair, 1);

  LLVMOrcRetainSymbolStringPoolEntry(TargetSym.Name);
  LLVMOrcCDependenceMapPair Dependency = {JD, {&DependencySymbol, 1}};
  LLVMOrcCSymbolDependenceGroup DependenceSet = {
      /*.Symbols = */ {/*.Symbols = */ &TargetSym.Name, /* .Length = */ 1},
      /* .Dependencies = */ &Dependency,
      /* .NumDependencies = */ 1};

  LLVMOrcMaterializationResponsibilityNotifyEmitted(MR, &DependenceSet, 1);
  LLVMOrcDisposeMaterializationResponsibility(MR);
}

TEST_F(OrcCAPITestBase, MaterializationResponsibility) {
  LLVMJITSymbolFlags Flags = {
      LLVMJITSymbolGenericFlagsExported | LLVMJITSymbolGenericFlagsCallable, 0};
  LLVMOrcCSymbolFlagsMapPair Sym = {LLVMOrcLLJITMangleAndIntern(Jit, "foo"),
                                    Flags};

  LLVMOrcMaterializationUnitRef MU = LLVMOrcCreateCustomMaterializationUnit(
      "MU", (void *)Jit, &Sym, 1, NULL, &Materialize, NULL, &Destroy);
  LLVMOrcJITDylibRef JD = LLVMOrcLLJITGetMainJITDylib(Jit);
  LLVMOrcJITDylibDefine(JD, MU);

  LLVMOrcJITTargetAddress Addr;
  if (LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &Addr, "foo")) {
    FAIL() << "foo was not materialized " << toString(Err);
  }
  ASSERT_TRUE(!!Addr);
  ASSERT_EQ(Addr, (LLVMOrcJITTargetAddress)&TargetFn);

  if (LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &Addr, "other")) {
    FAIL() << "other was not materialized " << toString(Err);
  }
  ASSERT_TRUE(!!Addr);
  ASSERT_EQ(Addr, (LLVMOrcJITTargetAddress)&TargetFn);

  if (LLVMErrorRef Err = LLVMOrcLLJITLookup(Jit, &Addr, "dependency")) {
    FAIL() << "dependency was not materialized " << toString(Err);
  }
  ASSERT_TRUE(!!Addr);
  ASSERT_EQ(Addr, (LLVMOrcJITTargetAddress)&TargetFn);
}

struct SuspendedLookupContext {
  std::function<void()> AsyncWork;
  LLVMOrcSymbolStringPoolEntryRef NameToGenerate;
  JITTargetAddress AddrToGenerate;

  bool Disposed = false;
  bool QueryCompleted = true;
};

static LLVMErrorRef TryToGenerateWithSuspendedLookup(
    LLVMOrcDefinitionGeneratorRef GeneratorObj, void *RawCtx,
    LLVMOrcLookupStateRef *LookupState, LLVMOrcLookupKind Kind,
    LLVMOrcJITDylibRef JD, LLVMOrcJITDylibLookupFlags JDLookupFlags,
    LLVMOrcCLookupSet LookupSet, size_t LookupSetSize) {

  auto *Ctx = static_cast<SuspendedLookupContext *>(RawCtx);

  assert(LookupSetSize == 1);
  assert(LookupSet[0].Name == Ctx->NameToGenerate);

  LLVMJITEvaluatedSymbol Sym = {0x1234, {LLVMJITSymbolGenericFlagsExported, 0}};
  LLVMOrcRetainSymbolStringPoolEntry(LookupSet[0].Name);
  LLVMOrcCSymbolMapPair Pair = {LookupSet[0].Name, Sym};
  LLVMOrcCSymbolMapPair Pairs[] = {Pair};
  LLVMOrcMaterializationUnitRef MU = LLVMOrcAbsoluteSymbols(Pairs, 1);

  // Capture and reset LookupState to suspend the lookup. We'll continue it in
  // the SuspendedLookup testcase below.
  Ctx->AsyncWork = [LS = *LookupState, JD, MU]() {
    LLVMErrorRef Err = LLVMOrcJITDylibDefine(JD, MU);
    LLVMOrcLookupStateContinueLookup(LS, Err);
  };
  *LookupState = nullptr;
  return LLVMErrorSuccess;
}

static void DisposeSuspendedLookupContext(void *Ctx) {
  static_cast<SuspendedLookupContext *>(Ctx)->Disposed = true;
}

static void
suspendLookupTestLookupHandlerCallback(LLVMErrorRef Err,
                                       LLVMOrcCSymbolMapPairs Result,
                                       size_t NumPairs, void *RawCtx) {
  if (Err) {
    FAIL() << "Suspended DefinitionGenerator did not create symbol \"foo\": "
           << toString(Err);
    return;
  }

  EXPECT_EQ(NumPairs, 1U)
      << "Unexpected number of result entries: expected 1, got " << NumPairs;

  auto *Ctx = static_cast<SuspendedLookupContext *>(RawCtx);
  EXPECT_EQ(Result[0].Name, Ctx->NameToGenerate);
  EXPECT_EQ(Result[0].Sym.Address, Ctx->AddrToGenerate);

  Ctx->QueryCompleted = true;
}

TEST_F(OrcCAPITestBase, SuspendedLookup) {
  // Test that we can suspend lookup in a custom generator.
  SuspendedLookupContext Ctx;
  Ctx.NameToGenerate = LLVMOrcLLJITMangleAndIntern(Jit, "foo");
  Ctx.AddrToGenerate = 0x1234;

  // Add generator.
  LLVMOrcJITDylibAddGenerator(MainDylib,
                              LLVMOrcCreateCustomCAPIDefinitionGenerator(
                                  &TryToGenerateWithSuspendedLookup, &Ctx,
                                  DisposeSuspendedLookupContext));

  // Expect no work to do before the lookup.
  EXPECT_FALSE(Ctx.AsyncWork) << "Unexpected generator work before lookup";

  // Issue lookup. This should trigger the generator, but generation should
  // be suspended.
  LLVMOrcCJITDylibSearchOrderElement SO[] = {
      {MainDylib, LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly}};
  LLVMOrcRetainSymbolStringPoolEntry(Ctx.NameToGenerate);
  LLVMOrcCLookupSetElement LS[] = {
      {Ctx.NameToGenerate, LLVMOrcSymbolLookupFlagsRequiredSymbol}};
  LLVMOrcExecutionSessionLookup(ExecutionSession, LLVMOrcLookupKindStatic, SO,
                                1, LS, 1,
                                suspendLookupTestLookupHandlerCallback, &Ctx);

  // Expect that we now have generator work to do.
  EXPECT_TRUE(Ctx.AsyncWork)
      << "Failed to generator (or failed to suspend generator)";

  // Do the work. This should allow the query to complete.
  Ctx.AsyncWork();

  // Check that the query completed.
  EXPECT_TRUE(Ctx.QueryCompleted);

  // Release our local copy of the string.
  LLVMOrcReleaseSymbolStringPoolEntry(Ctx.NameToGenerate);

  // Explicitly tear down the JIT.
  LLVMOrcDisposeLLJIT(Jit);
  Jit = nullptr;

  // Check that the generator context was "destroyed".
  EXPECT_TRUE(Ctx.Disposed);
}
