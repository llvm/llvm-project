//===------ OrcTestCommon.h - Utilities for Orc Unit Tests ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common utilities for the Orc unit tests.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_ORC_ORCTESTCOMMON_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_ORC_ORCTESTCOMMON_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

#include <memory>

namespace llvm {

namespace orc {
// CoreAPIsStandardTest that saves a bunch of boilerplate by providing the
// following:
//
// (1) ES -- An ExecutionSession
// (2) Foo, Bar, Baz, Qux -- SymbolStringPtrs for strings "foo", "bar", "baz",
//     and "qux" respectively.
// (3) FooAddr, BarAddr, BazAddr, QuxAddr -- Dummy addresses. Guaranteed
//     distinct and non-null.
// (4) FooSym, BarSym, BazSym, QuxSym -- JITEvaluatedSymbols with FooAddr,
//     BarAddr, BazAddr, and QuxAddr respectively. All with default strong,
//     linkage and non-hidden visibility.
// (5) V -- A JITDylib associated with ES.
class CoreAPIsBasedStandardTest : public testing::Test {
public:
  ~CoreAPIsBasedStandardTest() {
    if (auto Err = ES.endSession())
      ES.reportError(std::move(Err));
  }

protected:
  std::shared_ptr<SymbolStringPool> SSP = std::make_shared<SymbolStringPool>();
  ExecutionSession ES{SSP};
  JITDylib &JD = ES.createBareJITDylib("JD");
  SymbolStringPtr Foo = ES.intern("foo");
  SymbolStringPtr Bar = ES.intern("bar");
  SymbolStringPtr Baz = ES.intern("baz");
  SymbolStringPtr Qux = ES.intern("qux");
  static const JITTargetAddress FooAddr = 1U;
  static const JITTargetAddress BarAddr = 2U;
  static const JITTargetAddress BazAddr = 3U;
  static const JITTargetAddress QuxAddr = 4U;
  JITEvaluatedSymbol FooSym =
      JITEvaluatedSymbol(FooAddr, JITSymbolFlags::Exported);
  JITEvaluatedSymbol BarSym =
      JITEvaluatedSymbol(BarAddr, JITSymbolFlags::Exported);
  JITEvaluatedSymbol BazSym =
      JITEvaluatedSymbol(BazAddr, JITSymbolFlags::Exported);
  JITEvaluatedSymbol QuxSym =
      JITEvaluatedSymbol(QuxAddr, JITSymbolFlags::Exported);
};

} // end namespace orc

class OrcNativeTarget {
public:
  static void initialize() {
    if (!NativeTargetInitialized) {
      InitializeNativeTarget();
      InitializeNativeTargetAsmParser();
      InitializeNativeTargetAsmPrinter();
      NativeTargetInitialized = true;
    }
  }

private:
  static bool NativeTargetInitialized;
};

class SimpleMaterializationUnit : public orc::MaterializationUnit {
public:
  using MaterializeFunction =
      std::function<void(std::unique_ptr<orc::MaterializationResponsibility>)>;
  using DiscardFunction =
      std::function<void(const orc::JITDylib &, orc::SymbolStringPtr)>;
  using DestructorFunction = std::function<void()>;

  SimpleMaterializationUnit(
      orc::SymbolFlagsMap SymbolFlags, MaterializeFunction Materialize,
      orc::SymbolStringPtr InitSym = nullptr,
      DiscardFunction Discard = DiscardFunction(),
      DestructorFunction Destructor = DestructorFunction())
      : MaterializationUnit(std::move(SymbolFlags), std::move(InitSym)),
        Materialize(std::move(Materialize)), Discard(std::move(Discard)),
        Destructor(std::move(Destructor)) {}

  ~SimpleMaterializationUnit() override {
    if (Destructor)
      Destructor();
  }

  StringRef getName() const override { return "<Simple>"; }

  void
  materialize(std::unique_ptr<orc::MaterializationResponsibility> R) override {
    Materialize(std::move(R));
  }

  void discard(const orc::JITDylib &JD,
               const orc::SymbolStringPtr &Name) override {
    if (Discard)
      Discard(JD, std::move(Name));
    else
      llvm_unreachable("Discard not supported");
  }

private:
  MaterializeFunction Materialize;
  DiscardFunction Discard;
  DestructorFunction Destructor;
};

// Base class for Orc tests that will execute code.
class OrcExecutionTest {
public:

  OrcExecutionTest() {

    // Initialize the native target if it hasn't been done already.
    OrcNativeTarget::initialize();

    // Try to select a TargetMachine for the host.
    TM.reset(EngineBuilder().selectTarget());

    if (TM) {
      // If we found a TargetMachine, check that it's one that Orc supports.
      const Triple& TT = TM->getTargetTriple();

      // Bail out for windows platforms. We do not support these yet.
      if ((TT.getArch() != Triple::x86_64 && TT.getArch() != Triple::x86) ||
           TT.isOSWindows())
        return;

      // Target can JIT?
      SupportsJIT = TM->getTarget().hasJIT();
      // Use ability to create callback manager to detect whether Orc
      // has indirection support on this platform. This way the test
      // and Orc code do not get out of sync.
      SupportsIndirection = !!orc::createLocalCompileCallbackManager(TT, ES, 0);
    }
  };

  ~OrcExecutionTest() {
    if (auto Err = ES.endSession())
      ES.reportError(std::move(Err));
  }

protected:
  orc::ExecutionSession ES;
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  bool SupportsJIT = false;
  bool SupportsIndirection = false;
};

class ModuleBuilder {
public:
  ModuleBuilder(LLVMContext &Context, StringRef Triple,
                StringRef Name);

  Function *createFunctionDecl(FunctionType *FTy, StringRef Name) {
    return Function::Create(FTy, GlobalValue::ExternalLinkage, Name, M.get());
  }

  Module* getModule() { return M.get(); }
  const Module* getModule() const { return M.get(); }
  std::unique_ptr<Module> takeModule() { return std::move(M); }

private:
  std::unique_ptr<Module> M;
};

// Dummy struct type.
struct DummyStruct {
  int X[256];
};

inline StructType *getDummyStructTy(LLVMContext &Context) {
  return StructType::get(ArrayType::get(Type::getInt32Ty(Context), 256));
}

template <typename HandleT, typename ModuleT>
class MockBaseLayer {
public:

  using ModuleHandleT = HandleT;

  using AddModuleSignature =
    Expected<ModuleHandleT>(ModuleT M,
                            std::shared_ptr<JITSymbolResolver> R);

  using RemoveModuleSignature = Error(ModuleHandleT H);
  using FindSymbolSignature = JITSymbol(const std::string &Name,
                                        bool ExportedSymbolsOnly);
  using FindSymbolInSignature = JITSymbol(ModuleHandleT H,
                                          const std::string &Name,
                                          bool ExportedSymbolsONly);
  using EmitAndFinalizeSignature = Error(ModuleHandleT H);

  std::function<AddModuleSignature> addModuleImpl;
  std::function<RemoveModuleSignature> removeModuleImpl;
  std::function<FindSymbolSignature> findSymbolImpl;
  std::function<FindSymbolInSignature> findSymbolInImpl;
  std::function<EmitAndFinalizeSignature> emitAndFinalizeImpl;

  Expected<ModuleHandleT> addModule(ModuleT M,
                                    std::shared_ptr<JITSymbolResolver> R) {
    assert(addModuleImpl &&
           "addModule called, but no mock implementation was provided");
    return addModuleImpl(std::move(M), std::move(R));
  }

  Error removeModule(ModuleHandleT H) {
    assert(removeModuleImpl &&
           "removeModule called, but no mock implementation was provided");
    return removeModuleImpl(H);
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    assert(findSymbolImpl &&
           "findSymbol called, but no mock implementation was provided");
    return findSymbolImpl(Name, ExportedSymbolsOnly);
  }

  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    assert(findSymbolInImpl &&
           "findSymbolIn called, but no mock implementation was provided");
    return findSymbolInImpl(H, Name, ExportedSymbolsOnly);
  }

  Error emitAndFinaliez(ModuleHandleT H) {
    assert(emitAndFinalizeImpl &&
           "emitAndFinalize called, but no mock implementation was provided");
    return emitAndFinalizeImpl(H);
  }
};

class ReturnNullJITSymbol {
public:
  template <typename... Args>
  JITSymbol operator()(Args...) const {
    return nullptr;
  }
};

template <typename ReturnT>
class DoNothingAndReturn {
public:
  DoNothingAndReturn(ReturnT Ret) : Ret(std::move(Ret)) {}

  template <typename... Args>
  void operator()(Args...) const { return Ret; }
private:
  ReturnT Ret;
};

template <>
class DoNothingAndReturn<void> {
public:
  template <typename... Args>
  void operator()(Args...) const { }
};

} // namespace llvm

#endif
