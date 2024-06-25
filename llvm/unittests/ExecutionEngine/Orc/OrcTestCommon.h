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

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
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
  class OverridableDispatcher : public InPlaceTaskDispatcher {
  public:
    OverridableDispatcher(CoreAPIsBasedStandardTest &Parent) : Parent(Parent) {}
    void dispatch(std::unique_ptr<Task> T) override;

  private:
    CoreAPIsBasedStandardTest &Parent;
  };

  std::unique_ptr<llvm::orc::ExecutorProcessControl>
  makeEPC(std::shared_ptr<SymbolStringPool> SSP);

  std::shared_ptr<SymbolStringPool> SSP = std::make_shared<SymbolStringPool>();
  ExecutionSession ES{makeEPC(SSP)};
  JITDylib &JD = ES.createBareJITDylib("JD");
  SymbolStringPtr Foo = ES.intern("foo");
  SymbolStringPtr Bar = ES.intern("bar");
  SymbolStringPtr Baz = ES.intern("baz");
  SymbolStringPtr Qux = ES.intern("qux");
  static constexpr ExecutorAddr FooAddr{1};
  static constexpr ExecutorAddr BarAddr{2};
  static constexpr ExecutorAddr BazAddr{3};
  static constexpr ExecutorAddr QuxAddr{4};
  ExecutorSymbolDef FooSym{FooAddr, JITSymbolFlags::Exported};
  ExecutorSymbolDef BarSym{BarAddr, JITSymbolFlags::Exported};
  ExecutorSymbolDef BazSym{BazAddr, JITSymbolFlags::Exported};
  ExecutorSymbolDef QuxSym{QuxAddr, JITSymbolFlags::Exported};
  unique_function<void(std::unique_ptr<Task>)> DispatchOverride;
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
      : MaterializationUnit(
            Interface(std::move(SymbolFlags), std::move(InitSym))),
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

} // namespace llvm

#endif
