//===------ OrcTestCommon.h - Utilities for Orc Unit Tests ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>

namespace llvm {

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

      if ((TT.getArch() != Triple::x86_64 && TT.getArch() != Triple::x86) ||
          TT.isOSWindows())
        TM = nullptr;
    }
  };

protected:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
};

class ModuleBuilder {
public:
  ModuleBuilder(LLVMContext &Context, StringRef Triple,
                StringRef Name);

  template <typename FuncType>
  Function* createFunctionDecl(StringRef Name) {
    return Function::Create(
             TypeBuilder<FuncType, false>::get(M->getContext()),
             GlobalValue::ExternalLinkage, Name, M.get());
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

// TypeBuilder specialization for DummyStruct.
template <bool XCompile>
class TypeBuilder<DummyStruct, XCompile> {
public:
  static StructType *get(LLVMContext &Context) {
    return StructType::get(
        TypeBuilder<types::i<32>[256], XCompile>::get(Context));
  }
};

template <typename HandleT,
          typename AddModuleFtor,
          typename RemoveModuleFtor,
          typename FindSymbolFtor,
          typename FindSymbolInFtor>
class MockBaseLayer {
public:

  typedef HandleT ModuleHandleT;

  MockBaseLayer(AddModuleFtor &&AddModule,
                RemoveModuleFtor &&RemoveModule,
                FindSymbolFtor &&FindSymbol,
                FindSymbolInFtor &&FindSymbolIn)
      : AddModule(std::move(AddModule)),
        RemoveModule(std::move(RemoveModule)),
        FindSymbol(std::move(FindSymbol)),
        FindSymbolIn(std::move(FindSymbolIn))
  {}

  template <typename ModuleT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  Expected<ModuleHandleT> addModule(ModuleT Ms, MemoryManagerPtrT MemMgr,
                                    SymbolResolverPtrT Resolver) {
    return AddModule(std::move(Ms), std::move(MemMgr), std::move(Resolver));
  }

  Error removeModule(ModuleHandleT H) {
    return RemoveModule(H);
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return FindSymbol(Name, ExportedSymbolsOnly);
  }

  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return FindSymbolIn(H, Name, ExportedSymbolsOnly);
  }

private:
  AddModuleFtor AddModule;
  RemoveModuleFtor RemoveModule;
  FindSymbolFtor FindSymbol;
  FindSymbolInFtor FindSymbolIn;
};

template <typename ModuleHandleT,
          typename AddModuleFtor,
          typename RemoveModuleFtor,
          typename FindSymbolFtor,
          typename FindSymbolInFtor>
MockBaseLayer<ModuleHandleT, AddModuleFtor, RemoveModuleFtor,
              FindSymbolFtor, FindSymbolInFtor>
createMockBaseLayer(AddModuleFtor &&AddModule,
                    RemoveModuleFtor &&RemoveModule,
                    FindSymbolFtor &&FindSymbol,
                    FindSymbolInFtor &&FindSymbolIn) {
  return MockBaseLayer<ModuleHandleT, AddModuleFtor, RemoveModuleFtor,
                       FindSymbolFtor, FindSymbolInFtor>(
                         std::forward<AddModuleFtor>(AddModule),
                         std::forward<RemoveModuleFtor>(RemoveModule),
                         std::forward<FindSymbolFtor>(FindSymbol),
                         std::forward<FindSymbolInFtor>(FindSymbolIn));
}


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
