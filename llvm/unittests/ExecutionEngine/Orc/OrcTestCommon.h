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

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/JITSymbol.h"
#include "llvm/Support/TargetSelect.h"
#include <memory>

namespace llvm {

// Base class for Orc tests that will execute code.
class OrcExecutionTest {
public:

  OrcExecutionTest() {
    if (!NativeTargetInitialized) {
      InitializeNativeTarget();
      InitializeNativeTargetAsmParser();
      InitializeNativeTargetAsmPrinter();
      NativeTargetInitialized = true;
    }

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
  std::unique_ptr<TargetMachine> TM;
private:
  static bool NativeTargetInitialized;
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
      TypeBuilder<types::i<32>[256], XCompile>::get(Context), nullptr);
  }
};

template <typename HandleT,
          typename AddModuleSetFtor,
          typename RemoveModuleSetFtor,
          typename FindSymbolFtor,
          typename FindSymbolInFtor>
class MockBaseLayer {
public:

  typedef HandleT ModuleSetHandleT;

  MockBaseLayer(AddModuleSetFtor &&AddModuleSet,
                RemoveModuleSetFtor &&RemoveModuleSet,
                FindSymbolFtor &&FindSymbol,
                FindSymbolInFtor &&FindSymbolIn)
      : AddModuleSet(AddModuleSet), RemoveModuleSet(RemoveModuleSet),
        FindSymbol(FindSymbol), FindSymbolIn(FindSymbolIn)
  {}

  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms, MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {
    return AddModuleSet(std::move(Ms), std::move(MemMgr), std::move(Resolver));
  }

  void removeModuleSet(ModuleSetHandleT H) {
    RemoveModuleSet(H);
  }

  orc::JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return FindSymbol(Name, ExportedSymbolsOnly);
  }

  orc::JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return FindSymbolIn(H, Name, ExportedSymbolsOnly);
  }

private:
  AddModuleSetFtor AddModuleSet;
  RemoveModuleSetFtor RemoveModuleSet;
  FindSymbolFtor FindSymbol;
  FindSymbolInFtor FindSymbolIn;
};

template <typename ModuleSetHandleT,
          typename AddModuleSetFtor,
          typename RemoveModuleSetFtor,
          typename FindSymbolFtor,
          typename FindSymbolInFtor>
MockBaseLayer<ModuleSetHandleT, AddModuleSetFtor, RemoveModuleSetFtor,
              FindSymbolFtor, FindSymbolInFtor>
createMockBaseLayer(AddModuleSetFtor &&AddModuleSet,
                    RemoveModuleSetFtor &&RemoveModuleSet,
                    FindSymbolFtor &&FindSymbol,
                    FindSymbolInFtor &&FindSymbolIn) {
  return MockBaseLayer<ModuleSetHandleT, AddModuleSetFtor, RemoveModuleSetFtor,
                       FindSymbolFtor, FindSymbolInFtor>(
                         std::forward<AddModuleSetFtor>(AddModuleSet),
                         std::forward<RemoveModuleSetFtor>(RemoveModuleSet),
                         std::forward<FindSymbolFtor>(FindSymbol),
                         std::forward<FindSymbolInFtor>(FindSymbolIn));
}

template <typename ReturnT>
class DoNothingAndReturn {
public:
  DoNothingAndReturn(ReturnT Val) : Val(Val) {}

  template <typename... Args>
  ReturnT operator()(Args...) const { return Val; }
private:
  ReturnT Val;
};

template <>
class DoNothingAndReturn<void> {
public:
  template <typename... Args>
  void operator()(Args...) const { }
};

} // namespace llvm

#endif
