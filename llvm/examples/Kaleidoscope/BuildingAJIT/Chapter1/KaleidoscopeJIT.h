//===- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

namespace llvm {
namespace orc {

class KaleidoscopeJIT {
private:

  ExecutionSession ES;
  RTDyldObjectLinkingLayer ObjectLayer{ES, getMemoryMgr};
  IRCompileLayer CompileLayer{ES, ObjectLayer,
                              ConcurrentIRCompiler(getJTMB())};
  DataLayout DL{cantFail(getJTMB().getDefaultDataLayoutForTarget())};
  MangleAndInterner Mangle{ES, DL};
  ThreadSafeContext Ctx{llvm::make_unique<LLVMContext>()};

  static JITTargetMachineBuilder getJTMB() {
    return cantFail(JITTargetMachineBuilder::detectHost());
  }

  static std::unique_ptr<SectionMemoryManager> getMemoryMgr() {
    return llvm::make_unique<SectionMemoryManager>();
  }

public:

  KaleidoscopeJIT() {
    ES.getMainJITDylib().setGenerator(
      cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(DL)));
  }

  const DataLayout &getDataLayout() const { return DL; }

  LLVMContext &getContext() { return *Ctx.getContext(); }

  void addModule(std::unique_ptr<Module> M) {
    cantFail(CompileLayer.add(ES.getMainJITDylib(),
                              ThreadSafeModule(std::move(M), Ctx)));
  }

  Expected<JITEvaluatedSymbol> lookup(StringRef Name) {
    return ES.lookup({&ES.getMainJITDylib()}, Mangle(Name.str()));
  }
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
