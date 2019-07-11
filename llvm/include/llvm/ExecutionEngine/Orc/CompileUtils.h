//===- CompileUtils.h - Utilities for compiling IR in the JIT ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for compiling IR to object files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include <memory>

namespace llvm {

class MCContext;
class MemoryBuffer;
class Module;
class ObjectCache;
class TargetMachine;

namespace orc {

class JITTargetMachineBuilder;

/// Simple compile functor: Takes a single IR module and returns an ObjectFile.
/// This compiler supports a single compilation thread and LLVMContext only.
/// For multithreaded compilation, use ConcurrentIRCompiler below.
class SimpleCompiler {
public:
  using CompileResult = std::unique_ptr<MemoryBuffer>;

  /// Construct a simple compile functor with the given target.
  SimpleCompiler(TargetMachine &TM, ObjectCache *ObjCache = nullptr)
    : TM(TM), ObjCache(ObjCache) {}

  /// Set an ObjectCache to query before compiling.
  void setObjectCache(ObjectCache *NewCache) { ObjCache = NewCache; }

  /// Compile a Module to an ObjectFile.
  CompileResult operator()(Module &M);

private:
  CompileResult tryToLoadFromObjectCache(const Module &M);
  void notifyObjectCompiled(const Module &M, const MemoryBuffer &ObjBuffer);

  TargetMachine &TM;
  ObjectCache *ObjCache = nullptr;
};

/// A SimpleCompiler that owns its TargetMachine.
///
/// This convenient for clients who don't want to own their TargetMachines,
/// e.g. LLJIT.
class TMOwningSimpleCompiler : public SimpleCompiler {
public:
  TMOwningSimpleCompiler(std::unique_ptr<TargetMachine> TM,
                         ObjectCache *ObjCache = nullptr)
      : SimpleCompiler(*TM, ObjCache), TM(std::move(TM)) {}

private:
  // FIXME: shared because std::functions (and consequently
  // IRCompileLayer::CompileFunction) are not moveable.
  std::shared_ptr<llvm::TargetMachine> TM;
};

/// A thread-safe version of SimpleCompiler.
///
/// This class creates a new TargetMachine and SimpleCompiler instance for each
/// compile.
class ConcurrentIRCompiler {
public:
  ConcurrentIRCompiler(JITTargetMachineBuilder JTMB,
                       ObjectCache *ObjCache = nullptr);

  void setObjectCache(ObjectCache *ObjCache) { this->ObjCache = ObjCache; }

  std::unique_ptr<MemoryBuffer> operator()(Module &M);

private:
  JITTargetMachineBuilder JTMB;
  ObjectCache *ObjCache = nullptr;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEUTILS_H
