//===- ExecutionEngine.h - MLIR Execution engine and utils -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a JIT-backed execution engine for MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
#define MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>

namespace llvm {
template <typename T> class Expected;
class Module;
class ExecutionEngine;
class JITEventListener;
class MemoryBuffer;
} // namespace llvm

namespace mlir {

class ModuleOp;

/// A simple object cache following Lang's LLJITWithObjectCache example.
class SimpleObjectCache : public llvm::ObjectCache {
public:
  void notifyObjectCompiled(const llvm::Module *M,
                            llvm::MemoryBufferRef ObjBuffer) override;
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *M) override;

  /// Dump cached object to output file `filename`.
  void dumpToObjectFile(StringRef filename);

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cachedObjects;
};

/// JIT-backed execution engine for MLIR modules.  Assumes the module can be
/// converted to LLVM IR.  For each function, creates a wrapper function with
/// the fixed interface
///
///     void _mlir_funcName(void **)
///
/// where the only argument is interpreted as a list of pointers to the actual
/// arguments of the function, followed by a pointer to the result.  This allows
/// the engine to provide the caller with a generic function pointer that can
/// be used to invoke the JIT-compiled function.
class ExecutionEngine {
public:
  ExecutionEngine(bool enableObjectCache, bool enableGDBNotificationListener,
                  bool enablePerfNotificationListener);

  /// Creates an execution engine for the given module.
  ///
  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  ///
  /// `jitCodeGenOptLevel`, when provided, is used as the optimization level for
  /// target code generation.
  ///
  /// If `sharedLibPaths` are provided, the underlying JIT-compilation will
  /// open and link the shared libraries for symbol resolution.
  ///
  /// If `enableObjectCache` is set, the JIT compiler will create one to store
  /// the object generated for the given module.
  ///
  /// If enable `enableGDBNotificationListener` is set, the JIT compiler will
  /// notify the llvm's global GDB notification listener.
  ///
  /// If `enablePerfNotificationListener` is set, the JIT compiler will notify
  /// the llvm's global Perf notification listener.
  static llvm::Expected<std::unique_ptr<ExecutionEngine>>
  create(ModuleOp m,
         llvm::function_ref<llvm::Error(llvm::Module *)> transformer = {},
         Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel = llvm::None,
         ArrayRef<StringRef> sharedLibPaths = {}, bool enableObjectCache = true,
         bool enableGDBNotificationListener = true,
         bool enablePerfNotificationListener = true);

  /// Looks up a packed-argument function with the given name and returns a
  /// pointer to it.  Propagates errors in case of failure.
  llvm::Expected<void (*)(void **)> lookup(StringRef name) const;

  /// Invokes the function with the given name passing it the list of arguments
  /// as a list of opaque pointers.
  llvm::Error invoke(StringRef name, MutableArrayRef<void *> args = llvm::None);

  /// Set the target triple on the module. This is implicitly done when creating
  /// the engine.
  static bool setupTargetTriple(llvm::Module *llvmModule);

  /// Dump object code to output file `filename`.
  void dumpToObjectFile(StringRef filename);

  /// Register symbols with this ExecutionEngine.
  void registerSymbols(
      llvm::function_ref<llvm::orc::SymbolMap(llvm::orc::MangleAndInterner)>
          symbolMap);

private:
  /// Ordering of llvmContext and jit is important for destruction purposes: the
  /// jit must be destroyed before the context.
  llvm::LLVMContext llvmContext;

  /// Underlying LLJIT.
  std::unique_ptr<llvm::orc::LLJIT> jit;

  /// Underlying cache.
  std::unique_ptr<SimpleObjectCache> cache;

  /// GDB notification listener.
  llvm::JITEventListener *gdbListener;

  /// Perf notification listener.
  llvm::JITEventListener *perfListener;
};

} // end namespace mlir

#endif // MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
