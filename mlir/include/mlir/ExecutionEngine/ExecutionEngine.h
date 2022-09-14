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
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>

namespace llvm {
template <typename T>
class Expected;
class Module;
class ExecutionEngine;
class JITEventListener;
class MemoryBuffer;
} // namespace llvm

namespace mlir {

class Operation;

/// A simple object cache following Lang's LLJITWithObjectCache example.
class SimpleObjectCache : public llvm::ObjectCache {
public:
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef objBuffer) override;
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *m) override;

  /// Dump cached object to output file `filename`.
  void dumpToObjectFile(StringRef filename);

private:
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> cachedObjects;
};

struct ExecutionEngineOptions {
  /// If `llvmModuleBuilder` is provided, it will be used to create an LLVM
  /// module from the given MLIR IR. Otherwise, a default
  /// `translateModuleToLLVMIR` function will be used to translate to LLVM IR.
  llvm::function_ref<std::unique_ptr<llvm::Module>(Operation *,
                                                   llvm::LLVMContext &)>
      llvmModuleBuilder = nullptr;

  /// If `transformer` is provided, it will be called on the LLVM module during
  /// JIT-compilation and can be used, e.g., for reporting or optimization.
  llvm::function_ref<llvm::Error(llvm::Module *)> transformer = {};

  /// `jitCodeGenOptLevel`, when provided, is used as the optimization level for
  /// target code generation.
  Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel = llvm::None;

  /// If `sharedLibPaths` are provided, the underlying JIT-compilation will
  /// open and link the shared libraries for symbol resolution.
  ArrayRef<StringRef> sharedLibPaths = {};

  /// Specifies an existing `sectionMemoryMapper` to be associated with the
  /// compiled code. If none is provided, a default memory mapper that directly
  /// calls into the operating system is used.
  llvm::SectionMemoryManager::MemoryMapper *sectionMemoryMapper = nullptr;

  /// If `enableObjectCache` is set, the JIT compiler will create one to store
  /// the object generated for the given module. The contents of the cache can
  /// be dumped to a file via the `dumpToObjectfile` method.
  bool enableObjectCache = false;

  /// If enable `enableGDBNotificationListener` is set, the JIT compiler will
  /// notify the llvm's global GDB notification listener.
  bool enableGDBNotificationListener = true;

  /// If `enablePerfNotificationListener` is set, the JIT compiler will notify
  /// the llvm's global Perf notification listener.
  bool enablePerfNotificationListener = true;
};

/// JIT-backed execution engine for MLIR. Assumes the IR can be converted to
/// LLVM IR. For each function, creates a wrapper function with the fixed
/// interface
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

  /// Creates an execution engine for the given MLIR IR.
  static llvm::Expected<std::unique_ptr<ExecutionEngine>>
  create(Operation *op, const ExecutionEngineOptions &options = {});

  /// Looks up a packed-argument function wrapping the function with the given
  /// name and returns a pointer to it. Propagates errors in case of failure.
  llvm::Expected<void (*)(void **)> lookupPacked(StringRef name) const;

  /// Looks up the original function with the given name and returns a
  /// pointer to it. This is not necesarily a packed function. Propagates
  /// errors in case of failure.
  llvm::Expected<void *> lookup(StringRef name) const;

  /// Invokes the function with the given name passing it the list of opaque
  /// pointers to the actual arguments.
  llvm::Error invokePacked(StringRef name,
                           MutableArrayRef<void *> args = llvm::None);

  /// Trait that defines how a given type is passed to the JIT code. This
  /// defaults to passing the address but can be specialized.
  template <typename T>
  struct Argument {
    static void pack(SmallVectorImpl<void *> &args, T &val) {
      args.push_back(&val);
    }
  };

  /// Tag to wrap an output parameter when invoking a jitted function.
  template <typename T>
  struct Result {
    Result(T &result) : value(result) {}
    T &value;
  };

  /// Helper function to wrap an output operand when using
  /// ExecutionEngine::invoke.
  template <typename T>
  static Result<T> result(T &t) {
    return Result<T>(t);
  }

  // Specialization for output parameter: their address is forwarded directly to
  // the native code.
  template <typename T>
  struct Argument<Result<T>> {
    static void pack(SmallVectorImpl<void *> &args, Result<T> &result) {
      args.push_back(&result.value);
    }
  };

  /// Invokes the function with the given name passing it the list of arguments
  /// by value. Function result can be obtain through output parameter using the
  /// `Result` wrapper defined above. For example:
  ///
  ///     func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface }
  ///
  /// can be invoked:
  ///
  ///     int32_t result = 0;
  ///     llvm::Error error = jit->invoke("foo", 42,
  ///                                     result(result));
  template <typename... Args>
  llvm::Error invoke(StringRef funcName, Args... args) {
    const std::string adapterName =
        std::string("_mlir_ciface_") + funcName.str();
    llvm::SmallVector<void *> argsArray;
    // Pack every arguments in an array of pointers. Delegate the packing to a
    // trait so that it can be overridden per argument type.
    (Argument<Args>::pack(argsArray, args), ...);
    return invokePacked(adapterName, argsArray);
  }

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

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
