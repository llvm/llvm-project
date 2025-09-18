//===----------- ThreadSafeModule.h -- Layer interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thread safe wrappers and utilities for Module and LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULE_H
#define LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULE_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Compiler.h"

#include <functional>
#include <memory>
#include <mutex>

namespace llvm {
namespace orc {

/// An LLVMContext together with an associated mutex that can be used to lock
/// the context to prevent concurrent access by other threads.
class ThreadSafeContext {
private:
  struct State {
    State(std::unique_ptr<LLVMContext> Ctx) : Ctx(std::move(Ctx)) {}

    std::unique_ptr<LLVMContext> Ctx;
    std::recursive_mutex Mutex;
  };

public:
  /// Construct a null context.
  ThreadSafeContext() = default;

  /// Construct a ThreadSafeContext from the given LLVMContext.
  ThreadSafeContext(std::unique_ptr<LLVMContext> NewCtx)
      : S(std::make_shared<State>(std::move(NewCtx))) {
    assert(S->Ctx != nullptr &&
           "Can not construct a ThreadSafeContext from a nullptr");
  }

  template <typename Func> decltype(auto) withContextDo(Func &&F) {
    if (auto TmpS = S) {
      std::lock_guard<std::recursive_mutex> Lock(TmpS->Mutex);
      return F(TmpS->Ctx.get());
    } else
      return F((LLVMContext *)nullptr);
  }

  template <typename Func> decltype(auto) withContextDo(Func &&F) const {
    if (auto TmpS = S) {
      std::lock_guard<std::recursive_mutex> Lock(TmpS->Mutex);
      return F(const_cast<const LLVMContext *>(TmpS->Ctx.get()));
    } else
      return F((const LLVMContext *)nullptr);
  }

private:
  std::shared_ptr<State> S;
};

/// An LLVM Module together with a shared ThreadSafeContext.
class ThreadSafeModule {
public:
  /// Default construct a ThreadSafeModule. This results in a null module and
  /// null context.
  ThreadSafeModule() = default;

  ThreadSafeModule(ThreadSafeModule &&Other) = default;

  ThreadSafeModule &operator=(ThreadSafeModule &&Other) {
    // We have to explicitly define this move operator to copy the fields in
    // reverse order (i.e. module first) to ensure the dependencies are
    // protected: The old module that is being overwritten must be destroyed
    // *before* the context that it depends on.
    // We also need to lock the context to make sure the module tear-down
    // does not overlap any other work on the context.
    TSCtx.withContextDo([this](LLVMContext *Ctx) { M = nullptr; });
    M = std::move(Other.M);
    TSCtx = std::move(Other.TSCtx);
    return *this;
  }

  /// Construct a ThreadSafeModule from a unique_ptr<Module> and a
  /// unique_ptr<LLVMContext>. This creates a new ThreadSafeContext from the
  /// given context.
  ThreadSafeModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> Ctx)
      : M(std::move(M)), TSCtx(std::move(Ctx)) {}

  /// Construct a ThreadSafeModule from a unique_ptr<Module> and an
  /// existing ThreadSafeContext.
  ThreadSafeModule(std::unique_ptr<Module> M, ThreadSafeContext TSCtx)
      : M(std::move(M)), TSCtx(std::move(TSCtx)) {}

  ~ThreadSafeModule() {
    // We need to lock the context while we destruct the module.
    TSCtx.withContextDo([this](LLVMContext *Ctx) { M = nullptr; });
  }

  /// Boolean conversion: This ThreadSafeModule will evaluate to true if it
  /// wraps a non-null module.
  explicit operator bool() const { return !!M; }

  /// Locks the associated ThreadSafeContext and calls the given function
  /// on the contained Module.
  template <typename Func> decltype(auto) withModuleDo(Func &&F) {
    return TSCtx.withContextDo([&](LLVMContext *) {
      assert(M && "Can not call on null module");
      return F(*M);
    });
  }

  /// Locks the associated ThreadSafeContext and calls the given function
  /// on the contained Module.
  template <typename Func> decltype(auto) withModuleDo(Func &&F) const {
    return TSCtx.withContextDo([&](const LLVMContext *) {
      assert(M && "Can not call on null module");
      return F(*M);
    });
  }

  /// Locks the associated ThreadSafeContext and calls the given function,
  /// passing the contained std::unique_ptr<Module>. The given function should
  /// consume the Module.
  template <typename Func> decltype(auto) consumingModuleDo(Func &&F) {
    return TSCtx.withContextDo([&](LLVMContext *) {
      assert(M && "Can not call on null module");
      return F(std::move(M));
    });
  }

  /// Get a raw pointer to the contained module without locking the context.
  Module *getModuleUnlocked() { return M.get(); }

  /// Get a raw pointer to the contained module without locking the context.
  const Module *getModuleUnlocked() const { return M.get(); }

  /// Returns the context for this ThreadSafeModule.
  ThreadSafeContext getContext() const { return TSCtx; }

private:
  std::unique_ptr<Module> M;
  ThreadSafeContext TSCtx;
};

using GVPredicate = std::function<bool(const GlobalValue &)>;
using GVModifier = std::function<void(GlobalValue &)>;

/// Clones the given module onto the given context.
LLVM_ABI ThreadSafeModule
cloneToContext(const ThreadSafeModule &TSMW, ThreadSafeContext TSCtx,
               GVPredicate ShouldCloneDef = GVPredicate(),
               GVModifier UpdateClonedDefSource = GVModifier());

/// Clone the given module onto the given context.
/// The caller is responsible for ensuring that the source module and its
/// LLVMContext will not be concurrently accessed during the clone.
LLVM_ABI ThreadSafeModule
cloneExternalModuleToContext(const Module &M, ThreadSafeContext TSCtx,
                             GVPredicate ShouldCloneDef = GVPredicate(),
                             GVModifier UpdateClonedDefSource = GVModifier());

/// Clones the given module on to a new context.
LLVM_ABI ThreadSafeModule cloneToNewContext(
    const ThreadSafeModule &TSMW, GVPredicate ShouldCloneDef = GVPredicate(),
    GVModifier UpdateClonedDefSource = GVModifier());

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULE_H
