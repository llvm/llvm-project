//===----------- ThreadSafeModule.h -- Layer interfaces ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Thread safe wrappers and utilities for Module and LLVMContext.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULEWRAPPER_H
#define LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULEWRAPPER_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

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
    State(std::unique_ptr<LLVMContext> Ctx)
      : Ctx(std::move(Ctx)) {}

    std::unique_ptr<LLVMContext> Ctx;
    std::recursive_mutex Mutex;
  };

public:

  // RAII based lock for ThreadSafeContext.
  class Lock {
  private:
    using UnderlyingLock = std::lock_guard<std::recursive_mutex>;
  public:

    Lock(std::shared_ptr<State> S)
      : S(std::move(S)),
        L(llvm::make_unique<UnderlyingLock>(this->S->Mutex)) {}
  private:
    std::shared_ptr<State> S;
    std::unique_ptr<UnderlyingLock> L;
  };

  /// Construct a null context.
  ThreadSafeContext() = default;

  /// Construct a ThreadSafeContext from the given LLVMContext.
  ThreadSafeContext(std::unique_ptr<LLVMContext> NewCtx)
      : S(std::make_shared<State>(std::move(NewCtx))) {
    assert(S->Ctx != nullptr &&
           "Can not construct a ThreadSafeContext from a nullptr");
  }

  /// Returns a pointer to the LLVMContext that was used to construct this
  /// instance, or null if the instance was default constructed.
  LLVMContext* getContext() {
    return S ? S->Ctx.get() : nullptr;
  }

  Lock getLock() {
    assert(S && "Can not lock an empty ThreadSafeContext");
    return Lock(S);
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

  /// Construct a ThreadSafeModule from a unique_ptr<Module> and a
  /// unique_ptr<LLVMContext>. This creates a new ThreadSafeContext from the
  /// given context.
  ThreadSafeModule(std::unique_ptr<Module> M,
                   std::unique_ptr<LLVMContext> Ctx)
    : M(std::move(M)), TSCtx(std::move(Ctx)) {}

  ThreadSafeModule(std::unique_ptr<Module> M,
                   ThreadSafeContext TSCtx)
    : M(std::move(M)), TSCtx(std::move(TSCtx)) {}

  Module* getModule() { return M.get(); }

  ThreadSafeContext::Lock getContextLock() { return TSCtx.getLock(); }

  explicit operator bool() {
    if (M) {
      assert(TSCtx.getContext() && "Non-null module must have non-null context");
      return true;
    }
    return false;
  }

private:
  std::unique_ptr<Module> M;
  ThreadSafeContext TSCtx;
};

using GVPredicate = std::function<bool(const GlobalValue&)>;
using GVModifier = std::function<void(GlobalValue&)>;

/// Clones the given module on to a new context.
ThreadSafeModule
cloneToNewContext(ThreadSafeModule &TSMW,
                  GVPredicate ShouldCloneDef = GVPredicate(),
                  GVModifier UpdateClonedDefSource = GVModifier());

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_THREADSAFEMODULEWRAPPER_H
