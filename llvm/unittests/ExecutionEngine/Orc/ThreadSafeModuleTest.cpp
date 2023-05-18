//===--- ThreadSafeModuleTest.cpp - Test basic use of ThreadSafeModule ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "gtest/gtest.h"

#include <atomic>
#include <future>
#include <thread>

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(ThreadSafeModuleTest, ContextWhollyOwnedByOneModule) {
  // Test that ownership of a context can be transferred to a single
  // ThreadSafeModule.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  auto M = std::make_unique<Module>("M", *TSCtx.getContext());
  ThreadSafeModule TSM(std::move(M), std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedByTwoModules) {
  // Test that ownership of a context can be shared between more than one
  // ThreadSafeModule.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());

  auto M1 = std::make_unique<Module>("M1", *TSCtx.getContext());
  ThreadSafeModule TSM1(std::move(M1), TSCtx);

  auto M2 = std::make_unique<Module>("M2", *TSCtx.getContext());
  ThreadSafeModule TSM2(std::move(M2), std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedWithClient) {
  // Test that ownership of a context can be shared with a client-held
  // ThreadSafeContext so that it can be re-used for new modules.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());

  {
    // Create and destroy a module.
    auto M1 = std::make_unique<Module>("M1", *TSCtx.getContext());
    ThreadSafeModule TSM1(std::move(M1), TSCtx);
  }

  // Verify that the context is still available for re-use.
  auto M2 = std::make_unique<Module>("M2", *TSCtx.getContext());
  ThreadSafeModule TSM2(std::move(M2), std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ThreadSafeModuleMoveAssignment) {
  // Move assignment needs to move the module before the context (opposite
  // to the field order) to ensure that overwriting with an empty
  // ThreadSafeModule does not destroy the context early.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  auto M = std::make_unique<Module>("M", *TSCtx.getContext());
  ThreadSafeModule TSM(std::move(M), std::move(TSCtx));
  TSM = ThreadSafeModule();
}

TEST(ThreadSafeModuleTest, BasicContextLockAPI) {
  // Test that basic lock API calls work.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  auto M = std::make_unique<Module>("M", *TSCtx.getContext());
  ThreadSafeModule TSM(std::move(M), TSCtx);

  { auto L = TSCtx.getLock(); }

  { auto L = TSM.getContext().getLock(); }
}

TEST(ThreadSafeModuleTest, ContextLockPreservesContext) {
  // Test that the existence of a context lock preserves the attached
  // context.
  // The trick to verify this is a bit of a hack: We attach a Module
  // (without the ThreadSafeModule wrapper) to the context, then verify
  // that this Module destructs safely (which it will not if its context
  // has been destroyed) even though all references to the context have
  // been thrown away (apart from the lock).

  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  auto L = TSCtx.getLock();
  auto &Ctx = *TSCtx.getContext();
  auto M = std::make_unique<Module>("M", Ctx);
  TSCtx = ThreadSafeContext();
}

TEST(ThreadSafeModuleTest, WithModuleDo) {
  // Test non-const version of withModuleDo.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  ThreadSafeModule TSM(std::make_unique<Module>("M", *TSCtx.getContext()),
                       TSCtx);
  TSM.withModuleDo([](Module &M) {});
}

TEST(ThreadSafeModuleTest, WithModuleDoConst) {
  // Test const version of withModuleDo.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  const ThreadSafeModule TSM(std::make_unique<Module>("M", *TSCtx.getContext()),
                             TSCtx);
  TSM.withModuleDo([](const Module &M) {});
}

TEST(ThreadSafeModuleTest, ConsumingModuleDo) {
  // Test consumingModuleDo.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  ThreadSafeModule TSM(std::make_unique<Module>("M", *TSCtx.getContext()),
                       TSCtx);
  TSM.consumingModuleDo([](std::unique_ptr<Module> M) {});
}

} // end anonymous namespace
