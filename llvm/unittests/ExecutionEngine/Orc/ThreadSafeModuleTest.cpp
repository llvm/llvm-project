//===--- ThreadSafeModuleTest.cpp - Test basic use of ThreadSafeModule ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <atomic>
#include <future>
#include <thread>

using namespace llvm;
using namespace llvm::orc;

namespace {

const llvm::StringRef FooSrc = R"(
  define void @foo() {
    ret void
  }
)";

static ThreadSafeModule parseModule(llvm::StringRef Source,
                                    llvm::StringRef Name) {
  auto Ctx = std::make_unique<LLVMContext>();
  SMDiagnostic Err;
  auto M = parseIR(MemoryBufferRef(Source, Name), Err, *Ctx);
  if (!M) {
    Err.print("Testcase source failed to parse: ", errs());
    exit(1);
  }
  return ThreadSafeModule(std::move(M), std::move(Ctx));
}

TEST(ThreadSafeModuleTest, ContextWhollyOwnedByOneModule) {
  // Test that ownership of a context can be transferred to a single
  // ThreadSafeModule.
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("M", *Ctx);
  ThreadSafeModule TSM(std::move(M), std::move(Ctx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedByTwoModules) {
  // Test that ownership of a context can be shared between more than one
  // ThreadSafeModule.
  auto Ctx = std::make_unique<LLVMContext>();

  auto M1 = std::make_unique<Module>("M1", *Ctx);
  auto M2 = std::make_unique<Module>("M2", *Ctx);

  ThreadSafeContext TSCtx(std::move(Ctx));
  ThreadSafeModule TSM1(std::move(M1), TSCtx);
  ThreadSafeModule TSM2(std::move(M2), std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ContextOwnershipSharedWithClient) {
  // Test that ownership of a context can be shared with a client-held
  // ThreadSafeContext so that it can be re-used for new modules.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());

  {
    // Create and destroy a module.
    auto M1 = TSCtx.withContextDo(
        [](LLVMContext *Ctx) { return std::make_unique<Module>("M1", *Ctx); });
    ThreadSafeModule TSM1(std::move(M1), TSCtx);
  }

  // Verify that the context is still available for re-use.
  auto M2 = TSCtx.withContextDo(
      [](LLVMContext *Ctx) { return std::make_unique<Module>("M2", *Ctx); });
  ThreadSafeModule TSM2(std::move(M2), std::move(TSCtx));
}

TEST(ThreadSafeModuleTest, ThreadSafeModuleMoveAssignment) {
  // Move assignment needs to move the module before the context (opposite
  // to the field order) to ensure that overwriting with an empty
  // ThreadSafeModule does not destroy the context early.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  auto M = TSCtx.withContextDo(
      [](LLVMContext *Ctx) { return std::make_unique<Module>("M", *Ctx); });
  ThreadSafeModule TSM(std::move(M), std::move(TSCtx));
  TSM = ThreadSafeModule();
}

TEST(ThreadSafeModuleTest, WithContextDoPreservesContext) {
  // Test that withContextDo passes through the LLVMContext that was used
  // to create the ThreadSafeContext.

  auto Ctx = std::make_unique<LLVMContext>();
  LLVMContext *OriginalCtx = Ctx.get();
  ThreadSafeContext TSCtx(std::move(Ctx));
  TSCtx.withContextDo(
      [&](LLVMContext *ClosureCtx) { EXPECT_EQ(ClosureCtx, OriginalCtx); });
}

TEST(ThreadSafeModuleTest, WithModuleDo) {
  // Test non-const version of withModuleDo.
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("M", *Ctx);
  ThreadSafeModule TSM(std::move(M), std::move(Ctx));
  TSM.withModuleDo([](Module &M) {});
}

TEST(ThreadSafeModuleTest, WithModuleDoConst) {
  // Test const version of withModuleDo.
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("M", *Ctx);
  const ThreadSafeModule TSM(std::move(M), std::move(Ctx));
  TSM.withModuleDo([](const Module &M) {});
}

TEST(ThreadSafeModuleTest, ConsumingModuleDo) {
  // Test consumingModuleDo.
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = std::make_unique<Module>("M", *Ctx);
  ThreadSafeModule TSM(std::move(M), std::move(Ctx));
  TSM.consumingModuleDo([](std::unique_ptr<Module> M) {});
}

TEST(ThreadSafeModuleTest, CloneToNewContext) {
  auto TSM1 = parseModule(FooSrc, "foo.ll");
  auto TSM2 = cloneToNewContext(TSM1);
  TSM2.withModuleDo([&](Module &NewM) {
    EXPECT_FALSE(verifyModule(NewM, &errs()));
    TSM1.withModuleDo([&](Module &OrigM) {
      EXPECT_NE(&NewM.getContext(), &OrigM.getContext());
    });
  });
}

TEST(ObjectFormatsTest, CloneToContext) {
  auto TSM1 = parseModule(FooSrc, "foo.ll");

  auto TSCtx = ThreadSafeContext(std::make_unique<LLVMContext>());
  auto TSM2 = cloneToContext(TSM1, TSCtx);

  TSM2.withModuleDo([&](Module &M) {
    EXPECT_FALSE(verifyModule(M, &errs()));
    TSCtx.withContextDo(
        [&](LLVMContext *Ctx) { EXPECT_EQ(&M.getContext(), Ctx); });
  });
}

} // end anonymous namespace
