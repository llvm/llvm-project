//===- MLIRContextTest.cpp - MLIR context tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ThreadPool.h"
#include "gtest/gtest.h"
#include <array>
#include <string>
#include <thread>

using namespace mlir;

namespace {
TEST(MLIRContextTest, LockingWithoutThreadPool) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  EXPECT_FALSE(context.isMultithreadingEnabled());
  EXPECT_FALSE(context.isThreadSafeLockingEnabled());

  context.enableThreadSafeLocking();
  EXPECT_EQ(context.isThreadSafeLockingEnabled(), LLVM_ENABLE_THREADS != 0);
  EXPECT_FALSE(context.isMultithreadingEnabled());
  EXPECT_EQ(context.getNumThreads(), 1u);

  // MLIR's parallel helpers must still execute on the calling thread, without
  // trying to access a thread pool.
  auto callingThread = std::this_thread::get_id();
  parallelFor(&context, 0, 16, [&](size_t) {
    EXPECT_EQ(std::this_thread::get_id(), callingThread);
  });

  context.disableMultithreading();
  EXPECT_FALSE(context.isThreadSafeLockingEnabled());
  EXPECT_FALSE(context.isMultithreadingEnabled());
}

TEST(MLIRContextTest, MultithreadingEnablesLocking) {
  MLIRContext context;
  EXPECT_EQ(context.isMultithreadingEnabled(), LLVM_ENABLE_THREADS != 0);
  EXPECT_EQ(context.isThreadSafeLockingEnabled(), LLVM_ENABLE_THREADS != 0);

  context.disableMultithreading();
  EXPECT_FALSE(context.isMultithreadingEnabled());
  EXPECT_FALSE(context.isThreadSafeLockingEnabled());

  context.enableMultithreading();
  EXPECT_EQ(context.isMultithreadingEnabled(), LLVM_ENABLE_THREADS != 0);
  EXPECT_EQ(context.isThreadSafeLockingEnabled(), LLVM_ENABLE_THREADS != 0);

  context.disableMultithreading();
  context.enableThreadSafeLocking();
  context.enableMultithreading();
  EXPECT_EQ(context.isMultithreadingEnabled(), LLVM_ENABLE_THREADS != 0);
  EXPECT_EQ(context.isThreadSafeLockingEnabled(), LLVM_ENABLE_THREADS != 0);

  context.disableMultithreading();
  EXPECT_FALSE(context.isThreadSafeLockingEnabled());
}

TEST(MLIRContextTest, ExternalThreadPoolEnablesLocking) {
  llvm::SingleThreadExecutor pool;
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.setThreadPool(pool);
  EXPECT_EQ(context.isMultithreadingEnabled(), LLVM_ENABLE_THREADS != 0);
  EXPECT_EQ(context.isThreadSafeLockingEnabled(), LLVM_ENABLE_THREADS != 0);
}

#if LLVM_ENABLE_THREADS
TEST(MLIRContextTest, ConcurrentAccessWithoutThreadPool) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.enableThreadSafeLocking();

  struct UniquedEntities {
    IntegerType type;
    StringAttr attribute;
    AffineExpr expression;
    OperationName operationName;
  };
  constexpr unsigned numThreads = 4;
  constexpr unsigned numEntities = 256;
  std::array<SmallVector<UniquedEntities>, numThreads> results;

  // Use an external pool directly: the context itself has no pool and MLIR's
  // parallel helpers would execute sequentially.
  llvm::DefaultThreadPool pool(llvm::hardware_concurrency(numThreads));
  for (unsigned thread = 0; thread < numThreads; ++thread) {
    pool.async([&, thread] {
      for (unsigned i = 0; i < numEntities; ++i) {
        // Use the same keys on every thread, including keys not yet present in
        // the context, to exercise concurrent insertion as well as lookup.
        results[thread].push_back(
            {IntegerType::get(&context, i + 129),
             StringAttr::get(&context, "external.attr" + std::to_string(i)),
             getAffineDimExpr(i, &context) +
                 getAffineConstantExpr(i + 1, &context),
             OperationName("external.op" + std::to_string(i), &context)});
      }
    });
  }
  pool.wait();

  for (unsigned i = 0; i < numEntities; ++i) {
    const auto &expected = results[0][i];
    for (unsigned thread = 1; thread < numThreads; ++thread) {
      const auto &actual = results[thread][i];
      EXPECT_EQ(actual.type, expected.type);
      EXPECT_EQ(actual.attribute, expected.attribute);
      EXPECT_EQ(actual.expression, expected.expression);
      EXPECT_EQ(actual.operationName, expected.operationName);
    }
  }
  EXPECT_FALSE(context.isMultithreadingEnabled());
  EXPECT_TRUE(context.isThreadSafeLockingEnabled());
}
#endif
} // namespace
