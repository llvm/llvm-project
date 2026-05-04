//===-- EJitRuntimeTest.cpp - EmbeddedJIT Runtime Unit Tests ---------------===//
//
// NOTE: To call the C API functions, we need to include the C runtime header
// which is in the non-canonical include path. We declare the symbols we need
// via extern "C" declarations instead, since they're provided by libLLVMEJIT.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJit.h"
#include "llvm/ExecutionEngine/EJIT/EJitCache.h"
#include "llvm/ExecutionEngine/EJIT/EJitCommon.h"
#include "llvm/ExecutionEngine/EJIT/EJitLogger.h"
#include "llvm/ExecutionEngine/EJIT/EJitModuleLoader.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptions.h"
#include "llvm/ExecutionEngine/EJIT/EJitOptimizer.h"
#include "llvm/ExecutionEngine/EJIT/EJitRegistrationStore.h"
#include "llvm/ExecutionEngine/EJIT/EJitRuntimeState.h"
#include "llvm/ExecutionEngine/EJIT/EJitStructFieldPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"
#include <thread>

using namespace llvm;
using namespace llvm::ejit;

//===----------------------------------------------------------------------===//
// EJitRegistrationStore tests (T3-08)
//===----------------------------------------------------------------------===//

TEST(EJitRegistrationStore, RegisterAndConsumeBitcode) {
  EJitRegistrationStore &store = EJitRegistrationStore::instance();
  // consume any leftover data from previous tests
  store.consume();

  const uint8_t data[] = {0x01, 0x02, 0x03};
  store.registerBitcode("func_a", data, sizeof(data));

  StoredData result = store.consume();
  ASSERT_EQ(result.bitcodes.size(), 1u);
  EXPECT_EQ(result.bitcodes[0].funcName, "func_a");
  EXPECT_EQ(result.bitcodes[0].size, 3u);
  EXPECT_EQ(result.bitcodes[0].data[0], 0x01);

  // consume again should be empty
  StoredData empty = store.consume();
  EXPECT_TRUE(empty.bitcodes.empty());
  EXPECT_TRUE(empty.periodArrays.empty());
  EXPECT_TRUE(empty.staticVars.empty());
}

TEST(EJitRegistrationStore, RegisterAndConsumePeriodArrays) {
  EJitRegistrationStore &store = EJitRegistrationStore::instance();
  store.consume();

  int arr[10];
  store.registerPeriodArray("cell", "cells", arr, 10);
  store.registerPeriodArray("trp", "trps", arr, 5);

  StoredData result = store.consume();
  ASSERT_EQ(result.periodArrays.size(), 2u);
  EXPECT_EQ(result.periodArrays[0].periodName, "cell");
  EXPECT_EQ(result.periodArrays[0].varName, "cells");
  EXPECT_EQ(result.periodArrays[0].arraySize, 10u);
  EXPECT_EQ(result.periodArrays[1].periodName, "trp");
}

TEST(EJitRegistrationStore, RegisterAndConsumeStaticVars) {
  EJitRegistrationStore &store = EJitRegistrationStore::instance();
  store.consume();

  int val = 42;
  store.registerStaticVar("config", &val);

  StoredData result = store.consume();
  ASSERT_EQ(result.staticVars.size(), 1u);
  EXPECT_EQ(result.staticVars[0].varName, "config");
  EXPECT_EQ(result.staticVars[0].varAddr, &val);
}

TEST(EJitRegistrationStore, ConsumeClearsAllTypes) {
  EJitRegistrationStore &store = EJitRegistrationStore::instance();
  store.consume();

  const uint8_t d[] = {0};
  store.registerBitcode("f", d, 1);
  int arr;
  store.registerPeriodArray("p", "v", &arr, 1);
  int val;
  store.registerStaticVar("s", &val);

  StoredData result = store.consume();
  EXPECT_EQ(result.bitcodes.size(), 1u);
  EXPECT_EQ(result.periodArrays.size(), 1u);
  EXPECT_EQ(result.staticVars.size(), 1u);
}

TEST(EJitRegistrationStore, ThreadSafety) {
  EJitRegistrationStore &store = EJitRegistrationStore::instance();
  store.consume();

  std::thread t1([&]() {
    for (int i = 0; i < 100; ++i) {
      const uint8_t d[] = {0};
      store.registerBitcode("t1_func", d, 1);
    }
  });
  std::thread t2([&]() {
    for (int i = 0; i < 100; ++i) {
      int arr;
      store.registerPeriodArray("t2_period", "t2_var", &arr, 1);
    }
  });

  t1.join();
  t2.join();

  StoredData result = store.consume();
  EXPECT_EQ(result.bitcodes.size() + result.periodArrays.size(), 200u);
}

//===----------------------------------------------------------------------===//
// EJitModuleLoader tests (T3-09)
//===----------------------------------------------------------------------===//

TEST(EJitModuleLoader, RegisterAndGetBitcode) {
  EJitModuleLoader loader;
  const uint8_t data[] = {0xAA, 0xBB, 0xCC, 0xDD};
  loader.registerBitcode("my_func", data, sizeof(data));

  auto result = loader.getBitcode("my_func");
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->size(), 4u);
  EXPECT_EQ((uint8_t)(*result)[0], 0xAA);
}

TEST(EJitModuleLoader, GetBitcodeNotFound) {
  EJitModuleLoader loader;
  auto result = loader.getBitcode("nonexistent");
  EXPECT_FALSE(static_cast<bool>(result));
  // Consume the error to avoid unchecked-Expected assertion at destruction
  if (!result)
    llvm::consumeError(result.takeError());
}

TEST(EJitModuleLoader, MultipleFunctions) {
  EJitModuleLoader loader;
  const uint8_t d1[] = {0x11};
  const uint8_t d2[] = {0x22, 0x33};
  loader.registerBitcode("f1", d1, sizeof(d1));
  loader.registerBitcode("f2", d2, sizeof(d2));

  EXPECT_EQ(loader.getEntryCount(), 2u);

  auto r1 = loader.getBitcode("f1");
  ASSERT_TRUE(static_cast<bool>(r1));
  EXPECT_EQ(r1->size(), 1u);

  auto r2 = loader.getBitcode("f2");
  ASSERT_TRUE(static_cast<bool>(r2));
  EXPECT_EQ(r2->size(), 2u);
}

TEST(EJitModuleLoader, GetTotalBitcodeSize) {
  EJitModuleLoader loader;
  loader.registerBitcode("a", nullptr, 100);
  loader.registerBitcode("b", nullptr, 200);
  EXPECT_EQ(loader.getTotalBitcodeSize(), 300u);
}

//===----------------------------------------------------------------------===//
// EJitCache tests (T3-10)
//===----------------------------------------------------------------------===//

TEST(EJitCache, BasicPutAndGet) {
  EJitCache cache(10, 1024 * 1024);
  int dummy = 0;
  cache.put("key1", &dummy, 64);
  EXPECT_EQ(cache.getOrNull("key1"), &dummy);
  EXPECT_EQ(cache.getOrNull("nonexistent"), nullptr);
}

TEST(EJitCache, StatsTracking) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;
  cache.put("k1", &dummy, 64 );
  cache.getOrNull("k1");  // hit
  cache.getOrNull("k1");  // hit
  cache.getOrNull("k2");  // miss

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 1u);
  EXPECT_EQ(stats.hits, 2ull);
  EXPECT_EQ(stats.misses, 1ull);
}

TEST(EJitCache, LRUEvictionByEntryCount) {
  EJitCache cache(2, 1024 * 1024);
  int a, b, c;

  EXPECT_TRUE(cache.put("a", &a, 1));
  EXPECT_TRUE(cache.put("b", &b, 1));
  EXPECT_TRUE(cache.put("c", &c, 1)); // should evict 'a'

  EXPECT_EQ(cache.getOrNull("a"), nullptr);
  EXPECT_EQ(cache.getOrNull("b"), &b);
  EXPECT_EQ(cache.getOrNull("c"), &c);

  auto stats = cache.getStats();
  EXPECT_EQ(stats.evictions, 1ull);
}

TEST(EJitCache, LRUEvictionByTotalSize) {
  EJitCache cache(100, 200);
  int dummy;

  cache.put("a", &dummy, 120); // ok
  cache.put("b", &dummy, 90);  // should evict 'a'

  EXPECT_EQ(cache.getOrNull("a"), nullptr);
  EXPECT_EQ(cache.getOrNull("b"), &dummy);
}

TEST(EJitCache, SingleFuncSizeLimit) {
  EJitCache cache(10, 1024 * 1024, 100);
  int dummy;
  EXPECT_FALSE(cache.put("too_big", &dummy, 200));
  EXPECT_TRUE(cache.put("ok", &dummy, 50));

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 1u);
}

TEST(EJitCache, PeriodicInvalidation) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;

  std::set<std::string> depsA = {"cell=0"};
  std::set<std::string> depsB = {"cell=1"};
  std::set<std::string> depsC = {"trp=0"};

  cache.put("a", &dummy, 1, depsA);
  cache.put("b", &dummy, 1, depsB);
  cache.put("c", &dummy, 1, depsC);

  EXPECT_EQ(cache.getOrNull("a"), &dummy);
  EXPECT_EQ(cache.getOrNull("b"), &dummy);
  EXPECT_EQ(cache.getOrNull("c"), &dummy);

  cache.invalidateByPeriod("cell", 0);
  EXPECT_EQ(cache.getOrNull("a"), nullptr);  // invalidated
  EXPECT_EQ(cache.getOrNull("b"), &dummy);   // still valid (cell=1)
  EXPECT_EQ(cache.getOrNull("c"), &dummy);   // still valid (trp=0)
}

TEST(EJitCache, BuildCacheKey) {
  // No dimensions
  std::string key0 = EJitCache::buildCacheKey("myfunc", nullptr, 0);
  EXPECT_EQ(key0, "myfunc");

  // Single dimension
  std::pair<std::string, uint8_t> dims1[] = {{"cell", 3}};
  std::string key1 = EJitCache::buildCacheKey("myfunc", dims1, 1);
  EXPECT_EQ(key1, "myfunc|cell=3");

  // Multiple dimensions, sorted by name
  std::pair<std::string, uint8_t> dims2[] = {{"trp", 1}, {"cell", 5}};
  std::string key2 = EJitCache::buildCacheKey("myfunc", dims2, 2);
  EXPECT_EQ(key2, "myfunc|cell=5,trp=1");
}

TEST(EJitCache, Clear) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;
  cache.put("a", &dummy, 64 );
  cache.put("b", &dummy, 64 );
  cache.clear();

  EXPECT_EQ(cache.getOrNull("a"), nullptr);
  EXPECT_EQ(cache.getOrNull("b"), nullptr);

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 0u);
}

TEST(EJitCache, ThreadSafety) {
  EJitCache cache(1000, 1024 * 1024 * 100);
  int dummy[100]{};

  std::thread writer([&]() {
    for (int i = 0; i < 100; ++i)
      cache.put("key" + std::to_string(i), &dummy[i], 1);
  });

  std::thread reader([&]() {
    for (int i = 0; i < 1000; ++i)
      cache.getOrNull("key" + std::to_string(i % 100));
  });

  writer.join();
  reader.join();

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 100u);
}

//===----------------------------------------------------------------------===//
// PeriodArrayRegistry tests (T3-11)
//===----------------------------------------------------------------------===//

TEST(PeriodArrayRegistry, RegisterAndQueryArrays) {
  PeriodArrayRegistry reg;
  int data[10];
  reg.registerArray("cell", "cells", data, 10);

  const auto *arrs = reg.getArrays("cell");
  ASSERT_NE(arrs, nullptr);
  ASSERT_EQ(arrs->size(), 1u);
  EXPECT_EQ((*arrs)[0].varName, "cells");
  EXPECT_EQ((*arrs)[0].periodName, "cell");
  EXPECT_EQ((*arrs)[0].baseAddr, data);
  EXPECT_EQ((*arrs)[0].arraySize, 10u);

  const auto *arrs2 = reg.getArrays("nonexistent");
  EXPECT_EQ(arrs2, nullptr);
}

TEST(PeriodArrayRegistry, RegisterAndQueryStaticVars) {
  PeriodArrayRegistry reg;
  int val = 42;
  reg.registerStaticVar("config", &val);

  const auto &vars = reg.getStaticVars();
  ASSERT_EQ(vars.size(), 1u);
  EXPECT_EQ(vars[0].varName, "config");
  EXPECT_EQ(vars[0].varAddr, &val);
}

TEST(PeriodArrayRegistry, VarNameIndex) {
  PeriodArrayRegistry reg;
  int data[5];
  reg.registerArray("cell", "my_cells", data, 5);

  const auto *info = reg.getArrayInfo("my_cells");
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->periodName, "cell");
  EXPECT_EQ(info->arraySize, 5u);

  EXPECT_EQ(reg.getArrayInfo("unknown"), nullptr);
}

TEST(PeriodArrayRegistry, MultipleArraysSamePeriod) {
  PeriodArrayRegistry reg;
  int data1[5], data2[10];
  reg.registerArray("cell", "cells_a", data1, 5);
  reg.registerArray("cell", "cells_b", data2, 10);

  const auto *arrs = reg.getArrays("cell");
  ASSERT_NE(arrs, nullptr);
  EXPECT_EQ(arrs->size(), 2u);
}

//===----------------------------------------------------------------------===//
// EJitRuntimeState tests (T3-11)
//===----------------------------------------------------------------------===//

TEST(EJitRuntimeState, ActivateAndDeactivate) {
  EJitRuntimeState state;
  EXPECT_FALSE(state.isActive("cell", 0));

  state.activate("cell", 0);
  EXPECT_TRUE(state.isActive("cell", 0));

  state.deactivate("cell", 0);
  EXPECT_FALSE(state.isActive("cell", 0));
}

TEST(EJitRuntimeState, ActivateAllAndDeactivateAll) {
  EJitRuntimeState state;

  state.activateAll("cell");

  // After activateAll, specific cells should still track individually
  state.activate("cell", 5);
  EXPECT_TRUE(state.isActive("cell", 5));

  state.deactivateAll("cell");
  EXPECT_FALSE(state.isActive("cell", 5));
}

TEST(EJitRuntimeState, IndependentPeriods) {
  EJitRuntimeState state;

  state.activate("cell", 1);
  state.deactivate("trp", 2);

  EXPECT_TRUE(state.isActive("cell", 1));
  EXPECT_FALSE(state.isActive("trp", 2));
}

TEST(EJitRuntimeState, UninitializedReturnsFalse) {
  EJitRuntimeState state;
  EXPECT_FALSE(state.isActive("nonexistent", 99));
}

TEST(EJitRuntimeState, ThreadSafety) {
  EJitRuntimeState state;

  std::thread activator([&]() {
    for (int i = 0; i < 1000; ++i)
      state.activate("cell", i % 8);
  });

  std::thread checker([&]() {
    for (int i = 0; i < 1000; ++i)
      (void)state.isActive("cell", i % 8);
  });

  activator.join();
  checker.join();

  // After all activations, the last few should be active
  // (no guarantee about specific states due to interleaving)
  state.activateAll("cell");
  EXPECT_TRUE(state.isActive("cell", 0));
  EXPECT_TRUE(state.isActive("cell", 7));
}

//===----------------------------------------------------------------------===//
// EJitLogger tests (T3-12)
//===----------------------------------------------------------------------===//

TEST(EJitLogger, LogAndGetLastError) {
  EJitLogger logger;
  logger.log(ErrorCode::CompilationFailed, "test error", "myfunc", "mykey");

  const EJitError *err = logger.getLastError();
  ASSERT_NE(err, nullptr);
  EXPECT_EQ(err->code, ErrorCode::CompilationFailed);
  EXPECT_EQ(err->message, "test error");
  EXPECT_EQ(err->funcName, "myfunc");
  EXPECT_EQ(err->cacheKey, "mykey");
}

TEST(EJitLogger, GetLastErrorEmpty) {
  EJitLogger logger;
  EXPECT_EQ(logger.getLastError(), nullptr);
}

TEST(EJitLogger, RingBufferWrap) {
  EJitLogger logger;

  // Write more than kMaxErrors entries
  for (size_t i = 0; i < EJitLogger::kMaxErrors + 10; ++i) {
    logger.log(ErrorCode::Success, "msg" + std::to_string(i));
  }

  const EJitError *last = logger.getLastError();
  ASSERT_NE(last, nullptr);
  EXPECT_EQ(last->message, "msg" + std::to_string(EJitLogger::kMaxErrors + 9));
}

TEST(EJitLogger, GetErrors) {
  EJitLogger logger;

  for (size_t i = 0; i < 5; ++i)
    logger.log(ErrorCode::Success, "msg" + std::to_string(i));

  auto errors = logger.getErrors(3);
  EXPECT_EQ(errors.size(), 3u);
  EXPECT_EQ(errors[0].message, "msg0");
  EXPECT_EQ(errors[1].message, "msg1");
  EXPECT_EQ(errors[2].message, "msg2");
}

TEST(EJitLogger, Clear) {
  EJitLogger logger;
  logger.log(ErrorCode::Success, "test");
  logger.clear();
  EXPECT_EQ(logger.getLastError(), nullptr);
}

//===----------------------------------------------------------------------===//
// EJit end-to-end construction test (T3-20)
//===----------------------------------------------------------------------===//

TEST(EJit, ConstructionAndBasicOps) {
  Config config;
  config.compileMode = CompileMode::Sync;
  config.maxCacheEntries = 64;
  config.maxCacheSize = 1024 * 1024;

  EJit ejit(config);

  // Basic lifecycle operations should not crash
  ejit.activate("test_period", 0);
  EXPECT_TRUE(ejit.isActive("test_period", 0));

  ejit.deactivate("test_period", 0);
  EXPECT_FALSE(ejit.isActive("test_period", 0));

  // Cache should be empty initially
  auto stats = ejit.getStats();
  EXPECT_EQ(stats.entryCount, 0u);
}

TEST(EJit, ActivateAllAndDeactivateAll) {
  EJit ejit(Config{});

  ejit.activate("p1", 0);
  ejit.activate("p1", 1);
  ejit.activateAll("p1");

  // deactivateAll should clear all
  ejit.deactivateAll("p1");
  EXPECT_FALSE(ejit.isActive("p1", 0));
  EXPECT_FALSE(ejit.isActive("p1", 1));
}

TEST(EJit, CacheOperations) {
  EJit ejit(Config{});

  // clearCache should not crash
  ejit.clearCache();

  // invalidateByPeriod should not crash
  ejit.invalidateByPeriod("test", 0);
}

TEST(EJit, CompileMode) {
  EJit ejit(Config{});
  EXPECT_EQ(ejit.getCompileMode(), CompileMode::Sync);

  ejit.setCompileMode(CompileMode::Async);
  EXPECT_EQ(ejit.getCompileMode(), CompileMode::Async);
}

TEST(EJit, OptimizationLevel) {
  EJit ejit(Config{});
  EXPECT_EQ(ejit.getOptimizationLevel(), llvm::ejit::OptimizationLevel::L2);

  ejit.setOptimizationLevel(llvm::ejit::OptimizationLevel::L3);
  EXPECT_EQ(ejit.getOptimizationLevel(), llvm::ejit::OptimizationLevel::L3);
}

//===----------------------------------------------------------------------===//
// C API tests with runtime-dynamic cellIdx (T3-21)
//===----------------------------------------------------------------------===//

extern "C" {
  typedef enum { EJIT_OK_C = 0 } ejit_status_test_t;
  extern ejit_status_test_t ejit_init(const void *config);
  extern void ejit_shutdown(void);
  extern ejit_status_test_t ejit_activate(const char *, uint8_t);
  extern ejit_status_test_t ejit_deactivate(const char *, uint8_t);
  extern bool ejit_is_active(const char *, uint8_t);
  extern void ejit_invalidate(const char *, uint8_t);
  extern void ejit_clear_cache(void);
}

TEST(EJitCApi, ActivateWithDynamicCellIdx) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  uint8_t idx = 42;
  EXPECT_EQ(ejit_activate("dynamic", idx), EJIT_OK_C);
  EXPECT_TRUE(ejit_is_active("dynamic", idx));
  EXPECT_EQ(ejit_deactivate("dynamic", idx), EJIT_OK_C);
  EXPECT_FALSE(ejit_is_active("dynamic", idx));
  ejit_shutdown();
}

TEST(EJitCApi, LoopWithDynamicCellIdx) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  for (uint8_t i = 0; i < 10; i++)
    EXPECT_EQ(ejit_activate("loop", i), EJIT_OK_C);
  for (uint8_t i = 0; i < 10; i++)
    EXPECT_TRUE(ejit_is_active("loop", i));
  for (uint8_t i = 10; i > 0; i--)
    EXPECT_EQ(ejit_deactivate("loop", i - 1), EJIT_OK_C);
  for (uint8_t i = 0; i < 10; i++)
    EXPECT_FALSE(ejit_is_active("loop", i));
  ejit_shutdown();
}

static uint8_t computeCellIdx(int x, int y) {
  return static_cast<uint8_t>((x + y) % 256);
}

TEST(EJitCApi, ActivateWithComputedCellIdx) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  uint8_t idx = computeCellIdx(100, 55);
  EXPECT_EQ(idx, 155);
  EXPECT_EQ(ejit_activate("compute", idx), EJIT_OK_C);
  EXPECT_TRUE(ejit_is_active("compute", idx));
  idx = computeCellIdx(200, 200);
  EXPECT_EQ(idx, 144);
  EXPECT_EQ(ejit_activate("compute", idx), EJIT_OK_C);
  EXPECT_TRUE(ejit_is_active("compute", idx));
  EXPECT_FALSE(ejit_is_active("other", 155));
  ejit_shutdown();
}

TEST(EJitCApi, DynamicCellIdxBoundaries) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  EXPECT_EQ(ejit_activate("bound", (uint8_t)0), EJIT_OK_C);
  EXPECT_TRUE(ejit_is_active("bound", (uint8_t)0));
  EXPECT_EQ(ejit_activate("bound", (uint8_t)255), EJIT_OK_C);
  EXPECT_TRUE(ejit_is_active("bound", (uint8_t)255));
  ejit_deactivate("bound", 0);
  EXPECT_FALSE(ejit_is_active("bound", 0));
  EXPECT_TRUE(ejit_is_active("bound", (uint8_t)255));
  ejit_shutdown();
}

TEST(EJitCApi, MultiPeriodDynamicIndices) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  uint8_t indices[] = {3, 7, 15, 31, 63};
  for (size_t i = 0; i < sizeof(indices)/sizeof(indices[0]); i++) {
    EXPECT_EQ(ejit_activate("cell", indices[i]), EJIT_OK_C);
    EXPECT_EQ(ejit_activate("trp", indices[i]), EJIT_OK_C);
  }
  for (size_t i = 0; i < sizeof(indices)/sizeof(indices[0]); i++) {
    EXPECT_TRUE(ejit_is_active("cell", indices[i]));
    EXPECT_TRUE(ejit_is_active("trp", indices[i]));
  }
  for (size_t i = 0; i < sizeof(indices)/sizeof(indices[0]); i++)
    ejit_deactivate("trp", indices[i]);
  for (size_t i = 0; i < sizeof(indices)/sizeof(indices[0]); i++) {
    EXPECT_TRUE(ejit_is_active("cell", indices[i]));
    EXPECT_FALSE(ejit_is_active("trp", indices[i]));
  }
  ejit_shutdown();
}

TEST(EJitCApi, InvalidateWithDynamicCellIdx) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  for (uint8_t i = 0; i < 5; i++)
    ejit_activate("inv", i);
  for (uint8_t i = 0; i < 5; i++)
    EXPECT_TRUE(ejit_is_active("inv", i));
  for (uint8_t i = 0; i < 5; i++)
    ejit_invalidate("inv", i);
  for (uint8_t i = 0; i < 5; i++)
    EXPECT_TRUE(ejit_is_active("inv", i));
  ejit_shutdown();
}

TEST(EJitCApi, ActivationCycleWithRuntimeIndex) {
  ASSERT_EQ(ejit_init(nullptr), EJIT_OK_C);
  uint8_t workload[] = {10, 20, 30, 40, 50};
  for (size_t cycle = 0; cycle < 3; cycle++) {
    for (size_t i = 0; i < sizeof(workload)/sizeof(workload[0]); i++)
      EXPECT_EQ(ejit_activate("cycle", workload[i]), EJIT_OK_C);
    for (size_t i = 0; i < sizeof(workload)/sizeof(workload[0]); i++)
      EXPECT_TRUE(ejit_is_active("cycle", workload[i]));
    for (size_t i = 0; i < sizeof(workload)/sizeof(workload[0]); i++)
      EXPECT_EQ(ejit_deactivate("cycle", workload[i]), EJIT_OK_C);
    for (size_t i = 0; i < sizeof(workload)/sizeof(workload[0]); i++)
      EXPECT_FALSE(ejit_is_active("cycle", workload[i]));
  }
  ejit_shutdown();
}

//===----------------------------------------------------------------------===//
// PeriodArrayRegistry::getArrayByBaseAddr tests
//===----------------------------------------------------------------------===//

TEST(PeriodArrayRegistry, GetArrayByBaseAddr) {
  PeriodArrayRegistry reg;
  int data[10];
  reg.registerArray("cell", "my_cells", data, 10);

  const auto *info = reg.getArrayByBaseAddr(data);
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->varName, "my_cells");
  EXPECT_EQ(info->periodName, "cell");
  EXPECT_EQ(info->arraySize, 10u);

  // Non-registered pointer returns nullptr
  int other;
  EXPECT_EQ(reg.getArrayByBaseAddr(&other), nullptr);
  EXPECT_EQ(reg.getArrayByBaseAddr(nullptr), nullptr);
}

TEST(PeriodArrayRegistry, GetArrayByBaseAddrMultipleArrays) {
  PeriodArrayRegistry reg;
  int data1[5], data2[10];
  reg.registerArray("cell", "a", data1, 5);
  reg.registerArray("trp", "b", data2, 10);

  const auto *info1 = reg.getArrayByBaseAddr(data1);
  ASSERT_NE(info1, nullptr);
  EXPECT_EQ(info1->varName, "a");
  EXPECT_EQ(info1->periodName, "cell");

  const auto *info2 = reg.getArrayByBaseAddr(data2);
  ASSERT_NE(info2, nullptr);
  EXPECT_EQ(info2->varName, "b");
  EXPECT_EQ(info2->periodName, "trp");
}

//===----------------------------------------------------------------------===//
// EJitOptimizer tests
//===----------------------------------------------------------------------===//

static std::unique_ptr<Module> createTestModule(LLVMContext &Ctx,
                                                const std::string &Name) {
  auto M = std::make_unique<Module>("test", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  return M;
}

/// Create a simple function with a period-array-index argument metadata.
static Function *createPeriodIndFunc(LLVMContext &Ctx, Module &M,
                                     const std::string &name) {
  IRBuilder<> B(Ctx);
  Type *RetTy = B.getInt32Ty();
  Type *ParamTy = B.getInt32Ty();
  FunctionType *FT = FunctionType::get(RetTy, {ParamTy}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, name, &M);
  F->setName(name);

  auto &Arg = *F->arg_begin();
  Arg.setName("period_idx");

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);
  B.CreateRet(B.CreateAdd(&Arg, B.getInt32(1)));

  // Attach !ejit.metadata with a sub-node for period-array-index:
  //   !ejit.metadata = !{!0}
  //   !0 = !{!"ejit_period_arr_ind", !"cell", i32 0}
  Metadata *MDOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(B.getInt32Ty(), 0)),
  };
  MDNode *Sub = MDNode::get(Ctx, MDOps);
  F->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {Sub}));

  return F;
}

TEST(EJitOptimizer, PreReplacePeriodIndices) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "preReplaceTest");
  Function *F = createPeriodIndFunc(Ctx, *M, "test_func");
  ASSERT_NE(F, nullptr);

  PeriodArrayRegistry reg;
  SpecializationContext ctx;
  ctx.fnName = "test_func";
  ctx.dimensions.push_back({"cell", 42});

  // Before replacement: argument is used
  auto &Arg = *F->arg_begin();
  EXPECT_TRUE(Arg.hasNUsesOrMore(1));

  EJitOptimizer opt(reg);
  opt.preReplacePeriodIndices(*M, ctx);

  // After replacement: the arg should have zero uses (replaced by constant 42)
  EXPECT_EQ(Arg.getNumUses(), 0u);
}

TEST(EJitOptimizer, OptimizationPipelineL1) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "optL1");
  createPeriodIndFunc(Ctx, *M, "f");

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);

  // L1 should not crash
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L1);
}

TEST(EJitOptimizer, OptimizationPipelineL2) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "optL2");
  createPeriodIndFunc(Ctx, *M, "f");

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);

  // L2 should not crash
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L2);
}

TEST(EJitOptimizer, OptimizationPipelineL3) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "optL3");
  createPeriodIndFunc(Ctx, *M, "f");

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);

  // L3 should not crash
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L3);
}

TEST(EJitOptimizer, FullPipelineEndToEnd) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "fullPipeline");
  createPeriodIndFunc(Ctx, *M, "test_fn");

  PeriodArrayRegistry reg;
  SpecializationContext ctx;
  ctx.fnName = "test_fn";
  ctx.dimensions.push_back({"cell", 100});
  ctx.optLevel = llvm::ejit::OptimizationLevel::L3;

  EJitOptimizer opt(reg);

  // 1. Pre-replace
  opt.preReplacePeriodIndices(*M, ctx);

  // 2. InstCombine
  opt.runInstCombine(*M);

  // 3. Inline (no-op for a single function, but shouldn't crash)
  opt.runInline(*M);

  // 4. Optimization pipeline at L3
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L3);
}

//===----------------------------------------------------------------------===//
// EJitStructFieldPass tests
//===----------------------------------------------------------------------===//

/// Create a function with a GEP + load from a global array (simulating
/// a period array access marked with !ejit.may_const).
/// The GEP index is a constant so the pass can compute the offset.
/// Also adds !ejit.metadata to the global to identify it as a period array.
static Function *createStructFieldFunc(LLVMContext &Ctx, Module &M,
                                       uint64_t gepIdx = 2) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();

  // Create a global array: int32_t g_arr[4]
  auto *ArrTy = ArrayType::get(Int32Ty, 4);
  auto *GVar = new GlobalVariable(M, ArrTy, false,
                                   GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(ArrTy), "g_arr");

  // Add !ejit.metadata to GVar: !g_arr = !{!"ejit_period_arr", !"cell", i32 4}
  Metadata *ArrMDOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar->setMetadata(MD_EJIT_METADATA,
                    MDNode::get(Ctx, {MDNode::get(Ctx, ArrMDOps)}));

  // Function: int32_t test_load()
  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test_load", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // GEP: &g_arr[gepIdx] with constant index
  Value *IdxList[] = {B.getInt32(0), B.getInt64(gepIdx)};
  auto *GEP = B.CreateInBoundsGEP(ArrTy, GVar, IdxList, "gep");
  auto *Load = B.CreateLoad(Int32Ty, GEP, "load");

  // Add !ejit.may_const metadata to the load
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  B.CreateRet(Load);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitution) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_struct", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createStructFieldFunc(Ctx, *M, 2); // g_arr[2] = 30
  ASSERT_NE(F, nullptr);

  // Create real memory representing the period array data.
  // The StructFieldPass reads from the registry's baseAddr at the GEP offset.
  int32_t mockArr[4] = {10, 20, 30, 40};

  PeriodArrayRegistry reg;
  GlobalVariable *GV = M->getGlobalVariable("g_arr", true);
  ASSERT_NE(GV, nullptr) << "g_arr not found in module";
  reg.registerArray("cell", "g_arr", mockArr, 4);

  // Run the StructFieldPass
  EJitStructFieldPass structPass(reg);
  FunctionAnalysisManager FAM;
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  auto PA = structPass.run(*F, FAM);

  // The pass should find the GEP+load, compute offset for g_arr[2] = 8 bytes,
  // read mockArr + 8 = mockArr[2] = 30, and replace the load.
  EXPECT_FALSE(PA.areAllPreserved());

  // Verify the load was replaced: the ret should now use a ConstantInt(30)
  bool loadRemoved = true;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (isa<LoadInst>(&I))
        loadRemoved = false;
  EXPECT_TRUE(loadRemoved);

  // Check that the return value is constant 30
  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 30);
}

TEST(EJitStructFieldPass, NoMayConstNoChange) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_nochange", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));

  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();
  FunctionType *FT = FunctionType::get(Int32Ty, {Int32Ty}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "noop", M.get());

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);
  auto *Add = B.CreateAdd(F->getArg(0), B.getInt32(1));
  B.CreateRet(Add);

  PeriodArrayRegistry reg;
  EJitStructFieldPass structPass(reg);
  FunctionAnalysisManager FAM;
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  auto PA = structPass.run(*F, FAM);

  // Pass should preserve all analyses when there's nothing to do
  EXPECT_TRUE(PA.areAllPreserved());
}


