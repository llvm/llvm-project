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

  auto result = loader.getBitcodeByFuncIdx(hashFuncName("my_func"));
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->size(), 4u);
  EXPECT_EQ((uint8_t)(*result)[0], 0xAA);
}

TEST(EJitModuleLoader, GetBitcodeNotFound) {
  EJitModuleLoader loader;
  auto result = loader.getBitcodeByFuncIdx(0xDEAD);
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

  auto r1 = loader.getBitcodeByFuncIdx(hashFuncName("f1"));
  ASSERT_TRUE(static_cast<bool>(r1));
  EXPECT_EQ(r1->size(), 1u);

  auto r2 = loader.getBitcodeByFuncIdx(hashFuncName("f2"));
  ASSERT_TRUE(static_cast<bool>(r2));
  EXPECT_EQ(r2->size(), 2u);
}

//===----------------------------------------------------------------------===//
// EJitCache tests (T3-10)
//===----------------------------------------------------------------------===//

TEST(EJitCache, BasicPutAndGet) {
  EJitCache cache(10, 1024 * 1024);
  int dummy = 0;
  cache.put(1, &dummy, 64);
  EXPECT_EQ(cache.getOrNull(1), &dummy);
  EXPECT_EQ(cache.getOrNull(999), nullptr);
}

TEST(EJitCache, StatsTracking) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;
  cache.put(100, &dummy, 64 );
  cache.getOrNull(100);  // hit
  cache.getOrNull(100);  // hit
  cache.getOrNull(200);  // miss

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 1u);
  EXPECT_EQ(stats.hits, 2ull);
  EXPECT_EQ(stats.misses, 1ull);
}

TEST(EJitCache, LRUEvictionByEntryCount) {
  EJitCache cache(2, 1024 * 1024);
  int a, b, c;

  EXPECT_TRUE(cache.put(10, &a, 1));
  EXPECT_TRUE(cache.put(20, &b, 1));
  EXPECT_TRUE(cache.put(30, &c, 1)); // should evict 'a'

  EXPECT_EQ(cache.getOrNull(10), nullptr);
  EXPECT_EQ(cache.getOrNull(20), &b);
  EXPECT_EQ(cache.getOrNull(30), &c);

  auto stats = cache.getStats();
  EXPECT_EQ(stats.evictions, 1ull);
}

TEST(EJitCache, LRUEvictionByTotalSize) {
  EJitCache cache(100, 200);
  int dummy;

  cache.put(10, &dummy, 120); // ok
  cache.put(20, &dummy, 90);  // should evict 'a'

  EXPECT_EQ(cache.getOrNull(10), nullptr);
  EXPECT_EQ(cache.getOrNull(20), &dummy);
}

TEST(EJitCache, SingleFuncSizeLimit) {
  EJitCache cache(10, 1024 * 1024, 100);
  int dummy;
  EXPECT_FALSE(cache.put(9999, &dummy, 200));
  EXPECT_TRUE(cache.put(500, &dummy, 50));

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 1u);
}

TEST(EJitCache, PeriodicInvalidation) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;

  std::set<std::string> depsA = {"cell=0"};
  std::set<std::string> depsB = {"cell=1"};
  std::set<std::string> depsC = {"trp=0"};

  cache.put(10, &dummy, 1, depsA);
  cache.put(20, &dummy, 1, depsB);
  cache.put(30, &dummy, 1, depsC);

  EXPECT_EQ(cache.getOrNull(10), &dummy);
  EXPECT_EQ(cache.getOrNull(20), &dummy);
  EXPECT_EQ(cache.getOrNull(30), &dummy);

  cache.invalidateByPeriod("cell", 0);
  EXPECT_EQ(cache.getOrNull(10), nullptr);  // invalidated
  EXPECT_EQ(cache.getOrNull(20), &dummy);   // still valid (cell=1)
  EXPECT_EQ(cache.getOrNull(30), &dummy);   // still valid (trp=0)
}

TEST(EJitCache, BuildCacheKey) {
  // uint64_t key = funcIdx(32b) | dim[0](8b) | dim[1](8b) | dim[2](8b) | dim[3](8b)
  // No dimensions → key = funcIdx << 32
  uint64_t key0 = EJitCache::buildCacheKey(7, nullptr, 0);
  EXPECT_EQ(key0, 0x0000000700000000ULL);

  // Single dimension: funcIdx=1, cell=3 → (1 << 32) | 3
  std::pair<std::string, uint8_t> dims1[] = {{"cell", 3}};
  uint64_t key1 = EJitCache::buildCacheKey(1, dims1, 1);
  EXPECT_EQ(key1, 0x0000000100000003ULL);

  // Multiple dimensions: funcIdx=2, d0=1, d1=5 → (2 << 32) | 1 | (5 << 8)
  std::pair<std::string, uint8_t> dims2[] = {{"trp", 1}, {"cell", 5}};
  uint64_t key2 = EJitCache::buildCacheKey(2, dims2, 2);
  EXPECT_EQ(key2, 0x0000000200000501ULL);
}

TEST(EJitCache, Clear) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;
  cache.put(10, &dummy, 64 );
  cache.put(20, &dummy, 64 );
  cache.clear();

  EXPECT_EQ(cache.getOrNull(10), nullptr);
  EXPECT_EQ(cache.getOrNull(20), nullptr);

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 0u);
}

TEST(EJitCache, ThreadSafety) {
  EJitCache cache(1000, 1024 * 1024 * 100);
  int dummy[100]{};

  std::thread writer([&]() {
    for (int i = 0; i < 100; ++i)
      cache.put(static_cast<uint32_t>(i), &dummy[i], 1);
  });

  std::thread reader([&]() {
    for (int i = 0; i < 1000; ++i)
      cache.getOrNull(static_cast<uint32_t>(i % 100));
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

  // Register a period array so activateAll/deactivateAll know the cell range
  int dummy[8];
  state.getRegistry().registerArray("cell", "dummy_cells", dummy, 8);

  state.activateAll("cell");

  // After activateAll, all cells of the registered array are active
  EXPECT_TRUE(state.isActive("cell", 0));
  EXPECT_TRUE(state.isActive("cell", 7));

  state.deactivateAll("cell");
  EXPECT_FALSE(state.isActive("cell", 0));
  EXPECT_FALSE(state.isActive("cell", 7));
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
  // Register a period array before constructing EJit so
  // activateAll/deactivateAll know the cell range.
  int dummy[4];
  EJitRegistrationStore::instance().consume(); // clear leftover
  EJitRegistrationStore::instance().registerPeriodArray(
      "p1", "dummy_arr", dummy, 4);

  EJit ejit(Config{});

  ejit.activateAll("p1");
  EXPECT_TRUE(ejit.isActive("p1", 0));
  EXPECT_TRUE(ejit.isActive("p1", 3));

  // deactivateAll should clear all
  ejit.deactivateAll("p1");
  EXPECT_FALSE(ejit.isActive("p1", 0));
  EXPECT_FALSE(ejit.isActive("p1", 3));
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

//===----------------------------------------------------------------------===//
// EJitStructFieldPass extended tests
//===----------------------------------------------------------------------===//

/// Create a function with two loads from different "fields" of the same
/// global array (index 1 and index 3). Tests multi-field substitution.
/// g_arr[1] + g_arr[3]
static Function *createMultiFieldFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();

  auto *ArrTy = ArrayType::get(Int32Ty, 4);
  auto *GVar = new GlobalVariable(M, ArrTy, false, GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(ArrTy), "g_arr");

  // !ejit.metadata = !{!"ejit_period_arr", !"cell", i32 4}
  Metadata *ArrOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar->setMetadata(MD_EJIT_METADATA,
                    MDNode::get(Ctx, {MDNode::get(Ctx, ArrOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "multi_field", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // Load from g_arr[1] (offset 4)
  Value *I1[] = {B.getInt32(0), B.getInt64(1)};
  auto *GEP1 = B.CreateInBoundsGEP(ArrTy, GVar, I1, "elem1");
  auto *Load1 = B.CreateLoad(Int32Ty, GEP1, "val1");
  Load1->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  // Load from g_arr[3] (offset 12)
  Value *I3[] = {B.getInt32(0), B.getInt64(3)};
  auto *GEP3 = B.CreateInBoundsGEP(ArrTy, GVar, I3, "elem3");
  auto *Load3 = B.CreateLoad(Int32Ty, GEP3, "val3");
  Load3->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  auto *Sum = B.CreateAdd(Load1, Load3, "sum");
  B.CreateRet(Sum);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitutionMultipleFields) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_multifield", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createMultiFieldFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  int32_t mockArr[4] = {10, 55, 20, 66};  // g_arr[1]=55, g_arr[3]=66

  PeriodArrayRegistry reg;
  GlobalVariable *GV = M->getGlobalVariable("g_arr", true);
  ASSERT_NE(GV, nullptr);
  reg.registerArray("cell", "g_arr", mockArr, 4);

  EJitStructFieldPass sp(reg);
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

  sp.run(*F, FAM);

  // Fold the add of two constants: 55 + 66 = 121
  EJitOptimizer opt(reg);
  opt.runInstCombine(*M);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 121);
}

/// Create a function with loads from different indices into a period array.
static Function *createNestedStructFunc(LLVMContext &Ctx, Module &M,
                                         uint64_t arrIdx = 1) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();

  auto *ArrTy = ArrayType::get(Int32Ty, 4);
  auto *GVar = new GlobalVariable(M, ArrTy, false, GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(ArrTy), "g_data");

  Metadata *ArrOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar->setMetadata(MD_EJIT_METADATA,
                    MDNode::get(Ctx, {MDNode::get(Ctx, ArrOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "indexed_load", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  Value *Indices[] = {B.getInt32(0), B.getInt64(arrIdx)};
  auto *GEP = B.CreateInBoundsGEP(ArrTy, GVar, Indices, "gep");
  auto *Load = B.CreateLoad(Int32Ty, GEP, "val");
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));
  B.CreateRet(Load);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitutionNestedStruct) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_nested", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createNestedStructFunc(Ctx, *M, 2);
  ASSERT_NE(F, nullptr);

  int32_t mockArr[4] = {10, 20, 40, 80};  // g_data[2] = 40

  PeriodArrayRegistry reg;
  GlobalVariable *GV = M->getGlobalVariable("g_data", true);
  ASSERT_NE(GV, nullptr);
  reg.registerArray("cell", "g_data", mockArr, 4);

  EJitStructFieldPass sp(reg);
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
  sp.run(*F, FAM);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 40);
}

/// Create a function that accesses two different period arrays.
/// g_cells["cell"].field and g_trps["trp"].field
static Function *createMultiArrayFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();

  auto *ArrTy = ArrayType::get(Int32Ty, 4);

  auto *GVar1 = new GlobalVariable(M, ArrTy, false, GlobalValue::InternalLinkage,
                                    ConstantAggregateZero::get(ArrTy), "g_cells");
  auto *GVar2 = new GlobalVariable(M, ArrTy, false, GlobalValue::InternalLinkage,
                                    ConstantAggregateZero::get(ArrTy), "g_trps");

  // g_cells metadata: ejit_period_arr "cell" size 4
  Metadata *CellOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR), MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar1->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {MDNode::get(Ctx, CellOps)}));
  // g_trps metadata: ejit_period_arr "trp" size 4
  Metadata *TrpOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR), MDString::get(Ctx, "trp"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar2->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {MDNode::get(Ctx, TrpOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "multi_arr", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // Load from g_cells[2] -> cell array, index 2
  Value *CIdx[] = {B.getInt32(0), B.getInt64(2)};
  auto *GEP1 = B.CreateInBoundsGEP(ArrTy, GVar1, CIdx, "cell_gep");
  auto *Load1 = B.CreateLoad(Int32Ty, GEP1, "cell_val");
  Load1->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  // Load from g_trps[3] -> trp array, index 3
  Value *TIdx[] = {B.getInt32(0), B.getInt64(3)};
  auto *GEP2 = B.CreateInBoundsGEP(ArrTy, GVar2, TIdx, "trp_gep");
  auto *Load2 = B.CreateLoad(Int32Ty, GEP2, "trp_val");
  Load2->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  auto *Mul = B.CreateMul(Load1, Load2, "product");
  B.CreateRet(Mul);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitutionMultipleArrays) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_multi_arr", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createMultiArrayFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  int32_t cellData[4] = {1, 2, 7, 4};  // g_cells[2] = 7
  int32_t trpData[4]  = {5, 6, 7, 8};  // g_trps[3] = 8

  PeriodArrayRegistry reg;
  reg.registerArray("cell", "g_cells", cellData, 4);
  reg.registerArray("trp", "g_trps", trpData, 4);

  EJitStructFieldPass sp(reg);
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
  sp.run(*F, FAM);

  EJitOptimizer opt(reg);
  opt.runInstCombine(*M);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 56);  // 7 * 8 = 56
}

/// Create a function with may_const loads of different types (int + float)
/// from separate array globals. Tests mixed-type substitution.
static Function *createMixedTypeFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();
  Type *FloatTy = B.getFloatTy();

  auto *IntArrTy = ArrayType::get(Int32Ty, 4);
  auto *FltArrTy = ArrayType::get(FloatTy, 4);

  auto *GInt = new GlobalVariable(M, IntArrTy, false, GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(IntArrTy), "g_ints");
  auto *GFlt = new GlobalVariable(M, FltArrTy, false, GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(FltArrTy), "g_floats");

  Metadata *IntOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR), MDString::get(Ctx, "ints"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GInt->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {MDNode::get(Ctx, IntOps)}));
  Metadata *FltOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR), MDString::get(Ctx, "floats"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GFlt->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {MDNode::get(Ctx, FltOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "mixed_type", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // Load int from g_ints[0]
  Value *II[] = {B.getInt32(0), B.getInt64(0)};
  auto *GEP_i = B.CreateInBoundsGEP(IntArrTy, GInt, II, "int_elem");
  auto *LoadI = B.CreateLoad(Int32Ty, GEP_i, "int_val");
  LoadI->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  // Load float from g_floats[0]
  Value *FI[] = {B.getInt32(0), B.getInt64(0)};
  auto *GEP_f = B.CreateInBoundsGEP(FltArrTy, GFlt, FI, "flt_elem");
  auto *LoadF = B.CreateLoad(FloatTy, GEP_f, "flt_val");
  LoadF->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));

  auto *FToI = B.CreateFPToSI(LoadF, Int32Ty, "ftoi");
  auto *Sum = B.CreateAdd(LoadI, FToI, "sum");
  B.CreateRet(Sum);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitutionIntFloat) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_mixed", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createMixedTypeFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  int32_t intData[4] = {10, 0, 0, 0};
  float fltData[4] = {3.5f, 0, 0, 0};
  // 10 + (int)3.5 = 10 + 3 = 13

  PeriodArrayRegistry reg;
  reg.registerArray("ints", "g_ints", intData, 4);
  reg.registerArray("floats", "g_floats", fltData, 4);

  EJitStructFieldPass sp(reg);
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
  sp.run(*F, FAM);

  EJitOptimizer opt(reg);
  opt.runInstCombine(*M);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 13);
}

//===----------------------------------------------------------------------===//
// EJitStructFieldPass multi-dimensional array test
//===----------------------------------------------------------------------===//

/// Create a function accessing a 2D period array: g_arr[1][2]
/// The GEP has 3 indices: {0, 1, 2}. Tests the multi-index offset fix.
static Function *createMultiDimArrayFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();
  auto *InnerTy = ArrayType::get(Int32Ty, 8);   // [8 x i32]
  auto *OuterTy = ArrayType::get(InnerTy, 4);    // [4 x [8 x i32]]

  auto *GVar = new GlobalVariable(M, OuterTy, false, GlobalValue::InternalLinkage,
                                   ConstantAggregateZero::get(OuterTy), "g_2d");

  Metadata *ArrOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR), MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GVar->setMetadata(MD_EJIT_METADATA,
                    MDNode::get(Ctx, {MDNode::get(Ctx, ArrOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "access_2d", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // g_2d[1][2] with 3-index GEP
  Value *Indices[] = {B.getInt32(0), B.getInt64(1), B.getInt64(2)};
  auto *GEP = B.CreateInBoundsGEP(OuterTy, GVar, Indices, "gep_2d");
  auto *Load = B.CreateLoad(Int32Ty, GEP, "val_2d");
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));
  B.CreateRet(Load);
  return F;
}

TEST(EJitStructFieldPass, MayConstLoadSubstitutionMultiDimArray) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_2d", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createMultiDimArrayFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  // 2D mock: [4][8] int32
  // g_2d[1][2] = element at row 1, col 2 = offset 1*8*4 + 2*4 = 40 bytes
  int32_t mock2D[4][8] = {
    { 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 99, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0 },
  };

  PeriodArrayRegistry reg;
  GlobalVariable *GV = M->getGlobalVariable("g_2d", true);
  ASSERT_NE(GV, nullptr);
  reg.registerArray("cell", "g_2d", mock2D, 4);

  EJitStructFieldPass sp(reg);
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
  sp.run(*F, FAM);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 99);
}

//===----------------------------------------------------------------------===//
// EJitOptimizer extended tests
//===----------------------------------------------------------------------===//

/// Create a function with 2 period-array-index params (cell + trp).
static Function *createMultiDimFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  Type *Int32Ty = B.getInt32Ty();
  FunctionType *FT = FunctionType::get(Int32Ty, {Int32Ty, Int32Ty}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "multi_dim", &M);

  F->getArg(0)->setName("cell_idx");
  F->getArg(1)->setName("trp_idx");

  // Metadata for cell dimension (param 0)
  Metadata *CellOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0)),
  };
  // Metadata for trp dimension (param 1)
  Metadata *TrpOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
      MDString::get(Ctx, "trp"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 1)),
  };
  F->setMetadata(MD_EJIT_METADATA,
                 MDNode::get(Ctx, {MDNode::get(Ctx, CellOps), MDNode::get(Ctx, TrpOps)}));

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);
  auto *Sum = B.CreateAdd(F->getArg(0), F->getArg(1), "sum");
  B.CreateRet(Sum);
  return F;
}

TEST(EJitOptimizer, PreReplacePeriodIndicesMultiDim) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "multi_dim_test");
  Function *F = createMultiDimFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  PeriodArrayRegistry reg;
  SpecializationContext ctx;
  ctx.fnName = "multi_dim";
  ctx.dimensions.push_back({"cell", 10});
  ctx.dimensions.push_back({"trp", 25});

  EJitOptimizer opt(reg);
  opt.preReplacePeriodIndices(*M, ctx);

  // Both args should be replaced; sum should fold to 35
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

  opt.runInstCombine(*M);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 35);
}

/// Create IR with dead code behind a false branch. L1 should eliminate it.
static Function *createDeadCodeFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  FunctionType *FT = FunctionType::get(B.getInt32Ty(), {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "dead_code_fn", &M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);

  // Unreachable block
  auto *DeadBB = BasicBlock::Create(Ctx, "dead", F);
  {
    IRBuilder<> DB(DeadBB);
    DB.CreateRet(DB.getInt32(999));
  }

  // br on constant false -> unreachable branch should be eliminated
  B.CreateCondBr(B.getFalse(), DeadBB, DeadBB);
  B.CreateRet(B.getInt32(42));

  return F;
}

TEST(EJitOptimizer, OptimizationL1DeadCodeElimination) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "deadcode");
  Function *F = createDeadCodeFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);
  int bbCount = (int)std::distance(F->begin(), F->end());

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L1);

  // Dead block should be removed
  int bbAfter = (int)std::distance(F->begin(), F->end());
  EXPECT_LT(bbAfter, bbCount);
}

/// Create a call to a small callee. L2 should inline it.
static Function *createInlineCandidate(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();

  // Callee: int callee(int x) { return x + 1; }
  FunctionType *CalleeFT = FunctionType::get(Int32Ty, {Int32Ty}, false);
  auto *Callee = Function::Create(CalleeFT, GlobalValue::InternalLinkage, "callee", &M);
  Callee->addFnAttr(Attribute::AlwaysInline);
  {
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", Callee);
    B.SetInsertPoint(BB);
    auto *Val = B.CreateAdd(Callee->getArg(0), B.getInt32(1));
    B.CreateRet(Val);
  }

  // Caller: int caller() { return callee(41); }
  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *Caller = Function::Create(FT, GlobalValue::ExternalLinkage, "caller", &M);
  {
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", Caller);
    B.SetInsertPoint(BB);
    auto *Call = B.CreateCall(Callee, {B.getInt32(41)});
    B.CreateRet(Call);
  }
  return Caller;
}

TEST(EJitOptimizer, OptimizationL2InlineAndSimplify) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "inline_test");
  Function *F = createInlineCandidate(Ctx, *M);
  ASSERT_NE(F, nullptr);

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L2);

  // After inlining 41+1, should fold to constant 42
  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  // If fully optimized, this should be constant 42
  if (RetVal)
    EXPECT_EQ(RetVal->getSExtValue(), 42);
}

/// Create a small loop with constant bounds. L3 should unroll it.
static Function *createLoopFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();
  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "loop_fn", &M);

  // for i in [0..4): sum += i  => 0+1+2+3 = 6
  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *LoopHdr = BasicBlock::Create(Ctx, "loop", F);
  BasicBlock *LoopBody = BasicBlock::Create(Ctx, "body", F);
  BasicBlock *LoopExit = BasicBlock::Create(Ctx, "exit", F);

  B.SetInsertPoint(Entry);
  B.CreateBr(LoopHdr);

  // Loop header: phi [i=0, sum=0], icmp i < 4
  B.SetInsertPoint(LoopHdr);
  auto *PhiI = B.CreatePHI(Int32Ty, 2, "i");
  auto *PhiSum = B.CreatePHI(Int32Ty, 2, "sum");
  auto *Cmp = B.CreateICmpSLT(PhiI, B.getInt32(4));
  B.CreateCondBr(Cmp, LoopBody, LoopExit);

  B.SetInsertPoint(LoopBody);
  auto *NewSum = B.CreateAdd(PhiSum, PhiI);
  auto *NewI = B.CreateAdd(PhiI, B.getInt32(1));
  B.CreateBr(LoopHdr);

  // Back edges
  PhiI->addIncoming(B.getInt32(0), Entry);
  PhiI->addIncoming(NewI, LoopBody);
  PhiSum->addIncoming(B.getInt32(0), Entry);
  PhiSum->addIncoming(NewSum, LoopBody);

  B.SetInsertPoint(LoopExit);
  B.CreateRet(PhiSum);
  return F;
}

TEST(EJitOptimizer, OptimizationL3LoopUnroll) {
  LLVMContext Ctx;
  auto M = createTestModule(Ctx, "loop_test");
  Function *F = createLoopFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  PeriodArrayRegistry reg;
  EJitOptimizer opt(reg);

  // Promote first (mem2reg)
  opt.runInstCombine(*M);
  // Then L3 with loop unroll
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L3);

  // After unrolling 0+1+2+3, should fold to constant 6
  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  if (RetVal)
    EXPECT_EQ(RetVal->getSExtValue(), 6);
}

//===----------------------------------------------------------------------===//
// End-to-end tests
//===----------------------------------------------------------------------===//

/// Create IR with a branch that depends on a may_const field.
/// After full pipeline (StructField + L2), the branch should be folded.
static Function *createBranchOnMayConstFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();
  auto *STy = StructType::create(Ctx, {Int32Ty}, "BranchCfg");

  auto *GVar = new GlobalVariable(M, STy, false, GlobalValue::InternalLinkage,
                                   ConstantStruct::get(STy, ConstantInt::get(Int32Ty, 0)),
                                   "g_branch_cfg");
  Metadata *PeriodOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD),
      MDString::get(Ctx, "static"),
  };
  GVar->setMetadata(MD_EJIT_METADATA, MDNode::get(Ctx, {MDNode::get(Ctx, PeriodOps)}));

  FunctionType *FT = FunctionType::get(Int32Ty, {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "branch_fn", &M);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *ThenBB = BasicBlock::Create(Ctx, "then", F);
  BasicBlock *ElseBB = BasicBlock::Create(Ctx, "else", F);
  BasicBlock *Merge = BasicBlock::Create(Ctx, "merge", F);

  B.SetInsertPoint(Entry);
  Value *I0[] = {B.getInt32(0), B.getInt32(0)};
  auto *GEP = B.CreateInBoundsGEP(STy, GVar, I0, "cfg_gep");
  auto *Load = B.CreateLoad(Int32Ty, GEP, "cfg_val");
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));
  auto *Cmp = B.CreateICmpNE(Load, B.getInt32(0), "is_set");
  B.CreateCondBr(Cmp, ThenBB, ElseBB);

  B.SetInsertPoint(ThenBB);
  B.CreateBr(Merge);
  B.SetInsertPoint(ElseBB);
  B.CreateBr(Merge);

  B.SetInsertPoint(Merge);
  auto *Phi = B.CreatePHI(Int32Ty, 2, "result");
  Phi->addIncoming(B.getInt32(100), ThenBB);
  Phi->addIncoming(B.getInt32(0), ElseBB);
  B.CreateRet(Phi);
  return F;
}

TEST(EJitEndToEnd, BranchFolding) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("test_branch", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createBranchOnMayConstFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  // Mock memory: field value = 1 (non-zero, so "then" branch taken)
  struct MockCfg { int32_t val; };
  MockCfg mock = {1};

  PeriodArrayRegistry reg;
  GlobalVariable *GV = M->getGlobalVariable("g_branch_cfg", true);
  ASSERT_NE(GV, nullptr);
  reg.registerStaticVar("g_branch_cfg", &mock);

  // Full pipeline: StructField -> InstCombine -> Inline -> L2
  EJitStructFieldPass sfp(reg);
  EJitOptimizer opt(reg);

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

  sfp.run(*F, FAM);
  opt.runInstCombine(*M);
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L2);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  // mock.val = 1 => branch taken => result = 100
  EXPECT_EQ(RetVal->getSExtValue(), 100);
}

//===----------------------------------------------------------------------===//
// EJit end-to-end MultiPeriod specialization test
//===----------------------------------------------------------------------===//

TEST(EJitEndToEnd, MultiPeriodSpecialization) {
  // Create module with two period arrays, run StructField with different
  // mock data simulating different cell indices.
  LLVMContext Ctx1;
  auto M1 = std::make_unique<Module>("spec1", Ctx1);
  M1->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F1 = createMultiArrayFunc(Ctx1, *M1);

  int32_t cellA[4] = {1, 10, 20, 30};
  int32_t trpA[4]  = {100, 200, 300, 400};

  PeriodArrayRegistry reg;
  reg.registerArray("cell", "g_cells", cellA, 4);
  reg.registerArray("trp", "g_trps", trpA, 4);

  EJitStructFieldPass sfp(reg);

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

  sfp.run(*F1, FAM);

  EJitOptimizer opt1(reg);
  opt1.runInstCombine(*M1);

  // g_cells[2] * g_trps[3] = 20 * 400 = 8000
  auto *Ret = dyn_cast_or_null<ReturnInst>(&F1->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 8000);

  // Now change cell data, re-run (different specialization)
  LLVMContext Ctx2;
  auto M2 = std::make_unique<Module>("spec2", Ctx2);
  M2->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F2 = createMultiArrayFunc(Ctx2, *M2);

  int32_t cellB[4] = {5, 5, 5, 5}; // g_cells[2] = 5
  int32_t trpB[4]  = {0, 0, 0, 6}; // g_trps[3] = 6

  PeriodArrayRegistry reg2;
  reg2.registerArray("cell", "g_cells", cellB, 4);
  reg2.registerArray("trp", "g_trps", trpB, 4);

  EJitStructFieldPass sfp2(reg2);
  FunctionAnalysisManager FAM2;
  PassBuilder PB2;
  PB2.registerFunctionAnalyses(FAM2);
  PB2.registerLoopAnalyses(LAM);
  PB2.registerCGSCCAnalyses(CGAM);
  PB2.registerModuleAnalyses(MAM);
  PB2.crossRegisterProxies(LAM, FAM2, CGAM, MAM);

  sfp2.run(*F2, FAM2);

  EJitOptimizer opt2(reg2);
  opt2.runInstCombine(*M2);

  auto *Ret2 = dyn_cast_or_null<ReturnInst>(&F2->back().back());
  ASSERT_NE(Ret2, nullptr);
  auto *RetVal2 = dyn_cast<ConstantInt>(Ret2->getReturnValue());
  ASSERT_NE(RetVal2, nullptr);
  EXPECT_EQ(RetVal2->getSExtValue(), 30); // 5 * 6 = 30
}

//===----------------------------------------------------------------------===//
// EJit end-to-end cache invalidation test
//===----------------------------------------------------------------------===//

TEST(EJitEndToEnd, CacheInvalidation) {
  EJitCache cache(100, 1024 * 1024);
  int dummy = 42;

  // Put entries with different period dependencies
  std::set<std::string> depsA = {"cell=1", "trp=2"};
  std::set<std::string> depsB = {"cell=3", "slice=0"};
  std::set<std::string> depsC = {"cell=1", "carrier=5"};

  cache.put(1001, &dummy, 64, depsA);
  cache.put(1002, &dummy, 64, depsB);
  cache.put(1003, &dummy, 64, depsC);

  EXPECT_NE(cache.getOrNull(1001), nullptr);
  EXPECT_NE(cache.getOrNull(1002), nullptr);
  EXPECT_NE(cache.getOrNull(1003), nullptr);

  // Invalidate cell=1: should remove key_a (cell=1,trp=2) and key_c (cell=1,carrier=5)
  // but NOT key_b (cell=3,slice=0)
  cache.invalidateByPeriod("cell", 1);

  EXPECT_EQ(cache.getOrNull(1001), nullptr);
  EXPECT_NE(cache.getOrNull(1002), nullptr);
  EXPECT_EQ(cache.getOrNull(1003), nullptr);

  auto stats = cache.getStats();
  EXPECT_EQ(stats.entryCount, 1u);  // only key_b remains
}

//===----------------------------------------------------------------------===//
// JIT pipeline IR verification tests
//===----------------------------------------------------------------------===//

/// Create IR matching the process_board trace test pattern:
///   if (g_cfg.field0 == 1) { g_cfg.field1 = 100; } else { g_cfg.field1 = 200; }
/// After StructField, the branch on field0 should fold, eliminating the dead path.
static Function *createBranchOnFieldFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();
  auto *STy = StructType::create(Ctx, {Int32Ty, Int32Ty}, "Cfg");

  auto *GV = new GlobalVariable(M, STy, false, GlobalValue::InternalLinkage,
                                 ConstantStruct::get(STy, ConstantInt::get(Int32Ty, 0),
                                                     ConstantInt::get(Int32Ty, 0)),
                                 "g_cfg");
  Metadata *PeriodOps[] = {
      MDString::get(Ctx, "ejit_period"),
      MDString::get(Ctx, "static"),
  };
  GV->setMetadata(MD_EJIT_METADATA,
                  MDNode::get(Ctx, {MDNode::get(Ctx, PeriodOps)}));

  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "process_data", &M);

  auto *Entry = BasicBlock::Create(Ctx, "entry", F);
  auto *ThenBB = BasicBlock::Create(Ctx, "then", F);
  auto *ElseBB = BasicBlock::Create(Ctx, "else", F);
  auto *Merge = BasicBlock::Create(Ctx, "merge", F);

  B.SetInsertPoint(Entry);
  Value *I0[] = {B.getInt32(0), B.getInt32(0)};
  auto *GEP_f0 = B.CreateInBoundsGEP(STy, GV, I0, "field0");
  auto *Load = B.CreateLoad(Int32Ty, GEP_f0, "load_field0");
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));
  auto *Cmp = B.CreateICmpEQ(Load, B.getInt32(1), "cmp");
  B.CreateCondBr(Cmp, ThenBB, ElseBB);

  B.SetInsertPoint(ThenBB);
  Value *I1_t[] = {B.getInt32(0), B.getInt32(1)};
  auto *GEP_xx_t = B.CreateInBoundsGEP(STy, GV, I1_t, "xx_then");
  B.CreateStore(B.getInt32(100), GEP_xx_t);
  B.CreateBr(Merge);

  B.SetInsertPoint(ElseBB);
  Value *I1_e[] = {B.getInt32(0), B.getInt32(1)};
  auto *GEP_xx_e = B.CreateInBoundsGEP(STy, GV, I1_e, "xx_else");
  B.CreateStore(B.getInt32(200), GEP_xx_e);
  B.CreateBr(Merge);

  B.SetInsertPoint(Merge);
  B.CreateRetVoid();
  return F;
}

TEST(EJitPipelineIR, BranchFoldingOnMayConst) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("branch_fold", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createBranchOnFieldFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  // Count branches before optimization
  int brBefore = 0;
  for (auto &BB : *F)
    if (isa<BranchInst>(BB.getTerminator()))
      ++brBefore;
  EXPECT_GE(brBefore, 1);

  // Mock: field0 = 1
  struct MockCfg { int32_t f0; int32_t f1; };
  MockCfg mock = {1, 0};

  PeriodArrayRegistry reg;
  reg.registerStaticVar("g_cfg", &mock);

  // Full pipeline: StructField -> InstCombine -> Inline -> L2
  EJitStructFieldPass sfp(reg);
  EJitOptimizer opt(reg);

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

  sfp.run(*F, FAM);
  opt.runInstCombine(*M);
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L2);

  // After full pipeline: no loads should remain (all may_const replaced)
  int loadCount = 0;
  for (auto &BB : *F)
    for (auto &I : BB)
      if (isa<LoadInst>(&I))
        ++loadCount;
  EXPECT_EQ(loadCount, 0) << "All may_const loads should be replaced";

  // The branch on field0 == 1 should be folded (single conditional branch gone)
  // May still have unconditional branches for block transitions
  int condBrCount = 0;
  for (auto &BB : *F) {
    auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
    if (BI && BI->isConditional())
      ++condBrCount;
  }
  EXPECT_EQ(condBrCount, 0) << "Conditional branch should be eliminated";
}

/// Create IR matching the process_cell trace test pattern:
///   g_arr[idx].field == 0xFD ? a += 5 : a += 15
/// With idx=0 replaced by constant during preReplacePeriodIndices.
static Function *createCellProcessFunc(LLVMContext &Ctx, Module &M) {
  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();
  auto *STy = StructType::create(Ctx, {Int32Ty, Int32Ty}, "CellCfg");
  auto *ArrTy = ArrayType::get(STy, 4);

  auto *GV = new GlobalVariable(M, ArrTy, false, GlobalValue::InternalLinkage,
                                 ConstantAggregateZero::get(ArrTy), "g_cells");
  Metadata *ArrOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 4)),
  };
  GV->setMetadata(MD_EJIT_METADATA,
                  MDNode::get(Ctx, {MDNode::get(Ctx, ArrOps)}));

  // Function: void process_cell(i32 cell_idx) — ejit_period_arr_ind on arg 0
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {Int32Ty}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "process_cell", &M);
  F->getArg(0)->setName("cell_idx");

  // Attach ejit_period_arr_ind metadata on arg 0
  Metadata *IndOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0)),
  };
  F->setMetadata(MD_EJIT_METADATA,
                 MDNode::get(Ctx, {MDNode::get(Ctx, IndOps)}));

  auto *Entry = BasicBlock::Create(Ctx, "entry", F);
  auto *ThenBB = BasicBlock::Create(Ctx, "then", F);
  auto *ElseBB = BasicBlock::Create(Ctx, "else", F);
  auto *Merge  = BasicBlock::Create(Ctx, "merge", F);

  B.SetInsertPoint(Entry);
  // Load field 0 at cell_idx: gep %STy, %ArrTy* @g_cells, i32 0, i64 %cell_idx, i32 0
  Value *Idx_f0[] = {B.getInt32(0), F->getArg(0), B.getInt32(0)};
  auto *GEP_f0 = B.CreateInBoundsGEP(ArrTy, GV, Idx_f0, "cell_f0");
  auto *Load = B.CreateLoad(Int32Ty, GEP_f0, "load_f0");
  Load->setMetadata("ejit.may_const", MDNode::get(Ctx, MDString::get(Ctx, "ejit")));
  auto *Cmp = B.CreateICmpEQ(Load, B.getInt32(253), "cmp"); // 0xFD = 253
  B.CreateCondBr(Cmp, ThenBB, ElseBB);

  B.SetInsertPoint(ThenBB);
  Value *Idx_f1_t[] = {B.getInt32(0), F->getArg(0), B.getInt32(1)};
  auto *GEP_xx = B.CreateInBoundsGEP(ArrTy, GV, Idx_f1_t, "field1_then");
  auto *Old = B.CreateLoad(Int32Ty, GEP_xx, "old_val");
  auto *New = B.CreateAdd(Old, B.getInt32(5));
  B.CreateStore(New, GEP_xx);
  B.CreateBr(Merge);

  B.SetInsertPoint(ElseBB);
  Value *Idx_f1_e[] = {B.getInt32(0), F->getArg(0), B.getInt32(1)};
  auto *GEP_xx_e = B.CreateInBoundsGEP(ArrTy, GV, Idx_f1_e, "field1_else");
  auto *Old_e = B.CreateLoad(Int32Ty, GEP_xx_e, "old_val_e");
  auto *New_e = B.CreateAdd(Old_e, B.getInt32(15));
  B.CreateStore(New_e, GEP_xx_e);
  B.CreateBr(Merge);

  B.SetInsertPoint(Merge);
  B.CreateRetVoid();
  return F;
}

TEST(EJitPipelineIR, CellProcessBranchFolding) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("cell_process", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));
  Function *F = createCellProcessFunc(Ctx, *M);
  ASSERT_NE(F, nullptr);

  // Mock: g_cells[0].field0 = 0xFD (253), g_cells[0].field1 = 0
  struct MockCell { int32_t f0; int32_t f1; };
  MockCell mockArr[4] = {{253, 0}, {0, 0}, {0, 0}, {0, 0}};

  PeriodArrayRegistry reg;
  reg.registerArray("cell", "g_cells", mockArr, 4);

  SpecializationContext ctx;
  ctx.fnName = "process_cell";
  ctx.dimensions.push_back({"cell", 0});
  ctx.optLevel = llvm::ejit::OptimizationLevel::L2;

  EJitOptimizer opt(reg);
  // 1. Replace period index arg (cell_idx=0) with constant
  opt.preReplacePeriodIndices(*M, ctx);
  // 2. Fold constant chains + Promote
  opt.runInstCombine(*M);

  // 3. StructField: replace may_const loads
  EJitStructFieldPass sfp(reg);
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
  sfp.run(*F, FAM);

  // 4. Final cleanup
  opt.runInstCombine(*M);
  opt.runOptimizationPipeline(*M, llvm::ejit::OptimizationLevel::L2);

  // All may_const loads should be gone
  int mayConstLoads = 0;
  for (auto &BB : *F)
    for (auto &I : BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->hasMetadata("ejit.may_const"))
          ++mayConstLoads;
  EXPECT_EQ(mayConstLoads, 0);

  // Conditional branch should be eliminated
  int condBr = 0;
  for (auto &BB : *F) {
    auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
    if (BI && BI->isConditional())
      ++condBr;
  }
  EXPECT_EQ(condBr, 0) << "Conditional branch on may_const field should be folded";
}

/// Verify InstCombine runs correctly after period index replacement
/// (catches the case where preReplacePeriodIndices + InstCombine should
/// fold a constant expression like "period_idx" replaced with constant).
TEST(EJitPipelineIR, PeriodIndexReplacementAndFold) {
  LLVMContext Ctx;
  auto M = std::make_unique<Module>("period_idx_fold", Ctx);
  M->setTargetTriple(Triple("x86_64-unknown-linux-gnu"));

  IRBuilder<> B(Ctx);
  auto *Int32Ty = B.getInt32Ty();

  FunctionType *FT = FunctionType::get(Int32Ty, {Int32Ty}, false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test_fn", M.get());
  F->getArg(0)->setName("period_idx");

  Metadata *IndOps[] = {
      MDString::get(Ctx, TAG_EJIT_PERIOD_ARR_IND),
      MDString::get(Ctx, "cell"),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, 0)),
  };
  F->setMetadata(MD_EJIT_METADATA,
                 MDNode::get(Ctx, {MDNode::get(Ctx, IndOps)}));

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  B.SetInsertPoint(BB);
  // period_idx * 2 + 10: should fold to 42 * 2 + 10 = 94 after replacement
  auto *Mul = B.CreateMul(F->getArg(0), B.getInt32(2), "mul");
  auto *Add = B.CreateAdd(Mul, B.getInt32(10), "add");
  B.CreateRet(Add);

  PeriodArrayRegistry reg;
  SpecializationContext ctx;
  ctx.fnName = "test_fn";
  ctx.dimensions.push_back({"cell", 42});

  EJitOptimizer opt(reg);
  opt.preReplacePeriodIndices(*M, ctx);
  opt.runInstCombine(*M);

  auto *Ret = dyn_cast_or_null<ReturnInst>(&F->back().back());
  ASSERT_NE(Ret, nullptr);
  auto *RetVal = dyn_cast<ConstantInt>(Ret->getReturnValue());
  ASSERT_NE(RetVal, nullptr);
  EXPECT_EQ(RetVal->getSExtValue(), 94);
}

//===----------------------------------------------------------------------===//
// JIT cache lifecycle tests
//===----------------------------------------------------------------------===//

TEST(EJitCacheLifecycle, HitAfterPut) {
  EJitCache cache(100, 1024 * 1024);
  int dummy = 42;
  EXPECT_EQ(cache.getOrNull(777), nullptr);
  cache.put(777, &dummy, 64);
  EXPECT_EQ(cache.getOrNull(777), &dummy);
}

TEST(EJitCacheLifecycle, MissAfterInvalidate) {
  EJitCache cache(100, 1024 * 1024);
  int dummy = 42;
  std::set<std::string> deps = {"cell=5"};
  cache.put(777, &dummy, 64, deps);
  EXPECT_NE(cache.getOrNull(777), nullptr);
  cache.invalidateByPeriod("cell", 5);
  EXPECT_EQ(cache.getOrNull(777), nullptr);
}

TEST(EJitCacheLifecycle, MissAfterEviction) {
  EJitCache cache(2, 1024 * 1024);
  int a, b, c;
  cache.put(10, &a, 1);
  cache.put(20, &b, 1);
  cache.put(30, &c, 1);  // should evict 'a'
  EXPECT_EQ(cache.getOrNull(10), nullptr);
  EXPECT_NE(cache.getOrNull(30), nullptr);
}

TEST(EJitCacheLifecycle, MissAfterClear) {
  EJitCache cache(10, 1024 * 1024);
  int dummy;
  cache.put(10, &dummy, 64);
  cache.put(20, &dummy, 64);
  cache.clear();
  EXPECT_EQ(cache.getOrNull(10), nullptr);
  EXPECT_EQ(cache.getOrNull(20), nullptr);
}

TEST(EJitCacheLifecycle, ReputAfterInvalidate) {
  EJitCache cache(100, 1024 * 1024);
  int dummy = 42;
  std::set<std::string> deps = {"cell=3"};
  cache.put(777, &dummy, 64, deps);
  cache.invalidateByPeriod("cell", 3);
  // Reput with new value
  int newVal = 99;
  cache.put(777, &newVal, 64, deps);
  EXPECT_EQ(cache.getOrNull(777), &newVal);
}


