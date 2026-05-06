//===- JSONParserBM.cpp - JSON parser benchmarks --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Benchmarks for LLVM's JSON parser.
// Runs parsing, tree iteration, and object key lookup with generated inputs.
// Measures time performance and memory consumption.
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <atomic>
#include <random>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Memory tracking via global operator new
//
// These are global overrides so the benchmark might be over-counting memory
// allocations and usage. The data is still useful for comparisons with this
// benchmark itself.
//===----------------------------------------------------------------------===//

static std::atomic_size_t TotalAllocatedBytes{0};
static std::atomic_size_t NumAllocs{0};
static bool TrackMemory = false;

// Single-object new/delete.
void *operator new(std::size_t Size) {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new(std::size_t Size, std::align_val_t) {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new(std::size_t Size, const std::nothrow_t &) noexcept {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new(std::size_t Size, std::align_val_t,
                   const std::nothrow_t &) noexcept {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void operator delete(void *Ptr) noexcept { std::free(Ptr); }
void operator delete(void *Ptr, std::align_val_t) noexcept { std::free(Ptr); }
void operator delete(void *Ptr, std::size_t) noexcept { std::free(Ptr); }
void operator delete(void *Ptr, std::size_t, std::align_val_t) noexcept {
  std::free(Ptr);
}

// Array new/delete.
void *operator new[](std::size_t Size) {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new[](std::size_t Size, std::align_val_t) {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new[](std::size_t Size, const std::nothrow_t &) noexcept {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void *operator new[](std::size_t Size, std::align_val_t,
                     const std::nothrow_t &) noexcept {
  if (TrackMemory) {
    TotalAllocatedBytes += Size;
    ++NumAllocs;
  }
  return std::malloc(Size);
}

void operator delete[](void *Ptr) noexcept { std::free(Ptr); }
void operator delete[](void *Ptr, std::align_val_t) noexcept { std::free(Ptr); }
void operator delete[](void *Ptr, std::size_t) noexcept { std::free(Ptr); }
void operator delete[](void *Ptr, std::size_t, std::align_val_t) noexcept {
  std::free(Ptr);
}

//===----------------------------------------------------------------------===//
// Test data generation
//===----------------------------------------------------------------------===//

/// Generate a JSON string with \p N entries in an array. Each entry is a nested
/// structure with objects and arrays to exercise parsing, iteration, and lookup
/// at multiple depths.
///
/// Structure:
///   {"items": [
///     {
///       "name": "item_I",
///       "value": I,
///       "tags": [
///         {"label": "tag_0", "priority": 0},
///         ...
///       ],
///       "details": {
///         "description": "description text for item I",
///         "active": true/false,
///         "nested": { "x": I, "y": I*100 }
///       }
///     },
///     ...
///   ]}
static std::string generateJSON(int N) {
  std::string S;
  raw_string_ostream OS(S);
  OS << "{\"items\": [\n";
  for (int I = 0; I < N; ++I) {
    if (I > 0)
      OS << ",\n";
    OS << "  {\n"
       << "    \"name\": \"item_" << I << "\",\n"
       << "    \"value\": " << I << ",\n"
       << "    \"tags\": [\n"
       << "      {\"label\": \"tag_0\", \"priority\": 0},\n"
       << "      {\"label\": \"tag_1\", \"priority\": 1},\n"
       << "      {\"label\": \"tag_2\", \"priority\": 2}\n"
       << "    ],\n"
       << "    \"details\": {\n"
       << "      \"description\": \"description text for item " << I << "\",\n"
       << "      \"active\": " << (I % 2 == 0 ? "true" : "false") << ",\n"
       << "      \"nested\": {\"x\": " << I << ", \"y\": " << I * 100 << "}\n"
       << "    }\n"
       << "  }";
  }
  OS << "\n]}";
  return S;
}

//===----------------------------------------------------------------------===//
// Tree traversal helpers
//===----------------------------------------------------------------------===//

/// Walk the JSON value tree, visiting every node. Returns the number of
/// nodes visited.
static size_t walkTree(const json::Value &V) {
  size_t Count = 1;
  if (const auto *Obj = V.getAsObject()) {
    for (const auto &KV : *Obj)
      Count += walkTree(KV.second);
  } else if (const auto *Arr = V.getAsArray()) {
    for (const auto &Elem : *Arr)
      Count += walkTree(Elem);
  }
  return Count;
}

/// An Object paired with its own keys, for lookup benchmarks.
struct ObjectWithKeys {
  const json::Object *Obj;
  SmallVector<std::string> Keys;
};

/// Collect every Object in the tree together with its own keys.
static void collectObjectsWithKeys(const json::Value &V,
                                   SmallVectorImpl<ObjectWithKeys> &Result) {
  if (const auto *Obj = V.getAsObject()) {
    ObjectWithKeys Entry;
    Entry.Obj = Obj;
    for (const auto &KV : *Obj) {
      Entry.Keys.push_back(std::string(StringRef(KV.first)));
      collectObjectsWithKeys(KV.second, Result);
    }
    Result.push_back(std::move(Entry));
  } else if (const auto *Arr = V.getAsArray()) {
    for (const auto &Elem : *Arr)
      collectObjectsWithKeys(Elem, Result);
  }
}

//===----------------------------------------------------------------------===//
// Benchmarks
//===----------------------------------------------------------------------===//

/// Benchmark json::parse(). Reports parse throughput and memory allocated.
static void BM_JSONParse(benchmark::State &State) {
  std::string JSON = generateJSON(State.range(0));

  // Measure memory for a single parse before the timed loop.
  TotalAllocatedBytes = 0;
  NumAllocs = 0;
  TrackMemory = true;
  {
    auto V = json::parse(JSON);
    benchmark::DoNotOptimize(V);
  }
  TrackMemory = false;

  State.counters["AllocBytes"] = TotalAllocatedBytes.load();
  State.counters["Allocs"] = NumAllocs.load();

  for (auto _ : State) {
    auto V = json::parse(JSON);
    benchmark::DoNotOptimize(V);
  }
  State.counters["ParseByteRate"] = benchmark::Counter(
      State.iterations() * JSON.size(), benchmark::Counter::kIsRate,
      benchmark::Counter::kIs1024);
}
BENCHMARK(BM_JSONParse)->Arg(10)->Arg(1000)->Arg(100000);

/// Benchmark recursive tree iteration over a parsed JSON value.
static void BM_JSONIterate(benchmark::State &State) {
  std::string JSON = generateJSON(State.range(0));
  json::Value Root = cantFail(json::parse(JSON));
  size_t NodeCount = 0;
  for (auto _ : State) {
    NodeCount = walkTree(Root);
    benchmark::DoNotOptimize(NodeCount);
  }
  State.SetItemsProcessed(State.iterations() * NodeCount);
}
BENCHMARK(BM_JSONIterate)->Arg(10)->Arg(1000)->Arg(100000);

/// Benchmark Object::get() with each object's own keys in insertion order.
static void BM_JSONLookupSequential(benchmark::State &State) {
  std::string JSON = generateJSON(State.range(0));
  json::Value Root = cantFail(json::parse(JSON));
  SmallVector<ObjectWithKeys> ObjKeys;
  collectObjectsWithKeys(Root, ObjKeys);

  size_t TotalLookups = 0;
  for (const auto &OK : ObjKeys)
    TotalLookups += OK.Keys.size();

  for (auto _ : State) {
    for (const auto &OK : ObjKeys)
      for (const auto &K : OK.Keys)
        benchmark::DoNotOptimize(OK.Obj->get(K));
  }
  State.SetItemsProcessed(State.iterations() * TotalLookups);
}
BENCHMARK(BM_JSONLookupSequential)->Arg(10)->Arg(1000)->Arg(100000);

/// Benchmark Object::get() with each object's own keys in random order.
static void BM_JSONLookupRandom(benchmark::State &State) {
  std::string JSON = generateJSON(State.range(0));
  json::Value Root = cantFail(json::parse(JSON));
  SmallVector<ObjectWithKeys> ObjKeys;
  collectObjectsWithKeys(Root, ObjKeys);

  std::mt19937 RNG(42);
  size_t TotalLookups = 0;
  for (auto &OK : ObjKeys) {
    TotalLookups += OK.Keys.size();
    std::shuffle(OK.Keys.begin(), OK.Keys.end(), RNG);
  }

  for (auto _ : State) {
    for (const auto &OK : ObjKeys)
      for (const auto &K : OK.Keys)
        benchmark::DoNotOptimize(OK.Obj->get(K));
  }
  State.SetItemsProcessed(State.iterations() * TotalLookups);
}
BENCHMARK(BM_JSONLookupRandom)->Arg(10)->Arg(1000)->Arg(100000);

BENCHMARK_MAIN();
