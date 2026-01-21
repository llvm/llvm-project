//===- writeToOutputInParallel.cpp - Parallel file writing benchmark ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llvm;

// Benchmark parallel file writing using writeToOutput.
// This simulates scenarios where multiple threads are writing files
// concurrently, which is common in parallel compilation scenarios.
static void BM_WriteToOutputInParallel(benchmark::State &State) {
  const int NumThreads = State.range(0);
  const int FilesPerThread = State.range(1);
  const int BytesPerFile = 40 * 1024; // ~40KB per file

  for (auto _ : State) {
    // Pause timing while we set up temporary directories
    State.PauseTiming();
    
    // Create temporary directories for each thread
    std::vector<std::unique_ptr<llvm::unittest::TempDir>> TempDirs;
    for (int i = 0; i < NumThreads; ++i) {
      TempDirs.push_back(std::make_unique<llvm::unittest::TempDir>(
          "writeToOutputBM", /*Unique*/ true));
    }
    
    State.ResumeTiming();

    // Launch threads, each writing multiple files
    std::vector<std::thread> Threads;
    for (int threadIdx = 0; threadIdx < NumThreads; ++threadIdx) {
      Threads.emplace_back([&, threadIdx]() {
        const auto &TempDir = TempDirs[threadIdx];
        for (int fileIdx = 0; fileIdx < FilesPerThread; ++fileIdx) {
          SmallString<128> Path(TempDir->path());
          sys::path::append(Path, "file_" + std::to_string(fileIdx) + ".bin");

          // Ignore errors in benchmark - we're measuring performance, not
          // correctness
          (void)writeToOutput(Path, [BytesPerFile](raw_ostream &Out) -> Error {
            // Write 32-bit integers up to BytesPerFile
            const int NumInts = BytesPerFile / sizeof(int32_t);
            for (int32_t i = 0; i < NumInts; ++i) {
              Out.write(reinterpret_cast<const char *>(&i), sizeof(i));
            }
            return Error::success();
          });
        }
      });
    }

    // Wait for all threads to complete
    for (auto &Thread : Threads) {
      Thread.join();
    }

    // Cleanup happens automatically when TempDirs goes out of scope
  }

  // Report throughput metrics
  const int64_t TotalFiles = NumThreads * FilesPerThread;
  const int64_t TotalBytes = TotalFiles * BytesPerFile;
  
  State.SetItemsProcessed(State.iterations() * TotalFiles);
  State.SetBytesProcessed(State.iterations() * TotalBytes);
  State.counters["threads"] = NumThreads;
  State.counters["files_per_thread"] = FilesPerThread;
  State.counters["total_files"] = TotalFiles;
}

// Test various combinations of thread counts and files per thread
// These represent different parallelism scenarios:
// - Low parallelism, many files per thread (serial-like workload)
// - High parallelism, few files per thread (highly parallel workload)
// - Balanced scenarios

BENCHMARK(BM_WriteToOutputInParallel)
    ->Args({1, 1000})      // 1 thread, 1000 files
    ->Args({2, 500})       // 2 threads, 500 files each
    ->Args({4, 250})       // 4 threads, 250 files each
    ->Args({8, 125})       // 8 threads, 125 files each
    ->Args({10, 100})      // 10 threads, 100 files each
    ->Args({10, 1000})     // 10 threads, 1000 files each (stress test)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
