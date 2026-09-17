//===- writeToOutputInParallel.cpp - Parallel file writing benchmark ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmark/benchmark.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <thread>
#include <vector>

using namespace llvm;

// Benchmark parallel file writing using writeToOutput. This simulates scenarios
// where multiple threads are writing files concurrently, which is common in
// parallel compilation scenarios. The goal of this benchmark is to ensure that
// LLVM's global signal handling state updates aren't too expensive.
static void BM_WriteToOutputInParallel(benchmark::State &State) {
  const int NumThreads = State.range(0);
  const int FilesPerThread = State.range(1);
  const int BytesPerFile = 40 * 1024; // ~40KB per file

  for (auto _ : State) {
    // Create one top-level unique directory
    SmallString<128> TopLevelDir;
    if (sys::fs::createUniqueDirectory("writeToOutputBM", TopLevelDir)) {
      State.SkipWithError("Failed to create temporary directory");
      return;
    }
    auto Cleanup =
        llvm::scope_exit([&]() { sys::fs::remove_directories(TopLevelDir); });

    // Create subdirectories for each thread within the top-level directory
    std::vector<SmallString<128>> ThreadDirs;
    for (int I = 0; I < NumThreads; ++I) {
      SmallString<128> ThreadDir(TopLevelDir);
      sys::path::append(ThreadDir, "thread_" + std::to_string(I));
      if (sys::fs::create_directory(ThreadDir)) {
        State.SkipWithError("Failed to create thread directory");
        return;
      }
      ThreadDirs.push_back(ThreadDir);
    }

    // Launch threads, each writing multiple files
    std::vector<std::thread> Threads;
    for (int ThreadIdx = 0; ThreadIdx < NumThreads; ++ThreadIdx) {
      Threads.emplace_back([&, ThreadIdx]() {
        const auto &ThreadDir = ThreadDirs[ThreadIdx];
        for (int FileIdx = 0; FileIdx < FilesPerThread; ++FileIdx) {
          SmallString<128> Path(ThreadDir);
          sys::path::append(Path, "file_" + std::to_string(FileIdx) + ".bin");

          Error E = writeToOutput(Path, [=](raw_ostream &Out) -> Error {
            // Write 32-bit integers up to BytesPerFile
            const int NumInts = BytesPerFile / sizeof(int32_t);
            for (int32_t I = 0; I < NumInts; ++I) {
              Out.write(reinterpret_cast<const char *>(&I), sizeof(I));
            }
            return Error::success();
          });
          if (E) {
            State.SkipWithError("Failed to create outputfile " +
                                Path.str().str());
            return;
          }
        }
      });
    }

    // Wait for all threads to complete
    for (auto &Thread : Threads) {
      Thread.join();
    }

    // Cleanup happens automatically via scope_exit
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
    ->Args({1, 1000})  // 1 thread, 1000 files
    ->Args({2, 500})   // 2 threads, 500 files each
    ->Args({4, 250})   // 4 threads, 250 files each
    ->Args({8, 125})   // 8 threads, 125 files each
    ->Args({10, 100})  // 10 threads, 100 files each
    ->Args({10, 1000}) // 10 threads, 1000 files each (stress test)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
