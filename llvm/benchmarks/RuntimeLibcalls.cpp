//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "benchmark/benchmark.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Triple.h"
#include <random>
#include <string>
using namespace llvm;

static constexpr unsigned MaxFuncNameSize = 53;

static std::vector<StringRef> getLibcallNameStringRefs() {
  std::vector<StringRef> Names(RTLIB::NumLibcallImpls);
  // Keep the strlens on the StringRef construction out of the benchmark loop.
  for (RTLIB::LibcallImpl LC : RTLIB::libcall_impls())
    Names[LC] = RTLIB::RuntimeLibcallsInfo::getLibcallImplName(LC);

  return Names;
}

static std::vector<std::string> getRandomFuncNames() {
  std::mt19937_64 Rng;
  std::uniform_int_distribution<> StringLengthDistribution(1, MaxFuncNameSize);
  std::uniform_int_distribution<> CharDistribution(1, 255);
  int NumTestFuncs = 1 << 10;
  std::vector<std::string> TestFuncNames(NumTestFuncs);

  for (std::string &TestFuncName : TestFuncNames) {
    for (int I = 0, E = StringLengthDistribution(Rng); I != E; ++I)
      TestFuncName += static_cast<char>(CharDistribution(Rng));
  }

  return TestFuncNames;
}

#ifdef SYMBOL_TEST_DATA_FILE
static std::vector<std::string> readSymbolsFromFile(StringRef InputFile) {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFile, /*IsText=*/true);
  if (!BufOrError) {
    reportFatalUsageError("failed to open \'" + Twine(InputFile) +
                          "\': " + BufOrError.getError().message());
  }

  // Hackily figure out if there's a prefix on the symbol names - llvm-nm
  // appears to not have a flag to skip this.
  llvm::Triple HostTriple(LLVM_HOST_TRIPLE);
  std::string DummyDatalayout = "e";
  DummyDatalayout += DataLayout::getManglingComponent(HostTriple);

  DataLayout DL(DummyDatalayout);
  char GlobalPrefix = DL.getGlobalPrefix();

  std::vector<std::string> Lines;
  for (line_iterator LineIt(**BufOrError, /*SkipBlanks=*/true);
       !LineIt.is_at_eof(); ++LineIt) {
    StringRef SymbolName = *LineIt;
    SymbolName.consume_front(StringRef(&GlobalPrefix, 1));

    Lines.push_back(SymbolName.str());
  }
  return Lines;
}
#endif

static void BM_LookupRuntimeLibcallByNameKnownCalls(benchmark::State &State) {
  std::vector<StringRef> Names = getLibcallNameStringRefs();

  for (auto _ : State) {
    for (StringRef Name : Names) {
      benchmark::DoNotOptimize(
          RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName(Name).empty());
    }
  }
}

static void BM_LookupRuntimeLibcallByNameRandomCalls(benchmark::State &State) {
  std::vector<std::string> TestFuncNames = getRandomFuncNames();

  for (auto _ : State) {
    for (const std::string &Name : TestFuncNames) {
      benchmark::DoNotOptimize(
          RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName(StringRef(Name))
              .empty());
    }
  }
}

#ifdef SYMBOL_TEST_DATA_FILE
// This isn't fully representative, it doesn't include any anonymous functions.
// nm -n --no-demangle --format=just-symbols sample-binary > sample.txt
static void BM_LookupRuntimeLibcallByNameSampleData(benchmark::State &State) {
  std::vector<std::string> TestFuncNames =
      readSymbolsFromFile(SYMBOL_TEST_DATA_FILE);
  for (auto _ : State) {
    for (const std::string &Name : TestFuncNames) {
      benchmark::DoNotOptimize(
          RTLIB::RuntimeLibcallsInfo::lookupLibcallImplName(StringRef(Name))
              .empty());
    }
  }
}
#endif

BENCHMARK(BM_LookupRuntimeLibcallByNameKnownCalls);
BENCHMARK(BM_LookupRuntimeLibcallByNameRandomCalls);

#ifdef SYMBOL_TEST_DATA_FILE
BENCHMARK(BM_LookupRuntimeLibcallByNameSampleData);
#endif

BENCHMARK_MAIN();
