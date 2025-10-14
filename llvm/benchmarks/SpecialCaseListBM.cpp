#include "benchmark/benchmark.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {
constexpr int RNG_SEED = 123456;
constexpr int MAX_LIST_MIN = 10;
constexpr int MAX_LIST_MAX = 1000000;
constexpr int MAX_LIST_MUL = 10;

std::unique_ptr<SpecialCaseList> makeSpecialCaseList(StringRef List) {
  std::string Error;
  std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(List);
  auto SCL = SpecialCaseList::create(MB.get(), Error);
  assert(SCL);
  assert(Error == "");
  return SCL;
}

static const std::string Dictionary[] = {
    "orange",   "tabby",  "tortie", "tuxedo", "void",
    "multiple", "spaces", "cute",   "fluffy", "kittens",
};

std::vector<std::string> genFiles(size_t NumFiles) {
  std::vector<std::string> R;
  R.reserve(NumFiles);
  std::minstd_rand Rng(RNG_SEED);
  std::uniform_int_distribution<> DepthDistrib(8, 16);
  std::uniform_int_distribution<> WordDistrib(0, std::size(Dictionary) - 1);

  std::string S;
  for (size_t I = 0; I < NumFiles; ++I) {
    for (size_t D = DepthDistrib(Rng); D; --D) {
      S += Dictionary[WordDistrib(Rng)];
      if (D > 1)
        S += "/";
    }
    R.push_back(std::move(S));
    S.clear();
  }
  return R;
}

std::string genGlobNone(const std::vector<std::string> &Files) {
  std::string S;
  for (const auto &F : Files) {
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobInMid(const std::vector<std::string> &Files) {
  std::string S;
  std::minstd_rand Rng(RNG_SEED);
  for (std::string F : Files) {
    std::uniform_int_distribution<> PosDistrib(0, F.size() - 1);
    F[PosDistrib(Rng)] = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobAtStart(const std::vector<std::string> &Files) {
  std::string S;
  for (std::string F : Files) {
    F.front() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobAtEnd(const std::vector<std::string> &Files) {
  std::string S;
  for (std::string F : Files) {
    F.back() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobAtBothSides(const std::vector<std::string> &Files) {
  std::string S;
  for (std::string F : Files) {
    F.back() = '*';
    F.front() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

void BM_Make_(
    benchmark::State &state,
    std::string (*GenerateCaseList)(const std::vector<std::string> &Files)) {
  std::vector<std::string> BigFileList = genFiles(state.range(0));
  std::string BigCaseList = GenerateCaseList(BigFileList);
  for (auto _ : state) {
    auto SCL = makeSpecialCaseList(BigCaseList);
    benchmark::DoNotOptimize(SCL);
  }
}
void BM_True_(
    benchmark::State &state,
    std::string (*GenerateCaseList)(const std::vector<std::string> &Files)) {
  std::vector<std::string> BigFileList = genFiles(state.range(0));
  std::string BigCaseList = GenerateCaseList(BigFileList);
  auto SCL = makeSpecialCaseList(BigCaseList);
  std::minstd_rand Rng(RNG_SEED);
  std::uniform_int_distribution<> LineDistrib(0, BigFileList.size() - 1);
  for (auto _ : state) {
    auto &Q = BigFileList[LineDistrib(Rng)];
    bool R = SCL->inSection("", "src", Q);
    if (!R)
      abort();
    benchmark::DoNotOptimize(R);
  }
}

void BM_False(
    benchmark::State &state,
    std::string (*GenerateCaseList)(const std::vector<std::string> &Files)) {
  std::vector<std::string> BigFileList = genFiles(state.range(0));
  std::string BigCaseList = GenerateCaseList(BigFileList);
  auto SCL = makeSpecialCaseList(BigCaseList);
  std::minstd_rand Rng(RNG_SEED);
  std::uniform_int_distribution<> LineDistrib(0, BigFileList.size() - 1);
  for (auto _ : state) {
    std::string Q = BigFileList[LineDistrib(Rng)];
    std::uniform_int_distribution<> PosDistrib(0, Q.size() - 1);
    Q[PosDistrib(Rng)] = '_';
    bool R = SCL->inSection("", "src", Q);
    benchmark::DoNotOptimize(R);
  }
}

} // namespace

BENCHMARK_CAPTURE(BM_Make_, None_, genGlobNone)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Start, genGlobAtStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, End__, genGlobAtEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Mid__, genGlobInMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Both_, genGlobAtBothSides)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_CAPTURE(BM_True_, None_, genGlobNone)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Start, genGlobAtStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, End__, genGlobAtEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Mid__, genGlobInMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Both_, genGlobAtBothSides)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_CAPTURE(BM_False, None_, genGlobNone)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Start, genGlobAtStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, End__, genGlobAtEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Mid__, genGlobInMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Both_, genGlobAtBothSides)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_MAIN();
