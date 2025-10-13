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

std::vector<std::string> genFiles(size_t N) {
  std::vector<std::string> R;
  R.reserve(N);
  std::mt19937 Rng(RNG_SEED);
  std::uniform_int_distribution<> DepthDistrib(8, 16);
  std::uniform_int_distribution<> WordDistrib(0, std::size(Dictionary) - 1);

  std::string S;
  for (size_t i = 0; i < N; ++i) {
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

std::string genFixed(const std::vector<std::string> &Files) {
  std::string S;
  for (const auto &F : Files) {
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genMid(const std::vector<std::string> &Files) {
  std::string S;
  std::mt19937 Rng(RNG_SEED);
  for (std::string F : Files) {
    std::uniform_int_distribution<> PosDistrib(0, F.size() - 1);
    F[PosDistrib(Rng)] = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genStart(const std::vector<std::string> &Files) {
  std::string S;
  for (auto F : Files) {
    F.front() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genEnd(const std::vector<std::string> &Files) {
  std::string S;
  for (auto F : Files) {
    F.back() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genBoth(const std::vector<std::string> &Files) {
  std::string S;
  for (auto F : Files) {
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
  std::mt19937 Rng(RNG_SEED);
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
  std::mt19937 Rng(RNG_SEED);
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

BENCHMARK_CAPTURE(BM_Make_, None_, genFixed)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Start, genStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, End__, genEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Mid__, genMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_Make_, Both_, genBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_CAPTURE(BM_True_, None_, genFixed)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Start, genStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, End__, genEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Mid__, genMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_True_, Both_, genBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_CAPTURE(BM_False, None_, genFixed)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Start, genStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, End__, genEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Mid__, genMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);
BENCHMARK_CAPTURE(BM_False, Both_, genBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(MAX_LIST_MIN, MAX_LIST_MAX);

BENCHMARK_MAIN();
