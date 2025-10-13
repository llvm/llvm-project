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
constexpr int MAX_LIST_SIZE = 100000;
constexpr int MAX_LIST_MUL = 1000;

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

std::string genFixedPath(const std::vector<std::string> &Files) {
  std::string S;
  for (const auto &F : Files) {
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobMid(const std::vector<std::string> &Files) {
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

std::string genGlobStart(const std::vector<std::string> &Files) {
  std::string S;
  for (auto F : Files) {
    F.front() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobEnd(const std::vector<std::string> &Files) {
  std::string S;
  for (auto F : Files) {
    F.back() = '*';
    S += "src:";
    S += F;
    S += "\n";
  }
  return S;
}

std::string genGlobBoth(const std::vector<std::string> &Files) {
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

void BM_Create(
    benchmark::State &state,
    std::string (*GenerateCaseList)(const std::vector<std::string> &Files)) {
  std::vector<std::string> BigFileList = genFiles(state.range(0));
  std::string BigCaseList = GenerateCaseList(BigFileList);
  for (auto _ : state) {
    auto SCL = makeSpecialCaseList(BigCaseList);
    benchmark::DoNotOptimize(SCL);
  }
}
void BM_Pos(
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

void BM_Neg(
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

BENCHMARK_CAPTURE(BM_Create, Exact, genFixedPath)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Create, Start, genGlobStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Create, End, genGlobEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Create, Mid, genGlobMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Create, EndBoth, genGlobBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);

BENCHMARK_CAPTURE(BM_Pos, Exact, genFixedPath)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Pos, Start, genGlobStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Pos, End, genGlobEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Pos, Mid, genGlobMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Pos, Both, genGlobBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);

BENCHMARK_CAPTURE(BM_Neg, Exact, genFixedPath)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Neg, Start, genGlobStart)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Neg, End, genGlobEnd)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Neg, Mid, genGlobMid)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);
BENCHMARK_CAPTURE(BM_Neg, End, genGlobBoth)
    ->RangeMultiplier(MAX_LIST_MUL)
    ->Range(1, MAX_LIST_SIZE);

BENCHMARK_MAIN();
