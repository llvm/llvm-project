#include "llvm/ADT/SparseBitVector.h"
#include "benchmark/benchmark.h"
using namespace llvm;

static unsigned xorshift(unsigned State) {
  State ^= State << 13;
  State ^= State >> 17;
  State ^= State << 5;
  return State;
}

static void BM_SparseBitVectorIterator(benchmark::State &State) {
  SparseBitVector<> BV;

  unsigned Prev = 0xcafebabe;
  for (unsigned I = 0, E = State.range(0); I != E; ++I)
    BV.set((Prev = xorshift(Prev)) % 10000);

  for (auto _ : State) {
    unsigned Total = 0;
    for (auto I = BV.begin(), E = BV.end(); I != E; ++I)
      Total += *I;
    benchmark::DoNotOptimize(Total);
  }
}

BENCHMARK(BM_SparseBitVectorIterator)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

BENCHMARK_MAIN();
