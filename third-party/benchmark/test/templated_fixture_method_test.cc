
#include <cassert>
#include <memory>

#include "benchmark/benchmark.h"

template <typename T>
class MyFixture : public ::benchmark::Fixture {
 public:
  MyFixture() : data(0) {}

  T data;

  using type = T;
};

BENCHMARK_TEMPLATE_METHOD_F(MyFixture, Foo)(benchmark::State& st) {
  for (auto _ : st) {
    this->data += typename Base::type(1);
  }
}

BENCHMARK_TEMPLATE_INSTANTIATE_F(MyFixture, Foo, int);
BENCHMARK_TEMPLATE_INSTANTIATE_F(MyFixture, Foo, double);

BENCHMARK_MAIN();
