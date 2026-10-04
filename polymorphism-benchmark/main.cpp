//===- main.cpp - Polymorphism overhead benchmark driver ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compares the runtime overhead of four styles of polymorphism, each
// implemented in its own file as a minimal, self-contained model:
//
//  Virtual.cpp: C++ virtual methods and dynamic_cast.
//  Classof.cpp: LLVM/Clang-style kind enum, classof(), isa<>/dyn_cast<>, and
//               switch-based visitors.
//  Variant.cpp: Flang-style std::variant with std::visit and common::visit.
//  MLIR.cpp:    MLIR-style Operation/Op/Trait/Interface with Concept/Model.
//
// Benchmarks are named <Measure>/<Pattern>/<Style>/<inline|noinline>.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <algorithm>
#include <string>
#include <utility>

using namespace polybench;

static void registerBenchmarks() {
  Registry R;
  registerVirtual_inline(R);
  registerVirtual_noinline(R);
  registerClassof_inline(R);
  registerClassof_noinline(R);
  registerVariant_inline(R);
  registerVariant_noinline(R);
  registerMLIR_inline(R);
  registerMLIR_noinline(R);

  // Order of first appearance, so that inline/noinline pairs are adjacent.
  std::vector<std::string> Names;
  for (const BenchmarkEntry &E : R)
    if (std::find(Names.begin(), Names.end(), E.Name) == Names.end())
      Names.push_back(E.Name);

  const Pattern Patterns[] = {Pattern::Monomorphic, Pattern::RoundRobin,
                              Pattern::Random};
  const std::pair<Measure, const char *> Measures[] = {
      {Measure::Dispatch, "Dispatch"}, {Measure::TypeTest, "TypeTest"}};

  for (auto [M, MeasureName] : Measures)
    for (Pattern P : Patterns)
      for (const std::string &Name : Names)
        for (bool Inline : {true, false})
          for (const BenchmarkEntry &E : R)
            if (E.M == M && E.Name == Name && E.Inline == Inline)
              benchmark::RegisterBenchmark(
                  std::string(MeasureName) + "/" + getPatternName(P) + "/" +
                      Name + "/" + (Inline ? "inline" : "noinline"),
                  E.Fn, P);
}

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  registerBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
