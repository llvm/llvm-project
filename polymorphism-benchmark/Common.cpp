//===- Common.cpp - Shared infrastructure for polymorphism benchmarks -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <random>

namespace polybench {

const char *getPatternName(Pattern P) {
  switch (P) {
  case Pattern::Monomorphic:
    return "Monomorphic";
  case Pattern::RoundRobin:
    return "RoundRobin";
  case Pattern::Random:
    return "Random";
  }
  return "";
}

std::vector<NodeSpec> makeSpecs(Pattern P) {
  std::vector<NodeSpec> Specs;
  Specs.reserve(NumNodes);
  std::mt19937 Rng(42);
  std::uniform_int_distribution<unsigned> Dist(0, NumKinds - 1);
  for (std::size_t I = 0; I < NumNodes; ++I) {
    Kind K = Kind::Add;
    switch (P) {
    case Pattern::Monomorphic:
      K = Kind::Add;
      break;
    case Pattern::RoundRobin:
      K = static_cast<Kind>(I % NumKinds);
      break;
    case Pattern::Random:
      K = static_cast<Kind>(Dist(Rng));
      break;
    }
    Specs.push_back(
        {K, static_cast<int>(I % 7) + 1, static_cast<int>(I % 13) + 1});
  }
  return Specs;
}

} // namespace polybench
