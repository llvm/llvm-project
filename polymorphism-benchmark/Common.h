//===- Common.h - Shared infrastructure for polymorphism benchmarks -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every style models the same AST-like hierarchy:
//
//   Node
//   +- Const
//   +- UnaryOperation
//   |  +- Neg
//   |  +- Not
//   +- BinaryOperation
//   |  +- Add
//   |  +- Sub
//   |  +- Mul
//   |  +- Div
//   +- Call
//
// The dispatched operation is getCost(), a toy cost model:
//
//   Const:           1
//   Neg:             Operand + 1
//   Not:             Operand + 2
//   BinaryOperation: LHS + RHS + 1   (shared by Add, Sub, Mul)
//   Div:             LHS + RHS + 20  (overrides BinaryOperation)
//   Call:            NumArgs * 4 + 10
//
// The type test checks whether a node is a BinaryOperation and, if so, reads
// its operands.
//
// Each style's source file is compiled twice: with POLY_INLINE=1 the
// node-specific methods (getCost implementations, classof, visitors used for
// type tests) can be inlined; with POLY_INLINE=0 they are marked as if
// defined in another translation unit.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMORPHISM_BENCHMARK_COMMON_H
#define POLYMORPHISM_BENCHMARK_COMMON_H

#include "benchmark/benchmark.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace polybench {

constexpr std::size_t NumNodes = 4096;

enum class Kind : std::uint8_t { Const, Neg, Not, Add, Sub, Mul, Div, Call };
constexpr unsigned NumKinds = 8;

enum class Pattern { Monomorphic, RoundRobin, Random };

const char *getPatternName(Pattern P);

/// Style-independent description of a node. A is the operand/LHS/value/number
/// of arguments, B is the RHS/callee.
struct NodeSpec {
  Kind K;
  int A;
  int B;
};

/// Monomorphic: every node is an Add.
/// RoundRobin:  Const, Neg, Not, Add, Sub, Mul, Div, Call, Const, ...
/// Random:      uniformly distributed kinds, identical for all styles.
std::vector<NodeSpec> makeSpecs(Pattern P);

/// Runs \p Body over all nodes once per benchmark iteration.
template <typename Container, typename BodyT>
void runBenchmark(benchmark::State &State, Pattern P, const Container &Nodes,
                  BodyT Body) {
  std::int64_t Checksum = 0;
  for (auto _ : State) {
    std::int64_t Sum = 0;
    for (const auto &N : Nodes)
      Sum += Body(N);
    benchmark::DoNotOptimize(Sum);
    benchmark::ClobberMemory();
    Checksum = Sum;
  }
  State.SetItemsProcessed(static_cast<std::int64_t>(State.iterations()) *
                          static_cast<std::int64_t>(Nodes.size()));
  // Must be identical for all styles of the same measure and pattern.
  State.counters["checksum"] = static_cast<double>(Checksum);
  State.SetLabel(getPatternName(P));
}

using BenchmarkFn = void (*)(benchmark::State &, Pattern);

enum class Measure { Dispatch, TypeTest };

struct BenchmarkEntry {
  Measure M;
  const char *Name;
  bool Inline;
  BenchmarkFn Fn;
};

using Registry = std::vector<BenchmarkEntry>;

void registerVirtual_inline(Registry &R);
void registerVirtual_noinline(Registry &R);
void registerClassof_inline(Registry &R);
void registerClassof_noinline(Registry &R);
void registerVariant_inline(Registry &R);
void registerVariant_noinline(Registry &R);
void registerMLIR_inline(Registry &R);
void registerMLIR_noinline(Registry &R);

} // namespace polybench

#define POLY_CONCAT_IMPL(A, B) A##B
#define POLY_CONCAT(A, B) POLY_CONCAT_IMPL(A, B)

#ifdef POLY_INLINE
#if POLY_INLINE
#define POLY_METHOD
#define POLY_SUFFIX _inline
#else
// noipa (GCC) also prevents cloning and interprocedural propagation, which a
// definition in another translation unit would prevent as well.
#if defined(__clang__)
#define POLY_METHOD __attribute__((noinline))
#else
#define POLY_METHOD __attribute__((noipa))
#endif
#define POLY_SUFFIX _noinline
#endif

/// Namespace for a style; distinct per POLY_INLINE to avoid ODR violations.
#define POLY_NS(Style) POLY_CONCAT(Style, POLY_SUFFIX)
/// Name of the registration function of a style.
#define POLY_REGISTER_FN(Style) POLY_CONCAT(register##Style, POLY_SUFFIX)
#endif

#endif // POLYMORPHISM_BENCHMARK_COMMON_H
