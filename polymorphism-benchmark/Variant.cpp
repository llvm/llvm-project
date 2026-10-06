//===- Variant.cpp - Flang-style std::variant polymorphism ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A flat std::variant of all concrete node types, stored by value. Abstract
// classes are ordinary base structs of the alternatives that share data and
// member functions (like parser::Expr::IntrinsicBinary or
// evaluate::Operation<>). Dispatch and type tests use std::visit and Flang's
// common::visit; membership in an abstract class is decided at compile time
// in the visitor.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <type_traits>
#include <utility>
#include <variant>

namespace polybench {
namespace POLY_NS(flangstyle) {

/// Copy of Fortran::common::log2visit from flang/Common/visit.h, restricted
/// to a single variant argument.
namespace log2visit {

template <std::size_t LOW, std::size_t HIGH, typename RESULT, typename VISITOR,
          typename VARIANT>
inline RESULT Log2VisitHelper(VISITOR &&visitor, std::size_t which,
                              VARIANT &&u) {
  if constexpr (LOW + 7 >= HIGH) {
    switch (which - LOW) {
#define VISIT_CASE_N(N)                                                        \
  case N:                                                                      \
    if constexpr (LOW + N <= HIGH) {                                           \
      return visitor(std::get<(LOW + N)>(std::forward<VARIANT>(u)));           \
    }
      VISIT_CASE_N(1)
      [[fallthrough]];
      VISIT_CASE_N(2)
      [[fallthrough]];
      VISIT_CASE_N(3)
      [[fallthrough]];
      VISIT_CASE_N(4)
      [[fallthrough]];
      VISIT_CASE_N(5)
      [[fallthrough]];
      VISIT_CASE_N(6)
      [[fallthrough]];
      VISIT_CASE_N(7)
#undef VISIT_CASE_N
    }
    return visitor(std::get<LOW>(std::forward<VARIANT>(u)));
  } else {
    static constexpr std::size_t mid{(HIGH + LOW) / 2};
    if (which <= mid) {
      return Log2VisitHelper<LOW, mid, RESULT>(std::forward<VISITOR>(visitor),
                                               which, std::forward<VARIANT>(u));
    } else {
      return Log2VisitHelper<(mid + 1), HIGH, RESULT>(
          std::forward<VISITOR>(visitor), which, std::forward<VARIANT>(u));
    }
  }
}

template <typename VISITOR, typename VARIANT>
inline auto visit(VISITOR &&visitor, VARIANT &&u)
    -> decltype(visitor(std::get<0>(std::forward<VARIANT>(u)))) {
  using Result = decltype(visitor(std::get<0>(std::forward<VARIANT>(u))));
  static constexpr std::size_t high{std::variant_size_v<std::decay_t<VARIANT>> -
                                    1};
  return Log2VisitHelper<0, high, Result>(std::forward<VISITOR>(visitor),
                                          u.index(), std::forward<VARIANT>(u));
}

} // namespace log2visit

namespace common {
using log2visit::visit;
} // namespace common

struct Const {
  int Value;
  POLY_METHOD std::int64_t getCost() const { return 1; }
};

struct UnaryOperation {
  int Operand;
};

struct Neg : UnaryOperation {
  POLY_METHOD std::int64_t getCost() const { return Operand + 1; }
};

struct Not : UnaryOperation {
  POLY_METHOD std::int64_t getCost() const { return Operand + 2; }
};

struct BinaryOperation {
  int LHS, RHS;
  POLY_METHOD std::int64_t getCost() const { return LHS + RHS + 1; }
};

struct Add : BinaryOperation {};
struct Sub : BinaryOperation {};
struct Mul : BinaryOperation {};

struct Div : BinaryOperation {
  POLY_METHOD std::int64_t getCost() const { return LHS + RHS + 20; }
};

struct Call {
  int NumArgs, Callee;
  POLY_METHOD std::int64_t getCost() const { return NumArgs * 4 + 10; }
};

using Node = std::variant<Const, Neg, Not, Add, Sub, Mul, Div, Call>;

Node create(const NodeSpec &S) {
  switch (S.K) {
  case Kind::Const:
    return Const{S.A};
  case Kind::Neg:
    return Neg{{S.A}};
  case Kind::Not:
    return Not{{S.A}};
  case Kind::Add:
    return Add{{S.A, S.B}};
  case Kind::Sub:
    return Sub{{S.A, S.B}};
  case Kind::Mul:
    return Mul{{S.A, S.B}};
  case Kind::Div:
    return Div{{S.A, S.B}};
  case Kind::Call:
    return Call{S.A, S.B};
  }
  return Const{0};
}

std::vector<Node> createNodes(Pattern P) {
  std::vector<Node> Result;
  for (const NodeSpec &S : makeSpecs(P))
    Result.push_back(create(S));
  return Result;
}

struct CostVisitor {
  template <typename T> std::int64_t operator()(const T &X) const {
    return X.getCost();
  }
};

/// The per-alternative handler of the type test.
struct BinaryOperandsVisitor {
  template <typename T> std::int64_t operator()(const T &X) const {
    if constexpr (std::is_base_of_v<BinaryOperation, T>)
      return X.LHS + X.RHS;
    else
      return 0;
  }
};

void BM_Dispatch_StdVisit(benchmark::State &State, Pattern P) {
  std::vector<Node> Nodes = createNodes(P);
  runBenchmark(State, P, Nodes, [](const Node &Nd) -> std::int64_t {
    return std::visit(CostVisitor{}, Nd);
  });
}

void BM_Dispatch_CommonVisit(benchmark::State &State, Pattern P) {
  std::vector<Node> Nodes = createNodes(P);
  runBenchmark(State, P, Nodes, [](const Node &Nd) -> std::int64_t {
    return common::visit(CostVisitor{}, Nd);
  });
}

void BM_TypeTest_StdVisit(benchmark::State &State, Pattern P) {
  std::vector<Node> Nodes = createNodes(P);
  runBenchmark(State, P, Nodes, [](const Node &Nd) -> std::int64_t {
    return std::visit(BinaryOperandsVisitor{}, Nd);
  });
}

void BM_TypeTest_CommonVisit(benchmark::State &State, Pattern P) {
  std::vector<Node> Nodes = createNodes(P);
  runBenchmark(State, P, Nodes, [](const Node &Nd) -> std::int64_t {
    return common::visit(BinaryOperandsVisitor{}, Nd);
  });
}

} // namespace POLY_NS(flangstyle)

void POLY_REGISTER_FN(Variant)(Registry &R) {
  using namespace POLY_NS(flangstyle);
  R.push_back({Measure::Dispatch, "Variant_StdVisit", POLY_INLINE,
               BM_Dispatch_StdVisit});
  R.push_back({Measure::Dispatch, "Variant_CommonVisit", POLY_INLINE,
               BM_Dispatch_CommonVisit});
  R.push_back({Measure::TypeTest, "Variant_StdVisit", POLY_INLINE,
               BM_TypeTest_StdVisit});
  R.push_back({Measure::TypeTest, "Variant_CommonVisit", POLY_INLINE,
               BM_TypeTest_CommonVisit});
}

} // namespace polybench
