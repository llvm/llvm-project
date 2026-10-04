//===- Virtual.cpp - C++ virtual methods ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dispatch through virtual methods; type tests with dynamic_cast.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <memory>

namespace polybench {
namespace POLY_NS(virt) {

struct Node {
  virtual ~Node() = default;
  virtual std::int64_t getCost() const = 0;
};

struct Const final : Node {
  int Value;
  explicit Const(int Value) : Value(Value) {}
  POLY_METHOD std::int64_t getCost() const override { return 1; }
};

struct UnaryOperation : Node {
  int Operand;
  explicit UnaryOperation(int Operand) : Operand(Operand) {}
  int getOperand() const { return Operand; }
};

struct Neg final : UnaryOperation {
  using UnaryOperation::UnaryOperation;
  POLY_METHOD std::int64_t getCost() const override { return Operand + 1; }
};

struct Not final : UnaryOperation {
  using UnaryOperation::UnaryOperation;
  POLY_METHOD std::int64_t getCost() const override { return Operand + 2; }
};

struct BinaryOperation : Node {
  int LHS, RHS;
  BinaryOperation(int LHS, int RHS) : LHS(LHS), RHS(RHS) {}
  int getLHS() const { return LHS; }
  int getRHS() const { return RHS; }
  POLY_METHOD std::int64_t getCost() const override { return LHS + RHS + 1; }
};

struct Add final : BinaryOperation {
  using BinaryOperation::BinaryOperation;
};

struct Sub final : BinaryOperation {
  using BinaryOperation::BinaryOperation;
};

struct Mul final : BinaryOperation {
  using BinaryOperation::BinaryOperation;
};

struct Div final : BinaryOperation {
  using BinaryOperation::BinaryOperation;
  POLY_METHOD std::int64_t getCost() const override { return LHS + RHS + 20; }
};

struct Call final : Node {
  int NumArgs, Callee;
  Call(int NumArgs, int Callee) : NumArgs(NumArgs), Callee(Callee) {}
  POLY_METHOD std::int64_t getCost() const override { return NumArgs * 4 + 10; }
};

std::unique_ptr<Node> create(const NodeSpec &S) {
  switch (S.K) {
  case Kind::Const:
    return std::make_unique<Const>(S.A);
  case Kind::Neg:
    return std::make_unique<Neg>(S.A);
  case Kind::Not:
    return std::make_unique<Not>(S.A);
  case Kind::Add:
    return std::make_unique<Add>(S.A, S.B);
  case Kind::Sub:
    return std::make_unique<Sub>(S.A, S.B);
  case Kind::Mul:
    return std::make_unique<Mul>(S.A, S.B);
  case Kind::Div:
    return std::make_unique<Div>(S.A, S.B);
  case Kind::Call:
    return std::make_unique<Call>(S.A, S.B);
  }
  return nullptr;
}

struct Nodes {
  std::vector<std::unique_ptr<Node>> Owner;
  std::vector<const Node *> Ptrs;

  explicit Nodes(Pattern P) {
    for (const NodeSpec &S : makeSpecs(P)) {
      Owner.push_back(create(S));
      Ptrs.push_back(Owner.back().get());
    }
  }
};

void BM_Dispatch(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs,
               [](const Node *Nd) -> std::int64_t { return Nd->getCost(); });
}

void BM_TypeTest(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](const Node *Nd) -> std::int64_t {
    if (const auto *B = dynamic_cast<const BinaryOperation *>(Nd))
      return B->getLHS() + B->getRHS();
    return 0;
  });
}

} // namespace POLY_NS(virt)

void POLY_REGISTER_FN(Virtual)(Registry &R) {
  using namespace POLY_NS(virt);
  R.push_back({Measure::Dispatch, "Virtual", POLY_INLINE, BM_Dispatch});
  R.push_back(
      {Measure::TypeTest, "Virtual_DynamicCast", POLY_INLINE, BM_TypeTest});
}

} // namespace polybench
