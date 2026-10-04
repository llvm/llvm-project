//===- Classof.cpp - LLVM/Clang-style hand-rolled RTTI --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kind enum in the base class with contiguous ranges for abstract classes,
// static classof() methods, isa<>/cast<>/dyn_cast<>, and a switch-based
// visitor as generated from a .def file (InstVisitor, StmtVisitor) where
// cases without a specific implementation use the superclass' one.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <memory>

namespace polybench {
namespace POLY_NS(llvmstyle) {

template <typename To, typename From> bool isa(const From *Val) {
  return To::classof(Val);
}

template <typename To, typename From> const To *cast(const From *Val) {
  return static_cast<const To *>(Val);
}

template <typename To, typename From> const To *dyn_cast(const From *Val) {
  return isa<To>(Val) ? cast<To>(Val) : nullptr;
}

class Node {
public:
  enum NodeKind : std::uint8_t {
    NK_Const,
    NK_Neg,
    NK_FirstUnary = NK_Neg,
    NK_Not,
    NK_LastUnary = NK_Not,
    NK_Add,
    NK_FirstBinary = NK_Add,
    NK_Sub,
    NK_Mul,
    NK_Div,
    NK_LastBinary = NK_Div,
    NK_Call,
  };

  NodeKind getKind() const { return TheKind; }

protected:
  explicit Node(NodeKind K) : TheKind(K) {}

private:
  const NodeKind TheKind;
};

class Const : public Node {
public:
  explicit Const(int Value) : Node(NK_Const), Value(Value) {}
  int getValue() const { return Value; }
  static bool classof(const Node *N) { return N->getKind() == NK_Const; }
  POLY_METHOD std::int64_t getCost() const { return 1; }

private:
  int Value;
};

class UnaryOperation : public Node {
public:
  int getOperand() const { return Operand; }

  static bool classof(const Node *N) {
    return N->getKind() >= NK_FirstUnary && N->getKind() <= NK_LastUnary;
  }

protected:
  UnaryOperation(NodeKind K, int Operand) : Node(K), Operand(Operand) {}

private:
  int Operand;
};

class Neg : public UnaryOperation {
public:
  explicit Neg(int Operand) : UnaryOperation(NK_Neg, Operand) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Neg; }
  POLY_METHOD std::int64_t getCost() const { return getOperand() + 1; }
};

class Not : public UnaryOperation {
public:
  explicit Not(int Operand) : UnaryOperation(NK_Not, Operand) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Not; }
  POLY_METHOD std::int64_t getCost() const { return getOperand() + 2; }
};

class BinaryOperation : public Node {
public:
  int getLHS() const { return LHS; }
  int getRHS() const { return RHS; }
  static bool classof(const Node *N) {
    return N->getKind() >= NK_FirstBinary && N->getKind() <= NK_LastBinary;
  }
  POLY_METHOD std::int64_t getCost() const { return LHS + RHS + 1; }

protected:
  BinaryOperation(NodeKind K, int LHS, int RHS) : Node(K), LHS(LHS), RHS(RHS) {}

private:
  int LHS, RHS;
};

class Add : public BinaryOperation {
public:
  Add(int LHS, int RHS) : BinaryOperation(NK_Add, LHS, RHS) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Add; }
};

class Sub : public BinaryOperation {
public:
  Sub(int LHS, int RHS) : BinaryOperation(NK_Sub, LHS, RHS) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Sub; }
};

class Mul : public BinaryOperation {
public:
  Mul(int LHS, int RHS) : BinaryOperation(NK_Mul, LHS, RHS) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Mul; }
};

class Div : public BinaryOperation {
public:
  Div(int LHS, int RHS) : BinaryOperation(NK_Div, LHS, RHS) {}
  static bool classof(const Node *N) { return N->getKind() == NK_Div; }
  POLY_METHOD std::int64_t getCost() const { return getLHS() + getRHS() + 20; }
};

class Call : public Node {
public:
  Call(int NumArgs, int Callee)
      : Node(NK_Call), NumArgs(NumArgs), Callee(Callee) {}
  int getNumArgs() const { return NumArgs; }
  int getCallee() const { return Callee; }
  static bool classof(const Node *N) { return N->getKind() == NK_Call; }
  POLY_METHOD std::int64_t getCost() const { return NumArgs * 4 + 10; }

private:
  int NumArgs, Callee;
};

std::int64_t getCost(const Node *N) {
  switch (N->getKind()) {
  case Node::NK_Const:
    return cast<Const>(N)->getCost();
  case Node::NK_Neg:
    return cast<Neg>(N)->getCost();
  case Node::NK_Not:
    return cast<Not>(N)->getCost();
  case Node::NK_Add:
    return cast<Add>(N)->getCost();
  case Node::NK_Sub:
    return cast<Sub>(N)->getCost();
  case Node::NK_Mul:
    return cast<Mul>(N)->getCost();
  case Node::NK_Div:
    return cast<Div>(N)->getCost();
  case Node::NK_Call:
    return cast<Call>(N)->getCost();
  }
  __builtin_unreachable();
}

/// Without a virtual destructor, deletion must dispatch on the kind as well.
struct NodeDeleter {
  void operator()(Node *N) const {
    switch (N->getKind()) {
    case Node::NK_Const:
      delete static_cast<Const *>(N);
      return;
    case Node::NK_Neg:
      delete static_cast<Neg *>(N);
      return;
    case Node::NK_Not:
      delete static_cast<Not *>(N);
      return;
    case Node::NK_Add:
      delete static_cast<Add *>(N);
      return;
    case Node::NK_Sub:
      delete static_cast<Sub *>(N);
      return;
    case Node::NK_Mul:
      delete static_cast<Mul *>(N);
      return;
    case Node::NK_Div:
      delete static_cast<Div *>(N);
      return;
    case Node::NK_Call:
      delete static_cast<Call *>(N);
      return;
    }
  }
};

Node *create(const NodeSpec &S) {
  switch (S.K) {
  case Kind::Const:
    return new Const(S.A);
  case Kind::Neg:
    return new Neg(S.A);
  case Kind::Not:
    return new Not(S.A);
  case Kind::Add:
    return new Add(S.A, S.B);
  case Kind::Sub:
    return new Sub(S.A, S.B);
  case Kind::Mul:
    return new Mul(S.A, S.B);
  case Kind::Div:
    return new Div(S.A, S.B);
  case Kind::Call:
    return new Call(S.A, S.B);
  }
  return nullptr;
}

struct Nodes {
  std::vector<std::unique_ptr<Node, NodeDeleter>> Owner;
  std::vector<const Node *> Ptrs;

  explicit Nodes(Pattern P) {
    for (const NodeSpec &S : makeSpecs(P)) {
      Owner.emplace_back(create(S));
      Ptrs.push_back(Owner.back().get());
    }
  }
};

void BM_Dispatch(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs,
               [](const Node *Nd) -> std::int64_t { return getCost(Nd); });
}

void BM_TypeTest(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](const Node *Nd) -> std::int64_t {
    if (const auto *B = dyn_cast<BinaryOperation>(Nd))
      return B->getLHS() + B->getRHS();
    return 0;
  });
}

} // namespace POLY_NS(llvmstyle)

void POLY_REGISTER_FN(Classof)(Registry &R) {
  using namespace POLY_NS(llvmstyle);
  R.push_back({Measure::Dispatch, "Classof_Switch", POLY_INLINE, BM_Dispatch});
  R.push_back({Measure::TypeTest, "Classof_DynCast", POLY_INLINE, BM_TypeTest});
}

} // namespace polybench
