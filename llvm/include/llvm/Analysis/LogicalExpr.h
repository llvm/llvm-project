//===------------------- LogicalExpr.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines LogicalExpr, a class that represent a logical value by
/// a set of bitsets.
///
/// For a logical expression represented by bitset, the "and" logic
/// operator represented by "&" is translated to "*" and is then evaluated as
/// the "or" of the bitset. For example, pattern "a & b" is represented by the
/// logical expression "01 * 10", and the expression is reduced to "11". So the
/// operation "&" between two logical expressions (not "xor", only "and" chain)
/// is actually bitwise "or" of the masks. There are two exceptions:
///    If one of the operands is constant 0, the entire bitset represents 0.
///    If one of the operands is constant -1, the result is the other one.
///
/// The evaluation of a pattern for bitwise "xor" is represented by a "+" math
/// operator. But it also has one exception to normal math rules: if two masks
/// are identical, we remove them. For example with "a ^ a", the logical
/// expression is "1 + 1". We eliminate them from the logical expression.
///
/// We use commutative, associative, and distributive laws of arithmetic
/// multiplication and addition to reduce the expression. An example for the
/// LogicalExpr caculation:
///     ((a & b) | (a ^ c)) ^ (!(b & c) & a)
/// Mask for the leafs are: a --> 001, b --> 010, c -->100
/// First step is expand the pattern to:
///      (((a & b) & (a ^ c)) ^ (a & b) ^ (a ^ c)) ^ (((b & c) ^ -1) & a)
/// Use logical expression to represent the pattern:
///      001 * 010 * (001 + 100) + 001 * 010 + 001 + 100 + (010 * 100 + -1C) *
///      001
/// Expression after distributive laws:
///      001 * 010 * 001 + 001 * 010 * 100 + 001 * 010 + 001 + 100 + 010 * 100 *
///      001 + -1C * 001
/// Calculate multiplication:
///      011 + 111 + 011 + 001 + 100 + 111 + 001
/// Calculate addition:
///      100
/// Restore to value
///      c
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOGICALEXPR_H
#define LLVM_ANALYSIS_LOGICALEXPR_H

#include "llvm/ADT/DenseSet.h"

namespace llvm {
// TODO: can we use APInt define the mask to enlarge the max leaf number?
typedef SmallDenseSet<uint64_t, 8> ExprAddChain;

class LogicalExpr {
private:
  ExprAddChain AddChain;

public:
  static const uint64_t ExprAllOne = 0x8000000000000000;

  LogicalExpr() {}
  LogicalExpr(uint64_t BitSet) {
    if (BitSet != 0)
      AddChain.insert(BitSet);
  }
  LogicalExpr(const ExprAddChain &SrcAddChain) : AddChain(SrcAddChain) {
  }

  unsigned size() const { return AddChain.size(); }
  ExprAddChain::iterator begin() { return AddChain.begin(); }
  ExprAddChain::iterator end() { return AddChain.end(); }
  ExprAddChain::const_iterator begin() const { return AddChain.begin(); }
  ExprAddChain::const_iterator end() const { return AddChain.end(); }

  LogicalExpr &operator*=(const LogicalExpr &RHS) {
    ExprAddChain NewChain;
    for (auto LHS : AddChain) {
      for (auto RHS : RHS.AddChain) {
        uint64_t NewBitSet;
        // Except the special case one value "*" -1 is just return itself, the
        // other "*" operation is actually "|" LHS and RHS 's bitset. For
        // example: ab * bd  = abd The expression ab * bd convert to bitset will
        // be 0b0011 * 0b1010. The result abd convert to bitset will become
        // 0b1011.
        if (LHS == ExprAllOne)
          NewBitSet = RHS;
        else if (RHS == ExprAllOne)
          NewBitSet = LHS;
        else
          NewBitSet = LHS | RHS;
        assert(NewBitSet == ExprAllOne || (NewBitSet & ExprAllOne) == 0);
        // a ^ a -> 0
        auto InsertPair = NewChain.insert(NewBitSet);
        if (!InsertPair.second)
          NewChain.erase(InsertPair.first);
      }
    }

    AddChain = NewChain;
    return *this;
  }

  LogicalExpr &operator+=(const LogicalExpr &RHS) {
    for (auto RHS : RHS.AddChain) {
      // a ^ a -> 0
      auto InsertPair = AddChain.insert(RHS);
      if (!InsertPair.second)
        AddChain.erase(InsertPair.first);
    }
    return *this;
  }
};

inline LogicalExpr operator*(LogicalExpr a, const LogicalExpr &b) {
  a *= b;
  return a;
}

inline LogicalExpr operator+(LogicalExpr a, const LogicalExpr &b) {
  a += b;
  return a;
}

inline LogicalExpr operator&(const LogicalExpr &a, const LogicalExpr &b) {
  return a * b;
}

inline LogicalExpr operator^(const LogicalExpr &a, const LogicalExpr &b) {
  return a + b;
}

inline LogicalExpr operator|(const LogicalExpr &a, const LogicalExpr &b) {
  return a * b + a + b;
}

inline LogicalExpr operator~(const LogicalExpr &a) {
  LogicalExpr AllOneExpr(LogicalExpr::ExprAllOne);
  return a + AllOneExpr;
}

} // namespace llvm

#endif // LLVM_ANALYSIS_LOGICALEXPR_H
