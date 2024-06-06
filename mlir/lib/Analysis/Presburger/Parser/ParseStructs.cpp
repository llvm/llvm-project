//===- ParseStructs.cpp - Presburger Parse Structrures ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParseStructs class that the parser for the
// Presburger library parses into.
//
//===----------------------------------------------------------------------===//

#include "ParseStructs.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::presburger;
using llvm::dbgs;
using llvm::divideFloorSigned;
using llvm::mod;

CoefficientVector PureAffineExprImpl::collectLinearTerms() const {
  CoefficientVector nestedLinear = std::accumulate(
      nestedDivTerms.begin(), nestedDivTerms.end(), CoefficientVector(info),
      [](const CoefficientVector &acc, const PureAffineExpr &div) {
        return acc + div->getLinearDividend();
      });
  return nestedLinear += linearDividend;
}

SmallVector<std::tuple<size_t, int64_t, CoefficientVector>, 8>
PureAffineExprImpl::getNonLinearCoeffs() const {
  SmallVector<std::tuple<size_t, int64_t, CoefficientVector>, 8> ret;
  // dividend `floordiv` divisor <=> q; adjustedMulFactor = 1,
  // adjustedLinearTerm is empty.
  //
  // dividend `mod` divisor <=> dividend - divisor*q; adjustedMulFactor =
  // -divisor, adjustedLinearTerm is the linear part of the dividend.
  //
  // where q is a floordiv id added by the flattener.
  auto adjustedMulFactor = [](const PureAffineExprImpl &div) {
    return div.mulFactor * (div.kind == DivKind::Mod ? -div.divisor : 1);
  };
  auto adjustedLinearTerm = [](PureAffineExprImpl &div) {
    return div.kind == DivKind::Mod ? div.linearDividend *= div.mulFactor
                                    : CoefficientVector(div.info);
  };
  if (hasDivisor())
    for (const auto &toplevel : nestedDivTerms)
      for (const auto &div : toplevel->getNestedDivTerms())
        ret.emplace_back(div->hash(), adjustedMulFactor(*div),
                         adjustedLinearTerm(*div));
  else
    for (const auto &div : nestedDivTerms)
      ret.emplace_back(div->hash(), adjustedMulFactor(*div),
                       adjustedLinearTerm(*div));
  return ret;
}

unsigned PureAffineExprImpl::countNestedDivs() const {
  return hasDivisor() +
         std::accumulate(getNestedDivTerms().begin(), getNestedDivTerms().end(),
                         0, [](unsigned acc, const PureAffineExpr &div) {
                           return acc + div->countNestedDivs();
                         });
}

PureAffineExpr mlir::presburger::operator+(PureAffineExpr &&lhs,
                                           PureAffineExpr &&rhs) {
  if (lhs->isLinear() && rhs->isLinear())
    return std::make_unique<PureAffineExprImpl>(lhs->getLinearDividend() +
                                                rhs->getLinearDividend());
  if (lhs->isLinear())
    return std::make_unique<PureAffineExprImpl>(
        std::move(rhs->addLinearTerm(lhs->getLinearDividend())));
  if (rhs->isLinear())
    return std::make_unique<PureAffineExprImpl>(
        std::move(lhs->addLinearTerm(rhs->getLinearDividend())));

  if (!(lhs->hasDivisor() ^ rhs->hasDivisor())) {
    auto ret = PureAffineExprImpl(lhs->info);
    ret.addDivTerm(std::move(*lhs));
    ret.addDivTerm(std::move(*rhs));
    return std::make_unique<PureAffineExprImpl>(std::move(ret));
  }
  if (lhs->hasDivisor()) {
    rhs->addDivTerm(std::move(*lhs));
    return rhs;
  }
  if (rhs->hasDivisor()) {
    lhs->addDivTerm(std::move(*rhs));
    return lhs;
  }
  llvm_unreachable("Malformed AffineExpr");
}

PureAffineExpr mlir::presburger::operator*(PureAffineExpr &&expr, int64_t c) {
  if (expr->isLinear())
    return std::make_unique<PureAffineExprImpl>(expr->getLinearDividend() * c);
  return std::make_unique<PureAffineExprImpl>(std::move(expr->mulConstant(c)));
}

PureAffineExpr mlir::presburger::operator+(PureAffineExpr &&expr, int64_t c) {
  return std::move(expr) + std::make_unique<PureAffineExprImpl>(expr->info, c);
}

PureAffineExpr mlir::presburger::div(PureAffineExpr &&dividend, int64_t divisor,
                                     DivKind kind) {
  assert(divisor > 0 && "floorDiv or mod with a negative divisor");

  // Constant fold.
  if (dividend->isConstant()) {
    int64_t c = kind == DivKind::FloorDiv
                    ? divideFloorSigned(dividend->getConstant(), divisor)
                    : mod(dividend->getConstant(), divisor);
    return std::make_unique<PureAffineExprImpl>(dividend->info, c);
  }

  // Factor out mul, using gcd internally.
  uint64_t exprMultiple;
  if (dividend->isLinear()) {
    exprMultiple = dividend->getLinearDividend().factorMulFromLinearTerm();
  } else {
    // Canonicalize the div.
    uint64_t constMultiple =
        dividend->getLinearDividend().factorMulFromLinearTerm();
    dividend->divLinearDividend(static_cast<int64_t>(constMultiple));
    dividend->mulFactor *= constMultiple;
    exprMultiple = dividend->mulFactor;
  }

  // Perform gcd with divisor.
  uint64_t gcd = std::gcd(std::abs(divisor), exprMultiple);

  // Divide according to the type.
  if (gcd > 1) {
    if (dividend->isLinear())
      dividend->linearDividend /= static_cast<int64_t>(gcd);
    else
      dividend->mulFactor /= (static_cast<int64_t>(gcd));

    divisor /= static_cast<int64_t>(gcd);
  }

  if (dividend->isLinear())
    return std::make_unique<PureAffineExprImpl>(dividend->getLinearDividend(),
                                                divisor, kind);

  // x floordiv 1 <=> x, x % 1 <=> 0
  return divisor == 1
             ? (dividend->isMod()
                    ? std::make_unique<PureAffineExprImpl>(dividend->info)
                    : std::move(dividend))
             : std::make_unique<PureAffineExprImpl>(std::move(*dividend),
                                                    divisor, kind);
  ;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
enum class BindingStrength {
  Weak,   // + and -
  Strong, // All other binary operators.
};

static void printCoefficient(int64_t c, bool &isExprBegin,
                             const ParseInfo &info, int idx = -1) {
  bool isConstant = idx != -1 && info.isConstantIdx(idx);
  bool isDimOrSymbol = idx != -1 && !info.isConstantIdx(idx);
  if (!c)
    return;
  if (!isExprBegin)
    dbgs() << " ";
  if (c < 0) {
    dbgs() << "- ";
    if (c != -1 || isConstant)
      dbgs() << std::abs(c);
  } else {
    if (!isExprBegin)
      dbgs() << "+ ";
    if (c != 1 || isConstant) {
      dbgs() << c;
      isExprBegin = false;
    }
  }
  if (isDimOrSymbol) {
    if (std::abs(c) != 1)
      dbgs() << " * ";
    dbgs() << (info.isDimIdx(idx) ? 'd' : 's') << idx;
    isExprBegin = false;
  }
}

static bool printCoefficientVec(
    const CoefficientVector &linear, bool isExprBegin = true,
    BindingStrength enclosingTightness = BindingStrength::Weak) {
  if (enclosingTightness == BindingStrength::Strong &&
      linear.hasMultipleCoefficients()) {
    dbgs() << '(';
    isExprBegin = true;
  }
  for (auto [idx, c] : enumerate(linear.getCoefficients()))
    printCoefficient(c, isExprBegin, linear.info, idx);

  if (enclosingTightness == BindingStrength::Strong &&
      linear.hasMultipleCoefficients())
    dbgs() << ')';
  return isExprBegin;
}

static bool
printAffineExpr(const PureAffineExprImpl &expr, bool isExprBegin = true,
                BindingStrength enclosingTightness = BindingStrength::Weak) {
  if (expr.isLinear())
    return printCoefficientVec(expr.getLinearDividend(), isExprBegin,
                               enclosingTightness);

  const auto &div = expr;
  const auto &linearDividend = div.getLinearDividend();
  const auto &divisor = div.getDivisor();
  const auto &mulFactor = div.getMulFactor();
  const auto &nestedDivs = div.getNestedDivTerms();

  printCoefficient(mulFactor, isExprBegin, expr.info);
  if (std::abs(mulFactor) != 1)
    dbgs() << " * ";

  if (div.hasDivisor() || enclosingTightness == BindingStrength::Strong) {
    dbgs() << '(';
    isExprBegin = true;
  }

  isExprBegin = printCoefficientVec(linearDividend, isExprBegin);

  for (const auto &div : nestedDivs)
    isExprBegin = printAffineExpr(*div, isExprBegin, BindingStrength::Strong);

  if (div.hasDivisor())
    dbgs() << (expr.isMod() ? " % " : " floordiv ") << divisor;

  if (div.hasDivisor() || enclosingTightness == BindingStrength::Strong)
    dbgs() << ')';

  return isExprBegin;
}

LLVM_DUMP_METHOD void CoefficientVector::dump() const {
  printCoefficientVec(*this);
  dbgs() << '\n';
}

LLVM_DUMP_METHOD void PureAffineExprImpl::dump() const {
  printAffineExpr(*this);
  dbgs() << '\n';
}

LLVM_DUMP_METHOD void FinalParseResult::dump() const {
  dbgs() << "Exprs:\n";
  for (const auto &expr : exprs)
    expr->dump();
  dbgs() << "EqFlags: ";
  for (bool eqF : eqFlags)
    dbgs() << eqF << ' ';
  dbgs() << '\n';
}
#endif
