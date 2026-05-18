//===-- InstrumentorUtils.cpp - Highly configurable instrumentation pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/InstrumentorUtils.h"
#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/DiagnosticInfo.h"

using namespace llvm;
using namespace llvm::instrumentor;

namespace {
enum PropertyType { INT, STRING, POINTER, UNKNOWN };

/// Simple filter expression evaluator for instrumentation opportunities.
/// Supports integer comparisons (==, !=, <, >, <=, >=), string comparisons
/// (==, !=), pointer comparisons (==, !=) against null, string prefix checks
/// (startswith), and logical operators (&&, ||).
class FilterEvaluator {
  StringRef Expr;
  DenseMap<StringRef, int64_t> &IntPropertyValues;
  DenseMap<StringRef, StringRef> &StringPropertyValues;
  DenseMap<StringRef, Value *> &PointerPropertyValues;
  DenseMap<StringRef, PropertyType> &DynamicProperties;
  size_t Pos = 0;

public:
  FilterEvaluator(StringRef Expr,
                  DenseMap<StringRef, int64_t> &IntPropertyValues,
                  DenseMap<StringRef, StringRef> &StringPropertyValues,
                  DenseMap<StringRef, Value *> &PointerPropertyValues,
                  DenseMap<StringRef, PropertyType> &DynamicProperties)
      : Expr(Expr), IntPropertyValues(IntPropertyValues),
        StringPropertyValues(StringPropertyValues),
        PointerPropertyValues(PointerPropertyValues),
        DynamicProperties(DynamicProperties) {}

  Expected<bool> evaluate() {
    if (Expr.empty())
      return true;

    Expected<bool> Result = parseOrExpr();

    // Check if we consumed the entire expression.
    skipWhitespace();
    if (Pos < Expr.size() && Result)
      return createStringError(
          "unexpected characters at position " + std::to_string(Pos) + ": '" +
          Expr.substr(Pos, std::min<size_t>(10, Expr.size() - Pos)).str() +
          "'");

    return Result;
  }

private:
  void skipWhitespace() {
    while (Pos < Expr.size() && std::isspace(Expr[Pos]))
      ++Pos;
  }

  Expected<bool> parseOrExpr() {
    Expected<bool> Result = parseAndExpr();
    while (Result) {
      skipWhitespace();
      if (Pos + 1 < Expr.size() && Expr[Pos] == '|' && Expr[Pos + 1] == '|') {
        Pos += 2;
        Expected<bool> NextResult = parseAndExpr();
        if (!NextResult)
          return NextResult;
        *Result |= *NextResult;
      } else {
        break;
      }
    }
    return Result;
  }

  Expected<bool> parseAndExpr() {
    Expected<bool> Result = parsePrimary();
    while (Result) {
      skipWhitespace();
      if (Pos + 1 < Expr.size() && Expr[Pos] == '&' && Expr[Pos + 1] == '&') {
        Pos += 2;
        Expected<bool> NextResult = parsePrimary();
        if (!NextResult)
          return NextResult;
        *Result &= *NextResult;
      } else {
        break;
      }
    }
    return Result;
  }

  Expected<bool> parsePrimary() {
    skipWhitespace();

    // Check for opening parenthesis.
    if (Pos < Expr.size() && Expr[Pos] == '(') {
      ++Pos; // Skip '('
      Expected<bool> Result = parseOrExpr();

      skipWhitespace();
      if (Result && (Pos >= Expr.size() || Expr[Pos] != ')'))
        return createStringError("expected ')' at position " +
                                 std::to_string(Pos));

      // Skip ')'.
      ++Pos;
      return Result;
    }

    // Otherwise parse a comparison.
    return parseComparison();
  }

  // Parse a quoted string literal.
  Expected<StringRef> parseStringLiteral() {
    skipWhitespace();
    if (Pos >= Expr.size() || Expr[Pos] != '"')
      return createStringError("expected string literal at position " +
                               std::to_string(Pos));

    // Skip opening quote.
    ++Pos;
    size_t Start = Pos;
    while (Pos < Expr.size() && Expr[Pos] != '"')
      ++Pos;

    if (Pos >= Expr.size())
      return createStringError("unclosed string literal starting at position " +
                               std::to_string(Start - 1));

    StringRef Result = Expr.slice(Start, Pos);
    // Skip closing quote.
    ++Pos;
    return Result;
  }

  Expected<bool> parseComparison() {
    skipWhitespace();

    // Parse left-hand side (property name).
    size_t Start = Pos;
    while (Pos < Expr.size() && (std::isalnum(Expr[Pos]) || Expr[Pos] == '_'))
      ++Pos;

    StringRef PropName = Expr.slice(Start, Pos);
    if (PropName.empty())
      return createStringError("expected property name at position " +
                               std::to_string(Pos));

    skipWhitespace();

    // Check for .startswith() method call.
    if (Pos < Expr.size() && Expr[Pos] == '.') {
      ++Pos;
      skipWhitespace();

      // Parse method name.
      Start = Pos;
      while (Pos < Expr.size() && std::isalpha(Expr[Pos]))
        ++Pos;

      StringRef MethodName = Expr.slice(Start, Pos);
      skipWhitespace();

      if (MethodName == "startswith") {
        // Parse (.
        if (Pos >= Expr.size() || Expr[Pos] != '(')
          return createStringError(
              "expected '(' after 'startswith' at position " +
              std::to_string(Pos));

        ++Pos;

        // Parse string argument.
        auto Prefix = parseStringLiteral();
        if (!Prefix)
          return Prefix.takeError();

        skipWhitespace();

        // Parse )
        if (Pos >= Expr.size() || Expr[Pos] != ')')
          return createStringError(
              "expected ')' to close 'startswith' call at position " +
              std::to_string(Pos));

        ++Pos;

        // Evaluate startswith.
        auto StrIt = StringPropertyValues.find(PropName);
        if (StrIt != StringPropertyValues.end())
          return StrIt->second.starts_with(*Prefix);

        // If this is a dynamic string property, assume the filter passes.
        if (DynamicProperties.lookup_or(PropName, UNKNOWN) == STRING)
          return true;

        return createStringError(
            "startswith is only valid on string properties not '" + PropName +
            "'");
      }

      return createStringError("unknown method '" + MethodName.str() +
                               "' on property '" + PropName.str() + "'");
    }

    // Check if this is an integer property.
    auto IntIt = IntPropertyValues.find(PropName);
    if (IntIt != IntPropertyValues.end()) {
      int64_t LHS = IntIt->second;

      // Parse operator.
      enum OpKind { EQ, NE, LT, GT, LE, GE } Op;
      if (Pos < Expr.size()) {
        if (Expr[Pos] == '=' && Pos + 1 < Expr.size() && Expr[Pos + 1] == '=') {
          Op = EQ;
          Pos += 2;
        } else if (Expr[Pos] == '!' && Pos + 1 < Expr.size() &&
                   Expr[Pos + 1] == '=') {
          Op = NE;
          Pos += 2;
        } else if (Expr[Pos] == '<' && Pos + 1 < Expr.size() &&
                   Expr[Pos + 1] == '=') {
          Op = LE;
          Pos += 2;
        } else if (Expr[Pos] == '>' && Pos + 1 < Expr.size() &&
                   Expr[Pos + 1] == '=') {
          Op = GE;
          Pos += 2;
        } else if (Expr[Pos] == '<') {
          Op = LT;
          Pos += 1;
        } else if (Expr[Pos] == '>') {
          Op = GT;
          Pos += 1;
        } else {
          return createStringError("expected comparison operator (==, !=, <, "
                                   ">, <=, >=) at position " +
                                   std::to_string(Pos));
        }
      } else {
        return createStringError(
            "expected comparison operator after property '" + PropName.str() +
            "'");
      }

      skipWhitespace();

      // Parse right-hand side (constant value).
      Start = Pos;
      bool Negative = false;
      if (Pos < Expr.size() && Expr[Pos] == '-') {
        Negative = true;
        ++Pos;
      }

      size_t DigitStart = Pos;
      while (Pos < Expr.size() && std::isdigit(Expr[Pos]))
        ++Pos;

      if (Pos == DigitStart)
        return createStringError("expected integer value at position " +
                                 std::to_string(Pos));

      StringRef ValueStr = Expr.slice(Start, Pos);
      int64_t RHS = 0;
      if (ValueStr.getAsInteger(10, RHS))
        return createStringError("invalid integer value '" + ValueStr.str() +
                                 "'");

      if (Negative)
        RHS = -RHS;

      // Evaluate comparison.
      switch (Op) {
      case EQ:
        return LHS == RHS;
      case NE:
        return LHS != RHS;
      case LT:
        return LHS < RHS;
      case GT:
        return LHS > RHS;
      case LE:
        return LHS <= RHS;
      case GE:
        return LHS >= RHS;
      }
      return true;
    }

    // Check if this is a string property.
    auto StrIt = StringPropertyValues.find(PropName);
    if (StrIt != StringPropertyValues.end()) {
      StringRef LHS = StrIt->second;

      // Parse operator (only == and != for strings).
      enum OpKind { EQ, NE } Op;
      if (Pos < Expr.size()) {
        if (Expr[Pos] == '=' && Pos + 1 < Expr.size() && Expr[Pos + 1] == '=') {
          Op = EQ;
          Pos += 2;
        } else if (Expr[Pos] == '!' && Pos + 1 < Expr.size() &&
                   Expr[Pos + 1] == '=') {
          Op = NE;
          Pos += 2;
        } else {
          return createStringError("string property '" + PropName.str() +
                                   "' only supports == and != operators");
        }
      } else {
        return createStringError(
            "expected comparison operator after string property '" +
            PropName.str() + "'");
      }

      skipWhitespace();

      // Parse right-hand side (string literal).
      auto RHS = parseStringLiteral();
      if (!RHS)
        return RHS.takeError();

      // Evaluate comparison.
      switch (Op) {
      case EQ:
        return LHS == *RHS;
      case NE:
        return LHS != *RHS;
      }
      return true;
    }

    // Check if this is a pointer property.
    auto PtrIt = PointerPropertyValues.find(PropName);
    if (PtrIt != PointerPropertyValues.end()) {
      Value *LHS = PtrIt->second;

      // Parse operator (only == and != for pointers).
      enum OpKind { EQ, NE } Op;
      if (Pos < Expr.size()) {
        if (Expr[Pos] == '=' && Pos + 1 < Expr.size() && Expr[Pos + 1] == '=') {
          Op = EQ;
          Pos += 2;
        } else if (Expr[Pos] == '!' && Pos + 1 < Expr.size() &&
                   Expr[Pos + 1] == '=') {
          Op = NE;
          Pos += 2;
        } else {
          return createStringError("pointer property '" + PropName.str() +
                                   "' only supports == and != operators");
        }
      } else {
        return createStringError(
            "expected comparison operator after pointer property '" +
            PropName.str() + "'");
      }

      skipWhitespace();

      // Parse right-hand side (must be "null").
      Start = Pos;
      while (Pos < Expr.size() && std::isalpha(Expr[Pos]))
        ++Pos;

      StringRef RHS = Expr.slice(Start, Pos);
      if (RHS != "null")
        return createStringError("pointer comparisons only support 'null' as "
                                 "right-hand side, got '" +
                                 RHS.str() + "'");

      // Check if the pointer is a constant null.
      bool IsNull = false;
      if (auto *C = dyn_cast<Constant>(LHS)) {
        IsNull = C->isNullValue();
      } else {
        // Non-constant pointer - assume filter passes (conservative)
        return true;
      }

      // Evaluate comparison
      switch (Op) {
      case EQ:
        return IsNull;
      case NE:
        return !IsNull;
      }
      return true;
    }

    // Dynamic property value, assume filter passes.
    if (DynamicProperties.count(PropName))
      return true;

    // Unknown property, record an error.
    return createStringError("expected enabled property name, got '" +
                             PropName.str() + "'");
  }
};
} // anonymous namespace

bool llvm::instrumentor::evaluateFilter(Value &V,
                                        InstrumentationOpportunity &IO,
                                        InstrumentationConfig &IConf,
                                        InstrumentorIRBuilderTy &IIRB) {
  if (IO.Filter.empty())
    return true;

  // Collect constant property values for filter evaluation.
  DenseMap<StringRef, int64_t> IntPropertyValues;
  DenseMap<StringRef, StringRef> StringPropertyValues;
  DenseMap<StringRef, Value *> PointerPropertyValues;
  DenseMap<StringRef, PropertyType> DynamicProperties;

  for (auto &Arg : IO.IRTArgs) {
    if (!Arg.Enabled)
      continue;

    // Get the value for this argument.
    Value *ArgValue = Arg.GetterCB(V, *Arg.Ty, IConf, IIRB);
    if (!ArgValue)
      continue;

    if (auto *CI = dyn_cast<ConstantInt>(ArgValue)) {
      // Check for constant integer values.
      IntPropertyValues[Arg.Name] = CI->getSExtValue();
    } else if ((Arg.Flags & IRTArg::STRING) && isa<Constant>(ArgValue)) {
      // Check for constant string values (marked with STRING flag).
      if (auto *GV = dyn_cast<GlobalVariable>(ArgValue))
        if (GV->isConstant() && GV->hasInitializer())
          if (auto *CDA = dyn_cast<ConstantDataArray>(GV->getInitializer()))
            if (CDA->isCString())
              StringPropertyValues[Arg.Name] = CDA->getAsCString();
    } else if (ArgValue->getType()->isPointerTy()) {
      // Check for pointer values (for null comparisons), after the strings.
      PointerPropertyValues[Arg.Name] = ArgValue;
    } else {
      // If the value is not constant, we skip it - the filter will pass
      // for dynamic values - but we still want to report broken filters.
      DynamicProperties[Arg.Name] =
          Arg.Ty->isIntegerTy()
              ? INT
              : (Arg.Flags & IRTArg::STRING
                     ? STRING
                     : (Arg.Ty->isPointerTy() ? POINTER : UNKNOWN));
    }
  }

  FilterEvaluator Evaluator(IO.Filter, IntPropertyValues, StringPropertyValues,
                            PointerPropertyValues, DynamicProperties);

  Expected<bool> Result = Evaluator.evaluate();
  if (!Result) {
    // Emit an error if the filter is malformed.
    IIRB.Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("malformed filter expression for instrumentation opportunity '") +
            IO.getName() + Twine("': ") + toString(Result.takeError()) +
            Twine("\nFilter: ") + IO.Filter,
        DS_Error));
    return false;
  }

  return Result.get();
}
