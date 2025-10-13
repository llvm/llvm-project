//===--- StringSwitch.h - Switch-on-literal-string Construct --------------===/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===/
///
/// \file
///  This file implements the StringSwitch template, which mimics a switch()
///  statement whose cases are string literals.
///
//===----------------------------------------------------------------------===/
#ifndef LLVM_ADT_STRINGSWITCH_H
#define LLVM_ADT_STRINGSWITCH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <optional>

namespace llvm {

/// A switch()-like statement whose cases are string literals.
///
/// The StringSwitch class is a simple form of a switch() statement that
/// determines whether the given string matches one of the given string
/// literals. The template type parameter \p T is the type of the value that
/// will be returned from the string-switch expression. For example,
/// the following code switches on the name of a color in \c argv[i]:
///
/// \code
/// Color color = StringSwitch<Color>(argv[i])
///   .Case("red", Red)
///   .Case("orange", Orange)
///   .Case("yellow", Yellow)
///   .Case("green", Green)
///   .Case("blue", Blue)
///   .Case("indigo", Indigo)
///   .Cases("violet", "purple", Violet)
///   .Default(UnknownColor);
/// \endcode
template<typename T, typename R = T>
class StringSwitch {
  /// The string we are matching.
  const StringRef Str;

  /// The pointer to the result of this switch statement, once known,
  /// null before that.
  std::optional<T> Result;

public:
  explicit StringSwitch(StringRef S)
  : Str(S), Result() { }

  // StringSwitch is not copyable.
  StringSwitch(const StringSwitch &) = delete;

  // StringSwitch is not assignable due to 'Str' being 'const'.
  void operator=(const StringSwitch &) = delete;
  void operator=(StringSwitch &&other) = delete;

  StringSwitch(StringSwitch &&other)
    : Str(other.Str), Result(std::move(other.Result)) { }

  ~StringSwitch() = default;

  // Case-sensitive case matchers
  StringSwitch &Case(StringLiteral S, T Value) {
    CaseImpl(Value, S);
    return *this;
  }

  StringSwitch& EndsWith(StringLiteral S, T Value) {
    if (!Result && Str.ends_with(S)) {
      Result = std::move(Value);
    }
    return *this;
  }

  StringSwitch& StartsWith(StringLiteral S, T Value) {
    if (!Result && Str.starts_with(S)) {
      Result = std::move(Value);
    }
    return *this;
  }

  StringSwitch &Cases(std::initializer_list<StringLiteral> CaseStrings,
                      T Value) {
    return CasesImpl(Value, CaseStrings);
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, T Value) {
    return CasesImpl(Value, {S0, S1});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      T Value) {
    return CasesImpl(Value, {S0, S1, S2});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4, S5});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4, S5, S6});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4, S5, S6, S7});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, StringLiteral S8,
                      T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4, S5, S6, S7, S8});
  }

  StringSwitch &Cases(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                      StringLiteral S3, StringLiteral S4, StringLiteral S5,
                      StringLiteral S6, StringLiteral S7, StringLiteral S8,
                      StringLiteral S9, T Value) {
    return CasesImpl(Value, {S0, S1, S2, S3, S4, S5, S6, S7, S8, S9});
  }

  // Case-insensitive case matchers.
  StringSwitch &CaseLower(StringLiteral S, T Value) {
    CaseLowerImpl(Value, S);
    return *this;
  }

  StringSwitch &EndsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.ends_with_insensitive(S))
      Result = Value;

    return *this;
  }

  StringSwitch &StartsWithLower(StringLiteral S, T Value) {
    if (!Result && Str.starts_with_insensitive(S))
      Result = std::move(Value);

    return *this;
  }

  StringSwitch &CasesLower(std::initializer_list<StringLiteral> CaseStrings,
                           T Value) {
    return CasesLowerImpl(Value, CaseStrings);
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, T Value) {
    return CasesLowerImpl(Value, {S0, S1});
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           T Value) {
    return CasesLowerImpl(Value, {S0, S1, S2});
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           StringLiteral S3, T Value) {
    return CasesLowerImpl(Value, {S0, S1, S2, S3});
  }

  StringSwitch &CasesLower(StringLiteral S0, StringLiteral S1, StringLiteral S2,
                           StringLiteral S3, StringLiteral S4, T Value) {
    return CasesLowerImpl(Value, {S0, S1, S2, S3, S4});
  }

  [[nodiscard]] R Default(T Value) {
    if (Result)
      return std::move(*Result);
    return Value;
  }

  /// Declare default as unreachable, making sure that all cases were handled.
  [[nodiscard]] R DefaultUnreachable(
      const char *Message = "Fell off the end of a string-switch") {
    if (Result)
      return std::move(*Result);
    llvm_unreachable(Message);
  }

  [[nodiscard]] operator R() { return DefaultUnreachable(); }

private:
  // Returns true when `Str` matches the `S` argument, and stores the result.
  bool CaseImpl(T &Value, StringLiteral S) {
    if (!Result && Str == S) {
      Result = std::move(Value);
      return true;
    }
    return false;
  }

  // Returns true when `Str` matches the `S` argument (case-insensitive), and
  // stores the result.
  bool CaseLowerImpl(T &Value, StringLiteral S) {
    if (!Result && Str.equals_insensitive(S)) {
      Result = std::move(Value);
      return true;
    }
    return false;
  }

  StringSwitch &CasesImpl(T &Value,
                          std::initializer_list<StringLiteral> Cases) {
    // Stop matching after the string is found.
    for (StringLiteral S : Cases)
      if (CaseImpl(Value, S))
        break;
    return *this;
  }

  StringSwitch &CasesLowerImpl(T &Value,
                               std::initializer_list<StringLiteral> Cases) {
    // Stop matching after the string is found.
    for (StringLiteral S : Cases)
      if (CaseLowerImpl(Value, S))
        break;
    return *this;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGSWITCH_H
