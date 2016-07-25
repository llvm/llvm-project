//===--- StringSwitch.h - Switch-on-literal-string Construct --------------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements the StringSwitch template, which mimics a switch()
//  statement whose cases are string literals.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_ADT_STRINGSWITCH_H
#define LLVM_ADT_STRINGSWITCH_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstring>

namespace llvm {

/// \brief A switch()-like statement whose cases are string literals.
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
  /// \brief The string we are matching.
  StringRef Str;

  /// \brief The pointer to the result of this switch statement, once known,
  /// null before that.
  const T *Result;

public:
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  explicit StringSwitch(StringRef S)
  : Str(S), Result(nullptr) { }

  // StringSwitch is not copyable.
  StringSwitch(const StringSwitch &) = delete;
  void operator=(const StringSwitch &) = delete;

  StringSwitch(StringSwitch &&other) {
    *this = std::move(other);
  }
  StringSwitch &operator=(StringSwitch &&other) {
    Str = other.Str;
    Result = other.Result;
    return *this;
  }

  ~StringSwitch() = default;

  template<unsigned N>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& Case(const char (&S)[N], const T& Value) {
    if (!Result && N-1 == Str.size() &&
        (std::memcmp(S, Str.data(), N-1) == 0)) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& EndsWith(const char (&S)[N], const T &Value) {
    if (!Result && Str.size() >= N-1 &&
        std::memcmp(S, Str.data() + Str.size() + 1 - N, N-1) == 0) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& StartsWith(const char (&S)[N], const T &Value) {
    if (!Result && Str.size() >= N-1 &&
        std::memcmp(S, Str.data(), N-1) == 0) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N0, unsigned N1>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const T& Value) {
    if (!Result && (
        (N0-1 == Str.size() && std::memcmp(S0, Str.data(), N0-1) == 0) ||
        (N1-1 == Str.size() && std::memcmp(S1, Str.data(), N1-1) == 0))) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N0, unsigned N1, unsigned N2>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const T& Value) {
    if (!Result && (
        (N0-1 == Str.size() && std::memcmp(S0, Str.data(), N0-1) == 0) ||
        (N1-1 == Str.size() && std::memcmp(S1, Str.data(), N1-1) == 0) ||
        (N2-1 == Str.size() && std::memcmp(S2, Str.data(), N2-1) == 0))) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N0, unsigned N1, unsigned N2, unsigned N3>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const char (&S3)[N3],
                      const T& Value) {
    if (!Result && (
        (N0-1 == Str.size() && std::memcmp(S0, Str.data(), N0-1) == 0) ||
        (N1-1 == Str.size() && std::memcmp(S1, Str.data(), N1-1) == 0) ||
        (N2-1 == Str.size() && std::memcmp(S2, Str.data(), N2-1) == 0) ||
        (N3-1 == Str.size() && std::memcmp(S3, Str.data(), N3-1) == 0))) {
      Result = &Value;
    }

    return *this;
  }

  template<unsigned N0, unsigned N1, unsigned N2, unsigned N3, unsigned N4>
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  StringSwitch& Cases(const char (&S0)[N0], const char (&S1)[N1],
                      const char (&S2)[N2], const char (&S3)[N3],
                      const char (&S4)[N4], const T& Value) {
    if (!Result && (
        (N0-1 == Str.size() && std::memcmp(S0, Str.data(), N0-1) == 0) ||
        (N1-1 == Str.size() && std::memcmp(S1, Str.data(), N1-1) == 0) ||
        (N2-1 == Str.size() && std::memcmp(S2, Str.data(), N2-1) == 0) ||
        (N3-1 == Str.size() && std::memcmp(S3, Str.data(), N3-1) == 0) ||
        (N4-1 == Str.size() && std::memcmp(S4, Str.data(), N4-1) == 0))) {
      Result = &Value;
    }

    return *this;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  R Default(const T& Value) const {
    if (Result)
      return *Result;

    return Value;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  operator R() const {
    assert(Result && "Fell off the end of a string-switch");
    return *Result;
  }
};

} // end namespace llvm

#endif // LLVM_ADT_STRINGSWITCH_H
