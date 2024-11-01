//===------- Utilities.h - Target independent OpenMP target RTL -- C++ ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Routines and classes used to provide useful functionalities like string
// parsing and environment variables.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_INCLUDE_UTILITIES_H
#define OPENMP_LIBOMPTARGET_INCLUDE_UTILITIES_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "Debug.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace llvm {
namespace omp {
namespace target {

/// Utility class for parsing strings to other types.
struct StringParser {
  /// Parse a string to another type.
  template <typename Ty> static bool parse(const char *Value, Ty &Result);
};

/// Class for reading and checking environment variables. Currently working with
/// integer, floats, std::string and bool types.
template <typename Ty> class Envar {
  Ty Data;
  bool IsPresent;
  bool Initialized;

public:
  /// Auxiliary function to safely create envars. This static function safely
  /// creates envars using fallible constructors. See the constructors to know
  /// more details about the creation parameters.
  template <typename... ArgsTy>
  static Expected<Envar> create(ArgsTy &&...Args) {
    Error Err = Error::success();
    Envar Envar(std::forward<ArgsTy>(Args)..., Err);
    if (Err)
      return std::move(Err);
    return std::move(Envar);
  }

  /// Create an empty envar. Cannot be consulted. This constructor is merely
  /// for convenience. This constructor is not fallible.
  Envar() : Data(Ty()), IsPresent(false), Initialized(false) {}

  /// Create an envar with a name and an optional default. The Envar object will
  /// take the value read from the environment variable, or the default if it
  /// was not set or not correct. This constructor is not fallible.
  Envar(StringRef Name, Ty Default = Ty())
      : Data(Default), IsPresent(false), Initialized(true) {

    if (const char *EnvStr = getenv(Name.data())) {
      // Check whether the envar is defined and valid.
      IsPresent = StringParser::parse<Ty>(EnvStr, Data);

      if (!IsPresent) {
        DP("Ignoring invalid value %s for envar %s\n", EnvStr, Name.data());
        Data = Default;
      }
    }
  }

  /// Get the definitive value.
  const Ty &get() const {
    // Throw a runtime error in case this envar is not initialized.
    if (!Initialized)
      FATAL_MESSAGE0(1, "Consulting envar before initialization");

    return Data;
  }

  /// Get the definitive value.
  operator Ty() const { return get(); }

  /// Indicate whether the environment variable was defined and valid.
  bool isPresent() const { return IsPresent; }

private:
  /// This constructor should never fail but we provide it for convenience. This
  /// way, the constructor can be used by the Envar::create() static function
  /// to safely create this kind of envars.
  Envar(StringRef Name, Ty Default, Error &Err) : Envar(Name, Default) {
    ErrorAsOutParameter EAO(&Err);
    Err = Error::success();
  }

  /// Create an envar with a name, getter function and a setter function. The
  /// Envar object will take the value read from the environment variable if
  /// this value is accepted by the setter function. Otherwise, the getter
  /// function will be executed to get the default value. The getter should be
  /// of the form Error GetterFunctionTy(Ty &Value) and the setter should
  /// be of the form Error SetterFunctionTy(Ty Value). This constructor has a
  /// private visibility because is a fallible constructor. Please use the
  /// Envar::create() static function to safely create this object instead.
  template <typename GetterFunctor, typename SetterFunctor>
  Envar(StringRef Name, GetterFunctor Getter, SetterFunctor Setter, Error &Err)
      : Data(Ty()), IsPresent(false), Initialized(true) {
    ErrorAsOutParameter EAO(&Err);
    Err = init(Name, Getter, Setter);
  }

  template <typename GetterFunctor, typename SetterFunctor>
  Error init(StringRef Name, GetterFunctor Getter, SetterFunctor Setter);
};

/// Define some common envar types.
using IntEnvar = Envar<int>;
using Int32Envar = Envar<int32_t>;
using Int64Envar = Envar<int64_t>;
using UInt32Envar = Envar<uint32_t>;
using UInt64Envar = Envar<uint64_t>;
using StringEnvar = Envar<std::string>;
using BoolEnvar = Envar<bool>;

/// Utility class for thread-safe reference counting. Any class that needs
/// objects' reference counting can inherit from this entity or have it as a
/// class data member.
template <typename Ty = uint32_t,
          std::memory_order MemoryOrder = std::memory_order_relaxed>
struct RefCountTy {
  /// Create a refcount object initialized to zero.
  RefCountTy() : Refs(0) {}

  ~RefCountTy() { assert(Refs == 0 && "Destroying with non-zero refcount"); }

  /// Increase the reference count atomically.
  void increase() { Refs.fetch_add(1, MemoryOrder); }

  /// Decrease the reference count and return whether it became zero. Decreasing
  /// the counter in more units than it was previously increased results in
  /// undefined behavior.
  bool decrease() {
    Ty Prev = Refs.fetch_sub(1, MemoryOrder);
    assert(Prev > 0 && "Invalid refcount");
    return (Prev == 1);
  }

  Ty get() const { return Refs.load(MemoryOrder); }

private:
  /// The atomic reference counter.
  std::atomic<Ty> Refs;
};

template <>
inline bool StringParser::parse(const char *ValueStr, bool &Result) {
  std::string Value(ValueStr);

  // Convert the string to lowercase.
  std::transform(Value.begin(), Value.end(), Value.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // May be implemented with fancier C++ features, but let's keep it simple.
  if (Value == "true" || Value == "yes" || Value == "on" || Value == "1")
    Result = true;
  else if (Value == "false" || Value == "no" || Value == "off" || Value == "0")
    Result = false;
  else
    return false;

  // Parsed correctly.
  return true;
}

template <typename Ty>
inline bool StringParser::parse(const char *Value, Ty &Result) {
  assert(Value && "Parsed value cannot be null");

  std::istringstream Stream(Value);
  Stream >> Result;

  return !Stream.fail();
}

template <typename Ty>
template <typename GetterFunctor, typename SetterFunctor>
inline Error Envar<Ty>::init(StringRef Name, GetterFunctor Getter,
                             SetterFunctor Setter) {
  // Get the default value.
  Ty Default;
  if (Error Err = Getter(Default))
    return Err;

  if (const char *EnvStr = getenv(Name.data())) {
    IsPresent = StringParser::parse<Ty>(EnvStr, Data);
    if (IsPresent) {
      // Check whether the envar value is actually valid.
      Error Err = Setter(Data);
      if (Err) {
        // The setter reported an invalid value. Mark the user-defined value as
        // not present and reset to the getter value (default).
        IsPresent = false;
        Data = Default;
        DP("Setter of envar %s failed, resetting to %s\n", Name.data(),
           std::to_string(Data).data());
        consumeError(std::move(Err));
      }
    } else {
      DP("Ignoring invalid value %s for envar %s\n", EnvStr, Name.data());
      Data = Default;
    }
  } else {
    Data = Default;
  }

  return Error::success();
}

/// Return the difference (in bytes) between \p Begin and \p End.
template <typename Ty = char>
ptrdiff_t getPtrDiff(const void *End, const void *Begin) {
  return reinterpret_cast<const Ty *>(End) -
         reinterpret_cast<const Ty *>(Begin);
}

/// Return \p Ptr advanced by \p Offset bytes.
template <typename Ty> Ty *advanceVoidPtr(Ty *Ptr, int64_t Offset) {
  static_assert(std::is_void<Ty>::value);
  return const_cast<char *>(reinterpret_cast<const char *>(Ptr) + Offset);
}

/// Return \p Ptr aligned to \p Alignment bytes.
template <typename Ty> Ty *alignPtr(Ty *Ptr, int64_t Alignment) {
  size_t Space = std::numeric_limits<size_t>::max();
  return std::align(Alignment, sizeof(char), Ptr, Space);
}

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_INCLUDE_UTILITIES_H
