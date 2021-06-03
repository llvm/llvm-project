//===--- RustDemangle.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEMANGLE_RUSTDEMANGLE_H
#define LLVM_DEMANGLE_RUSTDEMANGLE_H

#include "llvm/Demangle/DemangleConfig.h"
#include "llvm/Demangle/StringView.h"
#include "llvm/Demangle/Utility.h"
#include <cstdint>

namespace llvm {
namespace rust_demangle {

using llvm::itanium_demangle::OutputStream;
using llvm::itanium_demangle::StringView;
using llvm::itanium_demangle::SwapAndRestore;

struct Identifier {
  StringView Name;
  bool Punycode;

  bool empty() const { return Name.empty(); }
};

enum class BasicType {
  Bool,
  Char,
  I8,
  I16,
  I32,
  I64,
  I128,
  ISize,
  U8,
  U16,
  U32,
  U64,
  U128,
  USize,
  F32,
  F64,
  Str,
  Placeholder,
  Unit,
  Variadic,
  Never,
};

enum class InType {
  No,
  Yes,
};

class Demangler {
  // Maximum recursion level. Used to avoid stack overflow.
  size_t MaxRecursionLevel;
  // Current recursion level.
  size_t RecursionLevel;
  size_t BoundLifetimes;
  // Input string that is being demangled with "_R" prefix removed.
  StringView Input;
  // Position in the input string.
  size_t Position;

  // When true, print methods append the output to the stream.
  // When false, the output is suppressed.
  bool Print;

  // True if an error occurred.
  bool Error;

public:
  // Demangled output.
  OutputStream Output;

  Demangler(size_t MaxRecursionLevel = 500);

  bool demangle(StringView MangledName);

private:
  void demanglePath(InType InType);
  void demangleImplPath(InType InType);
  void demangleGenericArg();
  void demangleType();
  void demangleFnSig();
  void demangleOptionalBinder();
  void demangleConst();
  void demangleConstInt();
  void demangleConstBool();
  void demangleConstChar();

  Identifier parseIdentifier();
  uint64_t parseOptionalBase62Number(char Tag);
  uint64_t parseBase62Number();
  uint64_t parseDecimalNumber();
  uint64_t parseHexNumber(StringView &HexDigits);

  void print(char C) {
    if (Error || !Print)
      return;

    Output += C;
  }

  void print(StringView S) {
    if (Error || !Print)
      return;

    Output += S;
  }

  void printDecimalNumber(uint64_t N) {
    if (Error || !Print)
      return;

    Output << N;
  }

  void printBasicType(BasicType);
  void printLifetime(uint64_t Index);

  char look() const {
    if (Error || Position >= Input.size())
      return 0;

    return Input[Position];
  }

  char consume() {
    if (Error || Position >= Input.size()) {
      Error = true;
      return 0;
    }

    return Input[Position++];
  }

  bool consumeIf(char Prefix) {
    if (Error || Position >= Input.size() || Input[Position] != Prefix)
      return false;

    Position += 1;
    return true;
  }

  /// Computes A + B. When computation wraps around sets the error and returns
  /// false. Otherwise assigns the result to A and returns true.
  bool addAssign(uint64_t &A, const uint64_t B) {
    if (A > std::numeric_limits<uint64_t>::max() - B) {
      Error = true;
      return false;
    }

    A += B;
    return true;
  }

  /// Computes A * B. When computation wraps around sets the error and returns
  /// false. Otherwise assigns the result to A and returns true.
  bool mulAssign(uint64_t &A, const uint64_t B) {
    if (B != 0 && A > std::numeric_limits<uint64_t>::max() / B) {
      Error = true;
      return false;
    }

    A *= B;
    return true;
  }
};

} // namespace rust_demangle
} // namespace llvm

#endif
