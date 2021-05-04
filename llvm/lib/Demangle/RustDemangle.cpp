//===--- RustDemangle.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a demangler for Rust v0 mangled symbols as specified in
// https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/RustDemangle.h"
#include "llvm/Demangle/Demangle.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

using namespace llvm;
using namespace rust_demangle;

char *llvm::rustDemangle(const char *MangledName, char *Buf, size_t *N,
                         int *Status) {
  if (MangledName == nullptr || (Buf != nullptr && N == nullptr)) {
    if (Status != nullptr)
      *Status = demangle_invalid_args;
    return nullptr;
  }

  // Return early if mangled name doesn't look like a Rust symbol.
  StringView Mangled(MangledName);
  if (!Mangled.startsWith("_R")) {
    if (Status != nullptr)
      *Status = demangle_invalid_mangled_name;
    return nullptr;
  }

  Demangler D;
  if (!initializeOutputStream(nullptr, nullptr, D.Output, 1024)) {
    if (Status != nullptr)
      *Status = demangle_memory_alloc_failure;
    return nullptr;
  }

  if (!D.demangle(Mangled)) {
    if (Status != nullptr)
      *Status = demangle_invalid_mangled_name;
    std::free(D.Output.getBuffer());
    return nullptr;
  }

  D.Output += '\0';
  char *Demangled = D.Output.getBuffer();
  size_t DemangledLen = D.Output.getCurrentPosition();

  if (Buf != nullptr) {
    if (DemangledLen <= *N) {
      std::memcpy(Buf, Demangled, DemangledLen);
      std::free(Demangled);
      Demangled = Buf;
    } else {
      std::free(Buf);
    }
  }

  if (N != nullptr)
    *N = DemangledLen;

  if (Status != nullptr)
    *Status = demangle_success;

  return Demangled;
}

Demangler::Demangler(size_t MaxRecursionLevel)
    : MaxRecursionLevel(MaxRecursionLevel) {}

static inline bool isDigit(const char C) { return '0' <= C && C <= '9'; }

static inline bool isLower(const char C) { return 'a' <= C && C <= 'z'; }

static inline bool isUpper(const char C) { return 'A' <= C && C <= 'Z'; }

/// Returns true if C is a valid mangled character: <0-9a-zA-Z_>.
static inline bool isValid(const char C) {
  return isDigit(C) || isLower(C) || isUpper(C) || C == '_';
}

// Demangles Rust v0 mangled symbol. Returns true when successful, and false
// otherwise. The demangled symbol is stored in Output field. It is
// responsibility of the caller to free the memory behind the output stream.
//
// <symbol-name> = "_R" <path> [<instantiating-crate>]
bool Demangler::demangle(StringView Mangled) {
  Position = 0;
  Error = false;
  RecursionLevel = 0;

  if (!Mangled.consumeFront("_R")) {
    Error = true;
    return false;
  }
  Input = Mangled;

  demanglePath();

  // FIXME parse optional <instantiating-crate>.

  if (Position != Input.size())
    Error = true;

  return !Error;
}

// <path> = "C" <identifier>               // crate root
//        | "M" <impl-path> <type>         // <T> (inherent impl)
//        | "X" <impl-path> <type> <path>  // <T as Trait> (trait impl)
//        | "Y" <type> <path>              // <T as Trait> (trait definition)
//        | "N" <ns> <path> <identifier>   // ...::ident (nested path)
//        | "I" <path> {<generic-arg>} "E" // ...<T, U> (generic args)
//        | <backref>
// <identifier> = [<disambiguator>] <undisambiguated-identifier>
// <ns> = "C"      // closure
//      | "S"      // shim
//      | <A-Z>    // other special namespaces
//      | <a-z>    // internal namespaces
void Demangler::demanglePath() {
  if (Error || RecursionLevel >= MaxRecursionLevel) {
    Error = true;
    return;
  }
  RecursionLevel += 1;

  switch (consume()) {
  case 'C': {
    parseOptionalBase62Number('s');
    Identifier Ident = parseIdentifier();
    print(Ident.Name);
    break;
  }
  case 'N': {
    char NS = consume();
    if (!isLower(NS) && !isUpper(NS)) {
      Error = true;
      break;
    }
    demanglePath();

    parseOptionalBase62Number('s');
    Identifier Ident = parseIdentifier();

    if (!Ident.empty()) {
      // FIXME print special namespaces:
      // * "C" closures
      // * "S" shim
      print("::");
      print(Ident.Name);
    }
    break;
  }
  default:
    // FIXME parse remaining productions.
    Error = true;
    break;
  }

  RecursionLevel -= 1;
}

// <undisambiguated-identifier> = ["u"] <decimal-number> ["_"] <bytes>
Identifier Demangler::parseIdentifier() {
  bool Punycode = consumeIf('u');
  uint64_t Bytes = parseDecimalNumber();

  // Underscore resolves the ambiguity when identifier starts with a decimal
  // digit or another underscore.
  consumeIf('_');

  if (Error || Bytes > Input.size() - Position) {
    Error = true;
    return {};
  }
  StringView S = Input.substr(Position, Bytes);
  Position += Bytes;

  if (!std::all_of(S.begin(), S.end(), isValid)) {
    Error = true;
    return {};
  }

  return {S, Punycode};
}

// Parses optional base 62 number. The presence of a number is determined using
// Tag.
void Demangler::parseOptionalBase62Number(char Tag) {
  // Parsing result is currently unused.
  if (consumeIf(Tag))
    parseBase62Number();
}

// Parses base 62 number with <0-9a-zA-Z> as digits. Number is terminated by
// "_". All values are offset by 1, so that "_" encodes 0, "0_" encodes 1,
// "1_" encodes 2, etc.
//
// <base-62-number> = {<0-9a-zA-Z>} "_"
uint64_t Demangler::parseBase62Number() {
  if (consumeIf('_'))
    return 0;

  uint64_t Value = 0;

  while (true) {
    uint64_t Digit;
    char C = consume();

    if (C == '_') {
      break;
    } else if (isDigit(C)) {
      Digit = C - '0';
    } else if (isLower(C)) {
      Digit = 10 + (C - 'a');
    } else if (isUpper(C)) {
      Digit = 10 + 26 + (C - 'A');
    } else {
      Error = true;
      return 0;
    }

    if (!mulAssign(Value, 62))
      return 0;

    if (!addAssign(Value, Digit))
      return 0;
  }

  if (!addAssign(Value, 1))
    return 0;

  return Value;
}

// Parses a decimal number that had been encoded without any leading zeros.
//
// <decimal-number> = "0"
//                  | <1-9> {<0-9>}
uint64_t Demangler::parseDecimalNumber() {
  char C = look();
  if (!isDigit(C)) {
    Error = true;
    return 0;
  }

  if (C == '0') {
    consume();
    return 0;
  }

  uint64_t Value = 0;

  while (isDigit(look())) {
    if (!mulAssign(Value, 10)) {
      Error = true;
      return 0;
    }

    uint64_t D = consume() - '0';
    if (!addAssign(Value, D))
      return 0;
  }

  return Value;
}
