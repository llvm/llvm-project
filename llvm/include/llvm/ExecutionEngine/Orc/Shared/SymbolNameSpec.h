//===- SymbolNameSpec.h - A symbol name plus its mangling kind --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A symbol name paired with the naming level it is expressed in.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLNAMESPEC_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLNAMESPEC_H

#include "llvm/ADT/StringRef.h"

namespace llvm::orc {

/// The naming level a symbol name is expressed in, which determines how it is
/// mangled before being interned for lookup.
enum class SymbolNameKind {
  Verbatim, // Use the name as given, with no mangling.
  Linker,   // An already-decorated linker name; a synonym for Verbatim.
  IR,       // An IR global name; mangled to linker level for the target.
  C,        // A C source name; handled identically to IR.
};

/// A symbol name together with the naming level (SymbolNameKind) it is
/// expressed in, so that a Mangler / MangleAndInterner can mangle it to linker
/// level rather than requiring callers to pre-mangle.
///
/// The name is not copied: a SymbolNameSpec must not outlive the string it
/// refers to.
///
/// Implicitly constructible from a StringRef, defaulting to Verbatim, so that
/// APIs taking a SymbolNameSpec stay drop-in replacements for ones that
/// previously took an already-mangled StringRef.
class SymbolNameSpec {
public:
  constexpr SymbolNameSpec(StringRef Name,
                           SymbolNameKind Kind = SymbolNameKind::Verbatim)
      : Name(Name), Kind(Kind) {}

  static constexpr SymbolNameSpec verbatim(StringRef Name) {
    return {Name, SymbolNameKind::Verbatim};
  }
  static constexpr SymbolNameSpec linker(StringRef Name) {
    return {Name, SymbolNameKind::Linker};
  }
  static constexpr SymbolNameSpec ir(StringRef Name) {
    return {Name, SymbolNameKind::IR};
  }
  static constexpr SymbolNameSpec c(StringRef Name) {
    return {Name, SymbolNameKind::C};
  }

  constexpr StringRef getName() const { return Name; }
  constexpr SymbolNameKind getKind() const { return Kind; }

private:
  StringRef Name;
  SymbolNameKind Kind;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SYMBOLNAMESPEC_H
