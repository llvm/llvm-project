//===- llvm/TextAPI/Symbol.h - TAPI Symbol ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_SYMBOL_H
#define LLVM_TEXTAPI_MACHO_SYMBOL_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TextAPI/MachO/ArchitectureSet.h"

namespace llvm {
namespace MachO {

// clang-format off

/// Symbol flags.
enum class SymbolFlags : uint8_t {
  /// No flags
  None             = 0,

  /// Thread-local value symbol
  ThreadLocalValue = 1U << 0,

  /// Weak defined symbol
  WeakDefined      = 1U << 1,

  /// Weak referenced symbol
  WeakReferenced   = 1U << 2,

  /// Undefined
  Undefined        = 1U << 3,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Undefined),
};

// clang-format on

enum class SymbolKind : uint8_t {
  GlobalSymbol,
  ObjectiveCClass,
  ObjectiveCClassEHType,
  ObjectiveCInstanceVariable,
};

class Symbol {
public:
  constexpr Symbol(SymbolKind Kind, StringRef Name,
                   ArchitectureSet Architectures, SymbolFlags Flags)
      : Name(Name), Architectures(Architectures), Kind(Kind), Flags(Flags) {}

  SymbolKind getKind() const { return Kind; }
  StringRef getName() const { return Name; }
  ArchitectureSet getArchitectures() const { return Architectures; }
  void addArchitectures(ArchitectureSet Archs) { Architectures |= Archs; }
  SymbolFlags getFlags() const { return Flags; }

  bool isWeakDefined() const {
    return (Flags & SymbolFlags::WeakDefined) == SymbolFlags::WeakDefined;
  }

  bool isWeakReferenced() const {
    return (Flags & SymbolFlags::WeakReferenced) == SymbolFlags::WeakReferenced;
  }

  bool isThreadLocalValue() const {
    return (Flags & SymbolFlags::ThreadLocalValue) ==
           SymbolFlags::ThreadLocalValue;
  }

  bool isUndefined() const {
    return (Flags & SymbolFlags::Undefined) == SymbolFlags::Undefined;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump(raw_ostream &OS) const;
  void dump() const { dump(llvm::errs()); }
#endif

private:
  StringRef Name;
  ArchitectureSet Architectures;
  SymbolKind Kind;
  SymbolFlags Flags;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_SYMBOL_H
