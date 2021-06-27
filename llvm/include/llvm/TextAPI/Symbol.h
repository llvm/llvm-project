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
#include "llvm/TextAPI/ArchitectureSet.h"
#include "llvm/TextAPI/Target.h"

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

  /// Rexported
  Rexported        = 1U << 4,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Rexported),
};

// clang-format on

enum class SymbolKind : uint8_t {
  GlobalSymbol,
  ObjectiveCClass,
  ObjectiveCClassEHType,
  ObjectiveCInstanceVariable,
};

constexpr StringLiteral ObjC1ClassNamePrefix = ".objc_class_name_";
constexpr StringLiteral ObjC2ClassNamePrefix = "_OBJC_CLASS_$_";
constexpr StringLiteral ObjC2MetaClassNamePrefix = "_OBJC_METACLASS_$_";
constexpr StringLiteral ObjC2EHTypePrefix = "_OBJC_EHTYPE_$_";
constexpr StringLiteral ObjC2IVarPrefix = "_OBJC_IVAR_$_";

using TargetList = SmallVector<Target, 5>;
class Symbol {
public:
  Symbol(SymbolKind Kind, StringRef Name, TargetList Targets, SymbolFlags Flags)
      : Name(Name), Targets(std::move(Targets)), Kind(Kind), Flags(Flags) {}

  void addTarget(Target target) { Targets.emplace_back(target); }
  SymbolKind getKind() const { return Kind; }
  StringRef getName() const { return Name; }
  ArchitectureSet getArchitectures() const {
    return mapToArchitectureSet(Targets);
  }
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

  bool isReexported() const {
    return (Flags & SymbolFlags::Rexported) == SymbolFlags::Rexported;
  }

  using const_target_iterator = TargetList::const_iterator;
  using const_target_range = llvm::iterator_range<const_target_iterator>;
  const_target_range targets() const { return {Targets}; }

  using const_filtered_target_iterator =
      llvm::filter_iterator<const_target_iterator,
                            std::function<bool(const Target &)>>;
  using const_filtered_target_range =
      llvm::iterator_range<const_filtered_target_iterator>;
  const_filtered_target_range targets(ArchitectureSet architectures) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void dump(raw_ostream &OS) const;
  void dump() const { dump(llvm::errs()); }
#endif

  bool operator==(const Symbol &O) const {
    return std::tie(Name, Kind, Targets, Flags) ==
           std::tie(O.Name, O.Kind, O.Targets, O.Flags);
  }

  bool operator!=(const Symbol &O) const { return !(*this == O); }

  bool operator<(const Symbol &O) const {
    return std::tie(Name, Kind, Targets, Flags) <
           std::tie(O.Name, O.Kind, O.Targets, O.Flags);
  }

private:
  StringRef Name;
  TargetList Targets;
  SymbolKind Kind;
  SymbolFlags Flags;
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_MACHO_SYMBOL_H
