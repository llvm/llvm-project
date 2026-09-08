//===------- Mangler.h -- Linker name mangling for ORC ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Linker name mangling for ORC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_MANGLER_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_MANGLER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Shared/SymbolNameSpec.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Triple;

namespace orc {

/// Applies linker name-mangling for a target.
class Mangler {
public:
  /// The linker name-mangling scheme for a target, determined by its object
  /// format. This captures the platform decoration applied to symbol names
  /// (e.g. a leading '_' on MachO), independently of any ExecutionSession.
  enum class ManglingMode {
    None,
    ELF,
    MachO,
    WinCOFF,
    WinCOFFX86,
    GOFF,
    Mips,
    XCOFF
  };

  explicit Mangler(ManglingMode Mode) : Mode(Mode) {}
  explicit Mangler(StringRef DLStr) : Mode(fromDataLayoutStr(DLStr)) {}
  explicit Mangler(const Triple &TT, StringRef ABIName = "")
      : Mode(fromTriple(TT, ABIName)) {}

  /// Calls the given callback with the mangled version of Name as a StringRef.
  /// The mangled name is only valid for the duration of the callback and must
  /// not escape. This allows withMangledNameDo to avoid allocations when
  /// mangling is a no-op.
  template <typename HandlerFn>
  decltype(auto) withMangledNameDo(HandlerFn &&H,
                                   const SymbolNameSpec &Name) const {
    if (Name.getKind() == SymbolNameKind::Verbatim ||
        Name.getKind() == SymbolNameKind::Linker)
      return H(Name.getName());

    if (Name.getName().empty())
      return H(Name.getName());

    if (Name.getName()[0] == '\1')
      return H(Name.getName().substr(1));

    if (Name.getName()[0] == '?' && doNotMangleLeadingQuestionMark())
      return H(Name.getName());

    if (Mode == ManglingMode::MachO || Mode == ManglingMode::WinCOFFX86) {
      SmallString<1024> MangledName;
      MangledName.append({StringRef("_"), Name.getName()});
      return H(StringRef(MangledName));
    }

    return H(Name.getName());
  }

  /// Construct a mangled version of the given name as a std::string.
  /// This always produces a copy, even for no-op manglings. Prefer
  /// withMangledNameDo in any performance-sensitive context.
  std::string mangledCopy(const SymbolNameSpec &Name) const {
    return withMangledNameDo(
        [](StringRef MangledName) { return MangledName.str(); }, Name);
  }

private:
  LLVM_ABI static ManglingMode fromDataLayoutStr(StringRef DLStr);
  LLVM_ABI static ManglingMode fromTriple(const Triple &TT, StringRef ABIName);
  bool doNotMangleLeadingQuestionMark() const {
    return Mode == ManglingMode::WinCOFF || Mode == ManglingMode::WinCOFFX86;
  }

  ManglingMode Mode;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MANGLING_H
