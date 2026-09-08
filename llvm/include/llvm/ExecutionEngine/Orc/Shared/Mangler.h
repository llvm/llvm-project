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
  enum class Mode { None, ELF, MachO, WinCOFF, WinCOFFX86, GOFF, Mips, XCOFF };

  explicit Mangler(Mode MM) : MM(MM) {}
  explicit Mangler(StringRef DLStr) : MM(fromDataLayoutStr(DLStr)) {}
  explicit Mangler(const Triple &TT, StringRef ABIName = "")
      : MM(fromTriple(TT, ABIName)) {}

  /// Calls the given callback with the mangled version of NameSpec as a
  /// StringRef. The mangled name is only valid for the duration of the callback
  /// and must not escape. This allows withMangledNameDo to avoid allocations
  /// when mangling is a no-op.
  template <typename HandlerFn>
  decltype(auto) withMangledNameDo(HandlerFn &&H,
                                   const SymbolNameSpec &NameSpec) const {
    if (NameSpec.getKind() == SymbolNameKind::Verbatim ||
        NameSpec.getKind() == SymbolNameKind::Linker)
      return H(NameSpec.getName());

    if (NameSpec.getName().empty())
      return H(NameSpec.getName());

    if (NameSpec.getName()[0] == '\1')
      return H(NameSpec.getName().substr(1));

    if (NameSpec.getName()[0] == '?' && doNotMangleLeadingQuestionMark())
      return H(NameSpec.getName());

    if (MM == Mode::MachO || MM == Mode::WinCOFFX86) {
      SmallString<1024> MangledName;
      MangledName.append({StringRef("_"), NameSpec.getName()});
      return H(StringRef(MangledName));
    }

    return H(NameSpec.getName());
  }

  /// Construct a mangled version of the given name as a std::string.
  /// This always produces a copy, even for no-op manglings. Prefer
  /// withMangledNameDo in any performance-sensitive context.
  std::string mangledCopy(const SymbolNameSpec &Name) const {
    return withMangledNameDo(
        [](StringRef MangledName) { return MangledName.str(); }, Name);
  }

private:
  LLVM_ABI static Mode fromDataLayoutStr(StringRef DLStr);
  LLVM_ABI static Mode fromTriple(const Triple &TT, StringRef ABIName);
  bool doNotMangleLeadingQuestionMark() const {
    return MM == Mode::WinCOFF || MM == Mode::WinCOFFX86;
  }

  Mode MM;
};

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_MANGLING_H
