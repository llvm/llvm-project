//===------------ Mangling.h - Name mangling for ORC RT ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mangling of symbol names to the linker-level names used by the platform.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_MANGLING_H
#define ORC_RT_SUPPORT_MANGLING_H

#include <cstring>
#include <memory>
#include <string>
#include <string_view>

namespace orc_rt {

/// The level a symbol name is written at, which determines whether the platform
/// decorates it.
///
/// Linker names are linker-level already and are used as written. C names are
/// C identifiers, and may be decorated: on Darwin they gain a leading
/// underscore.
enum class SymbolNameKind { Linker, C };

/// Pairs a symbol name with the kind of name it is, so that mangling can be
/// applied where -- and only where -- the platform calls for it.
///
/// This does not own the name: it holds a string_view. It is intended for
/// passing names to APIs that mangle them immediately, and for arrays built
/// from string literals. Holding one alongside a name with a shorter lifetime
/// leaves a dangling view.
class SymbolNameSpec {
public:
  constexpr SymbolNameSpec(std::string_view Name, SymbolNameKind Kind) noexcept
      : Name(Name), Kind(Kind) {}

  /// Returns a spec for a name that is already linker-level.
  static constexpr SymbolNameSpec linker(std::string_view Name) noexcept {
    return SymbolNameSpec(Name, SymbolNameKind::Linker);
  }

  /// Returns a spec for a C identifier, to be decorated by the platform.
  static constexpr SymbolNameSpec c(std::string_view Name) noexcept {
    return SymbolNameSpec(Name, SymbolNameKind::C);
  }

  constexpr std::string_view name() const noexcept { return Name; }
  constexpr SymbolNameKind kind() const noexcept { return Kind; }

private:
  std::string_view Name;
  SymbolNameKind Kind = SymbolNameKind::Linker;
};

/// Mangles NameSpec and passes the result to H, returning whatever H returns.
///
/// The mangled name is passed rather than returned so that the common case --
/// a name short enough to build in a stack buffer -- needs no allocation. The
/// string_view H receives is valid only for the duration of the call.
template <typename HandlerT>
decltype(auto) withMangledNameDo(HandlerT &&H,
                                 const SymbolNameSpec &NameSpec) noexcept {
#if defined(__APPLE__)
  if (NameSpec.kind() == SymbolNameKind::Linker || NameSpec.name().empty())
    return H(NameSpec.name());

  constexpr size_t InlineStorageSize = 1024;
  char InlineStorage[InlineStorageSize];
  char *Buffer = InlineStorage;
  std::unique_ptr<char[]> BigBuffer;
  size_t NewNameSize = NameSpec.name().size() + 1;

  // The buffer holds the underscore, the name, and a NUL, so it needs
  // name().size() + 2 bytes.
  if (NameSpec.name().size() + 2 > InlineStorageSize) {
    BigBuffer = std::make_unique<char[]>(NewNameSize + 1);
    Buffer = &BigBuffer[0];
  }
  Buffer[0] = '_';
  memcpy(&Buffer[1], NameSpec.name().data(), NameSpec.name().size());
  Buffer[NewNameSize] = '\0';

  return H(std::string_view(Buffer, NewNameSize));
#else
  // No mangling.
  return H(NameSpec.name());
#endif
}

/// Returns the mangled form of NameSpec as a std::string.
///
/// Prefer withMangledNameDo where the name does not need to outlive the call.
inline std::string mangledCopy(const SymbolNameSpec &NameSpec) noexcept {
  return withMangledNameDo(
      [](std::string_view MangledName) { return std::string(MangledName); },
      NameSpec);
}

} // namespace orc_rt

#endif // ORC_RT_SUPPORT_MANGLING_H
