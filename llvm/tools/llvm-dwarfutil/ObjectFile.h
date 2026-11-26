//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_DWARFUTIL_OBJECTFILE_H
#define LLVM_TOOLS_LLVM_DWARFUTIL_OBJECTFILE_H

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <system_error>
#include <utility>

namespace llvm {
namespace dwarfutil {

template <typename... Types, typename Fn>
Error dispatchImpl(object::ObjectFile &File, Fn &&Apply) {
  std::optional<Error> Result;
  bool Matched = ([&] {
    if (auto *TypedFile = dyn_cast<Types>(&File)) {
      Result.emplace(Apply(*TypedFile));
      return true;
    }
    return false;
  }() || ...);

  if (!Matched)
    return createStringError(std::errc::invalid_argument,
                             "unsupported file format");
  return std::move(*Result);
}

// Keep the supported formats in one place. Callbacks receive the concrete
// object type so format-specific operations can be selected by overloads.
template <typename Fn>
Error dispatchObject(object::ObjectFile &File, Fn &&Apply) {
  return dispatchImpl<object::ELF32LEObjectFile, object::ELF64LEObjectFile,
                      object::ELF32BEObjectFile, object::ELF64BEObjectFile>(
      File, std::forward<Fn>(Apply));
}

} // namespace dwarfutil
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_DWARFUTIL_OBJECTFILE_H
