//===- DynamicLibrary.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/DynamicLibrary.h"

namespace orc_rt::sys {

void *globalLookupHandle() { return nullptr; }

Expected<void *> loadLibrary(const std::string &) {
  return make_error<StringError>("Windows loadLibrary not implemented");
}

Error unloadLibrary(void *) {
  return make_error<StringError>("Windows unloadLibrary not implemented");
}

std::vector<std::optional<void *>>
lookupLibrarySymbols(void *, const std::vector<std::string> &Names) {
  return std::vector<std::optional<void *>>(Names.size(), std::nullopt);
}

} // namespace orc_rt::sys
