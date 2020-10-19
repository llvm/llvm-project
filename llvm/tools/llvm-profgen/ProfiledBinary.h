//===-- ProfiledBinary.h - Binary decoder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H
#define LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace sampleprof {

class ProfiledBinary {
  std::string Path;
  mutable uint64_t BaseAddress = 0;

public:
  ProfiledBinary(StringRef Path) : Path(Path) { load(); }

  const StringRef getPath() const { return Path; }
  const StringRef getName() const { return llvm::sys::path::filename(Path); }
  uint64_t getBaseAddress() const { return BaseAddress; }
  void setBaseAddress(uint64_t Address) { BaseAddress = Address; }

private:
  void load() {
    // TODO:
  }
};

} // end namespace sampleprof
} // end namespace llvm

#endif
