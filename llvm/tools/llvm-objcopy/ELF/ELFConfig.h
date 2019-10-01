//===- ELFConfig.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_ELFCONFIG_H
#define LLVM_TOOLS_OBJCOPY_ELFCONFIG_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace llvm {
namespace objcopy {
struct CopyConfig;

namespace elf {

struct NewSymbolInfo {
  StringRef SymbolName;
  StringRef SectionName;
  uint64_t Value = 0;
  uint8_t Type = ELF::STT_NOTYPE;
  uint8_t Bind = ELF::STB_GLOBAL;
  uint8_t Visibility = ELF::STV_DEFAULT;
};

struct ELFCopyConfig {
  Optional<uint8_t> NewSymbolVisibility;
  std::vector<NewSymbolInfo> SymbolsToAdd;
};

Expected<ELFCopyConfig> parseConfig(const CopyConfig &Config);

} // namespace elf
} // namespace objcopy
} // namespace llvm

#endif
