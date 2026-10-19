//===- ELFConfig.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_ELF_ELFCONFIG_H
#define LLVM_OBJCOPY_ELF_ELFCONFIG_H

#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/ObjCopy/CommonConfig.h"
#include "llvm/Object/ELFTypes.h"

namespace llvm {
namespace objcopy {

// Note to remove info specified by --remove-note option.
struct RemoveNoteInfo {
  StringRef Name;
  uint32_t TypeId;
};

// ELF specific configuration for copying/stripping a single file.
struct ELFConfig {
  uint8_t NewSymbolVisibility = (uint8_t)ELF::STV_DEFAULT;

  std::vector<std::pair<NameMatcher, uint8_t>> SymbolsToSetVisibility;

  // ELF entry point address expression. The input parameter is an entry point
  // address in the input ELF file. The entry address in the output file is
  // calculated with EntryExpr(input_address), when either --set-start or
  // --change-start is used.
  std::function<uint64_t(uint64_t)> EntryExpr;

  bool getAllowBrokenLinks() const { return NotesToRemove.getFlag(0); }
  void setAllowBrokenLinks(bool Value) { NotesToRemove.setFlag(0, Value); }
  bool getKeepFileSymbols() const { return NotesToRemove.getFlag(1); }
  void setKeepFileSymbols(bool Value) { NotesToRemove.setFlag(1, Value); }
  bool getLocalizeHidden() const { return NotesToRemove.getFlag(2); }
  void setLocalizeHidden(bool Value) { NotesToRemove.setFlag(2, Value); }
  bool getVerifyNoteSections() const { return !NotesToRemove.getFlag(3); }
  void setVerifyNoteSections(bool Value) { NotesToRemove.setFlag(3, !Value); }

  // Notes specified by --remove-note option.
  llvm::SmallVectorWithFlags<RemoveNoteInfo, 0, 4> NotesToRemove;
};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_OBJCOPY_ELF_ELFCONFIG_H
