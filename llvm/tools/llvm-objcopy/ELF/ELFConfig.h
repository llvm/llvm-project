//===- ELFConfig.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJCOPY_ELF_ELFCONFIG_H
#define LLVM_TOOLS_LLVM_OBJCOPY_ELF_ELFCONFIG_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFTypes.h"
#include <vector>

namespace llvm {
namespace objcopy {

// ELF specific configuration for copying/stripping a single file.
struct ELFConfig {
  uint8_t NewSymbolVisibility = (uint8_t)ELF::STV_DEFAULT;
};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_OBJCOPY_ELF_ELFCONFIG_H
