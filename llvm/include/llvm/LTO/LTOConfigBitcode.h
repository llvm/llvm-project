//===- LTOConfigBitcode.h - lto::Config in bitcode ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility for embedding the serializable fields of lto::Config as JSON in
// ThinLTO summary indexes.
//
// Non-serializable fields (callbacks, loaded plugin pointers, stream handles)
// are omitted.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTOCONFIG_BITCODE_H
#define LLVM_LTO_LTOCONFIG_BITCODE_H

#include "llvm/LTO/Config.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace llvm {
namespace lto {

LLVM_ABI std::string serializeLTOConfig(const Config &Config);

/// Read the configuration JSON blob from the parsed index, if present.
LLVM_ABI Expected<std::optional<Config>>
readLTOConfigFromSummaryIndexIfPresent(const ModuleSummaryIndex &Index);

} // namespace lto
} // namespace llvm

#endif
