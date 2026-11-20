//===- LTOConfigBitcode.h - lto::Config in bitcode ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility for embedding the serializable fields of lto::Config as JSON in
// ThinLTO summary indexes and standalone bitcode files.
//
// Non-serializable fields (callbacks, loaded plugin pointers, stream handles)
// are omitted. Readers also accept the previous structured metadata format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_LTOCONFIG_BITCODE_H
#define LLVM_LTO_LTOCONFIG_BITCODE_H

#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/LTO/Config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <optional>

namespace llvm {
namespace lto {

inline constexpr StringLiteral LTOConfigMetadataName = "llvm.lto.config";

LLVM_ABI std::string serializeLTOConfig(const Config &Config);

/// Serialize all serializable fields of \p Config into \p M.
LLVM_ABI Error encodeLTOConfigToModule(Module &M, const Config &Config);

/// Deserialize lto::Config previously stored by encodeLTOConfigToModule.
LLVM_ABI Expected<Config> decodeLTOConfigFromModule(const Module &M);

/// Serialize \p Config into a standalone LLVM bitcode file at \p Path.
LLVM_ABI Error writeLTOConfigToFile(StringRef Path, const Config &Config);

/// Read a Config from a file written by writeLTOConfigToFile().
LLVM_ABI Expected<Config> readLTOConfigFromFile(StringRef Path);

/// Read the Config from a ThinLTO summary index.
LLVM_ABI Expected<Config> readLTOConfigFromSummaryIndex(MemoryBufferRef Buffer);

/// Read the Config from a ThinLTO summary index, if present.
LLVM_ABI Expected<std::optional<Config>>
readLTOConfigFromSummaryIndexIfPresent(MemoryBufferRef Buffer);

/// Use the parsed index and consult its original bitcode for legacy metadata.
LLVM_ABI Expected<std::optional<Config>>
readLTOConfigFromSummaryIndexIfPresent(const ModuleSummaryIndex &Index,
                                       MemoryBufferRef Buffer);

/// Returns true if \p M contains serialized lto::Config metadata.
LLVM_ABI bool hasEncodedLTOConfig(const Module &M);

} // namespace lto
} // namespace llvm

#endif
