//===- LTOConfigBitcode.h - lto::Config in bitcode ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility for embedding serializable fields of lto::Config in LLVM IR bitcode
// via module metadata. Intended for LTO / DTLTO configuration transport.
//
// Non-serializable fields (callbacks, loaded plugin pointers, stream handles)
// are omitted. See encodeLTOConfigToModule() documentation in the .cpp file.
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

/// Serialize all serializable fields of \p Config into \p M.
LLVM_ABI Error encodeLTOConfigToModule(Module &M, const Config &Config);

/// Deserialize lto::Config previously stored by encodeLTOConfigToModule.
LLVM_ABI Expected<Config> decodeLTOConfigFromModule(const Module &M);

/// Serialize \p Config into a standalone LLVM bitcode file at \p Path.
LLVM_ABI Error writeLTOConfigToFile(StringRef Path, const Config &Config);

/// Read a Config from a file written by writeLTOConfigToFile().
LLVM_ABI Expected<Config> readLTOConfigFromFile(StringRef Path);

/// Write a ThinLTO summary index containing serialized Config metadata.
LLVM_ABI Error writeIndexWithLTOConfigToFile(
    const ModuleSummaryIndex &Index, const Config &Config, raw_ostream &Out,
    const ModuleToSummariesForIndexTy *ModuleToSummariesForIndex = nullptr,
    const GVSummaryPtrSet *DecSummaries = nullptr);

/// Read Config metadata from a ThinLTO summary index.
LLVM_ABI Expected<Config> readLTOConfigFromSummaryIndex(MemoryBufferRef Buffer);

/// Read Config metadata from a ThinLTO summary index, or return std::nullopt if
/// the index has no Config metadata.
LLVM_ABI Expected<std::optional<Config>>
readLTOConfigFromSummaryIndexIfPresent(MemoryBufferRef Buffer);

/// Returns true if \p M contains serialized lto::Config metadata.
LLVM_ABI bool hasEncodedLTOConfig(const Module &M);

} // namespace lto
} // namespace llvm

#endif
