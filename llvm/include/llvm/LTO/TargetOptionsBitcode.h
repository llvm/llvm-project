//===- TargetOptionsBitcode.h - TargetOptions in bitcode --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility for embedding llvm::TargetOptions in LLVM IR bitcode via module
// metadata. Intended for LTO / DTLTO configuration transport.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LTO_TARGETOPTIONS_BITCODE_H
#define LLVM_LTO_TARGETOPTIONS_BITCODE_H

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {
namespace lto {

/// Metadata name written into the module and persisted in bitcode.
inline constexpr StringLiteral TargetOptionsMetadataName =
    "llvm.lto.target_options";

/// Serialize \p Options into \p M as named module metadata.
/// Non-serializable fields are skipped.
LLVM_ABI Error encodeTargetOptionsToModule(Module &M,
                                           const TargetOptions &Options);

/// Deserialize TargetOptions previously stored by encodeTargetOptionsToModule.
/// Returns an error if metadata is missing or malformed.
LLVM_ABI Expected<TargetOptions> decodeTargetOptionsFromModule(const Module &M);

/// Returns true if \p M contains serialized TargetOptions metadata.
LLVM_ABI bool hasEncodedTargetOptions(const Module &M);

/// Encode TargetOptions as a standalone metadata node (for nesting).
LLVM_ABI MDNode *encodeTargetOptionsAsNode(LLVMContext &Ctx,
                                           const TargetOptions &Options);

/// Decode TargetOptions from a node produced by encodeTargetOptionsAsNode.
LLVM_ABI Expected<TargetOptions>
decodeTargetOptionsFromNode(const MDNode *Root);

} // namespace lto
} // namespace llvm

#endif
