//===- AMDGPUMetadataVerifier.h - MsgPack Types -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This is a verifier for AMDGPU HSA metadata, which can verify both
/// well-typed metadata and untyped metadata. When verifying in the non-strict
/// mode, untyped metadata is coerced into the correct type if possible.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_AMDGPUMETADATAVERIFIER_H
#define LLVM_BINARYFORMAT_AMDGPUMETADATAVERIFIER_H

#include "llvm/BinaryFormat/MsgPackTypes.h"

namespace llvm {
namespace AMDGPU {
namespace HSAMD {
namespace V3 {

/// Verifier for AMDGPU HSA metadata.
///
/// Operates in two modes:
///
/// In strict mode, metadata must already be well-typed.
///
/// In non-strict mode, metadata is coerced into expected types when possible.
class MetadataVerifier {
  bool Strict;

  bool verifyScalar(msgpack::Node &Node, msgpack::ScalarNode::ScalarKind SKind,
                    function_ref<bool(msgpack::ScalarNode &)> verifyValue = {});
  bool verifyInteger(msgpack::Node &Node);
  bool verifyArray(msgpack::Node &Node,
                   function_ref<bool(msgpack::Node &)> verifyNode,
                   Optional<size_t> Size = None);
  bool verifyEntry(msgpack::MapNode &MapNode, StringRef Key, bool Required,
                   function_ref<bool(msgpack::Node &)> verifyNode);
  bool
  verifyScalarEntry(msgpack::MapNode &MapNode, StringRef Key, bool Required,
                    msgpack::ScalarNode::ScalarKind SKind,
                    function_ref<bool(msgpack::ScalarNode &)> verifyValue = {});
  bool verifyIntegerEntry(msgpack::MapNode &MapNode, StringRef Key,
                          bool Required);
  bool verifyKernelArgs(msgpack::Node &Node);
  bool verifyKernel(msgpack::Node &Node);

public:
  /// Construct a MetadataVerifier, specifying whether it will operate in \p
  /// Strict mode.
  MetadataVerifier(bool Strict) : Strict(Strict) {}

  /// Verify given HSA metadata.
  ///
  /// \returns True when successful, false when metadata is invalid.
  bool verify(msgpack::Node &HSAMetadataRoot);
};

} // end namespace V3
} // end namespace HSAMD
} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_BINARYFORMAT_AMDGPUMETADATAVERIFIER_H
