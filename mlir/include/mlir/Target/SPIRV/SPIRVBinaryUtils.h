//===- SPIRVBinaryUtils.cpp - SPIR-V Binary Module Utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities for SPIR-V binary module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_BINARY_UTILS_H_
#define MLIR_TARGET_SPIRV_BINARY_UTILS_H_

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
namespace spirv {

/// SPIR-V binary header word count
constexpr unsigned kHeaderWordCount = 5;

/// SPIR-V magic number
constexpr uint32_t kMagicNumber = 0x07230203;

/// The serializer tool ID registered to the Khronos Group
constexpr uint32_t kGeneratorNumber = 22;

/// Appends a SPRI-V module header to `header` with the given `version` and
/// `idBound`.
void appendModuleHeader(SmallVectorImpl<uint32_t> &header,
                        spirv::Version version, uint32_t idBound);

/// Returns the word-count-prefixed opcode for an SPIR-V instruction.
uint32_t getPrefixedOpcode(uint32_t wordCount, spirv::Opcode opcode);

/// Encodes an SPIR-V `literal` string into the given `binary` vector.
LogicalResult encodeStringLiteralInto(SmallVectorImpl<uint32_t> &binary,
                                      StringRef literal);
} // end namespace spirv
} // end namespace mlir

#endif // MLIR_TARGET_SPIRV_BINARY_UTILS_H_
