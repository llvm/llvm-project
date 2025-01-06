//===- SPIRVBinaryUtils.cpp - MLIR SPIR-V Binary Module Utilities ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for SPIR-V binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "llvm/Config/llvm-config.h" // for LLVM_VERSION_MAJOR

using namespace mlir;

void spirv::appendModuleHeader(SmallVectorImpl<uint32_t> &header,
                               spirv::Version version, uint32_t idBound) {
  uint32_t majorVersion = 1;
  uint32_t minorVersion = 0;
  switch (version) {
#define MIN_VERSION_CASE(v)                                                    \
  case spirv::Version::V_1_##v:                                                \
    minorVersion = v;                                                          \
    break

    MIN_VERSION_CASE(0);
    MIN_VERSION_CASE(1);
    MIN_VERSION_CASE(2);
    MIN_VERSION_CASE(3);
    MIN_VERSION_CASE(4);
    MIN_VERSION_CASE(5);
    MIN_VERSION_CASE(6);
#undef MIN_VERSION_CASE
  }

  // See "2.3. Physical Layout of a SPIR-V Module and Instruction" in the SPIR-V
  // spec for the definition of the binary module header.
  //
  // The first five words of a SPIR-V module must be:
  // +-------------------------------------------------------------------------+
  // | Magic number                                                            |
  // +-------------------------------------------------------------------------+
  // | Version number (bytes: 0 | major number | minor number | 0)             |
  // +-------------------------------------------------------------------------+
  // | Generator magic number                                                  |
  // +-------------------------------------------------------------------------+
  // | Bound (all result <id>s in the module guaranteed to be less than it)    |
  // +-------------------------------------------------------------------------+
  // | 0 (reserved for instruction schema)                                     |
  // +-------------------------------------------------------------------------+
  header.push_back(spirv::kMagicNumber);
  header.push_back((majorVersion << 16) | (minorVersion << 8));
  header.push_back((kGeneratorNumber << 16) | LLVM_VERSION_MAJOR);
  header.push_back(idBound); // <id> bound
  header.push_back(0);       // Schema (reserved word)
}

/// Returns the word-count-prefixed opcode for an SPIR-V instruction.
uint32_t spirv::getPrefixedOpcode(uint32_t wordCount, spirv::Opcode opcode) {
  assert(((wordCount >> 16) == 0) && "word count out of range!");
  return (wordCount << 16) | static_cast<uint32_t>(opcode);
}

void spirv::encodeStringLiteralInto(SmallVectorImpl<uint32_t> &binary,
                                    StringRef literal) {
  // We need to encode the literal and the null termination.
  auto encodingSize = literal.size() / 4 + 1;
  auto bufferStartSize = binary.size();
  binary.resize(bufferStartSize + encodingSize, 0);
  std::memcpy(binary.data() + bufferStartSize, literal.data(), literal.size());
}
