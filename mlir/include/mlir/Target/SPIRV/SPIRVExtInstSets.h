//===- SPIRVExtInstSets.h - SPIR-V ext inst sets ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares extended instruction set constants used by SPIR-V
// (de)serialization.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_SPIRV_SPIRVEXTINSTSETS_H
#define MLIR_TARGET_SPIRV_SPIRVEXTINSTSETS_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir::spirv {

/// Extension set name for TOSA ops.
constexpr StringLiteral extTosa{"TOSA.001000.1"};

/// Extension set name for non-semantic graph debug info.
constexpr StringLiteral extDebugInfo{"NonSemantic.Graph.DebugInfo.1"};

/// Instruction opcodes in the NonSemantic.Graph.DebugInfo.1 extended
/// instruction set.
enum class GraphDebugInfoExtInst : uint32_t {
  DebugGraph = 1,
  DebugOperation = 2,
  DebugTensor = 3,
};

constexpr bool isValidGraphDebugInfoExtInst(uint32_t opcode) {
  return llvm::is_contained(
      {
          llvm::to_underlying(GraphDebugInfoExtInst::DebugGraph),
          llvm::to_underlying(GraphDebugInfoExtInst::DebugOperation),
          llvm::to_underlying(GraphDebugInfoExtInst::DebugTensor),
      },
      opcode);
}

} // namespace mlir::spirv

#endif // MLIR_TARGET_SPIRV_SPIRVEXTINSTSETS_H
