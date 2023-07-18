//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class RewritePatternSet;

namespace arm_sme {
//===----------------------------------------------------------------------===//
// The EnableArmStreaming pass.
//===----------------------------------------------------------------------===//
// Options for Armv9 Streaming SVE mode. By default, streaming-mode is part of
// the function interface (ABI) and the caller manages PSTATE.SM on entry/exit.
// In a locally streaming function PSTATE.SM is kept internal and the callee
// manages it on entry/exit.
enum class ArmStreaming { Default = 0, Locally = 1 };

#define GEN_PASS_DECL
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"

/// Pass to enable Armv9 Streaming SVE mode.
std::unique_ptr<Pass>
createEnableArmStreamingPass(const ArmStreaming mode = ArmStreaming::Default,
                             const bool enableZA = false);

/// Pass that replaces 'arm_sme.get_tile_id' ops with actual tiles.
std::unique_ptr<Pass> createTileAllocationPass();

//===----------------------------------------------------------------------===//
// Type ArmSMETypeConverter pass.
//===----------------------------------------------------------------------===//
class ArmSMETypeConverter : public LLVMTypeConverter {
public:
  ArmSMETypeConverter(MLIRContext *ctx, const LowerToLLVMOptions &options);
};

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"

} // namespace arm_sme
} // namespace mlir

#endif // MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H
