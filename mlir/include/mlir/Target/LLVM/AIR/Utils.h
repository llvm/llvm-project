//===-- Utils.h - MLIR AIR target utils -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares AIR target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_AIR_UTILS_H
#define MLIR_TARGET_LLVM_AIR_UTILS_H

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace mlir {
namespace air {

/// Base class for all AIR serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  SerializeGPUModuleBase(Operation &module, AIRTargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  /// Initializes the LLVM AIR target. Can be called multiple times.
  static void init();

  /// Returns the target attribute.
  AIRTargetAttr getTarget() const;

  /// Returns the gpu module being serialized.
  gpu::GPUModuleOp getGPUModuleOp();

protected:
  /// AIR target attribute.
  AIRTargetAttr target;

  /// GPU compilation target options.
  gpu::TargetOptions targetOptions;
};
} // namespace air
} // namespace mlir

#endif // MLIR_TARGET_LLVM_AIR_UTILS_H
