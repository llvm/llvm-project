//===- SPIRVModule.h - SPIR-V Module Utilities ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVMODULE_H
#define MLIR_DIALECT_SPIRV_SPIRVMODULE_H

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/OwningOpRefBase.h"

namespace mlir {
namespace spirv {

/// This class acts as an owning reference to a SPIR-V module, and will
/// automatically destroy the held module on destruction if the held module
/// is valid.
class OwningSPIRVModuleRef : public OwningOpRefBase<spirv::ModuleOp> {
public:
  using OwningOpRefBase<spirv::ModuleOp>::OwningOpRefBase;
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVMODULE_H
