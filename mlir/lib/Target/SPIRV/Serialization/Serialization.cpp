//===- Serialization.cpp - MLIR SPIR-V Serialization ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MLIR SPIR-V module to SPIR-V binary serialization entry
// point.
//
//===----------------------------------------------------------------------===//

#include "Serializer.h"

#include "mlir/Target/SPIRV/Serialization.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "spirv-serialization"

namespace mlir {
LogicalResult spirv::serialize(spirv::ModuleOp module,
                               SmallVectorImpl<uint32_t> &binary,
                               const SerializationOptions &options) {
  if (!module.vce_triple().hasValue())
    return module.emitError(
        "module must have 'vce_triple' attribute to be serializeable");

  Serializer serializer(module, options);

  if (failed(serializer.serialize()))
    return failure();

  LLVM_DEBUG(serializer.printValueIDMap(llvm::dbgs()));

  serializer.collect(binary);
  return success();
}
} // namespace mlir
