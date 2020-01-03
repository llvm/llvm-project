//===- SPIRVLowering.cpp - Standard to SPIR-V dialect conversion--===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir {
#include "mlir/Dialect/SPIRV/TargetAndABI.cpp.inc"
}

StringRef mlir::spirv::getInterfaceVarABIAttrName() {
  return "spv.interface_var_abi";
}

mlir::spirv::InterfaceVarABIAttr
mlir::spirv::getInterfaceVarABIAttr(unsigned descriptorSet, unsigned binding,
                                    spirv::StorageClass storageClass,
                                    MLIRContext *context) {
  Type i32Type = IntegerType::get(32, context);
  return mlir::spirv::InterfaceVarABIAttr::get(
      IntegerAttr::get(i32Type, descriptorSet),
      IntegerAttr::get(i32Type, binding),
      IntegerAttr::get(i32Type, static_cast<int64_t>(storageClass)), context);
}

StringRef mlir::spirv::getEntryPointABIAttrName() {
  return "spv.entry_point_abi";
}

mlir::spirv::EntryPointABIAttr
mlir::spirv::getEntryPointABIAttr(ArrayRef<int32_t> localSize,
                                  MLIRContext *context) {
  assert(localSize.size() == 3);
  return mlir::spirv::EntryPointABIAttr::get(
      DenseElementsAttr::get<int32_t>(
          VectorType::get(3, IntegerType::get(32, context)), localSize)
          .cast<DenseIntElementsAttr>(),
      context);
}
