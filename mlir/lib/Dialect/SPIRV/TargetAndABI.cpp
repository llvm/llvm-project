//===- TargetAndABI.cpp - SPIR-V target and ABI utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

StringRef spirv::getInterfaceVarABIAttrName() {
  return "spv.interface_var_abi";
}

spirv::InterfaceVarABIAttr
spirv::getInterfaceVarABIAttr(unsigned descriptorSet, unsigned binding,
                              spirv::StorageClass storageClass,
                              MLIRContext *context) {
  Type i32Type = IntegerType::get(32, context);
  return spirv::InterfaceVarABIAttr::get(
      IntegerAttr::get(i32Type, descriptorSet),
      IntegerAttr::get(i32Type, binding),
      IntegerAttr::get(i32Type, static_cast<int64_t>(storageClass)), context);
}

StringRef spirv::getEntryPointABIAttrName() { return "spv.entry_point_abi"; }

spirv::EntryPointABIAttr
spirv::getEntryPointABIAttr(ArrayRef<int32_t> localSize, MLIRContext *context) {
  assert(localSize.size() == 3);
  return spirv::EntryPointABIAttr::get(
      DenseElementsAttr::get<int32_t>(
          VectorType::get(3, IntegerType::get(32, context)), localSize)
          .cast<DenseIntElementsAttr>(),
      context);
}

spirv::EntryPointABIAttr spirv::lookupEntryPointABI(Operation *op) {
  while (op && !op->hasTrait<OpTrait::FunctionLike>())
    op = op->getParentOp();
  if (!op)
    return {};

  if (auto attr = op->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName()))
    return attr;

  return {};
}

DenseIntElementsAttr spirv::lookupLocalWorkGroupSize(Operation *op) {
  if (auto entryPoint = spirv::lookupEntryPointABI(op))
    return entryPoint.local_size();

  return {};
}

spirv::ResourceLimitsAttr
spirv::getDefaultResourceLimits(MLIRContext *context) {
  auto i32Type = IntegerType::get(32, context);
  auto v3i32Type = VectorType::get(3, i32Type);

  // These numbers are from "Table 46. Required Limits" of the Vulkan spec.
  return spirv::ResourceLimitsAttr ::get(
      IntegerAttr::get(i32Type, 128),
      DenseIntElementsAttr::get<int32_t>(v3i32Type, {128, 128, 64}), context);
}

StringRef spirv::getTargetEnvAttrName() { return "spv.target_env"; }

spirv::TargetEnvAttr spirv::getDefaultTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_0,
                                          {spirv::Capability::Shader},
                                          ArrayRef<Extension>(), context);
  return spirv::TargetEnvAttr::get(triple,
                                   spirv::getDefaultResourceLimits(context));
}

spirv::TargetEnvAttr spirv::lookupTargetEnv(Operation *op) {
  while (op) {
    op = SymbolTable::getNearestSymbolTable(op);
    if (!op)
      break;

    if (auto attr = op->getAttrOfType<spirv::TargetEnvAttr>(
            spirv::getTargetEnvAttrName()))
      return attr;

    op = op->getParentOp();
  }

  return {};
}

spirv::TargetEnvAttr spirv::lookupTargetEnvOrDefault(Operation *op) {
  if (spirv::TargetEnvAttr attr = spirv::lookupTargetEnv(op))
    return attr;

  return getDefaultTargetEnv(op->getContext());
}
