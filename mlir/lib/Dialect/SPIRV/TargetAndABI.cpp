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

//===----------------------------------------------------------------------===//
// TargetEnv
//===----------------------------------------------------------------------===//

spirv::TargetEnv::TargetEnv(spirv::TargetEnvAttr targetAttr)
    : targetAttr(targetAttr) {
  for (spirv::Extension ext : targetAttr.getExtensions())
    givenExtensions.insert(ext);

  // Add extensions implied by the current version.
  for (spirv::Extension ext :
       spirv::getImpliedExtensions(targetAttr.getVersion()))
    givenExtensions.insert(ext);

  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    givenCapabilities.insert(cap);

    // Add capabilities implied by the current capability.
    for (spirv::Capability c : spirv::getRecursiveImpliedCapabilities(cap))
      givenCapabilities.insert(c);
  }
}

spirv::Version spirv::TargetEnv::getVersion() {
  return targetAttr.getVersion();
}

bool spirv::TargetEnv::allows(spirv::Capability capability) const {
  return givenCapabilities.count(capability);
}

Optional<spirv::Capability>
spirv::TargetEnv::allows(ArrayRef<spirv::Capability> caps) const {
  auto chosen = llvm::find_if(caps, [this](spirv::Capability cap) {
    return givenCapabilities.count(cap);
  });
  if (chosen != caps.end())
    return *chosen;
  return llvm::None;
}

bool spirv::TargetEnv::allows(spirv::Extension extension) const {
  return givenExtensions.count(extension);
}

Optional<spirv::Extension>
spirv::TargetEnv::allows(ArrayRef<spirv::Extension> exts) const {
  auto chosen = llvm::find_if(exts, [this](spirv::Extension ext) {
    return givenExtensions.count(ext);
  });
  if (chosen != exts.end())
    return *chosen;
  return llvm::None;
}

MLIRContext *spirv::TargetEnv::getContext() const {
  return targetAttr.getContext();
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

StringRef spirv::getInterfaceVarABIAttrName() {
  return "spv.interface_var_abi";
}

spirv::InterfaceVarABIAttr
spirv::getInterfaceVarABIAttr(unsigned descriptorSet, unsigned binding,
                              Optional<spirv::StorageClass> storageClass,
                              MLIRContext *context) {
  return spirv::InterfaceVarABIAttr::get(descriptorSet, binding, storageClass,
                                         context);
}

bool spirv::needsInterfaceVarABIAttrs(spirv::TargetEnvAttr targetAttr) {
  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    if (cap == spirv::Capability::Kernel)
      return false;
    if (cap == spirv::Capability::Shader)
      return true;
  }
  return false;
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

spirv::AddressingModel
spirv::getAddressingModel(spirv::TargetEnvAttr targetAttr) {
  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    // TODO: Physical64 is hard-coded here, but some information should come
    // from TargetEnvAttr to selected between Physical32 and Physical64.
    if (cap == Capability::Kernel)
      return spirv::AddressingModel::Physical64;
  }
  // Logical addressing doesn't need any capabilities so return it as default.
  return spirv::AddressingModel::Logical;
}

FailureOr<spirv::ExecutionModel>
spirv::getExecutionModel(spirv::TargetEnvAttr targetAttr) {
  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    if (cap == spirv::Capability::Kernel)
      return spirv::ExecutionModel::Kernel;
    if (cap == spirv::Capability::Shader)
      return spirv::ExecutionModel::GLCompute;
  }
  return failure();
}

FailureOr<spirv::MemoryModel>
spirv::getMemoryModel(spirv::TargetEnvAttr targetAttr) {
  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    if (cap == spirv::Capability::Addresses)
      return spirv::MemoryModel::OpenCL;
    if (cap == spirv::Capability::Shader)
      return spirv::MemoryModel::GLSL450;
  }
  return failure();
}
