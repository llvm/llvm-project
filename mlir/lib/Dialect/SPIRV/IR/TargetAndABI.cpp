//===- TargetAndABI.cpp - SPIR-V target and ABI utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
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

spirv::Version spirv::TargetEnv::getVersion() const {
  return targetAttr.getVersion();
}

bool spirv::TargetEnv::allows(spirv::Capability capability) const {
  return givenCapabilities.count(capability);
}

Optional<spirv::Capability>
spirv::TargetEnv::allows(ArrayRef<spirv::Capability> caps) const {
  const auto *chosen = llvm::find_if(caps, [this](spirv::Capability cap) {
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
  const auto *chosen = llvm::find_if(exts, [this](spirv::Extension ext) {
    return givenExtensions.count(ext);
  });
  if (chosen != exts.end())
    return *chosen;
  return llvm::None;
}

spirv::Vendor spirv::TargetEnv::getVendorID() const {
  return targetAttr.getVendorID();
}

spirv::DeviceType spirv::TargetEnv::getDeviceType() const {
  return targetAttr.getDeviceType();
}

uint32_t spirv::TargetEnv::getDeviceID() const {
  return targetAttr.getDeviceID();
}

spirv::ResourceLimitsAttr spirv::TargetEnv::getResourceLimits() const {
  return targetAttr.getResourceLimits();
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
  if (localSize.empty())
    return spirv::EntryPointABIAttr::get(context, nullptr);

  assert(localSize.size() == 3);
  return spirv::EntryPointABIAttr::get(
      context, DenseElementsAttr::get<int32_t>(
                   VectorType::get(3, IntegerType::get(context, 32)), localSize)
                   .cast<DenseIntElementsAttr>());
}

spirv::EntryPointABIAttr spirv::lookupEntryPointABI(Operation *op) {
  while (op && !isa<FunctionOpInterface>(op))
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
    return entryPoint.getLocal_size();

  return {};
}

spirv::ResourceLimitsAttr
spirv::getDefaultResourceLimits(MLIRContext *context) {
  // All the fields have default values. Here we just provide a nicer way to
  // construct a default resource limit attribute.
  Builder b(context);
  return spirv::ResourceLimitsAttr::get(
      context,
      /*max_compute_shared_memory_size=*/16384,
      /*max_compute_workgroup_invocations=*/128,
      /*max_compute_workgroup_size=*/b.getI32ArrayAttr({128, 128, 64}),
      /*subgroup_size=*/32,
      /*cooperative_matrix_properties_nv=*/ArrayAttr());
}

StringRef spirv::getTargetEnvAttrName() { return "spv.target_env"; }

spirv::TargetEnvAttr spirv::getDefaultTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_0,
                                          {spirv::Capability::Shader},
                                          ArrayRef<Extension>(), context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
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
    // TODO PhysicalStorageBuffer64 is hard-coded here, but some information
    // should come from TargetEnvAttr to select between PhysicalStorageBuffer64
    // and PhysicalStorageBuffer64EXT
    if (cap == Capability::PhysicalStorageBufferAddresses)
      return spirv::AddressingModel::PhysicalStorageBuffer64;
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
