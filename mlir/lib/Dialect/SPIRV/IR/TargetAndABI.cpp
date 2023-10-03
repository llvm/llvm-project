//===- TargetAndABI.cpp - SPIR-V target and ABI utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <optional>

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

std::optional<spirv::Capability>
spirv::TargetEnv::allows(ArrayRef<spirv::Capability> caps) const {
  const auto *chosen = llvm::find_if(caps, [this](spirv::Capability cap) {
    return givenCapabilities.count(cap);
  });
  if (chosen != caps.end())
    return *chosen;
  return std::nullopt;
}

bool spirv::TargetEnv::allows(spirv::Extension extension) const {
  return givenExtensions.count(extension);
}

std::optional<spirv::Extension>
spirv::TargetEnv::allows(ArrayRef<spirv::Extension> exts) const {
  const auto *chosen = llvm::find_if(exts, [this](spirv::Extension ext) {
    return givenExtensions.count(ext);
  });
  if (chosen != exts.end())
    return *chosen;
  return std::nullopt;
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
  return "spirv.interface_var_abi";
}

spirv::InterfaceVarABIAttr
spirv::getInterfaceVarABIAttr(unsigned descriptorSet, unsigned binding,
                              std::optional<spirv::StorageClass> storageClass,
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

StringRef spirv::getEntryPointABIAttrName() { return "spirv.entry_point_abi"; }

spirv::EntryPointABIAttr
spirv::getEntryPointABIAttr(MLIRContext *context,
                            ArrayRef<int32_t> workgroupSize,
                            std::optional<int> subgroupSize) {
  DenseI32ArrayAttr workgroupSizeAttr;
  if (!workgroupSize.empty()) {
    assert(workgroupSize.size() == 3);
    workgroupSizeAttr = DenseI32ArrayAttr::get(context, workgroupSize);
  }
  return spirv::EntryPointABIAttr::get(context, workgroupSizeAttr,
                                       subgroupSize);
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

DenseI32ArrayAttr spirv::lookupLocalWorkGroupSize(Operation *op) {
  if (auto entryPoint = spirv::lookupEntryPointABI(op))
    return entryPoint.getWorkgroupSize();

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
      /*min_subgroup_size=*/std::nullopt,
      /*max_subgroup_size=*/std::nullopt,
      /*cooperative_matrix_properties_khr=*/ArrayAttr{},
      /*cooperative_matrix_properties_nv=*/ArrayAttr{});
}

StringRef spirv::getTargetEnvAttrName() { return "spirv.target_env"; }

spirv::TargetEnvAttr spirv::getDefaultTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_0,
                                          {spirv::Capability::Shader},
                                          ArrayRef<Extension>(), context);
  return spirv::TargetEnvAttr::get(
      triple, spirv::getDefaultResourceLimits(context),
      spirv::ClientAPI::Unknown, spirv::Vendor::Unknown,
      spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
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
spirv::getAddressingModel(spirv::TargetEnvAttr targetAttr,
                          bool use64bitAddress) {
  for (spirv::Capability cap : targetAttr.getCapabilities()) {
    if (cap == Capability::Kernel)
      return use64bitAddress ? spirv::AddressingModel::Physical64
                             : spirv::AddressingModel::Physical32;
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
    if (cap == spirv::Capability::Kernel)
      return spirv::MemoryModel::OpenCL;
    if (cap == spirv::Capability::Shader)
      return spirv::MemoryModel::GLSL450;
  }
  return failure();
}
