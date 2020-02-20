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

using namespace mlir;

namespace mlir {
#include "mlir/Dialect/SPIRV/TargetAndABI.cpp.inc"

namespace spirv {
namespace detail {
struct TargetEnvAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Attribute, Attribute, Attribute, Attribute>;

  TargetEnvAttributeStorage(Attribute version, Attribute extensions,
                            Attribute capabilities, Attribute limits)
      : version(version), extensions(extensions), capabilities(capabilities),
        limits(limits) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == version && std::get<1>(key) == extensions &&
           std::get<2>(key) == capabilities && std::get<3>(key) == limits;
  }

  static TargetEnvAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetEnvAttributeStorage>())
        TargetEnvAttributeStorage(std::get<0>(key), std::get<1>(key),
                                  std::get<2>(key), std::get<3>(key));
  }

  Attribute version;
  Attribute extensions;
  Attribute capabilities;
  Attribute limits;
};
} // namespace detail
} // namespace spirv
} // namespace mlir

spirv::TargetEnvAttr spirv::TargetEnvAttr::get(IntegerAttr version,
                                               ArrayAttr extensions,
                                               ArrayAttr capabilities,
                                               DictionaryAttr limits) {
  assert(version && extensions && capabilities && limits);
  MLIRContext *context = version.getContext();
  return Base::get(context, spirv::AttrKind::TargetEnv, version, extensions,
                   capabilities, limits);
}

StringRef spirv::TargetEnvAttr::getKindName() { return "target_env"; }

spirv::Version spirv::TargetEnvAttr::getVersion() {
  return static_cast<spirv::Version>(
      getImpl()->version.cast<IntegerAttr>().getValue().getZExtValue());
}

spirv::TargetEnvAttr::ext_iterator::ext_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator,
                            spirv::Extension (*)(Attribute)>(
          it, [](Attribute attr) {
            return *symbolizeExtension(attr.cast<StringAttr>().getValue());
          }) {}

spirv::TargetEnvAttr::ext_range spirv::TargetEnvAttr::getExtensions() {
  auto range = getExtensionsAttr().getValue();
  return {ext_iterator(range.begin()), ext_iterator(range.end())};
}

ArrayAttr spirv::TargetEnvAttr::getExtensionsAttr() {
  return getImpl()->extensions.cast<ArrayAttr>();
}

spirv::TargetEnvAttr::cap_iterator::cap_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator,
                            spirv::Capability (*)(Attribute)>(
          it, [](Attribute attr) {
            return *symbolizeCapability(
                attr.cast<IntegerAttr>().getValue().getZExtValue());
          }) {}

spirv::TargetEnvAttr::cap_range spirv::TargetEnvAttr::getCapabilities() {
  auto range = getCapabilitiesAttr().getValue();
  return {cap_iterator(range.begin()), cap_iterator(range.end())};
}

ArrayAttr spirv::TargetEnvAttr::getCapabilitiesAttr() {
  return getImpl()->capabilities.cast<ArrayAttr>();
}

DictionaryAttr spirv::TargetEnvAttr::getResourceLimits() {
  return getImpl()->limits.cast<DictionaryAttr>();
}

LogicalResult spirv::TargetEnvAttr::verifyConstructionInvariants(
    Location loc, IntegerAttr version, ArrayAttr extensions,
    ArrayAttr capabilities, DictionaryAttr limits) {
  if (!version.getType().isInteger(32))
    return emitError(loc, "expected 32-bit integer for version");

  if (!llvm::all_of(extensions.getValue(), [](Attribute attr) {
        if (auto strAttr = attr.dyn_cast<StringAttr>())
          if (spirv::symbolizeExtension(strAttr.getValue()))
            return true;
        return false;
      }))
    return emitError(loc, "unknown extension in extension list");

  if (!llvm::all_of(capabilities.getValue(), [](Attribute attr) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>())
          if (spirv::symbolizeCapability(intAttr.getValue().getZExtValue()))
            return true;
        return false;
      }))
    return emitError(loc, "unknown capability in capability list");

  if (!limits.isa<spirv::ResourceLimitsAttr>())
    return emitError(loc, "expected spirv::ResourceLimitsAttr for limits");

  return success();
}

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
  Builder builder(context);
  return spirv::TargetEnvAttr::get(
      builder.getI32IntegerAttr(static_cast<uint32_t>(spirv::Version::V_1_0)),
      builder.getI32ArrayAttr({}),
      builder.getI32ArrayAttr(
          {static_cast<uint32_t>(spirv::Capability::Shader)}),
      spirv::getDefaultResourceLimits(context));
}

spirv::TargetEnvAttr spirv::lookupTargetEnvOrDefault(Operation *op) {
  if (auto attr = op->getAttrOfType<spirv::TargetEnvAttr>(
          spirv::getTargetEnvAttrName()))
    return attr;
  return getDefaultTargetEnv(op->getContext());
}
