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
// DictionaryDict derived attributes
//===----------------------------------------------------------------------===//

namespace mlir {
#include "mlir/Dialect/SPIRV/TargetAndABI.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute storage classes
//===----------------------------------------------------------------------===//

namespace spirv {
namespace detail {
struct VerCapExtAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Attribute, Attribute, Attribute>;

  VerCapExtAttributeStorage(Attribute version, Attribute capabilities,
                            Attribute extensions)
      : version(version), capabilities(capabilities), extensions(extensions) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == version && std::get<1>(key) == capabilities &&
           std::get<2>(key) == extensions;
  }

  static VerCapExtAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<VerCapExtAttributeStorage>())
        VerCapExtAttributeStorage(std::get<0>(key), std::get<1>(key),
                                  std::get<2>(key));
  }

  Attribute version;
  Attribute capabilities;
  Attribute extensions;
};

struct TargetEnvAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<Attribute, Attribute>;

  TargetEnvAttributeStorage(Attribute triple, Attribute limits)
      : triple(triple), limits(limits) {}

  bool operator==(const KeyTy &key) const {
    return key.first == triple && key.second == limits;
  }

  static TargetEnvAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetEnvAttributeStorage>())
        TargetEnvAttributeStorage(key.first, key.second);
  }

  Attribute triple;
  Attribute limits;
};
} // namespace detail
} // namespace spirv
} // namespace mlir

//===----------------------------------------------------------------------===//
// VerCapExtAttr
//===----------------------------------------------------------------------===//

spirv::VerCapExtAttr spirv::VerCapExtAttr::get(
    spirv::Version version, ArrayRef<spirv::Capability> capabilities,
    ArrayRef<spirv::Extension> extensions, MLIRContext *context) {
  Builder b(context);

  auto versionAttr = b.getI32IntegerAttr(static_cast<uint32_t>(version));

  SmallVector<Attribute, 4> capAttrs;
  capAttrs.reserve(capabilities.size());
  for (spirv::Capability cap : capabilities)
    capAttrs.push_back(b.getI32IntegerAttr(static_cast<uint32_t>(cap)));

  SmallVector<Attribute, 4> extAttrs;
  extAttrs.reserve(extensions.size());
  for (spirv::Extension ext : extensions)
    extAttrs.push_back(b.getStringAttr(spirv::stringifyExtension(ext)));

  return get(versionAttr, b.getArrayAttr(capAttrs), b.getArrayAttr(extAttrs));
}

spirv::VerCapExtAttr spirv::VerCapExtAttr::get(IntegerAttr version,
                                               ArrayAttr capabilities,
                                               ArrayAttr extensions) {
  assert(version && capabilities && extensions);
  MLIRContext *context = version.getContext();
  return Base::get(context, spirv::AttrKind::VerCapExt, version, capabilities,
                   extensions);
}

StringRef spirv::VerCapExtAttr::getKindName() { return "vce"; }

spirv::Version spirv::VerCapExtAttr::getVersion() {
  return static_cast<spirv::Version>(
      getImpl()->version.cast<IntegerAttr>().getValue().getZExtValue());
}

spirv::VerCapExtAttr::ext_iterator::ext_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator,
                            spirv::Extension (*)(Attribute)>(
          it, [](Attribute attr) {
            return *symbolizeExtension(attr.cast<StringAttr>().getValue());
          }) {}

spirv::VerCapExtAttr::ext_range spirv::VerCapExtAttr::getExtensions() {
  auto range = getExtensionsAttr().getValue();
  return {ext_iterator(range.begin()), ext_iterator(range.end())};
}

ArrayAttr spirv::VerCapExtAttr::getExtensionsAttr() {
  return getImpl()->extensions.cast<ArrayAttr>();
}

spirv::VerCapExtAttr::cap_iterator::cap_iterator(ArrayAttr::iterator it)
    : llvm::mapped_iterator<ArrayAttr::iterator,
                            spirv::Capability (*)(Attribute)>(
          it, [](Attribute attr) {
            return *symbolizeCapability(
                attr.cast<IntegerAttr>().getValue().getZExtValue());
          }) {}

spirv::VerCapExtAttr::cap_range spirv::VerCapExtAttr::getCapabilities() {
  auto range = getCapabilitiesAttr().getValue();
  return {cap_iterator(range.begin()), cap_iterator(range.end())};
}

ArrayAttr spirv::VerCapExtAttr::getCapabilitiesAttr() {
  return getImpl()->capabilities.cast<ArrayAttr>();
}

LogicalResult spirv::VerCapExtAttr::verifyConstructionInvariants(
    Location loc, IntegerAttr version, ArrayAttr capabilities,
    ArrayAttr extensions) {
  if (!version.getType().isSignlessInteger(32))
    return emitError(loc, "expected 32-bit integer for version");

  if (!llvm::all_of(capabilities.getValue(), [](Attribute attr) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>())
          if (spirv::symbolizeCapability(intAttr.getValue().getZExtValue()))
            return true;
        return false;
      }))
    return emitError(loc, "unknown capability in capability list");

  if (!llvm::all_of(extensions.getValue(), [](Attribute attr) {
        if (auto strAttr = attr.dyn_cast<StringAttr>())
          if (spirv::symbolizeExtension(strAttr.getValue()))
            return true;
        return false;
      }))
    return emitError(loc, "unknown extension in extension list");

  return success();
}

//===----------------------------------------------------------------------===//
// TargetEnvAttr
//===----------------------------------------------------------------------===//

spirv::TargetEnvAttr spirv::TargetEnvAttr::get(spirv::VerCapExtAttr triple,
                                               DictionaryAttr limits) {
  assert(triple && limits && "expected valid triple and limits");
  MLIRContext *context = triple.getContext();
  return Base::get(context, spirv::AttrKind::TargetEnv, triple, limits);
}

StringRef spirv::TargetEnvAttr::getKindName() { return "target_env"; }

spirv::VerCapExtAttr spirv::TargetEnvAttr::getTripleAttr() {
  return getImpl()->triple.cast<spirv::VerCapExtAttr>();
}

spirv::Version spirv::TargetEnvAttr::getVersion() {
  return getTripleAttr().getVersion();
}

spirv::VerCapExtAttr::ext_range spirv::TargetEnvAttr::getExtensions() {
  return getTripleAttr().getExtensions();
}

ArrayAttr spirv::TargetEnvAttr::getExtensionsAttr() {
  return getTripleAttr().getExtensionsAttr();
}

spirv::VerCapExtAttr::cap_range spirv::TargetEnvAttr::getCapabilities() {
  return getTripleAttr().getCapabilities();
}

ArrayAttr spirv::TargetEnvAttr::getCapabilitiesAttr() {
  return getTripleAttr().getCapabilitiesAttr();
}

spirv::ResourceLimitsAttr spirv::TargetEnvAttr::getResourceLimits() {
  return getImpl()->limits.cast<spirv::ResourceLimitsAttr>();
}

LogicalResult spirv::TargetEnvAttr::verifyConstructionInvariants(
    Location loc, spirv::VerCapExtAttr triple, DictionaryAttr limits) {
  if (!limits.isa<spirv::ResourceLimitsAttr>())
    return emitError(loc, "expected spirv::ResourceLimitsAttr for limits");

  return success();
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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
