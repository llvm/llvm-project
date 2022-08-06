//===- SPIRVAttributes.cpp - SPIR-V attribute definitions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::spirv;

//===----------------------------------------------------------------------===//
// TableGen'erated attribute utility functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace spirv {
#include "mlir/Dialect/SPIRV/IR/SPIRVAttrUtils.inc"
} // namespace spirv

//===----------------------------------------------------------------------===//
// Attribute storage classes
//===----------------------------------------------------------------------===//

namespace spirv {
namespace detail {

struct InterfaceVarABIAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Attribute, Attribute, Attribute>;

  InterfaceVarABIAttributeStorage(Attribute descriptorSet, Attribute binding,
                                  Attribute storageClass)
      : descriptorSet(descriptorSet), binding(binding),
        storageClass(storageClass) {}

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == descriptorSet && std::get<1>(key) == binding &&
           std::get<2>(key) == storageClass;
  }

  static InterfaceVarABIAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<InterfaceVarABIAttributeStorage>())
        InterfaceVarABIAttributeStorage(std::get<0>(key), std::get<1>(key),
                                        std::get<2>(key));
  }

  Attribute descriptorSet;
  Attribute binding;
  Attribute storageClass;
};

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
  using KeyTy = std::tuple<Attribute, Vendor, DeviceType, uint32_t, Attribute>;

  TargetEnvAttributeStorage(Attribute triple, Vendor vendorID,
                            DeviceType deviceType, uint32_t deviceID,
                            Attribute limits)
      : triple(triple), limits(limits), vendorID(vendorID),
        deviceType(deviceType), deviceID(deviceID) {}

  bool operator==(const KeyTy &key) const {
    return key ==
           std::make_tuple(triple, vendorID, deviceType, deviceID, limits);
  }

  static TargetEnvAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetEnvAttributeStorage>())
        TargetEnvAttributeStorage(std::get<0>(key), std::get<1>(key),
                                  std::get<2>(key), std::get<3>(key),
                                  std::get<4>(key));
  }

  Attribute triple;
  Attribute limits;
  Vendor vendorID;
  DeviceType deviceType;
  uint32_t deviceID;
};
} // namespace detail
} // namespace spirv
} // namespace mlir

//===----------------------------------------------------------------------===//
// InterfaceVarABIAttr
//===----------------------------------------------------------------------===//

spirv::InterfaceVarABIAttr
spirv::InterfaceVarABIAttr::get(uint32_t descriptorSet, uint32_t binding,
                                Optional<spirv::StorageClass> storageClass,
                                MLIRContext *context) {
  Builder b(context);
  auto descriptorSetAttr = b.getI32IntegerAttr(descriptorSet);
  auto bindingAttr = b.getI32IntegerAttr(binding);
  auto storageClassAttr =
      storageClass ? b.getI32IntegerAttr(static_cast<uint32_t>(*storageClass))
                   : IntegerAttr();
  return get(descriptorSetAttr, bindingAttr, storageClassAttr);
}

spirv::InterfaceVarABIAttr
spirv::InterfaceVarABIAttr::get(IntegerAttr descriptorSet, IntegerAttr binding,
                                IntegerAttr storageClass) {
  assert(descriptorSet && binding);
  MLIRContext *context = descriptorSet.getContext();
  return Base::get(context, descriptorSet, binding, storageClass);
}

StringRef spirv::InterfaceVarABIAttr::getKindName() {
  return "interface_var_abi";
}

uint32_t spirv::InterfaceVarABIAttr::getBinding() {
  return getImpl()->binding.cast<IntegerAttr>().getInt();
}

uint32_t spirv::InterfaceVarABIAttr::getDescriptorSet() {
  return getImpl()->descriptorSet.cast<IntegerAttr>().getInt();
}

Optional<spirv::StorageClass> spirv::InterfaceVarABIAttr::getStorageClass() {
  if (getImpl()->storageClass)
    return static_cast<spirv::StorageClass>(
        getImpl()->storageClass.cast<IntegerAttr>().getValue().getZExtValue());
  return llvm::None;
}

LogicalResult spirv::InterfaceVarABIAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, IntegerAttr descriptorSet,
    IntegerAttr binding, IntegerAttr storageClass) {
  if (!descriptorSet.getType().isSignlessInteger(32))
    return emitError() << "expected 32-bit integer for descriptor set";

  if (!binding.getType().isSignlessInteger(32))
    return emitError() << "expected 32-bit integer for binding";

  if (storageClass) {
    if (auto storageClassAttr = storageClass.cast<IntegerAttr>()) {
      auto storageClassValue =
          spirv::symbolizeStorageClass(storageClassAttr.getInt());
      if (!storageClassValue)
        return emitError() << "unknown storage class";
    } else {
      return emitError() << "expected valid storage class";
    }
  }

  return success();
}

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
  return Base::get(context, version, capabilities, extensions);
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

LogicalResult
spirv::VerCapExtAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             IntegerAttr version, ArrayAttr capabilities,
                             ArrayAttr extensions) {
  if (!version.getType().isSignlessInteger(32))
    return emitError() << "expected 32-bit integer for version";

  if (!llvm::all_of(capabilities.getValue(), [](Attribute attr) {
        if (auto intAttr = attr.dyn_cast<IntegerAttr>())
          if (spirv::symbolizeCapability(intAttr.getValue().getZExtValue()))
            return true;
        return false;
      }))
    return emitError() << "unknown capability in capability list";

  if (!llvm::all_of(extensions.getValue(), [](Attribute attr) {
        if (auto strAttr = attr.dyn_cast<StringAttr>())
          if (spirv::symbolizeExtension(strAttr.getValue()))
            return true;
        return false;
      }))
    return emitError() << "unknown extension in extension list";

  return success();
}

//===----------------------------------------------------------------------===//
// TargetEnvAttr
//===----------------------------------------------------------------------===//

spirv::TargetEnvAttr spirv::TargetEnvAttr::get(spirv::VerCapExtAttr triple,
                                               Vendor vendorID,
                                               DeviceType deviceType,
                                               uint32_t deviceID,
                                               ResourceLimitsAttr limits) {
  assert(triple && limits && "expected valid triple and limits");
  MLIRContext *context = triple.getContext();
  return Base::get(context, triple, vendorID, deviceType, deviceID, limits);
}

StringRef spirv::TargetEnvAttr::getKindName() { return "target_env"; }

spirv::VerCapExtAttr spirv::TargetEnvAttr::getTripleAttr() const {
  return getImpl()->triple.cast<spirv::VerCapExtAttr>();
}

spirv::Version spirv::TargetEnvAttr::getVersion() const {
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

spirv::Vendor spirv::TargetEnvAttr::getVendorID() const {
  return getImpl()->vendorID;
}

spirv::DeviceType spirv::TargetEnvAttr::getDeviceType() const {
  return getImpl()->deviceType;
}

uint32_t spirv::TargetEnvAttr::getDeviceID() const {
  return getImpl()->deviceID;
}

spirv::ResourceLimitsAttr spirv::TargetEnvAttr::getResourceLimits() const {
  return getImpl()->limits.cast<spirv::ResourceLimitsAttr>();
}

//===----------------------------------------------------------------------===//
// ODS Generated Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Parsing
//===----------------------------------------------------------------------===//

/// Parses a comma-separated list of keywords, invokes `processKeyword` on each
/// of the parsed keyword, and returns failure if any error occurs.
static ParseResult
parseKeywordList(DialectAsmParser &parser,
                 function_ref<LogicalResult(SMLoc, StringRef)> processKeyword) {
  if (parser.parseLSquare())
    return failure();

  // Special case for empty list.
  if (succeeded(parser.parseOptionalRSquare()))
    return success();

  // Keep parsing the keyword and an optional comma following it. If the comma
  // is successfully parsed, then we have more keywords to parse.
  if (failed(parser.parseCommaSeparatedList([&]() {
        auto loc = parser.getCurrentLocation();
        StringRef keyword;
        if (parser.parseKeyword(&keyword) ||
            failed(processKeyword(loc, keyword)))
          return failure();
        return success();
      })))
    return failure();
  return parser.parseRSquare();
}

/// Parses a spirv::InterfaceVarABIAttr.
static Attribute parseInterfaceVarABIAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  Builder &builder = parser.getBuilder();

  if (parser.parseLParen())
    return {};

  IntegerAttr descriptorSetAttr;
  {
    auto loc = parser.getCurrentLocation();
    uint32_t descriptorSet = 0;
    auto descriptorSetParseResult = parser.parseOptionalInteger(descriptorSet);

    if (!descriptorSetParseResult.has_value() ||
        failed(*descriptorSetParseResult)) {
      parser.emitError(loc, "missing descriptor set");
      return {};
    }
    descriptorSetAttr = builder.getI32IntegerAttr(descriptorSet);
  }

  if (parser.parseComma())
    return {};

  IntegerAttr bindingAttr;
  {
    auto loc = parser.getCurrentLocation();
    uint32_t binding = 0;
    auto bindingParseResult = parser.parseOptionalInteger(binding);

    if (!bindingParseResult.has_value() || failed(*bindingParseResult)) {
      parser.emitError(loc, "missing binding");
      return {};
    }
    bindingAttr = builder.getI32IntegerAttr(binding);
  }

  if (parser.parseRParen())
    return {};

  IntegerAttr storageClassAttr;
  {
    if (succeeded(parser.parseOptionalComma())) {
      auto loc = parser.getCurrentLocation();
      StringRef storageClass;
      if (parser.parseKeyword(&storageClass))
        return {};

      if (auto storageClassSymbol =
              spirv::symbolizeStorageClass(storageClass)) {
        storageClassAttr = builder.getI32IntegerAttr(
            static_cast<uint32_t>(*storageClassSymbol));
      } else {
        parser.emitError(loc, "unknown storage class: ") << storageClass;
        return {};
      }
    }
  }

  if (parser.parseGreater())
    return {};

  return spirv::InterfaceVarABIAttr::get(descriptorSetAttr, bindingAttr,
                                         storageClassAttr);
}

static Attribute parseVerCapExtAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  Builder &builder = parser.getBuilder();

  IntegerAttr versionAttr;
  {
    auto loc = parser.getCurrentLocation();
    StringRef version;
    if (parser.parseKeyword(&version) || parser.parseComma())
      return {};

    if (auto versionSymbol = spirv::symbolizeVersion(version)) {
      versionAttr =
          builder.getI32IntegerAttr(static_cast<uint32_t>(*versionSymbol));
    } else {
      parser.emitError(loc, "unknown version: ") << version;
      return {};
    }
  }

  ArrayAttr capabilitiesAttr;
  {
    SmallVector<Attribute, 4> capabilities;
    SMLoc errorloc;
    StringRef errorKeyword;

    auto processCapability = [&](SMLoc loc, StringRef capability) {
      if (auto capSymbol = spirv::symbolizeCapability(capability)) {
        capabilities.push_back(
            builder.getI32IntegerAttr(static_cast<uint32_t>(*capSymbol)));
        return success();
      }
      return errorloc = loc, errorKeyword = capability, failure();
    };
    if (parseKeywordList(parser, processCapability) || parser.parseComma()) {
      if (!errorKeyword.empty())
        parser.emitError(errorloc, "unknown capability: ") << errorKeyword;
      return {};
    }

    capabilitiesAttr = builder.getArrayAttr(capabilities);
  }

  ArrayAttr extensionsAttr;
  {
    SmallVector<Attribute, 1> extensions;
    SMLoc errorloc;
    StringRef errorKeyword;

    auto processExtension = [&](SMLoc loc, StringRef extension) {
      if (spirv::symbolizeExtension(extension)) {
        extensions.push_back(builder.getStringAttr(extension));
        return success();
      }
      return errorloc = loc, errorKeyword = extension, failure();
    };
    if (parseKeywordList(parser, processExtension)) {
      if (!errorKeyword.empty())
        parser.emitError(errorloc, "unknown extension: ") << errorKeyword;
      return {};
    }

    extensionsAttr = builder.getArrayAttr(extensions);
  }

  if (parser.parseGreater())
    return {};

  return spirv::VerCapExtAttr::get(versionAttr, capabilitiesAttr,
                                   extensionsAttr);
}

/// Parses a spirv::TargetEnvAttr.
static Attribute parseTargetEnvAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  spirv::VerCapExtAttr tripleAttr;
  if (parser.parseAttribute(tripleAttr) || parser.parseComma())
    return {};

  // Parse [vendor[:device-type[:device-id]]]
  Vendor vendorID = Vendor::Unknown;
  DeviceType deviceType = DeviceType::Unknown;
  uint32_t deviceID = spirv::TargetEnvAttr::kUnknownDeviceID;
  {
    auto loc = parser.getCurrentLocation();
    StringRef vendorStr;
    if (succeeded(parser.parseOptionalKeyword(&vendorStr))) {
      if (auto vendorSymbol = spirv::symbolizeVendor(vendorStr)) {
        vendorID = *vendorSymbol;
      } else {
        parser.emitError(loc, "unknown vendor: ") << vendorStr;
      }

      if (succeeded(parser.parseOptionalColon())) {
        loc = parser.getCurrentLocation();
        StringRef deviceTypeStr;
        if (parser.parseKeyword(&deviceTypeStr))
          return {};
        if (auto deviceTypeSymbol = spirv::symbolizeDeviceType(deviceTypeStr)) {
          deviceType = *deviceTypeSymbol;
        } else {
          parser.emitError(loc, "unknown device type: ") << deviceTypeStr;
        }

        if (succeeded(parser.parseOptionalColon())) {
          loc = parser.getCurrentLocation();
          if (parser.parseInteger(deviceID))
            return {};
        }
      }
      if (parser.parseComma())
        return {};
    }
  }

  ResourceLimitsAttr limitsAttr;
  if (parser.parseAttribute(limitsAttr) || parser.parseGreater())
    return {};

  return spirv::TargetEnvAttr::get(tripleAttr, vendorID, deviceType, deviceID,
                                   limitsAttr);
}

Attribute SPIRVDialect::parseAttribute(DialectAsmParser &parser,
                                       Type type) const {
  // SPIR-V attributes are dictionaries so they do not have type.
  if (type) {
    parser.emitError(parser.getNameLoc(), "unexpected type");
    return {};
  }

  // Parse the kind keyword first.
  StringRef attrKind;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &attrKind, type, attr);
  if (result.has_value())
    return attr;

  if (attrKind == spirv::TargetEnvAttr::getKindName())
    return parseTargetEnvAttr(parser);
  if (attrKind == spirv::VerCapExtAttr::getKindName())
    return parseVerCapExtAttr(parser);
  if (attrKind == spirv::InterfaceVarABIAttr::getKindName())
    return parseInterfaceVarABIAttr(parser);

  parser.emitError(parser.getNameLoc(), "unknown SPIR-V attribute kind: ")
      << attrKind;
  return {};
}

//===----------------------------------------------------------------------===//
// Attribute Printing
//===----------------------------------------------------------------------===//

static void print(spirv::VerCapExtAttr triple, DialectAsmPrinter &printer) {
  auto &os = printer.getStream();
  printer << spirv::VerCapExtAttr::getKindName() << "<"
          << spirv::stringifyVersion(triple.getVersion()) << ", [";
  llvm::interleaveComma(
      triple.getCapabilities(), os,
      [&](spirv::Capability cap) { os << spirv::stringifyCapability(cap); });
  printer << "], [";
  llvm::interleaveComma(triple.getExtensionsAttr(), os, [&](Attribute attr) {
    os << attr.cast<StringAttr>().getValue();
  });
  printer << "]>";
}

static void print(spirv::TargetEnvAttr targetEnv, DialectAsmPrinter &printer) {
  printer << spirv::TargetEnvAttr::getKindName() << "<#spv.";
  print(targetEnv.getTripleAttr(), printer);
  spirv::Vendor vendorID = targetEnv.getVendorID();
  spirv::DeviceType deviceType = targetEnv.getDeviceType();
  uint32_t deviceID = targetEnv.getDeviceID();
  if (vendorID != spirv::Vendor::Unknown) {
    printer << ", " << spirv::stringifyVendor(vendorID);
    if (deviceType != spirv::DeviceType::Unknown) {
      printer << ":" << spirv::stringifyDeviceType(deviceType);
      if (deviceID != spirv::TargetEnvAttr::kUnknownDeviceID)
        printer << ":" << deviceID;
    }
  }
  printer << ", " << targetEnv.getResourceLimits() << ">";
}

static void print(spirv::InterfaceVarABIAttr interfaceVarABIAttr,
                  DialectAsmPrinter &printer) {
  printer << spirv::InterfaceVarABIAttr::getKindName() << "<("
          << interfaceVarABIAttr.getDescriptorSet() << ", "
          << interfaceVarABIAttr.getBinding() << ")";
  auto storageClass = interfaceVarABIAttr.getStorageClass();
  if (storageClass)
    printer << ", " << spirv::stringifyStorageClass(*storageClass);
  printer << ">";
}

void SPIRVDialect::printAttribute(Attribute attr,
                                  DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;

  if (auto targetEnv = attr.dyn_cast<TargetEnvAttr>())
    print(targetEnv, printer);
  else if (auto vceAttr = attr.dyn_cast<VerCapExtAttr>())
    print(vceAttr, printer);
  else if (auto interfaceVarABIAttr = attr.dyn_cast<InterfaceVarABIAttr>())
    print(interfaceVarABIAttr, printer);
  else
    llvm_unreachable("unhandled SPIR-V attribute kind");
}

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

void spirv::SPIRVDialect::registerAttributes() {
  addAttributes<InterfaceVarABIAttr, TargetEnvAttr, VerCapExtAttr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.cpp.inc"
      >();
}
