//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tensor;

//===----------------------------------------------------------------------===//
// TableGen'd Attributes Methods
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tensor/IR/TensorAttrDefs.cpp.inc"

// Dictionary keys.
static constexpr StringRef getSparseDimLevelTypeAttrName() {
  return "sparseDimLevelType";
}
static constexpr StringRef getSparseDimOrderingAttrName() {
  return "sparseDimOrdering";
}
static constexpr StringRef getSparsePointerBitWidthAttrName() {
  return "sparsePointerBitWidth";
}
static constexpr StringRef getSparseIndexBitWidthAttrName() {
  return "sparseIndexBitWidth";
}

// Dictionary values.
static constexpr StringRef getDenseDimLevelTypeVal() { return "dense"; }
static constexpr StringRef getCompressedDimLevelTypeVal() {
  return "compressed";
}
static constexpr StringRef getSingletonDimLevelTypeVal() { return "singleton"; }

Attribute SparseTensorEncodingAttr::parse(MLIRContext *context,
                                          DialectAsmParser &parser, Type type) {
  if (failed(parser.parseLess()))
    return {};
  DictionaryAttr dict;
  if (failed(parser.parseAttribute(dict)))
    return {};
  if (failed(parser.parseGreater()))
    return {};
  return SparseTensorEncodingAttr::get(context, dict);
}

void SparseTensorEncodingAttr::print(DialectAsmPrinter &printer) const {
  printer << "sparse<" << getDict() << ">";
}

LogicalResult SparseTensorEncodingAttr::verifyEncoding(
    llvm::ArrayRef<int64_t> shape, Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  unsigned size = shape.size();
  for (const NamedAttribute &attr : getDict()) {
    if (attr.first == getSparseDimLevelTypeAttrName()) {
      // Dimension level type verification.
      auto arrayAttr = attr.second.dyn_cast<ArrayAttr>();
      if (!arrayAttr || size != static_cast<int64_t>(arrayAttr.size()))
        return emitError() << "expected an array of size " << size
                           << " for dimension level types";
      for (unsigned i = 0; i < size; i++) {
        auto strAttr = arrayAttr[i].dyn_cast<StringAttr>();
        if (!strAttr)
          return emitError()
                 << "expected string value in dimension level types";
        auto strVal = strAttr.getValue();
        if (strVal != getDenseDimLevelTypeVal() &&
            strVal != getCompressedDimLevelTypeVal() &&
            strVal != getSingletonDimLevelTypeVal())
          return emitError() << "unexpected dimension level type: " << strAttr;
      }
    } else if (attr.first == getSparseDimOrderingAttrName()) {
      // Dimension order verification.
      auto affineAttr = attr.second.dyn_cast<AffineMapAttr>();
      if (!affineAttr)
        return emitError() << "expected an affine map for dimension ordering";
      AffineMap map = affineAttr.getValue();
      if (size != map.getNumResults() || !map.isPermutation())
        return emitError() << "expected a permutation affine map of size "
                           << size << " for dimension ordering";
    } else if (attr.first == getSparsePointerBitWidthAttrName() ||
               attr.first == getSparseIndexBitWidthAttrName()) {
      // Pointer or index bitwidth verification.
      auto intAttr = attr.second.dyn_cast<IntegerAttr>();
      if (!intAttr)
        return emitError() << "expected an integral bitwidth";
      switch (intAttr.getInt()) {
      case 0:
      case 8:
      case 16:
      case 32:
      case 64:
        continue;
      default:
        return emitError() << "unexpected bitwidth: " << intAttr.getInt();
      }
    } else {
      return emitError() << "unexpected key: " << attr.first.str();
    }
  }
  return success();
}

SparseTensorEncodingAttr::DimLevelType
SparseTensorEncodingAttr::getDimLevelType(unsigned dim) const {
  if (auto value = getDict().get(getSparseDimLevelTypeAttrName())) {
    auto strVal =
        value.dyn_cast<ArrayAttr>()[dim].cast<StringAttr>().getValue();
    if (strVal == getCompressedDimLevelTypeVal())
      return DimLevelType::Compressed;
    if (strVal == getSingletonDimLevelTypeVal())
      return DimLevelType::Singleton;
  }
  return DimLevelType::Dense;
}

AffineMap SparseTensorEncodingAttr::getDimOrdering() const {
  if (auto value = getDict().get(getSparseDimOrderingAttrName()))
    return value.cast<AffineMapAttr>().getValue();
  return {};
}

unsigned SparseTensorEncodingAttr::getPointerBitWidth() const {
  if (auto value = getDict().get(getSparsePointerBitWidthAttrName()))
    return value.cast<IntegerAttr>().getInt();
  return 0;
}

unsigned SparseTensorEncodingAttr::getIndexBitWidth() const {
  if (auto value = getDict().get(getSparseIndexBitWidthAttrName()))
    return value.cast<IntegerAttr>().getInt();
  return 0;
}

//===----------------------------------------------------------------------===//
// TensorDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TensorInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TensorDialect Methods
//===----------------------------------------------------------------------===//

void TensorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Tensor/IR/TensorAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tensor/IR/TensorOps.cpp.inc"
      >();
  addInterfaces<TensorInlinerInterface>();
}

Attribute TensorDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult =
      generatedAttributeParser(getContext(), parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown tensor attribute");
  return Attribute();
}

void TensorDialect::printAttribute(::mlir::Attribute attr,
                                   ::mlir::DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}
