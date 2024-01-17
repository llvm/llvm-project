//===- XeGPUDialect.cpp - MLIR XeGPU dialect implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeUtilities.h>

#include <numeric>

namespace mlir {
namespace xegpu {

void XeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
      >();
}

bool printDefaultValues() {
  auto *env = getenv("MLIR_XEGPU_PRINT_DEFAULTS");
  if (env && std::string(env) == "true")
    return true;
  return false;
}

SubGroupMapAttr SubGroupMapAttr::get(mlir::MLIRContext *context,
                                     llvm::ArrayRef<int32_t> wiLayout,
                                     llvm::ArrayRef<int32_t> wiData) {
  assert(wiLayout.size() == 2 && wiData.size() == 2 &&
         "wiLayout and wiData should be 2D arrays.\n");
  return Base::get(context, mlir::DenseI32ArrayAttr::get(context, wiLayout),
                   mlir::DenseI32ArrayAttr::get(context, wiData));
}

mlir::LogicalResult SubGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr layout, mlir::DenseI32ArrayAttr data) {

  if (layout.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_layout which "
                   "is to be an integer array of size 2.\n";
    return mlir::failure();
  }

  if (data.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_data which is "
                   "to be an integer array of size 2.\n";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::Attribute TensorDescAttr::parse(mlir::AsmParser &parser,
                                      mlir::Type type) {
  mlir::FailureOr<xegpu::MemoryScopeKind> memory_scope;
  mlir::FailureOr<int> array_length;
  mlir::FailureOr<bool> boundary_check;
  mlir::FailureOr<xegpu::ScatteredAttr> scattered;
  mlir::FailureOr<xegpu::SubGroupMapAttr> map;

  bool seen_memory_scope = false;
  bool seen_array_length = false;
  bool seen_boundary_check = false;
  bool seen_scattered = false;
  bool seen_map = false;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  // Parse elements
  auto parseElt = [&]() -> mlir::ParseResult {
    llvm::StringRef paramKey;

    if (!parser.parseOptionalKeyword(&paramKey)) {
      if (parser.parseEqual())
        return mlir::failure();

      if (!seen_memory_scope && paramKey == "memory_scope") {
        seen_memory_scope = true;
        // Parse variable 'memory_scope'
        memory_scope =
            mlir::FieldParser<mlir::xegpu::MemoryScopeKind>::parse(parser);
        if (mlir::failed(memory_scope))
          return parser.emitError(
              parser.getCurrentLocation(),
              "Failed to parse the 'memory_scope' of TensorDescAttr, which is "
              "to be a `xegpu::MemoryScope`");
      } else if (!seen_array_length && paramKey == "array_length") {
        seen_array_length = true;
        // Parse variable 'array_length'
        array_length = ::mlir::FieldParser<int>::parse(parser);
        if (mlir::failed(array_length))
          return parser.emitError(parser.getCurrentLocation(),
                                  "Failed to parse the 'array_length' of "
                                  "TensorDescAttr, which is to be a `int`");
      } else if (!seen_boundary_check && paramKey == "boundary_check") {
        seen_boundary_check = true;
        // Parse variable 'boundary_check'
        boundary_check = ::mlir::FieldParser<bool>::parse(parser);
        if (::mlir::failed(boundary_check))
          return parser.emitError(parser.getCurrentLocation(),
                                  "Failed to parse the 'boundary_check' of "
                                  "TensorDescAttr, which is to be a `bool`");
      } else if (!seen_map && paramKey == "map") {
        seen_map = true;
        // Parse variable 'map'
        map = ::mlir::FieldParser<xegpu::SubGroupMapAttr>::parse(parser);
        if (::mlir::failed(map))
          return parser.emitError(
              parser.getCurrentLocation(),
              "Failed to parse the 'map' of TensorDescAttr, which is to be a "
              "`xegpu::SubGroupMapAttr`");
      }
    } else if (!seen_scattered) {
      // parse scattered
      scattered = mlir::FieldParser<xegpu::ScatteredAttr>::parse(parser);
      if (mlir::failed(scattered))
        return parser.emitError(
            parser.getCurrentLocation(),
            "Failed to parse 'scattered' attr of TensorDescAttr, which is to "
            "be a `xegpu::ScatteredAttr`");
      seen_scattered = true;
    }
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return {};

  // Parse literal '>'
  if (parser.parseGreater())
    return {};
  return TensorDescAttr::get(
      parser.getContext(),
      memory_scope.value_or(xegpu::MemoryScopeKind::GLOBAL),
      array_length.value_or(1), boundary_check.value_or(true),
      scattered.value_or(xegpu::ScatteredAttr()),
      map.value_or(xegpu::SubGroupMapAttr()));
}

void TensorDescAttr::print(::mlir::AsmPrinter &printer) const {
  bool printSep = false;
  bool printDefaults = printDefaultValues();

  printer << "<";

  if (printDefaults || getMemoryScope() != xegpu::MemoryScopeKind::GLOBAL) {
    if (printSep)
      printer << ", ";
    printSep = true;
    printer << "memory_scope = ";
    printer.printStrippedAttrOrType(getMemoryScope());
  }
  if (printDefaults || getArrayLength() != 1) {
    if (printSep)
      printer << ", ";
    printSep = true;
    printer << "array_length = ";
    printer.printStrippedAttrOrType(getArrayLength());
  }
  if (printDefaults || getBoundaryCheck() != true) {
    if (printSep)
      printer << ", ";
    printSep = true;
    printer << "boundary_check = ";
    printer.printStrippedAttrOrType(getBoundaryCheck());
  }
  if (getScattered()) {
    if (printSep)
      printer << ", ";
    printSep = true;
    printer.printStrippedAttrOrType(getScattered());
  }
  if (getMap()) {
    if (printSep)
      printer << ", ";
    printSep = true;
    printer << "map = ";
    printer.printStrippedAttrOrType(getMap());
  }
  printer << ">";
}

bool TensorDescAttr::hasNonDefaultAttrs() {
  int count = 0;
  if (getMemoryScope() != MemoryScopeKind::GLOBAL)
    count++;
  if (getBoundaryCheck() != true)
    count++;
  if (getArrayLength() != 1)
    count++;
  if (getScattered())
    count++;
  if (getMap())
    count++;
  return count;
}

TensorDescAttr TensorDescAttr::get(mlir::MLIRContext *context,
                                   xegpu::MemoryScopeKind memory_scope,
                                   int array_length,
                                   xegpu::ScatteredAttr scattered,
                                   xegpu::SubGroupMapAttr map) {
  return Base::get(context, std::move(memory_scope), std::move(array_length),
                   true, std::move(scattered), std::move(map));
}

mlir::Type TensorDescType::parse(::mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<mlir::Attribute> encoding;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  auto elemTypeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  // parse optional attributes
  if (mlir::succeeded(parser.parseOptionalComma())) {
    encoding = mlir::FieldParser<mlir::Attribute>::parse(parser);
    if (mlir::failed(encoding)) {
      parser.emitError(
          parser.getCurrentLocation(),
          "Failed to parse the attribute field for TensorDescType.\n");
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return TensorDescType::get(parser.getContext(), shape, elementType,
                             encoding.value_or(mlir::Attribute()));
}

void TensorDescType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";

  auto shape = getShape();
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << getElementType();

  if (printDefaultValues()) {
    auto encoding = getEncoding();
    if (auto attr = getEncodingAsMapAttr()) {
      encoding = TensorDescAttr::get(getContext(), MemoryScopeKind::GLOBAL, 1,
                                     {}, attr);
    }
    if (auto attr = getEncodingAsScatteredAttr()) {
      encoding = TensorDescAttr::get(getContext(), MemoryScopeKind::GLOBAL, 1,
                                     attr, {});
    }
    printer << ", " << encoding;
  } else if (auto encoding = getEncodingAsTensorDescAttr()) {
    if (encoding.hasNonDefaultAttrs())
      printer << ", " << encoding;
  } else if (auto encoding = getEncoding()) {
    printer << ", " << encoding;
  }
  printer << ">";
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType,
                                   mlir::Attribute encoding) {
  return Base::get(elementType.getContext(), shape, elementType, encoding);
}

TensorDescType TensorDescType::get(mlir::MLIRContext *context,
                                   llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType,
                                   mlir::xegpu::MemoryScopeKind memory_scope,
                                   int array_length, bool boundary_check,
                                   mlir::xegpu::ScatteredAttr scattered,
                                   mlir::xegpu::SubGroupMapAttr mapping) {
  auto attr = TensorDescAttr::get(context, memory_scope, array_length,
                                  boundary_check, scattered, mapping);
  return Base::get(context, shape, elementType, attr);
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType,
                                   mlir::xegpu::MemoryScopeKind memory_scope,
                                   int array_length, bool boundary_check,
                                   mlir::xegpu::ScatteredAttr scattered,
                                   mlir::xegpu::SubGroupMapAttr mapping) {
  auto attr =
      TensorDescAttr::get(elementType.getContext(), memory_scope, array_length,
                          boundary_check, scattered, mapping);
  return Base::get(elementType.getContext(), shape, elementType, attr);
}

xegpu::MemoryScopeKind TensorDescType::getMemoryScope() {
  auto attr = getEncodingAsTensorDescAttr();
  if (attr)
    return attr.getMemoryScope();
  // return default value
  return MemoryScopeKind::GLOBAL;
}

int TensorDescType::getArrayLength() {
  auto attr = getEncodingAsTensorDescAttr();
  if (attr)
    return attr.getArrayLength();
  // return default value
  return 1;
}

bool TensorDescType::getBoundaryCheck() {
  auto attr = getEncodingAsTensorDescAttr();
  if (attr)
    return attr.getBoundaryCheck();
  // return default value
  return true;
}

xegpu::ScatteredAttr TensorDescType::getScattered() {
  if (auto attr = getEncodingAsTensorDescAttr())
    return attr.getScattered();
  if (auto attr = getEncodingAsScatteredAttr())
    return attr;
  // return default value
  return {};
}

xegpu::SubGroupMapAttr TensorDescType::getMapping() {
  if (auto attr = getEncodingAsTensorDescAttr())
    return attr.getMap();
  if (auto attr = getEncodingAsMapAttr())
    return attr;
  // return default value
  return xegpu::SubGroupMapAttr();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
