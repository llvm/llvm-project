//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// LinalgDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct LinalgInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  // Operations in Linalg dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

/// Trait to check if T provides a `regionBuilder` method.
template <typename T, typename... Args>
using has_region_builder = decltype(T::regionBuilder);
template <typename T>
using detect_has_region_builder = llvm::is_detected<has_region_builder, T>;

/// SFINAE helper for single C++ class without a `regionBuilder` method (e.g.
/// an OpInterface).
template <typename OpType, typename = std::enable_if_t<
                               !detect_has_region_builder<OpType>::value>>
void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  // Do nothing.
}

template <typename OpType,
          typename = std::enable_if_t<detect_has_region_builder<OpType>::value>,
          typename = void>
void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  map.insert(std::make_pair(
      OpType::getOperationName(),
      static_cast<LinalgDialect::RegionBuilderFunType>(OpType::regionBuilder)));
}

template <typename... OpTypes>
void addNamedOpBuilders(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  (void)std::initializer_list<int>{0,
                                   (addNamedOpBuilderImpl<OpTypes>(map), 0)...};
}

void mlir::linalg::LinalgDialect::initialize() {
  addTypes<RangeType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgSparseOps.cpp.inc"
      >();

  // Fill the Linalg-specific OpName to RegionBuilder map.
  addNamedOpBuilders<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(namedStructuredOpRegionBuilders);

  addInterfaces<LinalgInlinerInterface>();
}

Type mlir::linalg::LinalgDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "range")
    return RangeType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown Linalg type: " + keyword);
  return Type();
}

/// RangeType prints as just "range".
static void print(RangeType rt, DialectAsmPrinter &os) { os << "range"; }

void mlir::linalg::LinalgDialect::printType(Type type,
                                            DialectAsmPrinter &os) const {
  print(type.cast<RangeType>(), os);
}
