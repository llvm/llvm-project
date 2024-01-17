//===- UBOps.cpp - UB Dialect Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/UB/IR/UBOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::ub;

namespace {
/// This class defines the interface for handling inlining with UB
/// operations.
struct UBInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All UB ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// UBDialect
//===----------------------------------------------------------------------===//

void UBDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/UB/IR/UBOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/UB/IR/UBOpsAttributes.cpp.inc"
      >();
  addInterfaces<UBInlinerInterface>();
  declarePromisedInterface<UBDialect, ConvertToLLVMPatternInterface>();
}

Operation *UBDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  if (auto attr = dyn_cast<PoisonAttr>(value))
    return builder.create<PoisonOp>(loc, type, attr);

  return nullptr;
}

OpFoldResult PoisonOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

#include "mlir/Dialect/UB/IR/UBOpsInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/UB/IR/UBOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/UB/IR/UBOps.cpp.inc"
