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

  /// `ub.unreachable` does not return to the caller, so it stays in place when
  /// the inlined region has more than one block.
  void handleTerminator(Operation *op, Block *newDest) const final {
    assert(isa<UnreachableOp>(op) && "unexpected UB terminator");
  }

  /// The single-block fast path erases the terminator and continues the
  /// inlined block with the operations after the call, which is wrong for
  /// `ub.unreachable`.
  bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const final {
    return inlinedBlocks.empty() ||
           !isa<UnreachableOp>(inlinedBlocks.begin()->getTerminator());
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
  declarePromisedInterface<ConvertToLLVMPatternInterface, UBDialect>();
}

Operation *UBDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  if (auto attr = dyn_cast<PoisonAttr>(value))
    return PoisonOp::create(builder, loc, type, attr);

  return nullptr;
}

OpFoldResult PoisonOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

#include "mlir/Dialect/UB/IR/UBOpsInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/UB/IR/UBOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/UB/IR/UBOps.cpp.inc"
