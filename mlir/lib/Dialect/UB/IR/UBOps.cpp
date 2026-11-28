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
#include "llvm/ADT/STLExtras.h"

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
  /// `ub.unreachable`. This hook is only asked when the inlined block ends in a
  /// UB terminator, and `ub.unreachable` is the only one.
  bool allowSingleBlockOptimization(Block *) const final { return false; }
};
} // namespace

//===----------------------------------------------------------------------===//
// UBDialect
//===----------------------------------------------------------------------===//

void UBDialect::initialize() {
  addInterfaces<UBInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, UBDialect>();
}

Operation *UBDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  if (auto attr = dyn_cast<PoisonAttr>(value))
    return PoisonOp::create(builder, loc, type, attr);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

LogicalResult AssumeOp::verify() {
  return getPredicate().verifyAssumption(getValue().getType(),
                                         getPredicateOperands().getTypes(),
                                         [&] { return emitOpError(); });
}

OpFoldResult AssumeOp::fold(FoldAdaptor adaptor) {
  // Poison in => Poison out.
  if (isa_and_nonnull<PoisonAttr>(adaptor.getValue()))
    return PoisonAttr::get(getContext());

  // Folds identical chained assumptions.
  if (auto source = getValue().getDefiningOp<AssumeOp>()) {
    if (source.getPredicate() == getPredicate() &&
        llvm::equal(source.getPredicateOperands(), getPredicateOperands()))
      return getValue();
  }

  // If the predicate evaluates to "true" based on the constant-folded value,
  // drop the assumption. It didn't add any additional information.
  std::optional<bool> holds = getPredicate().evaluate(
      adaptor.getValue(), adaptor.getPredicateOperands());
  if (!holds)
    return {};
  if (*holds)
    return getValue();

  // If the predicate evaluates to "false" based on the constant-folded value,
  // fold it to poison.
  return PoisonAttr::get(getContext());
}

OpFoldResult PoisonOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

#include "mlir/Dialect/UB/IR/UBOpsInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/UB/IR/UBOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/UB/IR/UBOps.cpp.inc"
