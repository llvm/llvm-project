//===- ArithDialect.cpp - MLIR Arith dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::arith;

#include "mlir/Dialect/Arith/IR/ArithOpsDialect.cpp.inc"
#include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Arith/IR/ArithOpsAttributes.cpp.inc"

namespace {
/// This class defines the interface for handling inlining for arithmetic
/// dialect operations.
struct ArithInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All arithmetic dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void arith::ArithDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Arith/IR/ArithOpsAttributes.cpp.inc"
      >();
  addInterfaces<ArithInlinerInterface>();
  declarePromisedInterface<ConvertToEmitCPatternInterface, ArithDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, ArithDialect>();
  declarePromisedInterface<bufferization::BufferDeallocationOpInterface,
                           SelectOp>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, ConstantOp,
                            IndexCastOp, SelectOp>();
  declarePromisedInterfaces<ValueBoundsOpInterface, ConstantOp, ExtSIOp, AddIOp,
                            SubIOp, MulIOp, SelectOp, FloorDivSIOp, CeilDivSIOp,
                            MinSIOp, MaxSIOp, MinUIOp, MaxUIOp, RemSIOp,
                            RemUIOp, DivUIOp, DivSIOp>();
}

/// Materialize an integer or floating point constant.
Operation *arith::ArithDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return ub::PoisonOp::create(builder, loc, type, poison);

  return ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// AssumeCmpIAttr
//===----------------------------------------------------------------------===//

LogicalResult
AssumeCmpIAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       CmpIPredicate predicate, IntegerAttr rhs) {
  if (!symbolizeCmpIPredicate(static_cast<uint64_t>(predicate)))
    return emitError() << "invalid integer comparison predicate";
  auto type = dyn_cast<IntegerType>(rhs.getType());
  if (!type || !type.isSignless() || type.getWidth() == 0)
    return emitError() << "expected a non-zero-width signless integer RHS";
  return success();
}

LogicalResult AssumeCmpIAttr::verifyAssumption(
    Type valueType, TypeRange predicateOperandTypes,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (!predicateOperandTypes.empty())
    return emitError()
           << "arith.assume_cmpi does not accept predicate operands";
  if (valueType != getRhs().getType())
    return emitError() << "arith.assume_cmpi expected predicated value type "
                       << getRhs().getType() << ", but got " << valueType;
  return success();
}

std::optional<bool>
AssumeCmpIAttr::evaluate(Attribute value,
                         ArrayRef<Attribute> /*predicateOperands*/) const {
  auto integer = dyn_cast_if_present<IntegerAttr>(value);
  if (!integer)
    return std::nullopt;
  return applyCmpPredicate(getPredicate(), integer.getValue(),
                           getRhs().getValue());
}
