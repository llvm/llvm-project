//===- ArithDialect.cpp - MLIR Arith dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

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
  declarePromisedInterface<ConvertToLLVMPatternInterface, ArithDialect>();
  declarePromisedInterface<bufferization::BufferDeallocationOpInterface,
                           SelectOp>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, ConstantOp,
                            IndexCastOp, SelectOp>();
  declarePromisedInterfaces<ValueBoundsOpInterface, AddIOp, ConstantOp, SubIOp,
                            MulIOp>();
}

/// Materialize an integer or floating point constant.
Operation *arith::ArithDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return builder.create<ub::PoisonOp>(loc, type, poison);

  return ConstantOp::materialize(builder, value, type, loc);
}

/// Return true if the type is compatible with fast math, i.e.
/// it is a float type or contains a float type.
bool arith::ArithFastMathInterface::isCompatibleType(Type type) {
  if (isa<FloatType>(type))
    return true;

  // ShapeType's with ValueSemantics represent containers
  // passed around as values (not references), so look inside
  // them to see if the element type is compatible with FastMath.
  if (type.hasTrait<ValueSemantics>())
    if (auto shapedType = dyn_cast<ShapedType>(type))
      return isCompatibleType(shapedType.getElementType());

  // ComplexType's element type is always a FloatType.
  if (auto complexType = dyn_cast<ComplexType>(type))
    return true;

  // TODO: what about TupleType and custom dialect struct-like types?
  // It seems that they worth an interface to get to the list of element types.
  //
  // NOTE: LLVM only allows fast-math flags for instructions producing
  // structures with homogeneous floating point members. I think
  // this restriction must not be asserted here, because custom
  // MLIR operations may be converted such that the original operation's
  // FastMathFlags still need to be propagated to the target
  // operations.

  return false;
}

/// Return true if any of the results of the operation
/// has a type compatible with fast math, i.e. it is a float type
/// or contains a float type.
///
/// TODO: the results often have the same type, and traversing
/// the same type again and again is not very efficient.
/// We can cache it here for the duration of the processing.
/// Other ideas?
bool arith::ArithFastMathInterface::isApplicableImpl() {
  Operation *op = getOperation();
  if (llvm::any_of(op->getResults(),
                   [](Value v) { return isCompatibleType(v.getType()); }))
    return true;
  return false;
}
