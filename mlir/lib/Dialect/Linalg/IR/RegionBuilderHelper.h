//===- RegionBuilderHelper.h - Region-Builder-Helper class declaration ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper builds the unary, binary, and type conversion functions defined by the
// DSL. See LinalgNamedStructuredOps.yamlgen.cpp.inc for the code that uses this
// class.
//
// Implementations of the math functions must be polymorphic over numeric types,
// internally performing necessary casts. If the function application makes no
// sense, then the only recourse is to assert and return nullptr. This can be
// extended later if it becomes possible to fail construction of the region. The
// invariant should be enforced at a higher level.
//
// TODO: These helpers are currently type polymorphic over the class of integer
// and floating point types, but they will not internally cast within bit
// widths of a class (mixed precision such as i8->i32) or across classes
// (i.e. mixed float and integer). Many such combinations are ambiguous or need
// to be handled with care and work is being considered to extend the op
// language to make such cases explicit. In the mean-time, violating this will
// fail verification, which is deemed acceptable.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINALG_REGION_BUILDER_HELPER_H
#define MLIR_LINALG_REGION_BUILDER_HELPER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

namespace mlir {
namespace linalg {

class RegionBuilderHelper {
public:
  RegionBuilderHelper(OpBuilder &builder, Block &block)
      : builder(builder), block(block) {}

  // Build the unary functions.
  Value buildUnaryFn(UnaryFn unaryFn, Value arg,
                     function_ref<InFlightDiagnostic()> emitError = {});

  // Build the binary functions.
  // If emitError is provided, an error will be emitted if the operation is not
  // supported and a nullptr will be returned, otherwise an assertion is raised.
  Value buildBinaryFn(BinaryFn binaryFn, Value arg0, Value arg1,
                      function_ref<InFlightDiagnostic()> emitError = {});

  // Build the ternary functions defined by OpDSL.
  Value buildTernaryFn(TernaryFn ternaryFn, Value arg0, Value arg1, Value arg2,
                       function_ref<InFlightDiagnostic()> emitError = {});

  // Build the type functions defined by OpDSL.
  Value buildTypeFn(TypeFn typeFn, Type toType, Value operand,
                    function_ref<InFlightDiagnostic()> emitError = {});

  // Create a `yieldOp` to yield `values` passed in as arg.
  void yieldOutputs(ValueRange values);

  // Create a constant op with value parsed from string `value`.
  Value constant(const std::string &value);

  // Create an `index` op to extract iteration index `dim`.
  Value index(int64_t dim) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    return IndexOp::create(builder, builder.getUnknownLoc(), dim);
  }

  // Create an integer of size `width`.
  Type getIntegerType(unsigned width) {
    return IntegerType::get(builder.getContext(), width);
  }

  Type getFloat32Type() { return Float32Type::get(builder.getContext()); }
  Type getFloat64Type() { return Float64Type::get(builder.getContext()); }

private:
  // Generates operations to cast the given operand to a specified type.
  // If the cast cannot be performed, a warning will be issued and the
  // operand returned as-is (which will presumably yield a verification
  // issue downstream).
  Value cast(Type toType, Value operand, bool isUnsignedCast) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(&block);
    auto loc = operand.getLoc();
    if (isa<UnknownLoc>(loc)) {
      if (operand.getDefiningOp())
        loc = operand.getDefiningOp()->getLoc();
      else if (operand.getParentBlock() &&
               operand.getParentBlock()->getParentOp())
        loc = operand.getParentBlock()->getParentOp()->getLoc();
    }
    return convertScalarToDtype(builder, loc, operand, toType, isUnsignedCast);
  }

  bool isComplex(Value value) {
    return llvm::isa<ComplexType>(value.getType());
  }
  bool isFloatingPoint(Value value) {
    return llvm::isa<FloatType>(value.getType());
  }
  bool isInteger(Value value) {
    return llvm::isa<IntegerType>(value.getType());
  }

  OpBuilder &builder;
  Block &block;
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_REGION_BUILDER_HELPER_H
