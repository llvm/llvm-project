//===- Linalg.h - Linalg dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALG_H
#define MLIR_DIALECT_LINALG_IR_LINALG_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#include <optional>

namespace mlir {
namespace linalg {

class LinalgOp;

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%f, %A) : f32, memref<f32>
///   name mangles into `linalg_fill_f32_viewf32`
///
/// 2. linalg.dot %A, %B, %C :
///      (memref<?xf32, stride_specification>,
///       memref<?xf32, stride_specification>, memref<f32>)
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32`
///
/// 3. linalg.matmul(...) :
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32`
std::string generateLibraryCallName(Operation *op);

/// Returns `num` AffineDimExpr dimensions at positions
///   [startIdx, startIdx + num) and increments `startIdx` to `startIdx + num`.
SmallVector<AffineExpr, 4> makeAffineDimExprs(unsigned num, unsigned &startIdx,
                                              MLIRContext *context);

/// Returns `maybeMap.get()` if `maybeMap` is set, otherwise returns the
/// symbol-less identity map of `rank`.
AffineMap extractOrIdentityMap(std::optional<AffineMap> maybeMap, unsigned rank,
                               MLIRContext *context);

/// Return the vector that is the concatenation of `a` and `b`.
SmallVector<AffineExpr, 4> concat(ArrayRef<AffineExpr> a,
                                  ArrayRef<AffineExpr> b);

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value val, int64_t dim);

/// Create one memref::DimOp or tensor::DimOp depending on the type of `val`.
/// This is a polymorphic convenience function to abstract away the rank and
/// concrete type of `val`.
/// Asserts that `val` is a memref or tensor type.
OpFoldResult createFoldedDimOp(OpBuilder &b, Location loc, Value val,
                               int64_t dim);

} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Enums
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgOpsEnums.h.inc"

namespace mlir {
namespace linalg {

/// Converts the given `m` and `r` parameters to a WinogradConv2DFmr enumeration
/// value.
std::optional<WinogradConv2DFmr> getWinogradConv2DFmr(int64_t m, int64_t r);

/// Converts the given WinogradConv2DFmr enumeration value to a pair of
/// m and r parameters.
std::pair<int64_t, int64_t> getFmrFromWinogradConv2DFmr(WinogradConv2DFmr fmr);

} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// Linalg Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOpsAttrDefs.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

//===----------------------------------------------------------------------===//
// Linalg Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgRelayoutOps.h.inc"

namespace mlir::linalg {

/// Returns the outer shape in the packed domain before applying the
/// transposition.
template <typename OpTy,
          typename = std::enable_if_t<std::is_same_v<OpTy, linalg::PackOp> ||
                                      std::is_same_v<OpTy, linalg::UnPackOp>>>
SmallVector<int64_t> getPackedOuterShapeWithoutTransposition(OpTy packOrUnPack);

/// Specialization of `linalg.matmul` op that has a transpose map on A
class MatmulTransposeAOp : public MatmulOp {
  /// Create an affine map for a transpose-A matmul. Used only in the builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using MatmulOp::MatmulOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<MatmulOp>(); }

  /// Build a transpose A matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeAOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   Attribute cast,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.matmul` op that has a transpose map on B
class MatmulTransposeBOp : public MatmulOp {
  /// Create an affine map for a transpose-B matmul. Used only in the builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using MatmulOp::MatmulOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<MatmulOp>(); }

  /// Build a transpose B matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static MatmulTransposeBOp create(OpBuilder &builder, Location location,
                                   TypeRange resultTensorTypes,
                                   ValueRange inputs, ValueRange outputs,
                                   Attribute cast,
                                   ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.batch_matmul` op that has a transpose map on A
class BatchMatmulTransposeAOp : public BatchMatmulOp {
  /// Create an affine map for a transpose-A batch_matmul. Used only in the
  /// builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using BatchMatmulOp::BatchMatmulOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<BatchMatmulOp>(); }

  /// Build a transpose A matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, ValueRange inputs,
         ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs,
         ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose A matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeAOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs, Attribute cast,
         ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

/// Specialization of `linalg.batch_matmul` op that has a transpose map on B
class BatchMatmulTransposeBOp : public BatchMatmulOp {
  /// Create an affine map for a transpose-B batch_matmul. Used only in the
  /// builders.
  static SmallVector<AffineMap> getDefaultIndexingMaps(OpBuilder &builder);

public:
  using BatchMatmulOp::BatchMatmulOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<BatchMatmulOp>(); }

  /// Build a transpose B matmul.
  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, ValueRange inputs,
         ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs,
         ArrayRef<NamedAttribute> attributes = {});

  /// Build a transpose B matmul with a specific result type and a cast type.
  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs, Attribute cast,
                    ArrayRef<NamedAttribute> attributes = {});

  static BatchMatmulTransposeBOp
  create(OpBuilder &builder, Location location, TypeRange resultTensorTypes,
         ValueRange inputs, ValueRange outputs, Attribute cast,
         ArrayRef<NamedAttribute> attributes = {});

  /// Checks if the affine map is the expected one for this operation
  static bool isDefaultIndexingMaps(Attribute attr);

  static bool classof(Operation *op);
};

//===----------------------------------------------------------------------===//
// Unary specializations of `linalg.elementwise`.
//===----------------------------------------------------------------------===//

namespace detail {
/// Builds a `linalg.elementwise` op carrying the given unary `kind` and, unless
/// the caller provided `indexing_maps` in `attributes`, the default identity
/// indexing maps. Shared by the unary `ElementwiseOp` specializations below.
void buildElementwiseUnaryOp(OpBuilder &builder, OperationState &result,
                             std::optional<TypeRange> resultTensorTypes,
                             ValueRange inputs, ValueRange outputs,
                             ElementwiseKind kind,
                             ArrayRef<NamedAttribute> attributes);
} // namespace detail

/// CRTP base factoring the shared builders and `classof` for the hand-written
/// unary specializations of `linalg.elementwise`. Each concrete op fixes its
/// `ElementwiseKind` through a static `getElementwiseKind()`. Like
/// `MatmulTransposeAOp`, these are convenience views over the generic op and are
/// not registered as distinct operations.
template <typename ConcreteOp>
class ElementwiseUnaryOp : public ElementwiseOp {
public:
  using ElementwiseOp::ElementwiseOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ElementwiseOp>(); }

  /// Implicitly usable wherever the generic `LinalgOp` interface is expected,
  /// mirroring the registered named ops these views replace. Needed because the
  /// interface's converting constructor keys on the exact op type.
  operator LinalgOp() { return llvm::cast<LinalgOp>(getOperation()); }

  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange inputs, ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {}) {
    detail::buildElementwiseUnaryOp(builder, result, std::nullopt, inputs,
                                    outputs, ConcreteOp::getElementwiseKind(),
                                    attributes);
  }

  static void build(OpBuilder &builder, OperationState &result,
                    TypeRange resultTensorTypes, ValueRange inputs,
                    ValueRange outputs,
                    ArrayRef<NamedAttribute> attributes = {}) {
    detail::buildElementwiseUnaryOp(builder, result, resultTensorTypes, inputs,
                                    outputs, ConcreteOp::getElementwiseKind(),
                                    attributes);
  }

  static ConcreteOp create(OpBuilder &builder, Location location,
                           ValueRange inputs, ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(location, ElementwiseOp::getOperationName());
    ConcreteOp::build(builder, state, inputs, outputs, attributes);
    auto res = llvm::dyn_cast<ConcreteOp>(builder.create(state));
    assert(res && "builder didn't return the right type");
    return res;
  }

  static ConcreteOp create(OpBuilder &builder, Location location,
                           TypeRange resultTensorTypes, ValueRange inputs,
                           ValueRange outputs,
                           ArrayRef<NamedAttribute> attributes = {}) {
    OperationState state(location, ElementwiseOp::getOperationName());
    ConcreteOp::build(builder, state, resultTensorTypes, inputs, outputs,
                      attributes);
    auto res = llvm::dyn_cast<ConcreteOp>(builder.create(state));
    assert(res && "builder didn't return the right type");
    return res;
  }

  static bool classof(Operation *op) {
    auto elementwise = llvm::dyn_cast_or_null<ElementwiseOp>(op);
    return elementwise &&
           elementwise.getKind() == ConcreteOp::getElementwiseKind();
  }
};

/// Specialization of `linalg.elementwise` op for the unary `exp` kind.
class ExpOp : public ElementwiseUnaryOp<ExpOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::exp; }
};

/// Specialization of `linalg.elementwise` op for the unary `log` kind.
class LogOp : public ElementwiseUnaryOp<LogOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::log; }
};

/// Specialization of `linalg.elementwise` op for the unary `abs` kind.
class AbsOp : public ElementwiseUnaryOp<AbsOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::abs; }
};

/// Specialization of `linalg.elementwise` op for the unary `ceil` kind.
class CeilOp : public ElementwiseUnaryOp<CeilOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::ceil; }
};

/// Specialization of `linalg.elementwise` op for the unary `floor` kind.
class FloorOp : public ElementwiseUnaryOp<FloorOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::floor; }
};

/// Specialization of `linalg.elementwise` op for the unary `negf` kind.
class NegFOp : public ElementwiseUnaryOp<NegFOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::negf; }
};

/// Specialization of `linalg.elementwise` op for the unary `reciprocal` kind.
class ReciprocalOp : public ElementwiseUnaryOp<ReciprocalOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() {
    return ElementwiseKind::reciprocal;
  }
};

/// Specialization of `linalg.elementwise` op for the unary `round` kind.
class RoundOp : public ElementwiseUnaryOp<RoundOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::round; }
};

/// Specialization of `linalg.elementwise` op for the unary `sqrt` kind.
class SqrtOp : public ElementwiseUnaryOp<SqrtOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::sqrt; }
};

/// Specialization of `linalg.elementwise` op for the unary `rsqrt` kind.
class RsqrtOp : public ElementwiseUnaryOp<RsqrtOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::rsqrt; }
};

/// Specialization of `linalg.elementwise` op for the unary `square` kind.
class SquareOp : public ElementwiseUnaryOp<SquareOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::square; }
};

/// Specialization of `linalg.elementwise` op for the unary `tanh` kind.
class TanhOp : public ElementwiseUnaryOp<TanhOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::tanh; }
};

/// Specialization of `linalg.elementwise` op for the unary `erf` kind.
class ErfOp : public ElementwiseUnaryOp<ErfOp> {
public:
  using ElementwiseUnaryOp::ElementwiseUnaryOp;
  static ElementwiseKind getElementwiseKind() { return ElementwiseKind::erf; }
};

} // namespace mlir::linalg

#endif // MLIR_DIALECT_LINALG_IR_LINALG_H
