//===- Linalg.cpp - C Interface for Linalg dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Linalg.h"
#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::linalg;

/// Apply the special region builder for the builtin named Linalg op.
/// Assert that `op` is a builtin named Linalg op.
void mlirLinalgFillBuiltinNamedOpRegion(MlirOperation mlirOp) {
  Operation *op = unwrap(mlirOp);
  auto linalgOp = cast<LinalgOp>(op);
  auto *dialect = static_cast<LinalgDialect *>(linalgOp->getDialect());
  LinalgDialect::RegionBuilderFunType fun =
      dialect->getRegionBuilder(op->getName().getStringRef());

  assert(fun && "Expected a builtin named Linalg op.");
  assert(op->getNumRegions() == 1 && "Expected Linalg op with 1 region");
  assert(op->getRegion(0).getBlocks().empty() &&
         "Expected Linalg op with 0 blocks");

  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    argTypes.push_back(getElementTypeOrSelf(opOperand.get().getType()));
    argLocs.push_back(opOperand.get().getLoc());
  }

  ImplicitLocOpBuilder b(op->getLoc(), op->getContext());
  Region &region = op->getRegion(0);
  Block *body = b.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);
  b.setInsertionPointToStart(body);
  fun(b, *body, op->getAttrs(), /*emitError=*/{});
}

MLIR_CAPI_EXPORTED bool mlirLinalgIsAContractionOp(MlirOperation op) {
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  // isaContractionOpInterface handles null linalgOp internally.
  return linalg::isaContractionOpInterface(linalgOp);
}

static MlirLinalgContractionDimensions
toContractionDimensions(MLIRContext *ctx,
                        const linalg::ContractionDimensions &dims) {
  auto toAttr = [ctx](ArrayRef<unsigned> vals) -> MlirAttribute {
    return wrap(DenseI32ArrayAttr::get(ctx, llvm::to_vector_of<int32_t>(vals)));
  };
  return {toAttr(dims.batch), toAttr(dims.m), toAttr(dims.n), toAttr(dims.k)};
}

MLIR_CAPI_EXPORTED MlirLinalgContractionDimensions
mlirLinalgInferContractionDimensions(MlirOperation op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return {};

  FailureOr<linalg::ContractionDimensions> maybeDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(maybeDims))
    return {};

  const linalg::ContractionDimensions &contractionDims = *maybeDims;
  MLIRContext *ctx = linalgOp.getContext();
  return toContractionDimensions(ctx, contractionDims);
}

MLIR_CAPI_EXPORTED MlirLinalgContractionDimensions
mlirLinalgInferContractionDimensionsFromMaps(const MlirAffineMap *indexingMaps,
                                             size_t numMaps) {
  if (!indexingMaps || numMaps != 3)
    return {};

  SmallVector<AffineMap, 3> maps;
  for (size_t i = 0; i < numMaps; ++i) {
    maps.push_back(unwrap(indexingMaps[i]));
  }

  FailureOr<linalg::ContractionDimensions> maybeDims =
      linalg::inferContractionDims(maps);
  if (failed(maybeDims))
    return {};

  MLIRContext *ctx = maps[0].getContext();

  return toContractionDimensions(ctx, *maybeDims);
}

MLIR_CAPI_EXPORTED bool mlirLinalgIsAConvolutionOp(MlirOperation op) {
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return false;

  return linalg::isaConvolutionOpInterface(linalgOp);
}

static MlirLinalgConvolutionDimensions
toConvolutionDimensions(MLIRContext *ctx,
                        const linalg::ConvolutionDimensions &dims) {
  auto toI32Attr = [ctx](ArrayRef<unsigned> vals) -> MlirAttribute {
    return wrap(DenseI32ArrayAttr::get(ctx, llvm::to_vector_of<int32_t>(vals)));
  };
  auto toI64Attr = [ctx](ArrayRef<int64_t> vals) -> MlirAttribute {
    return wrap(DenseI64ArrayAttr::get(ctx, vals));
  };
  return {toI32Attr(dims.batch),         toI32Attr(dims.outputImage),
          toI32Attr(dims.outputChannel), toI32Attr(dims.filterLoop),
          toI32Attr(dims.inputChannel),  toI32Attr(dims.depth),
          toI64Attr(dims.strides),       toI64Attr(dims.dilations)};
}

MLIR_CAPI_EXPORTED MlirLinalgConvolutionDimensions
mlirLinalgInferConvolutionDimensions(MlirOperation op) {
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return {};

  FailureOr<linalg::ConvolutionDimensions> maybeDims =
      linalg::inferConvolutionDims(linalgOp);
  if (failed(maybeDims))
    return {};

  return toConvolutionDimensions(linalgOp.getContext(), *maybeDims);
}

MLIR_CAPI_EXPORTED MlirLinalgConvolutionDimensions
mlirLinalgInferConvolutionDimensionsFromMaps(const MlirAffineMap *indexingMaps,
                                             size_t numMaps) {
  // inferConvolutionDims requires exactly 3 maps (input, filter, output);
  // keep this check in sync with its contract
  if (!indexingMaps || numMaps != 3)
    return {};

  SmallVector<AffineMap, 3> maps;
  for (size_t i = 0; i < numMaps; ++i)
    maps.push_back(unwrap(indexingMaps[i]));

  FailureOr<linalg::ConvolutionDimensions> maybeDims =
      linalg::inferConvolutionDims(maps);
  if (failed(maybeDims))
    return {};

  return toConvolutionDimensions(maps[0].getContext(), *maybeDims);
}

MLIR_CAPI_EXPORTED MlirAttribute
mlirLinalgGetIndexingMapsAttribute(MlirOperation op) {
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp)
    return MlirAttribute{nullptr};

  ArrayAttr attr = linalgOp.getIndexingMaps();
  return wrap(attr);
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Linalg, linalg, LinalgDialect)
