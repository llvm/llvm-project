//===- XeGPUDialect.cpp - MLIR XeGPU dialect implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using std::optional;

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
#define GET_OP_INTERFACE_CLASSES
#include "mlir/Dialect/XeGPU/IR/XeGPUOpInterface.cpp.inc"

// A `srcShape` consists of N distribution units, each being `subShapesLayout` x
// `subShape`. A `delinearizedId` is used to identify a particular `subShape`
// within each distribution unit.
// Example:
// WG data is 128x256. SG data is 16x32, in 4x2 layout, this gives a
// distribution unit of shape 64x64, we have 2x4 such distribution units.
// `delinearizedId` is used to identify a 16x32 of a subgroup in each
// distribution unit.
static SmallVector<SmallVector<Value>>
genCoordinates(OpBuilder &builder, Location loc,
               SmallVector<Value> delinearizedId,
               ArrayRef<int64_t> subShapesLayout, ArrayRef<int64_t> subShape,
               ArrayRef<int64_t> srcShape) {
  SmallVector<SmallVector<Value>> coordinates;

  // A distribution unit must be less than or equal to `srcShape`
  SmallVector<int64_t> distUnitShape = llvm::map_to_vector(
      llvm::zip_equal(srcShape,
                      computeElementwiseMul(subShapesLayout, subShape)),
      [](const auto &t) { return std::min(std::get<0>(t), std::get<1>(t)); });

  // Get the offset of `subShape` within a distribution unit.
  SmallVector<Value> distUnitLocalOffset = llvm::map_to_vector(
      llvm::zip(delinearizedId, subShape), [&](const auto &t) -> Value {
        return builder.createOrFold<arith::MulIOp>(
            loc, std::get<0>(t),
            builder.createOrFold<arith::ConstantIndexOp>(loc, std::get<1>(t)));
      });

  // For each dist unit
  for (SmallVector<int64_t> unitOffs :
       StaticTileOffsetRange(srcShape, distUnitShape)) {
    // Get dist unit offset within `srcShape`.
    SmallVector<Value> base =
        llvm::map_to_vector(unitOffs, [&](int64_t d) -> Value {
          return arith::ConstantIndexOp::create(builder, loc, d);
        });
    // Calculate `subShape` offset within `srcShape`.
    SmallVector<Value> adds =
        llvm::map_to_vector(llvm::zip_equal(base, distUnitLocalOffset),
                            [&](const auto &t) -> Value {
                              return builder.createOrFold<arith::AddIOp>(
                                  loc, std::get<0>(t), std::get<1>(t));
                            });
    // Do not go beyond `srcShape` bounds.
    SmallVector<Value> mods = llvm::map_to_vector(
        llvm::zip_equal(adds, srcShape), [&](const auto &t) -> Value {
          return builder.createOrFold<arith::RemUIOp>(
              loc, std::get<0>(t),
              arith::ConstantIndexOp::create(builder, loc, std::get<1>(t)));
        });

    coordinates.push_back(mods);
  }
  return coordinates;
}

// Checks if the given shape can be evenly distributed based on the layout
// and data factors provided by the LayoutAttr.
bool XeGPUDialect::isEvenlyDistributable(llvm::ArrayRef<int64_t> shape,
                                         xegpu::DistributeLayoutAttr attr) {
  assert(attr && "Layout attribute is missing.");

  // Checks whether the given shape can be evenly distributed using the
  // specified layout and data attributes. If successful, it returns the work
  // size for each compute unit; otherwise, it returns `std::nullopt`. The work
  // size per compute unit is calculated as follows:
  //   - If `data` is null: newShape[i] = shape[i] / layout[i]
  //   - If `data` is not null: newShape[i] = data[i]
  // When round-robin distribution (`rr`) is enabled, `shape[i]` can be
  // smaller than `layout[i] * data[i]`, allowing multiple compute units to
  // share the data.
  auto tryDistribute = [&](llvm::ArrayRef<int64_t> shape,
                           SmallVector<int64_t> layout,
                           SmallVector<int64_t> data,
                           bool rr = true) -> optional<SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> newShape(shape);
    if (layout.size()) {
      if (layout.size() != shape.size())
        return std::nullopt;
      auto ratio = computeShapeRatio(shape, layout);
      if (ratio.has_value()) {
        newShape = ratio.value();
      } else if (!rr || !computeShapeRatio(layout, shape).has_value()) {
        return std::nullopt;
      }
      // Round-robin case: continue with original newShape
    }

    if (data.size()) {
      if (data.size() != shape.size())
        return std::nullopt;
      auto ratio = computeShapeRatio(newShape, data);
      if (!ratio.has_value() && rr)
        ratio = computeShapeRatio(data, newShape);
      if (!ratio.has_value())
        return std::nullopt;

      // if data is not null, we always return it for next phase.
      newShape = data;
    }
    return newShape;
  };

  // check the sgLayout and sgData
  auto maybeSgShape = tryDistribute(shape, attr.getEffectiveSgLayoutAsInt(),
                                    attr.getEffectiveSgDataAsInt());
  if (!maybeSgShape)
    return false;
  auto sgShape = maybeSgShape.value();

  // check InstData, it neither have layout nor need round-robin
  auto maybeInstShape =
      tryDistribute(sgShape, {}, attr.getEffectiveInstDataAsInt(), false);
  if (!maybeInstShape)
    return false;
  auto instShape = maybeInstShape.value();

  // check LaneLayout and LaneData
  auto maybeLaneShape =
      tryDistribute(instShape, attr.getEffectiveLaneLayoutAsInt(),
                    attr.getEffectiveLaneDataAsInt(), false);
  return maybeLaneShape.has_value();
}

//===----------------------------------------------------------------------===//
// XeGPU_BlockTensorDescAttr
//===----------------------------------------------------------------------===//
BlockTensorDescAttr BlockTensorDescAttr::get(mlir::MLIRContext *context,
                                             xegpu::MemorySpace memory_space,
                                             int array_length,
                                             bool boundary_check) {
  auto scopeAttr = MemorySpaceAttr::get(context, memory_space);
  auto lengthAttr =
      IntegerAttr::get(IntegerType::get(context, 64), array_length);
  auto boundaryAttr = BoolAttr::get(context, boundary_check);
  return Base::get(context, scopeAttr, lengthAttr, boundaryAttr);
}

bool BlockTensorDescAttr::hasDefaultsOnly() {
  return getMemorySpace().getValue() == xegpu::MemorySpace::Global &&
         getArrayLength().getInt() == 1 && getBoundaryCheck().getValue();
}

//===----------------------------------------------------------------------===//
// XeGPU_ScatterTensorDescAttr
//===----------------------------------------------------------------------===//
ScatterTensorDescAttr
ScatterTensorDescAttr::get(mlir::MLIRContext *context,
                           xegpu::MemorySpace memory_space, int chunk_size) {
  auto scopeAttr = MemorySpaceAttr::get(context, memory_space);
  auto chunkSizeAttr =
      IntegerAttr::get(IntegerType::get(context, 64), chunk_size);
  return Base::get(context, scopeAttr, chunkSizeAttr);
}

LogicalResult ScatterTensorDescAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MemorySpaceAttr memory_space, IntegerAttr chunk_size) {
  int64_t chunkSize = chunk_size.getInt();
  if (chunkSize <= 0)
    return emitError() << "invalid chunk size";

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LayoutAttr
//===----------------------------------------------------------------------===//
LogicalResult
LayoutAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data,
                   DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout,
                   DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {

  // Special case for store_matrix
  if (!sg_layout && !inst_data && !lane_layout)
    return success();

  // generate code to check sg_laout, inst_data and lane_layout having the same
  // rank if they are not null.

  if (sg_layout && inst_data && sg_layout.size() != inst_data.size()) {
    return emitError()
           << "expected sg_layout and inst_data to have the same rank";
  }

  if (sg_layout && lane_layout && sg_layout.size() != lane_layout.size()) {
    return emitError()
           << "expected sg_layout and lane_layout to have the same rank";
  }

  if (inst_data && lane_layout && inst_data.size() != lane_layout.size()) {
    return emitError() << "expected inst_data and lane_layout to have the same "
                          "rank, got inst_data "
                       << inst_data.size() << ", lane_layout "
                       << lane_layout.size();
  }

  // sg_data is optional for Workgroup layout, but its presence requires
  // sg_layout.
  if (sg_data) {
    if (!sg_layout)
      return emitError() << "expected sg_layout being used with sg_data";
    if (sg_data.size() != sg_layout.size())
      return emitError()
             << "expected sg_data and sg_layout to have the same rank";
  }

  // lane_data is optional for Subgroup layout, but its presence requires
  // lane_layout.
  if (lane_data) {
    if (!lane_layout)
      return emitError() << "expected lane_layout being used with lane_data";
    if (lane_data.size() != lane_layout.size())
      return emitError()
             << "expected lane_data and lane_layout to have the same rank";
  }

  if (order) {
    if (!sg_layout && !lane_layout)
      return emitError()
             << "expected sg_layout/lane_layout being used with order";

    if (sg_layout && order.size() != sg_layout.size())
      return emitError()
             << "expected order and sg_layout to have the same rank";

    if (lane_layout && order.size() != lane_layout.size())
      return emitError()
             << "expected order and lane_layout to have the same rank";
  }

  return success();
}

FailureOr<SmallVector<Value>>
LayoutAttr::delinearizeId(OpBuilder &builder, Location loc, Value linearId) {

  SmallVector<int64_t> sgLayoutInt;
  if (isForWorkgroup()) {
    sgLayoutInt = getEffectiveSgLayoutAsInt();
  } else if (isForSubgroup()) {
    sgLayoutInt = getEffectiveLaneLayoutAsInt();
  } else {
    return failure();
  }

  DenseI32ArrayAttr orderAttr = getOrder();

  // Handle order attribute
  SmallVector<int64_t> order;
  if (orderAttr && !orderAttr.empty()) {
    order = llvm::to_vector(
        llvm::map_range(orderAttr.asArrayRef(),
                        [](int32_t idx) { return static_cast<int64_t>(idx); }));
  } else {
    // Default order: [1, 0] for 2D (row-major), [2, 1, 0] for 3D, etc.
    order = llvm::to_vector(
        llvm::reverse(llvm::seq<int64_t>(0, sgLayoutInt.size())));
  }

  if (order.size() != sgLayoutInt.size()) {
    return failure();
  }

  SmallVector<Value> result(sgLayoutInt.size());
  Value remaining = linearId;

  /// Process dimensions in the order they appear in the order array
  /// The first dimension in order is the fastest-changing
  ///
  /// Example walkthrough for linearId=22, sgLayout=[2,4,4], order=[2,1,0]:
  ///
  /// Initial: remaining=22, dimIdx = order[i], dimSize = sgLayout[dimIdx],
  /// result=[?,?,?]
  ///
  /// i=0 (process columns, dimIdx=2, dimSize=4):
  ///   result[2] = 22 % 4 = 2  (column coordinate)
  ///   remaining = 22 / 4 = 5  (5 complete groups of 4 columns processed)
  ///
  /// i=1 (process rows, dimIdx=1, dimSize=4):
  ///   result[1] = 5 % 4 = 1   (row coordinate)
  ///   remaining = 5 / 4 = 1   (1 complete group of 4 rows processed)
  ///
  /// i=2 (process layers, dimIdx=0, dimSize=2):
  ///   result[0] = 1 % 2 = 1   (layer coordinate)
  ///   (no remaining update - last iteration)
  ///
  /// Final result: [1,1,2] = Layer 1, Row 1, Column 2
  for (size_t i = 0; i < order.size(); ++i) {
    int64_t dimIdx = order[i];
    int64_t dimSize = sgLayoutInt[dimIdx];

    Value dimSizeVal =
        builder.createOrFold<arith::ConstantIndexOp>(loc, dimSize);

    /// Extract the coordinate for this dimension using modulo operation
    /// This gives us "how far within this dimension" we are
    /// e.g., linearId=22, dimSize=4: 22 % 4 = 2 (we're at position 2 within
    /// this dimension)
    result[dimIdx] =
        builder.createOrFold<arith::RemUIOp>(loc, remaining, dimSizeVal);

    /// Update remaining for the next dimension by removing what we've already
    /// processed. Division tells us "how many complete groups of this dimension
    /// we've gone through" e.g., linearId=22, dimSize=4: 22 / 4 = 5 (we've
    /// completed 5 groups of 4) Skip this for the last iteration since there's
    /// no next dimension to process
    if (i < order.size() - 1) {
      remaining =
          builder.createOrFold<arith::DivUIOp>(loc, remaining, dimSizeVal);
    }
  }
  return result;
}

/// Implements DistributeLayoutAttr::computeDistributedCoords to generate
/// instructions for computing multi-dimensional offsets when distributed by
/// LayoutAttr.
FailureOr<SmallVector<SmallVector<Value>>>
LayoutAttr::computeDistributedCoords(OpBuilder &builder, Location loc,
                                     Value linearId, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> layout;
  SmallVector<int64_t> subShape;
  if (isForWorkgroup()) {
    layout = getEffectiveSgLayoutAsInt();
    subShape = getEffectiveSgDataAsInt();
  } else if (isForSubgroup()) {
    layout = getEffectiveLaneLayoutAsInt();
    subShape = getEffectiveLaneDataAsInt();
  } else {
    return failure();
  }
  if (subShape.empty()) {
    if (auto derivedShape = computeShapeRatio(shape, layout))
      subShape = derivedShape.value();
    else
      return failure();
  }

  // delinearize Ids
  auto maybeIds = delinearizeId(builder, loc, linearId);
  if (failed(maybeIds))
    return failure();
  SmallVector<Value> ids = *maybeIds;

  return genCoordinates(builder, loc, ids, layout, subShape, shape);
}

bool LayoutAttr::isEqualTo(const xegpu::DistributeLayoutAttr &other) {
  if (dyn_cast<xegpu::SliceAttr>(other))
    return false;

  return *this == dyn_cast<xegpu::LayoutAttr>(other);
}

// set the layout for unit dims: sg_data, inst_data and lane_data to 1
DistributeLayoutAttr
LayoutAttr::setUnitDimData(SetVector<int64_t> unitDims) const {
  auto sgDataOpt = getSgData();
  auto instDataOpt = getInstData();
  auto laneDataOpt = getLaneData();

  SmallVector<int32_t> sgData;
  SmallVector<int32_t> instData;
  SmallVector<int32_t> laneData;

  if (sgDataOpt) {
    sgData = llvm::to_vector(sgDataOpt.asArrayRef());
  }
  if (instDataOpt) {
    instData = llvm::to_vector(instDataOpt.asArrayRef());
  }
  if (laneDataOpt) {
    laneData = llvm::to_vector(laneDataOpt.asArrayRef());
  }

  for (auto dim : unitDims) {
    if (dim < static_cast<int64_t>(sgData.size()))
      sgData[dim] = 1;
    if (dim < static_cast<int64_t>(instData.size()))
      instData[dim] = 1;
    if (dim < static_cast<int64_t>(laneData.size()))
      laneData[dim] = 1;
  }

  return LayoutAttr::get(
      getContext(), getSgLayout(),
      sgData.empty() ? DenseI32ArrayAttr()
                     : DenseI32ArrayAttr::get(getContext(), sgData),
      instData.empty() ? DenseI32ArrayAttr()
                       : DenseI32ArrayAttr::get(getContext(), instData),
      getLaneLayout(),
      laneData.empty() ? DenseI32ArrayAttr()
                       : DenseI32ArrayAttr::get(getContext(), laneData),
      getOrder());
}

// set the layout for the sepcified unit dims: sg_lane and lane_layout to 1
DistributeLayoutAttr
LayoutAttr::setUnitDimLayout(SetVector<int64_t> unitDims) const {
  auto sgLayoutOpt = getSgLayout();
  auto laneLayoutOpt = getLaneLayout();

  SmallVector<int32_t> sgLayout;
  SmallVector<int32_t> laneLayout;

  if (sgLayoutOpt) {
    sgLayout = llvm::to_vector(sgLayoutOpt.asArrayRef());
  }
  if (laneLayoutOpt) {
    laneLayout = llvm::to_vector(laneLayoutOpt.asArrayRef());
  }

  for (auto dim : unitDims) {
    if (dim < static_cast<int64_t>(sgLayout.size()))
      sgLayout[dim] = 1;
    if (dim < static_cast<int64_t>(laneLayout.size()))
      laneLayout[dim] = 1;
  }

  return LayoutAttr::get(
      getContext(),
      sgLayout.empty() ? DenseI32ArrayAttr()
                       : DenseI32ArrayAttr::get(getContext(), sgLayout),
      getSgData(), getInstData(),
      laneLayout.empty() ? DenseI32ArrayAttr()
                         : DenseI32ArrayAttr::get(getContext(), laneLayout),
      getLaneData(), getOrder());
}

//===----------------------------------------------------------------------===//
// XeGPU_SliceAttr
//===----------------------------------------------------------------------===//
LogicalResult
SliceAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) {

  if (!dims)
    return emitError() << "expected dims attribute";

  // check every element in dims is unique and smaller than rank
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dim : dims.asArrayRef()) {
    if (dim < 0)
      return emitError() << "invalid dim (" << dim << ") in slice attribute.";
    if (!seen.insert(dim).second)
      return emitError() << "repeated dim (" << dim << ") in slice attribute.";
  }
  return success();
}

SliceAttr SliceAttr::flatten() const {
  xegpu::DistributeLayoutAttr parent = getParent();
  SmallVector<DenseI64ArrayAttr> slicedDims({getDims()});

  while (auto sliceAttr = dyn_cast<xegpu::SliceAttr>(parent)) {
    parent = sliceAttr.getParent();
    slicedDims.push_back(sliceAttr.getDims());
  }

  auto layoutAttr = dyn_cast<xegpu::LayoutAttr>(parent);
  SmallVector<int64_t> indices =
      llvm::to_vector(llvm::seq<int64_t>(0, layoutAttr.getRank()));

  // get remaining dims (flattend) by applying slice ops with all slicedDims
  SmallVector<int64_t> remainingDims(indices);
  for (auto dim : llvm::reverse(slicedDims))
    remainingDims = XeGPUDialect::slice(llvm::ArrayRef<int64_t>(remainingDims),
                                        dim.asArrayRef());

  // get flattend sliced dims by applying slice ops with the remaining dims
  SmallVector<int64_t> flattendDims = XeGPUDialect::slice(
      llvm::ArrayRef<int64_t>(indices), llvm::ArrayRef<int64_t>(remainingDims));

  return xegpu::SliceAttr::get(
      getContext(), layoutAttr,
      DenseI64ArrayAttr::get(getContext(), flattendDims));
}

FailureOr<SmallVector<Value>>
SliceAttr::delinearizeId(OpBuilder &builder, Location loc, Value linearId) {
  SliceAttr attr = flatten();
  auto parent = dyn_cast<LayoutAttr>(attr.getParent());
  return parent.delinearizeId(builder, loc, linearId);
}

// Implements DistributeLayoutAttr::computeDistributedCoords to generate
// instructions for computing multi-dimensional offsets when distributed by
// LayoutAttr.
FailureOr<SmallVector<SmallVector<Value>>>
SliceAttr::computeDistributedCoords(OpBuilder &builder, Location loc,
                                    Value linearId, ArrayRef<int64_t> shape) {
  assert(getRank() == static_cast<int64_t>(shape.size()) && "invalid shape.");
  if (!isForWorkgroup())
    return failure();

  SmallVector<int64_t> layout;
  SmallVector<int64_t> subShape;
  if (isForWorkgroup()) {
    layout = getEffectiveSgLayoutAsInt();
    subShape = getEffectiveSgDataAsInt();
  } else if (isForSubgroup()) {
    layout = getEffectiveLaneLayoutAsInt();
    subShape = getEffectiveLaneDataAsInt();
  } else {
    return failure();
  }

  if (subShape.empty()) {
    if (auto derivedShape = computeShapeRatio(shape, layout))
      subShape = derivedShape.value();
    else
      return failure();
  }

  // delinearize Ids
  auto maybeIds = delinearizeId(builder, loc, linearId);
  if (failed(maybeIds))
    return failure();

  // The effective sgIds for offsets computing correspond
  // to the dims that are not sliced.
  ArrayRef<int64_t> dims = flatten().getDims().asArrayRef();
  SmallVector<Value> sgIds =
      XeGPUDialect::slice(ArrayRef<Value>(*maybeIds), dims);

  return genCoordinates(builder, loc, sgIds, layout, subShape, shape);
}

bool SliceAttr::isSliceOf(const xegpu::DistributeLayoutAttr &other) {
  auto flattenedThis = flatten();
  // If other is a LayoutAttr, just compare directly with parent of
  // flattenedThis.
  if (auto otherLayout = dyn_cast<xegpu::LayoutAttr>(other))
    return flattenedThis.getParent() == otherLayout;
  // If other is a SliceAttr, flatten it first before comparing.
  auto flattenedOther = dyn_cast<xegpu::SliceAttr>(other).flatten();
  // Both must have common parent LayoutAttr.
  if (flattenedThis.getParent() != flattenedOther.getParent())
    return false;
  // otherFlattened's sliced dims must be a subset of flattenedThis's sliced
  // dims.
  llvm::SmallDenseSet<int64_t> thisDims(
      flattenedThis.getDims().asArrayRef().begin(),
      flattenedThis.getDims().asArrayRef().end());
  return llvm::all_of(flattenedOther.getDims().asArrayRef(),
                      [&](int64_t dim) { return thisDims.contains(dim); });
}

xegpu::SliceAttr SliceAttr::dropSliceDims(ArrayRef<int64_t> sliceDimsToDrop) {
  if (sliceDimsToDrop.empty())
    return *this;
  SmallVector<int64_t> sliceDims{getDims().asArrayRef()};
  for (auto dim : sliceDimsToDrop) {
    auto foundIt = std::find(sliceDims.begin(), sliceDims.end(), dim);
    assert(foundIt != sliceDims.end() &&
           "Expected to find the specified reduction dim in slice dims");
    sliceDims.erase(foundIt);
  }

  auto sliceWithoutDims = xegpu::SliceAttr::get(
      this->getContext(), getParent(),
      DenseI64ArrayAttr::get(this->getContext(), sliceDims));

  return sliceWithoutDims;
}

bool SliceAttr::isEqualTo(const xegpu::DistributeLayoutAttr &other) {
  if (dyn_cast<xegpu::LayoutAttr>(other))
    return false;

  auto flattenedThis = flatten();
  auto flattenedOther = dyn_cast<xegpu::SliceAttr>(other).flatten();

  return ((flattenedThis.getParent() == flattenedOther.getParent()) &&
          (flattenedThis.getDims() == flattenedOther.getDims()));
}

// Helper function to adjust dimensions from sliced space to parent space
// say we have a parent shape of rank 4, and slice dims [1,3], so the sliced
// shape is of rank 2, if we want to set unit dim [0] in sliced space, it maps
// to dim [0] in parent space; if we want to set unit dim [1] in sliced space,
// it maps to dim [2] in parent space.
static SetVector<int64_t>
mapSlicedDimsToParentSpace(const SetVector<int64_t> &dimsToMap,
                           ArrayRef<int64_t> sliceDims) {
  // Rather than recovering the exact parent rank, we compute a safe upper bound
  // so that dimsToMap can be adjusted safely. This upper bound is defined as
  // max(dimsToMap, sliceDims) + 1 + sliceDims.size().
  int64_t maxDim = -1;
  maxDim =
      std::max(maxDim, *std::max_element(sliceDims.begin(), sliceDims.end()));
  maxDim =
      std::max(maxDim, *std::max_element(dimsToMap.begin(), dimsToMap.end()));
  int64_t parentSpaceRank = maxDim + sliceDims.size() + 1;

  // get remaining dims in parent space after applying slicing with parent's
  // slice Dims
  llvm::SmallDenseSet<int64_t> slicedDimsSet(sliceDims.begin(),
                                             sliceDims.end());
  SmallVector<int64_t> remainingDims;
  for (int64_t i = 0; i < parentSpaceRank; ++i) {
    if (!slicedDimsSet.contains(i))
      remainingDims.push_back(i);
  }

  // Map unit dims from sliced space to parent space
  SetVector<int64_t> adjustUnitDims;
  for (auto dim : dimsToMap) {
    int64_t mappedDim = remainingDims[dim];
    adjustUnitDims.insert(mappedDim);
  }

  return adjustUnitDims;
}

// set the layout for unit dims: sg_data, inst_data and lane_data to 1
DistributeLayoutAttr
SliceAttr::setUnitDimData(SetVector<int64_t> unitDims) const {
  DistributeLayoutAttr parentLayout = getParent();

  ArrayRef<int64_t> sliceDims = getDims().asArrayRef();

  SetVector<int64_t> adjustUnitDims =
      mapSlicedDimsToParentSpace(unitDims, sliceDims);

  return SliceAttr::get(getContext(),
                        parentLayout.setUnitDimData(adjustUnitDims), getDims());
}

// set the layout for the sepcified unit dims: sg_lane and lane_layout to 1
DistributeLayoutAttr
SliceAttr::setUnitDimLayout(SetVector<int64_t> unitDims) const {
  DistributeLayoutAttr parentLayout = getParent();

  ArrayRef<int64_t> sliceDims = getDims().asArrayRef();

  SetVector<int64_t> adjustUnitDims =
      mapSlicedDimsToParentSpace(unitDims, sliceDims);

  return SliceAttr::get(
      getContext(), parentLayout.setUnitDimLayout(adjustUnitDims), getDims());
}

//===----------------------------------------------------------------------===//
// XeGPU_RangeAttr
//===----------------------------------------------------------------------===//

LogicalResult
RangeAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  IntegerAttr startOfRange, IntegerAttr endOfRange) {
  if (startOfRange.getInt() >= endOfRange.getInt())
    return emitError() << "'end' : " << endOfRange.getInt()
                       << " must be greater than 'start' : "
                       << startOfRange.getInt();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_TensorDescType
//===----------------------------------------------------------------------===//

mlir::Type TensorDescType::parse(AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<mlir::Attribute> encoding;
  mlir::FailureOr<mlir::Attribute> layout;

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
  while (mlir::succeeded(parser.parseOptionalComma())) {
    mlir::Attribute attr;
    ParseResult res = parser.parseAttribute(attr);
    if (mlir::succeeded(res)) {
      if (mlir::isa<LayoutAttr>(attr)) {
        layout = attr;
        continue;
      }
      if (mlir::isa<BlockTensorDescAttr, ScatterTensorDescAttr>(attr)) {
        encoding = attr;
        continue;
      }
    }
    return {};
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  MLIRContext *ctxt = parser.getContext();
  return TensorDescType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, ctxt, shape,
      elementType, encoding.value_or(BlockTensorDescAttr::get(ctxt)),
      layout.value_or(mlir::Attribute()));
}

void TensorDescType::print(AsmPrinter &printer) const {
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

  auto encoding = getEncoding();
  auto blockAttr = llvm::dyn_cast_if_present<BlockTensorDescAttr>(encoding);
  if (encoding && (!blockAttr || !blockAttr.hasDefaultsOnly()))
    printer << ", " << encoding;

  if (auto layout = getLayout())
    printer << ", " << layout;

  printer << ">";
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int array_length,
                                   bool boundary_check,
                                   MemorySpace memory_space,
                                   mlir::Attribute layout) {
  auto context = elementType.getContext();
  auto attr = BlockTensorDescAttr::get(context, memory_space, array_length,
                                       boundary_check);
  return Base::get(context, shape, elementType, attr, layout);
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int chunk_size,
                                   MemorySpace memory_space,
                                   mlir::Attribute layout) {
  auto context = elementType.getContext();
  auto attr = ScatterTensorDescAttr::get(context, memory_space, chunk_size);
  return Base::get(context, shape, elementType, attr, layout);
}

LogicalResult
TensorDescType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                       mlir::Attribute encoding, mlir::Attribute layout) {
  size_t rank = shape.size();

  if (rank == 0)
    return emitError() << "expected non-zero rank tensor";

  auto blockAttr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(encoding);
  if (blockAttr) {
    MemorySpaceAttr memorySpaceAttr = blockAttr.getMemorySpace();
    if (rank > 1 && memorySpaceAttr &&
        memorySpaceAttr.getValue() == MemorySpace::SLM)
      return emitError() << "SLM is only supported for 1D block tensor";
  }

  if (!elementType.isIntOrFloat())
    return emitError() << "unsupported element type " << elementType
                       << ": expected integer or float";

  // for gather and scatter ops, Low-precision types are packed in 32-bit units.
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  int chunkAlignmentFactor =
      bitWidth < xegpu::uArch::generalPackedFormatBitSize
          ? xegpu::uArch::generalPackedFormatBitSize / bitWidth
          : 1;
  auto scatterAttr = mlir::dyn_cast_if_present<ScatterTensorDescAttr>(encoding);
  if (scatterAttr) {
    int64_t chunkSize = scatterAttr.getChunkSizeAsInt();
    if (rank == 1 && chunkSize != 1)
      return emitError() << "expected non-contiguous elements for 1D tensor";

    // If chunk size > 1, the second dimension of the tensor shape must be
    // equal to chunk size and it must be a multiple of the
    // chunkAlignmentFactor.
    if (chunkSize > 1) {
      if (shape.back() != chunkSize)
        return emitError() << "expected last dim of tensor to match chunk size";
      if (shape.back() % chunkAlignmentFactor != 0)
        return emitError() << "expected last dim of tensor to be a multiple of "
                           << chunkAlignmentFactor;
    }
  }

  auto layoutAttr = llvm::dyn_cast_if_present<LayoutAttr>(layout);
  if (layoutAttr) {
    if (rank != (size_t)layoutAttr.getRank())
      return emitError() << "expected layout rank to match tensor rank";

    auto laneData = layoutAttr.getLaneData();
    if (scatterAttr && laneData) {
      // Validate subgroup mapping rules for scattered tensors.
      // if chunkSize > 1, the last dimension of the tensor should
      // be distributed in the units divisible by chunkAlignmentFactor.
      int64_t chunkSize = scatterAttr.getChunkSizeAsInt();
      if (chunkSize > 1 && laneData[rank - 1] % chunkAlignmentFactor)
        return emitError()
               << "expected last dim of lane_data to be a multiple of: "
               << chunkAlignmentFactor;
    }

    if (!XeGPUDialect::isEvenlyDistributable(shape, layoutAttr)) {
      std::string shapeStr;
      llvm::raw_string_ostream stream(shapeStr);
      llvm::interleaveComma(shape, stream);
      return emitError() << "cannot distribute [" << shapeStr << "] using "
                         << layoutAttr;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_MemDescType
//===----------------------------------------------------------------------===//
mlir::Type MemDescType::parse(AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<MemLayoutAttr> layout;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape, false, true))) {
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
    MemLayoutAttr attr;
    ParseResult res = parser.parseAttribute(attr);
    if (mlir::failed(res))
      return {};
    layout = attr;
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  MLIRContext *ctxt = parser.getContext();
  return MemDescType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, ctxt, shape,
      elementType, layout.value_or(MemLayoutAttr()));
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";

  printer.printDimensionList(getShape());
  printer << 'x';
  printer << getElementType();

  if (auto layout = getMemLayout())
    printer << ", " << layout;

  printer << ">";
}

//===----------------------------------------------------------------------===//
// XeGPU_MemDescType
//===----------------------------------------------------------------------===//

Attribute MemLayoutAttr::parse(AsmParser &parser, Type type) {

  auto context = parser.getContext();
  llvm::SMLoc loc = parser.getCurrentLocation();

  llvm::SmallDenseSet<StringRef> seenKeys;
  SmallVector<NamedAttribute> attributes;

  auto parseElt = [&]() -> ParseResult {
    StringRef nameId;
    if (failed(parser.parseKeyword(&nameId)))
      return parser.emitError(loc, "expected valid attribute name");

    if (!seenKeys.insert(nameId).second)
      return parser.emitError(loc, "duplicate key '")
             << nameId << " in mem layout attribute";

    if (failed(parser.parseEqual()))
      return failure();

    Attribute attr;
    if (failed(parser.parseAttribute(attr)))
      return failure();
    attributes.emplace_back(nameId, attr);
    return success();
  };

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  if (failed(parser.parseCommaSeparatedList(parseElt)))
    return {};

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return parser.getChecked<MemLayoutAttr>(
      loc, context, DictionaryAttr::get(context, attributes));
}

void MemLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<";
  ArrayRef<NamedAttribute> attrs = getAttrs().getValue();
  for (size_t i = 0; i < attrs.size(); i++) {
    printer << attrs[i].getName().str() << " = " << attrs[i].getValue();
    if (i < attrs.size() - 1)
      printer << ", ";
  }
  printer << ">";
}
// a helper utility to perform binary operation on OpFoldResult.
// If both a and b are attributes, it will simply return the result.
// Otherwise, the corresponding arith op will be generated, and an
// contant op will be created if one of them is an attribute.
template <typename ArithOp>
OpFoldResult genBinOp(OpFoldResult a, OpFoldResult b, Location loc,
                      OpBuilder &builder) {
  auto aVal = getValueOrCreateConstantIndexOp(builder, loc, a);
  auto bVal = getValueOrCreateConstantIndexOp(builder, loc, b);
  return ArithOp::create(builder, loc, aVal, bVal).getResult();
}

// a helper utility to perform division operation on OpFoldResult and int64_t.
#define div(a, b)                                                              \
  genBinOp<arith::DivSIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform reminder operation on OpFoldResult and int64_t.
#define rem(a, b)                                                              \
  genBinOp<arith::RemSIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform multiply operation on OpFoldResult and int64_t.
#define mul(a, b)                                                              \
  genBinOp<arith::MulIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform addition operation on two OpFoldResult.
#define add(a, b) genBinOp<arith::AddIOp>(a, b, loc, builder)

// block the given offsets according to the block shape
// say the original offset is [y, x], and the block shape is [By, Bx],
// then the blocked offset is [y/By, x/Bx, y%By, x%Bx]
SmallVector<OpFoldResult> getBlockedOffsets(OpBuilder &builder, Location loc,
                                            ArrayRef<OpFoldResult> offsets,
                                            ArrayRef<int64_t> blockShape) {

  assert(offsets.size() == blockShape.size() &&
         "offsets and blockShape must have the same size");
  SmallVector<OpFoldResult> blockedOffsets;
  SmallVector<OpFoldResult> divs, rems;

  for (auto [offset, block] : llvm::zip(offsets, blockShape)) {
    divs.push_back(div(offset, block));
    rems.push_back(rem(offset, block));
  }
  blockedOffsets.append(divs.begin(), divs.end());
  blockedOffsets.append(rems.begin(), rems.end());

  return blockedOffsets;
}

// Get strides as vector of integer for MemDesc.
SmallVector<int64_t> MemDescType::getStrideShape() {

  SmallVector<int64_t> matrixShape(getShape().begin(), getShape().end());

  ArrayAttr strideAttr = getStrideAttr();
  SmallVector<int64_t> strides;
  for (Attribute attr : strideAttr.getValue()) {
    strides.push_back(cast<IntegerAttr>(attr).getInt());
  }

  SmallVector<int64_t> innerBlkShape = getBlockShape();

  // get perm from FCD to LCD
  // perm[i] = the dim with i-th smallest stride
  SmallVector<int, 4> perm =
      llvm::to_vector<4>(llvm::seq<int>(0, strides.size()));
  llvm::sort(perm, [&](int a, int b) { return strides[a] < strides[b]; });

  assert(strides[perm[0]] == 1 && "inner most dim must have stride 1");

  SmallVector<int64_t> innerBlkStride(innerBlkShape.size());
  innerBlkStride[perm[0]] = 1;
  for (size_t i = 1; i < perm.size(); ++i)
    innerBlkStride[perm[i]] =
        innerBlkStride[perm[i - 1]] * innerBlkShape[perm[i - 1]];

  // compute the original matrix shape using the stride info
  // and compute the number of blocks in each dimension
  // The shape of highest dim can't be derived from stride info,
  // but doesn't impact the stride computation for blocked layout.
  SmallVector<int64_t> matrixShapeOrig(matrixShape.size());
  SmallVector<int64_t> BlkShapeOrig(matrixShape.size());
  for (size_t i = 0; i < perm.size() - 1; ++i) {
    matrixShapeOrig[perm[i]] = strides[perm[i + 1]] / strides[perm[i]];
    BlkShapeOrig[perm[i]] = matrixShapeOrig[perm[i]] / innerBlkShape[perm[i]];
  }

  int64_t innerBlkSize = 1;
  for (auto s : innerBlkShape)
    innerBlkSize *= s;

  SmallVector<int64_t> outerBlkStride(matrixShape.size());
  outerBlkStride[perm[0]] = innerBlkSize;
  for (size_t i = 0; i < perm.size() - 1; ++i) {
    outerBlkStride[perm[i + 1]] =
        outerBlkStride[perm[i]] * BlkShapeOrig[perm[i]];
  }

  // combine the inner and outer strides
  SmallVector<int64_t> blockedStrides;
  blockedStrides.append(outerBlkStride.begin(), outerBlkStride.end());
  blockedStrides.append(innerBlkStride.begin(), innerBlkStride.end());

  return blockedStrides;
}

// Calculate the linear offset using the blocked offsets and stride
Value MemDescType::getLinearOffsets(OpBuilder &builder, Location loc,
                                    ArrayRef<OpFoldResult> offsets) {

  SmallVector<int64_t> matrixShape(getShape().begin(), getShape().end());
  SmallVector<int64_t> blockShape = getBlockShape();
  SmallVector<int64_t> strides = getStrideShape();
  SmallVector<OpFoldResult> blockedOffsets;

  // blockshape equal to matrixshape means no blocking
  if (llvm::equal(blockShape, matrixShape)) {
    // remove the outer dims from strides
    strides.erase(strides.begin(), strides.begin() + matrixShape.size());
  } else {
    assert(offsets.size() == blockShape.size() &&
           "offsets and blockShape must have the same size");
    // say the original offset is [y, x], and the block shape is [By, Bx],
    // then the blocked offset is [y/By, x/Bx, y%By, x%Bx]

    SmallVector<OpFoldResult> divs, rems;

    for (auto [offset, block] : llvm::zip(offsets, blockShape)) {
      divs.push_back(div(offset, block));
      rems.push_back(rem(offset, block));
    }
    blockedOffsets.append(divs.begin(), divs.end());
    blockedOffsets.append(rems.begin(), rems.end());
    offsets = blockedOffsets;
  }

  // Start with initial value as matrix descriptor's base offset.
  Value linearOffset = arith::ConstantIndexOp::create(builder, loc, 0);
  for (size_t i = 0; i < offsets.size(); ++i) {
    OpFoldResult mulResult = mul(offsets[i], strides[i]);
    Value mulVal = getValueOrCreateConstantIndexOp(builder, loc, mulResult);
    linearOffset = arith::AddIOp::create(builder, loc, mulVal, linearOffset);
  }

  return linearOffset;
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
