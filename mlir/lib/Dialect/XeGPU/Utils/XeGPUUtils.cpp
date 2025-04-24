//===---- XeGPUUtils.cpp - MLIR Utilities for XeGPUOps   ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utility methods for working with the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/Utils/XeGPUUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include <numeric>

using namespace mlir;

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(xegpu::TensorDescType tdescTy) {
  auto layout = llvm::dyn_cast_if_present<LayoutAttr>(tdescTy.getLayout());
  // It only works for subgroup level layout, which only has lane_layout
  // and lane_data, and is to distribute a SIMD code into SIMT code.
  if (!layout || !layout.isSgLayout())
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = tdescTy.getShape();
  auto elementType = tdescTy.getElementType();

  // compute sgSize by multiply elements of laneLayout
  // e.g. for 2D layout, sgSize = laneLayout[0] * laneLayout[1]
  // e.g. for 1D layout, sgSize = laneLayout[0]
  auto sgSize = std::accumulate(laneLayout.begin(), laneLayout.end(), 1,
                                std::multiplies<int64_t>());

  // Case 1: regular loads/stores
  auto scatterAttr = tdescTy.getEncodingAsScatterTensorDescAttr();
  if (scatterAttr) {
    auto chunkSize = scatterAttr.getChunkSize().getInt();
    // Verify if the first dimension of the tensor descriptor shape is
    // distributable.
    assert(tdescShape[0] == laneLayout[0] &&
           "tensor descriptor shape is not distributable");
    return VectorType::get({chunkSize}, elementType);
  }

  // Case 2: block loads/stores
  // Check if the tensor descriptor shape is distributable.
  int64_t tensorSize = 1;
  for (auto [tdescDim, laneDim, laneDataDim] :
       llvm::zip_equal(tdescShape, laneLayout, laneData)) {
    assert((tdescDim % (laneDim * laneDataDim) == 0) &&
           "tensor descriptor shape is not distributable");
    tensorSize *= tdescDim;
  }
  // tensorSize must be adjusted for array_length.
  tensorSize *= tdescTy.getArrayLength();

  return VectorType::get({tensorSize / sgSize}, elementType);
}

FailureOr<VectorType>
mlir::xegpu::getDistributedVectorType(VectorType originalType,
                                      xegpu::LayoutAttr layout) {
  auto shape = originalType.getShape();
  auto helperTdescTy = xegpu::TensorDescType::get(
      shape, originalType.getElementType(),
      /*array_length=*/1, /*boundary_check=*/true,
      /*memory_space=*/xegpu::MemorySpace::Global, layout);
  return xegpu::getDistributedVectorType(helperTdescTy);
}
