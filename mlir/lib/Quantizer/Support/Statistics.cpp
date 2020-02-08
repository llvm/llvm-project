//===- Statistics.cpp - Collects statistics over tensors ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/Statistics.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;

//===----------------------------------------------------------------------===//
// AttributeTensorStatistics implementation
//===----------------------------------------------------------------------===//

static void collectElementsStatisticsDim(ElementsAttr attr,
                                         unsigned numElements,
                                         ArrayRef<int64_t> shape,
                                         SmallVectorImpl<uint64_t> &indices,
                                         uint64_t dim,
                                         TensorAxisStatistics &statistics) {
  // Recursive terminating condition.
  if (dim >= shape.size())
    return;

  if (dim < (shape.size() - 1)) {
    // Recurse past dim.
    for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
      indices[dim] = i;
      collectElementsStatisticsDim(attr, numElements, shape, indices, dim + 1,
                                   statistics);
    }
    return;
  }

  // Collection dim.
  for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
    indices[dim] = i;
    double value = attr.getValue<FloatAttr>(indices).getValueAsDouble();
    statistics.minValue = std::min(statistics.minValue, value);
    statistics.maxValue = std::max(statistics.maxValue, value);
    statistics.mean += value / numElements;
    // TODO: Calculate a running variance.
  }
}

static void collectElementsStatisticsDimForAxis(
    unsigned axis, ElementsAttr attr, unsigned numElements,
    ArrayRef<int64_t> shape, SmallVectorImpl<uint64_t> &indices, uint64_t dim,
    TensorAxisStatistics &statistics) {
  // Recursive terminating condition.
  if (dim >= shape.size())
    return;

  // Axis is passed separately
  if (dim == axis) {
    collectElementsStatisticsDimForAxis(axis, attr, numElements, shape, indices,
                                        dim + 1, statistics);
    return;
  }

  // Go to last not axis dim
  if (dim < (shape.size() - 2) ||
      (dim == (shape.size() - 2) && axis != (shape.size() - 1))) {
    // Recurse past dim.
    for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
      indices[dim] = i;
      collectElementsStatisticsDimForAxis(axis, attr, numElements, shape,
                                          indices, dim + 1, statistics);
    }
    return;
  }

  // Pass axis
  uint64_t axisSize = shape[axis];
  for (uint64_t axisIdx = 0; axisIdx < axisSize; ++axisIdx) {
    indices[axis] = axisIdx;
    // Collection dim.
    for (uint64_t i = 0, s = shape[dim]; i < s; ++i) {
      indices[dim] = i;
      double value = attr.getValue<FloatAttr>(indices).getValueAsDouble();
      statistics.minValuePerAxis[axisIdx] =
          std::min(statistics.minValuePerAxis[axisIdx], value);
      statistics.maxValuePerAxis[axisIdx] =
          std::max(statistics.maxValuePerAxis[axisIdx], value);
      statistics.meanPerAxis[axisIdx] += value / numElements;
      // TODO: Calculate a running variance.
    }
  }
}

static bool getElementsStatistics(ElementsAttr attr,
                                  TensorAxisStatistics &statistics) {
  ShapedType sType = attr.getType();
  if (!sType.hasStaticShape())
    return false;
  Type elementTy = sType.getElementType();
  if (!elementTy.isa<FloatType>())
    return false;

  SmallVector<uint64_t, 4> indices;
  indices.resize(sType.getRank());
  ArrayRef<int64_t> shape = sType.getShape();

  statistics.minValue = std::numeric_limits<double>::infinity();
  statistics.maxValue = -std::numeric_limits<double>::infinity();
  statistics.mean = 0;
  statistics.variance = 0;

  auto numElements = sType.getNumElements();
  collectElementsStatisticsDim(attr, numElements, shape, indices, 0,
                               statistics);
  statistics.sampleSize = numElements;

  return true;
}

static bool getElementsStatisticsForAxis(unsigned axis, ElementsAttr attr,
                                         TensorAxisStatistics &statistics) {
  ShapedType sType = attr.getType();
  if (!sType.hasStaticShape() || axis >= sType.getRank())
    return false;
  Type elementTy = sType.getElementType();
  if (!elementTy.isa<FloatType>())
    return false;

  SmallVector<uint64_t, 4> indices;
  indices.resize(sType.getRank());
  ArrayRef<int64_t> shape = sType.getShape();

  uint64_t axisSize = shape[axis];
  statistics.minValuePerAxis.assign(axisSize,
                                    std::numeric_limits<double>::infinity());
  statistics.maxValuePerAxis.assign(axisSize,
                                    -std::numeric_limits<double>::infinity());
  statistics.meanPerAxis.assign(axisSize, 0);
  statistics.variancePerAxis.assign(axisSize, 0);

  uint64_t numElements = sType.getNumElements() / shape[axis];
  collectElementsStatisticsDimForAxis(axis, attr, numElements, shape, indices,
                                      0, statistics);
  statistics.sampleSizePerAxis = numElements;

  return true;
}

bool AttributeTensorStatistics::get(TensorAxisStatistics &stats) const {
  if (FloatAttr floatAttr = attr.dyn_cast<FloatAttr>()) {
    double value = floatAttr.getValueAsDouble();
    stats = TensorAxisStatistics(1, value, value, value, 0);
    return true;
  } else if (auto eltAttr = attr.dyn_cast<ElementsAttr>()) {
    return getElementsStatistics(eltAttr, stats);
  }
  return false;
}

bool AttributeTensorStatistics::supportsPerAxis() const {
  if (auto eltAttr = attr.dyn_cast<ElementsAttr>())
    return eltAttr.getType().getRank() > 1;
  return false;
}

unsigned AttributeTensorStatistics::getAxisCount() const {
  if (!supportsPerAxis())
    return 0;
  return attr.cast<ElementsAttr>().getType().getRank();
}

bool AttributeTensorStatistics::getForAxis(unsigned axis,
                                           TensorAxisStatistics &stats) const {
  if (!supportsPerAxis())
    return false;
  auto eltAttr = attr.cast<ElementsAttr>();
  return getElementsStatisticsForAxis(axis, eltAttr, stats);
}

raw_ostream &mlir::quantizer::operator<<(raw_ostream &os,
                                         const TensorAxisStatistics &stats) {
  os << "STATS[sampleSizeLayer=" << stats.sampleSize
     << ", minValueLayer=" << stats.minValue
     << ", maxValueLayer=" << stats.maxValue << ", meanLayer=" << stats.mean
     << ", varianceLayer=" << stats.variance
     << ", sampleSizePerAxis=" << stats.sampleSizePerAxis << ", statsPerAxis={";
  for (unsigned i = 0, n = stats.minValuePerAxis.size(); i < n; ++i) {
    os << "minValue=" << stats.minValuePerAxis[i]
       << ", maxValue=" << stats.maxValuePerAxis[i]
       << ", mean=" << stats.meanPerAxis[i]
       << ", variance=" << stats.variancePerAxis[i];
    if (i != n - 1)
      os << "; ";
  }
  os << "}]";
  return os;
}
