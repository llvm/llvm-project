//===- Builders.h - MLIR EDSC Builders for StandardOps ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_STANDARDOPS_EDSC_BUILDERS_H_
#define MLIR_DIALECT_STANDARDOPS_EDSC_BUILDERS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace edsc {

/// Base class for MemRefBoundsCapture and VectorBoundsCapture.
class BoundsCapture {
public:
  unsigned rank() const { return lbs.size(); }
  Value lb(unsigned idx) const { return lbs[idx]; }
  Value ub(unsigned idx) const { return ubs[idx]; }
  int64_t step(unsigned idx) const { return steps[idx]; }
  std::tuple<Value, Value, int64_t> range(unsigned idx) const {
    return std::make_tuple(lbs[idx], ubs[idx], steps[idx]);
  }
  void swapRanges(unsigned i, unsigned j) {
    if (i == j)
      return;
    std::swap(lbs[i], lbs[j]);
    std::swap(ubs[i], ubs[j]);
    std::swap(steps[i], steps[j]);
  }

  ArrayRef<Value> getLbs() const { return lbs; }
  ArrayRef<Value> getUbs() const { return ubs; }
  ArrayRef<int64_t> getSteps() const { return steps; }

protected:
  SmallVector<Value, 8> lbs;
  SmallVector<Value, 8> ubs;
  SmallVector<int64_t, 8> steps;
};

/// A MemRefBoundsCapture represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO: Support MemRefs with layoutMaps.
class MemRefBoundsCapture : public BoundsCapture {
public:
  explicit MemRefBoundsCapture(Value v);

  unsigned fastestVarying() const { return rank() - 1; }

private:
  Value base;
};

/// A VectorBoundsCapture represents the information required to step through a
/// Vector accessing each scalar element at a time. It is the counterpart of
/// a MemRefBoundsCapture but for vectors. This exists purely for boilerplate
/// avoidance.
class VectorBoundsCapture : public BoundsCapture {
public:
  explicit VectorBoundsCapture(Value v);
  explicit VectorBoundsCapture(VectorType t);

private:
  Value base;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_BUILDERS_H_
