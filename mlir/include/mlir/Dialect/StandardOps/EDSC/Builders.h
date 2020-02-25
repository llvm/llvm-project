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
  ValueHandle lb(unsigned idx) { return lbs[idx]; }
  ValueHandle ub(unsigned idx) { return ubs[idx]; }
  int64_t step(unsigned idx) { return steps[idx]; }
  std::tuple<ValueHandle, ValueHandle, int64_t> range(unsigned idx) {
    return std::make_tuple(lbs[idx], ubs[idx], steps[idx]);
  }
  void swapRanges(unsigned i, unsigned j) {
    if (i == j)
      return;
    lbs[i].swap(lbs[j]);
    ubs[i].swap(ubs[j]);
    std::swap(steps[i], steps[j]);
  }

  ArrayRef<ValueHandle> getLbs() { return lbs; }
  ArrayRef<ValueHandle> getUbs() { return ubs; }
  ArrayRef<int64_t> getSteps() { return steps; }

protected:
  SmallVector<ValueHandle, 8> lbs;
  SmallVector<ValueHandle, 8> ubs;
  SmallVector<int64_t, 8> steps;
};

/// A MemRefBoundsCapture represents the information required to step through a
/// MemRef. It has placeholders for non-contiguous tensors that fit within the
/// Fortran subarray model.
/// At the moment it can only capture a MemRef with an identity layout map.
// TODO(ntv): Support MemRefs with layoutMaps.
class MemRefBoundsCapture : public BoundsCapture {
public:
  explicit MemRefBoundsCapture(Value v);
  MemRefBoundsCapture(const MemRefBoundsCapture &) = default;
  MemRefBoundsCapture &operator=(const MemRefBoundsCapture &) = default;

  unsigned fastestVarying() const { return rank() - 1; }

private:
  ValueHandle base;
};

/// A VectorBoundsCapture represents the information required to step through a
/// Vector accessing each scalar element at a time. It is the counterpart of
/// a MemRefBoundsCapture but for vectors. This exists purely for boilerplate
/// avoidance.
class VectorBoundsCapture : public BoundsCapture {
public:
  explicit VectorBoundsCapture(Value v);
  VectorBoundsCapture(const VectorBoundsCapture &) = default;
  VectorBoundsCapture &operator=(const VectorBoundsCapture &) = default;

private:
  ValueHandle base;
};

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_STANDARDOPS_EDSC_BUILDERS_H_
