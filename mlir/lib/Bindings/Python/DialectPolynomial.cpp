//===- DialectPolynomial.cpp - 'polynomial' dialect submodule -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Polynomial.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_mlirDialectsPolynomial, m) {
  m.doc() = "MLIR Polynomial dialect";
}
