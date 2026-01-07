//===- Rewrite.h - Rewrite Submodules of pybind module --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_REWRITE_H
#define MLIR_BINDINGS_PYTHON_REWRITE_H

#include "mlir-c/Rewrite.h"
#include "mlir/Bindings/Python/IRCore.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP Base class for rewriter wrappers.
template <typename DerivedTy>
class PyRewriterBase {
public:
  PyRewriterBase(MlirRewriterBase rewriter)
      : base(rewriter),
        ctx(PyMlirContext::forContext(mlirRewriterBaseGetContext(base))) {}

  PyInsertionPoint getInsertionPoint() const {
    MlirBlock block = mlirRewriterBaseGetInsertionBlock(base);
    MlirOperation op = mlirRewriterBaseGetOperationAfterInsertion(base);

    if (mlirOperationIsNull(op)) {
      MlirOperation owner = mlirBlockGetParentOperation(block);
      auto parent = PyOperation::forOperation(ctx, owner);
      return PyInsertionPoint(PyBlock(parent, block));
    }

    return PyInsertionPoint(PyOperation::forOperation(ctx, op));
  }

  void replaceOp(MlirOperation op, MlirOperation newOp) {
    mlirRewriterBaseReplaceOpWithOperation(base, op, newOp);
  }

  void replaceOp(MlirOperation op, const std::vector<MlirValue> &values) {
    mlirRewriterBaseReplaceOpWithValues(base, op, values.size(), values.data());
  }

  void eraseOp(MlirOperation op) { mlirRewriterBaseEraseOp(base, op); }

  static void bind(nanobind::module_ &m) {
    nb::class_<DerivedTy>(m, DerivedTy::pyClassName)
        .def_prop_ro("ip", &PyRewriterBase::getInsertionPoint,
                     "The current insertion point of the PatternRewriter.")
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op, PyOperationBase &newOp) {
              self.replaceOp(op.getOperation(), newOp.getOperation());
            },
            "Replace an operation with a new operation.", nb::arg("op"),
            nb::arg("new_op"))
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op,
               const std::vector<PyValue> &values) {
              std::vector<MlirValue> values_(values.size());
              std::copy(values.begin(), values.end(), values_.begin());
              self.replaceOp(op.getOperation(), values_);
            },
            "Replace an operation with a list of values.", nb::arg("op"),
            nb::arg("values"))
        .def("erase_op", &DerivedTy::eraseOp, "Erase an operation.",
             nb::arg("op"));
  }

private:
  MlirRewriterBase base;
  PyMlirContextRef ctx;
};

void populateRewriteSubmodule(nanobind::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_REWRITE_H
