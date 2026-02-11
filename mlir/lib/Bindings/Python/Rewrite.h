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

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP Base class for rewriter wrappers.
template <typename DerivedTy>
class MLIR_PYTHON_API_EXPORTED PyRewriterBase {
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

  static void bind(nanobind::module_ &m) {
    nanobind::class_<DerivedTy>(m, DerivedTy::pyClassName)
        .def_prop_ro("ip", &PyRewriterBase::getInsertionPoint,
                     "The current insertion point of the PatternRewriter.")
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op, PyOperationBase &newOp) {
              mlirRewriterBaseReplaceOpWithOperation(
                  self.base, op.getOperation(), newOp.getOperation());
            },
            "Replace an operation with a new operation.", nanobind::arg("op"),
            nanobind::arg("new_op"))
        .def(
            "replace_op",
            [](DerivedTy &self, PyOperationBase &op,
               const std::vector<PyValue> &values) {
              std::vector<MlirValue> values_(values.size());
              std::copy(values.begin(), values.end(), values_.begin());
              mlirRewriterBaseReplaceOpWithValues(
                  self.base, op.getOperation(), values_.size(), values_.data());
            },
            "Replace an operation with a list of values.", nanobind::arg("op"),
            nanobind::arg("values"))
        .def(
            "erase_op",
            [](DerivedTy &self, PyOperationBase &op) {
              mlirRewriterBaseEraseOp(self.base, op.getOperation());
            },
            "Erase an operation.", nanobind::arg("op"));
  }

private:
  MlirRewriterBase base;
  PyMlirContextRef ctx;
};

void MLIR_PYTHON_API_EXPORTED populateRewriteSubmodule(nanobind::module_ &m);
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_REWRITE_H
