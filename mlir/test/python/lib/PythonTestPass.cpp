//===- PythonTestPassDemo.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestPass.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct PythonTestPassDemo
    : public PassWrapper<PythonTestPassDemo, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PythonTestPassDemo)

  PythonTestPassDemo(PyObject *func) : func(func) {}
  StringRef getArgument() const final { return "python-pass-demo"; }

  void runOnOperation() override {
    this->getOperation()->walk([this](Operation *op) {
      PyObject *mlirModule =
          PyImport_ImportModule(MAKE_MLIR_PYTHON_QUALNAME("ir"));
      PyObject *cAPIFactory = PyObject_GetAttrString(
          PyObject_GetAttrString(mlirModule, "Operation"),
          MLIR_PYTHON_CAPI_FACTORY_ATTR);
      PyObject *opApiObject = PyObject_CallFunction(
          cAPIFactory, "(O)", mlirPythonOperationToCapsule(wrap(op)));
      (void)PyObject_CallFunction(func, "(O)", opApiObject);
      Py_DECREF(opApiObject);
    });
  }

  PyObject *func;
};

std::unique_ptr<OperationPass<ModuleOp>>
createPythonTestPassDemoPassWithFunc(PyObject *func) {
  return std::make_unique<PythonTestPassDemo>(func);
}

} // namespace

void registerPythonTestPassDemoPassWithFunc(PyObject *func) {
  registerPass([func]() { return createPythonTestPassDemoPassWithFunc(func); });
}
