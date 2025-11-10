//===- DialectPtr.cpp - Pybind module for Ptr dialect API support ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NanobindUtils.h"

#include "mlir-c/Dialect/PtrDialect.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Diagnostics.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

using namespace nanobind::literals;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

static void populateDialectPTRSubmodule(nanobind::module_ &m) {
  mlir_type_subclass(m, "PtrType", mlirPtrTypeIsAPtrType)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirAttribute memorySpace) {
            return cls(mlirPtrGetPtrType(memorySpace));
          },
          "Gets an instance of PtrType with memory_space in the same context",
          nb::arg("cls"), nb::arg("memory_space"));
}

NB_MODULE(_mlirDialectsPTR, m) {
  m.doc() = "MLIR PTR Dialect";

  populateDialectPTRSubmodule(m);
}
