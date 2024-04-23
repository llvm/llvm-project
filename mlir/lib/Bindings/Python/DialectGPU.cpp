//===- DialectGPU.cpp - Pybind module for the GPU passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirDialectsGPU, m) {
  m.doc() = "MLIR GPU Dialect";

  //===-------------------------------------------------------------------===//
  // ObjectAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(m, "ObjectAttr", mlirAttributeIsAGPUObjectAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute target, uint32_t format,
             py::bytes object, std::optional<MlirAttribute> mlirObjectProps) {
            py::buffer_info info(py::buffer(object).request());
            MlirStringRef objectStrRef =
                mlirStringRefCreate(static_cast<char *>(info.ptr), info.size);
            return cls(mlirGPUObjectAttrGet(
                mlirAttributeGetContext(target), target, format, objectStrRef,
                mlirObjectProps.has_value() ? *mlirObjectProps
                                            : MlirAttribute{nullptr}));
          },
          "cls"_a, "target"_a, "format"_a, "object"_a,
          "properties"_a = py::none(), "Gets a gpu.object from parameters.")
      .def_property_readonly(
          "target",
          [](MlirAttribute self) { return mlirGPUObjectAttrGetTarget(self); })
      .def_property_readonly(
          "format",
          [](MlirAttribute self) { return mlirGPUObjectAttrGetFormat(self); })
      .def_property_readonly(
          "object",
          [](MlirAttribute self) {
            MlirStringRef stringRef = mlirGPUObjectAttrGetObject(self);
            return py::bytes(stringRef.data, stringRef.length);
          })
      .def_property_readonly("properties", [](MlirAttribute self) {
        if (mlirGPUObjectAttrHasProperties(self))
          return py::cast(mlirGPUObjectAttrGetProperties(self));
        return py::none().cast<py::object>();
      });
}
