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
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace nanobind::literals;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::nanobind_adaptors;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirDialectsGPU, m) {
  m.doc() = "MLIR GPU Dialect";
  //===-------------------------------------------------------------------===//
  // AsyncTokenType
  //===-------------------------------------------------------------------===//

  auto mlirGPUAsyncTokenType =
      mlir_type_subclass(m, "AsyncTokenType", mlirTypeIsAGPUAsyncTokenType);

  mlirGPUAsyncTokenType.def_classmethod(
      "get",
      [](const nb::object &cls, MlirContext ctx) {
        return cls(mlirGPUAsyncTokenTypeGet(ctx));
      },
      "Gets an instance of AsyncTokenType in the same context", nb::arg("cls"),
      nb::arg("ctx").none() = nb::none());

  //===-------------------------------------------------------------------===//
  // ObjectAttr
  //===-------------------------------------------------------------------===//

  mlir_attribute_subclass(m, "ObjectAttr", mlirAttributeIsAGPUObjectAttr)
      .def_classmethod(
          "get",
          [](const nb::object &cls, MlirAttribute target, uint32_t format,
             const nb::bytes &object,
             std::optional<MlirAttribute> mlirObjectProps,
             std::optional<MlirAttribute> mlirKernelsAttr) {
            MlirStringRef objectStrRef = mlirStringRefCreate(
                static_cast<char *>(const_cast<void *>(object.data())),
                object.size());
            return cls(mlirGPUObjectAttrGetWithKernels(
                mlirAttributeGetContext(target), target, format, objectStrRef,
                mlirObjectProps.has_value() ? *mlirObjectProps
                                            : MlirAttribute{nullptr},
                mlirKernelsAttr.has_value() ? *mlirKernelsAttr
                                            : MlirAttribute{nullptr}));
          },
          "cls"_a, "target"_a, "format"_a, "object"_a,
          "properties"_a.none() = nb::none(), "kernels"_a.none() = nb::none(),
          "Gets a gpu.object from parameters.")
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
            return nb::bytes(stringRef.data, stringRef.length);
          })
      .def_property_readonly("properties",
                             [](MlirAttribute self) -> nb::object {
                               if (mlirGPUObjectAttrHasProperties(self))
                                 return nb::cast(
                                     mlirGPUObjectAttrGetProperties(self));
                               return nb::none();
                             })
      .def_property_readonly("kernels", [](MlirAttribute self) -> nb::object {
        if (mlirGPUObjectAttrHasKernels(self))
          return nb::cast(mlirGPUObjectAttrGetKernels(self));
        return nb::none();
      });
}
