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
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace nanobind::literals;
using namespace mlir::python::nanobind_adaptors;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace gpu {
// -----------------------------------------------------------------------------
// AsyncTokenType
// -----------------------------------------------------------------------------

struct AsyncTokenType : PyConcreteType<AsyncTokenType> {
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAGPUAsyncTokenType;
  static constexpr const char *pyClassName = "AsyncTokenType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AsyncTokenType(context->getRef(),
                                mlirGPUAsyncTokenTypeGet(context.get()->get()));
        },
        "Gets an instance of AsyncTokenType in the same context",
        nb::arg("context").none() = nb::none());
  }
};

//===-------------------------------------------------------------------===//
// ObjectAttr
//===-------------------------------------------------------------------===//

struct ObjectAttr : PyConcreteAttribute<ObjectAttr> {
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAGPUObjectAttr;
  static constexpr const char *pyClassName = "ObjectAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirAttribute target, uint32_t format, const nb::bytes &object,
           std::optional<MlirAttribute> mlirObjectProps,
           std::optional<MlirAttribute> mlirKernelsAttr,
           DefaultingPyMlirContext context) {
          MlirStringRef objectStrRef = mlirStringRefCreate(
              static_cast<char *>(const_cast<void *>(object.data())),
              object.size());
          return ObjectAttr(
              context->getRef(),
              mlirGPUObjectAttrGetWithKernels(
                  mlirAttributeGetContext(target), target, format, objectStrRef,
                  mlirObjectProps.has_value() ? *mlirObjectProps
                                              : MlirAttribute{nullptr},
                  mlirKernelsAttr.has_value() ? *mlirKernelsAttr
                                              : MlirAttribute{nullptr}));
        },
        "target"_a, "format"_a, "object"_a, "properties"_a = nb::none(),
        "kernels"_a = nb::none(), "context"_a = nb::none(),
        "Gets a gpu.object from parameters.");

    c.def_prop_ro("target", [](MlirAttribute self) {
      return mlirGPUObjectAttrGetTarget(self);
    });
    c.def_prop_ro("format", [](MlirAttribute self) {
      return mlirGPUObjectAttrGetFormat(self);
    });
    c.def_prop_ro("object", [](MlirAttribute self) {
      MlirStringRef stringRef = mlirGPUObjectAttrGetObject(self);
      return nb::bytes(stringRef.data, stringRef.length);
    });
    c.def_prop_ro("properties", [](MlirAttribute self) -> nb::object {
      if (mlirGPUObjectAttrHasProperties(self))
        return nb::cast(mlirGPUObjectAttrGetProperties(self));
      return nb::none();
    });
    c.def_prop_ro("kernels", [](MlirAttribute self) -> nb::object {
      if (mlirGPUObjectAttrHasKernels(self))
        return nb::cast(mlirGPUObjectAttrGetKernels(self));
      return nb::none();
    });
  }
};
} // namespace gpu
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlirDialectsGPU, m) {
  m.doc() = "MLIR GPU Dialect";

  mlir::python::mlir::gpu::AsyncTokenType::bind(m);
  mlir::python::mlir::gpu::ObjectAttr::bind(m);
}
