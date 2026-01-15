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
#include "mlir/Bindings/Python/IRAttributes.h"
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
  static inline const MlirStringRef name = mlirGPUAsyncTokenTypeGetName();
  using Base::Base;

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
  static inline const MlirStringRef name = mlirGPUObjectAttrGetName();
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const PyAttribute &target, uint32_t format, const nb::bytes &object,
           std::optional<PyDictAttribute> mlirObjectProps,
           std::optional<PyAttribute> mlirKernelsAttr,
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

    c.def_prop_ro("target", [](ObjectAttr &self) {
      return PyAttribute(self.getContext(), mlirGPUObjectAttrGetTarget(self))
          .maybeDownCast();
    });
    c.def_prop_ro("format", [](const ObjectAttr &self) {
      return mlirGPUObjectAttrGetFormat(self);
    });
    c.def_prop_ro("object", [](const ObjectAttr &self) {
      MlirStringRef stringRef = mlirGPUObjectAttrGetObject(self);
      return nb::bytes(stringRef.data, stringRef.length);
    });
    c.def_prop_ro(
        "properties", [](ObjectAttr &self) -> std::optional<PyDictAttribute> {
          if (mlirGPUObjectAttrHasProperties(self))
            return PyDictAttribute(self.getContext(),
                                   mlirGPUObjectAttrGetProperties(self));
          return std::nullopt;
        });
    c.def_prop_ro("kernels",
                  [](ObjectAttr &self)
                      -> std::optional<nb::typed<nb::object, PyAttribute>> {
                    if (mlirGPUObjectAttrHasKernels(self))
                      return PyAttribute(self.getContext(),
                                         mlirGPUObjectAttrGetKernels(self))
                          .maybeDownCast();
                    return std::nullopt;
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

  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::gpu::AsyncTokenType::bind(m);
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::gpu::ObjectAttr::bind(m);
}
