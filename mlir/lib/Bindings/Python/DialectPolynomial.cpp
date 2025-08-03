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

class PyIntMonomial {
public:
  PyIntMonomial(MlirIntMonomial intMonomial) : intMonomial(intMonomial) {}
  PyIntMonomial(int64_t coeff, uint64_t expo)
      : intMonomial(mlirPolynomialGetIntMonomial(coeff, expo)) {}
  operator MlirIntMonomial() const { return intMonomial; }
  MlirIntMonomial get() { return intMonomial; }

  int64_t getCoefficient() {
    return mlirPolynomialIntMonomialGetCoefficient(this->get());
  }

  uint64_t getExponent() {
    return mlirPolynomialIntMonomialGetExponent(this->get());
  }

private:
  MlirIntMonomial intMonomial;
};

#define MLIR_PYTHON_CAPSULE_INT_POLYNOMIAL                                     \
  MAKE_MLIR_PYTHON_QUALNAME("dialects.polynomial.IntMonomial._CAPIPtr")

static inline MlirIntMonomial
mlirPythonCapsuleToIntMonomial(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_INT_POLYNOMIAL);
  MlirIntMonomial intMonomial = {ptr};
  return intMonomial;
}

static inline PyObject *
mlirPythonIntMonomialToCapsule(MlirIntMonomial intMonomial) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(intMonomial),
                       MLIR_PYTHON_CAPSULE_INT_POLYNOMIAL, nullptr);
}

static inline bool mlirIntMonomialIsNull(MlirIntMonomial intMonomial) {
  return !intMonomial.ptr;
}

namespace pybind11 {
namespace detail {

/// Casts object <-> MlirIntMonomial.
template <>
struct type_caster<MlirIntMonomial> {
  PYBIND11_TYPE_CASTER(MlirIntMonomial, _("MlirIntMonomial"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToIntMonomial(capsule.ptr());
    return !mlirIntMonomialIsNull(value);
  }

  static handle cast(MlirIntMonomial v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonIntMonomialToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("dialects.polynomial"))
        .attr("IntMonomial")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};
} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(_mlirDialectsPolynomial, m) {
  m.doc() = "MLIR Polynomial dialect";

  py::class_<PyIntMonomial>(m, "IntMonomial", py::module_local())
      .def(py::init<PyIntMonomial &>())
      .def(py::init<MlirIntMonomial>())
      .def(py::init<int64_t, uint64_t>())
      .def_property_readonly("coefficient", &PyIntMonomial::getCoefficient)
      .def_property_readonly("exponent", &PyIntMonomial::getExponent)
      .def("__str__", [](PyIntMonomial &self) {
        return std::string("<")
            .append(std::to_string(self.getCoefficient()))
            .append(", ")
            .append(std::to_string(self.getExponent()))
            .append(">");
      });
}
