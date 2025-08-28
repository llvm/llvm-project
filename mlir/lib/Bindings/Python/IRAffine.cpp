//===- IRAffine.cpp - Exports 'ir' module affine related bindings ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "IRModule.h"
#include "NanobindUtils.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
#include "mlir-c/IntegerSet.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

/// Attempts to populate `result` with the content of `list` casted to the
/// appropriate type (Python and C types are provided as template arguments).
/// Throws errors in case of failure, using "action" to describe what the caller
/// was attempting to do.
template <typename PyType, typename CType>
static void pyListToVector(const nb::list &list,
                           llvm::SmallVectorImpl<CType> &result,
                           StringRef action) {
  result.reserve(nb::len(list));
  for (nb::handle item : list) {
    try {
      result.push_back(nb::cast<PyType>(item));
    } catch (nb::cast_error &err) {
      std::string msg = (llvm::Twine("Invalid expression when ") + action +
                         " (" + err.what() + ")")
                            .str();
      throw std::runtime_error(msg.c_str());
    } catch (std::runtime_error &err) {
      std::string msg = (llvm::Twine("Invalid expression (None?) when ") +
                         action + " (" + err.what() + ")")
                            .str();
      throw std::runtime_error(msg.c_str());
    }
  }
}

template <typename PermutationTy>
static bool isPermutation(const std::vector<PermutationTy> &permutation) {
  llvm::SmallVector<bool, 8> seen(permutation.size(), false);
  for (auto val : permutation) {
    if (val < permutation.size()) {
      if (seen[val])
        return false;
      seen[val] = true;
      continue;
    }
    return false;
  }
  return true;
}

namespace {

/// CRTP base class for Python MLIR affine expressions that subclass AffineExpr
/// and should be castable from it. Intermediate hierarchy classes can be
/// modeled by specifying BaseTy.
template <typename DerivedTy, typename BaseTy = PyAffineExpr>
class PyConcreteAffineExpr : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = nb::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirAffineExpr);

  PyConcreteAffineExpr() = default;
  PyConcreteAffineExpr(PyMlirContextRef contextRef, MlirAffineExpr affineExpr)
      : BaseTy(std::move(contextRef), affineExpr) {}
  PyConcreteAffineExpr(PyAffineExpr &orig)
      : PyConcreteAffineExpr(orig.getContext(), castFrom(orig)) {}

  static MlirAffineExpr castFrom(PyAffineExpr &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(orig)));
      throw nb::value_error((Twine("Cannot cast affine expression to ") +
                             DerivedTy::pyClassName + " (from " + origRepr +
                             ")")
                                .str()
                                .c_str());
    }
    return orig;
  }

  static void bind(nb::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(nb::init<PyAffineExpr &>(), nb::arg("expr"));
    cls.def_static(
        "isinstance",
        [](PyAffineExpr &otherAffineExpr) -> bool {
          return DerivedTy::isaFunction(otherAffineExpr);
        },
        nb::arg("other"));
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyAffineConstantExpr : public PyConcreteAffineExpr<PyAffineConstantExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAConstant;
  static constexpr const char *pyClassName = "AffineConstantExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineConstantExpr get(intptr_t value,
                                  DefaultingPyMlirContext context) {
    MlirAffineExpr affineExpr =
        mlirAffineConstantExprGet(context->get(), static_cast<int64_t>(value));
    return PyAffineConstantExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineConstantExpr::get, nb::arg("value"),
                 nb::arg("context").none() = nb::none());
    c.def_prop_ro("value", [](PyAffineConstantExpr &self) {
      return mlirAffineConstantExprGetValue(self);
    });
  }
};

class PyAffineDimExpr : public PyConcreteAffineExpr<PyAffineDimExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsADim;
  static constexpr const char *pyClassName = "AffineDimExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineDimExpr get(intptr_t pos, DefaultingPyMlirContext context) {
    MlirAffineExpr affineExpr = mlirAffineDimExprGet(context->get(), pos);
    return PyAffineDimExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineDimExpr::get, nb::arg("position"),
                 nb::arg("context").none() = nb::none());
    c.def_prop_ro("position", [](PyAffineDimExpr &self) {
      return mlirAffineDimExprGetPosition(self);
    });
  }
};

class PyAffineSymbolExpr : public PyConcreteAffineExpr<PyAffineSymbolExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsASymbol;
  static constexpr const char *pyClassName = "AffineSymbolExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineSymbolExpr get(intptr_t pos, DefaultingPyMlirContext context) {
    MlirAffineExpr affineExpr = mlirAffineSymbolExprGet(context->get(), pos);
    return PyAffineSymbolExpr(context->getRef(), affineExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineSymbolExpr::get, nb::arg("position"),
                 nb::arg("context").none() = nb::none());
    c.def_prop_ro("position", [](PyAffineSymbolExpr &self) {
      return mlirAffineSymbolExprGetPosition(self);
    });
  }
};

class PyAffineBinaryExpr : public PyConcreteAffineExpr<PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsABinary;
  static constexpr const char *pyClassName = "AffineBinaryExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  PyAffineExpr lhs() {
    MlirAffineExpr lhsExpr = mlirAffineBinaryOpExprGetLHS(get());
    return PyAffineExpr(getContext(), lhsExpr);
  }

  PyAffineExpr rhs() {
    MlirAffineExpr rhsExpr = mlirAffineBinaryOpExprGetRHS(get());
    return PyAffineExpr(getContext(), rhsExpr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("lhs", &PyAffineBinaryExpr::lhs);
    c.def_prop_ro("rhs", &PyAffineBinaryExpr::rhs);
  }
};

class PyAffineAddExpr
    : public PyConcreteAffineExpr<PyAffineAddExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAAdd;
  static constexpr const char *pyClassName = "AffineAddExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineAddExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    MlirAffineExpr expr = mlirAffineAddExprGet(lhs, rhs);
    return PyAffineAddExpr(lhs.getContext(), expr);
  }

  static PyAffineAddExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    MlirAffineExpr expr = mlirAffineAddExprGet(
        lhs, mlirAffineConstantExprGet(mlirAffineExprGetContext(lhs), rhs));
    return PyAffineAddExpr(lhs.getContext(), expr);
  }

  static PyAffineAddExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineAddExprGet(
        mlirAffineConstantExprGet(mlirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineAddExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineAddExpr::get);
  }
};

class PyAffineMulExpr
    : public PyConcreteAffineExpr<PyAffineMulExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAMul;
  static constexpr const char *pyClassName = "AffineMulExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineMulExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    MlirAffineExpr expr = mlirAffineMulExprGet(lhs, rhs);
    return PyAffineMulExpr(lhs.getContext(), expr);
  }

  static PyAffineMulExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    MlirAffineExpr expr = mlirAffineMulExprGet(
        lhs, mlirAffineConstantExprGet(mlirAffineExprGetContext(lhs), rhs));
    return PyAffineMulExpr(lhs.getContext(), expr);
  }

  static PyAffineMulExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineMulExprGet(
        mlirAffineConstantExprGet(mlirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineMulExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineMulExpr::get);
  }
};

class PyAffineModExpr
    : public PyConcreteAffineExpr<PyAffineModExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAMod;
  static constexpr const char *pyClassName = "AffineModExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineModExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    MlirAffineExpr expr = mlirAffineModExprGet(lhs, rhs);
    return PyAffineModExpr(lhs.getContext(), expr);
  }

  static PyAffineModExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    MlirAffineExpr expr = mlirAffineModExprGet(
        lhs, mlirAffineConstantExprGet(mlirAffineExprGetContext(lhs), rhs));
    return PyAffineModExpr(lhs.getContext(), expr);
  }

  static PyAffineModExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineModExprGet(
        mlirAffineConstantExprGet(mlirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineModExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineModExpr::get);
  }
};

class PyAffineFloorDivExpr
    : public PyConcreteAffineExpr<PyAffineFloorDivExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAFloorDiv;
  static constexpr const char *pyClassName = "AffineFloorDivExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineFloorDivExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    MlirAffineExpr expr = mlirAffineFloorDivExprGet(lhs, rhs);
    return PyAffineFloorDivExpr(lhs.getContext(), expr);
  }

  static PyAffineFloorDivExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    MlirAffineExpr expr = mlirAffineFloorDivExprGet(
        lhs, mlirAffineConstantExprGet(mlirAffineExprGetContext(lhs), rhs));
    return PyAffineFloorDivExpr(lhs.getContext(), expr);
  }

  static PyAffineFloorDivExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineFloorDivExprGet(
        mlirAffineConstantExprGet(mlirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineFloorDivExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineFloorDivExpr::get);
  }
};

class PyAffineCeilDivExpr
    : public PyConcreteAffineExpr<PyAffineCeilDivExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsACeilDiv;
  static constexpr const char *pyClassName = "AffineCeilDivExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineCeilDivExpr get(PyAffineExpr lhs, const PyAffineExpr &rhs) {
    MlirAffineExpr expr = mlirAffineCeilDivExprGet(lhs, rhs);
    return PyAffineCeilDivExpr(lhs.getContext(), expr);
  }

  static PyAffineCeilDivExpr getRHSConstant(PyAffineExpr lhs, intptr_t rhs) {
    MlirAffineExpr expr = mlirAffineCeilDivExprGet(
        lhs, mlirAffineConstantExprGet(mlirAffineExprGetContext(lhs), rhs));
    return PyAffineCeilDivExpr(lhs.getContext(), expr);
  }

  static PyAffineCeilDivExpr getLHSConstant(intptr_t lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineCeilDivExprGet(
        mlirAffineConstantExprGet(mlirAffineExprGetContext(rhs), lhs), rhs);
    return PyAffineCeilDivExpr(rhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineCeilDivExpr::get);
  }
};

} // namespace

bool PyAffineExpr::operator==(const PyAffineExpr &other) const {
  return mlirAffineExprEqual(affineExpr, other.affineExpr);
}

nb::object PyAffineExpr::getCapsule() {
  return nb::steal<nb::object>(mlirPythonAffineExprToCapsule(*this));
}

PyAffineExpr PyAffineExpr::createFromCapsule(const nb::object &capsule) {
  MlirAffineExpr rawAffineExpr = mlirPythonCapsuleToAffineExpr(capsule.ptr());
  if (mlirAffineExprIsNull(rawAffineExpr))
    throw nb::python_error();
  return PyAffineExpr(
      PyMlirContext::forContext(mlirAffineExprGetContext(rawAffineExpr)),
      rawAffineExpr);
}

//------------------------------------------------------------------------------
// PyAffineMap and utilities.
//------------------------------------------------------------------------------
namespace {

/// A list of expressions contained in an affine map. Internally these are
/// stored as a consecutive array leading to inexpensive random access. Both
/// the map and the expression are owned by the context so we need not bother
/// with lifetime extension.
class PyAffineMapExprList
    : public Sliceable<PyAffineMapExprList, PyAffineExpr> {
public:
  static constexpr const char *pyClassName = "AffineExprList";

  PyAffineMapExprList(const PyAffineMap &map, intptr_t startIndex = 0,
                      intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirAffineMapGetNumResults(map) : length,
                  step),
        affineMap(map) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyAffineMapExprList, PyAffineExpr>;

  intptr_t getRawNumElements() { return mlirAffineMapGetNumResults(affineMap); }

  PyAffineExpr getRawElement(intptr_t pos) {
    return PyAffineExpr(affineMap.getContext(),
                        mlirAffineMapGetResult(affineMap, pos));
  }

  PyAffineMapExprList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyAffineMapExprList(affineMap, startIndex, length, step);
  }

  PyAffineMap affineMap;
};
} // namespace

bool PyAffineMap::operator==(const PyAffineMap &other) const {
  return mlirAffineMapEqual(affineMap, other.affineMap);
}

nb::object PyAffineMap::getCapsule() {
  return nb::steal<nb::object>(mlirPythonAffineMapToCapsule(*this));
}

PyAffineMap PyAffineMap::createFromCapsule(const nb::object &capsule) {
  MlirAffineMap rawAffineMap = mlirPythonCapsuleToAffineMap(capsule.ptr());
  if (mlirAffineMapIsNull(rawAffineMap))
    throw nb::python_error();
  return PyAffineMap(
      PyMlirContext::forContext(mlirAffineMapGetContext(rawAffineMap)),
      rawAffineMap);
}

//------------------------------------------------------------------------------
// PyIntegerSet and utilities.
//------------------------------------------------------------------------------
namespace {

class PyIntegerSetConstraint {
public:
  PyIntegerSetConstraint(PyIntegerSet set, intptr_t pos)
      : set(std::move(set)), pos(pos) {}

  PyAffineExpr getExpr() {
    return PyAffineExpr(set.getContext(),
                        mlirIntegerSetGetConstraint(set, pos));
  }

  bool isEq() { return mlirIntegerSetIsConstraintEq(set, pos); }

  static void bind(nb::module_ &m) {
    nb::class_<PyIntegerSetConstraint>(m, "IntegerSetConstraint")
        .def_prop_ro("expr", &PyIntegerSetConstraint::getExpr)
        .def_prop_ro("is_eq", &PyIntegerSetConstraint::isEq);
  }

private:
  PyIntegerSet set;
  intptr_t pos;
};

class PyIntegerSetConstraintList
    : public Sliceable<PyIntegerSetConstraintList, PyIntegerSetConstraint> {
public:
  static constexpr const char *pyClassName = "IntegerSetConstraintList";

  PyIntegerSetConstraintList(const PyIntegerSet &set, intptr_t startIndex = 0,
                             intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirIntegerSetGetNumConstraints(set) : length,
                  step),
        set(set) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyIntegerSetConstraintList, PyIntegerSetConstraint>;

  intptr_t getRawNumElements() { return mlirIntegerSetGetNumConstraints(set); }

  PyIntegerSetConstraint getRawElement(intptr_t pos) {
    return PyIntegerSetConstraint(set, pos);
  }

  PyIntegerSetConstraintList slice(intptr_t startIndex, intptr_t length,
                                   intptr_t step) {
    return PyIntegerSetConstraintList(set, startIndex, length, step);
  }

  PyIntegerSet set;
};
} // namespace

bool PyIntegerSet::operator==(const PyIntegerSet &other) const {
  return mlirIntegerSetEqual(integerSet, other.integerSet);
}

nb::object PyIntegerSet::getCapsule() {
  return nb::steal<nb::object>(mlirPythonIntegerSetToCapsule(*this));
}

PyIntegerSet PyIntegerSet::createFromCapsule(const nb::object &capsule) {
  MlirIntegerSet rawIntegerSet = mlirPythonCapsuleToIntegerSet(capsule.ptr());
  if (mlirIntegerSetIsNull(rawIntegerSet))
    throw nb::python_error();
  return PyIntegerSet(
      PyMlirContext::forContext(mlirIntegerSetGetContext(rawIntegerSet)),
      rawIntegerSet);
}

void mlir::python::populateIRAffine(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Mapping of PyAffineExpr and derived classes.
  //----------------------------------------------------------------------------
  nb::class_<PyAffineExpr>(m, "AffineExpr")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyAffineExpr::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineExpr::createFromCapsule)
      .def("__add__", &PyAffineAddExpr::get)
      .def("__add__", &PyAffineAddExpr::getRHSConstant)
      .def("__radd__", &PyAffineAddExpr::getRHSConstant)
      .def("__mul__", &PyAffineMulExpr::get)
      .def("__mul__", &PyAffineMulExpr::getRHSConstant)
      .def("__rmul__", &PyAffineMulExpr::getRHSConstant)
      .def("__mod__", &PyAffineModExpr::get)
      .def("__mod__", &PyAffineModExpr::getRHSConstant)
      .def("__rmod__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineModExpr::get(
                 PyAffineConstantExpr::get(other, *self.getContext().get()),
                 self);
           })
      .def("__sub__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             auto negOne =
                 PyAffineConstantExpr::get(-1, *self.getContext().get());
             return PyAffineAddExpr::get(self,
                                         PyAffineMulExpr::get(negOne, other));
           })
      .def("__sub__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineAddExpr::get(
                 self,
                 PyAffineConstantExpr::get(-other, *self.getContext().get()));
           })
      .def("__rsub__",
           [](PyAffineExpr &self, intptr_t other) {
             return PyAffineAddExpr::getLHSConstant(
                 other, PyAffineMulExpr::getLHSConstant(-1, self));
           })
      .def("__eq__", [](PyAffineExpr &self,
                        PyAffineExpr &other) { return self == other; })
      .def("__eq__",
           [](PyAffineExpr &self, nb::object &other) { return false; })
      .def("__str__",
           [](PyAffineExpr &self) {
             PyPrintAccumulator printAccum;
             mlirAffineExprPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyAffineExpr &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("AffineExpr(");
             mlirAffineExprPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyAffineExpr &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def_prop_ro(
          "context",
          [](PyAffineExpr &self) { return self.getContext().getObject(); })
      .def("compose",
           [](PyAffineExpr &self, PyAffineMap &other) {
             return PyAffineExpr(self.getContext(),
                                 mlirAffineExprCompose(self, other));
           })
      .def(
          "shift_dims",
          [](PyAffineExpr &self, uint32_t numDims, uint32_t shift,
             uint32_t offset) {
            return PyAffineExpr(
                self.getContext(),
                mlirAffineExprShiftDims(self, numDims, shift, offset));
          },
          nb::arg("num_dims"), nb::arg("shift"), nb::arg("offset").none() = 0)
      .def(
          "shift_symbols",
          [](PyAffineExpr &self, uint32_t numSymbols, uint32_t shift,
             uint32_t offset) {
            return PyAffineExpr(
                self.getContext(),
                mlirAffineExprShiftSymbols(self, numSymbols, shift, offset));
          },
          nb::arg("num_symbols"), nb::arg("shift"),
          nb::arg("offset").none() = 0)
      .def_static(
          "simplify_affine_expr",
          [](PyAffineExpr &self, uint32_t numDims, uint32_t numSymbols) {
            return PyAffineExpr(
                self.getContext(),
                mlirSimplifyAffineExpr(self, numDims, numSymbols));
          },
          nb::arg("expr"), nb::arg("num_dims"), nb::arg("num_symbols"),
          "Simplify an affine expression by flattening and some amount of "
          "simple analysis.")
      .def_static(
          "get_add", &PyAffineAddExpr::get,
          "Gets an affine expression containing a sum of two expressions.")
      .def_static("get_add", &PyAffineAddExpr::getLHSConstant,
                  "Gets an affine expression containing a sum of a constant "
                  "and another expression.")
      .def_static("get_add", &PyAffineAddExpr::getRHSConstant,
                  "Gets an affine expression containing a sum of an expression "
                  "and a constant.")
      .def_static(
          "get_mul", &PyAffineMulExpr::get,
          "Gets an affine expression containing a product of two expressions.")
      .def_static("get_mul", &PyAffineMulExpr::getLHSConstant,
                  "Gets an affine expression containing a product of a "
                  "constant and another expression.")
      .def_static("get_mul", &PyAffineMulExpr::getRHSConstant,
                  "Gets an affine expression containing a product of an "
                  "expression and a constant.")
      .def_static("get_mod", &PyAffineModExpr::get,
                  "Gets an affine expression containing the modulo of dividing "
                  "one expression by another.")
      .def_static("get_mod", &PyAffineModExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the modulo of "
                  "dividing a constant by an expression.")
      .def_static("get_mod", &PyAffineModExpr::getRHSConstant,
                  "Gets an affine expression containing the module of dividing"
                  "an expression by a constant.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::get,
                  "Gets an affine expression containing the rounded-down "
                  "result of dividing one expression by another.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the rounded-down "
                  "result of dividing a constant by an expression.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::getRHSConstant,
                  "Gets an affine expression containing the rounded-down "
                  "result of dividing an expression by a constant.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::get,
                  "Gets an affine expression containing the rounded-up result "
                  "of dividing one expression by another.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::getLHSConstant,
                  "Gets a semi-affine expression containing the rounded-up "
                  "result of dividing a constant by an expression.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::getRHSConstant,
                  "Gets an affine expression containing the rounded-up result "
                  "of dividing an expression by a constant.")
      .def_static("get_constant", &PyAffineConstantExpr::get, nb::arg("value"),
                  nb::arg("context").none() = nb::none(),
                  "Gets a constant affine expression with the given value.")
      .def_static(
          "get_dim", &PyAffineDimExpr::get, nb::arg("position"),
          nb::arg("context").none() = nb::none(),
          "Gets an affine expression of a dimension at the given position.")
      .def_static(
          "get_symbol", &PyAffineSymbolExpr::get, nb::arg("position"),
          nb::arg("context").none() = nb::none(),
          "Gets an affine expression of a symbol at the given position.")
      .def(
          "dump", [](PyAffineExpr &self) { mlirAffineExprDump(self); },
          kDumpDocstring);
  PyAffineConstantExpr::bind(m);
  PyAffineDimExpr::bind(m);
  PyAffineSymbolExpr::bind(m);
  PyAffineBinaryExpr::bind(m);
  PyAffineAddExpr::bind(m);
  PyAffineMulExpr::bind(m);
  PyAffineModExpr::bind(m);
  PyAffineFloorDivExpr::bind(m);
  PyAffineCeilDivExpr::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyAffineMap.
  //----------------------------------------------------------------------------
  nb::class_<PyAffineMap>(m, "AffineMap")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyAffineMap::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineMap::createFromCapsule)
      .def("__eq__",
           [](PyAffineMap &self, PyAffineMap &other) { return self == other; })
      .def("__eq__", [](PyAffineMap &self, nb::object &other) { return false; })
      .def("__str__",
           [](PyAffineMap &self) {
             PyPrintAccumulator printAccum;
             mlirAffineMapPrint(self, printAccum.getCallback(),
                                printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyAffineMap &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("AffineMap(");
             mlirAffineMapPrint(self, printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyAffineMap &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def_static("compress_unused_symbols",
                  [](const nb::list &affineMaps,
                     DefaultingPyMlirContext context) {
                    SmallVector<MlirAffineMap> maps;
                    pyListToVector<PyAffineMap, MlirAffineMap>(
                        affineMaps, maps, "attempting to create an AffineMap");
                    std::vector<MlirAffineMap> compressed(affineMaps.size());
                    auto populate = [](void *result, intptr_t idx,
                                       MlirAffineMap m) {
                      static_cast<MlirAffineMap *>(result)[idx] = (m);
                    };
                    mlirAffineMapCompressUnusedSymbols(
                        maps.data(), maps.size(), compressed.data(), populate);
                    std::vector<PyAffineMap> res;
                    res.reserve(compressed.size());
                    for (auto m : compressed)
                      res.emplace_back(context->getRef(), m);
                    return res;
                  })
      .def_prop_ro(
          "context",
          [](PyAffineMap &self) { return self.getContext().getObject(); },
          "Context that owns the Affine Map")
      .def(
          "dump", [](PyAffineMap &self) { mlirAffineMapDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t dimCount, intptr_t symbolCount, const nb::list &exprs,
             DefaultingPyMlirContext context) {
            SmallVector<MlirAffineExpr> affineExprs;
            pyListToVector<PyAffineExpr, MlirAffineExpr>(
                exprs, affineExprs, "attempting to create an AffineMap");
            MlirAffineMap map =
                mlirAffineMapGet(context->get(), dimCount, symbolCount,
                                 affineExprs.size(), affineExprs.data());
            return PyAffineMap(context->getRef(), map);
          },
          nb::arg("dim_count"), nb::arg("symbol_count"), nb::arg("exprs"),
          nb::arg("context").none() = nb::none(),
          "Gets a map with the given expressions as results.")
      .def_static(
          "get_constant",
          [](intptr_t value, DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapConstantGet(context->get(), value);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("value"), nb::arg("context").none() = nb::none(),
          "Gets an affine map with a single constant result")
      .def_static(
          "get_empty",
          [](DefaultingPyMlirContext context) {
            MlirAffineMap affineMap = mlirAffineMapEmptyGet(context->get());
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("context").none() = nb::none(), "Gets an empty affine map.")
      .def_static(
          "get_identity",
          [](intptr_t nDims, DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapMultiDimIdentityGet(context->get(), nDims);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("n_dims"), nb::arg("context").none() = nb::none(),
          "Gets an identity map with the given number of dimensions.")
      .def_static(
          "get_minor_identity",
          [](intptr_t nDims, intptr_t nResults,
             DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapMinorIdentityGet(context->get(), nDims, nResults);
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("n_dims"), nb::arg("n_results"),
          nb::arg("context").none() = nb::none(),
          "Gets a minor identity map with the given number of dimensions and "
          "results.")
      .def_static(
          "get_permutation",
          [](std::vector<unsigned> permutation,
             DefaultingPyMlirContext context) {
            if (!isPermutation(permutation))
              throw std::runtime_error("Invalid permutation when attempting to "
                                       "create an AffineMap");
            MlirAffineMap affineMap = mlirAffineMapPermutationGet(
                context->get(), permutation.size(), permutation.data());
            return PyAffineMap(context->getRef(), affineMap);
          },
          nb::arg("permutation"), nb::arg("context").none() = nb::none(),
          "Gets an affine map that permutes its inputs.")
      .def(
          "get_submap",
          [](PyAffineMap &self, std::vector<intptr_t> &resultPos) {
            intptr_t numResults = mlirAffineMapGetNumResults(self);
            for (intptr_t pos : resultPos) {
              if (pos < 0 || pos >= numResults)
                throw nb::value_error("result position out of bounds");
            }
            MlirAffineMap affineMap = mlirAffineMapGetSubMap(
                self, resultPos.size(), resultPos.data());
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("result_positions"))
      .def(
          "get_major_submap",
          [](PyAffineMap &self, intptr_t nResults) {
            if (nResults >= mlirAffineMapGetNumResults(self))
              throw nb::value_error("number of results out of bounds");
            MlirAffineMap affineMap =
                mlirAffineMapGetMajorSubMap(self, nResults);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("n_results"))
      .def(
          "get_minor_submap",
          [](PyAffineMap &self, intptr_t nResults) {
            if (nResults >= mlirAffineMapGetNumResults(self))
              throw nb::value_error("number of results out of bounds");
            MlirAffineMap affineMap =
                mlirAffineMapGetMinorSubMap(self, nResults);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("n_results"))
      .def(
          "replace",
          [](PyAffineMap &self, PyAffineExpr &expression,
             PyAffineExpr &replacement, intptr_t numResultDims,
             intptr_t numResultSyms) {
            MlirAffineMap affineMap = mlirAffineMapReplace(
                self, expression, replacement, numResultDims, numResultSyms);
            return PyAffineMap(self.getContext(), affineMap);
          },
          nb::arg("expr"), nb::arg("replacement"), nb::arg("n_result_dims"),
          nb::arg("n_result_syms"))
      .def_prop_ro(
          "is_permutation",
          [](PyAffineMap &self) { return mlirAffineMapIsPermutation(self); })
      .def_prop_ro("is_projected_permutation",
                   [](PyAffineMap &self) {
                     return mlirAffineMapIsProjectedPermutation(self);
                   })
      .def_prop_ro(
          "n_dims",
          [](PyAffineMap &self) { return mlirAffineMapGetNumDims(self); })
      .def_prop_ro(
          "n_inputs",
          [](PyAffineMap &self) { return mlirAffineMapGetNumInputs(self); })
      .def_prop_ro(
          "n_symbols",
          [](PyAffineMap &self) { return mlirAffineMapGetNumSymbols(self); })
      .def_prop_ro("results",
                   [](PyAffineMap &self) { return PyAffineMapExprList(self); });
  PyAffineMapExprList::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyIntegerSet.
  //----------------------------------------------------------------------------
  nb::class_<PyIntegerSet>(m, "IntegerSet")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyIntegerSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyIntegerSet::createFromCapsule)
      .def("__eq__", [](PyIntegerSet &self,
                        PyIntegerSet &other) { return self == other; })
      .def("__eq__",
           [](PyIntegerSet &self, const nb::object &other) { return false; })
      .def("__str__",
           [](PyIntegerSet &self) {
             PyPrintAccumulator printAccum;
             mlirIntegerSetPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             return printAccum.join();
           })
      .def("__repr__",
           [](PyIntegerSet &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("IntegerSet(");
             mlirIntegerSetPrint(self, printAccum.getCallback(),
                                 printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def("__hash__",
           [](PyIntegerSet &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
      .def_prop_ro(
          "context",
          [](PyIntegerSet &self) { return self.getContext().getObject(); })
      .def(
          "dump", [](PyIntegerSet &self) { mlirIntegerSetDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t numDims, intptr_t numSymbols, const nb::list &exprs,
             std::vector<bool> eqFlags, DefaultingPyMlirContext context) {
            if (exprs.size() != eqFlags.size())
              throw nb::value_error(
                  "Expected the number of constraints to match "
                  "that of equality flags");
            if (exprs.size() == 0)
              throw nb::value_error("Expected non-empty list of constraints");

            // Copy over to a SmallVector because std::vector has a
            // specialization for booleans that packs data and does not
            // expose a `bool *`.
            SmallVector<bool, 8> flags(eqFlags.begin(), eqFlags.end());

            SmallVector<MlirAffineExpr> affineExprs;
            pyListToVector<PyAffineExpr>(exprs, affineExprs,
                                         "attempting to create an IntegerSet");
            MlirIntegerSet set = mlirIntegerSetGet(
                context->get(), numDims, numSymbols, exprs.size(),
                affineExprs.data(), flags.data());
            return PyIntegerSet(context->getRef(), set);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"), nb::arg("exprs"),
          nb::arg("eq_flags"), nb::arg("context").none() = nb::none())
      .def_static(
          "get_empty",
          [](intptr_t numDims, intptr_t numSymbols,
             DefaultingPyMlirContext context) {
            MlirIntegerSet set =
                mlirIntegerSetEmptyGet(context->get(), numDims, numSymbols);
            return PyIntegerSet(context->getRef(), set);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"),
          nb::arg("context").none() = nb::none())
      .def(
          "get_replaced",
          [](PyIntegerSet &self, const nb::list &dimExprs,
             const nb::list &symbolExprs, intptr_t numResultDims,
             intptr_t numResultSymbols) {
            if (static_cast<intptr_t>(dimExprs.size()) !=
                mlirIntegerSetGetNumDims(self))
              throw nb::value_error(
                  "Expected the number of dimension replacement expressions "
                  "to match that of dimensions");
            if (static_cast<intptr_t>(symbolExprs.size()) !=
                mlirIntegerSetGetNumSymbols(self))
              throw nb::value_error(
                  "Expected the number of symbol replacement expressions "
                  "to match that of symbols");

            SmallVector<MlirAffineExpr> dimAffineExprs, symbolAffineExprs;
            pyListToVector<PyAffineExpr>(
                dimExprs, dimAffineExprs,
                "attempting to create an IntegerSet by replacing dimensions");
            pyListToVector<PyAffineExpr>(
                symbolExprs, symbolAffineExprs,
                "attempting to create an IntegerSet by replacing symbols");
            MlirIntegerSet set = mlirIntegerSetReplaceGet(
                self, dimAffineExprs.data(), symbolAffineExprs.data(),
                numResultDims, numResultSymbols);
            return PyIntegerSet(self.getContext(), set);
          },
          nb::arg("dim_exprs"), nb::arg("symbol_exprs"),
          nb::arg("num_result_dims"), nb::arg("num_result_symbols"))
      .def_prop_ro("is_canonical_empty",
                   [](PyIntegerSet &self) {
                     return mlirIntegerSetIsCanonicalEmpty(self);
                   })
      .def_prop_ro(
          "n_dims",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumDims(self); })
      .def_prop_ro(
          "n_symbols",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumSymbols(self); })
      .def_prop_ro(
          "n_inputs",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumInputs(self); })
      .def_prop_ro("n_equalities",
                   [](PyIntegerSet &self) {
                     return mlirIntegerSetGetNumEqualities(self);
                   })
      .def_prop_ro("n_inequalities",
                   [](PyIntegerSet &self) {
                     return mlirIntegerSetGetNumInequalities(self);
                   })
      .def_prop_ro("constraints", [](PyIntegerSet &self) {
        return PyIntegerSetConstraintList(self);
      });
  PyIntegerSetConstraint::bind(m);
  PyIntegerSetConstraintList::bind(m);
}
