//===- IRAffine.cpp - Exports 'ir' module affine related bindings ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"

#include "PybindUtils.h"

#include "mlir-c/AffineMap.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IntegerSet.h"

namespace py = pybind11;
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
static void pyListToVector(py::list list, llvm::SmallVectorImpl<CType> &result,
                           StringRef action) {
  result.reserve(py::len(list));
  for (py::handle item : list) {
    try {
      result.push_back(item.cast<PyType>());
    } catch (py::cast_error &err) {
      std::string msg = (llvm::Twine("Invalid expression when ") + action +
                         " (" + err.what() + ")")
                            .str();
      throw py::cast_error(msg);
    } catch (py::reference_cast_error &err) {
      std::string msg = (llvm::Twine("Invalid expression (None?) when ") +
                         action + " (" + err.what() + ")")
                            .str();
      throw py::cast_error(msg);
    }
  }
}

template <typename PermutationTy>
static bool isPermutation(std::vector<PermutationTy> permutation) {
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
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirAffineExpr);

  PyConcreteAffineExpr() = default;
  PyConcreteAffineExpr(PyMlirContextRef contextRef, MlirAffineExpr affineExpr)
      : BaseTy(std::move(contextRef), affineExpr) {}
  PyConcreteAffineExpr(PyAffineExpr &orig)
      : PyConcreteAffineExpr(orig.getContext(), castFrom(orig)) {}

  static MlirAffineExpr castFrom(PyAffineExpr &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError,
                       Twine("Cannot cast affine expression to ") +
                           DerivedTy::pyClassName + " (from " + origRepr + ")");
    }
    return orig;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyAffineExpr &>());
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
    c.def_static("get", &PyAffineConstantExpr::get, py::arg("value"),
                 py::arg("context") = py::none());
    c.def_property_readonly("value", [](PyAffineConstantExpr &self) {
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
    c.def_static("get", &PyAffineDimExpr::get, py::arg("position"),
                 py::arg("context") = py::none());
    c.def_property_readonly("position", [](PyAffineDimExpr &self) {
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
    c.def_static("get", &PyAffineSymbolExpr::get, py::arg("position"),
                 py::arg("context") = py::none());
    c.def_property_readonly("position", [](PyAffineSymbolExpr &self) {
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
    c.def_property_readonly("lhs", &PyAffineBinaryExpr::lhs);
    c.def_property_readonly("rhs", &PyAffineBinaryExpr::rhs);
  }
};

class PyAffineAddExpr
    : public PyConcreteAffineExpr<PyAffineAddExpr, PyAffineBinaryExpr> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAffineExprIsAAdd;
  static constexpr const char *pyClassName = "AffineAddExpr";
  using PyConcreteAffineExpr::PyConcreteAffineExpr;

  static PyAffineAddExpr get(PyAffineExpr lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineAddExprGet(lhs, rhs);
    return PyAffineAddExpr(lhs.getContext(), expr);
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

  static PyAffineMulExpr get(PyAffineExpr lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineMulExprGet(lhs, rhs);
    return PyAffineMulExpr(lhs.getContext(), expr);
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

  static PyAffineModExpr get(PyAffineExpr lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineModExprGet(lhs, rhs);
    return PyAffineModExpr(lhs.getContext(), expr);
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

  static PyAffineFloorDivExpr get(PyAffineExpr lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineFloorDivExprGet(lhs, rhs);
    return PyAffineFloorDivExpr(lhs.getContext(), expr);
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

  static PyAffineCeilDivExpr get(PyAffineExpr lhs, PyAffineExpr rhs) {
    MlirAffineExpr expr = mlirAffineCeilDivExprGet(lhs, rhs);
    return PyAffineCeilDivExpr(lhs.getContext(), expr);
  }

  static void bindDerived(ClassTy &c) {
    c.def_static("get", &PyAffineCeilDivExpr::get);
  }
};

} // namespace

bool PyAffineExpr::operator==(const PyAffineExpr &other) {
  return mlirAffineExprEqual(affineExpr, other.affineExpr);
}

py::object PyAffineExpr::getCapsule() {
  return py::reinterpret_steal<py::object>(
      mlirPythonAffineExprToCapsule(*this));
}

PyAffineExpr PyAffineExpr::createFromCapsule(py::object capsule) {
  MlirAffineExpr rawAffineExpr = mlirPythonCapsuleToAffineExpr(capsule.ptr());
  if (mlirAffineExprIsNull(rawAffineExpr))
    throw py::error_already_set();
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

  PyAffineMapExprList(PyAffineMap map, intptr_t startIndex = 0,
                      intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirAffineMapGetNumResults(map) : length,
                  step),
        affineMap(map) {}

  intptr_t getNumElements() { return mlirAffineMapGetNumResults(affineMap); }

  PyAffineExpr getElement(intptr_t pos) {
    return PyAffineExpr(affineMap.getContext(),
                        mlirAffineMapGetResult(affineMap, pos));
  }

  PyAffineMapExprList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyAffineMapExprList(affineMap, startIndex, length, step);
  }

private:
  PyAffineMap affineMap;
};
} // end namespace

bool PyAffineMap::operator==(const PyAffineMap &other) {
  return mlirAffineMapEqual(affineMap, other.affineMap);
}

py::object PyAffineMap::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonAffineMapToCapsule(*this));
}

PyAffineMap PyAffineMap::createFromCapsule(py::object capsule) {
  MlirAffineMap rawAffineMap = mlirPythonCapsuleToAffineMap(capsule.ptr());
  if (mlirAffineMapIsNull(rawAffineMap))
    throw py::error_already_set();
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
  PyIntegerSetConstraint(PyIntegerSet set, intptr_t pos) : set(set), pos(pos) {}

  PyAffineExpr getExpr() {
    return PyAffineExpr(set.getContext(),
                        mlirIntegerSetGetConstraint(set, pos));
  }

  bool isEq() { return mlirIntegerSetIsConstraintEq(set, pos); }

  static void bind(py::module &m) {
    py::class_<PyIntegerSetConstraint>(m, "IntegerSetConstraint")
        .def_property_readonly("expr", &PyIntegerSetConstraint::getExpr)
        .def_property_readonly("is_eq", &PyIntegerSetConstraint::isEq);
  }

private:
  PyIntegerSet set;
  intptr_t pos;
};

class PyIntegerSetConstraintList
    : public Sliceable<PyIntegerSetConstraintList, PyIntegerSetConstraint> {
public:
  static constexpr const char *pyClassName = "IntegerSetConstraintList";

  PyIntegerSetConstraintList(PyIntegerSet set, intptr_t startIndex = 0,
                             intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirIntegerSetGetNumConstraints(set) : length,
                  step),
        set(set) {}

  intptr_t getNumElements() { return mlirIntegerSetGetNumConstraints(set); }

  PyIntegerSetConstraint getElement(intptr_t pos) {
    return PyIntegerSetConstraint(set, pos);
  }

  PyIntegerSetConstraintList slice(intptr_t startIndex, intptr_t length,
                                   intptr_t step) {
    return PyIntegerSetConstraintList(set, startIndex, length, step);
  }

private:
  PyIntegerSet set;
};
} // namespace

bool PyIntegerSet::operator==(const PyIntegerSet &other) {
  return mlirIntegerSetEqual(integerSet, other.integerSet);
}

py::object PyIntegerSet::getCapsule() {
  return py::reinterpret_steal<py::object>(
      mlirPythonIntegerSetToCapsule(*this));
}

PyIntegerSet PyIntegerSet::createFromCapsule(py::object capsule) {
  MlirIntegerSet rawIntegerSet = mlirPythonCapsuleToIntegerSet(capsule.ptr());
  if (mlirIntegerSetIsNull(rawIntegerSet))
    throw py::error_already_set();
  return PyIntegerSet(
      PyMlirContext::forContext(mlirIntegerSetGetContext(rawIntegerSet)),
      rawIntegerSet);
}

void mlir::python::populateIRAffine(py::module &m) {
  //----------------------------------------------------------------------------
  // Mapping of PyAffineExpr and derived classes.
  //----------------------------------------------------------------------------
  py::class_<PyAffineExpr>(m, "AffineExpr")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyAffineExpr::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineExpr::createFromCapsule)
      .def("__add__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             return PyAffineAddExpr::get(self, other);
           })
      .def("__mul__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             return PyAffineMulExpr::get(self, other);
           })
      .def("__mod__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             return PyAffineModExpr::get(self, other);
           })
      .def("__sub__",
           [](PyAffineExpr &self, PyAffineExpr &other) {
             auto negOne =
                 PyAffineConstantExpr::get(-1, *self.getContext().get());
             return PyAffineAddExpr::get(self,
                                         PyAffineMulExpr::get(negOne, other));
           })
      .def("__eq__", [](PyAffineExpr &self,
                        PyAffineExpr &other) { return self == other; })
      .def("__eq__",
           [](PyAffineExpr &self, py::object &other) { return false; })
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
      .def_property_readonly(
          "context",
          [](PyAffineExpr &self) { return self.getContext().getObject(); })
      .def_static(
          "get_add", &PyAffineAddExpr::get,
          "Gets an affine expression containing a sum of two expressions.")
      .def_static(
          "get_mul", &PyAffineMulExpr::get,
          "Gets an affine expression containing a product of two expressions.")
      .def_static("get_mod", &PyAffineModExpr::get,
                  "Gets an affine expression containing the modulo of dividing "
                  "one expression by another.")
      .def_static("get_floor_div", &PyAffineFloorDivExpr::get,
                  "Gets an affine expression containing the rounded-down "
                  "result of dividing one expression by another.")
      .def_static("get_ceil_div", &PyAffineCeilDivExpr::get,
                  "Gets an affine expression containing the rounded-up result "
                  "of dividing one expression by another.")
      .def_static("get_constant", &PyAffineConstantExpr::get, py::arg("value"),
                  py::arg("context") = py::none(),
                  "Gets a constant affine expression with the given value.")
      .def_static(
          "get_dim", &PyAffineDimExpr::get, py::arg("position"),
          py::arg("context") = py::none(),
          "Gets an affine expression of a dimension at the given position.")
      .def_static(
          "get_symbol", &PyAffineSymbolExpr::get, py::arg("position"),
          py::arg("context") = py::none(),
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
  py::class_<PyAffineMap>(m, "AffineMap")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyAffineMap::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAffineMap::createFromCapsule)
      .def("__eq__",
           [](PyAffineMap &self, PyAffineMap &other) { return self == other; })
      .def("__eq__", [](PyAffineMap &self, py::object &other) { return false; })
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
      .def_property_readonly(
          "context",
          [](PyAffineMap &self) { return self.getContext().getObject(); },
          "Context that owns the Affine Map")
      .def(
          "dump", [](PyAffineMap &self) { mlirAffineMapDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t dimCount, intptr_t symbolCount, py::list exprs,
             DefaultingPyMlirContext context) {
            SmallVector<MlirAffineExpr> affineExprs;
            pyListToVector<PyAffineExpr, MlirAffineExpr>(
                exprs, affineExprs, "attempting to create an AffineMap");
            MlirAffineMap map =
                mlirAffineMapGet(context->get(), dimCount, symbolCount,
                                 affineExprs.size(), affineExprs.data());
            return PyAffineMap(context->getRef(), map);
          },
          py::arg("dim_count"), py::arg("symbol_count"), py::arg("exprs"),
          py::arg("context") = py::none(),
          "Gets a map with the given expressions as results.")
      .def_static(
          "get_constant",
          [](intptr_t value, DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapConstantGet(context->get(), value);
            return PyAffineMap(context->getRef(), affineMap);
          },
          py::arg("value"), py::arg("context") = py::none(),
          "Gets an affine map with a single constant result")
      .def_static(
          "get_empty",
          [](DefaultingPyMlirContext context) {
            MlirAffineMap affineMap = mlirAffineMapEmptyGet(context->get());
            return PyAffineMap(context->getRef(), affineMap);
          },
          py::arg("context") = py::none(), "Gets an empty affine map.")
      .def_static(
          "get_identity",
          [](intptr_t nDims, DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapMultiDimIdentityGet(context->get(), nDims);
            return PyAffineMap(context->getRef(), affineMap);
          },
          py::arg("n_dims"), py::arg("context") = py::none(),
          "Gets an identity map with the given number of dimensions.")
      .def_static(
          "get_minor_identity",
          [](intptr_t nDims, intptr_t nResults,
             DefaultingPyMlirContext context) {
            MlirAffineMap affineMap =
                mlirAffineMapMinorIdentityGet(context->get(), nDims, nResults);
            return PyAffineMap(context->getRef(), affineMap);
          },
          py::arg("n_dims"), py::arg("n_results"),
          py::arg("context") = py::none(),
          "Gets a minor identity map with the given number of dimensions and "
          "results.")
      .def_static(
          "get_permutation",
          [](std::vector<unsigned> permutation,
             DefaultingPyMlirContext context) {
            if (!isPermutation(permutation))
              throw py::cast_error("Invalid permutation when attempting to "
                                   "create an AffineMap");
            MlirAffineMap affineMap = mlirAffineMapPermutationGet(
                context->get(), permutation.size(), permutation.data());
            return PyAffineMap(context->getRef(), affineMap);
          },
          py::arg("permutation"), py::arg("context") = py::none(),
          "Gets an affine map that permutes its inputs.")
      .def("get_submap",
           [](PyAffineMap &self, std::vector<intptr_t> &resultPos) {
             intptr_t numResults = mlirAffineMapGetNumResults(self);
             for (intptr_t pos : resultPos) {
               if (pos < 0 || pos >= numResults)
                 throw py::value_error("result position out of bounds");
             }
             MlirAffineMap affineMap = mlirAffineMapGetSubMap(
                 self, resultPos.size(), resultPos.data());
             return PyAffineMap(self.getContext(), affineMap);
           })
      .def("get_major_submap",
           [](PyAffineMap &self, intptr_t nResults) {
             if (nResults >= mlirAffineMapGetNumResults(self))
               throw py::value_error("number of results out of bounds");
             MlirAffineMap affineMap =
                 mlirAffineMapGetMajorSubMap(self, nResults);
             return PyAffineMap(self.getContext(), affineMap);
           })
      .def("get_minor_submap",
           [](PyAffineMap &self, intptr_t nResults) {
             if (nResults >= mlirAffineMapGetNumResults(self))
               throw py::value_error("number of results out of bounds");
             MlirAffineMap affineMap =
                 mlirAffineMapGetMinorSubMap(self, nResults);
             return PyAffineMap(self.getContext(), affineMap);
           })
      .def_property_readonly(
          "is_permutation",
          [](PyAffineMap &self) { return mlirAffineMapIsPermutation(self); })
      .def_property_readonly("is_projected_permutation",
                             [](PyAffineMap &self) {
                               return mlirAffineMapIsProjectedPermutation(self);
                             })
      .def_property_readonly(
          "n_dims",
          [](PyAffineMap &self) { return mlirAffineMapGetNumDims(self); })
      .def_property_readonly(
          "n_inputs",
          [](PyAffineMap &self) { return mlirAffineMapGetNumInputs(self); })
      .def_property_readonly(
          "n_symbols",
          [](PyAffineMap &self) { return mlirAffineMapGetNumSymbols(self); })
      .def_property_readonly("results", [](PyAffineMap &self) {
        return PyAffineMapExprList(self);
      });
  PyAffineMapExprList::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of PyIntegerSet.
  //----------------------------------------------------------------------------
  py::class_<PyIntegerSet>(m, "IntegerSet")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyIntegerSet::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyIntegerSet::createFromCapsule)
      .def("__eq__", [](PyIntegerSet &self,
                        PyIntegerSet &other) { return self == other; })
      .def("__eq__", [](PyIntegerSet &self, py::object other) { return false; })
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
      .def_property_readonly(
          "context",
          [](PyIntegerSet &self) { return self.getContext().getObject(); })
      .def(
          "dump", [](PyIntegerSet &self) { mlirIntegerSetDump(self); },
          kDumpDocstring)
      .def_static(
          "get",
          [](intptr_t numDims, intptr_t numSymbols, py::list exprs,
             std::vector<bool> eqFlags, DefaultingPyMlirContext context) {
            if (exprs.size() != eqFlags.size())
              throw py::value_error(
                  "Expected the number of constraints to match "
                  "that of equality flags");
            if (exprs.empty())
              throw py::value_error("Expected non-empty list of constraints");

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
          py::arg("num_dims"), py::arg("num_symbols"), py::arg("exprs"),
          py::arg("eq_flags"), py::arg("context") = py::none())
      .def_static(
          "get_empty",
          [](intptr_t numDims, intptr_t numSymbols,
             DefaultingPyMlirContext context) {
            MlirIntegerSet set =
                mlirIntegerSetEmptyGet(context->get(), numDims, numSymbols);
            return PyIntegerSet(context->getRef(), set);
          },
          py::arg("num_dims"), py::arg("num_symbols"),
          py::arg("context") = py::none())
      .def("get_replaced",
           [](PyIntegerSet &self, py::list dimExprs, py::list symbolExprs,
              intptr_t numResultDims, intptr_t numResultSymbols) {
             if (static_cast<intptr_t>(dimExprs.size()) !=
                 mlirIntegerSetGetNumDims(self))
               throw py::value_error(
                   "Expected the number of dimension replacement expressions "
                   "to match that of dimensions");
             if (static_cast<intptr_t>(symbolExprs.size()) !=
                 mlirIntegerSetGetNumSymbols(self))
               throw py::value_error(
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
           })
      .def_property_readonly("is_canonical_empty",
                             [](PyIntegerSet &self) {
                               return mlirIntegerSetIsCanonicalEmpty(self);
                             })
      .def_property_readonly(
          "n_dims",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumDims(self); })
      .def_property_readonly(
          "n_symbols",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumSymbols(self); })
      .def_property_readonly(
          "n_inputs",
          [](PyIntegerSet &self) { return mlirIntegerSetGetNumInputs(self); })
      .def_property_readonly("n_equalities",
                             [](PyIntegerSet &self) {
                               return mlirIntegerSetGetNumEqualities(self);
                             })
      .def_property_readonly("n_inequalities",
                             [](PyIntegerSet &self) {
                               return mlirIntegerSetGetNumInequalities(self);
                             })
      .def_property_readonly("constraints", [](PyIntegerSet &self) {
        return PyIntegerSetConstraintList(self);
      });
  PyIntegerSetConstraint::bind(m);
  PyIntegerSetConstraintList::bind(m);
}
