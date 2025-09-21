//===- Presburger.cpp - Presburger library --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Presburger.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "llvm/ADT/ScopeExit.h"
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static bool isSignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
         code == 'q';
}

namespace {
struct PyPresburgerIntegerRelation {
  PyPresburgerIntegerRelation(MlirPresburgerIntegerRelation relation)
      : relation(relation) {}
  PyPresburgerIntegerRelation(const PyPresburgerIntegerRelation &other) =
      default;
  PyPresburgerIntegerRelation(PyPresburgerIntegerRelation &&other) noexcept
      : relation(other.relation) {
    other.relation.ptr = nullptr;
  }
  virtual ~PyPresburgerIntegerRelation() {
    if (relation.ptr) {
      mlirPresburgerIntegerRelationDestroy(relation);
      relation.ptr = {nullptr};
    }
  }
  static std::unique_ptr<PyPresburgerIntegerRelation>
  getFromNumConstrainsAndVars(uint64_t numReservedInequalities,
                              uint64_t numReservedEqualities,
                              uint64_t numReservedCols);
  static std::unique_ptr<PyPresburgerIntegerRelation>
  getFromBuffers(py::buffer inequalitiesCoefficients,
                 py::buffer equalityCoefficients, unsigned numDomainVars,
                 unsigned numRangeVars);
  py::object getCapsule();
  static void bind(py::module &module);
  MlirPresburgerIntegerRelation relation{nullptr};
};

/// A utility that enables accessing/modifying the underlying coefficients
/// easier.
struct PyPresburgerTableau {
  enum class Kind { Equalities, Inequalities };
  PyPresburgerTableau(MlirPresburgerIntegerRelation relation, Kind kind)
      : relation(relation), kind(kind) {}
  static void bind(py::module &module);
  int64_t at64(int64_t row, int64_t col) const {
    if (kind == Kind::Equalities)
      return mlirPresburgerIntegerRelationAtEq64(relation, row, col);
    return mlirPresburgerIntegerRelationAtIneq64(relation, row, col);
  }
  MlirPresburgerIntegerRelation relation;
  Kind kind;
};

struct PyPresburgerMaybeOptimum {
  PyPresburgerMaybeOptimum(MlirMaybeOptimum optimum)
      : kind(optimum.kind), integerPoint(optimum.vector.data),
        integerPointSize(optimum.vector.size) {}
  ~PyPresburgerMaybeOptimum() {
    if (integerPoint) {
      delete[] integerPoint;
      integerPoint = nullptr;
    }
  }
  static void bind(py::module &module);
  MlirPresburgerOptimumKind kind;
  const int64_t *integerPoint{nullptr};
  int64_t integerPointSize;
};
} // namespace

std::unique_ptr<PyPresburgerIntegerRelation>
PyPresburgerIntegerRelation::getFromBuffers(py::buffer inequalitiesCoefficients,
                                            py::buffer equalityCoefficients,
                                            unsigned numDomainVars,
                                            unsigned numRangeVars) {
  // Request a contiguous view. In exotic cases, this will cause a copy.
  int flags = PyBUF_ND;
  flags |= PyBUF_FORMAT;
  // Get the view of the inequality coefficients.
  std::unique_ptr<Py_buffer> ineqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(inequalitiesCoefficients.ptr(), ineqView.get(),
                         flags) != 0)
    throw py::error_already_set();
  auto freeIneqBuffer = llvm::make_scope_exit([&]() {
    if (ineqView)
      PyBuffer_Release(ineqView.get());
  });
  if (!PyBuffer_IsContiguous(ineqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");
  if (!isSignedIntegerFormat(ineqView->format) || ineqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(ineqView->format));
  if (ineqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d inequality coefficients but got rank ") +
        std::to_string(ineqView->ndim));
  unsigned numInequalities = ineqView->shape[0];
  // Get the view of the eequality coefficients.
  std::unique_ptr<Py_buffer> eqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(equalityCoefficients.ptr(), eqView.get(), flags) != 0)
    throw py::error_already_set();
  auto freeEqBuffer = llvm::make_scope_exit([&]() {
    if (eqView)
      PyBuffer_Release(eqView.get());
  });
  if (!PyBuffer_IsContiguous(eqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");
  if (!isSignedIntegerFormat(eqView->format) || eqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(eqView->format));
  if (eqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d equality coefficients but got rank ") +
        std::to_string(eqView->ndim));
  unsigned numEqualities = eqView->shape[0];
  if (eqView->shape[1] != numDomainVars + numRangeVars + 1 ||
      eqView->shape[1] != ineqView->shape[1])
    throw std::invalid_argument(
        "expected number of columns of inequality and equality coefficient "
        "matrices to equal numRangeVars + numDomainVars + 1");
  MlirPresburgerIntegerRelation relation =
      mlirPresburgerIntegerRelationCreateFromCoefficients(
          reinterpret_cast<const int64_t *>(ineqView->buf), numInequalities,
          reinterpret_cast<const int64_t *>(eqView->buf), numEqualities,
          numDomainVars, numRangeVars);
  return std::make_unique<PyPresburgerIntegerRelation>(relation);
}

std::unique_ptr<PyPresburgerIntegerRelation>
PyPresburgerIntegerRelation::getFromNumConstrainsAndVars(
    uint64_t numReservedInequalities, uint64_t numReservedEqualities,
    uint64_t numReservedCols) {
  MlirPresburgerIntegerRelation relation = mlirPresburgerIntegerRelationCreate(
      numReservedInequalities, numReservedEqualities, numReservedCols);
  return std::make_unique<PyPresburgerIntegerRelation>(relation);
}

py::object PyPresburgerIntegerRelation::getCapsule() {
  throw std::invalid_argument("unimplemented");
}

void PyPresburgerTableau::bind(py::module &m) {
  py::class_<PyPresburgerTableau>(m, "IntegerRelationTableau",
                                  py::module_local())
      .def("__getitem__", [](PyPresburgerTableau &self,
                             const py::tuple &index) {
        return self.at64(index[0].cast<int64_t>(), index[1].cast<int64_t>());
      });
}

void PyPresburgerMaybeOptimum::bind(py::module &m) {
  py::class_<PyPresburgerMaybeOptimum>(m, "IntegerRelationMaybeOptimum",
                                       py::module_local())
      .def("is_empty",
           [](PyPresburgerMaybeOptimum &self) {
             return self.kind == MlirPresburgerOptimumKind::Empty;
           })
      .def("is_unbounded",
           [](PyPresburgerMaybeOptimum &self) {
             return self.kind == MlirPresburgerOptimumKind::Unbounded;
           })
      .def("is_bounded",
           [](PyPresburgerMaybeOptimum &self) {
             return self.kind == MlirPresburgerOptimumKind::Bounded;
           })
      .def("get_integer_point", [](PyPresburgerMaybeOptimum &self) {
        if (self.kind != MlirPresburgerOptimumKind::Bounded)
          return std::vector<int64_t>();
        std::vector<int64_t> r{self.integerPoint,
                               self.integerPoint + self.integerPointSize};
        return r;
      });
}

void PyPresburgerIntegerRelation::bind(py::module &m) {
  py::class_<PyPresburgerIntegerRelation>(m, "IntegerRelation",
                                          py::module_local())
      .def(py::init<>(&PyPresburgerIntegerRelation::getFromBuffers),
           py::arg("inequalities_coefficients"),
           py::arg("equalities_coefficients"), py::arg("num_domain_vars"),
           py::arg("num_range_vars"))
      .def(
          py::init<>(&PyPresburgerIntegerRelation::getFromNumConstrainsAndVars),
          py::arg("num_reserved_inequalities"),
          py::arg("num_reserved_equalities"), py::arg("num_reserved_cols"))
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyPresburgerIntegerRelation::getCapsule)
      .def("__eq__",
           [](PyPresburgerIntegerRelation &self,
              PyPresburgerIntegerRelation &other) {
             return mlirPresburgerIntegerRelationIsEqual(self.relation,
                                                         other.relation);
           })
      .def(
          "append",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationAppend(self.relation,
                                                       other.relation);
          },
          py::arg("other"))
      .def(
          "intersect",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            PyPresburgerIntegerRelation intersection(
                mlirPresburgerIntegerRelationIntersect(self.relation,
                                                       other.relation));
            return intersection;
          },
          py::arg("other"))
      .def(
          "is_equal",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationIsEqual(self.relation,
                                                        other.relation);
          },
          py::arg("other"))
      .def(
          "is_obviously_equal",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationIsObviouslyEqual(
                self.relation, other.relation);
          },
          py::arg("other"))
      .def(
          "is_subset_of",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationIsSubsetOf(self.relation,
                                                           other.relation);
          },
          py::arg("other"))
      .def(
          "merge_and_align_symbols",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationMergeAndAlignSymbols(
                self.relation, other.relation);
          },
          py::arg("other"))
      .def(
          "merge_local_vars",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationMergeLocalVars(self.relation,
                                                               other.relation);
          },
          py::arg("other"))
      .def(
          "compose",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationCompose(self.relation,
                                                        other.relation);
          },
          py::arg("other"))
      .def(
          "apply_domain",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationApplyDomain(self.relation,
                                                            other.relation);
          },
          py::arg("other"))
      .def(
          "apply_range",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationApplyRange(self.relation,
                                                           other.relation);
          },
          py::arg("other"))
      .def(
          "merge_and_composite",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            return mlirPresburgerIntegerRelationMergeAndCompose(self.relation,
                                                                other.relation);
          },
          py::arg("other"))
      .def(
          "union_bounding_box",
          [](PyPresburgerIntegerRelation &self,
             PyPresburgerIntegerRelation &other) {
            auto r = mlirPresburgerIntegerRelationUnionBoundingBox(
                self.relation, other.relation);
            return mlirLogicalResultIsSuccess(r);
          },
          py::arg("other"))
      .def(
          "inequalities",
          [](PyPresburgerIntegerRelation &self) {
            PyPresburgerTableau tableau(
                self.relation, PyPresburgerTableau::Kind::Inequalities);
            return tableau;
          },
          py::keep_alive<0, 1>())
      .def(
          "equalities",
          [](PyPresburgerIntegerRelation &self) {
            PyPresburgerTableau tableau(self.relation,
                                        PyPresburgerTableau::Kind::Equalities);
            return tableau;
          },
          py::keep_alive<0, 1>())
      .def("get_equality",
           [](PyPresburgerIntegerRelation &self, int64_t row) {
             unsigned numCol =
                 mlirPresburgerIntegerRelationNumCols(self.relation);
             std::vector<int64_t> result(numCol);
             for (unsigned i = 0; i < numCol; i++)
               result[i] =
                   mlirPresburgerIntegerRelationAtEq64(self.relation, row, i);
             return result;
           })
      .def("get_inequality",
           [](PyPresburgerIntegerRelation &self, int64_t row) {
             unsigned numCol =
                 mlirPresburgerIntegerRelationNumCols(self.relation);
             std::vector<int64_t> result(numCol);
             for (unsigned i = 0; i < numCol; i++)
               result[i] =
                   mlirPresburgerIntegerRelationAtIneq64(self.relation, row, i);
             return result;
           })
      .def(
          "get_num_vars_of_kind",
          [](PyPresburgerIntegerRelation &self,
             MlirPresburgerVariableKind kind) {
            return mlirPresburgerIntegerRelationGetNumVarKind(self.relation,
                                                              kind);
          },
          py::arg("kind"))
      .def(
          "get_var_kind_offset",
          [](PyPresburgerIntegerRelation &self,
             MlirPresburgerVariableKind kind) {
            return mlirPresburgerIntegerRelationGetVarKindOffset(self.relation,
                                                                 kind);
          },
          py::arg("kind"))
      .def(
          "get_var_kind_end",
          [](PyPresburgerIntegerRelation &self,
             MlirPresburgerVariableKind kind) {
            return mlirPresburgerIntegerRelationGetVarKindEnd(self.relation,
                                                              kind);
          },
          py::arg("kind"))
      .def(
          "get_var_kind_overlap",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerVariableKind kind,
             int64_t varStart, int64_t varLimit) {
            return mlirPresburgerIntegerRelationGetVarKindOverLap(
                self.relation, kind, varStart, varLimit);
          },
          py::arg("kind"), py::arg("start"), py::arg("limit"))
      .def(
          "get_var_kind_at",
          [](PyPresburgerIntegerRelation &self, uint64_t pos) {
            return mlirPresburgerIntegerRelationGetVarKindAt(self.relation,
                                                             pos);
          },
          py::arg("pos"))
      .def(
          "get_constant_bound",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerBoundType type,
             uint64_t pos) -> std::optional<int64_t> {
            auto r = mlirPresburgerIntegerRelationGetConstantBound64(
                self.relation, type, pos);
            if (!r.hasValue)
              return std::nullopt;
            return r.value;
          },
          py::arg("bound_type"), py::arg("pos"))
      .def("is_full_dim",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationIsFullDim(self.relation);
           })
      .def(
          "contains_point",
          [](PyPresburgerIntegerRelation &self, std::vector<int64_t> &point) {
            return mlirPresburgerIntegerRelationContainsPoint(
                self.relation, point.data(), point.size());
          },
          py::arg("point"))
      .def("has_only_div_locals",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationHasOnlyDivLocals(
                 self.relation);
           })
      .def("remove_trivial_equalities",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationRemoveTrivialEqualities(
                 self.relation);
           })
      .def(
          "insert_var",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerVariableKind kind,
             uint64_t pos, uint64_t num) {
            return mlirPresburgerIntegerRelationInsertVar(self.relation, kind,
                                                          pos, num);
          },
          py::arg("kind"), py::arg("pos"), py::arg("num") = 1)
      .def(
          "append_var",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerVariableKind kind,
             uint64_t num) {
            return mlirPresburgerIntegerRelationAppendVar(self.relation, kind,
                                                          num);
          },
          py::arg("kind"), py::arg("num") = 1)
      .def(
          "add_equality",
          [](PyPresburgerIntegerRelation &self,
             const std::vector<int64_t> &eq) {
            return mlirPresburgerIntegerRelationAddEquality(
                self.relation, eq.data(), eq.size());
          },
          py::arg("coefficients"))
      .def(
          "add_inequality",
          [](PyPresburgerIntegerRelation &self,
             const std::vector<int64_t> &inEq) {
            return mlirPresburgerIntegerRelationAddInequality(
                self.relation, inEq.data(), inEq.size());
          },
          py::arg("coefficients"))
      .def(
          "eliminate_redundant_local_var",
          [](PyPresburgerIntegerRelation &self, uint64_t posA, uint64_t posB) {
            return mlirPresburgerIntegerRelationEliminateRedundantLocalVar(
                self.relation, posA, posB);
          },
          py::arg("pos_a"), py::arg("pos_b"))
      .def(
          "remove_var_of_kind",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerVariableKind kind,
             uint64_t pos) {
            return mlirPresburgerIntegerRelationRemoveVarKind(self.relation,
                                                              kind, pos);
          },
          py::arg("kind"), py::arg("pos"))
      .def(
          "remove_var_range_of_kind",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerVariableKind kind,
             uint64_t varStart, uint64_t varLimit) {
            return mlirPresburgerIntegerRelationRemoveVarRangeKind(
                self.relation, kind, varStart, varLimit);
          },
          py::arg("kind"), py::arg("start"), py::arg("limit"))
      .def(
          "remove_var",
          [](PyPresburgerIntegerRelation &self, uint64_t pos) {
            return mlirPresburgerIntegerRelationRemoveVar(self.relation, pos);
          },
          py::arg("pos"))
      .def(
          "remove_equality",
          [](PyPresburgerIntegerRelation &self, uint64_t pos) {
            return mlirPresburgerIntegerRelationRemoveEquality(self.relation,
                                                               pos);
          },
          py::arg("pos"))
      .def(
          "remove_inequality",
          [](PyPresburgerIntegerRelation &self, uint64_t pos) {
            return mlirPresburgerIntegerRelationRemoveInequality(self.relation,
                                                                 pos);
          },
          py::arg("pos"))
      .def(
          "remove_equality_range",
          [](PyPresburgerIntegerRelation &self, uint64_t start, uint64_t end) {
            return mlirPresburgerIntegerRelationRemoveEqualityRange(
                self.relation, start, end);
          },
          py::arg("start"), py::arg("end"))
      .def(
          "remove_inequality_range",
          [](PyPresburgerIntegerRelation &self, uint64_t start, uint64_t end) {
            return mlirPresburgerIntegerRelationRemoveInequalityRange(
                self.relation, start, end);
          },
          py::arg("start"), py::arg("end"))
      .def(
          "find_integer_lex_min",
          [](PyPresburgerIntegerRelation &self) {
            auto r =
                mlirPresburgerIntegerRelationFindIntegerLexMin(self.relation);
            auto mayBeOptimum = std::make_unique<PyPresburgerMaybeOptimum>(r);
            return mayBeOptimum.release();
          },
          py::return_value_policy::take_ownership)
      .def("find_integer_sample",
           [](PyPresburgerIntegerRelation &self)
               -> std::optional<std::vector<int64_t>> {
             auto r =
                 mlirPresburgerIntegerRelationFindIntegerSample(self.relation);
             if (!r.hasValue)
               return std::nullopt;
             std::vector<int64_t> integerSample{r.data, r.data + r.size};
             return integerSample;
           })
      .def("compute_volume",
           [](PyPresburgerIntegerRelation &self) -> std::optional<int64_t> {
             auto r = mlirPresburgerIntegerRelationComputeVolume(self.relation);
             if (!r.hasValue)
               return std::nullopt;
             return r.value;
           })
      .def(
          "swap_var",
          [](PyPresburgerIntegerRelation &self, uint64_t posA, uint64_t posB) {
            return mlirPresburgerIntegerRelationSwapVar(self.relation, posA,
                                                        posB);
          },
          py::arg("pos_a"), py::arg("pos_b"))
      .def("clear_constraints",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationClearConstraints(
                 self.relation);
           })
      .def(
          "set_and_eliminate",
          [](PyPresburgerIntegerRelation &self, uint64_t pos,
             std::vector<int64_t> &values) {
            return mlirPresburgerIntegerRelationSetAndEliminate(
                self.relation, pos, values.data(), values.size());
          },
          py::arg("pos"), py::arg("values"))
      .def(
          "remove_independent_constraints",
          [](PyPresburgerIntegerRelation &self, uint64_t pos, uint64_t num) {
            return mlirPresburgerIntegerRelationRemoveIndependentConstraints(
                self.relation, pos, num);
          },
          py::arg("pos"), py::arg("num"))
      .def(
          "is_hyper_rectangular",
          [](PyPresburgerIntegerRelation &self, uint64_t pos, uint64_t num) {
            return mlirPresburgerIntegerRelationIsHyperRectangular(
                self.relation, pos, num);
          },
          py::arg("pos"), py::arg("num"))
      .def("remove_trivial_redundancy",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationRemoveTrivialRedundancy(
                 self.relation);
           })
      .def("remove_redundant_inequalities",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationRemoveRedundantInequalities(
                 self.relation);
           })
      .def("remove_redundant_constraints",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationRemoveRedundantConstraints(
                 self.relation);
           })
      .def("remove_duplicate_divs",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationRemoveDuplicateDivs(
                 self.relation);
           })
      .def("simplify",
           [](PyPresburgerIntegerRelation &self) {
             return mlirPresburgerIntegerRelationSimplify(self.relation);
           })
      .def(
          "convert_var_kind",
          [](PyPresburgerIntegerRelation &self,
             MlirPresburgerVariableKind srcKind, uint64_t vatStart,
             uint64_t varLimit, MlirPresburgerVariableKind dstKind,
             py::object &pos) {
            if (pos.is_none())
              return mlirPresburgerIntegerRelationConvertVarKindNoPos(
                  self.relation, srcKind, vatStart, varLimit, dstKind);
            return mlirPresburgerIntegerRelationConvertVarKind(
                self.relation, srcKind, vatStart, varLimit, dstKind,
                pos.cast<uint64_t>());
          },
          py::arg("src_kind"), py::arg("start"), py::arg("limit"),
          py::arg("dst_kind"), py::arg("pos") = py::none())
      .def(
          "convert_to_local",
          [](PyPresburgerIntegerRelation &self,
             MlirPresburgerVariableKind srcKind, uint64_t vatStart,
             uint64_t varLimit) {
            return mlirPresburgerIntegerRelationConvertVarKindNoPos(
                self.relation, srcKind, vatStart, varLimit,
                MlirPresburgerVariableKind::Local);
          },
          py::arg("src_kind"), py::arg("start"), py::arg("limit"))
      .def(
          "add_bound",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerBoundType type,
             uint64_t pos, int64_t value) {
            return mlirPresburgerIntegerRelationAddBound(self.relation, type,
                                                         pos, value);
          },
          py::arg("bound_type"), py::arg("pos"), py::arg("value"))
      .def(
          "add_bound",
          [](PyPresburgerIntegerRelation &self, MlirPresburgerBoundType type,
             std::vector<int64_t> &expr, int64_t value) {
            return mlirPresburgerIntegerRelationAddBoundExpr(
                self.relation, type, expr.data(), expr.size(), value);
          },
          py::arg("bound_type"), py::arg("expr"), py::arg("value"))
      .def(
          "constant_fold_var",
          [](PyPresburgerIntegerRelation &self, uint64_t pos) {
            auto r = mlirPresburgerIntegerRelationConstantFoldVar(self.relation,
                                                                  pos);
            return mlirLogicalResultIsSuccess(r);
          },
          py::arg("pos"))
      .def(
          "constant_fold_var_range",
          [](PyPresburgerIntegerRelation &self, uint64_t pos, uint64_t num) {
            return mlirPresburgerIntegerRelationConstantFoldVarRange(
                self.relation, pos, num);
          },
          py::arg("pos"), py::arg("num"))
      .def_property_readonly(
          "at_eq",
          [](PyPresburgerIntegerRelation &self) {
            return PyPresburgerTableau(self.relation,
                                       PyPresburgerTableau::Kind::Equalities);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "at_ineq",
          [](PyPresburgerIntegerRelation &self) {
            return PyPresburgerTableau(self.relation,
                                       PyPresburgerTableau::Kind::Inequalities);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "num_constraints",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumConstraints(self.relation);
          })
      .def_property_readonly(
          "num_domain_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumDomainVars(self.relation);
          })
      .def_property_readonly("num_range_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumRangeVars(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_symbol_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumSymbolVars(self.relation);
          })
      .def_property_readonly("num_local_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumLocalVars(
                                   self.relation);
                             })
      .def_property_readonly("num_dim_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumDimVars(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_dim_and_symbol_vars",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumDimAndSymbolVars(
                self.relation);
          })
      .def_property_readonly("num_vars",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumVars(
                                   self.relation);
                             })
      .def_property_readonly("num_columns",
                             [](const PyPresburgerIntegerRelation &self) {
                               return mlirPresburgerIntegerRelationNumCols(
                                   self.relation);
                             })
      .def_property_readonly(
          "num_equalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumEqualities(self.relation);
          })
      .def_property_readonly(
          "num_inequalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumInequalities(self.relation);
          })
      .def_property_readonly(
          "num_reserved_equalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumReservedEqualities(
                self.relation);
          })
      .def_property_readonly(
          "num_reserved_inequalities",
          [](const PyPresburgerIntegerRelation &self) {
            return mlirPresburgerIntegerRelationNumReservedInequalities(
                self.relation);
          })
      .def("__str__", [](const PyPresburgerIntegerRelation &self) {
        mlirPresburgerIntegerRelationDump(self.relation);
        return "";
      });
}

static inline void populateVarKindEnum(py::module &m) {
  py::enum_<MlirPresburgerVariableKind>(m, "VariableKind", py::module_local())
      .value("Symbol", MlirPresburgerVariableKind::Symbol)
      .value("Local", MlirPresburgerVariableKind::Local)
      .value("Domain", MlirPresburgerVariableKind::Domain)
      .value("Range", MlirPresburgerVariableKind::Range)
      .export_values();
}

static inline void populateBoundTypeEnum(py::module &m) {
  py::enum_<MlirPresburgerBoundType>(m, "BoundType", py::module_local())
      .value("EQ", MlirPresburgerBoundType::EQ)
      .value("LB", MlirPresburgerBoundType::LB)
      .value("UB", MlirPresburgerBoundType::UB)
      .export_values();
}

static void populatePresburgerModule(py::module &m) {
  populateVarKindEnum(m);
  populateBoundTypeEnum(m);
  PyPresburgerTableau::bind(m);
  PyPresburgerMaybeOptimum::bind(m);
  PyPresburgerIntegerRelation::bind(m);
}
// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------
PYBIND11_MODULE(_mlirPresburger, m) {
  m.doc() = "MLIR Presburger utilities";
  populatePresburgerModule(m);
}