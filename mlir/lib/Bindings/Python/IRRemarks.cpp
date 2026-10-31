//===- IRRemarks.cpp - Remark engine bindings -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Remarks.h"
#include "mlir/Bindings/Python/IRCore.h"

#include <cstdio>
#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace {
/// User data of the remark callback installed by `Context.enable_remarks`.
struct PyRemarkCallbackData {
  MlirContext context;
  nb::object callback;
};
} // namespace

static nb::str remarkStr(MlirStringRef ref) {
  return nb::str(ref.data, ref.length);
}

/// Forwards a reported remark to the Python callable in `userData`; the
/// PyRemark is invalidated once the callable returns, like a PyDiagnostic.
static void pyRemarkCallback(MlirRemark remark, void *userData) {
  auto *data = static_cast<PyRemarkCallbackData *>(userData);
  // Since this can be called from arbitrary C++ contexts, always get the gil.
  nb::gil_scoped_acquire gil;
  // A remark postponed until the context's destruction cannot be delivered:
  // the Python wrapper of the context is already gone.
  if (!PyMlirContext::isLiveContext(data->context))
    return;
  PyRemark *pyRemark = new PyRemark(remark);
  nb::object pyRemarkObject = nb::cast(pyRemark, nb::rv_policy::take_ownership);
  try {
    data->callback(pyRemarkObject);
  } catch (std::exception &e) {
    fprintf(stderr, "MLIR Python remark callback raised exception: %s\n",
            e.what());
  }
  pyRemark->invalidate();
}

static void pyRemarkCallbackDelete(void *userData) {
  nb::gil_scoped_acquire gil;
  delete static_cast<PyRemarkCallbackData *>(userData);
}

bool PyMlirContext::isLiveContext(MlirContext context) {
  nb::ft_lock_guard lock(live_contexts_mutex);
  return getLiveContexts().count(context.ptr) != 0;
}

void PyMlirContext::enableRemarks(
    PyRemarkPolicy policy, const std::optional<std::string> &outputFile,
    PyRemarkFormat format, const std::string &allFilter,
    const std::string &passedFilter, const std::string &missedFilter,
    const std::string &analysisFilter, const std::string &failedFilter,
    nb::object callback, std::optional<bool> printAsEmitRemarks) {
  bool hasCallback = !callback.is_none();
  if (outputFile && hasCallback)
    throw nb::value_error(
        "a remark callback cannot be combined with an output file");
  if (mlirContextHasRemarkEngine(get())) {
    throw nb::value_error("remarks are already enabled on this context; call "
                          "finalize_remarks() first");
  }

  MlirRemarkCategories categories{
      toMlirStringRef(allFilter), toMlirStringRef(passedFilter),
      toMlirStringRef(missedFilter), toMlirStringRef(analysisFilter),
      toMlirStringRef(failedFilter)};
  auto remarkPolicy = static_cast<MlirRemarkPolicy>(policy);
  // Without a sink of its own, the engine prints as MLIR remark diagnostics.
  bool printAsEmit =
      printAsEmitRemarks.value_or(!outputFile.has_value() && !hasCallback);

  MlirLogicalResult result;
  if (outputFile) {
    result = mlirContextEnableOptimizationRemarksToFile(
        get(), categories, remarkPolicy,
        static_cast<MlirRemarkFileFormat>(format), toMlirStringRef(*outputFile),
        printAsEmit);
    if (mlirLogicalResultIsFailure(result))
      throw nb::value_error(
          ("failed to enable remarks: cannot write '" + *outputFile + "'")
              .c_str());
  } else if (hasCallback) {
    auto *data = new PyRemarkCallbackData{get(), std::move(callback)};
    result = mlirContextEnableOptimizationRemarksWithCallback(
        get(), categories, remarkPolicy, pyRemarkCallback, data,
        pyRemarkCallbackDelete, printAsEmit);
  } else {
    result = mlirContextEnableOptimizationRemarks(get(), categories,
                                                  remarkPolicy, printAsEmit);
  }
  if (mlirLogicalResultIsFailure(result))
    throw nb::value_error("failed to enable remarks");
}

void PyMlirContext::finalizeRemarks() {
  mlirContextFinalizeOptimizationRemarks(get());
}

bool PyMlirContext::getRemarksEnabled() {
  return mlirContextHasRemarkEngine(get());
}

void PyRemark::checkValid() const {
  if (!valid)
    throw std::invalid_argument("Remark is invalid (used outside of callback)");
}

PyRemarkKind PyRemark::getKind() const {
  checkValid();
  return static_cast<PyRemarkKind>(mlirRemarkGetKind(remark));
}

nb::str PyRemark::getRemarkName() const {
  checkValid();
  return remarkStr(mlirRemarkGetRemarkName(remark));
}

nb::str PyRemark::getCategoryName() const {
  checkValid();
  return remarkStr(mlirRemarkGetCategoryName(remark));
}

nb::str PyRemark::getFullCategoryName() const {
  checkValid();
  return remarkStr(mlirRemarkGetFullCategoryName(remark));
}

nb::str PyRemark::getFunctionName() const {
  checkValid();
  return remarkStr(mlirRemarkGetFunctionName(remark));
}

nb::typed<nb::object, PyLocation> PyRemark::getLocation() const {
  checkValid();
  MlirLocation loc = mlirRemarkGetLocation(remark);
  MlirContext context = mlirLocationGetContext(loc);
  return PyLocation(PyMlirContext::forContext(context), loc).maybeDownCast();
}

uint64_t PyRemark::getId() const {
  checkValid();
  return mlirRemarkGetId(remark);
}

nb::list PyRemark::getArgs() const {
  checkValid();
  nb::list args;
  intptr_t numArgs = mlirRemarkGetNumArgs(remark);
  for (intptr_t i = 0; i < numArgs; ++i) {
    args.append(nb::make_tuple(remarkStr(mlirRemarkGetArgKey(remark, i)),
                               remarkStr(mlirRemarkGetArgValue(remark, i))));
  }
  return args;
}

nb::str PyRemark::getMessage() const {
  checkValid();
  std::string message;
  auto callback = +[](MlirStringRef ref, void *userData) {
    static_cast<std::string *>(userData)->append(ref.data, ref.length);
  };
  mlirRemarkPrint(remark, callback, &message);
  return nb::str(message.c_str(), message.size());
}

void populateIRRemarks(nb::module_ &m) {
  nb::enum_<PyRemarkKind>(m, "RemarkKind")
      .value("UNKNOWN", PyRemarkKind::Unknown)
      .value("PASSED", PyRemarkKind::Passed)
      .value("MISSED", PyRemarkKind::Missed)
      .value("FAILURE", PyRemarkKind::Failure)
      .value("ANALYSIS", PyRemarkKind::Analysis);

  nb::enum_<PyRemarkPolicy>(m, "RemarkPolicy")
      .value("ALL", PyRemarkPolicy::All)
      .value("FINAL", PyRemarkPolicy::Final);

  nb::enum_<PyRemarkFormat>(m, "RemarkFormat")
      .value("YAML", PyRemarkFormat::YAML)
      .value("BITSTREAM", PyRemarkFormat::Bitstream);

  nb::class_<PyRemark>(m, "Remark")
      .def_prop_ro("kind", &PyRemark::getKind,
                   "Returns the kind of the remark.")
      .def_prop_ro("remark_name", &PyRemark::getRemarkName,
                   "Returns the name identifying the remark.")
      .def_prop_ro("category_name", &PyRemark::getCategoryName,
                   "Returns the category of the remark (what the filters "
                   "match).")
      .def_prop_ro("full_category_name", &PyRemark::getFullCategoryName,
                   "Returns the combined `category:subcategory` name.")
      .def_prop_ro("function_name", &PyRemark::getFunctionName,
                   "Returns the name of the function the remark refers to.")
      .def_prop_ro("location", &PyRemark::getLocation,
                   "Returns the location associated with the remark.")
      .def_prop_ro("remark_id", &PyRemark::getId,
                   "Returns the id of the remark within its engine (0 when "
                   "unset).")
      .def_prop_ro("args", &PyRemark::getArgs,
                   "Returns the key/value arguments as a list of tuples.")
      .def_prop_ro("message", &PyRemark::getMessage,
                   "Returns the textual form of the remark, `[Kind] name | "
                   "Category:... | key=value, ...`.")
      .def(
          "__str__",
          [](PyRemark &self) -> nb::str {
            if (!self.isValid())
              return nb::str("<Invalid Remark>");
            return self.getMessage();
          },
          "Returns the remark message as a string.");
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
