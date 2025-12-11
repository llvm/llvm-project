//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Pass.h"
#include "Rewrite.h"
#include "mlir/Bindings/Python/Globals.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindUtils.h"

namespace nb = nanobind;
using namespace mlir;
using namespace nb::literals;
using namespace mlir::python;

static const char kModuleParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new MlirModule or raises an MLIRError if the parsing fails.

See also: https://mlir.llvm.org/docs/LangRef/
)";

static const char kDumpDocstring[] =
    "Dumps a debug representation of the object to stderr.";

static const char kValueReplaceAllUsesExceptDocstring[] =
    R"(Replace all uses of this value with the `with` value, except for those
in `exceptions`. `exceptions` can be either a single operation or a list of
operations.
)";

namespace {
// see
// https://raw.githubusercontent.com/python/pythoncapi_compat/master/pythoncapi_compat.h

#ifndef _Py_CAST
#define _Py_CAST(type, expr) ((type)(expr))
#endif

// Static inline functions should use _Py_NULL rather than using directly NULL
// to prevent C++ compiler warnings. On C23 and newer and on C++11 and newer,
// _Py_NULL is defined as nullptr.
#ifndef _Py_NULL
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ > 201710L) ||               \
    (defined(__cplusplus) && __cplusplus >= 201103)
#define _Py_NULL nullptr
#else
#define _Py_NULL NULL
#endif
#endif

// Python 3.10.0a3
#if PY_VERSION_HEX < 0x030A00A3

// bpo-42262 added Py_XNewRef()
#if !defined(Py_XNewRef)
[[maybe_unused]] PyObject *_Py_XNewRef(PyObject *obj) {
  Py_XINCREF(obj);
  return obj;
}
#define Py_XNewRef(obj) _Py_XNewRef(_PyObject_CAST(obj))
#endif

// bpo-42262 added Py_NewRef()
#if !defined(Py_NewRef)
[[maybe_unused]] PyObject *_Py_NewRef(PyObject *obj) {
  Py_INCREF(obj);
  return obj;
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#endif

#endif // Python 3.10.0a3

// Python 3.9.0b1
#if PY_VERSION_HEX < 0x030900B1 && !defined(PYPY_VERSION)

// bpo-40429 added PyThreadState_GetFrame()
PyFrameObject *PyThreadState_GetFrame(PyThreadState *tstate) {
  assert(tstate != _Py_NULL && "expected tstate != _Py_NULL");
  return _Py_CAST(PyFrameObject *, Py_XNewRef(tstate->frame));
}

// bpo-40421 added PyFrame_GetBack()
PyFrameObject *PyFrame_GetBack(PyFrameObject *frame) {
  assert(frame != _Py_NULL && "expected frame != _Py_NULL");
  return _Py_CAST(PyFrameObject *, Py_XNewRef(frame->f_back));
}

// bpo-40421 added PyFrame_GetCode()
PyCodeObject *PyFrame_GetCode(PyFrameObject *frame) {
  assert(frame != _Py_NULL && "expected frame != _Py_NULL");
  assert(frame->f_code != _Py_NULL && "expected frame->f_code != _Py_NULL");
  return _Py_CAST(PyCodeObject *, Py_NewRef(frame->f_code));
}

#endif // Python 3.9.0b1

MlirLocation tracebackToLocation(MlirContext ctx) {
  size_t framesLimit =
      PyGlobals::get().getTracebackLoc().locTracebackFramesLimit();
  // Use a thread_local here to avoid requiring a large amount of space.
  thread_local std::array<MlirLocation, PyGlobals::TracebackLoc::kMaxFrames>
      frames;
  size_t count = 0;

  nb::gil_scoped_acquire acquire;
  PyThreadState *tstate = PyThreadState_GET();
  PyFrameObject *next;
  PyFrameObject *pyFrame = PyThreadState_GetFrame(tstate);
  // In the increment expression:
  // 1. get the next prev frame;
  // 2. decrement the ref count on the current frame (in order that it can get
  //    gc'd, along with any objects in its closure and etc);
  // 3. set current = next.
  for (; pyFrame != nullptr && count < framesLimit;
       next = PyFrame_GetBack(pyFrame), Py_XDECREF(pyFrame), pyFrame = next) {
    PyCodeObject *code = PyFrame_GetCode(pyFrame);
    auto fileNameStr =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_filename));
    llvm::StringRef fileName(fileNameStr);
    if (!PyGlobals::get().getTracebackLoc().isUserTracebackFilename(fileName))
      continue;

    // co_qualname and PyCode_Addr2Location added in py3.11
#if PY_VERSION_HEX < 0x030B00F0
    std::string name =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_name));
    llvm::StringRef funcName(name);
    int startLine = PyFrame_GetLineNumber(pyFrame);
    MlirLocation loc =
        mlirLocationFileLineColGet(ctx, wrap(fileName), startLine, 0);
#else
    std::string name =
        nb::cast<std::string>(nb::borrow<nb::str>(code->co_qualname));
    llvm::StringRef funcName(name);
    int startLine, startCol, endLine, endCol;
    int lasti = PyFrame_GetLasti(pyFrame);
    if (!PyCode_Addr2Location(code, lasti, &startLine, &startCol, &endLine,
                              &endCol)) {
      throw nb::python_error();
    }
    MlirLocation loc = mlirLocationFileLineColRangeGet(
        ctx, wrap(fileName), startLine, startCol, endLine, endCol);
#endif

    frames[count] = mlirLocationNameGet(ctx, wrap(funcName), loc);
    ++count;
  }
  // When the loop breaks (after the last iter), current frame (if non-null)
  // is leaked without this.
  Py_XDECREF(pyFrame);

  if (count == 0)
    return mlirLocationUnknownGet(ctx);

  MlirLocation callee = frames[0];
  assert(!mlirLocationIsNull(callee) && "expected non-null callee location");
  if (count == 1)
    return callee;

  MlirLocation caller = frames[count - 1];
  assert(!mlirLocationIsNull(caller) && "expected non-null caller location");
  for (int i = count - 2; i >= 1; i--)
    caller = mlirLocationCallSiteGet(frames[i], caller);

  return mlirLocationCallSiteGet(callee, caller);
}

PyLocation
maybeGetTracebackLocation(const std::optional<PyLocation> &location) {
  if (location.has_value())
    return location.value();
  if (!PyGlobals::get().getTracebackLoc().locTracebacksEnabled())
    return DefaultingPyLocation::resolve();

  PyMlirContext &ctx = DefaultingPyMlirContext::resolve();
  MlirLocation mlirLoc = tracebackToLocation(ctx.get());
  PyMlirContextRef ref = PyMlirContext::forContext(ctx.get());
  return {ref, mlirLoc};
}
} // namespace

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------

static void populateIRCore(nb::module_ &m) {
  // disable leak warnings which tend to be false positives.
  nb::set_leak_warnings(false);
  //----------------------------------------------------------------------------
  // Enums.
  //----------------------------------------------------------------------------
  nb::enum_<MlirDiagnosticSeverity>(m, "DiagnosticSeverity")
      .value("ERROR", MlirDiagnosticError)
      .value("WARNING", MlirDiagnosticWarning)
      .value("NOTE", MlirDiagnosticNote)
      .value("REMARK", MlirDiagnosticRemark);

  nb::enum_<MlirWalkOrder>(m, "WalkOrder")
      .value("PRE_ORDER", MlirWalkPreOrder)
      .value("POST_ORDER", MlirWalkPostOrder);

  nb::enum_<MlirWalkResult>(m, "WalkResult")
      .value("ADVANCE", MlirWalkResultAdvance)
      .value("INTERRUPT", MlirWalkResultInterrupt)
      .value("SKIP", MlirWalkResultSkip);

  //----------------------------------------------------------------------------
  // Mapping of Diagnostics.
  //----------------------------------------------------------------------------
  nb::class_<PyDiagnostic>(m, "Diagnostic")
      .def_prop_ro("severity", &PyDiagnostic::getSeverity,
                   "Returns the severity of the diagnostic.")
      .def_prop_ro("location", &PyDiagnostic::getLocation,
                   "Returns the location associated with the diagnostic.")
      .def_prop_ro("message", &PyDiagnostic::getMessage,
                   "Returns the message text of the diagnostic.")
      .def_prop_ro("notes", &PyDiagnostic::getNotes,
                   "Returns a tuple of attached note diagnostics.")
      .def(
          "__str__",
          [](PyDiagnostic &self) -> nb::str {
            if (!self.isValid())
              return nb::str("<Invalid Diagnostic>");
            return self.getMessage();
          },
          "Returns the diagnostic message as a string.");

  nb::class_<PyDiagnostic::DiagnosticInfo>(m, "DiagnosticInfo")
      .def(
          "__init__",
          [](PyDiagnostic::DiagnosticInfo &self, PyDiagnostic diag) {
            new (&self) PyDiagnostic::DiagnosticInfo(diag.getInfo());
          },
          "diag"_a, "Creates a DiagnosticInfo from a Diagnostic.")
      .def_ro("severity", &PyDiagnostic::DiagnosticInfo::severity,
              "The severity level of the diagnostic.")
      .def_ro("location", &PyDiagnostic::DiagnosticInfo::location,
              "The location associated with the diagnostic.")
      .def_ro("message", &PyDiagnostic::DiagnosticInfo::message,
              "The message text of the diagnostic.")
      .def_ro("notes", &PyDiagnostic::DiagnosticInfo::notes,
              "List of attached note diagnostics.")
      .def(
          "__str__",
          [](PyDiagnostic::DiagnosticInfo &self) { return self.message; },
          "Returns the diagnostic message as a string.");

  nb::class_<PyDiagnosticHandler>(m, "DiagnosticHandler")
      .def("detach", &PyDiagnosticHandler::detach,
           "Detaches the diagnostic handler from the context.")
      .def_prop_ro("attached", &PyDiagnosticHandler::isAttached,
                   "Returns True if the handler is attached to a context.")
      .def_prop_ro("had_error", &PyDiagnosticHandler::getHadError,
                   "Returns True if an error was encountered during diagnostic "
                   "handling.")
      .def("__enter__", &PyDiagnosticHandler::contextEnter,
           "Enters the diagnostic handler as a context manager.")
      .def("__exit__", &PyDiagnosticHandler::contextExit,
           nb::arg("exc_type").none(), nb::arg("exc_value").none(),
           nb::arg("traceback").none(),
           "Exits the diagnostic handler context manager.");

  // Expose DefaultThreadPool to python
  nb::class_<PyThreadPool>(m, "ThreadPool")
      .def(
          "__init__", [](PyThreadPool &self) { new (&self) PyThreadPool(); },
          "Creates a new thread pool with default concurrency.")
      .def("get_max_concurrency", &PyThreadPool::getMaxConcurrency,
           "Returns the maximum number of threads in the pool.")
      .def("_mlir_thread_pool_ptr", &PyThreadPool::_mlir_thread_pool_ptr,
           "Returns the raw pointer to the LLVM thread pool as a string.");

  nb::class_<PyMlirContext>(m, "Context")
      .def(
          "__init__",
          [](PyMlirContext &self) {
            MlirContext context = mlirContextCreateWithThreading(false);
            new (&self) PyMlirContext(context);
          },
          R"(
            Creates a new MLIR context.

            The context is the top-level container for all MLIR objects. It owns the storage
            for types, attributes, locations, and other core IR objects. A context can be
            configured to allow or disallow unregistered dialects and can have dialects
            loaded on-demand.)")
      .def_static("_get_live_count", &PyMlirContext::getLiveCount,
                  "Gets the number of live Context objects.")
      .def(
          "_get_context_again",
          [](PyMlirContext &self) -> nb::typed<nb::object, PyMlirContext> {
            PyMlirContextRef ref = PyMlirContext::forContext(self.get());
            return ref.releaseObject();
          },
          "Gets another reference to the same context.")
      .def("_get_live_module_count", &PyMlirContext::getLiveModuleCount,
           "Gets the number of live modules owned by this context.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyMlirContext::getCapsule,
                   "Gets a capsule wrapping the MlirContext.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyMlirContext::createFromCapsule,
                  "Creates a Context from a capsule wrapping MlirContext.")
      .def("__enter__", &PyMlirContext::contextEnter,
           "Enters the context as a context manager.")
      .def("__exit__", &PyMlirContext::contextExit, nb::arg("exc_type").none(),
           nb::arg("exc_value").none(), nb::arg("traceback").none(),
           "Exits the context manager.")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/)
              -> std::optional<nb::typed<nb::object, PyMlirContext>> {
            auto *context = PyThreadContextEntry::getDefaultContext();
            if (!context)
              return {};
            return nb::cast(context);
          },
          nb::sig("def current(/) -> Context | None"),
          "Gets the Context bound to the current thread or returns None if no "
          "context is set.")
      .def_prop_ro(
          "dialects",
          [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Gets a container for accessing dialects by name.")
      .def_prop_ro(
          "d", [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Alias for `dialects`.")
      .def(
          "get_dialect_descriptor",
          [=](PyMlirContext &self, std::string &name) {
            MlirDialect dialect = mlirContextGetOrLoadDialect(
                self.get(), {name.data(), name.size()});
            if (mlirDialectIsNull(dialect)) {
              throw nb::value_error(
                  (Twine("Dialect '") + name + "' not found").str().c_str());
            }
            return PyDialectDescriptor(self.getRef(), dialect);
          },
          nb::arg("dialect_name"),
          "Gets or loads a dialect by name, returning its descriptor object.")
      .def_prop_rw(
          "allow_unregistered_dialects",
          [](PyMlirContext &self) -> bool {
            return mlirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyMlirContext &self, bool value) {
            mlirContextSetAllowUnregisteredDialects(self.get(), value);
          },
          "Controls whether unregistered dialects are allowed in this context.")
      .def("attach_diagnostic_handler", &PyMlirContext::attachDiagnosticHandler,
           nb::arg("callback"),
           "Attaches a diagnostic handler that will receive callbacks.")
      .def(
          "enable_multithreading",
          [](PyMlirContext &self, bool enable) {
            mlirContextEnableMultithreading(self.get(), enable);
          },
          nb::arg("enable"),
          R"(
            Enables or disables multi-threading support in the context.

            Args:
              enable: Whether to enable (True) or disable (False) multi-threading.
          )")
      .def(
          "set_thread_pool",
          [](PyMlirContext &self, PyThreadPool &pool) {
            // we should disable multi-threading first before setting
            // new thread pool otherwise the assert in
            // MLIRContext::setThreadPool will be raised.
            mlirContextEnableMultithreading(self.get(), false);
            mlirContextSetThreadPool(self.get(), pool.get());
          },
          R"(
            Sets a custom thread pool for the context to use.

            Args:
              pool: A ThreadPool object to use for parallel operations.

            Note:
              Multi-threading is automatically disabled before setting the thread pool.)")
      .def(
          "get_num_threads",
          [](PyMlirContext &self) {
            return mlirContextGetNumThreads(self.get());
          },
          "Gets the number of threads in the context's thread pool.")
      .def(
          "_mlir_thread_pool_ptr",
          [](PyMlirContext &self) {
            MlirLlvmThreadPool pool = mlirContextGetThreadPool(self.get());
            std::stringstream ss;
            ss << pool.ptr;
            return ss.str();
          },
          "Gets the raw pointer to the LLVM thread pool as a string.")
      .def(
          "is_registered_operation",
          [](PyMlirContext &self, std::string &name) {
            return mlirContextIsRegisteredOperation(
                self.get(), MlirStringRef{name.data(), name.size()});
          },
          nb::arg("operation_name"),
          R"(
            Checks whether an operation with the given name is registered.

            Args:
              operation_name: The fully qualified name of the operation (e.g., `arith.addf`).

            Returns:
              True if the operation is registered, False otherwise.)")
      .def(
          "append_dialect_registry",
          [](PyMlirContext &self, PyDialectRegistry &registry) {
            mlirContextAppendDialectRegistry(self.get(), registry);
          },
          nb::arg("registry"),
          R"(
            Appends the contents of a dialect registry to the context.

            Args:
              registry: A DialectRegistry containing dialects to append.)")
      .def_prop_rw("emit_error_diagnostics",
                   &PyMlirContext::getEmitErrorDiagnostics,
                   &PyMlirContext::setEmitErrorDiagnostics,
                   R"(
            Controls whether error diagnostics are emitted to diagnostic handlers.

            By default, error diagnostics are captured and reported through MLIRError exceptions.)")
      .def(
          "load_all_available_dialects",
          [](PyMlirContext &self) {
            mlirContextLoadAllAvailableDialects(self.get());
          },
          R"(
            Loads all dialects available in the registry into the context.

            This eagerly loads all dialects that have been registered, making them
            immediately available for use.)");

  //----------------------------------------------------------------------------
  // Mapping of PyDialectDescriptor
  //----------------------------------------------------------------------------
  nb::class_<PyDialectDescriptor>(m, "DialectDescriptor")
      .def_prop_ro(
          "namespace",
          [](PyDialectDescriptor &self) {
            MlirStringRef ns = mlirDialectGetNamespace(self.get());
            return nb::str(ns.data, ns.length);
          },
          "Returns the namespace of the dialect.")
      .def(
          "__repr__",
          [](PyDialectDescriptor &self) {
            MlirStringRef ns = mlirDialectGetNamespace(self.get());
            std::string repr("<DialectDescriptor ");
            repr.append(ns.data, ns.length);
            repr.append(">");
            return repr;
          },
          nb::sig("def __repr__(self) -> str"),
          "Returns a string representation of the dialect descriptor.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialects
  //----------------------------------------------------------------------------
  nb::class_<PyDialects>(m, "Dialects")
      .def(
          "__getitem__",
          [=](PyDialects &self, std::string keyName) {
            MlirDialect dialect =
                self.getDialectForKey(keyName, /*attrError=*/false);
            nb::object descriptor =
                nb::cast(PyDialectDescriptor{self.getContext(), dialect});
            return createCustomDialectWrapper(keyName, std::move(descriptor));
          },
          "Gets a dialect by name using subscript notation.")
      .def(
          "__getattr__",
          [=](PyDialects &self, std::string attrName) {
            MlirDialect dialect =
                self.getDialectForKey(attrName, /*attrError=*/true);
            nb::object descriptor =
                nb::cast(PyDialectDescriptor{self.getContext(), dialect});
            return createCustomDialectWrapper(attrName, std::move(descriptor));
          },
          "Gets a dialect by name using attribute notation.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialect
  //----------------------------------------------------------------------------
  nb::class_<PyDialect>(m, "Dialect")
      .def(nb::init<nb::object>(), nb::arg("descriptor"),
           "Creates a Dialect from a DialectDescriptor.")
      .def_prop_ro(
          "descriptor", [](PyDialect &self) { return self.getDescriptor(); },
          "Returns the DialectDescriptor for this dialect.")
      .def(
          "__repr__",
          [](const nb::object &self) {
            auto clazz = self.attr("__class__");
            return nb::str("<Dialect ") +
                   self.attr("descriptor").attr("namespace") +
                   nb::str(" (class ") + clazz.attr("__module__") +
                   nb::str(".") + clazz.attr("__name__") + nb::str(")>");
          },
          nb::sig("def __repr__(self) -> str"),
          "Returns a string representation of the dialect.");

  //----------------------------------------------------------------------------
  // Mapping of PyDialectRegistry
  //----------------------------------------------------------------------------
  nb::class_<PyDialectRegistry>(m, "DialectRegistry")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyDialectRegistry::getCapsule,
                   "Gets a capsule wrapping the MlirDialectRegistry.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyDialectRegistry::createFromCapsule,
                  "Creates a DialectRegistry from a capsule wrapping "
                  "`MlirDialectRegistry`.")
      .def(nb::init<>(), "Creates a new empty dialect registry.");

  //----------------------------------------------------------------------------
  // Mapping of Location
  //----------------------------------------------------------------------------
  nb::class_<PyLocation>(m, "Location")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyLocation::getCapsule,
                   "Gets a capsule wrapping the MlirLocation.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyLocation::createFromCapsule,
                  "Creates a Location from a capsule wrapping MlirLocation.")
      .def("__enter__", &PyLocation::contextEnter,
           "Enters the location as a context manager.")
      .def("__exit__", &PyLocation::contextExit, nb::arg("exc_type").none(),
           nb::arg("exc_value").none(), nb::arg("traceback").none(),
           "Exits the location context manager.")
      .def(
          "__eq__",
          [](PyLocation &self, PyLocation &other) -> bool {
            return mlirLocationEqual(self, other);
          },
          "Compares two locations for equality.")
      .def(
          "__eq__", [](PyLocation &self, nb::object other) { return false; },
          "Compares location with non-location object (always returns False).")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) -> std::optional<PyLocation *> {
            auto *loc = PyThreadContextEntry::getDefaultLocation();
            if (!loc)
              return std::nullopt;
            return loc;
          },
          // clang-format off
          nb::sig("def current(/) -> Location | None"),
          // clang-format on
          "Gets the Location bound to the current thread or raises ValueError.")
      .def_static(
          "unknown",
          [](DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationUnknownGet(context->get()));
          },
          nb::arg("context") = nb::none(),
          "Gets a Location representing an unknown location.")
      .def_static(
          "callsite",
          [](PyLocation callee, const std::vector<PyLocation> &frames,
             DefaultingPyMlirContext context) {
            if (frames.empty())
              throw nb::value_error("No caller frames provided.");
            MlirLocation caller = frames.back().get();
            for (const PyLocation &frame :
                 llvm::reverse(llvm::ArrayRef(frames).drop_back()))
              caller = mlirLocationCallSiteGet(frame.get(), caller);
            return PyLocation(context->getRef(),
                              mlirLocationCallSiteGet(callee.get(), caller));
          },
          nb::arg("callee"), nb::arg("frames"), nb::arg("context") = nb::none(),
          "Gets a Location representing a caller and callsite.")
      .def("is_a_callsite", mlirLocationIsACallSite,
           "Returns True if this location is a CallSiteLoc.")
      .def_prop_ro(
          "callee",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              mlirLocationCallSiteGetCallee(self));
          },
          "Gets the callee location from a CallSiteLoc.")
      .def_prop_ro(
          "caller",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              mlirLocationCallSiteGetCaller(self));
          },
          "Gets the caller location from a CallSiteLoc.")
      .def_static(
          "file",
          [](std::string filename, int line, int col,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationFileLineColGet(
                    context->get(), toMlirStringRef(filename), line, col));
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context") = nb::none(),
          "Gets a Location representing a file, line and column.")
      .def_static(
          "file",
          [](std::string filename, int startLine, int startCol, int endLine,
             int endCol, DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationFileLineColRangeGet(
                                  context->get(), toMlirStringRef(filename),
                                  startLine, startCol, endLine, endCol));
          },
          nb::arg("filename"), nb::arg("start_line"), nb::arg("start_col"),
          nb::arg("end_line"), nb::arg("end_col"),
          nb::arg("context") = nb::none(),
          "Gets a Location representing a file, line and column range.")
      .def("is_a_file", mlirLocationIsAFileLineColRange,
           "Returns True if this location is a FileLineColLoc.")
      .def_prop_ro(
          "filename",
          [](MlirLocation loc) {
            return mlirIdentifierStr(
                mlirLocationFileLineColRangeGetFilename(loc));
          },
          "Gets the filename from a FileLineColLoc.")
      .def_prop_ro("start_line", mlirLocationFileLineColRangeGetStartLine,
                   "Gets the start line number from a `FileLineColLoc`.")
      .def_prop_ro("start_col", mlirLocationFileLineColRangeGetStartColumn,
                   "Gets the start column number from a `FileLineColLoc`.")
      .def_prop_ro("end_line", mlirLocationFileLineColRangeGetEndLine,
                   "Gets the end line number from a `FileLineColLoc`.")
      .def_prop_ro("end_col", mlirLocationFileLineColRangeGetEndColumn,
                   "Gets the end column number from a `FileLineColLoc`.")
      .def_static(
          "fused",
          [](const std::vector<PyLocation> &pyLocations,
             std::optional<PyAttribute> metadata,
             DefaultingPyMlirContext context) {
            llvm::SmallVector<MlirLocation, 4> locations;
            locations.reserve(pyLocations.size());
            for (auto &pyLocation : pyLocations)
              locations.push_back(pyLocation.get());
            MlirLocation location = mlirLocationFusedGet(
                context->get(), locations.size(), locations.data(),
                metadata ? metadata->get() : MlirAttribute{0});
            return PyLocation(context->getRef(), location);
          },
          nb::arg("locations"), nb::arg("metadata") = nb::none(),
          nb::arg("context") = nb::none(),
          "Gets a Location representing a fused location with optional "
          "metadata.")
      .def("is_a_fused", mlirLocationIsAFused,
           "Returns True if this location is a `FusedLoc`.")
      .def_prop_ro(
          "locations",
          [](PyLocation &self) {
            unsigned numLocations = mlirLocationFusedGetNumLocations(self);
            std::vector<MlirLocation> locations(numLocations);
            if (numLocations)
              mlirLocationFusedGetLocations(self, locations.data());
            std::vector<PyLocation> pyLocations{};
            pyLocations.reserve(numLocations);
            for (unsigned i = 0; i < numLocations; ++i)
              pyLocations.emplace_back(self.getContext(), locations[i]);
            return pyLocations;
          },
          "Gets the list of locations from a `FusedLoc`.")
      .def_static(
          "name",
          [](std::string name, std::optional<PyLocation> childLoc,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationNameGet(
                    context->get(), toMlirStringRef(name),
                    childLoc ? childLoc->get()
                             : mlirLocationUnknownGet(context->get())));
          },
          nb::arg("name"), nb::arg("childLoc") = nb::none(),
          nb::arg("context") = nb::none(),
          "Gets a Location representing a named location with optional child "
          "location.")
      .def("is_a_name", mlirLocationIsAName,
           "Returns True if this location is a `NameLoc`.")
      .def_prop_ro(
          "name_str",
          [](MlirLocation loc) {
            return mlirIdentifierStr(mlirLocationNameGetName(loc));
          },
          "Gets the name string from a `NameLoc`.")
      .def_prop_ro(
          "child_loc",
          [](PyLocation &self) {
            return PyLocation(self.getContext(),
                              mlirLocationNameGetChildLoc(self));
          },
          "Gets the child location from a `NameLoc`.")
      .def_static(
          "from_attr",
          [](PyAttribute &attribute, DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationFromAttribute(attribute));
          },
          nb::arg("attribute"), nb::arg("context") = nb::none(),
          "Gets a Location from a `LocationAttr`.")
      .def_prop_ro(
          "context",
          [](PyLocation &self) -> nb::typed<nb::object, PyMlirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Location`.")
      .def_prop_ro(
          "attr",
          [](PyLocation &self) {
            return PyAttribute(self.getContext(),
                               mlirLocationGetAttribute(self));
          },
          "Get the underlying `LocationAttr`.")
      .def(
          "emit_error",
          [](PyLocation &self, std::string message) {
            mlirEmitError(self, message.c_str());
          },
          nb::arg("message"),
          R"(
            Emits an error diagnostic at this location.

            Args:
              message: The error message to emit.)")
      .def(
          "__repr__",
          [](PyLocation &self) {
            PyPrintAccumulator printAccum;
            mlirLocationPrint(self, printAccum.getCallback(),
                              printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly representation of the location.");

  //----------------------------------------------------------------------------
  // Mapping of Module
  //----------------------------------------------------------------------------
  nb::class_<PyModule>(m, "Module", nb::is_weak_referenceable())
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyModule::getCapsule,
                   "Gets a capsule wrapping the MlirModule.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyModule::createFromCapsule,
                  R"(
                    Creates a Module from a `MlirModule` wrapped by a capsule (i.e. `module._CAPIPtr`).

                    This returns a new object **BUT** `_clear_mlir_module(module)` must be called to
                    prevent double-frees (of the underlying `mlir::Module`).)")
      .def("_clear_mlir_module", &PyModule::clearMlirModule,
           R"(
             Clears the internal MLIR module reference.

             This is used internally to prevent double-free when ownership is transferred
             via the C API capsule mechanism. Not intended for normal use.)")
      .def_static(
          "parse",
          [](const std::string &moduleAsm, DefaultingPyMlirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirModule module = mlirModuleCreateParse(
                context->get(), toMlirStringRef(moduleAsm));
            if (mlirModuleIsNull(module))
              throw MLIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("asm"), nb::arg("context") = nb::none(),
          kModuleParseDocstring)
      .def_static(
          "parse",
          [](nb::bytes moduleAsm, DefaultingPyMlirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirModule module = mlirModuleCreateParse(
                context->get(), toMlirStringRef(moduleAsm));
            if (mlirModuleIsNull(module))
              throw MLIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("asm"), nb::arg("context") = nb::none(),
          kModuleParseDocstring)
      .def_static(
          "parseFile",
          [](const std::string &path, DefaultingPyMlirContext context)
              -> nb::typed<nb::object, PyModule> {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirModule module = mlirModuleCreateParseFromFile(
                context->get(), toMlirStringRef(path));
            if (mlirModuleIsNull(module))
              throw MLIRError("Unable to parse module assembly", errors.take());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("path"), nb::arg("context") = nb::none(),
          kModuleParseDocstring)
      .def_static(
          "create",
          [](const std::optional<PyLocation> &loc)
              -> nb::typed<nb::object, PyModule> {
            PyLocation pyLoc = maybeGetTracebackLocation(loc);
            MlirModule module = mlirModuleCreateEmpty(pyLoc.get());
            return PyModule::forModule(module).releaseObject();
          },
          nb::arg("loc") = nb::none(), "Creates an empty module.")
      .def_prop_ro(
          "context",
          [](PyModule &self) -> nb::typed<nb::object, PyMlirContext> {
            return self.getContext().getObject();
          },
          "Context that created the `Module`.")
      .def_prop_ro(
          "operation",
          [](PyModule &self) -> nb::typed<nb::object, PyOperation> {
            return PyOperation::forOperation(self.getContext(),
                                             mlirModuleGetOperation(self.get()),
                                             self.getRef().releaseObject())
                .releaseObject();
          },
          "Accesses the module as an operation.")
      .def_prop_ro(
          "body",
          [](PyModule &self) {
            PyOperationRef moduleOp = PyOperation::forOperation(
                self.getContext(), mlirModuleGetOperation(self.get()),
                self.getRef().releaseObject());
            PyBlock returnBlock(moduleOp, mlirModuleGetBody(self.get()));
            return returnBlock;
          },
          "Return the block for this module.")
      .def(
          "dump",
          [](PyModule &self) {
            mlirOperationDump(mlirModuleGetOperation(self.get()));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](const nb::object &self) {
            // Defer to the operation's __str__.
            return self.attr("operation").attr("__str__")();
          },
          nb::sig("def __str__(self) -> str"),
          R"(
            Gets the assembly form of the operation with default options.

            If more advanced control over the assembly formatting or I/O options is needed,
            use the dedicated print or get_asm method, which supports keyword arguments to
            customize behavior.
          )")
      .def(
          "__eq__",
          [](PyModule &self, PyModule &other) {
            return mlirModuleEqual(self.get(), other.get());
          },
          "other"_a, "Compares two modules for equality.")
      .def(
          "__hash__",
          [](PyModule &self) { return mlirModuleHashValue(self.get()); },
          "Returns the hash value of the module.");

  //----------------------------------------------------------------------------
  // Mapping of Operation.
  //----------------------------------------------------------------------------
  nb::class_<PyOperationBase>(m, "_OperationBase")
      .def_prop_ro(
          MLIR_PYTHON_CAPI_PTR_ATTR,
          [](PyOperationBase &self) {
            return self.getOperation().getCapsule();
          },
          "Gets a capsule wrapping the `MlirOperation`.")
      .def(
          "__eq__",
          [](PyOperationBase &self, PyOperationBase &other) {
            return mlirOperationEqual(self.getOperation().get(),
                                      other.getOperation().get());
          },
          "Compares two operations for equality.")
      .def(
          "__eq__",
          [](PyOperationBase &self, nb::object other) { return false; },
          "Compares operation with non-operation object (always returns "
          "False).")
      .def(
          "__hash__",
          [](PyOperationBase &self) {
            return mlirOperationHashValue(self.getOperation().get());
          },
          "Returns the hash value of the operation.")
      .def_prop_ro(
          "attributes",
          [](PyOperationBase &self) {
            return PyOpAttributeMap(self.getOperation().getRef());
          },
          "Returns a dictionary-like map of operation attributes.")
      .def_prop_ro(
          "context",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyMlirContext> {
            PyOperation &concreteOperation = self.getOperation();
            concreteOperation.checkValid();
            return concreteOperation.getContext().getObject();
          },
          "Context that owns the operation.")
      .def_prop_ro(
          "name",
          [](PyOperationBase &self) {
            auto &concreteOperation = self.getOperation();
            concreteOperation.checkValid();
            MlirOperation operation = concreteOperation.get();
            return mlirIdentifierStr(mlirOperationGetName(operation));
          },
          "Returns the fully qualified name of the operation.")
      .def_prop_ro(
          "operands",
          [](PyOperationBase &self) {
            return PyOpOperandList(self.getOperation().getRef());
          },
          "Returns the list of operation operands.")
      .def_prop_ro(
          "regions",
          [](PyOperationBase &self) {
            return PyRegionList(self.getOperation().getRef());
          },
          "Returns the list of operation regions.")
      .def_prop_ro(
          "results",
          [](PyOperationBase &self) {
            return PyOpResultList(self.getOperation().getRef());
          },
          "Returns the list of Operation results.")
      .def_prop_ro(
          "result",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyOpResult> {
            auto &operation = self.getOperation();
            return PyOpResult(operation.getRef(), getUniqueResult(operation))
                .maybeDownCast();
          },
          "Shortcut to get an op result if it has only one (throws an error "
          "otherwise).")
      .def_prop_rw(
          "location",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            return PyLocation(operation.getContext(),
                              mlirOperationGetLocation(operation.get()));
          },
          [](PyOperationBase &self, const PyLocation &location) {
            PyOperation &operation = self.getOperation();
            mlirOperationSetLocation(operation.get(), location.get());
          },
          nb::for_getter("Returns the source location the operation was "
                         "defined or derived from."),
          nb::for_setter("Sets the source location the operation was defined "
                         "or derived from."))
      .def_prop_ro(
          "parent",
          [](PyOperationBase &self)
              -> std::optional<nb::typed<nb::object, PyOperation>> {
            auto parent = self.getOperation().getParentOperation();
            if (parent)
              return parent->getObject();
            return {};
          },
          "Returns the parent operation, or `None` if at top level.")
      .def(
          "__str__",
          [](PyOperationBase &self) {
            return self.getAsm(/*binary=*/false,
                               /*largeElementsLimit=*/std::nullopt,
                               /*largeResourceLimit=*/std::nullopt,
                               /*enableDebugInfo=*/false,
                               /*prettyDebugInfo=*/false,
                               /*printGenericOpForm=*/false,
                               /*useLocalScope=*/false,
                               /*useNameLocAsPrefix=*/false,
                               /*assumeVerified=*/false,
                               /*skipRegions=*/false);
          },
          nb::sig("def __str__(self) -> str"),
          "Returns the assembly form of the operation.")
      .def("print",
           nb::overload_cast<PyAsmState &, nb::object, bool>(
               &PyOperationBase::print),
           nb::arg("state"), nb::arg("file") = nb::none(),
           nb::arg("binary") = false,
           R"(
             Prints the assembly form of the operation to a file like object.

             Args:
               state: `AsmState` capturing the operation numbering and flags.
               file: Optional file like object to write to. Defaults to sys.stdout.
               binary: Whether to write `bytes` (True) or `str` (False). Defaults to False.)")
      .def("print",
           nb::overload_cast<std::optional<int64_t>, std::optional<int64_t>,
                             bool, bool, bool, bool, bool, bool, nb::object,
                             bool, bool>(&PyOperationBase::print),
           // Careful: Lots of arguments must match up with print method.
           nb::arg("large_elements_limit") = nb::none(),
           nb::arg("large_resource_limit") = nb::none(),
           nb::arg("enable_debug_info") = false,
           nb::arg("pretty_debug_info") = false,
           nb::arg("print_generic_op_form") = false,
           nb::arg("use_local_scope") = false,
           nb::arg("use_name_loc_as_prefix") = false,
           nb::arg("assume_verified") = false, nb::arg("file") = nb::none(),
           nb::arg("binary") = false, nb::arg("skip_regions") = false,
           R"(
             Prints the assembly form of the operation to a file like object.

             Args:
               large_elements_limit: Whether to elide elements attributes above this
                 number of elements. Defaults to None (no limit).
               large_resource_limit: Whether to elide resource attributes above this
                 number of characters. Defaults to None (no limit). If large_elements_limit
                 is set and this is None, the behavior will be to use large_elements_limit
                 as large_resource_limit.
               enable_debug_info: Whether to print debug/location information. Defaults
                 to False.
               pretty_debug_info: Whether to format debug information for easier reading
                 by a human (warning: the result is unparseable). Defaults to False.
               print_generic_op_form: Whether to print the generic assembly forms of all
                 ops. Defaults to False.
               use_local_scope: Whether to print in a way that is more optimized for
                 multi-threaded access but may not be consistent with how the overall
                 module prints.
               use_name_loc_as_prefix: Whether to use location attributes (NameLoc) as
                 prefixes for the SSA identifiers. Defaults to False.
               assume_verified: By default, if not printing generic form, the verifier
                 will be run and if it fails, generic form will be printed with a comment
                 about failed verification. While a reasonable default for interactive use,
                 for systematic use, it is often better for the caller to verify explicitly
                 and report failures in a more robust fashion. Set this to True if doing this
                 in order to avoid running a redundant verification. If the IR is actually
                 invalid, behavior is undefined.
               file: The file like object to write to. Defaults to sys.stdout.
               binary: Whether to write bytes (True) or str (False). Defaults to False.
               skip_regions: Whether to skip printing regions. Defaults to False.)")
      .def("write_bytecode", &PyOperationBase::writeBytecode, nb::arg("file"),
           nb::arg("desired_version") = nb::none(),
           R"(
             Write the bytecode form of the operation to a file like object.

             Args:
               file: The file like object to write to.
               desired_version: Optional version of bytecode to emit.
             Returns:
               The bytecode writer status.)")
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           nb::arg("binary") = false,
           nb::arg("large_elements_limit") = nb::none(),
           nb::arg("large_resource_limit") = nb::none(),
           nb::arg("enable_debug_info") = false,
           nb::arg("pretty_debug_info") = false,
           nb::arg("print_generic_op_form") = false,
           nb::arg("use_local_scope") = false,
           nb::arg("use_name_loc_as_prefix") = false,
           nb::arg("assume_verified") = false, nb::arg("skip_regions") = false,
           R"(
            Gets the assembly form of the operation with all options available.

            Args:
              binary: Whether to return a bytes (True) or str (False) object. Defaults to
                False.
              ... others ...: See the print() method for common keyword arguments for
                configuring the printout.
            Returns:
              Either a bytes or str object, depending on the setting of the `binary`
              argument.)")
      .def("verify", &PyOperationBase::verify,
           "Verify the operation. Raises MLIRError if verification fails, and "
           "returns true otherwise.")
      .def("move_after", &PyOperationBase::moveAfter, nb::arg("other"),
           "Puts self immediately after the other operation in its parent "
           "block.")
      .def("move_before", &PyOperationBase::moveBefore, nb::arg("other"),
           "Puts self immediately before the other operation in its parent "
           "block.")
      .def("is_before_in_block", &PyOperationBase::isBeforeInBlock,
           nb::arg("other"),
           R"(
             Checks if this operation is before another in the same block.

             Args:
               other: Another operation in the same parent block.

             Returns:
               True if this operation is before `other` in the operation list of the parent block.)")
      .def(
          "clone",
          [](PyOperationBase &self,
             const nb::object &ip) -> nb::typed<nb::object, PyOperation> {
            return self.getOperation().clone(ip);
          },
          nb::arg("ip") = nb::none(),
          R"(
            Creates a deep copy of the operation.

            Args:
              ip: Optional insertion point where the cloned operation should be inserted.
                If None, the current insertion point is used. If False, the operation
                remains detached.

            Returns:
              A new Operation that is a clone of this operation.)")
      .def(
          "detach_from_parent",
          [](PyOperationBase &self) -> nb::typed<nb::object, PyOpView> {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            if (!operation.isAttached())
              throw nb::value_error("Detached operation has no parent.");

            operation.detachFromParent();
            return operation.createOpView();
          },
          "Detaches the operation from its parent block.")
      .def_prop_ro(
          "attached",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            return operation.isAttached();
          },
          "Reports if the operation is attached to its parent block.")
      .def(
          "erase", [](PyOperationBase &self) { self.getOperation().erase(); },
          R"(
            Erases the operation and frees its memory.

            Note:
              After erasing, any Python references to the operation become invalid.)")
      .def("walk", &PyOperationBase::walk, nb::arg("callback"),
           nb::arg("walk_order") = MlirWalkPostOrder,
           // clang-format off
          nb::sig("def walk(self, callback: Callable[[Operation], WalkResult], walk_order: WalkOrder) -> None"),
           // clang-format on
           R"(
             Walks the operation tree with a callback function.

             Args:
               callback: A callable that takes an Operation and returns a WalkResult.
               walk_order: The order of traversal (PRE_ORDER or POST_ORDER).)");

  nb::class_<PyOperation, PyOperationBase>(m, "Operation")
      .def_static(
          "create",
          [](std::string_view name,
             std::optional<std::vector<PyType *>> results,
             std::optional<std::vector<PyValue *>> operands,
             std::optional<nb::dict> attributes,
             std::optional<std::vector<PyBlock *>> successors, int regions,
             const std::optional<PyLocation> &location,
             const nb::object &maybeIp,
             bool inferType) -> nb::typed<nb::object, PyOperation> {
            // Unpack/validate operands.
            llvm::SmallVector<MlirValue, 4> mlirOperands;
            if (operands) {
              mlirOperands.reserve(operands->size());
              for (PyValue *operand : *operands) {
                if (!operand)
                  throw nb::value_error("operand value cannot be None");
                mlirOperands.push_back(operand->get());
              }
            }

            PyLocation pyLoc = maybeGetTracebackLocation(location);
            return PyOperation::create(name, results, mlirOperands, attributes,
                                       successors, regions, pyLoc, maybeIp,
                                       inferType);
          },
          nb::arg("name"), nb::arg("results") = nb::none(),
          nb::arg("operands") = nb::none(), nb::arg("attributes") = nb::none(),
          nb::arg("successors") = nb::none(), nb::arg("regions") = 0,
          nb::arg("loc") = nb::none(), nb::arg("ip") = nb::none(),
          nb::arg("infer_type") = false,
          R"(
            Creates a new operation.

            Args:
              name: Operation name (e.g. `dialect.operation`).
              results: Optional sequence of Type representing op result types.
              operands: Optional operands of the operation.
              attributes: Optional Dict of {str: Attribute}.
              successors: Optional List of Block for the operation's successors.
              regions: Number of regions to create (default = 0).
              location: Optional Location object (defaults to resolve from context manager).
              ip: Optional InsertionPoint (defaults to resolve from context manager or set to False to disable insertion, even with an insertion point set in the context manager).
              infer_type: Whether to infer result types (default = False).
            Returns:
              A new detached Operation object. Detached operations can be added to blocks, which causes them to become attached.)")
      .def_static(
          "parse",
          [](const std::string &sourceStr, const std::string &sourceName,
             DefaultingPyMlirContext context)
              -> nb::typed<nb::object, PyOpView> {
            return PyOperation::parse(context->getRef(), sourceStr, sourceName)
                ->createOpView();
          },
          nb::arg("source"), nb::kw_only(), nb::arg("source_name") = "",
          nb::arg("context") = nb::none(),
          "Parses an operation. Supports both text assembly format and binary "
          "bytecode format.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyOperation::getCapsule,
                   "Gets a capsule wrapping the MlirOperation.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR,
                  &PyOperation::createFromCapsule,
                  "Creates an Operation from a capsule wrapping MlirOperation.")
      .def_prop_ro(
          "operation",
          [](nb::object self) -> nb::typed<nb::object, PyOperation> {
            return self;
          },
          "Returns self (the operation).")
      .def_prop_ro(
          "opview",
          [](PyOperation &self) -> nb::typed<nb::object, PyOpView> {
            return self.createOpView();
          },
          R"(
            Returns an OpView of this operation.

            Note:
              If the operation has a registered and loaded dialect then this OpView will
              be concrete wrapper class.)")
      .def_prop_ro("block", &PyOperation::getBlock,
                   "Returns the block containing this operation.")
      .def_prop_ro(
          "successors",
          [](PyOperationBase &self) {
            return PyOpSuccessors(self.getOperation().getRef());
          },
          "Returns the list of Operation successors.")
      .def("_set_invalid", &PyOperation::setInvalid,
           "Invalidate the operation.");

  auto opViewClass =
      nb::class_<PyOpView, PyOperationBase>(m, "OpView")
          .def(nb::init<nb::typed<nb::object, PyOperation>>(),
               nb::arg("operation"))
          .def(
              "__init__",
              [](PyOpView *self, std::string_view name,
                 std::tuple<int, bool> opRegionSpec,
                 nb::object operandSegmentSpecObj,
                 nb::object resultSegmentSpecObj,
                 std::optional<nb::list> resultTypeList, nb::list operandList,
                 std::optional<nb::dict> attributes,
                 std::optional<std::vector<PyBlock *>> successors,
                 std::optional<int> regions,
                 const std::optional<PyLocation> &location,
                 const nb::object &maybeIp) {
                PyLocation pyLoc = maybeGetTracebackLocation(location);
                new (self) PyOpView(PyOpView::buildGeneric(
                    name, opRegionSpec, operandSegmentSpecObj,
                    resultSegmentSpecObj, resultTypeList, operandList,
                    attributes, successors, regions, pyLoc, maybeIp));
              },
              nb::arg("name"), nb::arg("opRegionSpec"),
              nb::arg("operandSegmentSpecObj") = nb::none(),
              nb::arg("resultSegmentSpecObj") = nb::none(),
              nb::arg("results") = nb::none(), nb::arg("operands") = nb::none(),
              nb::arg("attributes") = nb::none(),
              nb::arg("successors") = nb::none(),
              nb::arg("regions") = nb::none(), nb::arg("loc") = nb::none(),
              nb::arg("ip") = nb::none())
          .def_prop_ro(
              "operation",
              [](PyOpView &self) -> nb::typed<nb::object, PyOperation> {
                return self.getOperationObject();
              })
          .def_prop_ro("opview",
                       [](nb::object self) -> nb::typed<nb::object, PyOpView> {
                         return self;
                       })
          .def(
              "__str__",
              [](PyOpView &self) { return nb::str(self.getOperationObject()); })
          .def_prop_ro(
              "successors",
              [](PyOperationBase &self) {
                return PyOpSuccessors(self.getOperation().getRef());
              },
              "Returns the list of Operation successors.")
          .def(
              "_set_invalid",
              [](PyOpView &self) { self.getOperation().setInvalid(); },
              "Invalidate the operation.");
  opViewClass.attr("_ODS_REGIONS") = nb::make_tuple(0, true);
  opViewClass.attr("_ODS_OPERAND_SEGMENTS") = nb::none();
  opViewClass.attr("_ODS_RESULT_SEGMENTS") = nb::none();
  // It is faster to pass the operation_name, ods_regions, and
  // ods_operand_segments/ods_result_segments as arguments to the constructor,
  // rather than to access them as attributes.
  opViewClass.attr("build_generic") = classmethod(
      [](nb::handle cls, std::optional<nb::list> resultTypeList,
         nb::list operandList, std::optional<nb::dict> attributes,
         std::optional<std::vector<PyBlock *>> successors,
         std::optional<int> regions, std::optional<PyLocation> location,
         const nb::object &maybeIp) {
        std::string name = nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        std::tuple<int, bool> opRegionSpec =
            nb::cast<std::tuple<int, bool>>(cls.attr("_ODS_REGIONS"));
        nb::object operandSegmentSpec = cls.attr("_ODS_OPERAND_SEGMENTS");
        nb::object resultSegmentSpec = cls.attr("_ODS_RESULT_SEGMENTS");
        PyLocation pyLoc = maybeGetTracebackLocation(location);
        return PyOpView::buildGeneric(name, opRegionSpec, operandSegmentSpec,
                                      resultSegmentSpec, resultTypeList,
                                      operandList, attributes, successors,
                                      regions, pyLoc, maybeIp);
      },
      nb::arg("cls"), nb::arg("results") = nb::none(),
      nb::arg("operands") = nb::none(), nb::arg("attributes") = nb::none(),
      nb::arg("successors") = nb::none(), nb::arg("regions") = nb::none(),
      nb::arg("loc") = nb::none(), nb::arg("ip") = nb::none(),
      "Builds a specific, generated OpView based on class level attributes.");
  opViewClass.attr("parse") = classmethod(
      [](const nb::object &cls, const std::string &sourceStr,
         const std::string &sourceName,
         DefaultingPyMlirContext context) -> nb::typed<nb::object, PyOpView> {
        PyOperationRef parsed =
            PyOperation::parse(context->getRef(), sourceStr, sourceName);

        // Check if the expected operation was parsed, and cast to to the
        // appropriate `OpView` subclass if successful.
        // NOTE: This accesses attributes that have been automatically added to
        // `OpView` subclasses, and is not intended to be used on `OpView`
        // directly.
        std::string clsOpName =
            nb::cast<std::string>(cls.attr("OPERATION_NAME"));
        MlirStringRef identifier =
            mlirIdentifierStr(mlirOperationGetName(*parsed.get()));
        std::string_view parsedOpName(identifier.data, identifier.length);
        if (clsOpName != parsedOpName)
          throw MLIRError(Twine("Expected a '") + clsOpName + "' op, got: '" +
                          parsedOpName + "'");
        return PyOpView::constructDerived(cls, parsed.getObject());
      },
      nb::arg("cls"), nb::arg("source"), nb::kw_only(),
      nb::arg("source_name") = "", nb::arg("context") = nb::none(),
      "Parses a specific, generated OpView based on class level attributes.");

  //----------------------------------------------------------------------------
  // Mapping of PyRegion.
  //----------------------------------------------------------------------------
  nb::class_<PyRegion>(m, "Region")
      .def_prop_ro(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
      .def_prop_ro(
          "owner",
          [](PyRegion &self) -> nb::typed<nb::object, PyOpView> {
            return self.getParentOperation()->createOpView();
          },
          "Returns the operation owning this region.")
      .def(
          "__iter__",
          [](PyRegion &self) {
            self.checkValid();
            MlirBlock firstBlock = mlirRegionGetFirstBlock(self.get());
            return PyBlockIterator(self.getParentOperation(), firstBlock);
          },
          "Iterates over blocks in the region.")
      .def(
          "__eq__",
          [](PyRegion &self, PyRegion &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two regions for pointer equality.")
      .def(
          "__eq__", [](PyRegion &self, nb::object &other) { return false; },
          "Compares region with non-region object (always returns False).");

  //----------------------------------------------------------------------------
  // Mapping of PyBlock.
  //----------------------------------------------------------------------------
  nb::class_<PyBlock>(m, "Block")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyBlock::getCapsule,
                   "Gets a capsule wrapping the MlirBlock.")
      .def_prop_ro(
          "owner",
          [](PyBlock &self) -> nb::typed<nb::object, PyOpView> {
            return self.getParentOperation()->createOpView();
          },
          "Returns the owning operation of this block.")
      .def_prop_ro(
          "region",
          [](PyBlock &self) {
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            return PyRegion(self.getParentOperation(), region);
          },
          "Returns the owning region of this block.")
      .def_prop_ro(
          "arguments",
          [](PyBlock &self) {
            return PyBlockArgumentList(self.getParentOperation(), self.get());
          },
          "Returns a list of block arguments.")
      .def(
          "add_argument",
          [](PyBlock &self, const PyType &type, const PyLocation &loc) {
            return PyBlockArgument(self.getParentOperation(),
                                   mlirBlockAddArgument(self.get(), type, loc));
          },
          "type"_a, "loc"_a,
          R"(
            Appends an argument of the specified type to the block.

            Args:
              type: The type of the argument to add.
              loc: The source location for the argument.

            Returns:
              The newly added block argument.)")
      .def(
          "erase_argument",
          [](PyBlock &self, unsigned index) {
            return mlirBlockEraseArgument(self.get(), index);
          },
          nb::arg("index"),
          R"(
            Erases the argument at the specified index.

            Args:
              index: The index of the argument to erase.)")
      .def_prop_ro(
          "operations",
          [](PyBlock &self) {
            return PyOperationList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of operations.")
      .def_static(
          "create_at_start",
          [](PyRegion &parent, const nb::sequence &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            parent.checkValid();
            MlirBlock block = createBlock(pyArgTypes, pyArgLocs);
            mlirRegionInsertOwnedBlock(parent, 0, block);
            return PyBlock(parent.getParentOperation(), block);
          },
          nb::arg("parent"), nb::arg("arg_types") = nb::list(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block at the beginning of the given "
          "region (with given argument types and locations).")
      .def(
          "append_to",
          [](PyBlock &self, PyRegion &region) {
            MlirBlock b = self.get();
            if (!mlirRegionIsNull(mlirBlockGetParentRegion(b)))
              mlirBlockDetach(b);
            mlirRegionAppendOwnedBlock(region.get(), b);
          },
          nb::arg("region"),
          R"(
            Appends this block to a region.

            Transfers ownership if the block is currently owned by another region.

            Args:
              region: The region to append the block to.)")
      .def(
          "create_before",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            self.checkValid();
            MlirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockBefore(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          nb::arg("arg_types"), nb::kw_only(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block before this block "
          "(with given argument types and locations).")
      .def(
          "create_after",
          [](PyBlock &self, const nb::args &pyArgTypes,
             const std::optional<nb::sequence> &pyArgLocs) {
            self.checkValid();
            MlirBlock block =
                createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockAfter(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          nb::arg("arg_types"), nb::kw_only(),
          nb::arg("arg_locs") = std::nullopt,
          "Creates and returns a new Block after this block "
          "(with given argument types and locations).")
      .def(
          "__iter__",
          [](PyBlock &self) {
            self.checkValid();
            MlirOperation firstOperation =
                mlirBlockGetFirstOperation(self.get());
            return PyOperationIterator(self.getParentOperation(),
                                       firstOperation);
          },
          "Iterates over operations in the block.")
      .def(
          "__eq__",
          [](PyBlock &self, PyBlock &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two blocks for pointer equality.")
      .def(
          "__eq__", [](PyBlock &self, nb::object &other) { return false; },
          "Compares block with non-block object (always returns False).")
      .def(
          "__hash__",
          [](PyBlock &self) {
            return static_cast<size_t>(llvm::hash_value(self.get().ptr));
          },
          "Returns the hash value of the block.")
      .def(
          "__str__",
          [](PyBlock &self) {
            self.checkValid();
            PyPrintAccumulator printAccum;
            mlirBlockPrint(self.get(), printAccum.getCallback(),
                           printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the block.")
      .def(
          "append",
          [](PyBlock &self, PyOperationBase &operation) {
            if (operation.getOperation().isAttached())
              operation.getOperation().detachFromParent();

            MlirOperation mlirOperation = operation.getOperation().get();
            mlirBlockAppendOwnedOperation(self.get(), mlirOperation);
            operation.getOperation().setAttached(
                self.getParentOperation().getObject());
          },
          nb::arg("operation"),
          R"(
            Appends an operation to this block.

            If the operation is currently in another block, it will be moved.

            Args:
              operation: The operation to append to the block.)")
      .def_prop_ro(
          "successors",
          [](PyBlock &self) {
            return PyBlockSuccessors(self, self.getParentOperation());
          },
          "Returns the list of Block successors.")
      .def_prop_ro(
          "predecessors",
          [](PyBlock &self) {
            return PyBlockPredecessors(self, self.getParentOperation());
          },
          "Returns the list of Block predecessors.");

  //----------------------------------------------------------------------------
  // Mapping of PyInsertionPoint.
  //----------------------------------------------------------------------------

  nb::class_<PyInsertionPoint>(m, "InsertionPoint")
      .def(nb::init<PyBlock &>(), nb::arg("block"),
           "Inserts after the last operation but still inside the block.")
      .def("__enter__", &PyInsertionPoint::contextEnter,
           "Enters the insertion point as a context manager.")
      .def("__exit__", &PyInsertionPoint::contextExit,
           nb::arg("exc_type").none(), nb::arg("exc_value").none(),
           nb::arg("traceback").none(),
           "Exits the insertion point context manager.")
      .def_prop_ro_static(
          "current",
          [](nb::object & /*class*/) {
            auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
            if (!ip)
              throw nb::value_error("No current InsertionPoint");
            return ip;
          },
          nb::sig("def current(/) -> InsertionPoint"),
          "Gets the InsertionPoint bound to the current thread or raises "
          "ValueError if none has been set.")
      .def(nb::init<PyOperationBase &>(), nb::arg("beforeOperation"),
           "Inserts before a referenced operation.")
      .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin,
                  nb::arg("block"),
                  R"(
                    Creates an insertion point at the beginning of a block.

                    Args:
                      block: The block at whose beginning operations should be inserted.

                    Returns:
                      An InsertionPoint at the block's beginning.)")
      .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                  nb::arg("block"),
                  R"(
                    Creates an insertion point before a block's terminator.

                    Args:
                      block: The block whose terminator to insert before.

                    Returns:
                      An InsertionPoint before the terminator.

                    Raises:
                      ValueError: If the block has no terminator.)")
      .def_static("after", &PyInsertionPoint::after, nb::arg("operation"),
                  R"(
                    Creates an insertion point immediately after an operation.

                    Args:
                      operation: The operation after which to insert.

                    Returns:
                      An InsertionPoint after the operation.)")
      .def("insert", &PyInsertionPoint::insert, nb::arg("operation"),
           R"(
             Inserts an operation at this insertion point.

             Args:
               operation: The operation to insert.)")
      .def_prop_ro(
          "block", [](PyInsertionPoint &self) { return self.getBlock(); },
          "Returns the block that this `InsertionPoint` points to.")
      .def_prop_ro(
          "ref_operation",
          [](PyInsertionPoint &self)
              -> std::optional<nb::typed<nb::object, PyOperation>> {
            auto refOperation = self.getRefOperation();
            if (refOperation)
              return refOperation->getObject();
            return {};
          },
          "The reference operation before which new operations are "
          "inserted, or None if the insertion point is at the end of "
          "the block.");

  //----------------------------------------------------------------------------
  // Mapping of PyAttribute.
  //----------------------------------------------------------------------------
  nb::class_<PyAttribute>(m, "Attribute")
      // Delegate to the PyAttribute copy constructor, which will also lifetime
      // extend the backing context which owns the MlirAttribute.
      .def(nb::init<PyAttribute &>(), nb::arg("cast_from_type"),
           "Casts the passed attribute to the generic `Attribute`.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyAttribute::getCapsule,
                   "Gets a capsule wrapping the MlirAttribute.")
      .def_static(
          MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAttribute::createFromCapsule,
          "Creates an Attribute from a capsule wrapping `MlirAttribute`.")
      .def_static(
          "parse",
          [](const std::string &attrSpec, DefaultingPyMlirContext context)
              -> nb::typed<nb::object, PyAttribute> {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirAttribute attr = mlirAttributeParseGet(
                context->get(), toMlirStringRef(attrSpec));
            if (mlirAttributeIsNull(attr))
              throw MLIRError("Unable to parse attribute", errors.take());
            return PyAttribute(context.get()->getRef(), attr).maybeDownCast();
          },
          nb::arg("asm"), nb::arg("context") = nb::none(),
          "Parses an attribute from an assembly form. Raises an `MLIRError` on "
          "failure.")
      .def_prop_ro(
          "context",
          [](PyAttribute &self) -> nb::typed<nb::object, PyMlirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Attribute`.")
      .def_prop_ro(
          "type",
          [](PyAttribute &self) -> nb::typed<nb::object, PyType> {
            return PyType(self.getContext(), mlirAttributeGetType(self))
                .maybeDownCast();
          },
          "Returns the type of the `Attribute`.")
      .def(
          "get_named",
          [](PyAttribute &self, std::string name) {
            return PyNamedAttribute(self, std::move(name));
          },
          nb::keep_alive<0, 1>(),
          R"(
            Binds a name to the attribute, creating a `NamedAttribute`.

            Args:
              name: The name to bind to the `Attribute`.

            Returns:
              A `NamedAttribute` with the given name and this attribute.)")
      .def(
          "__eq__",
          [](PyAttribute &self, PyAttribute &other) { return self == other; },
          "Compares two attributes for equality.")
      .def(
          "__eq__", [](PyAttribute &self, nb::object &other) { return false; },
          "Compares attribute with non-attribute object (always returns "
          "False).")
      .def(
          "__hash__",
          [](PyAttribute &self) {
            return static_cast<size_t>(llvm::hash_value(self.get().ptr));
          },
          "Returns the hash value of the attribute.")
      .def(
          "dump", [](PyAttribute &self) { mlirAttributeDump(self); },
          kDumpDocstring)
      .def(
          "__str__",
          [](PyAttribute &self) {
            PyPrintAccumulator printAccum;
            mlirAttributePrint(self, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the Attribute.")
      .def(
          "__repr__",
          [](PyAttribute &self) {
            // Generally, assembly formats are not printed for __repr__ because
            // this can cause exceptionally long debug output and exceptions.
            // However, attribute values are generally considered useful and
            // are printed. This may need to be re-evaluated if debug dumps end
            // up being excessive.
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Attribute(");
            mlirAttributePrint(self, printAccum.getCallback(),
                               printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the attribute.")
      .def_prop_ro(
          "typeid",
          [](PyAttribute &self) {
            MlirTypeID mlirTypeID = mlirAttributeGetTypeID(self);
            assert(!mlirTypeIDIsNull(mlirTypeID) &&
                   "mlirTypeID was expected to be non-null.");
            return PyTypeID(mlirTypeID);
          },
          "Returns the `TypeID` of the attribute.")
      .def(
          MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyAttribute &self) -> nb::typed<nb::object, PyAttribute> {
            return self.maybeDownCast();
          },
          "Downcasts the attribute to a more specific attribute if possible.");

  //----------------------------------------------------------------------------
  // Mapping of PyNamedAttribute
  //----------------------------------------------------------------------------
  nb::class_<PyNamedAttribute>(m, "NamedAttribute")
      .def(
          "__repr__",
          [](PyNamedAttribute &self) {
            PyPrintAccumulator printAccum;
            printAccum.parts.append("NamedAttribute(");
            printAccum.parts.append(
                nb::str(mlirIdentifierStr(self.namedAttr.name).data,
                        mlirIdentifierStr(self.namedAttr.name).length));
            printAccum.parts.append("=");
            mlirAttributePrint(self.namedAttr.attribute,
                               printAccum.getCallback(),
                               printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the named attribute.")
      .def_prop_ro(
          "name",
          [](PyNamedAttribute &self) {
            return mlirIdentifierStr(self.namedAttr.name);
          },
          "The name of the `NamedAttribute` binding.")
      .def_prop_ro(
          "attr",
          [](PyNamedAttribute &self) { return self.namedAttr.attribute; },
          nb::keep_alive<0, 1>(), nb::sig("def attr(self) -> Attribute"),
          "The underlying generic attribute of the `NamedAttribute` binding.");

  //----------------------------------------------------------------------------
  // Mapping of PyType.
  //----------------------------------------------------------------------------
  nb::class_<PyType>(m, "Type")
      // Delegate to the PyType copy constructor, which will also lifetime
      // extend the backing context which owns the MlirType.
      .def(nb::init<PyType &>(), nb::arg("cast_from_type"),
           "Casts the passed type to the generic `Type`.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyType::getCapsule,
                   "Gets a capsule wrapping the `MlirType`.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyType::createFromCapsule,
                  "Creates a Type from a capsule wrapping `MlirType`.")
      .def_static(
          "parse",
          [](std::string typeSpec,
             DefaultingPyMlirContext context) -> nb::typed<nb::object, PyType> {
            PyMlirContext::ErrorCapture errors(context->getRef());
            MlirType type =
                mlirTypeParseGet(context->get(), toMlirStringRef(typeSpec));
            if (mlirTypeIsNull(type))
              throw MLIRError("Unable to parse type", errors.take());
            return PyType(context.get()->getRef(), type).maybeDownCast();
          },
          nb::arg("asm"), nb::arg("context") = nb::none(),
          R"(
            Parses the assembly form of a type.

            Returns a Type object or raises an `MLIRError` if the type cannot be parsed.

            See also: https://mlir.llvm.org/docs/LangRef/#type-system)")
      .def_prop_ro(
          "context",
          [](PyType &self) -> nb::typed<nb::object, PyMlirContext> {
            return self.getContext().getObject();
          },
          "Context that owns the `Type`.")
      .def(
          "__eq__", [](PyType &self, PyType &other) { return self == other; },
          "Compares two types for equality.")
      .def(
          "__eq__", [](PyType &self, nb::object &other) { return false; },
          nb::arg("other").none(),
          "Compares type with non-type object (always returns False).")
      .def(
          "__hash__",
          [](PyType &self) {
            return static_cast<size_t>(llvm::hash_value(self.get().ptr));
          },
          "Returns the hash value of the `Type`.")
      .def(
          "dump", [](PyType &self) { mlirTypeDump(self); }, kDumpDocstring)
      .def(
          "__str__",
          [](PyType &self) {
            PyPrintAccumulator printAccum;
            mlirTypePrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
            return printAccum.join();
          },
          "Returns the assembly form of the `Type`.")
      .def(
          "__repr__",
          [](PyType &self) {
            // Generally, assembly formats are not printed for __repr__ because
            // this can cause exceptionally long debug output and exceptions.
            // However, types are an exception as they typically have compact
            // assembly forms and printing them is useful.
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Type(");
            mlirTypePrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          "Returns a string representation of the `Type`.")
      .def(
          MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyType &self) -> nb::typed<nb::object, PyType> {
            return self.maybeDownCast();
          },
          "Downcasts the Type to a more specific `Type` if possible.")
      .def_prop_ro(
          "typeid",
          [](PyType &self) {
            MlirTypeID mlirTypeID = mlirTypeGetTypeID(self);
            if (!mlirTypeIDIsNull(mlirTypeID))
              return PyTypeID(mlirTypeID);
            auto origRepr = nb::cast<std::string>(nb::repr(nb::cast(self)));
            throw nb::value_error(
                (origRepr + llvm::Twine(" has no typeid.")).str().c_str());
          },
          "Returns the `TypeID` of the `Type`, or raises `ValueError` if "
          "`Type` has no "
          "`TypeID`.");

  //----------------------------------------------------------------------------
  // Mapping of PyTypeID.
  //----------------------------------------------------------------------------
  nb::class_<PyTypeID>(m, "TypeID")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyTypeID::getCapsule,
                   "Gets a capsule wrapping the `MlirTypeID`.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyTypeID::createFromCapsule,
                  "Creates a `TypeID` from a capsule wrapping `MlirTypeID`.")
      // Note, this tests whether the underlying TypeIDs are the same,
      // not whether the wrapper MlirTypeIDs are the same, nor whether
      // the Python objects are the same (i.e., PyTypeID is a value type).
      .def(
          "__eq__",
          [](PyTypeID &self, PyTypeID &other) { return self == other; },
          "Compares two `TypeID`s for equality.")
      .def(
          "__eq__",
          [](PyTypeID &self, const nb::object &other) { return false; },
          "Compares TypeID with non-TypeID object (always returns False).")
      // Note, this gives the hash value of the underlying TypeID, not the
      // hash value of the Python object, nor the hash value of the
      // MlirTypeID wrapper.
      .def(
          "__hash__",
          [](PyTypeID &self) {
            return static_cast<size_t>(mlirTypeIDHashValue(self));
          },
          "Returns the hash value of the `TypeID`.");

  //----------------------------------------------------------------------------
  // Mapping of Value.
  //----------------------------------------------------------------------------
  m.attr("_T") = nb::type_var("_T", nb::arg("bound") = m.attr("Type"));

  nb::class_<PyValue>(m, "Value", nb::is_generic(),
                      nb::sig("class Value(Generic[_T])"))
      .def(nb::init<PyValue &>(), nb::keep_alive<0, 1>(), nb::arg("value"),
           "Creates a Value reference from another `Value`.")
      .def_prop_ro(MLIR_PYTHON_CAPI_PTR_ATTR, &PyValue::getCapsule,
                   "Gets a capsule wrapping the `MlirValue`.")
      .def_static(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyValue::createFromCapsule,
                  "Creates a `Value` from a capsule wrapping `MlirValue`.")
      .def_prop_ro(
          "context",
          [](PyValue &self) -> nb::typed<nb::object, PyMlirContext> {
            return self.getParentOperation()->getContext().getObject();
          },
          "Context in which the value lives.")
      .def(
          "dump", [](PyValue &self) { mlirValueDump(self.get()); },
          kDumpDocstring)
      .def_prop_ro(
          "owner",
          [](PyValue &self) -> nb::object {
            MlirValue v = self.get();
            if (mlirValueIsAOpResult(v)) {
              assert(mlirOperationEqual(self.getParentOperation()->get(),
                                        mlirOpResultGetOwner(self.get())) &&
                     "expected the owner of the value in Python to match "
                     "that in "
                     "the IR");
              return self.getParentOperation().getObject();
            }

            if (mlirValueIsABlockArgument(v)) {
              MlirBlock block = mlirBlockArgumentGetOwner(self.get());
              return nb::cast(PyBlock(self.getParentOperation(), block));
            }

            assert(false && "Value must be a block argument or an op result");
            return nb::none();
          },
          "Returns the owner of the value (`Operation` for results, `Block` "
          "for "
          "arguments).")
      .def_prop_ro(
          "uses",
          [](PyValue &self) {
            return PyOpOperandIterator(mlirValueGetFirstUse(self.get()));
          },
          "Returns an iterator over uses of this value.")
      .def(
          "__eq__",
          [](PyValue &self, PyValue &other) {
            return self.get().ptr == other.get().ptr;
          },
          "Compares two values for pointer equality.")
      .def(
          "__eq__", [](PyValue &self, nb::object other) { return false; },
          "Compares value with non-value object (always returns False).")
      .def(
          "__hash__",
          [](PyValue &self) {
            return static_cast<size_t>(llvm::hash_value(self.get().ptr));
          },
          "Returns the hash value of the value.")
      .def(
          "__str__",
          [](PyValue &self) {
            PyPrintAccumulator printAccum;
            printAccum.parts.append("Value(");
            mlirValuePrint(self.get(), printAccum.getCallback(),
                           printAccum.getUserData());
            printAccum.parts.append(")");
            return printAccum.join();
          },
          R"(
            Returns the string form of the value.

            If the value is a block argument, this is the assembly form of its type and the
            position in the argument list. If the value is an operation result, this is
            equivalent to printing the operation that produced it.
          )")
      .def(
          "get_name",
          [](PyValue &self, bool useLocalScope, bool useNameLocAsPrefix) {
            PyPrintAccumulator printAccum;
            MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
            if (useLocalScope)
              mlirOpPrintingFlagsUseLocalScope(flags);
            if (useNameLocAsPrefix)
              mlirOpPrintingFlagsPrintNameLocAsPrefix(flags);
            MlirAsmState valueState =
                mlirAsmStateCreateForValue(self.get(), flags);
            mlirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            mlirOpPrintingFlagsDestroy(flags);
            mlirAsmStateDestroy(valueState);
            return printAccum.join();
          },
          nb::arg("use_local_scope") = false,
          nb::arg("use_name_loc_as_prefix") = false,
          R"(
            Returns the string form of value as an operand.

            Args:
              use_local_scope: Whether to use local scope for naming.
              use_name_loc_as_prefix: Whether to use the location attribute (NameLoc) as prefix.

            Returns:
              The value's name as it appears in IR (e.g., `%0`, `%arg0`).)")
      .def(
          "get_name",
          [](PyValue &self, PyAsmState &state) {
            PyPrintAccumulator printAccum;
            MlirAsmState valueState = state.get();
            mlirValuePrintAsOperand(self.get(), valueState,
                                    printAccum.getCallback(),
                                    printAccum.getUserData());
            return printAccum.join();
          },
          nb::arg("state"),
          "Returns the string form of value as an operand (i.e., the ValueID).")
      .def_prop_ro(
          "type",
          [](PyValue &self) -> nb::typed<nb::object, PyType> {
            return PyType(self.getParentOperation()->getContext(),
                          mlirValueGetType(self.get()))
                .maybeDownCast();
          },
          "Returns the type of the value.")
      .def(
          "set_type",
          [](PyValue &self, const PyType &type) {
            mlirValueSetType(self.get(), type);
          },
          nb::arg("type"), "Sets the type of the value.",
          nb::sig("def set_type(self, type: _T)"))
      .def(
          "replace_all_uses_with",
          [](PyValue &self, PyValue &with) {
            mlirValueReplaceAllUsesOfWith(self.get(), with.get());
          },
          "Replace all uses of value with the new value, updating anything in "
          "the IR that uses `self` to use the other value instead.")
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with, PyOperation &exception) {
            MlirOperation exceptedUser = exception.get();
            mlirValueReplaceAllUsesExcept(self, with, 1, &exceptedUser);
          },
          nb::arg("with_"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with, const nb::list &exceptions) {
            // Convert Python list to a SmallVector of MlirOperations
            llvm::SmallVector<MlirOperation> exceptionOps;
            for (nb::handle exception : exceptions) {
              exceptionOps.push_back(nb::cast<PyOperation &>(exception).get());
            }

            mlirValueReplaceAllUsesExcept(
                self, with, static_cast<intptr_t>(exceptionOps.size()),
                exceptionOps.data());
          },
          nb::arg("with_"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with, PyOperation &exception) {
            MlirOperation exceptedUser = exception.get();
            mlirValueReplaceAllUsesExcept(self, with, 1, &exceptedUser);
          },
          nb::arg("with_"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with,
             std::vector<PyOperation> &exceptions) {
            // Convert Python list to a SmallVector of MlirOperations
            llvm::SmallVector<MlirOperation> exceptionOps;
            for (PyOperation &exception : exceptions)
              exceptionOps.push_back(exception);
            mlirValueReplaceAllUsesExcept(
                self, with, static_cast<intptr_t>(exceptionOps.size()),
                exceptionOps.data());
          },
          nb::arg("with_"), nb::arg("exceptions"),
          kValueReplaceAllUsesExceptDocstring)
      .def(
          MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyValue &self) -> nb::typed<nb::object, PyValue> {
            return self.maybeDownCast();
          },
          "Downcasts the `Value` to a more specific kind if possible.")
      .def_prop_ro(
          "location",
          [](MlirValue self) {
            return PyLocation(
                PyMlirContext::forContext(mlirValueGetContext(self)),
                mlirValueGetLocation(self));
          },
          "Returns the source location of the value.");

  PyBlockArgument::bind(m);
  PyOpResult::bind(m);
  PyOpOperand::bind(m);

  nb::class_<PyAsmState>(m, "AsmState")
      .def(nb::init<PyValue &, bool>(), nb::arg("value"),
           nb::arg("use_local_scope") = false,
           R"(
             Creates an `AsmState` for consistent SSA value naming.

             Args:
               value: The value to create state for.
               use_local_scope: Whether to use local scope for naming.)")
      .def(nb::init<PyOperationBase &, bool>(), nb::arg("op"),
           nb::arg("use_local_scope") = false,
           R"(
             Creates an AsmState for consistent SSA value naming.

             Args:
               op: The operation to create state for.
               use_local_scope: Whether to use local scope for naming.)");

  //----------------------------------------------------------------------------
  // Mapping of SymbolTable.
  //----------------------------------------------------------------------------
  nb::class_<PySymbolTable>(m, "SymbolTable")
      .def(nb::init<PyOperationBase &>(),
           R"(
             Creates a symbol table for an operation.

             Args:
               operation: The `Operation` that defines a symbol table (e.g., a `ModuleOp`).

             Raises:
               TypeError: If the operation is not a symbol table.)")
      .def(
          "__getitem__",
          [](PySymbolTable &self,
             const std::string &name) -> nb::typed<nb::object, PyOpView> {
            return self.dunderGetItem(name);
          },
          R"(
            Looks up a symbol by name in the symbol table.

            Args:
              name: The name of the symbol to look up.

            Returns:
              The operation defining the symbol.

            Raises:
              KeyError: If the symbol is not found.)")
      .def("insert", &PySymbolTable::insert, nb::arg("operation"),
           R"(
             Inserts a symbol operation into the symbol table.

             Args:
               operation: An operation with a symbol name to insert.

             Returns:
               The symbol name attribute of the inserted operation.

             Raises:
               ValueError: If the operation does not have a symbol name.)")
      .def("erase", &PySymbolTable::erase, nb::arg("operation"),
           R"(
             Erases a symbol operation from the symbol table.

             Args:
               operation: The symbol operation to erase.

             Note:
               The operation is also erased from the IR and invalidated.)")
      .def("__delitem__", &PySymbolTable::dunderDel,
           "Deletes a symbol by name from the symbol table.")
      .def(
          "__contains__",
          [](PySymbolTable &table, const std::string &name) {
            return !mlirOperationIsNull(mlirSymbolTableLookup(
                table, mlirStringRefCreate(name.data(), name.length())));
          },
          "Checks if a symbol with the given name exists in the table.")
      // Static helpers.
      .def_static("set_symbol_name", &PySymbolTable::setSymbolName,
                  nb::arg("symbol"), nb::arg("name"),
                  "Sets the symbol name for a symbol operation.")
      .def_static("get_symbol_name", &PySymbolTable::getSymbolName,
                  nb::arg("symbol"),
                  "Gets the symbol name from a symbol operation.")
      .def_static("get_visibility", &PySymbolTable::getVisibility,
                  nb::arg("symbol"),
                  "Gets the visibility attribute of a symbol operation.")
      .def_static("set_visibility", &PySymbolTable::setVisibility,
                  nb::arg("symbol"), nb::arg("visibility"),
                  "Sets the visibility attribute of a symbol operation.")
      .def_static("replace_all_symbol_uses",
                  &PySymbolTable::replaceAllSymbolUses, nb::arg("old_symbol"),
                  nb::arg("new_symbol"), nb::arg("from_op"),
                  "Replaces all uses of a symbol with a new symbol name within "
                  "the given operation.")
      .def_static("walk_symbol_tables", &PySymbolTable::walkSymbolTables,
                  nb::arg("from_op"), nb::arg("all_sym_uses_visible"),
                  nb::arg("callback"),
                  "Walks symbol tables starting from an operation with a "
                  "callback function.");

  // Container bindings.
  PyBlockArgumentList::bind(m);
  PyBlockIterator::bind(m);
  PyBlockList::bind(m);
  PyBlockSuccessors::bind(m);
  PyBlockPredecessors::bind(m);
  PyOperationIterator::bind(m);
  PyOperationList::bind(m);
  PyOpAttributeMap::bind(m);
  PyOpOperandIterator::bind(m);
  PyOpOperandList::bind(m);
  PyOpResultList::bind(m);
  PyOpSuccessors::bind(m);
  PyRegionIterator::bind(m);
  PyRegionList::bind(m);

  // Debug bindings.
  PyGlobalDebugFlag::bind(m);

  // Attribute builder getter.
  PyAttrBuilderMap::bind(m);

  // nb::register_exception_translator([](const std::exception_ptr &p,
  //                                      void *payload) {
  //   // We can't define exceptions with custom fields through pybind, so
  //   instead
  //   // the exception class is defined in python and imported here.
  //   try {
  //     if (p)
  //       std::rethrow_exception(p);
  //   } catch (const MLIRError &e) {
  //     nb::object obj = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
  //                          .attr("MLIRError")(e.message, e.errorDiagnostics);
  //     PyErr_SetObject(PyExc_Exception, obj.ptr());
  //   }
  // });
}

namespace mlir::python {
void populateIRAffine(nb::module_ &m);
void populateIRAttributes(nb::module_ &m);
void populateIRInterfaces(nb::module_ &m);
void populateIRTypes(nb::module_ &m);
void registerMLIRErrorInIRCore();
} // namespace mlir::python

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";
  m.attr("T") = nb::type_var("T");
  m.attr("U") = nb::type_var("U");

  nb::class_<PyGlobals>(m, "_Globals")
      .def_prop_rw("dialect_search_modules",
                   &PyGlobals::getDialectSearchPrefixes,
                   &PyGlobals::setDialectSearchPrefixes)
      .def("append_dialect_search_prefix", &PyGlobals::addDialectSearchPrefix,
           "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, nb::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering an operation")
      .def("loc_tracebacks_enabled",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebacksEnabled();
           })
      .def("set_loc_tracebacks_enabled",
           [](PyGlobals &self, bool enabled) {
             self.getTracebackLoc().setLocTracebacksEnabled(enabled);
           })
      .def("loc_tracebacks_frame_limit",
           [](PyGlobals &self) {
             return self.getTracebackLoc().locTracebackFramesLimit();
           })
      .def("set_loc_tracebacks_frame_limit",
           [](PyGlobals &self, std::optional<int> n) {
             self.getTracebackLoc().setLocTracebackFramesLimit(
                 n.value_or(PyGlobals::TracebackLoc::kMaxFrames));
           })
      .def("register_traceback_file_inclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileInclusion(filename);
           })
      .def("register_traceback_file_exclusion",
           [](PyGlobals &self, const std::string &filename) {
             self.getTracebackLoc().registerTracebackFileExclusion(filename);
           });

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") = nb::cast(new PyGlobals, nb::rv_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](nb::type_object pyClass) {
        std::string dialectNamespace =
            nanobind::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a,
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "register_operation",
      [](const nb::type_object &dialectClass, bool replace) -> nb::object {
        return nb::cpp_function(
            [dialectClass,
             replace](nb::type_object opClass) -> nb::type_object {
              std::string operationName =
                  nanobind::cast<std::string>(opClass.attr("OPERATION_NAME"));
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);
              // Dict-stuff the new opClass by name onto the dialect class.
              nb::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      // clang-format off
      nb::sig("def register_operation(dialect_class: type, *, replace: bool = False) "
        "-> typing.Callable[[type[T]], type[T]]"),
      // clang-format on
      "dialect_class"_a, nb::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  m.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function([mlirTypeID, replace](
                                    nb::callable typeCaster) -> nb::object {
          PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      // clang-format off
      nb::sig("def register_type_caster(typeid: _mlir.ir.TypeID, *, replace: bool = False) "
                        "-> typing.Callable[[typing.Callable[[T], U]], typing.Callable[[T], U]]"),
      // clang-format on
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a type caster for casting MLIR types to custom user types.");
  m.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> nb::object {
        return nb::cpp_function(
            [mlirTypeID, replace](nb::callable valueCaster) -> nb::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      // clang-format off
      nb::sig("def register_value_caster(typeid: _mlir.ir.TypeID, *, replace: bool = False) "
                        "-> typing.Callable[[typing.Callable[[T], U]], typing.Callable[[T], U]]"),
      // clang-format on
      "typeid"_a, nb::kw_only(), "replace"_a = false,
      "Register a value caster for casting MLIR values to custom user values.");

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  auto rewriteModule = m.def_submodule("rewrite", "MLIR Rewrite Bindings");
  populateRewriteSubmodule(rewriteModule);

  // Define and populate PassManager submodule.
  auto passManagerModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passManagerModule);
  registerMLIRErrorInIRCore();
  nb::register_exception_translator([](const std::exception_ptr &p,
                                       void *payload) {
    // We can't define exceptions with custom fields through pybind, so
    // instead the exception class is defined in python and imported here.
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const MLIRError &e) {
      nb::object obj = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                           .attr("MLIRError")(e.message, e.errorDiagnostics);
      PyErr_SetObject(PyExc_Exception, obj.ptr());
    }
  });
}
