//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModule.h"

#include "Globals.h"
#include "PybindUtils.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Debug.h"
#include "mlir-c/IR.h"
//#include "mlir-c/Registration.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

//------------------------------------------------------------------------------
// Docstrings (trivial, non-duplicated docstrings are included inline).
//------------------------------------------------------------------------------

static const char kContextParseTypeDocstring[] =
    R"(Parses the assembly form of a type.

Returns a Type object or raises a ValueError if the type cannot be parsed.

See also: https://mlir.llvm.org/docs/LangRef/#type-system
)";

static const char kContextGetCallSiteLocationDocstring[] =
    R"(Gets a Location representing a caller and callsite)";

static const char kContextGetFileLocationDocstring[] =
    R"(Gets a Location representing a file, line and column)";

static const char kContextGetFusedLocationDocstring[] =
    R"(Gets a Location representing a fused location with optional metadata)";

static const char kContextGetNameLocationDocString[] =
    R"(Gets a Location representing a named location with optional child location)";

static const char kModuleParseDocstring[] =
    R"(Parses a module's assembly format from a string.

Returns a new MlirModule or raises a ValueError if the parsing fails.

See also: https://mlir.llvm.org/docs/LangRef/
)";

static const char kOperationCreateDocstring[] =
    R"(Creates a new operation.

Args:
  name: Operation name (e.g. "dialect.operation").
  results: Sequence of Type representing op result types.
  attributes: Dict of str:Attribute.
  successors: List of Block for the operation's successors.
  regions: Number of regions to create.
  location: A Location object (defaults to resolve from context manager).
  ip: An InsertionPoint (defaults to resolve from context manager or set to
    False to disable insertion, even with an insertion point set in the
    context manager).
Returns:
  A new "detached" Operation object. Detached operations can be added
  to blocks, which causes them to become "attached."
)";

static const char kOperationPrintDocstring[] =
    R"(Prints the assembly form of the operation to a file like object.

Args:
  file: The file like object to write to. Defaults to sys.stdout.
  binary: Whether to write bytes (True) or str (False). Defaults to False.
  large_elements_limit: Whether to elide elements attributes above this
    number of elements. Defaults to None (no limit).
  enable_debug_info: Whether to print debug/location information. Defaults
    to False.
  pretty_debug_info: Whether to format debug information for easier reading
    by a human (warning: the result is unparseable).
  print_generic_op_form: Whether to print the generic assembly forms of all
    ops. Defaults to False.
  use_local_Scope: Whether to print in a way that is more optimized for
    multi-threaded access but may not be consistent with how the overall
    module prints.
  assume_verified: By default, if not printing generic form, the verifier
    will be run and if it fails, generic form will be printed with a comment
    about failed verification. While a reasonable default for interactive use,
    for systematic use, it is often better for the caller to verify explicitly
    and report failures in a more robust fashion. Set this to True if doing this
    in order to avoid running a redundant verification. If the IR is actually
    invalid, behavior is undefined.
)";

static const char kOperationGetAsmDocstring[] =
    R"(Gets the assembly form of the operation with all options available.

Args:
  binary: Whether to return a bytes (True) or str (False) object. Defaults to
    False.
  ... others ...: See the print() method for common keyword arguments for
    configuring the printout.
Returns:
  Either a bytes or str object, depending on the setting of the 'binary'
  argument.
)";

static const char kOperationStrDunderDocstring[] =
    R"(Gets the assembly form of the operation with default options.

If more advanced control over the assembly formatting or I/O options is needed,
use the dedicated print or get_asm method, which supports keyword arguments to
customize behavior.
)";

static const char kDumpDocstring[] =
    R"(Dumps a debug representation of the object to stderr.)";

static const char kAppendBlockDocstring[] =
    R"(Appends a new block, with argument types as positional args.

Returns:
  The created block.
)";

static const char kValueDunderStrDocstring[] =
    R"(Returns the string form of the value.

If the value is a block argument, this is the assembly form of its type and the
position in the argument list. If the value is an operation result, this is
equivalent to printing the operation that produced it.
)";

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

/// Helper for creating an @classmethod.
template <class Func, typename... Args>
py::object classmethod(Func f, Args... args) {
  py::object cf = py::cpp_function(f, args...);
  return py::reinterpret_borrow<py::object>((PyClassMethod_New(cf.ptr())));
}

static py::object
createCustomDialectWrapper(const std::string &dialectNamespace,
                           py::object dialectDescriptor) {
  auto dialectClass = PyGlobals::get().lookupDialectClass(dialectNamespace);
  if (!dialectClass) {
    // Use the base class.
    return py::cast(PyDialect(std::move(dialectDescriptor)));
  }

  // Create the custom implementation.
  return (*dialectClass)(std::move(dialectDescriptor));
}

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

/// Wrapper for the global LLVM debugging flag.
struct PyGlobalDebugFlag {
  static void set(py::object &o, bool enable) { mlirEnableGlobalDebug(enable); }

  static bool get(const py::object &) { return mlirIsGlobalDebugEnabled(); }

  static void bind(py::module &m) {
    // Debug flags.
    py::class_<PyGlobalDebugFlag>(m, "_GlobalDebug", py::module_local())
        .def_property_static("flag", &PyGlobalDebugFlag::get,
                             &PyGlobalDebugFlag::set, "LLVM-wide debug flag");
  }
};

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

namespace {

class PyRegionIterator {
public:
  PyRegionIterator(PyOperationRef operation)
      : operation(std::move(operation)) {}

  PyRegionIterator &dunderIter() { return *this; }

  PyRegion dunderNext() {
    operation->checkValid();
    if (nextIndex >= mlirOperationGetNumRegions(operation->get())) {
      throw py::stop_iteration();
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), nextIndex++);
    return PyRegion(operation, region);
  }

  static void bind(py::module &m) {
    py::class_<PyRegionIterator>(m, "RegionIterator", py::module_local())
        .def("__iter__", &PyRegionIterator::dunderIter)
        .def("__next__", &PyRegionIterator::dunderNext);
  }

private:
  PyOperationRef operation;
  int nextIndex = 0;
};

/// Regions of an op are fixed length and indexed numerically so are represented
/// with a sequence-like container.
class PyRegionList {
public:
  PyRegionList(PyOperationRef operation) : operation(std::move(operation)) {}

  intptr_t dunderLen() {
    operation->checkValid();
    return mlirOperationGetNumRegions(operation->get());
  }

  PyRegion dunderGetItem(intptr_t index) {
    // dunderLen checks validity.
    if (index < 0 || index >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds region");
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), index);
    return PyRegion(operation, region);
  }

  static void bind(py::module &m) {
    py::class_<PyRegionList>(m, "RegionSequence", py::module_local())
        .def("__len__", &PyRegionList::dunderLen)
        .def("__getitem__", &PyRegionList::dunderGetItem);
  }

private:
  PyOperationRef operation;
};

class PyBlockIterator {
public:
  PyBlockIterator(PyOperationRef operation, MlirBlock next)
      : operation(std::move(operation)), next(next) {}

  PyBlockIterator &dunderIter() { return *this; }

  PyBlock dunderNext() {
    operation->checkValid();
    if (mlirBlockIsNull(next)) {
      throw py::stop_iteration();
    }

    PyBlock returnBlock(operation, next);
    next = mlirBlockGetNextInRegion(next);
    return returnBlock;
  }

  static void bind(py::module &m) {
    py::class_<PyBlockIterator>(m, "BlockIterator", py::module_local())
        .def("__iter__", &PyBlockIterator::dunderIter)
        .def("__next__", &PyBlockIterator::dunderNext);
  }

private:
  PyOperationRef operation;
  MlirBlock next;
};

/// Blocks are exposed by the C-API as a forward-only linked list. In Python,
/// we present them as a more full-featured list-like container but optimize
/// it for forward iteration. Blocks are always owned by a region.
class PyBlockList {
public:
  PyBlockList(PyOperationRef operation, MlirRegion region)
      : operation(std::move(operation)), region(region) {}

  PyBlockIterator dunderIter() {
    operation->checkValid();
    return PyBlockIterator(operation, mlirRegionGetFirstBlock(region));
  }

  intptr_t dunderLen() {
    operation->checkValid();
    intptr_t count = 0;
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      count += 1;
      block = mlirBlockGetNextInRegion(block);
    }
    return count;
  }

  PyBlock dunderGetItem(intptr_t index) {
    operation->checkValid();
    if (index < 0) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds block");
    }
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      if (index == 0) {
        return PyBlock(operation, block);
      }
      block = mlirBlockGetNextInRegion(block);
      index -= 1;
    }
    throw SetPyError(PyExc_IndexError, "attempt to access out of bounds block");
  }

  PyBlock appendBlock(const py::args &pyArgTypes) {
    operation->checkValid();
    llvm::SmallVector<MlirType, 4> argTypes;
    llvm::SmallVector<MlirLocation, 4> argLocs;
    argTypes.reserve(pyArgTypes.size());
    argLocs.reserve(pyArgTypes.size());
    for (auto &pyArg : pyArgTypes) {
      argTypes.push_back(pyArg.cast<PyType &>());
      // TODO: Pass in a proper location here.
      argLocs.push_back(
          mlirLocationUnknownGet(mlirTypeGetContext(argTypes.back())));
    }

    MlirBlock block =
        mlirBlockCreate(argTypes.size(), argTypes.data(), argLocs.data());
    mlirRegionAppendOwnedBlock(region, block);
    return PyBlock(operation, block);
  }

  static void bind(py::module &m) {
    py::class_<PyBlockList>(m, "BlockList", py::module_local())
        .def("__getitem__", &PyBlockList::dunderGetItem)
        .def("__iter__", &PyBlockList::dunderIter)
        .def("__len__", &PyBlockList::dunderLen)
        .def("append", &PyBlockList::appendBlock, kAppendBlockDocstring);
  }

private:
  PyOperationRef operation;
  MlirRegion region;
};

class PyOperationIterator {
public:
  PyOperationIterator(PyOperationRef parentOperation, MlirOperation next)
      : parentOperation(std::move(parentOperation)), next(next) {}

  PyOperationIterator &dunderIter() { return *this; }

  py::object dunderNext() {
    parentOperation->checkValid();
    if (mlirOperationIsNull(next)) {
      throw py::stop_iteration();
    }

    PyOperationRef returnOperation =
        PyOperation::forOperation(parentOperation->getContext(), next);
    next = mlirOperationGetNextInBlock(next);
    return returnOperation->createOpView();
  }

  static void bind(py::module &m) {
    py::class_<PyOperationIterator>(m, "OperationIterator", py::module_local())
        .def("__iter__", &PyOperationIterator::dunderIter)
        .def("__next__", &PyOperationIterator::dunderNext);
  }

private:
  PyOperationRef parentOperation;
  MlirOperation next;
};

/// Operations are exposed by the C-API as a forward-only linked list. In
/// Python, we present them as a more full-featured list-like container but
/// optimize it for forward iteration. Iterable operations are always owned
/// by a block.
class PyOperationList {
public:
  PyOperationList(PyOperationRef parentOperation, MlirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {}

  PyOperationIterator dunderIter() {
    parentOperation->checkValid();
    return PyOperationIterator(parentOperation,
                               mlirBlockGetFirstOperation(block));
  }

  intptr_t dunderLen() {
    parentOperation->checkValid();
    intptr_t count = 0;
    MlirOperation childOp = mlirBlockGetFirstOperation(block);
    while (!mlirOperationIsNull(childOp)) {
      count += 1;
      childOp = mlirOperationGetNextInBlock(childOp);
    }
    return count;
  }

  py::object dunderGetItem(intptr_t index) {
    parentOperation->checkValid();
    if (index < 0) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds operation");
    }
    MlirOperation childOp = mlirBlockGetFirstOperation(block);
    while (!mlirOperationIsNull(childOp)) {
      if (index == 0) {
        return PyOperation::forOperation(parentOperation->getContext(), childOp)
            ->createOpView();
      }
      childOp = mlirOperationGetNextInBlock(childOp);
      index -= 1;
    }
    throw SetPyError(PyExc_IndexError,
                     "attempt to access out of bounds operation");
  }

  static void bind(py::module &m) {
    py::class_<PyOperationList>(m, "OperationList", py::module_local())
        .def("__getitem__", &PyOperationList::dunderGetItem)
        .def("__iter__", &PyOperationList::dunderIter)
        .def("__len__", &PyOperationList::dunderLen);
  }

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

} // namespace

//------------------------------------------------------------------------------
// PyMlirContext
//------------------------------------------------------------------------------

PyMlirContext::PyMlirContext(MlirContext context) : context(context) {
  py::gil_scoped_acquire acquire;
  auto &liveContexts = getLiveContexts();
  liveContexts[context.ptr] = this;
}

PyMlirContext::~PyMlirContext() {
  // Note that the only public way to construct an instance is via the
  // forContext method, which always puts the associated handle into
  // liveContexts.
  py::gil_scoped_acquire acquire;
  getLiveContexts().erase(context.ptr);
  mlirContextDestroy(context);
}

py::object PyMlirContext::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonContextToCapsule(get()));
}

py::object PyMlirContext::createFromCapsule(py::object capsule) {
  MlirContext rawContext = mlirPythonCapsuleToContext(capsule.ptr());
  if (mlirContextIsNull(rawContext))
    throw py::error_already_set();
  return forContext(rawContext).releaseObject();
}

PyMlirContext *PyMlirContext::createNewContextForInit() {
  MlirContext context = mlirContextCreate();
  return new PyMlirContext(context);
}

PyMlirContextRef PyMlirContext::forContext(MlirContext context) {
  py::gil_scoped_acquire acquire;
  auto &liveContexts = getLiveContexts();
  auto it = liveContexts.find(context.ptr);
  if (it == liveContexts.end()) {
    // Create.
    PyMlirContext *unownedContextWrapper = new PyMlirContext(context);
    py::object pyRef = py::cast(unownedContextWrapper);
    assert(pyRef && "cast to py::object failed");
    liveContexts[context.ptr] = unownedContextWrapper;
    return PyMlirContextRef(unownedContextWrapper, std::move(pyRef));
  }
  // Use existing.
  py::object pyRef = py::cast(it->second);
  return PyMlirContextRef(it->second, std::move(pyRef));
}

PyMlirContext::LiveContextMap &PyMlirContext::getLiveContexts() {
  static LiveContextMap liveContexts;
  return liveContexts;
}

size_t PyMlirContext::getLiveCount() { return getLiveContexts().size(); }

size_t PyMlirContext::getLiveOperationCount() { return liveOperations.size(); }

size_t PyMlirContext::clearLiveOperations() {
  for (auto &op : liveOperations)
    op.second.second->setInvalid();
  size_t numInvalidated = liveOperations.size();
  liveOperations.clear();
  return numInvalidated;
}

size_t PyMlirContext::getLiveModuleCount() { return liveModules.size(); }

pybind11::object PyMlirContext::contextEnter() {
  return PyThreadContextEntry::pushContext(*this);
}

void PyMlirContext::contextExit(const pybind11::object &excType,
                                const pybind11::object &excVal,
                                const pybind11::object &excTb) {
  PyThreadContextEntry::popContext(*this);
}

py::object PyMlirContext::attachDiagnosticHandler(py::object callback) {
  // Note that ownership is transferred to the delete callback below by way of
  // an explicit inc_ref (borrow).
  PyDiagnosticHandler *pyHandler =
      new PyDiagnosticHandler(get(), std::move(callback));
  py::object pyHandlerObject =
      py::cast(pyHandler, py::return_value_policy::take_ownership);
  pyHandlerObject.inc_ref();

  // In these C callbacks, the userData is a PyDiagnosticHandler* that is
  // guaranteed to be known to pybind.
  auto handlerCallback =
      +[](MlirDiagnostic diagnostic, void *userData) -> MlirLogicalResult {
    PyDiagnostic *pyDiagnostic = new PyDiagnostic(diagnostic);
    py::object pyDiagnosticObject =
        py::cast(pyDiagnostic, py::return_value_policy::take_ownership);

    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    bool result = false;
    {
      // Since this can be called from arbitrary C++ contexts, always get the
      // gil.
      py::gil_scoped_acquire gil;
      try {
        result = py::cast<bool>(pyHandler->callback(pyDiagnostic));
      } catch (std::exception &e) {
        fprintf(stderr, "MLIR Python Diagnostic handler raised exception: %s\n",
                e.what());
        pyHandler->hadError = true;
      }
    }

    pyDiagnostic->invalidate();
    return result ? mlirLogicalResultSuccess() : mlirLogicalResultFailure();
  };
  auto deleteCallback = +[](void *userData) {
    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    assert(pyHandler->registeredID && "handler is not registered");
    pyHandler->registeredID.reset();

    // Decrement reference, balancing the inc_ref() above.
    py::object pyHandlerObject =
        py::cast(pyHandler, py::return_value_policy::reference);
    pyHandlerObject.dec_ref();
  };

  pyHandler->registeredID = mlirContextAttachDiagnosticHandler(
      get(), handlerCallback, static_cast<void *>(pyHandler), deleteCallback);
  return pyHandlerObject;
}

PyMlirContext &DefaultingPyMlirContext::resolve() {
  PyMlirContext *context = PyThreadContextEntry::getDefaultContext();
  if (!context) {
    throw SetPyError(
        PyExc_RuntimeError,
        "An MLIR function requires a Context but none was provided in the call "
        "or from the surrounding environment. Either pass to the function with "
        "a 'context=' argument or establish a default using 'with Context():'");
  }
  return *context;
}

//------------------------------------------------------------------------------
// PyThreadContextEntry management
//------------------------------------------------------------------------------

std::vector<PyThreadContextEntry> &PyThreadContextEntry::getStack() {
  static thread_local std::vector<PyThreadContextEntry> stack;
  return stack;
}

PyThreadContextEntry *PyThreadContextEntry::getTopOfStack() {
  auto &stack = getStack();
  if (stack.empty())
    return nullptr;
  return &stack.back();
}

void PyThreadContextEntry::push(FrameKind frameKind, py::object context,
                                py::object insertionPoint,
                                py::object location) {
  auto &stack = getStack();
  stack.emplace_back(frameKind, std::move(context), std::move(insertionPoint),
                     std::move(location));
  // If the new stack has more than one entry and the context of the new top
  // entry matches the previous, copy the insertionPoint and location from the
  // previous entry if missing from the new top entry.
  if (stack.size() > 1) {
    auto &prev = *(stack.rbegin() + 1);
    auto &current = stack.back();
    if (current.context.is(prev.context)) {
      // Default non-context objects from the previous entry.
      if (!current.insertionPoint)
        current.insertionPoint = prev.insertionPoint;
      if (!current.location)
        current.location = prev.location;
    }
  }
}

PyMlirContext *PyThreadContextEntry::getContext() {
  if (!context)
    return nullptr;
  return py::cast<PyMlirContext *>(context);
}

PyInsertionPoint *PyThreadContextEntry::getInsertionPoint() {
  if (!insertionPoint)
    return nullptr;
  return py::cast<PyInsertionPoint *>(insertionPoint);
}

PyLocation *PyThreadContextEntry::getLocation() {
  if (!location)
    return nullptr;
  return py::cast<PyLocation *>(location);
}

PyMlirContext *PyThreadContextEntry::getDefaultContext() {
  auto *tos = getTopOfStack();
  return tos ? tos->getContext() : nullptr;
}

PyInsertionPoint *PyThreadContextEntry::getDefaultInsertionPoint() {
  auto *tos = getTopOfStack();
  return tos ? tos->getInsertionPoint() : nullptr;
}

PyLocation *PyThreadContextEntry::getDefaultLocation() {
  auto *tos = getTopOfStack();
  return tos ? tos->getLocation() : nullptr;
}

py::object PyThreadContextEntry::pushContext(PyMlirContext &context) {
  py::object contextObj = py::cast(context);
  push(FrameKind::Context, /*context=*/contextObj,
       /*insertionPoint=*/py::object(),
       /*location=*/py::object());
  return contextObj;
}

void PyThreadContextEntry::popContext(PyMlirContext &context) {
  auto &stack = getStack();
  if (stack.empty())
    throw SetPyError(PyExc_RuntimeError, "Unbalanced Context enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Context && tos.getContext() != &context)
    throw SetPyError(PyExc_RuntimeError, "Unbalanced Context enter/exit");
  stack.pop_back();
}

py::object
PyThreadContextEntry::pushInsertionPoint(PyInsertionPoint &insertionPoint) {
  py::object contextObj =
      insertionPoint.getBlock().getParentOperation()->getContext().getObject();
  py::object insertionPointObj = py::cast(insertionPoint);
  push(FrameKind::InsertionPoint,
       /*context=*/contextObj,
       /*insertionPoint=*/insertionPointObj,
       /*location=*/py::object());
  return insertionPointObj;
}

void PyThreadContextEntry::popInsertionPoint(PyInsertionPoint &insertionPoint) {
  auto &stack = getStack();
  if (stack.empty())
    throw SetPyError(PyExc_RuntimeError,
                     "Unbalanced InsertionPoint enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::InsertionPoint &&
      tos.getInsertionPoint() != &insertionPoint)
    throw SetPyError(PyExc_RuntimeError,
                     "Unbalanced InsertionPoint enter/exit");
  stack.pop_back();
}

py::object PyThreadContextEntry::pushLocation(PyLocation &location) {
  py::object contextObj = location.getContext().getObject();
  py::object locationObj = py::cast(location);
  push(FrameKind::Location, /*context=*/contextObj,
       /*insertionPoint=*/py::object(),
       /*location=*/locationObj);
  return locationObj;
}

void PyThreadContextEntry::popLocation(PyLocation &location) {
  auto &stack = getStack();
  if (stack.empty())
    throw SetPyError(PyExc_RuntimeError, "Unbalanced Location enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Location && tos.getLocation() != &location)
    throw SetPyError(PyExc_RuntimeError, "Unbalanced Location enter/exit");
  stack.pop_back();
}

//------------------------------------------------------------------------------
// PyDiagnostic*
//------------------------------------------------------------------------------

void PyDiagnostic::invalidate() {
  valid = false;
  if (materializedNotes) {
    for (auto &noteObject : *materializedNotes) {
      PyDiagnostic *note = py::cast<PyDiagnostic *>(noteObject);
      note->invalidate();
    }
  }
}

PyDiagnosticHandler::PyDiagnosticHandler(MlirContext context,
                                         py::object callback)
    : context(context), callback(std::move(callback)) {}

PyDiagnosticHandler::~PyDiagnosticHandler() = default;

void PyDiagnosticHandler::detach() {
  if (!registeredID)
    return;
  MlirDiagnosticHandlerID localID = *registeredID;
  mlirContextDetachDiagnosticHandler(context, localID);
  assert(!registeredID && "should have unregistered");
  // Not strictly necessary but keeps stale pointers from being around to cause
  // issues.
  context = {nullptr};
}

void PyDiagnostic::checkValid() {
  if (!valid) {
    throw std::invalid_argument(
        "Diagnostic is invalid (used outside of callback)");
  }
}

MlirDiagnosticSeverity PyDiagnostic::getSeverity() {
  checkValid();
  return mlirDiagnosticGetSeverity(diagnostic);
}

PyLocation PyDiagnostic::getLocation() {
  checkValid();
  MlirLocation loc = mlirDiagnosticGetLocation(diagnostic);
  MlirContext context = mlirLocationGetContext(loc);
  return PyLocation(PyMlirContext::forContext(context), loc);
}

py::str PyDiagnostic::getMessage() {
  checkValid();
  py::object fileObject = py::module::import("io").attr("StringIO")();
  PyFileAccumulator accum(fileObject, /*binary=*/false);
  mlirDiagnosticPrint(diagnostic, accum.getCallback(), accum.getUserData());
  return fileObject.attr("getvalue")();
}

py::tuple PyDiagnostic::getNotes() {
  checkValid();
  if (materializedNotes)
    return *materializedNotes;
  intptr_t numNotes = mlirDiagnosticGetNumNotes(diagnostic);
  materializedNotes = py::tuple(numNotes);
  for (intptr_t i = 0; i < numNotes; ++i) {
    MlirDiagnostic noteDiag = mlirDiagnosticGetNote(diagnostic, i);
    py::object pyNoteDiag = py::cast(PyDiagnostic(noteDiag));
    PyTuple_SET_ITEM(materializedNotes->ptr(), i, pyNoteDiag.ptr());
  }
  return *materializedNotes;
}

//------------------------------------------------------------------------------
// PyDialect, PyDialectDescriptor, PyDialects, PyDialectRegistry
//------------------------------------------------------------------------------

MlirDialect PyDialects::getDialectForKey(const std::string &key,
                                         bool attrError) {
  MlirDialect dialect = mlirContextGetOrLoadDialect(getContext()->get(),
                                                    {key.data(), key.size()});
  if (mlirDialectIsNull(dialect)) {
    throw SetPyError(attrError ? PyExc_AttributeError : PyExc_IndexError,
                     Twine("Dialect '") + key + "' not found");
  }
  return dialect;
}

py::object PyDialectRegistry::getCapsule() {
  return py::reinterpret_steal<py::object>(
      mlirPythonDialectRegistryToCapsule(*this));
}

PyDialectRegistry PyDialectRegistry::createFromCapsule(py::object capsule) {
  MlirDialectRegistry rawRegistry =
      mlirPythonCapsuleToDialectRegistry(capsule.ptr());
  if (mlirDialectRegistryIsNull(rawRegistry))
    throw py::error_already_set();
  return PyDialectRegistry(rawRegistry);
}

//------------------------------------------------------------------------------
// PyLocation
//------------------------------------------------------------------------------

py::object PyLocation::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonLocationToCapsule(*this));
}

PyLocation PyLocation::createFromCapsule(py::object capsule) {
  MlirLocation rawLoc = mlirPythonCapsuleToLocation(capsule.ptr());
  if (mlirLocationIsNull(rawLoc))
    throw py::error_already_set();
  return PyLocation(PyMlirContext::forContext(mlirLocationGetContext(rawLoc)),
                    rawLoc);
}

py::object PyLocation::contextEnter() {
  return PyThreadContextEntry::pushLocation(*this);
}

void PyLocation::contextExit(const pybind11::object &excType,
                             const pybind11::object &excVal,
                             const pybind11::object &excTb) {
  PyThreadContextEntry::popLocation(*this);
}

PyLocation &DefaultingPyLocation::resolve() {
  auto *location = PyThreadContextEntry::getDefaultLocation();
  if (!location) {
    throw SetPyError(
        PyExc_RuntimeError,
        "An MLIR function requires a Location but none was provided in the "
        "call or from the surrounding environment. Either pass to the function "
        "with a 'loc=' argument or establish a default using 'with loc:'");
  }
  return *location;
}

//------------------------------------------------------------------------------
// PyModule
//------------------------------------------------------------------------------

PyModule::PyModule(PyMlirContextRef contextRef, MlirModule module)
    : BaseContextObject(std::move(contextRef)), module(module) {}

PyModule::~PyModule() {
  py::gil_scoped_acquire acquire;
  auto &liveModules = getContext()->liveModules;
  assert(liveModules.count(module.ptr) == 1 &&
         "destroying module not in live map");
  liveModules.erase(module.ptr);
  mlirModuleDestroy(module);
}

PyModuleRef PyModule::forModule(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  PyMlirContextRef contextRef = PyMlirContext::forContext(context);

  py::gil_scoped_acquire acquire;
  auto &liveModules = contextRef->liveModules;
  auto it = liveModules.find(module.ptr);
  if (it == liveModules.end()) {
    // Create.
    PyModule *unownedModule = new PyModule(std::move(contextRef), module);
    // Note that the default return value policy on cast is automatic_reference,
    // which does not take ownership (delete will not be called).
    // Just be explicit.
    py::object pyRef =
        py::cast(unownedModule, py::return_value_policy::take_ownership);
    unownedModule->handle = pyRef;
    liveModules[module.ptr] =
        std::make_pair(unownedModule->handle, unownedModule);
    return PyModuleRef(unownedModule, std::move(pyRef));
  }
  // Use existing.
  PyModule *existing = it->second.second;
  py::object pyRef = py::reinterpret_borrow<py::object>(it->second.first);
  return PyModuleRef(existing, std::move(pyRef));
}

py::object PyModule::createFromCapsule(py::object capsule) {
  MlirModule rawModule = mlirPythonCapsuleToModule(capsule.ptr());
  if (mlirModuleIsNull(rawModule))
    throw py::error_already_set();
  return forModule(rawModule).releaseObject();
}

py::object PyModule::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(get()));
}

//------------------------------------------------------------------------------
// PyOperation
//------------------------------------------------------------------------------

PyOperation::PyOperation(PyMlirContextRef contextRef, MlirOperation operation)
    : BaseContextObject(std::move(contextRef)), operation(operation) {}

PyOperation::~PyOperation() {
  // If the operation has already been invalidated there is nothing to do.
  if (!valid)
    return;
  auto &liveOperations = getContext()->liveOperations;
  assert(liveOperations.count(operation.ptr) == 1 &&
         "destroying operation not in live map");
  liveOperations.erase(operation.ptr);
  if (!isAttached()) {
    mlirOperationDestroy(operation);
  }
}

PyOperationRef PyOperation::createInstance(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  // Create.
  PyOperation *unownedOperation =
      new PyOperation(std::move(contextRef), operation);
  // Note that the default return value policy on cast is automatic_reference,
  // which does not take ownership (delete will not be called).
  // Just be explicit.
  py::object pyRef =
      py::cast(unownedOperation, py::return_value_policy::take_ownership);
  unownedOperation->handle = pyRef;
  if (parentKeepAlive) {
    unownedOperation->parentKeepAlive = std::move(parentKeepAlive);
  }
  liveOperations[operation.ptr] = std::make_pair(pyRef, unownedOperation);
  return PyOperationRef(unownedOperation, std::move(pyRef));
}

PyOperationRef PyOperation::forOperation(PyMlirContextRef contextRef,
                                         MlirOperation operation,
                                         py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  auto it = liveOperations.find(operation.ptr);
  if (it == liveOperations.end()) {
    // Create.
    return createInstance(std::move(contextRef), operation,
                          std::move(parentKeepAlive));
  }
  // Use existing.
  PyOperation *existing = it->second.second;
  py::object pyRef = py::reinterpret_borrow<py::object>(it->second.first);
  return PyOperationRef(existing, std::move(pyRef));
}

PyOperationRef PyOperation::createDetached(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           py::object parentKeepAlive) {
  auto &liveOperations = contextRef->liveOperations;
  assert(liveOperations.count(operation.ptr) == 0 &&
         "cannot create detached operation that already exists");
  (void)liveOperations;

  PyOperationRef created = createInstance(std::move(contextRef), operation,
                                          std::move(parentKeepAlive));
  created->attached = false;
  return created;
}

void PyOperation::checkValid() const {
  if (!valid) {
    throw SetPyError(PyExc_RuntimeError, "the operation has been invalidated");
  }
}

void PyOperationBase::print(py::object fileObject, bool binary,
                            llvm::Optional<int64_t> largeElementsLimit,
                            bool enableDebugInfo, bool prettyDebugInfo,
                            bool printGenericOpForm, bool useLocalScope,
                            bool assumeVerified) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = py::module::import("sys").attr("stdout");

  if (!assumeVerified && !printGenericOpForm &&
      !mlirOperationVerify(operation)) {
    std::string message("// Verification failed, printing generic form\n");
    if (binary) {
      fileObject.attr("write")(py::bytes(message));
    } else {
      fileObject.attr("write")(py::str(message));
    }
    printGenericOpForm = true;
  }

  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  if (largeElementsLimit)
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
  if (enableDebugInfo)
    mlirOpPrintingFlagsEnableDebugInfo(flags, /*prettyForm=*/prettyDebugInfo);
  if (printGenericOpForm)
    mlirOpPrintingFlagsPrintGenericOpForm(flags);
  if (useLocalScope)
    mlirOpPrintingFlagsUseLocalScope(flags);

  PyFileAccumulator accum(fileObject, binary);
  mlirOperationPrintWithFlags(operation, flags, accum.getCallback(),
                              accum.getUserData());
  mlirOpPrintingFlagsDestroy(flags);
}

py::object PyOperationBase::getAsm(bool binary,
                                   llvm::Optional<int64_t> largeElementsLimit,
                                   bool enableDebugInfo, bool prettyDebugInfo,
                                   bool printGenericOpForm, bool useLocalScope,
                                   bool assumeVerified) {
  py::object fileObject;
  if (binary) {
    fileObject = py::module::import("io").attr("BytesIO")();
  } else {
    fileObject = py::module::import("io").attr("StringIO")();
  }
  print(fileObject, /*binary=*/binary,
        /*largeElementsLimit=*/largeElementsLimit,
        /*enableDebugInfo=*/enableDebugInfo,
        /*prettyDebugInfo=*/prettyDebugInfo,
        /*printGenericOpForm=*/printGenericOpForm,
        /*useLocalScope=*/useLocalScope,
        /*assumeVerified=*/assumeVerified);

  return fileObject.attr("getvalue")();
}

void PyOperationBase::moveAfter(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  mlirOperationMoveAfter(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

void PyOperationBase::moveBefore(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  mlirOperationMoveBefore(operation, otherOp);
  operation.parentKeepAlive = otherOp.parentKeepAlive;
}

llvm::Optional<PyOperationRef> PyOperation::getParentOperation() {
  checkValid();
  if (!isAttached())
    throw SetPyError(PyExc_ValueError, "Detached operations have no parent");
  MlirOperation operation = mlirOperationGetParentOperation(get());
  if (mlirOperationIsNull(operation))
    return {};
  return PyOperation::forOperation(getContext(), operation);
}

PyBlock PyOperation::getBlock() {
  checkValid();
  llvm::Optional<PyOperationRef> parentOperation = getParentOperation();
  MlirBlock block = mlirOperationGetBlock(get());
  assert(!mlirBlockIsNull(block) && "Attached operation has null parent");
  assert(parentOperation && "Operation has no parent");
  return PyBlock{std::move(*parentOperation), block};
}

py::object PyOperation::getCapsule() {
  checkValid();
  return py::reinterpret_steal<py::object>(mlirPythonOperationToCapsule(get()));
}

py::object PyOperation::createFromCapsule(py::object capsule) {
  MlirOperation rawOperation = mlirPythonCapsuleToOperation(capsule.ptr());
  if (mlirOperationIsNull(rawOperation))
    throw py::error_already_set();
  MlirContext rawCtxt = mlirOperationGetContext(rawOperation);
  return forOperation(PyMlirContext::forContext(rawCtxt), rawOperation)
      .releaseObject();
}

static void maybeInsertOperation(PyOperationRef &op,
                                 const py::object &maybeIp) {
  // InsertPoint active?
  if (!maybeIp.is(py::cast(false))) {
    PyInsertionPoint *ip;
    if (maybeIp.is_none()) {
      ip = PyThreadContextEntry::getDefaultInsertionPoint();
    } else {
      ip = py::cast<PyInsertionPoint *>(maybeIp);
    }
    if (ip)
      ip->insert(*op.get());
  }
}

py::object PyOperation::create(
    const std::string &name, llvm::Optional<std::vector<PyType *>> results,
    llvm::Optional<std::vector<PyValue *>> operands,
    llvm::Optional<py::dict> attributes,
    llvm::Optional<std::vector<PyBlock *>> successors, int regions,
    DefaultingPyLocation location, const py::object &maybeIp) {
  llvm::SmallVector<MlirValue, 4> mlirOperands;
  llvm::SmallVector<MlirType, 4> mlirResults;
  llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
  llvm::SmallVector<std::pair<std::string, MlirAttribute>, 4> mlirAttributes;

  // General parameter validation.
  if (regions < 0)
    throw SetPyError(PyExc_ValueError, "number of regions must be >= 0");

  // Unpack/validate operands.
  if (operands) {
    mlirOperands.reserve(operands->size());
    for (PyValue *operand : *operands) {
      if (!operand)
        throw SetPyError(PyExc_ValueError, "operand value cannot be None");
      mlirOperands.push_back(operand->get());
    }
  }

  // Unpack/validate results.
  if (results) {
    mlirResults.reserve(results->size());
    for (PyType *result : *results) {
      // TODO: Verify result type originate from the same context.
      if (!result)
        throw SetPyError(PyExc_ValueError, "result type cannot be None");
      mlirResults.push_back(*result);
    }
  }
  // Unpack/validate attributes.
  if (attributes) {
    mlirAttributes.reserve(attributes->size());
    for (auto &it : *attributes) {
      std::string key;
      try {
        key = it.first.cast<std::string>();
      } catch (py::cast_error &err) {
        std::string msg = "Invalid attribute key (not a string) when "
                          "attempting to create the operation \"" +
                          name + "\" (" + err.what() + ")";
        throw py::cast_error(msg);
      }
      try {
        auto &attribute = it.second.cast<PyAttribute &>();
        // TODO: Verify attribute originates from the same context.
        mlirAttributes.emplace_back(std::move(key), attribute);
      } catch (py::reference_cast_error &) {
        // This exception seems thrown when the value is "None".
        std::string msg =
            "Found an invalid (`None`?) attribute value for the key \"" + key +
            "\" when attempting to create the operation \"" + name + "\"";
        throw py::cast_error(msg);
      } catch (py::cast_error &err) {
        std::string msg = "Invalid attribute value for the key \"" + key +
                          "\" when attempting to create the operation \"" +
                          name + "\" (" + err.what() + ")";
        throw py::cast_error(msg);
      }
    }
  }
  // Unpack/validate successors.
  if (successors) {
    mlirSuccessors.reserve(successors->size());
    for (auto *successor : *successors) {
      // TODO: Verify successor originate from the same context.
      if (!successor)
        throw SetPyError(PyExc_ValueError, "successor block cannot be None");
      mlirSuccessors.push_back(successor->get());
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  MlirOperationState state =
      mlirOperationStateGet(toMlirStringRef(name), location);
  if (!mlirOperands.empty())
    mlirOperationStateAddOperands(&state, mlirOperands.size(),
                                  mlirOperands.data());
  if (!mlirResults.empty())
    mlirOperationStateAddResults(&state, mlirResults.size(),
                                 mlirResults.data());
  if (!mlirAttributes.empty()) {
    // Note that the attribute names directly reference bytes in
    // mlirAttributes, so that vector must not be changed from here
    // on.
    llvm::SmallVector<MlirNamedAttribute, 4> mlirNamedAttributes;
    mlirNamedAttributes.reserve(mlirAttributes.size());
    for (auto &it : mlirAttributes)
      mlirNamedAttributes.push_back(mlirNamedAttributeGet(
          mlirIdentifierGet(mlirAttributeGetContext(it.second),
                            toMlirStringRef(it.first)),
          it.second));
    mlirOperationStateAddAttributes(&state, mlirNamedAttributes.size(),
                                    mlirNamedAttributes.data());
  }
  if (!mlirSuccessors.empty())
    mlirOperationStateAddSuccessors(&state, mlirSuccessors.size(),
                                    mlirSuccessors.data());
  if (regions) {
    llvm::SmallVector<MlirRegion, 4> mlirRegions;
    mlirRegions.resize(regions);
    for (int i = 0; i < regions; ++i)
      mlirRegions[i] = mlirRegionCreate();
    mlirOperationStateAddOwnedRegions(&state, mlirRegions.size(),
                                      mlirRegions.data());
  }

  // Construct the operation.
  MlirOperation operation = mlirOperationCreate(&state);
  PyOperationRef created =
      PyOperation::createDetached(location->getContext(), operation);
  maybeInsertOperation(created, maybeIp);

  return created->createOpView();
}

py::object PyOperation::clone(const py::object &maybeIp) {
  MlirOperation clonedOperation = mlirOperationClone(operation);
  PyOperationRef cloned =
      PyOperation::createDetached(getContext(), clonedOperation);
  maybeInsertOperation(cloned, maybeIp);

  return cloned->createOpView();
}

py::object PyOperation::createOpView() {
  checkValid();
  MlirIdentifier ident = mlirOperationGetName(get());
  MlirStringRef identStr = mlirIdentifierStr(ident);
  auto opViewClass = PyGlobals::get().lookupRawOpViewClass(
      StringRef(identStr.data, identStr.length));
  if (opViewClass)
    return (*opViewClass)(getRef().getObject());
  return py::cast(PyOpView(getRef().getObject()));
}

void PyOperation::erase() {
  checkValid();
  // TODO: Fix memory hazards when erasing a tree of operations for which a deep
  // Python reference to a child operation is live. All children should also
  // have their `valid` bit set to false.
  auto &liveOperations = getContext()->liveOperations;
  if (liveOperations.count(operation.ptr))
    liveOperations.erase(operation.ptr);
  mlirOperationDestroy(operation);
  valid = false;
}

//------------------------------------------------------------------------------
// PyOpView
//------------------------------------------------------------------------------

py::object PyOpView::buildGeneric(
    const py::object &cls, py::list resultTypeList, py::list operandList,
    llvm::Optional<py::dict> attributes,
    llvm::Optional<std::vector<PyBlock *>> successors,
    llvm::Optional<int> regions, DefaultingPyLocation location,
    const py::object &maybeIp) {
  PyMlirContextRef context = location->getContext();
  // Class level operation construction metadata.
  std::string name = py::cast<std::string>(cls.attr("OPERATION_NAME"));
  // Operand and result segment specs are either none, which does no
  // variadic unpacking, or a list of ints with segment sizes, where each
  // element is either a positive number (typically 1 for a scalar) or -1 to
  // indicate that it is derived from the length of the same-indexed operand
  // or result (implying that it is a list at that position).
  py::object operandSegmentSpecObj = cls.attr("_ODS_OPERAND_SEGMENTS");
  py::object resultSegmentSpecObj = cls.attr("_ODS_RESULT_SEGMENTS");

  std::vector<uint32_t> operandSegmentLengths;
  std::vector<uint32_t> resultSegmentLengths;

  // Validate/determine region count.
  auto opRegionSpec = py::cast<std::tuple<int, bool>>(cls.attr("_ODS_REGIONS"));
  int opMinRegionCount = std::get<0>(opRegionSpec);
  bool opHasNoVariadicRegions = std::get<1>(opRegionSpec);
  if (!regions) {
    regions = opMinRegionCount;
  }
  if (*regions < opMinRegionCount) {
    throw py::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a minimum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str());
  }
  if (opHasNoVariadicRegions && *regions > opMinRegionCount) {
    throw py::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a maximum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str());
  }

  // Unpack results.
  std::vector<PyType *> resultTypes;
  resultTypes.reserve(resultTypeList.size());
  if (resultSegmentSpecObj.is_none()) {
    // Non-variadic result unpacking.
    for (const auto &it : llvm::enumerate(resultTypeList)) {
      try {
        resultTypes.push_back(py::cast<PyType *>(it.value()));
        if (!resultTypes.back())
          throw py::cast_error();
      } catch (py::cast_error &err) {
        throw py::value_error((llvm::Twine("Result ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Type (" + err.what() + ")")
                                  .str());
      }
    }
  } else {
    // Sized result unpacking.
    auto resultSegmentSpec = py::cast<std::vector<int>>(resultSegmentSpecObj);
    if (resultSegmentSpec.size() != resultTypeList.size()) {
      throw py::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(resultSegmentSpec.size()) +
                             " result segments but was provided " +
                             llvm::Twine(resultTypeList.size()))
                                .str());
    }
    resultSegmentLengths.reserve(resultTypeList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(resultTypeList, resultSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto *resultType = py::cast<PyType *>(std::get<0>(it.value()));
          if (resultType) {
            resultTypes.push_back(resultType);
            resultSegmentLengths.push_back(1);
          } else if (segmentSpec == 0) {
            // Allowed to be optional.
            resultSegmentLengths.push_back(0);
          } else {
            throw py::cast_error("was None and result is not optional");
          }
        } catch (py::cast_error &err) {
          throw py::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Type (" + err.what() +
                                 ")")
                                    .str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            resultSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = py::cast<py::sequence>(std::get<0>(it.value()));
            for (py::object segmentItem : segment) {
              resultTypes.push_back(py::cast<PyType *>(segmentItem));
              if (!resultTypes.back()) {
                throw py::cast_error("contained a None item");
              }
            }
            resultSegmentLengths.push_back(segment.size());
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw py::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Types (" +
                                 err.what() + ")")
                                    .str());
        }
      } else {
        throw py::value_error("Unexpected segment spec");
      }
    }
  }

  // Unpack operands.
  std::vector<PyValue *> operands;
  operands.reserve(operands.size());
  if (operandSegmentSpecObj.is_none()) {
    // Non-sized operand unpacking.
    for (const auto &it : llvm::enumerate(operandList)) {
      try {
        operands.push_back(py::cast<PyValue *>(it.value()));
        if (!operands.back())
          throw py::cast_error();
      } catch (py::cast_error &err) {
        throw py::value_error((llvm::Twine("Operand ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Value (" + err.what() + ")")
                                  .str());
      }
    }
  } else {
    // Sized operand unpacking.
    auto operandSegmentSpec = py::cast<std::vector<int>>(operandSegmentSpecObj);
    if (operandSegmentSpec.size() != operandList.size()) {
      throw py::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(operandSegmentSpec.size()) +
                             "operand segments but was provided " +
                             llvm::Twine(operandList.size()))
                                .str());
    }
    operandSegmentLengths.reserve(operandList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(operandList, operandSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto *operandValue = py::cast<PyValue *>(std::get<0>(it.value()));
          if (operandValue) {
            operands.push_back(operandValue);
            operandSegmentLengths.push_back(1);
          } else if (segmentSpec == 0) {
            // Allowed to be optional.
            operandSegmentLengths.push_back(0);
          } else {
            throw py::cast_error("was None and operand is not optional");
          }
        } catch (py::cast_error &err) {
          throw py::value_error((llvm::Twine("Operand ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Value (" + err.what() +
                                 ")")
                                    .str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            operandSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = py::cast<py::sequence>(std::get<0>(it.value()));
            for (py::object segmentItem : segment) {
              operands.push_back(py::cast<PyValue *>(segmentItem));
              if (!operands.back()) {
                throw py::cast_error("contained a None item");
              }
            }
            operandSegmentLengths.push_back(segment.size());
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw py::value_error((llvm::Twine("Operand ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Values (" +
                                 err.what() + ")")
                                    .str());
        }
      } else {
        throw py::value_error("Unexpected segment spec");
      }
    }
  }

  // Merge operand/result segment lengths into attributes if needed.
  if (!operandSegmentLengths.empty() || !resultSegmentLengths.empty()) {
    // Dup.
    if (attributes) {
      attributes = py::dict(*attributes);
    } else {
      attributes = py::dict();
    }
    if (attributes->contains("result_segment_sizes") ||
        attributes->contains("operand_segment_sizes")) {
      throw py::value_error("Manually setting a 'result_segment_sizes' or "
                            "'operand_segment_sizes' attribute is unsupported. "
                            "Use Operation.create for such low-level access.");
    }

    // Add result_segment_sizes attribute.
    if (!resultSegmentLengths.empty()) {
      int64_t size = resultSegmentLengths.size();
      MlirAttribute segmentLengthAttr = mlirDenseElementsAttrUInt32Get(
          mlirVectorTypeGet(1, &size, mlirIntegerTypeGet(context->get(), 32)),
          resultSegmentLengths.size(), resultSegmentLengths.data());
      (*attributes)["result_segment_sizes"] =
          PyAttribute(context, segmentLengthAttr);
    }

    // Add operand_segment_sizes attribute.
    if (!operandSegmentLengths.empty()) {
      int64_t size = operandSegmentLengths.size();
      MlirAttribute segmentLengthAttr = mlirDenseElementsAttrUInt32Get(
          mlirVectorTypeGet(1, &size, mlirIntegerTypeGet(context->get(), 32)),
          operandSegmentLengths.size(), operandSegmentLengths.data());
      (*attributes)["operand_segment_sizes"] =
          PyAttribute(context, segmentLengthAttr);
    }
  }

  // Delegate to create.
  return PyOperation::create(name,
                             /*results=*/std::move(resultTypes),
                             /*operands=*/std::move(operands),
                             /*attributes=*/std::move(attributes),
                             /*successors=*/std::move(successors),
                             /*regions=*/*regions, location, maybeIp);
}

PyOpView::PyOpView(const py::object &operationObject)
    // Casting through the PyOperationBase base-class and then back to the
    // Operation lets us accept any PyOperationBase subclass.
    : operation(py::cast<PyOperationBase &>(operationObject).getOperation()),
      operationObject(operation.getRef().getObject()) {}

py::object PyOpView::createRawSubclass(const py::object &userClass) {
  // This is... a little gross. The typical pattern is to have a pure python
  // class that extends OpView like:
  //   class AddFOp(_cext.ir.OpView):
  //     def __init__(self, loc, lhs, rhs):
  //       operation = loc.context.create_operation(
  //           "addf", lhs, rhs, results=[lhs.type])
  //       super().__init__(operation)
  //
  // I.e. The goal of the user facing type is to provide a nice constructor
  // that has complete freedom for the op under construction. This is at odds
  // with our other desire to sometimes create this object by just passing an
  // operation (to initialize the base class). We could do *arg and **kwargs
  // munging to try to make it work, but instead, we synthesize a new class
  // on the fly which extends this user class (AddFOp in this example) and
  // *give it* the base class's __init__ method, thus bypassing the
  // intermediate subclass's __init__ method entirely. While slightly,
  // underhanded, this is safe/legal because the type hierarchy has not changed
  // (we just added a new leaf) and we aren't mucking around with __new__.
  // Typically, this new class will be stored on the original as "_Raw" and will
  // be used for casts and other things that need a variant of the class that
  // is initialized purely from an operation.
  py::object parentMetaclass =
      py::reinterpret_borrow<py::object>((PyObject *)&PyType_Type);
  py::dict attributes;
  // TODO: pybind11 2.6 supports a more direct form. Upgrade many years from
  // now.
  //   auto opViewType = py::type::of<PyOpView>();
  auto opViewType = py::detail::get_type_handle(typeid(PyOpView), true);
  attributes["__init__"] = opViewType.attr("__init__");
  py::str origName = userClass.attr("__name__");
  py::str newName = py::str("_") + origName;
  return parentMetaclass(newName, py::make_tuple(userClass), attributes);
}

//------------------------------------------------------------------------------
// PyInsertionPoint.
//------------------------------------------------------------------------------

PyInsertionPoint::PyInsertionPoint(PyBlock &block) : block(block) {}

PyInsertionPoint::PyInsertionPoint(PyOperationBase &beforeOperationBase)
    : refOperation(beforeOperationBase.getOperation().getRef()),
      block((*refOperation)->getBlock()) {}

void PyInsertionPoint::insert(PyOperationBase &operationBase) {
  PyOperation &operation = operationBase.getOperation();
  if (operation.isAttached())
    throw SetPyError(PyExc_ValueError,
                     "Attempt to insert operation that is already attached");
  block.getParentOperation()->checkValid();
  MlirOperation beforeOp = {nullptr};
  if (refOperation) {
    // Insert before operation.
    (*refOperation)->checkValid();
    beforeOp = (*refOperation)->get();
  } else {
    // Insert at end (before null) is only valid if the block does not
    // already end in a known terminator (violating this will cause assertion
    // failures later).
    if (!mlirOperationIsNull(mlirBlockGetTerminator(block.get()))) {
      throw py::index_error("Cannot insert operation at the end of a block "
                            "that already has a terminator. Did you mean to "
                            "use 'InsertionPoint.at_block_terminator(block)' "
                            "versus 'InsertionPoint(block)'?");
    }
  }
  mlirBlockInsertOwnedOperationBefore(block.get(), beforeOp, operation);
  operation.setAttached();
}

PyInsertionPoint PyInsertionPoint::atBlockBegin(PyBlock &block) {
  MlirOperation firstOp = mlirBlockGetFirstOperation(block.get());
  if (mlirOperationIsNull(firstOp)) {
    // Just insert at end.
    return PyInsertionPoint(block);
  }

  // Insert before first op.
  PyOperationRef firstOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), firstOp);
  return PyInsertionPoint{block, std::move(firstOpRef)};
}

PyInsertionPoint PyInsertionPoint::atBlockTerminator(PyBlock &block) {
  MlirOperation terminator = mlirBlockGetTerminator(block.get());
  if (mlirOperationIsNull(terminator))
    throw SetPyError(PyExc_ValueError, "Block has no terminator");
  PyOperationRef terminatorOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), terminator);
  return PyInsertionPoint{block, std::move(terminatorOpRef)};
}

py::object PyInsertionPoint::contextEnter() {
  return PyThreadContextEntry::pushInsertionPoint(*this);
}

void PyInsertionPoint::contextExit(const pybind11::object &excType,
                                   const pybind11::object &excVal,
                                   const pybind11::object &excTb) {
  PyThreadContextEntry::popInsertionPoint(*this);
}

//------------------------------------------------------------------------------
// PyAttribute.
//------------------------------------------------------------------------------

bool PyAttribute::operator==(const PyAttribute &other) {
  return mlirAttributeEqual(attr, other.attr);
}

py::object PyAttribute::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonAttributeToCapsule(*this));
}

PyAttribute PyAttribute::createFromCapsule(py::object capsule) {
  MlirAttribute rawAttr = mlirPythonCapsuleToAttribute(capsule.ptr());
  if (mlirAttributeIsNull(rawAttr))
    throw py::error_already_set();
  return PyAttribute(
      PyMlirContext::forContext(mlirAttributeGetContext(rawAttr)), rawAttr);
}

//------------------------------------------------------------------------------
// PyNamedAttribute.
//------------------------------------------------------------------------------

PyNamedAttribute::PyNamedAttribute(MlirAttribute attr, std::string ownedName)
    : ownedName(new std::string(std::move(ownedName))) {
  namedAttr = mlirNamedAttributeGet(
      mlirIdentifierGet(mlirAttributeGetContext(attr),
                        toMlirStringRef(*this->ownedName)),
      attr);
}

//------------------------------------------------------------------------------
// PyType.
//------------------------------------------------------------------------------

bool PyType::operator==(const PyType &other) {
  return mlirTypeEqual(type, other.type);
}

py::object PyType::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonTypeToCapsule(*this));
}

PyType PyType::createFromCapsule(py::object capsule) {
  MlirType rawType = mlirPythonCapsuleToType(capsule.ptr());
  if (mlirTypeIsNull(rawType))
    throw py::error_already_set();
  return PyType(PyMlirContext::forContext(mlirTypeGetContext(rawType)),
                rawType);
}

//------------------------------------------------------------------------------
// PyValue and subclases.
//------------------------------------------------------------------------------

pybind11::object PyValue::getCapsule() {
  return py::reinterpret_steal<py::object>(mlirPythonValueToCapsule(get()));
}

PyValue PyValue::createFromCapsule(pybind11::object capsule) {
  MlirValue value = mlirPythonCapsuleToValue(capsule.ptr());
  if (mlirValueIsNull(value))
    throw py::error_already_set();
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  if (mlirOperationIsNull(owner))
    throw py::error_already_set();
  MlirContext ctx = mlirOperationGetContext(owner);
  PyOperationRef ownerRef =
      PyOperation::forOperation(PyMlirContext::forContext(ctx), owner);
  return PyValue(ownerRef, value);
}

//------------------------------------------------------------------------------
// PySymbolTable.
//------------------------------------------------------------------------------

PySymbolTable::PySymbolTable(PyOperationBase &operation)
    : operation(operation.getOperation().getRef()) {
  symbolTable = mlirSymbolTableCreate(operation.getOperation().get());
  if (mlirSymbolTableIsNull(symbolTable)) {
    throw py::cast_error("Operation is not a Symbol Table.");
  }
}

py::object PySymbolTable::dunderGetItem(const std::string &name) {
  operation->checkValid();
  MlirOperation symbol = mlirSymbolTableLookup(
      symbolTable, mlirStringRefCreate(name.data(), name.length()));
  if (mlirOperationIsNull(symbol))
    throw py::key_error("Symbol '" + name + "' not in the symbol table.");

  return PyOperation::forOperation(operation->getContext(), symbol,
                                   operation.getObject())
      ->createOpView();
}

void PySymbolTable::erase(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  mlirSymbolTableErase(symbolTable, symbol.getOperation().get());
  // The operation is also erased, so we must invalidate it. There may be Python
  // references to this operation so we don't want to delete it from the list of
  // live operations here.
  symbol.getOperation().valid = false;
}

void PySymbolTable::dunderDel(const std::string &name) {
  py::object operation = dunderGetItem(name);
  erase(py::cast<PyOperationBase &>(operation));
}

PyAttribute PySymbolTable::insert(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  MlirAttribute symbolAttr = mlirOperationGetAttributeByName(
      symbol.getOperation().get(), mlirSymbolTableGetSymbolAttributeName());
  if (mlirAttributeIsNull(symbolAttr))
    throw py::value_error("Expected operation to have a symbol name.");
  return PyAttribute(
      symbol.getOperation().getContext(),
      mlirSymbolTableInsert(symbolTable, symbol.getOperation().get()));
}

PyAttribute PySymbolTable::getSymbolName(PyOperationBase &symbol) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetSymbolAttributeName();
  MlirAttribute existingNameAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingNameAttr))
    throw py::value_error("Expected operation to have a symbol name.");
  return PyAttribute(symbol.getOperation().getContext(), existingNameAttr);
}

void PySymbolTable::setSymbolName(PyOperationBase &symbol,
                                  const std::string &name) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetSymbolAttributeName();
  MlirAttribute existingNameAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingNameAttr))
    throw py::value_error("Expected operation to have a symbol name.");
  MlirAttribute newNameAttr =
      mlirStringAttrGet(operation.getContext()->get(), toMlirStringRef(name));
  mlirOperationSetAttributeByName(operation.get(), attrName, newNameAttr);
}

PyAttribute PySymbolTable::getVisibility(PyOperationBase &symbol) {
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw py::value_error("Expected operation to have a symbol visibility.");
  return PyAttribute(symbol.getOperation().getContext(), existingVisAttr);
}

void PySymbolTable::setVisibility(PyOperationBase &symbol,
                                  const std::string &visibility) {
  if (visibility != "public" && visibility != "private" &&
      visibility != "nested")
    throw py::value_error(
        "Expected visibility to be 'public', 'private' or 'nested'");
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw py::value_error("Expected operation to have a symbol visibility.");
  MlirAttribute newVisAttr = mlirStringAttrGet(operation.getContext()->get(),
                                               toMlirStringRef(visibility));
  mlirOperationSetAttributeByName(operation.get(), attrName, newVisAttr);
}

void PySymbolTable::replaceAllSymbolUses(const std::string &oldSymbol,
                                         const std::string &newSymbol,
                                         PyOperationBase &from) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  if (mlirLogicalResultIsFailure(mlirSymbolTableReplaceAllSymbolUses(
          toMlirStringRef(oldSymbol), toMlirStringRef(newSymbol),
          from.getOperation())))

    throw py::value_error("Symbol rename failed");
}

void PySymbolTable::walkSymbolTables(PyOperationBase &from,
                                     bool allSymUsesVisible,
                                     py::object callback) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  struct UserData {
    PyMlirContextRef context;
    py::object callback;
    bool gotException;
    std::string exceptionWhat;
    py::object exceptionType;
  };
  UserData userData{
      fromOperation.getContext(), std::move(callback), false, {}, {}};
  mlirSymbolTableWalkSymbolTables(
      fromOperation.get(), allSymUsesVisible,
      [](MlirOperation foundOp, bool isVisible, void *calleeUserDataVoid) {
        UserData *calleeUserData = static_cast<UserData *>(calleeUserDataVoid);
        auto pyFoundOp =
            PyOperation::forOperation(calleeUserData->context, foundOp);
        if (calleeUserData->gotException)
          return;
        try {
          calleeUserData->callback(pyFoundOp.getObject(), isVisible);
        } catch (py::error_already_set &e) {
          calleeUserData->gotException = true;
          calleeUserData->exceptionWhat = e.what();
          calleeUserData->exceptionType = e.type();
        }
      },
      static_cast<void *>(&userData));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

namespace {
/// CRTP base class for Python MLIR values that subclass Value and should be
/// castable from it. The value hierarchy is one level deep and is not supposed
/// to accommodate other levels unless core MLIR changes.
template <typename DerivedTy>
class PyConcreteValue : public PyValue {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = py::class_<DerivedTy, PyValue>;
  using IsAFunctionTy = bool (*)(MlirValue);

  PyConcreteValue() = default;
  PyConcreteValue(PyOperationRef operationRef, MlirValue value)
      : PyValue(operationRef, value) {}
  PyConcreteValue(PyValue &orig)
      : PyConcreteValue(orig.getParentOperation(), castFrom(orig)) {}

  /// Attempts to cast the original value to the derived type and throws on
  /// type mismatches.
  static MlirValue castFrom(PyValue &orig) {
    if (!DerivedTy::isaFunction(orig.get())) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, Twine("Cannot cast value to ") +
                                             DerivedTy::pyClassName +
                                             " (from " + origRepr + ")");
    }
    return orig.get();
  }

  /// Binds the Python module objects to functions of this class.
  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName, py::module_local());
    cls.def(py::init<PyValue &>(), py::keep_alive<0, 1>(), py::arg("value"));
    cls.def_static(
        "isinstance",
        [](PyValue &otherValue) -> bool {
          return DerivedTy::isaFunction(otherValue);
        },
        py::arg("other_value"));
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Python wrapper for MlirBlockArgument.
class PyBlockArgument : public PyConcreteValue<PyBlockArgument> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsABlockArgument;
  static constexpr const char *pyClassName = "BlockArgument";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly("owner", [](PyBlockArgument &self) {
      return PyBlock(self.getParentOperation(),
                     mlirBlockArgumentGetOwner(self.get()));
    });
    c.def_property_readonly("arg_number", [](PyBlockArgument &self) {
      return mlirBlockArgumentGetArgNumber(self.get());
    });
    c.def(
        "set_type",
        [](PyBlockArgument &self, PyType type) {
          return mlirBlockArgumentSetType(self.get(), type);
        },
        py::arg("type"));
  }
};

/// Python wrapper for MlirOpResult.
class PyOpResult : public PyConcreteValue<PyOpResult> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsAOpResult;
  static constexpr const char *pyClassName = "OpResult";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly("owner", [](PyOpResult &self) {
      assert(
          mlirOperationEqual(self.getParentOperation()->get(),
                             mlirOpResultGetOwner(self.get())) &&
          "expected the owner of the value in Python to match that in the IR");
      return self.getParentOperation().getObject();
    });
    c.def_property_readonly("result_number", [](PyOpResult &self) {
      return mlirOpResultGetResultNumber(self.get());
    });
  }
};

/// Returns the list of types of the values held by container.
template <typename Container>
static std::vector<PyType> getValueTypes(Container &container,
                                         PyMlirContextRef &context) {
  std::vector<PyType> result;
  result.reserve(container.getNumElements());
  for (int i = 0, e = container.getNumElements(); i < e; ++i) {
    result.push_back(
        PyType(context, mlirValueGetType(container.getElement(i).get())));
  }
  return result;
}

/// A list of block arguments. Internally, these are stored as consecutive
/// elements, random access is cheap. The argument list is associated with the
/// operation that contains the block (detached blocks are not allowed in
/// Python bindings) and extends its lifetime.
class PyBlockArgumentList
    : public Sliceable<PyBlockArgumentList, PyBlockArgument> {
public:
  static constexpr const char *pyClassName = "BlockArgumentList";

  PyBlockArgumentList(PyOperationRef operation, MlirBlock block,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirBlockGetNumArguments(block) : length,
                  step),
        operation(std::move(operation)), block(block) {}

  /// Returns the number of arguments in the list.
  intptr_t getNumElements() {
    operation->checkValid();
    return mlirBlockGetNumArguments(block);
  }

  /// Returns `pos`-the element in the list. Asserts on out-of-bounds.
  PyBlockArgument getElement(intptr_t pos) {
    MlirValue argument = mlirBlockGetArgument(block, pos);
    return PyBlockArgument(operation, argument);
  }

  /// Returns a sublist of this list.
  PyBlockArgumentList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyBlockArgumentList(operation, block, startIndex, length, step);
  }

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly("types", [](PyBlockArgumentList &self) {
      return getValueTypes(self, self.operation->getContext());
    });
  }

private:
  PyOperationRef operation;
  MlirBlock block;
};

/// A list of operation operands. Internally, these are stored as consecutive
/// elements, random access is cheap. The result list is associated with the
/// operation whose results these are, and extends the lifetime of this
/// operation.
class PyOpOperandList : public Sliceable<PyOpOperandList, PyValue> {
public:
  static constexpr const char *pyClassName = "OpOperandList";

  PyOpOperandList(PyOperationRef operation, intptr_t startIndex = 0,
                  intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumOperands(operation->get())
                               : length,
                  step),
        operation(operation) {}

  intptr_t getNumElements() {
    operation->checkValid();
    return mlirOperationGetNumOperands(operation->get());
  }

  PyValue getElement(intptr_t pos) {
    MlirValue operand = mlirOperationGetOperand(operation->get(), pos);
    MlirOperation owner;
    if (mlirValueIsAOpResult(operand))
      owner = mlirOpResultGetOwner(operand);
    else if (mlirValueIsABlockArgument(operand))
      owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(operand));
    else
      assert(false && "Value must be an block arg or op result.");
    PyOperationRef pyOwner =
        PyOperation::forOperation(operation->getContext(), owner);
    return PyValue(pyOwner, operand);
  }

  PyOpOperandList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpOperandList(operation, startIndex, length, step);
  }

  void dunderSetItem(intptr_t index, PyValue value) {
    index = wrapIndex(index);
    mlirOperationSetOperand(operation->get(), index, value.get());
  }

  static void bindDerived(ClassTy &c) {
    c.def("__setitem__", &PyOpOperandList::dunderSetItem);
  }

private:
  PyOperationRef operation;
};

/// A list of operation results. Internally, these are stored as consecutive
/// elements, random access is cheap. The result list is associated with the
/// operation whose results these are, and extends the lifetime of this
/// operation.
class PyOpResultList : public Sliceable<PyOpResultList, PyOpResult> {
public:
  static constexpr const char *pyClassName = "OpResultList";

  PyOpResultList(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumResults(operation->get())
                               : length,
                  step),
        operation(operation) {}

  intptr_t getNumElements() {
    operation->checkValid();
    return mlirOperationGetNumResults(operation->get());
  }

  PyOpResult getElement(intptr_t index) {
    PyValue value(operation, mlirOperationGetResult(operation->get(), index));
    return PyOpResult(value);
  }

  PyOpResultList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpResultList(operation, startIndex, length, step);
  }

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly("types", [](PyOpResultList &self) {
      return getValueTypes(self, self.operation->getContext());
    });
  }

private:
  PyOperationRef operation;
};

/// A list of operation attributes. Can be indexed by name, producing
/// attributes, or by index, producing named attributes.
class PyOpAttributeMap {
public:
  PyOpAttributeMap(PyOperationRef operation)
      : operation(std::move(operation)) {}

  PyAttribute dunderGetItemNamed(const std::string &name) {
    MlirAttribute attr = mlirOperationGetAttributeByName(operation->get(),
                                                         toMlirStringRef(name));
    if (mlirAttributeIsNull(attr)) {
      throw SetPyError(PyExc_KeyError,
                       "attempt to access a non-existent attribute");
    }
    return PyAttribute(operation->getContext(), attr);
  }

  PyNamedAttribute dunderGetItemIndexed(intptr_t index) {
    if (index < 0 || index >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds attribute");
    }
    MlirNamedAttribute namedAttr =
        mlirOperationGetAttribute(operation->get(), index);
    return PyNamedAttribute(
        namedAttr.attribute,
        std::string(mlirIdentifierStr(namedAttr.name).data,
                    mlirIdentifierStr(namedAttr.name).length));
  }

  void dunderSetItem(const std::string &name, const PyAttribute &attr) {
    mlirOperationSetAttributeByName(operation->get(), toMlirStringRef(name),
                                    attr);
  }

  void dunderDelItem(const std::string &name) {
    int removed = mlirOperationRemoveAttributeByName(operation->get(),
                                                     toMlirStringRef(name));
    if (!removed)
      throw SetPyError(PyExc_KeyError,
                       "attempt to delete a non-existent attribute");
  }

  intptr_t dunderLen() {
    return mlirOperationGetNumAttributes(operation->get());
  }

  bool dunderContains(const std::string &name) {
    return !mlirAttributeIsNull(mlirOperationGetAttributeByName(
        operation->get(), toMlirStringRef(name)));
  }

  static void bind(py::module &m) {
    py::class_<PyOpAttributeMap>(m, "OpAttributeMap", py::module_local())
        .def("__contains__", &PyOpAttributeMap::dunderContains)
        .def("__len__", &PyOpAttributeMap::dunderLen)
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemNamed)
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemIndexed)
        .def("__setitem__", &PyOpAttributeMap::dunderSetItem)
        .def("__delitem__", &PyOpAttributeMap::dunderDelItem);
  }

private:
  PyOperationRef operation;
};

} // namespace

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRCore(py::module &m) {
  //----------------------------------------------------------------------------
  // Enums.
  //----------------------------------------------------------------------------
  py::enum_<MlirDiagnosticSeverity>(m, "DiagnosticSeverity", py::module_local())
      .value("ERROR", MlirDiagnosticError)
      .value("WARNING", MlirDiagnosticWarning)
      .value("NOTE", MlirDiagnosticNote)
      .value("REMARK", MlirDiagnosticRemark);

  //----------------------------------------------------------------------------
  // Mapping of Diagnostics.
  //----------------------------------------------------------------------------
  py::class_<PyDiagnostic>(m, "Diagnostic", py::module_local())
      .def_property_readonly("severity", &PyDiagnostic::getSeverity)
      .def_property_readonly("location", &PyDiagnostic::getLocation)
      .def_property_readonly("message", &PyDiagnostic::getMessage)
      .def_property_readonly("notes", &PyDiagnostic::getNotes)
      .def("__str__", [](PyDiagnostic &self) -> py::str {
        if (!self.isValid())
          return "<Invalid Diagnostic>";
        return self.getMessage();
      });

  py::class_<PyDiagnosticHandler>(m, "DiagnosticHandler", py::module_local())
      .def("detach", &PyDiagnosticHandler::detach)
      .def_property_readonly("attached", &PyDiagnosticHandler::isAttached)
      .def_property_readonly("had_error", &PyDiagnosticHandler::getHadError)
      .def("__enter__", &PyDiagnosticHandler::contextEnter)
      .def("__exit__", &PyDiagnosticHandler::contextExit);

  //----------------------------------------------------------------------------
  // Mapping of MlirContext.
  // Note that this is exported as _BaseContext. The containing, Python level
  // __init__.py will subclass it with site-specific functionality and set a
  // "Context" attribute on this module.
  //----------------------------------------------------------------------------
  py::class_<PyMlirContext>(m, "_BaseContext", py::module_local())
      .def(py::init<>(&PyMlirContext::createNewContextForInit))
      .def_static("_get_live_count", &PyMlirContext::getLiveCount)
      .def("_get_context_again",
           [](PyMlirContext &self) {
             PyMlirContextRef ref = PyMlirContext::forContext(self.get());
             return ref.releaseObject();
           })
      .def("_get_live_operation_count", &PyMlirContext::getLiveOperationCount)
      .def("_clear_live_operations", &PyMlirContext::clearLiveOperations)
      .def("_get_live_module_count", &PyMlirContext::getLiveModuleCount)
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyMlirContext::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyMlirContext::createFromCapsule)
      .def("__enter__", &PyMlirContext::contextEnter)
      .def("__exit__", &PyMlirContext::contextExit)
      .def_property_readonly_static(
          "current",
          [](py::object & /*class*/) {
            auto *context = PyThreadContextEntry::getDefaultContext();
            if (!context)
              throw SetPyError(PyExc_ValueError, "No current Context");
            return context;
          },
          "Gets the Context bound to the current thread or raises ValueError")
      .def_property_readonly(
          "dialects",
          [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Gets a container for accessing dialects by name")
      .def_property_readonly(
          "d", [](PyMlirContext &self) { return PyDialects(self.getRef()); },
          "Alias for 'dialect'")
      .def(
          "get_dialect_descriptor",
          [=](PyMlirContext &self, std::string &name) {
            MlirDialect dialect = mlirContextGetOrLoadDialect(
                self.get(), {name.data(), name.size()});
            if (mlirDialectIsNull(dialect)) {
              throw SetPyError(PyExc_ValueError,
                               Twine("Dialect '") + name + "' not found");
            }
            return PyDialectDescriptor(self.getRef(), dialect);
          },
          py::arg("dialect_name"),
          "Gets or loads a dialect by name, returning its descriptor object")
      .def_property(
          "allow_unregistered_dialects",
          [](PyMlirContext &self) -> bool {
            return mlirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyMlirContext &self, bool value) {
            mlirContextSetAllowUnregisteredDialects(self.get(), value);
          })
      .def("attach_diagnostic_handler", &PyMlirContext::attachDiagnosticHandler,
           py::arg("callback"),
           "Attaches a diagnostic handler that will receive callbacks")
      .def(
          "enable_multithreading",
          [](PyMlirContext &self, bool enable) {
            mlirContextEnableMultithreading(self.get(), enable);
          },
          py::arg("enable"))
      .def(
          "is_registered_operation",
          [](PyMlirContext &self, std::string &name) {
            return mlirContextIsRegisteredOperation(
                self.get(), MlirStringRef{name.data(), name.size()});
          },
          py::arg("operation_name"))
      .def(
          "append_dialect_registry",
          [](PyMlirContext &self, PyDialectRegistry &registry) {
            mlirContextAppendDialectRegistry(self.get(), registry);
          },
          py::arg("registry"))
      .def("load_all_available_dialects", [](PyMlirContext &self) {
        mlirContextLoadAllAvailableDialects(self.get());
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialectDescriptor
  //----------------------------------------------------------------------------
  py::class_<PyDialectDescriptor>(m, "DialectDescriptor", py::module_local())
      .def_property_readonly("namespace",
                             [](PyDialectDescriptor &self) {
                               MlirStringRef ns =
                                   mlirDialectGetNamespace(self.get());
                               return py::str(ns.data, ns.length);
                             })
      .def("__repr__", [](PyDialectDescriptor &self) {
        MlirStringRef ns = mlirDialectGetNamespace(self.get());
        std::string repr("<DialectDescriptor ");
        repr.append(ns.data, ns.length);
        repr.append(">");
        return repr;
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialects
  //----------------------------------------------------------------------------
  py::class_<PyDialects>(m, "Dialects", py::module_local())
      .def("__getitem__",
           [=](PyDialects &self, std::string keyName) {
             MlirDialect dialect =
                 self.getDialectForKey(keyName, /*attrError=*/false);
             py::object descriptor =
                 py::cast(PyDialectDescriptor{self.getContext(), dialect});
             return createCustomDialectWrapper(keyName, std::move(descriptor));
           })
      .def("__getattr__", [=](PyDialects &self, std::string attrName) {
        MlirDialect dialect =
            self.getDialectForKey(attrName, /*attrError=*/true);
        py::object descriptor =
            py::cast(PyDialectDescriptor{self.getContext(), dialect});
        return createCustomDialectWrapper(attrName, std::move(descriptor));
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialect
  //----------------------------------------------------------------------------
  py::class_<PyDialect>(m, "Dialect", py::module_local())
      .def(py::init<py::object>(), py::arg("descriptor"))
      .def_property_readonly(
          "descriptor", [](PyDialect &self) { return self.getDescriptor(); })
      .def("__repr__", [](py::object self) {
        auto clazz = self.attr("__class__");
        return py::str("<Dialect ") +
               self.attr("descriptor").attr("namespace") + py::str(" (class ") +
               clazz.attr("__module__") + py::str(".") +
               clazz.attr("__name__") + py::str(")>");
      });

  //----------------------------------------------------------------------------
  // Mapping of PyDialectRegistry
  //----------------------------------------------------------------------------
  py::class_<PyDialectRegistry>(m, "DialectRegistry", py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyDialectRegistry::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyDialectRegistry::createFromCapsule)
      .def(py::init<>());

  //----------------------------------------------------------------------------
  // Mapping of Location
  //----------------------------------------------------------------------------
  py::class_<PyLocation>(m, "Location", py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR, &PyLocation::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyLocation::createFromCapsule)
      .def("__enter__", &PyLocation::contextEnter)
      .def("__exit__", &PyLocation::contextExit)
      .def("__eq__",
           [](PyLocation &self, PyLocation &other) -> bool {
             return mlirLocationEqual(self, other);
           })
      .def("__eq__", [](PyLocation &self, py::object other) { return false; })
      .def_property_readonly_static(
          "current",
          [](py::object & /*class*/) {
            auto *loc = PyThreadContextEntry::getDefaultLocation();
            if (!loc)
              throw SetPyError(PyExc_ValueError, "No current Location");
            return loc;
          },
          "Gets the Location bound to the current thread or raises ValueError")
      .def_static(
          "unknown",
          [](DefaultingPyMlirContext context) {
            return PyLocation(context->getRef(),
                              mlirLocationUnknownGet(context->get()));
          },
          py::arg("context") = py::none(),
          "Gets a Location representing an unknown location")
      .def_static(
          "callsite",
          [](PyLocation callee, const std::vector<PyLocation> &frames,
             DefaultingPyMlirContext context) {
            if (frames.empty())
              throw py::value_error("No caller frames provided");
            MlirLocation caller = frames.back().get();
            for (const PyLocation &frame :
                 llvm::reverse(llvm::makeArrayRef(frames).drop_back()))
              caller = mlirLocationCallSiteGet(frame.get(), caller);
            return PyLocation(context->getRef(),
                              mlirLocationCallSiteGet(callee.get(), caller));
          },
          py::arg("callee"), py::arg("frames"), py::arg("context") = py::none(),
          kContextGetCallSiteLocationDocstring)
      .def_static(
          "file",
          [](std::string filename, int line, int col,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationFileLineColGet(
                    context->get(), toMlirStringRef(filename), line, col));
          },
          py::arg("filename"), py::arg("line"), py::arg("col"),
          py::arg("context") = py::none(), kContextGetFileLocationDocstring)
      .def_static(
          "fused",
          [](const std::vector<PyLocation> &pyLocations,
             llvm::Optional<PyAttribute> metadata,
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
          py::arg("locations"), py::arg("metadata") = py::none(),
          py::arg("context") = py::none(), kContextGetFusedLocationDocstring)
      .def_static(
          "name",
          [](std::string name, llvm::Optional<PyLocation> childLoc,
             DefaultingPyMlirContext context) {
            return PyLocation(
                context->getRef(),
                mlirLocationNameGet(
                    context->get(), toMlirStringRef(name),
                    childLoc ? childLoc->get()
                             : mlirLocationUnknownGet(context->get())));
          },
          py::arg("name"), py::arg("childLoc") = py::none(),
          py::arg("context") = py::none(), kContextGetNameLocationDocString)
      .def_property_readonly(
          "context",
          [](PyLocation &self) { return self.getContext().getObject(); },
          "Context that owns the Location")
      .def(
          "emit_error",
          [](PyLocation &self, std::string message) {
            mlirEmitError(self, message.c_str());
          },
          py::arg("message"), "Emits an error at this location")
      .def("__repr__", [](PyLocation &self) {
        PyPrintAccumulator printAccum;
        mlirLocationPrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
        return printAccum.join();
      });

  //----------------------------------------------------------------------------
  // Mapping of Module
  //----------------------------------------------------------------------------
  py::class_<PyModule>(m, "Module", py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR, &PyModule::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyModule::createFromCapsule)
      .def_static(
          "parse",
          [](const std::string moduleAsm, DefaultingPyMlirContext context) {
            MlirModule module = mlirModuleCreateParse(
                context->get(), toMlirStringRef(moduleAsm));
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirModuleIsNull(module)) {
              throw SetPyError(
                  PyExc_ValueError,
                  "Unable to parse module assembly (see diagnostics)");
            }
            return PyModule::forModule(module).releaseObject();
          },
          py::arg("asm"), py::arg("context") = py::none(),
          kModuleParseDocstring)
      .def_static(
          "create",
          [](DefaultingPyLocation loc) {
            MlirModule module = mlirModuleCreateEmpty(loc);
            return PyModule::forModule(module).releaseObject();
          },
          py::arg("loc") = py::none(), "Creates an empty module")
      .def_property_readonly(
          "context",
          [](PyModule &self) { return self.getContext().getObject(); },
          "Context that created the Module")
      .def_property_readonly(
          "operation",
          [](PyModule &self) {
            return PyOperation::forOperation(self.getContext(),
                                             mlirModuleGetOperation(self.get()),
                                             self.getRef().releaseObject())
                .releaseObject();
          },
          "Accesses the module as an operation")
      .def_property_readonly(
          "body",
          [](PyModule &self) {
            PyOperationRef moduleOp = PyOperation::forOperation(
                self.getContext(), mlirModuleGetOperation(self.get()),
                self.getRef().releaseObject());
            PyBlock returnBlock(moduleOp, mlirModuleGetBody(self.get()));
            return returnBlock;
          },
          "Return the block for this module")
      .def(
          "dump",
          [](PyModule &self) {
            mlirOperationDump(mlirModuleGetOperation(self.get()));
          },
          kDumpDocstring)
      .def(
          "__str__",
          [](py::object self) {
            // Defer to the operation's __str__.
            return self.attr("operation").attr("__str__")();
          },
          kOperationStrDunderDocstring);

  //----------------------------------------------------------------------------
  // Mapping of Operation.
  //----------------------------------------------------------------------------
  py::class_<PyOperationBase>(m, "_OperationBase", py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             [](PyOperationBase &self) {
                               return self.getOperation().getCapsule();
                             })
      .def("__eq__",
           [](PyOperationBase &self, PyOperationBase &other) {
             return &self.getOperation() == &other.getOperation();
           })
      .def("__eq__",
           [](PyOperationBase &self, py::object other) { return false; })
      .def("__hash__",
           [](PyOperationBase &self) {
             return static_cast<size_t>(llvm::hash_value(&self.getOperation()));
           })
      .def_property_readonly("attributes",
                             [](PyOperationBase &self) {
                               return PyOpAttributeMap(
                                   self.getOperation().getRef());
                             })
      .def_property_readonly("operands",
                             [](PyOperationBase &self) {
                               return PyOpOperandList(
                                   self.getOperation().getRef());
                             })
      .def_property_readonly("regions",
                             [](PyOperationBase &self) {
                               return PyRegionList(
                                   self.getOperation().getRef());
                             })
      .def_property_readonly(
          "results",
          [](PyOperationBase &self) {
            return PyOpResultList(self.getOperation().getRef());
          },
          "Returns the list of Operation results.")
      .def_property_readonly(
          "result",
          [](PyOperationBase &self) {
            auto &operation = self.getOperation();
            auto numResults = mlirOperationGetNumResults(operation);
            if (numResults != 1) {
              auto name = mlirIdentifierStr(mlirOperationGetName(operation));
              throw SetPyError(
                  PyExc_ValueError,
                  Twine("Cannot call .result on operation ") +
                      StringRef(name.data, name.length) + " which has " +
                      Twine(numResults) +
                      " results (it is only valid for operations with a "
                      "single result)");
            }
            return PyOpResult(operation.getRef(),
                              mlirOperationGetResult(operation, 0));
          },
          "Shortcut to get an op result if it has only one (throws an error "
          "otherwise).")
      .def_property_readonly(
          "location",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            return PyLocation(operation.getContext(),
                              mlirOperationGetLocation(operation.get()));
          },
          "Returns the source location the operation was defined or derived "
          "from.")
      .def(
          "__str__",
          [](PyOperationBase &self) {
            return self.getAsm(/*binary=*/false,
                               /*largeElementsLimit=*/llvm::None,
                               /*enableDebugInfo=*/false,
                               /*prettyDebugInfo=*/false,
                               /*printGenericOpForm=*/false,
                               /*useLocalScope=*/false,
                               /*assumeVerified=*/false);
          },
          "Returns the assembly form of the operation.")
      .def("print", &PyOperationBase::print,
           // Careful: Lots of arguments must match up with print method.
           py::arg("file") = py::none(), py::arg("binary") = false,
           py::arg("large_elements_limit") = py::none(),
           py::arg("enable_debug_info") = false,
           py::arg("pretty_debug_info") = false,
           py::arg("print_generic_op_form") = false,
           py::arg("use_local_scope") = false,
           py::arg("assume_verified") = false, kOperationPrintDocstring)
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           py::arg("binary") = false,
           py::arg("large_elements_limit") = py::none(),
           py::arg("enable_debug_info") = false,
           py::arg("pretty_debug_info") = false,
           py::arg("print_generic_op_form") = false,
           py::arg("use_local_scope") = false,
           py::arg("assume_verified") = false, kOperationGetAsmDocstring)
      .def(
          "verify",
          [](PyOperationBase &self) {
            return mlirOperationVerify(self.getOperation());
          },
          "Verify the operation and return true if it passes, false if it "
          "fails.")
      .def("move_after", &PyOperationBase::moveAfter, py::arg("other"),
           "Puts self immediately after the other operation in its parent "
           "block.")
      .def("move_before", &PyOperationBase::moveBefore, py::arg("other"),
           "Puts self immediately before the other operation in its parent "
           "block.")
      .def(
          "detach_from_parent",
          [](PyOperationBase &self) {
            PyOperation &operation = self.getOperation();
            operation.checkValid();
            if (!operation.isAttached())
              throw py::value_error("Detached operation has no parent.");

            operation.detachFromParent();
            return operation.createOpView();
          },
          "Detaches the operation from its parent block.");

  py::class_<PyOperation, PyOperationBase>(m, "Operation", py::module_local())
      .def_static("create", &PyOperation::create, py::arg("name"),
                  py::arg("results") = py::none(),
                  py::arg("operands") = py::none(),
                  py::arg("attributes") = py::none(),
                  py::arg("successors") = py::none(), py::arg("regions") = 0,
                  py::arg("loc") = py::none(), py::arg("ip") = py::none(),
                  kOperationCreateDocstring)
      .def_property_readonly("parent",
                             [](PyOperation &self) -> py::object {
                               auto parent = self.getParentOperation();
                               if (parent)
                                 return parent->getObject();
                               return py::none();
                             })
      .def("erase", &PyOperation::erase)
      .def("clone", &PyOperation::clone, py::arg("ip") = py::none())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyOperation::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyOperation::createFromCapsule)
      .def_property_readonly("name",
                             [](PyOperation &self) {
                               self.checkValid();
                               MlirOperation operation = self.get();
                               MlirStringRef name = mlirIdentifierStr(
                                   mlirOperationGetName(operation));
                               return py::str(name.data, name.length);
                             })
      .def_property_readonly(
          "context",
          [](PyOperation &self) {
            self.checkValid();
            return self.getContext().getObject();
          },
          "Context that owns the Operation")
      .def_property_readonly("opview", &PyOperation::createOpView);

  auto opViewClass =
      py::class_<PyOpView, PyOperationBase>(m, "OpView", py::module_local())
          .def(py::init<py::object>(), py::arg("operation"))
          .def_property_readonly("operation", &PyOpView::getOperationObject)
          .def_property_readonly(
              "context",
              [](PyOpView &self) {
                return self.getOperation().getContext().getObject();
              },
              "Context that owns the Operation")
          .def("__str__", [](PyOpView &self) {
            return py::str(self.getOperationObject());
          });
  opViewClass.attr("_ODS_REGIONS") = py::make_tuple(0, true);
  opViewClass.attr("_ODS_OPERAND_SEGMENTS") = py::none();
  opViewClass.attr("_ODS_RESULT_SEGMENTS") = py::none();
  opViewClass.attr("build_generic") = classmethod(
      &PyOpView::buildGeneric, py::arg("cls"), py::arg("results") = py::none(),
      py::arg("operands") = py::none(), py::arg("attributes") = py::none(),
      py::arg("successors") = py::none(), py::arg("regions") = py::none(),
      py::arg("loc") = py::none(), py::arg("ip") = py::none(),
      "Builds a specific, generated OpView based on class level attributes.");

  //----------------------------------------------------------------------------
  // Mapping of PyRegion.
  //----------------------------------------------------------------------------
  py::class_<PyRegion>(m, "Region", py::module_local())
      .def_property_readonly(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
      .def_property_readonly(
          "owner",
          [](PyRegion &self) {
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
      .def("__eq__",
           [](PyRegion &self, PyRegion &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyRegion &self, py::object &other) { return false; });

  //----------------------------------------------------------------------------
  // Mapping of PyBlock.
  //----------------------------------------------------------------------------
  py::class_<PyBlock>(m, "Block", py::module_local())
      .def_property_readonly(
          "owner",
          [](PyBlock &self) {
            return self.getParentOperation()->createOpView();
          },
          "Returns the owning operation of this block.")
      .def_property_readonly(
          "region",
          [](PyBlock &self) {
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            return PyRegion(self.getParentOperation(), region);
          },
          "Returns the owning region of this block.")
      .def_property_readonly(
          "arguments",
          [](PyBlock &self) {
            return PyBlockArgumentList(self.getParentOperation(), self.get());
          },
          "Returns a list of block arguments.")
      .def_property_readonly(
          "operations",
          [](PyBlock &self) {
            return PyOperationList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of operations.")
      .def_static(
          "create_at_start",
          [](PyRegion &parent, py::list pyArgTypes) {
            parent.checkValid();
            llvm::SmallVector<MlirType, 4> argTypes;
            llvm::SmallVector<MlirLocation, 4> argLocs;
            argTypes.reserve(pyArgTypes.size());
            argLocs.reserve(pyArgTypes.size());
            for (auto &pyArg : pyArgTypes) {
              argTypes.push_back(pyArg.cast<PyType &>());
              // TODO: Pass in a proper location here.
              argLocs.push_back(
                  mlirLocationUnknownGet(mlirTypeGetContext(argTypes.back())));
            }

            MlirBlock block = mlirBlockCreate(argTypes.size(), argTypes.data(),
                                              argLocs.data());
            mlirRegionInsertOwnedBlock(parent, 0, block);
            return PyBlock(parent.getParentOperation(), block);
          },
          py::arg("parent"), py::arg("arg_types") = py::list(),
          "Creates and returns a new Block at the beginning of the given "
          "region (with given argument types).")
      .def(
          "append_to",
          [](PyBlock &self, PyRegion &region) {
            MlirBlock b = self.get();
            if (!mlirRegionIsNull(mlirBlockGetParentRegion(b)))
              mlirBlockDetach(b);
            mlirRegionAppendOwnedBlock(region.get(), b);
          },
          "Append this block to a region, transferring ownership if necessary")
      .def(
          "create_before",
          [](PyBlock &self, py::args pyArgTypes) {
            self.checkValid();
            llvm::SmallVector<MlirType, 4> argTypes;
            llvm::SmallVector<MlirLocation, 4> argLocs;
            argTypes.reserve(pyArgTypes.size());
            argLocs.reserve(pyArgTypes.size());
            for (auto &pyArg : pyArgTypes) {
              argTypes.push_back(pyArg.cast<PyType &>());
              // TODO: Pass in a proper location here.
              argLocs.push_back(
                  mlirLocationUnknownGet(mlirTypeGetContext(argTypes.back())));
            }

            MlirBlock block = mlirBlockCreate(argTypes.size(), argTypes.data(),
                                              argLocs.data());
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockBefore(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          "Creates and returns a new Block before this block "
          "(with given argument types).")
      .def(
          "create_after",
          [](PyBlock &self, py::args pyArgTypes) {
            self.checkValid();
            llvm::SmallVector<MlirType, 4> argTypes;
            llvm::SmallVector<MlirLocation, 4> argLocs;
            argTypes.reserve(pyArgTypes.size());
            argLocs.reserve(pyArgTypes.size());
            for (auto &pyArg : pyArgTypes) {
              argTypes.push_back(pyArg.cast<PyType &>());

              // TODO: Pass in a proper location here.
              argLocs.push_back(
                  mlirLocationUnknownGet(mlirTypeGetContext(argTypes.back())));
            }
            MlirBlock block = mlirBlockCreate(argTypes.size(), argTypes.data(),
                                              argLocs.data());
            MlirRegion region = mlirBlockGetParentRegion(self.get());
            mlirRegionInsertOwnedBlockAfter(region, self.get(), block);
            return PyBlock(self.getParentOperation(), block);
          },
          "Creates and returns a new Block after this block "
          "(with given argument types).")
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
      .def("__eq__",
           [](PyBlock &self, PyBlock &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyBlock &self, py::object &other) { return false; })
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
          py::arg("operation"),
          "Appends an operation to this block. If the operation is currently "
          "in another block, it will be moved.");

  //----------------------------------------------------------------------------
  // Mapping of PyInsertionPoint.
  //----------------------------------------------------------------------------

  py::class_<PyInsertionPoint>(m, "InsertionPoint", py::module_local())
      .def(py::init<PyBlock &>(), py::arg("block"),
           "Inserts after the last operation but still inside the block.")
      .def("__enter__", &PyInsertionPoint::contextEnter)
      .def("__exit__", &PyInsertionPoint::contextExit)
      .def_property_readonly_static(
          "current",
          [](py::object & /*class*/) {
            auto *ip = PyThreadContextEntry::getDefaultInsertionPoint();
            if (!ip)
              throw SetPyError(PyExc_ValueError, "No current InsertionPoint");
            return ip;
          },
          "Gets the InsertionPoint bound to the current thread or raises "
          "ValueError if none has been set")
      .def(py::init<PyOperationBase &>(), py::arg("beforeOperation"),
           "Inserts before a referenced operation.")
      .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin,
                  py::arg("block"), "Inserts at the beginning of the block.")
      .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                  py::arg("block"), "Inserts before the block terminator.")
      .def("insert", &PyInsertionPoint::insert, py::arg("operation"),
           "Inserts an operation.")
      .def_property_readonly(
          "block", [](PyInsertionPoint &self) { return self.getBlock(); },
          "Returns the block that this InsertionPoint points to.");

  //----------------------------------------------------------------------------
  // Mapping of PyAttribute.
  //----------------------------------------------------------------------------
  py::class_<PyAttribute>(m, "Attribute", py::module_local())
      // Delegate to the PyAttribute copy constructor, which will also lifetime
      // extend the backing context which owns the MlirAttribute.
      .def(py::init<PyAttribute &>(), py::arg("cast_from_type"),
           "Casts the passed attribute to the generic Attribute")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyAttribute::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyAttribute::createFromCapsule)
      .def_static(
          "parse",
          [](std::string attrSpec, DefaultingPyMlirContext context) {
            MlirAttribute type = mlirAttributeParseGet(
                context->get(), toMlirStringRef(attrSpec));
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirAttributeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               Twine("Unable to parse attribute: '") +
                                   attrSpec + "'");
            }
            return PyAttribute(context->getRef(), type);
          },
          py::arg("asm"), py::arg("context") = py::none(),
          "Parses an attribute from an assembly form")
      .def_property_readonly(
          "context",
          [](PyAttribute &self) { return self.getContext().getObject(); },
          "Context that owns the Attribute")
      .def_property_readonly("type",
                             [](PyAttribute &self) {
                               return PyType(self.getContext()->getRef(),
                                             mlirAttributeGetType(self));
                             })
      .def(
          "get_named",
          [](PyAttribute &self, std::string name) {
            return PyNamedAttribute(self, std::move(name));
          },
          py::keep_alive<0, 1>(), "Binds a name to the attribute")
      .def("__eq__",
           [](PyAttribute &self, PyAttribute &other) { return self == other; })
      .def("__eq__", [](PyAttribute &self, py::object &other) { return false; })
      .def("__hash__",
           [](PyAttribute &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
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
      .def("__repr__", [](PyAttribute &self) {
        // Generally, assembly formats are not printed for __repr__ because
        // this can cause exceptionally long debug output and exceptions.
        // However, attribute values are generally considered useful and are
        // printed. This may need to be re-evaluated if debug dumps end up
        // being excessive.
        PyPrintAccumulator printAccum;
        printAccum.parts.append("Attribute(");
        mlirAttributePrint(self, printAccum.getCallback(),
                           printAccum.getUserData());
        printAccum.parts.append(")");
        return printAccum.join();
      });

  //----------------------------------------------------------------------------
  // Mapping of PyNamedAttribute
  //----------------------------------------------------------------------------
  py::class_<PyNamedAttribute>(m, "NamedAttribute", py::module_local())
      .def("__repr__",
           [](PyNamedAttribute &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("NamedAttribute(");
             printAccum.parts.append(
                 py::str(mlirIdentifierStr(self.namedAttr.name).data,
                         mlirIdentifierStr(self.namedAttr.name).length));
             printAccum.parts.append("=");
             mlirAttributePrint(self.namedAttr.attribute,
                                printAccum.getCallback(),
                                printAccum.getUserData());
             printAccum.parts.append(")");
             return printAccum.join();
           })
      .def_property_readonly(
          "name",
          [](PyNamedAttribute &self) {
            return py::str(mlirIdentifierStr(self.namedAttr.name).data,
                           mlirIdentifierStr(self.namedAttr.name).length);
          },
          "The name of the NamedAttribute binding")
      .def_property_readonly(
          "attr",
          [](PyNamedAttribute &self) {
            // TODO: When named attribute is removed/refactored, also remove
            // this constructor (it does an inefficient table lookup).
            auto contextRef = PyMlirContext::forContext(
                mlirAttributeGetContext(self.namedAttr.attribute));
            return PyAttribute(std::move(contextRef), self.namedAttr.attribute);
          },
          py::keep_alive<0, 1>(),
          "The underlying generic attribute of the NamedAttribute binding");

  //----------------------------------------------------------------------------
  // Mapping of PyType.
  //----------------------------------------------------------------------------
  py::class_<PyType>(m, "Type", py::module_local())
      // Delegate to the PyType copy constructor, which will also lifetime
      // extend the backing context which owns the MlirType.
      .def(py::init<PyType &>(), py::arg("cast_from_type"),
           "Casts the passed type to the generic Type")
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR, &PyType::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyType::createFromCapsule)
      .def_static(
          "parse",
          [](std::string typeSpec, DefaultingPyMlirContext context) {
            MlirType type =
                mlirTypeParseGet(context->get(), toMlirStringRef(typeSpec));
            // TODO: Rework error reporting once diagnostic engine is exposed
            // in C API.
            if (mlirTypeIsNull(type)) {
              throw SetPyError(PyExc_ValueError,
                               Twine("Unable to parse type: '") + typeSpec +
                                   "'");
            }
            return PyType(context->getRef(), type);
          },
          py::arg("asm"), py::arg("context") = py::none(),
          kContextParseTypeDocstring)
      .def_property_readonly(
          "context", [](PyType &self) { return self.getContext().getObject(); },
          "Context that owns the Type")
      .def("__eq__", [](PyType &self, PyType &other) { return self == other; })
      .def("__eq__", [](PyType &self, py::object &other) { return false; })
      .def("__hash__",
           [](PyType &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
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
          "Returns the assembly form of the type.")
      .def("__repr__", [](PyType &self) {
        // Generally, assembly formats are not printed for __repr__ because
        // this can cause exceptionally long debug output and exceptions.
        // However, types are an exception as they typically have compact
        // assembly forms and printing them is useful.
        PyPrintAccumulator printAccum;
        printAccum.parts.append("Type(");
        mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());
        printAccum.parts.append(")");
        return printAccum.join();
      });

  //----------------------------------------------------------------------------
  // Mapping of Value.
  //----------------------------------------------------------------------------
  py::class_<PyValue>(m, "Value", py::module_local())
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR, &PyValue::getCapsule)
      .def(MLIR_PYTHON_CAPI_FACTORY_ATTR, &PyValue::createFromCapsule)
      .def_property_readonly(
          "context",
          [](PyValue &self) { return self.getParentOperation()->getContext(); },
          "Context in which the value lives.")
      .def(
          "dump", [](PyValue &self) { mlirValueDump(self.get()); },
          kDumpDocstring)
      .def_property_readonly(
          "owner",
          [](PyValue &self) {
            assert(mlirOperationEqual(self.getParentOperation()->get(),
                                      mlirOpResultGetOwner(self.get())) &&
                   "expected the owner of the value in Python to match that in "
                   "the IR");
            return self.getParentOperation().getObject();
          })
      .def("__eq__",
           [](PyValue &self, PyValue &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyValue &self, py::object other) { return false; })
      .def("__hash__",
           [](PyValue &self) {
             return static_cast<size_t>(llvm::hash_value(self.get().ptr));
           })
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
          kValueDunderStrDocstring)
      .def_property_readonly("type", [](PyValue &self) {
        return PyType(self.getParentOperation()->getContext(),
                      mlirValueGetType(self.get()));
      });
  PyBlockArgument::bind(m);
  PyOpResult::bind(m);

  //----------------------------------------------------------------------------
  // Mapping of SymbolTable.
  //----------------------------------------------------------------------------
  py::class_<PySymbolTable>(m, "SymbolTable", py::module_local())
      .def(py::init<PyOperationBase &>())
      .def("__getitem__", &PySymbolTable::dunderGetItem)
      .def("insert", &PySymbolTable::insert, py::arg("operation"))
      .def("erase", &PySymbolTable::erase, py::arg("operation"))
      .def("__delitem__", &PySymbolTable::dunderDel)
      .def("__contains__",
           [](PySymbolTable &table, const std::string &name) {
             return !mlirOperationIsNull(mlirSymbolTableLookup(
                 table, mlirStringRefCreate(name.data(), name.length())));
           })
      // Static helpers.
      .def_static("set_symbol_name", &PySymbolTable::setSymbolName,
                  py::arg("symbol"), py::arg("name"))
      .def_static("get_symbol_name", &PySymbolTable::getSymbolName,
                  py::arg("symbol"))
      .def_static("get_visibility", &PySymbolTable::getVisibility,
                  py::arg("symbol"))
      .def_static("set_visibility", &PySymbolTable::setVisibility,
                  py::arg("symbol"), py::arg("visibility"))
      .def_static("replace_all_symbol_uses",
                  &PySymbolTable::replaceAllSymbolUses, py::arg("old_symbol"),
                  py::arg("new_symbol"), py::arg("from_op"))
      .def_static("walk_symbol_tables", &PySymbolTable::walkSymbolTables,
                  py::arg("from_op"), py::arg("all_sym_uses_visible"),
                  py::arg("callback"));

  // Container bindings.
  PyBlockArgumentList::bind(m);
  PyBlockIterator::bind(m);
  PyBlockList::bind(m);
  PyOperationIterator::bind(m);
  PyOperationList::bind(m);
  PyOpAttributeMap::bind(m);
  PyOpOperandList::bind(m);
  PyOpResultList::bind(m);
  PyRegionIterator::bind(m);
  PyRegionList::bind(m);

  // Debug bindings.
  PyGlobalDebugFlag::bind(m);
}
