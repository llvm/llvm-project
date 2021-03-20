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
#include "mlir-c/Registration.h"
#include "llvm/ADT/SmallVector.h"
#include <pybind11/stl.h>

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

static const char kContextGetFileLocationDocstring[] =
    R"(Gets a Location representing a file, line and column)";

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

// Helper for creating an @classmethod.
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
    py::class_<PyRegionIterator>(m, "RegionIterator")
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
    py::class_<PyRegionList>(m, "RegionSequence")
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
    py::class_<PyBlockIterator>(m, "BlockIterator")
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

  PyBlock appendBlock(py::args pyArgTypes) {
    operation->checkValid();
    llvm::SmallVector<MlirType, 4> argTypes;
    argTypes.reserve(pyArgTypes.size());
    for (auto &pyArg : pyArgTypes) {
      argTypes.push_back(pyArg.cast<PyType &>());
    }

    MlirBlock block = mlirBlockCreate(argTypes.size(), argTypes.data());
    mlirRegionAppendOwnedBlock(region, block);
    return PyBlock(operation, block);
  }

  static void bind(py::module &m) {
    py::class_<PyBlockList>(m, "BlockList")
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
    py::class_<PyOperationIterator>(m, "OperationIterator")
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
    py::class_<PyOperationList>(m, "OperationList")
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
  mlirRegisterAllDialects(context);
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

size_t PyMlirContext::getLiveModuleCount() { return liveModules.size(); }

pybind11::object PyMlirContext::contextEnter() {
  return PyThreadContextEntry::pushContext(*this);
}

void PyMlirContext::contextExit(pybind11::object excType,
                                pybind11::object excVal,
                                pybind11::object excTb) {
  PyThreadContextEntry::popContext(*this);
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
// PyDialect, PyDialectDescriptor, PyDialects
//------------------------------------------------------------------------------

MlirDialect PyDialects::getDialectForKey(const std::string &key,
                                         bool attrError) {
  // If the "std" dialect was asked for, substitute the empty namespace :(
  static const std::string emptyKey;
  const std::string *canonKey = key == "std" ? &emptyKey : &key;
  MlirDialect dialect = mlirContextGetOrLoadDialect(
      getContext()->get(), {canonKey->data(), canonKey->size()});
  if (mlirDialectIsNull(dialect)) {
    throw SetPyError(attrError ? PyExc_AttributeError : PyExc_IndexError,
                     Twine("Dialect '") + key + "' not found");
  }
  return dialect;
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

void PyLocation::contextExit(py::object excType, py::object excVal,
                             py::object excTb) {
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
                            bool printGenericOpForm, bool useLocalScope) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = py::module::import("sys").attr("stdout");

  if (!printGenericOpForm && !mlirOperationVerify(operation)) {
    fileObject.attr("write")("// Verification failed, printing generic form\n");
    printGenericOpForm = true;
  }

  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  if (largeElementsLimit)
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
  if (enableDebugInfo)
    mlirOpPrintingFlagsEnableDebugInfo(flags, /*prettyForm=*/prettyDebugInfo);
  if (printGenericOpForm)
    mlirOpPrintingFlagsPrintGenericOpForm(flags);

  PyFileAccumulator accum(fileObject, binary);
  py::gil_scoped_release();
  mlirOperationPrintWithFlags(operation, flags, accum.getCallback(),
                              accum.getUserData());
  mlirOpPrintingFlagsDestroy(flags);
}

py::object PyOperationBase::getAsm(bool binary,
                                   llvm::Optional<int64_t> largeElementsLimit,
                                   bool enableDebugInfo, bool prettyDebugInfo,
                                   bool printGenericOpForm,
                                   bool useLocalScope) {
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
        /*useLocalScope=*/useLocalScope);

  return fileObject.attr("getvalue")();
}

PyOperationRef PyOperation::getParentOperation() {
  if (!isAttached())
    throw SetPyError(PyExc_ValueError, "Detached operations have no parent");
  MlirOperation operation = mlirOperationGetParentOperation(get());
  if (mlirOperationIsNull(operation))
    throw SetPyError(PyExc_ValueError, "Operation has no parent.");
  return PyOperation::forOperation(getContext(), operation);
}

PyBlock PyOperation::getBlock() {
  PyOperationRef parentOperation = getParentOperation();
  MlirBlock block = mlirOperationGetBlock(get());
  assert(!mlirBlockIsNull(block) && "Attached operation has null parent");
  return PyBlock{std::move(parentOperation), block};
}

py::object PyOperation::create(
    std::string name, llvm::Optional<std::vector<PyType *>> results,
    llvm::Optional<std::vector<PyValue *>> operands,
    llvm::Optional<py::dict> attributes,
    llvm::Optional<std::vector<PyBlock *>> successors, int regions,
    DefaultingPyLocation location, py::object maybeIp) {
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
    llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
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

  // InsertPoint active?
  if (!maybeIp.is(py::cast(false))) {
    PyInsertionPoint *ip;
    if (maybeIp.is_none()) {
      ip = PyThreadContextEntry::getDefaultInsertionPoint();
    } else {
      ip = py::cast<PyInsertionPoint *>(maybeIp);
    }
    if (ip)
      ip->insert(*created.get());
  }

  return created->createOpView();
}

py::object PyOperation::createOpView() {
  MlirIdentifier ident = mlirOperationGetName(get());
  MlirStringRef identStr = mlirIdentifierStr(ident);
  auto opViewClass = PyGlobals::get().lookupRawOpViewClass(
      StringRef(identStr.data, identStr.length));
  if (opViewClass)
    return (*opViewClass)(getRef().getObject());
  return py::cast(PyOpView(getRef().getObject()));
}

//------------------------------------------------------------------------------
// PyOpView
//------------------------------------------------------------------------------

py::object
PyOpView::buildGeneric(py::object cls, py::list resultTypeList,
                       py::list operandList,
                       llvm::Optional<py::dict> attributes,
                       llvm::Optional<std::vector<PyBlock *>> successors,
                       llvm::Optional<int> regions,
                       DefaultingPyLocation location, py::object maybeIp) {
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
    for (auto it : llvm::enumerate(resultTypeList)) {
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
                             "result segments but was provided " +
                             llvm::Twine(resultTypeList.size()))
                                .str());
    }
    resultSegmentLengths.reserve(resultTypeList.size());
    for (auto it :
         llvm::enumerate(llvm::zip(resultTypeList, resultSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto resultType = py::cast<PyType *>(std::get<0>(it.value()));
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
    for (auto it : llvm::enumerate(operandList)) {
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
    for (auto it :
         llvm::enumerate(llvm::zip(operandList, operandSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto operandValue = py::cast<PyValue *>(std::get<0>(it.value()));
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
  return PyOperation::create(std::move(name),
                             /*results=*/std::move(resultTypes),
                             /*operands=*/std::move(operands),
                             /*attributes=*/std::move(attributes),
                             /*successors=*/std::move(successors),
                             /*regions=*/*regions, location, maybeIp);
}

PyOpView::PyOpView(py::object operationObject)
    // Casting through the PyOperationBase base-class and then back to the
    // Operation lets us accept any PyOperationBase subclass.
    : operation(py::cast<PyOperationBase &>(operationObject).getOperation()),
      operationObject(operation.getRef().getObject()) {}

py::object PyOpView::createRawSubclass(py::object userClass) {
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

void PyInsertionPoint::contextExit(pybind11::object excType,
                                   pybind11::object excVal,
                                   pybind11::object excTb) {
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
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyValue &>(), py::keep_alive<0, 1>());
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
    c.def("set_type", [](PyBlockArgument &self, PyType type) {
      return mlirBlockArgumentSetType(self.get(), type);
    });
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
      return self.getParentOperation();
    });
    c.def_property_readonly("result_number", [](PyOpResult &self) {
      return mlirOpResultGetResultNumber(self.get());
    });
  }
};

/// A list of block arguments. Internally, these are stored as consecutive
/// elements, random access is cheap. The argument list is associated with the
/// operation that contains the block (detached blocks are not allowed in
/// Python bindings) and extends its lifetime.
class PyBlockArgumentList {
public:
  PyBlockArgumentList(PyOperationRef operation, MlirBlock block)
      : operation(std::move(operation)), block(block) {}

  /// Returns the length of the block argument list.
  intptr_t dunderLen() {
    operation->checkValid();
    return mlirBlockGetNumArguments(block);
  }

  /// Returns `index`-th element of the block argument list.
  PyBlockArgument dunderGetItem(intptr_t index) {
    if (index < 0 || index >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds region");
    }
    PyValue value(operation, mlirBlockGetArgument(block, index));
    return PyBlockArgument(value);
  }

  /// Defines a Python class in the bindings.
  static void bind(py::module &m) {
    py::class_<PyBlockArgumentList>(m, "BlockArgumentList")
        .def("__len__", &PyBlockArgumentList::dunderLen)
        .def("__getitem__", &PyBlockArgumentList::dunderGetItem);
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
    return PyValue(operation, mlirOperationGetOperand(operation->get(), pos));
  }

  PyOpOperandList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpOperandList(operation, startIndex, length, step);
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

private:
  PyOperationRef operation;
};

/// A list of operation attributes. Can be indexed by name, producing
/// attributes, or by index, producing named attributes.
class PyOpAttributeMap {
public:
  PyOpAttributeMap(PyOperationRef operation) : operation(operation) {}

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
        std::string(mlirIdentifierStr(namedAttr.name).data));
  }

  void dunderSetItem(const std::string &name, PyAttribute attr) {
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
    py::class_<PyOpAttributeMap>(m, "OpAttributeMap")
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

} // end namespace

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRCore(py::module &m) {
  //----------------------------------------------------------------------------
  // Mapping of MlirContext
  //----------------------------------------------------------------------------
  py::class_<PyMlirContext>(m, "Context")
      .def(py::init<>(&PyMlirContext::createNewContextForInit))
      .def_static("_get_live_count", &PyMlirContext::getLiveCount)
      .def("_get_context_again",
           [](PyMlirContext &self) {
             PyMlirContextRef ref = PyMlirContext::forContext(self.get());
             return ref.releaseObject();
           })
      .def("_get_live_operation_count", &PyMlirContext::getLiveOperationCount)
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
          "Gets or loads a dialect by name, returning its descriptor object")
      .def_property(
          "allow_unregistered_dialects",
          [](PyMlirContext &self) -> bool {
            return mlirContextGetAllowUnregisteredDialects(self.get());
          },
          [](PyMlirContext &self, bool value) {
            mlirContextSetAllowUnregisteredDialects(self.get(), value);
          });

  //----------------------------------------------------------------------------
  // Mapping of PyDialectDescriptor
  //----------------------------------------------------------------------------
  py::class_<PyDialectDescriptor>(m, "DialectDescriptor")
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
  py::class_<PyDialects>(m, "Dialects")
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
  py::class_<PyDialect>(m, "Dialect")
      .def(py::init<py::object>(), "descriptor")
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
  // Mapping of Location
  //----------------------------------------------------------------------------
  py::class_<PyLocation>(m, "Location")
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
      .def_property_readonly(
          "context",
          [](PyLocation &self) { return self.getContext().getObject(); },
          "Context that owns the Location")
      .def("__repr__", [](PyLocation &self) {
        PyPrintAccumulator printAccum;
        mlirLocationPrint(self, printAccum.getCallback(),
                          printAccum.getUserData());
        return printAccum.join();
      });

  //----------------------------------------------------------------------------
  // Mapping of Module
  //----------------------------------------------------------------------------
  py::class_<PyModule>(m, "Module")
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
            PyOperationRef module_op = PyOperation::forOperation(
                self.getContext(), mlirModuleGetOperation(self.get()),
                self.getRef().releaseObject());
            PyBlock returnBlock(module_op, mlirModuleGetBody(self.get()));
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
          [](PyModule &self) {
            MlirOperation operation = mlirModuleGetOperation(self.get());
            PyPrintAccumulator printAccum;
            mlirOperationPrint(operation, printAccum.getCallback(),
                               printAccum.getUserData());
            return printAccum.join();
          },
          kOperationStrDunderDocstring);

  //----------------------------------------------------------------------------
  // Mapping of Operation.
  //----------------------------------------------------------------------------
  py::class_<PyOperationBase>(m, "_OperationBase")
      .def("__eq__",
           [](PyOperationBase &self, PyOperationBase &other) {
             return &self.getOperation() == &other.getOperation();
           })
      .def("__eq__",
           [](PyOperationBase &self, py::object other) { return false; })
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
      .def("__iter__",
           [](PyOperationBase &self) {
             return PyRegionIterator(self.getOperation().getRef());
           })
      .def(
          "__str__",
          [](PyOperationBase &self) {
            return self.getAsm(/*binary=*/false,
                               /*largeElementsLimit=*/llvm::None,
                               /*enableDebugInfo=*/false,
                               /*prettyDebugInfo=*/false,
                               /*printGenericOpForm=*/false,
                               /*useLocalScope=*/false);
          },
          "Returns the assembly form of the operation.")
      .def("print", &PyOperationBase::print,
           // Careful: Lots of arguments must match up with print method.
           py::arg("file") = py::none(), py::arg("binary") = false,
           py::arg("large_elements_limit") = py::none(),
           py::arg("enable_debug_info") = false,
           py::arg("pretty_debug_info") = false,
           py::arg("print_generic_op_form") = false,
           py::arg("use_local_scope") = false, kOperationPrintDocstring)
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           py::arg("binary") = false,
           py::arg("large_elements_limit") = py::none(),
           py::arg("enable_debug_info") = false,
           py::arg("pretty_debug_info") = false,
           py::arg("print_generic_op_form") = false,
           py::arg("use_local_scope") = false, kOperationGetAsmDocstring)
      .def(
          "verify",
          [](PyOperationBase &self) {
            return mlirOperationVerify(self.getOperation());
          },
          "Verify the operation and return true if it passes, false if it "
          "fails.");

  py::class_<PyOperation, PyOperationBase>(m, "Operation")
      .def_static("create", &PyOperation::create, py::arg("name"),
                  py::arg("results") = py::none(),
                  py::arg("operands") = py::none(),
                  py::arg("attributes") = py::none(),
                  py::arg("successors") = py::none(), py::arg("regions") = 0,
                  py::arg("loc") = py::none(), py::arg("ip") = py::none(),
                  kOperationCreateDocstring)
      .def_property_readonly("name",
                             [](PyOperation &self) {
                               MlirOperation operation = self.get();
                               MlirStringRef name = mlirIdentifierStr(
                                   mlirOperationGetName(operation));
                               return py::str(name.data, name.length);
                             })
      .def_property_readonly(
          "context",
          [](PyOperation &self) { return self.getContext().getObject(); },
          "Context that owns the Operation")
      .def_property_readonly("opview", &PyOperation::createOpView);

  auto opViewClass =
      py::class_<PyOpView, PyOperationBase>(m, "OpView")
          .def(py::init<py::object>())
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
  py::class_<PyRegion>(m, "Region")
      .def_property_readonly(
          "blocks",
          [](PyRegion &self) {
            return PyBlockList(self.getParentOperation(), self.get());
          },
          "Returns a forward-optimized sequence of blocks.")
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
  py::class_<PyBlock>(m, "Block")
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
          "Returns the assembly form of the block.");

  //----------------------------------------------------------------------------
  // Mapping of PyInsertionPoint.
  //----------------------------------------------------------------------------

  py::class_<PyInsertionPoint>(m, "InsertionPoint")
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
           "Inserts an operation.");

  //----------------------------------------------------------------------------
  // Mapping of PyAttribute.
  //----------------------------------------------------------------------------
  py::class_<PyAttribute>(m, "Attribute")
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
  py::class_<PyNamedAttribute>(m, "NamedAttribute")
      .def("__repr__",
           [](PyNamedAttribute &self) {
             PyPrintAccumulator printAccum;
             printAccum.parts.append("NamedAttribute(");
             printAccum.parts.append(
                 mlirIdentifierStr(self.namedAttr.name).data);
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
  py::class_<PyType>(m, "Type")
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
  py::class_<PyValue>(m, "Value")
      .def_property_readonly(
          "context",
          [](PyValue &self) { return self.getParentOperation()->getContext(); },
          "Context in which the value lives.")
      .def(
          "dump", [](PyValue &self) { mlirValueDump(self.get()); },
          kDumpDocstring)
      .def("__eq__",
           [](PyValue &self, PyValue &other) {
             return self.get().ptr == other.get().ptr;
           })
      .def("__eq__", [](PyValue &self, py::object other) { return false; })
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
}
