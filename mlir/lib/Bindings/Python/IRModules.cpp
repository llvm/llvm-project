//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRModules.h"

#include "Globals.h"
#include "PybindUtils.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardTypes.h"
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

/// Checks whether the given type is an integer or float type.
static int mlirTypeIsAIntegerOrFloat(MlirType type) {
  return mlirTypeIsAInteger(type) || mlirTypeIsABF16(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type);
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
    py::class_<PyRegionList>(m, "ReqionSequence")
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
    std::string name, llvm::Optional<std::vector<PyValue *>> operands,
    llvm::Optional<std::vector<PyType *>> results,
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

      auto name = it.first.cast<std::string>();
      auto &attribute = it.second.cast<PyAttribute &>();
      // TODO: Verify attribute originates from the same context.
      mlirAttributes.emplace_back(std::move(name), attribute);
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
      mlirNamedAttributes.push_back(
          mlirNamedAttributeGet(toMlirStringRef(it.first), it.second));
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
  namedAttr = mlirNamedAttributeGet(toMlirStringRef(*this->ownedName), attr);
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
    return PyNamedAttribute(namedAttr.attribute,
                            std::string(namedAttr.name.data));
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
// Standard attribute subclasses.
//------------------------------------------------------------------------------

namespace {

/// CRTP base classes for Python attributes that subclass Attribute and should
/// be castable from it (i.e. via something like StringAttr(attr)).
/// By default, attribute class hierarchies are one level deep (i.e. a
/// concrete attribute class extends PyAttribute); however, intermediate
/// python-visible base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyAttribute>
class PyConcreteAttribute : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirAttribute);

  PyConcreteAttribute() = default;
  PyConcreteAttribute(PyMlirContextRef contextRef, MlirAttribute attr)
      : BaseTy(std::move(contextRef), attr) {}
  PyConcreteAttribute(PyAttribute &orig)
      : PyConcreteAttribute(orig.getContext(), castFrom(orig)) {}

  static MlirAttribute castFrom(PyAttribute &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, Twine("Cannot cast attribute to ") +
                                             DerivedTy::pyClassName +
                                             " (from " + origRepr + ")");
    }
    return orig;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName, py::buffer_protocol());
    cls.def(py::init<PyAttribute &>(), py::keep_alive<0, 1>());
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Float Point Attribute subclass - FloatAttr.
class PyFloatAttribute : public PyConcreteAttribute<PyFloatAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAFloat;
  static constexpr const char *pyClassName = "FloatAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, double value, DefaultingPyLocation loc) {
          MlirAttribute attr = mlirFloatAttrDoubleGetChecked(type, value, loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirAttributeIsNull(attr)) {
            throw SetPyError(PyExc_ValueError,
                             Twine("invalid '") +
                                 py::repr(py::cast(type)).cast<std::string>() +
                                 "' and expected floating point type.");
          }
          return PyFloatAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"), py::arg("loc") = py::none(),
        "Gets an uniqued float point attribute associated to a type");
    c.def_static(
        "get_f32",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF32TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f32 type");
    c.def_static(
        "get_f64",
        [](double value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirFloatAttrDoubleGet(
              context->get(), mlirF64TypeGet(context->get()), value);
          return PyFloatAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued float point attribute associated to a f64 type");
    c.def_property_readonly(
        "value",
        [](PyFloatAttribute &self) {
          return mlirFloatAttrGetValueDouble(self);
        },
        "Returns the value of the float point attribute");
  }
};

/// Integer Attribute subclass - IntegerAttr.
class PyIntegerAttribute : public PyConcreteAttribute<PyIntegerAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAInteger;
  static constexpr const char *pyClassName = "IntegerAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &type, int64_t value) {
          MlirAttribute attr = mlirIntegerAttrGet(type, value);
          return PyIntegerAttribute(type.getContext(), attr);
        },
        py::arg("type"), py::arg("value"),
        "Gets an uniqued integer attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyIntegerAttribute &self) {
          return mlirIntegerAttrGetValueInt(self);
        },
        "Returns the value of the integer attribute");
  }
};

/// Bool Attribute subclass - BoolAttr.
class PyBoolAttribute : public PyConcreteAttribute<PyBoolAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsABool;
  static constexpr const char *pyClassName = "BoolAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](bool value, DefaultingPyMlirContext context) {
          MlirAttribute attr = mlirBoolAttrGet(context->get(), value);
          return PyBoolAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets an uniqued bool attribute");
    c.def_property_readonly(
        "value",
        [](PyBoolAttribute &self) { return mlirBoolAttrGetValue(self); },
        "Returns the value of the bool attribute");
  }
};

class PyStringAttribute : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::string value, DefaultingPyMlirContext context) {
          MlirAttribute attr =
              mlirStringAttrGet(context->get(), value.size(), &value[0]);
          return PyStringAttribute(context->getRef(), attr);
        },
        py::arg("value"), py::arg("context") = py::none(),
        "Gets a uniqued string attribute");
    c.def_static(
        "get_typed",
        [](PyType &type, std::string value) {
          MlirAttribute attr =
              mlirStringAttrTypedGet(type, value.size(), &value[0]);
          return PyStringAttribute(type.getContext(), attr);
        },

        "Gets a uniqued string attribute associated to a type");
    c.def_property_readonly(
        "value",
        [](PyStringAttribute &self) {
          MlirStringRef stringRef = mlirStringAttrGetValue(self);
          return py::str(stringRef.data, stringRef.length);
        },
        "Returns the value of the string attribute");
  }
};

// TODO: Support construction of bool elements.
// TODO: Support construction of string elements.
class PyDenseElementsAttribute
    : public PyConcreteAttribute<PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseElements;
  static constexpr const char *pyClassName = "DenseElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static PyDenseElementsAttribute
  getFromBuffer(py::buffer array, bool signless,
                DefaultingPyMlirContext contextWrapper) {
    // Request a contiguous view. In exotic cases, this will cause a copy.
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    Py_buffer *view = new Py_buffer();
    if (PyObject_GetBuffer(array.ptr(), view, flags) != 0) {
      delete view;
      throw py::error_already_set();
    }
    py::buffer_info arrayInfo(view);

    MlirContext context = contextWrapper->get();
    // Switch on the types that can be bulk loaded between the Python and
    // MLIR-C APIs.
    // See: https://docs.python.org/3/library/struct.html#format-characters
    if (arrayInfo.format == "f") {
      // f32
      assert(arrayInfo.itemsize == 4 && "mismatched array itemsize");
      return PyDenseElementsAttribute(
          contextWrapper->getRef(),
          bulkLoad(context, mlirDenseElementsAttrFloatGet,
                   mlirF32TypeGet(context), arrayInfo));
    } else if (arrayInfo.format == "d") {
      // f64
      assert(arrayInfo.itemsize == 8 && "mismatched array itemsize");
      return PyDenseElementsAttribute(
          contextWrapper->getRef(),
          bulkLoad(context, mlirDenseElementsAttrDoubleGet,
                   mlirF64TypeGet(context), arrayInfo));
    } else if (isSignedIntegerFormat(arrayInfo.format)) {
      if (arrayInfo.itemsize == 4) {
        // i32
        MlirType elementType = signless ? mlirIntegerTypeGet(context, 32)
                                        : mlirIntegerTypeSignedGet(context, 32);
        return PyDenseElementsAttribute(contextWrapper->getRef(),
                                        bulkLoad(context,
                                                 mlirDenseElementsAttrInt32Get,
                                                 elementType, arrayInfo));
      } else if (arrayInfo.itemsize == 8) {
        // i64
        MlirType elementType = signless ? mlirIntegerTypeGet(context, 64)
                                        : mlirIntegerTypeSignedGet(context, 64);
        return PyDenseElementsAttribute(contextWrapper->getRef(),
                                        bulkLoad(context,
                                                 mlirDenseElementsAttrInt64Get,
                                                 elementType, arrayInfo));
      }
    } else if (isUnsignedIntegerFormat(arrayInfo.format)) {
      if (arrayInfo.itemsize == 4) {
        // unsigned i32
        MlirType elementType = signless
                                   ? mlirIntegerTypeGet(context, 32)
                                   : mlirIntegerTypeUnsignedGet(context, 32);
        return PyDenseElementsAttribute(contextWrapper->getRef(),
                                        bulkLoad(context,
                                                 mlirDenseElementsAttrUInt32Get,
                                                 elementType, arrayInfo));
      } else if (arrayInfo.itemsize == 8) {
        // unsigned i64
        MlirType elementType = signless
                                   ? mlirIntegerTypeGet(context, 64)
                                   : mlirIntegerTypeUnsignedGet(context, 64);
        return PyDenseElementsAttribute(contextWrapper->getRef(),
                                        bulkLoad(context,
                                                 mlirDenseElementsAttrUInt64Get,
                                                 elementType, arrayInfo));
      }
    }

    // TODO: Fall back to string-based get.
    std::string message = "unimplemented array format conversion from format: ";
    message.append(arrayInfo.format);
    throw SetPyError(PyExc_ValueError, message);
  }

  static PyDenseElementsAttribute getSplat(PyType shapedType,
                                           PyAttribute &elementAttr) {
    auto contextWrapper =
        PyMlirContext::forContext(mlirTypeGetContext(shapedType));
    if (!mlirAttributeIsAInteger(elementAttr) &&
        !mlirAttributeIsAFloat(elementAttr)) {
      std::string message = "Illegal element type for DenseElementsAttr: ";
      message.append(py::repr(py::cast(elementAttr)));
      throw SetPyError(PyExc_ValueError, message);
    }
    if (!mlirTypeIsAShaped(shapedType) ||
        !mlirShapedTypeHasStaticShape(shapedType)) {
      std::string message =
          "Expected a static ShapedType for the shaped_type parameter: ";
      message.append(py::repr(py::cast(shapedType)));
      throw SetPyError(PyExc_ValueError, message);
    }
    MlirType shapedElementType = mlirShapedTypeGetElementType(shapedType);
    MlirType attrType = mlirAttributeGetType(elementAttr);
    if (!mlirTypeEqual(shapedElementType, attrType)) {
      std::string message =
          "Shaped element type and attribute type must be equal: shaped=";
      message.append(py::repr(py::cast(shapedType)));
      message.append(", element=");
      message.append(py::repr(py::cast(elementAttr)));
      throw SetPyError(PyExc_ValueError, message);
    }

    MlirAttribute elements =
        mlirDenseElementsAttrSplatGet(shapedType, elementAttr);
    return PyDenseElementsAttribute(contextWrapper->getRef(), elements);
  }

  intptr_t dunderLen() { return mlirElementsAttrGetNumElements(*this); }

  py::buffer_info accessBuffer() {
    MlirType shapedType = mlirAttributeGetType(*this);
    MlirType elementType = mlirShapedTypeGetElementType(shapedType);

    if (mlirTypeIsAF32(elementType)) {
      // f32
      return bufferInfo(shapedType, mlirDenseElementsAttrGetFloatValue);
    } else if (mlirTypeIsAF64(elementType)) {
      // f64
      return bufferInfo(shapedType, mlirDenseElementsAttrGetDoubleValue);
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 32) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i32
        return bufferInfo(shapedType, mlirDenseElementsAttrGetInt32Value);
      } else if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i32
        return bufferInfo(shapedType, mlirDenseElementsAttrGetUInt32Value);
      }
    } else if (mlirTypeIsAInteger(elementType) &&
               mlirIntegerTypeGetWidth(elementType) == 64) {
      if (mlirIntegerTypeIsSignless(elementType) ||
          mlirIntegerTypeIsSigned(elementType)) {
        // i64
        return bufferInfo(shapedType, mlirDenseElementsAttrGetInt64Value);
      } else if (mlirIntegerTypeIsUnsigned(elementType)) {
        // unsigned i64
        return bufferInfo(shapedType, mlirDenseElementsAttrGetUInt64Value);
      }
    }

    std::string message = "unimplemented array format.";
    throw SetPyError(PyExc_ValueError, message);
  }

  static void bindDerived(ClassTy &c) {
    c.def("__len__", &PyDenseElementsAttribute::dunderLen)
        .def_static("get", PyDenseElementsAttribute::getFromBuffer,
                    py::arg("array"), py::arg("signless") = true,
                    py::arg("context") = py::none(),
                    "Gets from a buffer or ndarray")
        .def_static("get_splat", PyDenseElementsAttribute::getSplat,
                    py::arg("shaped_type"), py::arg("element_attr"),
                    "Gets a DenseElementsAttr where all values are the same")
        .def_property_readonly("is_splat",
                               [](PyDenseElementsAttribute &self) -> bool {
                                 return mlirDenseElementsAttrIsSplat(self);
                               })
        .def_buffer(&PyDenseElementsAttribute::accessBuffer);
  }

private:
  template <typename ElementTy>
  static MlirAttribute
  bulkLoad(MlirContext context,
           MlirAttribute (*ctor)(MlirType, intptr_t, ElementTy *),
           MlirType mlirElementType, py::buffer_info &arrayInfo) {
    SmallVector<int64_t, 4> shape(arrayInfo.shape.begin(),
                                  arrayInfo.shape.begin() + arrayInfo.ndim);
    auto shapedType =
        mlirRankedTensorTypeGet(shape.size(), shape.data(), mlirElementType);
    intptr_t numElements = arrayInfo.size;
    const ElementTy *contents = static_cast<const ElementTy *>(arrayInfo.ptr);
    return ctor(shapedType, numElements, contents);
  }

  static bool isUnsignedIntegerFormat(const std::string &format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'I' || code == 'B' || code == 'H' || code == 'L' ||
           code == 'Q';
  }

  static bool isSignedIntegerFormat(const std::string &format) {
    if (format.empty())
      return false;
    char code = format[0];
    return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
           code == 'q';
  }

  template <typename Type>
  py::buffer_info bufferInfo(MlirType shapedType,
                             Type (*value)(MlirAttribute, intptr_t)) {
    intptr_t rank = mlirShapedTypeGetRank(shapedType);
    // Prepare the data for the buffer_info.
    // Buffer is configured for read-only access below.
    Type *data = static_cast<Type *>(
        const_cast<void *>(mlirDenseElementsAttrGetRawData(*this)));
    // Prepare the shape for the buffer_info.
    SmallVector<intptr_t, 4> shape;
    for (intptr_t i = 0; i < rank; ++i)
      shape.push_back(mlirShapedTypeGetDimSize(shapedType, i));
    // Prepare the strides for the buffer_info.
    SmallVector<intptr_t, 4> strides;
    intptr_t strideFactor = 1;
    for (intptr_t i = 1; i < rank; ++i) {
      strideFactor = 1;
      for (intptr_t j = i; j < rank; ++j) {
        strideFactor *= mlirShapedTypeGetDimSize(shapedType, j);
      }
      strides.push_back(sizeof(Type) * strideFactor);
    }
    strides.push_back(sizeof(Type));
    return py::buffer_info(data, sizeof(Type),
                           py::format_descriptor<Type>::format(), rank, shape,
                           strides, /*readonly=*/true);
  }
}; // namespace

/// Refinement of the PyDenseElementsAttribute for attributes containing integer
/// (and boolean) values. Supports element access.
class PyDenseIntElementsAttribute
    : public PyConcreteAttribute<PyDenseIntElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseIntElements;
  static constexpr const char *pyClassName = "DenseIntElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  /// Returns the element at the given linear position. Asserts if the index is
  /// out of range.
  py::int_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    assert(mlirTypeIsAInteger(type) &&
           "expected integer element type in dense int elements attribute");
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::int_ is implicitly constructible
    // from any C++ integral type and handles bitwidth correctly.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    unsigned width = mlirIntegerTypeGetWidth(type);
    bool isUnsigned = mlirIntegerTypeIsUnsigned(type);
    if (isUnsigned) {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetUInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetUInt64Value(*this, pos);
      }
    } else {
      if (width == 1) {
        return mlirDenseElementsAttrGetBoolValue(*this, pos);
      }
      if (width == 32) {
        return mlirDenseElementsAttrGetInt32Value(*this, pos);
      }
      if (width == 64) {
        return mlirDenseElementsAttrGetInt64Value(*this, pos);
      }
    }
    throw SetPyError(PyExc_TypeError, "Unsupported integer type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseIntElementsAttribute::dunderGetItem);
  }
};

/// Refinement of PyDenseElementsAttribute for attributes containing
/// floating-point values. Supports element access.
class PyDenseFPElementsAttribute
    : public PyConcreteAttribute<PyDenseFPElementsAttribute,
                                 PyDenseElementsAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsADenseFPElements;
  static constexpr const char *pyClassName = "DenseFPElementsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  py::float_ dunderGetItem(intptr_t pos) {
    if (pos < 0 || pos >= dunderLen()) {
      throw SetPyError(PyExc_IndexError,
                       "attempt to access out of bounds element");
    }

    MlirType type = mlirAttributeGetType(*this);
    type = mlirShapedTypeGetElementType(type);
    // Dispatch element extraction to an appropriate C function based on the
    // elemental type of the attribute. py::float_ is implicitly constructible
    // from float and double.
    // TODO: consider caching the type properties in the constructor to avoid
    // querying them on each element access.
    if (mlirTypeIsAF32(type)) {
      return mlirDenseElementsAttrGetFloatValue(*this, pos);
    }
    if (mlirTypeIsAF64(type)) {
      return mlirDenseElementsAttrGetDoubleValue(*this, pos);
    }
    throw SetPyError(PyExc_TypeError, "Unsupported floating-point type");
  }

  static void bindDerived(ClassTy &c) {
    c.def("__getitem__", &PyDenseFPElementsAttribute::dunderGetItem);
  }
};

/// Unit Attribute subclass. Unit attributes don't have values.
class PyUnitAttribute : public PyConcreteAttribute<PyUnitAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAUnit;
  static constexpr const char *pyClassName = "UnitAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return PyUnitAttribute(context->getRef(),
                                 mlirUnitAttrGet(context->get()));
        },
        py::arg("context") = py::none(), "Create a Unit attribute.");
  }
};

} // namespace

//------------------------------------------------------------------------------
// Standard type subclasses.
//------------------------------------------------------------------------------

namespace {

/// CRTP base classes for Python types that subclass Type and should be
/// castable from it (i.e. via something like IntegerType(t)).
/// By default, type class hierarchies are one level deep (i.e. a
/// concrete type class extends PyType); however, intermediate python-visible
/// base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyType>
class PyConcreteType : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = py::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirType);

  PyConcreteType() = default;
  PyConcreteType(PyMlirContextRef contextRef, MlirType t)
      : BaseTy(std::move(contextRef), t) {}
  PyConcreteType(PyType &orig)
      : PyConcreteType(orig.getContext(), castFrom(orig)) {}

  static MlirType castFrom(PyType &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr = py::repr(py::cast(orig)).cast<std::string>();
      throw SetPyError(PyExc_ValueError, Twine("Cannot cast type to ") +
                                             DerivedTy::pyClassName +
                                             " (from " + origRepr + ")");
    }
    return orig;
  }

  static void bind(py::module &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(py::init<PyType &>(), py::keep_alive<0, 1>());
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class PyIntegerType : public PyConcreteType<PyIntegerType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAInteger;
  static constexpr const char *pyClassName = "IntegerType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_signless",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create a signless integer type");
    c.def_static(
        "get_signed",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeSignedGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create a signed integer type");
    c.def_static(
        "get_unsigned",
        [](unsigned width, DefaultingPyMlirContext context) {
          MlirType t = mlirIntegerTypeUnsignedGet(context->get(), width);
          return PyIntegerType(context->getRef(), t);
        },
        py::arg("width"), py::arg("context") = py::none(),
        "Create an unsigned integer type");
    c.def_property_readonly(
        "width",
        [](PyIntegerType &self) { return mlirIntegerTypeGetWidth(self); },
        "Returns the width of the integer type");
    c.def_property_readonly(
        "is_signless",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSignless(self);
        },
        "Returns whether this is a signless integer");
    c.def_property_readonly(
        "is_signed",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsSigned(self);
        },
        "Returns whether this is a signed integer");
    c.def_property_readonly(
        "is_unsigned",
        [](PyIntegerType &self) -> bool {
          return mlirIntegerTypeIsUnsigned(self);
        },
        "Returns whether this is an unsigned integer");
  }
};

/// Index Type subclass - IndexType.
class PyIndexType : public PyConcreteType<PyIndexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAIndex;
  static constexpr const char *pyClassName = "IndexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirIndexTypeGet(context->get());
          return PyIndexType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a index type.");
  }
};

/// Floating Point Type subclass - BF16Type.
class PyBF16Type : public PyConcreteType<PyBF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsABF16;
  static constexpr const char *pyClassName = "BF16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirBF16TypeGet(context->get());
          return PyBF16Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a bf16 type.");
  }
};

/// Floating Point Type subclass - F16Type.
class PyF16Type : public PyConcreteType<PyF16Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF16;
  static constexpr const char *pyClassName = "F16Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF16TypeGet(context->get());
          return PyF16Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f16 type.");
  }
};

/// Floating Point Type subclass - F32Type.
class PyF32Type : public PyConcreteType<PyF32Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF32;
  static constexpr const char *pyClassName = "F32Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF32TypeGet(context->get());
          return PyF32Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f32 type.");
  }
};

/// Floating Point Type subclass - F64Type.
class PyF64Type : public PyConcreteType<PyF64Type> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAF64;
  static constexpr const char *pyClassName = "F64Type";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirF64TypeGet(context->get());
          return PyF64Type(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a f64 type.");
  }
};

/// None Type subclass - NoneType.
class PyNoneType : public PyConcreteType<PyNoneType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsANone;
  static constexpr const char *pyClassName = "NoneType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = mlirNoneTypeGet(context->get());
          return PyNoneType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create a none type.");
  }
};

/// Complex Type subclass - ComplexType.
class PyComplexType : public PyConcreteType<PyComplexType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAComplex;
  static constexpr const char *pyClassName = "ComplexType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType) {
          // The element must be a floating point or integer scalar type.
          if (mlirTypeIsAIntegerOrFloat(elementType)) {
            MlirType t = mlirComplexTypeGet(elementType);
            return PyComplexType(elementType.getContext(), t);
          }
          throw SetPyError(
              PyExc_ValueError,
              Twine("invalid '") +
                  py::repr(py::cast(elementType)).cast<std::string>() +
                  "' and expected floating point or integer type.");
        },
        "Create a complex type");
    c.def_property_readonly(
        "element_type",
        [](PyComplexType &self) -> PyType {
          MlirType t = mlirComplexTypeGetElementType(self);
          return PyType(self.getContext(), t);
        },
        "Returns element type.");
  }
};

class PyShapedType : public PyConcreteType<PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAShaped;
  static constexpr const char *pyClassName = "ShapedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_property_readonly(
        "element_type",
        [](PyShapedType &self) {
          MlirType t = mlirShapedTypeGetElementType(self);
          return PyType(self.getContext(), t);
        },
        "Returns the element type of the shaped type.");
    c.def_property_readonly(
        "has_rank",
        [](PyShapedType &self) -> bool { return mlirShapedTypeHasRank(self); },
        "Returns whether the given shaped type is ranked.");
    c.def_property_readonly(
        "rank",
        [](PyShapedType &self) {
          self.requireHasRank();
          return mlirShapedTypeGetRank(self);
        },
        "Returns the rank of the given ranked shaped type.");
    c.def_property_readonly(
        "has_static_shape",
        [](PyShapedType &self) -> bool {
          return mlirShapedTypeHasStaticShape(self);
        },
        "Returns whether the given shaped type has a static shape.");
    c.def(
        "is_dynamic_dim",
        [](PyShapedType &self, intptr_t dim) -> bool {
          self.requireHasRank();
          return mlirShapedTypeIsDynamicDim(self, dim);
        },
        "Returns whether the dim-th dimension of the given shaped type is "
        "dynamic.");
    c.def(
        "get_dim_size",
        [](PyShapedType &self, intptr_t dim) {
          self.requireHasRank();
          return mlirShapedTypeGetDimSize(self, dim);
        },
        "Returns the dim-th dimension of the given ranked shaped type.");
    c.def_static(
        "is_dynamic_size",
        [](int64_t size) -> bool { return mlirShapedTypeIsDynamicSize(size); },
        "Returns whether the given dimension size indicates a dynamic "
        "dimension.");
    c.def(
        "is_dynamic_stride_or_offset",
        [](PyShapedType &self, int64_t val) -> bool {
          self.requireHasRank();
          return mlirShapedTypeIsDynamicStrideOrOffset(val);
        },
        "Returns whether the given value is used as a placeholder for dynamic "
        "strides and offsets in shaped types.");
  }

private:
  void requireHasRank() {
    if (!mlirShapedTypeHasRank(*this)) {
      throw SetPyError(
          PyExc_ValueError,
          "calling this method requires that the type has a rank.");
    }
  }
};

/// Vector Type subclass - VectorType.
class PyVectorType : public PyConcreteType<PyVectorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAVector;
  static constexpr const char *pyClassName = "VectorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           DefaultingPyLocation loc) {
          MlirType t = mlirVectorTypeGetChecked(shape.size(), shape.data(),
                                                elementType, loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point or integer type.");
          }
          return PyVectorType(elementType.getContext(), t);
        },
        py::arg("shape"), py::arg("elementType"), py::arg("loc") = py::none(),
        "Create a vector type");
  }
};

/// Ranked Tensor Type subclass - RankedTensorType.
class PyRankedTensorType
    : public PyConcreteType<PyRankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr const char *pyClassName = "RankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, PyType &elementType,
           DefaultingPyLocation loc) {
          MlirType t = mlirRankedTensorTypeGetChecked(
              shape.size(), shape.data(), elementType, loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
          return PyRankedTensorType(elementType.getContext(), t);
        },
        py::arg("shape"), py::arg("element_type"), py::arg("loc") = py::none(),
        "Create a ranked tensor type");
  }
};

/// Unranked Tensor Type subclass - UnrankedTensorType.
class PyUnrankedTensorType
    : public PyConcreteType<PyUnrankedTensorType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedTensor;
  static constexpr const char *pyClassName = "UnrankedTensorType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elementType, DefaultingPyLocation loc) {
          MlirType t = mlirUnrankedTensorTypeGetChecked(elementType, loc);
          // TODO: Rework error reporting once diagnostic engine is exposed
          // in C API.
          if (mlirTypeIsNull(t)) {
            throw SetPyError(
                PyExc_ValueError,
                Twine("invalid '") +
                    py::repr(py::cast(elementType)).cast<std::string>() +
                    "' and expected floating point, integer, vector or "
                    "complex "
                    "type.");
          }
          return PyUnrankedTensorType(elementType.getContext(), t);
        },
        py::arg("element_type"), py::arg("loc") = py::none(),
        "Create a unranked tensor type");
  }
};

/// Ranked MemRef Type subclass - MemRefType.
class PyMemRefType : public PyConcreteType<PyMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsARankedTensor;
  static constexpr const char *pyClassName = "MemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    // TODO: Add mlirMemRefTypeGet and mlirMemRefTypeGetAffineMap binding
    // once the affine map binding is completed.
    c.def_static(
         "get_contiguous_memref",
         // TODO: Make the location optional and create a default location.
         [](PyType &elementType, std::vector<int64_t> shape,
            unsigned memorySpace, DefaultingPyLocation loc) {
           MlirType t = mlirMemRefTypeContiguousGetChecked(
               elementType, shape.size(), shape.data(), memorySpace, loc);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyMemRefType(elementType.getContext(), t);
         },
         py::arg("element_type"), py::arg("shape"), py::arg("memory_space"),
         py::arg("loc") = py::none(), "Create a memref type")
        .def_property_readonly(
            "num_affine_maps",
            [](PyMemRefType &self) -> intptr_t {
              return mlirMemRefTypeGetNumAffineMaps(self);
            },
            "Returns the number of affine layout maps in the given MemRef "
            "type.")
        .def_property_readonly(
            "memory_space",
            [](PyMemRefType &self) -> unsigned {
              return mlirMemRefTypeGetMemorySpace(self);
            },
            "Returns the memory space of the given MemRef type.");
  }
};

/// Unranked MemRef Type subclass - UnrankedMemRefType.
class PyUnrankedMemRefType
    : public PyConcreteType<PyUnrankedMemRefType, PyShapedType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAUnrankedMemRef;
  static constexpr const char *pyClassName = "UnrankedMemRefType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
         "get",
         [](PyType &elementType, unsigned memorySpace,
            DefaultingPyLocation loc) {
           MlirType t =
               mlirUnrankedMemRefTypeGetChecked(elementType, memorySpace, loc);
           // TODO: Rework error reporting once diagnostic engine is exposed
           // in C API.
           if (mlirTypeIsNull(t)) {
             throw SetPyError(
                 PyExc_ValueError,
                 Twine("invalid '") +
                     py::repr(py::cast(elementType)).cast<std::string>() +
                     "' and expected floating point, integer, vector or "
                     "complex "
                     "type.");
           }
           return PyUnrankedMemRefType(elementType.getContext(), t);
         },
         py::arg("element_type"), py::arg("memory_space"),
         py::arg("loc") = py::none(), "Create a unranked memref type")
        .def_property_readonly(
            "memory_space",
            [](PyUnrankedMemRefType &self) -> unsigned {
              return mlirUnrankedMemrefGetMemorySpace(self);
            },
            "Returns the memory space of the given Unranked MemRef type.");
  }
};

/// Tuple Type subclass - TupleType.
class PyTupleType : public PyConcreteType<PyTupleType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATuple;
  static constexpr const char *pyClassName = "TupleType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get_tuple",
        [](py::list elementList, DefaultingPyMlirContext context) {
          intptr_t num = py::len(elementList);
          // Mapping py::list to SmallVector.
          SmallVector<MlirType, 4> elements;
          for (auto element : elementList)
            elements.push_back(element.cast<PyType>());
          MlirType t = mlirTupleTypeGet(context->get(), num, elements.data());
          return PyTupleType(context->getRef(), t);
        },
        py::arg("elements"), py::arg("context") = py::none(),
        "Create a tuple type");
    c.def(
        "get_type",
        [](PyTupleType &self, intptr_t pos) -> PyType {
          MlirType t = mlirTupleTypeGetType(self, pos);
          return PyType(self.getContext(), t);
        },
        "Returns the pos-th type in the tuple type.");
    c.def_property_readonly(
        "num_types",
        [](PyTupleType &self) -> intptr_t {
          return mlirTupleTypeGetNumTypes(self);
        },
        "Returns the number of types contained in a tuple.");
  }
};

/// Function type.
class PyFunctionType : public PyConcreteType<PyFunctionType> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsAFunction;
  static constexpr const char *pyClassName = "FunctionType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<PyType> inputs, std::vector<PyType> results,
           DefaultingPyMlirContext context) {
          SmallVector<MlirType, 4> inputsRaw(inputs.begin(), inputs.end());
          SmallVector<MlirType, 4> resultsRaw(results.begin(), results.end());
          MlirType t = mlirFunctionTypeGet(context->get(), inputsRaw.size(),
                                           inputsRaw.data(), resultsRaw.size(),
                                           resultsRaw.data());
          return PyFunctionType(context->getRef(), t);
        },
        py::arg("inputs"), py::arg("results"), py::arg("context") = py::none(),
        "Gets a FunctionType from a list of input and result types");
    c.def_property_readonly(
        "inputs",
        [](PyFunctionType &self) {
          MlirType t = self;
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumInputs(self); i < e;
               ++i) {
            types.append(PyType(contextRef, mlirFunctionTypeGetInput(t, i)));
          }
          return types;
        },
        "Returns the list of input types in the FunctionType.");
    c.def_property_readonly(
        "results",
        [](PyFunctionType &self) {
          auto contextRef = self.getContext();
          py::list types;
          for (intptr_t i = 0, e = mlirFunctionTypeGetNumResults(self); i < e;
               ++i) {
            types.append(
                PyType(contextRef, mlirFunctionTypeGetResult(self, i)));
          }
          return types;
        },
        "Returns the list of result types in the FunctionType.");
  }
};

} // namespace

//------------------------------------------------------------------------------
// Populates the pybind11 IR submodule.
//------------------------------------------------------------------------------

void mlir::python::populateIRSubmodule(py::module &m) {
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
           py::arg("use_local_scope") = false, kOperationGetAsmDocstring);

  py::class_<PyOperation, PyOperationBase>(m, "Operation")
      .def_static("create", &PyOperation::create, py::arg("name"),
                  py::arg("operands") = py::none(),
                  py::arg("results") = py::none(),
                  py::arg("attributes") = py::none(),
                  py::arg("successors") = py::none(), py::arg("regions") = 0,
                  py::arg("loc") = py::none(), py::arg("ip") = py::none(),
                  kOperationCreateDocstring)
      .def_property_readonly(
          "context",
          [](PyOperation &self) { return self.getContext().getObject(); },
          "Context that owns the Operation")
      .def_property_readonly("opview", &PyOperation::createOpView);

  py::class_<PyOpView, PyOperationBase>(m, "OpView")
      .def(py::init<py::object>())
      .def_property_readonly("operation", &PyOpView::getOperationObject)
      .def_property_readonly(
          "context",
          [](PyOpView &self) {
            return self.getOperation().getContext().getObject();
          },
          "Context that owns the Operation")
      .def("__str__",
           [](PyOpView &self) { return py::str(self.getOperationObject()); });

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
             printAccum.parts.append(self.namedAttr.name.data);
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
            return py::str(self.namedAttr.name.data,
                           self.namedAttr.name.length);
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

  // Standard attribute bindings.
  PyFloatAttribute::bind(m);
  PyIntegerAttribute::bind(m);
  PyBoolAttribute::bind(m);
  PyStringAttribute::bind(m);
  PyDenseElementsAttribute::bind(m);
  PyDenseIntElementsAttribute::bind(m);
  PyDenseFPElementsAttribute::bind(m);
  PyUnitAttribute::bind(m);

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

  // Standard type bindings.
  PyIntegerType::bind(m);
  PyIndexType::bind(m);
  PyBF16Type::bind(m);
  PyF16Type::bind(m);
  PyF32Type::bind(m);
  PyF64Type::bind(m);
  PyNoneType::bind(m);
  PyComplexType::bind(m);
  PyShapedType::bind(m);
  PyVectorType::bind(m);
  PyRankedTensorType::bind(m);
  PyUnrankedTensorType::bind(m);
  PyMemRefType::bind(m);
  PyUnrankedMemRefType::bind(m);
  PyTupleType::bind(m);
  PyFunctionType::bind(m);

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
