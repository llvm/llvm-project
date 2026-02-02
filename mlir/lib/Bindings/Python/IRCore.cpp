//===- IRModules.cpp - IR Submodules of pybind module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "mlir/Bindings/Python/Globals.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/NanobindUtils.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Debug.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

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

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

/// Helper for creating an @classmethod.
template <class Func, typename... Args>
static nb::object classmethod(Func f, Args... args) {
  nb::object cf = nb::cpp_function(f, args...);
  return nb::borrow<nb::object>((PyClassMethod_New(cf.ptr())));
}

static nb::object
createCustomDialectWrapper(const std::string &dialectNamespace,
                           nb::object dialectDescriptor) {
  auto dialectClass =
      python::MLIR_BINDINGS_PYTHON_DOMAIN::PyGlobals::get().lookupDialectClass(
          dialectNamespace);
  if (!dialectClass) {
    // Use the base class.
    return nb::cast(python::MLIR_BINDINGS_PYTHON_DOMAIN::PyDialect(
        std::move(dialectDescriptor)));
  }

  // Create the custom implementation.
  return (*dialectClass)(std::move(dialectDescriptor));
}

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

MlirBlock createBlock(const nb::sequence &pyArgTypes,
                      const std::optional<nb::sequence> &pyArgLocs) {
  SmallVector<MlirType> argTypes;
  argTypes.reserve(nb::len(pyArgTypes));
  for (const auto &pyType : pyArgTypes)
    argTypes.push_back(
        nb::cast<python::MLIR_BINDINGS_PYTHON_DOMAIN::PyType &>(pyType));

  SmallVector<MlirLocation> argLocs;
  if (pyArgLocs) {
    argLocs.reserve(nb::len(*pyArgLocs));
    for (const auto &pyLoc : *pyArgLocs)
      argLocs.push_back(
          nb::cast<python::MLIR_BINDINGS_PYTHON_DOMAIN::PyLocation &>(pyLoc));
  } else if (!argTypes.empty()) {
    argLocs.assign(
        argTypes.size(),
        python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyLocation::resolve());
  }

  if (argTypes.size() != argLocs.size())
    throw nb::value_error(("Expected " + Twine(argTypes.size()) +
                           " locations, got: " + Twine(argLocs.size()))
                              .str()
                              .c_str());
  return mlirBlockCreate(argTypes.size(), argTypes.data(), argLocs.data());
}

void PyGlobalDebugFlag::set(nb::object &o, bool enable) {
  nb::ft_lock_guard lock(mutex);
  mlirEnableGlobalDebug(enable);
}

bool PyGlobalDebugFlag::get(const nb::object &) {
  nb::ft_lock_guard lock(mutex);
  return mlirIsGlobalDebugEnabled();
}

void PyGlobalDebugFlag::bind(nb::module_ &m) {
  // Debug flags.
  nb::class_<PyGlobalDebugFlag>(m, "_GlobalDebug")
      .def_prop_rw_static("flag", &PyGlobalDebugFlag::get,
                          &PyGlobalDebugFlag::set, "LLVM-wide debug flag.")
      .def_static(
          "set_types",
          [](const std::string &type) {
            nb::ft_lock_guard lock(mutex);
            mlirSetGlobalDebugType(type.c_str());
          },
          "types"_a, "Sets specific debug types to be produced by LLVM.")
      .def_static(
          "set_types",
          [](const std::vector<std::string> &types) {
            std::vector<const char *> pointers;
            pointers.reserve(types.size());
            for (const std::string &str : types)
              pointers.push_back(str.c_str());
            nb::ft_lock_guard lock(mutex);
            mlirSetGlobalDebugTypes(pointers.data(), pointers.size());
          },
          "types"_a,
          "Sets multiple specific debug types to be produced by LLVM.");
}

nb::ft_mutex PyGlobalDebugFlag::mutex;

bool PyAttrBuilderMap::dunderContains(const std::string &attributeKind) {
  return PyGlobals::get().lookupAttributeBuilder(attributeKind).has_value();
}

nb::callable
PyAttrBuilderMap::dunderGetItemNamed(const std::string &attributeKind) {
  auto builder = PyGlobals::get().lookupAttributeBuilder(attributeKind);
  if (!builder)
    throw nb::key_error(attributeKind.c_str());
  return *builder;
}

void PyAttrBuilderMap::dunderSetItemNamed(const std::string &attributeKind,
                                          nb::callable func, bool replace) {
  PyGlobals::get().registerAttributeBuilder(attributeKind, std::move(func),
                                            replace);
}

void PyAttrBuilderMap::bind(nb::module_ &m) {
  nb::class_<PyAttrBuilderMap>(m, "AttrBuilder")
      .def_static("contains", &PyAttrBuilderMap::dunderContains,
                  "attribute_kind"_a,
                  "Checks whether an attribute builder is registered for the "
                  "given attribute kind.")
      .def_static("get", &PyAttrBuilderMap::dunderGetItemNamed,
                  "attribute_kind"_a,
                  "Gets the registered attribute builder for the given "
                  "attribute kind.")
      .def_static("insert", &PyAttrBuilderMap::dunderSetItemNamed,
                  "attribute_kind"_a, "attr_builder"_a, "replace"_a = false,
                  "Register an attribute builder for building MLIR "
                  "attributes from Python values.");
}

//------------------------------------------------------------------------------
// PyBlock
//------------------------------------------------------------------------------

nb::object PyBlock::getCapsule() {
  return nb::steal<nb::object>(mlirPythonBlockToCapsule(get()));
}

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

nb::typed<nb::object, PyRegion> PyRegionIterator::dunderNext() {
  operation->checkValid();
  if (nextIndex >= mlirOperationGetNumRegions(operation->get())) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }
  MlirRegion region = mlirOperationGetRegion(operation->get(), nextIndex++);
  return nb::cast(PyRegion(operation, region));
}

void PyRegionIterator::bind(nb::module_ &m) {
  nb::class_<PyRegionIterator>(m, "RegionIterator")
      .def("__iter__", &PyRegionIterator::dunderIter,
           "Returns an iterator over the regions in the operation.")
      .def("__next__", &PyRegionIterator::dunderNext,
           "Returns the next region in the iteration.");
}

PyRegionList::PyRegionList(PyOperationRef operation, intptr_t startIndex,
                           intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirOperationGetNumRegions(operation->get())
                             : length,
                step),
      operation(std::move(operation)) {}

PyRegionIterator PyRegionList::dunderIter() {
  operation->checkValid();
  return PyRegionIterator(operation, startIndex);
}

void PyRegionList::bindDerived(ClassTy &c) {
  c.def("__iter__", &PyRegionList::dunderIter,
        "Returns an iterator over the regions in the sequence.");
}

intptr_t PyRegionList::getRawNumElements() {
  operation->checkValid();
  return mlirOperationGetNumRegions(operation->get());
}

PyRegion PyRegionList::getRawElement(intptr_t pos) {
  operation->checkValid();
  return PyRegion(operation, mlirOperationGetRegion(operation->get(), pos));
}

PyRegionList PyRegionList::slice(intptr_t startIndex, intptr_t length,
                                 intptr_t step) const {
  return PyRegionList(operation, startIndex, length, step);
}

nb::typed<nb::object, PyBlock> PyBlockIterator::dunderNext() {
  operation->checkValid();
  if (mlirBlockIsNull(next)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyBlock returnBlock(operation, next);
  next = mlirBlockGetNextInRegion(next);
  return nb::cast(returnBlock);
}

void PyBlockIterator::bind(nb::module_ &m) {
  nb::class_<PyBlockIterator>(m, "BlockIterator")
      .def("__iter__", &PyBlockIterator::dunderIter,
           "Returns an iterator over the blocks in the operation's region.")
      .def("__next__", &PyBlockIterator::dunderNext,
           "Returns the next block in the iteration.");
}

PyBlockIterator PyBlockList::dunderIter() {
  operation->checkValid();
  return PyBlockIterator(operation, mlirRegionGetFirstBlock(region));
}

intptr_t PyBlockList::dunderLen() {
  operation->checkValid();
  intptr_t count = 0;
  MlirBlock block = mlirRegionGetFirstBlock(region);
  while (!mlirBlockIsNull(block)) {
    count += 1;
    block = mlirBlockGetNextInRegion(block);
  }
  return count;
}

PyBlock PyBlockList::dunderGetItem(intptr_t index) {
  operation->checkValid();
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0) {
    throw nb::index_error("attempt to access out of bounds block");
  }
  MlirBlock block = mlirRegionGetFirstBlock(region);
  while (!mlirBlockIsNull(block)) {
    if (index == 0) {
      return PyBlock(operation, block);
    }
    block = mlirBlockGetNextInRegion(block);
    index -= 1;
  }
  throw nb::index_error("attempt to access out of bounds block");
}

PyBlock PyBlockList::appendBlock(const nb::args &pyArgTypes,
                                 const std::optional<nb::sequence> &pyArgLocs) {
  operation->checkValid();
  MlirBlock block = createBlock(nb::cast<nb::sequence>(pyArgTypes), pyArgLocs);
  mlirRegionAppendOwnedBlock(region, block);
  return PyBlock(operation, block);
}

void PyBlockList::bind(nb::module_ &m) {
  nb::class_<PyBlockList>(m, "BlockList")
      .def("__getitem__", &PyBlockList::dunderGetItem,
           "Returns the block at the specified index.")
      .def("__iter__", &PyBlockList::dunderIter,
           "Returns an iterator over blocks in the operation's region.")
      .def("__len__", &PyBlockList::dunderLen,
           "Returns the number of blocks in the operation's region.")
      .def("append", &PyBlockList::appendBlock,
           R"(
              Appends a new block, with argument types as positional args.

              Returns:
                The created block.
             )",
           "args"_a, nb::kw_only(), "arg_locs"_a = std::nullopt);
}

nb::typed<nb::object, PyOpView> PyOperationIterator::dunderNext() {
  parentOperation->checkValid();
  if (mlirOperationIsNull(next)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyOperationRef returnOperation =
      PyOperation::forOperation(parentOperation->getContext(), next);
  next = mlirOperationGetNextInBlock(next);
  return returnOperation->createOpView();
}

void PyOperationIterator::bind(nb::module_ &m) {
  nb::class_<PyOperationIterator>(m, "OperationIterator")
      .def("__iter__", &PyOperationIterator::dunderIter,
           "Returns an iterator over the operations in an operation's block.")
      .def("__next__", &PyOperationIterator::dunderNext,
           "Returns the next operation in the iteration.");
}

PyOperationIterator PyOperationList::dunderIter() {
  parentOperation->checkValid();
  return PyOperationIterator(parentOperation,
                             mlirBlockGetFirstOperation(block));
}

intptr_t PyOperationList::dunderLen() {
  parentOperation->checkValid();
  intptr_t count = 0;
  MlirOperation childOp = mlirBlockGetFirstOperation(block);
  while (!mlirOperationIsNull(childOp)) {
    count += 1;
    childOp = mlirOperationGetNextInBlock(childOp);
  }
  return count;
}

nb::typed<nb::object, PyOpView> PyOperationList::dunderGetItem(intptr_t index) {
  parentOperation->checkValid();
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0) {
    throw nb::index_error("attempt to access out of bounds operation");
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
  throw nb::index_error("attempt to access out of bounds operation");
}

void PyOperationList::bind(nb::module_ &m) {
  nb::class_<PyOperationList>(m, "OperationList")
      .def("__getitem__", &PyOperationList::dunderGetItem,
           "Returns the operation at the specified index.")
      .def("__iter__", &PyOperationList::dunderIter,
           "Returns an iterator over operations in the list.")
      .def("__len__", &PyOperationList::dunderLen,
           "Returns the number of operations in the list.");
}

nb::typed<nb::object, PyOpView> PyOpOperand::getOwner() const {
  MlirOperation owner = mlirOpOperandGetOwner(opOperand);
  PyMlirContextRef context =
      PyMlirContext::forContext(mlirOperationGetContext(owner));
  return PyOperation::forOperation(context, owner)->createOpView();
}

size_t PyOpOperand::getOperandNumber() const {
  return mlirOpOperandGetOperandNumber(opOperand);
}

void PyOpOperand::bind(nb::module_ &m) {
  nb::class_<PyOpOperand>(m, "OpOperand")
      .def_prop_ro("owner", &PyOpOperand::getOwner,
                   "Returns the operation that owns this operand.")
      .def_prop_ro("operand_number", &PyOpOperand::getOperandNumber,
                   "Returns the operand number in the owning operation.");
}

nb::typed<nb::object, PyOpOperand> PyOpOperandIterator::dunderNext() {
  if (mlirOpOperandIsNull(opOperand)) {
    PyErr_SetNone(PyExc_StopIteration);
    // python functions should return NULL after setting any exception
    return nb::object();
  }

  PyOpOperand returnOpOperand(opOperand);
  opOperand = mlirOpOperandGetNextUse(opOperand);
  return nb::cast(returnOpOperand);
}

void PyOpOperandIterator::bind(nb::module_ &m) {
  nb::class_<PyOpOperandIterator>(m, "OpOperandIterator")
      .def("__iter__", &PyOpOperandIterator::dunderIter,
           "Returns an iterator over operands.")
      .def("__next__", &PyOpOperandIterator::dunderNext,
           "Returns the next operand in the iteration.");
}

//------------------------------------------------------------------------------
// PyThreadPool
//------------------------------------------------------------------------------

PyThreadPool::PyThreadPool() {
  ownedThreadPool = std::make_unique<llvm::DefaultThreadPool>();
}

std::string PyThreadPool::_mlir_thread_pool_ptr() const {
  std::stringstream ss;
  ss << ownedThreadPool.get();
  return ss.str();
}

//------------------------------------------------------------------------------
// PyMlirContext
//------------------------------------------------------------------------------

PyMlirContext::PyMlirContext(MlirContext context) : context(context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  liveContexts[context.ptr] = this;
}

PyMlirContext::~PyMlirContext() {
  // Note that the only public way to construct an instance is via the
  // forContext method, which always puts the associated handle into
  // liveContexts.
  nb::gil_scoped_acquire acquire;
  {
    nb::ft_lock_guard lock(live_contexts_mutex);
    getLiveContexts().erase(context.ptr);
  }
  mlirContextDestroy(context);
}

PyMlirContextRef PyMlirContext::getRef() {
  return PyMlirContextRef(this, nb::cast(this));
}

nb::object PyMlirContext::getCapsule() {
  return nb::steal<nb::object>(mlirPythonContextToCapsule(get()));
}

nb::object PyMlirContext::createFromCapsule(nb::object capsule) {
  MlirContext rawContext = mlirPythonCapsuleToContext(capsule.ptr());
  if (mlirContextIsNull(rawContext))
    throw nb::python_error();
  return forContext(rawContext).releaseObject();
}

PyMlirContextRef PyMlirContext::forContext(MlirContext context) {
  nb::gil_scoped_acquire acquire;
  nb::ft_lock_guard lock(live_contexts_mutex);
  auto &liveContexts = getLiveContexts();
  auto it = liveContexts.find(context.ptr);
  if (it == liveContexts.end()) {
    // Create.
    PyMlirContext *unownedContextWrapper = new PyMlirContext(context);
    nb::object pyRef = nb::cast(unownedContextWrapper);
    assert(pyRef && "cast to nb::object failed");
    liveContexts[context.ptr] = unownedContextWrapper;
    return PyMlirContextRef(unownedContextWrapper, std::move(pyRef));
  }
  // Use existing.
  nb::object pyRef = nb::cast(it->second);
  return PyMlirContextRef(it->second, std::move(pyRef));
}

nb::ft_mutex PyMlirContext::live_contexts_mutex;

PyMlirContext::LiveContextMap &PyMlirContext::getLiveContexts() {
  static LiveContextMap liveContexts;
  return liveContexts;
}

size_t PyMlirContext::getLiveCount() {
  nb::ft_lock_guard lock(live_contexts_mutex);
  return getLiveContexts().size();
}

nb::object PyMlirContext::contextEnter(nb::object context) {
  return PyThreadContextEntry::pushContext(context);
}

void PyMlirContext::contextExit(const nb::object &excType,
                                const nb::object &excVal,
                                const nb::object &excTb) {
  PyThreadContextEntry::popContext(*this);
}

nb::object PyMlirContext::attachDiagnosticHandler(nb::object callback) {
  // Note that ownership is transferred to the delete callback below by way of
  // an explicit inc_ref (borrow).
  PyDiagnosticHandler *pyHandler =
      new PyDiagnosticHandler(get(), std::move(callback));
  nb::object pyHandlerObject =
      nb::cast(pyHandler, nb::rv_policy::take_ownership);
  (void)pyHandlerObject.inc_ref();

  // In these C callbacks, the userData is a PyDiagnosticHandler* that is
  // guaranteed to be known to pybind.
  auto handlerCallback =
      +[](MlirDiagnostic diagnostic, void *userData) -> MlirLogicalResult {
    PyDiagnostic *pyDiagnostic = new PyDiagnostic(diagnostic);
    nb::object pyDiagnosticObject =
        nb::cast(pyDiagnostic, nb::rv_policy::take_ownership);

    auto *pyHandler = static_cast<PyDiagnosticHandler *>(userData);
    bool result = false;
    {
      // Since this can be called from arbitrary C++ contexts, always get the
      // gil.
      nb::gil_scoped_acquire gil;
      try {
        result = nb::cast<bool>(pyHandler->callback(pyDiagnostic));
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
    nb::object pyHandlerObject = nb::cast(pyHandler, nb::rv_policy::reference);
    pyHandlerObject.dec_ref();
  };

  pyHandler->registeredID = mlirContextAttachDiagnosticHandler(
      get(), handlerCallback, static_cast<void *>(pyHandler), deleteCallback);
  return pyHandlerObject;
}

MlirLogicalResult PyMlirContext::ErrorCapture::handler(MlirDiagnostic diag,
                                                       void *userData) {
  auto *self = static_cast<ErrorCapture *>(userData);
  // Check if the context requested we emit errors instead of capturing them.
  if (self->ctx->emitErrorDiagnostics)
    return mlirLogicalResultFailure();

  if (mlirDiagnosticGetSeverity(diag) !=
      MlirDiagnosticSeverity::MlirDiagnosticError)
    return mlirLogicalResultFailure();

  self->errors.emplace_back(PyDiagnostic(diag).getInfo());
  return mlirLogicalResultSuccess();
}

PyMlirContext &DefaultingPyMlirContext::resolve() {
  PyMlirContext *context = PyThreadContextEntry::getDefaultContext();
  if (!context) {
    throw std::runtime_error(
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

void PyThreadContextEntry::push(FrameKind frameKind, nb::object context,
                                nb::object insertionPoint,
                                nb::object location) {
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
  return nb::cast<PyMlirContext *>(context);
}

PyInsertionPoint *PyThreadContextEntry::getInsertionPoint() {
  if (!insertionPoint)
    return nullptr;
  return nb::cast<PyInsertionPoint *>(insertionPoint);
}

PyLocation *PyThreadContextEntry::getLocation() {
  if (!location)
    return nullptr;
  return nb::cast<PyLocation *>(location);
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

nb::object PyThreadContextEntry::pushContext(nb::object context) {
  push(FrameKind::Context, /*context=*/context,
       /*insertionPoint=*/nb::object(),
       /*location=*/nb::object());
  return context;
}

void PyThreadContextEntry::popContext(PyMlirContext &context) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Context enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Context && tos.getContext() != &context)
    throw std::runtime_error("Unbalanced Context enter/exit");
  stack.pop_back();
}

nb::object
PyThreadContextEntry::pushInsertionPoint(nb::object insertionPointObj) {
  PyInsertionPoint &insertionPoint =
      nb::cast<PyInsertionPoint &>(insertionPointObj);
  nb::object contextObj =
      insertionPoint.getBlock().getParentOperation()->getContext().getObject();
  push(FrameKind::InsertionPoint,
       /*context=*/contextObj,
       /*insertionPoint=*/insertionPointObj,
       /*location=*/nb::object());
  return insertionPointObj;
}

void PyThreadContextEntry::popInsertionPoint(PyInsertionPoint &insertionPoint) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::InsertionPoint &&
      tos.getInsertionPoint() != &insertionPoint)
    throw std::runtime_error("Unbalanced InsertionPoint enter/exit");
  stack.pop_back();
}

nb::object PyThreadContextEntry::pushLocation(nb::object locationObj) {
  PyLocation &location = nb::cast<PyLocation &>(locationObj);
  nb::object contextObj = location.getContext().getObject();
  push(FrameKind::Location, /*context=*/contextObj,
       /*insertionPoint=*/nb::object(),
       /*location=*/locationObj);
  return locationObj;
}

void PyThreadContextEntry::popLocation(PyLocation &location) {
  auto &stack = getStack();
  if (stack.empty())
    throw std::runtime_error("Unbalanced Location enter/exit");
  auto &tos = stack.back();
  if (tos.frameKind != FrameKind::Location && tos.getLocation() != &location)
    throw std::runtime_error("Unbalanced Location enter/exit");
  stack.pop_back();
}

//------------------------------------------------------------------------------
// PyDiagnostic*
//------------------------------------------------------------------------------

void PyDiagnostic::invalidate() {
  valid = false;
  if (materializedNotes) {
    for (nb::handle noteObject : *materializedNotes) {
      PyDiagnostic *note = nb::cast<PyDiagnostic *>(noteObject);
      note->invalidate();
    }
  }
}

PyDiagnosticHandler::PyDiagnosticHandler(MlirContext context,
                                         nb::object callback)
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

PyDiagnosticSeverity PyDiagnostic::getSeverity() {
  checkValid();
  return static_cast<PyDiagnosticSeverity>(
      mlirDiagnosticGetSeverity(diagnostic));
}

PyLocation PyDiagnostic::getLocation() {
  checkValid();
  MlirLocation loc = mlirDiagnosticGetLocation(diagnostic);
  MlirContext context = mlirLocationGetContext(loc);
  return PyLocation(PyMlirContext::forContext(context), loc);
}

nb::str PyDiagnostic::getMessage() {
  checkValid();
  nb::object fileObject = nb::module_::import_("io").attr("StringIO")();
  PyFileAccumulator accum(fileObject, /*binary=*/false);
  mlirDiagnosticPrint(diagnostic, accum.getCallback(), accum.getUserData());
  return nb::cast<nb::str>(fileObject.attr("getvalue")());
}

nb::tuple PyDiagnostic::getNotes() {
  checkValid();
  if (materializedNotes)
    return *materializedNotes;
  intptr_t numNotes = mlirDiagnosticGetNumNotes(diagnostic);
  nb::tuple notes = nb::steal<nb::tuple>(PyTuple_New(numNotes));
  for (intptr_t i = 0; i < numNotes; ++i) {
    MlirDiagnostic noteDiag = mlirDiagnosticGetNote(diagnostic, i);
    nb::object diagnostic = nb::cast(PyDiagnostic(noteDiag));
    PyTuple_SET_ITEM(notes.ptr(), i, diagnostic.release().ptr());
  }
  materializedNotes = std::move(notes);

  return *materializedNotes;
}

PyDiagnostic::DiagnosticInfo PyDiagnostic::getInfo() {
  std::vector<DiagnosticInfo> notes;
  for (nb::handle n : getNotes())
    notes.emplace_back(nb::cast<PyDiagnostic>(n).getInfo());
  return {getSeverity(), getLocation(), nb::cast<std::string>(getMessage()),
          std::move(notes)};
}

//------------------------------------------------------------------------------
// PyDialect, PyDialectDescriptor, PyDialects, PyDialectRegistry
//------------------------------------------------------------------------------

MlirDialect PyDialects::getDialectForKey(const std::string &key,
                                         bool attrError) {
  MlirDialect dialect = mlirContextGetOrLoadDialect(getContext()->get(),
                                                    {key.data(), key.size()});
  if (mlirDialectIsNull(dialect)) {
    std::string msg = (Twine("Dialect '") + key + "' not found").str();
    if (attrError)
      throw nb::attribute_error(msg.c_str());
    throw nb::index_error(msg.c_str());
  }
  return dialect;
}

nb::object PyDialectRegistry::getCapsule() {
  return nb::steal<nb::object>(mlirPythonDialectRegistryToCapsule(*this));
}

PyDialectRegistry PyDialectRegistry::createFromCapsule(nb::object capsule) {
  MlirDialectRegistry rawRegistry =
      mlirPythonCapsuleToDialectRegistry(capsule.ptr());
  if (mlirDialectRegistryIsNull(rawRegistry))
    throw nb::python_error();
  return PyDialectRegistry(rawRegistry);
}

//------------------------------------------------------------------------------
// PyLocation
//------------------------------------------------------------------------------

nb::object PyLocation::getCapsule() {
  return nb::steal<nb::object>(mlirPythonLocationToCapsule(*this));
}

PyLocation PyLocation::createFromCapsule(nb::object capsule) {
  MlirLocation rawLoc = mlirPythonCapsuleToLocation(capsule.ptr());
  if (mlirLocationIsNull(rawLoc))
    throw nb::python_error();
  return PyLocation(PyMlirContext::forContext(mlirLocationGetContext(rawLoc)),
                    rawLoc);
}

nb::object PyLocation::contextEnter(nb::object locationObj) {
  return PyThreadContextEntry::pushLocation(locationObj);
}

void PyLocation::contextExit(const nb::object &excType,
                             const nb::object &excVal,
                             const nb::object &excTb) {
  PyThreadContextEntry::popLocation(*this);
}

PyLocation &DefaultingPyLocation::resolve() {
  auto *location = PyThreadContextEntry::getDefaultLocation();
  if (!location) {
    throw std::runtime_error(
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
  nb::gil_scoped_acquire acquire;
  auto &liveModules = getContext()->liveModules;
  assert(liveModules.count(module.ptr) == 1 &&
         "destroying module not in live map");
  liveModules.erase(module.ptr);
  mlirModuleDestroy(module);
}

PyModuleRef PyModule::forModule(MlirModule module) {
  MlirContext context = mlirModuleGetContext(module);
  PyMlirContextRef contextRef = PyMlirContext::forContext(context);

  nb::gil_scoped_acquire acquire;
  auto &liveModules = contextRef->liveModules;
  auto it = liveModules.find(module.ptr);
  if (it == liveModules.end()) {
    // Create.
    PyModule *unownedModule = new PyModule(std::move(contextRef), module);
    // Note that the default return value policy on cast is automatic_reference,
    // which does not take ownership (delete will not be called).
    // Just be explicit.
    nb::object pyRef = nb::cast(unownedModule, nb::rv_policy::take_ownership);
    unownedModule->handle = pyRef;
    liveModules[module.ptr] =
        std::make_pair(unownedModule->handle, unownedModule);
    return PyModuleRef(unownedModule, std::move(pyRef));
  }
  // Use existing.
  PyModule *existing = it->second.second;
  nb::object pyRef = nb::borrow<nb::object>(it->second.first);
  return PyModuleRef(existing, std::move(pyRef));
}

nb::object PyModule::createFromCapsule(nb::object capsule) {
  MlirModule rawModule = mlirPythonCapsuleToModule(capsule.ptr());
  if (mlirModuleIsNull(rawModule))
    throw nb::python_error();
  return forModule(rawModule).releaseObject();
}

nb::object PyModule::getCapsule() {
  return nb::steal<nb::object>(mlirPythonModuleToCapsule(get()));
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
  // Otherwise, invalidate the operation when it is attached.
  if (isAttached())
    setInvalid();
  else {
    // And destroy it when it is detached, i.e. owned by Python.
    erase();
  }
}

namespace {

// Constructs a new object of type T in-place on the Python heap, returning a
// PyObjectRef to it, loosely analogous to std::make_shared<T>().
template <typename T, class... Args>
PyObjectRef<T> makeObjectRef(Args &&...args) {
  nb::handle type = nb::type<T>();
  nb::object instance = nb::inst_alloc(type);
  T *ptr = nb::inst_ptr<T>(instance);
  new (ptr) T(std::forward<Args>(args)...);
  nb::inst_mark_ready(instance);
  return PyObjectRef<T>(ptr, std::move(instance));
}

} // namespace

PyOperationRef PyOperation::createInstance(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           nb::object parentKeepAlive) {
  // Create.
  PyOperationRef unownedOperation =
      makeObjectRef<PyOperation>(std::move(contextRef), operation);
  unownedOperation->handle = unownedOperation.getObject();
  if (parentKeepAlive) {
    unownedOperation->parentKeepAlive = std::move(parentKeepAlive);
  }
  return unownedOperation;
}

PyOperationRef PyOperation::forOperation(PyMlirContextRef contextRef,
                                         MlirOperation operation,
                                         nb::object parentKeepAlive) {
  return createInstance(std::move(contextRef), operation,
                        std::move(parentKeepAlive));
}

PyOperationRef PyOperation::createDetached(PyMlirContextRef contextRef,
                                           MlirOperation operation,
                                           nb::object parentKeepAlive) {
  PyOperationRef created = createInstance(std::move(contextRef), operation,
                                          std::move(parentKeepAlive));
  created->attached = false;
  return created;
}

PyOperationRef PyOperation::parse(PyMlirContextRef contextRef,
                                  const std::string &sourceStr,
                                  const std::string &sourceName) {
  PyMlirContext::ErrorCapture errors(contextRef);
  MlirOperation op =
      mlirOperationCreateParse(contextRef->get(), toMlirStringRef(sourceStr),
                               toMlirStringRef(sourceName));
  if (mlirOperationIsNull(op))
    throw MLIRError("Unable to parse operation assembly", errors.take());
  return PyOperation::createDetached(std::move(contextRef), op);
}

void PyOperation::detachFromParent() {
  mlirOperationRemoveFromParent(getOperation());
  setDetached();
  parentKeepAlive = nb::object();
}

MlirOperation PyOperation::get() const {
  checkValid();
  return operation;
}

PyOperationRef PyOperation::getRef() {
  return PyOperationRef(this, nb::borrow<nb::object>(handle));
}

void PyOperation::setAttached(const nb::object &parent) {
  assert(!attached && "operation already attached");
  attached = true;
}

void PyOperation::setDetached() {
  assert(attached && "operation already detached");
  attached = false;
}

void PyOperation::checkValid() const {
  if (!valid) {
    throw std::runtime_error("the operation has been invalidated");
  }
}

void PyOperationBase::print(std::optional<int64_t> largeElementsLimit,
                            std::optional<int64_t> largeResourceLimit,
                            bool enableDebugInfo, bool prettyDebugInfo,
                            bool printGenericOpForm, bool useLocalScope,
                            bool useNameLocAsPrefix, bool assumeVerified,
                            nb::object fileObject, bool binary,
                            bool skipRegions) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");

  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  if (largeElementsLimit)
    mlirOpPrintingFlagsElideLargeElementsAttrs(flags, *largeElementsLimit);
  if (largeResourceLimit)
    mlirOpPrintingFlagsElideLargeResourceString(flags, *largeResourceLimit);
  if (enableDebugInfo)
    mlirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/true,
                                       /*prettyForm=*/prettyDebugInfo);
  if (printGenericOpForm)
    mlirOpPrintingFlagsPrintGenericOpForm(flags);
  if (useLocalScope)
    mlirOpPrintingFlagsUseLocalScope(flags);
  if (assumeVerified)
    mlirOpPrintingFlagsAssumeVerified(flags);
  if (skipRegions)
    mlirOpPrintingFlagsSkipRegions(flags);
  if (useNameLocAsPrefix)
    mlirOpPrintingFlagsPrintNameLocAsPrefix(flags);

  PyFileAccumulator accum(fileObject, binary);
  mlirOperationPrintWithFlags(operation, flags, accum.getCallback(),
                              accum.getUserData());
  mlirOpPrintingFlagsDestroy(flags);
}

void PyOperationBase::print(PyAsmState &state, nb::object fileObject,
                            bool binary) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  if (fileObject.is_none())
    fileObject = nb::module_::import_("sys").attr("stdout");
  PyFileAccumulator accum(fileObject, binary);
  mlirOperationPrintWithState(operation, state.get(), accum.getCallback(),
                              accum.getUserData());
}

void PyOperationBase::writeBytecode(const nb::object &fileOrStringObject,
                                    std::optional<int64_t> bytecodeVersion) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  PyFileAccumulator accum(fileOrStringObject, /*binary=*/true);

  if (!bytecodeVersion.has_value())
    return mlirOperationWriteBytecode(operation, accum.getCallback(),
                                      accum.getUserData());

  MlirBytecodeWriterConfig config = mlirBytecodeWriterConfigCreate();
  mlirBytecodeWriterConfigDesiredEmitVersion(config, *bytecodeVersion);
  MlirLogicalResult res = mlirOperationWriteBytecodeWithConfig(
      operation, config, accum.getCallback(), accum.getUserData());
  mlirBytecodeWriterConfigDestroy(config);
  if (mlirLogicalResultIsFailure(res))
    throw nb::value_error((Twine("Unable to honor desired bytecode version ") +
                           Twine(*bytecodeVersion))
                              .str()
                              .c_str());
}

void PyOperationBase::walk(std::function<PyWalkResult(MlirOperation)> callback,
                           PyWalkOrder walkOrder) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  struct UserData {
    std::function<PyWalkResult(MlirOperation)> callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{callback, false, {}, {}};
  MlirOperationWalkCallback walkCallback = [](MlirOperation op,
                                              void *userData) {
    UserData *calleeUserData = static_cast<UserData *>(userData);
    try {
      return static_cast<MlirWalkResult>((calleeUserData->callback)(op));
    } catch (nb::python_error &e) {
      calleeUserData->gotException = true;
      calleeUserData->exceptionWhat = std::string(e.what());
      calleeUserData->exceptionType = nb::borrow(e.type());
      return MlirWalkResult::MlirWalkResultInterrupt;
    }
  };
  mlirOperationWalk(operation, walkCallback, &userData,
                    static_cast<MlirWalkOrder>(walkOrder));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

nb::object PyOperationBase::getAsm(bool binary,
                                   std::optional<int64_t> largeElementsLimit,
                                   std::optional<int64_t> largeResourceLimit,
                                   bool enableDebugInfo, bool prettyDebugInfo,
                                   bool printGenericOpForm, bool useLocalScope,
                                   bool useNameLocAsPrefix, bool assumeVerified,
                                   bool skipRegions) {
  nb::object fileObject;
  if (binary) {
    fileObject = nb::module_::import_("io").attr("BytesIO")();
  } else {
    fileObject = nb::module_::import_("io").attr("StringIO")();
  }
  print(/*largeElementsLimit=*/largeElementsLimit,
        /*largeResourceLimit=*/largeResourceLimit,
        /*enableDebugInfo=*/enableDebugInfo,
        /*prettyDebugInfo=*/prettyDebugInfo,
        /*printGenericOpForm=*/printGenericOpForm,
        /*useLocalScope=*/useLocalScope,
        /*useNameLocAsPrefix=*/useNameLocAsPrefix,
        /*assumeVerified=*/assumeVerified,
        /*fileObject=*/fileObject,
        /*binary=*/binary,
        /*skipRegions=*/skipRegions);

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

bool PyOperationBase::isBeforeInBlock(PyOperationBase &other) {
  PyOperation &operation = getOperation();
  PyOperation &otherOp = other.getOperation();
  operation.checkValid();
  otherOp.checkValid();
  return mlirOperationIsBeforeInBlock(operation, otherOp);
}

bool PyOperationBase::verify() {
  PyOperation &op = getOperation();
  PyMlirContext::ErrorCapture errors(op.getContext());
  if (!mlirOperationVerify(op.get()))
    throw MLIRError("Verification failed", errors.take());
  return true;
}

std::optional<PyOperationRef> PyOperation::getParentOperation() {
  checkValid();
  if (!isAttached())
    throw nb::value_error("Detached operations have no parent");
  MlirOperation operation = mlirOperationGetParentOperation(get());
  if (mlirOperationIsNull(operation))
    return {};
  return PyOperation::forOperation(getContext(), operation);
}

PyBlock PyOperation::getBlock() {
  checkValid();
  std::optional<PyOperationRef> parentOperation = getParentOperation();
  MlirBlock block = mlirOperationGetBlock(get());
  assert(!mlirBlockIsNull(block) && "Attached operation has null parent");
  assert(parentOperation && "Operation has no parent");
  return PyBlock{std::move(*parentOperation), block};
}

nb::object PyOperation::getCapsule() {
  checkValid();
  return nb::steal<nb::object>(mlirPythonOperationToCapsule(get()));
}

nb::object PyOperation::createFromCapsule(const nb::object &capsule) {
  MlirOperation rawOperation = mlirPythonCapsuleToOperation(capsule.ptr());
  if (mlirOperationIsNull(rawOperation))
    throw nb::python_error();
  MlirContext rawCtxt = mlirOperationGetContext(rawOperation);
  return forOperation(PyMlirContext::forContext(rawCtxt), rawOperation)
      .releaseObject();
}

static void maybeInsertOperation(PyOperationRef &op,
                                 const nb::object &maybeIp) {
  // InsertPoint active?
  if (!maybeIp.is(nb::cast(false))) {
    PyInsertionPoint *ip;
    if (maybeIp.is_none()) {
      ip = PyThreadContextEntry::getDefaultInsertionPoint();
    } else {
      ip = nb::cast<PyInsertionPoint *>(maybeIp);
    }
    if (ip)
      ip->insert(*op.get());
  }
}

nb::object PyOperation::create(std::string_view name,
                               std::optional<std::vector<PyType *>> results,
                               llvm::ArrayRef<MlirValue> operands,
                               std::optional<nb::dict> attributes,
                               std::optional<std::vector<PyBlock *>> successors,
                               int regions, PyLocation &location,
                               const nb::object &maybeIp, bool inferType) {
  llvm::SmallVector<MlirType, 4> mlirResults;
  llvm::SmallVector<MlirBlock, 4> mlirSuccessors;
  llvm::SmallVector<std::pair<std::string, MlirAttribute>, 4> mlirAttributes;

  // General parameter validation.
  if (regions < 0)
    throw nb::value_error("number of regions must be >= 0");

  // Unpack/validate results.
  if (results) {
    mlirResults.reserve(results->size());
    for (PyType *result : *results) {
      // TODO: Verify result type originate from the same context.
      if (!result)
        throw nb::value_error("result type cannot be None");
      mlirResults.push_back(*result);
    }
  }
  // Unpack/validate attributes.
  if (attributes) {
    mlirAttributes.reserve(attributes->size());
    for (std::pair<nb::handle, nb::handle> it : *attributes) {
      std::string key;
      try {
        key = nb::cast<std::string>(it.first);
      } catch (nb::cast_error &err) {
        std::string msg = "Invalid attribute key (not a string) when "
                          "attempting to create the operation \"" +
                          std::string(name) + "\" (" + err.what() + ")";
        throw nb::type_error(msg.c_str());
      }
      try {
        auto &attribute = nb::cast<PyAttribute &>(it.second);
        // TODO: Verify attribute originates from the same context.
        mlirAttributes.emplace_back(std::move(key), attribute);
      } catch (nb::cast_error &err) {
        std::string msg = "Invalid attribute value for the key \"" + key +
                          "\" when attempting to create the operation \"" +
                          std::string(name) + "\" (" + err.what() + ")";
        throw nb::type_error(msg.c_str());
      } catch (std::runtime_error &) {
        // This exception seems thrown when the value is "None".
        std::string msg =
            "Found an invalid (`None`?) attribute value for the key \"" + key +
            "\" when attempting to create the operation \"" +
            std::string(name) + "\"";
        throw std::runtime_error(msg);
      }
    }
  }
  // Unpack/validate successors.
  if (successors) {
    mlirSuccessors.reserve(successors->size());
    for (auto *successor : *successors) {
      // TODO: Verify successor originate from the same context.
      if (!successor)
        throw nb::value_error("successor block cannot be None");
      mlirSuccessors.push_back(successor->get());
    }
  }

  // Apply unpacked/validated to the operation state. Beyond this
  // point, exceptions cannot be thrown or else the state will leak.
  MlirOperationState state =
      mlirOperationStateGet(toMlirStringRef(name), location);
  if (!operands.empty())
    mlirOperationStateAddOperands(&state, operands.size(), operands.data());
  state.enableResultTypeInference = inferType;
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
  PyMlirContext::ErrorCapture errors(location.getContext());
  MlirOperation operation = mlirOperationCreate(&state);
  if (!operation.ptr)
    throw MLIRError("Operation creation failed", errors.take());
  PyOperationRef created =
      PyOperation::createDetached(location.getContext(), operation);
  maybeInsertOperation(created, maybeIp);

  return created.getObject();
}

nb::object PyOperation::clone(const nb::object &maybeIp) {
  MlirOperation clonedOperation = mlirOperationClone(operation);
  PyOperationRef cloned =
      PyOperation::createDetached(getContext(), clonedOperation);
  maybeInsertOperation(cloned, maybeIp);

  return cloned->createOpView();
}

nb::object PyOperation::createOpView() {
  checkValid();
  MlirIdentifier ident = mlirOperationGetName(get());
  MlirStringRef identStr = mlirIdentifierStr(ident);
  auto operationCls = PyGlobals::get().lookupOperationClass(
      StringRef(identStr.data, identStr.length));
  if (operationCls)
    return PyOpView::constructDerived(*operationCls, getRef().getObject());
  return nb::cast(PyOpView(getRef().getObject()));
}

void PyOperation::erase() {
  checkValid();
  setInvalid();
  mlirOperationDestroy(operation);
}

void PyOpResult::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "owner",
      [](PyOpResult &self) -> nb::typed<nb::object, PyOpView> {
        assert(mlirOperationEqual(self.getParentOperation()->get(),
                                  mlirOpResultGetOwner(self.get())) &&
               "expected the owner of the value in Python to match that in "
               "the IR");
        return self.getParentOperation()->createOpView();
      },
      "Returns the operation that produces this result.");
  c.def_prop_ro(
      "result_number",
      [](PyOpResult &self) { return mlirOpResultGetResultNumber(self.get()); },
      "Returns the position of this result in the operation's result list.");
}

/// Returns the list of types of the values held by container.
template <typename Container>
static std::vector<nb::typed<nb::object, PyType>>
getValueTypes(Container &container, PyMlirContextRef &context) {
  std::vector<nb::typed<nb::object, PyType>> result;
  result.reserve(container.size());
  for (int i = 0, e = container.size(); i < e; ++i) {
    result.push_back(PyType(context->getRef(),
                            mlirValueGetType(container.getElement(i).get()))
                         .maybeDownCast());
  }
  return result;
}

PyOpResultList::PyOpResultList(PyOperationRef operation, intptr_t startIndex,
                               intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirOperationGetNumResults(operation->get())
                             : length,
                step),
      operation(std::move(operation)) {}

void PyOpResultList::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "types",
      [](PyOpResultList &self) {
        return getValueTypes(self, self.operation->getContext());
      },
      "Returns a list of types for all results in this result list.");
  c.def_prop_ro(
      "owner",
      [](PyOpResultList &self) -> nb::typed<nb::object, PyOpView> {
        return self.operation->createOpView();
      },
      "Returns the operation that owns this result list.");
}

intptr_t PyOpResultList::getRawNumElements() {
  operation->checkValid();
  return mlirOperationGetNumResults(operation->get());
}

PyOpResult PyOpResultList::getRawElement(intptr_t index) {
  PyValue value(operation, mlirOperationGetResult(operation->get(), index));
  return PyOpResult(value);
}

PyOpResultList PyOpResultList::slice(intptr_t startIndex, intptr_t length,
                                     intptr_t step) const {
  return PyOpResultList(operation, startIndex, length, step);
}

//------------------------------------------------------------------------------
// PyOpView
//------------------------------------------------------------------------------

static void populateResultTypes(StringRef name, nb::list resultTypeList,
                                const nb::object &resultSegmentSpecObj,
                                std::vector<int32_t> &resultSegmentLengths,
                                std::vector<PyType *> &resultTypes) {
  resultTypes.reserve(resultTypeList.size());
  if (resultSegmentSpecObj.is_none()) {
    // Non-variadic result unpacking.
    for (const auto &it : llvm::enumerate(resultTypeList)) {
      try {
        resultTypes.push_back(nb::cast<PyType *>(it.value()));
        if (!resultTypes.back())
          throw nb::cast_error();
      } catch (nb::cast_error &err) {
        throw nb::value_error((llvm::Twine("Result ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Type (" + err.what() + ")")
                                  .str()
                                  .c_str());
      }
    }
  } else {
    // Sized result unpacking.
    auto resultSegmentSpec = nb::cast<std::vector<int>>(resultSegmentSpecObj);
    if (resultSegmentSpec.size() != resultTypeList.size()) {
      throw nb::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(resultSegmentSpec.size()) +
                             " result segments but was provided " +
                             llvm::Twine(resultTypeList.size()))
                                .str()
                                .c_str());
    }
    resultSegmentLengths.reserve(resultTypeList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(resultTypeList, resultSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        try {
          auto *resultType = nb::cast<PyType *>(std::get<0>(it.value()));
          if (resultType) {
            resultTypes.push_back(resultType);
            resultSegmentLengths.push_back(1);
          } else if (segmentSpec == 0) {
            // Allowed to be optional.
            resultSegmentLengths.push_back(0);
          } else {
            throw nb::value_error(
                (llvm::Twine("Result ") + llvm::Twine(it.index()) +
                 " of operation \"" + name +
                 "\" must be a Type (was None and result is not optional)")
                    .str()
                    .c_str());
          }
        } catch (nb::cast_error &err) {
          throw nb::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Type (" + err.what() +
                                 ")")
                                    .str()
                                    .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            resultSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(std::get<0>(it.value()));
            for (nb::handle segmentItem : segment) {
              resultTypes.push_back(nb::cast<PyType *>(segmentItem));
              if (!resultTypes.back()) {
                throw nb::type_error("contained a None item");
              }
            }
            resultSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error((llvm::Twine("Result ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Types (" +
                                 err.what() + ")")
                                    .str()
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }
}

MlirValue getUniqueResult(MlirOperation operation) {
  auto numResults = mlirOperationGetNumResults(operation);
  if (numResults != 1) {
    auto name = mlirIdentifierStr(mlirOperationGetName(operation));
    throw nb::value_error((Twine("Cannot call .result on operation ") +
                           StringRef(name.data, name.length) + " which has " +
                           Twine(numResults) +
                           " results (it is only valid for operations with a "
                           "single result)")
                              .str()
                              .c_str());
  }
  return mlirOperationGetResult(operation, 0);
}

static MlirValue getOpResultOrValue(nb::handle operand) {
  if (operand.is_none()) {
    throw nb::value_error("contained a None item");
  }
  PyOperationBase *op;
  if (nb::try_cast<PyOperationBase *>(operand, op)) {
    return getUniqueResult(op->getOperation());
  }
  PyOpResultList *opResultList;
  if (nb::try_cast<PyOpResultList *>(operand, opResultList)) {
    return getUniqueResult(opResultList->getOperation()->get());
  }
  PyValue *value;
  if (nb::try_cast<PyValue *>(operand, value)) {
    return value->get();
  }
  throw nb::value_error("is not a Value");
}

nb::object PyOpView::buildGeneric(
    std::string_view name, std::tuple<int, bool> opRegionSpec,
    nb::object operandSegmentSpecObj, nb::object resultSegmentSpecObj,
    std::optional<nb::list> resultTypeList, nb::list operandList,
    std::optional<nb::dict> attributes,
    std::optional<std::vector<PyBlock *>> successors,
    std::optional<int> regions, PyLocation &location,
    const nb::object &maybeIp) {
  PyMlirContextRef context = location.getContext();

  // Class level operation construction metadata.
  // Operand and result segment specs are either none, which does no
  // variadic unpacking, or a list of ints with segment sizes, where each
  // element is either a positive number (typically 1 for a scalar) or -1 to
  // indicate that it is derived from the length of the same-indexed operand
  // or result (implying that it is a list at that position).
  std::vector<int32_t> operandSegmentLengths;
  std::vector<int32_t> resultSegmentLengths;

  // Validate/determine region count.
  int opMinRegionCount = std::get<0>(opRegionSpec);
  bool opHasNoVariadicRegions = std::get<1>(opRegionSpec);
  if (!regions) {
    regions = opMinRegionCount;
  }
  if (*regions < opMinRegionCount) {
    throw nb::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a minimum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str()
            .c_str());
  }
  if (opHasNoVariadicRegions && *regions > opMinRegionCount) {
    throw nb::value_error(
        (llvm::Twine("Operation \"") + name + "\" requires a maximum of " +
         llvm::Twine(opMinRegionCount) +
         " regions but was built with regions=" + llvm::Twine(*regions))
            .str()
            .c_str());
  }

  // Unpack results.
  std::vector<PyType *> resultTypes;
  if (resultTypeList.has_value()) {
    populateResultTypes(name, *resultTypeList, resultSegmentSpecObj,
                        resultSegmentLengths, resultTypes);
  }

  // Unpack operands.
  llvm::SmallVector<MlirValue, 4> operands;
  operands.reserve(operands.size());
  if (operandSegmentSpecObj.is_none()) {
    // Non-sized operand unpacking.
    for (const auto &it : llvm::enumerate(operandList)) {
      try {
        operands.push_back(getOpResultOrValue(it.value()));
      } catch (nb::builtin_exception &err) {
        throw nb::value_error((llvm::Twine("Operand ") +
                               llvm::Twine(it.index()) + " of operation \"" +
                               name + "\" must be a Value (" + err.what() + ")")
                                  .str()
                                  .c_str());
      }
    }
  } else {
    // Sized operand unpacking.
    auto operandSegmentSpec = nb::cast<std::vector<int>>(operandSegmentSpecObj);
    if (operandSegmentSpec.size() != operandList.size()) {
      throw nb::value_error((llvm::Twine("Operation \"") + name +
                             "\" requires " +
                             llvm::Twine(operandSegmentSpec.size()) +
                             "operand segments but was provided " +
                             llvm::Twine(operandList.size()))
                                .str()
                                .c_str());
    }
    operandSegmentLengths.reserve(operandList.size());
    for (const auto &it :
         llvm::enumerate(llvm::zip(operandList, operandSegmentSpec))) {
      int segmentSpec = std::get<1>(it.value());
      if (segmentSpec == 1 || segmentSpec == 0) {
        // Unpack unary element.
        auto &operand = std::get<0>(it.value());
        if (!operand.is_none()) {
          try {

            operands.push_back(getOpResultOrValue(operand));
          } catch (nb::builtin_exception &err) {
            throw nb::value_error((llvm::Twine("Operand ") +
                                   llvm::Twine(it.index()) +
                                   " of operation \"" + name +
                                   "\" must be a Value (" + err.what() + ")")
                                      .str()
                                      .c_str());
          }

          operandSegmentLengths.push_back(1);
        } else if (segmentSpec == 0) {
          // Allowed to be optional.
          operandSegmentLengths.push_back(0);
        } else {
          throw nb::value_error(
              (llvm::Twine("Operand ") + llvm::Twine(it.index()) +
               " of operation \"" + name +
               "\" must be a Value (was None and operand is not optional)")
                  .str()
                  .c_str());
        }
      } else if (segmentSpec == -1) {
        // Unpack sequence by appending.
        try {
          if (std::get<0>(it.value()).is_none()) {
            // Treat it as an empty list.
            operandSegmentLengths.push_back(0);
          } else {
            // Unpack the list.
            auto segment = nb::cast<nb::sequence>(std::get<0>(it.value()));
            for (nb::handle segmentItem : segment) {
              operands.push_back(getOpResultOrValue(segmentItem));
            }
            operandSegmentLengths.push_back(nb::len(segment));
          }
        } catch (std::exception &err) {
          // NOTE: Sloppy to be using a catch-all here, but there are at least
          // three different unrelated exceptions that can be thrown in the
          // above "casts". Just keep the scope above small and catch them all.
          throw nb::value_error((llvm::Twine("Operand ") +
                                 llvm::Twine(it.index()) + " of operation \"" +
                                 name + "\" must be a Sequence of Values (" +
                                 err.what() + ")")
                                    .str()
                                    .c_str());
        }
      } else {
        throw nb::value_error("Unexpected segment spec");
      }
    }
  }

  // Merge operand/result segment lengths into attributes if needed.
  if (!operandSegmentLengths.empty() || !resultSegmentLengths.empty()) {
    // Dup.
    if (attributes) {
      attributes = nb::dict(*attributes);
    } else {
      attributes = nb::dict();
    }
    if (attributes->contains("resultSegmentSizes") ||
        attributes->contains("operandSegmentSizes")) {
      throw nb::value_error("Manually setting a 'resultSegmentSizes' or "
                            "'operandSegmentSizes' attribute is unsupported. "
                            "Use Operation.create for such low-level access.");
    }

    // Add resultSegmentSizes attribute.
    if (!resultSegmentLengths.empty()) {
      MlirAttribute segmentLengthAttr =
          mlirDenseI32ArrayGet(context->get(), resultSegmentLengths.size(),
                               resultSegmentLengths.data());
      (*attributes)["resultSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }

    // Add operandSegmentSizes attribute.
    if (!operandSegmentLengths.empty()) {
      MlirAttribute segmentLengthAttr =
          mlirDenseI32ArrayGet(context->get(), operandSegmentLengths.size(),
                               operandSegmentLengths.data());
      (*attributes)["operandSegmentSizes"] =
          PyAttribute(context, segmentLengthAttr);
    }
  }

  // Delegate to create.
  return PyOperation::create(name,
                             /*results=*/std::move(resultTypes),
                             /*operands=*/operands,
                             /*attributes=*/std::move(attributes),
                             /*successors=*/std::move(successors),
                             /*regions=*/*regions, location, maybeIp,
                             !resultTypeList);
}

nb::object PyOpView::constructDerived(const nb::object &cls,
                                      const nb::object &operation) {
  nb::handle opViewType = nb::type<PyOpView>();
  nb::object instance = cls.attr("__new__")(cls);
  opViewType.attr("__init__")(instance, operation);
  return instance;
}

PyOpView::PyOpView(const nb::object &operationObject)
    // Casting through the PyOperationBase base-class and then back to the
    // Operation lets us accept any PyOperationBase subclass.
    : operation(nb::cast<PyOperationBase &>(operationObject).getOperation()),
      operationObject(operation.getRef().getObject()) {}

//------------------------------------------------------------------------------
// PyAsmState
//------------------------------------------------------------------------------

PyAsmState::PyAsmState(MlirValue value, bool useLocalScope) {
  flags = mlirOpPrintingFlagsCreate();
  // The OpPrintingFlags are not exposed Python side, create locally and
  // associate lifetime with the state.
  if (useLocalScope)
    mlirOpPrintingFlagsUseLocalScope(flags);
  state = mlirAsmStateCreateForValue(value, flags);
}

PyAsmState::PyAsmState(PyOperationBase &operation, bool useLocalScope) {
  flags = mlirOpPrintingFlagsCreate();
  // The OpPrintingFlags are not exposed Python side, create locally and
  // associate lifetime with the state.
  if (useLocalScope)
    mlirOpPrintingFlagsUseLocalScope(flags);
  state = mlirAsmStateCreateForOperation(operation.getOperation().get(), flags);
}

//------------------------------------------------------------------------------
// PyInsertionPoint.
//------------------------------------------------------------------------------

PyInsertionPoint::PyInsertionPoint(const PyBlock &block) : block(block) {}

PyInsertionPoint::PyInsertionPoint(PyOperationBase &beforeOperationBase)
    : refOperation(beforeOperationBase.getOperation().getRef()),
      block((*refOperation)->getBlock()) {}

PyInsertionPoint::PyInsertionPoint(PyOperationRef beforeOperationRef)
    : refOperation(beforeOperationRef), block((*refOperation)->getBlock()) {}

void PyInsertionPoint::insert(PyOperationBase &operationBase) {
  PyOperation &operation = operationBase.getOperation();
  if (operation.isAttached())
    throw nb::value_error(
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
      throw nb::index_error("Cannot insert operation at the end of a block "
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
    throw nb::value_error("Block has no terminator");
  PyOperationRef terminatorOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), terminator);
  return PyInsertionPoint{block, std::move(terminatorOpRef)};
}

PyInsertionPoint PyInsertionPoint::after(PyOperationBase &op) {
  PyOperation &operation = op.getOperation();
  PyBlock block = operation.getBlock();
  MlirOperation nextOperation = mlirOperationGetNextInBlock(operation);
  if (mlirOperationIsNull(nextOperation))
    return PyInsertionPoint(block);
  PyOperationRef nextOpRef = PyOperation::forOperation(
      block.getParentOperation()->getContext(), nextOperation);
  return PyInsertionPoint{block, std::move(nextOpRef)};
}

size_t PyMlirContext::getLiveModuleCount() { return liveModules.size(); }

nb::object PyInsertionPoint::contextEnter(nb::object insertPoint) {
  return PyThreadContextEntry::pushInsertionPoint(std::move(insertPoint));
}

void PyInsertionPoint::contextExit(const nb::object &excType,
                                   const nb::object &excVal,
                                   const nb::object &excTb) {
  PyThreadContextEntry::popInsertionPoint(*this);
}

//------------------------------------------------------------------------------
// PyAttribute.
//------------------------------------------------------------------------------

bool PyAttribute::operator==(const PyAttribute &other) const {
  return mlirAttributeEqual(attr, other.attr);
}

nb::object PyAttribute::getCapsule() {
  return nb::steal<nb::object>(mlirPythonAttributeToCapsule(*this));
}

PyAttribute PyAttribute::createFromCapsule(const nb::object &capsule) {
  MlirAttribute rawAttr = mlirPythonCapsuleToAttribute(capsule.ptr());
  if (mlirAttributeIsNull(rawAttr))
    throw nb::python_error();
  return PyAttribute(
      PyMlirContext::forContext(mlirAttributeGetContext(rawAttr)), rawAttr);
}

nb::typed<nb::object, PyAttribute> PyAttribute::maybeDownCast() {
  MlirTypeID mlirTypeID = mlirAttributeGetTypeID(this->get());
  assert(!mlirTypeIDIsNull(mlirTypeID) &&
         "mlirTypeID was expected to be non-null.");
  std::optional<nb::callable> typeCaster = PyGlobals::get().lookupTypeCaster(
      mlirTypeID, mlirAttributeGetDialect(this->get()));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!typeCaster)
    return thisObj;
  return typeCaster.value()(thisObj);
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

bool PyType::operator==(const PyType &other) const {
  return mlirTypeEqual(type, other.type);
}

nb::object PyType::getCapsule() {
  return nb::steal<nb::object>(mlirPythonTypeToCapsule(*this));
}

PyType PyType::createFromCapsule(nb::object capsule) {
  MlirType rawType = mlirPythonCapsuleToType(capsule.ptr());
  if (mlirTypeIsNull(rawType))
    throw nb::python_error();
  return PyType(PyMlirContext::forContext(mlirTypeGetContext(rawType)),
                rawType);
}

nb::typed<nb::object, PyType> PyType::maybeDownCast() {
  MlirTypeID mlirTypeID = mlirTypeGetTypeID(this->get());
  assert(!mlirTypeIDIsNull(mlirTypeID) &&
         "mlirTypeID was expected to be non-null.");
  std::optional<nb::callable> typeCaster = PyGlobals::get().lookupTypeCaster(
      mlirTypeID, mlirTypeGetDialect(this->get()));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!typeCaster)
    return thisObj;
  return typeCaster.value()(thisObj);
}

//------------------------------------------------------------------------------
// PyTypeID.
//------------------------------------------------------------------------------

nb::object PyTypeID::getCapsule() {
  return nb::steal<nb::object>(mlirPythonTypeIDToCapsule(*this));
}

PyTypeID PyTypeID::createFromCapsule(nb::object capsule) {
  MlirTypeID mlirTypeID = mlirPythonCapsuleToTypeID(capsule.ptr());
  if (mlirTypeIDIsNull(mlirTypeID))
    throw nb::python_error();
  return PyTypeID(mlirTypeID);
}
bool PyTypeID::operator==(const PyTypeID &other) const {
  return mlirTypeIDEqual(typeID, other.typeID);
}

//------------------------------------------------------------------------------
// PyValue and subclasses.
//------------------------------------------------------------------------------

nb::object PyValue::getCapsule() {
  return nb::steal<nb::object>(mlirPythonValueToCapsule(get()));
}

static PyOperationRef getValueOwnerRef(MlirValue value) {
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  else if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  else
    assert(false && "Value must be an block arg or op result.");
  if (mlirOperationIsNull(owner))
    throw nb::python_error();
  MlirContext ctx = mlirOperationGetContext(owner);
  return PyOperation::forOperation(PyMlirContext::forContext(ctx), owner);
}

nb::typed<nb::object, std::variant<PyBlockArgument, PyOpResult, PyValue>>
PyValue::maybeDownCast() {
  MlirType type = mlirValueGetType(get());
  MlirTypeID mlirTypeID = mlirTypeGetTypeID(type);
  assert(!mlirTypeIDIsNull(mlirTypeID) &&
         "mlirTypeID was expected to be non-null.");
  std::optional<nb::callable> valueCaster =
      PyGlobals::get().lookupValueCaster(mlirTypeID, mlirTypeGetDialect(type));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj;
  if (mlirValueIsAOpResult(value))
    thisObj = nb::cast<PyOpResult>(*this, nb::rv_policy::move);
  else if (mlirValueIsABlockArgument(value))
    thisObj = nb::cast<PyBlockArgument>(*this, nb::rv_policy::move);
  else
    assert(false && "Value must be an block arg or op result.");
  if (valueCaster)
    return valueCaster.value()(thisObj);
  return thisObj;
}

PyValue PyValue::createFromCapsule(nb::object capsule) {
  MlirValue value = mlirPythonCapsuleToValue(capsule.ptr());
  if (mlirValueIsNull(value))
    throw nb::python_error();
  PyOperationRef ownerRef = getValueOwnerRef(value);
  return PyValue(ownerRef, value);
}

//------------------------------------------------------------------------------
// PySymbolTable.
//------------------------------------------------------------------------------

PySymbolTable::PySymbolTable(PyOperationBase &operation)
    : operation(operation.getOperation().getRef()) {
  symbolTable = mlirSymbolTableCreate(operation.getOperation().get());
  if (mlirSymbolTableIsNull(symbolTable)) {
    throw nb::type_error("Operation is not a Symbol Table.");
  }
}

nb::object PySymbolTable::dunderGetItem(const std::string &name) {
  operation->checkValid();
  MlirOperation symbol = mlirSymbolTableLookup(
      symbolTable, mlirStringRefCreate(name.data(), name.length()));
  if (mlirOperationIsNull(symbol))
    throw nb::key_error(
        ("Symbol '" + name + "' not in the symbol table.").c_str());

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
  nb::object operation = dunderGetItem(name);
  erase(nb::cast<PyOperationBase &>(operation));
}

PyStringAttribute PySymbolTable::insert(PyOperationBase &symbol) {
  operation->checkValid();
  symbol.getOperation().checkValid();
  MlirAttribute symbolAttr = mlirOperationGetAttributeByName(
      symbol.getOperation().get(), mlirSymbolTableGetSymbolAttributeName());
  if (mlirAttributeIsNull(symbolAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return PyStringAttribute(
      symbol.getOperation().getContext(),
      mlirSymbolTableInsert(symbolTable, symbol.getOperation().get()));
}

PyStringAttribute PySymbolTable::getSymbolName(PyOperationBase &symbol) {
  // Op must already be a symbol.
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetSymbolAttributeName();
  MlirAttribute existingNameAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingNameAttr))
    throw nb::value_error("Expected operation to have a symbol name.");
  return PyStringAttribute(symbol.getOperation().getContext(),
                           existingNameAttr);
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
    throw nb::value_error("Expected operation to have a symbol name.");
  MlirAttribute newNameAttr =
      mlirStringAttrGet(operation.getContext()->get(), toMlirStringRef(name));
  mlirOperationSetAttributeByName(operation.get(), attrName, newNameAttr);
}

PyStringAttribute PySymbolTable::getVisibility(PyOperationBase &symbol) {
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
  return PyStringAttribute(symbol.getOperation().getContext(), existingVisAttr);
}

void PySymbolTable::setVisibility(PyOperationBase &symbol,
                                  const std::string &visibility) {
  if (visibility != "public" && visibility != "private" &&
      visibility != "nested")
    throw nb::value_error(
        "Expected visibility to be 'public', 'private' or 'nested'");
  PyOperation &operation = symbol.getOperation();
  operation.checkValid();
  MlirStringRef attrName = mlirSymbolTableGetVisibilityAttributeName();
  MlirAttribute existingVisAttr =
      mlirOperationGetAttributeByName(operation.get(), attrName);
  if (mlirAttributeIsNull(existingVisAttr))
    throw nb::value_error("Expected operation to have a symbol visibility.");
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

    throw nb::value_error("Symbol rename failed");
}

void PySymbolTable::walkSymbolTables(PyOperationBase &from,
                                     bool allSymUsesVisible,
                                     nb::object callback) {
  PyOperation &fromOperation = from.getOperation();
  fromOperation.checkValid();
  struct UserData {
    PyMlirContextRef context;
    nb::object callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
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
        } catch (nb::python_error &e) {
          calleeUserData->gotException = true;
          calleeUserData->exceptionWhat = e.what();
          calleeUserData->exceptionType = nb::borrow(e.type());
        }
      },
      static_cast<void *>(&userData));
  if (userData.gotException) {
    std::string message("Exception raised in callback: ");
    message.append(userData.exceptionWhat);
    throw std::runtime_error(message);
  }
}

void PyBlockArgument::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "owner",
      [](PyBlockArgument &self) {
        return PyBlock(self.getParentOperation(),
                       mlirBlockArgumentGetOwner(self.get()));
      },
      "Returns the block that owns this argument.");
  c.def_prop_ro(
      "arg_number",
      [](PyBlockArgument &self) {
        return mlirBlockArgumentGetArgNumber(self.get());
      },
      "Returns the position of this argument in the block's argument list.");
  c.def(
      "set_type",
      [](PyBlockArgument &self, PyType type) {
        return mlirBlockArgumentSetType(self.get(), type);
      },
      "type"_a, "Sets the type of this block argument.");
  c.def(
      "set_location",
      [](PyBlockArgument &self, PyLocation loc) {
        return mlirBlockArgumentSetLocation(self.get(), loc);
      },
      "loc"_a, "Sets the location of this block argument.");
}

PyBlockArgumentList::PyBlockArgumentList(PyOperationRef operation,
                                         MlirBlock block, intptr_t startIndex,
                                         intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirBlockGetNumArguments(block) : length, step),
      operation(std::move(operation)), block(block) {}

void PyBlockArgumentList::bindDerived(ClassTy &c) {
  c.def_prop_ro(
      "types",
      [](PyBlockArgumentList &self) {
        return getValueTypes(self, self.operation->getContext());
      },
      "Returns a list of types for all arguments in this argument list.");
}

intptr_t PyBlockArgumentList::getRawNumElements() {
  operation->checkValid();
  return mlirBlockGetNumArguments(block);
}

PyBlockArgument PyBlockArgumentList::getRawElement(intptr_t pos) const {
  MlirValue argument = mlirBlockGetArgument(block, pos);
  return PyBlockArgument(operation, argument);
}

PyBlockArgumentList PyBlockArgumentList::slice(intptr_t startIndex,
                                               intptr_t length,
                                               intptr_t step) const {
  return PyBlockArgumentList(operation, block, startIndex, length, step);
}

PyOpOperandList::PyOpOperandList(PyOperationRef operation, intptr_t startIndex,
                                 intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirOperationGetNumOperands(operation->get())
                             : length,
                step),
      operation(operation) {}

void PyOpOperandList::dunderSetItem(intptr_t index, PyValue value) {
  index = wrapIndex(index);
  mlirOperationSetOperand(operation->get(), index, value.get());
}

void PyOpOperandList::bindDerived(ClassTy &c) {
  c.def("__setitem__", &PyOpOperandList::dunderSetItem, "index"_a, "value"_a,
        "Sets the operand at the specified index to a new value.");
}

intptr_t PyOpOperandList::getRawNumElements() {
  operation->checkValid();
  return mlirOperationGetNumOperands(operation->get());
}

PyValue PyOpOperandList::getRawElement(intptr_t pos) {
  MlirValue operand = mlirOperationGetOperand(operation->get(), pos);
  PyOperationRef pyOwner = getValueOwnerRef(operand);
  return PyValue(pyOwner, operand);
}

PyOpOperandList PyOpOperandList::slice(intptr_t startIndex, intptr_t length,
                                       intptr_t step) const {
  return PyOpOperandList(operation, startIndex, length, step);
}

PyOpSuccessors::PyOpSuccessors(PyOperationRef operation, intptr_t startIndex,
                               intptr_t length, intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirOperationGetNumSuccessors(operation->get())
                             : length,
                step),
      operation(operation) {}

void PyOpSuccessors::dunderSetItem(intptr_t index, PyBlock block) {
  index = wrapIndex(index);
  mlirOperationSetSuccessor(operation->get(), index, block.get());
}

void PyOpSuccessors::bindDerived(ClassTy &c) {
  c.def("__setitem__", &PyOpSuccessors::dunderSetItem, "index"_a, "block"_a,
        "Sets the successor block at the specified index.");
}

intptr_t PyOpSuccessors::getRawNumElements() {
  operation->checkValid();
  return mlirOperationGetNumSuccessors(operation->get());
}

PyBlock PyOpSuccessors::getRawElement(intptr_t pos) {
  MlirBlock block = mlirOperationGetSuccessor(operation->get(), pos);
  return PyBlock(operation, block);
}

PyOpSuccessors PyOpSuccessors::slice(intptr_t startIndex, intptr_t length,
                                     intptr_t step) const {
  return PyOpSuccessors(operation, startIndex, length, step);
}

PyBlockSuccessors::PyBlockSuccessors(PyBlock block, PyOperationRef operation,
                                     intptr_t startIndex, intptr_t length,
                                     intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirBlockGetNumSuccessors(block.get()) : length,
                step),
      operation(operation), block(block) {}

intptr_t PyBlockSuccessors::getRawNumElements() {
  block.checkValid();
  return mlirBlockGetNumSuccessors(block.get());
}

PyBlock PyBlockSuccessors::getRawElement(intptr_t pos) {
  MlirBlock block = mlirBlockGetSuccessor(this->block.get(), pos);
  return PyBlock(operation, block);
}

PyBlockSuccessors PyBlockSuccessors::slice(intptr_t startIndex, intptr_t length,
                                           intptr_t step) const {
  return PyBlockSuccessors(block, operation, startIndex, length, step);
}

PyBlockPredecessors::PyBlockPredecessors(PyBlock block,
                                         PyOperationRef operation,
                                         intptr_t startIndex, intptr_t length,
                                         intptr_t step)
    : Sliceable(startIndex,
                length == -1 ? mlirBlockGetNumPredecessors(block.get())
                             : length,
                step),
      operation(operation), block(block) {}

intptr_t PyBlockPredecessors::getRawNumElements() {
  block.checkValid();
  return mlirBlockGetNumPredecessors(block.get());
}

PyBlock PyBlockPredecessors::getRawElement(intptr_t pos) {
  MlirBlock block = mlirBlockGetPredecessor(this->block.get(), pos);
  return PyBlock(operation, block);
}

PyBlockPredecessors PyBlockPredecessors::slice(intptr_t startIndex,
                                               intptr_t length,
                                               intptr_t step) const {
  return PyBlockPredecessors(block, operation, startIndex, length, step);
}

nb::typed<nb::object, PyAttribute>
PyOpAttributeMap::dunderGetItemNamed(const std::string &name) {
  MlirAttribute attr =
      mlirOperationGetAttributeByName(operation->get(), toMlirStringRef(name));
  if (mlirAttributeIsNull(attr)) {
    throw nb::key_error("attempt to access a non-existent attribute");
  }
  return PyAttribute(operation->getContext(), attr).maybeDownCast();
}

nb::typed<nb::object, std::optional<PyAttribute>>
PyOpAttributeMap::get(const std::string &key, nb::object defaultValue) {
  MlirAttribute attr =
      mlirOperationGetAttributeByName(operation->get(), toMlirStringRef(key));
  if (mlirAttributeIsNull(attr))
    return defaultValue;
  return PyAttribute(operation->getContext(), attr).maybeDownCast();
}

PyNamedAttribute PyOpAttributeMap::dunderGetItemIndexed(intptr_t index) {
  if (index < 0) {
    index += dunderLen();
  }
  if (index < 0 || index >= dunderLen()) {
    throw nb::index_error("attempt to access out of bounds attribute");
  }
  MlirNamedAttribute namedAttr =
      mlirOperationGetAttribute(operation->get(), index);
  return PyNamedAttribute(
      namedAttr.attribute,
      std::string(mlirIdentifierStr(namedAttr.name).data,
                  mlirIdentifierStr(namedAttr.name).length));
}

void PyOpAttributeMap::dunderSetItem(const std::string &name,
                                     const PyAttribute &attr) {
  mlirOperationSetAttributeByName(operation->get(), toMlirStringRef(name),
                                  attr);
}

void PyOpAttributeMap::dunderDelItem(const std::string &name) {
  int removed = mlirOperationRemoveAttributeByName(operation->get(),
                                                   toMlirStringRef(name));
  if (!removed)
    throw nb::key_error("attempt to delete a non-existent attribute");
}

intptr_t PyOpAttributeMap::dunderLen() {
  return mlirOperationGetNumAttributes(operation->get());
}

bool PyOpAttributeMap::dunderContains(const std::string &name) {
  return !mlirAttributeIsNull(
      mlirOperationGetAttributeByName(operation->get(), toMlirStringRef(name)));
}

void PyOpAttributeMap::forEachAttr(
    MlirOperation op,
    llvm::function_ref<void(MlirStringRef, MlirAttribute)> fn) {
  intptr_t n = mlirOperationGetNumAttributes(op);
  for (intptr_t i = 0; i < n; ++i) {
    MlirNamedAttribute na = mlirOperationGetAttribute(op, i);
    MlirStringRef name = mlirIdentifierStr(na.name);
    fn(name, na.attribute);
  }
}

void PyOpAttributeMap::bind(nb::module_ &m) {
  nb::class_<PyOpAttributeMap>(m, "OpAttributeMap")
      .def("__contains__", &PyOpAttributeMap::dunderContains, "name"_a,
           "Checks if an attribute with the given name exists in the map.")
      .def("__len__", &PyOpAttributeMap::dunderLen,
           "Returns the number of attributes in the map.")
      .def("__getitem__", &PyOpAttributeMap::dunderGetItemNamed, "name"_a,
           "Gets an attribute by name.")
      .def("__getitem__", &PyOpAttributeMap::dunderGetItemIndexed, "index"_a,
           "Gets a named attribute by index.")
      .def("__setitem__", &PyOpAttributeMap::dunderSetItem, "name"_a, "attr"_a,
           "Sets an attribute with the given name.")
      .def("__delitem__", &PyOpAttributeMap::dunderDelItem, "name"_a,
           "Deletes an attribute with the given name.")
      .def("get", &PyOpAttributeMap::get, nb::arg("key"),
           nb::arg("default") = nb::none(),
           "Gets an attribute by name or the default value, if it does not "
           "exist.")
      .def(
          "__iter__",
          [](PyOpAttributeMap &self) {
            nb::list keys;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](MlirStringRef name, MlirAttribute) {
                  keys.append(nb::str(name.data, name.length));
                });
            return nb::iter(keys);
          },
          "Iterates over attribute names.")
      .def(
          "keys",
          [](PyOpAttributeMap &self) {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](MlirStringRef name, MlirAttribute) {
                  out.append(nb::str(name.data, name.length));
                });
            return out;
          },
          "Returns a list of attribute names.")
      .def(
          "values",
          [](PyOpAttributeMap &self) {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(), [&](MlirStringRef, MlirAttribute attr) {
                  out.append(PyAttribute(self.operation->getContext(), attr)
                                 .maybeDownCast());
                });
            return out;
          },
          "Returns a list of attribute values.")
      .def(
          "items",
          [](PyOpAttributeMap &self) {
            nb::list out;
            PyOpAttributeMap::forEachAttr(
                self.operation->get(),
                [&](MlirStringRef name, MlirAttribute attr) {
                  out.append(nb::make_tuple(
                      nb::str(name.data, name.length),
                      PyAttribute(self.operation->getContext(), attr)
                          .maybeDownCast()));
                });
            return out;
          },
          "Returns a list of `(name, attribute)` tuples.");
}

} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

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

using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

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

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

void populateRoot(nb::module_ &m) {
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
            nb::cast<std::string>(pyClass.attr("DIALECT_NAMESPACE"));
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
                  nb::cast<std::string>(opClass.attr("OPERATION_NAME"));
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
      [](PyTypeID mlirTypeID, bool replace) -> nb::object {
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
      [](PyTypeID mlirTypeID, bool replace) -> nb::object {
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
}

//------------------------------------------------------------------------------
// Populates the core exports of the 'ir' submodule.
//------------------------------------------------------------------------------
void populateIRCore(nb::module_ &m) {
  //----------------------------------------------------------------------------
  // Enums.
  //----------------------------------------------------------------------------
  nb::enum_<PyDiagnosticSeverity>(m, "DiagnosticSeverity")
      .value("ERROR", PyDiagnosticSeverity::Error)
      .value("WARNING", PyDiagnosticSeverity::Warning)
      .value("NOTE", PyDiagnosticSeverity::Note)
      .value("REMARK", PyDiagnosticSeverity::Remark);

  nb::enum_<PyWalkOrder>(m, "WalkOrder")
      .value("PRE_ORDER", PyWalkOrder::PreOrder)
      .value("POST_ORDER", PyWalkOrder::PostOrder);
  nb::enum_<PyWalkResult>(m, "WalkResult")
      .value("ADVANCE", PyWalkResult::Advance)
      .value("INTERRUPT", PyWalkResult::Interrupt)
      .value("SKIP", PyWalkResult::Skip);

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
      .def("__exit__", &PyDiagnosticHandler::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
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
      .def("__exit__", &PyMlirContext::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
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
          "dialect_name"_a,
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
           "callback"_a,
           "Attaches a diagnostic handler that will receive callbacks.")
      .def(
          "enable_multithreading",
          [](PyMlirContext &self, bool enable) {
            mlirContextEnableMultithreading(self.get(), enable);
          },
          "enable"_a,
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
          "operation_name"_a,
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
          "registry"_a,
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
      .def(nb::init<nb::object>(), "descriptor"_a,
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
      .def("__exit__", &PyLocation::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
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
          "context"_a = nb::none(),
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
          "callee"_a, "frames"_a, "context"_a = nb::none(),
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
          "filename"_a, "line"_a, "col"_a, "context"_a = nb::none(),
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
          "filename"_a, "start_line"_a, "start_col"_a, "end_line"_a,
          "end_col"_a, "context"_a = nb::none(),
          "Gets a Location representing a file, line and column range.")
      .def("is_a_file", mlirLocationIsAFileLineColRange,
           "Returns True if this location is a FileLineColLoc.")
      .def_prop_ro(
          "filename",
          [](PyLocation loc) {
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
          "locations"_a, "metadata"_a = nb::none(), "context"_a = nb::none(),
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
          "name"_a, "childLoc"_a = nb::none(), "context"_a = nb::none(),
          "Gets a Location representing a named location with optional child "
          "location.")
      .def("is_a_name", mlirLocationIsAName,
           "Returns True if this location is a `NameLoc`.")
      .def_prop_ro(
          "name_str",
          [](PyLocation loc) {
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
          "attribute"_a, "context"_a = nb::none(),
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
          "message"_a,
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
          "asm"_a, "context"_a = nb::none(), kModuleParseDocstring)
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
          "asm"_a, "context"_a = nb::none(), kModuleParseDocstring)
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
          "path"_a, "context"_a = nb::none(), kModuleParseDocstring)
      .def_static(
          "create",
          [](const std::optional<PyLocation> &loc)
              -> nb::typed<nb::object, PyModule> {
            PyLocation pyLoc = maybeGetTracebackLocation(loc);
            MlirModule module = mlirModuleCreateEmpty(pyLoc.get());
            return PyModule::forModule(module).releaseObject();
          },
          "loc"_a = nb::none(), "Creates an empty module.")
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
           "state"_a, "file"_a = nb::none(), "binary"_a = false,
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
           "large_elements_limit"_a = nb::none(),
           "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
           "pretty_debug_info"_a = false, "print_generic_op_form"_a = false,
           "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
           "assume_verified"_a = false, "file"_a = nb::none(),
           "binary"_a = false, "skip_regions"_a = false,
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
      .def("write_bytecode", &PyOperationBase::writeBytecode, "file"_a,
           "desired_version"_a = nb::none(),
           R"(
             Write the bytecode form of the operation to a file like object.

             Args:
               file: The file like object to write to.
               desired_version: Optional version of bytecode to emit.
             Returns:
               The bytecode writer status.)")
      .def("get_asm", &PyOperationBase::getAsm,
           // Careful: Lots of arguments must match up with get_asm method.
           "binary"_a = false, "large_elements_limit"_a = nb::none(),
           "large_resource_limit"_a = nb::none(), "enable_debug_info"_a = false,
           "pretty_debug_info"_a = false, "print_generic_op_form"_a = false,
           "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
           "assume_verified"_a = false, "skip_regions"_a = false,
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
      .def("move_after", &PyOperationBase::moveAfter, "other"_a,
           "Puts self immediately after the other operation in its parent "
           "block.")
      .def("move_before", &PyOperationBase::moveBefore, "other"_a,
           "Puts self immediately before the other operation in its parent "
           "block.")
      .def("is_before_in_block", &PyOperationBase::isBeforeInBlock, "other"_a,
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
          "ip"_a = nb::none(),
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
      .def("walk", &PyOperationBase::walk, "callback"_a,
           "walk_order"_a = PyWalkOrder::PostOrder,
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
          "name"_a, "results"_a = nb::none(), "operands"_a = nb::none(),
          "attributes"_a = nb::none(), "successors"_a = nb::none(),
          "regions"_a = 0, "loc"_a = nb::none(), "ip"_a = nb::none(),
          "infer_type"_a = false,
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
          "source"_a, nb::kw_only(), "source_name"_a = "",
          "context"_a = nb::none(),
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
      .def(
          "replace_uses_of_with",
          [](PyOperation &self, PyValue &of, PyValue &with) {
            mlirOperationReplaceUsesOfWith(self.get(), of.get(), with.get());
          },
          "of"_a, "with_"_a,
          "Replaces uses of the 'of' value with the 'with' value inside the "
          "operation.")
      .def("_set_invalid", &PyOperation::setInvalid,
           "Invalidate the operation.");

  auto opViewClass =
      nb::class_<PyOpView, PyOperationBase>(m, "OpView")
          .def(nb::init<nb::typed<nb::object, PyOperation>>(), "operation"_a)
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
              "name"_a, "opRegionSpec"_a,
              "operandSegmentSpecObj"_a = nb::none(),
              "resultSegmentSpecObj"_a = nb::none(), "results"_a = nb::none(),
              "operands"_a = nb::none(), "attributes"_a = nb::none(),
              "successors"_a = nb::none(), "regions"_a = nb::none(),
              "loc"_a = nb::none(), "ip"_a = nb::none())
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
      "cls"_a, "results"_a = nb::none(), "operands"_a = nb::none(),
      "attributes"_a = nb::none(), "successors"_a = nb::none(),
      "regions"_a = nb::none(), "loc"_a = nb::none(), "ip"_a = nb::none(),
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
      "cls"_a, "source"_a, nb::kw_only(), "source_name"_a = "",
      "context"_a = nb::none(),
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
          "index"_a,
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
          "parent"_a, "arg_types"_a = nb::list(), "arg_locs"_a = std::nullopt,
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
          "region"_a,
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
          "arg_types"_a, nb::kw_only(), "arg_locs"_a = std::nullopt,
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
          "arg_types"_a, nb::kw_only(), "arg_locs"_a = std::nullopt,
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
          "operation"_a,
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
      .def(nb::init<PyBlock &>(), "block"_a,
           "Inserts after the last operation but still inside the block.")
      .def("__enter__", &PyInsertionPoint::contextEnter,
           "Enters the insertion point as a context manager.")
      .def("__exit__", &PyInsertionPoint::contextExit, "exc_type"_a.none(),
           "exc_value"_a.none(), "traceback"_a.none(),
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
      .def(nb::init<PyOperationBase &>(), "beforeOperation"_a,
           "Inserts before a referenced operation.")
      .def_static("at_block_begin", &PyInsertionPoint::atBlockBegin, "block"_a,
                  R"(
                    Creates an insertion point at the beginning of a block.

                    Args:
                      block: The block at whose beginning operations should be inserted.

                    Returns:
                      An InsertionPoint at the block's beginning.)")
      .def_static("at_block_terminator", &PyInsertionPoint::atBlockTerminator,
                  "block"_a,
                  R"(
                    Creates an insertion point before a block's terminator.

                    Args:
                      block: The block whose terminator to insert before.

                    Returns:
                      An InsertionPoint before the terminator.

                    Raises:
                      ValueError: If the block has no terminator.)")
      .def_static("after", &PyInsertionPoint::after, "operation"_a,
                  R"(
                    Creates an insertion point immediately after an operation.

                    Args:
                      operation: The operation after which to insert.

                    Returns:
                      An InsertionPoint after the operation.)")
      .def("insert", &PyInsertionPoint::insert, "operation"_a,
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
      .def(nb::init<PyAttribute &>(), "cast_from_type"_a,
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
          "asm"_a, "context"_a = nb::none(),
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
      .def(nb::init<PyType &>(), "cast_from_type"_a,
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
          "asm"_a, "context"_a = nb::none(),
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
          "other"_a.none(),
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
  m.attr("_T") = nb::type_var("_T", "bound"_a = m.attr("Type"));

  nb::class_<PyValue>(m, "Value", nb::is_generic(),
                      nb::sig("class Value(Generic[_T])"))
      .def(nb::init<PyValue &>(), nb::keep_alive<0, 1>(), "value"_a,
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
          [](PyValue &self)
              -> nb::typed<nb::object, std::variant<PyOpView, PyBlock>> {
            MlirValue v = self.get();
            if (mlirValueIsAOpResult(v)) {
              assert(mlirOperationEqual(self.getParentOperation()->get(),
                                        mlirOpResultGetOwner(self.get())) &&
                     "expected the owner of the value in Python to match "
                     "that in "
                     "the IR");
              return self.getParentOperation()->createOpView();
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
          "use_local_scope"_a = false, "use_name_loc_as_prefix"_a = false,
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
          "state"_a,
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
          "type"_a, "Sets the type of the value.",
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
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
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
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
      .def(
          "replace_all_uses_except",
          [](PyValue &self, PyValue &with, PyOperation &exception) {
            MlirOperation exceptedUser = exception.get();
            mlirValueReplaceAllUsesExcept(self, with, 1, &exceptedUser);
          },
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
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
          "with_"_a, "exceptions"_a, kValueReplaceAllUsesExceptDocstring)
      .def(
          MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
          [](PyValue &self) { return self.maybeDownCast(); },
          "Downcasts the `Value` to a more specific kind if possible.")
      .def_prop_ro(
          "location",
          [](PyValue self) {
            return PyLocation(
                PyMlirContext::forContext(mlirValueGetContext(self)),
                mlirValueGetLocation(self));
          },
          "Returns the source location of the value.");

  PyBlockArgument::bind(m);
  PyOpResult::bind(m);
  PyOpOperand::bind(m);

  nb::class_<PyAsmState>(m, "AsmState")
      .def(nb::init<PyValue &, bool>(), "value"_a, "use_local_scope"_a = false,
           R"(
             Creates an `AsmState` for consistent SSA value naming.

             Args:
               value: The value to create state for.
               use_local_scope: Whether to use local scope for naming.)")
      .def(nb::init<PyOperationBase &, bool>(), "op"_a,
           "use_local_scope"_a = false,
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
      .def("insert", &PySymbolTable::insert, "operation"_a,
           R"(
             Inserts a symbol operation into the symbol table.

             Args:
               operation: An operation with a symbol name to insert.

             Returns:
               The symbol name attribute of the inserted operation.

             Raises:
               ValueError: If the operation does not have a symbol name.)")
      .def("erase", &PySymbolTable::erase, "operation"_a,
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
      .def_static("set_symbol_name", &PySymbolTable::setSymbolName, "symbol"_a,
                  "name"_a, "Sets the symbol name for a symbol operation.")
      .def_static("get_symbol_name", &PySymbolTable::getSymbolName, "symbol"_a,
                  "Gets the symbol name from a symbol operation.")
      .def_static("get_visibility", &PySymbolTable::getVisibility, "symbol"_a,
                  "Gets the visibility attribute of a symbol operation.")
      .def_static("set_visibility", &PySymbolTable::setVisibility, "symbol"_a,
                  "visibility"_a,
                  "Sets the visibility attribute of a symbol operation.")
      .def_static("replace_all_symbol_uses",
                  &PySymbolTable::replaceAllSymbolUses, "old_symbol"_a,
                  "new_symbol"_a, "from_op"_a,
                  "Replaces all uses of a symbol with a new symbol name within "
                  "the given operation.")
      .def_static("walk_symbol_tables", &PySymbolTable::walkSymbolTables,
                  "from_op"_a, "all_sym_uses_visible"_a, "callback"_a,
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
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
