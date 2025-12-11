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
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
// clang-format on
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Debug.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "nanobind/typing.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <iostream>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace mlir::python;

using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir::python {
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

  if (mlirDiagnosticGetSeverity(diag) != MlirDiagnosticError)
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

void PyOperationBase::walk(
    std::function<MlirWalkResult(MlirOperation)> callback,
    MlirWalkOrder walkOrder) {
  PyOperation &operation = getOperation();
  operation.checkValid();
  struct UserData {
    std::function<MlirWalkResult(MlirOperation)> callback;
    bool gotException;
    std::string exceptionWhat;
    nb::object exceptionType;
  };
  UserData userData{callback, false, {}, {}};
  MlirOperationWalkCallback walkCallback = [](MlirOperation op,
                                              void *userData) {
    UserData *calleeUserData = static_cast<UserData *>(userData);
    try {
      return (calleeUserData->callback)(op);
    } catch (nb::python_error &e) {
      calleeUserData->gotException = true;
      calleeUserData->exceptionWhat = std::string(e.what());
      calleeUserData->exceptionType = nb::borrow(e.type());
      return MlirWalkResult::MlirWalkResultInterrupt;
    }
  };
  mlirOperationWalk(operation, walkCallback, &userData, walkOrder);
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
  if (!operation.ptr) {
    for (auto take : errors.take()) {
      std::cout << take.message << "\n";
    }
    throw MLIRError("Operation creation failed", errors.take());
  }
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

nb::object PyAttribute::maybeDownCast() {
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

nb::object PyType::maybeDownCast() {
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

nb::object PyValue::maybeDownCast() {
  MlirType type = mlirValueGetType(get());
  MlirTypeID mlirTypeID = mlirTypeGetTypeID(type);
  assert(!mlirTypeIDIsNull(mlirTypeID) &&
         "mlirTypeID was expected to be non-null.");
  std::optional<nb::callable> valueCaster =
      PyGlobals::get().lookupValueCaster(mlirTypeID, mlirTypeGetDialect(type));
  // nb::rv_policy::move means use std::move to move the return value
  // contents into a new instance that will be owned by Python.
  nb::object thisObj = nb::cast(this, nb::rv_policy::move);
  if (!valueCaster)
    return thisObj;
  return valueCaster.value()(thisObj);
}

PyValue PyValue::createFromCapsule(nb::object capsule) {
  MlirValue value = mlirPythonCapsuleToValue(capsule.ptr());
  if (mlirValueIsNull(value))
    throw nb::python_error();
  MlirOperation owner;
  if (mlirValueIsAOpResult(value))
    owner = mlirOpResultGetOwner(value);
  if (mlirValueIsABlockArgument(value))
    owner = mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  if (mlirOperationIsNull(owner))
    throw nb::python_error();
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

void registerMLIRErrorInIRCore() {
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
} // namespace mlir::python
