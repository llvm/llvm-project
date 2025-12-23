//===- IRCore.h - IR helpers of python bindings ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRCORE_H
#define MLIR_BINDINGS_PYTHON_IRCORE_H

#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include "Globals.h"
#include "NanobindUtils.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Debug.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ThreadPool.h"

namespace mlir {
namespace python {

class PyBlock;
class PyDiagnostic;
class PyDiagnosticHandler;
class PyInsertionPoint;
class PyLocation;
class DefaultingPyLocation;
class PyMlirContext;
class DefaultingPyMlirContext;
class PyModule;
class PyOperation;
class PyOperationBase;
class PyType;
class PySymbolTable;
class PyValue;

/// Template for a reference to a concrete type which captures a python
/// reference to its underlying python object.
template <typename T>
class MLIR_PYTHON_API_EXPORTED PyObjectRef {
public:
  PyObjectRef(T *referrent, nanobind::object object)
      : referrent(referrent), object(std::move(object)) {
    assert(this->referrent &&
           "cannot construct PyObjectRef with null referrent");
    assert(this->object && "cannot construct PyObjectRef with null object");
  }
  PyObjectRef(PyObjectRef &&other) noexcept
      : referrent(other.referrent), object(std::move(other.object)) {
    other.referrent = nullptr;
    assert(!other.object);
  }
  PyObjectRef(const PyObjectRef &other)
      : referrent(other.referrent), object(other.object /* copies */) {}
  ~PyObjectRef() = default;

  int getRefCount() {
    if (!object)
      return 0;
    return Py_REFCNT(object.ptr());
  }

  /// Releases the object held by this instance, returning it.
  /// This is the proper thing to return from a function that wants to return
  /// the reference. Note that this does not work from initializers.
  nanobind::object releaseObject() {
    assert(referrent && object);
    referrent = nullptr;
    auto stolen = std::move(object);
    return stolen;
  }

  T *get() { return referrent; }
  T *operator->() {
    assert(referrent && object);
    return referrent;
  }
  nanobind::object getObject() {
    assert(referrent && object);
    return object;
  }
  operator bool() const { return referrent && object; }

  using NBTypedT = nanobind::typed<nanobind::object, T>;

private:
  T *referrent;
  nanobind::object object;
};

/// Tracks an entry in the thread context stack. New entries are pushed onto
/// here for each with block that activates a new InsertionPoint, Context or
/// Location.
///
/// Pushing either a Location or InsertionPoint also pushes its associated
/// Context. Pushing a Context will not modify the Location or InsertionPoint
/// unless if they are from a different context, in which case, they are
/// cleared.
class MLIR_PYTHON_API_EXPORTED PyThreadContextEntry {
public:
  enum class FrameKind {
    Context,
    InsertionPoint,
    Location,
  };

  PyThreadContextEntry(FrameKind frameKind, nanobind::object context,
                       nanobind::object insertionPoint,
                       nanobind::object location)
      : context(std::move(context)), insertionPoint(std::move(insertionPoint)),
        location(std::move(location)), frameKind(frameKind) {}

  /// Gets the top of stack context and return nullptr if not defined.
  static PyMlirContext *getDefaultContext();

  /// Gets the top of stack insertion point and return nullptr if not defined.
  static PyInsertionPoint *getDefaultInsertionPoint();

  /// Gets the top of stack location and returns nullptr if not defined.
  static PyLocation *getDefaultLocation();

  PyMlirContext *getContext();
  PyInsertionPoint *getInsertionPoint();
  PyLocation *getLocation();
  FrameKind getFrameKind() { return frameKind; }

  /// Stack management.
  static PyThreadContextEntry *getTopOfStack();
  static nanobind::object pushContext(nanobind::object context);
  static void popContext(PyMlirContext &context);
  static nanobind::object pushInsertionPoint(nanobind::object insertionPoint);
  static void popInsertionPoint(PyInsertionPoint &insertionPoint);
  static nanobind::object pushLocation(nanobind::object location);
  static void popLocation(PyLocation &location);

  /// Gets the thread local stack.
  static std::vector<PyThreadContextEntry> &getStack();

private:
  static void push(FrameKind frameKind, nanobind::object context,
                   nanobind::object insertionPoint, nanobind::object location);

  /// An object reference to the PyContext.
  nanobind::object context;
  /// An object reference to the current insertion point.
  nanobind::object insertionPoint;
  /// An object reference to the current location.
  nanobind::object location;
  // The kind of push that was performed.
  FrameKind frameKind;
};

/// Wrapper around MlirLlvmThreadPool
/// Python object owns the C++ thread pool
class MLIR_PYTHON_API_EXPORTED PyThreadPool {
public:
  PyThreadPool() {
    ownedThreadPool = std::make_unique<llvm::DefaultThreadPool>();
  }
  PyThreadPool(const PyThreadPool &) = delete;
  PyThreadPool(PyThreadPool &&) = delete;

  int getMaxConcurrency() const { return ownedThreadPool->getMaxConcurrency(); }
  MlirLlvmThreadPool get() { return wrap(ownedThreadPool.get()); }

  std::string _mlir_thread_pool_ptr() const {
    std::stringstream ss;
    ss << ownedThreadPool.get();
    return ss.str();
  }

private:
  std::unique_ptr<llvm::ThreadPoolInterface> ownedThreadPool;
};

/// Wrapper around MlirContext.
using PyMlirContextRef = PyObjectRef<PyMlirContext>;
class MLIR_PYTHON_API_EXPORTED PyMlirContext {
public:
  PyMlirContext() = delete;
  PyMlirContext(MlirContext context);
  PyMlirContext(const PyMlirContext &) = delete;
  PyMlirContext(PyMlirContext &&) = delete;

  /// Returns a context reference for the singleton PyMlirContext wrapper for
  /// the given context.
  static PyMlirContextRef forContext(MlirContext context);
  ~PyMlirContext();

  /// Accesses the underlying MlirContext.
  MlirContext get() { return context; }

  /// Gets a strong reference to this context, which will ensure it is kept
  /// alive for the life of the reference.
  PyMlirContextRef getRef() {
    return PyMlirContextRef(this, nanobind::cast(this));
  }

  /// Gets a capsule wrapping the void* within the MlirContext.
  nanobind::object getCapsule();

  /// Creates a PyMlirContext from the MlirContext wrapped by a capsule.
  /// Note that PyMlirContext instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirContext
  /// is taken by calling this function.
  static nanobind::object createFromCapsule(nanobind::object capsule);

  /// Gets the count of live context objects. Used for testing.
  static size_t getLiveCount();

  /// Gets the count of live modules associated with this context.
  /// Used for testing.
  size_t getLiveModuleCount();

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object context);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  /// Attaches a Python callback as a diagnostic handler, returning a
  /// registration object (internally a PyDiagnosticHandler).
  nanobind::object attachDiagnosticHandler(nanobind::object callback);

  /// Controls whether error diagnostics should be propagated to diagnostic
  /// handlers, instead of being captured by `ErrorCapture`.
  void setEmitErrorDiagnostics(bool value) { emitErrorDiagnostics = value; }
  bool getEmitErrorDiagnostics() { return emitErrorDiagnostics; }
  struct ErrorCapture;

private:
  // Interns the mapping of live MlirContext::ptr to PyMlirContext instances,
  // preserving the relationship that an MlirContext maps to a single
  // PyMlirContext wrapper. This could be replaced in the future with an
  // extension mechanism on the MlirContext for stashing user pointers.
  // Note that this holds a handle, which does not imply ownership.
  // Mappings will be removed when the context is destructed.
  using LiveContextMap = llvm::DenseMap<void *, PyMlirContext *>;
  static nanobind::ft_mutex live_contexts_mutex;
  static LiveContextMap &getLiveContexts();

  // Interns all live modules associated with this context. Modules tracked
  // in this map are valid. When a module is invalidated, it is removed
  // from this map, and while it still exists as an instance, any
  // attempt to access it will raise an error.
  using LiveModuleMap =
      llvm::DenseMap<const void *, std::pair<nanobind::handle, PyModule *>>;
  LiveModuleMap liveModules;

  bool emitErrorDiagnostics = false;

  MlirContext context;
  friend class PyModule;
  friend class PyOperation;
};

/// Used in function arguments when None should resolve to the current context
/// manager set instance.
class MLIR_PYTHON_API_EXPORTED DefaultingPyMlirContext
    : public Defaulting<DefaultingPyMlirContext, PyMlirContext> {
public:
  using Defaulting::Defaulting;
  static constexpr const char kTypeDescription[] = "Context";
  static PyMlirContext &resolve();
};

/// Base class for all objects that directly or indirectly depend on an
/// MlirContext. The lifetime of the context will extend at least to the
/// lifetime of these instances.
/// Immutable objects that depend on a context extend this directly.
class MLIR_PYTHON_API_EXPORTED BaseContextObject {
public:
  BaseContextObject(PyMlirContextRef ref) : contextRef(std::move(ref)) {
    assert(this->contextRef &&
           "context object constructed with null context ref");
  }

  /// Accesses the context reference.
  PyMlirContextRef &getContext() { return contextRef; }

private:
  PyMlirContextRef contextRef;
};

/// Wrapper around an MlirLocation.
class MLIR_PYTHON_API_EXPORTED PyLocation : public BaseContextObject {
public:
  PyLocation(PyMlirContextRef contextRef, MlirLocation loc)
      : BaseContextObject(std::move(contextRef)), loc(loc) {}

  operator MlirLocation() const { return loc; }
  MlirLocation get() const { return loc; }

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object location);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  /// Gets a capsule wrapping the void* within the MlirLocation.
  nanobind::object getCapsule();

  /// Creates a PyLocation from the MlirLocation wrapped by a capsule.
  /// Note that PyLocation instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirLocation
  /// is taken by calling this function.
  static PyLocation createFromCapsule(nanobind::object capsule);

private:
  MlirLocation loc;
};

/// Python class mirroring the C MlirDiagnostic struct. Note that these structs
/// are only valid for the duration of a diagnostic callback and attempting
/// to access them outside of that will raise an exception. This applies to
/// nested diagnostics (in the notes) as well.
class MLIR_PYTHON_API_EXPORTED PyDiagnostic {
public:
  PyDiagnostic(MlirDiagnostic diagnostic) : diagnostic(diagnostic) {}
  void invalidate();
  bool isValid() { return valid; }
  MlirDiagnosticSeverity getSeverity();
  PyLocation getLocation();
  nanobind::str getMessage();
  nanobind::tuple getNotes();

  /// Materialized diagnostic information. This is safe to access outside the
  /// diagnostic callback.
  struct DiagnosticInfo {
    MlirDiagnosticSeverity severity;
    PyLocation location;
    std::string message;
    std::vector<DiagnosticInfo> notes;
  };
  DiagnosticInfo getInfo();

private:
  MlirDiagnostic diagnostic;

  void checkValid();
  /// If notes have been materialized from the diagnostic, then this will
  /// be populated with the corresponding objects (all castable to
  /// PyDiagnostic).
  std::optional<nanobind::tuple> materializedNotes;
  bool valid = true;
};

/// Represents a diagnostic handler attached to the context. The handler's
/// callback will be invoked with PyDiagnostic instances until the detach()
/// method is called or the context is destroyed. A diagnostic handler can be
/// the subject of a `with` block, which will detach it when the block exits.
///
/// Since diagnostic handlers can call back into Python code which can do
/// unsafe things (i.e. recursively emitting diagnostics, raising exceptions,
/// etc), this is generally not deemed to be a great user-level API. Users
/// should generally use some form of DiagnosticCollector. If the handler raises
/// any exceptions, they will just be emitted to stderr and dropped.
///
/// The unique usage of this class means that its lifetime management is
/// different from most other parts of the API. Instances are always created
/// in an attached state and can transition to a detached state by either:
///   a) The context being destroyed and unregistering all handlers.
///   b) An explicit call to detach().
/// The object may remain live from a Python perspective for an arbitrary time
/// after detachment, but there is nothing the user can do with it (since there
/// is no way to attach an existing handler object).
class MLIR_PYTHON_API_EXPORTED PyDiagnosticHandler {
public:
  PyDiagnosticHandler(MlirContext context, nanobind::object callback);
  ~PyDiagnosticHandler();

  bool isAttached() { return registeredID.has_value(); }
  bool getHadError() { return hadError; }

  /// Detaches the handler. Does nothing if not attached.
  void detach();

  nanobind::object contextEnter() { return nanobind::cast(this); }
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb) {
    detach();
  }

private:
  MlirContext context;
  nanobind::object callback;
  std::optional<MlirDiagnosticHandlerID> registeredID;
  bool hadError = false;
  friend class PyMlirContext;
};

/// RAII object that captures any error diagnostics emitted to the provided
/// context.
struct MLIR_PYTHON_API_EXPORTED PyMlirContext::ErrorCapture {
  ErrorCapture(PyMlirContextRef ctx)
      : ctx(ctx), handlerID(mlirContextAttachDiagnosticHandler(
                      ctx->get(), handler, /*userData=*/this,
                      /*deleteUserData=*/nullptr)) {}
  ~ErrorCapture() {
    mlirContextDetachDiagnosticHandler(ctx->get(), handlerID);
    assert(errors.empty() && "unhandled captured errors");
  }

  std::vector<PyDiagnostic::DiagnosticInfo> take() {
    return std::move(errors);
  };

private:
  PyMlirContextRef ctx;
  MlirDiagnosticHandlerID handlerID;
  std::vector<PyDiagnostic::DiagnosticInfo> errors;

  static MlirLogicalResult handler(MlirDiagnostic diag, void *userData);
};

/// Wrapper around an MlirDialect. This is exported as `DialectDescriptor` in
/// order to differentiate it from the `Dialect` base class which is extended by
/// plugins which extend dialect functionality through extension python code.
/// This should be seen as the "low-level" object and `Dialect` as the
/// high-level, user facing object.
class MLIR_PYTHON_API_EXPORTED PyDialectDescriptor : public BaseContextObject {
public:
  PyDialectDescriptor(PyMlirContextRef contextRef, MlirDialect dialect)
      : BaseContextObject(std::move(contextRef)), dialect(dialect) {}

  MlirDialect get() { return dialect; }

private:
  MlirDialect dialect;
};

/// User-level object for accessing dialects with dotted syntax such as:
///   ctx.dialect.std
class MLIR_PYTHON_API_EXPORTED PyDialects : public BaseContextObject {
public:
  PyDialects(PyMlirContextRef contextRef)
      : BaseContextObject(std::move(contextRef)) {}

  MlirDialect getDialectForKey(const std::string &key, bool attrError);
};

/// User-level dialect object. For dialects that have a registered extension,
/// this will be the base class of the extension dialect type. For un-extended,
/// objects of this type will be returned directly.
class MLIR_PYTHON_API_EXPORTED PyDialect {
public:
  PyDialect(nanobind::object descriptor) : descriptor(std::move(descriptor)) {}

  nanobind::object getDescriptor() { return descriptor; }

private:
  nanobind::object descriptor;
};

/// Wrapper around an MlirDialectRegistry.
/// Upon construction, the Python wrapper takes ownership of the
/// underlying MlirDialectRegistry.
class MLIR_PYTHON_API_EXPORTED PyDialectRegistry {
public:
  PyDialectRegistry() : registry(mlirDialectRegistryCreate()) {}
  PyDialectRegistry(MlirDialectRegistry registry) : registry(registry) {}
  ~PyDialectRegistry() {
    if (!mlirDialectRegistryIsNull(registry))
      mlirDialectRegistryDestroy(registry);
  }
  PyDialectRegistry(PyDialectRegistry &) = delete;
  PyDialectRegistry(PyDialectRegistry &&other) noexcept
      : registry(other.registry) {
    other.registry = {nullptr};
  }

  operator MlirDialectRegistry() const { return registry; }
  MlirDialectRegistry get() const { return registry; }

  nanobind::object getCapsule();
  static PyDialectRegistry createFromCapsule(nanobind::object capsule);

private:
  MlirDialectRegistry registry;
};

/// Used in function arguments when None should resolve to the current context
/// manager set instance.
class MLIR_PYTHON_API_EXPORTED DefaultingPyLocation
    : public Defaulting<DefaultingPyLocation, PyLocation> {
public:
  using Defaulting::Defaulting;
  static constexpr const char kTypeDescription[] = "Location";
  static PyLocation &resolve();

  operator MlirLocation() const { return *get(); }
};

/// Wrapper around MlirModule.
/// This is the top-level, user-owned object that contains regions/ops/blocks.
class PyModule;
using PyModuleRef = PyObjectRef<PyModule>;
class MLIR_PYTHON_API_EXPORTED PyModule : public BaseContextObject {
public:
  /// Returns a PyModule reference for the given MlirModule. This always returns
  /// a new object.
  static PyModuleRef forModule(MlirModule module);
  PyModule(PyModule &) = delete;
  PyModule(PyMlirContext &&) = delete;
  ~PyModule();

  /// Gets the backing MlirModule.
  MlirModule get() { return module; }

  /// Gets a strong reference to this module.
  PyModuleRef getRef() {
    return PyModuleRef(this, nanobind::borrow<nanobind::object>(handle));
  }

  /// Gets a capsule wrapping the void* within the MlirModule.
  /// Note that the module does not (yet) provide a corresponding factory for
  /// constructing from a capsule as that would require uniquing PyModule
  /// instances, which is not currently done.
  nanobind::object getCapsule();

  /// Creates a PyModule from the MlirModule wrapped by a capsule.
  /// Note this returns a new object BUT clearMlirModule() must be called to
  /// prevent double-frees (of the underlying mlir::Module).
  static nanobind::object createFromCapsule(nanobind::object capsule);

  void clearMlirModule() { module = {nullptr}; }

private:
  PyModule(PyMlirContextRef contextRef, MlirModule module);
  MlirModule module;
  nanobind::handle handle;
};

class PyAsmState;

/// Base class for PyOperation and PyOpView which exposes the primary, user
/// visible methods for manipulating it.
class MLIR_PYTHON_API_EXPORTED PyOperationBase {
public:
  virtual ~PyOperationBase() = default;
  /// Implements the bound 'print' method and helps with others.
  void print(std::optional<int64_t> largeElementsLimit,
             std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
             bool prettyDebugInfo, bool printGenericOpForm, bool useLocalScope,
             bool useNameLocAsPrefix, bool assumeVerified,
             nanobind::object fileObject, bool binary, bool skipRegions);
  void print(PyAsmState &state, nanobind::object fileObject, bool binary);

  nanobind::object
  getAsm(bool binary, std::optional<int64_t> largeElementsLimit,
         std::optional<int64_t> largeResourceLimit, bool enableDebugInfo,
         bool prettyDebugInfo, bool printGenericOpForm, bool useLocalScope,
         bool useNameLocAsPrefix, bool assumeVerified, bool skipRegions);

  // Implement the bound 'writeBytecode' method.
  void writeBytecode(const nanobind::object &fileObject,
                     std::optional<int64_t> bytecodeVersion);

  // Implement the walk method.
  void walk(std::function<MlirWalkResult(MlirOperation)> callback,
            MlirWalkOrder walkOrder);

  /// Moves the operation before or after the other operation.
  void moveAfter(PyOperationBase &other);
  void moveBefore(PyOperationBase &other);

  /// Given an operation 'other' that is within the same parent block, return
  /// whether the current operation is before 'other' in the operation list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of operations within the parent block.
  bool isBeforeInBlock(PyOperationBase &other);

  /// Verify the operation. Throws `MLIRError` if verification fails, and
  /// returns `true` otherwise.
  bool verify();

  /// Each must provide access to the raw Operation.
  virtual PyOperation &getOperation() = 0;
};

/// Wrapper around PyOperation.
/// Operations exist in either an attached (dependent) or detached (top-level)
/// state. In the detached state (as on creation), an operation is owned by
/// the creator and its lifetime extends either until its reference count
/// drops to zero or it is attached to a parent, at which point its lifetime
/// is bounded by its top-level parent reference.
class PyOperation;
class PyOpView;
using PyOperationRef = PyObjectRef<PyOperation>;
class MLIR_PYTHON_API_EXPORTED PyOperation : public PyOperationBase,
                                             public BaseContextObject {
public:
  ~PyOperation() override;
  PyOperation &getOperation() override { return *this; }

  /// Returns a PyOperation for the given MlirOperation, optionally associating
  /// it with a parentKeepAlive.
  static PyOperationRef
  forOperation(PyMlirContextRef contextRef, MlirOperation operation,
               nanobind::object parentKeepAlive = nanobind::object());

  /// Creates a detached operation. The operation must not be associated with
  /// any existing live operation.
  static PyOperationRef
  createDetached(PyMlirContextRef contextRef, MlirOperation operation,
                 nanobind::object parentKeepAlive = nanobind::object());

  /// Parses a source string (either text assembly or bytecode), creating a
  /// detached operation.
  static PyOperationRef parse(PyMlirContextRef contextRef,
                              const std::string &sourceStr,
                              const std::string &sourceName);

  /// Detaches the operation from its parent block and updates its state
  /// accordingly.
  void detachFromParent() {
    mlirOperationRemoveFromParent(getOperation());
    setDetached();
    parentKeepAlive = nanobind::object();
  }

  /// Gets the backing operation.
  operator MlirOperation() const { return get(); }
  MlirOperation get() const {
    checkValid();
    return operation;
  }

  PyOperationRef getRef() {
    return PyOperationRef(this, nanobind::borrow<nanobind::object>(handle));
  }

  bool isAttached() { return attached; }
  void setAttached(const nanobind::object &parent = nanobind::object()) {
    assert(!attached && "operation already attached");
    attached = true;
  }
  void setDetached() {
    assert(attached && "operation already detached");
    attached = false;
  }
  void checkValid() const;

  /// Gets the owning block or raises an exception if the operation has no
  /// owning block.
  PyBlock getBlock();

  /// Gets the parent operation or raises an exception if the operation has
  /// no parent.
  std::optional<PyOperationRef> getParentOperation();

  /// Gets a capsule wrapping the void* within the MlirOperation.
  nanobind::object getCapsule();

  /// Creates a PyOperation from the MlirOperation wrapped by a capsule.
  /// Ownership of the underlying MlirOperation is taken by calling this
  /// function.
  static nanobind::object createFromCapsule(const nanobind::object &capsule);

  /// Creates an operation. See corresponding python docstring.
  static nanobind::object
  create(std::string_view name, std::optional<std::vector<PyType *>> results,
         llvm::ArrayRef<MlirValue> operands,
         std::optional<nanobind::dict> attributes,
         std::optional<std::vector<PyBlock *>> successors, int regions,
         PyLocation &location, const nanobind::object &ip, bool inferType);

  /// Creates an OpView suitable for this operation.
  nanobind::object createOpView();

  /// Erases the underlying MlirOperation, removes its pointer from the
  /// parent context's live operations map, and sets the valid bit false.
  void erase();

  /// Invalidate the operation.
  void setInvalid() { valid = false; }

  /// Clones this operation.
  nanobind::object clone(const nanobind::object &ip);

  PyOperation(PyMlirContextRef contextRef, MlirOperation operation);

private:
  static PyOperationRef createInstance(PyMlirContextRef contextRef,
                                       MlirOperation operation,
                                       nanobind::object parentKeepAlive);

  MlirOperation operation;
  nanobind::handle handle;
  // Keeps the parent alive, regardless of whether it is an Operation or
  // Module.
  // TODO: As implemented, this facility is only sufficient for modeling the
  // trivial module parent back-reference. Generalize this to also account for
  // transitions from detached to attached and address TODOs in the
  // ir_operation.py regarding testing corresponding lifetime guarantees.
  nanobind::object parentKeepAlive;
  bool attached = true;
  bool valid = true;

  friend class PyOperationBase;
  friend class PySymbolTable;
};

/// A PyOpView is equivalent to the C++ "Op" wrappers: these are the basis for
/// providing more instance-specific accessors and serve as the base class for
/// custom ODS-style operation classes. Since this class is subclass on the
/// python side, it must present an __init__ method that operates in pure
/// python types.
class MLIR_PYTHON_API_EXPORTED PyOpView : public PyOperationBase {
public:
  PyOpView(const nanobind::object &operationObject);
  PyOperation &getOperation() override { return operation; }

  nanobind::object getOperationObject() { return operationObject; }

  static nanobind::object
  buildGeneric(std::string_view name, std::tuple<int, bool> opRegionSpec,
               nanobind::object operandSegmentSpecObj,
               nanobind::object resultSegmentSpecObj,
               std::optional<nanobind::list> resultTypeList,
               nanobind::list operandList,
               std::optional<nanobind::dict> attributes,
               std::optional<std::vector<PyBlock *>> successors,
               std::optional<int> regions, PyLocation &location,
               const nanobind::object &maybeIp);

  /// Construct an instance of a class deriving from OpView, bypassing its
  /// `__init__` method. The derived class will typically define a constructor
  /// that provides a convenient builder, but we need to side-step this when
  /// constructing an `OpView` for an already-built operation.
  ///
  /// The caller is responsible for verifying that `operation` is a valid
  /// operation to construct `cls` with.
  static nanobind::object constructDerived(const nanobind::object &cls,
                                           const nanobind::object &operation);

private:
  PyOperation &operation;           // For efficient, cast-free access from C++
  nanobind::object operationObject; // Holds the reference.
};

/// Wrapper around an MlirRegion.
/// Regions are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached regions.
class MLIR_PYTHON_API_EXPORTED PyRegion {
public:
  PyRegion(PyOperationRef parentOperation, MlirRegion region)
      : parentOperation(std::move(parentOperation)), region(region) {
    assert(!mlirRegionIsNull(region) && "python region cannot be null");
  }
  operator MlirRegion() const { return region; }

  MlirRegion get() { return region; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

private:
  PyOperationRef parentOperation;
  MlirRegion region;
};

/// Wrapper around an MlirAsmState.
class MLIR_PYTHON_API_EXPORTED PyAsmState {
public:
  PyAsmState(MlirValue value, bool useLocalScope) {
    flags = mlirOpPrintingFlagsCreate();
    // The OpPrintingFlags are not exposed Python side, create locally and
    // associate lifetime with the state.
    if (useLocalScope)
      mlirOpPrintingFlagsUseLocalScope(flags);
    state = mlirAsmStateCreateForValue(value, flags);
  }

  PyAsmState(PyOperationBase &operation, bool useLocalScope) {
    flags = mlirOpPrintingFlagsCreate();
    // The OpPrintingFlags are not exposed Python side, create locally and
    // associate lifetime with the state.
    if (useLocalScope)
      mlirOpPrintingFlagsUseLocalScope(flags);
    state =
        mlirAsmStateCreateForOperation(operation.getOperation().get(), flags);
  }
  ~PyAsmState() { mlirOpPrintingFlagsDestroy(flags); }
  // Delete copy constructors.
  PyAsmState(PyAsmState &other) = delete;
  PyAsmState(const PyAsmState &other) = delete;

  MlirAsmState get() { return state; }

private:
  MlirAsmState state;
  MlirOpPrintingFlags flags;
};

/// Wrapper around an MlirBlock.
/// Blocks are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached blocks.
class MLIR_PYTHON_API_EXPORTED PyBlock {
public:
  PyBlock(PyOperationRef parentOperation, MlirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {
    assert(!mlirBlockIsNull(block) && "python block cannot be null");
  }

  MlirBlock get() { return block; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

  /// Gets a capsule wrapping the void* within the MlirBlock.
  nanobind::object getCapsule();

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

/// An insertion point maintains a pointer to a Block and a reference operation.
/// Calls to insert() will insert a new operation before the
/// reference operation. If the reference operation is null, then appends to
/// the end of the block.
class MLIR_PYTHON_API_EXPORTED PyInsertionPoint {
public:
  /// Creates an insertion point positioned after the last operation in the
  /// block, but still inside the block.
  PyInsertionPoint(const PyBlock &block);
  /// Creates an insertion point positioned before a reference operation.
  PyInsertionPoint(PyOperationBase &beforeOperationBase);
  /// Creates an insertion point positioned before a reference operation.
  PyInsertionPoint(PyOperationRef beforeOperationRef);

  /// Shortcut to create an insertion point at the beginning of the block.
  static PyInsertionPoint atBlockBegin(PyBlock &block);
  /// Shortcut to create an insertion point before the block terminator.
  static PyInsertionPoint atBlockTerminator(PyBlock &block);
  /// Shortcut to create an insertion point to the node after the specified
  /// operation.
  static PyInsertionPoint after(PyOperationBase &op);

  /// Inserts an operation.
  void insert(PyOperationBase &operationBase);

  /// Enter and exit the context manager.
  static nanobind::object contextEnter(nanobind::object insertionPoint);
  void contextExit(const nanobind::object &excType,
                   const nanobind::object &excVal,
                   const nanobind::object &excTb);

  PyBlock &getBlock() { return block; }
  std::optional<PyOperationRef> &getRefOperation() { return refOperation; }

private:
  // Trampoline constructor that avoids null initializing members while
  // looking up parents.
  PyInsertionPoint(PyBlock block, std::optional<PyOperationRef> refOperation)
      : refOperation(std::move(refOperation)), block(std::move(block)) {}

  std::optional<PyOperationRef> refOperation;
  PyBlock block;
};
/// Wrapper around the generic MlirType.
/// The lifetime of a type is bound by the PyContext that created it.
class MLIR_PYTHON_API_EXPORTED PyType : public BaseContextObject {
public:
  PyType(PyMlirContextRef contextRef, MlirType type)
      : BaseContextObject(std::move(contextRef)), type(type) {}
  bool operator==(const PyType &other) const;
  operator MlirType() const { return type; }
  MlirType get() const { return type; }

  /// Gets a capsule wrapping the void* within the MlirType.
  nanobind::object getCapsule();

  /// Creates a PyType from the MlirType wrapped by a capsule.
  /// Note that PyType instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirType
  /// is taken by calling this function.
  static PyType createFromCapsule(nanobind::object capsule);

  nanobind::object maybeDownCast();

private:
  MlirType type;
};

/// A TypeID provides an efficient and unique identifier for a specific C++
/// type. This allows for a C++ type to be compared, hashed, and stored in an
/// opaque context. This class wraps around the generic MlirTypeID.
class MLIR_PYTHON_API_EXPORTED PyTypeID {
public:
  PyTypeID(MlirTypeID typeID) : typeID(typeID) {}
  // Note, this tests whether the underlying TypeIDs are the same,
  // not whether the wrapper MlirTypeIDs are the same, nor whether
  // the PyTypeID objects are the same (i.e., PyTypeID is a value type).
  bool operator==(const PyTypeID &other) const;
  operator MlirTypeID() const { return typeID; }
  MlirTypeID get() { return typeID; }

  /// Gets a capsule wrapping the void* within the MlirTypeID.
  nanobind::object getCapsule();

  /// Creates a PyTypeID from the MlirTypeID wrapped by a capsule.
  static PyTypeID createFromCapsule(nanobind::object capsule);

private:
  MlirTypeID typeID;
};

/// CRTP base classes for Python types that subclass Type and should be
/// castable from it (i.e. via something like IntegerType(t)).
/// By default, type class hierarchies are one level deep (i.e. a
/// concrete type class extends PyType); however, intermediate python-visible
/// base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyType>
class MLIR_PYTHON_API_EXPORTED PyConcreteType : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = nanobind::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirType);
  using GetTypeIDFunctionTy = MlirTypeID (*)();
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = nullptr;

  PyConcreteType() = default;
  PyConcreteType(PyMlirContextRef contextRef, MlirType t)
      : BaseTy(std::move(contextRef), t) {}
  PyConcreteType(PyType &orig)
      : PyConcreteType(orig.getContext(), castFrom(orig)) {}

  static MlirType castFrom(PyType &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((llvm::Twine("Cannot cast type to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .str()
                                      .c_str());
    }
    return orig;
  }

  static void bind(nanobind::module_ &m) {
    auto cls = ClassTy(m, DerivedTy::pyClassName);
    cls.def(nanobind::init<PyType &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("cast_from_type"));
    cls.def_static(
        "isinstance",
        [](PyType &otherType) -> bool {
          return DerivedTy::isaFunction(otherType);
        },
        nanobind::arg("other"));
    cls.def_prop_ro_static(
        "static_typeid",
        [](nanobind::object & /*class*/) {
          if (DerivedTy::getTypeIdFunction)
            return PyTypeID(DerivedTy::getTypeIdFunction());
          throw nanobind::attribute_error(
              (DerivedTy::pyClassName + llvm::Twine(" has no typeid."))
                  .str()
                  .c_str());
        },
        nanobind::sig("def static_typeid(/) -> TypeID"));
    cls.def_prop_ro("typeid", [](PyType &self) {
      return nanobind::cast<PyTypeID>(nanobind::cast(self).attr("typeid"));
    });
    cls.def("__repr__", [](DerivedTy &self) {
      PyPrintAccumulator printAccum;
      printAccum.parts.append(DerivedTy::pyClassName);
      printAccum.parts.append("(");
      mlirTypePrint(self, printAccum.getCallback(), printAccum.getUserData());
      printAccum.parts.append(")");
      return printAccum.join();
    });

    if (DerivedTy::getTypeIdFunction) {
      PyGlobals::get().registerTypeCaster(
          DerivedTy::getTypeIdFunction(),
          nanobind::cast<nanobind::callable>(nanobind::cpp_function(
              [](PyType pyType) -> DerivedTy { return pyType; })));
    }

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Wrapper around the generic MlirAttribute.
/// The lifetime of a type is bound by the PyContext that created it.
class MLIR_PYTHON_API_EXPORTED PyAttribute : public BaseContextObject {
public:
  PyAttribute(PyMlirContextRef contextRef, MlirAttribute attr)
      : BaseContextObject(std::move(contextRef)), attr(attr) {}
  bool operator==(const PyAttribute &other) const;
  operator MlirAttribute() const { return attr; }
  MlirAttribute get() const { return attr; }

  /// Gets a capsule wrapping the void* within the MlirAttribute.
  nanobind::object getCapsule();

  /// Creates a PyAttribute from the MlirAttribute wrapped by a capsule.
  /// Note that PyAttribute instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirAttribute
  /// is taken by calling this function.
  static PyAttribute createFromCapsule(const nanobind::object &capsule);

  nanobind::object maybeDownCast();

private:
  MlirAttribute attr;
};

/// Represents a Python MlirNamedAttr, carrying an optional owned name.
/// TODO: Refactor this and the C-API to be based on an Identifier owned
/// by the context so as to avoid ownership issues here.
class MLIR_PYTHON_API_EXPORTED PyNamedAttribute {
public:
  /// Constructs a PyNamedAttr that retains an owned name. This should be
  /// used in any code that originates an MlirNamedAttribute from a python
  /// string.
  /// The lifetime of the PyNamedAttr must extend to the lifetime of the
  /// passed attribute.
  PyNamedAttribute(MlirAttribute attr, std::string ownedName);

  MlirNamedAttribute namedAttr;

private:
  // Since the MlirNamedAttr contains an internal pointer to the actual
  // memory of the owned string, it must be heap allocated to remain valid.
  // Otherwise, strings that fit within the small object optimization threshold
  // will have their memory address change as the containing object is moved,
  // resulting in an invalid aliased pointer.
  std::unique_ptr<std::string> ownedName;
};

/// CRTP base classes for Python attributes that subclass Attribute and should
/// be castable from it (i.e. via something like StringAttr(attr)).
/// By default, attribute class hierarchies are one level deep (i.e. a
/// concrete attribute class extends PyAttribute); however, intermediate
/// python-visible base classes can be modeled by specifying a BaseTy.
template <typename DerivedTy, typename BaseTy = PyAttribute>
class MLIR_PYTHON_API_EXPORTED PyConcreteAttribute : public BaseTy {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  using ClassTy = nanobind::class_<DerivedTy, BaseTy>;
  using IsAFunctionTy = bool (*)(MlirAttribute);
  using GetTypeIDFunctionTy = MlirTypeID (*)();
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = nullptr;

  PyConcreteAttribute() = default;
  PyConcreteAttribute(PyMlirContextRef contextRef, MlirAttribute attr)
      : BaseTy(std::move(contextRef), attr) {}
  PyConcreteAttribute(PyAttribute &orig)
      : PyConcreteAttribute(orig.getContext(), castFrom(orig)) {}

  static MlirAttribute castFrom(PyAttribute &orig) {
    if (!DerivedTy::isaFunction(orig)) {
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((llvm::Twine("Cannot cast attribute to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .str()
                                      .c_str());
    }
    return orig;
  }

  static void bind(nanobind::module_ &m, PyType_Slot *slots = nullptr) {
    ClassTy cls;
    if (slots) {
      cls = ClassTy(m, DerivedTy::pyClassName, nanobind::type_slots(slots));
    } else {
      cls = ClassTy(m, DerivedTy::pyClassName);
    }
    cls.def(nanobind::init<PyAttribute &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("cast_from_attr"));
    cls.def_static(
        "isinstance",
        [](PyAttribute &otherAttr) -> bool {
          return DerivedTy::isaFunction(otherAttr);
        },
        nanobind::arg("other"));
    cls.def_prop_ro(
        "type",
        [](PyAttribute &attr) -> nanobind::typed<nanobind::object, PyType> {
          return PyType(attr.getContext(), mlirAttributeGetType(attr))
              .maybeDownCast();
        });
    cls.def_prop_ro_static(
        "static_typeid",
        [](nanobind::object & /*class*/) -> PyTypeID {
          if (DerivedTy::getTypeIdFunction)
            return PyTypeID(DerivedTy::getTypeIdFunction());
          throw nanobind::attribute_error(
              (DerivedTy::pyClassName + llvm::Twine(" has no typeid."))
                  .str()
                  .c_str());
        },
        nanobind::sig("def static_typeid(/) -> TypeID"));
    cls.def_prop_ro("typeid", [](PyAttribute &self) {
      return nanobind::cast<PyTypeID>(nanobind::cast(self).attr("typeid"));
    });
    cls.def("__repr__", [](DerivedTy &self) {
      PyPrintAccumulator printAccum;
      printAccum.parts.append(DerivedTy::pyClassName);
      printAccum.parts.append("(");
      mlirAttributePrint(self, printAccum.getCallback(),
                         printAccum.getUserData());
      printAccum.parts.append(")");
      return printAccum.join();
    });

    if (DerivedTy::getTypeIdFunction) {
      PyGlobals::get().registerTypeCaster(
          DerivedTy::getTypeIdFunction(),
          nanobind::cast<nanobind::callable>(
              nanobind::cpp_function([](PyAttribute pyAttribute) -> DerivedTy {
                return pyAttribute;
              })));
    }

    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

class MLIR_PYTHON_API_EXPORTED PyStringAttribute
    : public PyConcreteAttribute<PyStringAttribute> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirAttributeIsAString;
  static constexpr const char *pyClassName = "StringAttr";
  using PyConcreteAttribute::PyConcreteAttribute;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirStringAttrGetTypeID;

  static void bindDerived(ClassTy &c);
};

/// Wrapper around the generic MlirValue.
/// Values are managed completely by the operation that resulted in their
/// definition. For op result value, this is the operation that defines the
/// value. For block argument values, this is the operation that contains the
/// block to which the value is an argument (blocks cannot be detached in Python
/// bindings so such operation always exists).
class MLIR_PYTHON_API_EXPORTED PyValue {
public:
  // The virtual here is "load bearing" in that it enables RTTI
  // for PyConcreteValue CRTP classes that support maybeDownCast.
  // See PyValue::maybeDownCast.
  virtual ~PyValue() = default;
  PyValue(PyOperationRef parentOperation, MlirValue value)
      : parentOperation(std::move(parentOperation)), value(value) {}
  operator MlirValue() const { return value; }

  MlirValue get() { return value; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

  /// Gets a capsule wrapping the void* within the MlirValue.
  nanobind::object getCapsule();

  nanobind::object maybeDownCast();

  /// Creates a PyValue from the MlirValue wrapped by a capsule. Ownership of
  /// the underlying MlirValue is still tied to the owning operation.
  static PyValue createFromCapsule(nanobind::object capsule);

private:
  PyOperationRef parentOperation;
  MlirValue value;
};

/// Wrapper around MlirAffineExpr. Affine expressions are owned by the context.
class MLIR_PYTHON_API_EXPORTED PyAffineExpr : public BaseContextObject {
public:
  PyAffineExpr(PyMlirContextRef contextRef, MlirAffineExpr affineExpr)
      : BaseContextObject(std::move(contextRef)), affineExpr(affineExpr) {}
  bool operator==(const PyAffineExpr &other) const;
  operator MlirAffineExpr() const { return affineExpr; }
  MlirAffineExpr get() const { return affineExpr; }

  /// Gets a capsule wrapping the void* within the MlirAffineExpr.
  nanobind::object getCapsule();

  /// Creates a PyAffineExpr from the MlirAffineExpr wrapped by a capsule.
  /// Note that PyAffineExpr instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirAffineExpr
  /// is taken by calling this function.
  static PyAffineExpr createFromCapsule(const nanobind::object &capsule);

  PyAffineExpr add(const PyAffineExpr &other) const;
  PyAffineExpr mul(const PyAffineExpr &other) const;
  PyAffineExpr floorDiv(const PyAffineExpr &other) const;
  PyAffineExpr ceilDiv(const PyAffineExpr &other) const;
  PyAffineExpr mod(const PyAffineExpr &other) const;

private:
  MlirAffineExpr affineExpr;
};

class MLIR_PYTHON_API_EXPORTED PyAffineMap : public BaseContextObject {
public:
  PyAffineMap(PyMlirContextRef contextRef, MlirAffineMap affineMap)
      : BaseContextObject(std::move(contextRef)), affineMap(affineMap) {}
  bool operator==(const PyAffineMap &other) const;
  operator MlirAffineMap() const { return affineMap; }
  MlirAffineMap get() const { return affineMap; }

  /// Gets a capsule wrapping the void* within the MlirAffineMap.
  nanobind::object getCapsule();

  /// Creates a PyAffineMap from the MlirAffineMap wrapped by a capsule.
  /// Note that PyAffineMap instances are uniqued, so the returned object
  /// may be a pre-existing object. Ownership of the underlying MlirAffineMap
  /// is taken by calling this function.
  static PyAffineMap createFromCapsule(const nanobind::object &capsule);

private:
  MlirAffineMap affineMap;
};

class MLIR_PYTHON_API_EXPORTED PyIntegerSet : public BaseContextObject {
public:
  PyIntegerSet(PyMlirContextRef contextRef, MlirIntegerSet integerSet)
      : BaseContextObject(std::move(contextRef)), integerSet(integerSet) {}
  bool operator==(const PyIntegerSet &other) const;
  operator MlirIntegerSet() const { return integerSet; }
  MlirIntegerSet get() const { return integerSet; }

  /// Gets a capsule wrapping the void* within the MlirIntegerSet.
  nanobind::object getCapsule();

  /// Creates a PyIntegerSet from the MlirAffineMap wrapped by a capsule.
  /// Note that PyIntegerSet instances may be uniqued, so the returned object
  /// may be a pre-existing object. Integer sets are owned by the context.
  static PyIntegerSet createFromCapsule(const nanobind::object &capsule);

private:
  MlirIntegerSet integerSet;
};

/// Bindings for MLIR symbol tables.
class MLIR_PYTHON_API_EXPORTED PySymbolTable {
public:
  /// Constructs a symbol table for the given operation.
  explicit PySymbolTable(PyOperationBase &operation);

  /// Destroys the symbol table.
  ~PySymbolTable() { mlirSymbolTableDestroy(symbolTable); }

  /// Returns the symbol (opview) with the given name, throws if there is no
  /// such symbol in the table.
  nanobind::object dunderGetItem(const std::string &name);

  /// Removes the given operation from the symbol table and erases it.
  void erase(PyOperationBase &symbol);

  /// Removes the operation with the given name from the symbol table and erases
  /// it, throws if there is no such symbol in the table.
  void dunderDel(const std::string &name);

  /// Inserts the given operation into the symbol table. The operation must have
  /// the symbol trait.
  PyStringAttribute insert(PyOperationBase &symbol);

  /// Gets and sets the name of a symbol op.
  static PyStringAttribute getSymbolName(PyOperationBase &symbol);
  static void setSymbolName(PyOperationBase &symbol, const std::string &name);

  /// Gets and sets the visibility of a symbol op.
  static PyStringAttribute getVisibility(PyOperationBase &symbol);
  static void setVisibility(PyOperationBase &symbol,
                            const std::string &visibility);

  /// Replaces all symbol uses within an operation. See the API
  /// mlirSymbolTableReplaceAllSymbolUses for all caveats.
  static void replaceAllSymbolUses(const std::string &oldSymbol,
                                   const std::string &newSymbol,
                                   PyOperationBase &from);

  /// Walks all symbol tables under and including 'from'.
  static void walkSymbolTables(PyOperationBase &from, bool allSymUsesVisible,
                               nanobind::object callback);

  /// Casts the bindings class into the C API structure.
  operator MlirSymbolTable() { return symbolTable; }

private:
  PyOperationRef operation;
  MlirSymbolTable symbolTable;
};

/// Custom exception that allows access to error diagnostic information. This is
/// converted to the `ir.MLIRError` python exception when thrown.
struct MLIR_PYTHON_API_EXPORTED MLIRError {
  MLIRError(llvm::Twine message,
            std::vector<PyDiagnostic::DiagnosticInfo> &&errorDiagnostics = {})
      : message(message.str()), errorDiagnostics(std::move(errorDiagnostics)) {}
  std::string message;
  std::vector<PyDiagnostic::DiagnosticInfo> errorDiagnostics;
};

inline void registerMLIRError() {
  nanobind::register_exception_translator(
      [](const std::exception_ptr &p, void *payload) {
        // We can't define exceptions with custom fields through pybind, so
        // instead the exception class is defined in python and imported here.
        try {
          if (p)
            std::rethrow_exception(p);
        } catch (const MLIRError &e) {
          nanobind::object obj =
              nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                  .attr("MLIRError")(e.message, e.errorDiagnostics);
          PyErr_SetObject(PyExc_Exception, obj.ptr());
        }
      });
}

MLIR_PYTHON_API_EXPORTED void registerMLIRErrorInCore();

//------------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------------

/// Helper for creating an @classmethod.
template <class Func, typename... Args>
nanobind::object classmethod(Func f, Args... args) {
  nanobind::object cf = nanobind::cpp_function(f, args...);
  return nanobind::borrow<nanobind::object>((PyClassMethod_New(cf.ptr())));
}

inline nanobind::object
createCustomDialectWrapper(const std::string &dialectNamespace,
                           nanobind::object dialectDescriptor) {
  auto dialectClass = PyGlobals::get().lookupDialectClass(dialectNamespace);
  if (!dialectClass) {
    // Use the base class.
    return nanobind::cast(PyDialect(std::move(dialectDescriptor)));
  }

  // Create the custom implementation.
  return (*dialectClass)(std::move(dialectDescriptor));
}

inline MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

inline MlirStringRef toMlirStringRef(std::string_view s) {
  return mlirStringRefCreate(s.data(), s.size());
}

inline MlirStringRef toMlirStringRef(const nanobind::bytes &s) {
  return mlirStringRefCreate(static_cast<const char *>(s.data()), s.size());
}

/// Create a block, using the current location context if no locations are
/// specified.
inline MlirBlock
createBlock(const nanobind::sequence &pyArgTypes,
            const std::optional<nanobind::sequence> &pyArgLocs) {
  SmallVector<MlirType> argTypes;
  argTypes.reserve(nanobind::len(pyArgTypes));
  for (const auto &pyType : pyArgTypes)
    argTypes.push_back(nanobind::cast<PyType &>(pyType));

  SmallVector<MlirLocation> argLocs;
  if (pyArgLocs) {
    argLocs.reserve(nanobind::len(*pyArgLocs));
    for (const auto &pyLoc : *pyArgLocs)
      argLocs.push_back(nanobind::cast<PyLocation &>(pyLoc));
  } else if (!argTypes.empty()) {
    argLocs.assign(argTypes.size(), DefaultingPyLocation::resolve());
  }

  if (argTypes.size() != argLocs.size())
    throw nanobind::value_error(("Expected " + Twine(argTypes.size()) +
                                 " locations, got: " + Twine(argLocs.size()))
                                    .str()
                                    .c_str());
  return mlirBlockCreate(argTypes.size(), argTypes.data(), argLocs.data());
}

struct PyAttrBuilderMap {
  static bool dunderContains(const std::string &attributeKind) {
    return PyGlobals::get().lookupAttributeBuilder(attributeKind).has_value();
  }
  static nanobind::callable
  dunderGetItemNamed(const std::string &attributeKind) {
    auto builder = PyGlobals::get().lookupAttributeBuilder(attributeKind);
    if (!builder)
      throw nanobind::key_error(attributeKind.c_str());
    return *builder;
  }
  static void dunderSetItemNamed(const std::string &attributeKind,
                                 nanobind::callable func, bool replace) {
    PyGlobals::get().registerAttributeBuilder(attributeKind, std::move(func),
                                              replace);
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyAttrBuilderMap>(m, "AttrBuilder")
        .def_static("contains", &PyAttrBuilderMap::dunderContains,
                    nanobind::arg("attribute_kind"),
                    "Checks whether an attribute builder is registered for the "
                    "given attribute kind.")
        .def_static("get", &PyAttrBuilderMap::dunderGetItemNamed,
                    nanobind::arg("attribute_kind"),
                    "Gets the registered attribute builder for the given "
                    "attribute kind.")
        .def_static("insert", &PyAttrBuilderMap::dunderSetItemNamed,
                    nanobind::arg("attribute_kind"),
                    nanobind::arg("attr_builder"),
                    nanobind::arg("replace") = false,
                    "Register an attribute builder for building MLIR "
                    "attributes from Python values.");
  }
};

//------------------------------------------------------------------------------
// PyBlock
//------------------------------------------------------------------------------

inline nanobind::object PyBlock::getCapsule() {
  return nanobind::steal<nanobind::object>(mlirPythonBlockToCapsule(get()));
}

//------------------------------------------------------------------------------
// Collections.
//------------------------------------------------------------------------------

class MLIR_PYTHON_API_EXPORTED PyRegionIterator {
public:
  PyRegionIterator(PyOperationRef operation, int nextIndex)
      : operation(std::move(operation)), nextIndex(nextIndex) {}

  PyRegionIterator &dunderIter() { return *this; }

  PyRegion dunderNext() {
    operation->checkValid();
    if (nextIndex >= mlirOperationGetNumRegions(operation->get())) {
      throw nanobind::stop_iteration();
    }
    MlirRegion region = mlirOperationGetRegion(operation->get(), nextIndex++);
    return PyRegion(operation, region);
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyRegionIterator>(m, "RegionIterator")
        .def("__iter__", &PyRegionIterator::dunderIter,
             "Returns an iterator over the regions in the operation.")
        .def("__next__", &PyRegionIterator::dunderNext,
             "Returns the next region in the iteration.");
  }

private:
  PyOperationRef operation;
  intptr_t nextIndex = 0;
};

/// Regions of an op are fixed length and indexed numerically so are represented
/// with a sequence-like container.
class MLIR_PYTHON_API_EXPORTED PyRegionList
    : public Sliceable<PyRegionList, PyRegion> {
public:
  static constexpr const char *pyClassName = "RegionSequence";

  PyRegionList(PyOperationRef operation, intptr_t startIndex = 0,
               intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumRegions(operation->get())
                               : length,
                  step),
        operation(std::move(operation)) {}

  PyRegionIterator dunderIter() {
    operation->checkValid();
    return PyRegionIterator(operation, startIndex);
  }

  static void bindDerived(ClassTy &c) {
    c.def("__iter__", &PyRegionList::dunderIter,
          "Returns an iterator over the regions in the sequence.");
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyRegionList, PyRegion>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumRegions(operation->get());
  }

  PyRegion getRawElement(intptr_t pos) {
    operation->checkValid();
    return PyRegion(operation, mlirOperationGetRegion(operation->get(), pos));
  }

  PyRegionList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyRegionList(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

class MLIR_PYTHON_API_EXPORTED PyBlockIterator {
public:
  PyBlockIterator(PyOperationRef operation, MlirBlock next)
      : operation(std::move(operation)), next(next) {}

  PyBlockIterator &dunderIter() { return *this; }

  PyBlock dunderNext() {
    operation->checkValid();
    if (mlirBlockIsNull(next)) {
      throw nanobind::stop_iteration();
    }

    PyBlock returnBlock(operation, next);
    next = mlirBlockGetNextInRegion(next);
    return returnBlock;
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyBlockIterator>(m, "BlockIterator")
        .def("__iter__", &PyBlockIterator::dunderIter,
             "Returns an iterator over the blocks in the operation's region.")
        .def("__next__", &PyBlockIterator::dunderNext,
             "Returns the next block in the iteration.");
  }

private:
  PyOperationRef operation;
  MlirBlock next;
};

/// Blocks are exposed by the C-API as a forward-only linked list. In Python,
/// we present them as a more full-featured list-like container but optimize
/// it for forward iteration. Blocks are always owned by a region.
class MLIR_PYTHON_API_EXPORTED PyBlockList {
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
      index += dunderLen();
    }
    if (index < 0) {
      throw nanobind::index_error("attempt to access out of bounds block");
    }
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (!mlirBlockIsNull(block)) {
      if (index == 0) {
        return PyBlock(operation, block);
      }
      block = mlirBlockGetNextInRegion(block);
      index -= 1;
    }
    throw nanobind::index_error("attempt to access out of bounds block");
  }

  PyBlock appendBlock(const nanobind::args &pyArgTypes,
                      const std::optional<nanobind::sequence> &pyArgLocs) {
    operation->checkValid();
    MlirBlock block =
        createBlock(nanobind::cast<nanobind::sequence>(pyArgTypes), pyArgLocs);
    mlirRegionAppendOwnedBlock(region, block);
    return PyBlock(operation, block);
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyBlockList>(m, "BlockList")
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
             nanobind::arg("args"), nanobind::kw_only(),
             nanobind::arg("arg_locs") = std::nullopt);
  }

private:
  PyOperationRef operation;
  MlirRegion region;
};

class MLIR_PYTHON_API_EXPORTED PyOperationIterator {
public:
  PyOperationIterator(PyOperationRef parentOperation, MlirOperation next)
      : parentOperation(std::move(parentOperation)), next(next) {}

  PyOperationIterator &dunderIter() { return *this; }

  nanobind::typed<nanobind::object, PyOpView> dunderNext() {
    parentOperation->checkValid();
    if (mlirOperationIsNull(next)) {
      throw nanobind::stop_iteration();
    }

    PyOperationRef returnOperation =
        PyOperation::forOperation(parentOperation->getContext(), next);
    next = mlirOperationGetNextInBlock(next);
    return returnOperation->createOpView();
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyOperationIterator>(m, "OperationIterator")
        .def("__iter__", &PyOperationIterator::dunderIter,
             "Returns an iterator over the operations in an operation's block.")
        .def("__next__", &PyOperationIterator::dunderNext,
             "Returns the next operation in the iteration.");
  }

private:
  PyOperationRef parentOperation;
  MlirOperation next;
};

/// Operations are exposed by the C-API as a forward-only linked list. In
/// Python, we present them as a more full-featured list-like container but
/// optimize it for forward iteration. Iterable operations are always owned
/// by a block.
class MLIR_PYTHON_API_EXPORTED PyOperationList {
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

  nanobind::typed<nanobind::object, PyOpView> dunderGetItem(intptr_t index) {
    parentOperation->checkValid();
    if (index < 0) {
      index += dunderLen();
    }
    if (index < 0) {
      throw nanobind::index_error("attempt to access out of bounds operation");
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
    throw nanobind::index_error("attempt to access out of bounds operation");
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyOperationList>(m, "OperationList")
        .def("__getitem__", &PyOperationList::dunderGetItem,
             "Returns the operation at the specified index.")
        .def("__iter__", &PyOperationList::dunderIter,
             "Returns an iterator over operations in the list.")
        .def("__len__", &PyOperationList::dunderLen,
             "Returns the number of operations in the list.");
  }

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

class MLIR_PYTHON_API_EXPORTED PyOpOperand {
public:
  PyOpOperand(MlirOpOperand opOperand) : opOperand(opOperand) {}

  nanobind::typed<nanobind::object, PyOpView> getOwner() {
    MlirOperation owner = mlirOpOperandGetOwner(opOperand);
    PyMlirContextRef context =
        PyMlirContext::forContext(mlirOperationGetContext(owner));
    return PyOperation::forOperation(context, owner)->createOpView();
  }

  size_t getOperandNumber() { return mlirOpOperandGetOperandNumber(opOperand); }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyOpOperand>(m, "OpOperand")
        .def_prop_ro("owner", &PyOpOperand::getOwner,
                     "Returns the operation that owns this operand.")
        .def_prop_ro("operand_number", &PyOpOperand::getOperandNumber,
                     "Returns the operand number in the owning operation.");
  }

private:
  MlirOpOperand opOperand;
};

class MLIR_PYTHON_API_EXPORTED PyOpOperandIterator {
public:
  PyOpOperandIterator(MlirOpOperand opOperand) : opOperand(opOperand) {}

  PyOpOperandIterator &dunderIter() { return *this; }

  PyOpOperand dunderNext() {
    if (mlirOpOperandIsNull(opOperand))
      throw nanobind::stop_iteration();

    PyOpOperand returnOpOperand(opOperand);
    opOperand = mlirOpOperandGetNextUse(opOperand);
    return returnOpOperand;
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyOpOperandIterator>(m, "OpOperandIterator")
        .def("__iter__", &PyOpOperandIterator::dunderIter,
             "Returns an iterator over operands.")
        .def("__next__", &PyOpOperandIterator::dunderNext,
             "Returns the next operand in the iteration.");
  }

private:
  MlirOpOperand opOperand;
};

/// CRTP base class for Python MLIR values that subclass Value and should be
/// castable from it. The value hierarchy is one level deep and is not supposed
/// to accommodate other levels unless core MLIR changes.
template <typename DerivedTy>
class MLIR_PYTHON_API_EXPORTED PyConcreteValue : public PyValue {
public:
  // Derived classes must define statics for:
  //   IsAFunctionTy isaFunction
  //   const char *pyClassName
  // and redefine bindDerived.
  using ClassTy = nanobind::class_<DerivedTy, PyValue>;
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
      auto origRepr =
          nanobind::cast<std::string>(nanobind::repr(nanobind::cast(orig)));
      throw nanobind::value_error((Twine("Cannot cast value to ") +
                                   DerivedTy::pyClassName + " (from " +
                                   origRepr + ")")
                                      .str()
                                      .c_str());
    }
    return orig.get();
  }

  /// Binds the Python module objects to functions of this class.
  static void bind(nanobind::module_ &m) {
    auto cls = ClassTy(
        m, DerivedTy::pyClassName, nanobind::is_generic(),
        nanobind::sig((Twine("class ") + DerivedTy::pyClassName + "(Value[_T])")
                          .str()
                          .c_str()));
    cls.def(nanobind::init<PyValue &>(), nanobind::keep_alive<0, 1>(),
            nanobind::arg("value"));
    cls.def_static(
        "isinstance",
        [](PyValue &otherValue) -> bool {
          return DerivedTy::isaFunction(otherValue);
        },
        nanobind::arg("other_value"));
    cls.def(
        MLIR_PYTHON_MAYBE_DOWNCAST_ATTR,
        [](DerivedTy &self) -> nanobind::typed<nanobind::object, DerivedTy> {
          return self.maybeDownCast();
        });
    DerivedTy::bindDerived(cls);
  }

  /// Implemented by derived classes to add methods to the Python subclass.
  static void bindDerived(ClassTy &m) {}
};

/// Python wrapper for MlirOpResult.
class MLIR_PYTHON_API_EXPORTED PyOpResult : public PyConcreteValue<PyOpResult> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsAOpResult;
  static constexpr const char *pyClassName = "OpResult";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro(
        "owner",
        [](PyOpResult &self) -> nanobind::typed<nanobind::object, PyOpView> {
          assert(mlirOperationEqual(self.getParentOperation()->get(),
                                    mlirOpResultGetOwner(self.get())) &&
                 "expected the owner of the value in Python to match that in "
                 "the IR");
          return self.getParentOperation()->createOpView();
        },
        "Returns the operation that produces this result.");
    c.def_prop_ro(
        "result_number",
        [](PyOpResult &self) {
          return mlirOpResultGetResultNumber(self.get());
        },
        "Returns the position of this result in the operation's result list.");
  }
};

/// Returns the list of types of the values held by container.
template <typename Container>
std::vector<nanobind::typed<nanobind::object, PyType>>
getValueTypes(Container &container, PyMlirContextRef &context) {
  std::vector<nanobind::typed<nanobind::object, PyType>> result;
  result.reserve(container.size());
  for (int i = 0, e = container.size(); i < e; ++i) {
    result.push_back(PyType(context->getRef(),
                            mlirValueGetType(container.getElement(i).get()))
                         .maybeDownCast());
  }
  return result;
}

/// A list of operation results. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) result list is associated
/// with the operation whose results these are, and thus extends the lifetime of
/// this operation.
class MLIR_PYTHON_API_EXPORTED PyOpResultList
    : public Sliceable<PyOpResultList, PyOpResult> {
public:
  static constexpr const char *pyClassName = "OpResultList";
  using SliceableT = Sliceable<PyOpResultList, PyOpResult>;

  PyOpResultList(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumResults(operation->get())
                               : length,
                  step),
        operation(std::move(operation)) {}

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro(
        "types",
        [](PyOpResultList &self) {
          return getValueTypes(self, self.operation->getContext());
        },
        "Returns a list of types for all results in this result list.");
    c.def_prop_ro(
        "owner",
        [](PyOpResultList &self)
            -> nanobind::typed<nanobind::object, PyOpView> {
          return self.operation->createOpView();
        },
        "Returns the operation that owns this result list.");
  }

  PyOperationRef &getOperation() { return operation; }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpResultList, PyOpResult>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumResults(operation->get());
  }

  PyOpResult getRawElement(intptr_t index) {
    PyValue value(operation, mlirOperationGetResult(operation->get(), index));
    return PyOpResult(value);
  }

  PyOpResultList slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpResultList(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

/// Python wrapper for MlirBlockArgument.
class MLIR_PYTHON_API_EXPORTED PyBlockArgument
    : public PyConcreteValue<PyBlockArgument> {
public:
  static constexpr IsAFunctionTy isaFunction = mlirValueIsABlockArgument;
  static constexpr const char *pyClassName = "BlockArgument";
  using PyConcreteValue::PyConcreteValue;

  static void bindDerived(ClassTy &c) {
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
        nanobind::arg("type"), "Sets the type of this block argument.");
    c.def(
        "set_location",
        [](PyBlockArgument &self, PyLocation loc) {
          return mlirBlockArgumentSetLocation(self.get(), loc);
        },
        nanobind::arg("loc"), "Sets the location of this block argument.");
  }
};

/// A list of block arguments. Internally, these are stored as consecutive
/// elements, random access is cheap. The argument list is associated with the
/// operation that contains the block (detached blocks are not allowed in
/// Python bindings) and extends its lifetime.
class MLIR_PYTHON_API_EXPORTED PyBlockArgumentList
    : public Sliceable<PyBlockArgumentList, PyBlockArgument> {
public:
  static constexpr const char *pyClassName = "BlockArgumentList";
  using SliceableT = Sliceable<PyBlockArgumentList, PyBlockArgument>;

  PyBlockArgumentList(PyOperationRef operation, MlirBlock block,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirBlockGetNumArguments(block) : length,
                  step),
        operation(std::move(operation)), block(block) {}

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro(
        "types",
        [](PyBlockArgumentList &self) {
          return getValueTypes(self, self.operation->getContext());
        },
        "Returns a list of types for all arguments in this argument list.");
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockArgumentList, PyBlockArgument>;

  /// Returns the number of arguments in the list.
  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirBlockGetNumArguments(block);
  }

  /// Returns `pos`-the element in the list.
  PyBlockArgument getRawElement(intptr_t pos) {
    MlirValue argument = mlirBlockGetArgument(block, pos);
    return PyBlockArgument(operation, argument);
  }

  /// Returns a sublist of this list.
  PyBlockArgumentList slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyBlockArgumentList(operation, block, startIndex, length, step);
  }

  PyOperationRef operation;
  MlirBlock block;
};

/// A list of operation operands. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) operand list is associated
/// with the operation whose operands these are, and thus extends the lifetime
/// of this operation.
class MLIR_PYTHON_API_EXPORTED PyOpOperandList
    : public Sliceable<PyOpOperandList, PyValue> {
public:
  static constexpr const char *pyClassName = "OpOperandList";
  using SliceableT = Sliceable<PyOpOperandList, PyValue>;

  PyOpOperandList(PyOperationRef operation, intptr_t startIndex = 0,
                  intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumOperands(operation->get())
                               : length,
                  step),
        operation(operation) {}

  void dunderSetItem(intptr_t index, PyValue value) {
    index = wrapIndex(index);
    mlirOperationSetOperand(operation->get(), index, value.get());
  }

  static void bindDerived(ClassTy &c) {
    c.def("__setitem__", &PyOpOperandList::dunderSetItem,
          nanobind::arg("index"), nanobind::arg("value"),
          "Sets the operand at the specified index to a new value.");
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpOperandList, PyValue>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumOperands(operation->get());
  }

  PyValue getRawElement(intptr_t pos) {
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

  PyOperationRef operation;
};

/// A list of operation successors. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) successor list is
/// associated with the operation whose successors these are, and thus extends
/// the lifetime of this operation.
class MLIR_PYTHON_API_EXPORTED PyOpSuccessors
    : public Sliceable<PyOpSuccessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "OpSuccessors";

  PyOpSuccessors(PyOperationRef operation, intptr_t startIndex = 0,
                 intptr_t length = -1, intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirOperationGetNumSuccessors(operation->get())
                               : length,
                  step),
        operation(operation) {}

  void dunderSetItem(intptr_t index, PyBlock block) {
    index = wrapIndex(index);
    mlirOperationSetSuccessor(operation->get(), index, block.get());
  }

  static void bindDerived(ClassTy &c) {
    c.def("__setitem__", &PyOpSuccessors::dunderSetItem, nanobind::arg("index"),
          nanobind::arg("block"),
          "Sets the successor block at the specified index.");
  }

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyOpSuccessors, PyBlock>;

  intptr_t getRawNumElements() {
    operation->checkValid();
    return mlirOperationGetNumSuccessors(operation->get());
  }

  PyBlock getRawElement(intptr_t pos) {
    MlirBlock block = mlirOperationGetSuccessor(operation->get(), pos);
    return PyBlock(operation, block);
  }

  PyOpSuccessors slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyOpSuccessors(operation, startIndex, length, step);
  }

  PyOperationRef operation;
};

/// A list of block successors. Internally, these are stored as consecutive
/// elements, random access is cheap. The (returned) successor list is
/// associated with the operation and block whose successors these are, and thus
/// extends the lifetime of this operation and block.
class MLIR_PYTHON_API_EXPORTED PyBlockSuccessors
    : public Sliceable<PyBlockSuccessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "BlockSuccessors";

  PyBlockSuccessors(PyBlock block, PyOperationRef operation,
                    intptr_t startIndex = 0, intptr_t length = -1,
                    intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirBlockGetNumSuccessors(block.get())
                               : length,
                  step),
        operation(operation), block(block) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockSuccessors, PyBlock>;

  intptr_t getRawNumElements() {
    block.checkValid();
    return mlirBlockGetNumSuccessors(block.get());
  }

  PyBlock getRawElement(intptr_t pos) {
    MlirBlock block = mlirBlockGetSuccessor(this->block.get(), pos);
    return PyBlock(operation, block);
  }

  PyBlockSuccessors slice(intptr_t startIndex, intptr_t length, intptr_t step) {
    return PyBlockSuccessors(block, operation, startIndex, length, step);
  }

  PyOperationRef operation;
  PyBlock block;
};

/// A list of block predecessors. The (returned) predecessor list is
/// associated with the operation and block whose predecessors these are, and
/// thus extends the lifetime of this operation and block.
///
/// WARNING: This Sliceable is more expensive than the others here because
/// mlirBlockGetPredecessor actually iterates the use-def chain (of block
/// operands) anew for each indexed access.
class MLIR_PYTHON_API_EXPORTED PyBlockPredecessors
    : public Sliceable<PyBlockPredecessors, PyBlock> {
public:
  static constexpr const char *pyClassName = "BlockPredecessors";

  PyBlockPredecessors(PyBlock block, PyOperationRef operation,
                      intptr_t startIndex = 0, intptr_t length = -1,
                      intptr_t step = 1)
      : Sliceable(startIndex,
                  length == -1 ? mlirBlockGetNumPredecessors(block.get())
                               : length,
                  step),
        operation(operation), block(block) {}

private:
  /// Give the parent CRTP class access to hook implementations below.
  friend class Sliceable<PyBlockPredecessors, PyBlock>;

  intptr_t getRawNumElements() {
    block.checkValid();
    return mlirBlockGetNumPredecessors(block.get());
  }

  PyBlock getRawElement(intptr_t pos) {
    MlirBlock block = mlirBlockGetPredecessor(this->block.get(), pos);
    return PyBlock(operation, block);
  }

  PyBlockPredecessors slice(intptr_t startIndex, intptr_t length,
                            intptr_t step) {
    return PyBlockPredecessors(block, operation, startIndex, length, step);
  }

  PyOperationRef operation;
  PyBlock block;
};

/// A list of operation attributes. Can be indexed by name, producing
/// attributes, or by index, producing named attributes.
class MLIR_PYTHON_API_EXPORTED PyOpAttributeMap {
public:
  PyOpAttributeMap(PyOperationRef operation)
      : operation(std::move(operation)) {}

  nanobind::typed<nanobind::object, PyAttribute>
  dunderGetItemNamed(const std::string &name) {
    MlirAttribute attr = mlirOperationGetAttributeByName(operation->get(),
                                                         toMlirStringRef(name));
    if (mlirAttributeIsNull(attr)) {
      throw nanobind::key_error("attempt to access a non-existent attribute");
    }
    return PyAttribute(operation->getContext(), attr).maybeDownCast();
  }

  PyNamedAttribute dunderGetItemIndexed(intptr_t index) {
    if (index < 0) {
      index += dunderLen();
    }
    if (index < 0 || index >= dunderLen()) {
      throw nanobind::index_error("attempt to access out of bounds attribute");
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
      throw nanobind::key_error("attempt to delete a non-existent attribute");
  }

  intptr_t dunderLen() {
    return mlirOperationGetNumAttributes(operation->get());
  }

  bool dunderContains(const std::string &name) {
    return !mlirAttributeIsNull(mlirOperationGetAttributeByName(
        operation->get(), toMlirStringRef(name)));
  }

  static void
  forEachAttr(MlirOperation op,
              llvm::function_ref<void(MlirStringRef, MlirAttribute)> fn) {
    intptr_t n = mlirOperationGetNumAttributes(op);
    for (intptr_t i = 0; i < n; ++i) {
      MlirNamedAttribute na = mlirOperationGetAttribute(op, i);
      MlirStringRef name = mlirIdentifierStr(na.name);
      fn(name, na.attribute);
    }
  }

  static void bind(nanobind::module_ &m) {
    nanobind::class_<PyOpAttributeMap>(m, "OpAttributeMap")
        .def("__contains__", &PyOpAttributeMap::dunderContains,
             nanobind::arg("name"),
             "Checks if an attribute with the given name exists in the map.")
        .def("__len__", &PyOpAttributeMap::dunderLen,
             "Returns the number of attributes in the map.")
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemNamed,
             nanobind::arg("name"), "Gets an attribute by name.")
        .def("__getitem__", &PyOpAttributeMap::dunderGetItemIndexed,
             nanobind::arg("index"), "Gets a named attribute by index.")
        .def("__setitem__", &PyOpAttributeMap::dunderSetItem,
             nanobind::arg("name"), nanobind::arg("attr"),
             "Sets an attribute with the given name.")
        .def("__delitem__", &PyOpAttributeMap::dunderDelItem,
             nanobind::arg("name"), "Deletes an attribute with the given name.")
        .def(
            "__iter__",
            [](PyOpAttributeMap &self) {
              nanobind::list keys;
              PyOpAttributeMap::forEachAttr(
                  self.operation->get(),
                  [&](MlirStringRef name, MlirAttribute) {
                    keys.append(nanobind::str(name.data, name.length));
                  });
              return nanobind::iter(keys);
            },
            "Iterates over attribute names.")
        .def(
            "keys",
            [](PyOpAttributeMap &self) {
              nanobind::list out;
              PyOpAttributeMap::forEachAttr(
                  self.operation->get(),
                  [&](MlirStringRef name, MlirAttribute) {
                    out.append(nanobind::str(name.data, name.length));
                  });
              return out;
            },
            "Returns a list of attribute names.")
        .def(
            "values",
            [](PyOpAttributeMap &self) {
              nanobind::list out;
              PyOpAttributeMap::forEachAttr(
                  self.operation->get(),
                  [&](MlirStringRef, MlirAttribute attr) {
                    out.append(PyAttribute(self.operation->getContext(), attr)
                                   .maybeDownCast());
                  });
              return out;
            },
            "Returns a list of attribute values.")
        .def(
            "items",
            [](PyOpAttributeMap &self) {
              nanobind::list out;
              PyOpAttributeMap::forEachAttr(
                  self.operation->get(),
                  [&](MlirStringRef name, MlirAttribute attr) {
                    out.append(nanobind::make_tuple(
                        nanobind::str(name.data, name.length),
                        PyAttribute(self.operation->getContext(), attr)
                            .maybeDownCast()));
                  });
              return out;
            },
            "Returns a list of `(name, attribute)` tuples.");
  }

private:
  PyOperationRef operation;
};

MLIR_PYTHON_API_EXPORTED MlirValue getUniqueResult(MlirOperation operation);
} // namespace python
} // namespace mlir

namespace nanobind {
namespace detail {

template <>
struct type_caster<mlir::python::DefaultingPyMlirContext>
    : MlirDefaultingCaster<mlir::python::DefaultingPyMlirContext> {};
template <>
struct type_caster<mlir::python::DefaultingPyLocation>
    : MlirDefaultingCaster<mlir::python::DefaultingPyLocation> {};

} // namespace detail
} // namespace nanobind

#endif // MLIR_BINDINGS_PYTHON_IRCORE_H
