//===- IRModules.h - IR Submodules of pybind module -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRMODULES_H
#define MLIR_BINDINGS_PYTHON_IRMODULES_H

#include <pybind11/pybind11.h>

#include "mlir-c/IR.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace python {

class PyBlock;
class PyLocation;
class PyMlirContext;
class PyModule;
class PyOperation;
class PyType;

/// Template for a reference to a concrete type which captures a python
/// reference to its underlying python object.
template <typename T>
class PyObjectRef {
public:
  PyObjectRef(T *referrent, pybind11::object object)
      : referrent(referrent), object(std::move(object)) {
    assert(this->referrent &&
           "cannot construct PyObjectRef with null referrent");
    assert(this->object && "cannot construct PyObjectRef with null object");
  }
  PyObjectRef(PyObjectRef &&other)
      : referrent(other.referrent), object(std::move(other.object)) {
    other.referrent = nullptr;
    assert(!other.object);
  }
  PyObjectRef(const PyObjectRef &other)
      : referrent(other.referrent), object(other.object /* copies */) {}
  ~PyObjectRef() {}

  int getRefCount() {
    if (!object)
      return 0;
    return object.ref_count();
  }

  /// Releases the object held by this instance, returning it.
  /// This is the proper thing to return from a function that wants to return
  /// the reference. Note that this does not work from initializers.
  pybind11::object releaseObject() {
    assert(referrent && object);
    referrent = nullptr;
    auto stolen = std::move(object);
    return stolen;
  }

  T *operator->() {
    assert(referrent && object);
    return referrent;
  }
  pybind11::object getObject() {
    assert(referrent && object);
    return object;
  }
  operator bool() const { return referrent && object; }

private:
  T *referrent;
  pybind11::object object;
};

using PyMlirContextRef = PyObjectRef<PyMlirContext>;

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() = delete;
  PyMlirContext(const PyMlirContext &) = delete;
  PyMlirContext(PyMlirContext &&) = delete;

  /// For the case of a python __init__ (py::init) method, pybind11 is quite
  /// strict about needing to return a pointer that is not yet associated to
  /// an py::object. Since the forContext() method acts like a pool, possibly
  /// returning a recycled context, it does not satisfy this need. The usual
  /// way in python to accomplish such a thing is to override __new__, but
  /// that is also not supported by pybind11. Instead, we use this entry
  /// point which always constructs a fresh context (which cannot alias an
  /// existing one because it is fresh).
  static PyMlirContext *createNewContextForInit();

  /// Returns a context reference for the singleton PyMlirContext wrapper for
  /// the given context.
  static PyMlirContextRef forContext(MlirContext context);
  ~PyMlirContext();

  /// Accesses the underlying MlirContext.
  MlirContext get() { return context; }

  /// Gets a strong reference to this context, which will ensure it is kept
  /// alive for the life of the reference.
  PyMlirContextRef getRef() {
    return PyMlirContextRef(this, pybind11::cast(this));
  }

  /// Gets the count of live context objects. Used for testing.
  static size_t getLiveCount();

  /// Gets the count of live operations associated with this context.
  /// Used for testing.
  size_t getLiveOperationCount();

  /// Creates an operation. See corresponding python docstring.
  pybind11::object
  createOperation(std::string name, PyLocation location,
                  llvm::Optional<std::vector<PyType *>> results,
                  llvm::Optional<pybind11::dict> attributes,
                  llvm::Optional<std::vector<PyBlock *>> successors,
                  int regions);

private:
  PyMlirContext(MlirContext context);
  // Interns the mapping of live MlirContext::ptr to PyMlirContext instances,
  // preserving the relationship that an MlirContext maps to a single
  // PyMlirContext wrapper. This could be replaced in the future with an
  // extension mechanism on the MlirContext for stashing user pointers.
  // Note that this holds a handle, which does not imply ownership.
  // Mappings will be removed when the context is destructed.
  using LiveContextMap = llvm::DenseMap<void *, PyMlirContext *>;
  static LiveContextMap &getLiveContexts();

  // Interns all live operations associated with this context. Operations
  // tracked in this map are valid. When an operation is invalidated, it is
  // removed from this map, and while it still exists as an instance, any
  // attempt to access it will raise an error.
  using LiveOperationMap =
      llvm::DenseMap<void *, std::pair<pybind11::handle, PyOperation *>>;
  LiveOperationMap liveOperations;

  MlirContext context;
  friend class PyOperation;
};

/// Base class for all objects that directly or indirectly depend on an
/// MlirContext. The lifetime of the context will extend at least to the
/// lifetime of these instances.
/// Immutable objects that depend on a context extend this directly.
class BaseContextObject {
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
class PyLocation : public BaseContextObject {
public:
  PyLocation(PyMlirContextRef contextRef, MlirLocation loc)
      : BaseContextObject(std::move(contextRef)), loc(loc) {}
  MlirLocation loc;
};

/// Wrapper around MlirModule.
/// This is the top-level, user-owned object that contains regions/ops/blocks.
class PyModule;
using PyModuleRef = PyObjectRef<PyModule>;
class PyModule : public BaseContextObject {
public:
  /// Creates a reference to the module
  static PyModuleRef create(PyMlirContextRef contextRef, MlirModule module);
  PyModule(PyModule &) = delete;
  ~PyModule() {
    if (module.ptr)
      mlirModuleDestroy(module);
  }

  /// Gets the backing MlirModule.
  MlirModule get() { return module; }

  /// Gets a strong reference to this module.
  PyModuleRef getRef() {
    return PyModuleRef(this,
                       pybind11::reinterpret_borrow<pybind11::object>(handle));
  }

private:
  PyModule(PyMlirContextRef contextRef, MlirModule module)
      : BaseContextObject(std::move(contextRef)), module(module) {}
  MlirModule module;
  pybind11::handle handle;
};

/// Wrapper around PyOperation.
/// Operations exist in either an attached (dependent) or detached (top-level)
/// state. In the detached state (as on creation), an operation is owned by
/// the creator and its lifetime extends either until its reference count
/// drops to zero or it is attached to a parent, at which point its lifetime
/// is bounded by its top-level parent reference.
class PyOperation;
using PyOperationRef = PyObjectRef<PyOperation>;
class PyOperation : public BaseContextObject {
public:
  ~PyOperation();
  /// Returns a PyOperation for the given MlirOperation, optionally associating
  /// it with a parentKeepAlive (which must match on all such calls for the
  /// same operation).
  static PyOperationRef
  forOperation(PyMlirContextRef contextRef, MlirOperation operation,
               pybind11::object parentKeepAlive = pybind11::object());

  /// Creates a detached operation. The operation must not be associated with
  /// any existing live operation.
  static PyOperationRef
  createDetached(PyMlirContextRef contextRef, MlirOperation operation,
                 pybind11::object parentKeepAlive = pybind11::object());

  /// Gets the backing operation.
  MlirOperation get() {
    checkValid();
    return operation;
  }

  PyOperationRef getRef() {
    return PyOperationRef(
        this, pybind11::reinterpret_borrow<pybind11::object>(handle));
  }

  bool isAttached() { return attached; }
  void setAttached() {
    assert(!attached && "operation already attached");
    attached = true;
  }
  void checkValid();

private:
  PyOperation(PyMlirContextRef contextRef, MlirOperation operation);
  static PyOperationRef createInstance(PyMlirContextRef contextRef,
                                       MlirOperation operation,
                                       pybind11::object parentKeepAlive);

  MlirOperation operation;
  pybind11::handle handle;
  // Keeps the parent alive, regardless of whether it is an Operation or
  // Module.
  // TODO: As implemented, this facility is only sufficient for modeling the
  // trivial module parent back-reference. Generalize this to also account for
  // transitions from detached to attached and address TODOs in the
  // ir_operation.py regarding testing corresponding lifetime guarantees.
  pybind11::object parentKeepAlive;
  bool attached = true;
  bool valid = true;
};

/// Wrapper around an MlirRegion.
/// Regions are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached regions.
class PyRegion {
public:
  PyRegion(PyOperationRef parentOperation, MlirRegion region)
      : parentOperation(std::move(parentOperation)), region(region) {
    assert(!mlirRegionIsNull(region) && "python region cannot be null");
  }

  MlirRegion get() { return region; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

private:
  PyOperationRef parentOperation;
  MlirRegion region;
};

/// Wrapper around an MlirBlock.
/// Blocks are managed completely by their containing operation. Unlike the
/// C++ API, the python API does not support detached blocks.
class PyBlock {
public:
  PyBlock(PyOperationRef parentOperation, MlirBlock block)
      : parentOperation(std::move(parentOperation)), block(block) {
    assert(!mlirBlockIsNull(block) && "python block cannot be null");
  }

  MlirBlock get() { return block; }
  PyOperationRef &getParentOperation() { return parentOperation; }

  void checkValid() { return parentOperation->checkValid(); }

private:
  PyOperationRef parentOperation;
  MlirBlock block;
};

/// Wrapper around the generic MlirAttribute.
/// The lifetime of a type is bound by the PyContext that created it.
class PyAttribute : public BaseContextObject {
public:
  PyAttribute(PyMlirContextRef contextRef, MlirAttribute attr)
      : BaseContextObject(std::move(contextRef)), attr(attr) {}
  bool operator==(const PyAttribute &other);

  MlirAttribute attr;
};

/// Represents a Python MlirNamedAttr, carrying an optional owned name.
/// TODO: Refactor this and the C-API to be based on an Identifier owned
/// by the context so as to avoid ownership issues here.
class PyNamedAttribute {
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

/// Wrapper around the generic MlirType.
/// The lifetime of a type is bound by the PyContext that created it.
class PyType : public BaseContextObject {
public:
  PyType(PyMlirContextRef contextRef, MlirType type)
      : BaseContextObject(std::move(contextRef)), type(type) {}
  bool operator==(const PyType &other);
  operator MlirType() const { return type; }

  MlirType type;
};

void populateIRSubmodule(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRMODULES_H
