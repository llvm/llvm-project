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

class PyMlirContext;
class PyModule;

/// Holds a C++ PyMlirContext and associated py::object, making it convenient
/// to have an auto-releasing C++-side keep-alive reference to the context.
/// The reference to the PyMlirContext is a simple C++ reference and the
/// py::object holds the reference count which keeps it alive.
class PyMlirContextRef {
public:
  PyMlirContextRef(PyMlirContext &referrent, pybind11::object object)
      : referrent(referrent), object(std::move(object)) {}
  ~PyMlirContextRef() {}

  /// Releases the object held by this instance, causing its reference count
  /// to remain artifically inflated by one. This must be used to return
  /// the referenced PyMlirContext from a function. Otherwise, the destructor
  /// of this reference would be called prior to the default take_ownership
  /// policy assuming that the reference count has been transferred to it.
  PyMlirContext *release();

  PyMlirContext &operator->() { return referrent; }
  pybind11::object getObject() { return object; }

private:
  PyMlirContext &referrent;
  pybind11::object object;
};

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() = delete;
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
    return PyMlirContextRef(
        *this, pybind11::reinterpret_borrow<pybind11::object>(handle));
  }

  /// Gets the count of live context objects. Used for testing.
  static size_t getLiveCount();

private:
  PyMlirContext(MlirContext context);

  // Interns the mapping of live MlirContext::ptr to PyMlirContext instances,
  // preserving the relationship that an MlirContext maps to a single
  // PyMlirContext wrapper. This could be replaced in the future with an
  // extension mechanism on the MlirContext for stashing user pointers.
  // Note that this holds a handle, which does not imply ownership.
  // Mappings will be removed when the context is destructed.
  using LiveContextMap =
      llvm::DenseMap<void *, std::pair<pybind11::handle, PyMlirContext *>>;
  static LiveContextMap &getLiveContexts();

  MlirContext context;
  // The handle is set as part of lookup with forContext() (post construction).
  pybind11::handle handle;
};

/// Base class for all objects that directly or indirectly depend on an
/// MlirContext. The lifetime of the context will extend at least to the
/// lifetime of these instances.
/// Immutable objects that depend on a context extend this directly.
class BaseContextObject {
public:
  BaseContextObject(PyMlirContextRef ref) : contextRef(std::move(ref)) {}

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
class PyModule : public BaseContextObject {
public:
  PyModule(PyMlirContextRef contextRef, MlirModule module)
      : BaseContextObject(std::move(contextRef)), module(module) {}
  PyModule(PyModule &) = delete;
  PyModule(PyModule &&other)
      : BaseContextObject(std::move(other.getContext())) {
    module = other.module;
    other.module.ptr = nullptr;
  }
  ~PyModule() {
    if (module.ptr)
      mlirModuleDestroy(module);
  }

  MlirModule module;
};

/// Wrapper around an MlirRegion.
/// Note that region can exist in a detached state (where this instance is
/// responsible for clearing) or an attached state (where its owner is
/// responsible).
///
/// This python wrapper retains a redundant reference to its creating context
/// in order to facilitate checking that parts of the operation hierarchy
/// are only assembled from the same context.
class PyRegion {
public:
  PyRegion(MlirContext context, MlirRegion region, bool detached)
      : context(context), region(region), detached(detached) {}
  PyRegion(PyRegion &&other)
      : context(other.context), region(other.region), detached(other.detached) {
    other.detached = false;
  }
  ~PyRegion() {
    if (detached)
      mlirRegionDestroy(region);
  }

  // Call prior to attaching the region to a parent.
  // This will transition to the attached state and will throw an exception
  // if already attached.
  void attachToParent();

  MlirContext context;
  MlirRegion region;

private:
  bool detached;
};

/// Wrapper around an MlirBlock.
/// Note that blocks can exist in a detached state (where this instance is
/// responsible for clearing) or an attached state (where its owner is
/// responsible).
///
/// This python wrapper retains a redundant reference to its creating context
/// in order to facilitate checking that parts of the operation hierarchy
/// are only assembled from the same context.
class PyBlock {
public:
  PyBlock(MlirContext context, MlirBlock block, bool detached)
      : context(context), block(block), detached(detached) {}
  PyBlock(PyBlock &&other)
      : context(other.context), block(other.block), detached(other.detached) {
    other.detached = false;
  }
  ~PyBlock() {
    if (detached)
      mlirBlockDestroy(block);
  }

  // Call prior to attaching the block to a parent.
  // This will transition to the attached state and will throw an exception
  // if already attached.
  void attachToParent();

  MlirContext context;
  MlirBlock block;

private:
  bool detached;
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
