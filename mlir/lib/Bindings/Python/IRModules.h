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

namespace mlir {
namespace python {

class PyMlirContext;
class PyModule;

/// Wrapper around MlirContext.
class PyMlirContext {
public:
  PyMlirContext() { context = mlirContextCreate(); }
  ~PyMlirContext() { mlirContextDestroy(context); }

  MlirContext context;
};

/// Wrapper around an MlirLocation.
class PyLocation {
public:
  PyLocation(MlirLocation loc) : loc(loc) {}
  MlirLocation loc;
};

/// Wrapper around MlirModule.
class PyModule {
public:
  PyModule(MlirModule module) : module(module) {}
  PyModule(PyModule &) = delete;
  PyModule(PyModule &&other) {
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
class PyAttribute {
public:
  PyAttribute(MlirAttribute attr) : attr(attr) {}
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
class PyType {
public:
  PyType(MlirType type) : type(type) {}
  bool operator==(const PyType &other);
  operator MlirType() const { return type; }

  MlirType type;
};

void populateIRSubmodule(pybind11::module &m);

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRMODULES_H
