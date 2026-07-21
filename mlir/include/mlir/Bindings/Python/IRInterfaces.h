//===- IRInterfaces.h - IR Interfaces for Python Bindings -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_IRINTERFACES_H
#define MLIR_BINDINGS_PYTHON_IRINTERFACES_H

#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"

#include <nanobind/nanobind.h>

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {

/// CRTP base class for Python classes representing MLIR Op interfaces.
/// Interface hierarchies are flat so no base class is expected here. The
/// derived class is expected to define the following static fields:
///  - `const char *pyClassName` - the name of the Python class to create;
///  - `GetTypeIDFunctionTy getInterfaceID` - the function producing the TypeID
///    of the interface.
/// Derived classes may redefine the `bindDerived(ClassTy &)` method to bind
/// interface-specific methods.
///
/// An interface class may be constructed from either an Operation/OpView object
/// or from a subclass of OpView. In the latter case, only the static interface
/// methods are available, similarly to calling ConcereteOp::staticMethod on the
/// C++ side. Implementations of concrete interfaces can use the `isStatic`
/// method to check whether the interface object was constructed from a class or
/// an operation/opview instance. The `getOpName` always succeeds and returns a
/// canonical name of the operation suitable for lookups.
template <typename ConcreteIface>
class PyConcreteOpInterface {
protected:
  using ClassTy = nanobind::class_<ConcreteIface>;
  using GetTypeIDFunctionTy = MlirTypeID (*)();

public:
  /// Constructs an interface instance from an object that is either an
  /// operation or a subclass of OpView. In the latter case, only the static
  /// methods of the interface are accessible to the caller.
  PyConcreteOpInterface(nanobind::object object,
                        DefaultingPyMlirContext context)
      : obj(std::move(object)) {
    if (!nanobind::try_cast<PyOperation *>(obj, operation)) {
      PyOpView *opview;
      if (nanobind::try_cast<PyOpView *>(obj, opview)) {
        operation = &opview->getOperation();
      };
    }

    if (operation != nullptr) {
      if (!mlirOperationImplementsInterface(*operation,
                                            ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nanobind::value_error((msg + ConcreteIface::pyClassName).c_str());
      }

      MlirIdentifier identifier = mlirOperationGetName(*operation);
      MlirStringRef stringRef = mlirIdentifierStr(identifier);
      opName = std::string(stringRef.data, stringRef.length);
    } else {
      if (!nanobind::try_cast<std::string>(obj.attr("OPERATION_NAME"), opName))
        throw nanobind::type_error(
            "Op interface does not refer to an operation or OpView class");

      if (!mlirOperationImplementsInterfaceStatic(
              mlirStringRefCreate(opName.data(), opName.length()),
              context.resolve().get(), ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nanobind::value_error((msg + ConcreteIface::pyClassName).c_str());
      }
    }
  }

  /// Creates the Python bindings for this class in the given module.
  static void bind(nanobind::module_ &m) {
    nanobind::class_<ConcreteIface> cls(m, ConcreteIface::pyClassName);
    cls.def(nanobind::init<nanobind::object, DefaultingPyMlirContext>(),
            nanobind::arg("object"),
            nanobind::arg("context") = nanobind::none(),
            "Creates an interface from a given operation/opview object or from "
            "a subclass of OpView. Raises ValueError if the operation does not "
            "implement the interface.")
        .def_prop_ro(
            "operation", &PyConcreteOpInterface::getOperationObject,
            "Returns an Operation for which the interface was constructed.")
        .def_prop_ro("opview", &PyConcreteOpInterface::getOpView,
                     "Returns an OpView subclass _instance_ for which the "
                     "interface was constructed");
    ConcreteIface::bindDerived(cls);
  }

  /// Hook for derived classes to add class-specific bindings.
  static void bindDerived(ClassTy &cls) {}

  /// Returns `true` if this object was constructed from a subclass of OpView
  /// rather than from an operation instance.
  bool isStatic() { return operation == nullptr; }

  /// Returns the operation instance from which this object was constructed.
  /// Throws a type error if this object was constructed from a subclass of
  /// OpView.
  nanobind::typed<nanobind::object, PyOperation> getOperationObject() {
    if (operation == nullptr)
      throw nanobind::type_error(
          "Cannot get an operation from a static interface");
    return operation->getRef().releaseObject();
  }

  /// Returns the opview of the operation instance from which this object was
  /// constructed. Throws a type error if this object was constructed form a
  /// subclass of OpView.
  nanobind::typed<nanobind::object, PyOpView> getOpView() {
    if (operation == nullptr)
      throw nanobind::type_error(
          "Cannot get an opview from a static interface");
    return operation->createOpView();
  }

  /// Returns the canonical name of the operation this interface is constructed
  /// from.
  const std::string &getOpName() { return opName; }

private:
  PyOperation *operation = nullptr;
  std::string opName;
  nanobind::object obj;
};

struct PyMemoryEffectsInstanceList {
  MlirMemoryEffectInstancesList effects;
};

} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_IRINTERFACES_H
