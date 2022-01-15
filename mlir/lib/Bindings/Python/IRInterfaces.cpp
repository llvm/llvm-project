//===- IRInterfaces.cpp - MLIR IR interfaces pybind -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "IRModule.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/Interfaces.h"

namespace py = pybind11;

namespace mlir {
namespace python {

constexpr static const char *constructorDoc =
    R"(Creates an interface from a given operation/opview object or from a
subclass of OpView. Raises ValueError if the operation does not implement the
interface.)";

constexpr static const char *operationDoc =
    R"(Returns an Operation for which the interface was constructed.)";

constexpr static const char *opviewDoc =
    R"(Returns an OpView subclass _instance_ for which the interface was
constructed)";

constexpr static const char *inferReturnTypesDoc =
    R"(Given the arguments required to build an operation, attempts to infer
its return types. Raises ValueError on failure.)";

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
  using ClassTy = py::class_<ConcreteIface>;
  using GetTypeIDFunctionTy = MlirTypeID (*)();

public:
  /// Constructs an interface instance from an object that is either an
  /// operation or a subclass of OpView. In the latter case, only the static
  /// methods of the interface are accessible to the caller.
  PyConcreteOpInterface(py::object object, DefaultingPyMlirContext context)
      : obj(std::move(object)) {
    try {
      operation = &py::cast<PyOperation &>(obj);
    } catch (py::cast_error &err) {
      // Do nothing.
    }

    try {
      operation = &py::cast<PyOpView &>(obj).getOperation();
    } catch (py::cast_error &err) {
      // Do nothing.
    }

    if (operation != nullptr) {
      if (!mlirOperationImplementsInterface(*operation,
                                            ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw py::value_error(msg + ConcreteIface::pyClassName);
      }

      MlirIdentifier identifier = mlirOperationGetName(*operation);
      MlirStringRef stringRef = mlirIdentifierStr(identifier);
      opName = std::string(stringRef.data, stringRef.length);
    } else {
      try {
        opName = obj.attr("OPERATION_NAME").template cast<std::string>();
      } catch (py::cast_error &err) {
        throw py::type_error(
            "Op interface does not refer to an operation or OpView class");
      }

      if (!mlirOperationImplementsInterfaceStatic(
              mlirStringRefCreate(opName.data(), opName.length()),
              context.resolve().get(), ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw py::value_error(msg + ConcreteIface::pyClassName);
      }
    }
  }

  /// Creates the Python bindings for this class in the given module.
  static void bind(py::module &m) {
    py::class_<ConcreteIface> cls(m, "InferTypeOpInterface",
                                  py::module_local());
    cls.def(py::init<py::object, DefaultingPyMlirContext>(), py::arg("object"),
            py::arg("context") = py::none(), constructorDoc)
        .def_property_readonly("operation",
                               &PyConcreteOpInterface::getOperationObject,
                               operationDoc)
        .def_property_readonly("opview", &PyConcreteOpInterface::getOpView,
                               opviewDoc);
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
  py::object getOperationObject() {
    if (operation == nullptr) {
      throw py::type_error("Cannot get an operation from a static interface");
    }

    return operation->getRef().releaseObject();
  }

  /// Returns the opview of the operation instance from which this object was
  /// constructed. Throws a type error if this object was constructed form a
  /// subclass of OpView.
  py::object getOpView() {
    if (operation == nullptr) {
      throw py::type_error("Cannot get an opview from a static interface");
    }

    return operation->createOpView();
  }

  /// Returns the canonical name of the operation this interface is constructed
  /// from.
  const std::string &getOpName() { return opName; }

private:
  PyOperation *operation = nullptr;
  std::string opName;
  py::object obj;
};

/// Python wrapper for InterTypeOpInterface. This interface has only static
/// methods.
class PyInferTypeOpInterface
    : public PyConcreteOpInterface<PyInferTypeOpInterface> {
public:
  using PyConcreteOpInterface<PyInferTypeOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "InferTypeOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &mlirInferTypeOpInterfaceTypeID;

  /// C-style user-data structure for type appending callback.
  struct AppendResultsCallbackData {
    std::vector<PyType> &inferredTypes;
    PyMlirContext &pyMlirContext;
  };

  /// Appends the types provided as the two first arguments to the user-data
  /// structure (expects AppendResultsCallbackData).
  static void appendResultsCallback(intptr_t nTypes, MlirType *types,
                                    void *userData) {
    auto *data = static_cast<AppendResultsCallbackData *>(userData);
    data->inferredTypes.reserve(data->inferredTypes.size() + nTypes);
    for (intptr_t i = 0; i < nTypes; ++i) {
      data->inferredTypes.emplace_back(data->pyMlirContext.getRef(), types[i]);
    }
  }

  /// Given the arguments required to build an operation, attempts to infer its
  /// return types. Throws value_error on faliure.
  std::vector<PyType>
  inferReturnTypes(llvm::Optional<std::vector<PyValue>> operands,
                   llvm::Optional<PyAttribute> attributes,
                   llvm::Optional<std::vector<PyRegion>> regions,
                   DefaultingPyMlirContext context,
                   DefaultingPyLocation location) {
    llvm::SmallVector<MlirValue> mlirOperands;
    llvm::SmallVector<MlirRegion> mlirRegions;

    if (operands) {
      mlirOperands.reserve(operands->size());
      for (PyValue &value : *operands) {
        mlirOperands.push_back(value);
      }
    }

    if (regions) {
      mlirRegions.reserve(regions->size());
      for (PyRegion &region : *regions) {
        mlirRegions.push_back(region);
      }
    }

    std::vector<PyType> inferredTypes;
    PyMlirContext &pyContext = context.resolve();
    AppendResultsCallbackData data{inferredTypes, pyContext};
    MlirStringRef opNameRef =
        mlirStringRefCreate(getOpName().data(), getOpName().length());
    MlirAttribute attributeDict =
        attributes ? attributes->get() : mlirAttributeGetNull();

    MlirLogicalResult result = mlirInferTypeOpInterfaceInferReturnTypes(
        opNameRef, pyContext.get(), location.resolve(), mlirOperands.size(),
        mlirOperands.data(), attributeDict, mlirRegions.size(),
        mlirRegions.data(), &appendResultsCallback, &data);

    if (mlirLogicalResultIsFailure(result)) {
      throw py::value_error("Failed to infer result types");
    }

    return inferredTypes;
  }

  static void bindDerived(ClassTy &cls) {
    cls.def("inferReturnTypes", &PyInferTypeOpInterface::inferReturnTypes,
            py::arg("operands") = py::none(),
            py::arg("attributes") = py::none(), py::arg("regions") = py::none(),
            py::arg("context") = py::none(), py::arg("loc") = py::none(),
            inferReturnTypesDoc);
  }
};

void populateIRInterfaces(py::module &m) { PyInferTypeOpInterface::bind(m); }

} // namespace python
} // namespace mlir
