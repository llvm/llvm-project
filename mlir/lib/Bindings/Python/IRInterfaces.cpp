//===- IRInterfaces.cpp - MLIR IR interfaces pybind -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "IRModule.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace nb = nanobind;

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

constexpr static const char *inferReturnTypeComponentsDoc =
    R"(Given the arguments required to build an operation, attempts to infer
its return shaped type components. Raises ValueError on failure.)";

namespace {

/// Takes in an optional ist of operands and converts them into a SmallVector
/// of MlirVlaues. Returns an empty SmallVector if the list is empty.
llvm::SmallVector<MlirValue> wrapOperands(std::optional<nb::list> operandList) {
  llvm::SmallVector<MlirValue> mlirOperands;

  if (!operandList || operandList->size() == 0) {
    return mlirOperands;
  }

  // Note: as the list may contain other lists this may not be final size.
  mlirOperands.reserve(operandList->size());
  for (const auto &&it : llvm::enumerate(*operandList)) {
    if (it.value().is_none())
      continue;

    PyValue *val;
    try {
      val = nb::cast<PyValue *>(it.value());
      if (!val)
        throw nb::cast_error();
      mlirOperands.push_back(val->get());
      continue;
    } catch (nb::cast_error &err) {
      // Intentionally unhandled to try sequence below first.
      (void)err;
    }

    try {
      auto vals = nb::cast<nb::sequence>(it.value());
      for (nb::handle v : vals) {
        try {
          val = nb::cast<PyValue *>(v);
          if (!val)
            throw nb::cast_error();
          mlirOperands.push_back(val->get());
        } catch (nb::cast_error &err) {
          throw nb::value_error(
              (llvm::Twine("Operand ") + llvm::Twine(it.index()) +
               " must be a Value or Sequence of Values (" + err.what() + ")")
                  .str()
                  .c_str());
        }
      }
      continue;
    } catch (nb::cast_error &err) {
      throw nb::value_error((llvm::Twine("Operand ") + llvm::Twine(it.index()) +
                             " must be a Value or Sequence of Values (" +
                             err.what() + ")")
                                .str()
                                .c_str());
    }

    throw nb::cast_error();
  }

  return mlirOperands;
}

/// Takes in an optional vector of PyRegions and returns a SmallVector of
/// MlirRegion. Returns an empty SmallVector if the list is empty.
llvm::SmallVector<MlirRegion>
wrapRegions(std::optional<std::vector<PyRegion>> regions) {
  llvm::SmallVector<MlirRegion> mlirRegions;

  if (regions) {
    mlirRegions.reserve(regions->size());
    for (PyRegion &region : *regions) {
      mlirRegions.push_back(region);
    }
  }

  return mlirRegions;
}

} // namespace

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
  using ClassTy = nb::class_<ConcreteIface>;
  using GetTypeIDFunctionTy = MlirTypeID (*)();

public:
  /// Constructs an interface instance from an object that is either an
  /// operation or a subclass of OpView. In the latter case, only the static
  /// methods of the interface are accessible to the caller.
  PyConcreteOpInterface(nb::object object, DefaultingPyMlirContext context)
      : obj(std::move(object)) {
    try {
      operation = &nb::cast<PyOperation &>(obj);
    } catch (nb::cast_error &) {
      // Do nothing.
    }

    try {
      operation = &nb::cast<PyOpView &>(obj).getOperation();
    } catch (nb::cast_error &) {
      // Do nothing.
    }

    if (operation != nullptr) {
      if (!mlirOperationImplementsInterface(*operation,
                                            ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nb::value_error((msg + ConcreteIface::pyClassName).c_str());
      }

      MlirIdentifier identifier = mlirOperationGetName(*operation);
      MlirStringRef stringRef = mlirIdentifierStr(identifier);
      opName = std::string(stringRef.data, stringRef.length);
    } else {
      try {
        opName = nb::cast<std::string>(obj.attr("OPERATION_NAME"));
      } catch (nb::cast_error &) {
        throw nb::type_error(
            "Op interface does not refer to an operation or OpView class");
      }

      if (!mlirOperationImplementsInterfaceStatic(
              mlirStringRefCreate(opName.data(), opName.length()),
              context.resolve().get(), ConcreteIface::getInterfaceID())) {
        std::string msg = "the operation does not implement ";
        throw nb::value_error((msg + ConcreteIface::pyClassName).c_str());
      }
    }
  }

  /// Creates the Python bindings for this class in the given module.
  static void bind(nb::module_ &m) {
    nb::class_<ConcreteIface> cls(m, ConcreteIface::pyClassName);
    cls.def(nb::init<nb::object, DefaultingPyMlirContext>(), nb::arg("object"),
            nb::arg("context") = nb::none(), constructorDoc)
        .def_prop_ro("operation", &PyConcreteOpInterface::getOperationObject,
                     operationDoc)
        .def_prop_ro("opview", &PyConcreteOpInterface::getOpView, opviewDoc);
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
  nb::object getOperationObject() {
    if (operation == nullptr) {
      throw nb::type_error("Cannot get an operation from a static interface");
    }

    return operation->getRef().releaseObject();
  }

  /// Returns the opview of the operation instance from which this object was
  /// constructed. Throws a type error if this object was constructed form a
  /// subclass of OpView.
  nb::object getOpView() {
    if (operation == nullptr) {
      throw nb::type_error("Cannot get an opview from a static interface");
    }

    return operation->createOpView();
  }

  /// Returns the canonical name of the operation this interface is constructed
  /// from.
  const std::string &getOpName() { return opName; }

private:
  PyOperation *operation = nullptr;
  std::string opName;
  nb::object obj;
};

/// Python wrapper for InferTypeOpInterface. This interface has only static
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
  /// return types. Throws value_error on failure.
  std::vector<PyType>
  inferReturnTypes(std::optional<nb::list> operandList,
                   std::optional<PyAttribute> attributes, void *properties,
                   std::optional<std::vector<PyRegion>> regions,
                   DefaultingPyMlirContext context,
                   DefaultingPyLocation location) {
    llvm::SmallVector<MlirValue> mlirOperands =
        wrapOperands(std::move(operandList));
    llvm::SmallVector<MlirRegion> mlirRegions = wrapRegions(std::move(regions));

    std::vector<PyType> inferredTypes;
    PyMlirContext &pyContext = context.resolve();
    AppendResultsCallbackData data{inferredTypes, pyContext};
    MlirStringRef opNameRef =
        mlirStringRefCreate(getOpName().data(), getOpName().length());
    MlirAttribute attributeDict =
        attributes ? attributes->get() : mlirAttributeGetNull();

    MlirLogicalResult result = mlirInferTypeOpInterfaceInferReturnTypes(
        opNameRef, pyContext.get(), location.resolve(), mlirOperands.size(),
        mlirOperands.data(), attributeDict, properties, mlirRegions.size(),
        mlirRegions.data(), &appendResultsCallback, &data);

    if (mlirLogicalResultIsFailure(result)) {
      throw nb::value_error("Failed to infer result types");
    }

    return inferredTypes;
  }

  static void bindDerived(ClassTy &cls) {
    cls.def("inferReturnTypes", &PyInferTypeOpInterface::inferReturnTypes,
            nb::arg("operands") = nb::none(),
            nb::arg("attributes") = nb::none(),
            nb::arg("properties") = nb::none(), nb::arg("regions") = nb::none(),
            nb::arg("context") = nb::none(), nb::arg("loc") = nb::none(),
            inferReturnTypesDoc);
  }
};

/// Wrapper around an shaped type components.
class PyShapedTypeComponents {
public:
  PyShapedTypeComponents(MlirType elementType) : elementType(elementType) {}
  PyShapedTypeComponents(nb::list shape, MlirType elementType)
      : shape(std::move(shape)), elementType(elementType), ranked(true) {}
  PyShapedTypeComponents(nb::list shape, MlirType elementType,
                         MlirAttribute attribute)
      : shape(std::move(shape)), elementType(elementType), attribute(attribute),
        ranked(true) {}
  PyShapedTypeComponents(PyShapedTypeComponents &) = delete;
  PyShapedTypeComponents(PyShapedTypeComponents &&other) noexcept
      : shape(other.shape), elementType(other.elementType),
        attribute(other.attribute), ranked(other.ranked) {}

  static void bind(nb::module_ &m) {
    nb::class_<PyShapedTypeComponents>(m, "ShapedTypeComponents")
        .def_prop_ro(
            "element_type",
            [](PyShapedTypeComponents &self) { return self.elementType; },
            "Returns the element type of the shaped type components.")
        .def_static(
            "get",
            [](PyType &elementType) {
              return PyShapedTypeComponents(elementType);
            },
            nb::arg("element_type"),
            "Create an shaped type components object with only the element "
            "type.")
        .def_static(
            "get",
            [](nb::list shape, PyType &elementType) {
              return PyShapedTypeComponents(std::move(shape), elementType);
            },
            nb::arg("shape"), nb::arg("element_type"),
            "Create a ranked shaped type components object.")
        .def_static(
            "get",
            [](nb::list shape, PyType &elementType, PyAttribute &attribute) {
              return PyShapedTypeComponents(std::move(shape), elementType,
                                            attribute);
            },
            nb::arg("shape"), nb::arg("element_type"), nb::arg("attribute"),
            "Create a ranked shaped type components object with attribute.")
        .def_prop_ro(
            "has_rank",
            [](PyShapedTypeComponents &self) -> bool { return self.ranked; },
            "Returns whether the given shaped type component is ranked.")
        .def_prop_ro(
            "rank",
            [](PyShapedTypeComponents &self) -> nb::object {
              if (!self.ranked) {
                return nb::none();
              }
              return nb::int_(self.shape.size());
            },
            "Returns the rank of the given ranked shaped type components. If "
            "the shaped type components does not have a rank, None is "
            "returned.")
        .def_prop_ro(
            "shape",
            [](PyShapedTypeComponents &self) -> nb::object {
              if (!self.ranked) {
                return nb::none();
              }
              return nb::list(self.shape);
            },
            "Returns the shape of the ranked shaped type components as a list "
            "of integers. Returns none if the shaped type component does not "
            "have a rank.");
  }

  nb::object getCapsule();
  static PyShapedTypeComponents createFromCapsule(nb::object capsule);

private:
  nb::list shape;
  MlirType elementType;
  MlirAttribute attribute;
  bool ranked{false};
};

/// Python wrapper for InferShapedTypeOpInterface. This interface has only
/// static methods.
class PyInferShapedTypeOpInterface
    : public PyConcreteOpInterface<PyInferShapedTypeOpInterface> {
public:
  using PyConcreteOpInterface<
      PyInferShapedTypeOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "InferShapedTypeOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &mlirInferShapedTypeOpInterfaceTypeID;

  /// C-style user-data structure for type appending callback.
  struct AppendResultsCallbackData {
    std::vector<PyShapedTypeComponents> &inferredShapedTypeComponents;
  };

  /// Appends the shaped type components provided as unpacked shape, element
  /// type, attribute to the user-data.
  static void appendResultsCallback(bool hasRank, intptr_t rank,
                                    const int64_t *shape, MlirType elementType,
                                    MlirAttribute attribute, void *userData) {
    auto *data = static_cast<AppendResultsCallbackData *>(userData);
    if (!hasRank) {
      data->inferredShapedTypeComponents.emplace_back(elementType);
    } else {
      nb::list shapeList;
      for (intptr_t i = 0; i < rank; ++i) {
        shapeList.append(shape[i]);
      }
      data->inferredShapedTypeComponents.emplace_back(shapeList, elementType,
                                                      attribute);
    }
  }

  /// Given the arguments required to build an operation, attempts to infer the
  /// shaped type components. Throws value_error on failure.
  std::vector<PyShapedTypeComponents> inferReturnTypeComponents(
      std::optional<nb::list> operandList,
      std::optional<PyAttribute> attributes, void *properties,
      std::optional<std::vector<PyRegion>> regions,
      DefaultingPyMlirContext context, DefaultingPyLocation location) {
    llvm::SmallVector<MlirValue> mlirOperands =
        wrapOperands(std::move(operandList));
    llvm::SmallVector<MlirRegion> mlirRegions = wrapRegions(std::move(regions));

    std::vector<PyShapedTypeComponents> inferredShapedTypeComponents;
    PyMlirContext &pyContext = context.resolve();
    AppendResultsCallbackData data{inferredShapedTypeComponents};
    MlirStringRef opNameRef =
        mlirStringRefCreate(getOpName().data(), getOpName().length());
    MlirAttribute attributeDict =
        attributes ? attributes->get() : mlirAttributeGetNull();

    MlirLogicalResult result = mlirInferShapedTypeOpInterfaceInferReturnTypes(
        opNameRef, pyContext.get(), location.resolve(), mlirOperands.size(),
        mlirOperands.data(), attributeDict, properties, mlirRegions.size(),
        mlirRegions.data(), &appendResultsCallback, &data);

    if (mlirLogicalResultIsFailure(result)) {
      throw nb::value_error("Failed to infer result shape type components");
    }

    return inferredShapedTypeComponents;
  }

  static void bindDerived(ClassTy &cls) {
    cls.def("inferReturnTypeComponents",
            &PyInferShapedTypeOpInterface::inferReturnTypeComponents,
            nb::arg("operands") = nb::none(),
            nb::arg("attributes") = nb::none(), nb::arg("regions") = nb::none(),
            nb::arg("properties") = nb::none(), nb::arg("context") = nb::none(),
            nb::arg("loc") = nb::none(), inferReturnTypeComponentsDoc);
  }
};

void populateIRInterfaces(nb::module_ &m) {
  PyInferTypeOpInterface::bind(m);
  PyShapedTypeComponents::bind(m);
  PyInferShapedTypeOpInterface::bind(m);
}

} // namespace python
} // namespace mlir
