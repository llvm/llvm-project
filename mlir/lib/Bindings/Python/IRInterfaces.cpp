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

#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRInterfaces.h"

namespace nb = nanobind;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
constexpr static const char *inferReturnTypesDoc =
    R"(Given the arguments required to build an operation, attempts to infer
its return types. Raises ValueError on failure.)";

constexpr static const char *inferReturnTypeComponentsDoc =
    R"(Given the arguments required to build an operation, attempts to infer
its return shaped type components. Raises ValueError on failure.)";

namespace {

/// Takes in an optional ist of operands and converts them into a std::vector
/// of MlirVlaues. Returns an empty std::vector if the list is empty.
std::vector<MlirValue> wrapOperands(std::optional<nb::list> operandList) {
  std::vector<MlirValue> mlirOperands;

  if (!operandList || operandList->size() == 0) {
    return mlirOperands;
  }

  // Note: as the list may contain other lists this may not be final size.
  mlirOperands.reserve(operandList->size());
  for (size_t i = 0, e = operandList->size(); i < e; ++i) {
    nb::handle operand = (*operandList)[i];
    intptr_t index = static_cast<intptr_t>(i);
    if (operand.is_none())
      continue;

    PyValue *val;
    try {
      val = nb::cast<PyValue *>(operand);
      if (!val)
        throw nb::cast_error();
      mlirOperands.push_back(val->get());
      continue;
    } catch (nb::cast_error &err) {
      // Intentionally unhandled to try sequence below first.
      (void)err;
    }

    try {
      auto vals = nb::cast<nb::sequence>(operand);
      for (nb::handle v : vals) {
        try {
          val = nb::cast<PyValue *>(v);
          if (!val)
            throw nb::cast_error();
          mlirOperands.push_back(val->get());
        } catch (nb::cast_error &err) {
          throw nb::value_error(
              nanobind::detail::join("Operand ", index,
                                     " must be a Value or Sequence of Values (",
                                     err.what(), ")")
                  .c_str());
        }
      }
      continue;
    } catch (nb::cast_error &err) {
      throw nb::value_error(
          nanobind::detail::join("Operand ", index,
                                 " must be a Value or Sequence of Values (",
                                 err.what(), ")")
              .c_str());
    }

    throw nb::cast_error();
  }

  return mlirOperands;
}

/// Takes in an optional vector of PyRegions and returns a std::vector of
/// MlirRegion. Returns an empty std::vector if the list is empty.
std::vector<MlirRegion>
wrapRegions(std::optional<std::vector<PyRegion>> regions) {
  std::vector<MlirRegion> mlirRegions;

  if (regions) {
    mlirRegions.reserve(regions->size());
    for (PyRegion &region : *regions) {
      mlirRegions.push_back(region);
    }
  }

  return mlirRegions;
}

} // namespace

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
    std::vector<MlirValue> mlirOperands = wrapOperands(std::move(operandList));
    std::vector<MlirRegion> mlirRegions = wrapRegions(std::move(regions));

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
            nb::sig("def element_type(self) -> Type"),
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
            [](PyShapedTypeComponents &self) -> std::optional<nb::int_> {
              if (!self.ranked)
                return {};
              return nb::int_(self.shape.size());
            },
            "Returns the rank of the given ranked shaped type components. If "
            "the shaped type components does not have a rank, None is "
            "returned.")
        .def_prop_ro(
            "shape",
            [](PyShapedTypeComponents &self) -> std::optional<nb::list> {
              if (!self.ranked)
                return {};
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
    std::vector<MlirValue> mlirOperands = wrapOperands(std::move(operandList));
    std::vector<MlirRegion> mlirRegions = wrapRegions(std::move(regions));

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

/// Wrapper around the MemoryEffectsOpInterface.
class PyMemoryEffectsOpInterface
    : public PyConcreteOpInterface<PyMemoryEffectsOpInterface> {
public:
  using PyConcreteOpInterface<
      PyMemoryEffectsOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "MemoryEffectsOpInterface";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &mlirMemoryEffectsOpInterfaceTypeID;

  /// Attach a new MemoryEffectsOpInterface FallbackModel to the named
  /// operation. The FallbackModel acts as a trampoline for callbacks on the
  /// Python class.
  static void attach(nb::object &target, const std::string &opName,
                     DefaultingPyMlirContext ctx) {
    MlirMemoryEffectsOpInterfaceCallbacks callbacks;
    callbacks.userData = target.ptr();
    nb::handle(static_cast<PyObject *>(callbacks.userData)).inc_ref();
    callbacks.construct = nullptr;
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };
    callbacks.getEffects = [](MlirOperation op,
                              MlirMemoryEffectInstancesList effects,
                              void *userData) {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      // Get the 'get_effects' method from the Python class.
      auto pyGetEffects =
          nb::cast<nb::callable>(nb::getattr(pyClass, "get_effects"));

      PyMemoryEffectsInstanceList effectsWrapper{effects};

      PyMlirContextRef context =
          PyMlirContext::forContext(mlirOperationGetContext(op));
      auto opview = PyOperation::forOperation(context, op)->createOpView();

      // Invoke `pyClass.get_effects(op, effects)`.
      pyGetEffects(opview, effectsWrapper);
    };

    mlirMemoryEffectsOpInterfaceAttachFallbackModel(
        ctx->get(), mlirStringRefCreate(opName.c_str(), opName.size()),
        callbacks);
  }

  static void bindDerived(ClassTy &cls) {
    cls.attr("attach") = classmethod(
        [](const nb::object &cls, const nb::object &opName, nb::object target,
           DefaultingPyMlirContext context) {
          if (target.is_none())
            target = cls;
          return attach(target, nb::cast<std::string>(opName), context);
        },
        nb::arg("cls"), nb::arg("op_name"), nb::kw_only(),
        nb::arg("target").none() = nb::none(),
        nb::arg("context").none() = nb::none(),
        "Attach the interface subclass to the given operation name.");
  }
};

void populateIRInterfaces(nb::module_ &m) {
  nb::class_<PyMemoryEffectsInstanceList>(m, "MemoryEffectInstancesList");

  PyInferShapedTypeOpInterface::bind(m);
  PyInferTypeOpInterface::bind(m);
  PyMemoryEffectsOpInterface::bind(m);
  PyShapedTypeComponents::bind(m);
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
