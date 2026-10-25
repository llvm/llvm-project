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

MlirAttribute unwrapOptionalAttribute(const nb::object &attribute) {
  if (attribute.is_none())
    return mlirAttributeGetNull();

  PyAttribute *pyAttribute = nullptr;
  if (!nb::try_cast<PyAttribute *>(attribute, pyAttribute) || !pyAttribute)
    throw nb::type_error("parameters must be an Attribute or None");
  return pyAttribute->get();
}

MlirMemoryEffectInstance createMemoryEffectInstance(
    const PyMemoryEffect &effect, const nb::object &target,
    const nb::object &parameters, int stage, bool effectOnFullRegion,
    const PySideEffectResource &resource) {
  MlirAttribute unwrappedParameters = unwrapOptionalAttribute(parameters);

  MlirMemoryEffectInstance rawInstance{nullptr};
  if (target.is_none()) {
    rawInstance =
        mlirMemoryEffectInstanceCreate(effect.get(), unwrappedParameters, stage,
                                       effectOnFullRegion, resource.get());
  } else {
    PyOpOperand *opOperand = nullptr;
    PyValue *value = nullptr;
    PyAttribute *attribute = nullptr;
    if (nb::try_cast<PyOpOperand *>(target, opOperand) && opOperand) {
      rawInstance = mlirMemoryEffectInstanceCreateForOpOperand(
          effect.get(), *opOperand, unwrappedParameters, stage,
          effectOnFullRegion, resource.get());
    } else if (nb::try_cast<PyValue *>(target, value) && value) {
      MlirValue mlirValue = value->get();
      if (mlirValueIsAOpResult(mlirValue)) {
        rawInstance = mlirMemoryEffectInstanceCreateForOpResult(
            effect.get(), mlirValue, unwrappedParameters, stage,
            effectOnFullRegion, resource.get());
      } else if (mlirValueIsABlockArgument(mlirValue)) {
        rawInstance = mlirMemoryEffectInstanceCreateForBlockArgument(
            effect.get(), mlirValue, unwrappedParameters, stage,
            effectOnFullRegion, resource.get());
      } else {
        throw nb::type_error(
            "target Value must be an OpResult or BlockArgument");
      }
    } else if (nb::try_cast<PyAttribute *>(target, attribute) && attribute) {
      MlirAttribute symbol = attribute->get();
      if (!mlirAttributeIsASymbolRef(symbol))
        throw nb::type_error("target Attribute must be a SymbolRefAttr");
      rawInstance = mlirMemoryEffectInstanceCreateForSymbol(
          effect.get(), symbol, unwrappedParameters, stage, effectOnFullRegion,
          resource.get());
    } else {
      throw nb::type_error(
          "target must be an OpOperand, OpResult, BlockArgument, "
          "SymbolRefAttr, or None");
    }
  }
  return rawInstance;
}

/// Takes in an optional ist of operands and converts them into a std::vector
/// of MlirVlaues. Returns an empty std::vector if the list is empty.
std::vector<MlirValue> wrapOperands(std::optional<nb::sequence> operandList) {
  std::vector<MlirValue> mlirOperands;

  if (!operandList || nb::len(*operandList) == 0) {
    return mlirOperands;
  }

  // Note: as the list may contain other lists this may not be final size.
  mlirOperands.reserve(nb::len(*operandList));
  for (size_t i = 0, e = nb::len(*operandList); i < e; ++i) {
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

PyMemoryEffectInstance::PyMemoryEffectInstance(
    const PyMemoryEffect &effect, const nb::object &target,
    const nb::object &parameters, int stage, bool effectOnFullRegion,
    const PySideEffectResource &resource)
    : PyMemoryEffectInstance(createMemoryEffectInstance(
          effect, target, parameters, stage, effectOnFullRegion, resource)) {}

PyMemoryEffect PyMemoryEffectInstance::getEffect() const {
  return PyMemoryEffect(mlirMemoryEffectInstanceGetEffect(instance));
}

PySideEffectResource PyMemoryEffectInstance::getResource() const {
  return PySideEffectResource(mlirMemoryEffectInstanceGetResource(instance));
}

int PyMemoryEffectInstance::getStage() const {
  return mlirMemoryEffectInstanceGetStage(instance);
}

bool PyMemoryEffectInstance::getEffectOnFullRegion() const {
  return mlirMemoryEffectInstanceGetEffectOnFullRegion(instance);
}

nb::object PyMemoryEffectInstance::getParameters() const {
  MlirAttribute parameters = mlirMemoryEffectInstanceGetParameters(instance);
  if (mlirAttributeIsNull(parameters))
    return nb::none();
  PyMlirContextRef context =
      PyMlirContext::forContext(mlirAttributeGetContext(parameters));
  return PyAttribute(context, parameters).maybeDownCast();
}

nb::object PyMemoryEffectInstance::getValue() const {
  MlirValue value = mlirMemoryEffectInstanceGetValue(instance);
  if (mlirValueIsNull(value))
    return nb::none();
  MlirOperation owner =
      mlirValueIsAOpResult(value)
          ? mlirOpResultGetOwner(value)
          : mlirBlockGetParentOperation(mlirBlockArgumentGetOwner(value));
  PyMlirContextRef context =
      PyMlirContext::forContext(mlirOperationGetContext(owner));
  return PyValue(PyOperation::forOperation(context, owner), value)
      .maybeDownCast();
}

nb::object PyMemoryEffectInstance::getSymbolRef() const {
  MlirAttribute symbol = mlirMemoryEffectInstanceGetSymbolRef(instance);
  if (mlirAttributeIsNull(symbol))
    return nb::none();
  PyMlirContextRef context =
      PyMlirContext::forContext(mlirAttributeGetContext(symbol));
  return PyAttribute(context, symbol).maybeDownCast();
}

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
  inferReturnTypes(std::optional<nb::sequence> operandList,
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
            [](nb::typed<nb::list, nb::int_> shape, PyType &elementType) {
              return PyShapedTypeComponents(std::move(shape), elementType);
            },
            nb::arg("shape"), nb::arg("element_type"),
            "Create a ranked shaped type components object.")
        .def_static(
            "get",
            [](nb::typed<nb::list, nb::int_> shape, PyType &elementType,
               PyAttribute &attribute) {
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
      std::optional<nb::sequence> operandList,
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

/// Wrapper around the ConditionallySpeculatable interface.
class PyConditionallySpeculatableOpInterface
    : public PyConcreteOpInterface<PyConditionallySpeculatableOpInterface> {
public:
  using PyConcreteOpInterface<
      PyConditionallySpeculatableOpInterface>::PyConcreteOpInterface;

  constexpr static const char *pyClassName = "ConditionallySpeculatable";
  constexpr static GetTypeIDFunctionTy getInterfaceID =
      &mlirConditionallySpeculatableOpInterfaceTypeID;

  /// Attach a new ConditionallySpeculatable FallbackModel to the named
  /// operation. The FallbackModel acts as a trampoline for callbacks on the
  /// Python class.
  static void attach(nb::object &target, const std::string &opName,
                     DefaultingPyMlirContext ctx) {
    MlirConditionallySpeculatableOpInterfaceCallbacks callbacks;
    callbacks.userData = target.ptr();
    nb::handle(static_cast<PyObject *>(callbacks.userData)).inc_ref();
    callbacks.construct = nullptr;
    callbacks.destruct = [](void *userData) {
      nb::handle(static_cast<PyObject *>(userData)).dec_ref();
    };
    callbacks.getSpeculatability = [](MlirOperation op, void *userData) {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      auto pyGetSpeculatability =
          nb::cast<nb::callable>(nb::getattr(pyClass, "get_speculatability"));

      PyMlirContextRef context =
          PyMlirContext::forContext(mlirOperationGetContext(op));
      auto opview = PyOperation::forOperation(context, op)->createOpView();

      return nb::cast<MlirSpeculatability>(pyGetSpeculatability(opview));
    };

    mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(
        ctx->get(), mlirStringRefCreate(opName.c_str(), opName.size()),
        callbacks);
  }

  static void bindDerived(ClassTy &cls) {
    cls.def(
        "getSpeculatability",
        [](PyConditionallySpeculatableOpInterface &self) {
          if (self.isStatic())
            throw nb::type_error(
                "Cannot query speculatability on a static interface");
          auto operation = self.getOperationObject();
          auto *pyOperation = nb::cast<PyOperation *>(operation);
          return mlirConditionallySpeculatableOpInterfaceGetSpeculatability(
              pyOperation->get());
        },
        "Returns the speculatability of the given operation.");
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
                              MlirMemoryEffectInstancesCallback callback,
                              void *callbackUserData, void *userData) {
      nb::handle pyClass(static_cast<PyObject *>(userData));

      // Get the 'get_effects' method from the Python class.
      auto pyGetEffects =
          nb::cast<nb::callable>(nb::getattr(pyClass, "get_effects"));

      PyMlirContextRef context =
          PyMlirContext::forContext(mlirOperationGetContext(op));
      auto opview = PyOperation::forOperation(context, op)->createOpView();

      // Invoke `pyClass.get_effects(op)` and pass the resulting instances back
      // to the C++ interface as a borrowed array.
      nb::object result = pyGetEffects(opview);
      nb::iterable iterable;
      if (!nb::try_cast<nb::iterable>(result, iterable))
        throw nb::type_error("get_effects must return an iterable");

      std::vector<nb::object> effectObjects;
      std::vector<MlirMemoryEffectInstance> effects;
      for (nb::handle object : iterable) {
        PyMemoryEffectInstance *effect = nullptr;
        if (!nb::try_cast<PyMemoryEffectInstance *>(object, effect) ||
            !effect) {
          throw nb::type_error(
              "get_effects must return MemoryEffectInstance objects");
        }
        effectObjects.push_back(nb::borrow<nb::object>(object));
        effects.push_back(effect->get());
      }
      callback(effects.size(), effects.data(), callbackUserData);
    };

    mlirMemoryEffectsOpInterfaceAttachFallbackModel(
        ctx->get(), mlirStringRefCreate(opName.c_str(), opName.size()),
        callbacks);
  }

  std::vector<PyMemoryEffectInstance> getEffects() {
    if (isStatic())
      throw nb::type_error("Cannot query effects on a static interface");

    auto operationObject = getOperationObject();
    auto *operation = nb::cast<PyOperation *>(operationObject);
    std::vector<PyMemoryEffectInstance> effects;

    mlirMemoryEffectsOpInterfaceGetEffects(
        operation->get(),
        [](intptr_t numEffects, MlirMemoryEffectInstance *effects,
           void *userData) {
          auto *result =
              static_cast<std::vector<PyMemoryEffectInstance> *>(userData);
          result->reserve(result->size() + numEffects);
          for (intptr_t i = 0; i < numEffects; ++i) {
            result->emplace_back(mlirMemoryEffectInstanceClone(effects[i]));
          }
        },
        &effects);
    return effects;
  }

  static void bindDerived(ClassTy &cls) {
    cls.def("get_effects", &PyMemoryEffectsOpInterface::getEffects,
            nb::sig("def get_effects(self) -> list[MemoryEffectInstance]"),
            "Returns the memory effects of the operation.");
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
  nb::enum_<MlirSpeculatability>(m, "Speculatability")
      .value("NotSpeculatable", MlirSpeculatabilityNotSpeculatable)
      .value("Speculatable", MlirSpeculatabilitySpeculatable)
      .value("RecursivelySpeculatable",
             MlirSpeculatabilityRecursivelySpeculatable);
  nb::class_<PyMemoryEffect>(m, "MemoryEffect", "A memory effect.")
      .def(
          "__eq__",
          [](const PyMemoryEffect &self, const PyMemoryEffect &other) {
            return mlirTypeIDEqual(mlirMemoryEffectGetEffectID(self.get()),
                                   mlirMemoryEffectGetEffectID(other.get()));
          },
          nb::is_operator(), "Compares two memory effects for equality.")
      .def_prop_ro_static("Allocate",
                          [](nb::object & /*class*/) {
                            return PyMemoryEffect(
                                mlirMemoryEffectsAllocateGet());
                          })
      .def_prop_ro_static("Free",
                          [](nb::object & /*class*/) {
                            return PyMemoryEffect(mlirMemoryEffectsFreeGet());
                          })
      .def_prop_ro_static("Read",
                          [](nb::object & /*class*/) {
                            return PyMemoryEffect(mlirMemoryEffectsReadGet());
                          })
      .def_prop_ro_static("Write", [](nb::object & /*class*/) {
        return PyMemoryEffect(mlirMemoryEffectsWriteGet());
      });

  nb::class_<PySideEffectResource>(m, "SideEffectResource",
                                   "A side effect resource.")
      .def_prop_ro_static("Default", [](nb::object & /*class*/) {
        return PySideEffectResource(mlirSideEffectsDefaultResourceGet());
      });

  nb::class_<PyMemoryEffectInstance>(m, "MemoryEffectInstance",
                                     "A concrete instance of a memory effect.")
      .def(nb::init<const PyMemoryEffect &, const nb::object &,
                    const nb::object &, int, bool,
                    const PySideEffectResource &>(),
           nb::arg("effect"), nb::arg("target").none() = nb::none(),
           nb::kw_only(), nb::arg("parameters").none() = nb::none(),
           nb::arg("stage") = 0, nb::arg("effect_on_full_region") = false,
           nb::arg("resource") =
               PySideEffectResource(mlirSideEffectsDefaultResourceGet()),
           nb::sig("def __init__(self, effect: MemoryEffect, target: "
                   "OpOperand | OpResult | BlockArgument | SymbolRefAttr | "
                   "FlatSymbolRefAttr | None = None, *, parameters: Attribute "
                   "| None = None, stage: int = 0, "
                   "effect_on_full_region: bool = False, resource: "
                   "SideEffectResource = ...) -> None"),
           "Creates a memory effect instance. The target may be an OpOperand, "
           "OpResult, BlockArgument, SymbolRefAttr, or None.")
      .def_prop_ro("effect", &PyMemoryEffectInstance::getEffect,
                   "Returns the kind of memory effect.")
      .def_prop_ro("resource", &PyMemoryEffectInstance::getResource,
                   "Returns the affected side effect resource.")
      .def_prop_ro("stage", &PyMemoryEffectInstance::getStage,
                   "Returns the stage at which the effect occurs.")
      .def_prop_ro("effect_on_full_region",
                   &PyMemoryEffectInstance::getEffectOnFullRegion,
                   "Returns whether the effect applies to the full resource.")
      .def_prop_ro("parameters", &PyMemoryEffectInstance::getParameters,
                   nb::sig("def parameters(self) -> Attribute | None"),
                   "Returns the effect parameters, if any.")
      .def_prop_ro(
          "value", &PyMemoryEffectInstance::getValue,
          nb::sig("def value(self) -> OpResult | BlockArgument | None"),
          "Returns the affected value, if any.")
      .def_prop_ro("symbol_ref", &PyMemoryEffectInstance::getSymbolRef,
                   nb::sig("def symbol_ref(self) -> SymbolRefAttr | "
                           "FlatSymbolRefAttr | None"),
                   "Returns the affected symbol reference, if any.");

  PyConditionallySpeculatableOpInterface::bind(m);
  PyInferShapedTypeOpInterface::bind(m);
  PyInferTypeOpInterface::bind(m);
  PyMemoryEffectsOpInterface::bind(m);
  PyShapedTypeComponents::bind(m);
}
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir
