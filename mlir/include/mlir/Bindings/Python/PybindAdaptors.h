//===- PybindAdaptors.h - Adaptors for interop with MLIR APIs -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains adaptors for clients of the core MLIR Python APIs to
// interop via MLIR CAPI types. The facilities here do not depend on
// implementation details of the MLIR Python API and do not introduce C++-level
// dependencies with it (requiring only Python and CAPI-level dependencies).
//
// It is encouraged to be used both in-tree and out-of-tree. For in-tree use
// cases, it should be used for dialect implementations (versus relying on
// Pybind-based internals of the core libraries).
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_PYBINDADAPTORS_H
#define MLIR_BINDINGS_PYTHON_PYBINDADAPTORS_H

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"

#include "llvm/ADT/Twine.h"

namespace py = pybind11;
using namespace py::literals;

// Raw CAPI type casters need to be declared before use, so always include them
// first.
namespace pybind11 {
namespace detail {

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule (which can happen when two C APIs are communicating
/// directly via Python) or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute (through which supported MLIR Python API objects export their
/// contained API pointer as a capsule). Throws a type error if the object is
/// neither. This is intended to be used from type casters, which are invoked
/// with a raw handle (unowned). The returned object's lifetime may not extend
/// beyond the apiObject handle without explicitly having its refcount increased
/// (i.e. on return).
static py::object mlirApiObjectToCapsule(py::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return py::reinterpret_borrow<py::object>(apiObject);
  if (!py::hasattr(apiObject, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto repr = py::repr(apiObject).cast<std::string>();
    throw py::type_error(
        (llvm::Twine("Expected an MLIR object (got ") + repr + ").").str());
  }
  return apiObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

// Note: Currently all of the following support cast from py::object to the
// Mlir* C-API type, but only a few light-weight, context-bound ones
// implicitly cast the other way because the use case has not yet emerged and
// ownership is unclear.

/// Casts object <-> MlirAffineMap.
template <>
struct type_caster<MlirAffineMap> {
  PYBIND11_TYPE_CASTER(MlirAffineMap, _("MlirAffineMap"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAffineMap(capsule.ptr());
    if (mlirAffineMapIsNull(value)) {
      return false;
    }
    return !mlirAffineMapIsNull(value);
  }
  static handle cast(MlirAffineMap v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonAffineMapToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("AffineMap")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirAttribute.
template <>
struct type_caster<MlirAttribute> {
  PYBIND11_TYPE_CASTER(MlirAttribute, _("MlirAttribute"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAttribute(capsule.ptr());
    return !mlirAttributeIsNull(value);
  }
  static handle cast(MlirAttribute v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonAttributeToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts object -> MlirBlock.
template <>
struct type_caster<MlirBlock> {
  PYBIND11_TYPE_CASTER(MlirBlock, _("MlirBlock"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToBlock(capsule.ptr());
    return !mlirBlockIsNull(value);
  }
};

/// Casts object -> MlirContext.
template <>
struct type_caster<MlirContext> {
  PYBIND11_TYPE_CASTER(MlirContext, _("MlirContext"));
  bool load(handle src, bool) {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      // TODO: This raises an error of "No current context" currently.
      // Update the implementation to pretty-print the helpful error that the
      // core implementations print in this case.
      src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Context")
                .attr("current");
    }
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToContext(capsule.ptr());
    return !mlirContextIsNull(value);
  }
};

/// Casts object <-> MlirDialectRegistry.
template <>
struct type_caster<MlirDialectRegistry> {
  PYBIND11_TYPE_CASTER(MlirDialectRegistry, _("MlirDialectRegistry"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToDialectRegistry(capsule.ptr());
    return !mlirDialectRegistryIsNull(value);
  }
  static handle cast(MlirDialectRegistry v, return_value_policy, handle) {
    py::object capsule = py::reinterpret_steal<py::object>(
        mlirPythonDialectRegistryToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("DialectRegistry")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirLocation.
template <>
struct type_caster<MlirLocation> {
  PYBIND11_TYPE_CASTER(MlirLocation, _("MlirLocation"));
  bool load(handle src, bool) {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Location")
                .attr("current");
    }
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToLocation(capsule.ptr());
    return !mlirLocationIsNull(value);
  }
  static handle cast(MlirLocation v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonLocationToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Location")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirModule.
template <>
struct type_caster<MlirModule> {
  PYBIND11_TYPE_CASTER(MlirModule, _("MlirModule"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToModule(capsule.ptr());
    return !mlirModuleIsNull(value);
  }
  static handle cast(MlirModule v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonModuleToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Module")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirFrozenRewritePatternSet.
template <>
struct type_caster<MlirFrozenRewritePatternSet> {
  PYBIND11_TYPE_CASTER(MlirFrozenRewritePatternSet,
                       _("MlirFrozenRewritePatternSet"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToFrozenRewritePatternSet(capsule.ptr());
    return value.ptr != nullptr;
  }
  static handle cast(MlirFrozenRewritePatternSet v, return_value_policy,
                     handle) {
    py::object capsule = py::reinterpret_steal<py::object>(
        mlirPythonFrozenRewritePatternSetToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("rewrite"))
        .attr("FrozenRewritePatternSet")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirOperation.
template <>
struct type_caster<MlirOperation> {
  PYBIND11_TYPE_CASTER(MlirOperation, _("MlirOperation"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToOperation(capsule.ptr());
    return !mlirOperationIsNull(value);
  }
  static handle cast(MlirOperation v, return_value_policy, handle) {
    if (v.ptr == nullptr)
      return py::none();
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonOperationToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Operation")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirValue.
template <>
struct type_caster<MlirValue> {
  PYBIND11_TYPE_CASTER(MlirValue, _("MlirValue"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToValue(capsule.ptr());
    return !mlirValueIsNull(value);
  }
  static handle cast(MlirValue v, return_value_policy, handle) {
    if (v.ptr == nullptr)
      return py::none();
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonValueToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Value")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  };
};

/// Casts object -> MlirPassManager.
template <>
struct type_caster<MlirPassManager> {
  PYBIND11_TYPE_CASTER(MlirPassManager, _("MlirPassManager"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToPassManager(capsule.ptr());
    return !mlirPassManagerIsNull(value);
  }
};

/// Casts object <-> MlirTypeID.
template <>
struct type_caster<MlirTypeID> {
  PYBIND11_TYPE_CASTER(MlirTypeID, _("MlirTypeID"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToTypeID(capsule.ptr());
    return !mlirTypeIDIsNull(value);
  }
  static handle cast(MlirTypeID v, return_value_policy, handle) {
    if (v.ptr == nullptr)
      return py::none();
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonTypeIDToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("TypeID")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirType.
template <>
struct type_caster<MlirType> {
  PYBIND11_TYPE_CASTER(MlirType, _("MlirType"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToType(capsule.ptr());
    return !mlirTypeIsNull(value);
  }
  static handle cast(MlirType t, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonTypeToCapsule(t));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

} // namespace detail
} // namespace pybind11

namespace mlir {
namespace python {
namespace adaptors {

/// Provides a facility like py::class_ for defining a new class in a scope,
/// but this allows extension of an arbitrary Python class, defining methods
/// on it is a similar way. Classes defined in this way are very similar to
/// if defined in Python in the usual way but use Pybind11 machinery to do
/// it. These are not "real" Pybind11 classes but pure Python classes with no
/// relation to a concrete C++ class.
///
/// Derived from a discussion upstream:
///   https://github.com/pybind/pybind11/issues/1193
///   (plus a fair amount of extra curricular poking)
///   TODO: If this proves useful, see about including it in pybind11.
class pure_subclass {
public:
  pure_subclass(py::handle scope, const char *derivedClassName,
                const py::object &superClass) {
    py::object pyType =
        py::reinterpret_borrow<py::object>((PyObject *)&PyType_Type);
    py::object metaclass = pyType(superClass);
    py::dict attributes;

    thisClass =
        metaclass(derivedClassName, py::make_tuple(superClass), attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &...extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(thisClass),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &...extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(thisClass),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    auto builtinProperty =
        py::reinterpret_borrow<py::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_staticmethod(...) called with a non-static member "
                  "function pointer");
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::scope(thisClass),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) = py::staticmethod(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_classmethod(const char *name, Func &&f,
                                 const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_classmethod(...) called with a non-static member "
                  "function pointer");
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::scope(thisClass),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) =
        py::reinterpret_borrow<py::object>(PyClassMethod_New(cf.ptr()));
    return *this;
  }

  py::object get_class() const { return thisClass; }

protected:
  py::object superClass;
  py::object thisClass;
};

/// Creates a custom subclass of mlir.ir.Attribute, implementing a casting
/// constructor and type checking methods.
class mlir_attribute_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirAttribute);

  /// Subclasses by looking up the super-class dynamically.
  mlir_attribute_subclass(py::handle scope, const char *attrClassName,
                          IsAFunctionTy isaFunction)
      : mlir_attribute_subclass(
            scope, attrClassName, isaFunction,
            py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Attribute")) {}

  /// Subclasses with a provided mlir.ir.Attribute super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_attribute_subclass(py::handle scope, const char *typeClassName,
                          IsAFunctionTy isaFunction, const py::object &superCls)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in pybind11 due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    py::cpp_function newCf(
        [superCls, isaFunction, captureTypeName](py::object cls,
                                                 py::object otherAttribute) {
          MlirAttribute rawAttribute = py::cast<MlirAttribute>(otherAttribute);
          if (!isaFunction(rawAttribute)) {
            auto origRepr = py::repr(otherAttribute).cast<std::string>();
            throw std::invalid_argument(
                (llvm::Twine("Cannot cast attribute to ") + captureTypeName +
                 " (from " + origRepr + ")")
                    .str());
          }
          py::object self = superCls.attr("__new__")(cls, otherAttribute);
          return self;
        },
        py::name("__new__"), py::arg("cls"), py::arg("cast_from_attr"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirAttribute other) { return isaFunction(other); },
        py::arg("other_attribute"));
  }
};

/// Creates a custom subclass of mlir.ir.Type, implementing a casting
/// constructor and type checking methods.
class mlir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirType);
  using GetTypeIDFunctionTy = MlirTypeID (*)();

  /// Subclasses by looking up the super-class dynamically.
  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : mlir_type_subclass(
            scope, typeClassName, isaFunction,
            py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Type"),
            getTypeIDFunction) {}

  /// Subclasses with a provided mlir.ir.Type super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction, const py::object &superCls,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in pybind11 due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    py::cpp_function newCf(
        [superCls, isaFunction, captureTypeName](py::object cls,
                                                 py::object otherType) {
          MlirType rawType = py::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr = py::repr(otherType).cast<std::string>();
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          py::object self = superCls.attr("__new__")(cls, otherType);
          return self;
        },
        py::name("__new__"), py::arg("cls"), py::arg("cast_from_type"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        py::arg("other_type"));
    def("__repr__", [superCls, captureTypeName](py::object self) {
      return py::repr(superCls(self))
          .attr("replace")(superCls.attr("__name__"), captureTypeName);
    });
    if (getTypeIDFunction) {
      // 'get_static_typeid' method.
      // This is modeled as a static method instead of a static property because
      // `def_property_readonly_static` is not available in `pure_subclass` and
      // we do not want to introduce the complexity that pybind uses to
      // implement it.
      def_staticmethod("get_static_typeid",
                       [getTypeIDFunction]() { return getTypeIDFunction(); });
      py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
              getTypeIDFunction())(pybind11::cpp_function(
              [thisClass = thisClass](const py::object &mlirType) {
                return thisClass(mlirType);
              }));
    }
  }
};

/// Creates a custom subclass of mlir.ir.Value, implementing a casting
/// constructor and type checking methods.
class mlir_value_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirValue);

  /// Subclasses by looking up the super-class dynamically.
  mlir_value_subclass(py::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction)
      : mlir_value_subclass(
            scope, valueClassName, isaFunction,
            py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Value")) {
  }

  /// Subclasses with a provided mlir.ir.Value super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_value_subclass(py::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction, const py::object &superCls)
      : pure_subclass(scope, valueClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in pybind11 due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureValueName(
        valueClassName); // As string in case if valueClassName is not static.
    py::cpp_function newCf(
        [superCls, isaFunction, captureValueName](py::object cls,
                                                  py::object otherValue) {
          MlirValue rawValue = py::cast<MlirValue>(otherValue);
          if (!isaFunction(rawValue)) {
            auto origRepr = py::repr(otherValue).cast<std::string>();
            throw std::invalid_argument((llvm::Twine("Cannot cast value to ") +
                                         captureValueName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          py::object self = superCls.attr("__new__")(cls, otherValue);
          return self;
        },
        py::name("__new__"), py::arg("cls"), py::arg("cast_from_value"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirValue other) { return isaFunction(other); },
        py::arg("other_value"));
  }
};

} // namespace adaptors

/// RAII scope intercepting all diagnostics into a string. The message must be
/// checked before this goes out of scope.
class CollectDiagnosticsToStringScope {
public:
  explicit CollectDiagnosticsToStringScope(MlirContext ctx) : context(ctx) {
    handlerID = mlirContextAttachDiagnosticHandler(ctx, &handler, &errorMessage,
                                                   /*deleteUserData=*/nullptr);
  }
  ~CollectDiagnosticsToStringScope() {
    assert(errorMessage.empty() && "unchecked error message");
    mlirContextDetachDiagnosticHandler(context, handlerID);
  }

  [[nodiscard]] std::string takeMessage() { return std::move(errorMessage); }

private:
  static MlirLogicalResult handler(MlirDiagnostic diag, void *data) {
    auto printer = +[](MlirStringRef message, void *data) {
      *static_cast<std::string *>(data) +=
          llvm::StringRef(message.data, message.length);
    };
    MlirLocation loc = mlirDiagnosticGetLocation(diag);
    *static_cast<std::string *>(data) += "at ";
    mlirLocationPrint(loc, printer, data);
    *static_cast<std::string *>(data) += ": ";
    mlirDiagnosticPrint(diag, printer, data);
    return mlirLogicalResultSuccess();
  }

  MlirContext context;
  MlirDiagnosticHandlerID handlerID;
  std::string errorMessage = "";
};

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_PYBINDADAPTORS_H
