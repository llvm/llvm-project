//===- NanobindAdaptors.h - Interop with MLIR APIs via nanobind -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains adaptors for clients of the core MLIR Python APIs to
// interop via MLIR CAPI types, using nanobind. The facilities here do not
// depend on implementation details of the MLIR Python API and do not introduce
// C++-level dependencies with it (requiring only Python and CAPI-level
// dependencies).
//
// It is encouraged to be used both in-tree and out-of-tree. For in-tree use
// cases, it should be used for dialect implementations (versus relying on
// Pybind-based internals of the core libraries).
//===----------------------------------------------------------------------===//

#ifndef MLIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H
#define MLIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H

#include <cstdint>

#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir-c/Bindings/Python/Interop.h" // This is expected after nanobind.
#include "llvm/ADT/Twine.h"

// Raw CAPI type casters need to be declared before use, so always include them
// first.
namespace nanobind {
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
static nanobind::object mlirApiObjectToCapsule(nanobind::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return nanobind::borrow<nanobind::object>(apiObject);
  if (!nanobind::hasattr(apiObject, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    std::string repr = nanobind::cast<std::string>(nanobind::repr(apiObject));
    throw nanobind::type_error(
        (llvm::Twine("Expected an MLIR object (got ") + repr + ").")
            .str()
            .c_str());
  }
  return apiObject.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
}

// Note: Currently all of the following support cast from nanobind::object to
// the Mlir* C-API type, but only a few light-weight, context-bound ones
// implicitly cast the other way because the use case has not yet emerged and
// ownership is unclear.

/// Casts object <-> MlirAffineMap.
template <>
struct type_caster<MlirAffineMap> {
  NB_TYPE_CASTER(MlirAffineMap, const_name("MlirAffineMap"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAffineMap(capsule.ptr());
    if (mlirAffineMapIsNull(value)) {
      return false;
    }
    return !mlirAffineMapIsNull(value);
  }
  static handle from_cpp(MlirAffineMap v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonAffineMapToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("AffineMap")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirAttribute.
template <>
struct type_caster<MlirAttribute> {
  NB_TYPE_CASTER(MlirAttribute, const_name("MlirAttribute"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToAttribute(capsule.ptr());
    return !mlirAttributeIsNull(value);
  }
  static handle from_cpp(MlirAttribute v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonAttributeToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts object -> MlirBlock.
template <>
struct type_caster<MlirBlock> {
  NB_TYPE_CASTER(MlirBlock, const_name("MlirBlock"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToBlock(capsule.ptr());
    return !mlirBlockIsNull(value);
  }
};

/// Casts object -> MlirContext.
template <>
struct type_caster<MlirContext> {
  NB_TYPE_CASTER(MlirContext, const_name("MlirContext"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      // TODO: This raises an error of "No current context" currently.
      // Update the implementation to pretty-print the helpful error that the
      // core implementations print in this case.
      src = nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Context")
                .attr("current");
    }
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToContext(capsule.ptr());
    return !mlirContextIsNull(value);
  }
};

/// Casts object <-> MlirDialectRegistry.
template <>
struct type_caster<MlirDialectRegistry> {
  NB_TYPE_CASTER(MlirDialectRegistry, const_name("MlirDialectRegistry"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToDialectRegistry(capsule.ptr());
    return !mlirDialectRegistryIsNull(value);
  }
  static handle from_cpp(MlirDialectRegistry v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule = nanobind::steal<nanobind::object>(
        mlirPythonDialectRegistryToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("DialectRegistry")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirLocation.
template <>
struct type_caster<MlirLocation> {
  NB_TYPE_CASTER(MlirLocation, const_name("MlirLocation"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    if (src.is_none()) {
      // Gets the current thread-bound context.
      src = nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Location")
                .attr("current");
    }
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToLocation(capsule.ptr());
    return !mlirLocationIsNull(value);
  }
  static handle from_cpp(MlirLocation v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonLocationToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Location")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  }
};

/// Casts object <-> MlirModule.
template <>
struct type_caster<MlirModule> {
  NB_TYPE_CASTER(MlirModule, const_name("MlirModule"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToModule(capsule.ptr());
    return !mlirModuleIsNull(value);
  }
  static handle from_cpp(MlirModule v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonModuleToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Module")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirFrozenRewritePatternSet.
template <>
struct type_caster<MlirFrozenRewritePatternSet> {
  NB_TYPE_CASTER(MlirFrozenRewritePatternSet,
                 const_name("MlirFrozenRewritePatternSet"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToFrozenRewritePatternSet(capsule.ptr());
    return value.ptr != nullptr;
  }
  static handle from_cpp(MlirFrozenRewritePatternSet v, rv_policy, handle) {
    nanobind::object capsule = nanobind::steal<nanobind::object>(
        mlirPythonFrozenRewritePatternSetToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("rewrite"))
        .attr("FrozenRewritePatternSet")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirOperation.
template <>
struct type_caster<MlirOperation> {
  NB_TYPE_CASTER(MlirOperation, const_name("MlirOperation"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToOperation(capsule.ptr());
    return !mlirOperationIsNull(value);
  }
  static handle from_cpp(MlirOperation v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonOperationToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Operation")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirValue.
template <>
struct type_caster<MlirValue> {
  NB_TYPE_CASTER(MlirValue, const_name("MlirValue"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToValue(capsule.ptr());
    return !mlirValueIsNull(value);
  }
  static handle from_cpp(MlirValue v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonValueToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Value")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  };
};

/// Casts object -> MlirPassManager.
template <>
struct type_caster<MlirPassManager> {
  NB_TYPE_CASTER(MlirPassManager, const_name("MlirPassManager"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToPassManager(capsule.ptr());
    return !mlirPassManagerIsNull(value);
  }
};

/// Casts object <-> MlirTypeID.
template <>
struct type_caster<MlirTypeID> {
  NB_TYPE_CASTER(MlirTypeID, const_name("MlirTypeID"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToTypeID(capsule.ptr());
    return !mlirTypeIDIsNull(value);
  }
  static handle from_cpp(MlirTypeID v, rv_policy,
                         cleanup_list *cleanup) noexcept {
    if (v.ptr == nullptr)
      return nanobind::none();
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonTypeIDToCapsule(v));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("TypeID")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
  };
};

/// Casts object <-> MlirType.
template <>
struct type_caster<MlirType> {
  NB_TYPE_CASTER(MlirType, const_name("MlirType"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nanobind::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToType(capsule.ptr());
    return !mlirTypeIsNull(value);
  }
  static handle from_cpp(MlirType t, rv_policy,
                         cleanup_list *cleanup) noexcept {
    nanobind::object capsule =
        nanobind::steal<nanobind::object>(mlirPythonTypeToCapsule(t));
    return nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

} // namespace detail
} // namespace nanobind

namespace mlir {
namespace python {
namespace nanobind_adaptors {

/// Provides a facility like nanobind::class_ for defining a new class in a
/// scope, but this allows extension of an arbitrary Python class, defining
/// methods on it is a similar way. Classes defined in this way are very similar
/// to if defined in Python in the usual way but use nanobind machinery to
/// do it. These are not "real" nanobind classes but pure Python classes
/// with no relation to a concrete C++ class.
///
/// Derived from a discussion upstream:
///   https://github.com/pybind/pybind11/issues/1193
///   (plus a fair amount of extra curricular poking)
///   TODO: If this proves useful, see about including it in nanobind.
class pure_subclass {
public:
  pure_subclass(nanobind::handle scope, const char *derivedClassName,
                const nanobind::object &superClass) {
    nanobind::object pyType =
        nanobind::borrow<nanobind::object>((PyObject *)&PyType_Type);
    nanobind::object metaclass = pyType(superClass);
    nanobind::dict attributes;

    thisClass = metaclass(derivedClassName, nanobind::make_tuple(superClass),
                          attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &...extra) {
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f), nanobind::name(name), nanobind::is_method(),
        nanobind::scope(thisClass), extra...);
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &...extra) {
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f), nanobind::name(name), nanobind::is_method(),
        nanobind::scope(thisClass), extra...);
    auto builtinProperty =
        nanobind::borrow<nanobind::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_staticmethod(...) called with a non-static member "
                  "function pointer");
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f),
        nanobind::name(name), // nanobind::scope(thisClass),
        extra...);
    thisClass.attr(name) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_classmethod(const char *name, Func &&f,
                                 const Extra &...extra) {
    static_assert(!std::is_member_function_pointer<Func>::value,
                  "def_classmethod(...) called with a non-static member "
                  "function pointer");
    nanobind::object cf = nanobind::cpp_function(
        std::forward<Func>(f),
        nanobind::name(name), // nanobind::scope(thisClass),
        extra...);
    thisClass.attr(name) =
        nanobind::borrow<nanobind::object>(PyClassMethod_New(cf.ptr()));
    return *this;
  }

  nanobind::object get_class() const { return thisClass; }

protected:
  nanobind::object superClass;
  nanobind::object thisClass;
};

/// Creates a custom subclass of mlir.ir.Attribute, implementing a casting
/// constructor and type checking methods.
class mlir_attribute_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirAttribute);
  using GetTypeIDFunctionTy = MlirTypeID (*)();

  /// Subclasses by looking up the super-class dynamically.
  mlir_attribute_subclass(nanobind::handle scope, const char *attrClassName,
                          IsAFunctionTy isaFunction,
                          GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : mlir_attribute_subclass(
            scope, attrClassName, isaFunction,
            nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Attribute"),
            getTypeIDFunction) {}

  /// Subclasses with a provided mlir.ir.Attribute super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_attribute_subclass(nanobind::handle scope, const char *typeClassName,
                          IsAFunctionTy isaFunction,
                          const nanobind::object &superCls,
                          GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](
            nanobind::object cls, nanobind::object otherAttribute) {
          MlirAttribute rawAttribute =
              nanobind::cast<MlirAttribute>(otherAttribute);
          if (!isaFunction(rawAttribute)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherAttribute));
            throw std::invalid_argument(
                (llvm::Twine("Cannot cast attribute to ") + captureTypeName +
                 " (from " + origRepr + ")")
                    .str());
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherAttribute);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_attr"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirAttribute other) { return isaFunction(other); },
        nanobind::arg("other_attribute"));
    def("__repr__", [superCls, captureTypeName](nanobind::object self) {
      return nanobind::repr(superCls(self))
          .attr("replace")(superCls.attr("__name__"), captureTypeName);
    });
    if (getTypeIDFunction) {
      def_staticmethod("get_static_typeid",
                       [getTypeIDFunction]() { return getTypeIDFunction(); });
      nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
              getTypeIDFunction())(nanobind::cpp_function(
              [thisClass = thisClass](const nanobind::object &mlirAttribute) {
                return thisClass(mlirAttribute);
              }));
    }
  }
};

/// Creates a custom subclass of mlir.ir.Type, implementing a casting
/// constructor and type checking methods.
class mlir_type_subclass : public pure_subclass {
public:
  using IsAFunctionTy = bool (*)(MlirType);
  using GetTypeIDFunctionTy = MlirTypeID (*)();

  /// Subclasses by looking up the super-class dynamically.
  mlir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : mlir_type_subclass(
            scope, typeClassName, isaFunction,
            nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Type"),
            getTypeIDFunction) {}

  /// Subclasses with a provided mlir.ir.Type super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_type_subclass(nanobind::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction,
                     const nanobind::object &superCls,
                     GetTypeIDFunctionTy getTypeIDFunction = nullptr)
      : pure_subclass(scope, typeClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureTypeName](nanobind::object cls,
                                                 nanobind::object otherType) {
          MlirType rawType = nanobind::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherType));
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherType);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_type"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        nanobind::arg("other_type"));
    def("__repr__", [superCls, captureTypeName](nanobind::object self) {
      return nanobind::repr(superCls(self))
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
      nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
          .attr(MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR)(
              getTypeIDFunction())(nanobind::cpp_function(
              [thisClass = thisClass](const nanobind::object &mlirType) {
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
  mlir_value_subclass(nanobind::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction)
      : mlir_value_subclass(
            scope, valueClassName, isaFunction,
            nanobind::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Value")) {}

  /// Subclasses with a provided mlir.ir.Value super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_value_subclass(nanobind::handle scope, const char *valueClassName,
                      IsAFunctionTy isaFunction,
                      const nanobind::object &superCls)
      : pure_subclass(scope, valueClassName, superCls) {
    // Casting constructor. Note that it hard, if not impossible, to properly
    // call chain to parent `__init__` in nanobind due to its special handling
    // for init functions that don't have a fully constructed self-reference,
    // which makes it impossible to forward it to `__init__` of a superclass.
    // Instead, provide a custom `__new__` and call that of a superclass, which
    // eventually calls `__init__` of the superclass. Since attribute subclasses
    // have no additional members, we can just return the instance thus created
    // without amending it.
    std::string captureValueName(
        valueClassName); // As string in case if valueClassName is not static.
    nanobind::object newCf = nanobind::cpp_function(
        [superCls, isaFunction, captureValueName](nanobind::object cls,
                                                  nanobind::object otherValue) {
          MlirValue rawValue = nanobind::cast<MlirValue>(otherValue);
          if (!isaFunction(rawValue)) {
            auto origRepr =
                nanobind::cast<std::string>(nanobind::repr(otherValue));
            throw std::invalid_argument((llvm::Twine("Cannot cast value to ") +
                                         captureValueName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          nanobind::object self = superCls.attr("__new__")(cls, otherValue);
          return self;
        },
        nanobind::name("__new__"), nanobind::arg("cls"),
        nanobind::arg("cast_from_value"));
    thisClass.attr("__new__") = newCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirValue other) { return isaFunction(other); },
        nanobind::arg("other_value"));
  }
};

} // namespace nanobind_adaptors

} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_NANOBINDADAPTORS_H
