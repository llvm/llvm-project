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

#ifndef MLIR_BINDINGS_PYTHON_PYBIND_ADAPTORS_H
#define MLIR_BINDINGS_PYTHON_PYBIND_ADAPTORS_H

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"

namespace py = pybind11;

// Raw CAPI type casters need to be declared before use, so always include them
// first.
namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<llvm::Optional<T>> : optional_caster<llvm::Optional<T>> {};

/// Helper to convert a presumed MLIR API object to a capsule, accepting either
/// an explicit Capsule (which can happen when two C APIs are communicating
/// directly via Python) or indirectly by querying the MLIR_PYTHON_CAPI_PTR_ATTR
/// attribute (through which supported MLIR Python API objects export their
/// contained API pointer as a capsule). This is intended to be used from
/// type casters, which are invoked with a raw handle (unowned). The returned
/// object's lifetime may not extend beyond the apiObject handle without
/// explicitly having its refcount increased (i.e. on return).
static py::object mlirApiObjectToCapsule(py::handle apiObject) {
  if (PyCapsule_CheckExact(apiObject.ptr()))
    return py::reinterpret_borrow<py::object>(apiObject);
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
    if (mlirAttributeIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirAttribute v, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonAttributeToCapsule(v));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Attribute")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .release();
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
    if (mlirContextIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object <-> MlirLocation.
// TODO: Coerce None to default MlirLocation.
template <>
struct type_caster<MlirLocation> {
  PYBIND11_TYPE_CASTER(MlirLocation, _("MlirLocation"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToLocation(capsule.ptr());
    if (mlirLocationIsNull(value)) {
      return false;
    }
    return true;
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
    if (mlirModuleIsNull(value)) {
      return false;
    }
    return true;
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

/// Casts object <-> MlirOperation.
template <>
struct type_caster<MlirOperation> {
  PYBIND11_TYPE_CASTER(MlirOperation, _("MlirOperation"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToOperation(capsule.ptr());
    if (mlirOperationIsNull(value)) {
      return false;
    }
    return true;
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

/// Casts object -> MlirPassManager.
template <>
struct type_caster<MlirPassManager> {
  PYBIND11_TYPE_CASTER(MlirPassManager, _("MlirPassManager"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToPassManager(capsule.ptr());
    if (mlirPassManagerIsNull(value)) {
      return false;
    }
    return true;
  }
};

/// Casts object <-> MlirType.
template <>
struct type_caster<MlirType> {
  PYBIND11_TYPE_CASTER(MlirType, _("MlirType"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = mlirPythonCapsuleToType(capsule.ptr());
    if (mlirTypeIsNull(value)) {
      return false;
    }
    return true;
  }
  static handle cast(MlirType t, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonTypeToCapsule(t));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
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
                py::object superClass) {
    py::object pyType =
        py::reinterpret_borrow<py::object>((PyObject *)&PyType_Type);
    py::object metaclass = pyType(superClass);
    py::dict attributes;

    thisClass =
        metaclass(derivedClassName, py::make_tuple(superClass), attributes);
    scope.attr(derivedClassName) = thisClass;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def(const char *name, Func &&f, const Extra &... extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(py::none()),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    thisClass.attr(cf.name()) = cf;
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_property_readonly(const char *name, Func &&f,
                                       const Extra &... extra) {
    py::cpp_function cf(
        std::forward<Func>(f), py::name(name), py::is_method(py::none()),
        py::sibling(py::getattr(thisClass, name, py::none())), extra...);
    auto builtinProperty =
        py::reinterpret_borrow<py::object>((PyObject *)&PyProperty_Type);
    thisClass.attr(name) = builtinProperty(cf);
    return *this;
  }

  template <typename Func, typename... Extra>
  pure_subclass &def_staticmethod(const char *name, Func &&f,
                                  const Extra &... extra) {
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
                                 const Extra &... extra) {
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
                          IsAFunctionTy isaFunction, py::object superClass)
      : pure_subclass(scope, typeClassName, superClass) {
    // Casting constructor. Note that defining an __init__ method is special
    // and not yet generalized on pure_subclass (it requires a somewhat
    // different cpp_function and other requirements on chaining to super
    // __init__ make it more awkward to do generally).
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    py::cpp_function initCf(
        [superClass, isaFunction, captureTypeName](py::object self,
                                                   py::object otherType) {
          MlirAttribute rawAttribute = py::cast<MlirAttribute>(otherType);
          if (!isaFunction(rawAttribute)) {
            auto origRepr = py::repr(otherType).cast<std::string>();
            throw std::invalid_argument(
                (llvm::Twine("Cannot cast attribute to ") + captureTypeName +
                 " (from " + origRepr + ")")
                    .str());
          }
          superClass.attr("__init__")(self, otherType);
        },
        py::arg("cast_from_type"), py::is_method(py::none()),
        "Casts the passed type to this specific sub-type.");
    thisClass.attr("__init__") = initCf;

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

  /// Subclasses by looking up the super-class dynamically.
  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction)
      : mlir_type_subclass(
            scope, typeClassName, isaFunction,
            py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir")).attr("Type")) {}

  /// Subclasses with a provided mlir.ir.Type super-class. This must
  /// be used if the subclass is being defined in the same extension module
  /// as the mlir.ir class (otherwise, it will trigger a recursive
  /// initialization).
  mlir_type_subclass(py::handle scope, const char *typeClassName,
                     IsAFunctionTy isaFunction, py::object superClass)
      : pure_subclass(scope, typeClassName, superClass) {
    // Casting constructor. Note that defining an __init__ method is special
    // and not yet generalized on pure_subclass (it requires a somewhat
    // different cpp_function and other requirements on chaining to super
    // __init__ make it more awkward to do generally).
    std::string captureTypeName(
        typeClassName); // As string in case if typeClassName is not static.
    py::cpp_function initCf(
        [superClass, isaFunction, captureTypeName](py::object self,
                                                   py::object otherType) {
          MlirType rawType = py::cast<MlirType>(otherType);
          if (!isaFunction(rawType)) {
            auto origRepr = py::repr(otherType).cast<std::string>();
            throw std::invalid_argument((llvm::Twine("Cannot cast type to ") +
                                         captureTypeName + " (from " +
                                         origRepr + ")")
                                            .str());
          }
          superClass.attr("__init__")(self, otherType);
        },
        py::arg("cast_from_type"), py::is_method(py::none()),
        "Casts the passed type to this specific sub-type.");
    thisClass.attr("__init__") = initCf;

    // 'isinstance' method.
    def_staticmethod(
        "isinstance",
        [isaFunction](MlirType other) { return isaFunction(other); },
        py::arg("other_type"));
  }
};

} // namespace adaptors
} // namespace python
} // namespace mlir

#endif // MLIR_BINDINGS_PYTHON_PYBIND_ADAPTORS_H
