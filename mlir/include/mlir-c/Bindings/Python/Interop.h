//===-- mlir-c/Interop.h - Constants for Python/C-API interop -----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares constants and helpers necessary for C-level
// interop with the MLIR Python extension module. Since the Python bindings
// are a thin wrapper around the MLIR C-API, a further C-API is not provided
// specifically for the Python extension. Instead, simple facilities are
// provided for translating between Python types and corresponding MLIR C-API
// types.
//
// This header is standalone, requiring nothing beyond normal linking against
// the Python implementation.
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_BINDINGS_PYTHON_INTEROP_H
#define MLIR_C_BINDINGS_PYTHON_INTEROP_H

// We *should*, in theory, include Python.h here in order to import the correct
// definitions for what we need below, however, importing Python.h directly on
// Windows results in the enforcement of either pythonX.lib or pythonX_d.lib
// depending on the build flavor. Instead, we rely on the fact that this file
// (Interop.h) is always included AFTER pybind11 and will therefore have access
// to the definitions from Python.h in addition to having a workaround applied
// through the pybind11 headers that allows us to control which python library
// is used.
#if !defined(_MSC_VER)
#include <Python.h>
#endif

#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/ExecutionEngine.h"
#include "mlir-c/IR.h"
#include "mlir-c/IntegerSet.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Rewrite.h"

// The 'mlir' Python package is relocatable and supports co-existing in multiple
// projects. Each project must define its outer package prefix with this define
// in order to provide proper isolation and local name resolution.
// The default is for the upstream "import mlir" package layout.
// Note that this prefix is internally stringified, allowing it to be passed
// unquoted on the compiler command line without shell quote escaping issues.
#ifndef MLIR_PYTHON_PACKAGE_PREFIX
#define MLIR_PYTHON_PACKAGE_PREFIX mlir.
#endif

// Makes a fully-qualified name relative to the MLIR python package.
#define MLIR_PYTHON_STRINGIZE(s) #s
#define MLIR_PYTHON_STRINGIZE_ARG(arg) MLIR_PYTHON_STRINGIZE(arg)
#define MAKE_MLIR_PYTHON_QUALNAME(local)                                       \
  MLIR_PYTHON_STRINGIZE_ARG(MLIR_PYTHON_PACKAGE_PREFIX) local

#define MLIR_PYTHON_CAPSULE_AFFINE_EXPR                                        \
  MAKE_MLIR_PYTHON_QUALNAME("ir.AffineExpr._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_AFFINE_MAP                                         \
  MAKE_MLIR_PYTHON_QUALNAME("ir.AffineMap._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_ATTRIBUTE                                          \
  MAKE_MLIR_PYTHON_QUALNAME("ir.Attribute._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_BLOCK MAKE_MLIR_PYTHON_QUALNAME("ir.Block._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_CONTEXT                                            \
  MAKE_MLIR_PYTHON_QUALNAME("ir.Context._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_DIALECT_REGISTRY                                   \
  MAKE_MLIR_PYTHON_QUALNAME("ir.DialectRegistry._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE                                   \
  MAKE_MLIR_PYTHON_QUALNAME("execution_engine.ExecutionEngine._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_INTEGER_SET                                        \
  MAKE_MLIR_PYTHON_QUALNAME("ir.IntegerSet._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_LOCATION                                           \
  MAKE_MLIR_PYTHON_QUALNAME("ir.Location._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_MODULE                                             \
  MAKE_MLIR_PYTHON_QUALNAME("ir.Module._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_OPERATION                                          \
  MAKE_MLIR_PYTHON_QUALNAME("ir.Operation._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_TYPE MAKE_MLIR_PYTHON_QUALNAME("ir.Type._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_PASS_MANAGER                                       \
  MAKE_MLIR_PYTHON_QUALNAME("passmanager.PassManager._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_VALUE MAKE_MLIR_PYTHON_QUALNAME("ir.Value._CAPIPtr")
#define MLIR_PYTHON_CAPSULE_TYPEID                                             \
  MAKE_MLIR_PYTHON_QUALNAME("ir.TypeID._CAPIPtr")

/** Attribute on MLIR Python objects that expose their C-API pointer.
 * This will be a type-specific capsule created as per one of the helpers
 * below.
 *
 * Ownership is not transferred by acquiring a capsule in this way: the
 * validity of the pointer wrapped by the capsule will be bounded by the
 * lifetime of the Python object that produced it. Only the name and pointer
 * of the capsule are set. The caller is free to set a destructor and context
 * as needed to manage anything further. */
#define MLIR_PYTHON_CAPI_PTR_ATTR "_CAPIPtr"

/** Attribute on MLIR Python objects that exposes a factory function for
 * constructing the corresponding Python object from a type-specific
 * capsule wrapping the C-API pointer. The signature of the function is:
 *   def _CAPICreate(capsule) -> object
 * Calling such a function implies a transfer of ownership of the object the
 * capsule wraps: after such a call, the capsule should be considered invalid,
 * and its wrapped pointer must not be destroyed.
 *
 * Only a very small number of Python objects can be created in such a fashion
 * (i.e. top-level types such as Context where the lifetime can be cleanly
 * delineated). */
#define MLIR_PYTHON_CAPI_FACTORY_ATTR "_CAPICreate"

/** Attribute on MLIR Python objects that expose a function for downcasting the
 * corresponding Python object to a subclass if the object is in fact a subclass
 * (Concrete or mlir_type_subclass) of ir.Type. The signature of the function
 * is: def maybe_downcast(self) -> object where the resulting object will
 * (possibly) be an instance of the subclass.
 */
#define MLIR_PYTHON_MAYBE_DOWNCAST_ATTR "maybe_downcast"

/** Attribute on main C extension module (_mlir) that corresponds to the
 * type caster registration binding. The signature of the function is:
 *   def register_type_caster(MlirTypeID mlirTypeID, *, bool replace)
 * which then takes a typeCaster (register_type_caster is meant to be used as a
 * decorator from python), and where replace indicates the typeCaster should
 * replace any existing registered type casters (such as those for upstream
 * ConcreteTypes). The interface of the typeCaster is: def type_caster(ir.Type)
 * -> SubClassTypeT where SubClassTypeT indicates the result should be a
 * subclass (inherit from) ir.Type.
 */
#define MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR "register_type_caster"

/** Attribute on main C extension module (_mlir) that corresponds to the
 * value caster registration binding. The signature of the function is:
 *   def register_value_caster(MlirTypeID mlirTypeID, *, bool replace)
 * which then takes a valueCaster (register_value_caster is meant to be used as
 * a decorator, from python), and where replace indicates the valueCaster should
 * replace any existing registered value casters. The interface of the
 * valueCaster is: def value_caster(ir.Value) -> SubClassValueT where
 * SubClassValueT indicates the result should be a subclass (inherit from)
 * ir.Value.
 */
#define MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR "register_value_caster"

/// Gets a void* from a wrapped struct. Needed because const cast is different
/// between C/C++.
#ifdef __cplusplus
#define MLIR_PYTHON_GET_WRAPPED_POINTER(object)                                \
  (const_cast<void *>((object).ptr))
#else
#define MLIR_PYTHON_GET_WRAPPED_POINTER(object) (void *)(object.ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Creates a capsule object encapsulating the raw C-API MlirAffineExpr. The
 * returned capsule does not extend or affect ownership of any Python objects
 * that reference the expression in any way.
 */
static inline PyObject *mlirPythonAffineExprToCapsule(MlirAffineExpr expr) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(expr),
                       MLIR_PYTHON_CAPSULE_AFFINE_EXPR, NULL);
}

/** Extracts an MlirAffineExpr from a capsule as produced from
 * mlirPythonAffineExprToCapsule. If the capsule is not of the right type, then
 * a null expression is returned (as checked via mlirAffineExprIsNull). In such
 * a case, the Python APIs will have already set an error. */
static inline MlirAffineExpr mlirPythonCapsuleToAffineExpr(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_AFFINE_EXPR);
  MlirAffineExpr expr = {ptr};
  return expr;
}

/** Creates a capsule object encapsulating the raw C-API MlirAttribute.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the attribute in any way.
 */
static inline PyObject *mlirPythonAttributeToCapsule(MlirAttribute attribute) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(attribute),
                       MLIR_PYTHON_CAPSULE_ATTRIBUTE, NULL);
}

/** Extracts an MlirAttribute from a capsule as produced from
 * mlirPythonAttributeToCapsule. If the capsule is not of the right type, then
 * a null attribute is returned (as checked via mlirAttributeIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirAttribute mlirPythonCapsuleToAttribute(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_ATTRIBUTE);
  MlirAttribute attr = {ptr};
  return attr;
}

/** Creates a capsule object encapsulating the raw C-API MlirBlock.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *mlirPythonBlockToCapsule(MlirBlock block) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(block),
                       MLIR_PYTHON_CAPSULE_BLOCK, NULL);
}

/** Extracts an MlirBlock from a capsule as produced from
 * mlirPythonBlockToCapsule. If the capsule is not of the right type, then
 * a null pass manager is returned (as checked via mlirBlockIsNull). */
static inline MlirBlock mlirPythonCapsuleToBlock(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_BLOCK);
  MlirBlock block = {ptr};
  return block;
}

/** Creates a capsule object encapsulating the raw C-API MlirContext.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the context in any way.
 */
static inline PyObject *mlirPythonContextToCapsule(MlirContext context) {
  return PyCapsule_New(context.ptr, MLIR_PYTHON_CAPSULE_CONTEXT, NULL);
}

/** Extracts a MlirContext from a capsule as produced from
 * mlirPythonContextToCapsule. If the capsule is not of the right type, then
 * a null context is returned (as checked via mlirContextIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirContext mlirPythonCapsuleToContext(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_CONTEXT);
  MlirContext context = {ptr};
  return context;
}

/** Creates a capsule object encapsulating the raw C-API MlirDialectRegistry.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the context in any way.
 */
static inline PyObject *
mlirPythonDialectRegistryToCapsule(MlirDialectRegistry registry) {
  return PyCapsule_New(registry.ptr, MLIR_PYTHON_CAPSULE_DIALECT_REGISTRY,
                       NULL);
}

/** Extracts an MlirDialectRegistry from a capsule as produced from
 * mlirPythonDialectRegistryToCapsule. If the capsule is not of the right type,
 * then a null context is returned (as checked via mlirContextIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirDialectRegistry
mlirPythonCapsuleToDialectRegistry(PyObject *capsule) {
  void *ptr =
      PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_DIALECT_REGISTRY);
  MlirDialectRegistry registry = {ptr};
  return registry;
}

/** Creates a capsule object encapsulating the raw C-API MlirLocation.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the location in any way. */
static inline PyObject *mlirPythonLocationToCapsule(MlirLocation loc) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(loc),
                       MLIR_PYTHON_CAPSULE_LOCATION, NULL);
}

/** Extracts an MlirLocation from a capsule as produced from
 * mlirPythonLocationToCapsule. If the capsule is not of the right type, then
 * a null module is returned (as checked via mlirLocationIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirLocation mlirPythonCapsuleToLocation(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_LOCATION);
  MlirLocation loc = {ptr};
  return loc;
}

/** Creates a capsule object encapsulating the raw C-API MlirModule.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *mlirPythonModuleToCapsule(MlirModule module) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(module),
                       MLIR_PYTHON_CAPSULE_MODULE, NULL);
}

/** Extracts an MlirModule from a capsule as produced from
 * mlirPythonModuleToCapsule. If the capsule is not of the right type, then
 * a null module is returned (as checked via mlirModuleIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirModule mlirPythonCapsuleToModule(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_MODULE);
  MlirModule module = {ptr};
  return module;
}

/** Creates a capsule object encapsulating the raw C-API
 * MlirFrozenRewritePatternSet.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *
mlirPythonFrozenRewritePatternSetToCapsule(MlirFrozenRewritePatternSet pm) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(pm),
                       MLIR_PYTHON_CAPSULE_PASS_MANAGER, NULL);
}

/** Extracts an MlirFrozenRewritePatternSet from a capsule as produced from
 * mlirPythonFrozenRewritePatternSetToCapsule. If the capsule is not of the
 * right type, then a null module is returned. */
static inline MlirFrozenRewritePatternSet
mlirPythonCapsuleToFrozenRewritePatternSet(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_PASS_MANAGER);
  MlirFrozenRewritePatternSet pm = {ptr};
  return pm;
}

/** Creates a capsule object encapsulating the raw C-API MlirPassManager.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
static inline PyObject *mlirPythonPassManagerToCapsule(MlirPassManager pm) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(pm),
                       MLIR_PYTHON_CAPSULE_PASS_MANAGER, NULL);
}

/** Extracts an MlirPassManager from a capsule as produced from
 * mlirPythonPassManagerToCapsule. If the capsule is not of the right type, then
 * a null pass manager is returned (as checked via mlirPassManagerIsNull). */
static inline MlirPassManager
mlirPythonCapsuleToPassManager(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_PASS_MANAGER);
  MlirPassManager pm = {ptr};
  return pm;
}

/** Creates a capsule object encapsulating the raw C-API MlirOperation.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the operation in any way.
 */
static inline PyObject *mlirPythonOperationToCapsule(MlirOperation operation) {
  return PyCapsule_New(operation.ptr, MLIR_PYTHON_CAPSULE_OPERATION, NULL);
}

/** Extracts an MlirOperations from a capsule as produced from
 * mlirPythonOperationToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via mlirOperationIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirOperation mlirPythonCapsuleToOperation(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_OPERATION);
  MlirOperation op = {ptr};
  return op;
}

/** Creates a capsule object encapsulating the raw C-API MlirTypeID.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *mlirPythonTypeIDToCapsule(MlirTypeID typeID) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(typeID),
                       MLIR_PYTHON_CAPSULE_TYPEID, NULL);
}

/** Extracts an MlirTypeID from a capsule as produced from
 * mlirPythonTypeIDToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via mlirTypeIDIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirTypeID mlirPythonCapsuleToTypeID(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_TYPEID);
  MlirTypeID typeID = {ptr};
  return typeID;
}

/** Creates a capsule object encapsulating the raw C-API MlirType.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *mlirPythonTypeToCapsule(MlirType type) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(type),
                       MLIR_PYTHON_CAPSULE_TYPE, NULL);
}

/** Extracts an MlirType from a capsule as produced from
 * mlirPythonTypeToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via mlirTypeIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirType mlirPythonCapsuleToType(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_TYPE);
  MlirType type = {ptr};
  return type;
}

/** Creates a capsule object encapsulating the raw C-API MlirAffineMap.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the type in any way.
 */
static inline PyObject *mlirPythonAffineMapToCapsule(MlirAffineMap affineMap) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(affineMap),
                       MLIR_PYTHON_CAPSULE_AFFINE_MAP, NULL);
}

/** Extracts an MlirAffineMap from a capsule as produced from
 * mlirPythonAffineMapToCapsule. If the capsule is not of the right type, then
 * a null type is returned (as checked via mlirAffineMapIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirAffineMap mlirPythonCapsuleToAffineMap(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_AFFINE_MAP);
  MlirAffineMap affineMap = {ptr};
  return affineMap;
}

/** Creates a capsule object encapsulating the raw C-API MlirIntegerSet.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the set in any way. */
static inline PyObject *
mlirPythonIntegerSetToCapsule(MlirIntegerSet integerSet) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(integerSet),
                       MLIR_PYTHON_CAPSULE_INTEGER_SET, NULL);
}

/** Extracts an MlirIntegerSet from a capsule as produced from
 * mlirPythonIntegerSetToCapsule. If the capsule is not of the right type, then
 * a null set is returned (as checked via mlirIntegerSetIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirIntegerSet mlirPythonCapsuleToIntegerSet(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_INTEGER_SET);
  MlirIntegerSet integerSet = {ptr};
  return integerSet;
}

/** Creates a capsule object encapsulating the raw C-API MlirExecutionEngine.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the set in any way. */
static inline PyObject *
mlirPythonExecutionEngineToCapsule(MlirExecutionEngine jit) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(jit),
                       MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE, NULL);
}

/** Extracts an MlirExecutionEngine from a capsule as produced from
 * mlirPythonIntegerSetToCapsule. If the capsule is not of the right type, then
 * a null set is returned (as checked via mlirExecutionEngineIsNull). In such a
 * case, the Python APIs will have already set an error. */
static inline MlirExecutionEngine
mlirPythonCapsuleToExecutionEngine(PyObject *capsule) {
  void *ptr =
      PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_EXECUTION_ENGINE);
  MlirExecutionEngine jit = {ptr};
  return jit;
}

/** Creates a capsule object encapsulating the raw C-API MlirValue.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the operation in any way.
 */
static inline PyObject *mlirPythonValueToCapsule(MlirValue value) {
  return PyCapsule_New(MLIR_PYTHON_GET_WRAPPED_POINTER(value),
                       MLIR_PYTHON_CAPSULE_VALUE, NULL);
}

/** Extracts an MlirValue from a capsule as produced from
 * mlirPythonValueToCapsule. If the capsule is not of the right type, then a
 * null type is returned (as checked via mlirValueIsNull). In such a case, the
 * Python APIs will have already set an error. */
static inline MlirValue mlirPythonCapsuleToValue(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_VALUE);
  MlirValue value = {ptr};
  return value;
}

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BINDINGS_PYTHON_INTEROP_H
