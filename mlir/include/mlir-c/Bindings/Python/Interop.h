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

#include <Python.h>

#include "mlir-c/IR.h"

#define MLIR_PYTHON_CAPSULE_CONTEXT "mlir.ir.Context._CAPIPtr"
#define MLIR_PYTHON_CAPSULE_MODULE "mlir.ir.Module._CAPIPtr"

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

#ifdef __cplusplus
extern "C" {
#endif

/** Creates a capsule object encapsulating the raw C-API MlirContext.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the context in any way.
 */
inline PyObject *mlirPythonContextToCapsule(MlirContext context) {
  return PyCapsule_New(context.ptr, MLIR_PYTHON_CAPSULE_CONTEXT, NULL);
}

/** Extracts a MlirContext from a capsule as produced from
 * mlirPythonContextToCapsule. If the capsule is not of the right type, then
 * a null context is returned (as checked via mlirContextIsNull). In such a
 * case, the Python APIs will have already set an error. */
inline MlirContext mlirPythonCapsuleToContext(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_CONTEXT);
  MlirContext context = {ptr};
  return context;
}

/** Creates a capsule object encapsulating the raw C-API MlirModule.
 * The returned capsule does not extend or affect ownership of any Python
 * objects that reference the module in any way. */
inline PyObject *mlirPythonModuleToCapsule(MlirModule module) {
#ifdef __cplusplus
  void *ptr = const_cast<void *>(module.ptr);
#else
  void *ptr = (void *)ptr;
#endif
  return PyCapsule_New(ptr, MLIR_PYTHON_CAPSULE_MODULE, NULL);
}

/** Extracts an MlirModule from a capsule as produced from
 * mlirPythonModuleToCapsule. If the capsule is not of the right type, then
 * a null module is returned (as checked via mlirModuleIsNull). In such a
 * case, the Python APIs will have already set an error. */
inline MlirModule mlirPythonCapsuleToModule(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_MODULE);
  MlirModule module = {ptr};
  return module;
}

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BINDINGS_PYTHON_INTEROP_H
