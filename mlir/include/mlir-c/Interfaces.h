//===-- mlir-c/Interfaces.h - C API to Core MLIR IR interfaces ----*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR interface classes. It is
// intended to contain interfaces defined in lib/Interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_INTERFACES_H
#define MLIR_C_INTERFACES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirMemoryEffect, void);
DEFINE_C_API_STRUCT(MlirMemoryEffectInstance, void);
DEFINE_C_API_STRUCT(MlirMemoryEffectInstancesList, void);
DEFINE_C_API_STRUCT(MlirSideEffectResource, void);

#undef DEFINE_C_API_STRUCT

/// Returns `true` if the given operation implements an interface identified by
/// its TypeID.
MLIR_CAPI_EXPORTED bool
mlirOperationImplementsInterface(MlirOperation operation,
                                 MlirTypeID interfaceTypeID);

/// Returns `true` if the operation identified by its canonical string name
/// implements the interface identified by its TypeID in the given context.
/// Note that interfaces may be attached to operations in some contexts and not
/// others.
MLIR_CAPI_EXPORTED bool
mlirOperationImplementsInterfaceStatic(MlirStringRef operationName,
                                       MlirContext context,
                                       MlirTypeID interfaceTypeID);

//===----------------------------------------------------------------------===//
// InferTypeOpInterface.
//===----------------------------------------------------------------------===//

/// Returns the interface TypeID of the InferTypeOpInterface.
MLIR_CAPI_EXPORTED MlirTypeID mlirInferTypeOpInterfaceTypeID(void);

/// These callbacks are used to return multiple types from functions while
/// transferring ownership to the caller. The first argument is the number of
/// consecutive elements pointed to by the second argument. The third argument
/// is an opaque pointer forwarded to the callback by the caller.
typedef void (*MlirTypesCallback)(intptr_t, MlirType *, void *);

/// Infers the return types of the operation identified by its canonical given
/// the arguments that will be supplied to its generic builder. Calls `callback`
/// with the types of inferred arguments, potentially several times, on success.
/// Returns failure otherwise.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirInferTypeOpInterfaceInferReturnTypes(
    MlirStringRef opName, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    void *properties, intptr_t nRegions, MlirRegion *regions,
    MlirTypesCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// InferShapedTypeOpInterface.
//===----------------------------------------------------------------------===//

/// Returns the interface TypeID of the InferShapedTypeOpInterface.
MLIR_CAPI_EXPORTED MlirTypeID mlirInferShapedTypeOpInterfaceTypeID(void);

/// These callbacks are used to return multiple shaped type components from
/// functions while transferring ownership to the caller. The first argument is
/// the has rank boolean followed by the the rank and a pointer to the shape
/// (if applicable). The next argument is the element type, then the attribute.
/// The last argument is an opaque pointer forwarded to the callback by the
/// caller. This callback will be called potentially multiple times for each
/// shaped type components.
typedef void (*MlirShapedTypeComponentsCallback)(bool, intptr_t,
                                                 const int64_t *, MlirType,
                                                 MlirAttribute, void *);

/// Infers the return shaped type components of the operation. Calls `callback`
/// with the types of inferred arguments on success. Returns failure otherwise.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirInferShapedTypeOpInterfaceInferReturnTypes(
    MlirStringRef opName, MlirContext context, MlirLocation location,
    intptr_t nOperands, MlirValue *operands, MlirAttribute attributes,
    void *properties, intptr_t nRegions, MlirRegion *regions,
    MlirShapedTypeComponentsCallback callback, void *userData);

//===---------------------------------------------------------------------===//
// ConditionallySpeculatable
//===---------------------------------------------------------------------===//

/// Enum representing the speculatability of an operation.
typedef enum {
  /// The operation is not speculatable.
  MlirSpeculatabilityNotSpeculatable,
  /// The operation is speculatable.
  MlirSpeculatabilitySpeculatable,
  /// The operation is speculatable if all nested operations are speculatable.
  MlirSpeculatabilityRecursivelySpeculatable
} MlirSpeculatability;

/// Returns the interface TypeID of the ConditionallySpeculatable interface.
MLIR_CAPI_EXPORTED MlirTypeID
mlirConditionallySpeculatableOpInterfaceTypeID(void);

/// Callbacks for implementing ConditionallySpeculatable from external code.
typedef struct {
  /// Optional constructor for user data. Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for user data. Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Returns the speculatability of the given operation.
  MlirSpeculatability (*getSpeculatability)(MlirOperation op, void *userData);
  void *userData;
} MlirConditionallySpeculatableOpInterfaceCallbacks;

/// Attach a new FallbackModel for the ConditionallySpeculatable interface to
/// the named operation. The FallbackModel will call the provided callbacks.
MLIR_CAPI_EXPORTED void
mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(
    MlirContext ctx, MlirStringRef opName,
    MlirConditionallySpeculatableOpInterfaceCallbacks callbacks);

/// Returns the speculatability of the given operation.
///
/// The operation must implement the ConditionallySpeculatable interface.
MLIR_CAPI_EXPORTED MlirSpeculatability
mlirConditionallySpeculatableOpInterfaceGetSpeculatability(
    MlirOperation operation);

//===---------------------------------------------------------------------===//
// MemoryEffectsOpInterface
//===---------------------------------------------------------------------===//

/// Returns the borrowed singleton instance of the allocate memory effect.
MLIR_CAPI_EXPORTED MlirMemoryEffect mlirMemoryEffectsAllocateGet(void);

/// Returns the borrowed singleton instance of the free memory effect.
MLIR_CAPI_EXPORTED MlirMemoryEffect mlirMemoryEffectsFreeGet(void);

/// Returns the borrowed singleton instance of the read memory effect.
MLIR_CAPI_EXPORTED MlirMemoryEffect mlirMemoryEffectsReadGet(void);

/// Returns the borrowed singleton instance of the write memory effect.
MLIR_CAPI_EXPORTED MlirMemoryEffect mlirMemoryEffectsWriteGet(void);

/// Returns the borrowed singleton instance of the default side effect
/// resource.
MLIR_CAPI_EXPORTED MlirSideEffectResource
mlirSideEffectsDefaultResourceGet(void);

/// Creates a memory effect instance without an associated IR entity.
/// `parameters` may be a null attribute. The caller owns the returned instance
/// and must destroy it with `mlirMemoryEffectInstanceDestroy`.
MLIR_CAPI_EXPORTED MlirMemoryEffectInstance mlirMemoryEffectInstanceCreate(
    MlirMemoryEffect effect, MlirAttribute parameters, int stage,
    bool effectOnFullRegion, MlirSideEffectResource resource);

/// Creates a memory effect instance associated with an operation operand.
/// `parameters` may be a null attribute. The caller owns the returned instance
/// and must destroy it with `mlirMemoryEffectInstanceDestroy`.
MLIR_CAPI_EXPORTED MlirMemoryEffectInstance
mlirMemoryEffectInstanceCreateForOpOperand(MlirMemoryEffect effect,
                                           MlirOpOperand opOperand,
                                           MlirAttribute parameters, int stage,
                                           bool effectOnFullRegion,
                                           MlirSideEffectResource resource);

/// Creates a memory effect instance associated with an operation result.
/// `result` must wrap an OpResult. `parameters` may be a null attribute. The
/// caller owns the returned instance and must destroy it with
/// `mlirMemoryEffectInstanceDestroy`.
MLIR_CAPI_EXPORTED MlirMemoryEffectInstance
mlirMemoryEffectInstanceCreateForOpResult(MlirMemoryEffect effect,
                                          MlirValue result,
                                          MlirAttribute parameters, int stage,
                                          bool effectOnFullRegion,
                                          MlirSideEffectResource resource);

/// Creates a memory effect instance associated with a block argument.
/// `blockArgument` must wrap a BlockArgument. `parameters` may be a null
/// attribute. The caller owns the returned instance and must destroy it with
/// `mlirMemoryEffectInstanceDestroy`.
MLIR_CAPI_EXPORTED MlirMemoryEffectInstance
mlirMemoryEffectInstanceCreateForBlockArgument(
    MlirMemoryEffect effect, MlirValue blockArgument, MlirAttribute parameters,
    int stage, bool effectOnFullRegion, MlirSideEffectResource resource);

/// Creates a memory effect instance associated with a symbol. `symbol` must be
/// a SymbolRefAttr. `parameters` may be a null attribute. The caller owns the
/// returned instance and must destroy it with
/// `mlirMemoryEffectInstanceDestroy`.
MLIR_CAPI_EXPORTED MlirMemoryEffectInstance
mlirMemoryEffectInstanceCreateForSymbol(MlirMemoryEffect effect,
                                        MlirAttribute symbol,
                                        MlirAttribute parameters, int stage,
                                        bool effectOnFullRegion,
                                        MlirSideEffectResource resource);

/// Destroys a memory effect instance created by one of the functions above.
MLIR_CAPI_EXPORTED void
mlirMemoryEffectInstanceDestroy(MlirMemoryEffectInstance instance);

/// Appends a copy of `instance` to the given list. This does not take ownership
/// of `instance`; the caller remains responsible for destroying it.
MLIR_CAPI_EXPORTED void
mlirMemoryEffectInstancesListAppend(MlirMemoryEffectInstancesList list,
                                    MlirMemoryEffectInstance instance);

/// Returns the interface TypeID of the MemoryEffectsOpInterface.
MLIR_CAPI_EXPORTED MlirTypeID mlirMemoryEffectsOpInterfaceTypeID(void);

/// Callbacks for implementing MemoryEffectsOpInterface from external code.
typedef struct {
  /// Optional constructor for user data. Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for user data. Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Get memory effects callback.
  void (*getEffects)(MlirOperation op, MlirMemoryEffectInstancesList effects,
                     void *userData);
  void *userData;
} MlirMemoryEffectsOpInterfaceCallbacks;

/// Attach a new FallbackModel for the MemoryEffectsOpInterface to the named
/// operation. The FallbackModel will call the provided callbacks.
MLIR_CAPI_EXPORTED void mlirMemoryEffectsOpInterfaceAttachFallbackModel(
    MlirContext ctx, MlirStringRef opName,
    MlirMemoryEffectsOpInterfaceCallbacks callbacks);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_INTERFACES_H
