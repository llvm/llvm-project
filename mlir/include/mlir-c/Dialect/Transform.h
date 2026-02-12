//===-- mlir-c/Dialect/Transform.h - C API for Transform Dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_TRANSFORM_H
#define MLIR_C_DIALECT_TRANSFORM_H

#include "mlir-c/IR.h"
#include "mlir-c/Interfaces.h"
#include "mlir-c/Rewrite.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Transform, transform);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirTransformResults, void);
DEFINE_C_API_STRUCT(MlirTransformRewriter, void);
DEFINE_C_API_STRUCT(MlirTransformState, void);

#undef DEFINE_C_API_STRUCT

//===---------------------------------------------------------------------===//
// DiagnosedSilenceableFailure
//===---------------------------------------------------------------------===//

/// Enum representing the result of a transform operation.
typedef enum {
  /// The operation succeeded.
  MlirDiagnosedSilenceableFailureSuccess,
  /// The operation failed in a silenceable way.
  MlirDiagnosedSilenceableFailureSilenceableFailure,
  /// The operation failed definitively.
  MlirDiagnosedSilenceableFailureDefiniteFailure
} MlirDiagnosedSilenceableFailure;

//===---------------------------------------------------------------------===//
// AnyOpType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyOpType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformAnyOpTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyOpTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirStringRef mlirTransformAnyOpTypeGetName(void);

//===---------------------------------------------------------------------===//
// AnyParamType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyParamType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformAnyParamTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyParamTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirStringRef mlirTransformAnyParamTypeGetName(void);

//===---------------------------------------------------------------------===//
// AnyValueType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformAnyValueType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformAnyValueTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTransformAnyValueTypeGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirStringRef mlirTransformAnyValueTypeGetName(void);

//===---------------------------------------------------------------------===//
// OperationType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformOperationType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformOperationTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType
mlirTransformOperationTypeGet(MlirContext ctx, MlirStringRef operationName);

MLIR_CAPI_EXPORTED MlirStringRef mlirTransformOperationTypeGetName(void);

MLIR_CAPI_EXPORTED MlirStringRef
mlirTransformOperationTypeGetOperationName(MlirType type);

//===---------------------------------------------------------------------===//
// ParamType
//===---------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsATransformParamType(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirTransformParamTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirTransformParamTypeGet(MlirContext ctx,
                                                      MlirType type);

MLIR_CAPI_EXPORTED MlirStringRef mlirTransformParamTypeGetName(void);

MLIR_CAPI_EXPORTED MlirType mlirTransformParamTypeGetType(MlirType type);

//===---------------------------------------------------------------------===//
// TransformRewriter
//===---------------------------------------------------------------------===//

/// Cast the TransformRewriter to a RewriterBase
MLIR_CAPI_EXPORTED MlirRewriterBase
mlirTransformRewriterAsBase(MlirTransformRewriter rewriter);

//===---------------------------------------------------------------------===//
// TransformResults
//===---------------------------------------------------------------------===//

/// Set the payload operations for a transform result by iterating over a list.
MLIR_CAPI_EXPORTED void mlirTransformResultsSetOps(MlirTransformResults results,
                                                   MlirValue result,
                                                   intptr_t numOps,
                                                   MlirOperation *ops);

/// Set the payload values for a transform result by iterating over a list.
MLIR_CAPI_EXPORTED void
mlirTransformResultsSetValues(MlirTransformResults results, MlirValue result,
                              intptr_t numValues, MlirValue *values);

/// Set the parameters for a transform result by iterating over a list.
MLIR_CAPI_EXPORTED void
mlirTransformResultsSetParams(MlirTransformResults results, MlirValue result,
                              intptr_t numParams, MlirAttribute *params);

//===---------------------------------------------------------------------===//
// TransformState
//===---------------------------------------------------------------------===//

/// Callback for iterating over payload operations.
typedef void (*MlirOperationCallback)(MlirOperation, void *userData);

/// Iterate over payload operations associated with the transform IR value.
/// Calls the callback for each payload operation.
MLIR_CAPI_EXPORTED void
mlirTransformStateForEachPayloadOp(MlirTransformState state, MlirValue value,
                                   MlirOperationCallback callback,
                                   void *userData);

/// Callback for iterating over payload values.
typedef void (*MlirValueCallback)(MlirValue, void *userData);

/// Iterate over payload values associated with the transform IR value.
/// Calls the callback for each payload value.
MLIR_CAPI_EXPORTED void
mlirTransformStateForEachPayloadValue(MlirTransformState state, MlirValue value,
                                      MlirValueCallback callback,
                                      void *userData);

/// Callback for iterating over parameters.
typedef void (*MlirAttributeCallback)(MlirAttribute, void *userData);

/// Iterate over parameters associated with the transform IR value.
/// Calls the callback for each parameter.
MLIR_CAPI_EXPORTED void
mlirTransformStateForEachParam(MlirTransformState state, MlirValue value,
                               MlirAttributeCallback callback, void *userData);

//===---------------------------------------------------------------------===//
// TransformOpInterface
//===---------------------------------------------------------------------===//

/// Returns the interface TypeID of the TransformOpInterface.
MLIR_CAPI_EXPORTED MlirTypeID mlirTransformOpInterfaceTypeID(void);

/// Callbacks for implementing TransformOpInterface from external code.
typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// Apply callback that implements the transformation.
  MlirDiagnosedSilenceableFailure (*apply)(MlirOperation op,
                                           MlirTransformRewriter rewriter,
                                           MlirTransformResults results,
                                           MlirTransformState state,
                                           void *userData);
  /// Callback to check if repeated handle operands are allowed.
  bool (*allowsRepeatedHandleOperands)(MlirOperation op, void *userData);
  void *userData;
} MlirTransformOpInterfaceCallbacks;

/// Attach TransformOpInterface to the operation with the given name using
/// the provided callbacks.
MLIR_CAPI_EXPORTED void mlirTransformOpInterfaceAttachFallbackModel(
    MlirContext ctx, MlirStringRef opName,
    MlirTransformOpInterfaceCallbacks callbacks);

//===---------------------------------------------------------------------===//
// Transform-specifc MemoryEffectsOpInterface helpers
//===---------------------------------------------------------------------===//

/// Helper to mark operands as only reading handles.
MLIR_CAPI_EXPORTED void
mlirTransformOnlyReadsHandle(MlirOpOperand *operands, intptr_t numOperands,
                             MlirMemoryEffectInstancesList effects);

/// Helper to mark operands as consuming handles.
MLIR_CAPI_EXPORTED void
mlirTransformConsumesHandle(MlirOpOperand *operands, intptr_t numOperands,
                            MlirMemoryEffectInstancesList effects);

/// Helper to mark results as producing handles.
MLIR_CAPI_EXPORTED void
mlirTransformProducesHandle(MlirValue *results, intptr_t numResults,
                            MlirMemoryEffectInstancesList effects);

/// Helper to mark potential modifications to the payload IR.
MLIR_CAPI_EXPORTED void
mlirTransformModifiesPayload(MlirMemoryEffectInstancesList effects);

/// Helper to mark potential reads from the payload IR.
MLIR_CAPI_EXPORTED void
mlirTransformOnlyReadsPayload(MlirMemoryEffectInstancesList effects);

#ifdef __cplusplus
}
#endif

#include "mlir/Dialect/Transform/Transforms/Passes.capi.h.inc"

#endif // MLIR_C_DIALECT_TRANSFORM_H
