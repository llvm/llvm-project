//===-- mlir-c/Remarks.h - MLIR Remark Engine C API ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to the MLIR optimization remark engine:
// enabling the engine on a context with category filters, an emitting policy
// and a sink (MLIR diagnostics, a YAML/bitstream file, or a callback),
// emitting remarks, and inspecting a remark from inside a callback.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REMARKS_H
#define MLIR_C_REMARKS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Opaque types and enums.
//===----------------------------------------------------------------------===//

/// An opaque reference to a remark. A remark is only valid for the duration of
/// the callback it is passed to; it must not be stored.
struct MlirRemark {
  const void *ptr;
};
typedef struct MlirRemark MlirRemark;

/// The outcome a remark describes (see mlir::remark::RemarkKind).
enum MlirRemarkKind {
  MlirRemarkKindUnknown,
  /// An optimization was applied.
  MlirRemarkKindPassed,
  /// A profitable optimization opportunity was found but not applied.
  MlirRemarkKindMissed,
  /// The optimization was attempted but failed.
  MlirRemarkKindFailure,
  /// Informational context without a pass/fail outcome.
  MlirRemarkKindAnalysis
};
typedef enum MlirRemarkKind MlirRemarkKind;

/// The emitting policy of the remark engine.
enum MlirRemarkPolicy {
  /// Report every remark as it is emitted.
  MlirRemarkPolicyAll,
  /// Postpone remarks and report the final set when the engine is finalized,
  /// grouping related remarks under their parents.
  MlirRemarkPolicyFinal
};
typedef enum MlirRemarkPolicy MlirRemarkPolicy;

/// The serialization format of a remark output file.
enum MlirRemarkFileFormat {
  MlirRemarkFileFormatYAML,
  MlirRemarkFileFormatBitstream
};
typedef enum MlirRemarkFileFormat MlirRemarkFileFormat;

/// Regular expressions (llvm::Regex syntax, anchored by the engine) selecting
/// the remark categories to report: `all` applies to every kind, the others to
/// one kind each. A string of length 0 means the filter is not set, so the
/// corresponding kind is not reported unless `all` matches.
struct MlirRemarkCategories {
  MlirStringRef all;
  MlirStringRef passed;
  MlirStringRef missed;
  MlirStringRef analysis;
  MlirStringRef failed;
};
typedef struct MlirRemarkCategories MlirRemarkCategories;

/// Callback receiving every remark the engine reports.
typedef void (*MlirRemarkCallback)(MlirRemark remark, void *userData);

//===----------------------------------------------------------------------===//
// Enabling and finalizing the engine of a context.
//===----------------------------------------------------------------------===//

/// Enables the remark engine on the context without a streamer; when
/// `printAsEmitRemarks` is set, every reported remark is emitted as an MLIR
/// remark diagnostic at its location. Fails when the context already has a
/// remark engine (finalize it first).
MLIR_CAPI_EXPORTED MlirLogicalResult mlirContextEnableOptimizationRemarks(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, bool printAsEmitRemarks);

/// Enables the remark engine on the context, streaming the reported remarks to
/// the file at `path` in `format`. The file is written when the engine is
/// finalized. Fails when the file cannot be created or the context already has
/// a remark engine.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirContextEnableOptimizationRemarksToFile(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, MlirRemarkFileFormat format, MlirStringRef path,
    bool printAsEmitRemarks);

/// Enables the remark engine on the context, delivering every reported remark
/// to `callback`. `userData` is passed to the callback unchanged and released
/// through `deleteUserData` (may be null) when the engine is finalized or the
/// context is destroyed. Fails when the context already has a remark engine.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirContextEnableOptimizationRemarksWithCallback(
    MlirContext context, MlirRemarkCategories categories,
    MlirRemarkPolicy policy, MlirRemarkCallback callback, void *userData,
    void (*deleteUserData)(void *), bool printAsEmitRemarks);

/// Finalizes and removes the remark engine of the context: postponed remarks
/// are reported, the output file is written and the callback's user data is
/// released. Does nothing when no engine is enabled.
MLIR_CAPI_EXPORTED void
mlirContextFinalizeOptimizationRemarks(MlirContext context);

/// Returns whether the context has a remark engine enabled.
MLIR_CAPI_EXPORTED bool mlirContextHasRemarkEngine(MlirContext context);

//===----------------------------------------------------------------------===//
// Remark accessors (valid only inside a remark callback).
//===----------------------------------------------------------------------===//

/// Returns the kind of the remark.
MLIR_CAPI_EXPORTED MlirRemarkKind mlirRemarkGetKind(MlirRemark remark);

/// Returns the name identifying the remark.
MLIR_CAPI_EXPORTED MlirStringRef mlirRemarkGetRemarkName(MlirRemark remark);

/// Returns the category of the remark (the subject of the filters).
MLIR_CAPI_EXPORTED MlirStringRef mlirRemarkGetCategoryName(MlirRemark remark);

/// Returns the combined `category:subcategory` name of the remark.
MLIR_CAPI_EXPORTED MlirStringRef
mlirRemarkGetFullCategoryName(MlirRemark remark);

/// Returns the name of the function the remark refers to.
MLIR_CAPI_EXPORTED MlirStringRef mlirRemarkGetFunctionName(MlirRemark remark);

/// Returns the location of the remark.
MLIR_CAPI_EXPORTED MlirLocation mlirRemarkGetLocation(MlirRemark remark);

/// Returns the unique id of the remark within its engine (0 when unset).
MLIR_CAPI_EXPORTED uint64_t mlirRemarkGetId(MlirRemark remark);

/// Returns the number of key/value arguments attached to the remark.
MLIR_CAPI_EXPORTED intptr_t mlirRemarkGetNumArgs(MlirRemark remark);

/// Returns the key of the `pos`-th argument of the remark.
MLIR_CAPI_EXPORTED MlirStringRef mlirRemarkGetArgKey(MlirRemark remark,
                                                     intptr_t pos);

/// Returns the value of the `pos`-th argument of the remark.
MLIR_CAPI_EXPORTED MlirStringRef mlirRemarkGetArgValue(MlirRemark remark,
                                                       intptr_t pos);

/// Prints the remark in its textual form, `[Kind] name | Category:... |
/// Function=... | key=value, ...` (without its location).
MLIR_CAPI_EXPORTED void
mlirRemarkPrint(MlirRemark remark, MlirStringCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// Emission.
//===----------------------------------------------------------------------===//

/// Emits a remark of `kind` at `location` through the remark engine of the
/// location's context. `message` (when non-empty) and the `numArgs` key/value
/// pairs become the arguments of the remark. Returns false, emitting nothing,
/// when the context has no remark engine, the category is filtered out, or
/// `kind` is MlirRemarkKindUnknown.
MLIR_CAPI_EXPORTED bool mlirEmitOptimizationRemark(
    MlirLocation location, MlirRemarkKind kind, MlirStringRef remarkName,
    MlirStringRef categoryName, MlirStringRef subCategoryName,
    MlirStringRef functionName, MlirStringRef message, intptr_t numArgs,
    const MlirStringRef *argKeys, const MlirStringRef *argValues);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REMARKS_H
