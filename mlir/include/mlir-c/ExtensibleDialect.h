//===-- mlir-c/ExtensibleDialect.h - Extensible dialect APIs -----*- C -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides APIs for extensible dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_EXTENSIBLEDIALECT_H
#define MLIR_C_EXTENSIBLEDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see mlir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirDynamicDialect, void);
DEFINE_C_API_STRUCT(MlirDynamicOpDefinition, void);
DEFINE_C_API_STRUCT(MlirDynamicOpTrait, void);
DEFINE_C_API_STRUCT(MlirDynamicTypeDefinition, void);
DEFINE_C_API_STRUCT(MlirDynamicAttrDefinition, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
/// Dynamic dialect creation
//===----------------------------------------------------------------------===//

/// Create a new dynamic dialect with the given name and register it with the
/// context. If a dialect with the same name already exists, returns the
/// existing one. The context takes ownership of the dialect. The returned
/// handle is valid as long as the context is alive and must not be freed.
MLIR_CAPI_EXPORTED MlirDynamicDialect
mlirDynamicDialectCreate(MlirContext ctx, MlirStringRef name);

/// Get the underlying MlirDialect from a MlirDynamicDialect.
MLIR_CAPI_EXPORTED MlirDialect
mlirDynamicDialectAsDialect(MlirDynamicDialect dialect);

//===----------------------------------------------------------------------===//
/// Dynamic op definition creation
//===----------------------------------------------------------------------===//

/// Callbacks for a dynamic op definition. All fields may be NULL, in which
/// case the corresponding action is a no-op / always succeeds.
typedef struct {
  /// The callback function to verify the operation.
  MlirLogicalResult (*verify)(MlirOperation op, void *userData);
  /// The callback function to verify the operation with access to regions.
  MlirLogicalResult (*verifyRegion)(MlirOperation op, void *userData);
} MlirDynamicOpDefinitionCallbacks;

/// Create a dynamic op definition with the given name and callbacks.
/// The name should be the bare op name (e.g. "constant"), not the
/// dialect-qualified name. The definition is NOT yet registered with the
/// dialect; call mlirDynamicDialectRegisterOp to register it.
/// \p userData is shared between all callbacks.
/// If the definition is not registered, it must be destroyed with
/// mlirDynamicOpDefinitionDestroy to avoid a memory leak.
MLIR_CAPI_EXPORTED MlirDynamicOpDefinition mlirDynamicOpDefinitionCreate(
    MlirDynamicDialect dialect, MlirStringRef name,
    MlirDynamicOpDefinitionCallbacks callbacks, void *userData);

/// Destroy a dynamic op definition that was not registered with a dialect.
MLIR_CAPI_EXPORTED void
mlirDynamicOpDefinitionDestroy(MlirDynamicOpDefinition opDef);

/// Register a dynamic op definition with its parent dialect.
/// This transfers ownership of the definition to the dialect.
/// After this call, ops with this name can be created using
/// mlirOperationCreate with the dialect-qualified name.
MLIR_CAPI_EXPORTED void
mlirDynamicDialectRegisterOp(MlirDynamicDialect dialect,
                             MlirDynamicOpDefinition opDef);

//===----------------------------------------------------------------------===//
/// Dynamic type definition creation
//===----------------------------------------------------------------------===//

/// Callbacks for a dynamic type definition. All fields may be NULL, in which
/// case the corresponding action is a no-op / always succeeds.
typedef struct {
  /// The callback function to verify the type's parameters.
  MlirLogicalResult (*verify)(intptr_t nParams, MlirAttribute const *params,
                              void *userData);
} MlirDynamicTypeDefinitionCallbacks;

/// Create a dynamic type definition with the given name and callbacks.
/// The name should be the bare type name (e.g. "mystruct"), not the
/// dialect-qualified name. The definition is NOT yet registered with the
/// dialect; call mlirDynamicDialectRegisterType to register it.
/// Note: the verifier callback does not receive diagnostic context; failures
/// are reported without a custom error message.
/// If the definition is not registered, it must be destroyed with
/// mlirDynamicTypeDefinitionDestroy to avoid a memory leak.
MLIR_CAPI_EXPORTED MlirDynamicTypeDefinition mlirDynamicTypeDefinitionCreate(
    MlirDynamicDialect dialect, MlirStringRef name,
    MlirDynamicTypeDefinitionCallbacks callbacks, void *userData);

/// Destroy a dynamic type definition that was not registered with a dialect.
MLIR_CAPI_EXPORTED void
mlirDynamicTypeDefinitionDestroy(MlirDynamicTypeDefinition typeDef);

/// Register a dynamic type definition with its parent dialect.
/// This transfers ownership of the definition to the dialect.
MLIR_CAPI_EXPORTED void
mlirDynamicDialectRegisterType(MlirDynamicDialect dialect,
                               MlirDynamicTypeDefinition typeDef);

//===----------------------------------------------------------------------===//
/// Dynamic attribute definition creation
//===----------------------------------------------------------------------===//

/// Callbacks for a dynamic attribute definition. All fields may be NULL, in
/// which case the corresponding action is a no-op / always succeeds.
typedef struct {
  /// The callback function to verify the attribute's parameters.
  MlirLogicalResult (*verify)(intptr_t nParams, MlirAttribute const *params,
                              void *userData);
} MlirDynamicAttrDefinitionCallbacks;

/// Create a dynamic attribute definition with the given name and callbacks.
/// The name should be the bare attr name, not the dialect-qualified name.
/// The definition is NOT yet registered with the dialect; call
/// mlirDynamicDialectRegisterAttr to register it.
/// Note: the verifier callback does not receive diagnostic context; failures
/// are reported without a custom error message.
/// If the definition is not registered, it must be destroyed with
/// mlirDynamicAttrDefinitionDestroy to avoid a memory leak.
MLIR_CAPI_EXPORTED MlirDynamicAttrDefinition mlirDynamicAttrDefinitionCreate(
    MlirDynamicDialect dialect, MlirStringRef name,
    MlirDynamicAttrDefinitionCallbacks callbacks, void *userData);

/// Destroy a dynamic attr definition that was not registered with a dialect.
MLIR_CAPI_EXPORTED void
mlirDynamicAttrDefinitionDestroy(MlirDynamicAttrDefinition attrDef);

/// Register a dynamic attribute definition with its parent dialect.
/// This transfers ownership of the definition to the dialect.
MLIR_CAPI_EXPORTED void
mlirDynamicDialectRegisterAttr(MlirDynamicDialect dialect,
                               MlirDynamicAttrDefinition attrDef);

//===----------------------------------------------------------------------===//
/// Dynamic op trait APIs
//===----------------------------------------------------------------------===//

/// Attach a dynamic op trait to the given operation name.
/// Note that the operation name must be modeled by dynamic dialect and must be
/// registered.
/// The ownership of the trait will be transferred to the operation name
/// after this call.
MLIR_CAPI_EXPORTED bool
mlirDynamicOpTraitAttach(MlirDynamicOpTrait dynamicOpTrait,
                         MlirStringRef opName, MlirContext context);

/// Get the dynamic op trait that indicates the operation is a terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait
mlirDynamicOpTraitIsTerminatorCreate(void);

/// Get the type ID of the dynamic op trait that indicates the operation is a
/// terminator.
MLIR_CAPI_EXPORTED MlirTypeID mlirDynamicOpTraitIsTerminatorGetTypeID(void);

/// Get the dynamic op trait that indicates regions have no terminator.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait
mlirDynamicOpTraitNoTerminatorCreate(void);

/// Get the type ID of the dynamic op trait that indicates regions have no
/// terminator.
MLIR_CAPI_EXPORTED MlirTypeID mlirDynamicOpTraitNoTerminatorGetTypeID(void);

/// Destroy the dynamic op trait.
MLIR_CAPI_EXPORTED void
mlirDynamicOpTraitDestroy(MlirDynamicOpTrait dynamicOpTrait);

typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to verify the operation.
  MlirLogicalResult (*verifyTrait)(MlirOperation op, void *userData);
  /// The callback function to verify the operation with access to regions.
  MlirLogicalResult (*verifyRegionTrait)(MlirOperation op, void *userData);
} MlirDynamicOpTraitCallbacks;

/// Create a custom dynamic op trait with the given type ID and callbacks.
MLIR_CAPI_EXPORTED MlirDynamicOpTrait mlirDynamicOpTraitCreate(
    MlirTypeID typeID, MlirDynamicOpTraitCallbacks callbacks, void *userData);

/// Check if the given dialect is an extensible dialect.
MLIR_CAPI_EXPORTED bool mlirDialectIsAExtensibleDialect(MlirDialect dialect);

/// Look up a registered type definition by type name in the given dialect.
/// Note that the dialect must be an extensible dialect.
MLIR_CAPI_EXPORTED MlirDynamicTypeDefinition
mlirExtensibleDialectLookupTypeDefinition(MlirDialect dialect,
                                          MlirStringRef typeName);

/// Check if the given type is a dynamic type.
MLIR_CAPI_EXPORTED bool mlirTypeIsADynamicType(MlirType type);

/// Get a dynamic type by instantiating the given type definition with the
/// provided attributes.
MLIR_CAPI_EXPORTED MlirType mlirDynamicTypeGet(
    MlirDynamicTypeDefinition typeDef, MlirAttribute *attrs, intptr_t numAttrs);

/// Get the number of parameters in the given dynamic type.
MLIR_CAPI_EXPORTED intptr_t mlirDynamicTypeGetNumParams(MlirType type);

/// Get the parameter at the given index in the provided dynamic type.
MLIR_CAPI_EXPORTED MlirAttribute mlirDynamicTypeGetParam(MlirType type,
                                                         intptr_t index);

/// Get the type definition of the given dynamic type.
MLIR_CAPI_EXPORTED MlirDynamicTypeDefinition
mlirDynamicTypeGetTypeDef(MlirType type);

/// Get the type ID of a dynamic type definition.
MLIR_CAPI_EXPORTED MlirTypeID
mlirDynamicTypeDefinitionGetTypeID(MlirDynamicTypeDefinition typeDef);

/// Get the name of the given dynamic type definition.
MLIR_CAPI_EXPORTED MlirStringRef
mlirDynamicTypeDefinitionGetName(MlirDynamicTypeDefinition typeDef);

/// Get the dialect that the given dynamic type definition belongs to.
MLIR_CAPI_EXPORTED MlirDialect
mlirDynamicTypeDefinitionGetDialect(MlirDynamicTypeDefinition typeDef);

/// Look up a registered attribute definition by attribute name in the given
/// dialect. Note that the dialect must be an extensible dialect.
MLIR_CAPI_EXPORTED MlirDynamicAttrDefinition
mlirExtensibleDialectLookupAttrDefinition(MlirDialect dialect,
                                          MlirStringRef attrName);

/// Check if the given attribute is a dynamic attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsADynamicAttr(MlirAttribute attr);

/// Get a dynamic attribute by instantiating the given attribute definition with
/// the provided attributes.
MLIR_CAPI_EXPORTED MlirAttribute mlirDynamicAttrGet(
    MlirDynamicAttrDefinition attrDef, MlirAttribute *attrs, intptr_t numAttrs);

/// Get the number of parameters in the given dynamic attribute.
MLIR_CAPI_EXPORTED intptr_t mlirDynamicAttrGetNumParams(MlirAttribute attr);

/// Get the parameter at the given index in the provided dynamic attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirDynamicAttrGetParam(MlirAttribute attr,
                                                         intptr_t index);

/// Get the attribute definition of the given dynamic attribute.
MLIR_CAPI_EXPORTED MlirDynamicAttrDefinition
mlirDynamicAttrGetAttrDef(MlirAttribute attr);

/// Get the type ID of a dynamic attribute definition.
MLIR_CAPI_EXPORTED MlirTypeID
mlirDynamicAttrDefinitionGetTypeID(MlirDynamicAttrDefinition attrDef);

/// Get the name of the given dynamic attribute definition.
MLIR_CAPI_EXPORTED MlirStringRef
mlirDynamicAttrDefinitionGetName(MlirDynamicAttrDefinition attrDef);

/// Get the dialect that the given dynamic attribute definition belongs to.
MLIR_CAPI_EXPORTED MlirDialect
mlirDynamicAttrDefinitionGetDialect(MlirDynamicAttrDefinition attrDef);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_EXTENSIBLEDIALECT_H
