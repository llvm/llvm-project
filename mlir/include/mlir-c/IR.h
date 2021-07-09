//===-- mlir-c/IR.h - C API to Core MLIR IR classes ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to MLIR core IR classes.
//
// Many exotic languages can interoperate with C code but have a harder time
// with C++ due to name mangling. So in addition to C, this interface enables
// tools written in such languages.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_IR_H
#define MLIR_C_IR_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations.
///
/// Types are exposed to C bindings as structs containing opaque pointers. They
/// are not supposed to be inspected from C. This allows the underlying
/// representation to change without affecting the API users. The use of structs
/// instead of typedefs enables some type safety as structs are not implicitly
/// convertible to each other.
///
/// Instances of these types may or may not own the underlying object (most
/// often only point to an IR fragment without owning it). The ownership
/// semantics is defined by how an instance of the type was obtained.

//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirContext, void);
DEFINE_C_API_STRUCT(MlirDialect, void);
DEFINE_C_API_STRUCT(MlirOperation, void);
DEFINE_C_API_STRUCT(MlirOpPrintingFlags, void);
DEFINE_C_API_STRUCT(MlirBlock, void);
DEFINE_C_API_STRUCT(MlirRegion, void);

DEFINE_C_API_STRUCT(MlirAttribute, const void);
DEFINE_C_API_STRUCT(MlirIdentifier, const void);
DEFINE_C_API_STRUCT(MlirLocation, const void);
DEFINE_C_API_STRUCT(MlirModule, const void);
DEFINE_C_API_STRUCT(MlirType, const void);
DEFINE_C_API_STRUCT(MlirValue, const void);

#undef DEFINE_C_API_STRUCT

/// Named MLIR attribute.
///
/// A named attribute is essentially a (name, attribute) pair where the name is
/// a string.

struct MlirNamedAttribute {
  MlirIdentifier name;
  MlirAttribute attribute;
};
typedef struct MlirNamedAttribute MlirNamedAttribute;

//===----------------------------------------------------------------------===//
// Context API.
//===----------------------------------------------------------------------===//

/// Creates an MLIR context and transfers its ownership to the caller.
MLIR_CAPI_EXPORTED MlirContext mlirContextCreate();

/// Checks if two contexts are equal.
MLIR_CAPI_EXPORTED bool mlirContextEqual(MlirContext ctx1, MlirContext ctx2);

/// Checks whether a context is null.
static inline bool mlirContextIsNull(MlirContext context) {
  return !context.ptr;
}

/// Takes an MLIR context owned by the caller and destroys it.
MLIR_CAPI_EXPORTED void mlirContextDestroy(MlirContext context);

/// Sets whether unregistered dialects are allowed in this context.
MLIR_CAPI_EXPORTED void
mlirContextSetAllowUnregisteredDialects(MlirContext context, bool allow);

/// Returns whether the context allows unregistered dialects.
MLIR_CAPI_EXPORTED bool
mlirContextGetAllowUnregisteredDialects(MlirContext context);

/// Returns the number of dialects registered with the given context. A
/// registered dialect will be loaded if needed by the parser.
MLIR_CAPI_EXPORTED intptr_t
mlirContextGetNumRegisteredDialects(MlirContext context);

/// Returns the number of dialects loaded by the context.

MLIR_CAPI_EXPORTED intptr_t
mlirContextGetNumLoadedDialects(MlirContext context);

/// Gets the dialect instance owned by the given context using the dialect
/// namespace to identify it, loads (i.e., constructs the instance of) the
/// dialect if necessary. If the dialect is not registered with the context,
/// returns null. Use mlirContextLoad<Name>Dialect to load an unregistered
/// dialect.
MLIR_CAPI_EXPORTED MlirDialect mlirContextGetOrLoadDialect(MlirContext context,
                                                           MlirStringRef name);

/// Set threading mode (must be set to false to print-ir-after-all).
MLIR_CAPI_EXPORTED void mlirContextEnableMultithreading(MlirContext context,
                                                        bool enable);

/// Returns whether the given fully-qualified operation (i.e.
/// 'dialect.operation') is registered with the context. This will return true
/// if the dialect is loaded and the operation is registered within the
/// dialect.
MLIR_CAPI_EXPORTED bool mlirContextIsRegisteredOperation(MlirContext context,
                                                         MlirStringRef name);

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

/// Returns the context that owns the dialect.
MLIR_CAPI_EXPORTED MlirContext mlirDialectGetContext(MlirDialect dialect);

/// Checks if the dialect is null.
static inline bool mlirDialectIsNull(MlirDialect dialect) {
  return !dialect.ptr;
}

/// Checks if two dialects that belong to the same context are equal. Dialects
/// from different contexts will not compare equal.
MLIR_CAPI_EXPORTED bool mlirDialectEqual(MlirDialect dialect1,
                                         MlirDialect dialect2);

/// Returns the namespace of the given dialect.
MLIR_CAPI_EXPORTED MlirStringRef mlirDialectGetNamespace(MlirDialect dialect);

//===----------------------------------------------------------------------===//
// Location API.
//===----------------------------------------------------------------------===//

/// Creates an File/Line/Column location owned by the given context.
MLIR_CAPI_EXPORTED MlirLocation mlirLocationFileLineColGet(
    MlirContext context, MlirStringRef filename, unsigned line, unsigned col);

/// Creates a call site location with a callee and a caller.
MLIR_CAPI_EXPORTED MlirLocation mlirLocationCallSiteGet(MlirLocation callee,
                                                        MlirLocation caller);

/// Creates a location with unknown position owned by the given context.
MLIR_CAPI_EXPORTED MlirLocation mlirLocationUnknownGet(MlirContext context);

/// Gets the context that a location was created with.
MLIR_CAPI_EXPORTED MlirContext mlirLocationGetContext(MlirLocation location);

/// Checks if the location is null.
static inline bool mlirLocationIsNull(MlirLocation location) {
  return !location.ptr;
}

/// Checks if two locations are equal.
MLIR_CAPI_EXPORTED bool mlirLocationEqual(MlirLocation l1, MlirLocation l2);

/// Prints a location by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirLocationPrint(MlirLocation location,
                                          MlirStringCallback callback,
                                          void *userData);

//===----------------------------------------------------------------------===//
// Module API.
//===----------------------------------------------------------------------===//

/// Creates a new, empty module and transfers ownership to the caller.
MLIR_CAPI_EXPORTED MlirModule mlirModuleCreateEmpty(MlirLocation location);

/// Parses a module from the string and transfers ownership to the caller.
MLIR_CAPI_EXPORTED MlirModule mlirModuleCreateParse(MlirContext context,
                                                    MlirStringRef module);

/// Gets the context that a module was created with.
MLIR_CAPI_EXPORTED MlirContext mlirModuleGetContext(MlirModule module);

/// Gets the body of the module, i.e. the only block it contains.
MLIR_CAPI_EXPORTED MlirBlock mlirModuleGetBody(MlirModule module);

/// Checks whether a module is null.
static inline bool mlirModuleIsNull(MlirModule module) { return !module.ptr; }

/// Takes a module owned by the caller and deletes it.
MLIR_CAPI_EXPORTED void mlirModuleDestroy(MlirModule module);

/// Views the module as a generic operation.
MLIR_CAPI_EXPORTED MlirOperation mlirModuleGetOperation(MlirModule module);

/// Views the generic operation as a module.
/// The returned module is null when the input operation was not a ModuleOp.
MLIR_CAPI_EXPORTED MlirModule mlirModuleFromOperation(MlirOperation op);

//===----------------------------------------------------------------------===//
// Operation state.
//===----------------------------------------------------------------------===//

/// An auxiliary class for constructing operations.
///
/// This class contains all the information necessary to construct the
/// operation. It owns the MlirRegions it has pointers to and does not own
/// anything else. By default, the state can be constructed from a name and
/// location, the latter being also used to access the context, and has no other
/// components. These components can be added progressively until the operation
/// is constructed. Users are not expected to rely on the internals of this
/// class and should use mlirOperationState* functions instead.

struct MlirOperationState {
  MlirStringRef name;
  MlirLocation location;
  intptr_t nResults;
  MlirType *results;
  intptr_t nOperands;
  MlirValue *operands;
  intptr_t nRegions;
  MlirRegion *regions;
  intptr_t nSuccessors;
  MlirBlock *successors;
  intptr_t nAttributes;
  MlirNamedAttribute *attributes;
  bool enableResultTypeInference;
};
typedef struct MlirOperationState MlirOperationState;

/// Constructs an operation state from a name and a location.
MLIR_CAPI_EXPORTED MlirOperationState mlirOperationStateGet(MlirStringRef name,
                                                            MlirLocation loc);

/// Adds a list of components to the operation state.
MLIR_CAPI_EXPORTED void mlirOperationStateAddResults(MlirOperationState *state,
                                                     intptr_t n,
                                                     MlirType const *results);
MLIR_CAPI_EXPORTED void
mlirOperationStateAddOperands(MlirOperationState *state, intptr_t n,
                              MlirValue const *operands);
MLIR_CAPI_EXPORTED void
mlirOperationStateAddOwnedRegions(MlirOperationState *state, intptr_t n,
                                  MlirRegion const *regions);
MLIR_CAPI_EXPORTED void
mlirOperationStateAddSuccessors(MlirOperationState *state, intptr_t n,
                                MlirBlock const *successors);
MLIR_CAPI_EXPORTED void
mlirOperationStateAddAttributes(MlirOperationState *state, intptr_t n,
                                MlirNamedAttribute const *attributes);

/// Enables result type inference for the operation under construction. If
/// enabled, then the caller must not have called
/// mlirOperationStateAddResults(). Note that if enabled, the
/// mlirOperationCreate() call is failable: it will return a null operation
/// on inference failure and will emit diagnostics.
MLIR_CAPI_EXPORTED void
mlirOperationStateEnableResultTypeInference(MlirOperationState *state);

//===----------------------------------------------------------------------===//
// Op Printing flags API.
// While many of these are simple settings that could be represented in a
// struct, they are wrapped in a heap allocated object and accessed via
// functions to maximize the possibility of compatibility over time.
//===----------------------------------------------------------------------===//

/// Creates new printing flags with defaults, intended for customization.
/// Must be freed with a call to mlirOpPrintingFlagsDestroy().
MLIR_CAPI_EXPORTED MlirOpPrintingFlags mlirOpPrintingFlagsCreate();

/// Destroys printing flags created with mlirOpPrintingFlagsCreate.
MLIR_CAPI_EXPORTED void mlirOpPrintingFlagsDestroy(MlirOpPrintingFlags flags);

/// Enables the elision of large elements attributes by printing a lexically
/// valid but otherwise meaningless form instead of the element data. The
/// `largeElementLimit` is used to configure what is considered to be a "large"
/// ElementsAttr by providing an upper limit to the number of elements.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsElideLargeElementsAttrs(MlirOpPrintingFlags flags,
                                           intptr_t largeElementLimit);

/// Enable printing of debug information. If 'prettyForm' is set to true,
/// debug information is printed in a more readable 'pretty' form. Note: The
/// IR generated with 'prettyForm' is not parsable.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsEnableDebugInfo(MlirOpPrintingFlags flags, bool prettyForm);

/// Always print operations in the generic form.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsPrintGenericOpForm(MlirOpPrintingFlags flags);

/// Use local scope when printing the operation. This allows for using the
/// printer in a more localized and thread-safe setting, but may not
/// necessarily be identical to what the IR will look like when dumping
/// the full module.
MLIR_CAPI_EXPORTED void
mlirOpPrintingFlagsUseLocalScope(MlirOpPrintingFlags flags);

//===----------------------------------------------------------------------===//
// Operation API.
//===----------------------------------------------------------------------===//

/// Creates an operation and transfers ownership to the caller.
/// Note that caller owned child objects are transferred in this call and must
/// not be further used. Particularly, this applies to any regions added to
/// the state (the implementation may invalidate any such pointers).
///
/// This call can fail under the following conditions, in which case, it will
/// return a null operation and emit diagnostics:
///   - Result type inference is enabled and cannot be performed.
MLIR_CAPI_EXPORTED MlirOperation mlirOperationCreate(MlirOperationState *state);

/// Creates a deep copy of an operation. The operation is not inserted and
/// ownership is transferred to the caller.
MLIR_CAPI_EXPORTED MlirOperation mlirOperationClone(MlirOperation op);

/// Takes an operation owned by the caller and destroys it.
MLIR_CAPI_EXPORTED void mlirOperationDestroy(MlirOperation op);

/// Checks whether the underlying operation is null.
static inline bool mlirOperationIsNull(MlirOperation op) { return !op.ptr; }

/// Checks whether two operation handles point to the same operation. This does
/// not perform deep comparison.
MLIR_CAPI_EXPORTED bool mlirOperationEqual(MlirOperation op,
                                           MlirOperation other);

/// Gets the context this operation is associated with
MLIR_CAPI_EXPORTED MlirContext mlirOperationGetContext(MlirOperation op);

/// Gets the name of the operation as an identifier.
MLIR_CAPI_EXPORTED MlirIdentifier mlirOperationGetName(MlirOperation op);

/// Gets the block that owns this operation, returning null if the operation is
/// not owned.
MLIR_CAPI_EXPORTED MlirBlock mlirOperationGetBlock(MlirOperation op);

/// Gets the operation that owns this operation, returning null if the operation
/// is not owned.
MLIR_CAPI_EXPORTED MlirOperation
mlirOperationGetParentOperation(MlirOperation op);

/// Returns the number of regions attached to the given operation.
MLIR_CAPI_EXPORTED intptr_t mlirOperationGetNumRegions(MlirOperation op);

/// Returns `pos`-th region attached to the operation.
MLIR_CAPI_EXPORTED MlirRegion mlirOperationGetRegion(MlirOperation op,
                                                     intptr_t pos);

/// Returns an operation immediately following the given operation it its
/// enclosing block.
MLIR_CAPI_EXPORTED MlirOperation mlirOperationGetNextInBlock(MlirOperation op);

/// Returns the number of operands of the operation.
MLIR_CAPI_EXPORTED intptr_t mlirOperationGetNumOperands(MlirOperation op);

/// Returns `pos`-th operand of the operation.
MLIR_CAPI_EXPORTED MlirValue mlirOperationGetOperand(MlirOperation op,
                                                     intptr_t pos);

/// Sets the `pos`-th operand of the operation.
MLIR_CAPI_EXPORTED void mlirOperationSetOperand(MlirOperation op, intptr_t pos,
                                                MlirValue newValue);

/// Returns the number of results of the operation.
MLIR_CAPI_EXPORTED intptr_t mlirOperationGetNumResults(MlirOperation op);

/// Returns `pos`-th result of the operation.
MLIR_CAPI_EXPORTED MlirValue mlirOperationGetResult(MlirOperation op,
                                                    intptr_t pos);

/// Returns the number of successor blocks of the operation.
MLIR_CAPI_EXPORTED intptr_t mlirOperationGetNumSuccessors(MlirOperation op);

/// Returns `pos`-th successor of the operation.
MLIR_CAPI_EXPORTED MlirBlock mlirOperationGetSuccessor(MlirOperation op,
                                                       intptr_t pos);

/// Returns the number of attributes attached to the operation.
MLIR_CAPI_EXPORTED intptr_t mlirOperationGetNumAttributes(MlirOperation op);

/// Return `pos`-th attribute of the operation.
MLIR_CAPI_EXPORTED MlirNamedAttribute
mlirOperationGetAttribute(MlirOperation op, intptr_t pos);

/// Returns an attribute attached to the operation given its name.
MLIR_CAPI_EXPORTED MlirAttribute
mlirOperationGetAttributeByName(MlirOperation op, MlirStringRef name);

/// Sets an attribute by name, replacing the existing if it exists or
/// adding a new one otherwise.
MLIR_CAPI_EXPORTED void mlirOperationSetAttributeByName(MlirOperation op,
                                                        MlirStringRef name,
                                                        MlirAttribute attr);

/// Removes an attribute by name. Returns false if the attribute was not found
/// and true if removed.
MLIR_CAPI_EXPORTED bool mlirOperationRemoveAttributeByName(MlirOperation op,
                                                           MlirStringRef name);

/// Prints an operation by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirOperationPrint(MlirOperation op,
                                           MlirStringCallback callback,
                                           void *userData);

/// Same as mlirOperationPrint but accepts flags controlling the printing
/// behavior.
MLIR_CAPI_EXPORTED void mlirOperationPrintWithFlags(MlirOperation op,
                                                    MlirOpPrintingFlags flags,
                                                    MlirStringCallback callback,
                                                    void *userData);

/// Prints an operation to stderr.
MLIR_CAPI_EXPORTED void mlirOperationDump(MlirOperation op);

/// Verify the operation and return true if it passes, false if it fails.
MLIR_CAPI_EXPORTED bool mlirOperationVerify(MlirOperation op);

//===----------------------------------------------------------------------===//
// Region API.
//===----------------------------------------------------------------------===//

/// Creates a new empty region and transfers ownership to the caller.
MLIR_CAPI_EXPORTED MlirRegion mlirRegionCreate();

/// Takes a region owned by the caller and destroys it.
MLIR_CAPI_EXPORTED void mlirRegionDestroy(MlirRegion region);

/// Checks whether a region is null.
static inline bool mlirRegionIsNull(MlirRegion region) { return !region.ptr; }

/// Gets the first block in the region.
MLIR_CAPI_EXPORTED MlirBlock mlirRegionGetFirstBlock(MlirRegion region);

/// Takes a block owned by the caller and appends it to the given region.
MLIR_CAPI_EXPORTED void mlirRegionAppendOwnedBlock(MlirRegion region,
                                                   MlirBlock block);

/// Takes a block owned by the caller and inserts it at `pos` to the given
/// region. This is an expensive operation that linearly scans the region,
/// prefer insertAfter/Before instead.
MLIR_CAPI_EXPORTED void
mlirRegionInsertOwnedBlock(MlirRegion region, intptr_t pos, MlirBlock block);

/// Takes a block owned by the caller and inserts it after the (non-owned)
/// reference block in the given region. The reference block must belong to the
/// region. If the reference block is null, prepends the block to the region.
MLIR_CAPI_EXPORTED void mlirRegionInsertOwnedBlockAfter(MlirRegion region,
                                                        MlirBlock reference,
                                                        MlirBlock block);

/// Takes a block owned by the caller and inserts it before the (non-owned)
/// reference block in the given region. The reference block must belong to the
/// region. If the reference block is null, appends the block to the region.
MLIR_CAPI_EXPORTED void mlirRegionInsertOwnedBlockBefore(MlirRegion region,
                                                         MlirBlock reference,
                                                         MlirBlock block);

//===----------------------------------------------------------------------===//
// Block API.
//===----------------------------------------------------------------------===//

/// Creates a new empty block with the given argument types and transfers
/// ownership to the caller.
MLIR_CAPI_EXPORTED MlirBlock mlirBlockCreate(intptr_t nArgs,
                                             MlirType const *args);

/// Takes a block owned by the caller and destroys it.
MLIR_CAPI_EXPORTED void mlirBlockDestroy(MlirBlock block);

/// Checks whether a block is null.
static inline bool mlirBlockIsNull(MlirBlock block) { return !block.ptr; }

/// Checks whether two blocks handles point to the same block. This does not
/// perform deep comparison.
MLIR_CAPI_EXPORTED bool mlirBlockEqual(MlirBlock block, MlirBlock other);

/// Returns the closest surrounding operation that contains this block.
MLIR_CAPI_EXPORTED MlirOperation mlirBlockGetParentOperation(MlirBlock);

/// Returns the block immediately following the given block in its parent
/// region.
MLIR_CAPI_EXPORTED MlirBlock mlirBlockGetNextInRegion(MlirBlock block);

/// Returns the first operation in the block.
MLIR_CAPI_EXPORTED MlirOperation mlirBlockGetFirstOperation(MlirBlock block);

/// Returns the terminator operation in the block or null if no terminator.
MLIR_CAPI_EXPORTED MlirOperation mlirBlockGetTerminator(MlirBlock block);

/// Takes an operation owned by the caller and appends it to the block.
MLIR_CAPI_EXPORTED void mlirBlockAppendOwnedOperation(MlirBlock block,
                                                      MlirOperation operation);

/// Takes an operation owned by the caller and inserts it as `pos` to the block.
/// This is an expensive operation that scans the block linearly, prefer
/// insertBefore/After instead.
MLIR_CAPI_EXPORTED void mlirBlockInsertOwnedOperation(MlirBlock block,
                                                      intptr_t pos,
                                                      MlirOperation operation);

/// Takes an operation owned by the caller and inserts it after the (non-owned)
/// reference operation in the given block. If the reference is null, prepends
/// the operation. Otherwise, the reference must belong to the block.
MLIR_CAPI_EXPORTED void
mlirBlockInsertOwnedOperationAfter(MlirBlock block, MlirOperation reference,
                                   MlirOperation operation);

/// Takes an operation owned by the caller and inserts it before the (non-owned)
/// reference operation in the given block. If the reference is null, appends
/// the operation. Otherwise, the reference must belong to the block.
MLIR_CAPI_EXPORTED void
mlirBlockInsertOwnedOperationBefore(MlirBlock block, MlirOperation reference,
                                    MlirOperation operation);

/// Returns the number of arguments of the block.
MLIR_CAPI_EXPORTED intptr_t mlirBlockGetNumArguments(MlirBlock block);

/// Appends an argument of the specified type to the block. Returns the newly
/// added argument.
MLIR_CAPI_EXPORTED MlirValue mlirBlockAddArgument(MlirBlock block,
                                                  MlirType type);

/// Returns `pos`-th argument of the block.
MLIR_CAPI_EXPORTED MlirValue mlirBlockGetArgument(MlirBlock block,
                                                  intptr_t pos);

/// Prints a block by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void
mlirBlockPrint(MlirBlock block, MlirStringCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// Value API.
//===----------------------------------------------------------------------===//

/// Returns whether the value is null.
static inline bool mlirValueIsNull(MlirValue value) { return !value.ptr; }

/// Returns 1 if two values are equal, 0 otherwise.
MLIR_CAPI_EXPORTED bool mlirValueEqual(MlirValue value1, MlirValue value2);

/// Returns 1 if the value is a block argument, 0 otherwise.
MLIR_CAPI_EXPORTED bool mlirValueIsABlockArgument(MlirValue value);

/// Returns 1 if the value is an operation result, 0 otherwise.
MLIR_CAPI_EXPORTED bool mlirValueIsAOpResult(MlirValue value);

/// Returns the block in which this value is defined as an argument. Asserts if
/// the value is not a block argument.
MLIR_CAPI_EXPORTED MlirBlock mlirBlockArgumentGetOwner(MlirValue value);

/// Returns the position of the value in the argument list of its block.
MLIR_CAPI_EXPORTED intptr_t mlirBlockArgumentGetArgNumber(MlirValue value);

/// Sets the type of the block argument to the given type.
MLIR_CAPI_EXPORTED void mlirBlockArgumentSetType(MlirValue value,
                                                 MlirType type);

/// Returns an operation that produced this value as its result. Asserts if the
/// value is not an op result.
MLIR_CAPI_EXPORTED MlirOperation mlirOpResultGetOwner(MlirValue value);

/// Returns the position of the value in the list of results of the operation
/// that produced it.
MLIR_CAPI_EXPORTED intptr_t mlirOpResultGetResultNumber(MlirValue value);

/// Returns the type of the value.
MLIR_CAPI_EXPORTED MlirType mlirValueGetType(MlirValue value);

/// Prints the value to the standard error stream.
MLIR_CAPI_EXPORTED void mlirValueDump(MlirValue value);

/// Prints a value by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void
mlirValuePrint(MlirValue value, MlirStringCallback callback, void *userData);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Parses a type. The type is owned by the context.
MLIR_CAPI_EXPORTED MlirType mlirTypeParseGet(MlirContext context,
                                             MlirStringRef type);

/// Gets the context that a type was created with.
MLIR_CAPI_EXPORTED MlirContext mlirTypeGetContext(MlirType type);

/// Checks whether a type is null.
static inline bool mlirTypeIsNull(MlirType type) { return !type.ptr; }

/// Checks if two types are equal.
MLIR_CAPI_EXPORTED bool mlirTypeEqual(MlirType t1, MlirType t2);

/// Prints a location by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void
mlirTypePrint(MlirType type, MlirStringCallback callback, void *userData);

/// Prints the type to the standard error stream.
MLIR_CAPI_EXPORTED void mlirTypeDump(MlirType type);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

/// Parses an attribute. The attribute is owned by the context.
MLIR_CAPI_EXPORTED MlirAttribute mlirAttributeParseGet(MlirContext context,
                                                       MlirStringRef attr);

/// Gets the context that an attribute was created with.
MLIR_CAPI_EXPORTED MlirContext mlirAttributeGetContext(MlirAttribute attribute);

/// Gets the type of this attribute.
MLIR_CAPI_EXPORTED MlirType mlirAttributeGetType(MlirAttribute attribute);

/// Checks whether an attribute is null.
static inline bool mlirAttributeIsNull(MlirAttribute attr) { return !attr.ptr; }

/// Checks if two attributes are equal.
MLIR_CAPI_EXPORTED bool mlirAttributeEqual(MlirAttribute a1, MlirAttribute a2);

/// Prints an attribute by sending chunks of the string representation and
/// forwarding `userData to `callback`. Note that the callback may be called
/// several times with consecutive chunks of the string.
MLIR_CAPI_EXPORTED void mlirAttributePrint(MlirAttribute attr,
                                           MlirStringCallback callback,
                                           void *userData);

/// Prints the attribute to the standard error stream.
MLIR_CAPI_EXPORTED void mlirAttributeDump(MlirAttribute attr);

/// Associates an attribute with the name. Takes ownership of neither.
MLIR_CAPI_EXPORTED MlirNamedAttribute mlirNamedAttributeGet(MlirIdentifier name,
                                                            MlirAttribute attr);

//===----------------------------------------------------------------------===//
// Identifier API.
//===----------------------------------------------------------------------===//

/// Gets an identifier with the given string value.
MLIR_CAPI_EXPORTED MlirIdentifier mlirIdentifierGet(MlirContext context,
                                                    MlirStringRef str);

/// Returns the context associated with this identifier
MLIR_CAPI_EXPORTED MlirContext mlirIdentifierGetContext(MlirIdentifier);

/// Checks whether two identifiers are the same.
MLIR_CAPI_EXPORTED bool mlirIdentifierEqual(MlirIdentifier ident,
                                            MlirIdentifier other);

/// Gets the string value of the identifier.
MLIR_CAPI_EXPORTED MlirStringRef mlirIdentifierStr(MlirIdentifier ident);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_IR_H
