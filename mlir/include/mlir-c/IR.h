/*===-- mlir-c/IR.h - C API to Core MLIR IR classes ---------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to MLIR core IR classes.              *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef MLIR_C_IR_H
#define MLIR_C_IR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================*/
/** Opaque type declarations.
 *
 * Types are exposed to C bindings as structs containing opaque pointers. They
 * are not supposed to be inspected from C. This allows the underlying
 * representation to change without affecting the API users. The use of structs
 * instead of typedefs enables some type safety as structs are not implicitly
 * convertible to each other.
 *
 * Instances of these types may or may not own the underlying object (most often
 * only point to an IR fragment without owning it). The ownership semantics is
 * defined by how an instance of the type was obtained.
 */
/*============================================================================*/

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirContext, void);
DEFINE_C_API_STRUCT(MlirOperation, void);
DEFINE_C_API_STRUCT(MlirBlock, void);
DEFINE_C_API_STRUCT(MlirRegion, void);

DEFINE_C_API_STRUCT(MlirValue, const void);
DEFINE_C_API_STRUCT(MlirAttribute, const void);
DEFINE_C_API_STRUCT(MlirType, const void);
DEFINE_C_API_STRUCT(MlirLocation, const void);
DEFINE_C_API_STRUCT(MlirModule, const void);

/** Named MLIR attribute.
 *
 * A named attribute is essentially a (name, attribute) pair where the name is
 * a string.
 */
struct MlirNamedAttribute {
  const char *name;
  MlirAttribute attribute;
};
typedef struct MlirNamedAttribute MlirNamedAttribute;

/** A callback for returning string references.
 *
 * This function is called back by the functions that need to return a reference
 * to the portion of the string with the following arguments:
 *   - a pointer to the beginning of a string;
 *   - the length of the string (the pointer may point to a larger buffer, not
 *     necessarily null-terminated);
 *   - a pointer to user data forwarded from the printing call.
 */
typedef void (*MlirStringCallback)(const char *, intptr_t, void *);

/*============================================================================*/
/* Context API.                                                               */
/*============================================================================*/

/** Creates an MLIR context and transfers its ownership to the caller. */
MlirContext mlirContextCreate();

/** Checks if two contexts are equal. */
int mlirContextEqual(MlirContext ctx1, MlirContext ctx2);

/** Takes an MLIR context owned by the caller and destroys it. */
void mlirContextDestroy(MlirContext context);

/** Sets whether unregistered dialects are allowed in this context. */
void mlirContextSetAllowUnregisteredDialects(MlirContext context, int allow);

/** Returns whether the context allows unregistered dialects. */
int mlirContextGetAllowUnregisteredDialects(MlirContext context);

/*============================================================================*/
/* Location API.                                                              */
/*============================================================================*/

/** Creates an File/Line/Column location owned by the given context. */
MlirLocation mlirLocationFileLineColGet(MlirContext context,
                                        const char *filename, unsigned line,
                                        unsigned col);

/** Creates a location with unknown position owned by the given context. */
MlirLocation mlirLocationUnknownGet(MlirContext context);

/** Gets the context that a location was created with. */
MlirContext mlirLocationGetContext(MlirLocation location);

/** Prints a location by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirLocationPrint(MlirLocation location, MlirStringCallback callback,
                       void *userData);

/*============================================================================*/
/* Module API.                                                                */
/*============================================================================*/

/** Creates a new, empty module and transfers ownership to the caller. */
MlirModule mlirModuleCreateEmpty(MlirLocation location);

/** Parses a module from the string and transfers ownership to the caller. */
MlirModule mlirModuleCreateParse(MlirContext context, const char *module);

/** Gets the context that a module was created with. */
MlirContext mlirModuleGetContext(MlirModule module);

/** Checks whether a module is null. */
inline int mlirModuleIsNull(MlirModule module) { return !module.ptr; }

/** Takes a module owned by the caller and deletes it. */
void mlirModuleDestroy(MlirModule module);

/** Views the module as a generic operation. */
MlirOperation mlirModuleGetOperation(MlirModule module);

/*============================================================================*/
/* Operation state.                                                           */
/*============================================================================*/

/** An auxiliary class for constructing operations.
 *
 * This class contains all the information necessary to construct the operation.
 * It owns the MlirRegions it has pointers to and does not own anything else.
 * By default, the state can be constructed from a name and location, the latter
 * being also used to access the context, and has no other components. These
 * components can be added progressively until the operation is constructed.
 * Users are not expected to rely on the internals of this class and should use
 * mlirOperationState* functions instead.
 */
struct MlirOperationState {
  const char *name;
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
};
typedef struct MlirOperationState MlirOperationState;

/** Constructs an operation state from a name and a location. */
MlirOperationState mlirOperationStateGet(const char *name, MlirLocation loc);

/** Adds a list of components to the operation state. */
void mlirOperationStateAddResults(MlirOperationState *state, intptr_t n,
                                  MlirType *results);
void mlirOperationStateAddOperands(MlirOperationState *state, intptr_t n,
                                   MlirValue *operands);
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, intptr_t n,
                                       MlirRegion *regions);
void mlirOperationStateAddSuccessors(MlirOperationState *state, intptr_t n,
                                     MlirBlock *successors);
void mlirOperationStateAddAttributes(MlirOperationState *state, intptr_t n,
                                     MlirNamedAttribute *attributes);

/*============================================================================*/
/* Operation API.                                                             */
/*============================================================================*/

/** Creates an operation and transfers ownership to the caller. */
MlirOperation mlirOperationCreate(const MlirOperationState *state);

/** Takes an operation owned by the caller and destroys it. */
void mlirOperationDestroy(MlirOperation op);

/** Checks whether the underlying operation is null. */
int mlirOperationIsNull(MlirOperation op);

/** Returns the number of regions attached to the given operation. */
intptr_t mlirOperationGetNumRegions(MlirOperation op);

/** Returns `pos`-th region attached to the operation. */
MlirRegion mlirOperationGetRegion(MlirOperation op, intptr_t pos);

/** Returns an operation immediately following the given operation it its
 * enclosing block. */
MlirOperation mlirOperationGetNextInBlock(MlirOperation op);

/** Returns the number of operands of the operation. */
intptr_t mlirOperationGetNumOperands(MlirOperation op);

/** Returns `pos`-th operand of the operation. */
MlirValue mlirOperationGetOperand(MlirOperation op, intptr_t pos);

/** Returns the number of results of the operation. */
intptr_t mlirOperationGetNumResults(MlirOperation op);

/** Returns `pos`-th result of the operation. */
MlirValue mlirOperationGetResult(MlirOperation op, intptr_t pos);

/** Returns the number of successor blocks of the operation. */
intptr_t mlirOperationGetNumSuccessors(MlirOperation op);

/** Returns `pos`-th successor of the operation. */
MlirBlock mlirOperationGetSuccessor(MlirOperation op, intptr_t pos);

/** Returns the number of attributes attached to the operation. */
intptr_t mlirOperationGetNumAttributes(MlirOperation op);

/** Return `pos`-th attribute of the operation. */
MlirNamedAttribute mlirOperationGetAttribute(MlirOperation op, intptr_t pos);

/** Returns an attribute attached to the operation given its name. */
MlirAttribute mlirOperationGetAttributeByName(MlirOperation op,
                                              const char *name);

/** Prints an operation by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirOperationPrint(MlirOperation op, MlirStringCallback callback,
                        void *userData);

/** Prints an operation to stderr. */
void mlirOperationDump(MlirOperation op);

/*============================================================================*/
/* Region API.                                                                */
/*============================================================================*/

/** Creates a new empty region and transfers ownership to the caller. */
MlirRegion mlirRegionCreate();

/** Takes a region owned by the caller and destroys it. */
void mlirRegionDestroy(MlirRegion region);

/** Checks whether a region is null. */
int mlirRegionIsNull(MlirRegion region);

/** Gets the first block in the region. */
MlirBlock mlirRegionGetFirstBlock(MlirRegion region);

/** Takes a block owned by the caller and appends it to the given region. */
void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block);

/** Takes a block owned by the caller and inserts it at `pos` to the given
 * region. This is an expensive operation that linearly scans the region, prefer
 * insertAfter/Before instead. */
void mlirRegionInsertOwnedBlock(MlirRegion region, intptr_t pos,
                                MlirBlock block);

/** Takes a block owned by the caller and inserts it after the (non-owned)
 * reference block in the given region. The reference block must belong to the
 * region. If the reference block is null, prepends the block to the region. */
void mlirRegionInsertOwnedBlockAfter(MlirRegion region, MlirBlock reference,
                                     MlirBlock block);

/** Takes a block owned by the caller and inserts it before the (non-owned)
 * reference block in the given region. The reference block must belong to the
 * region. If the reference block is null, appends the block to the region. */
void mlirRegionInsertOwnedBlockBefore(MlirRegion region, MlirBlock reference,
                                      MlirBlock block);

/*============================================================================*/
/* Block API.                                                                 */
/*============================================================================*/

/** Creates a new empty block with the given argument types and transfers
 * ownership to the caller. */
MlirBlock mlirBlockCreate(intptr_t nArgs, MlirType *args);

/** Takes a block owned by the caller and destroys it. */
void mlirBlockDestroy(MlirBlock block);

/** Checks whether a block is null. */
int mlirBlockIsNull(MlirBlock block);

/** Returns the block immediately following the given block in its parent
 * region. */
MlirBlock mlirBlockGetNextInRegion(MlirBlock block);

/** Returns the first operation in the block. */
MlirOperation mlirBlockGetFirstOperation(MlirBlock block);

/** Takes an operation owned by the caller and appends it to the block. */
void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation operation);

/** Takes an operation owned by the caller and inserts it as `pos` to the block.
   This is an expensive operation that scans the block linearly, prefer
   insertBefore/After instead. */
void mlirBlockInsertOwnedOperation(MlirBlock block, intptr_t pos,
                                   MlirOperation operation);

/** Takes an operation owned by the caller and inserts it after the (non-owned)
 * reference operation in the given block. If the reference is null, prepends
 * the operation. Otherwise, the reference must belong to the block. */
void mlirBlockInsertOwnedOperationAfter(MlirBlock block,
                                        MlirOperation reference,
                                        MlirOperation operation);

/** Takes an operation owned by the caller and inserts it before the (non-owned)
 * reference operation in the given block. If the reference is null, appends the
 * operation. Otherwise, the reference must belong to the block. */
void mlirBlockInsertOwnedOperationBefore(MlirBlock block,
                                         MlirOperation reference,
                                         MlirOperation operation);

/** Returns the number of arguments of the block. */
intptr_t mlirBlockGetNumArguments(MlirBlock block);

/** Returns `pos`-th argument of the block. */
MlirValue mlirBlockGetArgument(MlirBlock block, intptr_t pos);

/** Prints a block by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirBlockPrint(MlirBlock block, MlirStringCallback callback,
                    void *userData);

/*============================================================================*/
/* Value API.                                                                 */
/*============================================================================*/

/** Returns the type of the value. */
MlirType mlirValueGetType(MlirValue value);

/** Prints a value by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirValuePrint(MlirValue value, MlirStringCallback callback,
                    void *userData);

/*============================================================================*/
/* Type API.                                                                  */
/*============================================================================*/

/** Parses a type. The type is owned by the context. */
MlirType mlirTypeParseGet(MlirContext context, const char *type);

/** Gets the context that a type was created with. */
MlirContext mlirTypeGetContext(MlirType type);

/** Checks whether a type is null. */
inline int mlirTypeIsNull(MlirType type) { return !type.ptr; }

/** Checks if two types are equal. */
int mlirTypeEqual(MlirType t1, MlirType t2);

/** Prints a location by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirTypePrint(MlirType type, MlirStringCallback callback, void *userData);

/** Prints the type to the standard error stream. */
void mlirTypeDump(MlirType type);

/*============================================================================*/
/* Attribute API.                                                             */
/*============================================================================*/

/** Parses an attribute. The attribute is owned by the context. */
MlirAttribute mlirAttributeParseGet(MlirContext context, const char *attr);

/** Gets the context that an attribute was created with. */
MlirContext mlirAttributeGetContext(MlirAttribute attribute);

/** Checks whether an attribute is null. */
inline int mlirAttributeIsNull(MlirAttribute attr) { return !attr.ptr; }

/** Checks if two attributes are equal. */
int mlirAttributeEqual(MlirAttribute a1, MlirAttribute a2);

/** Prints an attribute by sending chunks of the string representation and
 * forwarding `userData to `callback`. Note that the callback may be called
 * several times with consecutive chunks of the string. */
void mlirAttributePrint(MlirAttribute attr, MlirStringCallback callback,
                        void *userData);

/** Prints the attribute to the standard error stream. */
void mlirAttributeDump(MlirAttribute attr);

/** Associates an attribute with the name. Takes ownership of neither. */
MlirNamedAttribute mlirNamedAttributeGet(const char *name, MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_IR_H
