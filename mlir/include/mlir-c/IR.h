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
 * Instaces of these types may or may not own the underlying object (most often
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

#undef DEFINE_C_API_STRUCT

/** Named MLIR attribute.
 *
 * A named attribute is essentially a (name, attrbute) pair where the name is
 * a string.
 */
struct MlirNamedAttribute {
  const char *name;
  MlirAttribute attribute;
};
typedef struct MlirNamedAttribute MlirNamedAttribute;

/*============================================================================*/
/* Context API.                                                               */
/*============================================================================*/

/** Creates an MLIR context and transfers its ownership to the caller. */
MlirContext mlirContextCreate();

/** Takes an MLIR context owned by the caller and destroys it. */
void mlirContextDestroy(MlirContext context);

/*============================================================================*/
/* Location API.                                                              */
/*============================================================================*/

/** Creates an File/Line/Column location owned by the given context. */
MlirLocation mlirLocationFileLineColGet(MlirContext context,
                                        const char *filename, unsigned line,
                                        unsigned col);

/** Creates a location with unknown position owned by the given context. */
MlirLocation mlirLocationUnknownGet(MlirContext context);

/*============================================================================*/
/* Module API.                                                                */
/*============================================================================*/

/** Creates a new, empty module and transfers ownership to the caller. */
MlirModule mlirModuleCreateEmpty(MlirLocation location);

/** Parses a module from the string and transfers ownership to the caller. */
MlirModule mlirModuleCreateParse(MlirContext context, const char *module);

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
  unsigned nResults;
  MlirType *results;
  unsigned nOperands;
  MlirValue *operands;
  unsigned nRegions;
  MlirRegion *regions;
  unsigned nSuccessors;
  MlirBlock *successors;
  unsigned nAttributes;
  MlirNamedAttribute *attributes;
};
typedef struct MlirOperationState MlirOperationState;

/** Constructs an operation state from a name and a location. */
MlirOperationState mlirOperationStateGet(const char *name, MlirLocation loc);

/** Adds a list of components to the operation state. */
void mlirOperationStateAddResults(MlirOperationState *state, unsigned n,
                                  MlirType *results);
void mlirOperationStateAddOperands(MlirOperationState *state, unsigned n,
                                   MlirValue *operands);
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, unsigned n,
                                       MlirRegion *regions);
void mlirOperationStateAddSuccessors(MlirOperationState *state, unsigned n,
                                     MlirBlock *successors);
void mlirOperationStateAddAttributes(MlirOperationState *state, unsigned n,
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
unsigned mlirOperationGetNumRegions(MlirOperation op);

/** Returns `pos`-th region attached to the operation. */
MlirRegion mlirOperationGetRegion(MlirOperation op, unsigned pos);

/** Returns an operation immediately following the given operation it its
 * enclosing block. */
MlirOperation mlirOperationGetNextInBlock(MlirOperation op);

/** Returns the number of operands of the operation. */
unsigned mlirOperationGetNumOperands(MlirOperation op);

/** Returns `pos`-th operand of the operation. */
MlirValue mlirOperationGetOperand(MlirOperation op, unsigned pos);

/** Returns the number of results of the operation. */
unsigned mlirOperationGetNumResults(MlirOperation op);

/** Returns `pos`-th result of the operation. */
MlirValue mlirOperationGetResult(MlirOperation op, unsigned pos);

/** Returns the number of successor blocks of the operation. */
unsigned mlirOperationGetNumSuccessors(MlirOperation op);

/** Returns `pos`-th successor of the operation. */
MlirBlock mlirOperationGetSuccessor(MlirOperation op, unsigned pos);

/** Returns the number of attributes attached to the operation. */
unsigned mlirOperationGetNumAttributes(MlirOperation op);

/** Return `pos`-th attribute of the operation. */
MlirNamedAttribute mlirOperationGetAttribute(MlirOperation op, unsigned pos);

/** Returns an attrbute attached to the operation given its name. */
MlirAttribute mlirOperationGetAttributeByName(MlirOperation op,
                                              const char *name);
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
 * region. */
void mlirRegionInsertOwnedBlock(MlirRegion region, unsigned pos,
                                MlirBlock block);

/*============================================================================*/
/* Block API.                                                                 */
/*============================================================================*/

/** Creates a new empty block with the given argument types and transfers
 * ownership to the caller. */
MlirBlock mlirBlockCreate(unsigned nArgs, MlirType *args);

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
 */
void mlirBlockInsertOwnedOperation(MlirBlock block, unsigned pos,
                                   MlirOperation operation);

/** Returns the number of arguments of the block. */
unsigned mlirBlockGetNumArguments(MlirBlock block);

/** Returns `pos`-th argument of the block. */
MlirValue mlirBlockGetArgument(MlirBlock block, unsigned pos);

/*============================================================================*/
/* Value API.                                                                 */
/*============================================================================*/

/** Returns the type of the value. */
MlirType mlirValueGetType(MlirValue value);

/*============================================================================*/
/* Type API.                                                                  */
/*============================================================================*/

/** Parses a type. The type is owned by the context. */
MlirType mlirTypeParseGet(MlirContext context, const char *type);

/** Prints the type to the standard error stream. */
void mlirTypeDump(MlirType type);

/*============================================================================*/
/* Attribute API.                                                             */
/*============================================================================*/

/** Parses an attribute. The attribute is owned by the context. */
MlirAttribute mlirAttributeParseGet(MlirContext context, const char *attr);

/** Prints the attrbute to the standard error stream. */
void mlirAttributeDump(MlirAttribute attr);

/** Associates an attribute with the name. Takes ownership of neither. */
MlirNamedAttribute mlirNamedAttributeGet(const char *name, MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_IR_H
