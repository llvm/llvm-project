//===- IR.cpp - C Interface for Core MLIR APIs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"

using namespace mlir;

/* ========================================================================== */
/* Definitions of methods for non-owning structures used in C API.            */
/* ========================================================================== */

#define DEFINE_C_API_PTR_METHODS(name, cpptype)                                \
  static name wrap(cpptype *cpp) { return name{cpp}; }                         \
  static cpptype *unwrap(name c) { return static_cast<cpptype *>(c.ptr); }

DEFINE_C_API_PTR_METHODS(MlirContext, MLIRContext)
DEFINE_C_API_PTR_METHODS(MlirOperation, Operation)
DEFINE_C_API_PTR_METHODS(MlirBlock, Block)
DEFINE_C_API_PTR_METHODS(MlirRegion, Region)

#define DEFINE_C_API_METHODS(name, cpptype)                                    \
  static name wrap(cpptype cpp) { return name{cpp.getAsOpaquePointer()}; }     \
  static cpptype unwrap(name c) { return cpptype::getFromOpaquePointer(c.ptr); }

DEFINE_C_API_METHODS(MlirAttribute, Attribute)
DEFINE_C_API_METHODS(MlirLocation, Location);
DEFINE_C_API_METHODS(MlirType, Type)
DEFINE_C_API_METHODS(MlirValue, Value)
DEFINE_C_API_METHODS(MlirModule, ModuleOp)

template <typename CppTy, typename CTy>
static ArrayRef<CppTy> unwrapList(unsigned size, CTy *first,
                                  SmallVectorImpl<CppTy> &storage) {
  static_assert(
      std::is_same<decltype(unwrap(std::declval<CTy>())), CppTy>::value,
      "incompatible C and C++ types");

  if (size == 0)
    return llvm::None;

  assert(storage.empty() && "expected to populate storage");
  storage.reserve(size);
  for (unsigned i = 0; i < size; ++i)
    storage.push_back(unwrap(*(first + i)));
  return storage;
}

/* ========================================================================== */
/* Context API.                                                               */
/* ========================================================================== */

MlirContext mlirContextCreate() {
  auto *context = new MLIRContext;
  return wrap(context);
}

void mlirContextDestroy(MlirContext context) { delete unwrap(context); }

/* ========================================================================== */
/* Location API.                                                              */
/* ========================================================================== */

MlirLocation mlirLocationFileLineColGet(MlirContext context,
                                        const char *filename, unsigned line,
                                        unsigned col) {
  return wrap(FileLineColLoc::get(filename, line, col, unwrap(context)));
}

MlirLocation mlirLocationUnknownGet(MlirContext context) {
  return wrap(UnknownLoc::get(unwrap(context)));
}

/* ========================================================================== */
/* Module API.                                                                */
/* ========================================================================== */

MlirModule mlirModuleCreateEmpty(MlirLocation location) {
  return wrap(ModuleOp::create(unwrap(location)));
}

MlirModule mlirModuleCreateParse(MlirContext context, const char *module) {
  OwningModuleRef owning = parseSourceString(module, unwrap(context));
  return MlirModule{owning.release().getOperation()};
}

void mlirModuleDestroy(MlirModule module) {
  // Transfer ownership to an OwningModuleRef so that its destructor is called.
  OwningModuleRef(unwrap(module));
}

MlirOperation mlirModuleGetOperation(MlirModule module) {
  return wrap(unwrap(module).getOperation());
}

/* ========================================================================== */
/* Operation state API.                                                       */
/* ========================================================================== */

MlirOperationState mlirOperationStateGet(const char *name, MlirLocation loc) {
  MlirOperationState state;
  state.name = name;
  state.location = loc;
  state.nResults = 0;
  state.results = nullptr;
  state.nOperands = 0;
  state.operands = nullptr;
  state.nRegions = 0;
  state.regions = nullptr;
  state.nSuccessors = 0;
  state.successors = nullptr;
  state.nAttributes = 0;
  state.attributes = nullptr;
  return state;
}

#define APPEND_ELEMS(type, sizeName, elemName)                                 \
  state->elemName =                                                            \
      (type *)realloc(state->elemName, (state->sizeName + n) * sizeof(type));  \
  memcpy(state->elemName + state->sizeName, elemName, n * sizeof(type));       \
  state->sizeName += n;

void mlirOperationStateAddResults(MlirOperationState *state, unsigned n,
                                  MlirType *results) {
  APPEND_ELEMS(MlirType, nResults, results);
}

void mlirOperationStateAddOperands(MlirOperationState *state, unsigned n,
                                   MlirValue *operands) {
  APPEND_ELEMS(MlirValue, nOperands, operands);
}
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, unsigned n,
                                       MlirRegion *regions) {
  APPEND_ELEMS(MlirRegion, nRegions, regions);
}
void mlirOperationStateAddSuccessors(MlirOperationState *state, unsigned n,
                                     MlirBlock *successors) {
  APPEND_ELEMS(MlirBlock, nSuccessors, successors);
}
void mlirOperationStateAddAttributes(MlirOperationState *state, unsigned n,
                                     MlirNamedAttribute *attributes) {
  APPEND_ELEMS(MlirNamedAttribute, nAttributes, attributes);
}

/* ========================================================================== */
/* Operation API.                                                             */
/* ========================================================================== */

MlirOperation mlirOperationCreate(const MlirOperationState *state) {
  assert(state);
  OperationState cppState(unwrap(state->location), state->name);
  SmallVector<Type, 4> resultStorage;
  SmallVector<Value, 8> operandStorage;
  SmallVector<Block *, 2> successorStorage;
  cppState.addTypes(unwrapList(state->nResults, state->results, resultStorage));
  cppState.addOperands(
      unwrapList(state->nOperands, state->operands, operandStorage));
  cppState.addSuccessors(
      unwrapList(state->nSuccessors, state->successors, successorStorage));

  cppState.attributes.reserve(state->nAttributes);
  for (unsigned i = 0; i < state->nAttributes; ++i)
    cppState.addAttribute(state->attributes[i].name,
                          unwrap(state->attributes[i].attribute));

  for (unsigned i = 0; i < state->nRegions; ++i)
    cppState.addRegion(std::unique_ptr<Region>(unwrap(state->regions[i])));

  free(state->results);
  free(state->operands);
  free(state->regions);
  free(state->successors);
  free(state->attributes);

  return wrap(Operation::create(cppState));
}

void mlirOperationDestroy(MlirOperation op) { unwrap(op)->erase(); }

int mlirOperationIsNull(MlirOperation op) { return unwrap(op) == nullptr; }

unsigned mlirOperationGetNumRegions(MlirOperation op) {
  return unwrap(op)->getNumRegions();
}

MlirRegion mlirOperationGetRegion(MlirOperation op, unsigned pos) {
  return wrap(&unwrap(op)->getRegion(pos));
}

MlirOperation mlirOperationGetNextInBlock(MlirOperation op) {
  return wrap(unwrap(op)->getNextNode());
}

unsigned mlirOperationGetNumOperands(MlirOperation op) {
  return unwrap(op)->getNumOperands();
}

MlirValue mlirOperationGetOperand(MlirOperation op, unsigned pos) {
  return wrap(unwrap(op)->getOperand(pos));
}

unsigned mlirOperationGetNumResults(MlirOperation op) {
  return unwrap(op)->getNumResults();
}

MlirValue mlirOperationGetResult(MlirOperation op, unsigned pos) {
  return wrap(unwrap(op)->getResult(pos));
}

unsigned mlirOperationGetNumSuccessors(MlirOperation op) {
  return unwrap(op)->getNumSuccessors();
}

MlirBlock mlirOperationGetSuccessor(MlirOperation op, unsigned pos) {
  return wrap(unwrap(op)->getSuccessor(pos));
}

unsigned mlirOperationGetNumAttributes(MlirOperation op) {
  return unwrap(op)->getAttrs().size();
}

MlirNamedAttribute mlirOperationGetAttribute(MlirOperation op, unsigned pos) {
  NamedAttribute attr = unwrap(op)->getAttrs()[pos];
  return MlirNamedAttribute{attr.first.c_str(), wrap(attr.second)};
}

MlirAttribute mlirOperationGetAttributeByName(MlirOperation op,
                                              const char *name) {
  return wrap(unwrap(op)->getAttr(name));
}

void mlirOperationDump(MlirOperation op) { return unwrap(op)->dump(); }

/* ========================================================================== */
/* Region API.                                                                */
/* ========================================================================== */

MlirRegion mlirRegionCreate() { return wrap(new Region); }

MlirBlock mlirRegionGetFirstBlock(MlirRegion region) {
  Region *cppRegion = unwrap(region);
  if (cppRegion->empty())
    return wrap(static_cast<Block *>(nullptr));
  return wrap(&cppRegion->front());
}

void mlirRegionAppendOwnedBlock(MlirRegion region, MlirBlock block) {
  unwrap(region)->push_back(unwrap(block));
}

void mlirRegionInsertOwnedBlock(MlirRegion region, unsigned pos,
                                MlirBlock block) {
  auto &blockList = unwrap(region)->getBlocks();
  blockList.insert(std::next(blockList.begin(), pos), unwrap(block));
}

void mlirRegionDestroy(MlirRegion region) {
  delete static_cast<Region *>(region.ptr);
}

int mlirRegionIsNull(MlirRegion region) { return unwrap(region) == nullptr; }

/* ========================================================================== */
/* Block API.                                                                 */
/* ========================================================================== */

MlirBlock mlirBlockCreate(unsigned nArgs, MlirType *args) {
  Block *b = new Block;
  for (unsigned i = 0; i < nArgs; ++i)
    b->addArgument(unwrap(args[i]));
  return wrap(b);
}

MlirBlock mlirBlockGetNextInRegion(MlirBlock block) {
  return wrap(unwrap(block)->getNextNode());
}

MlirOperation mlirBlockGetFirstOperation(MlirBlock block) {
  Block *cppBlock = unwrap(block);
  if (cppBlock->empty())
    return wrap(static_cast<Operation *>(nullptr));
  return wrap(&cppBlock->front());
}

void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation operation) {
  unwrap(block)->push_back(unwrap(operation));
}

void mlirBlockInsertOwnedOperation(MlirBlock block, unsigned pos,
                                   MlirOperation operation) {
  auto &opList = unwrap(block)->getOperations();
  opList.insert(std::next(opList.begin(), pos), unwrap(operation));
}

void mlirBlockDestroy(MlirBlock block) { delete unwrap(block); }

int mlirBlockIsNull(MlirBlock block) { return unwrap(block) == nullptr; }

unsigned mlirBlockGetNumArguments(MlirBlock block) {
  return unwrap(block)->getNumArguments();
}

MlirValue mlirBlockGetArgument(MlirBlock block, unsigned pos) {
  return wrap(unwrap(block)->getArgument(pos));
}

/* ========================================================================== */
/* Value API.                                                                 */
/* ========================================================================== */

MlirType mlirValueGetType(MlirValue value) {
  return wrap(unwrap(value).getType());
}

/* ========================================================================== */
/* Type API.                                                                  */
/* ========================================================================== */

MlirType mlirTypeParseGet(MlirContext context, const char *type) {
  return wrap(mlir::parseType(type, unwrap(context)));
}

void mlirTypeDump(MlirType type) { unwrap(type).dump(); }

/* ========================================================================== */
/* Attribute API.                                                             */
/* ========================================================================== */

MlirAttribute mlirAttributeParseGet(MlirContext context, const char *attr) {
  return wrap(mlir::parseAttribute(attr, unwrap(context)));
}

void mlirAttributeDump(MlirAttribute attr) { unwrap(attr).dump(); }

MlirNamedAttribute mlirNamedAttributeGet(const char *name, MlirAttribute attr) {
  return MlirNamedAttribute{name, attr};
}
