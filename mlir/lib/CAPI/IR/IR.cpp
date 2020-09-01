//===- IR.cpp - C Interface for Core MLIR APIs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/* ========================================================================== */
/* Printing helper.                                                           */
/* ========================================================================== */

namespace {
/// A simple raw ostream subclass that forwards write_impl calls to the
/// user-supplied callback together with opaque user-supplied data.
class CallbackOstream : public llvm::raw_ostream {
public:
  CallbackOstream(std::function<void(const char *, intptr_t, void *)> callback,
                  void *opaqueData)
      : callback(callback), opaqueData(opaqueData), pos(0u) {}

  void write_impl(const char *ptr, size_t size) override {
    callback(ptr, size, opaqueData);
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

private:
  std::function<void(const char *, intptr_t, void *)> callback;
  void *opaqueData;
  uint64_t pos;
};
} // end namespace

/* ========================================================================== */
/* Context API.                                                               */
/* ========================================================================== */

MlirContext mlirContextCreate() {
  auto *context = new MLIRContext(/*loadAllDialects=*/false);
  return wrap(context);
}

int mlirContextEqual(MlirContext ctx1, MlirContext ctx2) {
  return unwrap(ctx1) == unwrap(ctx2);
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

void mlirLocationPrint(MlirLocation location, MlirStringCallback callback,
                       void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(location).print(stream);
  stream.flush();
}

/* ========================================================================== */
/* Module API.                                                                */
/* ========================================================================== */

MlirModule mlirModuleCreateEmpty(MlirLocation location) {
  return wrap(ModuleOp::create(unwrap(location)));
}

MlirModule mlirModuleCreateParse(MlirContext context, const char *module) {
  OwningModuleRef owning = parseSourceString(module, unwrap(context));
  if (!owning)
    return MlirModule{nullptr};
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

void mlirOperationStateAddResults(MlirOperationState *state, intptr_t n,
                                  MlirType *results) {
  APPEND_ELEMS(MlirType, nResults, results);
}

void mlirOperationStateAddOperands(MlirOperationState *state, intptr_t n,
                                   MlirValue *operands) {
  APPEND_ELEMS(MlirValue, nOperands, operands);
}
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, intptr_t n,
                                       MlirRegion *regions) {
  APPEND_ELEMS(MlirRegion, nRegions, regions);
}
void mlirOperationStateAddSuccessors(MlirOperationState *state, intptr_t n,
                                     MlirBlock *successors) {
  APPEND_ELEMS(MlirBlock, nSuccessors, successors);
}
void mlirOperationStateAddAttributes(MlirOperationState *state, intptr_t n,
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
  for (intptr_t i = 0; i < state->nAttributes; ++i)
    cppState.addAttribute(state->attributes[i].name,
                          unwrap(state->attributes[i].attribute));

  for (intptr_t i = 0; i < state->nRegions; ++i)
    cppState.addRegion(std::unique_ptr<Region>(unwrap(state->regions[i])));

  MlirOperation result = wrap(Operation::create(cppState));
  free(state->results);
  free(state->operands);
  free(state->successors);
  free(state->regions);
  free(state->attributes);
  return result;
}

void mlirOperationDestroy(MlirOperation op) { unwrap(op)->erase(); }

int mlirOperationIsNull(MlirOperation op) { return unwrap(op) == nullptr; }

intptr_t mlirOperationGetNumRegions(MlirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumRegions());
}

MlirRegion mlirOperationGetRegion(MlirOperation op, intptr_t pos) {
  return wrap(&unwrap(op)->getRegion(static_cast<unsigned>(pos)));
}

MlirOperation mlirOperationGetNextInBlock(MlirOperation op) {
  return wrap(unwrap(op)->getNextNode());
}

intptr_t mlirOperationGetNumOperands(MlirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumOperands());
}

MlirValue mlirOperationGetOperand(MlirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getOperand(static_cast<unsigned>(pos)));
}

intptr_t mlirOperationGetNumResults(MlirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumResults());
}

MlirValue mlirOperationGetResult(MlirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getResult(static_cast<unsigned>(pos)));
}

intptr_t mlirOperationGetNumSuccessors(MlirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getNumSuccessors());
}

MlirBlock mlirOperationGetSuccessor(MlirOperation op, intptr_t pos) {
  return wrap(unwrap(op)->getSuccessor(static_cast<unsigned>(pos)));
}

intptr_t mlirOperationGetNumAttributes(MlirOperation op) {
  return static_cast<intptr_t>(unwrap(op)->getAttrs().size());
}

MlirNamedAttribute mlirOperationGetAttribute(MlirOperation op, intptr_t pos) {
  NamedAttribute attr = unwrap(op)->getAttrs()[pos];
  return MlirNamedAttribute{attr.first.c_str(), wrap(attr.second)};
}

MlirAttribute mlirOperationGetAttributeByName(MlirOperation op,
                                              const char *name) {
  return wrap(unwrap(op)->getAttr(name));
}

void mlirOperationPrint(MlirOperation op, MlirStringCallback callback,
                        void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(op)->print(stream);
  stream.flush();
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

void mlirRegionInsertOwnedBlock(MlirRegion region, intptr_t pos,
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

MlirBlock mlirBlockCreate(intptr_t nArgs, MlirType *args) {
  Block *b = new Block;
  for (intptr_t i = 0; i < nArgs; ++i)
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

void mlirBlockInsertOwnedOperation(MlirBlock block, intptr_t pos,
                                   MlirOperation operation) {
  auto &opList = unwrap(block)->getOperations();
  opList.insert(std::next(opList.begin(), pos), unwrap(operation));
}

void mlirBlockDestroy(MlirBlock block) { delete unwrap(block); }

int mlirBlockIsNull(MlirBlock block) { return unwrap(block) == nullptr; }

intptr_t mlirBlockGetNumArguments(MlirBlock block) {
  return static_cast<intptr_t>(unwrap(block)->getNumArguments());
}

MlirValue mlirBlockGetArgument(MlirBlock block, intptr_t pos) {
  return wrap(unwrap(block)->getArgument(static_cast<unsigned>(pos)));
}

void mlirBlockPrint(MlirBlock block, MlirStringCallback callback,
                    void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(block)->print(stream);
  stream.flush();
}

/* ========================================================================== */
/* Value API.                                                                 */
/* ========================================================================== */

MlirType mlirValueGetType(MlirValue value) {
  return wrap(unwrap(value).getType());
}

void mlirValuePrint(MlirValue value, MlirStringCallback callback,
                    void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(value).print(stream);
  stream.flush();
}

/* ========================================================================== */
/* Type API.                                                                  */
/* ========================================================================== */

MlirType mlirTypeParseGet(MlirContext context, const char *type) {
  return wrap(mlir::parseType(type, unwrap(context)));
}

MlirContext mlirTypeGetContext(MlirType type) {
  return wrap(unwrap(type).getContext());
}

int mlirTypeEqual(MlirType t1, MlirType t2) { return unwrap(t1) == unwrap(t2); }

void mlirTypePrint(MlirType type, MlirStringCallback callback, void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(type).print(stream);
  stream.flush();
}

void mlirTypeDump(MlirType type) { unwrap(type).dump(); }

/* ========================================================================== */
/* Attribute API.                                                             */
/* ========================================================================== */

MlirAttribute mlirAttributeParseGet(MlirContext context, const char *attr) {
  return wrap(mlir::parseAttribute(attr, unwrap(context)));
}

int mlirAttributeEqual(MlirAttribute a1, MlirAttribute a2) {
  return unwrap(a1) == unwrap(a2);
}

void mlirAttributePrint(MlirAttribute attr, MlirStringCallback callback,
                        void *userData) {
  CallbackOstream stream(callback, userData);
  unwrap(attr).print(stream);
  stream.flush();
}

void mlirAttributeDump(MlirAttribute attr) { unwrap(attr).dump(); }

MlirNamedAttribute mlirNamedAttributeGet(const char *name, MlirAttribute attr) {
  return MlirNamedAttribute{name, attr};
}
