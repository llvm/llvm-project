//===- IR.cpp - C Interface for Core MLIR APIs ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Context API.
//===----------------------------------------------------------------------===//

MlirContext mlirContextCreate() {
  auto *context = new MLIRContext;
  return wrap(context);
}

bool mlirContextEqual(MlirContext ctx1, MlirContext ctx2) {
  return unwrap(ctx1) == unwrap(ctx2);
}

void mlirContextDestroy(MlirContext context) { delete unwrap(context); }

void mlirContextSetAllowUnregisteredDialects(MlirContext context, bool allow) {
  unwrap(context)->allowUnregisteredDialects(allow);
}

bool mlirContextGetAllowUnregisteredDialects(MlirContext context) {
  return unwrap(context)->allowsUnregisteredDialects();
}
intptr_t mlirContextGetNumRegisteredDialects(MlirContext context) {
  return static_cast<intptr_t>(unwrap(context)->getAvailableDialects().size());
}

// TODO: expose a cheaper way than constructing + sorting a vector only to take
// its size.
intptr_t mlirContextGetNumLoadedDialects(MlirContext context) {
  return static_cast<intptr_t>(unwrap(context)->getLoadedDialects().size());
}

MlirDialect mlirContextGetOrLoadDialect(MlirContext context,
                                        MlirStringRef name) {
  return wrap(unwrap(context)->getOrLoadDialect(unwrap(name)));
}

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MlirContext mlirDialectGetContext(MlirDialect dialect) {
  return wrap(unwrap(dialect)->getContext());
}

bool mlirDialectEqual(MlirDialect dialect1, MlirDialect dialect2) {
  return unwrap(dialect1) == unwrap(dialect2);
}

MlirStringRef mlirDialectGetNamespace(MlirDialect dialect) {
  return wrap(unwrap(dialect)->getNamespace());
}

//===----------------------------------------------------------------------===//
// Printing flags API.
//===----------------------------------------------------------------------===//

MlirOpPrintingFlags mlirOpPrintingFlagsCreate() {
  return wrap(new OpPrintingFlags());
}

void mlirOpPrintingFlagsDestroy(MlirOpPrintingFlags flags) {
  delete unwrap(flags);
}

void mlirOpPrintingFlagsElideLargeElementsAttrs(MlirOpPrintingFlags flags,
                                                intptr_t largeElementLimit) {
  unwrap(flags)->elideLargeElementsAttrs(largeElementLimit);
}

void mlirOpPrintingFlagsEnableDebugInfo(MlirOpPrintingFlags flags,
                                        bool prettyForm) {
  unwrap(flags)->enableDebugInfo(/*prettyForm=*/prettyForm);
}

void mlirOpPrintingFlagsPrintGenericOpForm(MlirOpPrintingFlags flags) {
  unwrap(flags)->printGenericOpForm();
}

void mlirOpPrintingFlagsUseLocalScope(MlirOpPrintingFlags flags) {
  unwrap(flags)->useLocalScope();
}

//===----------------------------------------------------------------------===//
// Location API.
//===----------------------------------------------------------------------===//

MlirLocation mlirLocationFileLineColGet(MlirContext context,
                                        MlirStringRef filename, unsigned line,
                                        unsigned col) {
  return wrap(
      FileLineColLoc::get(unwrap(filename), line, col, unwrap(context)));
}

MlirLocation mlirLocationCallSiteGet(MlirLocation callee, MlirLocation caller) {
  return wrap(CallSiteLoc::get(unwrap(callee), unwrap(caller)));
}

MlirLocation mlirLocationUnknownGet(MlirContext context) {
  return wrap(UnknownLoc::get(unwrap(context)));
}

bool mlirLocationEqual(MlirLocation l1, MlirLocation l2) {
  return unwrap(l1) == unwrap(l2);
}

MlirContext mlirLocationGetContext(MlirLocation location) {
  return wrap(unwrap(location).getContext());
}

void mlirLocationPrint(MlirLocation location, MlirStringCallback callback,
                       void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(location).print(stream);
}

//===----------------------------------------------------------------------===//
// Module API.
//===----------------------------------------------------------------------===//

MlirModule mlirModuleCreateEmpty(MlirLocation location) {
  return wrap(ModuleOp::create(unwrap(location)));
}

MlirModule mlirModuleCreateParse(MlirContext context, MlirStringRef module) {
  OwningModuleRef owning = parseSourceString(unwrap(module), unwrap(context));
  if (!owning)
    return MlirModule{nullptr};
  return MlirModule{owning.release().getOperation()};
}

MlirContext mlirModuleGetContext(MlirModule module) {
  return wrap(unwrap(module).getContext());
}

MlirBlock mlirModuleGetBody(MlirModule module) {
  return wrap(unwrap(module).getBody());
}

void mlirModuleDestroy(MlirModule module) {
  // Transfer ownership to an OwningModuleRef so that its destructor is called.
  OwningModuleRef(unwrap(module));
}

MlirOperation mlirModuleGetOperation(MlirModule module) {
  return wrap(unwrap(module).getOperation());
}

//===----------------------------------------------------------------------===//
// Operation state API.
//===----------------------------------------------------------------------===//

MlirOperationState mlirOperationStateGet(MlirStringRef name, MlirLocation loc) {
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
                                  MlirType const *results) {
  APPEND_ELEMS(MlirType, nResults, results);
}

void mlirOperationStateAddOperands(MlirOperationState *state, intptr_t n,
                                   MlirValue const *operands) {
  APPEND_ELEMS(MlirValue, nOperands, operands);
}
void mlirOperationStateAddOwnedRegions(MlirOperationState *state, intptr_t n,
                                       MlirRegion const *regions) {
  APPEND_ELEMS(MlirRegion, nRegions, regions);
}
void mlirOperationStateAddSuccessors(MlirOperationState *state, intptr_t n,
                                     MlirBlock const *successors) {
  APPEND_ELEMS(MlirBlock, nSuccessors, successors);
}
void mlirOperationStateAddAttributes(MlirOperationState *state, intptr_t n,
                                     MlirNamedAttribute const *attributes) {
  APPEND_ELEMS(MlirNamedAttribute, nAttributes, attributes);
}

//===----------------------------------------------------------------------===//
// Operation API.
//===----------------------------------------------------------------------===//

MlirOperation mlirOperationCreate(const MlirOperationState *state) {
  assert(state);
  OperationState cppState(unwrap(state->location), unwrap(state->name));
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
    cppState.addAttribute(unwrap(state->attributes[i].name),
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

bool mlirOperationEqual(MlirOperation op, MlirOperation other) {
  return unwrap(op) == unwrap(other);
}

MlirIdentifier mlirOperationGetName(MlirOperation op) {
  return wrap(unwrap(op)->getName().getIdentifier());
}

MlirBlock mlirOperationGetBlock(MlirOperation op) {
  return wrap(unwrap(op)->getBlock());
}

MlirOperation mlirOperationGetParentOperation(MlirOperation op) {
  return wrap(unwrap(op)->getParentOp());
}

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
  return MlirNamedAttribute{wrap(attr.first), wrap(attr.second)};
}

MlirAttribute mlirOperationGetAttributeByName(MlirOperation op,
                                              MlirStringRef name) {
  return wrap(unwrap(op)->getAttr(unwrap(name)));
}

void mlirOperationSetAttributeByName(MlirOperation op, MlirStringRef name,
                                     MlirAttribute attr) {
  unwrap(op)->setAttr(unwrap(name), unwrap(attr));
}

bool mlirOperationRemoveAttributeByName(MlirOperation op, MlirStringRef name) {
  return !!unwrap(op)->removeAttr(unwrap(name));
}

void mlirOperationPrint(MlirOperation op, MlirStringCallback callback,
                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(op)->print(stream);
}

void mlirOperationPrintWithFlags(MlirOperation op, MlirOpPrintingFlags flags,
                                 MlirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(op)->print(stream, *unwrap(flags));
}

void mlirOperationDump(MlirOperation op) { return unwrap(op)->dump(); }

bool mlirOperationVerify(MlirOperation op) {
  return succeeded(verify(unwrap(op)));
}

//===----------------------------------------------------------------------===//
// Region API.
//===----------------------------------------------------------------------===//

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

void mlirRegionInsertOwnedBlockAfter(MlirRegion region, MlirBlock reference,
                                     MlirBlock block) {
  Region *cppRegion = unwrap(region);
  if (mlirBlockIsNull(reference)) {
    cppRegion->getBlocks().insert(cppRegion->begin(), unwrap(block));
    return;
  }

  assert(unwrap(reference)->getParent() == unwrap(region) &&
         "expected reference block to belong to the region");
  cppRegion->getBlocks().insertAfter(Region::iterator(unwrap(reference)),
                                     unwrap(block));
}

void mlirRegionInsertOwnedBlockBefore(MlirRegion region, MlirBlock reference,
                                      MlirBlock block) {
  if (mlirBlockIsNull(reference))
    return mlirRegionAppendOwnedBlock(region, block);

  assert(unwrap(reference)->getParent() == unwrap(region) &&
         "expected reference block to belong to the region");
  unwrap(region)->getBlocks().insert(Region::iterator(unwrap(reference)),
                                     unwrap(block));
}

void mlirRegionDestroy(MlirRegion region) {
  delete static_cast<Region *>(region.ptr);
}

//===----------------------------------------------------------------------===//
// Block API.
//===----------------------------------------------------------------------===//

MlirBlock mlirBlockCreate(intptr_t nArgs, MlirType const *args) {
  Block *b = new Block;
  for (intptr_t i = 0; i < nArgs; ++i)
    b->addArgument(unwrap(args[i]));
  return wrap(b);
}

bool mlirBlockEqual(MlirBlock block, MlirBlock other) {
  return unwrap(block) == unwrap(other);
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

MlirOperation mlirBlockGetTerminator(MlirBlock block) {
  Block *cppBlock = unwrap(block);
  if (cppBlock->empty())
    return wrap(static_cast<Operation *>(nullptr));
  Operation &back = cppBlock->back();
  if (!back.isKnownTerminator())
    return wrap(static_cast<Operation *>(nullptr));
  return wrap(&back);
}

void mlirBlockAppendOwnedOperation(MlirBlock block, MlirOperation operation) {
  unwrap(block)->push_back(unwrap(operation));
}

void mlirBlockInsertOwnedOperation(MlirBlock block, intptr_t pos,
                                   MlirOperation operation) {
  auto &opList = unwrap(block)->getOperations();
  opList.insert(std::next(opList.begin(), pos), unwrap(operation));
}

void mlirBlockInsertOwnedOperationAfter(MlirBlock block,
                                        MlirOperation reference,
                                        MlirOperation operation) {
  Block *cppBlock = unwrap(block);
  if (mlirOperationIsNull(reference)) {
    cppBlock->getOperations().insert(cppBlock->begin(), unwrap(operation));
    return;
  }

  assert(unwrap(reference)->getBlock() == unwrap(block) &&
         "expected reference operation to belong to the block");
  cppBlock->getOperations().insertAfter(Block::iterator(unwrap(reference)),
                                        unwrap(operation));
}

void mlirBlockInsertOwnedOperationBefore(MlirBlock block,
                                         MlirOperation reference,
                                         MlirOperation operation) {
  if (mlirOperationIsNull(reference))
    return mlirBlockAppendOwnedOperation(block, operation);

  assert(unwrap(reference)->getBlock() == unwrap(block) &&
         "expected reference operation to belong to the block");
  unwrap(block)->getOperations().insert(Block::iterator(unwrap(reference)),
                                        unwrap(operation));
}

void mlirBlockDestroy(MlirBlock block) { delete unwrap(block); }

intptr_t mlirBlockGetNumArguments(MlirBlock block) {
  return static_cast<intptr_t>(unwrap(block)->getNumArguments());
}

MlirValue mlirBlockGetArgument(MlirBlock block, intptr_t pos) {
  return wrap(unwrap(block)->getArgument(static_cast<unsigned>(pos)));
}

void mlirBlockPrint(MlirBlock block, MlirStringCallback callback,
                    void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(block)->print(stream);
}

//===----------------------------------------------------------------------===//
// Value API.
//===----------------------------------------------------------------------===//

bool mlirValueEqual(MlirValue value1, MlirValue value2) {
  return unwrap(value1) == unwrap(value2);
}

bool mlirValueIsABlockArgument(MlirValue value) {
  return unwrap(value).isa<BlockArgument>();
}

bool mlirValueIsAOpResult(MlirValue value) {
  return unwrap(value).isa<OpResult>();
}

MlirBlock mlirBlockArgumentGetOwner(MlirValue value) {
  return wrap(unwrap(value).cast<BlockArgument>().getOwner());
}

intptr_t mlirBlockArgumentGetArgNumber(MlirValue value) {
  return static_cast<intptr_t>(
      unwrap(value).cast<BlockArgument>().getArgNumber());
}

void mlirBlockArgumentSetType(MlirValue value, MlirType type) {
  unwrap(value).cast<BlockArgument>().setType(unwrap(type));
}

MlirOperation mlirOpResultGetOwner(MlirValue value) {
  return wrap(unwrap(value).cast<OpResult>().getOwner());
}

intptr_t mlirOpResultGetResultNumber(MlirValue value) {
  return static_cast<intptr_t>(
      unwrap(value).cast<OpResult>().getResultNumber());
}

MlirType mlirValueGetType(MlirValue value) {
  return wrap(unwrap(value).getType());
}

void mlirValueDump(MlirValue value) { unwrap(value).dump(); }

void mlirValuePrint(MlirValue value, MlirStringCallback callback,
                    void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(value).print(stream);
}

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

MlirType mlirTypeParseGet(MlirContext context, MlirStringRef type) {
  return wrap(mlir::parseType(unwrap(type), unwrap(context)));
}

MlirContext mlirTypeGetContext(MlirType type) {
  return wrap(unwrap(type).getContext());
}

bool mlirTypeEqual(MlirType t1, MlirType t2) {
  return unwrap(t1) == unwrap(t2);
}

void mlirTypePrint(MlirType type, MlirStringCallback callback, void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(type).print(stream);
}

void mlirTypeDump(MlirType type) { unwrap(type).dump(); }

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MlirAttribute mlirAttributeParseGet(MlirContext context, MlirStringRef attr) {
  return wrap(mlir::parseAttribute(unwrap(attr), unwrap(context)));
}

MlirContext mlirAttributeGetContext(MlirAttribute attribute) {
  return wrap(unwrap(attribute).getContext());
}

MlirType mlirAttributeGetType(MlirAttribute attribute) {
  return wrap(unwrap(attribute).getType());
}

bool mlirAttributeEqual(MlirAttribute a1, MlirAttribute a2) {
  return unwrap(a1) == unwrap(a2);
}

void mlirAttributePrint(MlirAttribute attr, MlirStringCallback callback,
                        void *userData) {
  detail::CallbackOstream stream(callback, userData);
  unwrap(attr).print(stream);
}

void mlirAttributeDump(MlirAttribute attr) { unwrap(attr).dump(); }

MlirNamedAttribute mlirNamedAttributeGet(MlirIdentifier name,
                                         MlirAttribute attr) {
  return MlirNamedAttribute{name, attr};
}

//===----------------------------------------------------------------------===//
// Identifier API.
//===----------------------------------------------------------------------===//

MlirIdentifier mlirIdentifierGet(MlirContext context, MlirStringRef str) {
  return wrap(Identifier::get(unwrap(str), unwrap(context)));
}

bool mlirIdentifierEqual(MlirIdentifier ident, MlirIdentifier other) {
  return unwrap(ident) == unwrap(other);
}

MlirStringRef mlirIdentifierStr(MlirIdentifier ident) {
  return wrap(unwrap(ident).strref());
}
