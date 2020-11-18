//===- ir.c - Simple test of C APIs ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* RUN: mlir-capi-ir-test 2>&1 | FileCheck %s
 */

#include "mlir-c/IR.h"
#include "mlir-c/AffineExpr.h"
#include "mlir-c/AffineMap.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/Registration.h"
#include "mlir-c/StandardAttributes.h"
#include "mlir-c/StandardDialect.h"
#include "mlir-c/StandardTypes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void populateLoopBody(MlirContext ctx, MlirBlock loopBody,
                      MlirLocation location, MlirBlock funcBody) {
  MlirValue iv = mlirBlockGetArgument(loopBody, 0);
  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue funcArg1 = mlirBlockGetArgument(funcBody, 1);
  MlirType f32Type = mlirTypeParseGet(ctx, "f32");

  MlirOperationState loadLHSState = mlirOperationStateGet("std.load", location);
  MlirValue loadLHSOperands[] = {funcArg0, iv};
  mlirOperationStateAddOperands(&loadLHSState, 2, loadLHSOperands);
  mlirOperationStateAddResults(&loadLHSState, 1, &f32Type);
  MlirOperation loadLHS = mlirOperationCreate(&loadLHSState);
  mlirBlockAppendOwnedOperation(loopBody, loadLHS);

  MlirOperationState loadRHSState = mlirOperationStateGet("std.load", location);
  MlirValue loadRHSOperands[] = {funcArg1, iv};
  mlirOperationStateAddOperands(&loadRHSState, 2, loadRHSOperands);
  mlirOperationStateAddResults(&loadRHSState, 1, &f32Type);
  MlirOperation loadRHS = mlirOperationCreate(&loadRHSState);
  mlirBlockAppendOwnedOperation(loopBody, loadRHS);

  MlirOperationState addState = mlirOperationStateGet("std.addf", location);
  MlirValue addOperands[] = {mlirOperationGetResult(loadLHS, 0),
                             mlirOperationGetResult(loadRHS, 0)};
  mlirOperationStateAddOperands(&addState, 2, addOperands);
  mlirOperationStateAddResults(&addState, 1, &f32Type);
  MlirOperation add = mlirOperationCreate(&addState);
  mlirBlockAppendOwnedOperation(loopBody, add);

  MlirOperationState storeState = mlirOperationStateGet("std.store", location);
  MlirValue storeOperands[] = {mlirOperationGetResult(add, 0), funcArg0, iv};
  mlirOperationStateAddOperands(&storeState, 3, storeOperands);
  MlirOperation store = mlirOperationCreate(&storeState);
  mlirBlockAppendOwnedOperation(loopBody, store);

  MlirOperationState yieldState = mlirOperationStateGet("scf.yield", location);
  MlirOperation yield = mlirOperationCreate(&yieldState);
  mlirBlockAppendOwnedOperation(loopBody, yield);
}

MlirModule makeAndDumpAdd(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirBlock moduleBody = mlirModuleGetBody(moduleOp);

  MlirType memrefType = mlirTypeParseGet(ctx, "memref<?xf32>");
  MlirType funcBodyArgTypes[] = {memrefType, memrefType};
  MlirRegion funcBodyRegion = mlirRegionCreate();
  MlirBlock funcBody = mlirBlockCreate(
      sizeof(funcBodyArgTypes) / sizeof(MlirType), funcBodyArgTypes);
  mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);

  MlirAttribute funcTypeAttr =
      mlirAttributeParseGet(ctx, "(memref<?xf32>, memref<?xf32>) -> ()");
  MlirAttribute funcNameAttr = mlirAttributeParseGet(ctx, "\"add\"");
  MlirNamedAttribute funcAttrs[] = {
      mlirNamedAttributeGet("type", funcTypeAttr),
      mlirNamedAttributeGet("sym_name", funcNameAttr)};
  MlirOperationState funcState = mlirOperationStateGet("func", location);
  mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);
  mlirOperationStateAddOwnedRegions(&funcState, 1, &funcBodyRegion);
  MlirOperation func = mlirOperationCreate(&funcState);
  mlirBlockInsertOwnedOperation(moduleBody, 0, func);

  MlirType indexType = mlirTypeParseGet(ctx, "index");
  MlirAttribute indexZeroLiteral = mlirAttributeParseGet(ctx, "0 : index");
  MlirNamedAttribute indexZeroValueAttr =
      mlirNamedAttributeGet("value", indexZeroLiteral);
  MlirOperationState constZeroState =
      mlirOperationStateGet("std.constant", location);
  mlirOperationStateAddResults(&constZeroState, 1, &indexType);
  mlirOperationStateAddAttributes(&constZeroState, 1, &indexZeroValueAttr);
  MlirOperation constZero = mlirOperationCreate(&constZeroState);
  mlirBlockAppendOwnedOperation(funcBody, constZero);

  MlirValue funcArg0 = mlirBlockGetArgument(funcBody, 0);
  MlirValue constZeroValue = mlirOperationGetResult(constZero, 0);
  MlirValue dimOperands[] = {funcArg0, constZeroValue};
  MlirOperationState dimState = mlirOperationStateGet("std.dim", location);
  mlirOperationStateAddOperands(&dimState, 2, dimOperands);
  mlirOperationStateAddResults(&dimState, 1, &indexType);
  MlirOperation dim = mlirOperationCreate(&dimState);
  mlirBlockAppendOwnedOperation(funcBody, dim);

  MlirRegion loopBodyRegion = mlirRegionCreate();
  MlirBlock loopBody = mlirBlockCreate(/*nArgs=*/1, &indexType);
  mlirRegionAppendOwnedBlock(loopBodyRegion, loopBody);

  MlirAttribute indexOneLiteral = mlirAttributeParseGet(ctx, "1 : index");
  MlirNamedAttribute indexOneValueAttr =
      mlirNamedAttributeGet("value", indexOneLiteral);
  MlirOperationState constOneState =
      mlirOperationStateGet("std.constant", location);
  mlirOperationStateAddResults(&constOneState, 1, &indexType);
  mlirOperationStateAddAttributes(&constOneState, 1, &indexOneValueAttr);
  MlirOperation constOne = mlirOperationCreate(&constOneState);
  mlirBlockAppendOwnedOperation(funcBody, constOne);

  MlirValue dimValue = mlirOperationGetResult(dim, 0);
  MlirValue constOneValue = mlirOperationGetResult(constOne, 0);
  MlirValue loopOperands[] = {constZeroValue, dimValue, constOneValue};
  MlirOperationState loopState = mlirOperationStateGet("scf.for", location);
  mlirOperationStateAddOperands(&loopState, 3, loopOperands);
  mlirOperationStateAddOwnedRegions(&loopState, 1, &loopBodyRegion);
  MlirOperation loop = mlirOperationCreate(&loopState);
  mlirBlockAppendOwnedOperation(funcBody, loop);

  populateLoopBody(ctx, loopBody, location, funcBody);

  MlirOperationState retState = mlirOperationStateGet("std.return", location);
  MlirOperation ret = mlirOperationCreate(&retState);
  mlirBlockAppendOwnedOperation(funcBody, ret);

  MlirOperation module = mlirModuleGetOperation(moduleOp);
  mlirOperationDump(module);
  // clang-format off
  // CHECK: module {
  // CHECK:   func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
  // CHECK:     %[[C0:.*]] = constant 0 : index
  // CHECK:     %[[DIM:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
  // CHECK:     %[[C1:.*]] = constant 1 : index
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
  // CHECK:       %[[LHS:.*]] = load %[[ARG0]][%[[I]]] : memref<?xf32>
  // CHECK:       %[[RHS:.*]] = load %[[ARG1]][%[[I]]] : memref<?xf32>
  // CHECK:       %[[SUM:.*]] = addf %[[LHS]], %[[RHS]] : f32
  // CHECK:       store %[[SUM]], %[[ARG0]][%[[I]]] : memref<?xf32>
  // CHECK:     }
  // CHECK:     return
  // CHECK:   }
  // CHECK: }
  // clang-format on

  return moduleOp;
}

struct OpListNode {
  MlirOperation op;
  struct OpListNode *next;
};
typedef struct OpListNode OpListNode;

struct ModuleStats {
  unsigned numOperations;
  unsigned numAttributes;
  unsigned numBlocks;
  unsigned numRegions;
  unsigned numValues;
  unsigned numBlockArguments;
  unsigned numOpResults;
};
typedef struct ModuleStats ModuleStats;

int collectStatsSingle(OpListNode *head, ModuleStats *stats) {
  MlirOperation operation = head->op;
  stats->numOperations += 1;
  stats->numValues += mlirOperationGetNumResults(operation);
  stats->numAttributes += mlirOperationGetNumAttributes(operation);

  unsigned numRegions = mlirOperationGetNumRegions(operation);

  stats->numRegions += numRegions;

  intptr_t numResults = mlirOperationGetNumResults(operation);
  for (intptr_t i = 0; i < numResults; ++i) {
    MlirValue result = mlirOperationGetResult(operation, i);
    if (!mlirValueIsAOpResult(result))
      return 1;
    if (mlirValueIsABlockArgument(result))
      return 2;
    if (!mlirOperationEqual(operation, mlirOpResultGetOwner(result)))
      return 3;
    if (i != mlirOpResultGetResultNumber(result))
      return 4;
    ++stats->numOpResults;
  }

  for (unsigned i = 0; i < numRegions; ++i) {
    MlirRegion region = mlirOperationGetRegion(operation, i);
    for (MlirBlock block = mlirRegionGetFirstBlock(region);
         !mlirBlockIsNull(block); block = mlirBlockGetNextInRegion(block)) {
      ++stats->numBlocks;
      intptr_t numArgs = mlirBlockGetNumArguments(block);
      stats->numValues += numArgs;
      for (intptr_t j = 0; j < numArgs; ++j) {
        MlirValue arg = mlirBlockGetArgument(block, j);
        if (!mlirValueIsABlockArgument(arg))
          return 5;
        if (mlirValueIsAOpResult(arg))
          return 6;
        if (!mlirBlockEqual(block, mlirBlockArgumentGetOwner(arg)))
          return 7;
        if (j != mlirBlockArgumentGetArgNumber(arg))
          return 8;
        ++stats->numBlockArguments;
      }

      for (MlirOperation child = mlirBlockGetFirstOperation(block);
           !mlirOperationIsNull(child);
           child = mlirOperationGetNextInBlock(child)) {
        OpListNode *node = malloc(sizeof(OpListNode));
        node->op = child;
        node->next = head->next;
        head->next = node;
      }
    }
  }
  return 0;
}

int collectStats(MlirOperation operation) {
  OpListNode *head = malloc(sizeof(OpListNode));
  head->op = operation;
  head->next = NULL;

  ModuleStats stats;
  stats.numOperations = 0;
  stats.numAttributes = 0;
  stats.numBlocks = 0;
  stats.numRegions = 0;
  stats.numValues = 0;
  stats.numBlockArguments = 0;
  stats.numOpResults = 0;

  do {
    int retval = collectStatsSingle(head, &stats);
    if (retval)
      return retval;
    OpListNode *next = head->next;
    free(head);
    head = next;
  } while (head);

  if (stats.numValues != stats.numBlockArguments + stats.numOpResults)
    return 100;

  fprintf(stderr, "@stats\n");
  fprintf(stderr, "Number of operations: %u\n", stats.numOperations);
  fprintf(stderr, "Number of attributes: %u\n", stats.numAttributes);
  fprintf(stderr, "Number of blocks: %u\n", stats.numBlocks);
  fprintf(stderr, "Number of regions: %u\n", stats.numRegions);
  fprintf(stderr, "Number of values: %u\n", stats.numValues);
  fprintf(stderr, "Number of block arguments: %u\n", stats.numBlockArguments);
  fprintf(stderr, "Number of op results: %u\n", stats.numOpResults);
  // clang-format off
  // CHECK-LABEL: @stats
  // CHECK: Number of operations: 13
  // CHECK: Number of attributes: 4
  // CHECK: Number of blocks: 3
  // CHECK: Number of regions: 3
  // CHECK: Number of values: 9
  // CHECK: Number of block arguments: 3
  // CHECK: Number of op results: 6
  // clang-format on
  return 0;
}

static void printToStderr(const char *str, intptr_t len, void *userData) {
  (void)userData;
  fwrite(str, 1, len, stderr);
}

static void printFirstOfEach(MlirContext ctx, MlirOperation operation) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  MlirRegion region = mlirOperationGetRegion(operation, 0);
  MlirBlock block = mlirRegionGetFirstBlock(region);
  operation = mlirBlockGetFirstOperation(block);
  region = mlirOperationGetRegion(operation, 0);
  MlirOperation parentOperation = operation;
  block = mlirRegionGetFirstBlock(region);
  operation = mlirBlockGetFirstOperation(block);

  // Verify that parent operation and block report correctly.
  fprintf(stderr, "Parent operation eq: %d\n",
          mlirOperationEqual(mlirOperationGetParentOperation(operation),
                             parentOperation));
  fprintf(stderr, "Block eq: %d\n",
          mlirBlockEqual(mlirOperationGetBlock(operation), block));
  // CHECK: Parent operation eq: 1
  // CHECK: Block eq: 1

  // In the module we created, the first operation of the first function is
  // an "std.dim", which has an attribute and a single result that we can
  // use to test the printing mechanism.
  mlirBlockPrint(block, printToStderr, NULL);
  fprintf(stderr, "\n");
  fprintf(stderr, "First operation: ");
  mlirOperationPrint(operation, printToStderr, NULL);
  fprintf(stderr, "\n");
  // clang-format off
  // CHECK:   %[[C0:.*]] = constant 0 : index
  // CHECK:   %[[DIM:.*]] = dim %{{.*}}, %[[C0]] : memref<?xf32>
  // CHECK:   %[[C1:.*]] = constant 1 : index
  // CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
  // CHECK:     %[[LHS:.*]] = load %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:     %[[RHS:.*]] = load %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:     %[[SUM:.*]] = addf %[[LHS]], %[[RHS]] : f32
  // CHECK:     store %[[SUM]], %{{.*}}[%[[I]]] : memref<?xf32>
  // CHECK:   }
  // CHECK: return
  // CHECK: First operation: {{.*}} = constant 0 : index
  // clang-format on

  // Get the operation name and print it.
  MlirIdentifier ident = mlirOperationGetName(operation);
  MlirStringRef identStr = mlirIdentifierStr(ident);
  fprintf(stderr, "Operation name: '");
  for (size_t i = 0; i < identStr.length; ++i)
    fputc(identStr.data[i], stderr);
  fprintf(stderr, "'\n");
  // CHECK: Operation name: 'std.constant'

  // Get the identifier again and verify equal.
  MlirIdentifier identAgain = mlirIdentifierGet(ctx, identStr);
  fprintf(stderr, "Identifier equal: %d\n",
          mlirIdentifierEqual(ident, identAgain));
  // CHECK: Identifier equal: 1

  // Get the block terminator and print it.
  MlirOperation terminator = mlirBlockGetTerminator(block);
  fprintf(stderr, "Terminator: ");
  mlirOperationPrint(terminator, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Terminator: return

  // Get the attribute by index.
  MlirNamedAttribute namedAttr0 = mlirOperationGetAttribute(operation, 0);
  fprintf(stderr, "Get attr 0: ");
  mlirAttributePrint(namedAttr0.attribute, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Get attr 0: 0 : index

  // Now re-get the attribute by name.
  MlirAttribute attr0ByName =
      mlirOperationGetAttributeByName(operation, namedAttr0.name);
  fprintf(stderr, "Get attr 0 by name: ");
  mlirAttributePrint(attr0ByName, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Get attr 0 by name: 0 : index

  // Get a non-existing attribute and assert that it is null (sanity).
  fprintf(stderr, "does_not_exist is null: %d\n",
          mlirAttributeIsNull(
              mlirOperationGetAttributeByName(operation, "does_not_exist")));
  // CHECK: does_not_exist is null: 1

  // Get result 0 and its type.
  MlirValue value = mlirOperationGetResult(operation, 0);
  fprintf(stderr, "Result 0: ");
  mlirValuePrint(value, printToStderr, NULL);
  fprintf(stderr, "\n");
  fprintf(stderr, "Value is null: %d\n", mlirValueIsNull(value));
  // CHECK: Result 0: {{.*}} = constant 0 : index
  // CHECK: Value is null: 0

  MlirType type = mlirValueGetType(value);
  fprintf(stderr, "Result 0 type: ");
  mlirTypePrint(type, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Result 0 type: index

  // Set a custom attribute.
  mlirOperationSetAttributeByName(operation, "custom_attr",
                                  mlirBoolAttrGet(ctx, 1));
  fprintf(stderr, "Op with set attr: ");
  mlirOperationPrint(operation, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Op with set attr: {{.*}} {custom_attr = true}

  // Remove the attribute.
  fprintf(stderr, "Remove attr: %d\n",
          mlirOperationRemoveAttributeByName(operation, "custom_attr"));
  fprintf(stderr, "Remove attr again: %d\n",
          mlirOperationRemoveAttributeByName(operation, "custom_attr"));
  fprintf(stderr, "Removed attr is null: %d\n",
          mlirAttributeIsNull(
              mlirOperationGetAttributeByName(operation, "custom_attr")));
  // CHECK: Remove attr: 1
  // CHECK: Remove attr again: 0
  // CHECK: Removed attr is null: 1

  // Add a large attribute to verify printing flags.
  int64_t eltsShape[] = {4};
  int32_t eltsData[] = {1, 2, 3, 4};
  mlirOperationSetAttributeByName(
      operation, "elts",
      mlirDenseElementsAttrInt32Get(
          mlirRankedTensorTypeGet(1, eltsShape, mlirIntegerTypeGet(ctx, 32)), 4,
          eltsData));
  MlirOpPrintingFlags flags = mlirOpPrintingFlagsCreate();
  mlirOpPrintingFlagsElideLargeElementsAttrs(flags, 2);
  mlirOpPrintingFlagsPrintGenericOpForm(flags);
  mlirOpPrintingFlagsEnableDebugInfo(flags, /*prettyForm=*/0);
  mlirOpPrintingFlagsUseLocalScope(flags);
  fprintf(stderr, "Op print with all flags: ");
  mlirOperationPrintWithFlags(operation, flags, printToStderr, NULL);
  fprintf(stderr, "\n");
  // CHECK: Op print with all flags: %{{.*}} = "std.constant"() {elts = opaque<"", "0xDEADBEEF"> : tensor<4xi32>, value = 0 : index} : () -> index loc(unknown)

  mlirOpPrintingFlagsDestroy(flags);
}

static int constructAndTraverseIr(MlirContext ctx) {
  MlirLocation location = mlirLocationUnknownGet(ctx);

  MlirModule moduleOp = makeAndDumpAdd(ctx, location);
  MlirOperation module = mlirModuleGetOperation(moduleOp);

  int errcode = collectStats(module);
  if (errcode)
    return errcode;

  printFirstOfEach(ctx, module);

  mlirModuleDestroy(moduleOp);
  return 0;
}

/// Creates an operation with a region containing multiple blocks with
/// operations and dumps it. The blocks and operations are inserted using
/// block/operation-relative API and their final order is checked.
static void buildWithInsertionsAndPrint(MlirContext ctx) {
  MlirLocation loc = mlirLocationUnknownGet(ctx);

  MlirRegion owningRegion = mlirRegionCreate();
  MlirBlock nullBlock = mlirRegionGetFirstBlock(owningRegion);
  MlirOperationState state = mlirOperationStateGet("insertion.order.test", loc);
  mlirOperationStateAddOwnedRegions(&state, 1, &owningRegion);
  MlirOperation op = mlirOperationCreate(&state);
  MlirRegion region = mlirOperationGetRegion(op, 0);

  // Use integer types of different bitwidth as block arguments in order to
  // differentiate blocks.
  MlirType i1 = mlirIntegerTypeGet(ctx, 1);
  MlirType i2 = mlirIntegerTypeGet(ctx, 2);
  MlirType i3 = mlirIntegerTypeGet(ctx, 3);
  MlirType i4 = mlirIntegerTypeGet(ctx, 4);
  MlirBlock block1 = mlirBlockCreate(1, &i1);
  MlirBlock block2 = mlirBlockCreate(1, &i2);
  MlirBlock block3 = mlirBlockCreate(1, &i3);
  MlirBlock block4 = mlirBlockCreate(1, &i4);
  // Insert blocks so as to obtain the 1-2-3-4 order,
  mlirRegionInsertOwnedBlockBefore(region, nullBlock, block3);
  mlirRegionInsertOwnedBlockBefore(region, block3, block2);
  mlirRegionInsertOwnedBlockAfter(region, nullBlock, block1);
  mlirRegionInsertOwnedBlockAfter(region, block3, block4);

  MlirOperationState op1State = mlirOperationStateGet("dummy.op1", loc);
  MlirOperationState op2State = mlirOperationStateGet("dummy.op2", loc);
  MlirOperationState op3State = mlirOperationStateGet("dummy.op3", loc);
  MlirOperationState op4State = mlirOperationStateGet("dummy.op4", loc);
  MlirOperationState op5State = mlirOperationStateGet("dummy.op5", loc);
  MlirOperationState op6State = mlirOperationStateGet("dummy.op6", loc);
  MlirOperationState op7State = mlirOperationStateGet("dummy.op7", loc);
  MlirOperation op1 = mlirOperationCreate(&op1State);
  MlirOperation op2 = mlirOperationCreate(&op2State);
  MlirOperation op3 = mlirOperationCreate(&op3State);
  MlirOperation op4 = mlirOperationCreate(&op4State);
  MlirOperation op5 = mlirOperationCreate(&op5State);
  MlirOperation op6 = mlirOperationCreate(&op6State);
  MlirOperation op7 = mlirOperationCreate(&op7State);

  // Insert operations in the first block so as to obtain the 1-2-3-4 order.
  MlirOperation nullOperation = mlirBlockGetFirstOperation(block1);
  assert(mlirOperationIsNull(nullOperation));
  mlirBlockInsertOwnedOperationBefore(block1, nullOperation, op3);
  mlirBlockInsertOwnedOperationBefore(block1, op3, op2);
  mlirBlockInsertOwnedOperationAfter(block1, nullOperation, op1);
  mlirBlockInsertOwnedOperationAfter(block1, op3, op4);

  // Append operations to the rest of blocks to make them non-empty and thus
  // printable.
  mlirBlockAppendOwnedOperation(block2, op5);
  mlirBlockAppendOwnedOperation(block3, op6);
  mlirBlockAppendOwnedOperation(block4, op7);

  mlirOperationDump(op);
  mlirOperationDestroy(op);
  // clang-format off
  // CHECK-LABEL:  "insertion.order.test"
  // CHECK:      ^{{.*}}(%{{.*}}: i1
  // CHECK:        "dummy.op1"
  // CHECK-NEXT:   "dummy.op2"
  // CHECK-NEXT:   "dummy.op3"
  // CHECK-NEXT:   "dummy.op4"
  // CHECK:      ^{{.*}}(%{{.*}}: i2
  // CHECK:        "dummy.op5"
  // CHECK:      ^{{.*}}(%{{.*}}: i3
  // CHECK:        "dummy.op6"
  // CHECK:      ^{{.*}}(%{{.*}}: i4
  // CHECK:        "dummy.op7"
  // clang-format on
}

/// Dumps instances of all standard types to check that C API works correctly.
/// Additionally, performs simple identity checks that a standard type
/// constructed with C API can be inspected and has the expected type. The
/// latter achieves full coverage of C API for standard types. Returns 0 on
/// success and a non-zero error code on failure.
static int printStandardTypes(MlirContext ctx) {
  // Integer types.
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType si32 = mlirIntegerTypeSignedGet(ctx, 32);
  MlirType ui32 = mlirIntegerTypeUnsignedGet(ctx, 32);
  if (!mlirTypeIsAInteger(i32) || mlirTypeIsAF32(i32))
    return 1;
  if (!mlirTypeIsAInteger(si32) || !mlirIntegerTypeIsSigned(si32))
    return 2;
  if (!mlirTypeIsAInteger(ui32) || !mlirIntegerTypeIsUnsigned(ui32))
    return 3;
  if (mlirTypeEqual(i32, ui32) || mlirTypeEqual(i32, si32))
    return 4;
  if (mlirIntegerTypeGetWidth(i32) != mlirIntegerTypeGetWidth(si32))
    return 5;
  fprintf(stderr, "@types\n");
  mlirTypeDump(i32);
  fprintf(stderr, "\n");
  mlirTypeDump(si32);
  fprintf(stderr, "\n");
  mlirTypeDump(ui32);
  fprintf(stderr, "\n");
  // CHECK-LABEL: @types
  // CHECK: i32
  // CHECK: si32
  // CHECK: ui32

  // Index type.
  MlirType index = mlirIndexTypeGet(ctx);
  if (!mlirTypeIsAIndex(index))
    return 6;
  mlirTypeDump(index);
  fprintf(stderr, "\n");
  // CHECK: index

  // Floating-point types.
  MlirType bf16 = mlirBF16TypeGet(ctx);
  MlirType f16 = mlirF16TypeGet(ctx);
  MlirType f32 = mlirF32TypeGet(ctx);
  MlirType f64 = mlirF64TypeGet(ctx);
  if (!mlirTypeIsABF16(bf16))
    return 7;
  if (!mlirTypeIsAF16(f16))
    return 9;
  if (!mlirTypeIsAF32(f32))
    return 10;
  if (!mlirTypeIsAF64(f64))
    return 11;
  mlirTypeDump(bf16);
  fprintf(stderr, "\n");
  mlirTypeDump(f16);
  fprintf(stderr, "\n");
  mlirTypeDump(f32);
  fprintf(stderr, "\n");
  mlirTypeDump(f64);
  fprintf(stderr, "\n");
  // CHECK: bf16
  // CHECK: f16
  // CHECK: f32
  // CHECK: f64

  // None type.
  MlirType none = mlirNoneTypeGet(ctx);
  if (!mlirTypeIsANone(none))
    return 12;
  mlirTypeDump(none);
  fprintf(stderr, "\n");
  // CHECK: none

  // Complex type.
  MlirType cplx = mlirComplexTypeGet(f32);
  if (!mlirTypeIsAComplex(cplx) ||
      !mlirTypeEqual(mlirComplexTypeGetElementType(cplx), f32))
    return 13;
  mlirTypeDump(cplx);
  fprintf(stderr, "\n");
  // CHECK: complex<f32>

  // Vector (and Shaped) type. ShapedType is a common base class for vectors,
  // memrefs and tensors, one cannot create instances of this class so it is
  // tested on an instance of vector type.
  int64_t shape[] = {2, 3};
  MlirType vector =
      mlirVectorTypeGet(sizeof(shape) / sizeof(int64_t), shape, f32);
  if (!mlirTypeIsAVector(vector) || !mlirTypeIsAShaped(vector))
    return 14;
  if (!mlirTypeEqual(mlirShapedTypeGetElementType(vector), f32) ||
      !mlirShapedTypeHasRank(vector) || mlirShapedTypeGetRank(vector) != 2 ||
      mlirShapedTypeGetDimSize(vector, 0) != 2 ||
      mlirShapedTypeIsDynamicDim(vector, 0) ||
      mlirShapedTypeGetDimSize(vector, 1) != 3 ||
      !mlirShapedTypeHasStaticShape(vector))
    return 15;
  mlirTypeDump(vector);
  fprintf(stderr, "\n");
  // CHECK: vector<2x3xf32>

  // Ranked tensor type.
  MlirType rankedTensor =
      mlirRankedTensorTypeGet(sizeof(shape) / sizeof(int64_t), shape, f32);
  if (!mlirTypeIsATensor(rankedTensor) ||
      !mlirTypeIsARankedTensor(rankedTensor))
    return 16;
  mlirTypeDump(rankedTensor);
  fprintf(stderr, "\n");
  // CHECK: tensor<2x3xf32>

  // Unranked tensor type.
  MlirType unrankedTensor = mlirUnrankedTensorTypeGet(f32);
  if (!mlirTypeIsATensor(unrankedTensor) ||
      !mlirTypeIsAUnrankedTensor(unrankedTensor) ||
      mlirShapedTypeHasRank(unrankedTensor))
    return 17;
  mlirTypeDump(unrankedTensor);
  fprintf(stderr, "\n");
  // CHECK: tensor<*xf32>

  // MemRef type.
  MlirType memRef = mlirMemRefTypeContiguousGet(
      f32, sizeof(shape) / sizeof(int64_t), shape, 2);
  if (!mlirTypeIsAMemRef(memRef) ||
      mlirMemRefTypeGetNumAffineMaps(memRef) != 0 ||
      mlirMemRefTypeGetMemorySpace(memRef) != 2)
    return 18;
  mlirTypeDump(memRef);
  fprintf(stderr, "\n");
  // CHECK: memref<2x3xf32, 2>

  // Unranked MemRef type.
  MlirType unrankedMemRef = mlirUnrankedMemRefTypeGet(f32, 4);
  if (!mlirTypeIsAUnrankedMemRef(unrankedMemRef) ||
      mlirTypeIsAMemRef(unrankedMemRef) ||
      mlirUnrankedMemrefGetMemorySpace(unrankedMemRef) != 4)
    return 19;
  mlirTypeDump(unrankedMemRef);
  fprintf(stderr, "\n");
  // CHECK: memref<*xf32, 4>

  // Tuple type.
  MlirType types[] = {unrankedMemRef, f32};
  MlirType tuple = mlirTupleTypeGet(ctx, 2, types);
  if (!mlirTypeIsATuple(tuple) || mlirTupleTypeGetNumTypes(tuple) != 2 ||
      !mlirTypeEqual(mlirTupleTypeGetType(tuple, 0), unrankedMemRef) ||
      !mlirTypeEqual(mlirTupleTypeGetType(tuple, 1), f32))
    return 20;
  mlirTypeDump(tuple);
  fprintf(stderr, "\n");
  // CHECK: tuple<memref<*xf32, 4>, f32>

  // Function type.
  MlirType funcInputs[2] = {mlirIndexTypeGet(ctx), mlirIntegerTypeGet(ctx, 1)};
  MlirType funcResults[3] = {mlirIntegerTypeGet(ctx, 16),
                             mlirIntegerTypeGet(ctx, 32),
                             mlirIntegerTypeGet(ctx, 64)};
  MlirType funcType = mlirFunctionTypeGet(ctx, 2, funcInputs, 3, funcResults);
  if (mlirFunctionTypeGetNumInputs(funcType) != 2)
    return 21;
  if (mlirFunctionTypeGetNumResults(funcType) != 3)
    return 22;
  if (!mlirTypeEqual(funcInputs[0], mlirFunctionTypeGetInput(funcType, 0)) ||
      !mlirTypeEqual(funcInputs[1], mlirFunctionTypeGetInput(funcType, 1)))
    return 23;
  if (!mlirTypeEqual(funcResults[0], mlirFunctionTypeGetResult(funcType, 0)) ||
      !mlirTypeEqual(funcResults[1], mlirFunctionTypeGetResult(funcType, 1)) ||
      !mlirTypeEqual(funcResults[2], mlirFunctionTypeGetResult(funcType, 2)))
    return 24;
  mlirTypeDump(funcType);
  fprintf(stderr, "\n");
  // CHECK: (index, i1) -> (i16, i32, i64)

  return 0;
}

void callbackSetFixedLengthString(const char *data, intptr_t len,
                                  void *userData) {
  strncpy(userData, data, len);
}

int printStandardAttributes(MlirContext ctx) {
  MlirAttribute floating =
      mlirFloatAttrDoubleGet(ctx, mlirF64TypeGet(ctx), 2.0);
  if (!mlirAttributeIsAFloat(floating) ||
      fabs(mlirFloatAttrGetValueDouble(floating) - 2.0) > 1E-6)
    return 1;
  fprintf(stderr, "@attrs\n");
  mlirAttributeDump(floating);
  // CHECK-LABEL: @attrs
  // CHECK: 2.000000e+00 : f64

  // Exercise mlirAttributeGetType() just for the first one.
  MlirType floatingType = mlirAttributeGetType(floating);
  mlirTypeDump(floatingType);
  // CHECK: f64

  MlirAttribute integer = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), 42);
  if (!mlirAttributeIsAInteger(integer) ||
      mlirIntegerAttrGetValueInt(integer) != 42)
    return 2;
  mlirAttributeDump(integer);
  // CHECK: 42 : i32

  MlirAttribute boolean = mlirBoolAttrGet(ctx, 1);
  if (!mlirAttributeIsABool(boolean) || !mlirBoolAttrGetValue(boolean))
    return 3;
  mlirAttributeDump(boolean);
  // CHECK: true

  const char data[] = "abcdefghijklmnopqestuvwxyz";
  MlirAttribute opaque =
      mlirOpaqueAttrGet(ctx, "std", 3, data, mlirNoneTypeGet(ctx));
  if (!mlirAttributeIsAOpaque(opaque) ||
      strcmp("std", mlirOpaqueAttrGetDialectNamespace(opaque)))
    return 4;

  MlirStringRef opaqueData = mlirOpaqueAttrGetData(opaque);
  if (opaqueData.length != 3 ||
      strncmp(data, opaqueData.data, opaqueData.length))
    return 5;
  mlirAttributeDump(opaque);
  // CHECK: #std.abc

  MlirAttribute string = mlirStringAttrGet(ctx, 2, data + 3);
  if (!mlirAttributeIsAString(string))
    return 6;

  MlirStringRef stringValue = mlirStringAttrGetValue(string);
  if (stringValue.length != 2 ||
      strncmp(data + 3, stringValue.data, stringValue.length))
    return 7;
  mlirAttributeDump(string);
  // CHECK: "de"

  MlirAttribute flatSymbolRef = mlirFlatSymbolRefAttrGet(ctx, 3, data + 5);
  if (!mlirAttributeIsAFlatSymbolRef(flatSymbolRef))
    return 8;

  MlirStringRef flatSymbolRefValue =
      mlirFlatSymbolRefAttrGetValue(flatSymbolRef);
  if (flatSymbolRefValue.length != 3 ||
      strncmp(data + 5, flatSymbolRefValue.data, flatSymbolRefValue.length))
    return 9;
  mlirAttributeDump(flatSymbolRef);
  // CHECK: @fgh

  MlirAttribute symbols[] = {flatSymbolRef, flatSymbolRef};
  MlirAttribute symbolRef = mlirSymbolRefAttrGet(ctx, 2, data + 8, 2, symbols);
  if (!mlirAttributeIsASymbolRef(symbolRef) ||
      mlirSymbolRefAttrGetNumNestedReferences(symbolRef) != 2 ||
      !mlirAttributeEqual(mlirSymbolRefAttrGetNestedReference(symbolRef, 0),
                          flatSymbolRef) ||
      !mlirAttributeEqual(mlirSymbolRefAttrGetNestedReference(symbolRef, 1),
                          flatSymbolRef))
    return 10;

  MlirStringRef symbolRefLeaf = mlirSymbolRefAttrGetLeafReference(symbolRef);
  MlirStringRef symbolRefRoot = mlirSymbolRefAttrGetRootReference(symbolRef);
  if (symbolRefLeaf.length != 3 ||
      strncmp(data + 5, symbolRefLeaf.data, symbolRefLeaf.length) ||
      symbolRefRoot.length != 2 ||
      strncmp(data + 8, symbolRefRoot.data, symbolRefRoot.length))
    return 11;
  mlirAttributeDump(symbolRef);
  // CHECK: @ij::@fgh::@fgh

  MlirAttribute type = mlirTypeAttrGet(mlirF32TypeGet(ctx));
  if (!mlirAttributeIsAType(type) ||
      !mlirTypeEqual(mlirF32TypeGet(ctx), mlirTypeAttrGetValue(type)))
    return 12;
  mlirAttributeDump(type);
  // CHECK: f32

  MlirAttribute unit = mlirUnitAttrGet(ctx);
  if (!mlirAttributeIsAUnit(unit))
    return 13;
  mlirAttributeDump(unit);
  // CHECK: unit

  int64_t shape[] = {1, 2};

  int bools[] = {0, 1};
  uint32_t uints32[] = {0u, 1u};
  int32_t ints32[] = {0, 1};
  uint64_t uints64[] = {0u, 1u};
  int64_t ints64[] = {0, 1};
  float floats[] = {0.0f, 1.0f};
  double doubles[] = {0.0, 1.0};
  MlirAttribute boolElements = mlirDenseElementsAttrBoolGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 1)), 2, bools);
  MlirAttribute uint32Elements = mlirDenseElementsAttrUInt32Get(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeUnsignedGet(ctx, 32)), 2,
      uints32);
  MlirAttribute int32Elements = mlirDenseElementsAttrInt32Get(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 32)), 2,
      ints32);
  MlirAttribute uint64Elements = mlirDenseElementsAttrUInt64Get(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeUnsignedGet(ctx, 64)), 2,
      uints64);
  MlirAttribute int64Elements = mlirDenseElementsAttrInt64Get(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 64)), 2,
      ints64);
  MlirAttribute floatElements = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(2, shape, mlirF32TypeGet(ctx)), 2, floats);
  MlirAttribute doubleElements = mlirDenseElementsAttrDoubleGet(
      mlirRankedTensorTypeGet(2, shape, mlirF64TypeGet(ctx)), 2, doubles);

  if (!mlirAttributeIsADenseElements(boolElements) ||
      !mlirAttributeIsADenseElements(uint32Elements) ||
      !mlirAttributeIsADenseElements(int32Elements) ||
      !mlirAttributeIsADenseElements(uint64Elements) ||
      !mlirAttributeIsADenseElements(int64Elements) ||
      !mlirAttributeIsADenseElements(floatElements) ||
      !mlirAttributeIsADenseElements(doubleElements))
    return 14;

  if (mlirDenseElementsAttrGetBoolValue(boolElements, 1) != 1 ||
      mlirDenseElementsAttrGetUInt32Value(uint32Elements, 1) != 1 ||
      mlirDenseElementsAttrGetInt32Value(int32Elements, 1) != 1 ||
      mlirDenseElementsAttrGetUInt64Value(uint64Elements, 1) != 1 ||
      mlirDenseElementsAttrGetInt64Value(int64Elements, 1) != 1 ||
      fabsf(mlirDenseElementsAttrGetFloatValue(floatElements, 1) - 1.0f) >
          1E-6f ||
      fabs(mlirDenseElementsAttrGetDoubleValue(doubleElements, 1) - 1.0) > 1E-6)
    return 15;

  mlirAttributeDump(boolElements);
  mlirAttributeDump(uint32Elements);
  mlirAttributeDump(int32Elements);
  mlirAttributeDump(uint64Elements);
  mlirAttributeDump(int64Elements);
  mlirAttributeDump(floatElements);
  mlirAttributeDump(doubleElements);
  // CHECK: dense<{{\[}}[false, true]]> : tensor<1x2xi1>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui32>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi32>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui64>
  // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi64>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf32>
  // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf64>

  MlirAttribute splatBool = mlirDenseElementsAttrBoolSplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 1)), 1);
  MlirAttribute splatUInt32 = mlirDenseElementsAttrUInt32SplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 32)), 1);
  MlirAttribute splatInt32 = mlirDenseElementsAttrInt32SplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 32)), 1);
  MlirAttribute splatUInt64 = mlirDenseElementsAttrUInt64SplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 64)), 1);
  MlirAttribute splatInt64 = mlirDenseElementsAttrInt64SplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirIntegerTypeGet(ctx, 64)), 1);
  MlirAttribute splatFloat = mlirDenseElementsAttrFloatSplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirF32TypeGet(ctx)), 1.0f);
  MlirAttribute splatDouble = mlirDenseElementsAttrDoubleSplatGet(
      mlirRankedTensorTypeGet(2, shape, mlirF64TypeGet(ctx)), 1.0);

  if (!mlirAttributeIsADenseElements(splatBool) ||
      !mlirDenseElementsAttrIsSplat(splatBool) ||
      !mlirAttributeIsADenseElements(splatUInt32) ||
      !mlirDenseElementsAttrIsSplat(splatUInt32) ||
      !mlirAttributeIsADenseElements(splatInt32) ||
      !mlirDenseElementsAttrIsSplat(splatInt32) ||
      !mlirAttributeIsADenseElements(splatUInt64) ||
      !mlirDenseElementsAttrIsSplat(splatUInt64) ||
      !mlirAttributeIsADenseElements(splatInt64) ||
      !mlirDenseElementsAttrIsSplat(splatInt64) ||
      !mlirAttributeIsADenseElements(splatFloat) ||
      !mlirDenseElementsAttrIsSplat(splatFloat) ||
      !mlirAttributeIsADenseElements(splatDouble) ||
      !mlirDenseElementsAttrIsSplat(splatDouble))
    return 16;

  if (mlirDenseElementsAttrGetBoolSplatValue(splatBool) != 1 ||
      mlirDenseElementsAttrGetUInt32SplatValue(splatUInt32) != 1 ||
      mlirDenseElementsAttrGetInt32SplatValue(splatInt32) != 1 ||
      mlirDenseElementsAttrGetUInt64SplatValue(splatUInt64) != 1 ||
      mlirDenseElementsAttrGetInt64SplatValue(splatInt64) != 1 ||
      fabsf(mlirDenseElementsAttrGetFloatSplatValue(splatFloat) - 1.0f) >
          1E-6f ||
      fabs(mlirDenseElementsAttrGetDoubleSplatValue(splatDouble) - 1.0) > 1E-6)
    return 17;

  uint32_t *uint32RawData =
      (uint32_t *)mlirDenseElementsAttrGetRawData(uint32Elements);
  int32_t *int32RawData =
      (int32_t *)mlirDenseElementsAttrGetRawData(int32Elements);
  uint64_t *uint64RawData =
      (uint64_t *)mlirDenseElementsAttrGetRawData(uint64Elements);
  int64_t *int64RawData =
      (int64_t *)mlirDenseElementsAttrGetRawData(int64Elements);
  float *floatRawData =
      (float *)mlirDenseElementsAttrGetRawData(floatElements);
  double *doubleRawData =
      (double *)mlirDenseElementsAttrGetRawData(doubleElements);
  if (uint32RawData[0] != 0u || uint32RawData[1] != 1u ||
      int32RawData[0] != 0 || int32RawData[1] != 1 ||
      uint64RawData[0] != 0u || uint64RawData[1] != 1u ||
      int64RawData[0] != 0 || int64RawData[1] != 1 ||
      floatRawData[0] != 0.0f || floatRawData[1] != 1.0f ||
      doubleRawData[0] != 0.0 || doubleRawData[1] != 1.0)
    return 18;

  mlirAttributeDump(splatBool);
  mlirAttributeDump(splatUInt32);
  mlirAttributeDump(splatInt32);
  mlirAttributeDump(splatUInt64);
  mlirAttributeDump(splatInt64);
  mlirAttributeDump(splatFloat);
  mlirAttributeDump(splatDouble);
  // CHECK: dense<true> : tensor<1x2xi1>
  // CHECK: dense<1> : tensor<1x2xi32>
  // CHECK: dense<1> : tensor<1x2xi32>
  // CHECK: dense<1> : tensor<1x2xi64>
  // CHECK: dense<1> : tensor<1x2xi64>
  // CHECK: dense<1.000000e+00> : tensor<1x2xf32>
  // CHECK: dense<1.000000e+00> : tensor<1x2xf64>

  mlirAttributeDump(mlirElementsAttrGetValue(floatElements, 2, uints64));
  mlirAttributeDump(mlirElementsAttrGetValue(doubleElements, 2, uints64));
  // CHECK: 1.000000e+00 : f32
  // CHECK: 1.000000e+00 : f64

  int64_t indices[] = {4, 7};
  int64_t two = 2;
  MlirAttribute indicesAttr = mlirDenseElementsAttrInt64Get(
      mlirRankedTensorTypeGet(1, &two, mlirIntegerTypeGet(ctx, 64)), 2,
      indices);
  MlirAttribute valuesAttr = mlirDenseElementsAttrFloatGet(
      mlirRankedTensorTypeGet(1, &two, mlirF32TypeGet(ctx)), 2, floats);
  MlirAttribute sparseAttr = mlirSparseElementsAttribute(
      mlirRankedTensorTypeGet(2, shape, mlirF32TypeGet(ctx)), indicesAttr,
      valuesAttr);
  mlirAttributeDump(sparseAttr);
  // CHECK: sparse<[4, 7], [0.000000e+00, 1.000000e+00]> : tensor<1x2xf32>

  return 0;
}

int printAffineMap(MlirContext ctx) {
  MlirAffineMap emptyAffineMap = mlirAffineMapEmptyGet(ctx);
  MlirAffineMap affineMap = mlirAffineMapGet(ctx, 3, 2);
  MlirAffineMap constAffineMap = mlirAffineMapConstantGet(ctx, 2);
  MlirAffineMap multiDimIdentityAffineMap =
      mlirAffineMapMultiDimIdentityGet(ctx, 3);
  MlirAffineMap minorIdentityAffineMap =
      mlirAffineMapMinorIdentityGet(ctx, 3, 2);
  unsigned permutation[] = {1, 2, 0};
  MlirAffineMap permutationAffineMap = mlirAffineMapPermutationGet(
      ctx, sizeof(permutation) / sizeof(unsigned), permutation);

  fprintf(stderr, "@affineMap\n");
  mlirAffineMapDump(emptyAffineMap);
  mlirAffineMapDump(affineMap);
  mlirAffineMapDump(constAffineMap);
  mlirAffineMapDump(multiDimIdentityAffineMap);
  mlirAffineMapDump(minorIdentityAffineMap);
  mlirAffineMapDump(permutationAffineMap);
  // CHECK-LABEL: @affineMap
  // CHECK: () -> ()
  // CHECK: (d0, d1, d2)[s0, s1] -> ()
  // CHECK: () -> (2)
  // CHECK: (d0, d1, d2) -> (d0, d1, d2)
  // CHECK: (d0, d1, d2) -> (d1, d2)
  // CHECK: (d0, d1, d2) -> (d1, d2, d0)

  if (!mlirAffineMapIsIdentity(emptyAffineMap) ||
      mlirAffineMapIsIdentity(affineMap) ||
      mlirAffineMapIsIdentity(constAffineMap) ||
      !mlirAffineMapIsIdentity(multiDimIdentityAffineMap) ||
      mlirAffineMapIsIdentity(minorIdentityAffineMap) ||
      mlirAffineMapIsIdentity(permutationAffineMap))
    return 1;

  if (!mlirAffineMapIsMinorIdentity(emptyAffineMap) ||
      mlirAffineMapIsMinorIdentity(affineMap) ||
      !mlirAffineMapIsMinorIdentity(multiDimIdentityAffineMap) ||
      !mlirAffineMapIsMinorIdentity(minorIdentityAffineMap) ||
      mlirAffineMapIsMinorIdentity(permutationAffineMap))
    return 2;

  if (!mlirAffineMapIsEmpty(emptyAffineMap) ||
      mlirAffineMapIsEmpty(affineMap) || mlirAffineMapIsEmpty(constAffineMap) ||
      mlirAffineMapIsEmpty(multiDimIdentityAffineMap) ||
      mlirAffineMapIsEmpty(minorIdentityAffineMap) ||
      mlirAffineMapIsEmpty(permutationAffineMap))
    return 3;

  if (mlirAffineMapIsSingleConstant(emptyAffineMap) ||
      mlirAffineMapIsSingleConstant(affineMap) ||
      !mlirAffineMapIsSingleConstant(constAffineMap) ||
      mlirAffineMapIsSingleConstant(multiDimIdentityAffineMap) ||
      mlirAffineMapIsSingleConstant(minorIdentityAffineMap) ||
      mlirAffineMapIsSingleConstant(permutationAffineMap))
    return 4;

  if (mlirAffineMapGetSingleConstantResult(constAffineMap) != 2)
    return 5;

  if (mlirAffineMapGetNumDims(emptyAffineMap) != 0 ||
      mlirAffineMapGetNumDims(affineMap) != 3 ||
      mlirAffineMapGetNumDims(constAffineMap) != 0 ||
      mlirAffineMapGetNumDims(multiDimIdentityAffineMap) != 3 ||
      mlirAffineMapGetNumDims(minorIdentityAffineMap) != 3 ||
      mlirAffineMapGetNumDims(permutationAffineMap) != 3)
    return 6;

  if (mlirAffineMapGetNumSymbols(emptyAffineMap) != 0 ||
      mlirAffineMapGetNumSymbols(affineMap) != 2 ||
      mlirAffineMapGetNumSymbols(constAffineMap) != 0 ||
      mlirAffineMapGetNumSymbols(multiDimIdentityAffineMap) != 0 ||
      mlirAffineMapGetNumSymbols(minorIdentityAffineMap) != 0 ||
      mlirAffineMapGetNumSymbols(permutationAffineMap) != 0)
    return 7;

  if (mlirAffineMapGetNumResults(emptyAffineMap) != 0 ||
      mlirAffineMapGetNumResults(affineMap) != 0 ||
      mlirAffineMapGetNumResults(constAffineMap) != 1 ||
      mlirAffineMapGetNumResults(multiDimIdentityAffineMap) != 3 ||
      mlirAffineMapGetNumResults(minorIdentityAffineMap) != 2 ||
      mlirAffineMapGetNumResults(permutationAffineMap) != 3)
    return 8;

  if (mlirAffineMapGetNumInputs(emptyAffineMap) != 0 ||
      mlirAffineMapGetNumInputs(affineMap) != 5 ||
      mlirAffineMapGetNumInputs(constAffineMap) != 0 ||
      mlirAffineMapGetNumInputs(multiDimIdentityAffineMap) != 3 ||
      mlirAffineMapGetNumInputs(minorIdentityAffineMap) != 3 ||
      mlirAffineMapGetNumInputs(permutationAffineMap) != 3)
    return 9;

  if (!mlirAffineMapIsProjectedPermutation(emptyAffineMap) ||
      !mlirAffineMapIsPermutation(emptyAffineMap) ||
      mlirAffineMapIsProjectedPermutation(affineMap) ||
      mlirAffineMapIsPermutation(affineMap) ||
      mlirAffineMapIsProjectedPermutation(constAffineMap) ||
      mlirAffineMapIsPermutation(constAffineMap) ||
      !mlirAffineMapIsProjectedPermutation(multiDimIdentityAffineMap) ||
      !mlirAffineMapIsPermutation(multiDimIdentityAffineMap) ||
      !mlirAffineMapIsProjectedPermutation(minorIdentityAffineMap) ||
      mlirAffineMapIsPermutation(minorIdentityAffineMap) ||
      !mlirAffineMapIsProjectedPermutation(permutationAffineMap) ||
      !mlirAffineMapIsPermutation(permutationAffineMap))
    return 10;

  intptr_t sub[] = {1};

  MlirAffineMap subMap = mlirAffineMapGetSubMap(
      multiDimIdentityAffineMap, sizeof(sub) / sizeof(intptr_t), sub);
  MlirAffineMap majorSubMap =
      mlirAffineMapGetMajorSubMap(multiDimIdentityAffineMap, 1);
  MlirAffineMap minorSubMap =
      mlirAffineMapGetMinorSubMap(multiDimIdentityAffineMap, 1);

  mlirAffineMapDump(subMap);
  mlirAffineMapDump(majorSubMap);
  mlirAffineMapDump(minorSubMap);
  // CHECK: (d0, d1, d2) -> (d1)
  // CHECK: (d0, d1, d2) -> (d0)
  // CHECK: (d0, d1, d2) -> (d2)

  return 0;
}

int printAffineExpr(MlirContext ctx) {
  MlirAffineExpr affineDimExpr = mlirAffineDimExprGet(ctx, 5);
  MlirAffineExpr affineSymbolExpr = mlirAffineSymbolExprGet(ctx, 5);
  MlirAffineExpr affineConstantExpr = mlirAffineConstantExprGet(ctx, 5);
  MlirAffineExpr affineAddExpr =
      mlirAffineAddExprGet(affineDimExpr, affineSymbolExpr);
  MlirAffineExpr affineMulExpr =
      mlirAffineMulExprGet(affineDimExpr, affineSymbolExpr);
  MlirAffineExpr affineModExpr =
      mlirAffineModExprGet(affineDimExpr, affineSymbolExpr);
  MlirAffineExpr affineFloorDivExpr =
      mlirAffineFloorDivExprGet(affineDimExpr, affineSymbolExpr);
  MlirAffineExpr affineCeilDivExpr =
      mlirAffineCeilDivExprGet(affineDimExpr, affineSymbolExpr);

  // Tests mlirAffineExprDump.
  fprintf(stderr, "@affineExpr\n");
  mlirAffineExprDump(affineDimExpr);
  mlirAffineExprDump(affineSymbolExpr);
  mlirAffineExprDump(affineConstantExpr);
  mlirAffineExprDump(affineAddExpr);
  mlirAffineExprDump(affineMulExpr);
  mlirAffineExprDump(affineModExpr);
  mlirAffineExprDump(affineFloorDivExpr);
  mlirAffineExprDump(affineCeilDivExpr);
  // CHECK-LABEL: @affineExpr
  // CHECK: d5
  // CHECK: s5
  // CHECK: 5
  // CHECK: d5 + s5
  // CHECK: d5 * s5
  // CHECK: d5 mod s5
  // CHECK: d5 floordiv s5
  // CHECK: d5 ceildiv s5

  // Tests methods of affine binary operation expression, takes add expression
  // as an example.
  mlirAffineExprDump(mlirAffineBinaryOpExprGetLHS(affineAddExpr));
  mlirAffineExprDump(mlirAffineBinaryOpExprGetRHS(affineAddExpr));
  // CHECK: d5
  // CHECK: s5

  // Tests methods of affine dimension expression.
  if (mlirAffineDimExprGetPosition(affineDimExpr) != 5)
    return 1;

  // Tests methods of affine symbol expression.
  if (mlirAffineSymbolExprGetPosition(affineSymbolExpr) != 5)
    return 2;

  // Tests methods of affine constant expression.
  if (mlirAffineConstantExprGetValue(affineConstantExpr) != 5)
    return 3;

  // Tests methods of affine expression.
  if (mlirAffineExprIsSymbolicOrConstant(affineDimExpr) ||
      !mlirAffineExprIsSymbolicOrConstant(affineSymbolExpr) ||
      !mlirAffineExprIsSymbolicOrConstant(affineConstantExpr) ||
      mlirAffineExprIsSymbolicOrConstant(affineAddExpr) ||
      mlirAffineExprIsSymbolicOrConstant(affineMulExpr) ||
      mlirAffineExprIsSymbolicOrConstant(affineModExpr) ||
      mlirAffineExprIsSymbolicOrConstant(affineFloorDivExpr) ||
      mlirAffineExprIsSymbolicOrConstant(affineCeilDivExpr))
    return 4;

  if (!mlirAffineExprIsPureAffine(affineDimExpr) ||
      !mlirAffineExprIsPureAffine(affineSymbolExpr) ||
      !mlirAffineExprIsPureAffine(affineConstantExpr) ||
      !mlirAffineExprIsPureAffine(affineAddExpr) ||
      mlirAffineExprIsPureAffine(affineMulExpr) ||
      mlirAffineExprIsPureAffine(affineModExpr) ||
      mlirAffineExprIsPureAffine(affineFloorDivExpr) ||
      mlirAffineExprIsPureAffine(affineCeilDivExpr))
    return 5;

  if (mlirAffineExprGetLargestKnownDivisor(affineDimExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineSymbolExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineConstantExpr) != 5 ||
      mlirAffineExprGetLargestKnownDivisor(affineAddExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineMulExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineModExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineFloorDivExpr) != 1 ||
      mlirAffineExprGetLargestKnownDivisor(affineCeilDivExpr) != 1)
    return 6;

  if (!mlirAffineExprIsMultipleOf(affineDimExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineSymbolExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineConstantExpr, 5) ||
      !mlirAffineExprIsMultipleOf(affineAddExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineMulExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineModExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineFloorDivExpr, 1) ||
      !mlirAffineExprIsMultipleOf(affineCeilDivExpr, 1))
    return 7;

  if (!mlirAffineExprIsFunctionOfDim(affineDimExpr, 5) ||
      mlirAffineExprIsFunctionOfDim(affineSymbolExpr, 5) ||
      mlirAffineExprIsFunctionOfDim(affineConstantExpr, 5) ||
      !mlirAffineExprIsFunctionOfDim(affineAddExpr, 5) ||
      !mlirAffineExprIsFunctionOfDim(affineMulExpr, 5) ||
      !mlirAffineExprIsFunctionOfDim(affineModExpr, 5) ||
      !mlirAffineExprIsFunctionOfDim(affineFloorDivExpr, 5) ||
      !mlirAffineExprIsFunctionOfDim(affineCeilDivExpr, 5))
    return 8;

  // Tests 'IsA' methods of affine binary operation expression.
  if (!mlirAffineExprIsAAdd(affineAddExpr))
    return 9;

  if (!mlirAffineExprIsAMul(affineMulExpr))
    return 10;

  if (!mlirAffineExprIsAMod(affineModExpr))
    return 11;

  if (!mlirAffineExprIsAFloorDiv(affineFloorDivExpr))
    return 12;

  if (!mlirAffineExprIsACeilDiv(affineCeilDivExpr))
    return 13;

  return 0;
}

int registerOnlyStd() {
  MlirContext ctx = mlirContextCreate();
  // The built-in dialect is always loaded.
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 1;

  MlirDialect std =
      mlirContextGetOrLoadDialect(ctx, mlirStandardDialectGetNamespace());
  if (!mlirDialectIsNull(std))
    return 2;

  mlirContextRegisterStandardDialect(ctx);
  if (mlirContextGetNumRegisteredDialects(ctx) != 1)
    return 3;
  if (mlirContextGetNumLoadedDialects(ctx) != 1)
    return 4;

  std = mlirContextGetOrLoadDialect(ctx, mlirStandardDialectGetNamespace());
  if (mlirDialectIsNull(std))
    return 5;
  if (mlirContextGetNumLoadedDialects(ctx) != 2)
    return 6;

  MlirDialect alsoStd = mlirContextLoadStandardDialect(ctx);
  if (!mlirDialectEqual(std, alsoStd))
    return 7;

  MlirStringRef stdNs = mlirDialectGetNamespace(std);
  MlirStringRef alsoStdNs = mlirStandardDialectGetNamespace();
  if (stdNs.length != alsoStdNs.length ||
      strncmp(stdNs.data, alsoStdNs.data, stdNs.length))
    return 8;

  fprintf(stderr, "@registration\n");
  // CHECK-LABEL: @registration

  return 0;
}

// Wraps a diagnostic into additional text we can match against.
MlirLogicalResult errorHandler(MlirDiagnostic diagnostic) {
  fprintf(stderr, "processing diagnostic <<\n");
  mlirDiagnosticPrint(diagnostic, printToStderr, NULL);
  fprintf(stderr, "\n");
  MlirLocation loc = mlirDiagnosticGetLocation(diagnostic);
  mlirLocationPrint(loc, printToStderr, NULL);
  assert(mlirDiagnosticGetNumNotes(diagnostic) == 0);
  fprintf(stderr, ">> end of diagnostic\n");
  return mlirLogicalResultSuccess();
}

void testDiagnostics() {
  MlirContext ctx = mlirContextCreate();
  MlirDiagnosticHandlerID id =
      mlirContextAttachDiagnosticHandler(ctx, errorHandler);
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  fprintf(stderr, "@test_diagnostics\n");
  mlirEmitError(loc, "test diagnostics");
  mlirContextDetachDiagnosticHandler(ctx, id);
  mlirEmitError(loc, "more test diagnostics");
  // CHECK-LABEL: @test_diagnostics
  // CHECK: processing diagnostic <<
  // CHECK:   test diagnostics
  // CHECK:   loc(unknown)
  // CHECK: >> end of diagnostic
  // CHECK-NOT: processing diagnostic
  // CHECK:     more test diagnostics
}

int main() {
  MlirContext ctx = mlirContextCreate();
  mlirRegisterAllDialects(ctx);
  if (constructAndTraverseIr(ctx))
    return 1;
  buildWithInsertionsAndPrint(ctx);

  if (printStandardTypes(ctx))
    return 2;
  if (printStandardAttributes(ctx))
    return 3;
  if (printAffineMap(ctx))
    return 4;
  if (printAffineExpr(ctx))
    return 5;
  if (registerOnlyStd())
    return 6;

  mlirContextDestroy(ctx);

  testDiagnostics();
  return 0;
}
