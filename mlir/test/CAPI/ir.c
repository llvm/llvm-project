/*===- ir.c - Simple test of C APIs ---------------------------------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* RUN: mlir-capi-ir-test 2>&1 | FileCheck %s
 */

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

MlirModule makeAdd(MlirContext ctx, MlirLocation location) {
  MlirModule moduleOp = mlirModuleCreateEmpty(location);
  MlirOperation module = mlirModuleGetOperation(moduleOp);
  MlirRegion moduleBodyRegion = mlirOperationGetRegion(module, 0);
  MlirBlock moduleBody = mlirRegionGetFirstBlock(moduleBodyRegion);

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
};
typedef struct ModuleStats ModuleStats;

void collectStatsSingle(OpListNode *head, ModuleStats *stats) {
  MlirOperation operation = head->op;
  stats->numOperations += 1;
  stats->numValues += mlirOperationGetNumResults(operation);
  stats->numAttributes += mlirOperationGetNumAttributes(operation);

  unsigned numRegions = mlirOperationGetNumRegions(operation);

  stats->numRegions += numRegions;

  for (unsigned i = 0; i < numRegions; ++i) {
    MlirRegion region = mlirOperationGetRegion(operation, i);
    for (MlirBlock block = mlirRegionGetFirstBlock(region);
         !mlirBlockIsNull(block); block = mlirBlockGetNextInRegion(block)) {
      ++stats->numBlocks;
      stats->numValues += mlirBlockGetNumArguments(block);

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
}

void collectStats(MlirOperation operation) {
  OpListNode *head = malloc(sizeof(OpListNode));
  head->op = operation;
  head->next = NULL;

  ModuleStats stats;
  stats.numOperations = 0;
  stats.numAttributes = 0;
  stats.numBlocks = 0;
  stats.numRegions = 0;
  stats.numValues = 0;

  do {
    collectStatsSingle(head, &stats);
    OpListNode *next = head->next;
    free(head);
    head = next;
  } while (head);

  printf("Number of operations: %u\n", stats.numOperations);
  printf("Number of attributes: %u\n", stats.numAttributes);
  printf("Number of blocks: %u\n", stats.numBlocks);
  printf("Number of regions: %u\n", stats.numRegions);
  printf("Number of values: %u\n", stats.numValues);
}

int main() {
  mlirRegisterAllDialects();
  MlirContext ctx = mlirContextCreate();
  MlirLocation location = mlirLocationUnknownGet(ctx);

  MlirModule moduleOp = makeAdd(ctx, location);
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

  collectStats(module);
  // clang-format off
  // CHECK: Number of operations: 13
  // CHECK: Number of attributes: 4
  // CHECK: Number of blocks: 3
  // CHECK: Number of regions: 3
  // CHECK: Number of values: 9
  // clang-format on

  mlirModuleDestroy(moduleOp);
  mlirContextDestroy(ctx);

  return 0;
}
