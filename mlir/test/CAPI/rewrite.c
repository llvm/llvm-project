//===- rewrite.c - Test of the rewriting C API ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-capi-rewrite-test 2>&1 | FileCheck %s

#include "mlir-c/Rewrite.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

MlirOperation createOperationWithName(MlirContext ctx, const char *name) {
  MlirStringRef nameRef = mlirStringRefCreateFromCString(name);
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirOperationState state = mlirOperationStateGet(nameRef, loc);
  MlirType indexType = mlirIndexTypeGet(ctx);
  mlirOperationStateAddResults(&state, 1, &indexType);
  return mlirOperationCreate(&state);
}

void testInsertionPoint(MlirContext ctx) {
  // CHECK-LABEL: @testInsertionPoint
  fprintf(stderr, "@testInsertionPoint\n");

  const char *moduleString = "\"dialect.op1\"() : () -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);
  MlirOperation op1 = mlirBlockGetFirstOperation(body);

  // IRRewriter create
  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);

  // Insert before op
  mlirRewriterBaseSetInsertionPointBefore(rewriter, op1);
  MlirOperation op2 = createOperationWithName(ctx, "dialect.op2");
  mlirRewriterBaseInsert(rewriter, op2);

  // Insert after op
  mlirRewriterBaseSetInsertionPointAfter(rewriter, op2);
  MlirOperation op3 = createOperationWithName(ctx, "dialect.op3");
  mlirRewriterBaseInsert(rewriter, op3);
  MlirValue op3Res = mlirOperationGetResult(op3, 0);

  // Insert after value
  mlirRewriterBaseSetInsertionPointAfterValue(rewriter, op3Res);
  MlirOperation op4 = createOperationWithName(ctx, "dialect.op4");
  mlirRewriterBaseInsert(rewriter, op4);

  // Insert at beginning of block
  mlirRewriterBaseSetInsertionPointToStart(rewriter, body);
  MlirOperation op5 = createOperationWithName(ctx, "dialect.op5");
  mlirRewriterBaseInsert(rewriter, op5);

  // Insert at end of block
  mlirRewriterBaseSetInsertionPointToEnd(rewriter, body);
  MlirOperation op6 = createOperationWithName(ctx, "dialect.op6");
  mlirRewriterBaseInsert(rewriter, op6);

  // Get insertion blocks
  MlirBlock block1 = mlirRewriterBaseGetBlock(rewriter);
  MlirBlock block2 = mlirRewriterBaseGetInsertionBlock(rewriter);
  (void)block1;
  (void)block2;
  assert(body.ptr == block1.ptr);
  assert(body.ptr == block2.ptr);

  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %{{.*}} = "dialect.op5"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op2"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op3"() : () -> index
  // CHECK-NEXT:   %{{.*}} = "dialect.op4"() : () -> index
  // CHECK-NEXT:   "dialect.op1"() : () -> ()
  // CHECK-NEXT:   %{{.*}} = "dialect.op6"() : () -> index
  // CHECK-NEXT: }
  // clang-format on
  mlirOperationDump(op);

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testCreateBlock(MlirContext ctx) {
  // CHECK-LABEL: @testCreateBlock
  fprintf(stderr, "@testCreateBlock\n");

  const char *moduleString = "\"dialect.op1\"() ({^bb0:}) : () -> ()\n"
                             "\"dialect.op2\"() ({^bb0:}) : () -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  MlirOperation op1 = mlirBlockGetFirstOperation(body);
  MlirRegion region1 = mlirOperationGetRegion(op1, 0);
  MlirBlock block1 = mlirRegionGetFirstBlock(region1);

  MlirOperation op2 = mlirOperationGetNextInBlock(op1);
  MlirRegion region2 = mlirOperationGetRegion(op2, 0);
  MlirBlock block2 = mlirRegionGetFirstBlock(region2);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);

  // Create block before
  MlirType indexType = mlirIndexTypeGet(ctx);
  MlirLocation unknown = mlirLocationUnknownGet(ctx);
  mlirRewriterBaseCreateBlockBefore(rewriter, block1, 1, &indexType, &unknown);

  mlirRewriterBaseSetInsertionPointToEnd(rewriter, body);

  // Clone operation
  mlirRewriterBaseClone(rewriter, op1);

  // Clone without regions
  mlirRewriterBaseCloneWithoutRegions(rewriter, op1);

  // Clone region before
  mlirRewriterBaseCloneRegionBefore(rewriter, region1, block2);

  mlirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: "builtin.module"() ({
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:   ^{{.*}}:
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // clang-format on

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testInlineRegionBlock(MlirContext ctx) {
  // CHECK-LABEL: @testInlineRegionBlock
  fprintf(stderr, "@testInlineRegionBlock\n");

  const char *moduleString =
      "\"dialect.op1\"() ({\n"
      "  ^bb0(%arg0: index):\n"
      "    \"dialect.op1_in1\"(%arg0) [^bb1] : (index) -> ()\n"
      "  ^bb1():\n"
      "    \"dialect.op1_in2\"() : () -> ()\n"
      "}) : () -> ()\n"
      "\"dialect.op2\"() ({^bb0:}) : () -> ()\n"
      "\"dialect.op3\"() ({\n"
      "  ^bb0(%arg0: index):\n"
      "    \"dialect.op3_in1\"(%arg0) : (index) -> ()\n"
      "  ^bb1():\n"
      "    %x = \"dialect.op3_in2\"() : () -> index\n"
      "    %y = \"dialect.op3_in3\"() : () -> index\n"
      "}) : () -> ()\n"
      "\"dialect.op4\"() ({\n"
      "  ^bb0():\n"
      "    \"dialect.op4_in1\"() : () -> index\n"
      "  ^bb1(%arg0: index):\n"
      "    \"dialect.op4_in2\"(%arg0) : (index) -> ()\n"
      "}) : () -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  MlirOperation op1 = mlirBlockGetFirstOperation(body);
  MlirRegion region1 = mlirOperationGetRegion(op1, 0);

  MlirOperation op2 = mlirOperationGetNextInBlock(op1);
  MlirRegion region2 = mlirOperationGetRegion(op2, 0);
  MlirBlock block2 = mlirRegionGetFirstBlock(region2);

  MlirOperation op3 = mlirOperationGetNextInBlock(op2);
  MlirRegion region3 = mlirOperationGetRegion(op3, 0);
  MlirBlock block3_1 = mlirRegionGetFirstBlock(region3);
  MlirBlock block3_2 = mlirBlockGetNextInRegion(block3_1);
  MlirOperation op3_in2 = mlirBlockGetFirstOperation(block3_2);
  MlirValue op3_in2_res = mlirOperationGetResult(op3_in2, 0);
  MlirOperation op3_in3 = mlirOperationGetNextInBlock(op3_in2);

  MlirOperation op4 = mlirOperationGetNextInBlock(op3);
  MlirRegion region4 = mlirOperationGetRegion(op4, 0);
  MlirBlock block4_1 = mlirRegionGetFirstBlock(region4);
  MlirOperation op4_in1 = mlirBlockGetFirstOperation(block4_1);
  MlirValue op4_in1_res = mlirOperationGetResult(op4_in1, 0);
  MlirBlock block4_2 = mlirBlockGetNextInRegion(block4_1);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);

  // Test these three functions
  mlirRewriterBaseInlineRegionBefore(rewriter, region1, block2);
  mlirRewriterBaseInlineBlockBefore(rewriter, block3_1, op3_in3, 1,
                                    &op3_in2_res);
  mlirRewriterBaseMergeBlocks(rewriter, block4_2, block4_1, 1, &op4_in1_res);

  mlirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: "builtin.module"() ({
  // CHECK-NEXT:   "dialect.op1"() ({
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
  // CHECK-NEXT:     "dialect.op1_in1"(%{{.*}})[^[[bb:.*]]] : (index) -> ()
  // CHECK-NEXT:   ^[[bb]]:
  // CHECK-NEXT:     "dialect.op1_in2"() : () -> ()
  // CHECK-NEXT:   ^{{.*}}:  // no predecessors
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op3"() ({
  // CHECK-NEXT:     %{{.*}} = "dialect.op3_in2"() : () -> index
  // CHECK-NEXT:     "dialect.op3_in1"(%{{.*}}) : (index) -> ()
  // CHECK-NEXT:     %{{.*}} = "dialect.op3_in3"() : () -> index
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op4"() ({
  // CHECK-NEXT:     %{{.*}} = "dialect.op4_in1"() : () -> index
  // CHECK-NEXT:     "dialect.op4_in2"(%{{.*}}) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // clang-format on

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testReplaceOp(MlirContext ctx) {
  // CHECK-LABEL: @testReplaceOp
  fprintf(stderr, "@testReplaceOp\n");

  const char *moduleString =
      "%x, %y, %z = \"dialect.create_values\"() : () -> (index, index, index)\n"
      "%x_1, %y_1 = \"dialect.op1\"() : () -> (index, index)\n"
      "\"dialect.use_op1\"(%x_1, %y_1) : (index, index) -> ()\n"
      "%x_2, %y_2 = \"dialect.op2\"() : () -> (index, index)\n"
      "%x_3, %y_3 = \"dialect.op3\"() : () -> (index, index)\n"
      "\"dialect.use_op2\"(%x_2, %y_2) : (index, index) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  // get a handle to all operations/values
  MlirOperation createValues = mlirBlockGetFirstOperation(body);
  MlirValue x = mlirOperationGetResult(createValues, 0);
  MlirValue z = mlirOperationGetResult(createValues, 2);
  MlirOperation op1 = mlirOperationGetNextInBlock(createValues);
  MlirOperation useOp1 = mlirOperationGetNextInBlock(op1);
  MlirOperation op2 = mlirOperationGetNextInBlock(useOp1);
  MlirOperation op3 = mlirOperationGetNextInBlock(op2);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);

  // Test replace op with values
  MlirValue xz[2] = {x, z};
  mlirRewriterBaseReplaceOpWithValues(rewriter, op1, 2, xz);

  // Test replace op with op
  mlirRewriterBaseReplaceOpWithOperation(rewriter, op2, op3);

  mlirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %[[res:.*]]:3 = "dialect.create_values"() : () -> (index, index, index)
  // CHECK-NEXT:   "dialect.use_op1"(%[[res]]#0, %[[res]]#2) : (index, index) -> ()
  // CHECK-NEXT:   %[[res2:.*]]:2 = "dialect.op3"() : () -> (index, index)
  // CHECK-NEXT:   "dialect.use_op2"(%[[res2]]#0, %[[res2]]#1) : (index, index) -> ()
  // CHECK-NEXT: }
  // clang-format on

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testErase(MlirContext ctx) {
  // CHECK-LABEL: @testErase
  fprintf(stderr, "@testErase\n");

  const char *moduleString = "\"dialect.op_to_erase\"() : () -> ()\n"
                             "\"dialect.op2\"() ({\n"
                             "^bb0():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "^block_to_erase():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "^bb1():\n"
                             "  \"dialect.op2_nested\"() : () -> ()"
                             "}) : () -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  // get a handle to all operations/values
  MlirOperation opToErase = mlirBlockGetFirstOperation(body);
  MlirOperation op2 = mlirOperationGetNextInBlock(opToErase);
  MlirRegion op2Region = mlirOperationGetRegion(op2, 0);
  MlirBlock bb0 = mlirRegionGetFirstBlock(op2Region);
  MlirBlock blockToErase = mlirBlockGetNextInRegion(bb0);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);
  mlirRewriterBaseEraseOp(rewriter, opToErase);
  mlirRewriterBaseEraseBlock(rewriter, blockToErase);

  mlirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT: "dialect.op2"() ({
  // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
  // CHECK-NEXT: ^{{.*}}:
  // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
  // CHECK-NEXT: }) : () -> ()
  // CHECK-NEXT: }

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testMove(MlirContext ctx) {
  // CHECK-LABEL: @testMove
  fprintf(stderr, "@testMove\n");

  const char *moduleString = "\"dialect.op1\"() : () -> ()\n"
                             "\"dialect.op2\"() ({\n"
                             "^bb0(%arg0: index):\n"
                             "  \"dialect.op2_1\"(%arg0) : (index) -> ()"
                             "^bb1(%arg1: index):\n"
                             "  \"dialect.op2_2\"(%arg1) : (index) -> ()"
                             "}) : () -> ()\n"
                             "\"dialect.op3\"() : () -> ()\n"
                             "\"dialect.op4\"() : () -> ()\n";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  // get a handle to all operations/values
  MlirOperation op1 = mlirBlockGetFirstOperation(body);
  MlirOperation op2 = mlirOperationGetNextInBlock(op1);
  MlirOperation op3 = mlirOperationGetNextInBlock(op2);
  MlirOperation op4 = mlirOperationGetNextInBlock(op3);

  MlirRegion region2 = mlirOperationGetRegion(op2, 0);
  MlirBlock block0 = mlirRegionGetFirstBlock(region2);
  MlirBlock block1 = mlirBlockGetNextInRegion(block0);

  // Test move operations.
  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);
  mlirRewriterBaseMoveOpBefore(rewriter, op3, op1);
  mlirRewriterBaseMoveOpAfter(rewriter, op4, op1);
  mlirRewriterBaseMoveBlockBefore(rewriter, block1, block0);

  mlirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT:   "dialect.op3"() : () -> ()
  // CHECK-NEXT:   "dialect.op1"() : () -> ()
  // CHECK-NEXT:   "dialect.op4"() : () -> ()
  // CHECK-NEXT:   "dialect.op2"() ({
  // CHECK-NEXT:   ^{{.*}}(%[[arg0:.*]]: index):
  // CHECK-NEXT:     "dialect.op2_2"(%[[arg0]]) : (index) -> ()
  // CHECK-NEXT:   ^{{.*}}(%[[arg1:.*]]: index):  // no predecessors
  // CHECK-NEXT:     "dialect.op2_1"(%[[arg1]]) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT: }

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testOpModification(MlirContext ctx) {
  // CHECK-LABEL: @testOpModification
  fprintf(stderr, "@testOpModification\n");

  const char *moduleString =
      "%x, %y = \"dialect.op1\"() : () -> (index, index)\n"
      "\"dialect.op2\"(%x) : (index) -> ()\n";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  // get a handle to all operations/values
  MlirOperation op1 = mlirBlockGetFirstOperation(body);
  MlirValue y = mlirOperationGetResult(op1, 1);
  MlirOperation op2 = mlirOperationGetNextInBlock(op1);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);
  mlirRewriterBaseStartOpModification(rewriter, op1);
  mlirRewriterBaseCancelOpModification(rewriter, op1);

  mlirRewriterBaseStartOpModification(rewriter, op2);
  mlirOperationSetOperand(op2, 0, y);
  mlirRewriterBaseFinalizeOpModification(rewriter, op2);

  mlirOperationDump(op);
  // CHECK-NEXT: module {
  // CHECK-NEXT: %[[xy:.*]]:2 = "dialect.op1"() : () -> (index, index)
  // CHECK-NEXT: "dialect.op2"(%[[xy]]#1) : (index) -> ()
  // CHECK-NEXT: }

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testReplaceUses(MlirContext ctx) {
  // CHECK-LABEL: @testReplaceUses
  fprintf(stderr, "@testReplaceUses\n");

  const char *moduleString =
      // Replace values with values
      "%x1, %y1, %z1 = \"dialect.op1\"() : () -> (index, index, index)\n"
      "%x2, %y2, %z2 = \"dialect.op2\"() : () -> (index, index, index)\n"
      "\"dialect.op1_uses\"(%x1, %y1, %z1) : (index, index, index) -> ()\n"
      // Replace op with values
      "%x3 = \"dialect.op3\"() : () -> index\n"
      "%x4 = \"dialect.op4\"() : () -> index\n"
      "\"dialect.op3_uses\"(%x3) : (index) -> ()\n"
      // Replace op with op
      "%x5 = \"dialect.op5\"() : () -> index\n"
      "%x6 = \"dialect.op6\"() : () -> index\n"
      "\"dialect.op5_uses\"(%x5) : (index) -> ()\n"
      // Replace op in block;
      "%x7 = \"dialect.op7\"() : () -> index\n"
      "%x8 = \"dialect.op8\"() : () -> index\n"
      "\"dialect.op9\"() ({\n"
      "^bb0:\n"
      "   \"dialect.op7_uses\"(%x7) : (index) -> ()\n"
      "}): () -> ()\n"
      "\"dialect.op7_uses\"(%x7) : (index) -> ()\n"
      // Replace value with value except in op
      "%x10 = \"dialect.op10\"() : () -> index\n"
      "%x11 = \"dialect.op11\"() : () -> index\n"
      "\"dialect.op10_uses\"(%x10) : (index) -> ()\n"
      "\"dialect.op10_uses\"(%x10) : (index) -> ()\n";

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation op = mlirModuleGetOperation(module);
  MlirBlock body = mlirModuleGetBody(module);

  // get a handle to all operations/values
  MlirOperation op1 = mlirBlockGetFirstOperation(body);
  MlirValue x1 = mlirOperationGetResult(op1, 0);
  MlirValue y1 = mlirOperationGetResult(op1, 1);
  MlirValue z1 = mlirOperationGetResult(op1, 2);
  MlirOperation op2 = mlirOperationGetNextInBlock(op1);
  MlirValue x2 = mlirOperationGetResult(op2, 0);
  MlirValue y2 = mlirOperationGetResult(op2, 1);
  MlirValue z2 = mlirOperationGetResult(op2, 2);
  MlirOperation op1Uses = mlirOperationGetNextInBlock(op2);

  MlirOperation op3 = mlirOperationGetNextInBlock(op1Uses);
  MlirOperation op4 = mlirOperationGetNextInBlock(op3);
  MlirValue x4 = mlirOperationGetResult(op4, 0);
  MlirOperation op3Uses = mlirOperationGetNextInBlock(op4);

  MlirOperation op5 = mlirOperationGetNextInBlock(op3Uses);
  MlirOperation op6 = mlirOperationGetNextInBlock(op5);
  MlirOperation op5Uses = mlirOperationGetNextInBlock(op6);

  MlirOperation op7 = mlirOperationGetNextInBlock(op5Uses);
  MlirOperation op8 = mlirOperationGetNextInBlock(op7);
  MlirValue x8 = mlirOperationGetResult(op8, 0);
  MlirOperation op9 = mlirOperationGetNextInBlock(op8);
  MlirRegion region9 = mlirOperationGetRegion(op9, 0);
  MlirBlock block9 = mlirRegionGetFirstBlock(region9);
  MlirOperation op7Uses = mlirOperationGetNextInBlock(op9);

  MlirOperation op10 = mlirOperationGetNextInBlock(op7Uses);
  MlirValue x10 = mlirOperationGetResult(op10, 0);
  MlirOperation op11 = mlirOperationGetNextInBlock(op10);
  MlirValue x11 = mlirOperationGetResult(op11, 0);
  MlirOperation op10Uses1 = mlirOperationGetNextInBlock(op11);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);

  // Replace values
  mlirRewriterBaseReplaceAllUsesWith(rewriter, x1, x2);
  MlirValue y1z1[2] = {y1, z1};
  MlirValue y2z2[2] = {y2, z2};
  mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, 2, y1z1, y2z2);

  // Replace op with values
  mlirRewriterBaseReplaceOpWithValues(rewriter, op3, 1, &x4);

  // Replace op with op
  mlirRewriterBaseReplaceOpWithOperation(rewriter, op5, op6);

  // Replace op with op in block
  mlirRewriterBaseReplaceOpUsesWithinBlock(rewriter, op7, 1, &x8, block9);

  // Replace value with value except in op
  mlirRewriterBaseReplaceAllUsesExcept(rewriter, x10, x11, op10Uses1);

  mlirOperationDump(op);
  // clang-format off
  // CHECK-NEXT: module {
  // CHECK-NEXT:   %{{.*}}:3 = "dialect.op1"() : () -> (index, index, index)
  // CHECK-NEXT:   %[[res2:.*]]:3 = "dialect.op2"() : () -> (index, index, index)
  // CHECK-NEXT:   "dialect.op1_uses"(%[[res2]]#0, %[[res2]]#1, %[[res2]]#2) : (index, index, index) -> ()
  // CHECK-NEXT:   %[[res4:.*]] = "dialect.op4"() : () -> index
  // CHECK-NEXT:   "dialect.op3_uses"(%[[res4]]) : (index) -> ()
  // CHECK-NEXT:   %[[res6:.*]] = "dialect.op6"() : () -> index
  // CHECK-NEXT:   "dialect.op5_uses"(%[[res6]]) : (index) -> ()
  // CHECK-NEXT:   %[[res7:.*]] = "dialect.op7"() : () -> index
  // CHECK-NEXT:   %[[res8:.*]] = "dialect.op8"() : () -> index
  // CHECK-NEXT:   "dialect.op9"() ({
  // CHECK-NEXT:     "dialect.op7_uses"(%[[res8]]) : (index) -> ()
  // CHECK-NEXT:   }) : () -> ()
  // CHECK-NEXT:   "dialect.op7_uses"(%[[res7]]) : (index) -> ()
  // CHECK-NEXT:   %[[res10:.*]] = "dialect.op10"() : () -> index
  // CHECK-NEXT:   %[[res11:.*]] = "dialect.op11"() : () -> index
  // CHECK-NEXT:   "dialect.op10_uses"(%[[res10]]) : (index) -> ()
  // CHECK-NEXT:   "dialect.op10_uses"(%[[res11]]) : (index) -> ()
  // CHECK-NEXT: }
  // clang-format on

  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);
}

void testGreedyRewriteDriverConfig(MlirContext ctx) {
  // CHECK-LABEL: @testGreedyRewriteDriverConfig
  fprintf(stderr, "@testGreedyRewriteDriverConfig\n");

  // Test config creation and destruction
  MlirGreedyRewriteDriverConfig config = mlirGreedyRewriteDriverConfigCreate();

  // Test all configuration setters
  mlirGreedyRewriteDriverConfigSetMaxIterations(config, 5);
  mlirGreedyRewriteDriverConfigSetMaxNumRewrites(config, 100);
  mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(config, true);
  mlirGreedyRewriteDriverConfigEnableFolding(config, false);
  mlirGreedyRewriteDriverConfigSetStrictness(
      config, MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS);
  mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
      config, MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL);
  mlirGreedyRewriteDriverConfigEnableConstantCSE(config, false);

  // Test all configuration getters and verify values
  // CHECK: MaxIterations: 5
  fprintf(stderr, "MaxIterations: %" PRId64 "\n",
          mlirGreedyRewriteDriverConfigGetMaxIterations(config));
  // CHECK: MaxNumRewrites: 100
  fprintf(stderr, "MaxNumRewrites: %" PRId64 "\n",
          mlirGreedyRewriteDriverConfigGetMaxNumRewrites(config));
  // CHECK: UseTopDownTraversal: 1
  fprintf(stderr, "UseTopDownTraversal: %d\n",
          mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(config));
  // CHECK: FoldingEnabled: 0
  fprintf(stderr, "FoldingEnabled: %d\n",
          mlirGreedyRewriteDriverConfigIsFoldingEnabled(config));
  // CHECK: Strictness: 2
  fprintf(stderr, "Strictness: %d\n",
          mlirGreedyRewriteDriverConfigGetStrictness(config));
  // CHECK: RegionSimplificationLevel: 1
  fprintf(stderr, "RegionSimplificationLevel: %d\n",
          mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(config));
  // CHECK: ConstantCSEEnabled: 0
  fprintf(stderr, "ConstantCSEEnabled: %d\n",
          mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(config));

  // CHECK: Config test completed successfully
  fprintf(stderr, "Config test completed successfully\n");
  mlirGreedyRewriteDriverConfigDestroy(config);
}

void testCloneWithMapping(MlirContext ctx) {
  // CHECK-LABEL: @testCloneWithMapping
  fprintf(stderr, "@testCloneWithMapping\n");

  const char *moduleString =
      "%x, %y = \"dialect.create_values\"() : () -> (index, index)\n"
      "%sum = \"dialect.add\"(%x, %y) : (index, index) -> index\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirBlock body = mlirModuleGetBody(module);

  MlirOperation createValues = mlirBlockGetFirstOperation(body);
  MlirValue x = mlirOperationGetResult(createValues, 0);
  MlirValue y = mlirOperationGetResult(createValues, 1);
  MlirOperation addOp = mlirOperationGetNextInBlock(createValues);

  MlirRewriterBase rewriter = mlirIRRewriterCreate(ctx);
  mlirRewriterBaseSetInsertionPointAfter(rewriter, addOp);

  // Clone addOp with a mapping that swaps x -> y, y -> x
  MlirIRMapping mapping = mlirIRMappingCreate();
  mlirIRMappingMapValue(mapping, x, y);
  mlirIRMappingMapValue(mapping, y, x);

  MlirOperation cloned =
      mlirRewriterBaseCloneWithMapping(rewriter, addOp, mapping);
  assert(!mlirOperationIsNull(cloned));

  // Verify operands are remapped
  MlirValue clonedOp0 = mlirOperationGetOperand(cloned, 0);
  MlirValue clonedOp1 = mlirOperationGetOperand(cloned, 1);
  assert(mlirValueEqual(clonedOp0, y));
  assert(mlirValueEqual(clonedOp1, x));

  mlirIRMappingDestroy(mapping);
  mlirIRRewriterDestroy(rewriter);
  mlirModuleDestroy(module);

  // CHECK: testCloneWithMapping: PASSED
  fprintf(stderr, "testCloneWithMapping: PASSED\n");
}

static MlirConversionTargetLegality dynamicLegalityAlwaysLegal(MlirOperation op,
                                                               void *userData) {
  (void)op;
  intptr_t *counter = (intptr_t *)userData;
  (*counter)++;
  return MLIR_CONVERSION_TARGET_LEGALITY_LEGAL;
}

static MlirConversionTargetLegality
dynamicLegalityAlwaysIllegal(MlirOperation op, void *userData) {
  (void)op;
  intptr_t *counter = (intptr_t *)userData;
  (*counter)++;
  return MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
}

static MlirConversionTargetLegality dynamicLegalityNoOpinion(MlirOperation op,
                                                             void *userData) {
  (void)op;
  intptr_t *counter = (intptr_t *)userData;
  (*counter)++;
  return MLIR_CONVERSION_TARGET_LEGALITY_NO_OPINION;
}

// Runs a partial conversion of `moduleString` against `target` with an empty
// pattern set and returns whether it succeeded. This is what actually drives
// the registered dynamic-legality callbacks.
static bool runPartialConversion(MlirContext ctx, const char *moduleString,
                                 MlirConversionTarget target) {
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  assert(!mlirModuleIsNull(module) && "expected module to parse");
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);
  MlirConversionConfig config = mlirConversionConfigCreate();

  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);

  mlirConversionConfigDestroy(config);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirModuleDestroy(module);

  return mlirLogicalResultIsSuccess(result);
}

void testConversionTargetDynamicLegality(MlirContext ctx) {
  // CHECK-LABEL: @testConversionTargetDynamicLegality
  fprintf(stderr, "@testConversionTargetDynamicLegality\n");

  const char *opModule = "\"dialect.op1\"() : () -> ()\n";

  // addDynamicallyLegalOp: callback returning true makes the op legal, so the
  // (pattern-free) partial conversion succeeds and the callback is invoked.
  {
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t counter = 0;
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.op1"),
        dynamicLegalityAlwaysLegal, &counter);
    assert(runPartialConversion(ctx, opModule, target));
    assert(counter > 0 && "legality callback must be invoked");
    mlirConversionTargetDestroy(target);
  }

  // addDynamicallyLegalOp: callback returning false makes the op illegal. With
  // no pattern to legalize it, the partial conversion fails -- proving the
  // callback's return value actually drives the result.
  {
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t counter = 0;
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.op1"),
        dynamicLegalityAlwaysIllegal, &counter);
    assert(!runPartialConversion(ctx, opModule, target));
    assert(counter > 0 && "legality callback must be invoked");
    mlirConversionTargetDestroy(target);
  }

  // addDynamicallyLegalOp composition: callbacks registered for the same op are
  // chained, most-recent first. A callback returning NoOpinion abstains and
  // defers to the previously-registered callback. Here the first callback marks
  // the op illegal and the second abstains, so the op stays illegal (conversion
  // fails) and BOTH callbacks are invoked.
  {
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t illegalCounter = 0;
    intptr_t noOpinionCounter = 0;
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.op1"),
        dynamicLegalityAlwaysIllegal, &illegalCounter);
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.op1"),
        dynamicLegalityNoOpinion, &noOpinionCounter);
    assert(!runPartialConversion(ctx, opModule, target));
    assert(noOpinionCounter > 0 && "abstaining callback must be invoked");
    assert(illegalCounter > 0 && "deferred-to callback must be invoked");
    mlirConversionTargetDestroy(target);
  }

  // addDynamicallyLegalDialect: the callback applies to every op in the
  // dialect. Returning true keeps `dialect.op1` legal -> success.
  {
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t counter = 0;
    mlirConversionTargetAddDynamicallyLegalDialect(
        target, mlirStringRefCreateFromCString("dialect"),
        dynamicLegalityAlwaysLegal, &counter);
    assert(runPartialConversion(ctx, opModule, target));
    assert(counter > 0 && "dialect legality callback must be invoked");
    mlirConversionTargetDestroy(target);
  }

  // markUnknownOpDynamicallyLegal: `dialect.op1` is unregistered and otherwise
  // unmarked, so the unknown-op callback decides its legality.
  {
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t counter = 0;
    mlirConversionTargetMarkUnknownOpDynamicallyLegal(
        target, dynamicLegalityAlwaysLegal, &counter);
    assert(runPartialConversion(ctx, opModule, target));
    assert(counter > 0 && "unknown-op legality callback must be invoked");
    mlirConversionTargetDestroy(target);
  }

  // markOpRecursivelyLegal: an op marked recursively legal short-circuits the
  // walk so nested ops are never checked. Here `dialect.inner` is illegal, but
  // because `dialect.outer` is recursively legal the conversion still succeeds
  // and the inner op's (illegal) callback is never invoked.
  {
    const char *nestedModule = "\"dialect.outer\"() ({\n"
                               "  \"dialect.inner\"() : () -> ()\n"
                               "}) : () -> ()\n";
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t innerCounter = 0;
    intptr_t recursiveCounter = 0;
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.inner"),
        dynamicLegalityAlwaysIllegal, &innerCounter);
    mlirConversionTargetAddLegalOp(
        target, mlirStringRefCreateFromCString("dialect.outer"));
    mlirConversionTargetMarkOpRecursivelyLegal(
        target, mlirStringRefCreateFromCString("dialect.outer"),
        dynamicLegalityAlwaysLegal, &recursiveCounter);
    assert(runPartialConversion(ctx, nestedModule, target));
    assert(recursiveCounter > 0 && "recursive legality callback must run");
    assert(innerCounter == 0 &&
           "nested op must not be visited under recursive legality");
    mlirConversionTargetDestroy(target);
  }

  // markOpRecursivelyLegal with a NULL callback: the op is unconditionally
  // recursively legal (no per-instance check), so the nested illegal op is
  // still skipped and the conversion succeeds.
  {
    const char *nestedModule = "\"dialect.outer\"() ({\n"
                               "  \"dialect.inner\"() : () -> ()\n"
                               "}) : () -> ()\n";
    MlirConversionTarget target = mlirConversionTargetCreate(ctx);
    intptr_t innerCounter = 0;
    mlirConversionTargetAddDynamicallyLegalOp(
        target, mlirStringRefCreateFromCString("dialect.inner"),
        dynamicLegalityAlwaysIllegal, &innerCounter);
    mlirConversionTargetAddLegalOp(
        target, mlirStringRefCreateFromCString("dialect.outer"));
    mlirConversionTargetMarkOpRecursivelyLegal(
        target, mlirStringRefCreateFromCString("dialect.outer"), NULL, NULL);
    assert(runPartialConversion(ctx, nestedModule, target));
    assert(innerCounter == 0 &&
           "nested op must not be visited under recursive legality");
    mlirConversionTargetDestroy(target);
  }

  // CHECK: testConversionTargetDynamicLegality: PASSED
  fprintf(stderr, "testConversionTargetDynamicLegality: PASSED\n");
}

int main(void) {
  MlirContext ctx = mlirContextCreate();
  mlirContextSetAllowUnregisteredDialects(ctx, true);
  mlirContextGetOrLoadDialect(ctx, mlirStringRefCreateFromCString("builtin"));

  testInsertionPoint(ctx);
  testCreateBlock(ctx);
  testInlineRegionBlock(ctx);
  testReplaceOp(ctx);
  testErase(ctx);
  testMove(ctx);
  testOpModification(ctx);
  testReplaceUses(ctx);
  testGreedyRewriteDriverConfig(ctx);
  testCloneWithMapping(ctx);
  testConversionTargetDynamicLegality(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
