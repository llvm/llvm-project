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
  ++(*counter);
  return MLIR_CONVERSION_TARGET_LEGALITY_LEGAL;
}

static MlirConversionTargetLegality
dynamicLegalityAlwaysIllegal(MlirOperation op, void *userData) {
  (void)op;
  intptr_t *counter = (intptr_t *)userData;
  ++(*counter);
  return MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
}

static MlirConversionTargetLegality dynamicLegalityNoOpinion(MlirOperation op,
                                                             void *userData) {
  (void)op;
  intptr_t *counter = (intptr_t *)userData;
  ++(*counter);
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

// Type conversion callback: maps i32 -> i64 and leaves every other type
// unchanged (identity). Used by the materialization tests below.
static MlirTypeConverterConversionStatus
widenI32ToI64(MlirType type, MlirType *result, void *userData) {
  (void)userData;
  if (mlirTypeIsAInteger(type) && mlirIntegerTypeGetWidth(type) == 32)
    *result = mlirIntegerTypeGet(mlirTypeGetContext(type), 64);
  else
    *result = type;
  return MlirTypeConverterConversionStatusSuccess;
}

// Type conversion callback that declines i32 (returns Declined) and is the
// identity on every other type. A declining callback lets the converter fall
// through to another registered conversion function.
static MlirTypeConverterConversionStatus
declineI32(MlirType type, MlirType *result, void *userData) {
  (void)userData;
  if (mlirTypeIsAInteger(type) && mlirIntegerTypeGetWidth(type) == 32)
    return MlirTypeConverterConversionStatusDeclined;
  *result = type;
  return MlirTypeConverterConversionStatusSuccess;
}

// Type conversion callback that fails on i32 (returns Failure) and is the
// identity on every other type. A failure aborts the conversion without
// trying any earlier-registered conversion function.
static MlirTypeConverterConversionStatus
failI32(MlirType type, MlirType *result, void *userData) {
  (void)userData;
  if (mlirTypeIsAInteger(type) && mlirIntegerTypeGetWidth(type) == 32)
    return MlirTypeConverterConversionStatusFailure;
  *result = type;
  return MlirTypeConverterConversionStatusSuccess;
}

static MlirValue buildCastMaterialization(MlirRewriterBase rewriter,
                                          MlirType outputType, intptr_t nInputs,
                                          MlirValue *inputs, MlirLocation loc,
                                          void *userData) {
  intptr_t *counter = (intptr_t *)userData;
  if (counter)
    ++(*counter);
  MlirOperationState state =
      mlirOperationStateGet(mlirStringRefCreateFromCString("test.cast"), loc);
  mlirOperationStateAddOperands(&state, nInputs, inputs);
  mlirOperationStateAddResults(&state, 1, &outputType);
  MlirOperation castOp = mlirOperationCreate(&state);
  mlirRewriterBaseInsert(rewriter, castOp);
  return mlirOperationGetResult(castOp, 0);
}

// Source materialization callback that always declines (returns a null value)
// and records that it was consulted. Used to exercise the "this materialization
// declined, try the next one" fallback path.
static MlirValue
declineSourceMaterialization(MlirRewriterBase rewriter, MlirType outputType,
                             intptr_t nInputs, MlirValue *inputs,
                             MlirLocation loc, void *userData) {
  (void)rewriter;
  (void)outputType;
  (void)nInputs;
  (void)inputs;
  (void)loc;
  intptr_t *declined = (intptr_t *)userData;
  if (declined)
    ++(*declined);
  return (MlirValue){NULL};
}

// Source materialization callback that records the number of inputs it was
// invoked with (into userData) before building the `test.cast`. Used to verify
// that a 1:N replacement drives a source materialization with nInputs > 1.
static MlirValue buildSourceCastRecordInputs(MlirRewriterBase rewriter,
                                             MlirType outputType,
                                             intptr_t nInputs,
                                             MlirValue *inputs,
                                             MlirLocation loc, void *userData) {
  intptr_t *observedInputs = (intptr_t *)userData;
  if (observedInputs)
    *observedInputs = nInputs;
  MlirOperationState state =
      mlirOperationStateGet(mlirStringRefCreateFromCString("test.cast"), loc);
  mlirOperationStateAddOperands(&state, nInputs, inputs);
  mlirOperationStateAddResults(&state, 1, &outputType);
  MlirOperation castOp = mlirOperationCreate(&state);
  mlirRewriterBaseInsert(rewriter, castOp);
  return mlirOperationGetResult(castOp, 0);
}

// 1:1 target materialization callback. Builds a `test.cast` like
// buildCastMaterialization, but additionally inspects `originalType`: the
// counter passed as userData is only bumped when `originalType` is the expected
// original (i32) type, so a passing test proves the original type is observable
// from C.
static MlirValue buildTargetCast(MlirRewriterBase rewriter, MlirType outputType,
                                 intptr_t nInputs, MlirValue *inputs,
                                 MlirLocation loc, MlirType originalType,
                                 void *userData) {
  intptr_t *counter = (intptr_t *)userData;
  if (counter && mlirTypeIsAInteger(originalType) &&
      mlirIntegerTypeGetWidth(originalType) == 32)
    ++(*counter);
  MlirOperationState state =
      mlirOperationStateGet(mlirStringRefCreateFromCString("test.cast"), loc);
  mlirOperationStateAddOperands(&state, nInputs, inputs);
  mlirOperationStateAddResults(&state, 1, &outputType);
  MlirOperation castOp = mlirOperationCreate(&state);
  mlirRewriterBaseInsert(rewriter, castOp);
  return mlirOperationGetResult(castOp, 0);
}

// 1:N type conversion callback: maps i32 -> (i16, i16) and leaves every other
// type unchanged (1:1 identity).
static MlirTypeConverterConversionStatus
splitI32(MlirType type, MlirTypeConverterConversionResults results,
         void *userData) {
  (void)userData;
  if (mlirTypeIsAInteger(type) && mlirIntegerTypeGetWidth(type) == 32) {
    MlirType i16 = mlirIntegerTypeGet(mlirTypeGetContext(type), 16);
    mlirTypeConverterConversionResultsAppend(results, i16);
    mlirTypeConverterConversionResultsAppend(results, i16);
  } else {
    mlirTypeConverterConversionResultsAppend(results, type);
  }
  return MlirTypeConverterConversionStatusSuccess;
}

// 1:N type conversion callback: erases i32 (converts it to zero types) and
// leaves every other type unchanged (1:1 identity).
static MlirTypeConverterConversionStatus
eraseI32(MlirType type, MlirTypeConverterConversionResults results,
         void *userData) {
  (void)userData;
  if (mlirTypeIsAInteger(type) && mlirIntegerTypeGetWidth(type) == 32)
    return MlirTypeConverterConversionStatusSuccess; // append nothing -> erase
  mlirTypeConverterConversionResultsAppend(results, type);
  return MlirTypeConverterConversionStatusSuccess;
}

// 1:N type conversion callback: builds one `test.cast` per requested
// output type, each consuming the (single) input, and fills `outputs`
// accordingly. Bumps the counter passed as userData.
static MlirLogicalResult buildSplitCast(MlirRewriterBase rewriter,
                                        intptr_t nOutputTypes,
                                        MlirType *outputTypes, intptr_t nInputs,
                                        MlirValue *inputs, MlirLocation loc,
                                        MlirType originalType,
                                        MlirValue *outputs, void *userData) {
  intptr_t *counter = (intptr_t *)userData;
  // The original type must be the i32 operand type; it cannot be recovered from
  // the (i16) output types alone, so observing it here proves it is propagated.
  if (counter && mlirTypeIsAInteger(originalType) &&
      mlirIntegerTypeGetWidth(originalType) == 32)
    ++(*counter);
  for (intptr_t i = 0; i < nOutputTypes; ++i) {
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString("test.cast"), loc);
    mlirOperationStateAddOperands(&state, nInputs, inputs);
    mlirOperationStateAddResults(&state, 1, &outputTypes[i]);
    MlirOperation castOp = mlirOperationCreate(&state);
    mlirRewriterBaseInsert(rewriter, castOp);
    outputs[i] = mlirOperationGetResult(castOp, 0);
  }
  return mlirLogicalResultSuccess();
}

// 1:N type conversion callback that appends bogus result types and then
// declines (returns Declined). The binding must roll back the appended types so
// they do not pollute a subsequently-tried conversion function.
static MlirTypeConverterConversionStatus appendThenDeclineConversion(
    MlirType type, MlirTypeConverterConversionResults results, void *userData) {
  intptr_t *counter = (intptr_t *)userData;
  if (counter)
    ++(*counter);
  MlirType i8 = mlirIntegerTypeGet(mlirTypeGetContext(type), 8);
  // Append more (and differently-typed) entries than the real conversion would,
  // so any leak is observable as a wrong type/arity downstream.
  mlirTypeConverterConversionResultsAppend(results, i8);
  mlirTypeConverterConversionResultsAppend(results, i8);
  mlirTypeConverterConversionResultsAppend(results, i8);
  return MlirTypeConverterConversionStatusDeclined;
}

// 1:N type conversion callback that appends bogus result types and then
// fails (returns Failure). Unlike a decline, this must abort the conversion
// without trying any earlier-registered conversion function; the binding must
// still roll back the appended types.
static MlirTypeConverterConversionStatus appendThenFailConversion(
    MlirType type, MlirTypeConverterConversionResults results, void *userData) {
  intptr_t *counter = (intptr_t *)userData;
  if (counter)
    ++(*counter);
  MlirType i8 = mlirIntegerTypeGet(mlirTypeGetContext(type), 8);
  mlirTypeConverterConversionResultsAppend(results, i8);
  mlirTypeConverterConversionResultsAppend(results, i8);
  mlirTypeConverterConversionResultsAppend(results, i8);
  return MlirTypeConverterConversionStatusFailure;
}

// 1:N target materialization that always declines (returns failure) and records
// that it was consulted. Exercises the "decline, try the next" fallback.
static MlirLogicalResult declineTargetMaterialization(
    MlirRewriterBase rewriter, intptr_t nOutputTypes, MlirType *outputTypes,
    intptr_t nInputs, MlirValue *inputs, MlirLocation loc,
    MlirType originalType, MlirValue *outputs, void *userData) {
  (void)rewriter;
  (void)nOutputTypes;
  (void)outputTypes;
  (void)nInputs;
  (void)inputs;
  (void)loc;
  (void)originalType;
  (void)outputs;
  intptr_t *counter = (intptr_t *)userData;
  if (counter)
    ++(*counter);
  return mlirLogicalResultFailure();
}

// Conversion pattern for `test.source`: replaces it with a `test.source_i64`
// op whose result has the widened (i64) type. Because the original result type
// (i32) differs from the replacement type (i64), persisting uses force the
// framework to insert a source materialization.
static MlirLogicalResult convertSource(MlirConversionPattern pattern,
                                       MlirOperation op, intptr_t nOperands,
                                       MlirValue *operands,
                                       MlirConversionPatternRewriter rewriter,
                                       void *userData) {
  (void)pattern;
  (void)nOperands;
  (void)operands;
  (void)userData;
  MlirContext ctx = mlirOperationGetContext(op);
  MlirLocation loc = mlirOperationGetLocation(op);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("test.source_i64"), loc);
  mlirOperationStateAddResults(&state, 1, &i64);
  MlirOperation newOp = mlirOperationCreate(&state);

  MlirRewriterBase base = mlirPatternRewriterAsBase(
      mlirConversionPatternRewriterAsPatternRewriter(rewriter));
  mlirRewriterBaseInsert(base, newOp);
  MlirValue newVal = mlirOperationGetResult(newOp, 0);
  mlirRewriterBaseReplaceOpWithValues(base, op, 1, &newVal);
  return mlirLogicalResultSuccess();
}

void testTypeConverterSourceMaterialization(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverterSourceMaterialization
  fprintf(stderr, "@testTypeConverterSourceMaterialization\n");

  // `test.source` produces an i32 that is consumed by the (legal) `test.user`.
  // Converting `test.source` to an i64-producing op leaves `test.user` wanting
  // the original i32, which triggers a source materialization back to i32.
  const char *moduleString = "%0 = \"test.source\"() : () -> i32\n"
                             "\"test.user\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAddConversion(converter, widenI32ToI64, NULL);
  intptr_t materializationCounter = 0;
  mlirTypeConverterAddSourceMaterialization(converter, buildCastMaterialization,
                                            &materializationCounter);
  // Register a second materialization that always declines. Because
  // materializations are tried most-recently-added first, this one runs first,
  // returns null, and the framework must fall back to buildCastMaterialization.
  intptr_t declinedCounter = 0;
  mlirTypeConverterAddSourceMaterialization(
      converter, declineSourceMaterialization, &declinedCounter);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, convertSource, NULL};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.source"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.source"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.source_i64"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.user"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(materializationCounter > 0 &&
         "source materialization callback must be invoked");
  assert(declinedCounter > 0 &&
         "declining materialization must be consulted before the fallback");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK:      module {
  // CHECK-NEXT:   %[[v:.*]] = "test.source_i64"() : () -> i64
  // CHECK-NEXT:   %[[c:.*]] = "test.cast"(%[[v]]) : (i64) -> i32
  // CHECK-NEXT:   "test.user"(%[[c]]) : (i32) -> ()
  // CHECK-NEXT: }
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverterSourceMaterialization: PASSED
  fprintf(stderr, "testTypeConverterSourceMaterialization: PASSED\n");
}

// Conversion pattern for `test.consumer`: replaces it with a
// `test.consumer_legal` op that consumes the (already remapped) operands. The
// operand of the original op has type i32 but its producer is not converted, so
// the framework inserts a target materialization to i64 before invoking this
// pattern -- the remapped `operands` are therefore the i64 cast results.
static MlirLogicalResult convertConsumer(MlirConversionPattern pattern,
                                         MlirOperation op, intptr_t nOperands,
                                         MlirValue *operands,
                                         MlirConversionPatternRewriter rewriter,
                                         void *userData) {
  (void)pattern;
  (void)userData;
  MlirLocation loc = mlirOperationGetLocation(op);
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("test.consumer_legal"), loc);
  mlirOperationStateAddOperands(&state, nOperands, operands);
  MlirOperation newOp = mlirOperationCreate(&state);

  MlirRewriterBase base = mlirPatternRewriterAsBase(
      mlirConversionPatternRewriterAsPatternRewriter(rewriter));
  mlirRewriterBaseInsert(base, newOp);
  mlirRewriterBaseEraseOp(base, op);
  return mlirLogicalResultSuccess();
}

// 1:N conversion pattern for `test.consumer`: builds `test.consumer_legal` from
// the flattened remapped operands. Under the i32 -> (i16, i16) conversion the
// single operand is remapped to two values, so this is invoked via the 1:N
// matchAndRewrite hook with nOperands == 2.
static MlirLogicalResult
convertConsumer1ToN(MlirConversionPattern pattern, MlirOperation op,
                    intptr_t nRanges, intptr_t *rangeSizes, intptr_t nOperands,
                    MlirValue *operands, MlirConversionPatternRewriter rewriter,
                    void *userData) {
  (void)pattern;
  (void)nRanges;
  (void)rangeSizes;
  (void)userData;
  MlirLocation loc = mlirOperationGetLocation(op);
  MlirOperationState state = mlirOperationStateGet(
      mlirStringRefCreateFromCString("test.consumer_legal"), loc);
  // Under an erasure conversion (i32 -> zero types) the operand range is empty;
  // avoid passing a null pointer with count 0 (UB caught by UBSan's memcpy
  // nonnull check).
  if (nOperands)
    mlirOperationStateAddOperands(&state, nOperands, operands);
  MlirOperation newOp = mlirOperationCreate(&state);

  MlirRewriterBase base = mlirPatternRewriterAsBase(
      mlirConversionPatternRewriterAsPatternRewriter(rewriter));
  mlirRewriterBaseInsert(base, newOp);
  mlirRewriterBaseEraseOp(base, op);
  return mlirLogicalResultSuccess();
}

void testTypeConverterTargetMaterialization(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverterTargetMaterialization
  fprintf(stderr, "@testTypeConverterTargetMaterialization\n");

  // `test.consumer` takes an i32 from the (legal, unconverted) `test.producer`.
  // Converting `test.consumer` requires its operand as i64, which triggers a
  // target materialization from i32 to i64.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAddConversion(converter, widenI32ToI64, NULL);
  intptr_t materializationCounter = 0;
  mlirTypeConverterAddTargetMaterialization(converter, buildTargetCast,
                                            &materializationCounter);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, convertConsumer,
                                              NULL};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(materializationCounter > 0 &&
         "target materialization callback must be invoked");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK:      module {
  // CHECK-NEXT:   %[[v:.*]] = "test.producer"() : () -> i32
  // CHECK-NEXT:   %[[c:.*]] = "test.cast"(%[[v]]) : (i32) -> i64
  // CHECK-NEXT:   "test.consumer_legal"(%[[c]]) : (i64) -> ()
  // CHECK-NEXT: }
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverterTargetMaterialization: PASSED
  fprintf(stderr, "testTypeConverterTargetMaterialization: PASSED\n");
}

// Unit test for the three MlirTypeConverterConversionStatus values, exercised
// directly through mlirTypeConverterConvertType (no conversion driver needed).
// Conversion functions are consulted most-recently-registered first, so the
// second-registered callback is tried before the first (`widenI32ToI64`).
void testTypeConverterConversionStatus(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverterConversionStatus
  fprintf(stderr, "@testTypeConverterConversionStatus\n");

  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);

  // Success: the sole conversion widens i32 -> i64.
  MlirTypeConverter success = mlirTypeConverterCreate();
  mlirTypeConverterAddConversion(success, widenI32ToI64, NULL);
  MlirType successResult = mlirTypeConverterConvertType(success, i32);
  assert(!mlirTypeIsNull(successResult) && mlirTypeEqual(successResult, i64) &&
         "Success must yield the converted type");
  mlirTypeConverterDestroy(success);

  // Declined: `declineI32` is tried first and declines, so the converter falls
  // through to `widenI32ToI64`, which converts i32 -> i64.
  MlirTypeConverter declined = mlirTypeConverterCreate();
  mlirTypeConverterAddConversion(declined, widenI32ToI64, NULL);
  mlirTypeConverterAddConversion(declined, declineI32, NULL);
  MlirType declinedResult = mlirTypeConverterConvertType(declined, i32);
  assert(!mlirTypeIsNull(declinedResult) &&
         mlirTypeEqual(declinedResult, i64) &&
         "Declined must fall through to the next conversion function");
  mlirTypeConverterDestroy(declined);

  // Failure: `failI32` is tried first and fails, which aborts the
  // conversion without consulting `widenI32ToI64`; convertType returns null.
  MlirTypeConverter failure = mlirTypeConverterCreate();
  mlirTypeConverterAddConversion(failure, widenI32ToI64, NULL);
  mlirTypeConverterAddConversion(failure, failI32, NULL);
  MlirType failureResult = mlirTypeConverterConvertType(failure, i32);
  assert(mlirTypeIsNull(failureResult) &&
         "Failure must abort without trying another conversion function");
  mlirTypeConverterDestroy(failure);

  // CHECK: testTypeConverterConversionStatus: PASSED
  fprintf(stderr, "testTypeConverterConversionStatus: PASSED\n");
}

void testTypeConverter1ToNTargetMaterialization(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNTargetMaterialization
  fprintf(stderr, "@testTypeConverter1ToNTargetMaterialization\n");

  // `test.consumer` takes an i32 from the (legal, unconverted) `test.producer`.
  // The 1:N conversion maps i32 -> (i16, i16), so converting `test.consumer`
  // requires its operand as two i16 values, which triggers a 1:N target
  // materialization producing two values from the single i32.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, splitI32, NULL);
  intptr_t materializationCounter = 0;
  mlirTypeConverterAdd1ToNTargetMaterialization(converter, buildSplitCast,
                                                &materializationCounter);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, NULL,
                                              convertConsumer1ToN};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(materializationCounter > 0 &&
         "1:N target materialization callback must be invoked");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[v:.*]] = "test.producer"() : () -> i32
  // CHECK: %[[c0:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: %[[c1:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: "test.consumer_legal"(%[[c0]], %[[c1]]) : (i16, i16) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNTargetMaterialization: PASSED
  fprintf(stderr, "testTypeConverter1ToNTargetMaterialization: PASSED\n");
}

void testTypeConverter1ToNConversionDeclineRollback(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNConversionDeclineRollback
  fprintf(stderr, "@testTypeConverter1ToNConversionDeclineRollback\n");

  // Same setup as the 1:N target materialization test, but with an extra
  // conversion that appends bogus types and then declines. It is registered
  // last, so it is tried first; the binding must roll back its appended types
  // so the i32 -> (i16, i16) conversion still produces exactly two i16 values
  // (not i8s, and not three of them).
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, splitI32, NULL);
  intptr_t declineCounter = 0;
  mlirTypeConverterAdd1ToNConversion(converter, appendThenDeclineConversion,
                                     &declineCounter);
  intptr_t materializationCounter = 0;
  mlirTypeConverterAdd1ToNTargetMaterialization(converter, buildSplitCast,
                                                &materializationCounter);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, NULL,
                                              convertConsumer1ToN};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(declineCounter > 0 && "declining conversion must be consulted");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[v:.*]] = "test.producer"() : () -> i32
  // CHECK: %[[c0:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: %[[c1:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: "test.consumer_legal"(%[[c0]], %[[c1]]) : (i16, i16) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNConversionDeclineRollback: PASSED
  fprintf(stderr, "testTypeConverter1ToNConversionDeclineRollback: PASSED\n");
}

void testTypeConverter1ToNConversionFailure(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNConversionFailure
  fprintf(stderr, "@testTypeConverter1ToNConversionFailure\n");

  // Same setup as the decline-rollback test, but the extra conversion returns a
  // failure instead of a decline. Because it is registered last (tried
  // first), the failure must abort the conversion of i32 without falling back
  // to the i32 -> (i16, i16) conversion, so the whole conversion fails and the
  // IR is left unchanged.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, splitI32, NULL);
  intptr_t failCounter = 0;
  mlirTypeConverterAdd1ToNConversion(converter, appendThenFailConversion,
                                     &failCounter);
  mlirTypeConverterAdd1ToNTargetMaterialization(converter, buildSplitCast,
                                                NULL);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, NULL,
                                              convertConsumer1ToN};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsFailure(result) &&
         "failure must abort the conversion");
  assert(failCounter > 0 && "failing conversion must be consulted");

  // The conversion failed, so the IR is rolled back and left unchanged: i32 is
  // never split and no `test.cast` is produced.
  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[v:.*]] = "test.producer"() : () -> i32
  // CHECK: "test.consumer"(%[[v]]) : (i32) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNConversionFailure: PASSED
  fprintf(stderr, "testTypeConverter1ToNConversionFailure: PASSED\n");
}

void testTypeConverter1ToNTargetMaterializationDecline(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNTargetMaterializationDecline
  fprintf(stderr, "@testTypeConverter1ToNTargetMaterializationDecline\n");

  // Register two 1:N target materializations. Tried most-recently-added
  // first: one that returns failure (declining), and finally the real one.
  // The conversion must still succeed via the last, and both must run.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, splitI32, NULL);
  intptr_t buildCounter = 0, declineCounter = 0;
  mlirTypeConverterAdd1ToNTargetMaterialization(converter, buildSplitCast,
                                                &buildCounter);
  mlirTypeConverterAdd1ToNTargetMaterialization(
      converter, declineTargetMaterialization, &declineCounter);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, NULL,
                                              convertConsumer1ToN};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(declineCounter > 0 && "failing materialization must be consulted");
  assert(buildCounter > 0 && "fallback materialization must run");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[v:.*]] = "test.producer"() : () -> i32
  // CHECK: %[[c0:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: %[[c1:.*]] = "test.cast"(%[[v]]) : (i32) -> i16
  // CHECK: "test.consumer_legal"(%[[c0]], %[[c1]]) : (i16, i16) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNTargetMaterializationDecline: PASSED
  fprintf(stderr,
          "testTypeConverter1ToNTargetMaterializationDecline: PASSED\n");
}

// Conversion pattern for `test.producer`: replaces its single i32 result with
// two i16 values (built by two `test.piece` ops) via replaceOpWithMultiple.
// Because a legal use (`test.user`) still wants the original i32, the framework
// must insert a source materialization with two inputs to reconcile the 1:N
// replacement.
static MlirLogicalResult
convertProducerToMultiple(MlirConversionPattern pattern, MlirOperation op,
                          intptr_t nOperands, MlirValue *operands,
                          MlirConversionPatternRewriter rewriter,
                          void *userData) {
  (void)pattern;
  (void)nOperands;
  (void)operands;
  (void)userData;
  MlirContext ctx = mlirOperationGetContext(op);
  MlirLocation loc = mlirOperationGetLocation(op);
  MlirType i16 = mlirIntegerTypeGet(ctx, 16);

  MlirRewriterBase base = mlirPatternRewriterAsBase(
      mlirConversionPatternRewriterAsPatternRewriter(rewriter));

  MlirValue pieces[2];
  for (int i = 0; i < 2; ++i) {
    MlirOperationState state = mlirOperationStateGet(
        mlirStringRefCreateFromCString("test.piece"), loc);
    mlirOperationStateAddResults(&state, 1, &i16);
    MlirOperation pieceOp = mlirOperationCreate(&state);
    mlirRewriterBaseInsert(base, pieceOp);
    pieces[i] = mlirOperationGetResult(pieceOp, 0);
  }

  // The op has a single result, so there is one range, carrying two values.
  intptr_t rangeSizes[1] = {2};
  mlirConversionPatternRewriterReplaceOpWithMultiple(rewriter, op, 1,
                                                     rangeSizes, pieces);
  return mlirLogicalResultSuccess();
}

void testTypeConverterMultiInputSourceMaterialization(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverterMultiInputSourceMaterialization
  fprintf(stderr, "@testTypeConverterMultiInputSourceMaterialization\n");

  // `test.producer` produces an i32 consumed by the (legal) `test.user`.
  // The pattern replaces the producer's result with two i16 values, so the
  // i32-wanting `test.user` forces a source materialization that receives both
  // values as inputs (nInputs == 2).
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.user\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  intptr_t observedInputs = 0;
  mlirTypeConverterAddSourceMaterialization(
      converter, buildSourceCastRecordInputs, &observedInputs);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL,
                                              convertProducerToMultiple, NULL};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.producer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.piece"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.cast"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.user"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));
  assert(observedInputs == 2 &&
         "source materialization must receive both replacement values");

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[a:.*]] = "test.piece"() : () -> i16
  // CHECK: %[[b:.*]] = "test.piece"() : () -> i16
  // CHECK: %[[c:.*]] = "test.cast"(%[[a]], %[[b]]) : (i16, i16) -> i32
  // CHECK: "test.user"(%[[c]]) : (i32) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverterMultiInputSourceMaterialization: PASSED
  fprintf(stderr, "testTypeConverterMultiInputSourceMaterialization: PASSED\n");
}

// Conversion pattern for a two-result `test.multi`: replaces it via
// replaceOpWithMultiple with two ranges (one per result), exercising the
// nRanges > 1 marshalling path. Each result is replaced 1:1 with a same-typed
// value so no materialization is needed.
static MlirLogicalResult
convertMultiResult(MlirConversionPattern pattern, MlirOperation op,
                   intptr_t nOperands, MlirValue *operands,
                   MlirConversionPatternRewriter rewriter, void *userData) {
  (void)pattern;
  (void)nOperands;
  (void)operands;
  (void)userData;
  MlirContext ctx = mlirOperationGetContext(op);
  MlirLocation loc = mlirOperationGetLocation(op);
  MlirType i32 = mlirIntegerTypeGet(ctx, 32);
  MlirType i64 = mlirIntegerTypeGet(ctx, 64);
  MlirRewriterBase base = mlirPatternRewriterAsBase(
      mlirConversionPatternRewriterAsPatternRewriter(rewriter));

  MlirValue vals[2];
  MlirType types[2] = {i32, i64};
  const char *names[2] = {"test.r0", "test.r1"};
  for (int i = 0; i < 2; ++i) {
    MlirOperationState state =
        mlirOperationStateGet(mlirStringRefCreateFromCString(names[i]), loc);
    mlirOperationStateAddResults(&state, 1, &types[i]);
    MlirOperation newOp = mlirOperationCreate(&state);
    mlirRewriterBaseInsert(base, newOp);
    vals[i] = mlirOperationGetResult(newOp, 0);
  }

  // Two ranges (one per result), each carrying a single value.
  intptr_t rangeSizes[2] = {1, 1};
  mlirConversionPatternRewriterReplaceOpWithMultiple(rewriter, op, 2,
                                                     rangeSizes, vals);
  return mlirLogicalResultSuccess();
}

void testConversionReplaceOpWithMultipleRanges(MlirContext ctx) {
  // CHECK-LABEL: @testConversionReplaceOpWithMultipleRanges
  fprintf(stderr, "@testConversionReplaceOpWithMultipleRanges\n");

  const char *moduleString = "%0:2 = \"test.multi\"() : () -> (i32, i64)\n"
                             "\"test.use\"(%0#0, %0#1) : (i32, i64) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, convertMultiResult,
                                              NULL};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.multi"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.multi"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.r0"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.r1"));
  mlirConversionTargetAddLegalOp(target,
                                 mlirStringRefCreateFromCString("test.use"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: %[[a:.*]] = "test.r0"() : () -> i32
  // CHECK: %[[b:.*]] = "test.r1"() : () -> i64
  // CHECK: "test.use"(%[[a]], %[[b]]) : (i32, i64) -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testConversionReplaceOpWithMultipleRanges: PASSED
  fprintf(stderr, "testConversionReplaceOpWithMultipleRanges: PASSED\n");
}

void testTypeConverter1ToNOperandRequires1ToNCallback(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNOperandRequires1ToNCallback
  fprintf(stderr, "@testTypeConverter1ToNOperandRequires1ToNCallback\n");

  // The operand of `test.consumer` converts 1:N (i32 -> (i16, i16)), but the
  // pattern only provides the 1:1 matchAndRewrite (matchAndRewrite1ToN is
  // null). The driver's 1:1 dispatch cannot represent a 1:N-remapped operand,
  // so the pattern fails to match and the conversion fails.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, splitI32, NULL);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  // Only the 1:1 matchAndRewrite is set; matchAndRewrite1ToN is null.
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, convertConsumer,
                                              NULL};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsFailure(result) &&
         "1:1-only pattern must fail to legalize a 1:N-remapped operand");

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNOperandRequires1ToNCallback: PASSED
  fprintf(stderr, "testTypeConverter1ToNOperandRequires1ToNCallback: PASSED\n");
}

void testTypeConverter1ToNConversionErasure(MlirContext ctx) {
  // CHECK-LABEL: @testTypeConverter1ToNConversionErasure
  fprintf(stderr, "@testTypeConverter1ToNConversionErasure\n");

  // The i32 operand of `test.consumer` is erased (converted to zero types), so
  // the 1:N pattern is invoked with no operands and builds a nullary
  // `test.consumer_legal`.
  const char *moduleString = "%0 = \"test.producer\"() : () -> i32\n"
                             "\"test.consumer\"(%0) : (i32) -> ()\n";
  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
  MlirOperation moduleOp = mlirModuleGetOperation(module);

  MlirTypeConverter converter = mlirTypeConverterCreate();
  mlirTypeConverterAdd1ToNConversion(converter, eraseI32, NULL);

  MlirRewritePatternSet patterns = mlirRewritePatternSetCreate(ctx);
  MlirConversionPatternCallbacks callbacks = {NULL, NULL, NULL,
                                              convertConsumer1ToN};
  MlirConversionPattern pattern = mlirOpConversionPatternCreate(
      mlirStringRefCreateFromCString("test.consumer"), 1, ctx, converter,
      callbacks, NULL, 0, NULL);
  mlirRewritePatternSetAdd(patterns,
                           mlirConversionPatternAsRewritePattern(pattern));
  MlirFrozenRewritePatternSet frozen = mlirFreezeRewritePattern(patterns);
  mlirRewritePatternSetDestroy(patterns);

  MlirConversionTarget target = mlirConversionTargetCreate(ctx);
  mlirConversionTargetAddIllegalOp(
      target, mlirStringRefCreateFromCString("test.consumer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.producer"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("test.consumer_legal"));
  mlirConversionTargetAddLegalOp(
      target, mlirStringRefCreateFromCString("builtin.module"));

  MlirConversionConfig config = mlirConversionConfigCreate();
  MlirLogicalResult result =
      mlirApplyPartialConversion(moduleOp, target, frozen, config);
  assert(mlirLogicalResultIsSuccess(result));

  mlirOperationDump(moduleOp);
  // clang-format off
  // CHECK: "test.consumer_legal"() : () -> ()
  // clang-format on

  mlirConversionConfigDestroy(config);
  mlirConversionTargetDestroy(target);
  mlirFrozenRewritePatternSetDestroy(frozen);
  mlirTypeConverterDestroy(converter);
  mlirModuleDestroy(module);

  // CHECK: testTypeConverter1ToNConversionErasure: PASSED
  fprintf(stderr, "testTypeConverter1ToNConversionErasure: PASSED\n");
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
  testTypeConverterSourceMaterialization(ctx);
  testTypeConverterTargetMaterialization(ctx);
  testTypeConverterConversionStatus(ctx);
  testTypeConverter1ToNTargetMaterialization(ctx);
  testTypeConverter1ToNConversionDeclineRollback(ctx);
  testTypeConverter1ToNConversionFailure(ctx);
  testTypeConverter1ToNTargetMaterializationDecline(ctx);
  testTypeConverterMultiInputSourceMaterialization(ctx);
  testConversionReplaceOpWithMultipleRanges(ctx);
  testTypeConverter1ToNOperandRequires1ToNCallback(ctx);
  testTypeConverter1ToNConversionErasure(ctx);

  mlirContextDestroy(ctx);
  return 0;
}
