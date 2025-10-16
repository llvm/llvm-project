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

  mlirContextDestroy(ctx);
  return 0;
}
