//===- WrapInZeroTripCheck.cpp - Loop transforms to add zero-trip-check ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

/// Create zero-trip-check around a `while` op and return the new loop op in the
/// check. The while loop is rotated to avoid evaluating the condition twice.
///
/// Given an example below:
///
///   scf.while (%arg0 = %init) : (i32) -> i64 {
///     %val = .., %arg0 : i64
///     %cond = arith.cmpi .., %arg0 : i32
///     scf.condition(%cond) %val : i64
///   } do {
///   ^bb0(%arg1: i64):
///     %next = .., %arg1 : i32
///     scf.yield %next : i32
///   }
///
/// First clone before block to the front of the loop:
///
///   %pre_val = .., %init : i64
///   %pre_cond = arith.cmpi .., %init : i32
///   scf.while (%arg0 = %init) : (i32) -> i64 {
///     %val = .., %arg0 : i64
///     %cond = arith.cmpi .., %arg0 : i32
///     scf.condition(%cond) %val : i64
///   } do {
///   ^bb0(%arg1: i64):
///     %next = .., %arg1 : i32
///     scf.yield %next : i32
///   }
///
/// Create `if` op with the condition, rotate and move the loop into the else
/// branch:
///
///   %pre_val = .., %init : i64
///   %pre_cond = arith.cmpi .., %init : i32
///   scf.if %pre_cond -> i64 {
///     %res = scf.while (%arg1 = %va0) : (i64) -> i64 {
///       // Original after block
///       %next = .., %arg1 : i32
///       // Original before block
///       %val = .., %next : i64
///       %cond = arith.cmpi .., %next : i32
///       scf.condition(%cond) %val : i64
///     } do {
///     ^bb0(%arg2: i64):
///       %scf.yield %arg2 : i32
///     }
///     scf.yield %res : i64
///   } else {
///     scf.yield %pre_val : i64
///   }
FailureOr<scf::WhileOp> mlir::scf::wrapWhileLoopInZeroTripCheck(
    scf::WhileOp whileOp, RewriterBase &rewriter, bool forceCreateCheck) {
  // If the loop is in do-while form (after block only passes through values),
  // there is no need to create a zero-trip-check as before block is always run.
  if (!forceCreateCheck && isa<scf::YieldOp>(whileOp.getAfterBody()->front())) {
    return whileOp;
  }

  OpBuilder::InsertionGuard insertion_guard(rewriter);

  IRMapping mapper;
  Block *beforeBlock = whileOp.getBeforeBody();
  // Clone before block before the loop for zero-trip-check.
  for (auto [arg, init] :
       llvm::zip_equal(beforeBlock->getArguments(), whileOp.getInits())) {
    mapper.map(arg, init);
  }
  rewriter.setInsertionPoint(whileOp);
  for (auto &op : *beforeBlock) {
    if (isa<scf::ConditionOp>(op)) {
      break;
    }
    // Safe to clone everything as in a single block all defs have been cloned
    // and added to mapper in order.
    rewriter.insert(op.clone(mapper));
  }

  scf::ConditionOp condOp = whileOp.getConditionOp();
  Value clonedCondition = mapper.lookupOrDefault(condOp.getCondition());
  SmallVector<Value> clonedCondArgs = llvm::map_to_vector(
      condOp.getArgs(), [&](Value arg) { return mapper.lookupOrDefault(arg); });

  // Create rotated while loop.
  auto newLoopOp = rewriter.create<scf::WhileOp>(
      whileOp.getLoc(), whileOp.getResultTypes(), clonedCondArgs,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        // Rotate and move the loop body into before block.
        auto newBlock = builder.getBlock();
        rewriter.mergeBlocks(whileOp.getAfterBody(), newBlock, args);
        auto yieldOp = cast<scf::YieldOp>(newBlock->getTerminator());
        rewriter.mergeBlocks(whileOp.getBeforeBody(), newBlock,
                             yieldOp.getResults());
        rewriter.eraseOp(yieldOp);
      },
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        // Pass through values.
        builder.create<scf::YieldOp>(loc, args);
      });

  // Create zero-trip-check and move the while loop in.
  auto ifOp = rewriter.create<scf::IfOp>(
      whileOp.getLoc(), clonedCondition,
      [&](OpBuilder &builder, Location loc) {
        // Then runs the while loop.
        rewriter.moveOpBefore(newLoopOp, builder.getInsertionBlock(),
                              builder.getInsertionPoint());
        builder.create<scf::YieldOp>(loc, newLoopOp.getResults());
      },
      [&](OpBuilder &builder, Location loc) {
        // Else returns the results from precondition.
        builder.create<scf::YieldOp>(loc, clonedCondArgs);
      });

  rewriter.replaceOp(whileOp, ifOp);

  return newLoopOp;
}
