//===- LowerEarlyExit.cpp - Lower `scf.break` to yield-only SCF -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes all early-exit control flow from SCF: after the pass runs,
// the IR contains no `scf.break` ops and no `token`-typed block arguments on
// `scf.execute_region` ops.
//
// The lowering is intentionally split into a handful of small, local rewrite
// rules that can each be understood in isolation. Their fixed-point combination
// (run by the greedy pattern driver) eliminates arbitrary early-exit nests:
//
//   * BreakToYield        -- a break that targets the closest enclosing
//                            `scf.execute_region` and is already in "tail
//                            position" becomes a plain `scf.yield`.
//   * SinkContinuationIntoIf
//                         -- an `scf.if` that may break out is rewritten so the
//                            code following it is sunk into the fall-through
//                            branch(es) and the `scf.if` forwards the target's
//                            results. This drives breaks into tail position.
//   * HoistBreakThroughExecuteRegion
//                         -- a break that targets a *non-closest* enclosing
//                            `scf.execute_region` is hoisted out of the closest
//                            one by threading an (i1 flag, values...) suffix
//                            through it and re-breaking (guarded by the flag)
//                            in the parent region. Applied repeatedly, this
//                            moves a break up one `scf.execute_region` at a
//                            time.
//
// Once the greedy rewrite reaches a fixed point, the pass removes every
// remaining token block argument from `scf.execute_region` ops. At that point
// all such tokens are expected to be unused.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFLOWEREARLYEXIT
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Returns the closest enclosing token-bearing `scf.execute_region` of `op`, or
/// null if there is none.
///
/// Example: for the `scf.break` below, this returns the *inner* execute_region
/// (`%tok1`), regardless of which region the break ultimately targets.
///
///   scf.execute_region {            // <- not the closest
///   ^bb0(%tok0: token):
///     scf.execute_region {          // <- returned (closest)
///     ^bb1(%tok1: token):
///       scf.break %tok0             // op = this break
///     }
///   }
static ExecuteRegionOp getClosestTokenExecuteRegion(Operation *op) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
    if (auto er = dyn_cast<ExecuteRegionOp>(p))
      if (er.getToken())
        return er;
  return nullptr;
}

/// Returns true if yielding `op`'s results (in order) from its enclosing block
/// would make them flow unchanged into `target`'s results, i.e. `op` is in
/// "tail position" with respect to `target`. This is the case when `op` is
/// immediately followed by an `scf.yield` that forwards exactly `op`'s results,
/// and (recursively) the op owning that block is either `target` itself or an
/// `scf.if` that is also in tail position.
///
/// Example: `%inner` is in tail position w.r.t. the execute_region, because its
/// result is forwarded unchanged up to the target through forwarding yields:
///
///   %r = scf.execute_region -> f32 {          // target
///     %outer = scf.if %c -> f32 {
///       %inner = scf.if %d -> f32 { ... }      // op (tail position)
///       scf.yield %inner : f32                 // forwards %inner
///     } else { ... }
///     scf.yield %outer : f32                   // forwards %outer
///   }
static bool isTailToTarget(Operation *op, ExecuteRegionOp target) {
  Block *block = op->getBlock();
  Operation *term = block->getTerminator();
  if (op->getNextNode() != term)
    return false;
  auto yield = dyn_cast<YieldOp>(term);
  if (!yield || !llvm::equal(yield.getOperands(), op->getResults()))
    return false;

  Operation *parent = block->getParentOp();
  if (parent == target.getOperation())
    return true;
  if (auto parentIf = dyn_cast<IfOp>(parent))
    return isTailToTarget(parentIf, target);
  return false;
}

/// Returns true if `ifOp` transitively contains an `scf.break`.
static bool hasBreak(IfOp ifOp) {
  return ifOp.walk([](BreakOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Materializes `count` poison values for the given `types`.
static void appendPoison(RewriterBase &rewriter, Location loc, TypeRange types,
                         SmallVectorImpl<Value> &out) {
  for (Type t : types)
    out.push_back(ub::PoisonOp::create(rewriter, loc, t));
}

/// Materializes an `i1` constant.
static Value makeBool(RewriterBase &rewriter, Location loc, bool value) {
  return arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(value));
}

//===----------------------------------------------------------------------===//
// BreakToYield
//===----------------------------------------------------------------------===//

/// Replaces a `scf.break` with a `scf.yield` when the break already reaches its
/// target by simple fall-through. This is the terminal rule: it is the only one
/// that actually removes a break.
///
/// It fires when either:
///   (a) the break is a terminator of a block directly inside its target
///       `scf.execute_region` (so yielding returns the same values), or
///   (b) the break terminates a branch of an `scf.if` that is in tail position
///       with respect to the target (so the values are forwarded unchanged).
///
/// Case (a), break directly in the target:
///
///   scf.execute_region -> f32 {            scf.execute_region -> f32 {
///   ^bb0(%tok: token):              ==>     ^bb0(%tok: token):
///     scf.break %tok, %v : f32                scf.yield %v : f32
///   }                                       }
///
/// Case (b), break in a tail-position `scf.if`:
///
///   %r = scf.if %c -> f32 {                 %r = scf.if %c -> f32 {
///     scf.break %tok, %v : f32      ==>        scf.yield %v : f32
///   } else {                                } else {
///     scf.yield %w : f32                       scf.yield %w : f32
///   }                                       }
///   scf.yield %r : f32                      scf.yield %r : f32
struct BreakToYield : OpRewritePattern<BreakOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BreakOp breakOp,
                                PatternRewriter &rewriter) const override {
    auto target = dyn_cast_or_null<ExecuteRegionOp>(breakOp.getTarget());
    if (!target)
      return failure();

    Operation *parent = breakOp->getParentOp();
    bool resolved = false;
    if (parent == target.getOperation()) {
      resolved = true; // case (a)
    } else if (auto parentIf = dyn_cast<IfOp>(parent)) {
      resolved = isTailToTarget(parentIf, target); // case (b)
    }
    if (!resolved)
      return failure();

    rewriter.replaceOpWithNewOp<YieldOp>(breakOp, breakOp.getValues());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SinkContinuationIntoIf
//===----------------------------------------------------------------------===//

/// Rewrites an `scf.if` that may break out to its closest enclosing target so
/// that the code following it (its "continuation") is sunk into the
/// fall-through branch(es), and the `scf.if` itself forwards the values that
/// the continuation would have produced. This drives breaks toward tail
/// position.
///
/// Before:
///   scf.if %c { ...; scf.break %tok, %v } // (a fall-through branch may exist)
///   <continuation ops...>
///   scf.yield %ys
///
/// After:
///   %r... = scf.if %c -> (types(%ys)) {
///     ...; scf.break %tok, %v             // breaking branch unchanged
///   } else {
///     <continuation ops...>; scf.yield %ys
///   }
///   scf.yield %r...
///
/// A branch that itself breaks out is left untouched (its continuation is
/// unreachable). A branch that falls through (terminates with `scf.yield`, or
/// is an absent `else`) receives a copy of the continuation, with the old
/// `scf.if` results remapped to that branch's yielded values.
///
/// When *both* branches fall through (the break is nested deeper, so neither
/// branch terminator is a break), the continuation is cloned into *both* of
/// them -- it is tail-duplicated:
///
/// Before:
///   scf.if %c {
///     scf.if %d { scf.break %tok, %v }   // break nested; %c falls through
///   }
///   <continuation ops...>
///   scf.yield %ys
///
/// After (continuation duplicated into the then and else branches of %c):
///   %r... = scf.if %c -> (types(%ys)) {
///     scf.if %d { scf.break %tok, %v }
///     <continuation ops...>; scf.yield %ys
///   } else {
///     <continuation ops...>; scf.yield %ys
///   }
///   scf.yield %r...
struct SinkContinuationIntoIf : OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Progress gate: only fire on ifs that may actually early-exit.
    if (!hasBreak(ifOp))
      return failure();

    Block *block = ifOp->getBlock();
    Operation *term = block->getTerminator();
    auto yield = dyn_cast<YieldOp>(term);
    if (!yield)
      return failure();

    // If the if is already in canonical tail form, there is nothing to sink;
    // BreakToYield will finish the job.
    bool alreadyTail = ifOp->getNextNode() == term &&
                       llvm::equal(yield.getOperands(), ifOp->getResults());
    if (alreadyTail)
      return failure();

    SmallVector<Type> newResultTypes(yield.getOperandTypes());

    // Carve the continuation -- everything after the if, up to and including
    // the terminating yield -- into its own block. It serves as a template:
    // each fall-through branch receives a copy, and the copied yield becomes
    // that branch's new terminator.
    Block *contBlock =
        rewriter.splitBlock(block, std::next(ifOp->getIterator()));

    // Build the replacement `scf.if` with the continuation's result types and
    // move the existing regions into it.
    rewriter.setInsertionPoint(ifOp);
    auto newIf = IfOp::create(rewriter, ifOp.getLoc(), newResultTypes,
                              ifOp.getCondition(), /*addThenBlock=*/false,
                              /*addElseBlock=*/false);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    if (newIf.getElseRegion().empty())
      rewriter.createBlock(&newIf.getElseRegion());

    for (Region *region : {&newIf.getThenRegion(), &newIf.getElseRegion()}) {
      Block *body = &region->front();
      Operation *branchTerm = body->empty() ? nullptr : &body->back();

      // Leave breaking branches alone: their continuation is unreachable.
      if (branchTerm && isa<BreakOp>(branchTerm))
        continue;

      // Map the old if results to the values this branch yields, drop the
      // branch's yield, then append a copy of the continuation (whose own
      // yield becomes the new branch terminator).
      IRMapping mapping;
      if (auto branchYield = dyn_cast_or_null<YieldOp>(branchTerm)) {
        mapping.map(ifOp.getResults(), branchYield.getOperands());
        rewriter.eraseOp(branchYield);
      }

      rewriter.setInsertionPointToEnd(body);
      for (Operation &op : *contBlock)
        rewriter.clone(op, mapping);
    }

    // The continuation now lives in the fall-through branches; drop the
    // template block and forward the new if's results.
    rewriter.eraseBlock(contBlock);
    rewriter.setInsertionPointToEnd(block);
    YieldOp::create(rewriter, ifOp.getLoc(), newIf.getResults());

    rewriter.eraseOp(ifOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// HoistBreakThroughExecuteRegion
//===----------------------------------------------------------------------===//

/// Hoists breaks that target a *non-closest* enclosing `scf.execute_region` out
/// of the closest one. The closest region `E` is augmented with a suffix of
/// results `(i1 didBreak, <target results...>)`; breaks to the outer target are
/// re-pointed at `E` (driving the flag to true and forwarding the broken-out
/// values), all of `E`'s ordinary exits are extended with `(false, poison...)`,
/// and a flag-guarded re-break to the outer target is inserted right after `E`.
///
/// Applied repeatedly, this moves a break up one `scf.execute_region` at a time
/// until its target becomes the closest one, at which point the other rules
/// finish the lowering.
///
/// Before (the inner break targets the *outer* `%tok`):
///
///   %a = scf.execute_region -> f32 {
///   ^bb0(%tok: token):
///     %b = scf.execute_region -> f32 {
///     ^bb1(%tok1: token):
///       scf.break %tok, %v : f32
///     }
///     <continuation using %b...>
///   }
///
/// After (the break now targets the inner `%tok1`, threading out a flag and the
/// broken-out value; a flag-guarded re-break to `%tok` follows the region):
///
///   %a = scf.execute_region -> f32 {
///   ^bb0(%tok: token):
///     %b, %broke, %bv = scf.execute_region -> (f32, i1, f32) {
///     ^bb1(%tok1: token):
///       %p = ub.poison : f32
///       %t = arith.constant true
///       scf.break %tok1, %p, %t, %v : f32, i1, f32
///     }
///     scf.if %broke {
///       scf.break %tok, %bv : f32
///     }
///     <continuation using %b...>
///   }
struct HoistBreakThroughExecuteRegion : OpRewritePattern<BreakOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BreakOp breakToHoist,
                                PatternRewriter &rewriter) const override {
    auto target = dyn_cast_or_null<ExecuteRegionOp>(breakToHoist.getTarget());
    if (!target)
      return failure();

    ExecuteRegionOp region = getClosestTokenExecuteRegion(breakToHoist);
    if (!region || target == region)
      return failure();

    Value token = region.getToken();
    if (!token)
      return failure();

    TypeRange oldResultTypes = region.getResultTypes();
    TypeRange targetTypes = target.getResultTypes();
    unsigned numOld = oldResultTypes.size();
    Location loc = region.getLoc();

    SmallVector<Type> newResultTypes(oldResultTypes);
    newResultTypes.push_back(rewriter.getI1Type());
    newResultTypes.append(targetTypes.begin(), targetTypes.end());

    // Collect ordinary exits of `region` (its own yields and self-targeted
    // breaks); they are extended with the inactive suffix.
    SmallVector<Operation *> extended;
    for (Block &block : region.getRegion()) {
      if (auto yield = dyn_cast<YieldOp>(block.getTerminator()))
        extended.push_back(yield);
    }
    for (Operation *user : token.getUsers()) {
      if (auto b = dyn_cast<BreakOp>(user)) {
        extended.push_back(b);
      }
    }

    // Extend ordinary exits with `(false, poison...)`.
    for (Operation *op : extended) {
      rewriter.setInsertionPoint(op);
      SmallVector<Value> operands(op->getOperands());
      operands.push_back(makeBool(rewriter, loc, false));
      appendPoison(rewriter, loc, targetTypes, operands);
      rewriter.modifyOpInPlace(op, [&]() { op->setOperands(operands); });
    }

    // Redirect the selected outer-target break to `region`, driving the flag
    // and forwarding the broken-out values; the original-result slots become
    // poison.
    rewriter.setInsertionPoint(breakToHoist);
    SmallVector<Value> operands;
    operands.push_back(token);
    appendPoison(rewriter, loc, oldResultTypes, operands);
    operands.push_back(makeBool(rewriter, loc, true));
    operands.append(breakToHoist.getValues().begin(),
                    breakToHoist.getValues().end());
    rewriter.modifyOpInPlace(breakToHoist,
                             [&]() { breakToHoist->setOperands(operands); });

    // Build the widened execute_region and move the body over.
    rewriter.setInsertionPoint(region);
    auto newRegion = ExecuteRegionOp::create(rewriter, loc, newResultTypes);
    newRegion.getRegion().takeBody(region.getRegion());

    // Insert the guarded re-break to the outer target right after the region.
    rewriter.setInsertionPointAfter(newRegion);
    Value flag = newRegion.getResult(numOld);
    SmallVector<Value> targetVals(
        newRegion.getResults().slice(numOld + 1, targetTypes.size()));
    auto guard = IfOp::create(rewriter, loc, TypeRange{}, flag,
                              /*addThenBlock=*/true, /*addElseBlock=*/false);
    rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
    BreakOp::create(rewriter, loc, target.getToken(), targetVals);

    // Replace the original results with the leading slots of the new region.
    rewriter.replaceOp(region,
                       newRegion.getResults().take_front(numOld));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct SCFLowerEarlyExitPass
    : public impl::SCFLowerEarlyExitBase<SCFLowerEarlyExitPass> {
  void runOnOperation() override {
    WalkResult unsupportedTerminator =
        getOperation()->walk([](ExecuteRegionOp region) {
          for (Block &block : region.getRegion()) {
            Operation *term = block.getTerminator();
            if (isa<YieldOp, BreakOp>(term))
              continue;
            region.emitOpError()
                << "contains unsupported terminator '" << term->getName()
                << "'; expected 'scf.yield' or 'scf.break'";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (unsupportedTerminator.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BreakToYield, SinkContinuationIntoIf,
                 HoistBreakThroughExecuteRegion>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    getOperation()->walk([](ExecuteRegionOp region) {
      Value token = region.getToken();
      if (!token)
        return;
      assert(token.use_empty() &&
             "expected all scf.execute_region tokens to be unused after "
             "lowering early exits");
      region.getRegion().front().eraseArgument(0);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSCFLowerEarlyExitPass() {
  return std::make_unique<SCFLowerEarlyExitPass>();
}
