# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc
from mlir.ir import *
from mlir.passmanager import *
from mlir.dialects.builtin import ModuleOp
from mlir.dialects import arith
from mlir.rewrite import *


def run(f):
    print("\nTEST:", f.__name__)
    f()
    gc.collect()
    return f


# CHECK-LABEL: TEST: testRewritePattern
@run
def testRewritePattern():
    def to_muli(op, rewriter):
        with rewriter.ip:
            assert isinstance(op, arith.AddIOp)
            new_op = arith.muli(op.lhs, op.rhs, loc=op.location)
        rewriter.replace_op(op, new_op.owner)

    def constant_1_to_2(op, rewriter):
        c = op.value.value
        if c != 1:
            return True  # failed to match
        with rewriter.ip:
            new_op = arith.constant(op.type, 2, loc=op.location)
        rewriter.replace_op(op, [new_op])

    with Context():
        patterns = RewritePatternSet()
        patterns.add(arith.AddIOp, to_muli)
        patterns.add("arith.constant", constant_1_to_2)
        frozen = patterns.freeze()

        module = ModuleOp.parse(
            r"""
            module {
              func.func @add(%a: i64, %b: i64) -> i64 {
                %sum = arith.addi %a, %b : i64
                return %sum : i64
              }
            }
            """
        )

        apply_patterns_and_fold_greedily(module, frozen)
        # CHECK: %0 = arith.muli %arg0, %arg1 : i64
        # CHECK: return %0 : i64
        print(module)

        module = ModuleOp.parse(
            r"""
            module {
              func.func @const() -> (i64, i64) {
                %0 = arith.constant 1 : i64
                %1 = arith.constant 3 : i64
                return %0, %1 : i64, i64
              }
            }
            """
        )

        apply_patterns_and_fold_greedily(module, frozen)
        # CHECK: %c2_i64 = arith.constant 2 : i64
        # CHECK: %c3_i64 = arith.constant 3 : i64
        # CHECK: return %c2_i64, %c3_i64 : i64, i64
        print(module)

        module = ModuleOp.parse(
            r"""
            module {
              func.func @add(%a: i64, %b: i64) -> i64 {
                %sum = arith.addi %a, %b : i64
                return %sum : i64
              }
            }
            """
        )

        walk_and_apply_patterns(module, frozen)
        # CHECK: %0 = arith.muli %arg0, %arg1 : i64
        # CHECK: return %0 : i64
        print(module)


# CHECK-LABEL: TEST: testGreedyRewriteConfigCreation
@run
def testGreedyRewriteConfigCreation():
    # Test basic config creation and destruction
    config = GreedyRewriteConfig()
    # CHECK: Config created successfully
    print("Config created successfully")


# CHECK-LABEL: TEST: testGreedyRewriteConfigGetters
@run
def testGreedyRewriteConfigGetters():
    config = GreedyRewriteConfig()

    # Set some values
    config.max_iterations = 5
    config.max_num_rewrites = 50
    config.use_top_down_traversal = True
    config.enable_folding = False
    config.strictness = GreedyRewriteStrictness.EXISTING_AND_NEW_OPS
    config.region_simplification_level = GreedySimplifyRegionLevel.AGGRESSIVE
    config.enable_constant_cse = True

    # Test all getter methods and print results
    # CHECK: max_iterations: 5
    max_iterations = config.max_iterations
    print(f"max_iterations: {max_iterations}")
    # CHECK: max_rewrites: 50
    max_rewrites = config.max_num_rewrites
    print(f"max_rewrites: {max_rewrites}")
    # CHECK: use_top_down: True
    use_top_down = config.use_top_down_traversal
    print(f"use_top_down: {use_top_down}")
    # CHECK: folding_enabled: False
    folding_enabled = config.enable_folding
    print(f"folding_enabled: {folding_enabled}")
    # CHECK: strictness: GreedyRewriteStrictness.EXISTING_AND_NEW_OPS
    strictness = config.strictness
    print(f"strictness: {strictness}")
    # CHECK: region_level: GreedySimplifyRegionLevel.AGGRESSIVE
    region_level = config.region_simplification_level
    print(f"region_level: {region_level}")
    # CHECK: cse_enabled: True
    cse_enabled = config.enable_constant_cse
    print(f"cse_enabled: {cse_enabled}")


# CHECK-LABEL: TEST: testGreedyRewriteStrictnessEnum
@run
def testGreedyRewriteStrictnessEnum():
    config = GreedyRewriteConfig()

    # Test ANY_OP
    # CHECK: strictness ANY_OP: GreedyRewriteStrictness.ANY_OP
    config.strictness = GreedyRewriteStrictness.ANY_OP
    strictness = config.strictness
    print(f"strictness ANY_OP: {strictness}")

    # Test EXISTING_AND_NEW_OPS
    # CHECK: strictness EXISTING_AND_NEW_OPS: GreedyRewriteStrictness.EXISTING_AND_NEW_OPS
    config.strictness = GreedyRewriteStrictness.EXISTING_AND_NEW_OPS
    strictness = config.strictness
    print(f"strictness EXISTING_AND_NEW_OPS: {strictness}")

    # Test EXISTING_OPS
    # CHECK: strictness EXISTING_OPS: GreedyRewriteStrictness.EXISTING_OPS
    config.strictness = GreedyRewriteStrictness.EXISTING_OPS
    strictness = config.strictness
    print(f"strictness EXISTING_OPS: {strictness}")


# CHECK-LABEL: TEST: testGreedySimplifyRegionLevelEnum
@run
def testGreedySimplifyRegionLevelEnum():
    config = GreedyRewriteConfig()

    # Test DISABLED
    # CHECK: region_level DISABLED: GreedySimplifyRegionLevel.DISABLED
    config.region_simplification_level = GreedySimplifyRegionLevel.DISABLED
    level = config.region_simplification_level
    print(f"region_level DISABLED: {level}")

    # Test NORMAL
    # CHECK: region_level NORMAL: GreedySimplifyRegionLevel.NORMAL
    config.region_simplification_level = GreedySimplifyRegionLevel.NORMAL
    level = config.region_simplification_level
    print(f"region_level NORMAL: {level}")

    # Test AGGRESSIVE
    # CHECK: region_level AGGRESSIVE: GreedySimplifyRegionLevel.AGGRESSIVE
    config.region_simplification_level = GreedySimplifyRegionLevel.AGGRESSIVE
    level = config.region_simplification_level
    print(f"region_level AGGRESSIVE: {level}")


# CHECK-LABEL: TEST: testRewriteWithGreedyRewriteConfig
@run
def testRewriteWithGreedyRewriteConfig():
    def constant_1_to_2(op, rewriter):
        c = op.value.value
        if c != 1:
            return True  # failed to match
        with rewriter.ip:
            new_op = arith.constant(op.type, 2, loc=op.location)
        rewriter.replace_op(op, [new_op])

    with Context():
        patterns = RewritePatternSet()
        patterns.add(arith.ConstantOp, constant_1_to_2)
        frozen = patterns.freeze()

        module = ModuleOp.parse(
            r"""
            module {
              func.func @const() -> (i64, i64) {
                %0 = arith.constant 1 : i64
                %1 = arith.constant 1 : i64
                return %0, %1 : i64, i64
              }
            }
            """
        )

        config = GreedyRewriteConfig()
        config.enable_constant_cse = False
        apply_patterns_and_fold_greedily(module, frozen, config)
        # CHECK: %c2_i64 = arith.constant 2 : i64
        # CHECK: %c2_i64_0 = arith.constant 2 : i64
        # CHECK: return %c2_i64, %c2_i64_0 : i64, i64
        print(module)

        config = GreedyRewriteConfig()
        config.enable_constant_cse = True
        apply_patterns_and_fold_greedily(module, frozen, config)
        # CHECK: %c2_i64 = arith.constant 2 : i64
        # CHECK: return %c2_i64, %c2_i64 : i64
        print(module)
