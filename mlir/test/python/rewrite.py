# RUN: %PYTHON %s | FileCheck %s

import gc
from mlir.ir import *
from mlir.rewrite import *


def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  return f


# CHECK-LABEL: TEST: testGreedyRewriteDriverConfigCreation
@run
def testGreedyRewriteDriverConfigCreation():
  # Test basic config creation and destruction
  config = GreedyRewriteDriverConfig()
  # CHECK: Config created successfully
  print("Config created successfully")


# CHECK-LABEL: TEST: testGreedyRewriteDriverConfigGetters
@run
def testGreedyRewriteDriverConfigGetters():
  config = GreedyRewriteDriverConfig()

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
  config = GreedyRewriteDriverConfig()

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
  config = GreedyRewriteDriverConfig()

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
