# AMDGPU NextUseAnalysis

This directory contains comprehensive tests for the AMDGPU NextUseAnalysis pass, which implements a three-tier ranking system for spiller decisions with EdgeWeight parameter support.

## Overview

The NextUseAnalysis pass calculates next-use distances for virtual registers to help the spiller make optimal decisions about which registers to spill. The analysis implements a sophisticated three-tier ranking system:

- **Tier 1**: Finite distances (0 to LoopTag-1) - immediate uses
- **Tier 2**: Loop-exit distances (LoopTag to DeadTag-1) - mapped to high range [60000, 64999]  
- **Tier 3**: Dead registers (DeadTag+) - assigned maximum distance (65535)

## EdgeWeight Parameter

The analysis includes an `EdgeWeight` parameter that adds weight to distances when crossing loop exit edges:

```cpp
int64_t EdgeWeight = 0;
if (LoopExits.contains(MBB->getNumber())) {
  unsigned ExitTo = LoopExits[MBB->getNumber()];
  if (SuccNum == ExitTo)
    EdgeWeight = LoopTag;
}
```

This ensures that registers used after loop exits are deprioritized for spilling within the loop.

## Test Suite Structure

### Basic Control Flow Tests
- **`simple-linear-block-distances.mir`** - Linear control flow with conditional branches
- **`simple-loop-3blocks.mir`** - Simple loop with minimal basic blocks

### Single Loop Tests  
- **`complex-single-loop-a.mir`** - Complex single loop with multiple internal paths
- **`complex-single-loop-b.mir`** - Another variant of complex single loop
- **`complex-single-loop.mir`** - Additional single loop complexity test

### Sequential Loop Tests
- **`sequence_2_loops.mir`** - Two loops executed sequentially
- **`multi_exit_loop_followed_by_simple_loop.mir`** - Multi-exit loop followed by simple loop

### Nested Loop Tests (True Nesting)
- **`inner_cfg_in_2_nesteed_loops.mir`** - Inner control flow within 2 nested loops
- **`if_else_with_loops_nested_in_2_outer_loops.mir`** - Conditional logic with loops nested in outer loops
- **`loop_nested_in_3_outer_loops_complex_cfg.mir`** - Deep nesting with complex control flow
- **`three_loops_sequence_nested_in_outer_loop.mir`** - Sequential loops within an outer loop container
- **`triple-nested-loops.mir`** - Three levels of true loop nesting

### Side Exit Tests
- **`nested-loops-with-side-exits-a.mir`** - Nested loops with early exit paths
- **`nested-loops-with-side-exits-b.mir`** - Alternative nested loops with side exits

### Complex Control Flow Tests
- **`complex-control-flow-11blocks.mir`** - Complex control flow with 11 basic blocks
- **`complex-control-flow-14blocks.mir`** - More complex control flow with 14 basic blocks

### Ranking System Tests
- **`three-tier-ranking-nested-loops.mir`** - Specific test for the three-tier ranking system

## Test Validation

All tests include comprehensive CHECK patterns that validate:

1. **Analysis Output Format**: Proper structure of NextUseAnalysis results
2. **Distance Calculations**: Correct next-use distance computation
3. **EdgeWeight Application**: Proper handling of loop-exit edge weights
4. **Three-Tier Ranking**: Correct categorization into finite/loop-exit/dead tiers
5. **Register Tracking**: Accurate virtual register next-use information

## Running Tests

To run all NextUseAnalysis tests:

```bash
cd /path/to/llvm-project/llvm/test/CodeGen/AMDGPU/NextUseAnalysis
llvm-lit -v .
```

To run a specific test:

```bash
llvm-lit -v simple-loop-3blocks.mir
```

## Test Generation

Tests were generated using the automated `update_test_checks.py` script:

```bash
python3 update_test_checks.py <test_file.mir> <path_to_llc>
```

This ensures consistent CHECK pattern formatting and comprehensive coverage of the analysis output.

## Implementation Details

The tests validate the core functionality:

- **Loop Exit Detection**: Tests verify proper identification of loop exit edges
- **Distance Materialization**: Validates the `materializeForRank()` function
- **Offset Handling**: Ensures correct instruction offset calculations  
- **Successor Merging**: Tests proper merging of successor block distances
- **Three-Tier System**: Validates the ranking system across all complexity levels

## Coverage

The test suite provides comprehensive coverage across:
- 17 test files
- Simple to extremely complex control flow patterns
- All three ranking tiers
- Various loop nesting scenarios
- EdgeWeight parameter functionality
- 100% pass rate validation

This ensures the NextUseAnalysis implementation is production-ready and handles all expected AMDGPU spilling scenarios.
