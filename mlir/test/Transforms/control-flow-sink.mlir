// RUN: mlir-opt -control-flow-sink %s | FileCheck %s

// Test that operations can be sunk.

// CHECK-LABEL: @test_simple_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK-NEXT: %[[V0:.*]] = arith.subi %[[ARG2]], %[[ARG1]]
// CHECK-NEXT: %[[V1:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V2:.*]] = arith.subi %[[ARG1]], %[[ARG2]]
// CHECK-NEXT:   test.region_if_yield %[[V2]]
// CHECK-NEXT: } else {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V2:.*]] = arith.addi %[[ARG1]], %[[ARG1]]
// CHECK-NEXT:   %[[V3:.*]] = arith.addi %[[V0]], %[[V2]]
// CHECK-NEXT:   test.region_if_yield %[[V3]]
// CHECK-NEXT: } join {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V2:.*]] = arith.addi %[[ARG2]], %[[ARG2]]
// CHECK-NEXT:   %[[V3:.*]] = arith.addi %[[V2]], %[[V0]]
// CHECK-NEXT:   test.region_if_yield %[[V3]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V1]]
func @test_simple_sink(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %0 = arith.subi %arg1, %arg2 : i32
  %1 = arith.subi %arg2, %arg1 : i32
  %2 = arith.addi %arg1, %arg1 : i32
  %3 = arith.addi %arg2, %arg2 : i32
  %4 = test.region_if %arg0: i32 -> i32 then {
  ^bb0(%arg3: i32):
    test.region_if_yield %0 : i32
  } else {
  ^bb0(%arg3: i32):
    %5 = arith.addi %1, %2 : i32
    test.region_if_yield %5 : i32
  } join {
  ^bb0(%arg3: i32):
    %5 = arith.addi %3, %1 : i32
    test.region_if_yield %5 : i32
  }
  return %4 : i32
}

// Test that a region op can be sunk.

// CHECK-LABEL: @test_region_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK-NEXT: %[[V0:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V1:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     test.region_if_yield %[[ARG1]]
// CHECK-NEXT:   } else {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     %[[V2:.*]] = arith.subi %[[ARG1]], %[[ARG2]]
// CHECK-NEXT:     test.region_if_yield %[[V2]]
// CHECK-NEXT:   } join {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     test.region_if_yield %[[ARG2]]
// CHECK-NEXT:   }
// CHECK-NEXT:   test.region_if_yield %[[V1]]
// CHECK-NEXT: } else {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG1]]
// CHECK-NEXT: } join {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG2]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V0]]
func @test_region_sink(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %0 = arith.subi %arg1, %arg2 : i32
  %1 = test.region_if %arg0: i32 -> i32 then {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg1 : i32
  } else {
  ^bb0(%arg3: i32):
    test.region_if_yield %0 : i32
  } join {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg2 : i32
  }
  %2 = test.region_if %arg0: i32 -> i32 then {
  ^bb0(%arg3: i32):
    test.region_if_yield %1 : i32
  } else {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg1 : i32
  } join {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg2 : i32
  }
  return %2 : i32
}

// Test that an entire subgraph can be sunk.

// CHECK-LABEL: @test_subgraph_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK-NEXT: %[[V0:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V1:.*]] = arith.subi %[[ARG1]], %[[ARG2]]
// CHECK-NEXT:   %[[V2:.*]] = arith.addi %[[ARG1]], %[[ARG2]]
// CHECK-NEXT:   %[[V3:.*]] = arith.subi %[[ARG2]], %[[ARG1]]
// CHECK-NEXT:   %[[V4:.*]] = arith.muli %[[V3]], %[[V3]]
// CHECK-NEXT:   %[[V5:.*]] = arith.muli %[[V2]], %[[V1]]
// CHECK-NEXT:   %[[V6:.*]] = arith.addi %[[V5]], %[[V4]]
// CHECK-NEXT:   test.region_if_yield %[[V6]]
// CHECK-NEXT: } else {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG1]]
// CHECK-NEXT: } join {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG2]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V0]]
func @test_subgraph_sink(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %0 = arith.addi %arg1, %arg2 : i32
  %1 = arith.subi %arg1, %arg2 : i32
  %2 = arith.subi %arg2, %arg1 : i32
  %3 = arith.muli %0, %1 : i32
  %4 = arith.muli %2, %2 : i32
  %5 = arith.addi %3, %4 : i32
  %6 = test.region_if %arg0: i32 -> i32 then {
  ^bb0(%arg3: i32):
    test.region_if_yield %5 : i32
  } else {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg1 : i32
  } join {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg2 : i32
  }
  return %6 : i32
}

// Test that ops can be sunk into regions with multiple blocks.

// CHECK-LABEL: @test_multiblock_region_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
// CHECK-NEXT: %[[V0:.*]] = arith.addi %[[ARG1]], %[[ARG2]]
// CHECK-NEXT: %[[V1:.*]] = "test.any_cond"() ({
// CHECK-NEXT:   %[[V3:.*]] = arith.addi %[[V0]], %[[ARG2]]
// CHECK-NEXT:   %[[V4:.*]] = arith.addi %[[V3]], %[[ARG1]]
// CHECK-NEXT:   cf.br ^bb1(%[[V4]] : i32)
// CHECK-NEXT: ^bb1(%[[V5:.*]]: i32):
// CHECK-NEXT:   %[[V6:.*]] = arith.addi %[[V5]], %[[V4]]
// CHECK-NEXT:   "test.yield"(%[[V6]])
// CHECK-NEXT: })
// CHECK-NEXT: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]]
// CHECK-NEXT: return %[[V2]]
func @test_multiblock_region_sink(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %0 = arith.addi %arg1, %arg2 : i32
  %1 = arith.addi %0, %arg2 : i32
  %2 = arith.addi %1, %arg1 : i32
  %3 = "test.any_cond"() ({
    cf.br ^bb1(%2 : i32)
  ^bb1(%5: i32):
    %6 = arith.addi %5, %2 : i32
    "test.yield"(%6) : (i32) -> ()
  }) : () -> i32
  %4 = arith.addi %0, %3 : i32
  return %4 : i32
}

// Test that ops can be sunk recursively into nested regions.

// CHECK-LABEL: @test_nested_region_sink
// CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32 {
// CHECK-NEXT: %[[V0:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[V1:.*]] = test.region_if %[[ARG0]]: i32 -> i32 then {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     %[[V2:.*]] = arith.addi %[[ARG1]], %[[ARG1]]
// CHECK-NEXT:     test.region_if_yield %[[V2]]
// CHECK-NEXT:   } else {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     test.region_if_yield %[[ARG1]]
// CHECK-NEXT:   } join {
// CHECK-NEXT:   ^bb0(%{{.*}}: i32):
// CHECK-NEXT:     test.region_if_yield %[[ARG1]]
// CHECK-NEXT:   }
// CHECK-NEXT:   test.region_if_yield %[[V1]]
// CHECK-NEXT: } else {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG1]]
// CHECK-NEXT: } join {
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   test.region_if_yield %[[ARG1]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[V0]]
func @test_nested_region_sink(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg1, %arg1 : i32
  %1 = test.region_if %arg0: i32 -> i32 then {
  ^bb0(%arg3: i32):
    %2 = test.region_if %arg0: i32 -> i32 then {
    ^bb0(%arg4: i32):
      test.region_if_yield %0 : i32
    } else {
    ^bb0(%arg4: i32):
      test.region_if_yield %arg1 : i32
    } join {
    ^bb0(%arg4: i32):
      test.region_if_yield %arg1 : i32
    }
    test.region_if_yield %2 : i32
  } else {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg1 : i32
  } join {
  ^bb0(%arg3: i32):
    test.region_if_yield %arg1 : i32
  }
  return %1 : i32
}

// Test that ops are only moved into the entry block, even when their only uses
// are further along.

// CHECK-LABEL: @test_not_sunk_deeply
// CHECK-SAME:  (%[[ARG0:.*]]: i32) -> i32 {
// CHECK-NEXT: %[[V0:.*]] = "test.any_cond"() ({
// CHECK-NEXT:   %[[V1:.*]] = arith.addi %[[ARG0]], %[[ARG0]]
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "test.yield"(%[[V1]]) : (i32) -> ()
// CHECK-NEXT: })
// CHECK-NEXT: return %[[V0]]
func @test_not_sunk_deeply(%arg0: i32) -> i32 {
  %0 = arith.addi %arg0, %arg0 : i32
  %1 = "test.any_cond"() ({
    cf.br ^bb1
  ^bb1:
    "test.yield"(%0) : (i32) -> ()
  }) : () -> i32
  return %1 : i32
}
