// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(test.isolated_region_branch(remove-dead-values)))' > %t.parallel
// RUN: FileCheck %s --check-prefixes=CHECK,CANON < %t.parallel
// RUN: mlir-opt %s --mlir-disable-threading --pass-pipeline='builtin.module(func.func(test.isolated_region_branch(remove-dead-values)))' > %t.serial
// RUN: diff %t.parallel %t.serial
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(test.isolated_region_branch(remove-dead-values{canonicalize=false})))' | FileCheck %s

// CHECK-LABEL: func.func @live_result
// CHECK-SAME: %[[X:.*]]: i32
// CHECK: %[[ROOT:.*]] = "test.isolated_region_branch"(%[[X]])
// CHECK: ^bb0(%[[ARG:.*]]: i32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ARG]], %[[ARG]] : i32
// CHECK-NEXT: "test.isolated_region_yield"(%[[SUM]])
// CHECK: return %[[ROOT]] : i32
func.func @live_result(%x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%x) ({
  ^bb0(%arg: i32):
    %dead = arith.muli %arg, %arg : i32
    %sum = arith.addi %arg, %arg : i32
    "test.isolated_region_yield"(%sum) : (i32) -> ()
  }) : (i32) -> i32
  return %r : i32
}

// Unused root results and inputs are outside this pass's rewrite scope.
// CHECK-LABEL: func.func @unused_result
// CHECK-SAME: %[[X:.*]]: i32
// CHECK: "test.isolated_region_branch"(%[[X]], %[[X]])
// CHECK: ^bb0(%[[A:.*]]: i32, %[[UNUSED:.*]]: i32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[A]], %[[A]] : i32
// CHECK-NEXT: "test.isolated_region_yield"(%[[SUM]])
// CHECK: "test.isolated_region_branch"(%[[X]], %[[X]])
// CHECK: "test.isolated_region_yield"
// CHECK: return
func.func @unused_result(%x: i32) {
  %r = "test.isolated_region_branch"(%x, %x) ({
  ^bb0(%arg: i32, %unused: i32):
    %sum = arith.addi %arg, %arg : i32
    "test.isolated_region_yield"(%sum) : (i32) -> ()
  }) : (i32, i32) -> i32
  %other = "test.isolated_region_branch"(%x, %x) ({
  ^bb0(%arg: i32, %unused: i32):
    "test.isolated_region_yield"(%arg) : (i32) -> ()
  }) : (i32, i32) -> i32
  return
}

// CHECK-LABEL: func.func @multiple_regions
// CHECK: "test.isolated_region_branch"()
// CHECK-NEXT: "test.isolated_region_yield"()
// CHECK-NEXT: }, {
// CHECK-NEXT: "test.isolated_region_yield"()
// CHECK: return
func.func @multiple_regions() {
  "test.isolated_region_branch"() ({
    %dead = arith.constant 1 : i32
    "test.isolated_region_yield"() : () -> ()
  }, {
    %dead = arith.constant 2 : i32
    "test.isolated_region_yield"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func.func @effects
// CHECK: "test.isolated_region_branch"
// CHECK: ^bb0(%[[MEM:.*]]: memref<i32>, %[[VAL:.*]]: i32):
// CHECK-NEXT: memref.store %[[VAL]], %[[MEM]][] : memref<i32>
// CHECK-NEXT: "test.isolated_region_yield"()
func.func @effects(%mem: memref<i32>, %x: i32) {
  "test.isolated_region_branch"(%mem, %x) ({
  ^bb0(%arg: memref<i32>, %val: i32):
    %dead = arith.muli %val, %val : i32
    memref.store %val, %arg[] : memref<i32>
    "test.isolated_region_yield"() : () -> ()
  }) : (memref<i32>, i32) -> ()
  return
}

// Remove dead loop results in both root regions with the shared patterns.
// CANON-LABEL: func.func @dead_carry
// CANON: "test.isolated_region_branch"
// CANON: scf.for {{.*}} iter_args(%[[CARRY:.*]] = %{{.*}}) -> (i32)
// CANON: scf.yield %{{.*}} : i32
// CANON: "test.isolated_region_yield"
// CANON: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i32)
// CANON: scf.yield %{{.*}} : i32
// CANON: "test.isolated_region_yield"
func.func @dead_carry(%lb: index, %ub: index, %step: index, %x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%lb, %ub, %step, %x) ({
  ^bb0(%lower: index, %upper: index, %stride: index, %init: i32):
    %loop:2 = scf.for %iv = %lower to %upper step %stride
        iter_args(%live = %init, %dead = %init) -> (i32, i32) {
      %sum = arith.addi %live, %live : i32
      %unused = arith.muli %dead, %dead : i32
      scf.yield %sum, %unused : i32, i32
    }
    "test.isolated_region_yield"(%loop#0) : (i32) -> ()
  }, {
  ^bb0(%lower: index, %upper: index, %stride: index, %init: i32):
    %loop:2 = scf.for %iv = %lower to %upper step %stride
        iter_args(%live = %init, %dead = %init) -> (i32, i32) {
      %sum = arith.addi %live, %live : i32
      %unused = arith.muli %dead, %dead : i32
      scf.yield %sum, %unused : i32, i32
    }
    "test.isolated_region_yield"(%loop#0) : (i32) -> ()
  }) : (index, index, index, i32) -> i32
  return %r : i32
}

// CHECK-LABEL: func.func @isolated_dead_input
// CHECK-SAME: %[[X:.*]]: i32
// CHECK: "test.isolated_region_branch"(%[[X]], %[[X]])
// CHECK: ^bb0(%[[ARG:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT: "test.isolated_region_yield"(%[[ARG]])
func.func @isolated_dead_input(%x: i32) -> i32 {
  %r = "test.isolated_region_branch"(%x, %x) ({
  ^bb0(%arg: i32, %dead: i32):
    "test.isolated_region_yield"(%arg) : (i32) -> ()
  }) : (i32, i32) -> i32
  return %r : i32
}
