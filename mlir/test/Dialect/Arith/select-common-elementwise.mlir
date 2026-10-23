// RUN: mlir-opt %s --canonicalize="test-convergence" | FileCheck %s
// RUN: mlir-opt %s --canonicalize="test-convergence" --cse | FileCheck %s

// CHECK-LABEL: @scalar_unary
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] : f32
// CHECK-NEXT: return %[[R]] : f32
func.func @scalar_unary(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = math.sqrt %a : f32 loc("true")
  %y = math.sqrt %b : f32 loc("false")
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @vector_binary
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg3 : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg2, %arg4 : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: %[[R:.*]] = arith.mulf %[[S0]], %[[S1]] : vector<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @vector_binary(%c: vector<4xi1>, %a: vector<4xf32>, %b: vector<4xf32>, %d: vector<4xf32>, %e: vector<4xf32>) -> vector<4xf32> {
  %x = arith.mulf %a, %b : vector<4xf32>
  %y = arith.mulf %d, %e : vector<4xf32>
  %r = arith.select %c, %x, %y : vector<4xi1>, vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: @tensor_shared_operand
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = arith.mulf %[[S]], %arg3 : tensor<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @tensor_shared_operand(%c: tensor<4xi1>, %a: tensor<4xf32>, %b: tensor<4xf32>, %k: tensor<4xf32>) -> tensor<4xf32> {
  %x = arith.mulf %a, %k : tensor<4xf32>
  %y = arith.mulf %b, %k : tensor<4xf32>
  %r = arith.select %c, %x, %y : tensor<4xi1>, tensor<4xf32>
  return %r : tensor<4xf32>
}

// Three differing operand positions need only one select.
// CHECK-LABEL: @repeated_pair
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = math.fma %[[S]], %[[S]], %[[S]] : f32
// CHECK-NEXT: return %[[R]]
func.func @repeated_pair(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = math.fma %a, %a, %a : f32
  %y = math.fma %b, %b, %b : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// The full ordered pair is the key, not just the true operand.
// CHECK-LABEL: @same_true_different_false
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg1, %arg3 : f32
// CHECK-NEXT: %[[R:.*]] = math.fma %[[S0]], %[[S1]], %arg4 : f32
// CHECK-NEXT: return %[[R]]
func.func @same_true_different_false(%c: i1, %a: f32, %b: f32, %d: f32, %k: f32) -> f32 {
  %x = math.fma %a, %a, %k : f32
  %y = math.fma %b, %d, %k : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @reversed_pairs
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg2, %arg1 : f32
// CHECK-NEXT: %[[R:.*]] = arith.divf %[[S0]], %[[S1]] : f32
// CHECK-NEXT: return %[[R]]
func.func @reversed_pairs(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = arith.divf %a, %b : f32
  %y = arith.divf %b, %a : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Select the input type, even when its width differs from the result width.
// CHECK-LABEL: @conversion
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: %[[R:.*]] = arith.fptosi %[[S]] : vector<4xf32> to vector<4xi8>
// CHECK-NEXT: return %[[R]]
func.func @conversion(%c: vector<4xi1>, %a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xi8> {
  %x = arith.fptosi %a : vector<4xf32> to vector<4xi8>
  %y = arith.fptosi %b : vector<4xf32> to vector<4xi8>
  %r = arith.select %c, %x, %y : vector<4xi1>, vector<4xi8>
  return %r : vector<4xi8>
}

// CHECK-LABEL: @same_fastmath
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] fastmath<nnan,ninf> : f32
// CHECK-NEXT: return %[[R]]
func.func @same_fastmath(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = math.sqrt %a fastmath<nnan,ninf> : f32
  %y = math.sqrt %b fastmath<nnan,ninf> : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @different_fastmath
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg1 fastmath<nnan> : f32
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg2 fastmath<ninf> : f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @different_fastmath(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = math.sqrt %a fastmath<nnan> : f32
  %y = math.sqrt %b fastmath<ninf> : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Overflow flags are operation properties and must match as well.
// CHECK-LABEL: @different_overflow
// CHECK-NEXT: %[[X:.*]] = arith.muli %arg1, %arg3 overflow<nsw> : i32
// CHECK-NEXT: %[[Y:.*]] = arith.muli %arg2, %arg3 overflow<nuw> : i32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : i32
// CHECK-NEXT: return %[[R]]
func.func @different_overflow(%c: i1, %a: i32, %b: i32, %k: i32) -> i32 {
  %x = arith.muli %a, %k overflow<nsw> : i32
  %y = arith.muli %b, %k overflow<nuw> : i32
  %r = arith.select %c, %x, %y : i32
  return %r : i32
}

// CHECK-LABEL: @multiple_uses_true
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg1 : f32
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]], %[[X]]
func.func @multiple_uses_true(%c: i1, %a: f32, %b: f32) -> (f32, f32) {
  %x = math.sqrt %a : f32
  %y = math.sqrt %b : f32
  %r = arith.select %c, %x, %y : f32
  return %r, %x : f32, f32
}

// CHECK-LABEL: @multiple_uses_false
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg1 : f32
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]], %[[Y]]
func.func @multiple_uses_false(%c: i1, %a: f32, %b: f32) -> (f32, f32) {
  %x = math.sqrt %a : f32
  %y = math.sqrt %b : f32
  %r = arith.select %c, %x, %y : f32
  return %r, %y : f32, f32
}

// CHECK-LABEL: @three_distinct_pairs
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg4 : f32
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg2, %arg5 : f32
// CHECK-NEXT: %[[S2:.*]] = arith.select %arg0, %arg3, %arg6 : f32
// CHECK-NEXT: %[[R:.*]] = math.fma %[[S0]], %[[S1]], %[[S2]] : f32
// CHECK-NEXT: return %[[R]]
func.func @three_distinct_pairs(%c: i1, %a: f32, %b: f32, %d: f32, %e: f32, %f: f32, %g: f32) -> f32 {
  %x = math.fma %a, %b, %d : f32
  %y = math.fma %e, %f, %g : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Memory-effect freedom alone is insufficient: dynamic integer division cannot
// be speculated because the divisor can be zero.
// CHECK-LABEL: @not_speculatable
// CHECK-NEXT: %[[X:.*]] = arith.divsi %arg1, %arg3 : i32
// CHECK-NEXT: %[[Y:.*]] = arith.divsi %arg2, %arg3 : i32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : i32
// CHECK-NEXT: return %[[R]]
func.func @not_speculatable(%c: i1, %a: i32, %b: i32, %k: i32) -> i32 {
  %x = arith.divsi %a, %k : i32
  %y = arith.divsi %b, %k : i32
  %r = arith.select %c, %x, %y : i32
  return %r : i32
}

// CHECK-LABEL: @different_ops
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg1 : f32
// CHECK-NEXT: %[[Y:.*]] = math.exp %arg2 : f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @different_ops(%c: i1, %a: f32, %b: f32) -> f32 {
  %x = math.sqrt %a : f32
  %y = math.exp %b : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @scalar_condition_vector
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : vector<4xf32>
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] : vector<4xf32>
// CHECK-NEXT: return %[[R]]
func.func @scalar_condition_vector(%c: i1, %a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %x = math.sqrt %a : vector<4xf32>
  %y = math.sqrt %b : vector<4xf32>
  %r = arith.select %c, %x, %y : vector<4xf32>
  return %r : vector<4xf32>
}

// CHECK-LABEL: @dynamic_tensor
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<?x4xi1>, tensor<?x4xf32>
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] : tensor<?x4xf32>
// CHECK-NEXT: return %[[R]]
func.func @dynamic_tensor(%c: tensor<?x4xi1>, %a: tensor<?x4xf32>, %b: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %x = math.sqrt %a : tensor<?x4xf32>
  %y = math.sqrt %b : tensor<?x4xf32>
  %r = arith.select %c, %x, %y : tensor<?x4xi1>, tensor<?x4xf32>
  return %r : tensor<?x4xf32>
}

// CHECK-LABEL: @scalable_vector
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : vector<2x[4]xi1>, vector<2x[4]xf32>
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] : vector<2x[4]xf32>
// CHECK-NEXT: return %[[R]]
func.func @scalable_vector(%c: vector<2x[4]xi1>, %a: vector<2x[4]xf32>, %b: vector<2x[4]xf32>) -> vector<2x[4]xf32> {
  %x = math.sqrt %a : vector<2x[4]xf32>
  %y = math.sqrt %b : vector<2x[4]xf32>
  %r = arith.select %c, %x, %y : vector<2x[4]xi1>, vector<2x[4]xf32>
  return %r : vector<2x[4]xf32>
}

// CHECK-LABEL: @tensor_encoding
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : tensor<4xi1, "layout">, tensor<4xf32, "layout">
// CHECK-NEXT: %[[R:.*]] = math.sqrt %[[S]] : tensor<4xf32, "layout">
// CHECK-NEXT: return %[[R]]
func.func @tensor_encoding(%c: tensor<4xi1, "layout">, %a: tensor<4xf32, "layout">, %b: tensor<4xf32, "layout">) -> tensor<4xf32, "layout"> {
  %x = math.sqrt %a : tensor<4xf32, "layout">
  %y = math.sqrt %b : tensor<4xf32, "layout">
  %r = arith.select %c, %x, %y : tensor<4xi1, "layout">, tensor<4xf32, "layout">
  return %r : tensor<4xf32, "layout">
}

// Greedy canonicalization must remove dead conversions and then factor the
// muls, without relying on a subsequent CSE pass.
// CHECK-LABEL: @conversion_multiply_chain
// CHECK-NEXT: %[[S0:.*]] = arith.select %arg0, %arg1, %arg3 : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: %[[S1:.*]] = arith.select %arg0, %arg2, %arg4 : tensor<4xi1>, tensor<4xf32>
// CHECK-NEXT: %[[M:.*]] = arith.mulf %[[S0]], %[[S1]] : tensor<4xf32>
// CHECK-NEXT: %[[R:.*]] = arith.fptosi %[[M]] : tensor<4xf32> to tensor<4xi8>
// CHECK-NEXT: return %[[R]]
func.func @conversion_multiply_chain(%c: tensor<4xi1>, %a: tensor<4xf32>, %alpha: tensor<4xf32>, %b: tensor<4xf32>, %beta: tensor<4xf32>) -> tensor<4xi8> {
  %x = arith.mulf %a, %alpha : tensor<4xf32>
  %y = arith.mulf %b, %beta : tensor<4xf32>
  %ix = arith.fptosi %x : tensor<4xf32> to tensor<4xi8>
  %iy = arith.fptosi %y : tensor<4xf32> to tensor<4xi8>
  %r = arith.select %c, %ix, %iy : tensor<4xi1>, tensor<4xi8>
  return %r : tensor<4xi8>
}

// Select producers can also be factored through the outer select.
// CHECK-LABEL: @nested_selects
// CHECK-NEXT: %[[C:.*]] = arith.select %arg0, %arg1, %arg2 : i1
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg4, %arg5 : f32
// CHECK-NEXT: %[[R:.*]] = arith.select %[[C]], %arg3, %[[S]] : f32
// CHECK-NEXT: return %[[R]]
func.func @nested_selects(%c: i1, %d: i1, %e: i1, %a: f32, %b: f32, %k: f32) -> f32 {
  %x = arith.select %d, %a, %b : f32
  %y = arith.select %e, %a, %k : f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @memory_reads
// CHECK-NEXT: %[[X:.*]] = memref.load %arg1[%arg3] : memref<?xf32>
// CHECK-NEXT: %[[Y:.*]] = memref.load %arg2[%arg3] : memref<?xf32>
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @memory_reads(%c: i1, %a: memref<?xf32>, %b: memref<?xf32>, %i: index) -> f32 {
  %x = memref.load %a[%i] : memref<?xf32>
  %y = memref.load %b[%i] : memref<?xf32>
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// Pure reductions are not elementwise.
// CHECK-LABEL: @reductions
// CHECK-NEXT: %[[X:.*]] = vector.reduction <add>, %arg1 : vector<4xf32> into f32
// CHECK-NEXT: %[[Y:.*]] = vector.reduction <add>, %arg2 : vector<4xf32> into f32
// CHECK-NEXT: %[[R:.*]] = arith.select %arg0, %[[X]], %[[Y]] : f32
// CHECK-NEXT: return %[[R]]
func.func @reductions(%c: i1, %a: vector<4xf32>, %b: vector<4xf32>) -> f32 {
  %x = vector.reduction <add>, %a : vector<4xf32> into f32
  %y = vector.reduction <add>, %b : vector<4xf32> into f32
  %r = arith.select %c, %x, %y : f32
  return %r : f32
}

// CHECK-LABEL: @same_overflow
// CHECK-NEXT: %[[S:.*]] = arith.select %arg0, %arg1, %arg2 : i32
// CHECK-NEXT: %[[R:.*]] = arith.muli %[[S]], %arg3 overflow<nsw> : i32
// CHECK-NEXT: return %[[R]]
func.func @same_overflow(%c: i1, %a: i32, %b: i32, %k: i32) -> i32 {
  %x = arith.muli %a, %k overflow<nsw> : i32
  %y = arith.muli %b, %k overflow<nsw> : i32
  %r = arith.select %c, %x, %y : i32
  return %r : i32
}

// Single-use producers outside a loop must not be sunk into the loop.
// CHECK-LABEL: @producers_outside_loop
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg4 : f32
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg5 : f32
// CHECK-NEXT: %[[LOOP:.*]] = scf.for %[[I:.*]] = %arg0 to %arg1 step %arg2 iter_args(%[[ACC:.*]] = %arg6) -> (f32) {
// CHECK-NEXT: %[[C:.*]] = arith.cmpi slt, %[[I]], %arg3 : index
// CHECK-NEXT: %[[S:.*]] = arith.select %[[C]], %[[X]], %[[Y]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[ACC]], %[[S]] : f32
// CHECK-NEXT: scf.yield %[[SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]] : f32
func.func @producers_outside_loop(%lb: index, %ub: index, %step: index, %split: index, %a: f32, %b: f32, %init: f32) -> f32 {
  %x = math.sqrt %a : f32
  %y = math.sqrt %b : f32
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %c = arith.cmpi slt, %i, %split : index
    %r = arith.select %c, %x, %y : f32
    %sum = arith.addf %acc, %r : f32
    scf.yield %sum : f32
  }
  return %result : f32
}

// Check the block of each producer, not just whether the producers share one.
// CHECK-LABEL: @false_producer_outside_loop
// CHECK-NEXT: %[[Y:.*]] = math.sqrt %arg5 : f32
// CHECK-NEXT: %[[LOOP:.*]] = scf.for %[[I:.*]] = %arg0 to %arg1 step %arg2 iter_args(%[[ACC:.*]] = %arg6) -> (f32) {
// CHECK-NEXT: %[[X:.*]] = math.sqrt %arg4 : f32
// CHECK-NEXT: %[[C:.*]] = arith.cmpi slt, %[[I]], %arg3 : index
// CHECK-NEXT: %[[S:.*]] = arith.select %[[C]], %[[X]], %[[Y]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[ACC]], %[[S]] : f32
// CHECK-NEXT: scf.yield %[[SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]] : f32
func.func @false_producer_outside_loop(%lb: index, %ub: index, %step: index, %split: index, %a: f32, %b: f32, %init: f32) -> f32 {
  %y = math.sqrt %b : f32
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %x = math.sqrt %a : f32
    %c = arith.cmpi slt, %i, %split : index
    %r = arith.select %c, %x, %y : f32
    %sum = arith.addf %acc, %r : f32
    scf.yield %sum : f32
  }
  return %result : f32
}

// Factoring inside a loop is allowed when all three operations share a block.
// CHECK-LABEL: @producers_inside_loop
// CHECK-NEXT: %[[LOOP:.*]] = scf.for %[[I:.*]] = %arg0 to %arg1 step %arg2 iter_args(%[[ACC:.*]] = %arg6) -> (f32) {
// CHECK-NEXT: %[[C:.*]] = arith.cmpi slt, %[[I]], %arg3 : index
// CHECK-NEXT: %[[S:.*]] = arith.select %[[C]], %arg4, %arg5 : f32
// CHECK-NEXT: %[[SQRT:.*]] = math.sqrt %[[S]] : f32
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[ACC]], %[[SQRT]] : f32
// CHECK-NEXT: scf.yield %[[SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LOOP]] : f32
func.func @producers_inside_loop(%lb: index, %ub: index, %step: index, %split: index, %a: f32, %b: f32, %init: f32) -> f32 {
  %result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (f32) {
    %x = math.sqrt %a : f32
    %y = math.sqrt %b : f32
    %c = arith.cmpi slt, %i, %split : index
    %r = arith.select %c, %x, %y : f32
    %sum = arith.addf %acc, %r : f32
    scf.yield %sum : f32
  }
  return %result : f32
}
