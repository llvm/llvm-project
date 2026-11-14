// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// The old passthrough can be poison because the select overwrites every
// inactive lane.
// CHECK-LABEL: func.func @fold_select_into_maskedload(
// CHECK-SAME: %[[MASK:[^:]+]]: vector<4xi1>,
// CHECK: %[[NEW:.*]] = arith.addf
// CHECK: %[[LOAD:.*]] = vector.maskedload {{.*}}, %[[MASK]], %[[NEW]]
// CHECK-NEXT: return %[[LOAD]]
func.func @fold_select_into_maskedload(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %a: vector<4xf32>, %b: vector<4xf32>) -> vector<4xf32> {
  %old = ub.poison : vector<4xf32>
  %new = arith.addf %a, %b : vector<4xf32>
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// Sinking the load past the store would change the values of active lanes.
// CHECK-LABEL: func.func @fold_select_before_aliasing_store(
// CHECK-SAME: %[[BASE:[^:]+]]: memref<?xf32>, %[[IDX:[^:]+]]: index,
// CHECK-SAME: %[[MASK:[^:]+]]: vector<4xi1>, %{{[^:]+}}: vector<4xf32>, %[[NEW:[^:]+]]: vector<4xf32>
// CHECK-NEXT: %[[LOAD:.*]] = vector.maskedload %[[BASE]][%[[IDX]]], %[[MASK]], %[[NEW]] alignment = 16
// CHECK-NEXT: vector.store %[[NEW]], %[[BASE]][%[[IDX]]]
// CHECK-NEXT: return %[[LOAD]]
func.func @fold_select_before_aliasing_store(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %old: vector<4xf32>, %new: vector<4xf32>) -> vector<4xf32> {
  %load = vector.maskedload %base[%idx], %mask, %old alignment = 16
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  vector.store %new, %base[%idx] : memref<?xf32>, vector<4xf32>
  %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @fold_select_scalable_mask(
// CHECK-SAME: %[[MASK:[^:]+]]: vector<[4]xi1>, %{{[^:]+}}: vector<[4]xi32>, %[[NEW:[^:]+]]: vector<[4]xi32>
// CHECK: %[[LOAD:.*]] = vector.maskedload {{.*}}, %[[MASK]], %[[NEW]]
// CHECK-NEXT: return %[[LOAD]]
func.func @fold_select_scalable_mask(%base: memref<?xi32>, %idx: index,
    %mask: vector<[4]xi1>, %old: vector<[4]xi32>,
    %new: vector<[4]xi32>) -> vector<[4]xi32> {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xi32>, vector<[4]xi1>, vector<[4]xi32> into vector<[4]xi32>
  %result = arith.select %mask, %load, %new : vector<[4]xi1>, vector<[4]xi32>
  return %result : vector<[4]xi32>
}

// -----

// CHECK-LABEL: func.func @no_fold_select_different_mask(
// CHECK: %[[LOAD:.*]] = vector.maskedload
// CHECK: arith.select {{.*}}, %[[LOAD]],
func.func @no_fold_select_different_mask(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %other: vector<4xi1>, %old: vector<4xf32>,
    %new: vector<4xf32>) -> vector<4xf32> {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %result = arith.select %other, %load, %new : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @no_fold_select_multiple_load_uses(
// CHECK-SAME: %[[OLD:[^:]+]]: vector<4xf32>,
// CHECK: %[[LOAD:.*]] = vector.maskedload {{.*}}, %[[OLD]] :
// CHECK: %[[SELECT:.*]] = arith.select {{.*}}, %[[LOAD]],
// CHECK: return %[[SELECT]], %[[LOAD]]
func.func @no_fold_select_multiple_load_uses(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %old: vector<4xf32>,
    %new: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
  return %result, %load : vector<4xf32>, vector<4xf32>
}

// -----

// The replacement passthrough must be available at the original load.
// CHECK-LABEL: func.func @no_fold_select_late_passthrough(
// CHECK: %[[LOAD:.*]] = vector.maskedload
// CHECK: %[[NEW:.*]] = arith.addf
// CHECK: arith.select {{.*}}, %[[LOAD]], %[[NEW]]
func.func @no_fold_select_late_passthrough(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %old: vector<4xf32>, %a: vector<4xf32>,
    %b: vector<4xf32>) -> vector<4xf32> {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %new = arith.addf %a, %b : vector<4xf32>
  %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @no_fold_select_load_on_false_arm(
// CHECK: %[[LOAD:.*]] = vector.maskedload
// CHECK: arith.select {{.*}}, %[[LOAD]] :
func.func @no_fold_select_load_on_false_arm(%base: memref<?xf32>, %idx: index,
    %mask: vector<4xi1>, %old: vector<4xf32>, %other: vector<4xf32>) -> vector<4xf32> {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %result = arith.select %mask, %other, %load : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// Fold before the select's broadcast condition is canonicalized to a scalar.
// CHECK-LABEL: func.func @fold_select_broadcast_mask(
// CHECK-SAME: %[[COND:[^:]+]]: i1, %{{[^:]+}}: vector<4xf32>, %[[NEW:[^:]+]]: vector<4xf32>
// CHECK: %[[MASK:.*]] = vector.broadcast %[[COND]]
// CHECK: %[[LOAD:.*]] = vector.maskedload {{.*}}, %[[MASK]], %[[NEW]]
// CHECK-NEXT: return %[[LOAD]]
func.func @fold_select_broadcast_mask(%base: memref<?xf32>, %idx: index,
    %cond: i1, %old: vector<4xf32>, %new: vector<4xf32>) -> vector<4xf32> {
  %mask = vector.broadcast %cond : i1 to vector<4xi1>
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
  return %result : vector<4xf32>
}

// -----

// A rewrite scoped to the if region cannot update the enclosing load.
// CHECK-LABEL: func.func @no_fold_select_in_nested_region(
// CHECK-SAME: %[[MASK:[^:]+]]: vector<4xi1>, %[[OLD:[^:]+]]: vector<4xf32>, %[[NEW:[^:]+]]: vector<4xf32>
// CHECK: %[[LOAD:.*]] = vector.maskedload {{.*}}, %[[MASK]], %[[OLD]]
// CHECK: scf.if
// CHECK: %[[SELECT:.*]] = arith.select %[[MASK]], %[[LOAD]], %[[NEW]]
// CHECK: vector.store %[[SELECT]],
func.func @no_fold_select_in_nested_region(%base: memref<?xf32>, %idx: index,
    %cond: i1, %mask: vector<4xi1>, %old: vector<4xf32>, %new: vector<4xf32>) {
  %load = vector.maskedload %base[%idx], %mask, %old
      : memref<?xf32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  scf.if %cond {
    %result = arith.select %mask, %load, %new : vector<4xi1>, vector<4xf32>
    vector.store %result, %base[%idx] : memref<?xf32>, vector<4xf32>
  }
  return
}
