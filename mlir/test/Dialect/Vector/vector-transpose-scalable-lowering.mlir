// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// Check the deinterleave direction, including a tail because 37 is not a
// multiple of the minimum four f32 lanes in a 128-bit scalable vector.
// CHECK-LABEL: func.func @rows_to_columns(
// CHECK-NOT: vector.transpose
// CHECK: %[[RESULT:.*]] = memref.alloca_scope -> (vector<2x37xf32>) {
// CHECK:   %[[INPUT:.*]] = memref.alloca() : memref<74xf32>
// CHECK:   %[[OUTPUT:.*]] = memref.alloca() : memref<74xf32>
// CHECK:   %[[VSCALE:.*]] = vector.vscale
// CHECK:   %[[STEP:.*]] = arith.muli %[[VSCALE]], %{{.*}} : index
// CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %[[STEP]] {
// CHECK:     %[[PAIR:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<74xf32>, vector<[8]xi1>, vector<[8]xf32>
// CHECK:     %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave %[[PAIR]] : vector<[8]xf32> -> vector<[4]xf32>
// CHECK:     vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[EVEN]] : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:     vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[ODD]] : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:   }
// CHECK:   memref.alloca_scope.return %{{.*}} : vector<2x37xf32>
// CHECK: }
// CHECK: return %[[RESULT]] : vector<2x37xf32>
func.func @rows_to_columns(%arg: vector<37x2xf32>) -> vector<2x37xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<37x2xf32> to vector<2x37xf32>
  return %result : vector<2x37xf32>
}

// Check the inverse interleave direction.
// CHECK-LABEL: func.func @columns_to_rows(
// CHECK-NOT: vector.transpose
// CHECK: memref.alloca_scope -> (vector<37x2xf32>) {
// CHECK:   %[[INPUT:.*]] = memref.alloca() : memref<74xf32>
// CHECK:   %[[OUTPUT:.*]] = memref.alloca() : memref<74xf32>
// CHECK:   scf.for
// CHECK:     %[[LHS:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:     %[[RHS:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:     %[[PAIR:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<[4]xf32> -> vector<[8]xf32>
// CHECK:     vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[PAIR]] : memref<74xf32>, vector<[8]xi1>, vector<[8]xf32>
func.func @columns_to_rows(%arg: vector<2x37xf32>) -> vector<37x2xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<2x37xf32> to vector<37x2xf32>
  return %result : vector<37x2xf32>
}

// Leading dimensions are flattened into a batch loop. Each batch still uses
// the same scalable deinterleave kernel for the final two dimensions.
// CHECK-LABEL: func.func @batched_rows_to_columns(
// CHECK-NOT: vector.transpose
// CHECK: memref.alloca_scope -> (vector<2x5x2x37xf32>) {
// CHECK:   %[[INPUT:.*]] = memref.alloca() : memref<740xf32>
// CHECK:   %[[OUTPUT:.*]] = memref.alloca() : memref<740xf32>
// CHECK:   scf.for %[[BATCH:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     %[[BASE:.*]] = arith.muli %[[BATCH]], %{{.*}} : index
// CHECK:     scf.for
// CHECK:       %[[PAIR:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<740xf32>, vector<[8]xi1>, vector<[8]xf32>
// CHECK:       %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave %[[PAIR]] : vector<[8]xf32> -> vector<[4]xf32>
// CHECK:       vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[EVEN]] : memref<740xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:       vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[ODD]] : memref<740xf32>, vector<[4]xi1>, vector<[4]xf32>
func.func @batched_rows_to_columns(
    %arg: vector<2x5x37x2xf32>) -> vector<2x5x2x37xf32> {
  %result = vector.transpose %arg, [0, 1, 3, 2]
      : vector<2x5x37x2xf32> to vector<2x5x2x37xf32>
  return %result : vector<2x5x2x37xf32>
}

// Check the inverse batched interleave direction.
// CHECK-LABEL: func.func @batched_columns_to_rows(
// CHECK-NOT: vector.transpose
// CHECK: memref.alloca_scope -> (vector<3x37x2xf32>) {
// CHECK:   %[[INPUT:.*]] = memref.alloca() : memref<222xf32>
// CHECK:   %[[OUTPUT:.*]] = memref.alloca() : memref<222xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       %[[LHS:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<222xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:       %[[RHS:.*]] = vector.maskedload %[[INPUT]][%{{.*}}], %{{.*}}, %{{.*}} : memref<222xf32>, vector<[4]xi1>, vector<[4]xf32>
// CHECK:       %[[PAIR:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<[4]xf32> -> vector<[8]xf32>
// CHECK:       vector.maskedstore %[[OUTPUT]][%{{.*}}], %{{.*}}, %[[PAIR]] : memref<222xf32>, vector<[8]xi1>, vector<[8]xf32>
func.func @batched_columns_to_rows(
    %arg: vector<3x2x37xf32>) -> vector<3x37x2xf32> {
  %result = vector.transpose %arg, [0, 2, 1]
      : vector<3x2x37xf32> to vector<3x37x2xf32>
  return %result : vector<3x37x2xf32>
}

// The binary interleave/deinterleave implementation requires one dimension to
// be exactly 2, so a 7x3 transpose must remain unchanged.
// CHECK-LABEL: func.func @no_dimension_of_size_two(
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<7x3xf32> to vector<3x7xf32>
func.func @no_dimension_of_size_two(%arg: vector<7x3xf32>) -> vector<3x7xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<7x3xf32> to vector<3x7xf32>
  return %result : vector<3x7xf32>
}

// A dimension of size 2 is insufficient when dimensions other than the final
// two are transposed.
// CHECK-LABEL: func.func @non_trailing_transpose(
// CHECK: vector.transpose %{{.*}}, [1, 0, 2] : vector<2x3x4xf32> to vector<3x2x4xf32>
func.func @non_trailing_transpose(
    %arg: vector<2x3x4xf32>) -> vector<3x2x4xf32> {
  %result = vector.transpose %arg, [1, 0, 2]
      : vector<2x3x4xf32> to vector<3x2x4xf32>
  return %result : vector<3x2x4xf32>
}

// Already-scalable inputs are outside the scope of this fixed-to-scalable
// lowering.
// CHECK-LABEL: func.func @already_scalable(
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
func.func @already_scalable(%arg: vector<2x[4]xf32>) -> vector<[4]x2xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  return %result : vector<[4]x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.lower_transpose_to_scalable
        target_vector_bitwidth = 128
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// Check that the target vector width is configurable rather than implicitly
// assuming SVE's 128-bit minimum. With 256 bits, f32 chunks have eight lanes.
// CHECK-LABEL: func.func @target_width_256(
// CHECK: vector.maskedload {{.*}} : memref<38xf32>, vector<[16]xi1>, vector<[16]xf32>
// CHECK: vector.deinterleave {{.*}} : vector<[16]xf32> -> vector<[8]xf32>
func.func @target_width_256(%arg: vector<19x2xf32>) -> vector<2x19xf32> {
  %result = vector.transpose %arg, [1, 0] : vector<19x2xf32> to vector<2x19xf32>
  return %result : vector<2x19xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.lower_transpose_to_scalable
        target_vector_bitwidth = 256
    } : !transform.op<"func.func">
    transform.yield
  }
}
