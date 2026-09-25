// RUN: mlir-opt %s -split-input-file -test-eliminate-vector-masks | FileCheck %s --check-prefixes=ALL,WITH-RANGE
// RUN: mlir-opt %s -split-input-file -test-eliminate-vector-masks=fixed-size | FileCheck %s --check-prefixes=ALL,NO-RANGE

// Each scalable test below is paired with a fixed-size equivalent, so that both
// widths get the same coverage. Only the one case that needs a vscale range to
// be resolved differs between the two runs; everything else behaves identically
// with and without one.

// This tests a general pattern the vectorizer tends to emit.

// ALL-LABEL: @eliminate_redundant_masks_through_insert_and_extracts
//  WITH-RANGE: %[[ALL_TRUE_MASK:.*]] = vector.constant_mask [4] : vector<[4]xi1>
//  WITH-RANGE: vector.transfer_read {{.*}} %[[ALL_TRUE_MASK]]
//  WITH-RANGE: vector.mask %[[ALL_TRUE_MASK:.*]] {
//  WITH-RANGE-SAME:  vector.outerproduct
//  WITH-RANGE: vector.transfer_write {{.*}} %[[ALL_TRUE_MASK]]
// The mask dimension is scalable, so without a vscale range it cannot be
// resolved.
//  NO-RANGE-NOT: vector.constant_mask
//  NO-RANGE: vector.create_mask
#map = affine_map<()[s0] -> (-(1080 mod s0) + 1080)>

func.func @eliminate_redundant_masks_through_insert_and_extracts(%tensor: tensor<1x1000xf32>, %rhs : f32) {
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %ub = affine.apply #map()[%c4_vscale]

  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c0_f32 = arith.constant 0.0 : f32
  %extracted_slice_0 = tensor.extract_slice %tensor[0, 0] [1, %c4_vscale] [1, 1] : tensor<1x1000xf32> to tensor<1x?xf32>
  %output_tensor = scf.for %i = %c0 to %ub step %c4_vscale iter_args(%arg = %extracted_slice_0) -> tensor<1x?xf32> {
    // 1. Extract a slice.
    %extracted_slice_1 = tensor.extract_slice %arg[0, %i] [1, %c4_vscale] [1, 1] : tensor<1x?xf32> to tensor<?xf32>

    // 2. Create a mask for the slice.
    %dim_1 = tensor.dim %extracted_slice_1, %c0 : tensor<?xf32>
    %mask = vector.create_mask %dim_1 : vector<[4]xi1>

    // 3. Read the slice and do some computation.
    %lhs = vector.transfer_read %extracted_slice_1[%c0], %c0_f32, %mask {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32>
    %new_vec = vector.mask %mask { vector.outerproduct %lhs, %rhs {kind = #vector.kind<add>} : vector<[4]xf32>, f32 } : vector<[4]xi1> -> vector<[4]xf32>

    // 4. Write the new value.
    %write = vector.transfer_write %new_vec, %extracted_slice_1[%c0], %mask {in_bounds = [true]} : vector<[4]xf32>, tensor<?xf32>

    // 5. Insert and yield the new tensor value.
    %result = tensor.insert_slice %write into %arg[0, %i] [1, %c4_vscale] [1, 1] : tensor<?xf32> into tensor<1x?xf32>
    scf.yield %result : tensor<1x?xf32>
  }
  "test.some_use"(%output_tensor) : (tensor<1x?xf32>) -> ()
  return
}

// -----

// Fixed-size equivalent of the above. The mask bound comes from the induction
// variable, so it is not constant and no fold removes it, but value bounds
// proves `%i <= 1020`, hence `1024 - %i >= 4`.

// ALL-LABEL: @eliminate_redundant_masks_fixed_size
//       ALL: %[[ALL_TRUE_MASK:.*]] = vector.constant_mask [4] : vector<4xi1>
//       ALL: vector.transfer_read {{.*}}, %[[ALL_TRUE_MASK]]
func.func @eliminate_redundant_masks_fixed_size(%tensor: tensor<1024xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1024 = arith.constant 1024 : index
  %c0_f32 = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c1024 step %c4 iter_args(%acc = %c0_f32) -> f32 {
    %rem = arith.subi %c1024, %i : index
    %mask = vector.create_mask %rem : vector<4xi1>
    %vec = vector.transfer_read %tensor[%i], %c0_f32, %mask : tensor<1024xf32>, vector<4xf32>
    %sum = vector.reduction <add>, %vec : vector<4xf32> into f32
    %new_acc = arith.addf %acc, %sum : f32
    scf.yield %new_acc : f32
  }
  return %result : f32
}

// -----

// Test to ensure that functions without a body are skipped.
// ALL-LABEL: func.func private @negative_no_func_body()
func.func private @negative_no_func_body()

// -----

// ALL-LABEL: @negative_extract_slice_size_shrink
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<[4]xi1>) -> ()
func.func @negative_extract_slice_size_shrink(%tensor: tensor<1000xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1000 = arith.constant 1000 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  %extracted_slice = tensor.extract_slice %tensor[0] [%c4_vscale] [1] : tensor<1000xf32> to tensor<?xf32>
  %slice = scf.for %i = %c0 to %c1000 step %c4_vscale iter_args(%arg = %extracted_slice) -> tensor<?xf32> {
    // This mask cannot be eliminated even though looking at the operations above
    // (this comment) it appears `tensor.dim` will always be c4_vscale (so the mask all-true).
    %dim = tensor.dim %arg, %c0 : tensor<?xf32>
    %mask = vector.create_mask %dim : vector<[4]xi1>
    "test.some_use"(%mask) : (vector<[4]xi1>) -> ()
    // !!! Here the size of the mask could shrink in the next iteration.
    %next_num_elts = affine.min  affine_map<(d0)[s0] -> (-d0 + 1000, s0)>(%i)[%c4_vscale]
    %new_extracted_slice = tensor.extract_slice %tensor[%c4_vscale] [%next_num_elts] [1] : tensor<1000xf32> to tensor<?xf32>
    scf.yield %new_extracted_slice : tensor<?xf32>
  }
  "test.some_use"(%slice) : (tensor<?xf32>) -> ()
  return
}

// -----

// ALL-LABEL: @trivially_all_true_case
// ALL: %[[ALL_TRUE_MASK:.*]] = vector.constant_mask [2, 4] : vector<2x[4]xi1>
// ALL: "test.some_use"(%[[ALL_TRUE_MASK]]) : (vector<2x[4]xi1>) -> ()
func.func @trivially_all_true_case(%tensor: tensor<2x?xf32>)
{
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  // Is found to be all true _without_ value bounds analysis. A `vscale`
  // multiple is all-true whenever the multiplier is at least the mask dim
  // size, whatever `vscale` turns out to be, so no range is needed.
  %mask = vector.create_mask %c2, %c4_vscale : vector<2x[4]xi1>
  "test.some_use"(%mask) : (vector<2x[4]xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @trivially_all_true_case_fixed_size
// ALL: %[[ALL_TRUE_MASK:.*]] = vector.constant_mask [2, 4] : vector<2x4xi1>
// ALL: "test.some_use"(%[[ALL_TRUE_MASK]]) : (vector<2x4xi1>) -> ()
func.func @trivially_all_true_case_fixed_size()
{
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  // Also found to be all true _without_ value bounds analysis.
  %mask = vector.create_mask %c2, %c4 : vector<2x4xi1>
  "test.some_use"(%mask) : (vector<2x4xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_constant_dim_not_all_true
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<2x[4]xi1>) -> ()
func.func @negative_constant_dim_not_all_true()
{
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  // Since %c1 is a constant, this will be found not to be all-true via simple
  // pattern matching.
  %mask = vector.create_mask %c1, %c4_vscale : vector<2x[4]xi1>
  "test.some_use"(%mask) : (vector<2x[4]xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_constant_dim_not_all_true_fixed_size
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<2x4xi1>) -> ()
func.func @negative_constant_dim_not_all_true_fixed_size()
{
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // As above, found not to be all-true via simple pattern matching.
  %mask = vector.create_mask %c1, %c4 : vector<2x4xi1>
  "test.some_use"(%mask) : (vector<2x4xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_constant_vscale_multiple_not_all_true
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<2x[4]xi1>) -> ()
func.func @negative_constant_vscale_multiple_not_all_true() {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %vscale = vector.vscale
  %c3_vscale = arith.muli %vscale, %c3 : index
  // Since %c3_vscale is a constant vscale multiple, this will be found not to
  // be all-true via simple pattern matching. There is no fixed-size equivalent
  // of this case: it exercises the `vscale` multiple matching specifically.
  %mask = vector.create_mask %c2, %c3_vscale : vector<2x[4]xi1>
  "test.some_use"(%mask) : (vector<2x[4]xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_value_bounds_fixed_dim_not_all_true
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<3x[4]xi1>) -> ()
func.func @negative_value_bounds_fixed_dim_not_all_true(%tensor: tensor<2x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  // This is _very_ simple, but since tensor.dim is not a constant, value bounds
  // will be used to resolve it.
  %dim = tensor.dim %tensor, %c0 : tensor<2x?xf32>
  %mask = vector.create_mask %dim, %c4_vscale : vector<3x[4]xi1>
  "test.some_use"(%mask) : (vector<3x[4]xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_value_bounds_fixed_dim_not_all_true_fixed_size
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<3x4xi1>) -> ()
func.func @negative_value_bounds_fixed_dim_not_all_true_fixed_size(%tensor: tensor<2x?xf32>)
{
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // As above: a bound is found for %dim (2), but it is smaller than the mask
  // dimension (3).
  %dim = tensor.dim %tensor, %c0 : tensor<2x?xf32>
  %mask = vector.create_mask %dim, %c4 : vector<3x4xi1>
  "test.some_use"(%mask) : (vector<3x4xi1>) -> ()
  return
}

// -----

// ALL-LABEL: @negative_value_bounds_scalable_dim_not_all_true
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<3x[4]xi1>) -> ()
func.func @negative_value_bounds_scalable_dim_not_all_true(%tensor: tensor<2x100xf32>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %vscale = vector.vscale
  %c3_vscale = arith.muli %vscale, %c3 : index
  %slice = tensor.extract_slice %tensor[0, 0] [2, %c3_vscale] [1, 1] : tensor<2x100xf32> to tensor<2x?xf32>
  // Another simple example, but value bounds will be used to resolve the tensor.dim.
  // There is no fixed-size equivalent: the dimension under test is scalable.
  %dim = tensor.dim %slice, %c1 : tensor<2x?xf32>
  %mask = vector.create_mask %c3, %dim : vector<3x[4]xi1>
  "test.some_use"(%mask) : (vector<3x[4]xi1>) -> ()
  return
}

// -----

// Nothing bounds %n from below, so no bound is found at all.

// ALL-LABEL: @negative_unbounded_operand_fixed_size
// ALL-NOT: vector.constant_mask
// ALL: %[[MASK:.*]] = vector.create_mask
// ALL: "test.some_use"(%[[MASK]]) : (vector<4xi1>) -> ()
func.func @negative_unbounded_operand_fixed_size(%n: index) {
  %mask = vector.create_mask %n : vector<4xi1>
  "test.some_use"(%mask) : (vector<4xi1>) -> ()
  return
}

// -----

// A constant lower bound can never prove a scalable dimension all-true, in
// either run: `max(%n, 4) >= 4`, but `vector<[4]xi1>` holds `4 * vscale`
// elements, so the mask is only all-true when `vscale` is 1. Without the
// scalable-dimension check on the fixed-width path this is folded to an
// all-true mask, which is wrong for every `vscale > 1`.

// ALL-LABEL: @negative_scalable_dim_constant_lower_bound
//   ALL-NOT: vector.constant_mask
//       ALL: %[[MASK:.*]] = vector.create_mask
//       ALL: "test.some_use"(%[[MASK]]) : (vector<[4]xi1>) -> ()
func.func @negative_scalable_dim_constant_lower_bound(%n: index) {
  %c4 = arith.constant 4 : index
  // There is no fixed-size equivalent: the dimension under test is scalable.
  %m = arith.maxsi %n, %c4 : index
  %mask = vector.create_mask %m : vector<[4]xi1>
  "test.some_use"(%mask) : (vector<[4]xi1>) -> ()
  return
}
