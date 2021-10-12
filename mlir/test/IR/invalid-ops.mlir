// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func @dim(%arg : tensor<1x?xf32>) {
  %c2 = arith.constant 2 : index
  tensor.dim %arg, %c2 : tensor<1x?xf32> // expected-error {{'tensor.dim' op index is out of range}}
  return
}

// -----

func @rank(f32) {
^bb(%0: f32):
  "std.rank"(%0): (f32)->index // expected-error {{'std.rank' op operand #0 must be any memref or tensor type}}

  return
}

// -----
func @affine_apply_no_map() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{requires attribute 'map'}}
  return
}

// -----

func @affine_apply_wrong_operand_count() {
^bb0:
  %i = arith.constant 0 : index
  %x = "affine.apply" (%i) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index) -> (index) //  expected-error {{'affine.apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

func @affine_apply_wrong_result_count() {
^bb0:
  %i = arith.constant 0 : index
  %j = arith.constant 1 : index
  %x = "affine.apply" (%i, %j) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index,index) -> (index) //  expected-error {{'affine.apply' op mapping must produce one value}}
  return
}

// -----

func @unknown_custom_op() {
^bb0:
  %i = test.crazyThing() {value = 0} : () -> index  // expected-error {{custom op 'test.crazyThing' is unknown}}
  return
}

// -----

func @unknown_std_op() {
  // expected-error@+1 {{unregistered operation 'std.foo_bar_op' found in dialect ('std') that does not allow unknown operations}}
  %0 = "std.foo_bar_op"() : () -> index
  return
}

// -----

func @bad_alloc_wrong_dynamic_dim_count() {
^bb0:
  %0 = arith.constant 7 : index
  // Test alloc with wrong number of dynamic dimensions.
  // expected-error@+1 {{dimension operand count does not equal memref dynamic dimension count}}
  %1 = memref.alloc(%0)[%0] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = arith.constant 7 : index
  // Test alloc with wrong number of symbols
  // expected-error@+1 {{symbol operand count does not equal memref symbol count}}
  %1 = memref.alloc(%0) : memref<2x?xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func @test_store_zero_results() {
^bb0:
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  %1 = arith.constant 0 : index
  %2 = arith.constant 1 : index
  %3 = memref.load %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  // Test that store returns zero results.
  %4 = memref.store %3, %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1> // expected-error {{cannot name an operation with no results}}
  return
}

// -----

func @test_store_zero_results2(%x: i32, %p: memref<i32>) {
  "memref.store"(%x,%p) : (i32, memref<i32>) -> i32  // expected-error {{'memref.store' op requires zero results}}
  return
}

// -----

func @test_alloc_memref_map_rank_mismatch() {
^bb0:
  %0 = memref.alloc() : memref<1024x64xf32, affine_map<(d0) -> (d0)>, 1> // expected-error {{memref affine map dimension mismatch}}
  return
}

// -----

func @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{incorrect number of operands for callee}}
  return
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+2 {{different type than prior uses}}
  // expected-note@-2 {{prior use here}}
  %r = select %cond, %t, %f : i32
}

// -----

func @func_with_ops(i32, i32, i32) {
^bb0(%cond : i32, %t : i32, %f : i32):
  // expected-error@+1 {{op operand #0 must be bool-like}}
  %r = "std.select"(%cond, %t, %f) : (i32, i32, i32) -> i32
}

// -----

func @func_with_ops(i1, i32, i64) {
^bb0(%cond : i1, %t : i32, %f : i64):
  // expected-error@+1 {{all of {true_value, false_value, result} have same type}}
  %r = "std.select"(%cond, %t, %f) : (i1, i32, i64) -> i32
}

// -----

func @func_with_ops(vector<12xi1>, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : vector<12xi1>, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.select"(%cond, %t, %f) : (vector<12xi1>, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

func @func_with_ops(tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) {
^bb0(%cond : tensor<12xi1>, %t : tensor<42xi32>, %f : tensor<42xi32>):
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.select"(%cond, %t, %f) : (tensor<12xi1>, tensor<42xi32>, tensor<42xi32>) -> tensor<42xi32>
}

// -----

func @return_not_in_function() {
  "foo.region"() ({
    // expected-error@+1 {{'std.return' op expects parent op 'builtin.func'}}
    return
  }): () -> ()
  return
}

// -----

func @invalid_splat(%v : f32) {
  splat %v : memref<8xf32>
  // expected-error@-1 {{must be vector of any type values or statically shaped tensor of any type values}}
  return
}

// -----

func @invalid_splat(%v : vector<8xf32>) {
  %w = splat %v : tensor<8xvector<8xf32>>
  // expected-error@-1 {{must be integer/index/float type}}
  return
}

// -----

func @invalid_splat(%v : f32) { // expected-note {{prior use here}}
  splat %v : vector<8xf64>
  // expected-error@-1 {{expects different type than prior uses}}
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{expects 1 offset operand}}
  %1 = memref.view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>>
  // expected-error@+1 {{unsupported map for base memref type}}
  %1 = memref.view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>> to
      memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{unsupported map for result memref type}}
  %1 = memref.view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0, d1, s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = memref.view %0[%arg2][%arg0, %arg1] :  memref<2048xi8, 2> to memref<?x?xf32, 1>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<2048xi8>
  // expected-error@+1 {{incorrect number of size operands for type}}
  %1 = memref.view %0[%arg2][%arg0]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed offsets rank to match mixed sizes rank (2 vs 3) so the rank of the result type is well-formed}}
  %1 = memref.subview %0[0, 0][2, 2, 2][1, 1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = memref.subview %0[0, 0, 0][2, 2, 2][1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [2, 2, 2], strides:[1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32, offset: 0, strides: [64, 4, 1], 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = memref.subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, offset: 0, strides: [64, 4, 1], 2> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * s0 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>>
  // expected-error@+1 {{is not strided}}
  %1 = memref.subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>> to
      memref<8x?x4xf32, offset: 0, strides: [?, 4, 1]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected <= 3 offset values}}
  %1 = memref.subview %0[%arg0, %arg1, 0, 0][%arg2, 0, 0, 0][1, 1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x?x4xf32, offset: 0, strides:[?, ?, 4]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result element type to be 'f32'}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4xi32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result rank to be smaller or equal to the source rank.}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4x3xi32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = memref.subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = memref.alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = memref.subview %0[0, 2, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : memref<?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected result type to be 'memref<?x1xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' or a rank-reduced version. (mismatch of result sizes)}}
  %0 = memref.subview %arg0[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32>
  return
}

// -----

func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2)>>' and result type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 128 + d1 * 32 + d2 * 2)>>' are cast incompatible}}
  %0 = memref.cast %arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]> to memref<12x4x16xf32, offset:0, strides:[128, 32, 2]>
  return
}

// -----

func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2)>>' and result type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2 + 16)>>' are cast incompatible}}
  %0 = memref.cast %arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]> to memref<12x4x16xf32, offset:16, strides:[64, 16, 1]>
  return
}

// -----

// incompatible element types
func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xi32>' are cast incompatible}}
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xi32>
  return
}

// -----

func @invalid_prefetch_rw(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{rw specifier has to be 'read' or 'write'}}
  memref.prefetch %0[%i], rw, locality<0>, data  : memref<10xf32>
  return
}

// -----

func @invalid_prefetch_cache_type(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{cache type has to be 'data' or 'instr'}}
  memref.prefetch %0[%i], read, locality<0>, false  : memref<10xf32>
  return
}

// -----

func @invalid_prefetch_locality_hint(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3}}
  memref.prefetch %0[%i], read, locality<5>, data  : memref<10xf32>
  return
}

// -----

// incompatible memory space
func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xf32, 1>' are cast incompatible}}
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xf32, 1>
  return
}

// -----

// unranked to unranked
func @invalid_memref_cast() {
  %0 = memref.alloc() : memref<2x5xf32, 0>
  %1 = memref.cast %0 : memref<2x5xf32, 0> to memref<*xf32, 0>
  // expected-error@+1 {{operand type 'memref<*xf32>' and result type 'memref<*xf32>' are cast incompatible}}
  %2 = memref.cast %1 : memref<*xf32, 0> to memref<*xf32, 0>
  return
}

// -----

func @atomic_rmw_idxs_rank_mismatch(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects the number of subscripts to be equal to memref rank}}
  %x = atomic_rmw addf %val, %I[%i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func @atomic_rmw_expects_float(%I: memref<16x10xi32>, %i : index, %val : i32) {
  // expected-error@+1 {{expects a floating-point type}}
  %x = atomic_rmw addf %val, %I[%i, %i] : (i32, memref<16x10xi32>) -> i32
  return
}

// -----

func @atomic_rmw_expects_int(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects an integer type}}
  %x = atomic_rmw addi %val, %I[%i, %i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func @generic_atomic_rmw_wrong_arg_num(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected single number of entry block arguments}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%arg0 : f32, %arg1 : f32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_wrong_arg_type(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected block argument of the same type result type}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : i32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_result_type_mismatch(%I: memref<10xf32>, %i : index) {
 // expected-error@+1 {{failed to verify that result type matches element type of memref}}
 %0 = "std.generic_atomic_rmw"(%I, %i) ( {
    ^bb0(%old_value: f32):
      %c1 = arith.constant 1.0 : f32
      atomic_yield %c1 : f32
    }) : (memref<10xf32>, index) -> i32
  return
}

// -----

func @generic_atomic_rmw_has_side_effects(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{should contain only operations with no side effects}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1.0 : f32
      %buf = memref.alloc() : memref<2048xf32>
      atomic_yield %c1 : f32
  }
}

// -----

func @atomic_yield_type_mismatch(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{op types mismatch between yield op: 'i32' and its parent: 'f32'}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = arith.constant 1 : i32
      atomic_yield %c1 : i32
  }
  return
}

// -----

// alignment is not power of 2.
func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{alignment must be power of 2}}
  memref.assume_alignment %0, 12 : memref<4x4xf16>
  return
}

// -----

// 0 alignment value.
func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  memref.assume_alignment %0, 0 : memref<4x4xf16>
  return
}

// -----

"alloca_without_scoped_alloc_parent"() ( {
  memref.alloca() : memref<1xf32>
  // expected-error@-1 {{requires an ancestor op with AutomaticAllocationScope trait}}
  return
}) : () -> ()

// -----

func @slice_wrong_dynamic_type(%t: tensor<8x16x4xf32>, %idx : index) {
      // expected-error @+1 {{expected result type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (mismatch of result sizes)}}
  %0 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<?x4x4xf32>

  return
}

// -----

func @slice_wrong_static_type(%t: tensor<8x16x4xf32>, %idx : index) {
      // expected-error @+1 {{expected result type to be 'tensor<?x3x?xf32>' or a rank-reduced version. (mismatch of result sizes)}}
  %0 = tensor.extract_slice %t[0, 0, 0][%idx, 3, %idx][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  return
}

// -----

func @no_zero_bit_integer_attrs() {
  // expected-error @+1 {{integer constant out of range for attribute}}
  %x = "some.op"(){value = 0 : i0} : () -> f32
  return
}
