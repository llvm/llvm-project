// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func @dim(%arg : tensor<1x?xf32>) {
  %c2 = constant 2 : index
  dim %arg, %c2 : tensor<1x?xf32> // expected-error {{'std.dim' op index is out of range}}
  return
}

// -----

func @rank(f32) {
^bb(%0: f32):
  "std.rank"(%0): (f32)->index // expected-error {{'std.rank' op operand #0 must be any tensor or memref type}}
  return
}

// -----

func @constant() {
^bb:
  %x = "std.constant"(){value = "xyz"} : () -> i32 // expected-error {{unsupported 'value' attribute}}
  return
}

// -----

func @constant_out_of_range() {
^bb:
  %x = "std.constant"(){value = 100} : () -> i1 // expected-error {{requires attribute's type ('i64') to match op's return type ('i1')}}
  return
}

// -----

func @constant_wrong_type() {
^bb:
  %x = "std.constant"(){value = 10.} : () -> f32 // expected-error {{requires attribute's type ('f64') to match op's return type ('f32')}}
  return
}

// -----
func @affine_apply_no_map() {
^bb0:
  %i = constant 0 : index
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{requires attribute 'map'}}
  return
}

// -----

func @affine_apply_wrong_operand_count() {
^bb0:
  %i = constant 0 : index
  %x = "affine.apply" (%i) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index) -> (index) //  expected-error {{'affine.apply' op operand count and affine map dimension and symbol count must match}}
  return
}

// -----

func @affine_apply_wrong_result_count() {
^bb0:
  %i = constant 0 : index
  %j = constant 1 : index
  %x = "affine.apply" (%i, %j) {map = affine_map<(d0, d1) -> ((d0 + 1), (d1 + 2))>} : (index,index) -> (index) //  expected-error {{'affine.apply' op mapping must produce one value}}
  return
}

// -----

func @unknown_custom_op() {
^bb0:
  %i = crazyThing() {value = 0} : () -> index  // expected-error {{custom op 'crazyThing' is unknown}}
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
  %0 = constant 7 : index
  // Test alloc with wrong number of dynamic dimensions.
  // expected-error@+1 {{dimension operand count does not equal memref dynamic dimension count}}
  %1 = alloc(%0)[%0] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = constant 7 : index
  // Test alloc with wrong number of symbols
  // expected-error@+1 {{symbol operand count does not equal memref symbol count}}
  %1 = alloc(%0) : memref<2x?xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
  return
}

// -----

func @test_store_zero_results() {
^bb0:
  %0 = alloc() : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  %1 = constant 0 : index
  %2 = constant 1 : index
  %3 = load %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
  // Test that store returns zero results.
  %4 = store %3, %0[%1, %2] : memref<1024x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1> // expected-error {{cannot name an operation with no results}}
  return
}

// -----

func @test_store_zero_results2(%x: i32, %p: memref<i32>) {
  "std.store"(%x,%p) : (i32, memref<i32>) -> i32  // expected-error {{'std.store' op requires zero results}}
  return
}

// -----

func @test_alloc_memref_map_rank_mismatch() {
^bb0:
  %0 = alloc() : memref<1024x64xf32, affine_map<(d0) -> (d0)>, 1> // expected-error {{memref affine map dimension mismatch}}
  return
}

// -----

func @intlimit2() {
^bb:
  %0 = "std.constant"() {value = 0} : () -> i16777215
  %1 = "std.constant"() {value = 1} : () -> i16777216 // expected-error {{integer bitwidth is limited to 16777215 bits}}
  return
}

// -----

func @calls(%arg0: i32) {
  %x = call @calls() : () -> i32  // expected-error {{incorrect number of operands for callee}}
  return
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf %a, %a, %a : f32  // expected-error {{'std.addf' op expected 2 operands}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf(%a, %a) : f32  // expected-error {{expected ':'}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = addf{%a, %a} : f32  // expected-error {{expected attribute name}}
}

// -----

func @func_with_ops(f32) {
^bb0(%a : f32):
  // expected-error@+1 {{'std.addi' op operand #0 must be signless-integer-like}}
  %sf = addi %a, %a : f32
}

// -----

func @func_with_ops(i32) {
^bb0(%a : i32):
  %sf = addf %a, %a : i32  // expected-error {{'std.addf' op operand #0 must be floating-point-like}}
}

// -----

func @func_with_ops(i32) {
^bb0(%a : i32):
  // expected-error@+1 {{failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
  %r = "std.cmpi"(%a, %a) {predicate = 42} : (i32, i32) -> i1
}

// -----

// Comparison are defined for arguments of the same type.
func @func_with_ops(i32, i64) {
^bb0(%a : i32, %b : i64): // expected-note {{prior use here}}
  %r = cmpi eq, %a, %b : i32 // expected-error {{use of value '%b' expects different type than prior uses}}
}

// -----

// Comparisons must have the "predicate" attribute.
func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = cmpi %a, %b : i32 // expected-error {{expected string or keyword containing one of the following enum values}}
}

// -----

// Integer comparisons are not recognized for float types.
func @func_with_ops(f32, f32) {
^bb0(%a : f32, %b : f32):
  %r = cmpi eq, %a, %b : f32 // expected-error {{'lhs' must be signless-integer-like, but got 'f32'}}
}

// -----

// Result type must be boolean like.
func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = "std.cmpi"(%a, %b) {predicate = 0} : (i32, i32) -> i32 // expected-error {{op result #0 must be bool-like}}
}

// -----

func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  // expected-error@+1 {{requires attribute 'predicate'}}
  %r = "std.cmpi"(%a, %b) {foo = 1} : (i32, i32) -> i1
}

// -----

func @func_with_ops() {
^bb0:
  %c = constant dense<0> : vector<42 x i32>
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.cmpi"(%c, %c) {predicate = 0} : (vector<42 x i32>, vector<42 x i32>) -> vector<41 x i1>
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

func @invalid_cmp_shape(%idx : () -> ()) {
  // expected-error@+1 {{'lhs' must be signless-integer-like, but got '() -> ()'}}
  %cmp = cmpi eq, %idx, %idx : () -> ()

// -----

func @dma_start_not_enough_operands() {
  // expected-error@+1 {{expected at least 4 operands}}
  "std.dma_start"() : () -> ()
}

// -----

func @dma_no_src_memref(%m : f32, %tag : f32, %c0 : index) {
  // expected-error@+1 {{expected source to be of memref type}}
  dma_start %m[%c0], %m[%c0], %c0, %tag[%c0] : f32, f32, f32
}

// -----

func @dma_start_not_enough_operands_for_src(
    %src: memref<2x2x2xf32>, %idx: index) {
  // expected-error@+1 {{expected at least 7 operands}}
  "std.dma_start"(%src, %idx, %idx, %idx) : (memref<2x2x2xf32>, index, index, index) -> ()
}

// -----

func @dma_start_src_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected source indices to be of index type}}
  "std.dma_start"(%src, %idx, %flt, %dst, %idx, %tag, %idx)
      : (memref<2x2xf32>, index, f32, memref<2xf32,1>, index, memref<i32,2>, index) -> ()
}

// -----

func @dma_no_dst_memref(%m : f32, %tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected destination to be of memref type}}
  dma_start %mref[%c0], %m[%c0], %c0, %tag[%c0] : memref<8 x f32>, f32, f32
}

// -----

func @dma_start_not_enough_operands_for_dst(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>) {
  // expected-error@+1 {{expected at least 7 operands}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index) -> ()
}

// -----

func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected destination indices to be of index type}}
  "std.dma_start"(%src, %idx, %idx, %dst, %flt, %tag, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, f32, memref<i32,2>, index) -> ()
}

// -----

func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected num elements to be of index type}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %flt, %tag)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, f32, memref<i32,2>) -> ()
}

// -----

func @dma_no_tag_memref(%tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected tag to be of memref type}}
  dma_start %mref[%c0], %mref[%c0], %c0, %tag[%c0] : memref<8 x f32>, memref<8 x f32>, f32
}

// -----

func @dma_start_not_enough_operands_for_tag(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<2xi32,2>) {
  // expected-error@+1 {{expected at least 8 operands}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<2xi32,2>) -> ()
}

// -----

func @dma_start_dst_index_wrong_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<2xi32,2>, %flt: f32) {
  // expected-error@+1 {{expected tag indices to be of index type}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %flt)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<2xi32,2>, f32) -> ()
}

// -----

func @dma_start_same_space(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32>,
    %tag: memref<i32,2>) {
  // expected-error@+1 {{DMA should be between different memory spaces}}
  dma_start %src[%idx, %idx], %dst[%idx], %idx, %tag[] : memref<2x2xf32>, memref<2xf32>, memref<i32,2>
}

// -----

func @dma_start_too_many_operands(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>) {
  // expected-error@+1 {{incorrect number of operands}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %idx, %idx, %idx)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<i32,2>, index, index, index) -> ()
}


// -----

func @dma_start_wrong_stride_type(
    %src: memref<2x2xf32>, %idx: index, %dst: memref<2xf32,1>,
    %tag: memref<i32,2>, %flt: f32) {
  // expected-error@+1 {{expected stride and num elements per stride to be of type index}}
  "std.dma_start"(%src, %idx, %idx, %dst, %idx, %idx, %tag, %idx, %flt)
      : (memref<2x2xf32>, index, index, memref<2xf32,1>, index, index, memref<i32,2>, index, f32) -> ()
}

// -----

func @dma_wait_not_enough_operands() {
  // expected-error@+1 {{expected at least 2 operands}}
  "std.dma_wait"() : () -> ()
}

// -----

func @dma_wait_no_tag_memref(%tag : f32, %c0 : index) {
  // expected-error@+1 {{expected tag to be of memref type}}
  "std.dma_wait"(%tag, %c0, %c0) : (f32, index, index) -> ()
}

// -----

func @dma_wait_wrong_index_type(%tag : memref<2xi32>, %idx: index, %flt: f32) {
  // expected-error@+1 {{expected tag indices to be of index type}}
  "std.dma_wait"(%tag, %flt, %idx) : (memref<2xi32>, f32, index) -> ()
}

// -----

func @dma_wait_wrong_num_elements_type(%tag : memref<2xi32>, %idx: index, %flt: f32) {
  // expected-error@+1 {{expected the number of elements to be of index type}}
  "std.dma_wait"(%tag, %idx, %flt) : (memref<2xi32>, index, f32) -> ()
}

// -----

func @invalid_cmp_attr(%idx : i32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %cmp = cmpi i1, %idx, %idx : i32

// -----

func @cmpf_generic_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases}}
  %r = "std.cmpf"(%a, %a) {predicate = 42} : (f32, f32) -> i1
}

// -----

func @cmpf_canonical_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = cmpf foo, %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_signed(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = cmpf sge, %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_no_order(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = cmpf eq, %a, %a : f32
}

// -----

func @cmpf_canonical_no_predicate_attr(%a : f32, %b : f32) {
  %r = cmpf %a, %b : f32 // expected-error {{}}
}

// -----

func @cmpf_generic_no_predicate_attr(%a : f32, %b : f32) {
  // expected-error@+1 {{requires attribute 'predicate'}}
  %r = "std.cmpf"(%a, %b) {foo = 1} : (f32, f32) -> i1
}

// -----

func @cmpf_wrong_type(%a : i32, %b : i32) {
  %r = cmpf oeq, %a, %b : i32 // expected-error {{must be floating-point-like}}
}

// -----

func @cmpf_generic_wrong_result_type(%a : f32, %b : f32) {
  // expected-error@+1 {{result #0 must be bool-like}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (f32, f32) -> f32
}

// -----

func @cmpf_canonical_wrong_result_type(%a : f32, %b : f32) -> f32 {
  %r = cmpf oeq, %a, %b : f32 // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%r' expects different type than prior uses}}
  return %r : f32
}

// -----

func @cmpf_result_shape_mismatch(%a : vector<42xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %r = "std.cmpf"(%a, %a) {predicate = 0} : (vector<42 x f32>, vector<42 x f32>) -> vector<41 x i1>
}

// -----

func @cmpf_operand_shape_mismatch(%a : vector<42xf32>, %b : vector<41xf32>) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (vector<42 x f32>, vector<41 x f32>) -> vector<42 x i1>
}

// -----

func @cmpf_generic_operand_type_mismatch(%a : f32, %b : f64) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (f32, f64) -> i1
}

// -----

func @cmpf_canonical_type_mismatch(%a : f32, %b : f64) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = cmpf oeq, %a, %b : f32
}

// -----

func @index_cast_index_to_index(%arg0: index) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0: index to index
  return
}

// -----

func @index_cast_float(%arg0: index, %arg1: f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0 : index to f32
  return
}

// -----

func @index_cast_float_to_index(%arg0: f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = index_cast %arg0 : f32 to index
  return
}

// -----

func @sitofp_i32_to_i64(%arg0 : i32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = sitofp %arg0 : i32 to i64
  return
}

// -----

func @sitofp_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = sitofp %arg0 : f32 to i32
  return
}

// -----

func @fpext_f32_to_f16(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : f32 to f16
  return
}

// -----

func @fpext_f16_to_f16(%arg0 : f16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : f16 to f16
  return
}

// -----

func @fpext_i32_to_f32(%arg0 : i32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : i32 to f32
  return
}

// -----

func @fpext_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : f32 to i32
  return
}

// -----

func @fpext_vec(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = fpext %arg0 : vector<2xf16> to vector<3xf32>
  return
}

// -----

func @fpext_vec_f32_to_f16(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : vector<2xf32> to vector<2xf16>
  return
}

// -----

func @fpext_vec_f16_to_f16(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : vector<2xf16> to vector<2xf16>
  return
}

// -----

func @fpext_vec_i32_to_f32(%arg0 : vector<2xi32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : vector<2xi32> to vector<2xf32>
  return
}

// -----

func @fpext_vec_f32_to_i32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fpext %arg0 : vector<2xf32> to vector<2xi32>
  return
}

// -----

func @fptrunc_f16_to_f32(%arg0 : f16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : f16 to f32
  return
}

// -----

func @fptrunc_f32_to_f32(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : f32 to f32
  return
}

// -----

func @fptrunc_i32_to_f32(%arg0 : i32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : i32 to f32
  return
}

// -----

func @fptrunc_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : f32 to i32
  return
}

// -----

func @fptrunc_vec(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = fptrunc %arg0 : vector<2xf16> to vector<3xf32>
  return
}

// -----

func @fptrunc_vec_f16_to_f32(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : vector<2xf16> to vector<2xf32>
  return
}

// -----

func @fptrunc_vec_f32_to_f32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : vector<2xf32> to vector<2xf32>
  return
}

// -----

func @fptrunc_vec_i32_to_f32(%arg0 : vector<2xi32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : vector<2xi32> to vector<2xf32>
  return
}

// -----

func @fptrunc_vec_f32_to_i32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = fptrunc %arg0 : vector<2xf32> to vector<2xi32>
  return
}

// -----

func @sexti_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{'index' is not a valid operand type}}
  %0 = sexti %arg0 : index to i128
  return
}

// -----

func @zexti_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{'index' is not a valid operand type}}
  %0 = zexti %arg0 : index to i128
  return
}

// -----

func @trunci_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{'index' is not a valid operand type}}
  %2 = trunci %arg0 : index to i128
  return
}

// -----

func @sexti_index_as_result(%arg0 : i1) {
  // expected-error@+1 {{'index' is not a valid result type}}
  %0 = sexti %arg0 : i1 to index
  return
}

// -----

func @zexti_index_as_operand(%arg0 : i1) {
  // expected-error@+1 {{'index' is not a valid result type}}
  %0 = zexti %arg0 : i1 to index
  return
}

// -----

func @trunci_index_as_result(%arg0 : i128) {
  // expected-error@+1 {{'index' is not a valid result type}}
  %2 = trunci %arg0 : i128 to index
  return
}

// -----

func @sexti_cast_to_narrower(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = sexti %arg0 : i16 to i15
  return
}

// -----

func @zexti_cast_to_narrower(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = zexti %arg0 : i16 to i15
  return
}

// -----

func @trunci_cast_to_wider(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = trunci %arg0 : i16 to i17
  return
}

// -----

func @sexti_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = sexti %arg0 : i16 to i16
  return
}

// -----

func @zexti_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = zexti %arg0 : i16 to i16
  return
}

// -----

func @trunci_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{must be wider}}
  %0 = trunci %arg0 : i16 to i16
  return
}

// -----

func @return_not_in_function() {
  "foo.region"() ({
    // expected-error@+1 {{'std.return' op expects parent op 'func'}}
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
  // expected-error@-1 {{must be integer or float type}}
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
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{expects 1 offset operand}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>>
  // expected-error@+1 {{unsupported map for base memref type}}
  %1 = view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>> to
      memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{unsupported map for result memref type}}
  %1 = view %0[%arg2][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0, d1, s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = view %0[%arg2][%arg0, %arg1] :  memref<2048xi8, 2> to memref<?x?xf32, 1>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{incorrect number of size operands for type}}
  %1 = view %0[%arg2][%arg0]
    : memref<2048xi8> to memref<?x?xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed offsets rank to match mixed sizes rank (2 vs 3) so the rank of the result type is well-formed}}
  %1 = subview %0[0, 0][2, 2, 2][1, 1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = subview %0[0, 0, 0][2, 2, 2][1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected mixed sizes rank to match mixed strides rank (3 vs 2) so the rank of the result type is well-formed}}
  %1 = memref_reinterpret_cast %0 to offset: [0], sizes: [2, 2, 2], strides:[1, 1]
    : memref<8x16x4xf32> to memref<8x16x4xf32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32, offset: 0, strides: [64, 4, 1], 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, offset: 0, strides: [64, 4, 1], 2> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * s0 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>>
  // expected-error@+1 {{is not strided}}
  %1 = subview %0[0, 0, 0][%arg2, %arg2, %arg2][1, 1, 1]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>> to
      memref<8x?x4xf32, offset: 0, strides: [?, 4, 1]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected <= 3 offset values}}
  %1 = subview %0[%arg0, %arg1, 0, 0][%arg2, 0, 0, 0][1, 1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x?x4xf32, offset: 0, strides:[?, ?, 4]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>>' or a rank-reduced version. (mismatch of result affine map)}}
  %1 = subview %0[%arg0, %arg1, %arg2][%arg0, %arg1, %arg2][%arg0, %arg1, %arg2]
    : memref<8x16x4xf32> to
      memref<?x?x?xf32, offset: ?, strides: [64, 4, 1]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result element type to be 'f32'}}
  %1 = subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4xi32>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result rank to be smaller or equal to the source rank.}}
  %1 = subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to
      memref<8x16x4x3xi32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = subview %0[0, 0, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to be 'memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2 + 8)>>' or a rank-reduced version. (mismatch of result sizes)}}
  %1 = subview %0[0, 2, 0][8, 16, 4][1, 1, 1]
    : memref<8x16x4xf32> to memref<16x4xf32>
  return
}

// -----

func @invalid_rank_reducing_subview(%arg0 : memref<?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected result type to be 'memref<?x1xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' or a rank-reduced version. (mismatch of result affine map)}}
  %0 = subview %arg0[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32>
  return
}

// -----

// The affine map affine_map<(d0)[s0, s1, s2] -> (d0 * s1 + s0)> has an extra unused symbol.
func @invalid_rank_reducing_subview(%arg0 : memref<?x?xf32>, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expected result type to be 'memref<?x1xf32, affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>>' or a rank-reduced version. (mismatch of result affine map) inferred type: (d0)[s0, s1] -> (d0 * s1 + s0)}}
  %0 = subview %arg0[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, affine_map<(d0)[s0, s1, s2] -> (d0 * s1 + s0)>>
  return
}

// -----

func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2)>>' and result type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 128 + d1 * 32 + d2 * 2)>>' are cast incompatible}}
  %0 = memref_cast %arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]> to memref<12x4x16xf32, offset:0, strides:[128, 32, 2]>
  return
}

// -----

func @invalid_memref_cast(%arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]>) {
  // expected-error@+1{{operand type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2)>>' and result type 'memref<12x4x16xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2 + 16)>>' are cast incompatible}}
  %0 = memref_cast %arg0 : memref<12x4x16xf32, offset:0, strides:[64, 16, 1]> to memref<12x4x16xf32, offset:16, strides:[64, 16, 1]>
  return
}

// -----

// incompatible element types
func @invalid_memref_cast() {
  %0 = alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xi32>' are cast incompatible}}
  %1 = memref_cast %0 : memref<2x5xf32, 0> to memref<*xi32>
  return
}

// -----

func @invalid_prefetch_rw(%i : index) {
  %0 = alloc() : memref<10xf32>
  // expected-error@+1 {{rw specifier has to be 'read' or 'write'}}
  prefetch %0[%i], rw, locality<0>, data  : memref<10xf32>
  return
}

// -----

func @invalid_prefetch_cache_type(%i : index) {
  %0 = alloc() : memref<10xf32>
  // expected-error@+1 {{cache type has to be 'data' or 'instr'}}
  prefetch %0[%i], read, locality<0>, false  : memref<10xf32>
  return
}

// -----

func @invalid_prefetch_locality_hint(%i : index) {
  %0 = alloc() : memref<10xf32>
  // expected-error@+1 {{32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3}}
  prefetch %0[%i], read, locality<5>, data  : memref<10xf32>
  return
}

// -----

// incompatible memory space
func @invalid_memref_cast() {
  %0 = alloc() : memref<2x5xf32, 0>
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xf32, 1>' are cast incompatible}}
  %1 = memref_cast %0 : memref<2x5xf32, 0> to memref<*xf32, 1>
  return
}

// -----

// unranked to unranked
func @invalid_memref_cast() {
  %0 = alloc() : memref<2x5xf32, 0>
  %1 = memref_cast %0 : memref<2x5xf32, 0> to memref<*xf32, 0>
  // expected-error@+1 {{operand type 'memref<*xf32>' and result type 'memref<*xf32>' are cast incompatible}}
  %2 = memref_cast %1 : memref<*xf32, 0> to memref<*xf32, 0>
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
      %c1 = constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_wrong_arg_type(%I: memref<10xf32>, %i : index) {
  // expected-error@+1 {{expected block argument of the same type result type}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : i32):
      %c1 = constant 1.0 : f32
      atomic_yield %c1 : f32
  }
  return
}

// -----

func @generic_atomic_rmw_result_type_mismatch(%I: memref<10xf32>, %i : index) {
 // expected-error@+1 {{failed to verify that result type matches element type of memref}}
 %0 = "std.generic_atomic_rmw"(%I, %i) ( {
    ^bb0(%old_value: f32):
      %c1 = constant 1.0 : f32
      atomic_yield %c1 : f32
    }) : (memref<10xf32>, index) -> i32
  return
}

// -----

func @generic_atomic_rmw_has_side_effects(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{should contain only operations with no side effects}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = constant 1.0 : f32
      %buf = alloc() : memref<2048xf32>
      atomic_yield %c1 : f32
  }
}

// -----

func @atomic_yield_type_mismatch(%I: memref<10xf32>, %i : index) {
  // expected-error@+4 {{op types mismatch between yield op: 'i32' and its parent: 'f32'}}
  %x = generic_atomic_rmw %I[%i] : memref<10xf32> {
    ^bb0(%old_value : f32):
      %c1 = constant 1 : i32
      atomic_yield %c1 : i32
  }
  return
}

// -----

// alignment is not power of 2.
func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{alignment must be power of 2}}
  std.assume_alignment %0, 12 : memref<4x4xf16>
  return
}

// -----

// 0 alignment value.
func @assume_alignment(%0: memref<4x4xf16>) {
  // expected-error@+1 {{attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  std.assume_alignment %0, 0 : memref<4x4xf16>
  return
}

// -----

"alloca_without_scoped_alloc_parent"() ( {
  std.alloca() : memref<1xf32>
  // expected-error@-1 {{requires an ancestor op with AutomaticAllocationScope trait}}
  return
}) : () -> ()

// -----

func @subtensor_wrong_dynamic_type(%t: tensor<8x16x4xf32>, %idx : index) {
      // expected-error @+1 {{expected result type to be 'tensor<4x4x4xf32>' or a rank-reduced version. (mismatch of result sizes)}}
  %0 = subtensor %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<?x4x4xf32>

  return
}

// -----

func @subtensor_wrong_static_type(%t: tensor<8x16x4xf32>, %idx : index) {
      // expected-error @+1 {{expected result type to be 'tensor<?x3x?xf32>' or a rank-reduced version. (mismatch of result sizes)}}
  %0 = subtensor %t[0, 0, 0][%idx, 3, %idx][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  return
}

// -----

func @no_zero_bit_integer_attrs() {
  // expected-error @+1 {{integer constant out of range for attribute}}
  %x = "some.op"(){value = 0 : i0} : () -> f32
  return
}
