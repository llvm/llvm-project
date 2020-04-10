// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func @dim(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(%0){index = "xyz"} : (tensor<1xf32>)->index // expected-error {{attribute 'index' failed to satisfy constraint: arbitrary integer attribute}}
  return
}

// -----

func @dim2(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(){index = "xyz"} : ()->index // expected-error {{'std.dim' op requires a single operand}}
  return
}

// -----

func @dim3(tensor<1xf32>) {
^bb(%0: tensor<1xf32>):
  "std.dim"(%0){index = 1} : (tensor<1xf32>)->index // expected-error {{'std.dim' op index is out of range}}
  return
}

// -----

func @rank(f32) {
^bb(%0: f32):
  "std.rank"(%0): (f32)->index // expected-error {{'std.rank' op operand #0 must be tensor of any type values}}
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
  %x = "affine.apply" (%i) { } : (index) -> (index) //  expected-error {{'affine.apply' op requires attribute 'map'}}
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
  %1 = alloc(%0)[%1] : memref<2x4xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1> // expected-error {{op 'std.alloc' dimension operand count does not equal memref dynamic dimension count}}
  return
}

// -----

func @bad_alloc_wrong_symbol_count() {
^bb0:
  %0 = constant 7 : index
  // Test alloc with wrong number of symbols
  %1 = alloc(%0) : memref<2x?xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1> // expected-error {{operand count does not equal dimension plus symbol operand count}}
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
  %0 = "std.constant"() {value = 0} : () -> i4096
  %1 = "std.constant"() {value = 1} : () -> i4097 // expected-error {{integer bitwidth is limited to 4096 bits}}
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
  %r = cmpi "eq", %a, %b : i32 // expected-error {{use of value '%b' expects different type than prior uses}}
}

// -----

// Comparisons must have the "predicate" attribute.
func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = cmpi %a, %b : i32 // expected-error {{expected non-function type}}
}

// -----

// Integer comparisons are not recognized for float types.
func @func_with_ops(f32, f32) {
^bb0(%a : f32, %b : f32):
  %r = cmpi "eq", %a, %b : f32 // expected-error {{'lhs' must be signless-integer-like, but got 'f32'}}
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
  // expected-error@+1 {{op requires the same shape for all operands and results}}
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

func @func_with_ops(i1, vector<42xi32>, vector<42xi32>) {
^bb0(%cond : i1, %t : vector<42xi32>, %f : vector<42xi32>):
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %r = "std.select"(%cond, %t, %f) : (i1, vector<42xi32>, vector<42xi32>) -> vector<42xi32>
}

// -----

func @func_with_ops(i1, tensor<42xi32>, tensor<?xi32>) {
^bb0(%cond : i1, %t : tensor<42xi32>, %f : tensor<?xi32>):
  // expected-error@+1 {{ op requires the same shape for all operands and results}}
  %r = "std.select"(%cond, %t, %f) : (i1, tensor<42xi32>, tensor<?xi32>) -> tensor<42xi32>
}

// -----

func @invalid_select_shape(%cond : i1, %idx : () -> ()) {
  // expected-error@+1 {{'result' must be signless-integer-like or floating-point-like, but got '() -> ()'}}
  %sel = select %cond, %idx, %idx : () -> ()

// -----

func @invalid_cmp_shape(%idx : () -> ()) {
  // expected-error@+1 {{'lhs' must be signless-integer-like, but got '() -> ()'}}
  %cmp = cmpi "eq", %idx, %idx : () -> ()

// -----

func @dma_no_src_memref(%m : f32, %tag : f32, %c0 : index) {
  // expected-error@+1 {{expected source to be of memref type}}
  dma_start %m[%c0], %m[%c0], %c0, %tag[%c0] : f32, f32, f32
}

// -----

func @dma_no_dst_memref(%m : f32, %tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected destination to be of memref type}}
  dma_start %mref[%c0], %m[%c0], %c0, %tag[%c0] : memref<8 x f32>, f32, f32
}

// -----

func @dma_no_tag_memref(%tag : f32, %c0 : index) {
  %mref = alloc() : memref<8 x f32>
  // expected-error@+1 {{expected tag to be of memref type}}
  dma_start %mref[%c0], %mref[%c0], %c0, %tag[%c0] : memref<8 x f32>, memref<8 x f32>, f32
}

// -----

func @dma_wait_no_tag_memref(%tag : f32, %c0 : index) {
  // expected-error@+1 {{expected tag to be of memref type}}
  dma_wait %tag[%c0], %arg0 : f32
}

// -----

func @invalid_cmp_attr(%idx : i32) {
  // expected-error@+1 {{invalid kind of attribute specified}}
  %cmp = cmpi i1, %idx, %idx : i32

// -----

func @cmpf_generic_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases}}
  %r = "std.cmpf"(%a, %a) {predicate = 42} : (f32, f32) -> i1
}

// -----

func @cmpf_canonical_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{invalid predicate attribute specification: "foo"}}
  %r = cmpf "foo", %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_signed(%a : f32) {
  // expected-error@+1 {{invalid predicate attribute specification: "sge"}}
  %r = cmpf "sge", %a, %a : f32
}

// -----

func @cmpf_canonical_invalid_predicate_value_no_order(%a : f32) {
  // expected-error@+1 {{invalid predicate attribute specification: "eq"}}
  %r = cmpf "eq", %a, %a : f32
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
  %r = cmpf "oeq", %a, %b : i32 // expected-error {{must be floating-point-like}}
}

// -----

func @cmpf_generic_wrong_result_type(%a : f32, %b : f32) {
  // expected-error@+1 {{result #0 must be bool-like}}
  %r = "std.cmpf"(%a, %b) {predicate = 0} : (f32, f32) -> f32
}

// -----

func @cmpf_canonical_wrong_result_type(%a : f32, %b : f32) -> f32 {
  %r = cmpf "oeq", %a, %b : f32 // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%r' expects different type than prior uses}}
  return %r : f32
}

// -----

func @cmpf_result_shape_mismatch(%a : vector<42xf32>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
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
  %r = cmpf "oeq", %a, %b : f32
}

// -----

func @extract_element_no_operands() {
  // expected-error@+1 {{op expected 1 or more operands}}
  %0 = "std.extract_element"() : () -> f32
  return
}

// -----

func @extract_element_no_indices(%v : vector<3xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%v) : (vector<3xf32>) -> f32
  return
}

// -----

func @extract_element_invalid_index_type(%v : vector<3xf32>, %i : i32) {
  // expected-error@+1 {{operand #1 must be index}}
  %0 = "std.extract_element"(%v, %i) : (vector<3xf32>, i32) -> f32
  return
}

// -----

func @extract_element_element_result_type_mismatch(%v : vector<3xf32>, %i : index) {
  // expected-error@+1 {{result type matches element type of aggregate}}
  %0 = "std.extract_element"(%v, %i) : (vector<3xf32>, index) -> f64
  return
}

// -----

func @extract_element_vector_too_many_indices(%v : vector<3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%v, %i, %i) : (vector<3xf32>, index, index) -> f32
  return
}

// -----

func @extract_element_tensor_too_many_indices(%t : tensor<2x3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%t, %i, %i, %i) : (tensor<2x3xf32>, index, index, index) -> f32
  return
}

// -----

func @extract_element_tensor_too_few_indices(%t : tensor<2x3xf32>, %i : index) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = "std.extract_element"(%t, %i) : (tensor<2x3xf32>, index) -> f32
  return
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
  // expected-error@+1 {{are cast incompatible}}
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
  // expected-error@+1 {{are cast incompatible}}
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
  // expected-error@+1 {{incorrect number of operands for type}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{is not strided}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0, d1, s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xf32>
  // expected-error@+1 {{must be 1D memref of 8-bit signless integer values}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>>
  // expected-error@+1 {{unsupported map for base memref}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8, affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>> to
      memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8, 2> to
      memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d0 * 4 + d1 + s0)>, 1>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{incorrect dynamic strides}}
  %1 = view %0[][%arg0, %arg1]
    : memref<2048xi8> to
      memref<?x?x4xf32, affine_map<(d0, d1, d2) -> (d0 * 777 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_view(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{incorrect dynamic strides}}
  %1 = view %0[%arg0][]
    : memref<2048xi8> to
      memref<16x4x?xf32, affine_map<(d0, d1, d2) -> (d0 * 777 + d1 * 4 + d2)>>
  return
}

// -----

func @multiple_offsets(%arg0: index) {
  %0 = alloc() : memref<2048xi8>
  // expected-error@+1 {{expects 0 or 1 offset operand}}
  %1 = view %0[%arg0, %arg0][%arg0]
    : memref<2048xi8> to
      memref<?x?x4xf32, affine_map<(d0, d1, d2) -> (d0 * 777 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>, 2>
  // expected-error@+1 {{different memory spaces}}
  %1 = subview %0[][%arg2][]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>, 2> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * s0 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>>
  // expected-error@+1 {{is not strided}}
  %1 = subview %0[][%arg2][]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 4 + d2)>> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 + s0, d1, d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>>
  // expected-error@+1 {{is not strided}}
  %1 = subview %0[][%arg2][]
    : memref<8x16x4xf32, affine_map<(d0, d1, d2) -> (d0 + d1, d1 + d2, d2)>> to
      memref<8x?x4xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * s0 + d1 * 4 + d2)>>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected number of dynamic offsets specified to match the rank of the result type}}
  %1 = subview %0[%arg0, %arg1][%arg2][]
    : memref<8x16x4xf32> to
      memref<8x?x4xf32, offset: 0, strides:[?, ?, 4]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  // expected-error@+1 {{expected result type to have dynamic strides}}
  %1 = subview %0[%arg0, %arg1, %arg2][%arg0, %arg1, %arg2][%arg0, %arg1, %arg2]
    : memref<8x16x4xf32> to
      memref<?x?x?xf32, offset: ?, strides: [64, 4, 1]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = alloc() : memref<8x16x4xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  // expected-error@+1 {{expected result memref layout map to have dynamic offset}}
  %1 = subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32> to
      memref<?x?x?xf32, offset: 0, strides: [?, ?, ?]>
  return
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{expected rank of result type to match rank of base type}}
  %0 = subview %arg1[%arg0, %arg0][][%arg0, %arg0] : memref<?x?xf32> to memref<?xf32>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{expected number of dynamic offsets specified to match the rank of the result type}}
  %0 = subview %arg1[%arg0][][] : memref<?x?xf32> to memref<4x4xf32, offset: ?, strides: [4, 1]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{expected number of dynamic sizes specified to match the rank of the result type}}
  %0 = subview %arg1[][%arg0][] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{expected number of dynamic strides specified to match the rank of the result type}}
  %0 = subview %arg1[][][%arg0] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{invalid to specify dynamic sizes when subview result type is statically shaped and viceversa}}
  %0 = subview %arg1[][%arg0, %arg0][] : memref<?x?xf32> to memref<4x8xf32, offset: ?, strides: [?, ?]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x?xf32>) {
  // expected-error@+1 {{invalid to specify dynamic sizes when subview result type is statically shaped and viceversa}}
  %0 = subview %arg1[][][] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<16x4xf32>) {
  // expected-error@+1 {{expected result memref layout map to have dynamic offset}}
  %0 = subview %arg1[%arg0, %arg0][][] : memref<16x4xf32> to memref<4x2xf32>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<16x4xf32, offset: ?, strides: [4, 1]>) {
  // expected-error@+1 {{expected result memref layout map to have dynamic offset}}
  %0 = subview %arg1[][][] : memref<16x4xf32, offset: ?, strides: [4, 1]> to memref<4x2xf32>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<16x4xf32, offset: 8, strides:[?, 1]>) {
  // expected-error@+1 {{expected result memref layout map to have dynamic offset}}
  %0 = subview %arg1[][][] : memref<16x4xf32, offset: 8, strides:[?, 1]> to memref<4x2xf32>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<16x4xf32>) {
  // expected-error@+1 {{expected result type to have dynamic strides}}
  %0 = subview %arg1[][][%arg0, %arg0] : memref<16x4xf32> to memref<4x2xf32>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<16x4xf32, offset: 0, strides:[?, ?]>) {
  // expected-error@+1 {{expected result type to have dynamic stride along a dimension if the base memref type has dynamic stride along that dimension}}
  %0 = subview %arg1[][][] : memref<16x4xf32, offset: 0, strides:[?, ?]> to memref<4x2xf32, offset:?, strides:[2, 1]>
}

// -----

func @invalid_subview(%arg0 : index, %arg1 : memref<?x8x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  // expected-error@+1 {{expected shape of result type to be fully dynamic when sizes are specified}}
  %0 = subview %arg1[%c0, %c0, %c0][%c1, %arg0, %c1][%c1, %c1, %c1] : memref<?x8x?xf32> to memref<?x8x?xf32, offset:?, strides:[?, ?, ?]>
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
  // expected-error@+1 {{operand type 'memref<2x5xf32>' and result type 'memref<*xf32>' are cast incompatible}}
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
  %x = atomic_rmw "addf" %val, %I[%i] : (f32, memref<16x10xf32>) -> f32
  return
}

// -----

func @atomic_rmw_expects_float(%I: memref<16x10xi32>, %i : index, %val : i32) {
  // expected-error@+1 {{expects a floating-point type}}
  %x = atomic_rmw "addf" %val, %I[%i, %i] : (i32, memref<16x10xi32>) -> i32
  return
}

// -----

func @atomic_rmw_expects_int(%I: memref<16x10xf32>, %i : index, %val : f32) {
  // expected-error@+1 {{expects an integer type}}
  %x = atomic_rmw "addi" %val, %I[%i, %i] : (f32, memref<16x10xf32>) -> f32
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
  // expected-error@+1 {{'std.assume_alignment' op attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  std.assume_alignment %0, 0 : memref<4x4xf16>
  return
}

// -----

"alloca_without_scoped_alloc_parent"() ( {
  std.alloca() : memref<1xf32>
  // expected-error@-1 {{requires an ancestor op with AutomaticAllocationScope trait}}
  return
}) : () -> ()
