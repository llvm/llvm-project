// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func.func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{'arith.index_cast' op requires the same shape for all operands and results}}
  %0 = arith.index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

func.func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{'arith.index_cast' op requires the same shape for all operands and results}}
  %0 = arith.index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}

// -----

func.func @non_signless_constant() {
  // expected-error @+1 {{'arith.constant' op integer return type must be signless}}
  %0 = arith.constant 0 : ui32
  return
}

// -----

func.func @complex_constant_wrong_attribute_type() {
  // expected-error @+1 {{'arith.constant' op failed to verify that all of {value, result} have same type}}
  %0 = "arith.constant" () {value = 1.0 : f32} : () -> complex<f32>
  return
}

// -----

func.func @non_signless_constant() {
  // expected-error @+1 {{'arith.constant' op integer return type must be signless}}
  %0 = arith.constant 0 : si32
  return
}

// -----

func.func @bitcast_different_bit_widths(%arg : f16) -> f32 {
  // expected-error@+1 {{are cast incompatible}}
  %res = arith.bitcast %arg : f16 to f32
  return %res : f32
}

// -----

func.func @constant() {
^bb:
  %x = "arith.constant"(){value = "xyz"} : () -> i32 // expected-error {{'arith.constant' op failed to verify that all of {value, result} have same type}}
  return
}

// -----

func.func @constant_out_of_range() {
^bb:
  %x = "arith.constant"(){value = 100} : () -> i1 // expected-error {{'arith.constant' op failed to verify that all of {value, result} have same type}}
  return
}

// -----

func.func @constant_wrong_type() {
^bb:
  %x = "arith.constant"(){value = 10.} : () -> f32 // expected-error {{'arith.constant' op failed to verify that all of {value, result} have same type}}
  return
}

// -----

func.func @intlimit2() {
^bb:
  %0 = "arith.constant"() {value = 0} : () -> i16777215
  %1 = "arith.constant"() {value = 1} : () -> i16777216 // expected-error {{integer bitwidth is limited to 16777215 bits}}
  return
}

// -----

func.func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = arith.addf %a, %a, %a : f32  // expected-error {{expected ':'}}
}

// -----

func.func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = arith.addf(%a, %a) : f32  // expected-error {{expected SSA operand}}
}

// -----

func.func @func_with_ops(f32) {
^bb0(%a : f32):
  %sf = arith.addf{%a, %a} : f32  // expected-error {{expected SSA operand}}
}

// -----

func.func @func_with_ops(f32) {
^bb0(%a : f32):
  // expected-error@+1 {{'arith.addi' op operand #0 must be signless-integer-like}}
  %sf = arith.addi %a, %a : f32
}

// -----

func.func @func_with_ops(i32) {
^bb0(%a : i32):
  %sf = arith.addf %a, %a : i32  // expected-error {{'arith.addf' op operand #0 must be floating-point-like}}
}

// -----

func.func @func_with_ops(i32) {
^bb0(%a : i32):
  // expected-error@+1 {{failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
  %r = "arith.cmpi"(%a, %a) {predicate = 42} : (i32, i32) -> i1
}

// -----

// Comparison are defined for arguments of the same type.
func.func @func_with_ops(i32, i64) {
^bb0(%a : i32, %b : i64): // expected-note {{prior use here}}
  %r = arith.cmpi eq, %a, %b : i32 // expected-error {{use of value '%b' expects different type than prior uses}}
}

// -----

// Comparisons must have the "predicate" attribute.
func.func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = arith.cmpi %a, %b : i32 // expected-error {{expected string or keyword containing one of the following enum values}}
}

// -----

// Integer comparisons are not recognized for float types.
func.func @func_with_ops(f32, f32) {
^bb0(%a : f32, %b : f32):
  %r = arith.cmpi eq, %a, %b : f32 // expected-error {{'lhs' must be signless-integer-like, but got 'f32'}}
}

// -----

// Result type must be boolean like.
func.func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  %r = "arith.cmpi"(%a, %b) {predicate = 0} : (i32, i32) -> i32 // expected-error {{op result #0 must be bool-like}}
}

// -----

func.func @func_with_ops(i32, i32) {
^bb0(%a : i32, %b : i32):
  // expected-error@+1 {{requires attribute 'predicate'}}
  %r = "arith.cmpi"(%a, %b) {foo = 1} : (i32, i32) -> i1
}

// -----

func.func @func_with_ops() {
^bb0:
  %c = arith.constant dense<0> : vector<42 x i32>
  // expected-error@+1 {{op failed to verify that result type has i1 element type and same shape as operands}}
  %r = "arith.cmpi"(%c, %c) {predicate = 0} : (vector<42 x i32>, vector<42 x i32>) -> vector<41 x i1>
}

// -----

func.func @invalid_cmp_shape(%idx : () -> ()) {
  // expected-error@+1 {{'lhs' must be signless-integer-like, but got '() -> ()'}}
  %cmp = arith.cmpi eq, %idx, %idx : () -> ()

// -----

func.func @invalid_cmp_attr(%idx : i32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %cmp = arith.cmpi i1, %idx, %idx : i32

// -----

func.func @cmpf_generic_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases}}
  %r = "arith.cmpf"(%a, %a) {predicate = 42} : (f32, f32) -> i1
}

// -----

func.func @cmpf_canonical_invalid_predicate_value(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = arith.cmpf foo, %a, %a : f32
}

// -----

func.func @cmpf_canonical_invalid_predicate_value_signed(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = arith.cmpf sge, %a, %a : f32
}

// -----

func.func @cmpf_canonical_invalid_predicate_value_no_order(%a : f32) {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values}}
  %r = arith.cmpf eq, %a, %a : f32
}

// -----

func.func @cmpf_canonical_no_predicate_attr(%a : f32, %b : f32) {
  %r = arith.cmpf %a, %b : f32 // expected-error {{}}
}

// -----

func.func @cmpf_generic_no_predicate_attr(%a : f32, %b : f32) {
  // expected-error@+1 {{requires attribute 'predicate'}}
  %r = "arith.cmpf"(%a, %b) {foo = 1} : (f32, f32) -> i1
}

// -----

func.func @cmpf_wrong_type(%a : i32, %b : i32) {
  %r = arith.cmpf oeq, %a, %b : i32 // expected-error {{must be floating-point-like}}
}

// -----

func.func @cmpf_generic_wrong_result_type(%a : f32, %b : f32) {
  // expected-error@+1 {{result #0 must be bool-like}}
  %r = "arith.cmpf"(%a, %b) {predicate = 0} : (f32, f32) -> f32
}

// -----

func.func @cmpf_canonical_wrong_result_type(%a : f32, %b : f32) -> f32 {
  %r = arith.cmpf oeq, %a, %b : f32 // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%r' expects different type than prior uses}}
  return %r : f32
}

// -----

func.func @cmpf_result_shape_mismatch(%a : vector<42xf32>) {
  // expected-error@+1 {{op failed to verify that result type has i1 element type and same shape as operands}}
  %r = "arith.cmpf"(%a, %a) {predicate = 0} : (vector<42 x f32>, vector<42 x f32>) -> vector<41 x i1>
}

// -----

func.func @cmpf_operand_shape_mismatch(%a : vector<42xf32>, %b : vector<41xf32>) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "arith.cmpf"(%a, %b) {predicate = 0} : (vector<42 x f32>, vector<41 x f32>) -> vector<42 x i1>
}

// -----

func.func @cmpf_generic_operand_type_mismatch(%a : f32, %b : f64) {
  // expected-error@+1 {{op requires all operands to have the same type}}
  %r = "arith.cmpf"(%a, %b) {predicate = 0} : (f32, f64) -> i1
}

// -----

func.func @cmpf_canonical_type_mismatch(%a : f32, %b : f64) { // expected-note {{prior use here}}
  // expected-error@+1 {{use of value '%b' expects different type than prior uses}}
  %r = arith.cmpf oeq, %a, %b : f32
}

// -----

func.func @index_cast_index_to_index(%arg0: index) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.index_cast %arg0: index to index
  return
}

// -----

func.func @index_cast_float(%arg0: index, %arg1: f32) {
  // expected-error@+1 {{op result #0 must be signless-integer-like or memref of signless-integer, but got 'f32'}}
  %0 = arith.index_cast %arg0 : index to f32
  return
}

// -----

func.func @index_cast_float_to_index(%arg0: f32) {
  // expected-error@+1 {{op operand #0 must be signless-integer-like or memref of signless-integer, but got 'f32'}}
  %0 = arith.index_cast %arg0 : f32 to index
  return
}

// -----

func.func @sitofp_i32_to_i64(%arg0 : i32) {
  // expected-error@+1 {{op result #0 must be floating-point-like, but got 'i64'}}
  %0 = arith.sitofp %arg0 : i32 to i64
  return
}

// -----

func.func @sitofp_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{op operand #0 must be signless-fixed-width-integer-like, but got 'f32'}}
  %0 = arith.sitofp %arg0 : f32 to i32
  return
}

// -----

func.func @fpext_f32_to_f16(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extf %arg0 : f32 to f16
  return
}

// -----

func.func @fpext_f16_to_f16(%arg0 : f16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extf %arg0 : f16 to f16
  return
}

// -----

func.func @fpext_i32_to_f32(%arg0 : i32) {
  // expected-error@+1 {{op operand #0 must be floating-point-like, but got 'i32'}}
  %0 = arith.extf %arg0 : i32 to f32
  return
}

// -----

func.func @fpext_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{op result #0 must be floating-point-like, but got 'i32'}}
  %0 = arith.extf %arg0 : f32 to i32
  return
}

// -----

func.func @fpext_vec(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %0 = arith.extf %arg0 : vector<2xf16> to vector<3xf32>
  return
}

// -----

func.func @fpext_vec_f32_to_f16(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extf %arg0 : vector<2xf32> to vector<2xf16>
  return
}

// -----

func.func @fpext_vec_f16_to_f16(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extf %arg0 : vector<2xf16> to vector<2xf16>
  return
}

// -----

func.func @fpext_vec_i32_to_f32(%arg0 : vector<2xi32>) {
  // expected-error@+1 {{op operand #0 must be floating-point-like, but got 'vector<2xi32>'}}
  %0 = arith.extf %arg0 : vector<2xi32> to vector<2xf32>
  return
}

// -----

func.func @fpext_vec_f32_to_i32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{op result #0 must be floating-point-like, but got 'vector<2xi32>'}}
  %0 = arith.extf %arg0 : vector<2xf32> to vector<2xi32>
  return
}

// -----

func.func @fptrunc_f16_to_f32(%arg0 : f16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.truncf %arg0 : f16 to f32
  return
}

// -----

func.func @fptrunc_f32_to_f32(%arg0 : f32) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.truncf %arg0 : f32 to f32
  return
}

// -----

func.func @fptrunc_i32_to_f32(%arg0 : i32) {
  // expected-error@+1 {{op operand #0 must be floating-point-like, but got 'i32'}}
  %0 = arith.truncf %arg0 : i32 to f32
  return
}

// -----

func.func @fptrunc_f32_to_i32(%arg0 : f32) {
  // expected-error@+1 {{op result #0 must be floating-point-like, but got 'i32'}}
  %0 = arith.truncf %arg0 : f32 to i32
  return
}

// -----

func.func @fptrunc_vec(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{op requires the same shape for all operands and results}}
  %0 = arith.truncf %arg0 : vector<2xf16> to vector<3xf32>
  return
}

// -----

func.func @fptrunc_vec_f16_to_f32(%arg0 : vector<2xf16>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.truncf %arg0 : vector<2xf16> to vector<2xf32>
  return
}

// -----

func.func @fptrunc_vec_f32_to_f32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.truncf %arg0 : vector<2xf32> to vector<2xf32>
  return
}

// -----

func.func @fptrunc_vec_i32_to_f32(%arg0 : vector<2xi32>) {
  // expected-error@+1 {{op operand #0 must be floating-point-like, but got 'vector<2xi32>'}}
  %0 = arith.truncf %arg0 : vector<2xi32> to vector<2xf32>
  return
}

// -----

func.func @fptrunc_vec_f32_to_i32(%arg0 : vector<2xf32>) {
  // expected-error@+1 {{op result #0 must be floating-point-like, but got 'vector<2xi32>'}}
  %0 = arith.truncf %arg0 : vector<2xf32> to vector<2xi32>
  return
}

// -----

func.func @sexti_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{op operand #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %0 = arith.extsi %arg0 : index to i128
  return
}

// -----

func.func @zexti_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{op operand #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %0 = arith.extui %arg0 : index to i128
  return
}

// -----

func.func @trunci_index_as_operand(%arg0 : index) {
  // expected-error@+1 {{op operand #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %2 = arith.trunci %arg0 : index to i128
  return
}

// -----

func.func @sexti_index_as_result(%arg0 : i1) {
  // expected-error@+1 {{op result #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %0 = arith.extsi %arg0 : i1 to index
  return
}

// -----

func.func @zexti_index_as_operand(%arg0 : i1) {
  // expected-error@+1 {{op result #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %0 = arith.extui %arg0 : i1 to index
  return
}

// -----

func.func @trunci_index_as_result(%arg0 : i128) {
  // expected-error@+1 {{op result #0 must be signless-fixed-width-integer-like, but got 'index'}}
  %2 = arith.trunci %arg0 : i128 to index
  return
}

// -----

func.func @sexti_cast_to_narrower(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extsi %arg0 : i16 to i15
  return
}

// -----

func.func @zexti_cast_to_narrower(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extui %arg0 : i16 to i15
  return
}

// -----

func.func @trunci_cast_to_wider(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.trunci %arg0 : i16 to i17
  return
}

// -----

func.func @sexti_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extsi %arg0 : i16 to i16
  return
}

// -----

func.func @zexti_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.extui %arg0 : i16 to i16
  return
}

// -----

func.func @trunci_cast_to_same_width(%arg0 : i16) {
  // expected-error@+1 {{are cast incompatible}}
  %0 = arith.trunci %arg0 : i16 to i16
  return
}

// -----

func.func @trunci_scalable_to_fl(%arg0 : vector<[4]xi32>) {
  // expected-error@+1 {{'arith.trunci' op requires the same shape for all operands and results}}
  %0 = arith.trunci %arg0 : vector<[4]xi32> to vector<4xi8>
  return
}

// -----

func.func @truncf_scalable_to_fl(%arg0 : vector<[4]xf64>) {
  // expected-error@+1 {{'arith.truncf' op requires the same shape for all operands and results}}
  %0 = arith.truncf %arg0 : vector<[4]xf64> to vector<4xf32>
  return
}

// -----

func.func @extui_scalable_to_fl(%arg0 : vector<[4]xi32>) {
  // expected-error@+1 {{'arith.extui' op requires the same shape for all operands and results}}
  %0 = arith.extui %arg0 : vector<[4]xi32> to vector<4xi64>
  return
}

// -----

func.func @extsi_scalable_to_fl(%arg0 : vector<[4]xi32>) {
  // expected-error@+1 {{'arith.extsi' op requires the same shape for all operands and results}}
  %0 = arith.extsi %arg0 : vector<[4]xi32> to vector<4xi64>
  return
}

// -----

func.func @extf_scalable_to_fl(%arg0 : vector<[4]xf32>) {
  // expected-error@+1 {{'arith.extf' op requires the same shape for all operands and results}}
  %0 = arith.extf %arg0 : vector<[4]xf32> to vector<4xf64>
  return
}

// -----

func.func @fptoui_scalable_to_fl(%arg0 : vector<[4]xf64>) {
  // expected-error@+1 {{'arith.fptoui' op requires the same shape for all operands and results}}
  %0 = arith.fptoui %arg0 : vector<[4]xf64> to vector<4xi32>
  return
}

// -----

func.func @fptosi_scalable_to_fl(%arg0 : vector<[4]xf32>) {
  // expected-error@+1 {{'arith.fptosi' op requires the same shape for all operands and results}}
  %0 = arith.fptosi %arg0 : vector<[4]xf32> to vector<4xi32>
  return
}

// -----

func.func @uitofp_scalable_to_fl(%arg0 : vector<[4]xi32>) {
  // expected-error@+1 {{'arith.uitofp' op requires the same shape for all operands and results}}
  %0 = arith.uitofp %arg0 : vector<[4]xi32> to vector<4xf32>
  return
}

// -----

func.func @sitofp_scalable_to_fl(%arg0 : vector<[4]xi32>) {
  // expected-error@+1 {{'arith.sitofp' op requires the same shape for all operands and results}}
  %0 = arith.sitofp %arg0 : vector<[4]xi32> to vector<4xf32>
  return
}

// -----

func.func @bitcast_scalable_to_fl(%arg0 : vector<[4]xf32>) {
  // expected-error@+1 {{'arith.bitcast' op requires the same shape for all operands and results}}
  %0 = arith.bitcast %arg0 : vector<[4]xf32> to vector<4xi32>
  return
}

// -----

func.func @trunci_fl_to_scalable(%arg0 : vector<4xi32>) {
  // expected-error@+1 {{'arith.trunci' op requires the same shape for all operands and results}}
  %0 = arith.trunci %arg0 : vector<4xi32> to vector<[4]xi8>
  return
}

// -----

func.func @truncf_fl_to_scalable(%arg0 : vector<4xf64>) {
  // expected-error@+1 {{'arith.truncf' op requires the same shape for all operands and results}}
  %0 = arith.truncf %arg0 : vector<4xf64> to vector<[4]xf32>
  return
}

// -----

func.func @extui_fl_to_scalable(%arg0 : vector<4xi32>) {
  // expected-error@+1 {{'arith.extui' op requires the same shape for all operands and results}}
  %0 = arith.extui %arg0 : vector<4xi32> to vector<[4]xi64>
  return
}

// -----

func.func @extsi_fl_to_scalable(%arg0 : vector<4xi32>) {
  // expected-error@+1 {{'arith.extsi' op requires the same shape for all operands and results}}
  %0 = arith.extsi %arg0 : vector<4xi32> to vector<[4]xi64>
  return
}

// -----

func.func @extf_fl_to_scalable(%arg0 : vector<4xf32>) {
  // expected-error@+1 {{'arith.extf' op requires the same shape for all operands and results}}
  %0 = arith.extf %arg0 : vector<4xf32> to vector<[4]xf64>
  return
}

// -----

func.func @fptoui_fl_to_scalable(%arg0 : vector<4xf64>) {
  // expected-error@+1 {{'arith.fptoui' op requires the same shape for all operands and results}}
  %0 = arith.fptoui %arg0 : vector<4xf64> to vector<[4]xi32>
  return
}

// -----

func.func @fptosi_fl_to_scalable(%arg0 : vector<4xf32>) {
  // expected-error@+1 {{'arith.fptosi' op requires the same shape for all operands and results}}
  %0 = arith.fptosi %arg0 : vector<4xf32> to vector<[4]xi32>
  return
}

// -----

func.func @uitofp_fl_to_scalable(%arg0 : vector<4xi32>) {
  // expected-error@+1 {{'arith.uitofp' op requires the same shape for all operands and results}}
  %0 = arith.uitofp %arg0 : vector<4xi32> to vector<[4]xf32>
  return
}

// -----

func.func @sitofp_fl_to_scalable(%arg0 : vector<4xi32>) {
  // expected-error@+1 {{'arith.sitofp' op requires the same shape for all operands and results}}
  %0 = arith.sitofp %arg0 : vector<4xi32> to vector<[4]xf32>
  return
}

// -----

func.func @bitcast_fl_to_scalable(%arg0 : vector<4xf32>) {
  // expected-error@+1 {{'arith.bitcast' op requires the same shape for all operands and results}}
  %0 = arith.bitcast %arg0 : vector<4xf32> to vector<[4]xi32>
  return
}

// -----

func.func @func() {
  %c0 = arith.constant  // expected-error {{expected attribute value}}

  %x = arith.constant 1 : i32
}

// -----

func.func @delinearize(%idx: index, %basis0: index, %basis1 :index) {
  // expected-error@+1 {{'arith.delinearize_index' op should return an index for each basis element}}
  %1 = arith.delinearize_index %idx (%basis0, %basis1) : index
  return
}

// -----

func.func @delinearize(%idx: index, %basis0: index, %basis1 :index) {
  // expected-error@+1 {{'arith.delinearize_index' op basis should not be empty}}
  arith.delinearize_index %idx () : index
  return
}
