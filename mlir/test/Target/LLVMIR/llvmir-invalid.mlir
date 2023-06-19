// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// expected-error @below{{cannot be converted to LLVM IR}}
func.func @foo() {
  llvm.return
}

// -----

llvm.func @no_non_complex_struct() -> !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>> {
  // expected-error @below{{expected struct type to be a complex number}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
}

// -----

llvm.func @no_non_complex_struct() -> !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>> {
  // expected-error @below{{expected struct type to be a complex number}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
}

// -----

llvm.func @struct_wrong_attribute_element_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @below{{FloatAttr does not match expected type of the constant}}
  %0 = llvm.mlir.constant([1.0 : f32, 1.0 : f32]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

// expected-error @below{{unsupported constant value}}
llvm.mlir.global internal constant @test([2.5, 7.4]) : !llvm.array<2 x f64>

// -----

// expected-error @below{{LLVM attribute 'noinline' does not expect a value}}
llvm.func @passthrough_unexpected_value() attributes {passthrough = [["noinline", "42"]]}

// -----

// expected-error @below{{LLVM attribute 'alignstack' expects a value}}
llvm.func @passthrough_expected_value() attributes {passthrough = ["alignstack"]}

// -----

// expected-error @below{{expected 'passthrough' to contain string or array attributes}}
llvm.func @passthrough_wrong_type() attributes {passthrough = [42]}

// -----

// expected-error @below{{expected arrays within 'passthrough' to contain two strings}}
llvm.func @passthrough_wrong_type() attributes {
  passthrough = [[ 42, 42 ]]
}

// -----

llvm.func @unary_float_intr_wrong_type(%arg0 : i32) -> i32 {
  // expected-error @below{{op operand #0 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type}}
  %0 = "llvm.intr.exp"(%arg0) : (i32) -> i32
  llvm.return %0 : i32
}

// -----

llvm.func @binary_float_intr_wrong_type(%arg0 : f32, %arg1 : i32) -> i32 {
  // expected-error @below{{op operand #1 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type}}
  %0 = "llvm.intr.pow"(%arg0, %arg1) : (f32, i32) -> i32
  llvm.return %0 : i32
}

// -----

llvm.func @unary_int_intr_wrong_type(%arg0 : f32) -> f32 {
  // expected-error @below{{op operand #0 must be signless integer or LLVM dialect-compatible vector of signless integer}}
  %0 = "llvm.intr.ctpop"(%arg0) : (f32) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @binary_int_intr_wrong_type(%arg0 : i32, %arg1 : f32) -> f32 {
  // expected-error @below{{op operand #1 must be signless integer or LLVM dialect-compatible vector of signless integer}}
  %0 = "llvm.intr.smax"(%arg0, %arg1) : (i32, f32) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @ternary_float_intr_wrong_type(%arg0 : f32, %arg1 : f32, %arg2 : i32) -> f32 {
  // expected-error @below{{op operand #2 must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type}}
  %0 = "llvm.intr.fma"(%arg0, %arg1, %arg2) : (f32, f32, i32) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @powi_intr_wrong_type(%arg0 : f32, %arg1 : f32) -> f32 {
  // expected-error @below{{op operand #1 must be signless integer, but got 'f32'}}
  %0 = "llvm.intr.powi"(%arg0, %arg1) : (f32, f32) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @memcpy_intr_wrong_type(%src : i64, %dst : i64, %len : i64) {
  // expected-error @below{{op operand #0 must be LLVM pointer type, but got 'i64'}}
  "llvm.intr.memcpy"(%src, %dst, %len) <{isVolatile = false}> : (i64, i64, i64) -> ()
  llvm.return
}

// -----

llvm.func @memmove_intr_wrong_type(%src : !llvm.ptr, %dst : i64, %len : i64) {
  // expected-error @below{{op operand #1 must be LLVM pointer type, but got 'i64'}}
  "llvm.intr.memmove"(%src, %dst, %len) <{isVolatile = false}> : (!llvm.ptr, i64, i64) -> ()
  llvm.return
}

// -----

llvm.func @memset_intr_wrong_type(%dst : !llvm.ptr, %val : i32, %len : i64) {
  // expected-error @below{{op operand #1 must be 8-bit signless integer, but got 'i32'}}
  "llvm.intr.memset"(%dst, %val, %len) <{isVolatile = false}> : (!llvm.ptr, i32, i64) -> ()
  llvm.return
}

// -----

llvm.func @sadd_overflow_intr_wrong_type(%arg0 : i32, %arg1 : f32) -> !llvm.struct<(i32, i1)> {
  // expected-error @below{{op operand #1 must be signless integer or LLVM dialect-compatible vector of signless integer, but got 'f32'}}
  %0 = "llvm.intr.sadd.with.overflow"(%arg0, %arg1) : (i32, f32) -> !llvm.struct<(i32, i1)>
  llvm.return %0 : !llvm.struct<(i32, i1)>
}

// -----

llvm.func @assume_intr_wrong_type(%cond : i16) {
  // expected-error @below{{op operand #0 must be 1-bit signless integer, but got 'i16'}}
  "llvm.intr.assume"(%cond) : (i16) -> ()
  llvm.return
}

// -----

llvm.func @vec_reduce_add_intr_wrong_type(%arg0 : vector<4xi32>) -> f32 {
  // expected-error @below{{op requires the same element type for all operands and results}}
  %0 = "llvm.intr.vector.reduce.add"(%arg0) : (vector<4xi32>) -> f32
  llvm.return %0 : f32
}

// -----

llvm.func @vec_reduce_fmax_intr_wrong_type(%arg0 : vector<4xi32>) -> i32 {
  // expected-error @below{{op operand #0 must be LLVM dialect-compatible vector of floating-point}}
  %0 = llvm.intr.vector.reduce.fmax(%arg0) : (vector<4xi32>) -> i32
  llvm.return %0 : i32
}

// -----

llvm.func @matrix_load_intr_wrong_type(%ptr : !llvm.ptr, %stride : i32) -> f32 {
  // expected-error @below{{op result #0 must be LLVM dialect-compatible vector type, but got 'f32'}}
  %0 = llvm.intr.matrix.column.major.load %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} : f32 from !llvm.ptr stride i32
  llvm.return %0 : f32
}

// -----

llvm.func @matrix_store_intr_wrong_type(%matrix : vector<48xf32>, %ptr : i32, %stride : i64) {
  // expected-error @below {{op operand #1 must be LLVM pointer type, but got 'i32'}}
  llvm.intr.matrix.column.major.store %matrix, %ptr, <stride=%stride>
    { isVolatile = 0: i1, rows = 3: i32, columns = 16: i32} : vector<48xf32> to i32 stride i64
  llvm.return
}

// -----

llvm.func @matrix_multiply_intr_wrong_type(%arg0 : vector<64xf32>, %arg1 : f32) -> vector<12xf32> {
  // expected-error @below{{op operand #1 must be LLVM dialect-compatible vector type, but got 'f32'}}
  %0 = llvm.intr.matrix.multiply %arg0, %arg1
    { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32} : (vector<64xf32>, f32) -> vector<12xf32>
  llvm.return %0 : vector<12xf32>
}

// -----

llvm.func @matrix_transpose_intr_wrong_type(%matrix : f32) -> vector<48xf32> {
  // expected-error @below{{op operand #0 must be LLVM dialect-compatible vector type, but got 'f32'}}
  %0 = llvm.intr.matrix.transpose %matrix {rows = 3: i32, columns = 16: i32} : f32 into vector<48xf32>
  llvm.return %0 : vector<48xf32>
}

// -----

llvm.func @active_lane_intr_wrong_type(%base : i64, %n : vector<7xi64>) -> vector<7xi1> {
  // expected-error @below{{invalid kind of type specified}}
  %0 = llvm.intr.get.active.lane.mask %base, %n : i64, vector<7xi64> to vector<7xi1>
  llvm.return %0 : vector<7xi1>
}

// -----

llvm.func @masked_load_intr_wrong_type(%ptr : i64, %mask : vector<7xi1>) -> vector<7xf32> {
  // expected-error @below{{op operand #0 must be LLVM pointer type, but got 'i64'}}
  %0 = llvm.intr.masked.load %ptr, %mask { alignment = 1: i32} : (i64, vector<7xi1>) -> vector<7xf32>
  llvm.return %0 : vector<7xf32>
}

// -----

llvm.func @masked_store_intr_wrong_type(%vec : vector<7xf32>, %ptr : !llvm.ptr, %mask : vector<7xi32>) {
  // expected-error @below{{op operand #2 must be LLVM dialect-compatible vector of 1-bit signless integer, but got 'vector<7xi32>}}
  llvm.intr.masked.store %vec, %ptr, %mask { alignment = 1: i32} : vector<7xf32>, vector<7xi32> into !llvm.ptr
  llvm.return
}

// -----

llvm.func @masked_gather_intr_wrong_type(%ptrs : vector<7xf32>, %mask : vector<7xi1>) -> vector<7xf32> {
  // expected-error @below{{op operand #0 must be LLVM dialect-compatible vector of LLVM pointer type, but got 'vector<7xf32>'}}
  %0 = llvm.intr.masked.gather %ptrs, %mask { alignment = 1: i32} : (vector<7xf32>, vector<7xi1>) -> vector<7xf32>
  llvm.return %0 : vector<7xf32>
}

// -----

llvm.func @masked_scatter_intr_wrong_type(%vec : f32, %ptrs : !llvm.vec<7xptr<f32>>, %mask : vector<7xi1>) {
  // expected-error @below{{op operand #0 must be LLVM dialect-compatible vector type, but got 'f32'}}
  llvm.intr.masked.scatter %vec, %ptrs, %mask { alignment = 1: i32} : f32, vector<7xi1> into !llvm.vec<7xptr<f32>>
  llvm.return
}

// -----

llvm.func @stepvector_intr_wrong_type() -> vector<7xf32> {
  // expected-error @below{{op result #0 must be LLVM dialect-compatible vector of signless integer, but got 'vector<7xf32>'}}
  %0 = llvm.intr.experimental.stepvector : vector<7xf32>
  llvm.return %0 : vector<7xf32>
}

// -----

llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @foo any
}

llvm.comdat @__llvm_comdat_1 {
  // expected-error @below{{comdat selection symbols must be unique even in different comdat regions}}
  llvm.comdat_selector @foo any
}
