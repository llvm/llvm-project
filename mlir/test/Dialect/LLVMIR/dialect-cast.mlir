// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// These are the supported cases, just make sure they don't trigger errors, op
// syntax is tested elsewhere.

func @mlir_dialect_cast(%0: index, %1: i32, %2: bf16, %3: f16, %4: f32, %5: f64,
                        %6: vector<42xf32>, %7: memref<42xf32>,
                        %8: memref<?xf32>, %9: memref<f32>,
                        %10: memref<*xf32>) {
  llvm.mlir.cast %0 : index to !llvm.i64
  llvm.mlir.cast %0 : index to !llvm.i32
  llvm.mlir.cast %1 : i32 to !llvm.i32
  llvm.mlir.cast %2 : bf16 to !llvm.bfloat
  llvm.mlir.cast %3 : f16 to !llvm.half
  llvm.mlir.cast %4 : f32 to !llvm.float
  llvm.mlir.cast %5 : f64 to !llvm.double
  llvm.mlir.cast %6 : vector<42xf32> to !llvm.vec<42xfloat>
  llvm.mlir.cast %7 : memref<42xf32> to !llvm.ptr<float>
  llvm.mlir.cast %7 : memref<42xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<1xi64>, array<1xi64>)>
  llvm.mlir.cast %8 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<1xi64>, array<1xi64>)>
  llvm.mlir.cast %9 : memref<f32> to !llvm.struct<(ptr<float>, ptr<float>, i64)>
  llvm.mlir.cast %10 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
  return
}

// -----

func @mlir_dialect_cast_index_non_integer(%0 : index) {
  // expected-error@+1 {{invalid cast between index and non-integer type}}
  %1 = llvm.mlir.cast %0 : index to !llvm.float
}

// -----

// Cast verifier is symmetric, so we only check the symmetry once by having an
// std->llvm and llvm->std test. Everything else is std->llvm.

func @mlir_dialect_cast_index_non_integer_symmetry(%0: !llvm.float) {
  // expected-error@+1 {{invalid cast between index and non-integer type}}
  llvm.mlir.cast %0 : !llvm.float to index
}

// -----

func @mlir_dialect_cast_f16(%0 : f16) {
  // expected-error@+1 {{invalid cast between f16 and a type other than !llvm.half}}
  llvm.mlir.cast %0 : f16 to !llvm.float
}

// -----

func @mlir_dialect_cast_bf16(%0 : bf16) {
  // expected-error@+1 {{invalid cast between bf16 and a type other than !llvm.bfloat}}
  llvm.mlir.cast %0 : bf16 to !llvm.half
}

// -----

func @mlir_dialect_cast_f32(%0 : f32) {
  // expected-error@+1 {{invalid cast between f32 and a type other than !llvm.float}}
  llvm.mlir.cast %0 : f32 to !llvm.bfloat
}

// -----

func @mlir_dialect_cast_f64(%0 : f64) {
  // expected-error@+1 {{invalid cast between f64 and a type other than !llvm.double}}
  llvm.mlir.cast %0 : f64 to !llvm.float
}

// -----

func @mlir_dialect_cast_integer_non_integer(%0 : i16) {
  // expected-error@+1 {{invalid cast between integer and non-integer type}}
  llvm.mlir.cast %0 : i16 to !llvm.half
}

// -----

func @mlir_dialect_cast_integer_bitwidth_mismatch(%0 : i16) {
  // expected-error@+1 {{invalid cast between integers with mismatching bitwidth}}
  llvm.mlir.cast %0 : i16 to !llvm.i32
}

// -----

func @mlir_dialect_cast_nd_vector(%0 : vector<2x2xf32>) {
  // expected-error@+1 {{only 1-d vector is allowed}}
  llvm.mlir.cast %0 : vector<2x2xf32> to !llvm.vec<4xfloat>
}

// -----

func @mlir_dialect_cast_scalable_vector(%0 : vector<2xf32>) {
  // expected-error@+1 {{only fixed-sized vector is allowed}}
  llvm.mlir.cast %0 : vector<2xf32> to !llvm.vec<?x2xfloat>
}

// -----

func @mlir_dialect_cast_vector_size_mismatch(%0 : vector<2xf32>) {
  // expected-error@+1 {{invalid cast between vectors with mismatching sizes}}
  llvm.mlir.cast %0 : vector<2xf32> to !llvm.vec<4xfloat>
}

// -----

func @mlir_dialect_cast_dynamic_memref_bare_ptr(%0 : memref<?xf32>) {
  // expected-error@+1 {{unexpected bare pointer for dynamically shaped memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.ptr<float>
}

// -----

func @mlir_dialect_cast_memref_bare_ptr_space(%0 : memref<4xf32, 4>) {
  // expected-error@+1 {{invalid conversion between memref and pointer in different memory spaces}}
  llvm.mlir.cast %0 : memref<4xf32, 4> to !llvm.ptr<float, 3>
}

// -----

func @mlir_dialect_cast_memref_no_descriptor(%0 : memref<?xf32>) {
  // expected-error@+1 {{invalid cast between a memref and a type other than pointer or memref descriptor}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.float
}

// -----

func @mlir_dialect_cast_memref_descriptor_wrong_num_elements(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected memref descriptor with 5 elements}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<()>
}

// -----

func @mlir_dialect_cast_0d_memref_descriptor_wrong_num_elements(%0 : memref<f32>) {
  // expected-error@+1 {{expected memref descriptor with 3 elements}}
  llvm.mlir.cast %0 : memref<f32> to !llvm.struct<()>
}

// -----

func @mlir_dialect_cast_memref_descriptor_allocated(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected first element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(float, float, float, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_allocated_wrong_space(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected first element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float, 2>, float, float, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_aligned(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, float, float, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_aligned_wrong_space(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float, 2>, float, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_offset(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected third element of a memref descriptor to be index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, float, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, float, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes_wrong_type(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<10xfloat>, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes_wrong_rank(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<10xi64>, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<1xi64>, float)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides_wrong_type(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<1xi64>, array<10xfloat>)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides_wrong_rank(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<float>, ptr<float>, i64, array<1xi64>, array<10xi64>)>
}

// -----

func @mlir_dialect_cast_tensor(%0 : tensor<?xf32>) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : tensor<?xf32> to !llvm.float
}

// -----

func @mlir_dialect_cast_two_std_types(%0 : f32) {
  // expected-error@+1 {{expected one LLVM type and one built-in type}}
  llvm.mlir.cast %0 : f32 to f64
}

// -----

func @mlir_dialect_cast_unranked_memref(%0: memref<*xf32>) {
  // expected-error@+1 {{expected descriptor to be a struct with two elements}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.ptr<float>
}

// -----

func @mlir_dialect_cast_unranked_memref(%0: memref<*xf32>) {
  // expected-error@+1 {{expected descriptor to be a struct with two elements}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<()>
}

// -----

func @mlir_dialect_cast_unranked_rank(%0: memref<*xf32>) {
  // expected-error@+1 {{expected first element of a memref descriptor to be an index-compatible integer}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<(float, float)>
}

// -----

func @mlir_dialect_cast_unranked_rank(%0: memref<*xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be an !llvm.ptr<i8>}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<(i64, float)>
}
