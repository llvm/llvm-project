// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// These are the supported cases, just make sure they don't trigger errors, op
// syntax is tested elsewhere.

func @mlir_dialect_cast(%0: index, %1: vector<2x2x2xf32>,
                        %6: vector<42xf32>, %7: memref<42xf32>,
                        %8: memref<?xf32>, %9: memref<f32>,
                        %10: memref<*xf32>) {
  llvm.mlir.cast %0 : index to i64
  llvm.mlir.cast %0 : index to i32
  llvm.mlir.cast %1 : vector<2x2x2xf32> to !llvm.array<2 x array<2 x vector<2xf32>>>
  llvm.mlir.cast %7 : memref<42xf32> to !llvm.ptr<f32>
  llvm.mlir.cast %7 : memref<42xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<1xi64>)>
  llvm.mlir.cast %8 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<1xi64>)>
  llvm.mlir.cast %9 : memref<f32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  llvm.mlir.cast %10 : memref<*xf32> to !llvm.struct<(i64, ptr<i8>)>
  return
}

// -----

func @mlir_dialect_cast_index_non_integer(%0 : index) {
  // expected-error@+1 {{invalid cast between index and non-integer type}}
  %1 = llvm.mlir.cast %0 : index to f32
}

// -----

// Cast verifier is symmetric, so we only check the symmetry once by having an
// std->llvm and llvm->std test. Everything else is std->llvm.

func @mlir_dialect_cast_index_non_integer_symmetry(%0: f32) {
  // expected-error@+1 {{invalid cast between index and non-integer type}}
  llvm.mlir.cast %0 : f32 to index
}

// -----

func @mlir_dialect_cast_f16(%0 : f16) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : f16 to f32
}

// -----

func @mlir_dialect_cast_bf16(%0 : bf16) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : bf16 to f16
}

// -----

func @mlir_dialect_cast_f32(%0 : f32) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : f32 to bf16
}

// -----

func @mlir_dialect_cast_f64(%0 : f64) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : f64 to f32
}

// -----

func @mlir_dialect_cast_integer_non_integer(%0 : i16) {
  // expected-error@+1 {{invalid cast between integer and non-integer}}
  llvm.mlir.cast %0 : i16 to f16
}

// -----

func @mlir_dialect_cast_scalable_vector(%0 : vector<2xf32>) {
  // expected-error@+1 {{invalid cast for vector types}}
  llvm.mlir.cast %0 : vector<2xf32> to !llvm.vec<?x2xf32>
}

// -----

func @mlir_dialect_cast_vector_to_self(%0 : vector<2xf32>) {
  // expected-error@+1 {{vector types should not be casted}}
  llvm.mlir.cast %0 : vector<2xf32> to vector<2xf32>
}

// -----

func @mlir_dialect_cast_nd_vector(%0 : vector<2x2xf32>) {
  // expected-error@+1 {{invalid cast for vector, expected array}}
  llvm.mlir.cast %0 : vector<2x2xf32> to !llvm.struct<()>
}

// -----

func @mlir_dialect_cast_dynamic_memref_bare_ptr(%0 : memref<?xf32>) {
  // expected-error@+1 {{unexpected bare pointer for dynamically shaped memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.ptr<f32>
}

// -----

func @mlir_dialect_cast_memref_bare_ptr_space(%0 : memref<4xf32, 4>) {
  // expected-error@+1 {{invalid conversion between memref and pointer in different memory spaces}}
  llvm.mlir.cast %0 : memref<4xf32, 4> to !llvm.ptr<f32, 3>
}

// -----

func @mlir_dialect_cast_memref_no_descriptor(%0 : memref<?xf32>) {
  // expected-error@+1 {{invalid cast between a memref and a type other than pointer or memref descriptor}}
  llvm.mlir.cast %0 : memref<?xf32> to f32
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
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(f32, f32, f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_allocated_wrong_space(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected first element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32, 2>, f32, f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_aligned(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, f32, f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_aligned_wrong_space(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be a pointer in the address space of the memref}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32, 2>, f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_offset(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected third element of a memref descriptor to be index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, f32, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes_wrong_type(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<10xf32>, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_sizes_wrong_rank(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fourth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<10xi64>, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, f32)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides_wrong_type(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<10xf32>)>
}

// -----

func @mlir_dialect_cast_memref_descriptor_strides_wrong_rank(%0 : memref<?xf32>) {
  // expected-error@+1 {{expected fifth element of a memref descriptor to be an array of <rank> index-compatible integers}}
  llvm.mlir.cast %0 : memref<?xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1xi64>, array<10xi64>)>
}

// -----

func @mlir_dialect_cast_tensor(%0 : tensor<?xf32>) {
  // expected-error@+1 {{unsupported cast}}
  llvm.mlir.cast %0 : tensor<?xf32> to f32
}

// -----

func @mlir_dialect_cast_unranked_memref(%0: memref<*xf32>) {
  // expected-error@+1 {{expected descriptor to be a struct with two elements}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.ptr<f32>
}

// -----

func @mlir_dialect_cast_unranked_memref(%0: memref<*xf32>) {
  // expected-error@+1 {{expected descriptor to be a struct with two elements}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<()>
}

// -----

func @mlir_dialect_cast_unranked_rank(%0: memref<*xf32>) {
  // expected-error@+1 {{expected first element of a memref descriptor to be an index-compatible integer}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<(f32, f32)>
}

// -----

func @mlir_dialect_cast_unranked_rank(%0: memref<*xf32>) {
  // expected-error@+1 {{expected second element of a memref descriptor to be an !llvm.ptr<i8>}}
  llvm.mlir.cast %0 : memref<*xf32> to !llvm.struct<(i64, f32)>
}

// -----

func @mlir_dialect_cast_complex_non_struct(%0: complex<f32>) {
  // expected-error@+1 {{expected 'complex' to map to two-element struct with identical element types}}
  llvm.mlir.cast %0 : complex<f32> to f32
}

// -----

func @mlir_dialect_cast_complex_bad_size(%0: complex<f32>) {
  // expected-error@+1 {{expected 'complex' to map to two-element struct with identical element types}}
  llvm.mlir.cast %0 : complex<f32> to !llvm.struct<(f32, f32, f32)>
}

// -----

func @mlir_dialect_cast_complex_mismatching_type_struct(%0: complex<f32>) {
  // expected-error@+1 {{expected 'complex' to map to two-element struct with identical element types}}
  llvm.mlir.cast %0 : complex<f32> to !llvm.struct<(f32, f64)>
}

// -----

func @mlir_dialect_cast_complex_mismatching_element(%0: complex<f32>) {
  // expected-error@+1 {{expected 'complex' to map to two-element struct with identical element types}}
  llvm.mlir.cast %0 : complex<f32> to !llvm.struct<(f64, f64)>
}
