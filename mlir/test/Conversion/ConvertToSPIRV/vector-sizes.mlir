// RUN: mlir-opt --test-convert-to-spirv="run-signature-conversion=false run-vector-unrolling=true" --split-input-file %s | FileCheck %s

// This file tests the current behaviour of the SignatureConversion
// and the unrolling of vector.to_elements to vectors of valid SPIR-V
// sizes.

// vector's of rank 1 and size 1 will be changed
// to scalars. Since vector.to_elements will also produce
// a scalar, we expect the vector.to_elements to be folded
// away. Please note that even if run-signature-conversion=false
// The pattern FuncOpConversion will still run and change parameters
// which fit this constraint.

// CHECK-LABEL: spirv.func @vec_size_1
// CHECK-SAME: (%[[ARG0:.+]]: f32)
func.func @vec_size_1(%arg0: vector<1xf32>) -> (f32) {
  // CHECK-NEXT: spirv.ReturnValue %[[ARG0]] : f32
  %0:1 = vector.to_elements %arg0 : vector<1xf32>
  return %0#0 : f32
}

// -----

// vector's of rank 2, 3, 4 are allowed by SPIR-V.
// So they remain unchanged. FuncOpConversion will still
// run, but the signature converter will not convert these vectors.

// CHECK-LABEL: spirv.func @vec_size_2
// CHECK-SAME: (%[[ARG0:.+]]: vector<2xf32>)
func.func @vec_size_2(%arg0: vector<2xf32>) -> (f32) {
  // A single result type is enforced by the semantics

  // CHECK-NEXT: %[[VAL:.+]] = spirv.CompositeExtract %[[ARG0]][0 : i32] : vector<2xf32>
  %0:2 = vector.to_elements %arg0 : vector<2xf32>

  // CHECK-NEXT: spirv.ReturnValue %[[VAL]]
  return %0#0 : f32
}

// -----

// vector of rank 5 is the first one that doesn't fit
// into SPIR-V's vectors.

// run-signature-conversion=false means that
// this vector will not be unrolled.

// CHECK-LABEL: func.func @vec_size_5
// CHECK-SAME: (%[[ARG0:.+]]: vector<5xf32>)
func.func @vec_size_5(%arg0: vector<5xf32>) -> (f32) {

  // CHECK-NEXT: %[[VAL:.+]] = vector.extract_strided_slice %[[ARG0]] {offsets = [0], sizes = [1], strides = [1]} : vector<5xf32> to vector<1xf32>

  // We have the following comment in VectorConvertToElementOp
  //
  // // Input vectors of size 1 are converted to scalars by the type converter.
  // // We cannot use `spirv::CompositeExtractOp` directly in this case.
  // // For a scalar source, the result is just the scalar itself.
  //
  // Which in this case means an unrealized conversion cast.

  // CHECK-NEXT: %[[RETVAL:.+]] = builtin.unrealized_conversion_cast %[[VAL]] : vector<1xf32> to f32
  %0:5 = vector.to_elements %arg0 : vector<5xf32>

  // CHECK-NEXT: spirv.ReturnValue %[[RETVAL]] : f32
  return %0#0 : f32
}
