// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// TODO(b/133530217): Add more tests after switching to the generic parser.

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

// CHECK: func @scalar_array_type(!spv.array<16 x f32>, !spv.array<8 x i32>)
func @scalar_array_type(!spv.array<16xf32>, !spv.array<8 x i32>) -> ()

// CHECK: func @vector_array_type(!spv.array<32 x vector<4xf32>>)
func @vector_array_type(!spv.array< 32 x vector<4xf32> >) -> ()

// CHECK: func @array_type_stride(!spv.array<4 x !spv.array<4 x f32, stride=4>, stride=128>)
func @array_type_stride(!spv.array< 4 x !spv.array<4 x f32, stride=4>, stride = 128>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func @missing_left_angle_bracket(!spv.array 4xf32>) -> ()

// -----

// expected-error @+1 {{expected single integer for array element count}}
func @missing_count(!spv.array<f32>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func @missing_x(!spv.array<4 f32>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @missing_element_type(!spv.array<4x>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @cannot_parse_type(!spv.array<4xblabla>) -> ()

// -----

// expected-error @+1 {{expected single integer for array element count}}
func @more_than_one_dim(!spv.array<4x3xf32>) -> ()

// -----

// expected-error @+1 {{only 1-D vector allowed but found 'vector<4x3xf32>'}}
func @non_1D_vector(!spv.array<4xvector<4x3xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'tensor<4xf32>' to compose SPIR-V types}}
func @tensor_type(!spv.array<4xtensor<4xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'bf16' to compose SPIR-V types}}
func @bf16_type(!spv.array<4xbf16>) -> ()

// -----

// expected-error @+1 {{only 1/8/16/32/64-bit integer type allowed but found 'i256'}}
func @i256_type(!spv.array<4xi256>) -> ()

// -----

// expected-error @+1 {{cannot use 'index' to compose SPIR-V types}}
func @index_type(!spv.array<4xindex>) -> ()

// -----

// expected-error @+1 {{cannot use '!llvm.i32' to compose SPIR-V types}}
func @llvm_type(!spv.array<4x!llvm.i32>) -> ()

// -----

// expected-error @+1 {{ArrayStride must be greater than zero}}
func @array_type_zero_stride(!spv.array<4xi32, stride=0>) -> ()

// -----

// expected-error @+1 {{expected array length greater than 0}}
func @array_type_zero_length(!spv.array<0xf32>) -> ()

// -----

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// CHECK: @bool_ptr_type(!spv.ptr<i1, Uniform>)
func @bool_ptr_type(!spv.ptr<i1, Uniform>) -> ()

// CHECK: @scalar_ptr_type(!spv.ptr<f32, Uniform>)
func @scalar_ptr_type(!spv.ptr<f32, Uniform>) -> ()

// CHECK: @vector_ptr_type(!spv.ptr<vector<4xi32>, PushConstant>)
func @vector_ptr_type(!spv.ptr<vector<4xi32>,PushConstant>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func @missing_left_angle_bracket(!spv.ptr f32, Uniform>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @missing_comma(!spv.ptr<f32 Uniform>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @missing_pointee_type(!spv.ptr<, Uniform>) -> ()

// -----

// expected-error @+1 {{unknown storage class: SomeStorageClass}}
func @unknown_storage_class(!spv.ptr<f32, SomeStorageClass>) -> ()

// -----

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

// CHECK: func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>)
func @scalar_runtime_array_type(!spv.rtarray<f32>, !spv.rtarray<i32>) -> ()

// CHECK: func @vector_runtime_array_type(!spv.rtarray<vector<4xf32>>)
func @vector_runtime_array_type(!spv.rtarray< vector<4xf32> >) -> ()

// CHECK: func @runtime_array_type_stride(!spv.rtarray<f32, stride=4>)
func @runtime_array_type_stride(!spv.rtarray<f32, stride=4>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func @missing_left_angle_bracket(!spv.rtarray f32>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @missing_element_type(!spv.rtarray<>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func @redundant_count(!spv.rtarray<4xf32>) -> ()

// -----

// expected-error @+1 {{ArrayStride must be greater than zero}}
func @runtime_array_type_zero_stride(!spv.rtarray<i32, stride=0>) -> ()

// -----

//===----------------------------------------------------------------------===//
// ImageType
//===----------------------------------------------------------------------===//

// CHECK: func @image_parameters_1D(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>)
func @image_parameters_1D(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_one_element(!spv.image<f32>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_two_elements(!spv.image<f32, Dim1D>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_three_elements(!spv.image<f32, Dim1D, NoDepth>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_four_elements(!spv.image<f32, Dim1D, NoDepth, NonArrayed>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_five_elements(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_six_elements(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func @image_parameters_delimiter(!spv.image f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_nocomma_1(!spv.image<f32, Dim1D NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_nocomma_2(!spv.image<f32, Dim1D, NoDepth NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_nocomma_3(!spv.image<f32, Dim1D, NoDepth, NonArrayed SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_nocomma_4(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @image_parameters_nocomma_5(!spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown Unknown>) -> ()

// -----

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

// CHECK: func @struct_type(!spv.struct<f32>)
func @struct_type(!spv.struct<f32>) -> ()

// CHECK: func @struct_type2(!spv.struct<f32 [0]>)
func @struct_type2(!spv.struct<f32 [0]>) -> ()

// CHECK: func @struct_type_simple(!spv.struct<f32, !spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>>)
func @struct_type_simple(!spv.struct<f32, !spv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>>) -> ()

// CHECK: func @struct_type_with_offset(!spv.struct<f32 [0], i32 [4]>)
func @struct_type_with_offset(!spv.struct<f32 [0], i32 [4]>) -> ()

// CHECK: func @nested_struct(!spv.struct<f32, !spv.struct<f32, i32>>)
func @nested_struct(!spv.struct<f32, !spv.struct<f32, i32>>)

// CHECK: func @nested_struct_with_offset(!spv.struct<f32 [0], !spv.struct<f32 [0], i32 [4]> [4]>)
func @nested_struct_with_offset(!spv.struct<f32 [0], !spv.struct<f32 [0], i32 [4]> [4]>)

// CHECK: func @struct_type_with_decoration(!spv.struct<f32 [NonWritable]>)
func @struct_type_with_decoration(!spv.struct<f32 [NonWritable]>)

// CHECK: func @struct_type_with_decoration_and_offset(!spv.struct<f32 [0, NonWritable]>)
func @struct_type_with_decoration_and_offset(!spv.struct<f32 [0, NonWritable]>)

// CHECK: func @struct_type_with_decoration2(!spv.struct<f32 [NonWritable], i32 [NonReadable]>)
func @struct_type_with_decoration2(!spv.struct<f32 [NonWritable], i32 [NonReadable]>)

// CHECK: func @struct_type_with_decoration3(!spv.struct<f32, i32 [NonReadable]>)
func @struct_type_with_decoration3(!spv.struct<f32, i32 [NonReadable]>)

// CHECK: func @struct_type_with_decoration4(!spv.struct<f32 [0], i32 [4, NonReadable]>)
func @struct_type_with_decoration4(!spv.struct<f32 [0], i32 [4, NonReadable]>)

// CHECK: func @struct_type_with_decoration5(!spv.struct<f32 [NonWritable, NonReadable]>)
func @struct_type_with_decoration5(!spv.struct<f32 [NonWritable, NonReadable]>)

// CHECK: func @struct_type_with_decoration6(!spv.struct<f32, !spv.struct<i32 [NonWritable, NonReadable]>>)
func @struct_type_with_decoration6(!spv.struct<f32, !spv.struct<i32 [NonWritable, NonReadable]>>)

// CHECK: func @struct_type_with_decoration7(!spv.struct<f32 [0], !spv.struct<i32, f32 [NonReadable]> [4]>)
func @struct_type_with_decoration7(!spv.struct<f32 [0], !spv.struct<i32, f32 [NonReadable]> [4]>)

// CHECK: func @struct_type_with_decoration8(!spv.struct<f32, !spv.struct<i32 [0], f32 [4, NonReadable]>>)
func @struct_type_with_decoration8(!spv.struct<f32, !spv.struct<i32 [0], f32 [4, NonReadable]>>)

// CHECK: func @struct_empty(!spv.struct<>)
func @struct_empty(!spv.struct<>)

// -----

// expected-error @+1 {{layout specification must be given for all members}}
func @struct_type_missing_offset1((!spv.struct<f32, i32 [4]>) -> ()

// -----

// expected-error @+1 {{layout specification must be given for all members}}
func @struct_type_missing_offset2(!spv.struct<f32 [3], i32>) -> ()

// -----

// expected-error @+1 {{expected '>'}}
func @struct_type_missing_comma1(!spv.struct<f32 i32>) -> ()

// -----

// expected-error @+1 {{expected '>'}}
func @struct_type_missing_comma2(!spv.struct<f32 [0] i32>) -> ()

// -----

//  expected-error @+1 {{unbalanced '>' character in pretty dialect name}}
func @struct_type_neg_offset(!spv.struct<f32 [0>) -> ()

// -----

//  expected-error @+1 {{unbalanced ']' character in pretty dialect name}}
func @struct_type_neg_offset(!spv.struct<f32 0]>) -> ()

// -----

//  expected-error @+1 {{expected ']'}}
func @struct_type_neg_offset(!spv.struct<f32 [NonWritable 0]>) -> ()

// -----

//  expected-error @+1 {{expected valid keyword}}
func @struct_type_neg_offset(!spv.struct<f32 [NonWritable, 0]>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @struct_type_missing_comma(!spv.struct<f32 [0 NonWritable], i32 [4]>)

// -----

// expected-error @+1 {{expected ']'}}
func @struct_type_missing_comma(!spv.struct<f32 [0, NonWritable NonReadable], i32 [4]>)

// -----

//===----------------------------------------------------------------------===//
// CooperativeMatrix
//===----------------------------------------------------------------------===//

// CHECK: func @coop_matrix_type(!spv.coopmatrix<8x16xi32, Subgroup>, !spv.coopmatrix<8x8xf32, Workgroup>)
func @coop_matrix_type(!spv.coopmatrix<8x16xi32, Subgroup>, !spv.coopmatrix<8x8xf32, Workgroup>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func @missing_scope(!spv.coopmatrix<8x16xi32>) -> ()

// -----

// expected-error @+1 {{expected rows and columns size}}
func @missing_count(!spv.coopmatrix<8xi32, Subgroup>) -> ()

// -----

//===----------------------------------------------------------------------===//
// Matrix
//===----------------------------------------------------------------------===//
// CHECK: func @matrix_type(!spv.matrix<2 x vector<2xf16>>)
func @matrix_type(!spv.matrix<2 x vector<2xf16>>) -> ()

// -----

// CHECK: func @matrix_type(!spv.matrix<3 x vector<3xf32>>)
func @matrix_type(!spv.matrix<3 x vector<3xf32>>) -> ()

// -----

// CHECK: func @matrix_type(!spv.matrix<4 x vector<4xf16>>)
func @matrix_type(!spv.matrix<4 x vector<4xf16>>) -> ()

// -----

// expected-error @+1 {{matrix is expected to have 2, 3, or 4 columns}}
func @matrix_invalid_size(!spv.matrix<5 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{matrix is expected to have 2, 3, or 4 columns}}
func @matrix_invalid_size(!spv.matrix<1 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{matrix columns size has to be less than or equal to 4 and greater than or equal 2, but found 5}}
func @matrix_invalid_columns_size(!spv.matrix<3 x vector<5xf32>>) -> ()

// -----

// expected-error @+1 {{matrix columns size has to be less than or equal to 4 and greater than or equal 2, but found 1}}
func @matrix_invalid_columns_size(!spv.matrix<3 x vector<1xf32>>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func @matrix_invalid_format(!spv.matrix 3 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{unbalanced ')' character in pretty dialect name}}
func @matrix_invalid_format(!spv.matrix< 3 x vector<3xf32>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func @matrix_invalid_format(!spv.matrix<2 vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got 'i32'}}
func @matrix_invalid_type(!spv.matrix< 3 x i32>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got '!spv.array<16 x f32>'}}
func @matrix_invalid_type(!spv.matrix< 3 x !spv.array<16 x f32>>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got '!spv.rtarray<i32>'}}
func @matrix_invalid_type(!spv.matrix< 3 x !spv.rtarray<i32>>) -> ()

// -----

// expected-error @+1 {{matrix columns' elements must be of Float type, got 'i32'}}
func @matrix_invalid_type(!spv.matrix<2 x vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{expected single unsigned integer for number of columns}}
func @matrix_size_type(!spv.matrix< x vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{expected single unsigned integer for number of columns}}
func @matrix_size_type(!spv.matrix<2.0 x vector<3xi32>>) -> ()

// -----