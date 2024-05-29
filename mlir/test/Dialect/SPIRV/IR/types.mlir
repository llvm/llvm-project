// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// TODO: Add more tests after switching to the generic parser.

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

// CHECK: func private @scalar_array_type(!spirv.array<16 x f32>, !spirv.array<8 x i32>)
func.func private @scalar_array_type(!spirv.array<16xf32>, !spirv.array<8 x i32>) -> ()

// CHECK: func private @vector_array_type(!spirv.array<32 x vector<4xf32>>)
func.func private @vector_array_type(!spirv.array< 32 x vector<4xf32> >) -> ()

// CHECK: func private @array_type_stride(!spirv.array<4 x !spirv.array<4 x f32, stride=4>, stride=128>)
func.func private @array_type_stride(!spirv.array< 4 x !spirv.array<4 x f32, stride=4>, stride = 128>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func.func private @missing_left_angle_bracket(!spirv.array 4xf32>) -> ()

// -----

// expected-error @+1 {{expected single integer for array element count}}
func.func private @missing_count(!spirv.array<f32>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func.func private @missing_x(!spirv.array<4 f32>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @missing_element_type(!spirv.array<4x>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @cannot_parse_type(!spirv.array<4xblabla>) -> ()

// -----

// expected-error @+1 {{expected single integer for array element count}}
func.func private @more_than_one_dim(!spirv.array<4x3xf32>) -> ()

// -----

// expected-error @+1 {{only 1-D vector allowed but found 'vector<4x3xf32>'}}
func.func private @non_1D_vector(!spirv.array<4xvector<4x3xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'tensor<4xf32>' to compose SPIR-V types}}
func.func private @tensor_type(!spirv.array<4xtensor<4xf32>>) -> ()

// -----

// expected-error @+1 {{cannot use 'bf16' to compose SPIR-V types}}
func.func private @bf16_type(!spirv.array<4xbf16>) -> ()

// -----

// expected-error @+1 {{only 1/8/16/32/64-bit integer type allowed but found 'i256'}}
func.func private @i256_type(!spirv.array<4xi256>) -> ()

// -----

// expected-error @+1 {{cannot use 'index' to compose SPIR-V types}}
func.func private @index_type(!spirv.array<4xindex>) -> ()

// -----

// expected-error @+1 {{cannot use '!llvm.struct<()>' to compose SPIR-V types}}
func.func private @llvm_type(!spirv.array<4x!llvm.struct<()>>) -> ()

// -----

// expected-error @+1 {{ArrayStride must be greater than zero}}
func.func private @array_type_zero_stride(!spirv.array<4xi32, stride=0>) -> ()

// -----

// expected-error @+1 {{expected array length greater than 0}}
func.func private @array_type_zero_length(!spirv.array<0xf32>) -> ()

// -----

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// CHECK: @bool_ptr_type(!spirv.ptr<i1, Uniform>)
func.func private @bool_ptr_type(!spirv.ptr<i1, Uniform>) -> ()

// CHECK: @scalar_ptr_type(!spirv.ptr<f32, Uniform>)
func.func private @scalar_ptr_type(!spirv.ptr<f32, Uniform>) -> ()

// CHECK: @vector_ptr_type(!spirv.ptr<vector<4xi32>, PushConstant>)
func.func private @vector_ptr_type(!spirv.ptr<vector<4xi32>,PushConstant>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func.func private @missing_left_angle_bracket(!spirv.ptr f32, Uniform>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @missing_comma(!spirv.ptr<f32 Uniform>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @missing_pointee_type(!spirv.ptr<, Uniform>) -> ()

// -----

// expected-error @+1 {{unknown storage class: SomeStorageClass}}
func.func private @unknown_storage_class(!spirv.ptr<f32, SomeStorageClass>) -> ()

// -----

//===----------------------------------------------------------------------===//
// RuntimeArrayType
//===----------------------------------------------------------------------===//

// CHECK: func private @scalar_runtime_array_type(!spirv.rtarray<f32>, !spirv.rtarray<i32>)
func.func private @scalar_runtime_array_type(!spirv.rtarray<f32>, !spirv.rtarray<i32>) -> ()

// CHECK: func private @vector_runtime_array_type(!spirv.rtarray<vector<4xf32>>)
func.func private @vector_runtime_array_type(!spirv.rtarray< vector<4xf32> >) -> ()

// CHECK: func private @runtime_array_type_stride(!spirv.rtarray<f32, stride=4>)
func.func private @runtime_array_type_stride(!spirv.rtarray<f32, stride=4>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func.func private @missing_left_angle_bracket(!spirv.rtarray f32>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @missing_element_type(!spirv.rtarray<>) -> ()

// -----

// expected-error @+1 {{expected non-function type}}
func.func private @redundant_count(!spirv.rtarray<4xf32>) -> ()

// -----

// expected-error @+1 {{ArrayStride must be greater than zero}}
func.func private @runtime_array_type_zero_stride(!spirv.rtarray<i32, stride=0>) -> ()

// -----

//===----------------------------------------------------------------------===//
// ImageType
//===----------------------------------------------------------------------===//

// CHECK: func private @image_parameters_1D(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>)
func.func private @image_parameters_1D(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_one_element(!spirv.image<f32>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_two_elements(!spirv.image<f32, Dim1D>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_three_elements(!spirv.image<f32, Dim1D, NoDepth>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_four_elements(!spirv.image<f32, Dim1D, NoDepth, NonArrayed>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_five_elements(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_six_elements(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func.func private @image_parameters_delimiter(!spirv.image f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_nocomma_1(!spirv.image<f32, Dim1D NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_nocomma_2(!spirv.image<f32, Dim1D, NoDepth NonArrayed, SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_nocomma_3(!spirv.image<f32, Dim1D, NoDepth, NonArrayed SingleSampled, SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_nocomma_4(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled SamplerUnknown, Unknown>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @image_parameters_nocomma_5(!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown Unknown>) -> ()

// -----

//===----------------------------------------------------------------------===//
// SampledImageType
//===----------------------------------------------------------------------===//

// CHECK: func private @sampled_image_type(!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>)
func.func private @sampled_image_type(!spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>) -> ()

// -----

// expected-error @+1 {{sampled image must be composed using image type, got 'f32'}}
func.func private @samped_image_type_invaid_type(!spirv.sampled_image<f32>) -> ()

// -----

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

// CHECK: func private @struct_type(!spirv.struct<(f32)>)
func.func private @struct_type(!spirv.struct<(f32)>) -> ()

// CHECK: func private @struct_type2(!spirv.struct<(f32 [0])>)
func.func private @struct_type2(!spirv.struct<(f32 [0])>) -> ()

// CHECK: func private @struct_type_simple(!spirv.struct<(f32, !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>)>)
func.func private @struct_type_simple(!spirv.struct<(f32, !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>)>) -> ()

// CHECK: func private @struct_type_with_offset(!spirv.struct<(f32 [0], i32 [4])>)
func.func private @struct_type_with_offset(!spirv.struct<(f32 [0], i32 [4])>) -> ()

// CHECK: func private @nested_struct(!spirv.struct<(f32, !spirv.struct<(f32, i32)>)>)
func.func private @nested_struct(!spirv.struct<(f32, !spirv.struct<(f32, i32)>)>)

// CHECK: func private @nested_struct_with_offset(!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], i32 [4])> [4])>)
func.func private @nested_struct_with_offset(!spirv.struct<(f32 [0], !spirv.struct<(f32 [0], i32 [4])> [4])>)

// CHECK: func private @struct_type_with_decoration(!spirv.struct<(f32 [NonWritable])>)
func.func private @struct_type_with_decoration(!spirv.struct<(f32 [NonWritable])>)

// CHECK: func private @struct_type_with_decoration_and_offset(!spirv.struct<(f32 [0, NonWritable])>)
func.func private @struct_type_with_decoration_and_offset(!spirv.struct<(f32 [0, NonWritable])>)

// CHECK: func private @struct_type_with_decoration2(!spirv.struct<(f32 [NonWritable], i32 [NonReadable])>)
func.func private @struct_type_with_decoration2(!spirv.struct<(f32 [NonWritable], i32 [NonReadable])>)

// CHECK: func private @struct_type_with_decoration3(!spirv.struct<(f32, i32 [NonReadable])>)
func.func private @struct_type_with_decoration3(!spirv.struct<(f32, i32 [NonReadable])>)

// CHECK: func private @struct_type_with_decoration4(!spirv.struct<(f32 [0], i32 [4, NonReadable])>)
func.func private @struct_type_with_decoration4(!spirv.struct<(f32 [0], i32 [4, NonReadable])>)

// CHECK: func private @struct_type_with_decoration5(!spirv.struct<(f32 [NonWritable, NonReadable])>)
func.func private @struct_type_with_decoration5(!spirv.struct<(f32 [NonWritable, NonReadable])>)

// CHECK: func private @struct_type_with_decoration6(!spirv.struct<(f32, !spirv.struct<(i32 [NonWritable, NonReadable])>)>)
func.func private @struct_type_with_decoration6(!spirv.struct<(f32, !spirv.struct<(i32 [NonWritable, NonReadable])>)>)

// CHECK: func private @struct_type_with_decoration7(!spirv.struct<(f32 [0], !spirv.struct<(i32, f32 [NonReadable])> [4])>)
func.func private @struct_type_with_decoration7(!spirv.struct<(f32 [0], !spirv.struct<(i32, f32 [NonReadable])> [4])>)

// CHECK: func private @struct_type_with_decoration8(!spirv.struct<(f32, !spirv.struct<(i32 [0], f32 [4, NonReadable])>)>)
func.func private @struct_type_with_decoration8(!spirv.struct<(f32, !spirv.struct<(i32 [0], f32 [4, NonReadable])>)>)

// CHECK: func private @struct_type_with_matrix_1(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>)
func.func private @struct_type_with_matrix_1(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16])>)

// CHECK: func private @struct_type_with_matrix_2(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, RowMajor, MatrixStride=16])>)
func.func private @struct_type_with_matrix_2(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, RowMajor, MatrixStride=16])>)

// CHECK: func private @struct_empty(!spirv.struct<()>)
func.func private @struct_empty(!spirv.struct<()>)

// -----

// expected-error @+1 {{offset specification must be given for all members}}
func.func private @struct_type_missing_offset1((!spirv.struct<(f32, i32 [4])>) -> ()

// -----

// expected-error @+1 {{offset specification must be given for all members}}
func.func private @struct_type_missing_offset2(!spirv.struct<(f32 [3], i32)>) -> ()

// -----

// expected-error @+1 {{expected ')'}}
func.func private @struct_type_missing_comma1(!spirv.struct<(f32 i32)>) -> ()

// -----

// expected-error @+1 {{expected ')'}}
func.func private @struct_type_missing_comma2(!spirv.struct<(f32 [0] i32)>) -> ()

// -----

//  expected-error @+1 {{unbalanced '[' character in pretty dialect name}}
func.func private @struct_type_neg_offset(!spirv.struct<(f32 [0)>) -> ()

// -----

//  expected-error @+1 {{unbalanced '(' character in pretty dialect name}}
func.func private @struct_type_neg_offset(!spirv.struct<(f32 0])>) -> ()

// -----

//  expected-error @+1 {{expected ']'}}
func.func private @struct_type_neg_offset(!spirv.struct<(f32 [NonWritable 0])>) -> ()

// -----

//  expected-error @+1 {{expected valid keyword}}
func.func private @struct_type_neg_offset(!spirv.struct<(f32 [NonWritable, 0])>) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @struct_type_missing_comma(!spirv.struct<(f32 [0 NonWritable], i32 [4])>)

// -----

// expected-error @+1 {{expected ']'}}
func.func private @struct_type_missing_comma(!spirv.struct<(f32 [0, NonWritable NonReadable], i32 [4])>)

// -----

// expected-error @+1 {{expected ']'}}
func.func private @struct_type_missing_comma(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, RowMajor MatrixStride=16])>)

// -----

// expected-error @+1 {{expected integer value}}
func.func private @struct_missing_member_decorator_value(!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, RowMajor, MatrixStride=])>)

// -----

//===----------------------------------------------------------------------===//
// StructType (identified)
//===----------------------------------------------------------------------===//

// CHECK: func private @id_struct_empty(!spirv.struct<empty, ()>)
func.func private @id_struct_empty(!spirv.struct<empty, ()>) -> ()

// -----

// CHECK: func private @id_struct_simple(!spirv.struct<simple, (f32)>)
func.func private @id_struct_simple(!spirv.struct<simple, (f32)>) -> ()

// -----

// CHECK: func private @id_struct_multiple_elements(!spirv.struct<multi_elements, (f32, i32)>)
func.func private @id_struct_multiple_elements(!spirv.struct<multi_elements, (f32, i32)>) -> ()

// -----

// CHECK: func private @id_struct_nested_literal(!spirv.struct<a1, (!spirv.struct<()>)>)
func.func private @id_struct_nested_literal(!spirv.struct<a1, (!spirv.struct<()>)>) -> ()

// -----

// CHECK: func private @id_struct_nested_id(!spirv.struct<a2, (!spirv.struct<b2, ()>)>)
func.func private @id_struct_nested_id(!spirv.struct<a2, (!spirv.struct<b2, ()>)>) -> ()

// -----

// CHECK: func private @literal_struct_nested_id(!spirv.struct<(!spirv.struct<a3, ()>)>)
func.func private @literal_struct_nested_id(!spirv.struct<(!spirv.struct<a3, ()>)>) -> ()

// -----

// CHECK: func private @id_struct_self_recursive(!spirv.struct<a4, (!spirv.ptr<!spirv.struct<a4>, Uniform>)>)
func.func private @id_struct_self_recursive(!spirv.struct<a4, (!spirv.ptr<!spirv.struct<a4>, Uniform>)>) -> ()

// -----

// CHECK: func private @id_struct_self_recursive2(!spirv.struct<a5, (i32, !spirv.ptr<!spirv.struct<a5>, Uniform>)>)
func.func private @id_struct_self_recursive2(!spirv.struct<a5, (i32, !spirv.ptr<!spirv.struct<a5>, Uniform>)>) -> ()

// -----

// expected-error @+1 {{recursive struct reference not nested in struct definition}}
func.func private @id_wrong_recursive_reference(!spirv.struct<a6>) -> ()

// -----

// expected-error @+1 {{recursive struct reference not nested in struct definition}}
func.func private @id_struct_recursive_invalid(!spirv.struct<a7, (!spirv.ptr<!spirv.struct<b7>, Uniform>)>) -> ()

// -----

// expected-error @+1 {{identifier already used for an enclosing struct}}
func.func private @id_struct_redefinition(!spirv.struct<a8, (!spirv.ptr<!spirv.struct<a8, (!spirv.ptr<!spirv.struct<a8>, Uniform>)>, Uniform>)>) -> ()

// -----

// Equivalent to:
//   struct a { struct b *bPtr; };
//   struct b { struct a *aPtr; };
// CHECK: func private @id_struct_recursive(!spirv.struct<a9, (!spirv.ptr<!spirv.struct<b9, (!spirv.ptr<!spirv.struct<a9>, Uniform>)>, Uniform>)>)
func.func private @id_struct_recursive(!spirv.struct<a9, (!spirv.ptr<!spirv.struct<b9, (!spirv.ptr<!spirv.struct<a9>, Uniform>)>, Uniform>)>) -> ()

// -----

// Equivalent to:
//   struct a { struct b *bPtr; };
//   struct b { struct a *aPtr, struct b *bPtr; };
// CHECK: func private @id_struct_recursive(!spirv.struct<a10, (!spirv.ptr<!spirv.struct<b10, (!spirv.ptr<!spirv.struct<a10>, Uniform>, !spirv.ptr<!spirv.struct<b10>, Uniform>)>, Uniform>)>)
func.func private @id_struct_recursive(!spirv.struct<a10, (!spirv.ptr<!spirv.struct<b10, (!spirv.ptr<!spirv.struct<a10>, Uniform>, !spirv.ptr<!spirv.struct<b10>, Uniform>)>, Uniform>)>) -> ()

// -----

//===----------------------------------------------------------------------===//
// CooperativeMatrix (KHR)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func private @coop_matrix_types
// CHECK-SAME:    !spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>
// CHECK-SAME:    !spirv.coopmatrix<8x8xf32, Workgroup, MatrixB>
// CHECK-SAME:    !spirv.coopmatrix<4x8xf32, Workgroup, MatrixAcc>
func.func private @coop_matrix_types(!spirv.coopmatrix<8x16xi32, Subgroup, MatrixA>,
                                     !spirv.coopmatrix<8x8xf32, Workgroup, MatrixB>,
                                     !spirv.coopmatrix<4x8xf32, Workgroup, MatrixAcc>) -> ()

// -----

// expected-error @+1 {{expected valid keyword}}
func.func private @missing_scope(!spirv.coopmatrix<8x8xi32, >) -> ()

// -----

// expected-error @+1 {{expected ','}}
func.func private @missing_use(!spirv.coopmatrix<8x16xi32, Subgroup>) -> ()

// -----

// expected-error @+1 {{expected valid keyword}}
func.func private @missing_use2(!spirv.coopmatrix<8x8xi32, Subgroup,>) -> ()

// -----

// expected-error @+1 {{expected row and column count}}
func.func private @missing_count(!spirv.coopmatrix<8xi32, Subgroup, MatrixA>) -> ()

// -----

// expected-error @+1 {{expected row and column count}}
func.func private @too_many_dims(!spirv.coopmatrix<8x16x32xi32, Subgroup, MatrixB>) -> ()

// -----

// expected-error @+1 {{invalid use <id> attribute specification: Subgroup}}
func.func private @use_not_integer(!spirv.coopmatrix<8x8xi32, Subgroup, Subgroup>) -> ()

// -----

//===----------------------------------------------------------------------===//
// Matrix
//===----------------------------------------------------------------------===//
// CHECK: func private @matrix_type(!spirv.matrix<2 x vector<2xf16>>)
func.func private @matrix_type(!spirv.matrix<2 x vector<2xf16>>) -> ()

// -----

// CHECK: func private @matrix_type(!spirv.matrix<3 x vector<3xf32>>)
func.func private @matrix_type(!spirv.matrix<3 x vector<3xf32>>) -> ()

// -----

// CHECK: func private @matrix_type(!spirv.matrix<4 x vector<4xf16>>)
func.func private @matrix_type(!spirv.matrix<4 x vector<4xf16>>) -> ()

// -----

// expected-error @+1 {{matrix is expected to have 2, 3, or 4 columns}}
func.func private @matrix_invalid_size(!spirv.matrix<5 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{matrix is expected to have 2, 3, or 4 columns}}
func.func private @matrix_invalid_size(!spirv.matrix<1 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{matrix columns size has to be less than or equal to 4 and greater than or equal 2, but found 5}}
func.func private @matrix_invalid_columns_size(!spirv.matrix<3 x vector<5xf32>>) -> ()

// -----

// expected-error @+1 {{matrix columns size has to be less than or equal to 4 and greater than or equal 2, but found 1}}
func.func private @matrix_invalid_columns_size(!spirv.matrix<3 x vector<1xf32>>) -> ()

// -----

// expected-error @+1 {{expected '<'}}
func.func private @matrix_invalid_format(!spirv.matrix 3 x vector<3xf32>>) -> ()

// -----

// expected-error @+1 {{unbalanced '<' character in pretty dialect name}}
func.func private @matrix_invalid_format(!spirv.matrix< 3 x vector<3xf32>) -> ()

// -----

// expected-error @+1 {{expected 'x' in dimension list}}
func.func private @matrix_invalid_format(!spirv.matrix<2 vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got 'i32'}}
func.func private @matrix_invalid_type(!spirv.matrix< 3 x i32>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got '!spirv.array<16 x f32>'}}
func.func private @matrix_invalid_type(!spirv.matrix< 3 x !spirv.array<16 x f32>>) -> ()

// -----

// expected-error @+1 {{matrix must be composed using vector type, got '!spirv.rtarray<i32>'}}
func.func private @matrix_invalid_type(!spirv.matrix< 3 x !spirv.rtarray<i32>>) -> ()

// -----

// expected-error @+1 {{matrix columns' elements must be of Float type, got 'i32'}}
func.func private @matrix_invalid_type(!spirv.matrix<2 x vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{expected single unsigned integer for number of columns}}
func.func private @matrix_size_type(!spirv.matrix< x vector<3xi32>>) -> ()

// -----

// expected-error @+1 {{expected single unsigned integer for number of columns}}
func.func private @matrix_size_type(!spirv.matrix<2.0 x vector<3xi32>>) -> ()

// -----
