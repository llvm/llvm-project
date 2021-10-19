// RUN: mlir-opt -split-input-file -convert-std-to-spirv %s -o - | FileCheck %s
// RUN: mlir-opt -split-input-file -convert-std-to-spirv="emulate-non-32-bit-scalar-types=false" %s -o - | FileCheck %s --check-prefix=NOEMU

//===----------------------------------------------------------------------===//
// Integer types
//===----------------------------------------------------------------------===//

// Check that non-32-bit integer types are converted to 32-bit types if the
// corresponding capabilities are not available.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @integer8
// CHECK-SAME: i32
// CHECK-SAME: si32
// CHECK-SAME: ui32
// NOEMU-LABEL: func @integer8
// NOEMU-SAME: i8
// NOEMU-SAME: si8
// NOEMU-SAME: ui8
func @integer8(%arg0: i8, %arg1: si8, %arg2: ui8) { return }

// CHECK-LABEL: spv.func @integer16
// CHECK-SAME: i32
// CHECK-SAME: si32
// CHECK-SAME: ui32
// NOEMU-LABEL: func @integer16
// NOEMU-SAME: i16
// NOEMU-SAME: si16
// NOEMU-SAME: ui16
func @integer16(%arg0: i16, %arg1: si16, %arg2: ui16) { return }

// CHECK-LABEL: spv.func @integer64
// CHECK-SAME: i32
// CHECK-SAME: si32
// CHECK-SAME: ui32
// NOEMU-LABEL: func @integer64
// NOEMU-SAME: i64
// NOEMU-SAME: si64
// NOEMU-SAME: ui64
func @integer64(%arg0: i64, %arg1: si64, %arg2: ui64) { return }

} // end module

// -----

// Check that non-32-bit integer types are kept untouched if the corresponding
// capabilities are available.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Int8, Int16, Int64], []>, {}>
} {

// CHECK-LABEL: spv.func @integer8
// CHECK-SAME: i8
// CHECK-SAME: si8
// CHECK-SAME: ui8
// NOEMU-LABEL: spv.func @integer8
// NOEMU-SAME: i8
// NOEMU-SAME: si8
// NOEMU-SAME: ui8
func @integer8(%arg0: i8, %arg1: si8, %arg2: ui8) { return }

// CHECK-LABEL: spv.func @integer16
// CHECK-SAME: i16
// CHECK-SAME: si16
// CHECK-SAME: ui16
// NOEMU-LABEL: spv.func @integer16
// NOEMU-SAME: i16
// NOEMU-SAME: si16
// NOEMU-SAME: ui16
func @integer16(%arg0: i16, %arg1: si16, %arg2: ui16) { return }

// CHECK-LABEL: spv.func @integer64
// CHECK-SAME: i64
// CHECK-SAME: si64
// CHECK-SAME: ui64
// NOEMU-LABEL: spv.func @integer64
// NOEMU-SAME: i64
// NOEMU-SAME: si64
// NOEMU-SAME: ui64
func @integer64(%arg0: i64, %arg1: si64, %arg2: ui64) { return }

} // end module

// -----

// Check that weird bitwidths are not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-NOT: spv.func @integer4
func @integer4(%arg0: i4) { return }

// CHECK-NOT: spv.func @integer128
func @integer128(%arg0: i128) { return }

// CHECK-NOT: spv.func @integer42
func @integer42(%arg0: i42) { return }

} // end module
// -----

//===----------------------------------------------------------------------===//
// Index type
//===----------------------------------------------------------------------===//

// The index type is always converted into i32.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @index_type
// CHECK-SAME: %{{.*}}: i32
func @index_type(%arg0: index) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Float types
//===----------------------------------------------------------------------===//

// Check that non-32-bit float types are converted to 32-bit types if the
// corresponding capabilities are not available.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @float16
// CHECK-SAME: f32
// NOEMU-LABEL: func @float16
// NOEMU-SAME: f16
func @float16(%arg0: f16) { return }

// CHECK-LABEL: spv.func @float64
// CHECK-SAME: f32
// NOEMU-LABEL: func @float64
// NOEMU-SAME: f64
func @float64(%arg0: f64) { return }

} // end module

// -----

// Check that non-32-bit float types are kept untouched if the corresponding
// capabilities are available.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [Float16, Float64], []>, {}>
} {

// CHECK-LABEL: spv.func @float16
// CHECK-SAME: f16
// NOEMU-LABEL: spv.func @float16
// NOEMU-SAME: f16
func @float16(%arg0: f16) { return }

// CHECK-LABEL: spv.func @float64
// CHECK-SAME: f64
// NOEMU-LABEL: spv.func @float64
// NOEMU-SAME: f64
func @float64(%arg0: f64) { return }

} // end module

// -----

// Check that bf16 is not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-NOT: spv.func @bf16_type
func @bf16_type(%arg0: bf16) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Vector types
//===----------------------------------------------------------------------===//

// Check that capabilities for scalar types affects vector types too: no special
// capabilities available means using turning element types to 32-bit.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @int_vector
// CHECK-SAME: vector<2xi32>
// CHECK-SAME: vector<3xsi32>
// CHECK-SAME: vector<4xui32>
func @int_vector(
  %arg0: vector<2xi8>,
  %arg1: vector<3xsi16>,
  %arg2: vector<4xui64>
) { return }

// CHECK-LABEL: spv.func @float_vector
// CHECK-SAME: vector<2xf32>
// CHECK-SAME: vector<3xf32>
func @float_vector(
  %arg0: vector<2xf16>,
  %arg1: vector<3xf64>
) { return }

} // end module

// -----

// Check that capabilities for scalar types affects vector types too: having
// special capabilities means keep vector types untouched.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// CHECK-LABEL: spv.func @int_vector
// CHECK-SAME: vector<2xi8>
// CHECK-SAME: vector<3xsi16>
// CHECK-SAME: vector<4xui64>
func @int_vector(
  %arg0: vector<2xi8>,
  %arg1: vector<3xsi16>,
  %arg2: vector<4xui64>
) { return }

// CHECK-LABEL: spv.func @float_vector
// CHECK-SAME: vector<2xf16>
// CHECK-SAME: vector<3xf64>
func @float_vector(
  %arg0: vector<2xf16>,
  %arg1: vector<3xf64>
) { return }

// CHECK-LABEL: spv.func @one_element_vector
// CHECK-SAME: %{{.+}}: i32
func @one_element_vector(%arg0: vector<1xi32>) { return }

} // end module

// -----

// Check that > 4-element vectors are not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-NOT: spv.func @large_vector
func @large_vector(%arg0: vector<1024xi32>) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// MemRef types
//===----------------------------------------------------------------------===//

// Check memory spaces.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: func @memref_mem_space
// CHECK-SAME: StorageBuffer
// CHECK-SAME: Uniform
// CHECK-SAME: Workgroup
// CHECK-SAME: PushConstant
// CHECK-SAME: Private
// CHECK-SAME: Function
func @memref_mem_space(
    %arg0: memref<4xf32, 0>,
    %arg1: memref<4xf32, 4>,
    %arg2: memref<4xf32, 3>,
    %arg3: memref<4xf32, 7>,
    %arg4: memref<4xf32, 5>,
    %arg5: memref<4xf32, 6>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: convert them to 32-bit if not
// satisfied.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// An i1 is store in 8-bit, so 5xi1 has 40 bits, which is stored in 2xi32.
// CHECK-LABEL: spv.func @memref_1bit_type
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<2 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_1bit_type
// NOEMU-SAME: memref<5xi1>
func @memref_1bit_type(%arg0: memref<5xi1>) { return }

// CHECK-LABEL: spv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_8bit_StorageBuffer
// NOEMU-SAME: memref<16xi8>
func @memref_8bit_StorageBuffer(%arg0: memref<16xi8, 0>) { return }

// CHECK-LABEL: spv.func @memref_8bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_8bit_Uniform
// NOEMU-SAME: memref<16xsi8, 4>
func @memref_8bit_Uniform(%arg0: memref<16xsi8, 4>) { return }

// CHECK-LABEL: spv.func @memref_8bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_8bit_PushConstant
// NOEMU-SAME: memref<16xui8, 7>
func @memref_8bit_PushConstant(%arg0: memref<16xui8, 7>) { return }

// CHECK-LABEL: spv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_16bit_StorageBuffer
// NOEMU-SAME: memref<16xi16>
func @memref_16bit_StorageBuffer(%arg0: memref<16xi16, 0>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_16bit_Uniform
// NOEMU-SAME: memref<16xsi16, 4>
func @memref_16bit_Uniform(%arg0: memref<16xsi16, 4>) { return }

// CHECK-LABEL: spv.func @memref_16bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_16bit_PushConstant
// NOEMU-SAME: memref<16xui16, 7>
func @memref_16bit_PushConstant(%arg0: memref<16xui16, 7>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Input
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4>)>, Input>
// NOEMU-LABEL: func @memref_16bit_Input
// NOEMU-SAME: memref<16xf16, 9>
func @memref_16bit_Input(%arg3: memref<16xf16, 9>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Output
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f32, stride=4>)>, Output>
// NOEMU-LABEL: func @memref_16bit_Output
// NOEMU-SAME: memref<16xf16, 10>
func @memref_16bit_Output(%arg4: memref<16xf16, 10>) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [StoragePushConstant8, StoragePushConstant16],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: spv.func @memref_8bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, PushConstant>
// NOEMU-LABEL: spv.func @memref_8bit_PushConstant
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, PushConstant>
func @memref_8bit_PushConstant(%arg0: memref<16xi8, 7>) { return }

// CHECK-LABEL: spv.func @memref_16bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, PushConstant>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, PushConstant>
// NOEMU-LABEL: spv.func @memref_16bit_PushConstant
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, PushConstant>
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, PushConstant>
func @memref_16bit_PushConstant(
  %arg0: memref<16xi16, 7>,
  %arg1: memref<16xf16, 7>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [StorageBuffer8BitAccess, StorageBuffer16BitAccess],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: spv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, StorageBuffer>
// NOEMU-LABEL: spv.func @memref_8bit_StorageBuffer
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, StorageBuffer>
func @memref_8bit_StorageBuffer(%arg0: memref<16xi8, 0>) { return }

// CHECK-LABEL: spv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, StorageBuffer>
// NOEMU-LABEL: spv.func @memref_16bit_StorageBuffer
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, StorageBuffer>
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, StorageBuffer>
func @memref_16bit_StorageBuffer(
  %arg0: memref<16xi16, 0>,
  %arg1: memref<16xf16, 0>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageUniform16],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: spv.func @memref_8bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, Uniform>
// NOEMU-LABEL: spv.func @memref_8bit_Uniform
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i8, stride=1> [0])>, Uniform>
func @memref_8bit_Uniform(%arg0: memref<16xi8, 4>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, Uniform>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, Uniform>
// NOEMU-LABEL: spv.func @memref_16bit_Uniform
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2> [0])>, Uniform>
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2> [0])>, Uniform>
func @memref_16bit_Uniform(
  %arg0: memref<16xi16, 4>,
  %arg1: memref<16xf16, 4>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [StorageInputOutput16], [SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: spv.func @memref_16bit_Input
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2>)>, Input>
// NOEMU-LABEL: spv.func @memref_16bit_Input
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x f16, stride=2>)>, Input>
func @memref_16bit_Input(%arg3: memref<16xf16, 9>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Output
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2>)>, Output>
// NOEMU-LABEL: spv.func @memref_16bit_Output
// NOEMU-SAME: !spv.ptr<!spv.struct<(!spv.array<16 x i16, stride=2>)>, Output>
func @memref_16bit_Output(%arg4: memref<16xi16, 10>) { return }

} // end module

// -----

// Check that memref offset and strides affect the array size.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [StorageBuffer16BitAccess], [SPV_KHR_16bit_storage]>, {}>
} {

// CHECK-LABEL: spv.func @memref_offset_strides
func @memref_offset_strides(
// CHECK-SAME: !spv.array<64 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<72 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<256 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<64 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<88 x f32, stride=4> [0])>, StorageBuffer>
  %arg0: memref<16x4xf32, offset: 0, strides: [4, 1]>,  // tightly packed; row major
  %arg1: memref<16x4xf32, offset: 8, strides: [4, 1]>,  // offset 8
  %arg2: memref<16x4xf32, offset: 0, strides: [16, 1]>, // pad 12 after each row
  %arg3: memref<16x4xf32, offset: 0, strides: [1, 16]>, // tightly packed; col major
  %arg4: memref<16x4xf32, offset: 0, strides: [1, 22]>, // pad 4 after each col

// CHECK-SAME: !spv.array<64 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<72 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<256 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<64 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spv.array<88 x f16, stride=2> [0])>, StorageBuffer>
  %arg5: memref<16x4xf16, offset: 0, strides: [4, 1]>,
  %arg6: memref<16x4xf16, offset: 8, strides: [4, 1]>,
  %arg7: memref<16x4xf16, offset: 0, strides: [16, 1]>,
  %arg8: memref<16x4xf16, offset: 0, strides: [1, 16]>,
  %arg9: memref<16x4xf16, offset: 0, strides: [1, 22]>
) { return }

} // end module

// -----

// Dynamic shapes
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// Check that unranked shapes are not supported.
// CHECK-LABEL: func @unranked_memref
// CHECK-SAME: memref<*xi32>
func @unranked_memref(%arg0: memref<*xi32>) { return }

// CHECK-LABEL: func @memref_1bit_type
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_1bit_type
// NOEMU-SAME: memref<?xi1>
func @memref_1bit_type(%arg0: memref<?xi1>) { return }

// CHECK-LABEL: func @dynamic_dim_memref
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4> [0])>, StorageBuffer>
func @dynamic_dim_memref(%arg0: memref<8x?xi32>,
                         %arg1: memref<?x?xf32>) { return }

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: convert them to 32-bit if not
// satisfied.

// CHECK-LABEL: spv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_8bit_StorageBuffer
// NOEMU-SAME: memref<?xi8>
func @memref_8bit_StorageBuffer(%arg0: memref<?xi8, 0>) { return }

// CHECK-LABEL: spv.func @memref_8bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_8bit_Uniform
// NOEMU-SAME: memref<?xsi8, 4>
func @memref_8bit_Uniform(%arg0: memref<?xsi8, 4>) { return }

// CHECK-LABEL: spv.func @memref_8bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_8bit_PushConstant
// NOEMU-SAME: memref<?xui8, 7>
func @memref_8bit_PushConstant(%arg0: memref<?xui8, 7>) { return }

// CHECK-LABEL: spv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_16bit_StorageBuffer
// NOEMU-SAME: memref<?xi16>
func @memref_16bit_StorageBuffer(%arg0: memref<?xi16, 0>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Uniform
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_16bit_Uniform
// NOEMU-SAME: memref<?xsi16, 4>
func @memref_16bit_Uniform(%arg0: memref<?xsi16, 4>) { return }

// CHECK-LABEL: spv.func @memref_16bit_PushConstant
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_16bit_PushConstant
// NOEMU-SAME: memref<?xui16, 7>
func @memref_16bit_PushConstant(%arg0: memref<?xui16, 7>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Input
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4>)>, Input>
// NOEMU-LABEL: func @memref_16bit_Input
// NOEMU-SAME: memref<?xf16, 9>
func @memref_16bit_Input(%arg3: memref<?xf16, 9>) { return }

// CHECK-LABEL: spv.func @memref_16bit_Output
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<f32, stride=4>)>, Output>
// NOEMU-LABEL: func @memref_16bit_Output
// NOEMU-SAME: memref<?xf16, 10>
func @memref_16bit_Output(%arg4: memref<?xf16, 10>) { return }

} // end module

// -----

// Vector types
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: func @memref_vector
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<4 x vector<2xf32>, stride=8> [0])>, StorageBuffer>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16> [0])>, Uniform>
func @memref_vector(
    %arg0: memref<4xvector<2xf32>, 0>,
    %arg1: memref<4xvector<4xf32>, 4>)
{ return }

// CHECK-LABEL: func @dynamic_dim_memref_vector
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<vector<4xi32>, stride=16> [0])>, StorageBuffer>
// CHECK-SAME: !spv.ptr<!spv.struct<(!spv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>
func @dynamic_dim_memref_vector(%arg0: memref<8x?xvector<4xi32>>,
                         %arg1: memref<?x?xvector<2xf32>>)
{ return }

} // end module

// -----

// Vector types, check that sizes not available in SPIR-V are not transformed.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: func @memref_vector_wrong_size
// CHECK-SAME: memref<4xvector<5xf32>>
func @memref_vector_wrong_size(
    %arg0: memref<4xvector<5xf32>, 0>)
{ return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

// Check that tensor element types are kept untouched with proper capabilities.
module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, {}>
} {

// CHECK-LABEL: spv.func @int_tensor_types
// CHECK-SAME: !spv.array<32 x i64, stride=8>
// CHECK-SAME: !spv.array<32 x i32, stride=4>
// CHECK-SAME: !spv.array<32 x i16, stride=2>
// CHECK-SAME: !spv.array<32 x i8, stride=1>
func @int_tensor_types(
  %arg0: tensor<8x4xi64>,
  %arg1: tensor<8x4xi32>,
  %arg2: tensor<8x4xi16>,
  %arg3: tensor<8x4xi8>
) { return }

// CHECK-LABEL: spv.func @float_tensor_types
// CHECK-SAME: !spv.array<32 x f64, stride=8>
// CHECK-SAME: !spv.array<32 x f32, stride=4>
// CHECK-SAME: !spv.array<32 x f16, stride=2>
func @float_tensor_types(
  %arg0: tensor<8x4xf64>,
  %arg1: tensor<8x4xf32>,
  %arg2: tensor<8x4xf16>
) { return }

} // end module

// -----

// Check that tensor element types are changed to 32-bit without capabilities.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: spv.func @int_tensor_types
// CHECK-SAME: !spv.array<32 x i32, stride=4>
// CHECK-SAME: !spv.array<32 x i32, stride=4>
// CHECK-SAME: !spv.array<32 x i32, stride=4>
// CHECK-SAME: !spv.array<32 x i32, stride=4>
func @int_tensor_types(
  %arg0: tensor<8x4xi64>,
  %arg1: tensor<8x4xi32>,
  %arg2: tensor<8x4xi16>,
  %arg3: tensor<8x4xi8>
) { return }

// CHECK-LABEL: spv.func @float_tensor_types
// CHECK-SAME: !spv.array<32 x f32, stride=4>
// CHECK-SAME: !spv.array<32 x f32, stride=4>
// CHECK-SAME: !spv.array<32 x f32, stride=4>
func @float_tensor_types(
  %arg0: tensor<8x4xf64>,
  %arg1: tensor<8x4xf32>,
  %arg2: tensor<8x4xf16>
) { return }

} // end module

// -----

// Check that dynamic shapes are not supported.
module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, {}>
} {

// CHECK-LABEL: func @unranked_tensor
// CHECK-SAME: tensor<*xi32>
func @unranked_tensor(%arg0: tensor<*xi32>) { return }

// CHECK-LABEL: func @dynamic_dim_tensor
// CHECK-SAME: tensor<8x?xi32>
func @dynamic_dim_tensor(%arg0: tensor<8x?xi32>) { return }

} // end module
