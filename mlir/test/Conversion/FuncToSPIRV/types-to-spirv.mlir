// RUN: mlir-opt -split-input-file -convert-func-to-spirv %s -o - | FileCheck %s
// RUN: mlir-opt -split-input-file -convert-func-to-spirv="emulate-lt-32-bit-scalar-types=false" %s | \
// RUN:   FileCheck %s --check-prefix=NOEMU

//===----------------------------------------------------------------------===//
// Integer types
//===----------------------------------------------------------------------===//

// Check that non-32-bit integer types are converted to 32-bit types if the
// corresponding capabilities are not available.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @integer8
// CHECK-SAME: i32
// CHECK-SAME: si32
// CHECK-SAME: ui32
// NOEMU-LABEL: func.func @integer8
// NOEMU-SAME: i8
// NOEMU-SAME: si8
// NOEMU-SAME: ui8
func.func @integer8(%arg0: i8, %arg1: si8, %arg2: ui8) { return }

// CHECK-LABEL: spirv.func @integer16
// CHECK-SAME: i32
// CHECK-SAME: si32
// CHECK-SAME: ui32
// NOEMU-LABEL: func.func @integer16
// NOEMU-SAME: i16
// NOEMU-SAME: si16
// NOEMU-SAME: ui16
func.func @integer16(%arg0: i16, %arg1: si16, %arg2: ui16) { return }

// We do not truncate 64-bit types to 32-bit ones.
// CHECK-LABEL: func.func @integer64
// CHECK-SAME: i64
// CHECK-SAME: si64
// CHECK-SAME: ui64
// NOEMU-LABEL: func.func @integer64
// NOEMU-SAME: i64
// NOEMU-SAME: si64
// NOEMU-SAME: ui64
func.func @integer64(%arg0: i64, %arg1: si64, %arg2: ui64) { return }

// i128 is not supported by SPIR-V.
// CHECK-LABEL: func.func @integer128
// CHECK-SAME: i128
// NOEMU-LABEL: func.func @integer128
// NOEMU-SAME: i128
func.func @integer128(%arg0: i128) { return }

} // end module

// -----

// Check that non-32-bit integer types are kept untouched if the corresponding
// capabilities are available.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Int8, Int16, Int64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @integer8
// CHECK-SAME: i8
// CHECK-SAME: si8
// CHECK-SAME: ui8
// NOEMU-LABEL: spirv.func @integer8
// NOEMU-SAME: i8
// NOEMU-SAME: si8
// NOEMU-SAME: ui8
func.func @integer8(%arg0: i8, %arg1: si8, %arg2: ui8) { return }

// CHECK-LABEL: spirv.func @integer16
// CHECK-SAME: i16
// CHECK-SAME: si16
// CHECK-SAME: ui16
// NOEMU-LABEL: spirv.func @integer16
// NOEMU-SAME: i16
// NOEMU-SAME: si16
// NOEMU-SAME: ui16
func.func @integer16(%arg0: i16, %arg1: si16, %arg2: ui16) { return }

// CHECK-LABEL: spirv.func @integer64
// CHECK-SAME: i64
// CHECK-SAME: si64
// CHECK-SAME: ui64
// NOEMU-LABEL: spirv.func @integer64
// NOEMU-SAME: i64
// NOEMU-SAME: si64
// NOEMU-SAME: ui64
func.func @integer64(%arg0: i64, %arg1: si64, %arg2: ui64) { return }

} // end module

// -----

// Check that weird bitwidths are not supported.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-NOT: spirv.func @integer4
func.func @integer4(%arg0: i4) { return }

// CHECK-NOT: spirv.func @integer128
func.func @integer128(%arg0: i128) { return }

// CHECK-NOT: spirv.func @integer42
func.func @integer42(%arg0: i42) { return }

} // end module
// -----

//===----------------------------------------------------------------------===//
// Index type
//===----------------------------------------------------------------------===//

// The index type is always converted into i32.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @index_type
// CHECK-SAME: %{{.*}}: i32
func.func @index_type(%arg0: index) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Float types
//===----------------------------------------------------------------------===//

// Check that non-32-bit float types are converted to 32-bit types if the
// corresponding capabilities are not available.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @float16
// CHECK-SAME: f32
// NOEMU-LABEL: func.func @float16
// NOEMU-SAME: f16
func.func @float16(%arg0: f16) { return }

// CHECK-LABEL: func.func @float64
// CHECK-SAME: f64
// NOEMU-LABEL: func.func @float64
// NOEMU-SAME: f64
func.func @float64(%arg0: f64) { return }

// f80 is not supported by SPIR-V.
// CHECK-LABEL: func.func @float80
// CHECK-SAME: f80
// NOEMU-LABEL: func.func @float80
// NOEMU-SAME: f80
func.func @float80(%arg0: f80) { return }

} // end module

// -----

// Check that non-32-bit float types are kept untouched if the corresponding
// capabilities are available.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @float16
// CHECK-SAME: f16
// NOEMU-LABEL: spirv.func @float16
// NOEMU-SAME: f16
func.func @float16(%arg0: f16) { return }

// CHECK-LABEL: spirv.func @float64
// CHECK-SAME: f64
// NOEMU-LABEL: spirv.func @float64
// NOEMU-SAME: f64
func.func @float64(%arg0: f64) { return }

} // end module

// -----

// Check that bf16 is not supported.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-NOT: spirv.func @bf16_type
func.func @bf16_type(%arg0: bf16) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Vector types
//===----------------------------------------------------------------------===//

// Check that capabilities for scalar types affects vector types too: no special
// capabilities available means widening element types to 32-bit.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @int_vector
// CHECK-SAME: vector<2xi32>
// CHECK-SAME: vector<3xsi32>
func.func @int_vector(
  %arg0: vector<2xi8>,
  %arg1: vector<3xsi16>
) { return }

// CHECK-LABEL: spirv.func @float_vector
// CHECK-SAME: vector<2xf32>
func.func @float_vector(
  %arg0: vector<2xf16>
) { return }

// CHECK-LABEL: spirv.func @one_element_vector
// CHECK-SAME: %{{.+}}: i32
func.func @one_element_vector(%arg0: vector<1xi8>) { return }

} // end module

// -----

// Check that capabilities for scalar types affects vector types too: having
// special capabilities means keep vector types untouched.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @int_vector
// CHECK-SAME: vector<2xi8>
// CHECK-SAME: vector<3xsi16>
// CHECK-SAME: vector<4xui64>
func.func @int_vector(
  %arg0: vector<2xi8>,
  %arg1: vector<3xsi16>,
  %arg2: vector<4xui64>
) { return }

// CHECK-LABEL: spirv.func @float_vector
// CHECK-SAME: vector<2xf16>
// CHECK-SAME: vector<3xf64>
func.func @float_vector(
  %arg0: vector<2xf16>,
  %arg1: vector<3xf64>
) { return }

// CHECK-LABEL: spirv.func @one_element_vector
// CHECK-SAME: %{{.+}}: i32
func.func @one_element_vector(%arg0: vector<1xi32>) { return }

// CHECK-LABEL: spirv.func @zerod_vector
//  CHECK-SAME: %{{.+}}: f32
func.func @zerod_vector(%arg0: vector<f32>) { return }

} // end module

// -----

// Check that > 4-element vectors are not supported.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-NOT: spirv.func @large_vector
func.func @large_vector(%arg0: vector<1024xi32>) { return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// MemRef types
//===----------------------------------------------------------------------===//

// Check memory spaces.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @memref_mem_space
// CHECK-SAME: StorageBuffer
// CHECK-SAME: Uniform
// CHECK-SAME: Workgroup
// CHECK-SAME: PushConstant
// CHECK-SAME: Private
// CHECK-SAME: Function
func.func @memref_mem_space(
    %arg0: memref<4xf32, #spirv.storage_class<StorageBuffer>>,
    %arg1: memref<4xf32, #spirv.storage_class<Uniform>>,
    %arg2: memref<4xf32, #spirv.storage_class<Workgroup>>,
    %arg3: memref<4xf32, #spirv.storage_class<PushConstant>>,
    %arg4: memref<4xf32, #spirv.storage_class<Private>>,
    %arg5: memref<4xf32, #spirv.storage_class<Function>>
) { return }

// CHECK-LABEL: func @memref_1bit_type
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x i32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x i32>)>, Function>
// NOEMU-LABEL: func @memref_1bit_type
// NOEMU-SAME: memref<4x8xi1, #spirv.storage_class<StorageBuffer>>
// NOEMU-SAME: memref<4x8xi1, #spirv.storage_class<Function>>
func.func @memref_1bit_type(
    %arg0: memref<4x8xi1, #spirv.storage_class<StorageBuffer>>,
    %arg1: memref<4x8xi1, #spirv.storage_class<Function>>
) { return }

} // end module

// -----

// Reject memory spaces.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @numeric_memref_mem_space1
// CHECK-SAME: memref<4xf32>
// NOEMU-LABEL: func @numeric_memref_mem_space1
// NOEMU-SAME: memref<4xf32>
func.func @numeric_memref_mem_space1(%arg0: memref<4xf32>) { return }

// CHECK-LABEL: func @numeric_memref_mem_space2
// CHECK-SAME: memref<4xf32, 3>
// NOEMU-LABEL: func @numeric_memref_mem_space2
// NOEMU-SAME: memref<4xf32, 3>
func.func @numeric_memref_mem_space2(%arg0: memref<4xf32, 3>) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: convert them to 32-bit if not
// satisfied.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// An i1 is store in 8-bit, so 5xi1 has 40 bits, which is stored in 2xi32.
// CHECK-LABEL: spirv.func @memref_1bit_type
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<2 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_1bit_type
// NOEMU-SAME: memref<5xi1, #spirv.storage_class<StorageBuffer>>
func.func @memref_1bit_type(%arg0: memref<5xi1, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_8bit_StorageBuffer
// NOEMU-SAME: memref<16xi8, #spirv.storage_class<StorageBuffer>>
func.func @memref_8bit_StorageBuffer(%arg0: memref<16xi8, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_8bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<4 x si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_8bit_Uniform
// NOEMU-SAME: memref<16xsi8, #spirv.storage_class<Uniform>>
func.func @memref_8bit_Uniform(%arg0: memref<16xsi8, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: spirv.func @memref_8bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<4 x ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_8bit_PushConstant
// NOEMU-SAME: memref<16xui8, #spirv.storage_class<PushConstant>>
func.func @memref_8bit_PushConstant(%arg0: memref<16xui8, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_16bit_StorageBuffer
// NOEMU-SAME: memref<16xi16, #spirv.storage_class<StorageBuffer>>
func.func @memref_16bit_StorageBuffer(%arg0: memref<16xi16, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_16bit_Uniform
// NOEMU-SAME: memref<16xsi16, #spirv.storage_class<Uniform>>
func.func @memref_16bit_Uniform(%arg0: memref<16xsi16, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_16bit_PushConstant
// NOEMU-SAME: memref<16xui16, #spirv.storage_class<PushConstant>>
func.func @memref_16bit_PushConstant(%arg0: memref<16xui16, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Input
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x f32>)>, Input>
// NOEMU-LABEL: func @memref_16bit_Input
// NOEMU-SAME: memref<16xf16, #spirv.storage_class<Input>>
func.func @memref_16bit_Input(%arg3: memref<16xf16, #spirv.storage_class<Input>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Output
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<8 x f32>)>, Output>
// NOEMU-LABEL: func @memref_16bit_Output
// NOEMU-SAME: memref<16xf16, #spirv.storage_class<Output>>
func.func @memref_16bit_Output(%arg4: memref<16xf16, #spirv.storage_class<Output>>) { return }

// We do not truncate i64 to i32.

// CHECK-LABEL: func.func @memref_64bit_StorageBuffer
// CHECK-SAME: memref<16xi64, #spirv.storage_class<StorageBuffer>>
// NOEMU-LABEL: func.func @memref_64bit_StorageBuffer
// NOEMU-SAME: memref<16xi64, #spirv.storage_class<StorageBuffer>>
func.func @memref_64bit_StorageBuffer(%arg0: memref<16xi64, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: func.func @memref_64bit_Uniform
// CHECK-SAME: memref<16xsi64, #spirv.storage_class<Uniform>>
// NOEMU-LABEL: func.func @memref_64bit_Uniform
// NOEMU-SAME: memref<16xsi64, #spirv.storage_class<Uniform>>
func.func @memref_64bit_Uniform(%arg0: memref<16xsi64, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: func.func @memref_64bit_PushConstant
// CHECK-SAME: memref<16xui64, #spirv.storage_class<PushConstant>>
// NOEMU-LABEL: func.func @memref_64bit_PushConstant
// NOEMU-SAME: memref<16xui64, #spirv.storage_class<PushConstant>>
func.func @memref_64bit_PushConstant(%arg0: memref<16xui64, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: func.func @memref_64bit_Input
// CHECK-SAME: memref<16xf64, #spirv.storage_class<Input>>
// NOEMU-LABEL: func.func @memref_64bit_Input
// NOEMU-SAME: memref<16xf64, #spirv.storage_class<Input>>
func.func @memref_64bit_Input(%arg3: memref<16xf64, #spirv.storage_class<Input>>) { return }

// CHECK-LABEL: func.func @memref_64bit_Output
// CHECK-SAME: memref<16xf64, #spirv.storage_class<Output>>
// NOEMU-LABEL: func.func @memref_64bit_Output
// NOEMU-SAME: memref<16xf64, #spirv.storage_class<Output>>
func.func @memref_64bit_Output(%arg4: memref<16xf64, #spirv.storage_class<Output>>) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [StoragePushConstant8, StoragePushConstant16, Int64, Float64],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @memref_8bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, PushConstant>
// NOEMU-LABEL: spirv.func @memref_8bit_PushConstant
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, PushConstant>
func.func @memref_8bit_PushConstant(%arg0: memref<16xi8, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, PushConstant>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, PushConstant>
// NOEMU-LABEL: spirv.func @memref_16bit_PushConstant
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, PushConstant>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, PushConstant>
func.func @memref_16bit_PushConstant(
  %arg0: memref<16xi16, #spirv.storage_class<PushConstant>>,
  %arg1: memref<16xf16, #spirv.storage_class<PushConstant>>
) { return }

// CHECK-LABEL: spirv.func @memref_64bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, PushConstant>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, PushConstant>
// NOEMU-LABEL: spirv.func @memref_64bit_PushConstant
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, PushConstant>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, PushConstant>
func.func @memref_64bit_PushConstant(
  %arg0: memref<16xi64, #spirv.storage_class<PushConstant>>,
  %arg1: memref<16xf64, #spirv.storage_class<PushConstant>>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [StorageBuffer8BitAccess, StorageBuffer16BitAccess, Int64, Float64],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, StorageBuffer>
// NOEMU-LABEL: spirv.func @memref_8bit_StorageBuffer
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, StorageBuffer>
func.func @memref_8bit_StorageBuffer(%arg0: memref<16xi8, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, StorageBuffer>
// NOEMU-LABEL: spirv.func @memref_16bit_StorageBuffer
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, StorageBuffer>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, StorageBuffer>
func.func @memref_16bit_StorageBuffer(
  %arg0: memref<16xi16, #spirv.storage_class<StorageBuffer>>,
  %arg1: memref<16xf16, #spirv.storage_class<StorageBuffer>>
) { return }

// CHECK-LABEL: spirv.func @memref_64bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, StorageBuffer>
// NOEMU-LABEL: spirv.func @memref_64bit_StorageBuffer
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, StorageBuffer>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, StorageBuffer>
func.func @memref_64bit_StorageBuffer(
  %arg0: memref<16xi64, #spirv.storage_class<StorageBuffer>>,
  %arg1: memref<16xf64, #spirv.storage_class<StorageBuffer>>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [UniformAndStorageBuffer8BitAccess, StorageUniform16, Int64, Float64],
             [SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @memref_8bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, Uniform>
// NOEMU-LABEL: spirv.func @memref_8bit_Uniform
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i8, stride=1> [0])>, Uniform>
func.func @memref_8bit_Uniform(%arg0: memref<16xi8, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, Uniform>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, Uniform>
// NOEMU-LABEL: spirv.func @memref_16bit_Uniform
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16, stride=2> [0])>, Uniform>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16, stride=2> [0])>, Uniform>
func.func @memref_16bit_Uniform(
  %arg0: memref<16xi16, #spirv.storage_class<Uniform>>,
  %arg1: memref<16xf16, #spirv.storage_class<Uniform>>
) { return }

// CHECK-LABEL: spirv.func @memref_64bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, Uniform>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, Uniform>
// NOEMU-LABEL: spirv.func @memref_64bit_Uniform
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64, stride=8> [0])>, Uniform>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64, stride=8> [0])>, Uniform>
func.func @memref_64bit_Uniform(
  %arg0: memref<16xi64, #spirv.storage_class<Uniform>>,
  %arg1: memref<16xf64, #spirv.storage_class<Uniform>>
) { return }

} // end module

// -----

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: keep as-is when the capability
// and extension is available.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [StorageInputOutput16, Int64, Float64], [SPV_KHR_16bit_storage]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @memref_16bit_Input
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16>)>, Input>
// NOEMU-LABEL: spirv.func @memref_16bit_Input
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f16>)>, Input>
func.func @memref_16bit_Input(%arg3: memref<16xf16, #spirv.storage_class<Input>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Output
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16>)>, Output>
// NOEMU-LABEL: spirv.func @memref_16bit_Output
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i16>)>, Output>
func.func @memref_16bit_Output(%arg4: memref<16xi16, #spirv.storage_class<Output>>) { return }

// CHECK-LABEL: spirv.func @memref_64bit_Input
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64>)>, Input>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64>)>, Input>
// NOEMU-LABEL: spirv.func @memref_64bit_Input
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64>)>, Input>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64>)>, Input>
func.func @memref_64bit_Input(
  %arg0: memref<16xi64, #spirv.storage_class<Input>>,
  %arg1: memref<16xf64, #spirv.storage_class<Input>>
) { return }

// CHECK-LABEL: spirv.func @memref_64bit_Output
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64>)>, Output>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64>)>, Output>
// NOEMU-LABEL: spirv.func @memref_64bit_Output
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x i64>)>, Output>
// NOEMU-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<16 x f64>)>, Output>
func.func @memref_64bit_Output(
  %arg0: memref<16xi64, #spirv.storage_class<Output>>,
  %arg1: memref<16xf64, #spirv.storage_class<Output>>
) { return }

} // end module

// -----

// Check that memref offset and strides affect the array size.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [StorageBuffer16BitAccess], [SPV_KHR_16bit_storage]>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @memref_offset_strides
func.func @memref_offset_strides(
// CHECK-SAME: !spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<72 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<256 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<64 x f32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<88 x f32, stride=4> [0])>, StorageBuffer>
  %arg0: memref<16x4xf32, strided<[4, 1], offset: 0>, #spirv.storage_class<StorageBuffer>>,  // tightly packed; row major
  %arg1: memref<16x4xf32, strided<[4, 1], offset: 8>, #spirv.storage_class<StorageBuffer>>,  // offset 8
  %arg2: memref<16x4xf32, strided<[16, 1], offset: 0>, #spirv.storage_class<StorageBuffer>>, // pad 12 after each row
  %arg3: memref<16x4xf32, strided<[1, 16], offset: 0>, #spirv.storage_class<StorageBuffer>>, // tightly packed; col major
  %arg4: memref<16x4xf32, strided<[1, 22], offset: 0>, #spirv.storage_class<StorageBuffer>>, // pad 4 after each col

// CHECK-SAME: !spirv.array<64 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<72 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<256 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<64 x f16, stride=2> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.array<88 x f16, stride=2> [0])>, StorageBuffer>
  %arg5: memref<16x4xf16, strided<[4, 1], offset: 0>, #spirv.storage_class<StorageBuffer>>,
  %arg6: memref<16x4xf16, strided<[4, 1], offset: 8>, #spirv.storage_class<StorageBuffer>>,
  %arg7: memref<16x4xf16, strided<[16, 1], offset: 0>, #spirv.storage_class<StorageBuffer>>,
  %arg8: memref<16x4xf16, strided<[1, 16], offset: 0>, #spirv.storage_class<StorageBuffer>>,
  %arg9: memref<16x4xf16, strided<[1, 22], offset: 0>, #spirv.storage_class<StorageBuffer>>
) { return }

} // end module

// -----

// Dynamic shapes
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// Check that unranked shapes are not supported.
// CHECK-LABEL: func @unranked_memref
// CHECK-SAME: memref<*xi32>
func.func @unranked_memref(%arg0: memref<*xi32>) { return }

// CHECK-LABEL: func @memref_1bit_type
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_1bit_type
// NOEMU-SAME: memref<?xi1, #spirv.storage_class<StorageBuffer>>
func.func @memref_1bit_type(%arg0: memref<?xi1, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: func @dynamic_dim_memref
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
func.func @dynamic_dim_memref(
    %arg0: memref<8x?xi32, #spirv.storage_class<StorageBuffer>>,
    %arg1: memref<?x?xf32, #spirv.storage_class<StorageBuffer>>) { return }

// Check that using non-32-bit scalar types in interface storage classes
// requires special capability and extension: convert them to 32-bit if not
// satisfied.

// CHECK-LABEL: spirv.func @memref_8bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_8bit_StorageBuffer
// NOEMU-SAME: memref<?xi8, #spirv.storage_class<StorageBuffer>>
func.func @memref_8bit_StorageBuffer(%arg0: memref<?xi8, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_8bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_8bit_Uniform
// NOEMU-SAME: memref<?xsi8, #spirv.storage_class<Uniform>>
func.func @memref_8bit_Uniform(%arg0: memref<?xsi8, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: spirv.func @memref_8bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_8bit_PushConstant
// NOEMU-SAME: memref<?xui8, #spirv.storage_class<PushConstant>>
func.func @memref_8bit_PushConstant(%arg0: memref<?xui8, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_StorageBuffer
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
// NOEMU-LABEL: func @memref_16bit_StorageBuffer
// NOEMU-SAME: memref<?xi16, #spirv.storage_class<StorageBuffer>>
func.func @memref_16bit_StorageBuffer(%arg0: memref<?xi16, #spirv.storage_class<StorageBuffer>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Uniform
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<si32, stride=4> [0])>, Uniform>
// NOEMU-LABEL: func @memref_16bit_Uniform
// NOEMU-SAME: memref<?xsi16, #spirv.storage_class<Uniform>>
func.func @memref_16bit_Uniform(%arg0: memref<?xsi16, #spirv.storage_class<Uniform>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_PushConstant
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<ui32, stride=4> [0])>, PushConstant>
// NOEMU-LABEL: func @memref_16bit_PushConstant
// NOEMU-SAME: memref<?xui16, #spirv.storage_class<PushConstant>>
func.func @memref_16bit_PushConstant(%arg0: memref<?xui16, #spirv.storage_class<PushConstant>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Input
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, Input>
// NOEMU-LABEL: func @memref_16bit_Input
// NOEMU-SAME: memref<?xf16, #spirv.storage_class<Input>>
func.func @memref_16bit_Input(%arg3: memref<?xf16, #spirv.storage_class<Input>>) { return }

// CHECK-LABEL: spirv.func @memref_16bit_Output
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32>)>, Output>
// NOEMU-LABEL: func @memref_16bit_Output
// NOEMU-SAME: memref<?xf16, #spirv.storage_class<Output>>
func.func @memref_16bit_Output(%arg4: memref<?xf16, #spirv.storage_class<Output>>) { return }

} // end module

// -----

// Vector types
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @memref_vector
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<2xf32>, stride=8> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<4 x vector<4xf32>, stride=16> [0])>, Uniform>
func.func @memref_vector(
    %arg0: memref<4xvector<2xf32>, #spirv.storage_class<StorageBuffer>>,
    %arg1: memref<4xvector<4xf32>, #spirv.storage_class<Uniform>>)
{ return }

// CHECK-LABEL: func @dynamic_dim_memref_vector
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xi32>, stride=16> [0])>, StorageBuffer>
// CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<2xf32>, stride=8> [0])>, StorageBuffer>
func.func @dynamic_dim_memref_vector(
    %arg0: memref<8x?xvector<4xi32>, #spirv.storage_class<StorageBuffer>>,
    %arg1: memref<?x?xvector<2xf32>, #spirv.storage_class<StorageBuffer>>)
{ return }

} // end module

// -----

// Vector types, check that sizes not available in SPIR-V are not transformed.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @memref_vector_wrong_size
// CHECK-SAME: memref<4xvector<5xf32>, #spirv.storage_class<StorageBuffer>>
func.func @memref_vector_wrong_size(
    %arg0: memref<4xvector<5xf32>, #spirv.storage_class<StorageBuffer>>)
{ return }

} // end module

// -----

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

// Check that tensor element types are kept untouched with proper capabilities.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Int8, Int16, Int64, Float16, Float64], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @int_tensor_types
// CHECK-SAME: !spirv.array<32 x i64>
// CHECK-SAME: !spirv.array<32 x i32>
// CHECK-SAME: !spirv.array<32 x i16>
// CHECK-SAME: !spirv.array<32 x i8>
func.func @int_tensor_types(
  %arg0: tensor<8x4xi64>,
  %arg1: tensor<8x4xi32>,
  %arg2: tensor<8x4xi16>,
  %arg3: tensor<8x4xi8>
) { return }

// CHECK-LABEL: spirv.func @float_tensor_types
// CHECK-SAME: !spirv.array<32 x f64>
// CHECK-SAME: !spirv.array<32 x f32>
// CHECK-SAME: !spirv.array<32 x f16>
func.func @float_tensor_types(
  %arg0: tensor<8x4xf64>,
  %arg1: tensor<8x4xf32>,
  %arg2: tensor<8x4xf16>
) { return }

} // end module

// -----

// Check that tensor element types are changed to 32-bit without capabilities.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: spirv.func @int_tensor_types
// CHECK-SAME: !spirv.array<32 x i32>
// CHECK-SAME: !spirv.array<32 x i32>
// CHECK-SAME: !spirv.array<32 x i32>
func.func @int_tensor_types(
  %arg1: tensor<8x4xi32>,
  %arg2: tensor<8x4xi16>,
  %arg3: tensor<8x4xi8>
) { return }

// CHECK-LABEL: spirv.func @float_tensor_types
// CHECK-SAME: !spirv.array<32 x f32>
// CHECK-SAME: !spirv.array<32 x f32>
func.func @float_tensor_types(
  %arg1: tensor<8x4xf32>,
  %arg2: tensor<8x4xf16>
) { return }

} // end module

// -----

// Check that dynamic shapes are not supported.
module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, #spirv.resource_limits<>>
} {

// CHECK-LABEL: func @unranked_tensor
// CHECK-SAME: tensor<*xi32>
func.func @unranked_tensor(%arg0: tensor<*xi32>) { return }

// CHECK-LABEL: func @dynamic_dim_tensor
// CHECK-SAME: tensor<8x?xi32>
func.func @dynamic_dim_tensor(%arg0: tensor<8x?xi32>) { return }

} // end module
