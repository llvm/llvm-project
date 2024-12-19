// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t.bc --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
// RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %t.bc -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION

// CHECK-SPIRV: Capability BitInstructions
// CHECK-SPIRV: Extension "SPV_KHR_bit_instructions"
// CHECK-NO-EXTENSION: LLVM ERROR: bitfield_insert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long]] %[[#insertinsert_long]]
kernel void testInsert_long(long b, long i, global long *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong]] %[[#insertinsert_ulong]]
kernel void testInsert_ulong(ulong b, ulong i, global ulong *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int]] %[[#insertinsert_int]]
kernel void testInsert_int(int b, int i, global int *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint]] %[[#insertinsert_uint]]
kernel void testInsert_uint(uint b, uint i, global uint *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short]] %[[#insertinsert_short]]
kernel void testInsert_short(short b, short i, global short *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort]] %[[#insertinsert_ushort]]
kernel void testInsert_ushort(ushort b, ushort i, global ushort *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long2]] %[[#insertinsert_long2]]
kernel void testInsert_long2(long2 b, long2 i, global long2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong2]] %[[#insertinsert_ulong2]]
kernel void testInsert_ulong2(ulong2 b, ulong2 i, global ulong2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int2]] %[[#insertinsert_int2]]
kernel void testInsert_int2(int2 b, int2 i, global int2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint2]] %[[#insertinsert_uint2]]
kernel void testInsert_uint2(uint2 b, uint2 i, global uint2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short2]] %[[#insertinsert_short2]]
kernel void testInsert_short2(short2 b, short2 i, global short2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort2]] %[[#insertinsert_ushort2]]
kernel void testInsert_ushort2(ushort2 b, ushort2 i, global ushort2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_char2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_char2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char2]] %[[#insertinsert_char2]]
kernel void testInsert_char2(char2 b, char2 i, global char2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uchar2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uchar2:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar2]] %[[#insertinsert_uchar2]]
kernel void testInsert_uchar2(uchar2 b, uchar2 i, global uchar2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long3]] %[[#insertinsert_long3]]
kernel void testInsert_long3(long3 b, long3 i, global long3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong3]] %[[#insertinsert_ulong3]]
kernel void testInsert_ulong3(ulong3 b, ulong3 i, global ulong3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int3]] %[[#insertinsert_int3]]
kernel void testInsert_int3(int3 b, int3 i, global int3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint3]] %[[#insertinsert_uint3]]
kernel void testInsert_uint3(uint3 b, uint3 i, global uint3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short3]] %[[#insertinsert_short3]]
kernel void testInsert_short3(short3 b, short3 i, global short3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort3]] %[[#insertinsert_ushort3]]
kernel void testInsert_ushort3(ushort3 b, ushort3 i, global ushort3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_char3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_char3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char3]] %[[#insertinsert_char3]]
kernel void testInsert_char3(char3 b, char3 i, global char3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uchar3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uchar3:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar3]] %[[#insertinsert_uchar3]]
kernel void testInsert_uchar3(uchar3 b, uchar3 i, global uchar3 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long4]] %[[#insertinsert_long4]]
kernel void testInsert_long4(long4 b, long4 i, global long4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong4]] %[[#insertinsert_ulong4]]
kernel void testInsert_ulong4(ulong4 b, ulong4 i, global ulong4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int4]] %[[#insertinsert_int4]]
kernel void testInsert_int4(int4 b, int4 i, global int4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint4]] %[[#insertinsert_uint4]]
kernel void testInsert_uint4(uint4 b, uint4 i, global uint4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short4]] %[[#insertinsert_short4]]
kernel void testInsert_short4(short4 b, short4 i, global short4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort4]] %[[#insertinsert_ushort4]]
kernel void testInsert_ushort4(ushort4 b, ushort4 i, global ushort4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_char4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_char4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char4]] %[[#insertinsert_char4]]
kernel void testInsert_char4(char4 b, char4 i, global char4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uchar4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uchar4:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar4]] %[[#insertinsert_uchar4]]
kernel void testInsert_uchar4(uchar4 b, uchar4 i, global uchar4 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long8]] %[[#insertinsert_long8]]
kernel void testInsert_long8(long8 b, long8 i, global long8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong8]] %[[#insertinsert_ulong8]]
kernel void testInsert_ulong8(ulong8 b, ulong8 i, global ulong8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int8]] %[[#insertinsert_int8]]
kernel void testInsert_int8(int8 b, int8 i, global int8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint8]] %[[#insertinsert_uint8]]
kernel void testInsert_uint8(uint8 b, uint8 i, global uint8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short8]] %[[#insertinsert_short8]]
kernel void testInsert_short8(short8 b, short8 i, global short8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort8]] %[[#insertinsert_ushort8]]
kernel void testInsert_ushort8(ushort8 b, ushort8 i, global ushort8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_char8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_char8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char8]] %[[#insertinsert_char8]]
kernel void testInsert_char8(char8 b, char8 i, global char8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uchar8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uchar8:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar8]] %[[#insertinsert_uchar8]]
kernel void testInsert_uchar8(uchar8 b, uchar8 i, global uchar8 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_long16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_long16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long16]] %[[#insertinsert_long16]]
kernel void testInsert_long16(long16 b, long16 i, global long16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ulong16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ulong16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong16]] %[[#insertinsert_ulong16]]
kernel void testInsert_ulong16(ulong16 b, ulong16 i, global ulong16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_int16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_int16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int16]] %[[#insertinsert_int16]]
kernel void testInsert_int16(int16 b, int16 i, global int16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uint16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uint16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint16]] %[[#insertinsert_uint16]]
kernel void testInsert_uint16(uint16 b, uint16 i, global uint16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_short16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_short16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short16]] %[[#insertinsert_short16]]
kernel void testInsert_short16(short16 b, short16 i, global short16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_ushort16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_ushort16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort16]] %[[#insertinsert_ushort16]]
kernel void testInsert_ushort16(ushort16 b, ushort16 i, global ushort16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_char16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_char16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char16]] %[[#insertinsert_char16]]
kernel void testInsert_char16(char16 b, char16 i, global char16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#insertbase_uchar16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#insertinsert_uchar16:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar16]] %[[#insertinsert_uchar16]]
kernel void testInsert_uchar16(uchar16 b, uchar16 i, global uchar16 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long2(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int2(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short2(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char2(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long3(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int3(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short3(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char3(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long4(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int4(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short4(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char4(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long8(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int8(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short8(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char8(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_long16(long b, ulong bu, global long *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_int16(int b, uint bu, global int *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_short16(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
kernel void testExtractS_char16(char b, uchar bu, global char *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long(long b, ulong bu, global ulong *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int(int b, uint bu, global uint *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short(short b, ushort bu, global ushort *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char(char b, uchar bu, global uchar *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long2(long2 b, ulong2 bu, global ulong2 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int2(int2 b, uint2 bu, global uint2 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short2(short2 b, ushort2 bu, global ushort2 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char2(char2 b, uchar2 bu, global uchar2 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long3(long3 b, ulong3 bu, global ulong3 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int3(int3 b, uint3 bu, global uint3 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short3(short3 b, ushort3 bu, global ushort3 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char3(char3 b, uchar3 bu, global uchar3 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long4(long4 b, ulong4 bu, global ulong4 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int4(int4 b, uint4 bu, global uint4 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short4(short4 b, ushort4 bu, global ushort4 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char4(char4 b, uchar4 bu, global uchar4 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long8(long8 b, ulong8 bu, global ulong8 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int8(int8 b, uint8 bu, global uint8 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short8(short8 b, ushort8 bu, global ushort8 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char8(char8 b, uchar8 bu, global uchar8 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_long16(long16 b, ulong16 bu, global ulong16 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_int16(int16 b, uint16 bu, global uint16 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_short16(short16 b, ushort16 bu, global ushort16 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
// CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
kernel void testExtractU_char16(char16 b, uchar16 bu, global uchar16 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long(long b, global long *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong(ulong b, global ulong *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int(int b, global int *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint(uint b, global uint *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short(short b, global short *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort(ushort b, global ushort *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char(char b, global char *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar(uchar b, global uchar *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long2(long2 b, global long2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong2(ulong2 b, global ulong2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int2(int2 b, global int2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint2(uint2 b, global uint2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short2(short2 b, global short2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort2(ushort2 b, global ushort2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char2(char2 b, global char2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar2(uchar2 b, global uchar2 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long3(long3 b, global long3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong3(ulong3 b, global ulong3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int3(int3 b, global int3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint3(uint3 b, global uint3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short3(short3 b, global short3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort3(ushort3 b, global ushort3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char3(char3 b, global char3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar3(uchar3 b, global uchar3 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long4(long4 b, global long4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong4(ulong4 b, global ulong4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int4(int4 b, global int4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint4(uint4 b, global uint4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short4(short4 b, global short4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort4(ushort4 b, global ushort4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char4(char4 b, global char4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar4(uchar4 b, global uchar4 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long8(long8 b, global long8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong8(ulong8 b, global ulong8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int8(int8 b, global int8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint8(uint8 b, global uint8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short8(short8 b, global short8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort8(ushort8 b, global ushort8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char8(char8 b, global char8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar8(uchar8 b, global uchar8 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_long16(long16 b, global long16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ulong16(ulong16 b, global ulong16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_int16(int16 b, global int16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uint16(uint16 b, global uint16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_short16(short16 b, global short16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_ushort16(ushort16 b, global ushort16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_char16(char16 b, global char16 *res) {
  *res = bit_reverse(b);
}

// CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
// CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
// CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
kernel void testBitReverse_uchar16(uchar16 b, global uchar16 *res) {
  *res = bit_reverse(b);
}
