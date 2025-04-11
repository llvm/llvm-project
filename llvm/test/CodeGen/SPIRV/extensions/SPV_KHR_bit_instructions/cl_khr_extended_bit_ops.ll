; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %} 
;
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: LLVM ERROR: bitfield_insert: the builtin requires the following SPIR-V extension: SPV_KHR_bit_instructions
;
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long]] %[[#insertinsert_long]]
;
; OpenCL equivalent.
; kernel void testInsert_long(long b, long i, global long *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long(i64 noundef %b, i64 noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z15bitfield_insertlljj(i64 noundef %b, i64 noundef %i, i32 noundef 4, i32 noundef 2) #2
  store i64 %call, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z15bitfield_insertlljj(i64 noundef, i64 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong]] %[[#insertinsert_ulong]]
; OpenCL equivalent.
; kernel void testInsert_ulong(ulong b, ulong i, global ulong *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong(i64 noundef %b, i64 noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z15bitfield_insertmmjj(i64 noundef %b, i64 noundef %i, i32 noundef 4, i32 noundef 2) #2
  store i64 %call, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z15bitfield_insertmmjj(i64 noundef, i64 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int]] %[[#insertinsert_int]]
; OpenCL equivalent.
; kernel void testInsert_int(int b, int i, global int *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int(i32 noundef %b, i32 noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z15bitfield_insertiijj(i32 noundef %b, i32 noundef %i, i32 noundef 4, i32 noundef 2) #2
  store i32 %call, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z15bitfield_insertiijj(i32 noundef, i32 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint]] %[[#insertinsert_uint]]
; OpenCL equivalent.
; kernel void testInsert_uint(uint b, uint i, global uint *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint(i32 noundef %b, i32 noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z15bitfield_insertjjjj(i32 noundef %b, i32 noundef %i, i32 noundef 4, i32 noundef 2) #2
  store i32 %call, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z15bitfield_insertjjjj(i32 noundef, i32 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short]] %[[#insertinsert_short]]
; OpenCL equivalent.
; kernel void testInsert_short(short b, short i, global short *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short(i16 noundef signext %b, i16 noundef signext %i, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z15bitfield_insertssjj(i16 noundef signext %b, i16 noundef signext %i, i32 noundef 4, i32 noundef 2) #2
  store i16 %call, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func signext i16 @_Z15bitfield_insertssjj(i16 noundef signext, i16 noundef signext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort]] %[[#insertinsert_ushort]]
; OpenCL equivalentxr.
; kernel void testInsert_ushort(ushort b, ushort i, global ushort *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort(i16 noundef zeroext %b, i16 noundef zeroext %i, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func zeroext i16 @_Z15bitfield_insertttjj(i16 noundef zeroext %b, i16 noundef zeroext %i, i32 noundef 4, i32 noundef 2) #2
  store i16 %call, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func zeroext i16 @_Z15bitfield_insertttjj(i16 noundef zeroext, i16 noundef zeroext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long2]] %[[#insertinsert_long2]]
; OpenCL equivalent.
; kernel void testInsert_long2(long2 b, long2 i, global long2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long2(<2 x i64> noundef %b, <2 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <2 x i64> @_Z15bitfield_insertDv2_lS_jj(<2 x i64> noundef %b, <2 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i64> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <2 x i64> @_Z15bitfield_insertDv2_lS_jj(<2 x i64> noundef, <2 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong2]] %[[#insertinsert_ulong2]]
; OpenCL equivalent.
; kernel void testInsert_ulong2(ulong2 b, ulong2 i, global ulong2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong2(<2 x i64> noundef %b, <2 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <2 x i64> @_Z15bitfield_insertDv2_mS_jj(<2 x i64> noundef %b, <2 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i64> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <2 x i64> @_Z15bitfield_insertDv2_mS_jj(<2 x i64> noundef, <2 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int2]] %[[#insertinsert_int2]]
; OpenCL equivalent.
; kernel void testInsert_int2(int2 b, int2 i, global int2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int2(<2 x i32> noundef %b, <2 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <2 x i32> @_Z15bitfield_insertDv2_iS_jj(<2 x i32> noundef %b, <2 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i32> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <2 x i32> @_Z15bitfield_insertDv2_iS_jj(<2 x i32> noundef, <2 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint2]] %[[#insertinsert_uint2]]
; OpenCL equivalent.
; kernel void testInsert_uint2(uint2 b, uint2 i, global uint2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint2(<2 x i32> noundef %b, <2 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <2 x i32> @_Z15bitfield_insertDv2_jS_jj(<2 x i32> noundef %b, <2 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i32> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <2 x i32> @_Z15bitfield_insertDv2_jS_jj(<2 x i32> noundef, <2 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short2]] %[[#insertinsert_short2]]
; OpenCL equivalent.
; kernel void testInsert_short2(short2 b, short2 i, global short2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short2(<2 x i16> noundef %b, <2 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <2 x i16> @_Z15bitfield_insertDv2_sS_jj(<2 x i16> noundef %b, <2 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i16> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <2 x i16> @_Z15bitfield_insertDv2_sS_jj(<2 x i16> noundef, <2 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort2]] %[[#insertinsert_ushort2]]
; OpenCL equivalent.
; kernel void testInsert_ushort2(ushort2 b, ushort2 i, global ushort2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort2(<2 x i16> noundef %b, <2 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <2 x i16> @_Z15bitfield_insertDv2_tS_jj(<2 x i16> noundef %b, <2 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i16> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <2 x i16> @_Z15bitfield_insertDv2_tS_jj(<2 x i16> noundef, <2 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_char2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_char2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char2]] %[[#insertinsert_char2]]
; OpenCL equivalent.
; kernel void testInsert_char2(char2 b, char2 i, global char2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_char2(<2 x i8> noundef %b, <2 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func <2 x i8> @_Z15bitfield_insertDv2_cS_jj(<2 x i8> noundef %b, <2 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i8> %call, ptr addrspace(1) %res, align 2, !tbaa !22
  ret void
}

declare spir_func <2 x i8> @_Z15bitfield_insertDv2_cS_jj(<2 x i8> noundef, <2 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uchar2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uchar2:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar2]] %[[#insertinsert_uchar2]]
; OpenCL equivalent.
; kernel void testInsert_uchar2(uchar2 b, uchar2 i, global uchar2 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uchar2(<2 x i8> noundef %b, <2 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func <2 x i8> @_Z15bitfield_insertDv2_hS_jj(<2 x i8> noundef %b, <2 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <2 x i8> %call, ptr addrspace(1) %res, align 2, !tbaa !22
  ret void
}

declare spir_func <2 x i8> @_Z15bitfield_insertDv2_hS_jj(<2 x i8> noundef, <2 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long3]] %[[#insertinsert_long3]]
; OpenCL equivalent.
; kernel void testInsert_long3(long3 b, long3 i, global long3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long3(<3 x i64> noundef %b, <3 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <3 x i64> @_Z15bitfield_insertDv3_lS_jj(<3 x i64> noundef %b, <3 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i64> %call, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i64> %extractVec5, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <3 x i64> @_Z15bitfield_insertDv3_lS_jj(<3 x i64> noundef, <3 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong3]] %[[#insertinsert_ulong3]]
; OpenCL equivalent.
; kernel void testInsert_ulong3(ulong3 b, ulong3 i, global ulong3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong3(<3 x i64> noundef %b, <3 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <3 x i64> @_Z15bitfield_insertDv3_mS_jj(<3 x i64> noundef %b, <3 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i64> %call, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i64> %extractVec5, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <3 x i64> @_Z15bitfield_insertDv3_mS_jj(<3 x i64> noundef, <3 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int3]] %[[#insertinsert_int3]]
; OpenCL equivalent.
; kernel void testInsert_int3(int3 b, int3 i, global int3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int3(<3 x i32> noundef %b, <3 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <3 x i32> @_Z15bitfield_insertDv3_iS_jj(<3 x i32> noundef %b, <3 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i32> %call, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i32> %extractVec5, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <3 x i32> @_Z15bitfield_insertDv3_iS_jj(<3 x i32> noundef, <3 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint3]] %[[#insertinsert_uint3]]
; OpenCL equivalent.
; kernel void testInsert_uint3(uint3 b, uint3 i, global uint3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint3(<3 x i32> noundef %b, <3 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <3 x i32> @_Z15bitfield_insertDv3_jS_jj(<3 x i32> noundef %b, <3 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i32> %call, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i32> %extractVec5, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <3 x i32> @_Z15bitfield_insertDv3_jS_jj(<3 x i32> noundef, <3 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short3]] %[[#insertinsert_short3]]
; OpenCL equivalent.
; kernel void testInsert_short3(short3 b, short3 i, global short3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short3(<3 x i16> noundef %b, <3 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <3 x i16> @_Z15bitfield_insertDv3_sS_jj(<3 x i16> noundef %b, <3 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i16> %call, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i16> %extractVec5, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <3 x i16> @_Z15bitfield_insertDv3_sS_jj(<3 x i16> noundef, <3 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort3]] %[[#insertinsert_ushort3]]
; OpenCL equivalent.
; kernel void testInsert_ushort3(ushort3 b, ushort3 i, global ushort3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort3(<3 x i16> noundef %b, <3 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <3 x i16> @_Z15bitfield_insertDv3_tS_jj(<3 x i16> noundef %b, <3 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i16> %call, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i16> %extractVec5, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <3 x i16> @_Z15bitfield_insertDv3_tS_jj(<3 x i16> noundef, <3 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_char3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_char3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char3]] %[[#insertinsert_char3]]
; OpenCL equivalent.
; kernel void testInsert_char3(char3 b, char3 i, global char3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_char3(<3 x i8> noundef %b, <3 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <3 x i8> @_Z15bitfield_insertDv3_cS_jj(<3 x i8> noundef %b, <3 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i8> %call, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i8> %extractVec5, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <3 x i8> @_Z15bitfield_insertDv3_cS_jj(<3 x i8> noundef, <3 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uchar3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uchar3:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar3]] %[[#insertinsert_uchar3]]
; OpenCL equivalent.
; kernel void testInsert_uchar3(uchar3 b, uchar3 i, global uchar3 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uchar3(<3 x i8> noundef %b, <3 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <3 x i8> @_Z15bitfield_insertDv3_hS_jj(<3 x i8> noundef %b, <3 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  %extractVec5 = shufflevector <3 x i8> %call, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i8> %extractVec5, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <3 x i8> @_Z15bitfield_insertDv3_hS_jj(<3 x i8> noundef, <3 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long4]] %[[#insertinsert_long4]]
; OpenCL equivalent.
; kernel void testInsert_long4(long4 b, long4 i, global long4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long4(<4 x i64> noundef %b, <4 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <4 x i64> @_Z15bitfield_insertDv4_lS_jj(<4 x i64> noundef %b, <4 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i64> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <4 x i64> @_Z15bitfield_insertDv4_lS_jj(<4 x i64> noundef, <4 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong4]] %[[#insertinsert_ulong4]]
; OpenCL equivalent.
; kernel void testInsert_ulong4(ulong4 b, ulong4 i, global ulong4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong4(<4 x i64> noundef %b, <4 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <4 x i64> @_Z15bitfield_insertDv4_mS_jj(<4 x i64> noundef %b, <4 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i64> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <4 x i64> @_Z15bitfield_insertDv4_mS_jj(<4 x i64> noundef, <4 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int4]] %[[#insertinsert_int4]]
; OpenCL equivalent.
; kernel void testInsert_int4(int4 b, int4 i, global int4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int4(<4 x i32> noundef %b, <4 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <4 x i32> @_Z15bitfield_insertDv4_iS_jj(<4 x i32> noundef %b, <4 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i32> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <4 x i32> @_Z15bitfield_insertDv4_iS_jj(<4 x i32> noundef, <4 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint4]] %[[#insertinsert_uint4]]
; OpenCL equivalent.
; kernel void testInsert_uint4(uint4 b, uint4 i, global uint4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint4(<4 x i32> noundef %b, <4 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <4 x i32> @_Z15bitfield_insertDv4_jS_jj(<4 x i32> noundef %b, <4 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i32> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <4 x i32> @_Z15bitfield_insertDv4_jS_jj(<4 x i32> noundef, <4 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short4]] %[[#insertinsert_short4]]
; OpenCL equivalent.
; kernel void testInsert_short4(short4 b, short4 i, global short4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short4(<4 x i16> noundef %b, <4 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <4 x i16> @_Z15bitfield_insertDv4_sS_jj(<4 x i16> noundef %b, <4 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i16> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <4 x i16> @_Z15bitfield_insertDv4_sS_jj(<4 x i16> noundef, <4 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort4]] %[[#insertinsert_ushort4]]
; OpenCL equivalent.
; kernel void testInsert_ushort4(ushort4 b, ushort4 i, global ushort4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort4(<4 x i16> noundef %b, <4 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <4 x i16> @_Z15bitfield_insertDv4_tS_jj(<4 x i16> noundef %b, <4 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i16> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <4 x i16> @_Z15bitfield_insertDv4_tS_jj(<4 x i16> noundef, <4 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_char4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_char4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char4]] %[[#insertinsert_char4]]
; OpenCL equivalent.
; kernel void testInsert_char4(char4 b, char4 i, global char4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_char4(<4 x i8> noundef %b, <4 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <4 x i8> @_Z15bitfield_insertDv4_cS_jj(<4 x i8> noundef %b, <4 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i8> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <4 x i8> @_Z15bitfield_insertDv4_cS_jj(<4 x i8> noundef, <4 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uchar4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uchar4:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar4]] %[[#insertinsert_uchar4]]
; OpenCL equivalent.
; kernel void testInsert_uchar4(uchar4 b, uchar4 i, global uchar4 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uchar4(<4 x i8> noundef %b, <4 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <4 x i8> @_Z15bitfield_insertDv4_hS_jj(<4 x i8> noundef %b, <4 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <4 x i8> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <4 x i8> @_Z15bitfield_insertDv4_hS_jj(<4 x i8> noundef, <4 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long8]] %[[#insertinsert_long8]]
; OpenCL equivalent.
; kernel void testInsert_long8(long8 b, long8 i, global long8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long8(<8 x i64> noundef %b, <8 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <8 x i64> @_Z15bitfield_insertDv8_lS_jj(<8 x i64> noundef %b, <8 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i64> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <8 x i64> @_Z15bitfield_insertDv8_lS_jj(<8 x i64> noundef, <8 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong8]] %[[#insertinsert_ulong8]]
; OpenCL equivalent.
; kernel void testInsert_ulong8(ulong8 b, ulong8 i, global ulong8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong8(<8 x i64> noundef %b, <8 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <8 x i64> @_Z15bitfield_insertDv8_mS_jj(<8 x i64> noundef %b, <8 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i64> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <8 x i64> @_Z15bitfield_insertDv8_mS_jj(<8 x i64> noundef, <8 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int8]] %[[#insertinsert_int8]]
; OpenCL equivalent.
; kernel void testInsert_int8(int8 b, int8 i, global int8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int8(<8 x i32> noundef %b, <8 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <8 x i32> @_Z15bitfield_insertDv8_iS_jj(<8 x i32> noundef %b, <8 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i32> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <8 x i32> @_Z15bitfield_insertDv8_iS_jj(<8 x i32> noundef, <8 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint8]] %[[#insertinsert_uint8]]
; OpenCL equivalent.
; kernel void testInsert_uint8(uint8 b, uint8 i, global uint8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint8(<8 x i32> noundef %b, <8 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <8 x i32> @_Z15bitfield_insertDv8_jS_jj(<8 x i32> noundef %b, <8 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i32> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <8 x i32> @_Z15bitfield_insertDv8_jS_jj(<8 x i32> noundef, <8 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short8]] %[[#insertinsert_short8]]
; OpenCL equivalent.
; kernel void testInsert_short8(short8 b, short8 i, global short8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short8(<8 x i16> noundef %b, <8 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <8 x i16> @_Z15bitfield_insertDv8_sS_jj(<8 x i16> noundef %b, <8 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i16> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <8 x i16> @_Z15bitfield_insertDv8_sS_jj(<8 x i16> noundef, <8 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort8]] %[[#insertinsert_ushort8]]
; OpenCL equivalent.
; kernel void testInsert_ushort8(ushort8 b, ushort8 i, global ushort8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort8(<8 x i16> noundef %b, <8 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <8 x i16> @_Z15bitfield_insertDv8_tS_jj(<8 x i16> noundef %b, <8 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i16> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <8 x i16> @_Z15bitfield_insertDv8_tS_jj(<8 x i16> noundef, <8 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_char8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_char8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char8]] %[[#insertinsert_char8]]
; OpenCL equivalent.
; kernel void testInsert_char8(char8 b, char8 i, global char8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_char8(<8 x i8> noundef %b, <8 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <8 x i8> @_Z15bitfield_insertDv8_cS_jj(<8 x i8> noundef %b, <8 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i8> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <8 x i8> @_Z15bitfield_insertDv8_cS_jj(<8 x i8> noundef, <8 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uchar8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uchar8:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar8]] %[[#insertinsert_uchar8]]
; OpenCL equivalent.
; kernel void testInsert_uchar8(uchar8 b, uchar8 i, global uchar8 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uchar8(<8 x i8> noundef %b, <8 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <8 x i8> @_Z15bitfield_insertDv8_hS_jj(<8 x i8> noundef %b, <8 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <8 x i8> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <8 x i8> @_Z15bitfield_insertDv8_hS_jj(<8 x i8> noundef, <8 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_long16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_long16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_long16]] %[[#insertinsert_long16]]
; OpenCL equivalent.
; kernel void testInsert_long16(long16 b, long16 i, global long16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_long16(<16 x i64> noundef %b, <16 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 128 initializes((0, 128)) %res) {
entry:
  %call = tail call spir_func <16 x i64> @_Z15bitfield_insertDv16_lS_jj(<16 x i64> noundef %b, <16 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i64> %call, ptr addrspace(1) %res, align 128, !tbaa !22
  ret void
}

declare spir_func <16 x i64> @_Z15bitfield_insertDv16_lS_jj(<16 x i64> noundef, <16 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ulong16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ulong16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ulong16]] %[[#insertinsert_ulong16]]
; OpenCL equivalent.
; kernel void testInsert_ulong16(ulong16 b, ulong16 i, global ulong16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ulong16(<16 x i64> noundef %b, <16 x i64> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 128 initializes((0, 128)) %res) {
entry:
  %call = tail call spir_func <16 x i64> @_Z15bitfield_insertDv16_mS_jj(<16 x i64> noundef %b, <16 x i64> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i64> %call, ptr addrspace(1) %res, align 128, !tbaa !22
  ret void
}

declare spir_func <16 x i64> @_Z15bitfield_insertDv16_mS_jj(<16 x i64> noundef, <16 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_int16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_int16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_int16]] %[[#insertinsert_int16]]
; OpenCL equivalent.
; kernel void testInsert_int16(int16 b, int16 i, global int16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_int16(<16 x i32> noundef %b, <16 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <16 x i32> @_Z15bitfield_insertDv16_iS_jj(<16 x i32> noundef %b, <16 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i32> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <16 x i32> @_Z15bitfield_insertDv16_iS_jj(<16 x i32> noundef, <16 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uint16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uint16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uint16]] %[[#insertinsert_uint16]]
; OpenCL equivalent.
; kernel void testInsert_uint16(uint16 b, uint16 i, global uint16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uint16(<16 x i32> noundef %b, <16 x i32> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <16 x i32> @_Z15bitfield_insertDv16_jS_jj(<16 x i32> noundef %b, <16 x i32> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i32> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <16 x i32> @_Z15bitfield_insertDv16_jS_jj(<16 x i32> noundef, <16 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_short16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_short16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_short16]] %[[#insertinsert_short16]]
; OpenCL equivalent.
; kernel void testInsert_short16(short16 b, short16 i, global short16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_short16(<16 x i16> noundef %b, <16 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <16 x i16> @_Z15bitfield_insertDv16_sS_jj(<16 x i16> noundef %b, <16 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i16> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <16 x i16> @_Z15bitfield_insertDv16_sS_jj(<16 x i16> noundef, <16 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_ushort16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_ushort16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_ushort16]] %[[#insertinsert_ushort16]]
; OpenCL equivalent.
; kernel void testInsert_ushort16(ushort16 b, ushort16 i, global ushort16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_ushort16(<16 x i16> noundef %b, <16 x i16> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <16 x i16> @_Z15bitfield_insertDv16_tS_jj(<16 x i16> noundef %b, <16 x i16> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i16> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <16 x i16> @_Z15bitfield_insertDv16_tS_jj(<16 x i16> noundef, <16 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_char16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_char16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_char16]] %[[#insertinsert_char16]]
; OpenCL equivalent.
; kernel void testInsert_char16(char16 b, char16 i, global char16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_char16(<16 x i8> noundef %b, <16 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <16 x i8> @_Z15bitfield_insertDv16_cS_jj(<16 x i8> noundef %b, <16 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i8> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <16 x i8> @_Z15bitfield_insertDv16_cS_jj(<16 x i8> noundef, <16 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#insertbase_uchar16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#insertinsert_uchar16:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldInsert %[[#]] %[[#insertbase_uchar16]] %[[#insertinsert_uchar16]]
; OpenCL equivalent.
; kernel void testInsert_uchar16(uchar16 b, uchar16 i, global uchar16 *res) {
;   *res = bitfield_insert(b, i, 4, 2);
; }

define dso_local spir_kernel void @testInsert_uchar16(<16 x i8> noundef %b, <16 x i8> noundef %i, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <16 x i8> @_Z15bitfield_insertDv16_hS_jj(<16 x i8> noundef %b, <16 x i8> noundef %i, i32 noundef 4, i32 noundef 2) #2
  store <16 x i8> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <16 x i8> @_Z15bitfield_insertDv16_hS_jj(<16 x i8> noundef, <16 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef, i32 noundef, i32 noundef) 

declare spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef, i32 noundef, i32 noundef) 

declare spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext, i32 noundef, i32 noundef) 

declare spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

declare spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext, i32 noundef, i32 noundef) 

declare spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long2(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long2(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int2(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int2(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short2(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short2(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char2(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char2(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long3(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long3(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int3(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int3(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short3(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short3(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char3(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char3(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long4(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long4(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int4(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int4(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short4(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short4(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char4(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char4(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long8(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long8(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int8(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int8(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short8(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short8(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char8(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char8(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_long16(long b, ulong bu, global long *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_long16(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z23bitfield_extract_signedljj(i64 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z23bitfield_extract_signedmjj(i64 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_int16(int b, uint bu, global int *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_int16(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z23bitfield_extract_signedijj(i32 noundef %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z23bitfield_extract_signedjjj(i32 noundef %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add nsw i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_short16(short b, ushort bu, global short *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_short16(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z23bitfield_extract_signedsjj(i16 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i16 @_Z23bitfield_extract_signedtjj(i16 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#sextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#sextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldSExtract %[[#]] %[[#sextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractS_char16(char b, uchar bu, global char *res) {
;   *res = bitfield_extract_signed(b, 5, 4);
;   *res += bitfield_extract_signed(bu, 5, 4);
; }

define dso_local spir_kernel void @testExtractS_char16(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z23bitfield_extract_signedcjj(i8 noundef signext %b, i32 noundef 5, i32 noundef 4) #2
  %call1 = tail call spir_func signext i8 @_Z23bitfield_extract_signedhjj(i8 noundef zeroext %bu, i32 noundef 5, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long(long b, ulong bu, global ulong *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long(i64 noundef %b, i64 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z25bitfield_extract_unsignedljj(i64 noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func i64 @_Z25bitfield_extract_unsignedmjj(i64 noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add i64 %call1, %call
  store i64 %add, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z25bitfield_extract_unsignedljj(i64 noundef, i32 noundef, i32 noundef) 

declare spir_func i64 @_Z25bitfield_extract_unsignedmjj(i64 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int(int b, uint bu, global uint *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int(i32 noundef %b, i32 noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z25bitfield_extract_unsignedijj(i32 noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func i32 @_Z25bitfield_extract_unsignedjjj(i32 noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add i32 %call1, %call
  store i32 %add, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z25bitfield_extract_unsignedijj(i32 noundef, i32 noundef, i32 noundef) 

declare spir_func i32 @_Z25bitfield_extract_unsignedjjj(i32 noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short(short b, ushort bu, global ushort *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short(i16 noundef signext %b, i16 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func zeroext i16 @_Z25bitfield_extract_unsignedsjj(i16 noundef signext %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func zeroext i16 @_Z25bitfield_extract_unsignedtjj(i16 noundef zeroext %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add i16 %call1, %call
  store i16 %add, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func zeroext i16 @_Z25bitfield_extract_unsignedsjj(i16 noundef signext, i32 noundef, i32 noundef) 

declare spir_func zeroext i16 @_Z25bitfield_extract_unsignedtjj(i16 noundef zeroext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char(char b, uchar bu, global uchar *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char(i8 noundef signext %b, i8 noundef zeroext %bu, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func zeroext i8 @_Z25bitfield_extract_unsignedcjj(i8 noundef signext %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func zeroext i8 @_Z25bitfield_extract_unsignedhjj(i8 noundef zeroext %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add i8 %call1, %call
  store i8 %add, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

declare spir_func zeroext i8 @_Z25bitfield_extract_unsignedcjj(i8 noundef signext, i32 noundef, i32 noundef) 

declare spir_func zeroext i8 @_Z25bitfield_extract_unsignedhjj(i8 noundef zeroext, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long2(long2 b, ulong2 bu, global ulong2 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long2(<2 x i64> noundef %b, <2 x i64> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <2 x i64> @_Z25bitfield_extract_unsignedDv2_ljj(<2 x i64> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <2 x i64> @_Z25bitfield_extract_unsignedDv2_mjj(<2 x i64> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <2 x i64> %call1, %call
  store <2 x i64> %add, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <2 x i64> @_Z25bitfield_extract_unsignedDv2_ljj(<2 x i64> noundef, i32 noundef, i32 noundef) 

declare spir_func <2 x i64> @_Z25bitfield_extract_unsignedDv2_mjj(<2 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int2(int2 b, uint2 bu, global uint2 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int2(<2 x i32> noundef %b, <2 x i32> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <2 x i32> @_Z25bitfield_extract_unsignedDv2_ijj(<2 x i32> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <2 x i32> @_Z25bitfield_extract_unsignedDv2_jjj(<2 x i32> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <2 x i32> %call1, %call
  store <2 x i32> %add, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <2 x i32> @_Z25bitfield_extract_unsignedDv2_ijj(<2 x i32> noundef, i32 noundef, i32 noundef) 

declare spir_func <2 x i32> @_Z25bitfield_extract_unsignedDv2_jjj(<2 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short2(short2 b, ushort2 bu, global ushort2 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short2(<2 x i16> noundef %b, <2 x i16> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <2 x i16> @_Z25bitfield_extract_unsignedDv2_sjj(<2 x i16> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <2 x i16> @_Z25bitfield_extract_unsignedDv2_tjj(<2 x i16> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <2 x i16> %call1, %call
  store <2 x i16> %add, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <2 x i16> @_Z25bitfield_extract_unsignedDv2_sjj(<2 x i16> noundef, i32 noundef, i32 noundef) 

declare spir_func <2 x i16> @_Z25bitfield_extract_unsignedDv2_tjj(<2 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char2(char2 b, uchar2 bu, global uchar2 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char2(<2 x i8> noundef %b, <2 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func <2 x i8> @_Z25bitfield_extract_unsignedDv2_cjj(<2 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <2 x i8> @_Z25bitfield_extract_unsignedDv2_hjj(<2 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <2 x i8> %call1, %call
  store <2 x i8> %add, ptr addrspace(1) %res, align 2, !tbaa !22
  ret void
}

declare spir_func <2 x i8> @_Z25bitfield_extract_unsignedDv2_cjj(<2 x i8> noundef, i32 noundef, i32 noundef) 

declare spir_func <2 x i8> @_Z25bitfield_extract_unsignedDv2_hjj(<2 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long3(long3 b, ulong3 bu, global ulong3 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long3(<3 x i64> noundef %b, <3 x i64> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <3 x i64> @_Z25bitfield_extract_unsignedDv3_ljj(<3 x i64> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call6 = tail call spir_func <3 x i64> @_Z25bitfield_extract_unsignedDv3_mjj(<3 x i64> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <3 x i64> %call6, %call
  %extractVec9 = shufflevector <3 x i64> %add, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i64> %extractVec9, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <3 x i64> @_Z25bitfield_extract_unsignedDv3_ljj(<3 x i64> noundef, i32 noundef, i32 noundef) 

declare spir_func <3 x i64> @_Z25bitfield_extract_unsignedDv3_mjj(<3 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int3(int3 b, uint3 bu, global uint3 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int3(<3 x i32> noundef %b, <3 x i32> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <3 x i32> @_Z25bitfield_extract_unsignedDv3_ijj(<3 x i32> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call6 = tail call spir_func <3 x i32> @_Z25bitfield_extract_unsignedDv3_jjj(<3 x i32> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <3 x i32> %call6, %call
  %extractVec9 = shufflevector <3 x i32> %add, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i32> %extractVec9, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <3 x i32> @_Z25bitfield_extract_unsignedDv3_ijj(<3 x i32> noundef, i32 noundef, i32 noundef) 

declare spir_func <3 x i32> @_Z25bitfield_extract_unsignedDv3_jjj(<3 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short3(short3 b, ushort3 bu, global ushort3 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short3(<3 x i16> noundef %b, <3 x i16> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <3 x i16> @_Z25bitfield_extract_unsignedDv3_sjj(<3 x i16> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call6 = tail call spir_func <3 x i16> @_Z25bitfield_extract_unsignedDv3_tjj(<3 x i16> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <3 x i16> %call6, %call
  %extractVec9 = shufflevector <3 x i16> %add, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i16> %extractVec9, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <3 x i16> @_Z25bitfield_extract_unsignedDv3_sjj(<3 x i16> noundef, i32 noundef, i32 noundef) 

declare spir_func <3 x i16> @_Z25bitfield_extract_unsignedDv3_tjj(<3 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char3(char3 b, uchar3 bu, global uchar3 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char3(<3 x i8> noundef %b, <3 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <3 x i8> @_Z25bitfield_extract_unsignedDv3_cjj(<3 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call6 = tail call spir_func <3 x i8> @_Z25bitfield_extract_unsignedDv3_hjj(<3 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <3 x i8> %call6, %call
  %extractVec9 = shufflevector <3 x i8> %add, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i8> %extractVec9, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <3 x i8> @_Z25bitfield_extract_unsignedDv3_cjj(<3 x i8> noundef, i32 noundef, i32 noundef) 

declare spir_func <3 x i8> @_Z25bitfield_extract_unsignedDv3_hjj(<3 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long4(long4 b, ulong4 bu, global ulong4 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long4(<4 x i64> noundef %b, <4 x i64> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <4 x i64> @_Z25bitfield_extract_unsignedDv4_ljj(<4 x i64> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <4 x i64> @_Z25bitfield_extract_unsignedDv4_mjj(<4 x i64> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <4 x i64> %call1, %call
  store <4 x i64> %add, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <4 x i64> @_Z25bitfield_extract_unsignedDv4_ljj(<4 x i64> noundef, i32 noundef, i32 noundef) 

declare spir_func <4 x i64> @_Z25bitfield_extract_unsignedDv4_mjj(<4 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int4(int4 b, uint4 bu, global uint4 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int4(<4 x i32> noundef %b, <4 x i32> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <4 x i32> @_Z25bitfield_extract_unsignedDv4_ijj(<4 x i32> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <4 x i32> @_Z25bitfield_extract_unsignedDv4_jjj(<4 x i32> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <4 x i32> %call1, %call
  store <4 x i32> %add, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <4 x i32> @_Z25bitfield_extract_unsignedDv4_ijj(<4 x i32> noundef, i32 noundef, i32 noundef) 

declare spir_func <4 x i32> @_Z25bitfield_extract_unsignedDv4_jjj(<4 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short4(short4 b, ushort4 bu, global ushort4 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short4(<4 x i16> noundef %b, <4 x i16> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <4 x i16> @_Z25bitfield_extract_unsignedDv4_sjj(<4 x i16> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <4 x i16> @_Z25bitfield_extract_unsignedDv4_tjj(<4 x i16> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <4 x i16> %call1, %call
  store <4 x i16> %add, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <4 x i16> @_Z25bitfield_extract_unsignedDv4_sjj(<4 x i16> noundef, i32 noundef, i32 noundef) 

declare spir_func <4 x i16> @_Z25bitfield_extract_unsignedDv4_tjj(<4 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char4(char4 b, uchar4 bu, global uchar4 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char4(<4 x i8> noundef %b, <4 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <4 x i8> @_Z25bitfield_extract_unsignedDv4_cjj(<4 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <4 x i8> @_Z25bitfield_extract_unsignedDv4_hjj(<4 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <4 x i8> %call1, %call
  store <4 x i8> %add, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <4 x i8> @_Z25bitfield_extract_unsignedDv4_cjj(<4 x i8> noundef, i32 noundef, i32 noundef) 

declare spir_func <4 x i8> @_Z25bitfield_extract_unsignedDv4_hjj(<4 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long8(long8 b, ulong8 bu, global ulong8 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long8(<8 x i64> noundef %b, <8 x i64> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <8 x i64> @_Z25bitfield_extract_unsignedDv8_ljj(<8 x i64> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <8 x i64> @_Z25bitfield_extract_unsignedDv8_mjj(<8 x i64> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <8 x i64> %call1, %call
  store <8 x i64> %add, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <8 x i64> @_Z25bitfield_extract_unsignedDv8_ljj(<8 x i64> noundef, i32 noundef, i32 noundef) 

declare spir_func <8 x i64> @_Z25bitfield_extract_unsignedDv8_mjj(<8 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int8(int8 b, uint8 bu, global uint8 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int8(<8 x i32> noundef %b, <8 x i32> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <8 x i32> @_Z25bitfield_extract_unsignedDv8_ijj(<8 x i32> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <8 x i32> @_Z25bitfield_extract_unsignedDv8_jjj(<8 x i32> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <8 x i32> %call1, %call
  store <8 x i32> %add, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <8 x i32> @_Z25bitfield_extract_unsignedDv8_ijj(<8 x i32> noundef, i32 noundef, i32 noundef) 

declare spir_func <8 x i32> @_Z25bitfield_extract_unsignedDv8_jjj(<8 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short8(short8 b, ushort8 bu, global ushort8 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short8(<8 x i16> noundef %b, <8 x i16> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <8 x i16> @_Z25bitfield_extract_unsignedDv8_sjj(<8 x i16> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <8 x i16> @_Z25bitfield_extract_unsignedDv8_tjj(<8 x i16> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <8 x i16> %call1, %call
  store <8 x i16> %add, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <8 x i16> @_Z25bitfield_extract_unsignedDv8_sjj(<8 x i16> noundef, i32 noundef, i32 noundef) 

declare spir_func <8 x i16> @_Z25bitfield_extract_unsignedDv8_tjj(<8 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char8(char8 b, uchar8 bu, global uchar8 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char8(<8 x i8> noundef %b, <8 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj(<8 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_hjj(<8 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <8 x i8> %call1, %call
  store <8 x i8> %add, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj(<8 x i8> noundef, i32 noundef, i32 noundef) 

declare spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_hjj(<8 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_long16(long16 b, ulong16 bu, global ulong16 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_long16(<16 x i64> noundef %b, <16 x i64> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 128 initializes((0, 128)) %res) {
entry:
  %call = tail call spir_func <16 x i64> @_Z25bitfield_extract_unsignedDv16_ljj(<16 x i64> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <16 x i64> @_Z25bitfield_extract_unsignedDv16_mjj(<16 x i64> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <16 x i64> %call1, %call
  store <16 x i64> %add, ptr addrspace(1) %res, align 128, !tbaa !22
  ret void
}

declare spir_func <16 x i64> @_Z25bitfield_extract_unsignedDv16_ljj(<16 x i64> noundef, i32 noundef, i32 noundef) 

declare spir_func <16 x i64> @_Z25bitfield_extract_unsignedDv16_mjj(<16 x i64> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_int16(int16 b, uint16 bu, global uint16 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_int16(<16 x i32> noundef %b, <16 x i32> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <16 x i32> @_Z25bitfield_extract_unsignedDv16_ijj(<16 x i32> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <16 x i32> @_Z25bitfield_extract_unsignedDv16_jjj(<16 x i32> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <16 x i32> %call1, %call
  store <16 x i32> %add, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <16 x i32> @_Z25bitfield_extract_unsignedDv16_ijj(<16 x i32> noundef, i32 noundef, i32 noundef) 

declare spir_func <16 x i32> @_Z25bitfield_extract_unsignedDv16_jjj(<16 x i32> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_short16(short16 b, ushort16 bu, global ushort16 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_short16(<16 x i16> noundef %b, <16 x i16> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <16 x i16> @_Z25bitfield_extract_unsignedDv16_sjj(<16 x i16> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <16 x i16> @_Z25bitfield_extract_unsignedDv16_tjj(<16 x i16> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <16 x i16> %call1, %call
  store <16 x i16> %add, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <16 x i16> @_Z25bitfield_extract_unsignedDv16_sjj(<16 x i16> noundef, i32 noundef, i32 noundef) 

declare spir_func <16 x i16> @_Z25bitfield_extract_unsignedDv16_tjj(<16 x i16> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#uextractbase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#uextractbaseu:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbase]]
; CHECK-EXTENSION: %[[#]] = OpBitFieldUExtract %[[#]] %[[#uextractbaseu]]
; OpenCL equivalent.
; kernel void testExtractU_char16(char16 b, uchar16 bu, global uchar16 *res) {
;   *res = bitfield_extract_unsigned(b, 3, 4);
;   *res += bitfield_extract_unsigned(bu, 3, 4);
; }

define dso_local spir_kernel void @testExtractU_char16(<16 x i8> noundef %b, <16 x i8> noundef %bu, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <16 x i8> @_Z25bitfield_extract_unsignedDv16_cjj(<16 x i8> noundef %b, i32 noundef 3, i32 noundef 4) #2
  %call1 = tail call spir_func <16 x i8> @_Z25bitfield_extract_unsignedDv16_hjj(<16 x i8> noundef %bu, i32 noundef 3, i32 noundef 4) #2
  %add = add <16 x i8> %call1, %call
  store <16 x i8> %add, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <16 x i8> @_Z25bitfield_extract_unsignedDv16_cjj(<16 x i8> noundef, i32 noundef, i32 noundef) 

declare spir_func <16 x i8> @_Z25bitfield_extract_unsignedDv16_hjj(<16 x i8> noundef, i32 noundef, i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long(long b, global long *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long(i64 noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z11bit_reversel(i64 noundef %b) #2
  store i64 %call, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z11bit_reversel(i64 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong(ulong b, global ulong *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong(i64 noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func i64 @_Z11bit_reversem(i64 noundef %b) #2
  store i64 %call, ptr addrspace(1) %res, align 8, !tbaa !7
  ret void
}

declare spir_func i64 @_Z11bit_reversem(i64 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int(int b, global int *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int(i32 noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z11bit_reversei(i32 noundef %b) #2
  store i32 %call, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z11bit_reversei(i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint(uint b, global uint *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint(i32 noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func i32 @_Z11bit_reversej(i32 noundef %b) #2
  store i32 %call, ptr addrspace(1) %res, align 4, !tbaa !13
  ret void
}

declare spir_func i32 @_Z11bit_reversej(i32 noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short(short b, global short *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short(i16 noundef signext %b, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func signext i16 @_Z11bit_reverses(i16 noundef signext %b) #2
  store i16 %call, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func signext i16 @_Z11bit_reverses(i16 noundef signext) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort(ushort b, global ushort *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort(i16 noundef zeroext %b, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func zeroext i16 @_Z11bit_reverset(i16 noundef zeroext %b) #2
  store i16 %call, ptr addrspace(1) %res, align 2, !tbaa !17
  ret void
}

declare spir_func zeroext i16 @_Z11bit_reverset(i16 noundef zeroext) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char(char b, global char *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char(i8 noundef signext %b, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func signext i8 @_Z11bit_reversec(i8 noundef signext %b) #2
  store i8 %call, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

declare spir_func signext i8 @_Z11bit_reversec(i8 noundef signext) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar(uchar b, global uchar *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar(i8 noundef zeroext %b, ptr addrspace(1) nocapture noundef writeonly align 1 initializes((0, 1)) %res) {
entry:
  %call = tail call spir_func zeroext i8 @_Z11bit_reverseh(i8 noundef zeroext %b) #2
  store i8 %call, ptr addrspace(1) %res, align 1, !tbaa !22
  ret void
}

declare spir_func zeroext i8 @_Z11bit_reverseh(i8 noundef zeroext) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long2(long2 b, global long2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long2(<2 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <2 x i64> @_Z11bit_reverseDv2_l(<2 x i64> noundef %b) #2
  store <2 x i64> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <2 x i64> @_Z11bit_reverseDv2_l(<2 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong2(ulong2 b, global ulong2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong2(<2 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <2 x i64> @_Z11bit_reverseDv2_m(<2 x i64> noundef %b) #2
  store <2 x i64> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <2 x i64> @_Z11bit_reverseDv2_m(<2 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int2(int2 b, global int2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int2(<2 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <2 x i32> @_Z11bit_reverseDv2_i(<2 x i32> noundef %b) #2
  store <2 x i32> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <2 x i32> @_Z11bit_reverseDv2_i(<2 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint2(uint2 b, global uint2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint2(<2 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <2 x i32> @_Z11bit_reverseDv2_j(<2 x i32> noundef %b) #2
  store <2 x i32> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <2 x i32> @_Z11bit_reverseDv2_j(<2 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short2(short2 b, global short2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short2(<2 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <2 x i16> @_Z11bit_reverseDv2_s(<2 x i16> noundef %b) #2
  store <2 x i16> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <2 x i16> @_Z11bit_reverseDv2_s(<2 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort2(ushort2 b, global ushort2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort2(<2 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <2 x i16> @_Z11bit_reverseDv2_t(<2 x i16> noundef %b) #2
  store <2 x i16> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <2 x i16> @_Z11bit_reverseDv2_t(<2 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char2(char2 b, global char2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char2(<2 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func <2 x i8> @_Z11bit_reverseDv2_c(<2 x i8> noundef %b) #2
  store <2 x i8> %call, ptr addrspace(1) %res, align 2, !tbaa !22
  ret void
}

declare spir_func <2 x i8> @_Z11bit_reverseDv2_c(<2 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar2(uchar2 b, global uchar2 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar2(<2 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 2 initializes((0, 2)) %res) {
entry:
  %call = tail call spir_func <2 x i8> @_Z11bit_reverseDv2_h(<2 x i8> noundef %b) #2
  store <2 x i8> %call, ptr addrspace(1) %res, align 2, !tbaa !22
  ret void
}

declare spir_func <2 x i8> @_Z11bit_reverseDv2_h(<2 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long3(long3 b, global long3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long3(<3 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <3 x i64> @_Z11bit_reverseDv3_l(<3 x i64> noundef %b) #2
  %extractVec2 = shufflevector <3 x i64> %call, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i64> %extractVec2, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <3 x i64> @_Z11bit_reverseDv3_l(<3 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong3(ulong3 b, global ulong3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong3(<3 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <3 x i64> @_Z11bit_reverseDv3_m(<3 x i64> noundef %b) #2
  %extractVec2 = shufflevector <3 x i64> %call, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i64> %extractVec2, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <3 x i64> @_Z11bit_reverseDv3_m(<3 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int3(int3 b, global int3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int3(<3 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <3 x i32> @_Z11bit_reverseDv3_i(<3 x i32> noundef %b) #2
  %extractVec2 = shufflevector <3 x i32> %call, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i32> %extractVec2, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <3 x i32> @_Z11bit_reverseDv3_i(<3 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint3(uint3 b, global uint3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint3(<3 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <3 x i32> @_Z11bit_reverseDv3_j(<3 x i32> noundef %b) #2
  %extractVec2 = shufflevector <3 x i32> %call, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i32> %extractVec2, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <3 x i32> @_Z11bit_reverseDv3_j(<3 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short3(short3 b, global short3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short3(<3 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <3 x i16> @_Z11bit_reverseDv3_s(<3 x i16> noundef %b) #2
  %extractVec2 = shufflevector <3 x i16> %call, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i16> %extractVec2, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <3 x i16> @_Z11bit_reverseDv3_s(<3 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort3(ushort3 b, global ushort3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort3(<3 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <3 x i16> @_Z11bit_reverseDv3_t(<3 x i16> noundef %b) #2
  %extractVec2 = shufflevector <3 x i16> %call, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i16> %extractVec2, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <3 x i16> @_Z11bit_reverseDv3_t(<3 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char3(char3 b, global char3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char3(<3 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <3 x i8> @_Z11bit_reverseDv3_c(<3 x i8> noundef %b) #2
  %extractVec2 = shufflevector <3 x i8> %call, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i8> %extractVec2, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <3 x i8> @_Z11bit_reverseDv3_c(<3 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar3(uchar3 b, global uchar3 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar3(<3 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <3 x i8> @_Z11bit_reverseDv3_h(<3 x i8> noundef %b) #2
  %extractVec2 = shufflevector <3 x i8> %call, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  store <4 x i8> %extractVec2, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <3 x i8> @_Z11bit_reverseDv3_h(<3 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long4(long4 b, global long4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long4(<4 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <4 x i64> @_Z11bit_reverseDv4_l(<4 x i64> noundef %b) #2
  store <4 x i64> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <4 x i64> @_Z11bit_reverseDv4_l(<4 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong4(ulong4 b, global ulong4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong4(<4 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <4 x i64> @_Z11bit_reverseDv4_m(<4 x i64> noundef %b) #2
  store <4 x i64> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <4 x i64> @_Z11bit_reverseDv4_m(<4 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int4(int4 b, global int4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int4(<4 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <4 x i32> @_Z11bit_reverseDv4_i(<4 x i32> noundef %b) #2
  store <4 x i32> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <4 x i32> @_Z11bit_reverseDv4_i(<4 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint4(uint4 b, global uint4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint4(<4 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <4 x i32> @_Z11bit_reverseDv4_j(<4 x i32> noundef %b) #2
  store <4 x i32> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <4 x i32> @_Z11bit_reverseDv4_j(<4 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short4(short4 b, global short4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short4(<4 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <4 x i16> @_Z11bit_reverseDv4_s(<4 x i16> noundef %b) #2
  store <4 x i16> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <4 x i16> @_Z11bit_reverseDv4_s(<4 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort4(ushort4 b, global ushort4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort4(<4 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <4 x i16> @_Z11bit_reverseDv4_t(<4 x i16> noundef %b) #2
  store <4 x i16> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <4 x i16> @_Z11bit_reverseDv4_t(<4 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char4(char4 b, global char4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char4(<4 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <4 x i8> @_Z11bit_reverseDv4_c(<4 x i8> noundef %b) #2
  store <4 x i8> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <4 x i8> @_Z11bit_reverseDv4_c(<4 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar4(uchar4 b, global uchar4 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar4(<4 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 4 initializes((0, 4)) %res) {
entry:
  %call = tail call spir_func <4 x i8> @_Z11bit_reverseDv4_h(<4 x i8> noundef %b) #2
  store <4 x i8> %call, ptr addrspace(1) %res, align 4, !tbaa !22
  ret void
}

declare spir_func <4 x i8> @_Z11bit_reverseDv4_h(<4 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long8(long8 b, global long8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long8(<8 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <8 x i64> @_Z11bit_reverseDv8_l(<8 x i64> noundef %b) #2
  store <8 x i64> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <8 x i64> @_Z11bit_reverseDv8_l(<8 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong8(ulong8 b, global ulong8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong8(<8 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <8 x i64> @_Z11bit_reverseDv8_m(<8 x i64> noundef %b) #2
  store <8 x i64> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <8 x i64> @_Z11bit_reverseDv8_m(<8 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int8(int8 b, global int8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int8(<8 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <8 x i32> @_Z11bit_reverseDv8_i(<8 x i32> noundef %b) #2
  store <8 x i32> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <8 x i32> @_Z11bit_reverseDv8_i(<8 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint8(uint8 b, global uint8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint8(<8 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <8 x i32> @_Z11bit_reverseDv8_j(<8 x i32> noundef %b) #2
  store <8 x i32> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <8 x i32> @_Z11bit_reverseDv8_j(<8 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short8(short8 b, global short8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short8(<8 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <8 x i16> @_Z11bit_reverseDv8_s(<8 x i16> noundef %b) #2
  store <8 x i16> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <8 x i16> @_Z11bit_reverseDv8_s(<8 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort8(ushort8 b, global ushort8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort8(<8 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <8 x i16> @_Z11bit_reverseDv8_t(<8 x i16> noundef %b) #2
  store <8 x i16> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <8 x i16> @_Z11bit_reverseDv8_t(<8 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char8(char8 b, global char8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char8(<8 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <8 x i8> @_Z11bit_reverseDv8_c(<8 x i8> noundef %b) #2
  store <8 x i8> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <8 x i8> @_Z11bit_reverseDv8_c(<8 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar8(uchar8 b, global uchar8 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar8(<8 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 8 initializes((0, 8)) %res) {
entry:
  %call = tail call spir_func <8 x i8> @_Z11bit_reverseDv8_h(<8 x i8> noundef %b) #2
  store <8 x i8> %call, ptr addrspace(1) %res, align 8, !tbaa !22
  ret void
}

declare spir_func <8 x i8> @_Z11bit_reverseDv8_h(<8 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_long16(long16 b, global long16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_long16(<16 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 128 initializes((0, 128)) %res) {
entry:
  %call = tail call spir_func <16 x i64> @_Z11bit_reverseDv16_l(<16 x i64> noundef %b) #2
  store <16 x i64> %call, ptr addrspace(1) %res, align 128, !tbaa !22
  ret void
}

declare spir_func <16 x i64> @_Z11bit_reverseDv16_l(<16 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ulong16(ulong16 b, global ulong16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ulong16(<16 x i64> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 128 initializes((0, 128)) %res) {
entry:
  %call = tail call spir_func <16 x i64> @_Z11bit_reverseDv16_m(<16 x i64> noundef %b) #2
  store <16 x i64> %call, ptr addrspace(1) %res, align 128, !tbaa !22
  ret void
}

declare spir_func <16 x i64> @_Z11bit_reverseDv16_m(<16 x i64> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_int16(int16 b, global int16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_int16(<16 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <16 x i32> @_Z11bit_reverseDv16_i(<16 x i32> noundef %b) #2
  store <16 x i32> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <16 x i32> @_Z11bit_reverseDv16_i(<16 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uint16(uint16 b, global uint16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uint16(<16 x i32> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 64 initializes((0, 64)) %res) {
entry:
  %call = tail call spir_func <16 x i32> @_Z11bit_reverseDv16_j(<16 x i32> noundef %b) #2
  store <16 x i32> %call, ptr addrspace(1) %res, align 64, !tbaa !22
  ret void
}

declare spir_func <16 x i32> @_Z11bit_reverseDv16_j(<16 x i32> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_short16(short16 b, global short16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_short16(<16 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <16 x i16> @_Z11bit_reverseDv16_s(<16 x i16> noundef %b) #2
  store <16 x i16> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <16 x i16> @_Z11bit_reverseDv16_s(<16 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_ushort16(ushort16 b, global ushort16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_ushort16(<16 x i16> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 32 initializes((0, 32)) %res) {
entry:
  %call = tail call spir_func <16 x i16> @_Z11bit_reverseDv16_t(<16 x i16> noundef %b) #2
  store <16 x i16> %call, ptr addrspace(1) %res, align 32, !tbaa !22
  ret void
}

declare spir_func <16 x i16> @_Z11bit_reverseDv16_t(<16 x i16> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_char16(char16 b, global char16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_char16(<16 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <16 x i8> @_Z11bit_reverseDv16_c(<16 x i8> noundef %b) #2
  store <16 x i8> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <16 x i8> @_Z11bit_reverseDv16_c(<16 x i8> noundef) 

; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_uchar16(uchar16 b, global uchar16 *res) {
;   *res = bit_reverse(b);
; }

define dso_local spir_kernel void @testBitReverse_uchar16(<16 x i8> noundef %b, ptr addrspace(1) nocapture noundef writeonly align 16 initializes((0, 16)) %res) {
entry:
  %call = tail call spir_func <16 x i8> @_Z11bit_reverseDv16_h(<16 x i8> noundef %b) #2
  store <16 x i8> %call, ptr addrspace(1) %res, align 16, !tbaa !22
  ret void
}

declare spir_func <16 x i8> @_Z11bit_reverseDv16_h(<16 x i8> noundef) 

attributes #2 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git cc61409d353a40f62d3a137f3c7436aa00df779d)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!13 = !{!14, !14, i64 0}
!14 = !{!"int", !9, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"short", !9, i64 0}
!22 = !{!9, !9, i64 0}
