;; #pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;;
;; kernel void testNonUniformBroadcastChars()
;; {
;;     char16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastUChars()
;; {
;;     uchar16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastShorts()
;; {
;;     short16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastUShorts()
;; {
;;     ushort16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastInts()
;; {
;;     int16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastUInts()
;; {
;;     uint16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastLongs()
;; {
;;     long16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastULongs()
;; {
;;     ulong16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastFloats()
;; {
;;     float16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastHalfs()
;; {
;;     half16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testNonUniformBroadcastDoubles()
;; {
;;     double16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;;
;; kernel void testBallotOperations(global uint* dst)
;; {
;;     uint4 v = sub_group_ballot(0);
;;     dst[0] = sub_group_inverse_ballot(v);
;;     dst[1] = sub_group_ballot_bit_extract(v, 0);
;;     dst[2] = sub_group_ballot_bit_count(v);
;;     dst[3] = sub_group_ballot_inclusive_scan(v);
;;     dst[4] = sub_group_ballot_exclusive_scan(v);
;;     dst[5] = sub_group_ballot_find_lsb(v);
;;     dst[6] = sub_group_ballot_find_msb(v);
;; }
;;
;; kernel void testSubgroupMasks(global uint4* dst)
;; {
;;     dst[0] = get_sub_group_eq_mask();
;;     dst[1] = get_sub_group_ge_mask();
;;     dst[2] = get_sub_group_gt_mask();
;;     dst[3] = get_sub_group_le_mask();
;;     dst[4] = get_sub_group_lt_mask();
;; }

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability GroupNonUniformBallot

; CHECK-SPIRV-DAG: OpDecorate %[[#eqMask:]] BuiltIn SubgroupEqMask
; CHECK-SPIRV-DAG: OpDecorate %[[#geMask:]] BuiltIn SubgroupGeMask
; CHECK-SPIRV-DAG: OpDecorate %[[#gtMask:]] BuiltIn SubgroupGtMask
; CHECK-SPIRV-DAG: OpDecorate %[[#leMask:]] BuiltIn SubgroupLeMask
; CHECK-SPIRV-DAG: OpDecorate %[[#ltMask:]] BuiltIn SubgroupLtMask

; CHECK-SPIRV-DAG: %[[#bool:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#short:]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#half:]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#double:]] = OpTypeFloat 64

; CHECK-SPIRV-DAG: %[[#char2:]] = OpTypeVector %[[#char]] 2
; CHECK-SPIRV-DAG: %[[#char3:]] = OpTypeVector %[[#char]] 3
; CHECK-SPIRV-DAG: %[[#char4:]] = OpTypeVector %[[#char]] 4
; CHECK-SPIRV-DAG: %[[#char8:]] = OpTypeVector %[[#char]] 8
; CHECK-SPIRV-DAG: %[[#char16:]] = OpTypeVector %[[#char]] 16

; CHECK-SPIRV-DAG: %[[#short2:]] = OpTypeVector %[[#short]] 2
; CHECK-SPIRV-DAG: %[[#short3:]] = OpTypeVector %[[#short]] 3
; CHECK-SPIRV-DAG: %[[#short4:]] = OpTypeVector %[[#short]] 4
; CHECK-SPIRV-DAG: %[[#short8:]] = OpTypeVector %[[#short]] 8
; CHECK-SPIRV-DAG: %[[#short16:]] = OpTypeVector %[[#short]] 16

; CHECK-SPIRV-DAG: %[[#int2:]] = OpTypeVector %[[#int]] 2
; CHECK-SPIRV-DAG: %[[#int3:]] = OpTypeVector %[[#int]] 3
; CHECK-SPIRV-DAG: %[[#int4:]] = OpTypeVector %[[#int]] 4
; CHECK-SPIRV-DAG: %[[#int8:]] = OpTypeVector %[[#int]] 8
; CHECK-SPIRV-DAG: %[[#int16:]] = OpTypeVector %[[#int]] 16

; CHECK-SPIRV-DAG: %[[#long2:]] = OpTypeVector %[[#long]] 2
; CHECK-SPIRV-DAG: %[[#long3:]] = OpTypeVector %[[#long]] 3
; CHECK-SPIRV-DAG: %[[#long4:]] = OpTypeVector %[[#long]] 4
; CHECK-SPIRV-DAG: %[[#long8:]] = OpTypeVector %[[#long]] 8
; CHECK-SPIRV-DAG: %[[#long16:]] = OpTypeVector %[[#long]] 16

; CHECK-SPIRV-DAG: %[[#float2:]] = OpTypeVector %[[#float]] 2
; CHECK-SPIRV-DAG: %[[#float3:]] = OpTypeVector %[[#float]] 3
; CHECK-SPIRV-DAG: %[[#float4:]] = OpTypeVector %[[#float]] 4
; CHECK-SPIRV-DAG: %[[#float8:]] = OpTypeVector %[[#float]] 8
; CHECK-SPIRV-DAG: %[[#float16:]] = OpTypeVector %[[#float]] 16

; CHECK-SPIRV-DAG: %[[#half2:]] = OpTypeVector %[[#half]] 2
; CHECK-SPIRV-DAG: %[[#half3:]] = OpTypeVector %[[#half]] 3
; CHECK-SPIRV-DAG: %[[#half4:]] = OpTypeVector %[[#half]] 4
; CHECK-SPIRV-DAG: %[[#half8:]] = OpTypeVector %[[#half]] 8
; CHECK-SPIRV-DAG: %[[#half16:]] = OpTypeVector %[[#half]] 16

; CHECK-SPIRV-DAG: %[[#double2:]] = OpTypeVector %[[#double]] 2
; CHECK-SPIRV-DAG: %[[#double3:]] = OpTypeVector %[[#double]] 3
; CHECK-SPIRV-DAG: %[[#double4:]] = OpTypeVector %[[#double]] 4
; CHECK-SPIRV-DAG: %[[#double8:]] = OpTypeVector %[[#double]] 8
; CHECK-SPIRV-DAG: %[[#double16:]] = OpTypeVector %[[#double]] 16

; CHECK-SPIRV-DAG: %[[#false:]] = OpConstantFalse %[[#bool]]
; CHECK-SPIRV-DAG: %[[#ScopeSubgroup:]] = OpConstant %[[#int]] 3
; CHECK-SPIRV-DAG: %[[#char_0:]] = OpConstant %[[#char]] 0
; CHECK-SPIRV-DAG: %[[#short_0:]] = OpConstant %[[#short]] 0
; CHECK-SPIRV-DAG: %[[#int_0:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV-DAG: %[[#long_0:]] = OpConstantNull %[[#long]]
; CHECK-SPIRV-DAG: %[[#half_0:]] = OpConstant %[[#half]] 0
; CHECK-SPIRV-DAG: %[[#float_0:]] = OpConstant %[[#float]] 0
; CHECK-SPIRV-DAG: %[[#double_0:]] = OpConstant %[[#double]] 0

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char2_0:]] = OpVectorShuffle %[[#char2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char2]] %[[#ScopeSubgroup]] %[[#char2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char3_0:]] = OpVectorShuffle %[[#char3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char3]] %[[#ScopeSubgroup]] %[[#char3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char4_0:]] = OpVectorShuffle %[[#char4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char4]] %[[#ScopeSubgroup]] %[[#char4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char8_0:]] = OpVectorShuffle %[[#char8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char8]] %[[#ScopeSubgroup]] %[[#char8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#char16]]
; CHECK-SPIRV: %[[#char16_0:]] = OpVectorShuffle %[[#char16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char16]] %[[#ScopeSubgroup]] %[[#char16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char_value:]] = OpCompositeExtract %[[#char]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#char]] %[[#ScopeSubgroup]] %[[#char_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastChars() local_unnamed_addr {
  %1 = tail call spir_func signext i8 @_Z31sub_group_non_uniform_broadcastcj(i8 signext 0, i32 0)
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_cj(<2 x i8> %3, i32 0)
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_cj(<3 x i8> %7, i32 0)
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_cj(<4 x i8> %11, i32 0)
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_cj(<8 x i8> %15, i32 0)
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_cj(<16 x i8> %18, i32 0)
  %20 = extractelement <16 x i8> %19, i64 0
  %21 = tail call spir_func signext i8 @_Z25sub_group_broadcast_firstc(i8 signext %20)
  ret void
}

declare dso_local spir_func signext i8 @_Z31sub_group_non_uniform_broadcastcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_cj(<2 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_cj(<3 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_cj(<4 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_cj(<8 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_cj(<16 x i8>, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z25sub_group_broadcast_firstc(i8 signext) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char2_0:]] = OpVectorShuffle %[[#char2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char2]] %[[#ScopeSubgroup]] %[[#char2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char3_0:]] = OpVectorShuffle %[[#char3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char3]] %[[#ScopeSubgroup]] %[[#char3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char4_0:]] = OpVectorShuffle %[[#char4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char4]] %[[#ScopeSubgroup]] %[[#char4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char8_0:]] = OpVectorShuffle %[[#char8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char8]] %[[#ScopeSubgroup]] %[[#char8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#char16]]
; CHECK-SPIRV: %[[#char16_0:]] = OpVectorShuffle %[[#char16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#char16]] %[[#ScopeSubgroup]] %[[#char16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#char_value:]] = OpCompositeExtract %[[#char]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#char]] %[[#ScopeSubgroup]] %[[#char_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastUChars() local_unnamed_addr {
  %1 = tail call spir_func zeroext i8 @_Z31sub_group_non_uniform_broadcasthj(i8 zeroext 0, i32 0)
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8> %3, i32 0)
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8> %7, i32 0)
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8> %11, i32 0)
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8> %15, i32 0)
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8> %18, i32 0)
  %20 = extractelement <16 x i8> %19, i64 0
  %21 = tail call spir_func zeroext i8 @_Z25sub_group_broadcast_firsth(i8 zeroext %20)
  ret void
}

declare dso_local spir_func zeroext i8 @_Z31sub_group_non_uniform_broadcasthj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8>, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z25sub_group_broadcast_firsth(i8 zeroext) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short2_0:]] = OpVectorShuffle %[[#short2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short2]] %[[#ScopeSubgroup]] %[[#short2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short3_0:]] = OpVectorShuffle %[[#short3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short3]] %[[#ScopeSubgroup]] %[[#short3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short4_0:]] = OpVectorShuffle %[[#short4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short4]] %[[#ScopeSubgroup]] %[[#short4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short8_0:]] = OpVectorShuffle %[[#short8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short8]] %[[#ScopeSubgroup]] %[[#short8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#short16]]
; CHECK-SPIRV: %[[#short16_0:]] = OpVectorShuffle %[[#short16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short16]] %[[#ScopeSubgroup]] %[[#short16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short_value:]] = OpCompositeExtract %[[#short]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#short]] %[[#ScopeSubgroup]] %[[#short_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastShorts() local_unnamed_addr {
  %1 = tail call spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 signext 0, i32 0)
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_sj(<2 x i16> %3, i32 0)
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_sj(<3 x i16> %7, i32 0)
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_sj(<4 x i16> %11, i32 0)
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_sj(<8 x i16> %15, i32 0)
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_sj(<16 x i16> %18, i32 0)
  %20 = extractelement <16 x i16> %19, i64 0
  %21 = tail call spir_func signext i16 @_Z25sub_group_broadcast_firsts(i16 signext %20)
  ret void
}

declare dso_local spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_sj(<2 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_sj(<3 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_sj(<4 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_sj(<8 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_sj(<16 x i16>, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z25sub_group_broadcast_firsts(i16 signext) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short2_0:]] = OpVectorShuffle %[[#short2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short2]] %[[#ScopeSubgroup]] %[[#short2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short3_0:]] = OpVectorShuffle %[[#short3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short3]] %[[#ScopeSubgroup]] %[[#short3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short4_0:]] = OpVectorShuffle %[[#short4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short4]] %[[#ScopeSubgroup]] %[[#short4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short8_0:]] = OpVectorShuffle %[[#short8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short8]] %[[#ScopeSubgroup]] %[[#short8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#short16]]
; CHECK-SPIRV: %[[#short16_0:]] = OpVectorShuffle %[[#short16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#short16]] %[[#ScopeSubgroup]] %[[#short16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#short_value:]] = OpCompositeExtract %[[#short]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#short]] %[[#ScopeSubgroup]] %[[#short_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastUShorts() local_unnamed_addr {
  %1 = tail call spir_func zeroext i16 @_Z31sub_group_non_uniform_broadcasttj(i16 zeroext 0, i32 0)
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16> %3, i32 0)
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16> %7, i32 0)
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16> %11, i32 0)
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16> %15, i32 0)
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16> %18, i32 0)
  %20 = extractelement <16 x i16> %19, i64 0
  %21 = tail call spir_func zeroext i16 @_Z25sub_group_broadcast_firstt(i16 zeroext %20)
  ret void
}

declare dso_local spir_func zeroext i16 @_Z31sub_group_non_uniform_broadcasttj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16>, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z25sub_group_broadcast_firstt(i16 zeroext) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int2_0:]] = OpVectorShuffle %[[#int2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int2]] %[[#ScopeSubgroup]] %[[#int2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int3_0:]] = OpVectorShuffle %[[#int3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int3]] %[[#ScopeSubgroup]] %[[#int3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int4_0:]] = OpVectorShuffle %[[#int4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int4]] %[[#ScopeSubgroup]] %[[#int4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int8_0:]] = OpVectorShuffle %[[#int8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int8]] %[[#ScopeSubgroup]] %[[#int8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#int16]]
; CHECK-SPIRV: %[[#int16_0:]] = OpVectorShuffle %[[#int16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int16]] %[[#ScopeSubgroup]] %[[#int16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int_value:]] = OpCompositeExtract %[[#int]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#int]] %[[#ScopeSubgroup]] %[[#int_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastInts() local_unnamed_addr {
  %1 = tail call spir_func i32 @_Z31sub_group_non_uniform_broadcastij(i32 0, i32 0)
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_ij(<2 x i32> %3, i32 0)
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_ij(<3 x i32> %7, i32 0)
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_ij(<4 x i32> %11, i32 0)
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_ij(<8 x i32> %15, i32 0)
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_ij(<16 x i32> %18, i32 0)
  %20 = extractelement <16 x i32> %19, i64 0
  %21 = tail call spir_func i32 @_Z25sub_group_broadcast_firsti(i32 %20)
  ret void
}

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_broadcastij(i32, i32) local_unnamed_addr

declare dso_local spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_ij(<2 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_ij(<3 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_ij(<4 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_ij(<8 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_ij(<16 x i32>, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z25sub_group_broadcast_firsti(i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int2_0:]] = OpVectorShuffle %[[#int2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int2]] %[[#ScopeSubgroup]] %[[#int2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int3_0:]] = OpVectorShuffle %[[#int3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int3]] %[[#ScopeSubgroup]] %[[#int3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int4_0:]] = OpVectorShuffle %[[#int4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int4]] %[[#ScopeSubgroup]] %[[#int4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int8_0:]] = OpVectorShuffle %[[#int8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int8]] %[[#ScopeSubgroup]] %[[#int8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#int16]]
; CHECK-SPIRV: %[[#int16_0:]] = OpVectorShuffle %[[#int16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#int16]] %[[#ScopeSubgroup]] %[[#int16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#int_value:]] = OpCompositeExtract %[[#int]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#int]] %[[#ScopeSubgroup]] %[[#int_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastUInts() local_unnamed_addr {
  %1 = tail call spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32 0, i32 0)
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32> %3, i32 0)
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32> %7, i32 0)
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32> %11, i32 0)
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32> %15, i32 0)
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32> %18, i32 0)
  %20 = extractelement <16 x i32> %19, i64 0
  %21 = tail call spir_func i32 @_Z25sub_group_broadcast_firstj(i32 %20)
  ret void
}

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32, i32) local_unnamed_addr

declare dso_local spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32>, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z25sub_group_broadcast_firstj(i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long2_0:]] = OpVectorShuffle %[[#long2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long2]] %[[#ScopeSubgroup]] %[[#long2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long3_0:]] = OpVectorShuffle %[[#long3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long3]] %[[#ScopeSubgroup]] %[[#long3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long4_0:]] = OpVectorShuffle %[[#long4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long4]] %[[#ScopeSubgroup]] %[[#long4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long8_0:]] = OpVectorShuffle %[[#long8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long8]] %[[#ScopeSubgroup]] %[[#long8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#long16]]
; CHECK-SPIRV: %[[#long16_0:]] = OpVectorShuffle %[[#long16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long16]] %[[#ScopeSubgroup]] %[[#long16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long_value:]] = OpCompositeExtract %[[#long]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#long]] %[[#ScopeSubgroup]] %[[#long_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastLongs() local_unnamed_addr {
  %1 = tail call spir_func i64 @_Z31sub_group_non_uniform_broadcastlj(i64 0, i32 0)
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_lj(<2 x i64> %3, i32 0)
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_lj(<3 x i64> %7, i32 0)
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_lj(<4 x i64> %11, i32 0)
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_lj(<8 x i64> %15, i32 0)
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_lj(<16 x i64> %18, i32 0)
  %20 = extractelement <16 x i64> %19, i64 0
  %21 = tail call spir_func i64 @_Z25sub_group_broadcast_firstl(i64 %20)
  ret void
}

declare dso_local spir_func i64 @_Z31sub_group_non_uniform_broadcastlj(i64, i32) local_unnamed_addr

declare dso_local spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_lj(<2 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_lj(<3 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_lj(<4 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_lj(<8 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_lj(<16 x i64>, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z25sub_group_broadcast_firstl(i64) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long2_0:]] = OpVectorShuffle %[[#long2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long2]] %[[#ScopeSubgroup]] %[[#long2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long3_0:]] = OpVectorShuffle %[[#long3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long3]] %[[#ScopeSubgroup]] %[[#long3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long4_0:]] = OpVectorShuffle %[[#long4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long4]] %[[#ScopeSubgroup]] %[[#long4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long8_0:]] = OpVectorShuffle %[[#long8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long8]] %[[#ScopeSubgroup]] %[[#long8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#long16]]
; CHECK-SPIRV: %[[#long16_0:]] = OpVectorShuffle %[[#long16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#long16]] %[[#ScopeSubgroup]] %[[#long16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#long_value:]] = OpCompositeExtract %[[#long]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#long]] %[[#ScopeSubgroup]] %[[#long_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastULongs() local_unnamed_addr {
  %1 = tail call spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64 0, i32 0)
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64> %3, i32 0)
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64> %7, i32 0)
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64> %11, i32 0)
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64> %15, i32 0)
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64> %18, i32 0)
  %20 = extractelement <16 x i64> %19, i64 0
  %21 = tail call spir_func i64 @_Z25sub_group_broadcast_firstm(i64 %20)
  ret void
}

declare dso_local spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64, i32) local_unnamed_addr

declare dso_local spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64>, i32) local_unnamed_addr

declare dso_local spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64>, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z25sub_group_broadcast_firstm(i64) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float]] %[[#ScopeSubgroup]] %[[#float_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#float2_0:]] = OpVectorShuffle %[[#float2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float2]] %[[#ScopeSubgroup]] %[[#float2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#float3_0:]] = OpVectorShuffle %[[#float3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float3]] %[[#ScopeSubgroup]] %[[#float3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#float4_0:]] = OpVectorShuffle %[[#float4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float4]] %[[#ScopeSubgroup]] %[[#float4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#float8_0:]] = OpVectorShuffle %[[#float8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float8]] %[[#ScopeSubgroup]] %[[#float8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#float16]]
; CHECK-SPIRV: %[[#float16_0:]] = OpVectorShuffle %[[#float16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#float16]] %[[#ScopeSubgroup]] %[[#float16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#float_value:]] = OpCompositeExtract %[[#float]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#float]] %[[#ScopeSubgroup]] %[[#float_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastFloats() local_unnamed_addr {
  %1 = tail call spir_func float @_Z31sub_group_non_uniform_broadcastfj(float 0.000000e+00, i32 0)
  %2 = insertelement <16 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %1, i64 0
  %3 = shufflevector <16 x float> %2, <16 x float> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x float> @_Z31sub_group_non_uniform_broadcastDv2_fj(<2 x float> %3, i32 0)
  %5 = shufflevector <2 x float> %4, <2 x float> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x float> %5, <16 x float> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x float> %6, <16 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x float> @_Z31sub_group_non_uniform_broadcastDv3_fj(<3 x float> %7, i32 0)
  %9 = shufflevector <3 x float> %8, <3 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x float> %9, <16 x float> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x float> %10, <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x float> @_Z31sub_group_non_uniform_broadcastDv4_fj(<4 x float> %11, i32 0)
  %13 = shufflevector <4 x float> %12, <4 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x float> %13, <16 x float> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x float> %14, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x float> @_Z31sub_group_non_uniform_broadcastDv8_fj(<8 x float> %15, i32 0)
  %17 = shufflevector <8 x float> %16, <8 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x float> %17, <16 x float> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x float> @_Z31sub_group_non_uniform_broadcastDv16_fj(<16 x float> %18, i32 0)
  %20 = extractelement <16 x float> %19, i64 0
  %21 = tail call spir_func float @_Z25sub_group_broadcast_firstf(float %20)
  ret void
}

declare dso_local spir_func float @_Z31sub_group_non_uniform_broadcastfj(float, i32) local_unnamed_addr

declare dso_local spir_func <2 x float> @_Z31sub_group_non_uniform_broadcastDv2_fj(<2 x float>, i32) local_unnamed_addr

declare dso_local spir_func <3 x float> @_Z31sub_group_non_uniform_broadcastDv3_fj(<3 x float>, i32) local_unnamed_addr

declare dso_local spir_func <4 x float> @_Z31sub_group_non_uniform_broadcastDv4_fj(<4 x float>, i32) local_unnamed_addr

declare dso_local spir_func <8 x float> @_Z31sub_group_non_uniform_broadcastDv8_fj(<8 x float>, i32) local_unnamed_addr

declare dso_local spir_func <16 x float> @_Z31sub_group_non_uniform_broadcastDv16_fj(<16 x float>, i32) local_unnamed_addr

declare dso_local spir_func float @_Z25sub_group_broadcast_firstf(float) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half]] %[[#ScopeSubgroup]] %[[#half_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#half2_0:]] = OpVectorShuffle %[[#half2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half2]] %[[#ScopeSubgroup]] %[[#half2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#half3_0:]] = OpVectorShuffle %[[#half3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half3]] %[[#ScopeSubgroup]] %[[#half3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#half4_0:]] = OpVectorShuffle %[[#half4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half4]] %[[#ScopeSubgroup]] %[[#half4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#half8_0:]] = OpVectorShuffle %[[#half8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half8]] %[[#ScopeSubgroup]] %[[#half8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#half16]]
; CHECK-SPIRV: %[[#half16_0:]] = OpVectorShuffle %[[#half16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#half16]] %[[#ScopeSubgroup]] %[[#half16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#half_value:]] = OpCompositeExtract %[[#half]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#half]] %[[#ScopeSubgroup]] %[[#half_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastHalfs() local_unnamed_addr {
  %1 = tail call spir_func half @_Z31sub_group_non_uniform_broadcastDhj(half 0xH0000, i32 0)
  %2 = insertelement <16 x half> <half undef, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000>, half %1, i64 0
  %3 = shufflevector <16 x half> %2, <16 x half> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x half> @_Z31sub_group_non_uniform_broadcastDv2_Dhj(<2 x half> %3, i32 0)
  %5 = shufflevector <2 x half> %4, <2 x half> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x half> %5, <16 x half> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x half> %6, <16 x half> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x half> @_Z31sub_group_non_uniform_broadcastDv3_Dhj(<3 x half> %7, i32 0)
  %9 = shufflevector <3 x half> %8, <3 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x half> %9, <16 x half> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x half> %10, <16 x half> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x half> @_Z31sub_group_non_uniform_broadcastDv4_Dhj(<4 x half> %11, i32 0)
  %13 = shufflevector <4 x half> %12, <4 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x half> %13, <16 x half> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x half> %14, <16 x half> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x half> @_Z31sub_group_non_uniform_broadcastDv8_Dhj(<8 x half> %15, i32 0)
  %17 = shufflevector <8 x half> %16, <8 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x half> %17, <16 x half> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x half> @_Z31sub_group_non_uniform_broadcastDv16_Dhj(<16 x half> %18, i32 0)
  %20 = extractelement <16 x half> %19, i64 0
  %21 = tail call spir_func half @_Z25sub_group_broadcast_firstDh(half %20)
  ret void
}

declare dso_local spir_func half @_Z31sub_group_non_uniform_broadcastDhj(half, i32) local_unnamed_addr

declare dso_local spir_func <2 x half> @_Z31sub_group_non_uniform_broadcastDv2_Dhj(<2 x half>, i32) local_unnamed_addr

declare dso_local spir_func <3 x half> @_Z31sub_group_non_uniform_broadcastDv3_Dhj(<3 x half>, i32) local_unnamed_addr

declare dso_local spir_func <4 x half> @_Z31sub_group_non_uniform_broadcastDv4_Dhj(<4 x half>, i32) local_unnamed_addr

declare dso_local spir_func <8 x half> @_Z31sub_group_non_uniform_broadcastDv8_Dhj(<8 x half>, i32) local_unnamed_addr

declare dso_local spir_func <16 x half> @_Z31sub_group_non_uniform_broadcastDv16_Dhj(<16 x half>, i32) local_unnamed_addr

declare dso_local spir_func half @_Z25sub_group_broadcast_firstDh(half) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double]] %[[#ScopeSubgroup]] %[[#double_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#double2_0:]] = OpVectorShuffle %[[#double2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double2]] %[[#ScopeSubgroup]] %[[#double2_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#double3_0:]] = OpVectorShuffle %[[#double3]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double3]] %[[#ScopeSubgroup]] %[[#double3_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#double4_0:]] = OpVectorShuffle %[[#double4]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double4]] %[[#ScopeSubgroup]] %[[#double4_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#double8_0:]] = OpVectorShuffle %[[#double8]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double8]] %[[#ScopeSubgroup]] %[[#double8_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpVectorShuffle %[[#double16]]
; CHECK-SPIRV: %[[#double16_0:]] = OpVectorShuffle %[[#double16]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcast %[[#double16]] %[[#ScopeSubgroup]] %[[#double16_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#double_value:]] = OpCompositeExtract %[[#double]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#double]] %[[#ScopeSubgroup]] %[[#double_value]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testNonUniformBroadcastDoubles() local_unnamed_addr {
  %1 = tail call spir_func double @_Z31sub_group_non_uniform_broadcastdj(double 0.000000e+00, i32 0)
  %2 = insertelement <16 x double> <double undef, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00>, double %1, i64 0
  %3 = shufflevector <16 x double> %2, <16 x double> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x double> @_Z31sub_group_non_uniform_broadcastDv2_dj(<2 x double> %3, i32 0)
  %5 = shufflevector <2 x double> %4, <2 x double> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x double> %5, <16 x double> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x double> %6, <16 x double> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x double> @_Z31sub_group_non_uniform_broadcastDv3_dj(<3 x double> %7, i32 0)
  %9 = shufflevector <3 x double> %8, <3 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x double> %9, <16 x double> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x double> %10, <16 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x double> @_Z31sub_group_non_uniform_broadcastDv4_dj(<4 x double> %11, i32 0)
  %13 = shufflevector <4 x double> %12, <4 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x double> %13, <16 x double> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x double> %14, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x double> @_Z31sub_group_non_uniform_broadcastDv8_dj(<8 x double> %15, i32 0)
  %17 = shufflevector <8 x double> %16, <8 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x double> %17, <16 x double> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x double> @_Z31sub_group_non_uniform_broadcastDv16_dj(<16 x double> %18, i32 0)
  %20 = extractelement <16 x double> %19, i64 0
  %21 = tail call spir_func double @_Z25sub_group_broadcast_firstd(double %20)
  ret void
}

declare dso_local spir_func double @_Z31sub_group_non_uniform_broadcastdj(double, i32) local_unnamed_addr

declare dso_local spir_func <2 x double> @_Z31sub_group_non_uniform_broadcastDv2_dj(<2 x double>, i32) local_unnamed_addr

declare dso_local spir_func <3 x double> @_Z31sub_group_non_uniform_broadcastDv3_dj(<3 x double>, i32) local_unnamed_addr

declare dso_local spir_func <4 x double> @_Z31sub_group_non_uniform_broadcastDv4_dj(<4 x double>, i32) local_unnamed_addr

declare dso_local spir_func <8 x double> @_Z31sub_group_non_uniform_broadcastDv8_dj(<8 x double>, i32) local_unnamed_addr

declare dso_local spir_func <16 x double> @_Z31sub_group_non_uniform_broadcastDv16_dj(<16 x double>, i32) local_unnamed_addr

declare dso_local spir_func double @_Z25sub_group_broadcast_firstd(double) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#ballot:]] = OpGroupNonUniformBallot %[[#int4]] %[[#ScopeSubgroup]] %[[#false]]
; CHECK-SPIRV: %[[#ballot2:]] = OpGroupNonUniformBallot %[[#int4]] %[[#ScopeSubgroup]] %[[#false]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformInverseBallot %[[#bool]] %[[#ScopeSubgroup]] %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformInverseBallot %[[#bool]] %[[#ScopeSubgroup]] %[[#ballot2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitExtract %[[#bool]] %[[#ScopeSubgroup]] %[[#ballot]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitExtract %[[#bool]] %[[#ScopeSubgroup]] %[[#ballot2]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] Reduce %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] Reduce %[[#ballot2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] InclusiveScan %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] InclusiveScan %[[#ballot2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] ExclusiveScan %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotBitCount %[[#int]] %[[#ScopeSubgroup]] ExclusiveScan %[[#ballot2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotFindLSB %[[#int]] %[[#ScopeSubgroup]] %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotFindLSB %[[#int]] %[[#ScopeSubgroup]] %[[#ballot2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotFindMSB %[[#int]] %[[#ScopeSubgroup]] %[[#ballot]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBallotFindMSB %[[#int]] %[[#ScopeSubgroup]] %[[#ballot2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testBallotOperations(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func <4 x i32> @_Z16sub_group_balloti(i32 0)
  %r2 = tail call spir_func <4 x i32> @__spirv_GroupNonUniformBallot(i32 3, i1 false)
  %3 = tail call spir_func i32 @_Z24sub_group_inverse_ballotDv4_j(<4 x i32> %2)
  %r3 = tail call spir_func i1 @__spirv_GroupNonUniformInverseBallot(i32 3, <4 x i32> %r2)
  store i32 %3, i32 addrspace(1)* %0, align 4
  %4 = tail call spir_func i32 @_Z28sub_group_ballot_bit_extractDv4_jj(<4 x i32> %2, i32 0)
  %r4 = tail call spir_func i32 @__spirv_GroupNonUniformBallotBitExtract(i32 3, <4 x i32> %r2, i32 0)
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %4, i32 addrspace(1)* %5, align 4
  %6 = tail call spir_func i32 @_Z26sub_group_ballot_bit_countDv4_j(<4 x i32> %2)
  %r6 = tail call spir_func i32 @__spirv_GroupNonUniformBallotBitCount(i32 3, i32 0, <4 x i32> %r2)
  %7 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %6, i32 addrspace(1)* %7, align 4
  %8 = tail call spir_func i32 @_Z31sub_group_ballot_inclusive_scanDv4_j(<4 x i32> %2)
  %r8 = tail call spir_func i32 @__spirv_GroupNonUniformBallotBitCount(i32 3, i32 1, <4 x i32> %r2)
  %9 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 3
  store i32 %8, i32 addrspace(1)* %9, align 4
  %10 = tail call spir_func i32 @_Z31sub_group_ballot_exclusive_scanDv4_j(<4 x i32> %2)
  %r10 = tail call spir_func i32 @__spirv_GroupNonUniformBallotBitCount(i32 3, i32 2, <4 x i32> %r2)
  %11 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 4
  store i32 %10, i32 addrspace(1)* %11, align 4
  %12 = tail call spir_func i32 @_Z25sub_group_ballot_find_lsbDv4_j(<4 x i32> %2)
  %r12 = tail call spir_func i32 @__spirv_GroupNonUniformBallotFindLSB(i32 3, <4 x i32> %r2)
  %13 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 5
  store i32 %12, i32 addrspace(1)* %13, align 4
  %14 = tail call spir_func i32 @_Z25sub_group_ballot_find_msbDv4_j(<4 x i32> %2)
  %r14 = tail call spir_func i32 @__spirv_GroupNonUniformBallotFindMSB(i32 3, <4 x i32> %r2)
  %15 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 6
  store i32 %14, i32 addrspace(1)* %15, align 4
  ret void
}

declare dso_local spir_func <4 x i32> @_Z16sub_group_balloti(i32) local_unnamed_addr
declare dso_local spir_func <4 x i32> @__spirv_GroupNonUniformBallot(i32, i1)

declare dso_local spir_func i32 @_Z24sub_group_inverse_ballotDv4_j(<4 x i32>) local_unnamed_addr
declare dso_local spir_func i1 @__spirv_GroupNonUniformInverseBallot(i32, <4 x i32>)

declare dso_local spir_func i32 @_Z28sub_group_ballot_bit_extractDv4_jj(<4 x i32>, i32) local_unnamed_addr
declare dso_local spir_func i1 @__spirv_GroupNonUniformBallotBitExtract(i32, <4 x i32>, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z26sub_group_ballot_bit_countDv4_j(<4 x i32>) local_unnamed_addr
declare dso_local spir_func i32 @__spirv_GroupNonUniformBallotBitCount(i32, i32, <4 x i32>)

declare dso_local spir_func i32 @_Z31sub_group_ballot_inclusive_scanDv4_j(<4 x i32>) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_ballot_exclusive_scanDv4_j(<4 x i32>) local_unnamed_addr

declare dso_local spir_func i32 @_Z25sub_group_ballot_find_lsbDv4_j(<4 x i32>) local_unnamed_addr
declare dso_local spir_func i32 @__spirv_GroupNonUniformBallotFindLSB(i32, <4 x i32>)

declare dso_local spir_func i32 @_Z25sub_group_ballot_find_msbDv4_j(<4 x i32>) local_unnamed_addr
declare dso_local spir_func i32 @__spirv_GroupNonUniformBallotFindMSB(i32, <4 x i32>)

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int4]] %[[#eqMask]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int4]] %[[#geMask]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int4]] %[[#gtMask]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int4]] %[[#leMask]]
; CHECK-SPIRV: %[[#]] = OpLoad %[[#int4]] %[[#ltMask]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testSubgroupMasks(<4 x i32> addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func <4 x i32> @_Z21get_sub_group_eq_maskv()
  store <4 x i32> %2, <4 x i32> addrspace(1)* %0, align 16
  %3 = tail call spir_func <4 x i32> @_Z21get_sub_group_ge_maskv()
  %4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 1
  store <4 x i32> %3, <4 x i32> addrspace(1)* %4, align 16
  %5 = tail call spir_func <4 x i32> @_Z21get_sub_group_gt_maskv()
  %6 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 2
  store <4 x i32> %5, <4 x i32> addrspace(1)* %6, align 16
  %7 = tail call spir_func <4 x i32> @_Z21get_sub_group_le_maskv()
  %8 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 3
  store <4 x i32> %7, <4 x i32> addrspace(1)* %8, align 16
  %9 = tail call spir_func <4 x i32> @_Z21get_sub_group_lt_maskv()
  %10 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 4
  store <4 x i32> %9, <4 x i32> addrspace(1)* %10, align 16
  ret void
}

declare dso_local spir_func <4 x i32> @_Z21get_sub_group_eq_maskv() local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z21get_sub_group_ge_maskv() local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z21get_sub_group_gt_maskv() local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z21get_sub_group_le_maskv() local_unnamed_addr

declare dso_local spir_func <4 x i32> @_Z21get_sub_group_lt_maskv() local_unnamed_addr
