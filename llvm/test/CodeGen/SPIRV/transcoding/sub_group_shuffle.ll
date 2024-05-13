;; #pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;;
;; kernel void testShuffleChar(global char* dst)
;; {
;;     char v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleUChar(global uchar* dst)
;; {
;;     uchar v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleShort(global short* dst)
;; {
;;     short v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleUShort(global ushort* dst)
;; {
;;     ushort v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleInt(global int* dst)
;; {
;;     int v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleUInt(global uint* dst)
;; {
;;     uint v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleLong(global long* dst)
;; {
;;     long v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleULong(global ulong* dst)
;; {
;;     ulong v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleFloat(global float* dst)
;; {
;;     float v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleHalf(global half* dst)
;; {
;;     half v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;;
;; kernel void testShuffleDouble(global double* dst)
;; {
;;     double v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability GroupNonUniformShuffle

; CHECK-SPIRV-DAG: %[[#char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#short:]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#half:]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#double:]] = OpTypeFloat 64

; CHECK-SPIRV-DAG: %[[#ScopeSubgroup:]] = OpConstant %[[#int]] 3
; CHECK-SPIRV-DAG: %[[#char_0:]] = OpConstant %[[#char]] 0
; CHECK-SPIRV-DAG: %[[#short_0:]] = OpConstant %[[#short]] 0
; CHECK-SPIRV-DAG: %[[#int_0:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV-DAG: %[[#long_0:]] = OpConstantNull %[[#long]]
; CHECK-SPIRV-DAG: %[[#half_0:]] = OpConstant %[[#half]] 0
; CHECK-SPIRV-DAG: %[[#float_0:]] = OpConstant %[[#float]] 0
; CHECK-SPIRV-DAG: %[[#double_0:]] = OpConstant %[[#double]] 0

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i8 @_Z17sub_group_shufflecj(i8 signext 0, i32 0)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func signext i8 @_Z21sub_group_shuffle_xorcj(i8 signext 0, i32 0)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  ret void
}

declare dso_local spir_func signext i8 @_Z17sub_group_shufflecj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z21sub_group_shuffle_xorcj(i8 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleUChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i8 @_Z17sub_group_shufflehj(i8 zeroext 0, i32 0)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func zeroext i8 @_Z21sub_group_shuffle_xorhj(i8 zeroext 0, i32 0)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  ret void
}

declare dso_local spir_func zeroext i8 @_Z17sub_group_shufflehj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z21sub_group_shuffle_xorhj(i8 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i16 @_Z17sub_group_shufflesj(i16 signext 0, i32 0)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func signext i16 @_Z21sub_group_shuffle_xorsj(i16 signext 0, i32 0)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func signext i16 @_Z17sub_group_shufflesj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z21sub_group_shuffle_xorsj(i16 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleUShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i16 @_Z17sub_group_shuffletj(i16 zeroext 0, i32 0)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func zeroext i16 @_Z21sub_group_shuffle_xortj(i16 zeroext 0, i32 0)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func zeroext i16 @_Z17sub_group_shuffletj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z21sub_group_shuffle_xortj(i16 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z17sub_group_shuffleij(i32 0, i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func i32 @_Z17sub_group_shuffleij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorij(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleUInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z17sub_group_shufflejj(i32 0, i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z21sub_group_shuffle_xorjj(i32 0, i32 0)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func i32 @_Z17sub_group_shufflejj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorjj(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleLong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z17sub_group_shufflelj(i64 0, i32 0)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func i64 @_Z17sub_group_shufflelj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z21sub_group_shuffle_xorlj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleULong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z17sub_group_shufflemj(i64 0, i32 0)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z21sub_group_shuffle_xormj(i64 0, i32 0)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func i64 @_Z17sub_group_shufflemj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z21sub_group_shuffle_xormj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#float]] %[[#ScopeSubgroup]] %[[#float_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#float]] %[[#ScopeSubgroup]] %[[#float_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleFloat(float addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func float @_Z17sub_group_shufflefj(float 0.000000e+00, i32 0)
  store float %2, float addrspace(1)* %0, align 4
  %3 = tail call spir_func float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0)
  %4 = getelementptr inbounds float, float addrspace(1)* %0, i64 1
  store float %3, float addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func float @_Z17sub_group_shufflefj(float, i32) local_unnamed_addr

declare dso_local spir_func float @_Z21sub_group_shuffle_xorfj(float, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#half]] %[[#ScopeSubgroup]] %[[#half_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#half]] %[[#ScopeSubgroup]] %[[#half_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleHalf(half addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func half @_Z17sub_group_shuffleDhj(half 0xH0000, i32 0)
  store half %2, half addrspace(1)* %0, align 2
  %3 = tail call spir_func half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0)
  %4 = getelementptr inbounds half, half addrspace(1)* %0, i64 1
  store half %3, half addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func half @_Z17sub_group_shuffleDhj(half, i32) local_unnamed_addr

declare dso_local spir_func half @_Z21sub_group_shuffle_xorDhj(half, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffle %[[#double]] %[[#ScopeSubgroup]] %[[#double_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleXor %[[#double]] %[[#ScopeSubgroup]] %[[#double_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleDouble(double addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func double @_Z17sub_group_shuffledj(double 0.000000e+00, i32 0)
  store double %2, double addrspace(1)* %0, align 8
  %3 = tail call spir_func double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0)
  %4 = getelementptr inbounds double, double addrspace(1)* %0, i64 1
  store double %3, double addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func double @_Z17sub_group_shuffledj(double, i32) local_unnamed_addr

declare dso_local spir_func double @_Z21sub_group_shuffle_xordj(double, i32) local_unnamed_addr
