;; #pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;;
;; kernel void testShuffleRelativeChar(global char* dst)
;; {
;;     char v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeUChar(global uchar* dst)
;; {
;;     uchar v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeShort(global short* dst)
;; {
;;     short v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeUShort(global ushort* dst)
;; {
;;     ushort v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeInt(global int* dst)
;; {
;;     int v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeUInt(global uint* dst)
;; {
;;     uint v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeLong(global long* dst)
;; {
;;     long v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeULong(global ulong* dst)
;; {
;;     ulong v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeFloat(global float* dst)
;; {
;;     float v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeHalf(global half* dst)
;; {
;;     half v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;;
;; kernel void testShuffleRelativeDouble(global double* dst)
;; {
;;     double v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: OpCapability GroupNonUniformShuffleRelative

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
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i8 @_Z20sub_group_shuffle_upcj(i8 signext 0, i32 0)
  %w2 = tail call spir_func i8 @__spirv_GroupNonUniformShuffleUp(i32 3, i8 signext 0, i32 0)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func signext i8 @_Z22sub_group_shuffle_downcj(i8 signext 0, i32 0)
  %w3 = tail call spir_func i8 @__spirv_GroupNonUniformShuffleDown(i32 3, i8 signext 0, i32 0)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  ret void
}

declare dso_local spir_func signext i8 @_Z20sub_group_shuffle_upcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z22sub_group_shuffle_downcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func i8 @__spirv_GroupNonUniformShuffleUp(i32, i8, i32)

declare dso_local spir_func i8 @__spirv_GroupNonUniformShuffleDown(i32, i8, i32)

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#char]] %[[#ScopeSubgroup]] %[[#char_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeUChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i8 @_Z20sub_group_shuffle_uphj(i8 zeroext 0, i32 0)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func zeroext i8 @_Z22sub_group_shuffle_downhj(i8 zeroext 0, i32 0)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  ret void
}

declare dso_local spir_func zeroext i8 @_Z20sub_group_shuffle_uphj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z22sub_group_shuffle_downhj(i8 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i16 @_Z20sub_group_shuffle_upsj(i16 signext 0, i32 0)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func signext i16 @_Z22sub_group_shuffle_downsj(i16 signext 0, i32 0)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func signext i16 @_Z20sub_group_shuffle_upsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z22sub_group_shuffle_downsj(i16 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#short]] %[[#ScopeSubgroup]] %[[#short_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeUShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i16 @_Z20sub_group_shuffle_uptj(i16 zeroext 0, i32 0)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func zeroext i16 @_Z22sub_group_shuffle_downtj(i16 zeroext 0, i32 0)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func zeroext i16 @_Z20sub_group_shuffle_uptj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z22sub_group_shuffle_downtj(i16 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func i32 @_Z20sub_group_shuffle_upij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z22sub_group_shuffle_downij(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#int]] %[[#ScopeSubgroup]] %[[#int_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeUInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z20sub_group_shuffle_upjj(i32 0, i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z22sub_group_shuffle_downjj(i32 0, i32 0)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func i32 @_Z20sub_group_shuffle_upjj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z22sub_group_shuffle_downjj(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeLong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z20sub_group_shuffle_uplj(i64 0, i32 0)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z22sub_group_shuffle_downlj(i64 0, i32 0)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func i64 @_Z20sub_group_shuffle_uplj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z22sub_group_shuffle_downlj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#long]] %[[#ScopeSubgroup]] %[[#long_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeULong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z20sub_group_shuffle_upmj(i64 0, i32 0)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z22sub_group_shuffle_downmj(i64 0, i32 0)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func i64 @_Z20sub_group_shuffle_upmj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z22sub_group_shuffle_downmj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#float]] %[[#ScopeSubgroup]] %[[#float_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#float]] %[[#ScopeSubgroup]] %[[#float_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeFloat(float addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func float @_Z20sub_group_shuffle_upfj(float 0.000000e+00, i32 0)
  store float %2, float addrspace(1)* %0, align 4
  %3 = tail call spir_func float @_Z22sub_group_shuffle_downfj(float 0.000000e+00, i32 0)
  %4 = getelementptr inbounds float, float addrspace(1)* %0, i64 1
  store float %3, float addrspace(1)* %4, align 4
  ret void
}

declare dso_local spir_func float @_Z20sub_group_shuffle_upfj(float, i32) local_unnamed_addr

declare dso_local spir_func float @_Z22sub_group_shuffle_downfj(float, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#half]] %[[#ScopeSubgroup]] %[[#half_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#half]] %[[#ScopeSubgroup]] %[[#half_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeHalf(half addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func half @_Z20sub_group_shuffle_upDhj(half 0xH0000, i32 0)
  store half %2, half addrspace(1)* %0, align 2
  %3 = tail call spir_func half @_Z22sub_group_shuffle_downDhj(half 0xH0000, i32 0)
  %4 = getelementptr inbounds half, half addrspace(1)* %0, i64 1
  store half %3, half addrspace(1)* %4, align 2
  ret void
}

declare dso_local spir_func half @_Z20sub_group_shuffle_upDhj(half, i32) local_unnamed_addr

declare dso_local spir_func half @_Z22sub_group_shuffle_downDhj(half, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleUp %[[#double]] %[[#ScopeSubgroup]] %[[#double_0]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformShuffleDown %[[#double]] %[[#ScopeSubgroup]] %[[#double_0]] %[[#int_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testShuffleRelativeDouble(double addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func double @_Z20sub_group_shuffle_updj(double 0.000000e+00, i32 0)
  store double %2, double addrspace(1)* %0, align 8
  %3 = tail call spir_func double @_Z22sub_group_shuffle_downdj(double 0.000000e+00, i32 0)
  %4 = getelementptr inbounds double, double addrspace(1)* %0, i64 1
  store double %3, double addrspace(1)* %4, align 8
  ret void
}

declare dso_local spir_func double @_Z20sub_group_shuffle_updj(double, i32) local_unnamed_addr

declare dso_local spir_func double @_Z22sub_group_shuffle_downdj(double, i32) local_unnamed_addr
