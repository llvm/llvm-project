;; #pragma OPENCL EXTENSION cl_khr_subgroup_clustered_reduce : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;;
;; kernel void testClusteredArithmeticChar(global char* dst)
;; {
;;     char v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticUChar(global uchar* dst)
;; {
;;     uchar v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticShort(global short* dst)
;; {
;;     short v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticUShort(global ushort* dst)
;; {
;;     ushort v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticInt(global int* dst)
;; {
;;     int v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticUInt(global uint* dst)
;; {
;;     uint v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticLong(global long* dst)
;; {
;;     long v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticULong(global ulong* dst)
;; {
;;     ulong v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticFloat(global float* dst)
;; {
;;     float v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticHalf(global half* dst)
;; {
;;     half v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredArithmeticDouble(global double* dst)
;; {
;;     double v = 0;
;;     dst[0] = sub_group_clustered_reduce_add(v, 2);
;;     dst[1] = sub_group_clustered_reduce_mul(v, 2);
;;     dst[2] = sub_group_clustered_reduce_min(v, 2);
;;     dst[3] = sub_group_clustered_reduce_max(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseChar(global char* dst)
;; {
;;     char v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseUChar(global uchar* dst)
;; {
;;     uchar v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseShort(global short* dst)
;; {
;;     short v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseUShort(global ushort* dst)
;; {
;;     ushort v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseInt(global int* dst)
;; {
;;     int v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseUInt(global uint* dst)
;; {
;;     uint v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseLong(global long* dst)
;; {
;;     long v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredBitwiseULong(global ulong* dst)
;; {
;;     ulong v = 0;
;;     dst[0] = sub_group_clustered_reduce_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_xor(v, 2);
;; }
;;
;; kernel void testClusteredLogical(global int* dst)
;; {
;;     int v = 0;
;;     dst[0] = sub_group_clustered_reduce_logical_and(v, 2);
;;     dst[1] = sub_group_clustered_reduce_logical_or(v, 2);
;;     dst[2] = sub_group_clustered_reduce_logical_xor(v, 2);
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability GroupNonUniformClustered

; CHECK-SPIRV-DAG: %[[#bool:]] = OpTypeBool
; CHECK-SPIRV-DAG: %[[#char:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#short:]] = OpTypeInt 16 0
; CHECK-SPIRV-DAG: %[[#int:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#long:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#half:]] = OpTypeFloat 16
; CHECK-SPIRV-DAG: %[[#float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#double:]] = OpTypeFloat 64

; CHECK-SPIRV-DAG: %[[#false:]] = OpConstantFalse %[[#bool]]
; CHECK-SPIRV-DAG: %[[#ScopeSubgroup:]] = OpConstant %[[#int]] 3
; CHECK-SPIRV-DAG: %[[#char_0:]] = OpConstant %[[#char]] 0
; CHECK-SPIRV-DAG: %[[#short_0:]] = OpConstant %[[#short]] 0
; CHECK-SPIRV-DAG: %[[#int_0:]] = OpConstant %[[#int]] 0
; CHECK-SPIRV-DAG: %[[#int_2:]] = OpConstant %[[#int]] 2
; CHECK-SPIRV-DAG: %[[#long_0:]] = OpConstantNull %[[#long]]
; CHECK-SPIRV-DAG: %[[#half_0:]] = OpConstant %[[#half]] 0
; CHECK-SPIRV-DAG: %[[#float_0:]] = OpConstant %[[#float]] 0
; CHECK-SPIRV-DAG: %[[#double_0:]] = OpConstant %[[#double]] 0

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMin %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMax %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_addcj(i8 signext 0, i32 2)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_mulcj(i8 signext 0, i32 2)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  %5 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_mincj(i8 signext 0, i32 2)
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1
  %7 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_maxcj(i8 signext 0, i32 2)
  %8 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 3
  store i8 %7, i8 addrspace(1)* %8, align 1
  ret void
}

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_addcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_mulcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_mincj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_maxcj(i8 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMin %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMax %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticUChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_addhj(i8 zeroext 0, i32 2)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_mulhj(i8 zeroext 0, i32 2)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  %5 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_minhj(i8 zeroext 0, i32 2)
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1
  %7 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_maxhj(i8 zeroext 0, i32 2)
  %8 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 3
  store i8 %7, i8 addrspace(1)* %8, align 1
  ret void
}

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_addhj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_mulhj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_minhj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_maxhj(i8 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMin %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMax %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_addsj(i16 signext 0, i32 2)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_mulsj(i16 signext 0, i32 2)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  %5 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_minsj(i16 signext 0, i32 2)
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2
  %7 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_maxsj(i16 signext 0, i32 2)
  %8 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 3
  store i16 %7, i16 addrspace(1)* %8, align 2
  ret void
}

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_addsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_mulsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_minsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_maxsj(i16 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMin %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMax %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticUShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_addtj(i16 zeroext 0, i32 2)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_multj(i16 zeroext 0, i32 2)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  %5 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_mintj(i16 zeroext 0, i32 2)
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2
  %7 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_maxtj(i16 zeroext 0, i32 2)
  %8 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 3
  store i16 %7, i16 addrspace(1)* %8, align 2
  ret void
}

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_addtj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_multj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_mintj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_maxtj(i16 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMin %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMax %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_addij(i32 0, i32 2)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_mulij(i32 0, i32 2)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  %5 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_minij(i32 0, i32 2)
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %5, i32 addrspace(1)* %6, align 4
  %7 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_maxij(i32 0, i32 2)
  %8 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 3
  store i32 %7, i32 addrspace(1)* %8, align 4
  ret void
}

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_addij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_mulij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_minij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_maxij(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMin %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMax %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticUInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_addjj(i32 0, i32 2)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_muljj(i32 0, i32 2)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  %5 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_minjj(i32 0, i32 2)
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %5, i32 addrspace(1)* %6, align 4
  %7 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_maxjj(i32 0, i32 2)
  %8 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 3
  store i32 %7, i32 addrspace(1)* %8, align 4
  ret void
}

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_addjj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_muljj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_minjj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_maxjj(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMin %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformSMax %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticLong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_addlj(i64 0, i32 2)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_mullj(i64 0, i32 2)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  %5 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_minlj(i64 0, i32 2)
  %6 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
  store i64 %5, i64 addrspace(1)* %6, align 8
  %7 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_maxlj(i64 0, i32 2)
  %8 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 3
  store i64 %7, i64 addrspace(1)* %8, align 8
  ret void
}

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_addlj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_mullj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_minlj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_maxlj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIAdd %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformIMul %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMin %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformUMax %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticULong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_addmj(i64 0, i32 2)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_mulmj(i64 0, i32 2)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  %5 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_minmj(i64 0, i32 2)
  %6 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
  store i64 %5, i64 addrspace(1)* %6, align 8
  %7 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_maxmj(i64 0, i32 2)
  %8 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 3
  store i64 %7, i64 addrspace(1)* %8, align 8
  ret void
}

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_addmj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_mulmj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_minmj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_maxmj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFAdd %[[#float]] %[[#ScopeSubgroup]] ClusteredReduce %[[#float_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMul %[[#float]] %[[#ScopeSubgroup]] ClusteredReduce %[[#float_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMin %[[#float]] %[[#ScopeSubgroup]] ClusteredReduce %[[#float_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMax %[[#float]] %[[#ScopeSubgroup]] ClusteredReduce %[[#float_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticFloat(float addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func float @_Z30sub_group_clustered_reduce_addfj(float 0.000000e+00, i32 2)
  store float %2, float addrspace(1)* %0, align 4
  %3 = tail call spir_func float @_Z30sub_group_clustered_reduce_mulfj(float 0.000000e+00, i32 2)
  %4 = getelementptr inbounds float, float addrspace(1)* %0, i64 1
  store float %3, float addrspace(1)* %4, align 4
  %5 = tail call spir_func float @_Z30sub_group_clustered_reduce_minfj(float 0.000000e+00, i32 2)
  %6 = getelementptr inbounds float, float addrspace(1)* %0, i64 2
  store float %5, float addrspace(1)* %6, align 4
  %7 = tail call spir_func float @_Z30sub_group_clustered_reduce_maxfj(float 0.000000e+00, i32 2)
  %8 = getelementptr inbounds float, float addrspace(1)* %0, i64 3
  store float %7, float addrspace(1)* %8, align 4
  ret void
}

declare dso_local spir_func float @_Z30sub_group_clustered_reduce_addfj(float, i32) local_unnamed_addr

declare dso_local spir_func float @_Z30sub_group_clustered_reduce_mulfj(float, i32) local_unnamed_addr

declare dso_local spir_func float @_Z30sub_group_clustered_reduce_minfj(float, i32) local_unnamed_addr

declare dso_local spir_func float @_Z30sub_group_clustered_reduce_maxfj(float, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFAdd %[[#half]] %[[#ScopeSubgroup]] ClusteredReduce %[[#half_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMul %[[#half]] %[[#ScopeSubgroup]] ClusteredReduce %[[#half_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMin %[[#half]] %[[#ScopeSubgroup]] ClusteredReduce %[[#half_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMax %[[#half]] %[[#ScopeSubgroup]] ClusteredReduce %[[#half_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticHalf(half addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func half @_Z30sub_group_clustered_reduce_addDhj(half 0xH0000, i32 2)
  store half %2, half addrspace(1)* %0, align 2
  %3 = tail call spir_func half @_Z30sub_group_clustered_reduce_mulDhj(half 0xH0000, i32 2)
  %4 = getelementptr inbounds half, half addrspace(1)* %0, i64 1
  store half %3, half addrspace(1)* %4, align 2
  %5 = tail call spir_func half @_Z30sub_group_clustered_reduce_minDhj(half 0xH0000, i32 2)
  %6 = getelementptr inbounds half, half addrspace(1)* %0, i64 2
  store half %5, half addrspace(1)* %6, align 2
  %7 = tail call spir_func half @_Z30sub_group_clustered_reduce_maxDhj(half 0xH0000, i32 2)
  %8 = getelementptr inbounds half, half addrspace(1)* %0, i64 3
  store half %7, half addrspace(1)* %8, align 2
  ret void
}

declare dso_local spir_func half @_Z30sub_group_clustered_reduce_addDhj(half, i32) local_unnamed_addr

declare dso_local spir_func half @_Z30sub_group_clustered_reduce_mulDhj(half, i32) local_unnamed_addr

declare dso_local spir_func half @_Z30sub_group_clustered_reduce_minDhj(half, i32) local_unnamed_addr

declare dso_local spir_func half @_Z30sub_group_clustered_reduce_maxDhj(half, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFAdd %[[#double]] %[[#ScopeSubgroup]] ClusteredReduce %[[#double_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMul %[[#double]] %[[#ScopeSubgroup]] ClusteredReduce %[[#double_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMin %[[#double]] %[[#ScopeSubgroup]] ClusteredReduce %[[#double_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformFMax %[[#double]] %[[#ScopeSubgroup]] ClusteredReduce %[[#double_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredArithmeticDouble(double addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func double @_Z30sub_group_clustered_reduce_adddj(double 0.000000e+00, i32 2)
  store double %2, double addrspace(1)* %0, align 8
  %3 = tail call spir_func double @_Z30sub_group_clustered_reduce_muldj(double 0.000000e+00, i32 2)
  %4 = getelementptr inbounds double, double addrspace(1)* %0, i64 1
  store double %3, double addrspace(1)* %4, align 8
  %5 = tail call spir_func double @_Z30sub_group_clustered_reduce_mindj(double 0.000000e+00, i32 2)
  %6 = getelementptr inbounds double, double addrspace(1)* %0, i64 2
  store double %5, double addrspace(1)* %6, align 8
  %7 = tail call spir_func double @_Z30sub_group_clustered_reduce_maxdj(double 0.000000e+00, i32 2)
  %8 = getelementptr inbounds double, double addrspace(1)* %0, i64 3
  store double %7, double addrspace(1)* %8, align 8
  ret void
}

declare dso_local spir_func double @_Z30sub_group_clustered_reduce_adddj(double, i32) local_unnamed_addr

declare dso_local spir_func double @_Z30sub_group_clustered_reduce_muldj(double, i32) local_unnamed_addr

declare dso_local spir_func double @_Z30sub_group_clustered_reduce_mindj(double, i32) local_unnamed_addr

declare dso_local spir_func double @_Z30sub_group_clustered_reduce_maxdj(double, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_andcj(i8 signext 0, i32 2)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func signext i8 @_Z29sub_group_clustered_reduce_orcj(i8 signext 0, i32 2)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  %5 = tail call spir_func signext i8 @_Z30sub_group_clustered_reduce_xorcj(i8 signext 0, i32 2)
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1
  ret void
}

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_andcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z29sub_group_clustered_reduce_orcj(i8 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i8 @_Z30sub_group_clustered_reduce_xorcj(i8 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#char]] %[[#ScopeSubgroup]] ClusteredReduce %[[#char_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseUChar(i8 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_andhj(i8 zeroext 0, i32 2)
  store i8 %2, i8 addrspace(1)* %0, align 1
  %3 = tail call spir_func zeroext i8 @_Z29sub_group_clustered_reduce_orhj(i8 zeroext 0, i32 2)
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1
  %5 = tail call spir_func zeroext i8 @_Z30sub_group_clustered_reduce_xorhj(i8 zeroext 0, i32 2)
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1
  ret void
}

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_andhj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z29sub_group_clustered_reduce_orhj(i8 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i8 @_Z30sub_group_clustered_reduce_xorhj(i8 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_andsj(i16 signext 0, i32 2)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func signext i16 @_Z29sub_group_clustered_reduce_orsj(i16 signext 0, i32 2)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  %5 = tail call spir_func signext i16 @_Z30sub_group_clustered_reduce_xorsj(i16 signext 0, i32 2)
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2
  ret void
}

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_andsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z29sub_group_clustered_reduce_orsj(i16 signext, i32) local_unnamed_addr

declare dso_local spir_func signext i16 @_Z30sub_group_clustered_reduce_xorsj(i16 signext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#short]] %[[#ScopeSubgroup]] ClusteredReduce %[[#short_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseUShort(i16 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_andtj(i16 zeroext 0, i32 2)
  store i16 %2, i16 addrspace(1)* %0, align 2
  %3 = tail call spir_func zeroext i16 @_Z29sub_group_clustered_reduce_ortj(i16 zeroext 0, i32 2)
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2
  %5 = tail call spir_func zeroext i16 @_Z30sub_group_clustered_reduce_xortj(i16 zeroext 0, i32 2)
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2
  ret void
}

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_andtj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z29sub_group_clustered_reduce_ortj(i16 zeroext, i32) local_unnamed_addr

declare dso_local spir_func zeroext i16 @_Z30sub_group_clustered_reduce_xortj(i16 zeroext, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_andij(i32 0, i32 2)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z29sub_group_clustered_reduce_orij(i32 0, i32 2)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  %5 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_xorij(i32 0, i32 2)
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %5, i32 addrspace(1)* %6, align 4
  ret void
}

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_andij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z29sub_group_clustered_reduce_orij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_xorij(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#int]] %[[#ScopeSubgroup]] ClusteredReduce %[[#int_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseUInt(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_andjj(i32 0, i32 2)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z29sub_group_clustered_reduce_orjj(i32 0, i32 2)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  %5 = tail call spir_func i32 @_Z30sub_group_clustered_reduce_xorjj(i32 0, i32 2)
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %5, i32 addrspace(1)* %6, align 4
  ret void
}

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_andjj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z29sub_group_clustered_reduce_orjj(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z30sub_group_clustered_reduce_xorjj(i32, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseLong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_andlj(i64 0, i32 2)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z29sub_group_clustered_reduce_orlj(i64 0, i32 2)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  %5 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_xorlj(i64 0, i32 2)
  %6 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
  store i64 %5, i64 addrspace(1)* %6, align 8
  ret void
}

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_andlj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z29sub_group_clustered_reduce_orlj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_xorlj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseAnd %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseOr  %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformBitwiseXor %[[#long]] %[[#ScopeSubgroup]] ClusteredReduce %[[#long_0]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredBitwiseULong(i64 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_andmj(i64 0, i32 2)
  store i64 %2, i64 addrspace(1)* %0, align 8
  %3 = tail call spir_func i64 @_Z29sub_group_clustered_reduce_ormj(i64 0, i32 2)
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8
  %5 = tail call spir_func i64 @_Z30sub_group_clustered_reduce_xormj(i64 0, i32 2)
  %6 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
  store i64 %5, i64 addrspace(1)* %6, align 8
  ret void
}

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_andmj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z29sub_group_clustered_reduce_ormj(i64, i32) local_unnamed_addr

declare dso_local spir_func i64 @_Z30sub_group_clustered_reduce_xormj(i64, i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformLogicalAnd %[[#bool]] %[[#ScopeSubgroup]] ClusteredReduce %[[#false]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformLogicalOr  %[[#bool]] %[[#ScopeSubgroup]] ClusteredReduce %[[#false]] %[[#int_2]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformLogicalXor %[[#bool]] %[[#ScopeSubgroup]] ClusteredReduce %[[#false]] %[[#int_2]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testClusteredLogical(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z38sub_group_clustered_reduce_logical_andij(i32 0, i32 2)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z37sub_group_clustered_reduce_logical_orij(i32 0, i32 2)
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4
  %5 = tail call spir_func i32 @_Z38sub_group_clustered_reduce_logical_xorij(i32 0, i32 2)
  %6 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %5, i32 addrspace(1)* %6, align 4
  ret void
}

declare dso_local spir_func i32 @_Z38sub_group_clustered_reduce_logical_andij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z37sub_group_clustered_reduce_logical_orij(i32, i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z38sub_group_clustered_reduce_logical_xorij(i32, i32) local_unnamed_addr
