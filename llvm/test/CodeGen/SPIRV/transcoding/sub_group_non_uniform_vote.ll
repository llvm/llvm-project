;; #pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable
;;
;; kernel void testSubGroupElect(global int* dst) {
;; 	dst[0] = sub_group_elect();
;; }
;;
;; kernel void testSubGroupNonUniformAll(global int* dst) {
;; 	dst[0] = sub_group_non_uniform_all(0); 
;; }
;;
;; kernel void testSubGroupNonUniformAny(global int* dst) {
;; 	dst[0] = sub_group_non_uniform_any(0);
;; }
;;
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; kernel void testSubGroupNonUniformAllEqual(global int* dst) {
;;     {
;;         char v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         uchar v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         short v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         ushort v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         int v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         uint v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         long v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         ulong v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         float v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         half v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         double v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpCapability GroupNonUniformVote

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
; CHECK-SPIRV-DAG: %[[#long_0:]] = OpConstantNull %[[#long]]
; CHECK-SPIRV-DAG: %[[#half_0:]] = OpConstant %[[#half]] 0
; CHECK-SPIRV-DAG: %[[#float_0:]] = OpConstant %[[#float]] 0
; CHECK-SPIRV-DAG: %[[#double_0:]] = OpConstant %[[#double]] 0

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformElect %[[#bool]] %[[#ScopeSubgroup]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testSubGroupElect(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z15sub_group_electv()
  store i32 %2, i32 addrspace(1)* %0, align 4
  ret void
}

declare dso_local spir_func i32 @_Z15sub_group_electv() local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAll %[[#bool]] %[[#ScopeSubgroup]] %[[#false]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testSubGroupNonUniformAll(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z25sub_group_non_uniform_alli(i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  ret void
}

declare dso_local spir_func i32 @_Z25sub_group_non_uniform_alli(i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAny %[[#bool]] %[[#ScopeSubgroup]] %[[#false]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testSubGroupNonUniformAny(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z25sub_group_non_uniform_anyi(i32 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  ret void
}

declare dso_local spir_func i32 @_Z25sub_group_non_uniform_anyi(i32) local_unnamed_addr

; CHECK-SPIRV: OpFunction
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#char_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#char_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#short_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#short_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#int_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#long_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#long_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#float_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#half_0]]
; CHECK-SPIRV: %[[#]] = OpGroupNonUniformAllEqual %[[#bool]] %[[#ScopeSubgroup]] %[[#double_0]]
; CHECK-SPIRV: OpFunctionEnd

define dso_local spir_kernel void @testSubGroupNonUniformAllEqual(i32 addrspace(1)* nocapture) local_unnamed_addr {
  %2 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 signext 0)
  store i32 %2, i32 addrspace(1)* %0, align 4
  %3 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalh(i8 zeroext 0)
  store i32 %3, i32 addrspace(1)* %0, align 4
  %4 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 signext 0)
  store i32 %4, i32 addrspace(1)* %0, align 4
  %5 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalt(i16 zeroext 0)
  store i32 %5, i32 addrspace(1)* %0, align 4
  %6 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32 0)
  store i32 %6, i32 addrspace(1)* %0, align 4
  %7 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalj(i32 0)
  store i32 %7, i32 addrspace(1)* %0, align 4
  %8 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64 0)
  store i32 %8, i32 addrspace(1)* %0, align 4
  %9 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalm(i64 0)
  store i32 %9, i32 addrspace(1)* %0, align 4
  %10 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalf(float 0.000000e+00)
  store i32 %10, i32 addrspace(1)* %0, align 4
  %11 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalDh(half 0xH0000)
  store i32 %11, i32 addrspace(1)* %0, align 4
  %12 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equald(double 0.000000e+00)
  store i32 %12, i32 addrspace(1)* %0, align 4
  ret void
}

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 signext) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalh(i8 zeroext) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 signext) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalt(i16 zeroext) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalj(i32) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalm(i64) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalf(float) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalDh(half) local_unnamed_addr

declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equald(double) local_unnamed_addr
