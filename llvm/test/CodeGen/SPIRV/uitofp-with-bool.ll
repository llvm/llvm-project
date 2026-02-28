; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPV

;; The IR was generated from the following source:
;; void __kernel K(global float* A, int B) {
;;   bool Cmp = B > 0;
;;   A[0] = Cmp;
;; }
;; Command line:
;; clang -x cl -cl-std=CL2.0 -target spir64 -emit-llvm -S -c test.cl


; SPV-DAG: %[[#int_32:]] = OpTypeInt 32 0
; SPV-DAG: %[[#int_8:]] = OpTypeInt 8 0
; SPV-DAG: %[[#int_16:]] = OpTypeInt 16 0
; SPV-DAG: %[[#int_64:]] = OpTypeInt 64 0
; SPV-DAG: %[[#zero_32:]] = OpConstantNull %[[#int_32]]
; SPV-DAG: %[[#one_32:]] = OpConstant %[[#int_32]] 1
; SPV-DAG: %[[#zero_8:]] = OpConstantNull %[[#int_8]]
; SPV-DAG: %[[#mone_8:]] = OpConstant %[[#int_8]] 255
; SPV-DAG: %[[#zero_16:]] = OpConstantNull %[[#int_16]]
; SPV-DAG: %[[#mone_16:]] = OpConstant %[[#int_16]] 65535
; SPV-DAG: %[[#mone_32:]] = OpConstant %[[#int_32]] 4294967295
; SPV-DAG: %[[#zero_64:]] = OpConstantNull %[[#int_64]]
; SPV-DAG: %[[#mone_64:]] = OpConstant %[[#int_64]] 18446744073709551615
; SPV-DAG: %[[#one_8:]] = OpConstant %[[#int_8]] 1
; SPV-DAG: %[[#one_16:]] = OpConstant %[[#int_16]] 1
; SPV-DAG: %[[#one_64:]] = OpConstant %[[#int_64]] 1
; SPV-DAG: %[[#void:]] = OpTypeVoid
; SPV-DAG: %[[#float:]] = OpTypeFloat 32
; SPV-DAG: %[[#bool:]] = OpTypeBool
; SPV-DAG: %[[#vec_8:]] = OpTypeVector %[[#int_8]] 2
; SPV-DAG: %[[#vec_1:]] = OpTypeVector %[[#bool]] 2
; SPV-DAG: %[[#vec_16:]] = OpTypeVector %[[#int_16]] 2
; SPV-DAG: %[[#vec_32:]] = OpTypeVector %[[#int_32]] 2
; SPV-DAG: %[[#vec_64:]] = OpTypeVector %[[#int_64]] 2
; SPV-DAG: %[[#vec_float:]] = OpTypeVector %[[#float]] 2
; SPV-DAG: %[[#zeros_8:]] = OpConstantNull %[[#vec_8]]
; SPV-DAG: %[[#mones_8:]] = OpConstantComposite %[[#vec_8]] %[[#mone_8]] %[[#mone_8]]
; SPV-DAG: %[[#zeros_16:]] = OpConstantNull %[[#vec_16]]
; SPV-DAG: %[[#mones_16:]] = OpConstantComposite %[[#vec_16]] %[[#mone_16]] %[[#mone_16]]
; SPV-DAG: %[[#zeros_32:]] = OpConstantNull %[[#vec_32]]
; SPV-DAG: %[[#mones_32:]] = OpConstantComposite %[[#vec_32]] %[[#mone_32]] %[[#mone_32]]
; SPV-DAG: %[[#zeros_64:]] = OpConstantNull %[[#vec_64]]
; SPV-DAG: %[[#mones_64:]] = OpConstantComposite %[[#vec_64]] %[[#mone_64]] %[[#mone_64]]
; SPV-DAG: %[[#ones_8:]] = OpConstantComposite %[[#vec_8]] %[[#one_8]] %[[#one_8]]
; SPV-DAG: %[[#ones_16:]] = OpConstantComposite %[[#vec_16]] %[[#one_16]] %[[#one_16]]
; SPV-DAG: %[[#ones_32:]] = OpConstantComposite %[[#vec_32]] %[[#one_32]] %[[#one_32]]
; SPV-DAG: %[[#ones_64:]] = OpConstantComposite %[[#vec_64]] %[[#one_64]] %[[#one_64]]
; SPV-DAG: %[[#pointer:]] = OpTypePointer CrossWorkgroup %[[#float]]

@G_s1 = global i8 0
@G_s2 = global i16 0
@G_s3 = global i32 0
@G_s4 = global i64 0
@G_s5 = global <2 x i8> zeroinitializer
@G_s6 = global <2 x i16> zeroinitializer
@G_s7 = global <2 x i32> zeroinitializer
@G_s8 = global <2 x i64> zeroinitializer
@G_z1 = global i8 0
@G_z2 = global i16 0
@G_z3 = global i32 0
@G_z4 = global i64 0
@G_z5 = global <2 x i8> zeroinitializer
@G_z6 = global <2 x i16> zeroinitializer
@G_z7 = global <2 x i32> zeroinitializer
@G_z8 = global <2 x i64> zeroinitializer
@G_ufp1 = global float 0.0
@G_ufp2 = global <2 x float> zeroinitializer
@G_sfp1 = global float 0.0
@G_sfp2 = global <2 x float> zeroinitializer

; SPV-DAG: OpFunction
; SPV-DAG: %[[#A:]] = OpFunctionParameter %[[#pointer]]
; SPV-DAG: %[[#B:]] = OpFunctionParameter %[[#]]
; SPV-DAG: %[[#i1s:]] = OpFunctionParameter %[[#]]
; SPV-DAG: %[[#i1v:]] = OpFunctionParameter %[[#]]

define dso_local spir_kernel void @K(ptr addrspace(1) nocapture %A, i32 %B, i1 %i1s, <2 x i1> %i1v) local_unnamed_addr {
entry:

; SPV: %[[#cmp_res:]] = OpSGreaterThan %[[#bool]] %[[#B]] %[[#zero_32]]
  %cmp = icmp sgt i32 %B, 0
; SPV: %[[#select_res:]] = OpSelect %[[#int_32]] %[[#cmp_res]] %[[#one_32]] %[[#zero_32]]
; SPV: %[[#utof_res:]] = OpConvertUToF %[[#float]] %[[#select_res]]
  %conv = uitofp i1 %cmp to float
; SPV: OpStore %[[#A]] %[[#utof_res]]
  store float %conv, ptr addrspace(1) %A, align 4;

; SPV: %[[#s1:]] = OpSelect %[[#int_8]] %[[#i1s]] %[[#mone_8]] %[[#zero_8]]
  %s1 = sext i1 %i1s to i8
  store i8 %s1, ptr @G_s1
; SPV: %[[#s2:]] = OpSelect %[[#int_16]] %[[#i1s]] %[[#mone_16]] %[[#zero_16]]
  %s2 = sext i1 %i1s to i16
  store i16 %s2, ptr @G_s2
; SPV: %[[#s3:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#mone_32]] %[[#zero_32]]
  %s3 = sext i1 %i1s to i32
  store i32 %s3, ptr @G_s3
; SPV: %[[#s4:]] = OpSelect %[[#int_64]] %[[#i1s]] %[[#mone_64]] %[[#zero_64]]
  %s4 = sext i1 %i1s to i64
  store i64 %s4, ptr @G_s4
; SPV: %[[#s5:]] = OpSelect %[[#vec_8]] %[[#i1v]] %[[#mones_8]] %[[#zeros_8]]
  %s5 = sext <2 x i1> %i1v to <2 x i8>
  store <2 x i8> %s5, ptr @G_s5
; SPV: %[[#s6:]] = OpSelect %[[#vec_16]] %[[#i1v]] %[[#mones_16]] %[[#zeros_16]]
  %s6 = sext <2 x i1> %i1v to <2 x i16>
  store <2 x i16> %s6, ptr @G_s6
; SPV: %[[#s7:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#mones_32]] %[[#zeros_32]]
  %s7 = sext <2 x i1> %i1v to <2 x i32>
  store <2 x i32> %s7, ptr @G_s7
; SPV: %[[#s8:]] = OpSelect %[[#vec_64]] %[[#i1v]] %[[#mones_64]] %[[#zeros_64]]
  %s8 = sext <2 x i1> %i1v to <2 x i64>
  store <2 x i64> %s8, ptr @G_s8
; SPV: %[[#z1:]] = OpSelect %[[#int_8]] %[[#i1s]] %[[#one_8]] %[[#zero_8]]
  %z1 = zext i1 %i1s to i8
  store i8 %z1, ptr @G_z1
; SPV: %[[#z2:]] = OpSelect %[[#int_16]] %[[#i1s]] %[[#one_16]] %[[#zero_16]]
  %z2 = zext i1 %i1s to i16
  store i16 %z2, ptr @G_z2
; SPV: %[[#z3:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
  %z3 = zext i1 %i1s to i32
  store i32 %z3, ptr @G_z3
; SPV: %[[#z4:]] = OpSelect %[[#int_64]] %[[#i1s]] %[[#one_64]] %[[#zero_64]]
  %z4 = zext i1 %i1s to i64
  store i64 %z4, ptr @G_z4
; SPV: %[[#z5:]] = OpSelect %[[#vec_8]] %[[#i1v]] %[[#ones_8]] %[[#zeros_8]]
  %z5 = zext <2 x i1> %i1v to <2 x i8>
  store <2 x i8> %z5, ptr @G_z5
; SPV: %[[#z6:]] = OpSelect %[[#vec_16]] %[[#i1v]] %[[#ones_16]] %[[#zeros_16]]
  %z6 = zext <2 x i1> %i1v to <2 x i16>
  store <2 x i16> %z6, ptr @G_z6
; SPV: %[[#z7:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
  %z7 = zext <2 x i1> %i1v to <2 x i32>
  store <2 x i32> %z7, ptr @G_z7
; SPV: %[[#z8:]] = OpSelect %[[#vec_64]] %[[#i1v]] %[[#ones_64]] %[[#zeros_64]]
  %z8 = zext <2 x i1> %i1v to <2 x i64>
  store <2 x i64> %z8, ptr @G_z8
; SPV: %[[#ufp1_res:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
; SPV: %[[#ufp1:]] = OpConvertUToF %[[#float]] %[[#ufp1_res]]
  %ufp1 = uitofp i1 %i1s to float
  store float %ufp1, ptr @G_ufp1
; SPV: %[[#ufp2_res:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
; SPV: %[[#ufp2:]] = OpConvertUToF %[[#vec_float]] %[[#ufp2_res]]
  %ufp2 = uitofp <2 x i1> %i1v to <2 x float>
  store <2 x float> %ufp2, ptr @G_ufp2
; SPV: %[[#sfp1_res:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
; SPV: %[[#sfp1:]] = OpConvertSToF %[[#float]] %[[#sfp1_res]]
  %sfp1 = sitofp i1 %i1s to float
  store float %sfp1, ptr @G_sfp1
; SPV: %[[#sfp2_res:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
; SPV: %[[#sfp2:]] = OpConvertSToF %[[#vec_float]] %[[#sfp2_res]]
  %sfp2 = sitofp <2 x i1> %i1v to <2 x float>
  store <2 x float> %sfp2, ptr @G_sfp2
  ret void
}
