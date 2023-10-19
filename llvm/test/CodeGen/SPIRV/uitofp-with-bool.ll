; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPV

;; The IR was generated from the following source:
;; void __kernel K(global float* A, int B) {
;;   bool Cmp = B > 0;
;;   A[0] = Cmp;
;; }
;; Command line:
;; clang -x cl -cl-std=CL2.0 -target spir64 -emit-llvm -S -c test.cl


; SPV-DAG: OpName %[[#s1:]] "s1"
; SPV-DAG: OpName %[[#s2:]] "s2"
; SPV-DAG: OpName %[[#s3:]] "s3"
; SPV-DAG: OpName %[[#s4:]] "s4"
; SPV-DAG: OpName %[[#s5:]] "s5"
; SPV-DAG: OpName %[[#s6:]] "s6"
; SPV-DAG: OpName %[[#s7:]] "s7"
; SPV-DAG: OpName %[[#s8:]] "s8"
; SPV-DAG: OpName %[[#z1:]] "z1"
; SPV-DAG: OpName %[[#z2:]] "z2"
; SPV-DAG: OpName %[[#z3:]] "z3"
; SPV-DAG: OpName %[[#z4:]] "z4"
; SPV-DAG: OpName %[[#z5:]] "z5"
; SPV-DAG: OpName %[[#z6:]] "z6"
; SPV-DAG: OpName %[[#z7:]] "z7"
; SPV-DAG: OpName %[[#z8:]] "z8"
; SPV-DAG: OpName %[[#ufp1:]] "ufp1"
; SPV-DAG: OpName %[[#ufp2:]] "ufp2"
; SPV-DAG: OpName %[[#sfp1:]] "sfp1"
; SPV-DAG: OpName %[[#sfp2:]] "sfp2"
; SPV-DAG: %[[#int_32:]] = OpTypeInt 32 0
; SPV-DAG: %[[#int_8:]] = OpTypeInt 8 0
; SPV-DAG: %[[#int_16:]] = OpTypeInt 16 0
; SPV-DAG: %[[#int_64:]] = OpTypeInt 64 0
; SPV-DAG: %[[#zero_32:]] = OpConstant %[[#int_32]] 0
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

; SPV-DAG: OpFunction
; SPV-DAG: %[[#A:]] = OpFunctionParameter %[[#]]
; SPV-DAG: %[[#B:]] = OpFunctionParameter %[[#]]
; SPV-DAG: %[[#i1s:]] = OpFunctionParameter %[[#]]
; SPV-DAG: %[[#i1v:]] = OpFunctionParameter %[[#]]

define dso_local spir_kernel void @K(float addrspace(1)* nocapture %A, i32 %B, i1 %i1s, <2 x i1> %i1v) local_unnamed_addr {
entry:

; SPV-DAG: %[[#cmp_res:]] = OpSGreaterThan %[[#bool]] %[[#B]] %[[#zero_32]]
  %cmp = icmp sgt i32 %B, 0
; SPV-DAG: %[[#select_res:]] = OpSelect %[[#int_32]] %[[#cmp_res]] %[[#one_32]] %[[#zero_32]]
; SPV-DAG: %[[#utof_res:]] = OpConvertUToF %[[#float]] %[[#select_res]]
  %conv = uitofp i1 %cmp to float
; SPV-DAG: OpStore %[[#A]] %[[#utof_res]]
  store float %conv, float addrspace(1)* %A, align 4;

; SPV-DAG: %[[#s1]] = OpSelect %[[#int_8]] %[[#i1s]] %[[#mone_8]] %[[#zero_8]]
  %s1 = sext i1 %i1s to i8
; SPV-DAG: %[[#s2]] = OpSelect %[[#int_16]] %[[#i1s]] %[[#mone_16]] %[[#zero_16]]
  %s2 = sext i1 %i1s to i16
; SPV-DAG: %[[#s3]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#mone_32]] %[[#zero_32]]
  %s3 = sext i1 %i1s to i32
; SPV-DAG: %[[#s4]] = OpSelect %[[#int_64]] %[[#i1s]] %[[#mone_64]] %[[#zero_64]]
  %s4 = sext i1 %i1s to i64
; SPV-DAG: %[[#s5]] = OpSelect %[[#vec_8]] %[[#i1v]] %[[#mones_8]] %[[#zeros_8]]
  %s5 = sext <2 x i1> %i1v to <2 x i8>
; SPV-DAG: %[[#s6]] = OpSelect %[[#vec_16]] %[[#i1v]] %[[#mones_16]] %[[#zeros_16]]
  %s6 = sext <2 x i1> %i1v to <2 x i16>
; SPV-DAG: %[[#s7]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#mones_32]] %[[#zeros_32]]
  %s7 = sext <2 x i1> %i1v to <2 x i32>
; SPV-DAG: %[[#s8]] = OpSelect %[[#vec_64]] %[[#i1v]] %[[#mones_64]] %[[#zeros_64]]
  %s8 = sext <2 x i1> %i1v to <2 x i64>
; SPV-DAG: %[[#z1]] = OpSelect %[[#int_8]] %[[#i1s]] %[[#one_8]] %[[#zero_8]]
  %z1 = zext i1 %i1s to i8
; SPV-DAG: %[[#z2]] = OpSelect %[[#int_16]] %[[#i1s]] %[[#one_16]] %[[#zero_16]]
  %z2 = zext i1 %i1s to i16
; SPV-DAG: %[[#z3]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
  %z3 = zext i1 %i1s to i32
; SPV-DAG: %[[#z4]] = OpSelect %[[#int_64]] %[[#i1s]] %[[#one_64]] %[[#zero_64]]
  %z4 = zext i1 %i1s to i64
; SPV-DAG: %[[#z5]] = OpSelect %[[#vec_8]] %[[#i1v]] %[[#ones_8]] %[[#zeros_8]]
  %z5 = zext <2 x i1> %i1v to <2 x i8>
; SPV-DAG: %[[#z6]] = OpSelect %[[#vec_16]] %[[#i1v]] %[[#ones_16]] %[[#zeros_16]]
  %z6 = zext <2 x i1> %i1v to <2 x i16>
; SPV-DAG: %[[#z7]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
  %z7 = zext <2 x i1> %i1v to <2 x i32>
; SPV-DAG: %[[#z8]] = OpSelect %[[#vec_64]] %[[#i1v]] %[[#ones_64]] %[[#zeros_64]]
  %z8 = zext <2 x i1> %i1v to <2 x i64>
; SPV-DAG: %[[#ufp1_res:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
; SPV-DAG: %[[#ufp1]] = OpConvertUToF %[[#float]] %[[#ufp1_res]]
  %ufp1 = uitofp i1 %i1s to float
; SPV-DAG: %[[#ufp2_res:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
; SPV-DAG: %[[#ufp2]] = OpConvertUToF %[[#vec_float]] %[[#ufp2_res]]
  %ufp2 = uitofp <2 x i1> %i1v to <2 x float>
; SPV-DAG: %[[#sfp1_res:]] = OpSelect %[[#int_32]] %[[#i1s]] %[[#one_32]] %[[#zero_32]]
; SPV-DAG: %[[#sfp1]] = OpConvertSToF %[[#float]] %[[#sfp1_res]]
  %sfp1 = sitofp i1 %i1s to float
; SPV-DAG: %[[#sfp2_res:]] = OpSelect %[[#vec_32]] %[[#i1v]] %[[#ones_32]] %[[#zeros_32]]
; SPV-DAG: %[[#sfp2]] = OpConvertSToF %[[#vec_float]] %[[#sfp2_res]]
  %sfp2 = sitofp <2 x i1> %i1v to <2 x float>
  ret void
}
