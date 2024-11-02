; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpSatConvertSToU

;; kernel void testSToU(global int2 *a, global uchar2 *res) {
;;   res[0] = convert_uchar2_sat(*a);
;; }

define dso_local spir_kernel void @testSToU(<2 x i32> addrspace(1)* nocapture noundef readonly %a, <2 x i8> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load <2 x i32>, <2 x i32> addrspace(1)* %a, align 8
  %call = call spir_func <2 x i8> @_Z18convert_uchar2_satDv2_i(<2 x i32> noundef %0)
  store <2 x i8> %call, <2 x i8> addrspace(1)* %res, align 2
  ret void
}

declare spir_func <2 x i8> @_Z18convert_uchar2_satDv2_i(<2 x i32> noundef) local_unnamed_addr

; CHECK-SPIRV: OpSatConvertUToS

;; kernel void testUToS(global uint2 *a, global char2 *res) {
;;   res[0] = convert_char2_sat(*a);
;; }

define dso_local spir_kernel void @testUToS(<2 x i32> addrspace(1)* nocapture noundef readonly %a, <2 x i8> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load <2 x i32>, <2 x i32> addrspace(1)* %a, align 8
  %call = call spir_func <2 x i8> @_Z17convert_char2_satDv2_j(<2 x i32> noundef %0)
  store <2 x i8> %call, <2 x i8> addrspace(1)* %res, align 2
  ret void
}

declare spir_func <2 x i8> @_Z17convert_char2_satDv2_j(<2 x i32> noundef) local_unnamed_addr

; CHECK-SPIRV: OpConvertUToF

;; kernel void testUToF(global uint2 *a, global float2 *res) {
;;   res[0] = convert_float2_rtz(*a);
;; }

define dso_local spir_kernel void @testUToF(<2 x i32> addrspace(1)* nocapture noundef readonly %a, <2 x float> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load <2 x i32>, <2 x i32> addrspace(1)* %a, align 8
  %call = call spir_func <2 x float> @_Z18convert_float2_rtzDv2_j(<2 x i32> noundef %0)
  store <2 x float> %call, <2 x float> addrspace(1)* %res, align 8
  ret void
}

declare spir_func <2 x float> @_Z18convert_float2_rtzDv2_j(<2 x i32> noundef) local_unnamed_addr

; CHECK-SPIRV: OpConvertFToU

;; kernel void testFToUSat(global float2 *a, global uint2 *res) {
;;   res[0] = convert_uint2_sat_rtn(*a);
;; }

define dso_local spir_kernel void @testFToUSat(<2 x float> addrspace(1)* nocapture noundef readonly %a, <2 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load <2 x float>, <2 x float> addrspace(1)* %a, align 8
  %call = call spir_func <2 x i32> @_Z21convert_uint2_sat_rtnDv2_f(<2 x float> noundef %0)
  store <2 x i32> %call, <2 x i32> addrspace(1)* %res, align 8
  ret void
}

declare spir_func <2 x i32> @_Z21convert_uint2_sat_rtnDv2_f(<2 x float> noundef) local_unnamed_addr

; CHECK-SPIRV: OpSatConvertSToU

;; kernel void testUToUSat(global uchar *a, global uint *res) {
;;   res[0] = convert_uint_sat(*a);
;; }

define dso_local spir_kernel void @testUToUSat(i8 addrspace(1)* nocapture noundef readonly %a, i32 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load i8, i8 addrspace(1)* %a, align 1
  %call = call spir_func i32 @_Z16convert_uint_sath(i8 noundef zeroext %0)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare spir_func i32 @_Z16convert_uint_sath(i8 noundef zeroext) local_unnamed_addr

; CHECK-SPIRV: OpSatConvertSToU

;; kernel void testUToUSat1(global uint *a, global uchar *res) {
;;   res[0] = convert_uchar_sat(*a);
;; }

define dso_local spir_kernel void @testUToUSat1(i32 addrspace(1)* nocapture noundef readonly %a, i8 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = load i32, i32 addrspace(1)* %a, align 4
  %call = call spir_func zeroext i8 @_Z17convert_uchar_satj(i32 noundef %0)
  store i8 %call, i8 addrspace(1)* %res, align 1
  ret void
}

declare spir_func zeroext i8 @_Z17convert_uchar_satj(i32 noundef) local_unnamed_addr

; CHECK-SPIRV: OpConvertFToU

;; kernel void testFToU(global float3 *a, global uint3 *res) {
;;   res[0] = convert_uint3_rtp(*a);
;; }

define dso_local spir_kernel void @testFToU(<3 x float> addrspace(1)* nocapture noundef readonly %a, <3 x i32> addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %castToVec4 = bitcast <3 x float> addrspace(1)* %a to <4 x float> addrspace(1)*
  %loadVec4 = load <4 x float>, <4 x float> addrspace(1)* %castToVec4, align 16
  %extractVec = shufflevector <4 x float> %loadVec4, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %call = call spir_func <3 x i32> @_Z17convert_uint3_rtpDv3_f(<3 x float> noundef %extractVec)
  %extractVec1 = shufflevector <3 x i32> %call, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %storetmp = bitcast <3 x i32> addrspace(1)* %res to <4 x i32> addrspace(1)*
  store <4 x i32> %extractVec1, <4 x i32> addrspace(1)* %storetmp, align 16
  ret void
}

declare spir_func <3 x i32> @_Z17convert_uint3_rtpDv3_f(<3 x float> noundef) local_unnamed_addr
