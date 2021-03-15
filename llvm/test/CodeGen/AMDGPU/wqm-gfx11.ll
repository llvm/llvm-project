; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=CHECK %s

; Test that s_wqm is executed before lds.param.load.

;CHECK-LABEL: {{^}}test_param_load:
;CHECK: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
;CHECK: s_wqm_b32 exec_lo, exec_lo
;CHECK: lds_param_load
;CHECK: lds_param_load
;CHECK: lds_param_load
;CHECK: s_mov_b32 exec_lo, [[ORIG]]
;CHECK: v_add
;CHECK: v_add
;CHECK: v_add
define amdgpu_ps <3 x float> @test_param_load(i32 inreg %attr, <3 x float> %to_add) {
main_body:
  %a = call float @llvm.amdgcn.lds.param.load(i32 immarg 0, i32 immarg 0, i32 %attr) #1
  %b = call float @llvm.amdgcn.lds.param.load(i32 immarg 1, i32 immarg 0, i32 %attr) #1
  %c = call float @llvm.amdgcn.lds.param.load(i32 immarg 2, i32 immarg 0, i32 %attr) #1
  %tmp_0 = insertelement <3 x float> undef, float %a, i32 0
  %tmp_1 = insertelement <3 x float> %tmp_0, float %b, i32 1
  %tmp_2 = insertelement <3 x float> %tmp_1, float %c, i32 2
  %res = fadd <3 x float> %tmp_2, %to_add
  ret  <3 x float> %res
}

; Test that s_wqm is executed before lds.direct.load.

;CHECK-LABEL: {{^}}test_direct_load:
;CHECK: s_mov_b32 [[ORIG:s[0-9]+]], exec_lo
;CHECK: s_wqm_b32 exec_lo, exec_lo
;CHECK: lds_direct_load
;CHECK: lds_direct_load
;CHECK: lds_direct_load
;CHECK: s_mov_b32 exec_lo, [[ORIG]]
;CHECK: v_add
;CHECK: v_add
;CHECK: v_add
define amdgpu_ps <3 x float> @test_direct_load(i32 inreg %arg_0, i32 inreg %arg_1, i32 inreg %arg_2, <3 x float> %to_add) {
main_body:
  %a = call float @llvm.amdgcn.lds.direct.load(i32 %arg_0) #1
  %b = call float @llvm.amdgcn.lds.direct.load(i32 %arg_1) #1
  %c = call float @llvm.amdgcn.lds.direct.load(i32 %arg_2) #1
  %tmp_0 = insertelement <3 x float> undef, float %a, i32 0
  %tmp_1 = insertelement <3 x float> %tmp_0, float %b, i32 1
  %tmp_2 = insertelement <3 x float> %tmp_1, float %c, i32 2
  %res = fadd <3 x float> %tmp_2, %to_add
  ret  <3 x float> %res
}

attributes #1 = { nounwind readnone speculatable willreturn }
declare float @llvm.amdgcn.lds.param.load(i32 immarg, i32 immarg, i32) #1
declare float @llvm.amdgcn.lds.direct.load(i32) #1
