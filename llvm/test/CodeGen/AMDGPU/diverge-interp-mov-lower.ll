; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx810 -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX9 %s

; Testing for failures in divergence calculations when divergent intrinsic is lowered during instruction selection

@0 = external dso_local addrspace(4) constant [4 x <4 x float>]

; GCN-LABEL: {{^}}_amdgpu_ps_main:
; GCN-NOT: v_readfirstlane
; PRE-GFX9: flat_load_dword
; GFX9: global_load 
define dllexport amdgpu_ps void @_amdgpu_ps_main(i32 inreg %arg) local_unnamed_addr #0 {
.entry:
  %tmp = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %arg) #1
  %tmp1 = bitcast float %tmp to i32
  %tmp2 = srem i32 %tmp1, 4
  %tmp3 = select i1 false, i32 undef, i32 %tmp2
  %tmp4 = sext i32 %tmp3 to i64
  %tmp5 = getelementptr [4 x <4 x float>], ptr addrspace(4) @0, i64 0, i64 %tmp4
  %tmp6 = load <4 x float>, ptr addrspace(4) %tmp5, align 16
  %tmp7 = extractelement <4 x float> %tmp6, i32 3
  %tmp8 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float undef, float %tmp7) #1
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> undef, <2 x half> %tmp8, i1 true, i1 true) #2
  ret void
}

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #1
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #2

attributes #0 = { nounwind "InitialPSInputAddr"="0" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }
