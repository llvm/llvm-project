; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx810 -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX9 %s

; A test case that originally failed in divergence calculation
; Implementation has to identify all formal args that can be a source of divergence

@0 = external dso_local addrspace(4) constant [6 x <2 x float>]

; GCN-LABEL: {{^}}_amdgpu_vs_main:
; GCN-NOT: v_readfirstlane
; PRE-GFX9: flat_load_dword
; GFX9: global_load 
define dllexport amdgpu_vs void @_amdgpu_vs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8) local_unnamed_addr #0 {
.entry:
  %tmp = add i32 %arg4, %arg8
  %tmp9 = sext i32 %tmp to i64
  %tmp10 = getelementptr [6 x <2 x float>], [6 x <2 x float>] addrspace(4)* @0, i64 0, i64 %tmp9
  %tmp11 = load <2 x float>, <2 x float> addrspace(4)* %tmp10, align 8
  %tmp12 = fadd nnan arcp contract <2 x float> zeroinitializer, %tmp11
  %tmp13 = extractelement <2 x float> %tmp12, i32 1
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float undef, float %tmp13, float 0.000000e+00, float 1.000000e+00, i1 true, i1 false) #1
  ret void
}

declare i64 @llvm.amdgcn.s.getpc() #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
