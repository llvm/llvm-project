; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
;
; Check that PS is wave64
; GCN-LABEL: _amdgpu_ps_main:
; GCN: s_or_b64 exec, exec
;
; Check that VS is wave32
; GCN-LABEL: _amdgpu_vs_main:
; GCN: s_or_b32 exec_lo, exec_lo
;
; Check that GS is wave32
; GCN-LABEL: _amdgpu_gs_main:
; GCN: s_or_b32 exec_lo, exec_lo
;
; Check that HS is wave32
; GCN-LABEL: _amdgpu_hs_main:
; GCN: s_or_b32 exec_lo, exec_lo
;
; Check that CS is wave32
; GCN-LABEL: _amdgpu_cs_main:
; GCN: s_or_b32 exec_lo, exec_lo
;
; Check that:
; PS_W32_EN (bit 15) of SPI_PS_IN_CONTROL (0xa1b6) is 0;
; VS_W32_EN (bit 23) of VGT_SHADER_STAGES_EN (0xa2d5) is 1;
; GS_W32_EN (bit 22) of VGT_SHADER_STAGES_EN (0xa2d5) is 1;
; HS_W32_EN (bit 21) of VGT_SHADER_STAGES_EN (0xa2d5) is 1;
; CS_W32_EN (bit 15) of COMPUTE_DISPATCH_INITIATOR (0x2e00) is 1.
;
; GCN: .amd_amdgpu_pal_metadata{{.*}},0x2e00,0x8000,{{.*}}0xa1b6,0x1,{{.*}},0xa2d5,0xe00000,

define dllexport amdgpu_ps void @_amdgpu_ps_main(float %arg10) #0 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 true, i1 true)
  ret void
}

define dllexport amdgpu_vs void @_amdgpu_vs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8, float %arg10) local_unnamed_addr #2 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 false, i1 false)
  ret void
}

define dllexport amdgpu_gs void @_amdgpu_gs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8, float %arg10) local_unnamed_addr #2 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 false, i1 false)
  ret void
}

define dllexport amdgpu_hs void @_amdgpu_hs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8, float %arg10) local_unnamed_addr #2 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 false, i1 false)
  ret void
}

define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 %arg5, i32 %arg6, i32 %arg7, i32 %arg8, float %arg10) local_unnamed_addr #2 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 false, i1 false)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #2

attributes #0 = { nounwind "InitialPSInputAddr"="2" "target-features"="+wavefrontsize64" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind "target-features"="+wavefrontsize32" }
attributes #3 = { nounwind readonly }

!amdgpu.pal.metadata = !{!8}

!6 = !{}
!8 = !{i32 268435482, i32 1, i32 268435488, i32 -1, i32 268435480, i32 -322237066, i32 268435481, i32 717283096, i32 268435538, i32 4096, i32 268435539, i32 8192, i32 11338, i32 53215232, i32 11339, i32 10, i32 41411, i32 4, i32 41393, i32 0, i32 41479, i32 0, i32 41476, i32 17301504, i32 41478, i32 1087, i32 41721, i32 45, i32 41633, i32 0, i32 41702, i32 0, i32 41653, i32 0, i32 41657, i32 0, i32 41661, i32 0, i32 41665, i32 0, i32 41645, i32 0, i32 41750, i32 14, i32 268435528, i32 0, i32 268435493, i32 0, i32 268435500, i32 0, i32 268435536, i32 0, i32 11274, i32 2883584, i32 11275, i32 4, i32 41412, i32 0, i32 41413, i32 4, i32 41400, i32 16777216, i32 41398, i32 1, i32 41395, i32 0, i32 41396, i32 0, i32 41397, i32 0, i32 41619, i32 100794764, i32 41475, i32 16, i32 41103, i32 15, i32 268435485, i32 0, i32 268435529, i32 0, i32 268435494, i32 0, i32 268435501, i32 0, i32 41685, i32 0, i32 268435460, i32 -431267536, i32 268435461, i32 -366377628, i32 268435476, i32 352863062, i32 268435477, i32 1678737839, i32 268435532, i32 1, i32 41642, i32 127, i32 11343, i32 268435459, i32 11344, i32 268435460, i32 11340, i32 268435456, i32 11342, i32 0, i32 41361, i32 0, i32 11276, i32 268435456}
