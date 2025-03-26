;RUN: llc < %s -mtriple=amdgcn-pal -mcpu=gfx1010 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK
;RUN: llc < %s -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK

; ;CHECK-LABEL: {{^}}_amdgpu_ps_1_arg:
; ;CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_1_arg(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret2 = insertvalue { <4 x float> } poison, <4 x float> %ret1, 0
  ret { <4 x float> } %ret2
}

; CHECK-LABEL: {{^}}_amdgpu_ps_3_arg:
; CHECK: NumVgprs: 6
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_3_arg(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret2 = insertvalue { <4 x float> } poison, <4 x float> %ret1.2, 0
  ret { <4 x float> } %ret2
}

; CHECK-LABEL: {{^}}_amdgpu_ps_2_arg_gap:
; CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_2_arg_gap(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i3 = extractelement <2 x float> %arg5, i32 1
  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.2 = insertelement <4 x float> %ret1, float %i3, i32 1
  %ret2 = insertvalue { <4 x float> } poison, <4 x float> %ret1.2, 0
  ret { <4 x float> } %ret2
}

; Using InitialPSInputAddr of 0x2 causes the 2nd VGPR arg to be included in the packing - this increases the total number of VGPRs and in turn makes arg3 not be packed to be
; adjacent to arg1 (the only 2 used arguments)
; CHECK-LABEL: {{^}}_amdgpu_ps_2_arg_no_pack:
; CHECK: NumVgprs: 6
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_2_arg_no_pack(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #1 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i3 = extractelement <2 x float> %arg5, i32 1
  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.2 = insertelement <4 x float> %ret1, float %i3, i32 1
  %ret2 = insertvalue { <4 x float> } poison, <4 x float> %ret1.2, 0
  ret { <4 x float> } %ret2
}

; CHECK-LABEL: {{^}}_amdgpu_ps_all_arg:
; CHECK: NumVgprs: 24
define dllexport amdgpu_ps { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @_amdgpu_ps_all_arg(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %i4 = extractelement <3 x float> %arg6, i32 1
  %i5 = extractelement <2 x float> %arg7, i32 0
  %i6 = extractelement <2 x float> %arg8, i32 0
  %i7 = extractelement <2 x float> %arg9, i32 1

  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret1.3 = insertelement <4 x float> %ret1.2, float %i4, i32 3

  %ret2 = insertelement <4 x float> poison, float %i5, i32 0
  %ret2.1 = insertelement <4 x float> %ret2, float %i6, i32 1
  %ret2.2 = insertelement <4 x float> %ret2.1, float %i7, i32 2
  %ret2.3 = insertelement <4 x float> %ret2.2, float %arg10, i32 3

  %ret3 = insertelement <4 x float> poison, float %arg11, i32 0
  %ret3.1 = insertelement <4 x float> %ret3, float %arg12, i32 1
  %ret3.2 = insertelement <4 x float> %ret3.1, float %arg13, i32 2
  %ret3.3 = insertelement <4 x float> %ret3.2, float %arg14, i32 3

  %arg15.f = bitcast i32 %arg15 to float
  %arg16.f = bitcast i32 %arg16 to float
  %arg17.f = bitcast i32 %arg17 to float
  %arg18.f = bitcast i32 %arg18 to float

  %ret4 = insertelement <4 x float> poison, float %arg15.f, i32 0
  %ret4.1 = insertelement <4 x float> %ret4, float %arg16.f, i32 1
  %ret4.2 = insertelement <4 x float> %ret4.1, float %arg17.f, i32 2
  %ret4.3 = insertelement <4 x float> %ret4.2, float %arg18.f, i32 3

  %ret.res1 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } poison, <4 x float> %ret1.3, 0
  %ret.res2 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res1, <4 x float> %ret2.3, 1
  %ret.res3 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res2, <4 x float> %ret3.3, 2
  %ret.res  = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res3, <4 x float> %ret4.3, 3

  ret { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res
}

; Extra arguments have to be allocated even if they're unused
; CHECK-LABEL: {{^}}_amdgpu_ps_all_arg_extra_unused:
; CHECK: NumVgprs: 26
define dllexport amdgpu_ps { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @_amdgpu_ps_all_arg_extra_unused(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %i4 = extractelement <3 x float> %arg6, i32 1
  %i5 = extractelement <2 x float> %arg7, i32 0
  %i6 = extractelement <2 x float> %arg8, i32 0
  %i7 = extractelement <2 x float> %arg9, i32 1

  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret1.3 = insertelement <4 x float> %ret1.2, float %i4, i32 3

  %ret2 = insertelement <4 x float> poison, float %i5, i32 0
  %ret2.1 = insertelement <4 x float> %ret2, float %i6, i32 1
  %ret2.2 = insertelement <4 x float> %ret2.1, float %i7, i32 2
  %ret2.3 = insertelement <4 x float> %ret2.2, float %arg10, i32 3

  %ret3 = insertelement <4 x float> poison, float %arg11, i32 0
  %ret3.1 = insertelement <4 x float> %ret3, float %arg12, i32 1
  %ret3.2 = insertelement <4 x float> %ret3.1, float %arg13, i32 2
  %ret3.3 = insertelement <4 x float> %ret3.2, float %arg14, i32 3

  %arg15.f = bitcast i32 %arg15 to float
  %arg16.f = bitcast i32 %arg16 to float
  %arg17.f = bitcast i32 %arg17 to float
  %arg18.f = bitcast i32 %arg18 to float

  %ret4 = insertelement <4 x float> poison, float %arg15.f, i32 0
  %ret4.1 = insertelement <4 x float> %ret4, float %arg16.f, i32 1
  %ret4.2 = insertelement <4 x float> %ret4.1, float %arg17.f, i32 2
  %ret4.3 = insertelement <4 x float> %ret4.2, float %arg18.f, i32 3

  %ret.res1 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } poison, <4 x float> %ret1.3, 0
  %ret.res2 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res1, <4 x float> %ret2.3, 1
  %ret.res3 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res2, <4 x float> %ret3.3, 2
  %ret.res  = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res3, <4 x float> %ret4.3, 3

  ret { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res
}

; CHECK-LABEL: {{^}}_amdgpu_ps_all_arg_extra:
; CHECK: NumVgprs: 26
; CHECK: NumVGPRsForWavesPerEU: 26
define dllexport amdgpu_ps { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @_amdgpu_ps_all_arg_extra(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %i4 = extractelement <3 x float> %arg6, i32 1
  %i5 = extractelement <2 x float> %arg7, i32 0
  %i6 = extractelement <2 x float> %arg8, i32 0
  %i7 = extractelement <2 x float> %arg9, i32 1

  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret1.3 = insertelement <4 x float> %ret1.2, float %i4, i32 3

  %ret2 = insertelement <4 x float> poison, float %i5, i32 0
  %ret2.1 = insertelement <4 x float> %ret2, float %i6, i32 1
  %ret2.2 = insertelement <4 x float> %ret2.1, float %i7, i32 2
  %ret2.3 = insertelement <4 x float> %ret2.2, float %arg10, i32 3

  %ret3 = insertelement <4 x float> poison, float %arg11, i32 0
  %ret3.1 = insertelement <4 x float> %ret3, float %arg12, i32 1
  %ret3.2 = insertelement <4 x float> %ret3.1, float %arg13, i32 2
  %ret3.3 = insertelement <4 x float> %ret3.2, float %arg14, i32 3

  %arg15.f = bitcast i32 %arg15 to float
  %arg16.f = bitcast i32 %arg16 to float
  %arg17.f = bitcast i32 %arg17 to float
  %arg18.f = bitcast i32 %arg18 to float

  %arg15_16.f = fadd float %arg15.f, %arg16.f
  %arg17_18.f = fadd float %arg17.f, %arg18.f

  %ret4 = insertelement <4 x float> poison, float %extra_arg1, i32 0
  %ret4.1 = insertelement <4 x float> %ret4, float %extra_arg2, i32 1
  %ret4.2 = insertelement <4 x float> %ret4.1, float %arg15_16.f, i32 2
  %ret4.3 = insertelement <4 x float> %ret4.2, float %arg17_18.f, i32 3

  %ret.res1 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } poison, <4 x float> %ret1.3, 0
  %ret.res2 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res1, <4 x float> %ret2.3, 1
  %ret.res3 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res2, <4 x float> %ret3.3, 2
  %ret.res  = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res3, <4 x float> %ret4.3, 3

  ret { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res
}

; Check that when no input args are used we get the minimum allocation - note that we always enable the first input
; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused:
; CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #0 {
.entry:
  ret { <4 x float> } undef
}

; Check that when no input args are used we get the minimum allocation - note that we always enable the first input
; Additionally set the PSInputAddr to 0 via the metadata
; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused_ia0:
; CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused_ia0(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) local_unnamed_addr #3 {
.entry:
  ret { <4 x float> } undef
}

; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused_extra_used:
; CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused_extra_used(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %ret4.1 = insertelement <4 x float> poison, float %extra_arg1, i32 0
  %ret4.2 = insertelement <4 x float> %ret4.1, float %extra_arg2, i32 1

  %ret.res  = insertvalue { <4 x float> } poison, <4 x float> %ret4.2, 0

  ret { <4 x float> } %ret.res
}

; CHECK-LABEL: {{^}}_amdgpu_ps_part_unused_extra_used:
; CHECK: NumVgprs: 5
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_part_unused_extra_used(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %ret4.1 = insertelement <4 x float> poison, float %arg14, i32 0
  %ret4.2 = insertelement <4 x float> %ret4.1, float %extra_arg1, i32 1
  %ret4.3 = insertelement <4 x float> %ret4.2, float %extra_arg2, i32 2

  %ret.res  = insertvalue { <4 x float> } poison, <4 x float> %ret4.3, 0

  ret { <4 x float> } %ret.res
}

; CHECK-LABEL: {{^}}_amdgpu_ps_part_unused_extra_unused:
; CHECK: NumVgprs: 7
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_part_unused_extra_unused(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %ret4.1 = insertelement <4 x float> poison, float %arg12, i32 0
  %ret4.2 = insertelement <4 x float> %ret4.1, float %arg13, i32 1
  %ret4.3 = insertelement <4 x float> %ret4.2, float %arg14, i32 2

  %ret.res  = insertvalue { <4 x float> } poison, <4 x float> %ret4.3, 0

  ret { <4 x float> } %ret.res
}

; Extra unused inputs are always added to the allocation
; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused_extra_unused:
; CHECK: NumVgprs: 4
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused_extra_unused(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:

  ret { <4 x float> } undef
}

; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused_extra_used_no_packing:
; CHECK: NumVgprs: 26
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused_extra_used_no_packing(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #2 {
.entry:
  %ret4.1 = insertelement <4 x float> poison, float %extra_arg1, i32 0
  %ret4.2 = insertelement <4 x float> %ret4.1, float %extra_arg2, i32 1

  %ret.res  = insertvalue { <4 x float> } poison, <4 x float> %ret4.2, 0

  ret { <4 x float> } %ret.res
}

; CHECK-LABEL: {{^}}_amdgpu_ps_all_unused_extra_unused_no_packing:
; CHECK: NumVgprs: 26
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_all_unused_extra_unused_no_packing(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #2 {
.entry:
  ret { <4 x float> } undef
}

; CHECK-LABEL: {{^}}_amdgpu_ps_some_unused_arg_extra:
; CHECK: NumVgprs: 24
; CHECK: NumVGPRsForWavesPerEU: 24
define dllexport amdgpu_ps { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @_amdgpu_ps_some_unused_arg_extra(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #0 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %i4 = extractelement <3 x float> %arg6, i32 1
  %i5 = extractelement <2 x float> %arg7, i32 0
  %i6 = extractelement <2 x float> %arg8, i32 0
  %i7 = extractelement <2 x float> %arg9, i32 1

  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret1.3 = insertelement <4 x float> %ret1.2, float %i4, i32 3

  %ret2 = insertelement <4 x float> poison, float %i5, i32 0
  %ret2.1 = insertelement <4 x float> %ret2, float %i6, i32 1
  %ret2.2 = insertelement <4 x float> %ret2.1, float %i7, i32 2
  %ret2.3 = insertelement <4 x float> %ret2.2, float %arg10, i32 3

  %ret3 = insertelement <4 x float> poison, float %arg11, i32 0
  %ret3.1 = insertelement <4 x float> %ret3, float %arg12, i32 1
  %ret3.2 = insertelement <4 x float> %ret3.1, float %arg13, i32 2
  %ret3.3 = insertelement <4 x float> %ret3.2, float %arg14, i32 3

  %arg15.f = bitcast i32 %arg15 to float
  %arg16.f = bitcast i32 %arg16 to float

  %ret4 = insertelement <4 x float> poison, float %extra_arg1, i32 0
  %ret4.1 = insertelement <4 x float> %ret4, float %extra_arg2, i32 1
  %ret4.2 = insertelement <4 x float> %ret4.1, float %arg15.f, i32 2
  %ret4.3 = insertelement <4 x float> %ret4.2, float %arg16.f, i32 3

  %ret.res1 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } poison, <4 x float> %ret1.3, 0
  %ret.res2 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res1, <4 x float> %ret2.3, 1
  %ret.res3 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res2, <4 x float> %ret3.3, 2
  %ret.res  = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res3, <4 x float> %ret4.3, 3

  ret { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res
}

;CHECK-LABEL: {{^}}_amdgpu_ps_some_unused_no_packing_arg_extra:
;CHECK: NumVgprs: 26
;CHECK: NumVGPRsForWavesPerEU: 26
define dllexport amdgpu_ps { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @_amdgpu_ps_some_unused_no_packing_arg_extra(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18, float %extra_arg1, float %extra_arg2) local_unnamed_addr #2 {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %i2 = extractelement <2 x float> %arg4, i32 0
  %i3 = extractelement <2 x float> %arg5, i32 1
  %i4 = extractelement <3 x float> %arg6, i32 1
  %i5 = extractelement <2 x float> %arg7, i32 0
  %i6 = extractelement <2 x float> %arg8, i32 0
  %i7 = extractelement <2 x float> %arg9, i32 1

  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret1.1 = insertelement <4 x float> %ret1, float %i2, i32 1
  %ret1.2 = insertelement <4 x float> %ret1.1, float %i3, i32 2
  %ret1.3 = insertelement <4 x float> %ret1.2, float %i4, i32 3

  %ret2 = insertelement <4 x float> poison, float %i5, i32 0
  %ret2.1 = insertelement <4 x float> %ret2, float %i6, i32 1
  %ret2.2 = insertelement <4 x float> %ret2.1, float %i7, i32 2
  %ret2.3 = insertelement <4 x float> %ret2.2, float %arg10, i32 3

  %ret3 = insertelement <4 x float> poison, float %arg11, i32 0
  %ret3.1 = insertelement <4 x float> %ret3, float %arg12, i32 1
  %ret3.2 = insertelement <4 x float> %ret3.1, float %arg13, i32 2
  %ret3.3 = insertelement <4 x float> %ret3.2, float %arg14, i32 3

  %ret4 = insertelement <4 x float> poison, float %extra_arg1, i32 0
  %ret4.1 = insertelement <4 x float> %ret4, float %extra_arg2, i32 1

  %ret.res1 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } poison, <4 x float> %ret1.3, 0
  %ret.res2 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res1, <4 x float> %ret2.3, 1
  %ret.res3 = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res2, <4 x float> %ret3.3, 2
  %ret.res  = insertvalue { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res3, <4 x float> %ret4.1, 3

  ret { < 4 x float>, <4 x float>, <4 x float>, <4 x float> } %ret.res
}

attributes #0 = { nounwind "target-features"=",+wavefrontsize64,+cumode"  }
attributes #1 = { nounwind "InitialPSInputAddr"="2" "target-features"=",+wavefrontsize64,+cumode" }
attributes #2 = { nounwind "InitialPSInputAddr"="0xffff" "target-features"=",+wavefrontsize64,+cumode" }
attributes #3 = { nounwind "InitialPSInputAddr"="0" "target-features"=",+wavefrontsize64,+cumode" }
