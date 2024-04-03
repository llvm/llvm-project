; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}vgpr:
; GCN-DAG: v_mov_b32_e32 v1, v0
; GCN-DAG: exp mrt0 v0, v0, v0, v0 done vm
; GCN: s_waitcnt expcnt(0)
; GCN: v_add_f32_e32 v0, 1.0, v1
; GCN-NOT: s_endpgm
define amdgpu_vs { float, float } @vgpr(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %arg3, float %arg3, float %arg3, float %arg3, i1 true, i1 true) #0
  %x = fadd float %arg3, 1.000000e+00
  %a = insertvalue { float, float } undef, float %x, 0
  %b = insertvalue { float, float } %a, float %arg3, 1
  ret { float, float } %b
}

; GCN-LABEL: {{^}}vgpr_literal:
; GCN: exp mrt0 v0, v0, v0, v0 done vm

; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-DAG: v_mov_b32_e32 v3, -1.0
; GCN-DAG: s_waitcnt expcnt(0)
; GCN-NOT: s_endpgm
define amdgpu_vs { float, float, float, float } @vgpr_literal(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %arg3, float %arg3, float %arg3, float %arg3, i1 true, i1 true) #0
  ret { float, float, float, float } { float 1.000000e+00, float 2.000000e+00, float 4.000000e+00, float -1.000000e+00 }
}

; GCN: .long 165580
; GCN-NEXT: .long 562
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 562
; GCN-LABEL: {{^}}vgpr_ps_addr0:
; GCN-NOT: v_mov_b32_e32 v0
; GCN-NOT: v_mov_b32_e32 v1
; GCN-NOT: v_mov_b32_e32 v2
; GCN: v_mov_b32_e32 v3, v4
; GCN: v_mov_b32_e32 v4, v6
; GCN-NOT: s_endpgm
define amdgpu_ps { float, float, float, float, float } @vgpr_ps_addr0(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #1 {
bb:
  %i0 = extractelement <2 x i32> %arg4, i32 0
  %i1 = extractelement <2 x i32> %arg4, i32 1
  %i2 = extractelement <2 x i32> %arg7, i32 0
  %i3 = extractelement <2 x i32> %arg8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue { float, float, float, float, float } undef, float %f0, 0
  %r1 = insertvalue { float, float, float, float, float } %r0, float %f1, 1
  %r2 = insertvalue { float, float, float, float, float } %r1, float %f2, 2
  %r3 = insertvalue { float, float, float, float, float } %r2, float %f3, 3
  %r4 = insertvalue { float, float, float, float, float } %r3, float %arg12, 4
  ret { float, float, float, float, float } %r4
}

; GCN: .long 165580
; GCN-NEXT: .long 1
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 1
; GCN-LABEL: {{^}}ps_input_ena_no_inputs:
; GCN: v_mov_b32_e32 v0, 1.0
; GCN-NOT: s_endpgm
define amdgpu_ps float @ps_input_ena_no_inputs(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #1 {
bb:
  ret float 1.000000e+00
}

; GCN: .long 165580
; GCN-NEXT: .long 2081
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 2081
; GCN-LABEL: {{^}}ps_input_ena_pos_w:
; GCN-DAG: v_mov_b32_e32 v0, v4
; GCN-DAG: v_mov_b32_e32 v1, v2
; GCN-DAG: v_mov_b32_e32 v2, v3
; GCN-NOT: s_endpgm
define amdgpu_ps { float, <2 x float> } @ps_input_ena_pos_w(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #1 {
bb:
  %f = bitcast <2 x i32> %arg8 to <2 x float>
  %s = insertvalue { float, <2 x float> } undef, float %arg14, 0
  %s1 = insertvalue { float, <2 x float> } %s, <2 x float> %f, 1
  ret { float, <2 x float> } %s1
}

; GCN: .long 165580
; GCN-NEXT: .long 562
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 563
; GCN-LABEL: {{^}}vgpr_ps_addr1:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mov_b32_e32 v2, v4
; GCN-DAG: v_mov_b32_e32 v3, v6
; GCN-DAG: v_mov_b32_e32 v4, v8
; GCN-NOT: s_endpgm
define amdgpu_ps { float, float, float, float, float } @vgpr_ps_addr1(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #2 {
bb:
  %i0 = extractelement <2 x i32> %arg4, i32 0
  %i1 = extractelement <2 x i32> %arg4, i32 1
  %i2 = extractelement <2 x i32> %arg7, i32 0
  %i3 = extractelement <2 x i32> %arg8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue { float, float, float, float, float } undef, float %f0, 0
  %r1 = insertvalue { float, float, float, float, float } %r0, float %f1, 1
  %r2 = insertvalue { float, float, float, float, float } %r1, float %f2, 2
  %r3 = insertvalue { float, float, float, float, float } %r2, float %f3, 3
  %r4 = insertvalue { float, float, float, float, float } %r3, float %arg12, 4
  ret { float, float, float, float, float } %r4
}

; GCN: .long 165580
; GCN-NEXT: .long 562
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 631
; GCN-LABEL: {{^}}vgpr_ps_addr119:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN-DAG: v_mov_b32_e32 v2, v6
; GCN-DAG: v_mov_b32_e32 v3, v8
; GCN-DAG: v_mov_b32_e32 v4, v12
; GCN-NOT: s_endpgm
define amdgpu_ps { float, float, float, float, float } @vgpr_ps_addr119(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #3 {
bb:
  %i0 = extractelement <2 x i32> %arg4, i32 0
  %i1 = extractelement <2 x i32> %arg4, i32 1
  %i2 = extractelement <2 x i32> %arg7, i32 0
  %i3 = extractelement <2 x i32> %arg8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue { float, float, float, float, float } undef, float %f0, 0
  %r1 = insertvalue { float, float, float, float, float } %r0, float %f1, 1
  %r2 = insertvalue { float, float, float, float, float } %r1, float %f2, 2
  %r3 = insertvalue { float, float, float, float, float } %r2, float %f3, 3
  %r4 = insertvalue { float, float, float, float, float } %r3, float %arg12, 4
  ret { float, float, float, float, float } %r4
}

; GCN: .long 165580
; GCN-NEXT: .long 562
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 946
; GCN-LABEL: {{^}}vgpr_ps_addr418:
; GCN-NOT: v_mov_b32_e32 v0
; GCN-NOT: v_mov_b32_e32 v1
; GCN-NOT: v_mov_b32_e32 v2
; GCN: v_mov_b32_e32 v3, v4
; GCN: v_mov_b32_e32 v4, v8
; GCN-NOT: s_endpgm
define amdgpu_ps { float, float, float, float, float } @vgpr_ps_addr418(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <2 x i32> %arg4, <2 x i32> %arg5, <3 x i32> %arg6, <2 x i32> %arg7, <2 x i32> %arg8, <2 x i32> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, float %arg15, float %arg16, float %arg17, float %arg18) #4 {
bb:
  %i0 = extractelement <2 x i32> %arg4, i32 0
  %i1 = extractelement <2 x i32> %arg4, i32 1
  %i2 = extractelement <2 x i32> %arg7, i32 0
  %i3 = extractelement <2 x i32> %arg8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue { float, float, float, float, float } undef, float %f0, 0
  %r1 = insertvalue { float, float, float, float, float } %r0, float %f1, 1
  %r2 = insertvalue { float, float, float, float, float } %r1, float %f2, 2
  %r3 = insertvalue { float, float, float, float, float } %r2, float %f3, 3
  %r4 = insertvalue { float, float, float, float, float } %r3, float %arg12, 4
  ret { float, float, float, float, float } %r4
}

; GCN-LABEL: {{^}}sgpr:
; GCN-DAG: s_mov_b32 s2, s3
; GCN-DAG: s_add_{{i|u}}32 s0, s3, 2
; GCN-NOT: s_endpgm
define amdgpu_vs { i32, i32, i32 } @sgpr(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  %x = add i32 %arg2, 2
  %a = insertvalue { i32, i32, i32 } undef, i32 %x, 0
  %b = insertvalue { i32, i32, i32 } %a, i32 %arg1, 1
  %c = insertvalue { i32, i32, i32 } %a, i32 %arg2, 2
  ret { i32, i32, i32 } %c
}

; GCN-LABEL: {{^}}sgpr_literal:
; GCN: s_mov_b32 s0, 5
; GCN-NOT: s_mov_b32 s0, s0
; GCN-DAG: s_mov_b32 s1, 6
; GCN-DAG: s_mov_b32 s2, 7
; GCN-DAG: s_mov_b32 s3, 8
; GCN-NOT: s_endpgm
define amdgpu_vs { i32, i32, i32, i32 } @sgpr_literal(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  %x = add i32 %arg2, 2
  ret { i32, i32, i32, i32 } { i32 5, i32 6, i32 7, i32 8 }
}

; GCN-LABEL: {{^}}both:
; GCN-DAG: exp mrt0 v0, v0, v0, v0 done vm
; GCN-DAG: v_mov_b32_e32 v1, v0
; GCN-DAG: s_mov_b32 s1, s2
; GCN-DAG: s_waitcnt expcnt(0)
; GCN-DAG: v_add_f32_e32 v0, 1.0, v1
; GCN-DAG: s_add_{{i|u}}32 s0, s3, 2
; GCN-DAG: s_mov_b32 s2, s3
; GCN-NOT: s_endpgm
define amdgpu_vs { float, i32, float, i32, i32 } @both(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %arg3, float %arg3, float %arg3, float %arg3, i1 true, i1 true) #0
  %v = fadd float %arg3, 1.000000e+00
  %s = add i32 %arg2, 2
  %a0 = insertvalue { float, i32, float, i32, i32 } undef, float %v, 0
  %a1 = insertvalue { float, i32, float, i32, i32 } %a0, i32 %s, 1
  %a2 = insertvalue { float, i32, float, i32, i32 } %a1, float %arg3, 2
  %a3 = insertvalue { float, i32, float, i32, i32 } %a2, i32 %arg1, 3
  %a4 = insertvalue { float, i32, float, i32, i32 } %a3, i32 %arg2, 4
  ret { float, i32, float, i32, i32 } %a4
}

; GCN-LABEL: {{^}}structure_literal:
; GCN: exp mrt0 v0, v0, v0, v0 done vm

; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: s_mov_b32 s0, 2
; GCN-DAG: s_mov_b32 s1, 3
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-DAG: s_waitcnt expcnt(0)
define amdgpu_vs { { float, i32 }, { i32, <2 x float> } } @structure_literal(ptr addrspace(4) inreg %arg, i32 inreg %arg1, i32 inreg %arg2, float %arg3) #0 {
bb:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %arg3, float %arg3, float %arg3, float %arg3, i1 true, i1 true) #0
  ret { { float, i32 }, { i32, <2 x float> } } { { float, i32 } { float 1.000000e+00, i32 2 }, { i32, <2 x float> } { i32 3, <2 x float> <float 2.000000e+00, float 4.000000e+00> } }
}

; GCN-LABEL: {{^}}ret_return_to_epilog_pseudo_size:
; GCN: codeLenInByte = 0{{$}}
define amdgpu_ps float @ret_return_to_epilog_pseudo_size() #0 {
  ret float undef
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind "InitialPSInputAddr"="0" }
attributes #2 = { nounwind "InitialPSInputAddr"="1" }
attributes #3 = { nounwind "InitialPSInputAddr"="119" }
attributes #4 = { nounwind "InitialPSInputAddr"="418" }
