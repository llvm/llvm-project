; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

attributes #0 = { "ShaderType"="1" }

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

; GCN-LABEL: {{^}}vgpr:
; GCN: v_mov_b32_e32 v1, v0
; GCN-DAG: v_add_f32_e32 v0, 1.0, v1
; GCN-DAG: exp 15, 0, 1, 1, 1, v1, v1, v1, v1
; GCN: s_waitcnt expcnt(0)
; GCN-NOT: s_endpgm
define {float, float} @vgpr([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %3, float %3, float %3, float %3)
  %x = fadd float %3, 1.0
  %a = insertvalue {float, float} undef, float %x, 0
  %b = insertvalue {float, float} %a, float %3, 1
  ret {float, float} %b
}

; GCN-LABEL: {{^}}vgpr_literal:
; GCN: v_mov_b32_e32 v4, v0
; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-DAG: v_mov_b32_e32 v3, -1.0
; GCN: exp 15, 0, 1, 1, 1, v4, v4, v4, v4
; GCN: s_waitcnt expcnt(0)
; GCN-NOT: s_endpgm
define {float, float, float, float} @vgpr_literal([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %3, float %3, float %3, float %3)
  ret {float, float, float, float} {float 1.0, float 2.0, float 4.0, float -1.0}
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
attributes #1 = { "ShaderType"="0" "InitialPSInputAddr"="0" }
define {float, float, float, float, float} @vgpr_ps_addr0([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #1 {
  %i0 = extractelement <2 x i32> %4, i32 0
  %i1 = extractelement <2 x i32> %4, i32 1
  %i2 = extractelement <2 x i32> %7, i32 0
  %i3 = extractelement <2 x i32> %8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue {float, float, float, float, float} undef, float %f0, 0
  %r1 = insertvalue {float, float, float, float, float} %r0, float %f1, 1
  %r2 = insertvalue {float, float, float, float, float} %r1, float %f2, 2
  %r3 = insertvalue {float, float, float, float, float} %r2, float %f3, 3
  %r4 = insertvalue {float, float, float, float, float} %r3, float %12, 4
  ret {float, float, float, float, float} %r4
}


; GCN: .long 165580
; GCN-NEXT: .long 1
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 1
; GCN-LABEL: {{^}}ps_input_ena_no_inputs:
; GCN: v_mov_b32_e32 v0, 1.0
; GCN-NOT: s_endpgm
define float @ps_input_ena_no_inputs([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #1 {
  ret float 1.0
}


; GCN: .long 165580
; GCN-NEXT: .long 2081
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 2081
; GCN-LABEL: {{^}}ps_input_ena_pos_w:
; GCN-DAG: v_mov_b32_e32 v0, v4
; GCN-DAG: v_mov_b32_e32 v1, v2
; GCN: v_mov_b32_e32 v2, v3
; GCN-NOT: s_endpgm
define {float, <2 x float>} @ps_input_ena_pos_w([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #1 {
  %f = bitcast <2 x i32> %8 to <2 x float>
  %s = insertvalue {float, <2 x float>} undef, float %14, 0
  %s1 = insertvalue {float, <2 x float>} %s, <2 x float> %f, 1
  ret {float, <2 x float>} %s1
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
attributes #2 = { "ShaderType"="0" "InitialPSInputAddr"="1" }
define {float, float, float, float, float} @vgpr_ps_addr1([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #2 {
  %i0 = extractelement <2 x i32> %4, i32 0
  %i1 = extractelement <2 x i32> %4, i32 1
  %i2 = extractelement <2 x i32> %7, i32 0
  %i3 = extractelement <2 x i32> %8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue {float, float, float, float, float} undef, float %f0, 0
  %r1 = insertvalue {float, float, float, float, float} %r0, float %f1, 1
  %r2 = insertvalue {float, float, float, float, float} %r1, float %f2, 2
  %r3 = insertvalue {float, float, float, float, float} %r2, float %f3, 3
  %r4 = insertvalue {float, float, float, float, float} %r3, float %12, 4
  ret {float, float, float, float, float} %r4
}


; GCN: .long 165580
; GCN-NEXT: .long 562
; GCN-NEXT: .long 165584
; GCN-NEXT: .long 631
; GCN-LABEL: {{^}}vgpr_ps_addr119:
; GCN-DAG: v_mov_b32_e32 v0, v2
; GCN-DAG: v_mov_b32_e32 v1, v3
; GCN: v_mov_b32_e32 v2, v6
; GCN: v_mov_b32_e32 v3, v8
; GCN: v_mov_b32_e32 v4, v12
; GCN-NOT: s_endpgm
attributes #3 = { "ShaderType"="0" "InitialPSInputAddr"="119" }
define {float, float, float, float, float} @vgpr_ps_addr119([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #3 {
  %i0 = extractelement <2 x i32> %4, i32 0
  %i1 = extractelement <2 x i32> %4, i32 1
  %i2 = extractelement <2 x i32> %7, i32 0
  %i3 = extractelement <2 x i32> %8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue {float, float, float, float, float} undef, float %f0, 0
  %r1 = insertvalue {float, float, float, float, float} %r0, float %f1, 1
  %r2 = insertvalue {float, float, float, float, float} %r1, float %f2, 2
  %r3 = insertvalue {float, float, float, float, float} %r2, float %f3, 3
  %r4 = insertvalue {float, float, float, float, float} %r3, float %12, 4
  ret {float, float, float, float, float} %r4
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
attributes #4 = { "ShaderType"="0" "InitialPSInputAddr"="418" }
define {float, float, float, float, float} @vgpr_ps_addr418([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, float, float, float, float) #4 {
  %i0 = extractelement <2 x i32> %4, i32 0
  %i1 = extractelement <2 x i32> %4, i32 1
  %i2 = extractelement <2 x i32> %7, i32 0
  %i3 = extractelement <2 x i32> %8, i32 0
  %f0 = bitcast i32 %i0 to float
  %f1 = bitcast i32 %i1 to float
  %f2 = bitcast i32 %i2 to float
  %f3 = bitcast i32 %i3 to float
  %r0 = insertvalue {float, float, float, float, float} undef, float %f0, 0
  %r1 = insertvalue {float, float, float, float, float} %r0, float %f1, 1
  %r2 = insertvalue {float, float, float, float, float} %r1, float %f2, 2
  %r3 = insertvalue {float, float, float, float, float} %r2, float %f3, 3
  %r4 = insertvalue {float, float, float, float, float} %r3, float %12, 4
  ret {float, float, float, float, float} %r4
}


; GCN-LABEL: {{^}}sgpr:
; GCN: s_add_i32 s0, s3, 2
; GCN: s_mov_b32 s2, s3
; GCN-NOT: s_endpgm
define {i32, i32, i32} @sgpr([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  %x = add i32 %2, 2
  %a = insertvalue {i32, i32, i32} undef, i32 %x, 0
  %b = insertvalue {i32, i32, i32} %a, i32 %1, 1
  %c = insertvalue {i32, i32, i32} %a, i32 %2, 2
  ret {i32, i32, i32} %c
}


; GCN-LABEL: {{^}}sgpr_literal:
; GCN: s_mov_b32 s0, 5
; GCN-NOT: s_mov_b32 s0, s0
; GCN-DAG: s_mov_b32 s1, 6
; GCN-DAG: s_mov_b32 s2, 7
; GCN-DAG: s_mov_b32 s3, 8
; GCN-NOT: s_endpgm
define {i32, i32, i32, i32} @sgpr_literal([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  %x = add i32 %2, 2
  ret {i32, i32, i32, i32} {i32 5, i32 6, i32 7, i32 8}
}


; GCN-LABEL: {{^}}both:
; GCN: v_mov_b32_e32 v1, v0
; GCN-DAG: exp 15, 0, 1, 1, 1, v1, v1, v1, v1
; GCN-DAG: v_add_f32_e32 v0, 1.0, v1
; GCN-DAG: s_add_i32 s0, s3, 2
; GCN-DAG: s_mov_b32 s1, s2
; GCN: s_mov_b32 s2, s3
; GCN: s_waitcnt expcnt(0)
; GCN-NOT: s_endpgm
define {float, i32, float, i32, i32} @both([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %3, float %3, float %3, float %3)
  %v = fadd float %3, 1.0
  %s = add i32 %2, 2
  %a0 = insertvalue {float, i32, float, i32, i32} undef, float %v, 0
  %a1 = insertvalue {float, i32, float, i32, i32} %a0, i32 %s, 1
  %a2 = insertvalue {float, i32, float, i32, i32} %a1, float %3, 2
  %a3 = insertvalue {float, i32, float, i32, i32} %a2, i32 %1, 3
  %a4 = insertvalue {float, i32, float, i32, i32} %a3, i32 %2, 4
  ret {float, i32, float, i32, i32} %a4
}


; GCN-LABEL: {{^}}structure_literal:
; GCN: v_mov_b32_e32 v3, v0
; GCN-DAG: v_mov_b32_e32 v0, 1.0
; GCN-DAG: s_mov_b32 s0, 2
; GCN-DAG: s_mov_b32 s1, 3
; GCN-DAG: v_mov_b32_e32 v1, 2.0
; GCN-DAG: v_mov_b32_e32 v2, 4.0
; GCN-DAG: exp 15, 0, 1, 1, 1, v3, v3, v3, v3
define {{float, i32}, {i32, <2 x float>}} @structure_literal([9 x <16 x i8>] addrspace(2)* byval, i32 inreg, i32 inreg, float) #0 {
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %3, float %3, float %3, float %3)
  ret {{float, i32}, {i32, <2 x float>}} {{float, i32} {float 1.0, i32 2}, {i32, <2 x float>} {i32 3, <2 x float> <float 2.0, float 4.0>}}
}
