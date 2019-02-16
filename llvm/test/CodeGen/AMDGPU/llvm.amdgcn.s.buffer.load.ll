;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK-LABEL: {{^}}s_buffer_load_imm:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0x4
define amdgpu_ps void @s_buffer_load_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 4, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_load_index:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_load_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast i32 %load to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %bitcast, float undef, float undef, float undef, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_loadx2_imm:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x40
define amdgpu_ps void @s_buffer_loadx2_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 64, i32 0)
  %bitcast = bitcast <2 x i32> %load to <2 x float>
  %x = extractelement <2 x float> %bitcast, i32 0
  %y = extractelement <2 x float> %bitcast, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_loadx2_index:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_loadx2_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <2 x i32> %load to <2 x float>
  %x = extractelement <2 x float> %bitcast, i32 0
  %y = extractelement <2 x float> %bitcast, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_loadx4_imm:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0xc8
define amdgpu_ps void @s_buffer_loadx4_imm(<4 x i32> inreg %desc) {
main_body:
  %load = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 200, i32 0)
  %bitcast = bitcast <4 x i32> %load to <4 x float>
  %x = extractelement <4 x float> %bitcast, i32 0
  %y = extractelement <4 x float> %bitcast, i32 1
  %z = extractelement <4 x float> %bitcast, i32 2
  %w = extractelement <4 x float> %bitcast, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_loadx4_index:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}}
define amdgpu_ps void @s_buffer_loadx4_index(<4 x i32> inreg %desc, i32 inreg %index) {
main_body:
  %load = call <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32> %desc, i32 %index, i32 0)
  %bitcast = bitcast <4 x i32> %load to <4 x float>
  %x = extractelement <4 x float> %bitcast, i32 0
  %y = extractelement <4 x float> %bitcast, i32 1
  %z = extractelement <4 x float> %bitcast, i32 2
  %w = extractelement <4 x float> %bitcast, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_load_imm_mergex2:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x4
define amdgpu_ps void @s_buffer_load_imm_mergex2(<4 x i32> inreg %desc) {
main_body:
  %load0 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 4, i32 0)
  %load1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 8, i32 0)
  %x = bitcast i32 %load0 to float
  %y = bitcast i32 %load1 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float undef, float undef, i1 true, i1 true)
  ret void
}

;CHECK-LABEL: {{^}}s_buffer_load_imm_mergex4:
;CHECK-NOT: s_waitcnt;
;CHECK: s_buffer_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0x8
define amdgpu_ps void @s_buffer_load_imm_mergex4(<4 x i32> inreg %desc) {
main_body:
  %load0 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 8, i32 0)
  %load1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 12, i32 0)
  %load2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 16, i32 0)
  %load3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %desc, i32 20, i32 0)
  %x = bitcast i32 %load0 to float
  %y = bitcast i32 %load1 to float
  %z = bitcast i32 %load2 to float
  %w = bitcast i32 %load3 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %x, float %y, float %z, float %w, i1 true, i1 true)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32)
declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32)
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i32)
