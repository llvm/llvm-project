; RUN: llc -march=amdgcn -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN  %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

; sbuffer merging pass tests

; GCN-LABEL: {{^}}dwordx2:
; GCN: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
define amdgpu_kernel void @dwordx2(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 4, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.1.float, float %s.buffer.2.float, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}dwordx4:
; GCN: s_buffer_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
define amdgpu_kernel void @dwordx4(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 4, i32 0)
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 8, i32 0)
  %s.buffer.4 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 12, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast i32 %s.buffer.4 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, float %s.buffer.4.float, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}dwordx8:
; GCN: s_buffer_load_dwordx8 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
define amdgpu_kernel void @dwordx8(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 4, i32 0)
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 8, i32 0)
  %s.buffer.4 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 12, i32 0)
  %s.buffer.5 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 16, i32 0)
  %s.buffer.6 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 20, i32 0)
  %s.buffer.7 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 24, i32 0)
  %s.buffer.8 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 28, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast i32 %s.buffer.4 to float
  %s.buffer.5.float = bitcast i32 %s.buffer.5 to float
  %s.buffer.6.float = bitcast i32 %s.buffer.6 to float
  %s.buffer.7.float = bitcast i32 %s.buffer.7 to float
  %s.buffer.8.float = bitcast i32 %s.buffer.8 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, float %s.buffer.4.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.5.float, float %s.buffer.6.float, float %s.buffer.7.float, float %s.buffer.8.float, i1 true, i1 true)
  ret void
}

; GCN-LABEL: {{^}}dwordx16:
; GCN: s_buffer_load_dwordx16 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
define amdgpu_kernel void @dwordx16(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 4, i32 0)
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 8, i32 0)
  %s.buffer.4 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 12, i32 0)
  %s.buffer.5 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 16, i32 0)
  %s.buffer.6 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 20, i32 0)
  %s.buffer.7 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 24, i32 0)
  %s.buffer.8 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 28, i32 0)
  %s.buffer.9 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc,  i32 32, i32 0)
  %s.buffer.10 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 36, i32 0)
  %s.buffer.11 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 40, i32 0)
  %s.buffer.12 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 44, i32 0)
  %s.buffer.13 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 48, i32 0)
  %s.buffer.14 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 52, i32 0)
  %s.buffer.15 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 56, i32 0)
  %s.buffer.16 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 60, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast i32 %s.buffer.4 to float
  %s.buffer.5.float = bitcast i32 %s.buffer.5 to float
  %s.buffer.6.float = bitcast i32 %s.buffer.6 to float
  %s.buffer.7.float = bitcast i32 %s.buffer.7 to float
  %s.buffer.8.float = bitcast i32 %s.buffer.8 to float
  %s.buffer.9.float = bitcast i32 %s.buffer.9 to float
  %s.buffer.10.float = bitcast i32 %s.buffer.10 to float
  %s.buffer.11.float = bitcast i32 %s.buffer.11 to float
  %s.buffer.12.float = bitcast i32 %s.buffer.12 to float
  %s.buffer.13.float = bitcast i32 %s.buffer.13 to float
  %s.buffer.14.float = bitcast i32 %s.buffer.14 to float
  %s.buffer.15.float = bitcast i32 %s.buffer.15 to float
  %s.buffer.16.float = bitcast i32 %s.buffer.16 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, float %s.buffer.4.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.5.float, float %s.buffer.6.float, float %s.buffer.7.float, float %s.buffer.8.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.9.float, float %s.buffer.10.float, float %s.buffer.11.float, float %s.buffer.12.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.13.float, float %s.buffer.14.float, float %s.buffer.15.float, float %s.buffer.16.float, i1 true, i1 true)
  ret void
}

; Commented out the checks as this currently merges to 2-4-2
; Check for out-of-order
; XXGCN-LABEL: {{^}}dwordx8_ooo:
; XXGCN: s_buffer_load_dwordx8 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
define amdgpu_kernel void @dwordx8_ooo(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32  0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 20, i32 0)
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32  8, i32 0)
  %s.buffer.4 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 16, i32 0)
  %s.buffer.5 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 12, i32 0)
  %s.buffer.6 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 24, i32 0)
  %s.buffer.7 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32  4, i32 0)
  %s.buffer.8 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 28, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast i32 %s.buffer.4 to float
  %s.buffer.5.float = bitcast i32 %s.buffer.5 to float
  %s.buffer.6.float = bitcast i32 %s.buffer.6 to float
  %s.buffer.7.float = bitcast i32 %s.buffer.7 to float
  %s.buffer.8.float = bitcast i32 %s.buffer.8 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, float %s.buffer.4.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.5.float, float %s.buffer.6.float, float %s.buffer.7.float, float %s.buffer.8.float, i1 true, i1 true)
  ret void
}

; Check for distinct loads intermingled
; GCN-LABEL: {{^}}dwordx8_interm:
; VI-DAG: s_buffer_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x14
; SI-DAG: s_buffer_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x5
; CI-DAG: s_buffer_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x5
; GCN-DAG: s_buffer_load_dwordx4 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
; VI-DAG: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x10
; SI-DAG: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x4
; CI-DAG: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x4
define amdgpu_kernel void @dwordx8_interm(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %s.buffer.2 = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %rsrc, i32 20, i32 0)
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 4, i32 0)
  %s.buffer.4 = call <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32> %rsrc, i32 28, i32 0)
  %s.buffer.5 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 8, i32 0)
  %s.buffer.6 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 12, i32 0)
  %s.buffer.7 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 16, i32 0)
  %s.buffer.8 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 20, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast <2 x i32> %s.buffer.2 to <2 x float>
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast <2 x i32> %s.buffer.4 to <2 x float>
  %s.buffer.5.float = bitcast i32 %s.buffer.5 to float
  %s.buffer.6.float = bitcast i32 %s.buffer.6 to float
  %s.buffer.7.float = bitcast i32 %s.buffer.7 to float
  %s.buffer.8.float = bitcast i32 %s.buffer.8 to float
  %s.buffer.9.float = extractelement <2 x float> %s.buffer.2.float, i32 0
  %s.buffer.10.float = extractelement <2 x float> %s.buffer.4.float, i32 1
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.9.float, float %s.buffer.3.float, float %s.buffer.10.float, i1 true, i1 true)
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.5.float, float %s.buffer.6.float, float %s.buffer.7.float, float %s.buffer.8.float, i1 true, i1 true)
  ret void
}

; Check no merging across BB
; GCN-LABEL: {{^}}dwordx4_across_bb:
; GCN: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x0 
; VI: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x8 
; SI: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x2
; CI: s_buffer_load_dwordx2 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x2
define amdgpu_kernel void @dwordx4_across_bb(<4 x i32> addrspace(4)* inreg %in) #0 {
entry:
  %rsrc = load <4 x i32>, <4 x i32> addrspace(4)* %in
  %s.buffer.1 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 0, i32 0)
  %s.buffer.2 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 4, i32 0)
  br label %bb2
bb2:
  %s.buffer.3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 8, i32 0)
  %s.buffer.4 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %rsrc, i32 12, i32 0)
  %s.buffer.1.float = bitcast i32 %s.buffer.1 to float
  %s.buffer.2.float = bitcast i32 %s.buffer.2 to float
  %s.buffer.3.float = bitcast i32 %s.buffer.3 to float
  %s.buffer.4.float = bitcast i32 %s.buffer.4 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %s.buffer.1.float, float %s.buffer.2.float, float %s.buffer.3.float, float %s.buffer.4.float, i1 true, i1 true)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32)
declare <2 x i32> @llvm.amdgcn.s.buffer.load.v2i32(<4 x i32>, i32, i32)

attributes #0 = { nounwind }
