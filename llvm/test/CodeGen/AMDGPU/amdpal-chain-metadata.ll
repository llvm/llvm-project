; RUN: llc -mtriple=amdgpu11.00--amdpal < %s | FileCheck --check-prefixes=CHECK,EXTSTACK0 %s
; RUN: llc -mtriple=amdgpu12.00--amdpal -amdgpu-assume-external-call-stack-size=48 < %s | FileCheck --check-prefixes=CHECK,EXTSTACK48 %s

; CHECK-LABEL: {{^}}amdgpu_cs_chain_func:
; EXTSTACK0:     .set .Lamdgpu_cs_chain_func.private_seg_size, 16{{$}}
; EXTSTACK48:    .set .Lamdgpu_cs_chain_func.private_seg_size, 16+max(48){{$}}

; CHECK-LABEL: {{^}}chain_callee_smaller_stack:
; CHECK:         .set .Lchain_callee_smaller_stack.private_seg_size, 36{{$}}

; CHECK-LABEL: {{^}}chain_callee_larger_stack:
; CHECK:         .set .Lchain_callee_larger_stack.private_seg_size, 8{{$}}

; CHECK-LABEL: {{^}}chain_and_wwf:
; CHECK:         .set .Lchain_and_wwf.private_seg_size, 80+max(.Lwwf_callee_alloca.private_seg_size){{$}}

; CHECK-LABEL: {{^}}wwf_callee_alloca:
; CHECK:         .set .Lwwf_callee_alloca.private_seg_size, 28{{$}}

; CHECK-LABEL: {{^}}gfx_alloca:
; CHECK:         .set .Lgfx_alloca.private_seg_size, 28{{$}}

; CHECK-LABEL: {{^}}chain_and_gfx:
; CHECK:         .set .Lchain_and_gfx.private_seg_size, 32+max(.Lgfx_alloca.private_seg_size){{$}}

; CHECK-LABEL: {{^}}cs_to_chain_and_wwf:
; CHECK:         .set .Lcs_to_chain_and_wwf.private_seg_size, 12{{$}}

; Indirect chain calls don't contribute to the stack size.
; CHECK-LABEL: {{^}}chain_indirect_chain_and_wwf:
; CHECK:         .set .Lchain_indirect_chain_and_wwf.private_seg_size, 64+max(.Lwwf_callee_alloca.private_seg_size){{$}}
; CHECK:         .set .Lchain_indirect_chain_and_wwf.has_indirect_call, 1{{$}}

; A non-chain indirect call does contribute the assumed external stack size.
; CHECK-LABEL: {{^}}chain_indirect_gfx_and_chain:
; EXTSTACK0:     .set .Lchain_indirect_gfx_and_chain.private_seg_size, 32{{$}}
; EXTSTACK48:    .set .Lchain_indirect_gfx_and_chain.private_seg_size, 32+max(48){{$}}
; CHECK:         .set .Lchain_indirect_gfx_and_chain.has_indirect_call, 1{{$}}

; CHECK-LABEL: {{^}}chain_indirect_gfx_and_wwf:
; EXTSTACK0:     .set .Lchain_indirect_gfx_and_wwf.private_seg_size, 64+max(.Lwwf_callee_alloca.private_seg_size){{$}}
; EXTSTACK48:    .set .Lchain_indirect_gfx_and_wwf.private_seg_size, 64+max(48, .Lwwf_callee_alloca.private_seg_size){{$}}
; CHECK:         .set .Lchain_indirect_gfx_and_wwf.has_indirect_call, 1{{$}}

; CHECK-LABEL: {{^}}gfx_stack_only:
; CHECK:         .set .Lgfx_stack_only.private_seg_size, 20{{$}}

; CHECK-LABEL: {{^}}wwf_no_local_stack_to_gfx_with_stack:
; EXTSTACK0:     .set .Lwwf_no_local_stack_to_gfx_with_stack.private_seg_size, 16+max(.Lgfx_stack_only.private_seg_size){{$}}
; EXTSTACK48:    .set .Lwwf_no_local_stack_to_gfx_with_stack.private_seg_size, 16+max(48, .Lgfx_stack_only.private_seg_size){{$}}

; CHECK-LABEL: {{^}}chain_own_and_callee_stack:
; EXTSTACK0:     .set .Lchain_own_and_callee_stack.private_seg_size, 32+max(.Lwwf_no_local_stack_to_gfx_with_stack.private_seg_size){{$}}
; EXTSTACK48:    .set .Lchain_own_and_callee_stack.private_seg_size, 32+max(48, .Lwwf_no_local_stack_to_gfx_with_stack.private_seg_size){{$}}

; CHECK-LABEL: {{^}}chain_realign_stack:
; EXTSTACK0:     .set .Lchain_realign_stack.private_seg_size, 256+max(.Lgfx_stack_only.private_seg_size){{$}}
; EXTSTACK48:    .set .Lchain_realign_stack.private_seg_size, 256+max(48, .Lgfx_stack_only.private_seg_size){{$}}

; CHECK-LABEL: .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT: amdpal.pipelines:
; CHECK-NEXT:   - .api:            Vulkan
; CHECK-LABEL:         .hardware_stages:
; CHECK:      .cs:
; CHECK:             .entry_point_symbol: cs_to_chain_and_wwf
; CHECK:             .scratch_memory_size: 0x10{{$}}
; CHECK:         .shader_functions:
; CHECK-LABEL:      amdgpu_cs_chain_func:
; EXTSTACK0:         .backend_stack_size: 0x10{{$}}
; EXTSTACK48:        .backend_stack_size: 0x40{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x10{{$}}
; CHECK-LABEL:      chain_and_gfx:
; CHECK:             .backend_stack_size: 0x3c{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x20{{$}}
; CHECK-LABEL:      chain_and_wwf:
; CHECK:             .backend_stack_size: 0x6c{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x50{{$}}
; CHECK-LABEL:      chain_callee_larger_stack:
; CHECK:             .backend_stack_size: 0x8{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x8{{$}}
; CHECK-LABEL:      chain_callee_smaller_stack:
; CHECK:             .backend_stack_size: 0x24{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x24{{$}}
; CHECK-LABEL:      chain_indirect_chain_and_wwf:
; CHECK:             .backend_stack_size: 0x5c{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x40{{$}}
; CHECK-LABEL:      chain_indirect_gfx_and_chain:
; EXTSTACK0:         .backend_stack_size: 0x20{{$}}
; EXTSTACK48:        .backend_stack_size: 0x50{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x20{{$}}
; CHECK-LABEL:      chain_indirect_gfx_and_wwf:
; EXTSTACK0:         .backend_stack_size: 0x5c{{$}}
; EXTSTACK48:        .backend_stack_size: 0x70{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x40{{$}}
; CHECK-LABEL:      chain_own_and_callee_stack:
; EXTSTACK0:         .backend_stack_size: 0x44{{$}}
; EXTSTACK48:        .backend_stack_size: 0x60{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x20{{$}}
; CHECK-LABEL:      chain_realign_stack:
; EXTSTACK0:         .backend_stack_size: 0x114{{$}}
; EXTSTACK48:        .backend_stack_size: 0x130{{$}}
; FIXME: stack_frame_size_in_bytes would do well to include the realignment padding too...
; CHECK:             .stack_frame_size_in_bytes: 0x80{{$}}
; CHECK-LABEL:      gfx_alloca:
; CHECK:             .backend_stack_size: 0x1c{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x1c{{$}}
; CHECK-LABEL:      gfx_stack_only:
; CHECK:             .backend_stack_size: 0x14{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x14{{$}}
; CHECK:amdpal.version:
; CHECK-NEXT:  - 0x3
; CHECK-NEXT:  - 0
; CHECK-NEXT:...
; CHECK-NEXT:        .end_amdgpu_pal_metadata

define amdgpu_cs_chain void @amdgpu_cs_chain_func(<40 x i32> %should_spill) {
.entry:
  %v = alloca [3 x i32], addrspace(5)
  store i32 42, ptr addrspace(5) %v
  call amdgpu_gfx void @use(<40 x i32> %should_spill, ptr addrspace(5) %v)
  ret void
}

define amdgpu_cs_chain void @chain_callee_smaller_stack(<40 x i32> %v) {
.entry:
  %a = alloca [8 x i32], addrspace(5)
  store volatile i32 1, ptr addrspace(5) %a
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @amdgpu_cs_chain_func, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_cs_chain void @chain_callee_larger_stack(<40 x i32> %v) {
.entry:
  %a = alloca [1 x i32], addrspace(5)
  store volatile i32 1, ptr addrspace(5) %a
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @amdgpu_cs_chain_func, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_cs_chain void @chain_and_wwf(<40 x i32> %v) {
.entry:
  %a = alloca [16 x i32], addrspace(5)
  store i32 2, ptr addrspace(5) %a
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_callee_alloca, ptr addrspace(5) %a)
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @amdgpu_cs_chain_func, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_gfx_whole_wave void @wwf_callee_alloca(i1 %active, ptr addrspace(5) %p) norecurse {
  %a = alloca [4 x i32], addrspace(5)
  store volatile i32 3, ptr addrspace(5) %a
  store i32 4, ptr addrspace(5) %p
  ret void
}

define amdgpu_cs_chain void @chain_and_gfx(<40 x i32> %v) {
.entry:
  %a = alloca [4 x i32], addrspace(5)
  store i32 6, ptr addrspace(5) %a
  call amdgpu_gfx void @gfx_alloca(ptr addrspace(5) %a)
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @amdgpu_cs_chain_func, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_gfx void @gfx_alloca(ptr addrspace(5) %p) norecurse {
  %a = alloca [6 x i32], addrspace(5)
  store volatile i32 7, ptr addrspace(5) %a
  store i32 8, ptr addrspace(5) %p
  ret void
}

define amdgpu_cs void @cs_to_chain_and_wwf(<40 x i32> %v) {
.entry:
  %a = alloca [2 x i32], addrspace(5)
  store volatile i32 9, ptr addrspace(5) %a
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @chain_and_wwf, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_cs_chain void @chain_indirect_chain_and_wwf(ptr inreg %callee, <40 x i32> %v) {
.entry:
  %a = alloca [12 x i32], addrspace(5)
  store i32 10, ptr addrspace(5) %a
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_callee_alloca, ptr addrspace(5) %a)
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr %callee, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_cs_chain void @chain_indirect_gfx_and_chain(ptr inreg %fp, <40 x i32> %v) {
.entry:
  %a = alloca [5 x i32], addrspace(5)
  store i32 11, ptr addrspace(5) %a
  call amdgpu_gfx void %fp(ptr addrspace(5) %a)
  call void (ptr, i32, {}, <40 x i32>, i32, ...) @llvm.amdgcn.cs.chain.p0.i32.s.v40i32(ptr @amdgpu_cs_chain_func, i32 -1, {} inreg poison, <40 x i32> %v, i32 0)
  unreachable
}

define amdgpu_cs_chain void @chain_indirect_gfx_and_wwf(ptr inreg %callee) {
.entry:
  %a = alloca [12 x i32], addrspace(5)
  store i32 11, ptr addrspace(5) %a
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_callee_alloca, ptr addrspace(5) %a)
  call amdgpu_gfx void %callee()
  ret void
}

define amdgpu_cs_chain_preserve void @amdgpu_cs_chain_preserve_func() {
.entry:
  ret void
}

define amdgpu_gfx void @gfx_stack_only() {
  %a = alloca [4 x i32], addrspace(5)
  store volatile i32 12, ptr addrspace(5) %a
  ret void
}

define amdgpu_gfx_whole_wave void @wwf_no_local_stack_to_gfx_with_stack(i1 %active) {
  call amdgpu_gfx void @gfx_stack_only()
  ret void
}

define amdgpu_cs_chain void @chain_own_and_callee_stack() {
.entry:
  %a = alloca [4 x i32], addrspace(5)
  store volatile i32 13, ptr addrspace(5) %a
  call void (ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_no_local_stack_to_gfx_with_stack)
  ret void
}

define amdgpu_cs_chain void @chain_realign_stack() {
.entry:
  %a = alloca i32, align 128, addrspace(5)
  store volatile i32 13, ptr addrspace(5) %a
  call amdgpu_gfx void @gfx_stack_only()
  ret void
}

declare amdgpu_gfx void @use(...)

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\00"}
!1 = !{i32 7}
