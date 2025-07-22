; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck --check-prefixes=CHECK,GFX11 %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -verify-machineinstrs < %s | FileCheck --check-prefixes=CHECK,GFX12 %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -mattr=+dynamic-vgpr -verify-machineinstrs < %s | FileCheck --check-prefixes=CHECK,GFX12,DVGPR %s

; CHECK:           .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT: amdpal.pipelines:
; CHECK-NEXT:  - .api:            Vulkan
; CHECK-NEXT:    .compute_registers:
; DVGPR-NEXT:      .dynamic_vgpr_en:   true
; CHECK-NEXT:      .tg_size_en:     true
; CHECK-NEXT:      .tgid_x_en:      false
; CHECK-NEXT:      .tgid_y_en:      false
; CHECK-NEXT:      .tgid_z_en:      false
; CHECK-NEXT:      .tidig_comp_cnt: 0x1
; CHECK-NEXT:    .hardware_stages:
; CHECK-NEXT:      .cs:
; CHECK-NEXT:        .checksum_value: 0x9444d7d0
; CHECK-NEXT:        .debug_mode:     0
; CHECK-NEXT:        .excp_en:        0
; CHECK-NEXT:        .float_mode:     0xc0
; CHECK-NEXT:        .forward_progress: true
; GFX11-NEXT:        .ieee_mode:      true
; CHECK-NEXT:        .image_op:       false
; CHECK-NEXT:        .lds_size:       0x200
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .sgpr_limit:     0x6a
; CHECK-NEXT:        .threadgroup_dimensions:
; CHECK-NEXT:          - 0x1
; CHECK-NEXT:          - 0x400
; CHECK-NEXT:          - 0x1
; CHECK-NEXT:        .trap_present:   false
; CHECK-NEXT:        .user_data_reg_map:
; CHECK-NEXT:          - 0x10000000
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:          - 0xffffffff
; CHECK-NEXT:        .user_sgprs:     0x3
; CHECK-NEXT:        .vgpr_limit:     0x100
; CHECK-NEXT:        .wavefront_size: 0x40
; CHECK-NEXT:        .wgp_mode:       true
; CHECK:    .registers:      {}
; CHECK-NEXT:    .shader_functions:
; CHECK-NEXT:      dynamic_stack:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x22
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x2
; CHECK-NEXT:      dynamic_stack_loop:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x22
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x3
; CHECK-NEXT:      multiple_stack:
; CHECK-NEXT:        .backend_stack_size: 0x24
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x21
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x24
; CHECK-NEXT:        .vgpr_count:     0x3
; CHECK-NEXT:      no_stack:
; CHECK-NEXT:        .backend_stack_size: 0
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x20
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0
; CHECK-NEXT:        .vgpr_count:     0x1
; CHECK-NEXT:      no_stack_call:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x22
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x3
; CHECK-NEXT:      no_stack_extern_call:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0
; GFX11-NEXT:        .sgpr_count:     0x29
; GFX12-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x58
; CHECK-NEXT:      no_stack_extern_call_many_args:
; CHECK-NEXT:        .backend_stack_size: 0x90
; CHECK-NEXT:        .lds_size:       0
; GFX11-NEXT:        .sgpr_count:     0x29
; GFX12-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x90
; CHECK-NEXT:        .vgpr_count:     0x58
; CHECK-NEXT:      no_stack_indirect_call:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0
; GFX11-NEXT:        .sgpr_count:     0x29
; GFX12-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x58
; CHECK-NEXT:      simple_lds:
; CHECK-NEXT:        .backend_stack_size: 0
; CHECK-NEXT:        .lds_size:       0x100
; CHECK-NEXT:        .sgpr_count:     0x20
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0
; CHECK-NEXT:        .vgpr_count:     0x1
; CHECK-NEXT:      simple_lds_recurse:
; CHECK-NEXT:        .backend_stack_size: 0x10
; CHECK-NEXT:        .lds_size:       0x100
; CHECK-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x10
; CHECK-NEXT:        .vgpr_count:     0x29
; CHECK-NEXT:      simple_stack:
; CHECK-NEXT:        .backend_stack_size: 0x14
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x21
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x14
; CHECK-NEXT:        .vgpr_count:     0x2
; CHECK-NEXT:      simple_stack_call:
; CHECK-NEXT:        .backend_stack_size: 0x20
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x22
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x20
; CHECK-NEXT:        .vgpr_count:     0x4
; CHECK-NEXT:      simple_stack_extern_call:
; CHECK-NEXT:        .backend_stack_size: 0x20
; CHECK-NEXT:        .lds_size:       0
; GFX11-NEXT:        .sgpr_count:     0x29
; GFX12-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x20
; CHECK-NEXT:        .vgpr_count:     0x58
; CHECK-NEXT:      simple_stack_indirect_call:
; CHECK-NEXT:        .backend_stack_size: 0x20
; CHECK-NEXT:        .lds_size:       0
; GFX11-NEXT:        .sgpr_count:     0x29
; GFX12-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x20
; CHECK-NEXT:        .vgpr_count:     0x58
; CHECK-NEXT:      simple_stack_recurse:
; CHECK-NEXT:        .backend_stack_size: 0x20
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .sgpr_count:     0x24
; CHECK-NEXT:        .stack_frame_size_in_bytes: 0x20
; CHECK-NEXT:        .vgpr_count:     0x2a
; CHECK:amdpal.version:
; CHECK-NEXT:  - 0x3
; CHECK-NEXT:  - 0
; CHECK-NEXT:...
; CHECK-NEXT:        .end_amdgpu_pal_metadata

declare amdgpu_gfx float @extern_func(float) #0
declare amdgpu_gfx float @extern_func_many_args(<64 x float>) #0

@funcptr = external hidden unnamed_addr addrspace(4) constant ptr, align 4

define amdgpu_gfx float @no_stack(float %arg0) #0 {
  %add = fadd float %arg0, 1.0
  ret float %add
}

define amdgpu_gfx float @simple_stack(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %add = fadd float %arg0, %val
  ret float %add
}

define amdgpu_gfx float @multiple_stack(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %add = fadd float %arg0, %val
  %stack2 = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack2
  %val2 = load volatile float, ptr addrspace(5) %stack2
  %add2 = fadd float %add, %val2
  ret float %add2
}

define amdgpu_gfx float @dynamic_stack(float %arg0) #0 {
bb0:
  %cmp = fcmp ogt float %arg0, 0.0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %add = fadd float %arg0, %val
  br label %bb2

bb2:
  %res = phi float [ 0.0, %bb0 ], [ %add, %bb1 ]
  ret float %res
}

define amdgpu_gfx float @dynamic_stack_loop(float %arg0) #0 {
bb0:
  br label %bb1

bb1:
  %ctr = phi i32 [ 0, %bb0 ], [ %newctr, %bb1 ]
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %add = fadd float %arg0, %val
  %cmp = icmp sgt i32 %ctr, 0
  %newctr = sub i32 %ctr, 1
  br i1 %cmp, label %bb1, label %bb2

bb2:
  ret float %add
}

define amdgpu_gfx float @no_stack_call(float %arg0) #0 {
  %res = call amdgpu_gfx float @simple_stack(float %arg0)
  ret float %res
}

define amdgpu_gfx float @simple_stack_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %res = call amdgpu_gfx float @simple_stack(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

define amdgpu_gfx float @no_stack_extern_call(float %arg0) #0 {
  %res = call amdgpu_gfx float @extern_func(float %arg0)
  ret float %res
}

define amdgpu_gfx float @simple_stack_extern_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %res = call amdgpu_gfx float @extern_func(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

define amdgpu_gfx float @no_stack_extern_call_many_args(<64 x float> %arg0) #0 {
  %res = call amdgpu_gfx float @extern_func_many_args(<64 x float> %arg0)
  ret float %res
}

define amdgpu_gfx float @no_stack_indirect_call(float %arg0) #0 {
  %fptr = load ptr, ptr addrspace(4) @funcptr
  call amdgpu_gfx void %fptr()
  ret float %arg0
}

define amdgpu_gfx float @simple_stack_indirect_call(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %fptr = load ptr, ptr addrspace(4) @funcptr
  call amdgpu_gfx void %fptr()
  %add = fadd float %arg0, %val
  ret float %add
}

define amdgpu_gfx float @simple_stack_recurse(float %arg0) #0 {
  %stack = alloca float, i32 4, align 4, addrspace(5)
  store volatile float 2.0, ptr addrspace(5) %stack
  %val = load volatile float, ptr addrspace(5) %stack
  %res = call amdgpu_gfx float @simple_stack_recurse(float %arg0)
  %add = fadd float %res, %val
  ret float %add
}

@lds = internal addrspace(3) global [64 x float] poison

define amdgpu_gfx float @simple_lds(float %arg0) #0 {
  %val = load float, ptr addrspace(3) @lds
  ret float %val
}

define amdgpu_gfx float @simple_lds_recurse(float %arg0) #0 {
  %val = load float, ptr addrspace(3) @lds
  %res = call amdgpu_gfx float @simple_lds_recurse(float %val)
  ret float %res
}

attributes #0 = { nounwind }

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\00"}
!1 = !{i32 7}
