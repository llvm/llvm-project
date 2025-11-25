; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -amdgpu-enable-machine-level-inliner  < %s | FileCheck %s --check-prefixes=CHECK

; CHECK-LABEL: {{^}}cs_shader:
; CHECK: .set cs_shader.num_vgpr, 67{{$}}
; CHECK: .set cs_shader.numbered_sgpr, 67{{$}}
; CHECK: .set cs_shader.private_seg_size, 2064{{$}}
; CHECK: .set cs_shader.has_dyn_sized_stack, 0{{$}}
; CHECK: .set cs_shader.has_recursion, 0{{$}}
; CHECK: .set cs_shader.has_indirect_call, 0{{$}}
; CHECK-LABEL: {{^}}ps_shader:
; CHECK: .set ps_shader.num_vgpr, 1{{$}}
; CHECK: .set ps_shader.numbered_sgpr, 34{{$}}
; CHECK: .set ps_shader.private_seg_size, 16{{$}}
; CHECK: .set ps_shader.has_dyn_sized_stack, 1{{$}}
; CHECK: .set ps_shader.has_recursion, 0{{$}}
; CHECK: .set ps_shader.has_indirect_call, 0{{$}}
; CHECK-LABEL: {{^}}gs_shader:
; CHECK: .set gs_shader.num_vgpr, max(248, amdgpu.max_num_vgpr)
; CHECK: .set gs_shader.numbered_sgpr, max(96, amdgpu.max_num_sgpr)
; CHECK: .set gs_shader.private_seg_size, 592{{$}}
; CHECK: .set gs_shader.has_dyn_sized_stack, 1{{$}}
; CHECK: .set gs_shader.has_recursion, 1{{$}}
; CHECK: .set gs_shader.has_indirect_call, 1{{$}}
; CHECK-LABEL:           .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT: amdpal.pipelines:
; CHECK-NEXT:   - .api:            Vulkan
; CHECK-NEXT:     .compute_registers:
; CHECK-NEXT:       .tg_size_en:     true
; CHECK-NEXT:       .tgid_x_en:      false
; CHECK-NEXT:       .tgid_y_en:      false
; CHECK-NEXT:       .tgid_z_en:      false
; CHECK-NEXT:       .tidig_comp_cnt: 0x1{{$}}
; CHECK-NEXT:     .graphics_registers:
; CHECK-NEXT:      .ps_extra_lds_size: 0{{$}}
; CHECK-NEXT:      .spi_ps_input_addr:
; CHECK-NEXT:        .ancillary_ena:  false
; CHECK-NEXT:        .front_face_ena: false
; CHECK-NEXT:        .line_stipple_tex_ena: false
; CHECK-NEXT:        .linear_center_ena: false
; CHECK-NEXT:        .linear_centroid_ena: false
; CHECK-NEXT:        .linear_sample_ena: false
; CHECK-NEXT:        .persp_center_ena: false
; CHECK-NEXT:        .persp_centroid_ena: false
; CHECK-NEXT:        .persp_pull_model_ena: false
; CHECK-NEXT:        .persp_sample_ena: true
; CHECK-NEXT:        .pos_fixed_pt_ena: false
; CHECK-NEXT:        .pos_w_float_ena: false
; CHECK-NEXT:        .pos_x_float_ena: false
; CHECK-NEXT:        .pos_y_float_ena: false
; CHECK-NEXT:        .pos_z_float_ena: false
; CHECK-NEXT:        .sample_coverage_ena: false
; CHECK-NEXT:      .spi_ps_input_ena:
; CHECK-NEXT:        .ancillary_ena:  false
; CHECK-NEXT:        .front_face_ena: false
; CHECK-NEXT:        .line_stipple_tex_ena: false
; CHECK-NEXT:        .linear_center_ena: false
; CHECK-NEXT:        .linear_centroid_ena: false
; CHECK-NEXT:        .linear_sample_ena: false
; CHECK-NEXT:        .persp_center_ena: false
; CHECK-NEXT:        .persp_centroid_ena: false
; CHECK-NEXT:        .persp_pull_model_ena: false
; CHECK-NEXT:        .persp_sample_ena: true
; CHECK-NEXT:        .pos_fixed_pt_ena: false
; CHECK-NEXT:        .pos_w_float_ena: false
; CHECK-NEXT:        .pos_x_float_ena: false
; CHECK-NEXT:        .pos_y_float_ena: false
; CHECK-NEXT:        .pos_z_float_ena: false
; CHECK-NEXT:        .sample_coverage_ena: false
; CHECK-NEXT:    .hardware_stages:
; CHECK-NEXT:      .cs:
; CHECK-NEXT:        .checksum_value: 0x9444d7d0
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point_symbol:    cs_shader
; CHECK-NEXT:        .excp_en:        0{{$}}
; CHECK-NEXT:        .float_mode:     0xc0{{$}}
; CHECK-NEXT:        .forward_progress: true
; CHECK-NEXT:        .image_op:       false
; CHECK-NEXT:        .lds_size:       0{{$}}
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     true
; CHECK-NEXT:        .scratch_memory_size: 0x810{{$}}
; CHECK-NEXT:        .sgpr_count:     0x45{{$}}
; CHECK-NEXT:        .sgpr_limit:     0x6a{{$}}
; CHECK-NEXT:        .threadgroup_dimensions:
; CHECK-NEXT:          - 0x1{{$}}
; CHECK-NEXT:          - 0x400{{$}}
; CHECK-NEXT:          - 0x1{{$}}
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
; CHECK-NEXT:        .user_sgprs:     0x3{{$}}
; CHECK-NEXT:        .vgpr_count:     0x43{{$}}
; CHECK-NEXT:        .vgpr_limit:     0x100{{$}}
; CHECK-NEXT:        .wavefront_size: 0x40{{$}}
; CHECK-NEXT:        .wgp_mode:       true
; CHECK-NEXT:      .gs:
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point_symbol:    gs_shader
; CHECK-NEXT:        .forward_progress: true
; CHECK-NEXT:        .lds_size:       0{{$}}
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     true
; CHECK-NEXT:        .scratch_memory_size: 0x250{{$}}
; CHECK-NEXT:        .sgpr_count:     0x62{{$}}
; CHECK-NEXT:        .vgpr_count:     0xf8{{$}}
; CHECK-NEXT:        .wgp_mode:       true
; CHECK-NEXT:      .ps:
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point_symbol:    ps_shader
; CHECK-NEXT:        .forward_progress: true
; CHECK-NEXT:        .lds_size:       0{{$}}
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     true
; CHECK-NEXT:        .scratch_memory_size: 0x10{{$}}
; CHECK-NEXT:        .sgpr_count:     0x24{{$}}
; CHECK-NEXT:        .vgpr_count:     0x2{{$}}
; CHECK-NEXT:        .wgp_mode:       true
; CHECK:    .registers:      {}
; CHECK:amdpal.version:
; CHECK-NEXT:  - 0x3{{$}}
; CHECK-NEXT:  - 0x6{{$}}
; CHECK-NEXT:...
; CHECK-NEXT:        .end_amdgpu_pal_metadata

; Callee with high VGPR, SGPR and stack usage. The PAL metadata should reflect this.
define amdgpu_gfx_whole_wave i32 @wwf(i1 %active, i32 %x) {
  call void asm sideeffect "; touch high VGPR and SGPR", "~{v66},~{s66}"()
  %temp = alloca i32, align 1024, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %temp, align 1024
  %result = add i32 %x, 42
  ret i32 %result
}

define amdgpu_cs void @cs_shader(i32 %y) {
  %local = alloca i32, addrspace(5)
  %result = call i32(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf, i32 %y)
  %storable = mul i32 %result, %y
  store volatile i32 %storable, ptr addrspace(5) %local
  ret void
}

; Test that dynamic stack allocations in the callee are reported for the caller.
define amdgpu_gfx_whole_wave void @wwf_dyn_stack(i1 %active, i32 inreg %size, i32 %x) {
  %temp = alloca i32, i32 %size, addrspace(5)
  store volatile i32 %x, ptr addrspace(5) %temp
  ret void
}

define amdgpu_ps void @ps_shader() #1 {
  call void(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_dyn_stack, i32 inreg 12, i32 121)
  ret void
}

; Test that indirect calls in the callee are reported for the caller.
define amdgpu_gfx_whole_wave void @wwf_indirect(i1 %active, ptr inreg %func_ptr, i32 %x) {
  call void(i32) %func_ptr(i32 %x)
  ret void
}

define amdgpu_gs void @gs_shader(ptr inreg %func_ptr) {
  call void(ptr, ...) @llvm.amdgcn.call.whole.wave(ptr @wwf_indirect, ptr inreg %func_ptr, i32 42)
  ret void
}

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\06"}
