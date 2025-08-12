; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 <%s | FileCheck %s --check-prefixes=CHECK,GFX11,NODVGPR
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 <%s | FileCheck %s --check-prefixes=CHECK,NODVGPR
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 -mattr=+dynamic-vgpr <%s | FileCheck %s --check-prefixes=CHECK,DVGPR

; CHECK-LABEL: {{^}}_amdgpu_cs_main:
; NODVGPR: ; TotalNumSgprs: 4
; DVGPR: ; TotalNumSgprs: 34
; CHECK: ; NumVgprs: 2
; CHECK:           .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT: amdpal.pipelines:
; CHECK-NEXT:   - .api:            Vulkan
; CHECK-NEXT:     .compute_registers:
; DVGPR-NEXT:       .dynamic_vgpr_en:   true
; CHECK-NEXT:       .tg_size_en:     true
; CHECK-NEXT:       .tgid_x_en:      false
; CHECK-NEXT:       .tgid_y_en:      false
; CHECK-NEXT:       .tgid_z_en:      false
; CHECK-NEXT:       .tidig_comp_cnt: 0x1
; CHECK-NEXT:     .graphics_registers:
; CHECK-NEXT:      .ps_extra_lds_size: 0
; CHECK-NEXT:      .spi_ps_input_addr:
; CHECK-NEXT:        .ancillary_ena:  false
; CHECK-NEXT:        .front_face_ena: true
; CHECK-NEXT:        .line_stipple_tex_ena: false
; CHECK-NEXT:        .linear_center_ena: true
; CHECK-NEXT:        .linear_centroid_ena: true
; CHECK-NEXT:        .linear_sample_ena: true
; CHECK-NEXT:        .persp_center_ena: true
; CHECK-NEXT:        .persp_centroid_ena: true
; CHECK-NEXT:        .persp_pull_model_ena: false
; CHECK-NEXT:        .persp_sample_ena: true
; CHECK-NEXT:        .pos_fixed_pt_ena: true
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
; DVGPR-NEXT:        .dynamic_vgpr_saved_count: 0x70
; CHECK-NEXT:        .entry_point:    _amdgpu_cs_main
; CHECK-NEXT:        .entry_point_symbol:    _amdgpu_cs_main
; CHECK-NEXT:        .excp_en:        0
; CHECK-NEXT:        .float_mode:     0xc0
; CHECK-NEXT:        .forward_progress: true
; GFX11-NEXT:        .ieee_mode:      false
; CHECK-NEXT:        .image_op:       false
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     false
; CHECK-NEXT:        .scratch_memory_size: 0
; NODVGPR-NEXT:      .sgpr_count:     0x4
; DVGPR-NEXT:        .sgpr_count:     0x22
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
; CHECK-NEXT:        .vgpr_count:     0x2
; CHECK-NEXT:        .vgpr_limit:     0x100
; CHECK-NEXT:        .wavefront_size: 0x40
; CHECK-NEXT:        .wgp_mode:       false
; CHECK-NEXT:      .gs:
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point:    _amdgpu_gs_main
; CHECK-NEXT:        .entry_point_symbol:    gs_shader
; CHECK-NEXT:        .forward_progress: true
; GFX11-NEXT:        .ieee_mode:      false
; CHECK-NEXT:        .lds_size:       0x200
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     false
; CHECK-NEXT:        .scratch_memory_size: 0
; CHECK-NEXT:        .sgpr_count:     0x1
; CHECK-NEXT:        .vgpr_count:     0x1
; CHECK-NEXT:        .wgp_mode:       true
; CHECK-NEXT:      .hs:
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point:    _amdgpu_hs_main
; CHECK-NEXT:        .entry_point_symbol:    hs_shader
; CHECK-NEXT:        .forward_progress: true
; GFX11-NEXT:        .ieee_mode:      false
; CHECK-NEXT:        .lds_size:       0x1000
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     false
; CHECK-NEXT:        .scratch_memory_size: 0
; CHECK-NEXT:        .sgpr_count:     0x1
; CHECK-NEXT:        .vgpr_count:     0x1
; CHECK-NEXT:        .wgp_mode:       true
; CHECK-NEXT:      .ps:
; CHECK-NEXT:        .debug_mode:     false
; CHECK-NEXT:        .entry_point:    _amdgpu_ps_main
; CHECK-NEXT:        .entry_point_symbol:    ps_shader
; CHECK-NEXT:        .forward_progress: true
; GFX11-NEXT:        .ieee_mode:      false
; CHECK-NEXT:        .lds_size:       0
; CHECK-NEXT:        .mem_ordered:    true
; CHECK-NEXT:        .scratch_en:     false
; CHECK-NEXT:        .scratch_memory_size: 0
; CHECK-NEXT:        .sgpr_count:     0x1
; CHECK-NEXT:        .vgpr_count:     0x1
; CHECK-NEXT:        .wgp_mode:       true
; CHECK:    .registers:      {}
; CHECK:amdpal.version:
; CHECK-NEXT:  - 0x3
; CHECK-NEXT:  - 0
; CHECK-NEXT:...
; CHECK-NEXT:        .end_amdgpu_pal_metadata

define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg %arg1, i32 %arg2) #0 !lgc.shaderstage !1 {
.entry:
  %i = call i64 @llvm.amdgcn.s.getpc()
  %i1 = and i64 %i, -4294967296
  %i2 = zext i32 %arg1 to i64
  %i3 = or i64 %i1, %i2
  %i4 = inttoptr i64 %i3 to ptr addrspace(4)
  %i5 = and i32 %arg2, 1023
  %i6 = lshr i32 %arg2, 10
  %i7 = and i32 %i6, 1023
  %i8 = add nuw nsw i32 %i7, %i5
  %i9 = load <4 x i32>, ptr addrspace(4) %i4, align 16
  %.idx = shl nuw nsw i32 %i8, 2
  call void @llvm.amdgcn.raw.buffer.store.i32(i32 1, <4 x i32> %i9, i32 %.idx, i32 0, i32 0)
  ret void
}

define dllexport amdgpu_ps void @ps_shader() #1 {
  ret void
}

@LDS.GS = external addrspace(3) global [1 x i32], align 4

define dllexport amdgpu_gs void @gs_shader() {
  %ptr = getelementptr i32, ptr addrspace(3) @LDS.GS, i32 0
  store i32 0, ptr addrspace(3) %ptr, align 4
  ret void
}

@LDS.HS = external addrspace(3) global [1024 x i32], align 4

define dllexport amdgpu_hs void @hs_shader() {
  %ptr = getelementptr i32, ptr addrspace(3) @LDS.HS, i32 0
  store i32 0, ptr addrspace(3) %ptr, align 4
  ret void
}

!amdgpu.pal.metadata.msgpack = !{!0}

declare ptr addrspace(7) @lgc.buffer.desc.to.ptr(<4 x i32>) #1
declare i64 @llvm.amdgcn.s.getpc()
declare void @llvm.amdgcn.raw.buffer.store.i32(i32, <4 x i32>, i32, i32, i32 immarg) #3

attributes #0 = { nounwind memory(readwrite) "target-features"=",+wavefrontsize64,+cumode" }

attributes #1 = { nounwind memory(readwrite) "InitialPSInputAddr"="36983" }

!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\00"}
!1 = !{i32 7}
