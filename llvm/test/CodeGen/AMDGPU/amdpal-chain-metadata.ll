; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1100 <%s | FileCheck %s

; CHECK-LABEL: {{^}}amdgpu_cs_chain_func:
; CHECK:           .amdgpu_pal_metadata
; CHECK-NEXT: ---
; CHECK-NEXT: amdpal.pipelines:
; CHECK-NEXT:   - .api:            Vulkan
; CHECK:         .shader_functions:
; CHECK-NEXT:      amdgpu_cs_chain_func:
; CHECK:             .backend_stack_size: 0x10{{$}}
; CHECK:             .stack_frame_size_in_bytes: 0x10{{$}}
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

declare amdgpu_gfx void @use(...)

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\8A\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C2\AA.tgid_y_en\C2\AA.tgid_z_en\C2\AF.tidig_comp_cnt\01\B0.hardware_stages\81\A3.cs\8C\AF.checksum_value\CE\94D\D7\D0\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93\01\CD\04\00\01\AD.trap_present\00\B2.user_data_reg_map\DC\00 \CE\10\00\00\00\CE\FF\FF\FF\FF\00\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\CE\FF\FF\FF\FF\AB.user_sgprs\03\AB.vgpr_limit\CD\01\00\AF.wavefront_size@\B7.internal_pipeline_hash\92\CF\E7\10k\A6:\A6%\F7\CF\B2\1F\1A\D4{\DA\E1T\AA.registers\80\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\CF\E9Zn7}\1E\B9\E7\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CE\FF\FF\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4X\B8\11[\A4\88P\CF\A0;\B0\AF\FF\B4\BE\C0\AD.llpc_version\A461.1\AEamdpal.version\92\03\00"}
!1 = !{i32 7}
