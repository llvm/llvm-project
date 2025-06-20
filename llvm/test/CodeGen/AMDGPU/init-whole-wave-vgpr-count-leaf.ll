; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1200 < %s | FileCheck %s

; CHECK-LABEL: .shader_functions:

; Make sure that .vgpr_count doesn't include the %inactive.vgpr registers.
; CHECK-LABEL: leaf_shader:
; CHECK: .vgpr_count:{{.*}}0xc{{$}}

; Function without calls.
define amdgpu_cs_chain void @_leaf_shader(ptr %output.ptr, i32 inreg %input.value,
                              i32 %active.vgpr1, i32 %active.vgpr2,
                              i32 %inactive.vgpr1, i32 %inactive.vgpr2, i32 %inactive.vgpr3,
                              i32 %inactive.vgpr4, i32 %inactive.vgpr5, i32 %inactive.vgpr6)
                              local_unnamed_addr {
entry:
  %dead.val = call i32 @llvm.amdgcn.dead.i32()
  %is.whole.wave = call i1 @llvm.amdgcn.init.whole.wave()
  br i1 %is.whole.wave, label %compute, label %merge

compute:
  ; Perform a more complex computation using active VGPRs
  %square = mul i32 %active.vgpr1, %active.vgpr1
  %product = mul i32 %square, %active.vgpr2
  %sum = add i32 %product, %input.value
  %result = add i32 %sum, 42
  br label %merge

merge:
  %final.result = phi i32 [ 0, %entry ], [ %result, %compute ]
  %final.inactive1 = phi i32 [ %inactive.vgpr1, %entry ], [ %dead.val, %compute ]
  %final.inactive2 = phi i32 [ %inactive.vgpr2, %entry ], [ %dead.val, %compute ]
  %final.inactive3 = phi i32 [ %inactive.vgpr3, %entry ], [ %dead.val, %compute ]
  %final.inactive4 = phi i32 [ %inactive.vgpr4, %entry ], [ %dead.val, %compute ]
  %final.inactive5 = phi i32 [ %inactive.vgpr5, %entry ], [ %dead.val, %compute ]
  %final.inactive6 = phi i32 [ %inactive.vgpr6, %entry ], [ %dead.val, %compute ]

  store i32 %final.result, ptr %output.ptr, align 4

  ret void
}

declare i32 @llvm.amdgcn.dead.i32()
declare i1 @llvm.amdgcn.init.whole.wave()
declare void @llvm.amdgcn.cs.chain.p0.i32.v4i32.sl_i32i32i32i32i32i32i32i32i32i32i32i32s(ptr, i32, <4 x i32>, { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, i32 immarg, ...)

declare amdgpu_cs_chain void @retry_vgpr_alloc.v4i32(<4 x i32> inreg)

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\82\B0amdpal.pipelines\91\8B\A4.api\A6Vulkan\B2.compute_registers\85\AB.tg_size_en\C3\AA.tgid_x_en\C3\AA.tgid_y_en\C3\AA.tgid_z_en\C3\AF.tidig_comp_cnt\00\B0.hardware_stages\81\A3.cs\8D\AF.checksum_value\00\AB.debug_mode\00\AB.float_mode\CC\C0\A9.image_op\C2\AC.mem_ordered\C3\AB.sgpr_limitj\B7.threadgroup_dimensions\93 \01\01\AD.trap_present\00\B2.user_data_reg_map\90\AB.user_sgprs\10\AB.vgpr_limit\CD\01\00\AF.wavefront_size \AF.wg_round_robin\C2\B7.internal_pipeline_hash\92\CF|{2&\DCC\85M\CFep\8A\EDR\DE\D6\E1\B1.shader_functions\81\A7_miss_1\82\B4.frontend_stack_size\00\B4.outgoing_vgpr_countP\A8.shaders\81\A8.compute\82\B0.api_shader_hash\92\00\00\B1.hardware_mapping\91\A3.cs\B0.spill_threshold\CD\FF\FF\A5.type\A2Cs\B0.user_data_limit\01\A9.uses_cps\C3\AF.xgl_cache_info\82\B3.128_bit_cache_hash\92\CF\B4\AF\9D\0B\07\88\03\02\CF\01o\C9\CAf?)\DA\AD.llpc_version\A476.0\AEamdpal.version\92\03\00"}
