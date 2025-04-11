; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -amdgpu-kernarg-preload-count=16 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -amdgpu-kernarg-preload-count=16 < %s | FileCheck --check-prefix=CHECK %s

; CHECK:	amdhsa.kernels:
; CHECK-NEXT:    - .agpr_count:     0
; CHECK-NEXT:     .args:
; CHECK-NEXT:       - .name:           in
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .preload_registers: s8
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .preload_registers: 's[10:11]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .preload_registers: 's[12:13]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         24
; CHECK-NEXT:         .preload_registers: 's[14:15]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         36
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         40
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK-NEXT:       - .offset:         44
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_x
; CHECK-NEXT:       - .offset:         46
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_y
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_z
; CHECK-NEXT:       - .offset:         50
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_x
; CHECK-NEXT:       - .offset:         52
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_y
; CHECK-NEXT:       - .offset:         54
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_z
; CHECK-NEXT:       - .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         88
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .offset:         96
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_grid_dims
; CHECK-NEXT:       - .offset:         104
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:       - .offset:         112
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_hostcall_buffer
; CHECK-NEXT:       - .offset:         120
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_multigrid_sync_arg
; CHECK-NEXT:       - .offset:         128
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_heap_v1
; CHECK-NEXT:       - .offset:         136
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_default_queue
; CHECK-NEXT:       - .offset:         144
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_completion_action
; CHECK-NEXT:       - .offset:         152
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_dynamic_lds_size
; CHECK-NEXT:       - .offset:         232
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_queue_ptr
; CHECK-NEXT:   .group_segment_fixed_size: 0
; CHECK-NEXT:   .kernarg_segment_align: 8
; CHECK-NEXT:   .kernarg_segment_size: 288
; CHECK-NEXT:   .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           test_preload_v6
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK-NEXT:   .sgpr_count:     22
; CHECK-NEXT:   .sgpr_spill_count: 0
; CHECK-NEXT:   .symbol:         test_preload_v6.kd
; CHECK-NEXT:   .uses_dynamic_stack: false
; CHECK-NEXT:   .vgpr_count:     3
; CHECK-NEXT:   .vgpr_spill_count: 0
; CHECK-NEXT:   .wavefront_size: 64
; CHECK-NEXT: - .agpr_count:     0
; CHECK-NEXT:     .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           out
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .preload_registers: 's[2:3]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .preload_registers: s4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         12
; CHECK-NEXT:         .preload_registers: s5
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .preload_registers: s6
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK-NEXT:       - .offset:         20
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_x
; CHECK-NEXT:       - .offset:         22
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_z
; CHECK-NEXT:       - .offset:         26
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_x
; CHECK-NEXT:       - .offset:         28
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_y
; CHECK-NEXT:       - .offset:         30
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_z
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .offset:         72
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_grid_dims
; CHECK-NEXT:       - .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:   .group_segment_fixed_size: 0
; CHECK-NEXT:   .kernarg_segment_align: 8
; CHECK-NEXT:   .kernarg_segment_size: 264
; CHECK-NEXT:   .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           test_preload_v6_block_count_xyz
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK-NEXT:   .sgpr_count:     13
; CHECK-NEXT:   .sgpr_spill_count: 0
; CHECK-NEXT:   .symbol:         test_preload_v6_block_count_xyz.kd
; CHECK-NEXT:   .uses_dynamic_stack: false
; CHECK-NEXT:   .vgpr_count:     4
; CHECK-NEXT:   .vgpr_spill_count: 0
; CHECK-NEXT:   .wavefront_size: 64
; CHECK-NEXT: - .agpr_count:     0
; CHECK-NEXT:     .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           out
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .preload_registers: 's[2:3]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .offset:         8
; CHECK-NEXT:         .preload_registers: s4
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         12
; CHECK-NEXT:         .preload_registers: s5
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .preload_registers: s6
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK-NEXT:       - .offset:         20
; CHECK-NEXT:         .preload_registers: s7
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_x
; CHECK-NEXT:       - .offset:         22
; CHECK-NEXT:         .preload_registers: s7
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .preload_registers: s8
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_z
; CHECK-NEXT:       - .offset:         26
; CHECK-NEXT:         .preload_registers: s8
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_x
; CHECK-NEXT:       - .offset:         28
; CHECK-NEXT:         .preload_registers: s9
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_y
; CHECK-NEXT:       - .offset:         30
; CHECK-NEXT:         .preload_registers: s9
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_z
; CHECK-NEXT:       - .offset:         48
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .offset:         72
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_grid_dims
; CHECK-NEXT:       - .offset:         80
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:   .group_segment_fixed_size: 0
; CHECK-NEXT:   .kernarg_segment_align: 8
; CHECK-NEXT:   .kernarg_segment_size: 264
; CHECK-NEXT:   .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           test_preload_v6_block_count_z_workgroup_size_z_remainder_z
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK-NEXT:   .sgpr_count:     16
; CHECK-NEXT:   .sgpr_spill_count: 0
; CHECK-NEXT:   .symbol:         test_preload_v6_block_count_z_workgroup_size_z_remainder_z.kd
; CHECK-NEXT:   .uses_dynamic_stack: false
; CHECK-NEXT:   .vgpr_count:     4
; CHECK-NEXT:   .vgpr_spill_count: 0
; CHECK-NEXT:   .wavefront_size: 64
; CHECK-NEXT: - .agpr_count:     0
; CHECK-NEXT:     .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           out
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .preload_registers: 's[2:3]'
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .name:           arg0
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .preload_registers: s4
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .name:           arg1
; CHECK-NEXT:         .offset:         10
; CHECK-NEXT:         .preload_registers: s4
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     by_value
; CHECK-NEXT:       - .offset:         16
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_x
; CHECK-NEXT:       - .offset:         20
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_y
; CHECK-NEXT:       - .offset:         24
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     hidden_block_count_z
; CHECK-NEXT:       - .offset:         28
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_x
; CHECK-NEXT:       - .offset:         30
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_y
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_group_size_z
; CHECK-NEXT:       - .offset:         34
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_x
; CHECK-NEXT:       - .offset:         36
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_y
; CHECK-NEXT:       - .offset:         38
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_remainder_z
; CHECK-NEXT:       - .offset:         56
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_x
; CHECK-NEXT:       - .offset:         64
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_y
; CHECK-NEXT:       - .offset:         72
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_global_offset_z
; CHECK-NEXT:       - .offset:         80
; CHECK-NEXT:         .size:           2
; CHECK-NEXT:         .value_kind:     hidden_grid_dims
; CHECK-NEXT:       - .offset:         88
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     hidden_printf_buffer
; CHECK-NEXT:   .group_segment_fixed_size: 0
; CHECK-NEXT:   .kernarg_segment_align: 8
; CHECK-NEXT:   .kernarg_segment_size: 272
; CHECK-NEXT:   .max_flat_workgroup_size: 1024
; CHECK-NEXT:   .name:           test_prelaod_v6_ptr1_i16_i16
; CHECK-NEXT:   .private_segment_fixed_size: 0
; CHECK-NEXT:   .sgpr_count:     11
; CHECK-NEXT:   .sgpr_spill_count: 0
; CHECK-NEXT:   .symbol:         test_prelaod_v6_ptr1_i16_i16.kd
; CHECK-NEXT:   .uses_dynamic_stack: false
; CHECK-NEXT:   .vgpr_count:     2
; CHECK-NEXT:   .vgpr_spill_count: 0
; CHECK-NEXT:   .wavefront_size: 64
; CHECK-NEXT: amdhsa.printf:
; CHECK-NEXT:   - '1:1:4:%d\n'
; CHECK-NEXT:   - '2:1:8:%g\n'
; CHECK-NEXT: amdhsa.target:   amdgcn-amd-amdhsa--gfx942
; CHECK-NEXT: amdhsa.version:
; CHECK-NEXT:   - 1
; CHECK-NEXT:   - 3

@lds = external hidden addrspace(3) global [0 x i32], align 4

define amdgpu_kernel void @test_preload_v6(
    i32 inreg %in,
    ptr addrspace(1) inreg %r,
    ptr addrspace(1) inreg %a,
    ptr addrspace(1) inreg %b) #0 {
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  store i32 1234, ptr addrspacecast (ptr addrspace(3) @lds to ptr), align 4
  ret void
}

define amdgpu_kernel void @test_preload_v6_block_count_xyz(ptr addrspace(1) inreg %out) #1 {
  %imp_arg_ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep_x = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 0
  %load_x = load i32, ptr addrspace(4) %gep_x
  %gep_y = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 4
  %load_y = load i32, ptr addrspace(4) %gep_y
  %gep_z = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 8
  %load_z = load i32, ptr addrspace(4) %gep_z
  %ins.0 =  insertelement <3 x i32> poison, i32 %load_x, i32 0
  %ins.1 =  insertelement <3 x i32> %ins.0, i32 %load_y, i32 1
  %ins.2 =  insertelement <3 x i32> %ins.1, i32 %load_z, i32 2
  store <3 x i32> %ins.2, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_preload_v6_block_count_z_workgroup_size_z_remainder_z(ptr addrspace(1) inreg %out) #1 {
  %imp_arg_ptr = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %gep0 = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 8
  %gep1 = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 16
  %gep2 = getelementptr i8, ptr addrspace(4) %imp_arg_ptr, i32 22
  %load0 = load i32, ptr addrspace(4) %gep0
  %load1 = load i16, ptr addrspace(4) %gep1
  %load2 = load i16, ptr addrspace(4) %gep2
  %conv1 = zext i16 %load1 to i32
  %conv2 = zext i16 %load2 to i32
  %ins.0 =  insertelement <3 x i32> poison, i32 %load0, i32 0
  %ins.1 =  insertelement <3 x i32> %ins.0, i32 %conv1, i32 1
  %ins.2 =  insertelement <3 x i32> %ins.1, i32 %conv2, i32 2
  store <3 x i32> %ins.2, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_prelaod_v6_ptr1_i16_i16(ptr addrspace(1) inreg %out, i16 inreg %arg0, i16 inreg %arg1) #1 {
  %ext = zext i16 %arg0 to i32
  %ext1 = zext i16 %arg1 to i32
  %add = add i32 %ext, %ext1
  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}


!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!llvm.printf.fmts = !{!1, !2}
!1 = !{!"1:1:4:%d\5Cn"}
!2 = !{!"2:1:8:%g\5Cn"}

attributes #0 = { optnone noinline }
attributes #1 = { "amdgpu-agpr-alloc"="0" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "uniform-work-group-size"="false" }