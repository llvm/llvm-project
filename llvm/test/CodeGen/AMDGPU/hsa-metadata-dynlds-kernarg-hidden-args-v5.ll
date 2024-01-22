; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefix=CHECK %s


; CHECK:	amdhsa.kernels:
; CHECK-NEXT:       - .args:
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           r
; CHECK-NEXT:         .offset:         0
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           a
; CHECK-NEXT:         .offset:         8
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  global
; CHECK-NEXT:         .name:           b
; CHECK-NEXT:         .offset:         16
; CHECK-NEXT:         .size:           8
; CHECK-NEXT:         .value_kind:     global_buffer
; CHECK-NEXT:       - .address_space:  local
; CHECK-NEXT:         .name:           lds_ptr
; CHECK-NEXT:         .offset:         24
; CHECK-NEXT:         .pointee_align:  1
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:         .value_kind:     dynamic_shared_pointer
; CHECK-NEXT:       - .offset:         32
; CHECK-NEXT:         .size:           4
; CHECK-NEXT:        .value_kind:     hidden_block_count_x
; CHECK-NEXT:      - .offset:         36
; CHECK-NEXT:        .size:           4
; CHECK-NEXT:        .value_kind:     hidden_block_count_y
; CHECK-NEXT:      - .offset:         40
; CHECK-NEXT:        .size:           4
; CHECK-NEXT:        .value_kind:     hidden_block_count_z
; CHECK-NEXT:      - .offset:         44
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_group_size_x
; CHECK-NEXT:      - .offset:         46
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_group_size_y
; CHECK-NEXT:      - .offset:         48
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_group_size_z
; CHECK-NEXT:      - .offset:         50
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_remainder_x
; CHECK-NEXT:      - .offset:         52
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_remainder_y
; CHECK-NEXT:      - .offset:         54
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_remainder_z
; CHECK-NEXT:      - .offset:         72
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_global_offset_x
; CHECK-NEXT:      - .offset:         80
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_global_offset_y
; CHECK-NEXT:      - .offset:         88
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_global_offset_z
; CHECK-NEXT:      - .offset:         96
; CHECK-NEXT:        .size:           2
; CHECK-NEXT:        .value_kind:     hidden_grid_dims
; CHECK-NEXT:      - .offset:         104
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_printf_buffer
; CHECK-NEXT:      - .offset:         112
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_hostcall_buffer
; CHECK-NEXT:      - .offset:         120
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_multigrid_sync_arg
; CHECK-NEXT:      - .offset:         128
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_heap_v1
; CHECK-NEXT:      - .offset:         136
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_default_queue
; CHECK-NEXT:      - .offset:         144
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_completion_action
; CHECK:          - .offset:          152
; CHECK-NEXT:        .size:           4
; CHECK-NEXT:        .value_kind:     hidden_dynamic_lds_size
; CHECK:          - .offset:          232
; CHECK-NEXT:        .size:           8
; CHECK-NEXT:        .value_kind:     hidden_queue_ptr

; CHECK:          .name:           test_v5
; CHECK:          .symbol:         test_v5.kd

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 2

define amdgpu_kernel void @test_v5(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b,
    ptr addrspace(3) %lds_ptr) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  store i32 1234, ptr addrspace(3) %lds_ptr, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!llvm.printf.fmts = !{!1, !2}
!1 = !{!"1:1:4:%d\5Cn"}
!2 = !{!"2:1:8:%g\5Cn"}

attributes #0 = { optnone noinline }

