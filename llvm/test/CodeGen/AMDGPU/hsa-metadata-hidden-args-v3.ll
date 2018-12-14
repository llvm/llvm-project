; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX803 --check-prefix=NOTES %s
; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

; CHECK: ---
; CHECK:  amdhsa.kernels:
; CHECK:        .symbol:     test.kd
; CHECK:        .name:       test
; CHECK:        .args:
; CHECK-NEXT:     - .value_kind:      global_buffer
; CHECK-NEXT:       .name:            r
; CHECK-NEXT:       .offset:          0
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      f16
; CHECK-NEXT:       .address_space:   global
; CHECK-NEXT:     - .value_kind:      global_buffer
; CHECK-NEXT:       .name:            a
; CHECK-NEXT:       .offset:          8
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      f16
; CHECK-NEXT:       .address_space:   global
; CHECK-NEXT:     - .value_kind:      global_buffer
; CHECK-NEXT:       .name:            b
; CHECK-NEXT:       .offset:          16
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      f16
; CHECK-NEXT:       .address_space:   global
; CHECK-NEXT:     - .value_kind:      hidden_global_offset_x
; CHECK-NEXT:       .offset:          24
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i64
; CHECK-NEXT:     - .value_kind:      hidden_global_offset_y
; CHECK-NEXT:       .offset:          32
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i64
; CHECK-NEXT:     - .value_kind:      hidden_global_offset_z
; CHECK-NEXT:       .offset:          40
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i64
; CHECK-NEXT:     - .value_kind:      hidden_none
; CHECK-NEXT:       .offset:          48
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i8
; CHECK-NEXT:       .address_space:   global
; CHECK-NEXT:     - .value_kind:      hidden_none
; CHECK-NEXT:       .offset:          56
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i8
; CHECK-NEXT:       .address_space:   global
; CHECK-NEXT:     - .value_kind:      hidden_none
; CHECK-NEXT:       .offset:          64
; CHECK-NEXT:       .size:            8
; CHECK-NEXT:       .value_type:      i8
; CHECK-NEXT:       .address_space:   global
define amdgpu_kernel void @test(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}
