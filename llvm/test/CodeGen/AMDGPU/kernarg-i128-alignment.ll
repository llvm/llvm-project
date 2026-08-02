; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck %s

; The AMDGPU data layout has to give i128 an ABI alignment of 16, matching the
; ABI implemented by Clang, whose AMDGPUTargetInfo leaves Int128Align at its
; 128-bit default. Without an explicit entry the alignment would be inherited
; from the i64:64 entry, and the layout LLVM computes for an aggregate would
; then disagree with the one a frontend used when it emitted the field offsets.
;
; For kernel arguments that disagreement is an ABI break rather than a missed
; optimization: the kernarg slot is sized from the data layout, so the tail of
; the argument is never copied into the kernarg segment and loads of it run off
; the end of the segment.

; struct S { i64 a; i128 b; }: ABI align 16, size 32, offsetof(b) == 16.
; The byref slot must be 32 bytes, not 24, and `b` must be loaded from 0x20
; (kernarg base 16 plus a field offset of 16) which stays inside the segment.
; CHECK-LABEL: {{^}}kernarg_i128:
; CHECK: s_load_b128 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x20
; CHECK: .amdhsa_kernarg_size 48

; An i128 following a smaller member is padded out to offset 16 rather than
; packed at offset 8.
; CHECK-LABEL: {{^}}kernarg_i128_after_i8:
; CHECK: s_load_b128 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x20
; CHECK: .amdhsa_kernarg_size 48

; A bare i128 kernel argument is 16-byte aligned in the kernarg segment, so it
; starts at 16 (not 8) and the argument after it at 32 (not 24).
; CHECK-LABEL: {{^}}kernarg_i128_scalar:
; CHECK: s_load_b128 s[{{[0-9]+:[0-9]+}}], s[0:1], 0x10
; CHECK: .amdhsa_kernarg_size 40

; The kernel metadata is emitted once, after every function, so the per-kernel
; argument offsets are checked here in order rather than under each label.
; CHECK: .amdgpu_metadata

; CHECK:      .name:           s
; CHECK-NEXT: .offset:         16
; CHECK-NEXT: .size:           32
; CHECK: .kernarg_segment_size: 48
; CHECK: .name:           kernarg_i128
;
; CHECK:      .name:           s
; CHECK-NEXT: .offset:         16
; CHECK-NEXT: .size:           32
; CHECK: .kernarg_segment_size: 48
; CHECK: .name:           kernarg_i128_after_i8
;
; CHECK:      .name:           a
; CHECK-NEXT: .offset:         16
; CHECK-NEXT: .size:           16
; CHECK:      .name:           b
; CHECK-NEXT: .offset:         32
; CHECK-NEXT: .size:           8
; CHECK: .kernarg_segment_size: 40
; CHECK: .name:           kernarg_i128_scalar

define amdgpu_kernel void @kernarg_i128(ptr addrspace(1) %out,
                                        ptr addrspace(4) byref({ i64, i128 }) align 16 %s) {
  %pb = getelementptr inbounds i8, ptr addrspace(4) %s, i64 16
  %b = load i128, ptr addrspace(4) %pb, align 16
  store i128 %b, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_kernel void @kernarg_i128_after_i8(ptr addrspace(1) %out,
                                                 ptr addrspace(4) byref({ i8, i128 }) align 16 %s) {
  %pb = getelementptr inbounds i8, ptr addrspace(4) %s, i64 16
  %b = load i128, ptr addrspace(4) %pb, align 16
  store i128 %b, ptr addrspace(1) %out, align 16
  ret void
}

define amdgpu_kernel void @kernarg_i128_scalar(ptr addrspace(1) %out, i128 %a, i64 %b) {
  %ext = zext i64 %b to i128
  %sum = add i128 %a, %ext
  store i128 %sum, ptr addrspace(1) %out, align 16
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
