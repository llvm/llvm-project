; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1201 -stop-after=finalize-isel -o - < %s | FileCheck -check-prefix=ISEL %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1201 < %s | FileCheck -check-prefix=ISA %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 -amdgpu-scalarize-global-loads -stop-after=finalize-isel -o - < %s | FileCheck -check-prefix=GFX11-SCALAR %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 -amdgpu-scalarize-global-loads=false -stop-after=finalize-isel -o - < %s | FileCheck -check-prefix=GFX11-SCALAR %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx900 -mattr=+max-private-element-size-8 -stop-after=finalize-isel -o - < %s | FileCheck -check-prefix=PRIVATE8 %s

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @shrink_i64_load_to_i32_chunks(ptr addrspace(1) %out,
                                                         ptr addrspace(4) %p) {
; ISEL-LABEL: name: shrink_i64_load_to_i32_chunks
; ISEL: GLOBAL_LOAD_DWORDX3{{.*}}load (s96)
; ISEL: GLOBAL_LOAD_DWORDX3{{.*}}load (s96)
; ISEL: GLOBAL_LOAD_DWORDX3{{.*}}load (s96)
; ISEL-NOT: GLOBAL_LOAD_DWORDX4
;
; ISA-LABEL: shrink_i64_load_to_i32_chunks:
; ISA: global_load_b96
; ISA: global_load_b96
; ISA: global_load_b96
; ISA-NOT: global_load_b128
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 6
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <6 x i64>, ptr addrspace(4) %p.div, align 8
  %e0 = extractelement <6 x i64> %v, i32 0
  %e1 = extractelement <6 x i64> %v, i32 1
  %e2 = extractelement <6 x i64> %v, i32 2
  %e3 = extractelement <6 x i64> %v, i32 3
  %e4 = extractelement <6 x i64> %v, i32 4
  %e5 = extractelement <6 x i64> %v, i32 5

  %d0 = trunc i64 %e0 to i32
  %e0hi = lshr i64 %e0, 32
  %d1 = trunc i64 %e0hi to i32
  %d2 = trunc i64 %e1 to i32

  %d4 = trunc i64 %e2 to i32
  %e2hi = lshr i64 %e2, 32
  %d5 = trunc i64 %e2hi to i32
  %d6 = trunc i64 %e3 to i32

  %d8 = trunc i64 %e4 to i32
  %e4hi = lshr i64 %e4, 32
  %d9 = trunc i64 %e4hi to i32
  %d10 = trunc i64 %e5 to i32

  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %d0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %d1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out0, i32 2
  store volatile i32 %d2, ptr addrspace(1) %out2, align 4
  %out4 = getelementptr i32, ptr addrspace(1) %out0, i32 4
  store volatile i32 %d4, ptr addrspace(1) %out4, align 4
  %out5 = getelementptr i32, ptr addrspace(1) %out0, i32 5
  store volatile i32 %d5, ptr addrspace(1) %out5, align 4
  %out6 = getelementptr i32, ptr addrspace(1) %out0, i32 6
  store volatile i32 %d6, ptr addrspace(1) %out6, align 4
  %out8 = getelementptr i32, ptr addrspace(1) %out0, i32 8
  store volatile i32 %d8, ptr addrspace(1) %out8, align 4
  %out9 = getelementptr i32, ptr addrspace(1) %out0, i32 9
  store volatile i32 %d9, ptr addrspace(1) %out9, align 4
  %out10 = getelementptr i32, ptr addrspace(1) %out0, i32 10
  store volatile i32 %d10, ptr addrspace(1) %out10, align 4
  ret void
}

define amdgpu_kernel void @keep_i64_load_when_high_dword_used(ptr addrspace(1) %out,
                                                              ptr addrspace(4) %p) {
; ISEL-LABEL: name: keep_i64_load_when_high_dword_used
; ISEL: GLOBAL_LOAD_DWORDX4
; ISEL-NOT: GLOBAL_LOAD_DWORDX3
;
; ISA-LABEL: keep_i64_load_when_high_dword_used:
; ISA: global_load_b128
; ISA-NOT: global_load_b96
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 4
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <2 x i64>, ptr addrspace(4) %p.div, align 8
  %e0 = extractelement <2 x i64> %v, i32 0
  %e1 = extractelement <2 x i64> %v, i32 1
  %d0 = trunc i64 %e0 to i32
  %e1hi = lshr i64 %e1, 32
  %d3 = trunc i64 %e1hi to i32
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %d0, ptr addrspace(1) %out0, align 4
  %out3 = getelementptr i32, ptr addrspace(1) %out0, i32 3
  store volatile i32 %d3, ptr addrspace(1) %out3, align 4
  ret void
}

define amdgpu_kernel void @keep_non_contiguous_lanes(ptr addrspace(1) %out,
                                                     ptr addrspace(4) %p) {
; ISEL-LABEL: name: keep_non_contiguous_lanes
; ISEL: GLOBAL_LOAD_DWORDX4
; ISEL-NOT: GLOBAL_LOAD_DWORDX2
; ISEL-NOT: GLOBAL_LOAD_DWORDX3
;
; ISA-LABEL: keep_non_contiguous_lanes:
; ISA: global_load_b128
; ISA-NOT: global_load_b64
; ISA-NOT: global_load_b96
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 4
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <4 x i32>, ptr addrspace(4) %p.div, align 8
  %e0 = extractelement <4 x i32> %v, i32 0
  %e2 = extractelement <4 x i32> %v, i32 2
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %e0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %e2, ptr addrspace(1) %out1, align 4
  ret void
}

define amdgpu_kernel void @shrink_v3i32_load_to_v2i32(ptr addrspace(1) %out,
                                                      ptr addrspace(4) %p) {
; ISEL-LABEL: name: shrink_v3i32_load_to_v2i32
; ISEL: GLOBAL_LOAD_DWORDX2{{.*}}load (s64)
; ISEL-NOT: GLOBAL_LOAD_DWORDX3
;
; ISA-LABEL: shrink_v3i32_load_to_v2i32:
; ISA: global_load_b64
; ISA-NOT: global_load_b96
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 4
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <3 x i32>, ptr addrspace(4) %p.div, align 8
  %e0 = extractelement <3 x i32> %v, i32 0
  %e1 = extractelement <3 x i32> %v, i32 1
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %e0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  ret void
}

define amdgpu_kernel void @shrink_v4i32_middle_to_v2i32(ptr addrspace(1) %out,
                                                        ptr addrspace(4) %p) {
; ISEL-LABEL: name: shrink_v4i32_middle_to_v2i32
; ISEL: GLOBAL_LOAD_DWORDX2{{.*}}load (s64) from %ir.p.div + 4
; ISEL-NOT: GLOBAL_LOAD_DWORDX4
;
; ISA-LABEL: shrink_v4i32_middle_to_v2i32:
; ISA: global_load_b64
; ISA-NOT: global_load_b128
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 4
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <4 x i32>, ptr addrspace(4) %p.div, align 8
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %e1, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %e2, ptr addrspace(1) %out1, align 4
  ret void
}

define amdgpu_kernel void @keep_wide_v5i32_load(ptr addrspace(1) %out,
                                                ptr addrspace(4) %p) {
; ISEL-LABEL: name: keep_wide_v5i32_load
; ISEL: GLOBAL_LOAD_DWORDX4{{.*}}load (s128)
; ISEL-NOT: GLOBAL_LOAD_DWORDX3
;
; ISA-LABEL: keep_wide_v5i32_load:
; ISA: global_load_b128
; ISA-NOT: global_load_b96
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 5
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr addrspace(4) %p, i64 %byte.offset64
  %v = load <5 x i32>, ptr addrspace(4) %p.div, align 8
  %e0 = extractelement <5 x i32> %v, i32 0
  %e1 = extractelement <5 x i32> %v, i32 1
  %e2 = extractelement <5 x i32> %v, i32 2
  %e3 = extractelement <5 x i32> %v, i32 3
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %e0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out0, i32 2
  store volatile i32 %e2, ptr addrspace(1) %out2, align 4
  %out3 = getelementptr i32, ptr addrspace(1) %out0, i32 3
  store volatile i32 %e3, ptr addrspace(1) %out3, align 4
  ret void
}

define amdgpu_kernel void @shrink_local_v4i32_load_to_v3i32(ptr addrspace(1) %out,
                                                            ptr addrspace(3) %p) {
; ISEL-LABEL: name: shrink_local_v4i32_load_to_v3i32
; ISEL: DS_READ_B96{{.*}}load (s96)
; ISEL-NOT: DS_READ_B128
;
; ISA-LABEL: shrink_local_v4i32_load_to_v3i32:
; ISA: ds_load_b96
; ISA-NOT: ds_load_b128
entry:
  %v = load <4 x i32>, ptr addrspace(3) %p, align 16
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  store volatile i32 %e0, ptr addrspace(1) %out, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store volatile i32 %e2, ptr addrspace(1) %out2, align 4
  ret void
}

define amdgpu_kernel void @shrink_flat_v4i32_load_to_v3i32(ptr addrspace(1) %out,
                                                           ptr %p) {
; ISEL-LABEL: name: shrink_flat_v4i32_load_to_v3i32
; ISEL: FLAT_LOAD_DWORDX3{{.*}}load (s96)
; ISEL-NOT: FLAT_LOAD_DWORDX4
;
; ISA-LABEL: shrink_flat_v4i32_load_to_v3i32:
; ISA: flat_load_b96
; ISA-NOT: flat_load_b128
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %byte.offset = shl i32 %tid, 4
  %byte.offset64 = zext i32 %byte.offset to i64
  %p.div = getelementptr i8, ptr %p, i64 %byte.offset64
  %v = load <4 x i32>, ptr %p.div, align 8
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store volatile i32 %e0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out0, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out0, i32 2
  store volatile i32 %e2, ptr addrspace(1) %out2, align 4
  ret void
}

define amdgpu_kernel void @keep_underaligned_local_v4i32_load(ptr addrspace(1) %out,
                                                              ptr addrspace(3) %p) {
; ISEL-LABEL: name: keep_underaligned_local_v4i32_load
; ISEL-NOT: DS_READ_B96
; ISEL: DS_READ
;
; ISA-LABEL: keep_underaligned_local_v4i32_load:
; ISA-NOT: ds_load_b96
; ISA: ds_load
entry:
  %v = load <4 x i32>, ptr addrspace(3) %p, align 4
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  store volatile i32 %e0, ptr addrspace(1) %out, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store volatile i32 %e2, ptr addrspace(1) %out2, align 4
  ret void
}

define amdgpu_kernel void @shrink_private_v4i32_load_to_v3i32(ptr addrspace(1) %out,
                                                              ptr addrspace(5) %p) {
; ISEL-LABEL: name: shrink_private_v4i32_load_to_v3i32
; ISEL: SCRATCH_LOAD_DWORDX3{{.*}}load (s96)
; ISEL-NOT: SCRATCH_LOAD_DWORDX4
; PRIVATE8-LABEL: name: shrink_private_v4i32_load_to_v3i32
; PRIVATE8-NOT: SCRATCH_LOAD_DWORDX3
; PRIVATE8: BUFFER_LOAD_DWORDX2{{.*}}load (s64)
; PRIVATE8: BUFFER_LOAD_DWORD{{.*}}load (s32)
;
; ISA-LABEL: shrink_private_v4i32_load_to_v3i32:
; ISA: scratch_load_b96
; ISA-NOT: scratch_load_b128
entry:
  %v = load <4 x i32>, ptr addrspace(5) %p, align 16
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  store volatile i32 %e0, ptr addrspace(1) %out, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %e1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store volatile i32 %e2, ptr addrspace(1) %out2, align 4
  ret void
}

define amdgpu_kernel void @keep_smrd_constant_load(ptr addrspace(1) %out,
                                                   ptr addrspace(4) %p) {
; ISEL-LABEL: name: keep_smrd_constant_load
; ISEL-NOT: S_LOAD_DWORDX3
; ISEL: S_LOAD_DWORDX4{{.*}}load (s128){{.*}}addrspace 4
; ISEL-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-LABEL: name: keep_smrd_constant_load
; GFX11-SCALAR-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-NOT: GLOBAL_LOAD_DWORDX3
; GFX11-SCALAR: S_LOAD_DWORDX4{{.*}}load (s128){{.*}}addrspace 4
; GFX11-SCALAR-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-NOT: GLOBAL_LOAD_DWORDX3
entry:
  %v = load <2 x i64>, ptr addrspace(4) %p, align 16
  %e0 = extractelement <2 x i64> %v, i32 0
  %e1 = extractelement <2 x i64> %v, i32 1
  %d0 = trunc i64 %e0 to i32
  %e0hi = lshr i64 %e0, 32
  %d1 = trunc i64 %e0hi to i32
  %d2 = trunc i64 %e1 to i32
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store volatile i32 %d0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %d1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store volatile i32 %d2, ptr addrspace(1) %out2, align 4
  ret void
}

define amdgpu_kernel void @keep_smrd_uniform_global_load(ptr addrspace(1) %out,
                                                         ptr addrspace(1) %p) {
; ISEL-LABEL: name: keep_smrd_uniform_global_load
; ISEL-NOT: S_LOAD_DWORDX3
; ISEL: S_LOAD_DWORDX4{{.*}}load (s128) from %ir.2
; ISEL-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-LABEL: name: keep_smrd_uniform_global_load
; GFX11-SCALAR-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-NOT: GLOBAL_LOAD_DWORDX3
; GFX11-SCALAR: S_LOAD_DWORDX4{{.*}}load (s128) from %ir.2
; GFX11-SCALAR-NOT: S_LOAD_DWORDX3
; GFX11-SCALAR-NOT: GLOBAL_LOAD_DWORDX3
entry:
  %v = load <2 x i64>, ptr addrspace(1) %p, align 16, !invariant.load !0
  %e0 = extractelement <2 x i64> %v, i32 0
  %e1 = extractelement <2 x i64> %v, i32 1
  %d0 = trunc i64 %e0 to i32
  %e0hi = lshr i64 %e0, 32
  %d1 = trunc i64 %e0hi to i32
  %d2 = trunc i64 %e1 to i32
  %out0 = getelementptr i32, ptr addrspace(1) %out, i32 0
  store volatile i32 %d0, ptr addrspace(1) %out0, align 4
  %out1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store volatile i32 %d1, ptr addrspace(1) %out1, align 4
  %out2 = getelementptr i32, ptr addrspace(1) %out, i32 2
  store volatile i32 %d2, ptr addrspace(1) %out2, align 4
  ret void
}

!0 = !{}
