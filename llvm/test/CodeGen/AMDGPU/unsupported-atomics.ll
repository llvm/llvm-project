; RUN: not --crash llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GFX906 %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX908 %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX940 %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GFX1030 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1100 %s

; GFX906:       LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.atomic.fadd

; GFX908-LABEL: fadd_test:
; GFX908:       global_atomic_add_f32

; GFX90A-LABEL: fadd_test:
; GFX90A:       global_atomic_add_f32

; GFX940-LABEL: fadd_test:
; GFX940:       global_atomic_add_f32

; GFX1030:       LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.atomic.fadd

; GFX1100-LABEL: fadd_test:
; GFX1100:       global_atomic_add_f32

define fastcc void @fadd_test(ptr addrspace(1) nocapture noundef %0, float noundef %1) unnamed_addr {
  %3 = tail call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) noundef %0, float noundef %1)
  ret void
}
declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) nocapture, float)
