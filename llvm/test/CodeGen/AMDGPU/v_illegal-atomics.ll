; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX906-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX908-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX940-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1030-ASM %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1100-ASM %s

; GFX906-ASM-LABEL: fadd_test:
; GFX906-ASM-NOT:   global_atomic_add_f32
; GFX906-ASM:       v_illegal

; GFX908-ASM-LABEL: fadd_test:
; GFX908-ASM-NOT:   v_illegal
; GFX908-ASM:       global_atomic_add_f32

; GFX90A-ASM-LABEL: fadd_test:
; GFX90A-ASM-NOT:   v_illegal
; GFX90A-ASM:       global_atomic_add_f32

; GFX940-ASM-LABEL: fadd_test:
; GFX940-ASM-NOT:   v_illegal
; GFX940-ASM:       global_atomic_add_f32

; GFX1030-ASM-LABEL: fadd_test:
; GFX1030-ASM-NOT:   global_atomic_add_f32
; GFX1030-ASM:       v_illegal

; GFX1100-ASM-LABEL: fadd_test:
; GFX1100-ASM-NOT:   v_illegal
; GFX1100-ASM:       global_atomic_add_f32

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx906 -d - | FileCheck --check-prefix=GFX906-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx908 -d - | FileCheck --check-prefix=GFX908-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx90a -d - | FileCheck --check-prefix=GFX90A-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx940 -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx940 -d - | FileCheck --check-prefix=GFX940-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx1030 -d - | FileCheck --check-prefix=GFX1030-OBJ %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj -verify-machineinstrs < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx1100 -d - | FileCheck --check-prefix=GFX1100-OBJ %s

; GFX906-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX906-OBJ-NEXT: v_illegal // 000000000004: FFFFFFFF

; GFX908-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX908-OBJ-NEXT: global_atomic_add_f32

; GFX90A-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX90A-OBJ-NEXT: global_atomic_add_f32

; GFX940-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX940-OBJ-NEXT: global_atomic_add_f32

; GFX1030-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX1030-OBJ-NEXT: s_waitcnt_vscnt null, 0x0
; GFX1030-OBJ-NEXT: v_illegal // 000000000008: 00000000

; GFX1100-OBJ:      s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX1100-OBJ-NEXT: s_waitcnt_vscnt null, 0x0
; GFX1100-OBJ-NEXT: global_atomic_add_f32 v[0:1], v2, off

define fastcc void @fadd_test(ptr addrspace(1) nocapture noundef %0, float noundef %1) unnamed_addr {
  %3 = tail call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) noundef %0, float noundef %1)
  ret void
}
declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) nocapture, float)
