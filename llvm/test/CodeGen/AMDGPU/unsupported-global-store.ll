; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx9-generic    < %s 2>&1 | FileCheck -check-prefixes=GFX9-GENERIC    %s
; xxx: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx9-4-generic  < %s 2>&1 | FileCheck -check-prefixes=GFX9-4-GENERIC  %s
; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx10-1-generic < %s 2>&1 | FileCheck -check-prefixes=GFX10-1-GENERIC %s
; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx10-3-generic < %s 2>&1 | FileCheck -check-prefixes=GFX10-3-GENERIC %s
; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx11-generic   < %s 2>&1 | FileCheck -check-prefixes=GFX11-GENERIC   %s
; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx12-generic   < %s 2>&1 | FileCheck -check-prefixes=GFX12-GENERIC   %s

; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx9-generic    < %s 2>&1 | FileCheck -check-prefixes=GFX9-GENERIC-GBL-ISEL    %s
; xxx: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx9-4-generic  < %s 2>&1 | FileCheck -check-prefixes=GFX9-4-GENERIC-GBL-ISEL  %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx10-1-generic < %s 2>&1 | FileCheck -check-prefixes=GFX10-1-GENERIC-GBL-ISEL %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx10-3-generic < %s 2>&1 | FileCheck -check-prefixes=GFX10-3-GENERIC-GBL-ISEL %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx11-generic   < %s 2>&1 | FileCheck -check-prefixes=GFX11-GENERIC-GBL-ISEL   %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx12-generic   < %s 2>&1 | FileCheck -check-prefixes=GFX12-GENERIC-GBL-ISEL   %s

define void @global_store_b128(ptr addrspace(1) %addr, <4 x i32> %data) {
; GFX9-GENERIC:    LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX10-1-GENERIC: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX10-3-GENERIC: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX11-GENERIC:   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX12-GENERIC:   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128

; GFX9-GENERIC-GBL-ISEL:    LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX10-1-GENERIC-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX10-3-GENERIC-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX11-GENERIC-GBL-ISEL:   LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX12-GENERIC-GBL-ISEL:   LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
entry:
  call void @llvm.amdgcn.global.store.b128(ptr addrspace(1) %addr, <4 x i32> %data, metadata !0)
  ret void
}

!0 = !{!""}
