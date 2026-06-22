; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx602          < %s 2>&1 | FileCheck -check-prefixes=GFX602          %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx705          < %s 2>&1 | FileCheck -check-prefixes=GFX705          %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx810          < %s 2>&1 | FileCheck -check-prefixes=GFX810          %s

; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx602          < %s 2>&1 | FileCheck -check-prefixes=GFX602-GBL-ISEL          %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx705          < %s 2>&1 | FileCheck -check-prefixes=GFX705-GBL-ISEL          %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx810          < %s 2>&1 | FileCheck -check-prefixes=GFX810-GBL-ISEL          %s

define void @global_store_b128(ptr addrspace(1) %addr, <4 x i32> %data) {
; GFX602:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX705:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128
; GFX810:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.store.b128

; GFX602-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX705-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
; GFX810-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.store.b128)
entry:
  call void @llvm.amdgcn.global.store.b128(ptr addrspace(1) %addr, <4 x i32> %data, metadata !0)
  ret void
}

!0 = !{!""}
