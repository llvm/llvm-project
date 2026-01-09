; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx602          < %s 2>&1 | FileCheck -check-prefixes=GFX602          %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx705          < %s 2>&1 | FileCheck -check-prefixes=GFX705          %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx810          < %s 2>&1 | FileCheck -check-prefixes=GFX810          %s

; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx602          < %s 2>&1 | FileCheck -check-prefixes=GFX602-GBL-ISEL          %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx705          < %s 2>&1 | FileCheck -check-prefixes=GFX705-GBL-ISEL          %s
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx810          < %s 2>&1 | FileCheck -check-prefixes=GFX810-GBL-ISEL          %s

define <4 x i32> @global_load_b128(ptr addrspace(1) %addr) {
; GFX602:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.load.b128
; GFX705:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.load.b128
; GFX810:          LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.global.load.b128

; GFX602-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.load.b128)
; GFX705-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.load.b128)
; GFX810-GBL-ISEL: LLVM ERROR: cannot select: {{.*}} intrinsic(@llvm.amdgcn.global.load.b128)
entry:
  %data = call <4 x i32> @llvm.amdgcn.global.load.b128(ptr addrspace(1) %addr, metadata !0)
  ret <4 x i32> %data
}

!0 = !{!""}
