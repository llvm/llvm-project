; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-lower-ctor-dtor %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s

; Make sure we emit code for constructor entries that aren't direct
; function calls.

; Check a constructor that's an alias, and an integer literal.
@llvm.global_ctors = appending addrspace(1) global [2 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 1, ptr @foo.alias, i8* null },
  { i32, ptr, ptr } { i32 1, ptr inttoptr (i64 4096 to ptr), i8* null }
]

; Check a constantexpr addrspacecast
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 1, ptr addrspacecast (ptr addrspace(1) @bar to ptr), i8* null }
]

@foo.alias = hidden alias void (), ptr @foo

;.
; CHECK-NOT: @llvm.global_ctors
; CHECK-NOT: @llvm.global_dtors
; CHECK: @llvm.used = appending global [2 x ptr] [ptr @amdgcn.device.init, ptr @amdgcn.device.fini], section "llvm.metadata"
; CHECK: @foo.alias = hidden alias void (), ptr @foo
;.
define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    ret void
;
  ret void
}

define void @bar() addrspace(1) {
; CHECK-LABEL: @bar(
; CHECK-NEXT:    ret void
;
  ret void
}

; CHECK: define amdgpu_kernel void @amdgcn.device.init() #[[ATTR0:[0-9]+]] {
; CHECK-NEXT: call void @foo.alias()
; CHECK-NEXT: call void inttoptr (i64 4096 to ptr)()
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; CHECK: define amdgpu_kernel void @amdgcn.device.fini() #[[ATTR1:[0-9]+]] {
; CHECK-NEXT: call void addrspacecast (ptr addrspace(1) @bar to ptr)()
; CHECK-NEXT: ret void
; CHECK-NEXT: }

;.
; CHECK: attributes #[[ATTR0]] = { "device-init" }
; CHECK: attributes #[[ATTR1]] = { "device-fini" }


; GCN-LABEL: foo:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
; GCN-LABEL: bar:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
; GCN-LABEL: amdgcn.device.init:
; GCN:         s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT:    s_add_u32 s[[PC_LO]], s[[PC_LO]], foo.alias@rel32@lo+4
; GCN-NEXT:    s_addc_u32 s[[PC_HI]], s[[PC_HI]], foo.alias@rel32@hi+12
; GCN-NEXT:    s_swappc_b64 s[30:31], s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}

; GCN:         s_mov_b64 [[LIT_ADDR:s\[[0-9]+:[0-9]+\]]], 0x1000
; GCN:         s_swappc_b64 s[30:31], [[LIT_ADDR]]
; GCN-NEXT:    s_endpgm
;
; GCN-LABEL: amdgcn.device.fini:
; GCN:         s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT:    s_add_u32 s[[PC_LO]], s[[PC_LO]], bar@gotpcrel32@lo+4
; GCN-NEXT:    s_addc_u32 s[[PC_HI]], s[[PC_HI]], bar@gotpcrel32@hi+12
; GCN-NEXT:    s_load_dwordx2 s{{\[}}[[GOT_LO:[0-9]+]]:[[GOT_HI:[0-9]+]]{{\]}}, s{{\[}}[[PC_LO]]:[[PC_HI]]{{\]}}, 0x0
; GCN:         s_swappc_b64 s[30:31], s{{\[}}[[GOT_LO]]:[[GOT_HI]]{{\]}}
; GCN-NEXT:    s_endpgm
