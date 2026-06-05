; RUN: opt -S -passes='dxil-cbuffer-access' %s -o - | FileCheck %s --check-prefixes=CHECK
; RUN: llc %s -o - | FileCheck %s --check-prefixes=CHECK-LOWERED
; RUN: llc %s -O3 -o - | FileCheck %s --check-prefixes=CHECK-LOWERED

target triple = "dxil-unknown-shadermodel6.6-compute"

%__cblayout_CB = type <{ i32 }>
@CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
@i = external hidden local_unnamed_addr addrspace(2) global i32, align 4

@llvm.used = appending global [1 x ptr] [ptr @CB.cb], section "llvm.metadata"

; Check that DXILCBufferAccessPass removes the cbuffer global from @llvm.used
;
; CHECK: @CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison
; CHECK-NOT: @i
; CHECK-NOT: @llvm.used

; Check that the cbuffer global is removed during lowering to DXIL
;
; CHECK-LOWERED-NOT: %__cblayout_CB = type <{ i32 }>
; CHECK-LOWERED-NOT: @CB.cb = internal global target("dx.CBuffer", %__cblayout_CB) poison

define void @main() {
entry:
  %cb_handle = tail call target("dx.CBuffer", %__cblayout_CB) @llvm.dx.resource.handlefromimplicitbinding.tdx.CBuffer_s___cblayout_CBst(i32 0, i32 0, i32 1, i32 0, ptr nonnull null)
  store target("dx.CBuffer", %__cblayout_CB) %cb_handle, ptr @CB.cb, align 4
  %1 = load i32, ptr addrspace(2) @i, align 4
  ret void
}

; CHECK-NOT: !hlsl.cbs

!hlsl.cbs = !{!0}

!0 = !{ptr @CB.cb, ptr addrspace(2) @i}
!1 = !{i32 1, i32 8}

