; RUN: opt -S -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -passes=infer-address-spaces %s | FileCheck %s

; Make sure addrspace(0) is still treated as flat on targets without
; flat instructions. It's still flat, it just doesn't work.

; CHECK-LABEL: @load_flat_from_global(
; CHECK-NEXT: %tmp1 = load float, ptr addrspace(1) %ptr
; CHECK-NEXT: ret float %tmp1
define float @load_flat_from_global(ptr addrspace(1) %ptr) #0 {
  %tmp0 = addrspacecast ptr addrspace(1) %ptr to ptr
  %tmp1 = load float, ptr %tmp0
  ret float %tmp1
}
