; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -passes=atomic-expand %s | FileCheck %s

; CHECK: %preserve_me = phi float [ %{{[0-9]+}}, %atomicrmw.shared ], [ %loaded.private, %atomicrmw.private ], [ %{{[0-9]+}}, %atomicrmw.global ]
; CHECK: ret float %preserve_me
define float @expand_preserve_name(ptr %addr, float %val) {
  %preserve_me = atomicrmw fadd ptr %addr, float %val seq_cst, !amdgpu.no.fine.grained.memory !0, !amdgpu.ignore.denormal.mode !0
  ret float %preserve_me
}

!0 = !{}
