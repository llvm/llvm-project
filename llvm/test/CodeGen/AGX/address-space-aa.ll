; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s | FileCheck %s
; RUN: opt -mtriple=agx -passes='aa-eval' -aa-pipeline=agx-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AA

; Device and threadgroup memory are disjoint: the second device load is
; issued before the first threadgroup store, so both loads are in flight
; together (two scoreboard slots, two waits).

@tile = external addrspace(3) global [64 x float], align 4

; CHECK-LABEL: copy_two:
; CHECK: device_load 0
; CHECK-NEXT: device_load 1
; CHECK: wait 0
; CHECK-NEXT: threadgroup_store
; CHECK-NEXT: wait 1
; CHECK-NEXT: threadgroup_store

; AA: MayAlias: float addrspace(1)* %pa, float addrspace(1)* %pb
; AA: NoAlias: float addrspace(1)* %pb, float addrspace(3)* %ta
; AA: MayAlias: float* %g, float addrspace(3)* %ta
define agx_kernel void @copy_two(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr %g) {
  %i = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %i
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %i
  %ta = getelementptr float, ptr addrspace(3) @tile, i32 %i
  %j = add i32 %i, 32
  %tb = getelementptr float, ptr addrspace(3) @tile, i32 %j
  %va = load float, ptr addrspace(1) %pa
  store float %va, ptr addrspace(3) %ta
  %vb = load float, ptr addrspace(1) %pb
  store float %vb, ptr addrspace(3) %tb
  %vg = load float, ptr %g
  ret void
}

declare i32 @llvm.agx.thread.position.in.threadgroup.x()
