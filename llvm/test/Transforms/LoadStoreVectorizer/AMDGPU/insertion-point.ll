; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Check position of the inserted vector load/store.  Vectorized loads should be
; inserted at the position of the first load in the chain, and stores should be
; inserted at the position of the last store.

; CHECK-LABEL: @insert_load_point(
; CHECK: %z = add i32 %x, 4
; CHECK: load <2 x float>
; CHECK: %w = add i32 %y, 9
; CHECK: %foo = add i32 %z, %w
define amdgpu_kernel void @insert_load_point(float addrspace(1)* nocapture %a, float addrspace(1)* nocapture %b, float addrspace(1)* nocapture readonly %c, i64 %idx, i32 %x, i32 %y) #0 {
entry:
  %a.idx.x = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %c.idx.x = getelementptr inbounds float, float addrspace(1)* %c, i64 %idx
  %a.idx.x.1 = getelementptr inbounds float, float addrspace(1)* %a.idx.x, i64 1
  %c.idx.x.1 = getelementptr inbounds float, float addrspace(1)* %c.idx.x, i64 1

  %z = add i32 %x, 4
  %ld.c = load float, float addrspace(1)* %c.idx.x, align 4
  %w = add i32 %y, 9
  %ld.c.idx.1 = load float, float addrspace(1)* %c.idx.x.1, align 4
  %foo = add i32 %z, %w

  store float 0.0, float addrspace(1)* %a.idx.x, align 4
  store float 0.0, float addrspace(1)* %a.idx.x.1, align 4

  %add = fadd float %ld.c, %ld.c.idx.1
  store float %add, float addrspace(1)* %b, align 4
  store i32 %foo, i32 addrspace(3)* null, align 4
  ret void
}

; CHECK-LABEL: @insert_store_point(
; CHECK: %z = add i32 %x, 4
; CHECK: %w = add i32 %y, 9
; CHECK: store <2 x float>
; CHECK: %foo = add i32 %z, %w
define amdgpu_kernel void @insert_store_point(float addrspace(1)* nocapture %a, float addrspace(1)* nocapture %b, float addrspace(1)* nocapture readonly %c, i64 %idx, i32 %x, i32 %y) #0 {
entry:
  %a.idx.x = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %c.idx.x = getelementptr inbounds float, float addrspace(1)* %c, i64 %idx
  %a.idx.x.1 = getelementptr inbounds float, float addrspace(1)* %a.idx.x, i64 1
  %c.idx.x.1 = getelementptr inbounds float, float addrspace(1)* %c.idx.x, i64 1

  %ld.c = load float, float addrspace(1)* %c.idx.x, align 4
  %ld.c.idx.1 = load float, float addrspace(1)* %c.idx.x.1, align 4

  %z = add i32 %x, 4
  store float 0.0, float addrspace(1)* %a.idx.x, align 4
  %w = add i32 %y, 9
  store float 0.0, float addrspace(1)* %a.idx.x.1, align 4
  %foo = add i32 %z, %w

  %add = fadd float %ld.c, %ld.c.idx.1
  store float %add, float addrspace(1)* %b, align 4
  store i32 %foo, i32 addrspace(3)* null, align 4
  ret void
}

; Here we have four stores, with an aliasing load before the last one.  We can
; vectorize the first three stores as <3 x float>, but this vectorized store must
; be inserted at the location of the third scalar store, not the fourth one.
;
; CHECK-LABEL: @insert_store_point_alias
; CHECK: store <3 x float>
; CHECK: load float, float addrspace(1)* %a.idx.2
; CHECK: store float
; CHECK-SAME: %a.idx.3
define float @insert_store_point_alias(float addrspace(1)* nocapture %a, i64 %idx) {
  %a.idx = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %a.idx.1 = getelementptr inbounds float, float addrspace(1)* %a.idx, i64 1
  %a.idx.2 = getelementptr inbounds float, float addrspace(1)* %a.idx.1, i64 1
  %a.idx.3 = getelementptr inbounds float, float addrspace(1)* %a.idx.2, i64 1

  store float 0.0, float addrspace(1)* %a.idx, align 4
  store float 0.0, float addrspace(1)* %a.idx.1, align 4
  store float 0.0, float addrspace(1)* %a.idx.2, align 4
  %x = load float, float addrspace(1)* %a.idx.2, align 4
  store float 0.0, float addrspace(1)* %a.idx.3, align 4

  ret float %x
}

; Here we have four stores, with an aliasing load before the last one.  We
; could vectorize two of the stores before the load (although we currently
; don't), but the important thing is that we *don't* sink the store to
; a[idx + 1] below the load.
;
; CHECK-LABEL: @insert_store_point_alias_ooo
; CHECK: store float
; CHECK-SAME: %a.idx.3
; CHECK: store float
; CHECK-SAME: %a.idx.1
; CHECK: store float
; CHECK-SAME: %a.idx.2
; CHECK: load float, float addrspace(1)* %a.idx.2
; CHECK: store float
; CHECK-SAME: %a.idx
define float @insert_store_point_alias_ooo(float addrspace(1)* nocapture %a, i64 %idx) {
  %a.idx = getelementptr inbounds float, float addrspace(1)* %a, i64 %idx
  %a.idx.1 = getelementptr inbounds float, float addrspace(1)* %a.idx, i64 1
  %a.idx.2 = getelementptr inbounds float, float addrspace(1)* %a.idx.1, i64 1
  %a.idx.3 = getelementptr inbounds float, float addrspace(1)* %a.idx.2, i64 1

  store float 0.0, float addrspace(1)* %a.idx.3, align 4
  store float 0.0, float addrspace(1)* %a.idx.1, align 4
  store float 0.0, float addrspace(1)* %a.idx.2, align 4
  %x = load float, float addrspace(1)* %a.idx.2, align 4
  store float 0.0, float addrspace(1)* %a.idx, align 4

  ret float %x
}

attributes #0 = { nounwind }
