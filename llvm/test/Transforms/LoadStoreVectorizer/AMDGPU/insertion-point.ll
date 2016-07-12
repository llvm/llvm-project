; RUN: opt -mtriple=amdgcn-amd-amdhsa -basicaa -load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; Check relative position of the inserted vector load relative to the existing
; adds. Vectorized loads should be inserted at the position of the first load.

; CHECK-LABEL: @insert_load_point(
; CHECK: %z = add i32 %x, 4
; CHECK: load <2 x float>
; CHECK: %w = add i32 %y, 9
; CHECK: %foo = add i32 %z, %w
define void @insert_load_point(float addrspace(1)* nocapture %a, float addrspace(1)* nocapture %b, float addrspace(1)* nocapture readonly %c, i64 %idx, i32 %x, i32 %y) #0 {
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
define void @insert_store_point(float addrspace(1)* nocapture %a, float addrspace(1)* nocapture %b, float addrspace(1)* nocapture readonly %c, i64 %idx, i32 %x, i32 %y) #0 {
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

attributes #0 = { nounwind }
