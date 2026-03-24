; RUN: opt -mtriple=amdgcn--amdpal -disable-output -verify-each -passes=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 < %s
;
; Regression test for stale cross-alloca index analysis.
; The first loop builds entries in alloca %a using indices loaded from
; alloca %b. The second loop then indexes %b using values loaded back from %a.
; PromoteAlloca must not reuse invalid per-alloca index state across those
; two allocas.
;
; This is a pattern-only test: the pass only needs to finish without
; crashing while walking and rewriting the mixed %a/%b access graph.

define internal amdgpu_cs void @cross_alloca_indices_crash() {
entry:
  %a = alloca [4 x i32], align 4, addrspace(5)
  %b = alloca [4 x i32], align 4, addrspace(5)
  br label %map

  ; Build %a entries using indices loaded from %b.
map:                                               ; preds = %entry, %map
  %m = phi i32 [ 0, %entry ], [ %m.next, %map ]
  %b.ptr1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %m
  %b.val = load i32, ptr addrspace(5) %b.ptr1, align 4
  %idx = and i32 %b.val, 3
  %a.ptr1 = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 %idx
  store i32 %m, ptr addrspace(5) %a.ptr1, align 4
  %m.next = add nuw nsw i32 %m, 1
  %map.done = icmp eq i32 %m.next, 4
  br i1 %map.done, label %consume, label %map

  ; Revisit %b through indices loaded back from %a.
consume:                                           ; preds = %map, %consume
  %p = phi i32 [ 0, %map ], [ %p.next, %consume ]
  %a.ptr2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %a, i32 0, i32 %p
  %a.elem = load i32, ptr addrspace(5) %a.ptr2, align 4
  %b.ptr2 = getelementptr inbounds [4 x i32], ptr addrspace(5) %b, i32 0, i32 %a.elem
  %b.elem = load i32, ptr addrspace(5) %b.ptr2, align 4
  store i32 %b.elem, ptr addrspace(5) %b.ptr2, align 4
  %p.next = add nuw nsw i32 %p, 1
  %consume.done = icmp eq i32 %p.next, 4
  br i1 %consume.done, label %exit, label %consume

exit:                                              ; preds = %consume
  ret void
}
