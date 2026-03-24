; NOTE: Pattern-only regression test for a cross-alloca indices crash.
; Distilled from promote-alloca-crash-ce6eff-min.ll.
; RUN: opt -mtriple=amdgcn--amdpal -disable-output -verify-each -passes=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 < %s

define internal amdgpu_cs void @cross_alloca_indices_crash() {
entry:
  %a = alloca [16 x i32], align 4, addrspace(5)
  %b = alloca [16 x i32], align 4, addrspace(5)
  br label %64

64:                                                ; preds = %entry, %64
  %m = phi i32 [ 0, %entry ], [ %m.next, %64 ]
  %b.ptr2 = getelementptr inbounds [16 x i32], ptr addrspace(5) %b, i32 0, i32 %m
  %b.val = load i32, ptr addrspace(5) %b.ptr2, align 4
  %idx = and i32 %b.val, 15
  %a.ptr1 = getelementptr inbounds [16 x i32], ptr addrspace(5) %a, i32 0, i32 %idx
  store i32 %m, ptr addrspace(5) %a.ptr1, align 4
  %m.next = add nuw nsw i32 %m, 1
  %map.done = icmp eq i32 %m.next, 15
  br i1 %map.done, label %.loopexit4, label %64

.loopexit4:                                       ; preds = %64
  br label %79

79:                                                ; preds = %.loopexit4, %.critedge
  %in = phi i32 [ 15, %.loopexit4 ], [ %in.next, %.critedge ]
  %in.next = add nsw i32 %in, -1
  %b.ptr3 = getelementptr inbounds [16 x i32], ptr addrspace(5) %b, i32 0, i32 %in.next
  %b.val2 = load i32, ptr addrspace(5) %b.ptr3, align 4
  %start = and i32 %b.val2, 15
  %start.plus2 = add i32 %start, 2
  %has.candidates = icmp ult i32 %start.plus2, 16
  br i1 %has.candidates, label %.lr.ph, label %.critedge

.lr.ph:                                           ; preds = %79, %104
  %cand = phi i32 [ %start.plus2, %79 ], [ %cand.next, %104 ]
  %best.idx = phi i32 [ %start, %79 ], [ %cand, %104 ]
  %a.ptr2 = getelementptr inbounds [16 x i32], ptr addrspace(5) %a, i32 0, i32 %cand
  %a.val = load i32, ptr addrspace(5) %a.ptr2, align 4
  %is.free = icmp eq i32 %a.val, -1
  br i1 %is.free, label %109, label %.critedge

104:                                               ; preds = %109
  %cand.next = add i32 %cand, 2
  %cont.search = icmp ult i32 %cand.next, 16
  br i1 %cont.search, label %.lr.ph, label %.critedge

109:                                               ; preds = %.lr.ph
  %a.old.ptr = getelementptr inbounds [16 x i32], ptr addrspace(5) %a, i32 0, i32 %best.idx
  store i32 -1, ptr addrspace(5) %a.old.ptr, align 4
  store i32 %in.next, ptr addrspace(5) %a.ptr2, align 4
  br label %104

.critedge:                                        ; preds = %.lr.ph, %104, %79
  %more = icmp sgt i32 %in, 1
  br i1 %more, label %79, label %.loopexit2

.loopexit2:                                       ; preds = %.critedge, %.loopexit4
  br label %116

116:                                               ; preds = %.loopexit2, %128
  %p = phi i32 [ 0, %.loopexit2 ], [ %p.next, %128 ]
  %a.ptr3 = getelementptr inbounds [16 x i32], ptr addrspace(5) %a, i32 0, i32 %p
  %a.elem = load i32, ptr addrspace(5) %a.ptr3, align 4
  %skip = icmp eq i32 %a.elem, -1
  br i1 %skip, label %128, label %121

121:                                               ; preds = %116
  %b.ptr4 = getelementptr inbounds [16 x i32], ptr addrspace(5) %b, i32 0, i32 %a.elem
  %b.elem = load i32, ptr addrspace(5) %b.ptr4, align 4
  store i32 %b.elem, ptr addrspace(5) %b.ptr4, align 4
  br label %128

128:                                               ; preds = %121, %116
  %p.next = add nuw nsw i32 %p, 1
  %final.done = icmp eq i32 %p.next, 16
  br i1 %final.done, label %.loopexit, label %116

.loopexit:                                        ; preds = %128
  ret void
}
