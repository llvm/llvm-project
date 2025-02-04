; RUN: opt -S -passes=callsite-splitting < %s | FileCheck --check-prefix=CHECK %s
; RUN: opt -S -passes=callsite-splitting -callsite-splitting-duplication-threshold=0 < %s | FileCheck --check-prefix=NODUP %s

; Instructions before a call that will be pushed to its predecessors
; with uses after the callsite, must be patched up as PHI nodes in
; the join block.
define ptr @test_split_branch_phi(ptr %ptrarg, i32 %i) {
Header:
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                    ; preds = %Header
  %arrayidx = getelementptr inbounds i32, ptr %ptrarg, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %tobool1 = icmp ne i32 %0, 0
  br i1 %tobool1, label %CallSite, label %End

CallSite:                                          ; preds = %TBB, %Header
  %somepointer = getelementptr i32, ptr %ptrarg, i64 18
  call void @bar(ptr %ptrarg, i32 %i)
  br label %End

End:                                           ; preds = %CallSite, %TBB
  %somepointerphi = phi ptr [ %somepointer, %CallSite ], [ null, %TBB ]
  ret ptr %somepointerphi
}
; NODUP-LABEL: test_split_branch_phi
; NODUP-NOT: split
; CHECK-LABEL: Header.split
; CHECK: %[[V1:somepointer[0-9]+]] = getelementptr i32, ptr %ptrarg, i64 18
; CHECK: call void @bar(ptr null, i32 %i)
; CHECK: br label %CallSite
; CHECK-LABEL: TBB.split:
; CHECK: %[[V2:somepointer[0-9]+]] = getelementptr i32, ptr %ptrarg, i64 18
; CHECK: call void @bar(ptr nonnull %ptrarg, i32 %i)
; CHECK: br label %CallSite
; CHECK: CallSite:
; CHECK: phi ptr [ %[[V1]], %Header.split ], [ %[[V2]], %TBB.split ]


define void @split_branch_no_extra_phi(ptr %ptrarg, i32 %i) {
Header:
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                    ; preds = %Header
  %arrayidx = getelementptr inbounds i32, ptr %ptrarg, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %tobool1 = icmp ne i32 %0, 0
  br i1 %tobool1, label %CallSite, label %End

CallSite:                                          ; preds = %TBB, %Header
  %i.add = add i32 %i, 99
  call void @bar(ptr %ptrarg, i32 %i.add)
  br label %End

End:                                           ; preds = %CallSite, %TBB
  ret void
}
; NODUP-LABEL: split_branch_no_extra_phi
; NODUP-NOT: split
; CHECK-LABEL: split_branch_no_extra_phi
; CHECK-LABEL: Header.split
; CHECK: %[[V1:.+]] = add i32 %i, 99
; CHECK: call void @bar(ptr null, i32 %[[V1]])
; CHECK: br label %CallSite
; CHECK-LABEL: TBB.split:
; CHECK: %[[V2:.+]] = add i32 %i, 99
; CHECK: call void @bar(ptr nonnull %ptrarg, i32 %[[V2]])
; CHECK: br label %CallSite
; CHECK: CallSite:
; CHECK-NOT: phi


; In this test case, the codesize cost of the instructions before the call to
; bar() is equal to the default DuplicationThreshold of 5, because calls are
; more expensive.
define void @test_no_split_threshold(ptr %ptrarg, i32 %i) {
Header:
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                    ; preds = %Header
  %arrayidx = getelementptr inbounds i32, ptr %ptrarg, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %tobool1 = icmp ne i32 %0, 0
  br i1 %tobool1, label %CallSite, label %End

CallSite:                                          ; preds = %TBB, %Header
  %i2 = add i32 %i, 10
  call void @bari(i32 %i2)
  call void @bari(i32 %i2)
  call void @bar(ptr %ptrarg, i32 %i2)
  br label %End

End:                                           ; preds = %CallSite, %TBB
  ret void
}
; NODUP-LABEL: test_no_split_threshold
; NODUP-NOT: split
; CHECK-LABEL: test_no_split_threshold
; CHECK-NOT: split
; CHECK-LABEL: CallSite:
; CHECK: call void @bar(ptr %ptrarg, i32 %i2)

; In this test case, the phi node %l in CallSite should be removed, as after
; moving the call to the split blocks we can use the values directly.
define void @test_remove_unused_phi(ptr %ptrarg, i32 %i) {
Header:
  %l1 = load i32, ptr undef, align 16
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                    ; preds = %Header
  %arrayidx = getelementptr inbounds i32, ptr %ptrarg, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %l2 = load i32, ptr undef, align 16
  %tobool1 = icmp ne i32 %0, 0
  br i1 %tobool1, label %CallSite, label %End

CallSite:                                          ; preds = %TBB, %Header
  %l = phi i32 [ %l1, %Header ], [ %l2, %TBB ]
  call void @bar(ptr %ptrarg, i32 %l)
  br label %End

End:                                           ; preds = %CallSite, %TBB
  ret void
}
; NODUP-LABEL: test_remove_unused_phi
; NODUP-NOT: split
; CHECK-LABEL: test_remove_unused_phi
; CHECK-LABEL: Header.split
; CHECK: call void @bar(ptr null, i32 %l1)
; CHECK: br label %CallSite
; CHECK-LABEL: TBB.split:
; CHECK: call void @bar(ptr nonnull %ptrarg, i32 %l2)
; CHECK: br label %CallSite
; CHECK-LABEL: CallSite:
; CHECK-NOT: phi

; In this test case, we need to insert a new PHI node in TailBB to combine
; the loads we moved to the predecessors.
define void @test_add_new_phi(ptr %ptrarg, i32 %i, i1 %arg) {
Header:
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:
  br i1 %arg, label %CallSite, label %End

CallSite:
  %arrayidx112 = getelementptr inbounds i32, ptr undef, i64 1
  %0 = load i32, ptr %arrayidx112, align 4
  call void @bar(ptr %ptrarg, i32 %i)
  %sub = sub nsw i32 %0, undef
  br label %End

End:                                           ; preds = %CallSite, %TBB
  ret void
}
; NODUP-LABEL: test_add_new_phi
; NODUP-NOT: split
; CHECK-LABEL: test_add_new_phi
; CHECK-LABEL: Header.split
; CHECK: %[[V1:.+]] = load i32, ptr
; CHECK: call void @bar(ptr null, i32 %i)
; CHECK: br label %CallSite
; CHECK-LABEL: TBB.split:
; CHECK: %[[V2:.+]] = load i32, ptr
; CHECK: call void @bar(ptr nonnull %ptrarg, i32 %i)
; CHECK: br label %CallSite
; CHECK-LABEL: CallSite:
; CHECK-NEXT: %[[V3:.+]] = phi i32 [ %[[V1]], %Header.split ], [ %[[V2]], %TBB.split ]
; CHECK: %sub = sub nsw i32 %[[V3]], undef

define i32 @test_firstnophi(ptr %a, i32 %v) {
Header:
  %tobool1 = icmp eq ptr %a, null
  br i1 %tobool1, label %Tail, label %TBB

TBB:
  %cmp = icmp eq i32 %v, 1
  br i1 %cmp, label %Tail, label %End

Tail:
  %p = phi i32[1,%Header], [2, %TBB]
  store i32 %v, ptr %a
  %r = call i32 @callee(ptr %a, i32 %v, i32 %p)
  ret i32 %r

End:
  ret i32 %v
}
; NODUP-LABEL: @test_firstnophi
; NODUP-NOT: split:
; CHECK-LABEL: @test_firstnophi
; CHECK-LABEL: Header.split:
; CHECK-NEXT: store i32 %v, ptr %a
; CHECK-NEXT: %[[CALL1:.*]] = call i32 @callee(ptr null, i32 %v, i32 1)
; CHECK-NEXT: br label %Tail
; CHECK-LABEL: TBB.split:
; CHECK-NEXT: store i32 %v, ptr %a
; CHECK-NEXT: %[[CALL2:.*]] = call i32 @callee(ptr nonnull %a, i32 1, i32 2)
; CHECK-NEXT: br label %Tail
; CHECK-LABEL: Tail:
; CHECK: %[[MERGED:.*]] = phi i32 [ %[[CALL1]], %Header.split ], [ %[[CALL2]], %TBB.split ]
; CHECK: ret i32 %[[MERGED]]
define i32 @callee(ptr %a, i32 %v, i32 %p) {
    ret i32 0
}

define void @test_no_remove_used_phi(ptr %ptrarg, i32 %i) {
Header:
  %l1 = load i32, ptr undef, align 16
  %tobool = icmp ne ptr %ptrarg, null
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                    ; preds = %Header
  %arrayidx = getelementptr inbounds i32, ptr %ptrarg, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %l2 = load i32, ptr undef, align 16
  %tobool1 = icmp ne i32 %0, 0
  br i1 %tobool1, label %CallSite, label %End

CallSite:                                          ; preds = %TBB, %Header
  %l = phi i32 [ %l1, %Header ], [ %l2, %TBB ]
  call void @bar(ptr %ptrarg, i32 %l)
  call void @bari(i32 %l)
  br label %End

End:                                           ; preds = %CallSite, %TBB
  ret void
}
; NODUP-LABEL: @test_no_remove_used_phi
; NODUP-NOT: split
; CHECK-LABEL: @test_no_remove_used_phi
; CHECK-LABEL: Header.split:
; CHECK: call void @bar(ptr null, i32 %l1)
; CHECK-NEXT: br label %CallSite
; CHECK-LABEL: TBB.split:
; CHECK: call void @bar(ptr nonnull %ptrarg, i32 %l2)
; CHECK-NEXT: br label %CallSite
; CHECK-LABEL: CallSite:
; CHECK-NEXT:  %l = phi i32 [ %l1, %Header.split ], [ %l2, %TBB.split ]
; CHECK: call void @bari(i32 %l)

define void @bar(ptr, i32) {
    ret void
}

define  void @bari(i32) {
    ret void
}
