; RUN: opt -S -passes=jump-threading %s | FileCheck %s

; Value of predicate known on all inputs (trivial case)
; Note: InstCombine/EarlyCSE would also get this case
define void @test(ptr %p, ptr %addr) {
; CHECK-LABEL: @test
entry:
  %cmp0 = icmp eq ptr %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
; CHECK-LABEL: loop:
; CHECK-NEXT: phi
; CHECK-NEXT: br label %loop
  %p1 = phi ptr [%p, %entry], [%p1, %loop]
  %cmp1 = icmp eq ptr %p1, null
  br i1 %cmp1, label %exit, label %loop
exit:
  ret void
}

; Value of predicate known on all inputs (non-trivial)
define void @test2(ptr %p) {
; CHECK-LABEL: @test2
entry:
  %cmp0 = icmp eq ptr %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
  %p1 = phi ptr [%p, %entry], [%p2, %backedge]
  %cmp1 = icmp eq ptr %p1, null
  br i1 %cmp1, label %exit, label %backedge
backedge:
; CHECK-LABEL: backedge:
; CHECK-NEXT: phi
; CHECK-NEXT: load
; CHECK-NEXT: cmp
; CHECK-NEXT: br
; CHECK-DAG: label %backedge
  %p2 = load ptr, ptr %p1
  %cmp2 = icmp eq ptr %p2, null
  br i1 %cmp2, label %exit, label %loop
exit:
  ret void
}

; If the inputs don't branch the same way, we can't rewrite
; Well, we could unroll this loop exactly twice, but that's
; a different transform.
define void @test_mixed(ptr %p) {
; CHECK-LABEL: @test_mixed
entry:
  %cmp0 = icmp eq ptr %p, null
  br i1 %cmp0, label %exit, label %loop
loop:
; CHECK-LABEL: loop:
; CHECK-NEXT: phi
; CHECK-NEXT: %cmp1 = icmp
; CHECK-NEXT: br i1 %cmp1
  %p1 = phi ptr [%p, %entry], [%p1, %loop]
  %cmp1 = icmp ne ptr %p1, null
  br i1 %cmp1, label %exit, label %loop
exit:
  ret void
}

; The eq predicate is always true if we go through the path from
; L1 to L3, no matter the phi result %t5 is on the lhs or rhs of
; the predicate.
declare void @goo()
declare void @hoo()

define void @test3(i32 %m, ptr %t1) {
L1:
  %t0 = add i32 %m, 7
  %t2 = load ptr, ptr %t1, align 8
; CHECK-LABEL: @test3
; CHECK: %t3 = icmp eq ptr %t2, null
; CHECK: br i1 %t3, label %[[LABEL2:.*]], label %[[LABEL1:.*]]

  %t3 = icmp eq ptr %t2, null
  br i1 %t3, label %L3, label %L2

; CHECK: [[LABEL1]]:
; CHECK-NEXT: %t4 = load i32, ptr %t2, align 4
L2:
  %t4 = load i32, ptr %t2, align 4
  br label %L3

L3:
  %t5 = phi i32 [ %t0, %L1 ], [ %t4, %L2 ]
  %t6 = icmp eq i32 %t0, %t5
  br i1 %t6, label %L4, label %L5

; CHECK: [[LABEL2]]:
; CHECK-NEXT: call void @goo()
L4:
  call void @goo()
  ret void

L5:
  call void @hoo()
  ret void
}
