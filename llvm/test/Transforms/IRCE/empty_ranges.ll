; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S

; Make sure that IRCE doesn't apply in case of empty ranges.
; (i + 30 < 40) if i in [-30, 10).
; Intersected with iteration space, it is [0, 10).
; (i - 60 < 40) if i in [60 , 100).
; The intersection with safe iteration space is the empty range [60, 10).
; It is better to eliminate one range check than attempt to eliminate both given
; that we will never go to the main loop in the latter case and basically
; only duplicate code with no benefits.

define void @test_01(ptr %arr, ptr %a_len_ptr) #0 {

; CHECK-LABEL: test_01(
; CHECK-NOT:   preloop
; CHECK:       entry:
; CHECK-NEXT:    br i1 true, label %loop.preheader, label %main.pseudo.exit
; CHECK:       in.bounds.1:
; CHECK-NEXT:    %addr = getelementptr i32, ptr %arr, i32 %idx
; CHECK-NEXT:    store i32 0, ptr %addr
; CHECK-NEXT:    %off1 = add i32 %idx, 30
; CHECK-NEXT:    %c2 = icmp slt i32 %off1, 40
; CHECK-NEXT:    br i1 true, label %in.bounds.2, label %exit.loopexit2
; CHECK:       in.bounds.2:
; CHECK-NEXT:    %off2 = add i32 %idx, -60
; CHECK-NEXT:    %c3 = icmp slt i32 %off2, 40
; CHECK-NEXT:    br i1 %c3, label %in.bounds.3, label %exit.loopexit2
; CHECK:       in.bounds.3:
; CHECK-NEXT:    %next = icmp ult i32 %idx.next, 100
; CHECK-NEXT:    [[COND1:%[^ ]+]] = icmp ult i32 %idx.next, 10
; CHECK-NEXT:    br i1 [[COND1]], label %loop, label %main.exit.selector
; CHECK:       main.exit.selector:
; CHECK-NEXT:    %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds.3 ]
; CHECK-NEXT:    [[COND2:%[^ ]+]] = icmp ult i32 %idx.next.lcssa, 100
; CHECK-NEXT:    br i1 [[COND2]], label %main.pseudo.exit, label %exit
; CHECK:       postloop:

entry:
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds.3 ]
  %idx.next = add nsw nuw i32 %idx, 1
  %c1 = icmp slt i32 %idx, 20
  br i1 %c1, label %in.bounds.1, label %out.of.bounds

in.bounds.1:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %off1 = add i32 %idx, 30
  %c2 = icmp slt i32 %off1, 40
  br i1 %c2, label %in.bounds.2, label %exit

in.bounds.2:
  %off2 = add i32 %idx, -60
  %c3 = icmp slt i32 %off2, 40
  br i1 %c3, label %in.bounds.3, label %exit

in.bounds.3:
  %next = icmp ult i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}
