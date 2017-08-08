; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; Iterate from 0 to SINT_MAX, check that the post-loop is generated.
define void @test_01(i32* %arr, i32* %a_len_ptr) {

; CHECK:       test_01(
; CHECK:       entry:
; CHECK-NEXT:    %exit.mainloop.at = load i32, i32* %a_len_ptr
; CHECK:       loop:
; CHECK-NEXT:    %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:    %idx.next = add i32 %idx, 1
; CHECK-NEXT:    %abc = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:    br i1 true, label %in.bounds,
; CHECK:       in.bounds:
; CHECK-NEXT:    %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:    store i32 0, i32* %addr
; CHECK-NEXT:    %next = icmp slt i32 %idx.next, 2147483647
; CHECK-NEXT:    [[COND:%[^ ]+]] = icmp slt i32 %idx.next, %exit.mainloop.at
; CHECK-NEXT:    br i1 [[COND]], label %loop, label %main.exit.selector
; CHECK:       main.pseudo.exit:
; CHECK-NEXT:    %idx.copy = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    %indvar.end = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    br label %postloop
; CHECK:       postloop:
; CHECK-NEXT:    br label %loop.postloop
; CHECK:       loop.postloop:
; CHECK-NEXT:    %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:    %idx.next.postloop = add i32 %idx.postloop, 1
; CHECK-NEXT:    %abc.postloop = icmp slt i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:    br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit
; CHECK:       in.bounds.postloop:
; CHECK-NEXT:    %addr.postloop = getelementptr i32, i32* %arr, i32 %idx.postloop
; CHECK-NEXT:    store i32 0, i32* %addr.postloop
; CHECK-NEXT:    %next.postloop = icmp slt i32 %idx.next.postloop, 2147483647
; CHECK-NEXT:    br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 2147483647
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Iterate from SINT_MAX to 0, check that the pre-loop is generated.
define void @test_02(i32* %arr, i32* %a_len_ptr) {

; CHECK:      test_02(
; CHECK:      entry:
; CHECK-NEXT:   %len = load i32, i32* %a_len_ptr, !range !0
; CHECH-NEXT:    br i1 true, label %loop.preloop.preheader
; CHECK:      mainloop:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:   %idx.next = add i32 %idx, -1
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %len
; CHECK-NEXT:   br i1 true, label %in.bounds
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp sgt i32 %idx.next, -1
; CHECK-NEXT:   br i1 %next, label %loop, label %exit.loopexit
; CHECK:      loop.preloop:
; CHECK-NEXT:   %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 2147483647, %loop.preloop.preheader ]
; CHECK-NEXT:   %idx.next.preloop = add i32 %idx.preloop, -1
; CHECK-NEXT:   %abc.preloop = icmp slt i32 %idx.preloop, %len
; CHECK-NEXT:   br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.preloop:
; CHECK-NEXT:   %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:   store i32 0, i32* %addr.preloop
; CHECK-NEXT:   %next.preloop = icmp sgt i32 %idx.next.preloop, -1
; CHECK-NEXT:   [[COND:%[^ ]+]] = icmp sgt i32 %idx.next.preloop, -1
; CHECK-NEXT:   br i1 [[COND]], label %loop.preloop, label %preloop.exit.selector

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 2147483647, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp sgt i32 %idx.next, -1
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
