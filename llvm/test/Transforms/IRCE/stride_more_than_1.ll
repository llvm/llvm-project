; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,loop(irce)' -S < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_03: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_04: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_05: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_06: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_07: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_08: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; IV = 0; IV <s 100; IV += 7; 0 <= Len <= 50. IRCE is allowed.
define void @test_01(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_01(
; CHECK:      entry:
; CHECK-NEXT:   %exit.mainloop.at = load i32, i32* %a_len_ptr
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preheader, label %main.pseudo.exit
; CHECK:      loop.preheader:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:   %idx.next = add i32 %idx, 7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp slt i32 %idx.next, 100
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp slt i32 %idx.next, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop, label %main.exit.selector
; CHECK:      main.exit.selector:
; CHECK-NEXT:   %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp slt i32 %idx.next.lcssa, 100
; CHECK-NEXT:   br i1 [[COND3]], label %main.pseudo.exit, label %exit
; CHECK:      main.pseudo.exit:
; CHECK-NEXT:   %idx.copy = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    %indvar.end = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    br label %postloop
; CHECK:      postloop:
; CHECK-NEXT:   br label %loop.postloop
; CHECK:      loop.postloop:
; CHECK-NEXT:   %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:   %idx.next.postloop = add i32 %idx.postloop, 7
; CHECK-NEXT:   %abc.postloop = icmp slt i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:   br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.postloop:
; CHECK-NEXT:   %addr.postloop = getelementptr i32, i32* %arr, i32 %idx.postloop
; CHECK-NEXT:   store i32 0, i32* %addr.postloop
; CHECK-NEXT:   %next.postloop = icmp slt i32 %idx.next.postloop, 100
; CHECK-NEXT:   br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; IV = 0; IV <s MAX_INT - 7; IV += 7; 0 <= Len <= 50. IRCE is allowed.
define void @test_02(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_02(
; CHECK:      entry:
; CHECK-NEXT:   %exit.mainloop.at = load i32, i32* %a_len_ptr
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preheader, label %main.pseudo.exit
; CHECK:      loop.preheader:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:   %idx.next = add i32 %idx, 7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp slt i32 %idx.next, 2147483640
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp slt i32 %idx.next, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop, label %main.exit.selector
; CHECK:      main.exit.selector:
; CHECK-NEXT:   %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp slt i32 %idx.next.lcssa, 2147483640
; CHECK-NEXT:   br i1 [[COND3]], label %main.pseudo.exit, label %exit
; CHECK:      main.pseudo.exit:
; CHECK-NEXT:   %idx.copy = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    %indvar.end = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    br label %postloop
; CHECK:      postloop:
; CHECK-NEXT:   br label %loop.postloop
; CHECK:      loop.postloop:
; CHECK-NEXT:   %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:   %idx.next.postloop = add i32 %idx.postloop, 7
; CHECK-NEXT:   %abc.postloop = icmp slt i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:   br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.postloop:
; CHECK-NEXT:   %addr.postloop = getelementptr i32, i32* %arr, i32 %idx.postloop
; CHECK-NEXT:   store i32 0, i32* %addr.postloop
; CHECK-NEXT:   %next.postloop = icmp slt i32 %idx.next.postloop, 2147483640
; CHECK-NEXT:   br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 2147483640
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; IV = 0; IV <s MAX_INT; IV += 7; 0 <= Len <= MAX_INT - 7. This is the greatest
; value of Len for which IRCE is allowed.
define void @test_03(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_03(
; CHECK:      entry:
; CHECK-NEXT:   %exit.mainloop.at = load i32, i32* %a_len_ptr
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preheader, label %main.pseudo.exit
; CHECK:      loop.preheader:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:   %idx.next = add i32 %idx, 7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp slt i32 %idx.next, 2147483647
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp slt i32 %idx.next, %exit.mainloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop, label %main.exit.selector
; CHECK:      main.exit.selector:
; CHECK-NEXT:   %idx.next.lcssa = phi i32 [ %idx.next, %in.bounds ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp slt i32 %idx.next.lcssa, 2147483647
; CHECK-NEXT:   br i1 [[COND3]], label %main.pseudo.exit, label %exit
; CHECK:      main.pseudo.exit:
; CHECK-NEXT:   %idx.copy = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    %indvar.end = phi i32 [ 0, %entry ], [ %idx.next.lcssa, %main.exit.selector ]
; CHECK-NEXT:    br label %postloop
; CHECK:      postloop:
; CHECK-NEXT:   br label %loop.postloop
; CHECK:      loop.postloop:
; CHECK-NEXT:   %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:   %idx.next.postloop = add i32 %idx.postloop, 7
; CHECK-NEXT:   %abc.postloop = icmp slt i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:   br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.postloop:
; CHECK-NEXT:   %addr.postloop = getelementptr i32, i32* %arr, i32 %idx.postloop
; CHECK-NEXT:   store i32 0, i32* %addr.postloop
; CHECK-NEXT:   %next.postloop = icmp slt i32 %idx.next.postloop, 2147483647
; CHECK-NEXT:   br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !1
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 7
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

; IV = 0; IV <s MAX_INT; IV += 7; 0 <= Len <= MAX_INT - 6. IRCE is not allowed,
; because we cannot guarantee that IV + 7 will not exceed MAX_INT.
; Negative test.
define void @test_04(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_04(

entry:
  %len = load i32, i32* %a_len_ptr, !range !2
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 7
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

; IV = 100; IV >s -1; IV -= 7; 0 <= Len <= 50. IRCE is allowed.
define void @test_05(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_05(
; CHECK:      entry:
; CHECK-NEXT:   %len = load i32, i32* %a_len_ptr
; CHECK-NEXT:   %exit.preloop.at = add i32 %len, -1
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp sgt i32 100, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:      loop.preloop.preheader:
; CHECK-NEXT:   br label %loop.preloop
; CHECK:      mainloop:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:   %idx.next = add i32 %idx, -7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %len
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp sgt i32 %idx.next, -1
; CHECK-NEXT:   br i1 %next, label %loop, label %exit.loopexit
; CHECK:      loop.preloop:
; CHECK-NEXT:   %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 100, %loop.preloop.preheader ]
; CHECK-NEXT:   %idx.next.preloop = add i32 %idx.preloop, -7
; CHECK-NEXT:   %abc.preloop = icmp slt i32 %idx.preloop, %len
; CHECK-NEXT:   br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.preloop:
; CHECK-NEXT:   %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:   store i32 0, i32* %addr.preloop
; CHECK-NEXT:   %next.preloop = icmp sgt i32 %idx.next.preloop, -1
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp sgt i32 %idx.next.preloop, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop.preloop, label %preloop.exit.selector
; CHECK:      preloop.exit.selector:
; CHECK-NEXT:   %idx.next.preloop.lcssa = phi i32 [ %idx.next.preloop, %in.bounds.preloop ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp sgt i32 %idx.next.preloop.lcssa, -1
; CHECK-NEXT:   br i1 [[COND3]], label %preloop.pseudo.exit, label %exit
; CHECK:      preloop.pseudo.exit:
; CHECK-NEXT:   %idx.preloop.copy = phi i32 [ 100, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   %indvar.end = phi i32 [ 100, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   br label %mainloop

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -7
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

; IV = MAX_INT - 7; IV >u 6; IV -= 7; 10 <= Len <= 50. IRCE is allowed.
define void @test_06(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_06(
; CHECK:      entry:
; CHECK-NEXT:   %len = load i32, i32* %a_len_ptr
; CHECK-NEXT:   %exit.preloop.at = add i32 %len, -1
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp ugt i32 2147483640, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:      loop.preloop.preheader:
; CHECK-NEXT:   br label %loop.preloop
; CHECK:      mainloop:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:   %idx.next = add i32 %idx, -7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %len
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp ugt i32 %idx.next, 6
; CHECK-NEXT:   br i1 %next, label %loop, label %exit.loopexit
; CHECK:      loop.preloop:
; CHECK-NEXT:   %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 2147483640, %loop.preloop.preheader ]
; CHECK-NEXT:   %idx.next.preloop = add i32 %idx.preloop, -7
; CHECK-NEXT:   %abc.preloop = icmp slt i32 %idx.preloop, %len
; CHECK-NEXT:   br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.preloop:
; CHECK-NEXT:   %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:   store i32 0, i32* %addr.preloop
; CHECK-NEXT:   %next.preloop = icmp ugt i32 %idx.next.preloop, 6
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp ugt i32 %idx.next.preloop, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop.preloop, label %preloop.exit.selector
; CHECK:      preloop.exit.selector:
; CHECK-NEXT:   %idx.next.preloop.lcssa = phi i32 [ %idx.next.preloop, %in.bounds.preloop ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp ugt i32 %idx.next.preloop.lcssa, 6
; CHECK-NEXT:   br i1 [[COND3]], label %preloop.pseudo.exit, label %exit
; CHECK:      preloop.pseudo.exit:
; CHECK-NEXT:   %idx.preloop.copy = phi i32 [ 2147483640, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   %indvar.end = phi i32 [ 2147483640, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   br label %mainloop

entry:
  %len = load i32, i32* %a_len_ptr, !range !3
  br label %loop

loop:
  %idx = phi i32 [ 2147483640, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ugt i32 %idx.next, 6
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; IV = MAX_INT - 7; IV >u 5; IV -= 7; 10 <= Len <= 50. IRCE is not allowed,
; because we can cross the 0 border.
define void @test_07(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_07(

entry:
  %len = load i32, i32* %a_len_ptr, !range !3
  br label %loop

loop:
  %idx = phi i32 [ 2147483640, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ugt i32 %idx.next, 5
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; IV = MAX_INT; IV >u 6; IV -= 7; 10 <= Len <= 50. IRCE is allowed.
define void @test_08(i32* %arr, i32* %a_len_ptr) {

; CHECK:      @test_08(
; CHECK:      entry:
; CHECK-NEXT:   %len = load i32, i32* %a_len_ptr
; CHECK-NEXT:   %exit.preloop.at = add i32 %len, -1
; CHECK-NEXT:   [[COND1:%[^ ]+]] = icmp ugt i32 2147483647, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND1]], label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:      loop.preloop.preheader:
; CHECK-NEXT:   br label %loop.preloop
; CHECK:      mainloop:
; CHECK-NEXT:   br label %loop
; CHECK:      loop:
; CHECK-NEXT:   %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:   %idx.next = add i32 %idx, -7
; CHECK-NEXT:   %abc = icmp slt i32 %idx, %len
; CHECK-NEXT:   br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:      in.bounds:
; CHECK-NEXT:   %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:   store i32 0, i32* %addr
; CHECK-NEXT:   %next = icmp ugt i32 %idx.next, 6
; CHECK-NEXT:   br i1 %next, label %loop, label %exit.loopexit
; CHECK:      loop.preloop:
; CHECK-NEXT:   %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 2147483647, %loop.preloop.preheader ]
; CHECK-NEXT:   %idx.next.preloop = add i32 %idx.preloop, -7
; CHECK-NEXT:   %abc.preloop = icmp slt i32 %idx.preloop, %len
; CHECK-NEXT:   br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit
; CHECK:      in.bounds.preloop:
; CHECK-NEXT:   %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:   store i32 0, i32* %addr.preloop
; CHECK-NEXT:   %next.preloop = icmp ugt i32 %idx.next.preloop, 6
; CHECK-NEXT:   [[COND2:%[^ ]+]] = icmp ugt i32 %idx.next.preloop, %exit.preloop.at
; CHECK-NEXT:   br i1 [[COND2]], label %loop.preloop, label %preloop.exit.selector
; CHECK:      preloop.exit.selector:
; CHECK-NEXT:   %idx.next.preloop.lcssa = phi i32 [ %idx.next.preloop, %in.bounds.preloop ]
; CHECK-NEXT:   [[COND3:%[^ ]+]] = icmp ugt i32 %idx.next.preloop.lcssa, 6
; CHECK-NEXT:   br i1 [[COND3]], label %preloop.pseudo.exit, label %exit
; CHECK:      preloop.pseudo.exit:
; CHECK-NEXT:   %idx.preloop.copy = phi i32 [ 2147483647, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   %indvar.end = phi i32 [ 2147483647, %entry ], [ %idx.next.preloop.lcssa, %preloop.exit.selector ]
; CHECK-NEXT:   br label %mainloop

entry:
  %len = load i32, i32* %a_len_ptr, !range !3
  br label %loop

loop:
  %idx = phi i32 [ 2147483647, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -7
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ugt i32 %idx.next, 6
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
!1 = !{i32 0, i32 2147483640}
!2 = !{i32 0, i32 2147483641}
!3 = !{i32 10, i32 50}
