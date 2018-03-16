; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,loop(irce)' -S < %s 2>&1 | FileCheck %s

; Make sure we can eliminate range check with signed latch, unsigned IRC and
; positive offset. The safe iteration space is:
; No preloop,
; %exit.mainloop.at = smax (0, -1 - smax(12 - %len, -102)).
; Formula verification:
; %len = 10
; %exit.mainloop.at = 0
; %len = 50
; %exit.mainloop.at = 50 - 13 = 37.
; %len = 100
; %exit.mainloop.at = 100 - 13 = 87.
; %len = 150
; %exit.mainloop.at = 101.
; %len = SINT_MAX
; %exit.mainloop.at = 101

define void @test_01(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_01(
; CHECK-NOT:     preloop
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 12, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[SUB1]], -102
; CHECK-NEXT:      [[SMAX:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, [[SMAX]]
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp sgt i32 [[SUB2]], 0
; CHECK-NEXT:      %exit.mainloop.at = select i1 [[CMP2]], i32 [[SUB2]], i32 0
; CHECK-NEXT:      [[GOTO_LOOP:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[GOTO_LOOP]], label %loop.preheader, label %main.pseudo.exit
; CHECK:         loop
; CHECK:           br i1 true, label %in.bounds
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = add i32 %idx, 13
  %abc = icmp ult i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Make sure we can eliminate range check with signed latch, unsigned IRC and
; negative offset. The safe iteration space is:
; %exit.preloop.at = 13
; %exit.mainloop.at = smax(-1 - smax(smax(%len - SINT_MAX, -13) - 1 - %len, -102), 0)
; Formula verification:
; %len = 10
; %exit.mainloop.at = 0
; %len = 50
; %exit.mainloop.at = 63
; %len = 100
; %exit.mainloop.at = 101
; %len = 150
; %exit.mainloop.at = 101
; %len = SINT_MAX
; %exit.mainloop.at = 101

define void @test_02(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_02(
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[LEN_MINUS_SMAX:%[^ ]+]] = add i32 %len, -2147483647
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[LEN_MINUS_SMAX]], -13
; CHECK-NEXT:      [[SMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[LEN_MINUS_SMAX]], i32 -13
; CHECK-NEXT:      [[ADD1:%[^ ]+]] = add i32 [[SMAX1]], -1
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 [[ADD1]], %len
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp sgt i32 [[SUB1]], -102
; CHECK-NEXT:      [[SMAX2:%[^ ]+]] = select i1 [[CMP2]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, [[SMAX2]]
; CHECK-NEXT:      [[CMP3:%[^ ]+]] = icmp sgt i32 [[SUB2]], 0
; CHECK-NEXT:      %exit.mainloop.at = select i1 [[CMP3]], i32 [[SUB2]], i32 0
; CHECK-NEXT:      br i1 true, label %loop.preloop.preheader
; CHECK:         loop.preloop:
; CHECK-NEXT:      %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 0, %loop.preloop.preheader ]
; CHECK-NEXT:      %idx.next.preloop = add i32 %idx.preloop, 1
; CHECK-NEXT:      %idx.offset.preloop = sub i32 %idx.preloop, 13
; CHECK-NEXT:      %abc.preloop = icmp ult i32 %idx.offset.preloop, %len
; CHECK-NEXT:      br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit
; CHECK:         in.bounds.preloop:
; CHECK-NEXT:      %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:      store i32 0, i32* %addr.preloop
; CHECK-NEXT:      %next.preloop = icmp slt i32 %idx.next.preloop, 101
; CHECK-NEXT:      [[PRELOOP_COND:%[^ ]+]] = icmp slt i32 %idx.next.preloop, 13
; CHECK-NEXT:      br i1 [[PRELOOP_COND]], label %loop.preloop, label %preloop.exit.selector
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = sub i32 %idx, 13
  %abc = icmp ult i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Make sure we can eliminate range check with unsigned latch, signed IRC and
; positive offset. The safe iteration space is:
; No preloop,
; %exit.mainloop.at = -1 - umax(-2 - %len - smax(-1 - %len, -14), -102)
; Formula verification:
; %len = 10
; %exit.mainloop.at = 0
; %len = 50
; %exit.mainloop.at = 37
; %len = 100
; %exit.mainloop.at = 87
; %len = 150
; %exit.mainloop.at = 101
; %len = SINT_MAX
; %exit.mainloop.at = 101

define void @test_03(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_03(
; CHECK-NOT:     preloop
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 -2, %len
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[SUB2]], -14
; CHECK-NEXT:      [[SMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB2]], i32 -14
; CHECK-NEXT:      [[SUB3:%[^ ]+]] = sub i32 [[SUB1]], [[SMAX1]]
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp ugt i32 [[SUB3]], -102
; CHECK-NEXT:      [[UMAX1:%[^ ]+]] = select i1 [[CMP2]], i32 [[SUB3]], i32 -102
; CHECK-NEXT:      %exit.mainloop.at = sub i32 -1, [[UMAX1]]
; CHECK-NEXT:      [[CMP3:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[CMP3]], label %loop.preheader, label %main.pseudo.exit
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = add i32 %idx, 13
  %abc = icmp slt i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Make sure we can eliminate range check with unsigned latch, signed IRC and
; positive offset. The safe iteration space is:
; %exit.preloop.at = 13
; %exit.mainloop.at = -1 - umax(-14 - %len, -102)
; Formula verification:
; %len = 10
; %exit.mainloop.at = 23
; %len = 50
; %exit.mainloop.at = 63
; %len = 100
; %exit.mainloop.at = 101
; %len = 150
; %exit.mainloop.at = 101
; %len = SINT_MAX
; %exit.mainloop.at = 101

define void @test_04(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_04(
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 -14, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp ugt i32 [[SUB1]], -102
; CHECK-NEXT:      [[UMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      %exit.mainloop.at = sub i32 -1, [[UMAX1]]
; CHECK-NEXT:      br i1 true, label %loop.preloop.preheader
; CHECK:         in.bounds.preloop:
; CHECK-NEXT:      %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:      store i32 0, i32* %addr.preloop
; CHECK-NEXT:      %next.preloop = icmp ult i32 %idx.next.preloop, 101
; CHECK-NEXT:      [[PRELOOP_COND:%[^ ]+]] = icmp ult i32 %idx.next.preloop, 13
; CHECK-NEXT:      br i1 [[PRELOOP_COND]], label %loop.preloop, label %preloop.exit.selector
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = sub i32 %idx, 13
  %abc = icmp slt i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Signed latch, signed RC, positive offset. Same as test_01.
define void @test_05(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_05(
; CHECK-NOT:     preloop
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 12, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[SUB1]], -102
; CHECK-NEXT:      [[SMAX:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, [[SMAX]]
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp sgt i32 [[SUB2]], 0
; CHECK-NEXT:      %exit.mainloop.at = select i1 [[CMP2]], i32 [[SUB2]], i32 0
; CHECK-NEXT:      [[GOTO_LOOP:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[GOTO_LOOP]], label %loop.preheader, label %main.pseudo.exit
; CHECK:         loop
; CHECK:           br i1 true, label %in.bounds
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = add i32 %idx, 13
  %abc = icmp slt i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Signed latch, signed RC, negative offset. Same as test_02.
define void @test_06(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_06(
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[LEN_MINUS_SMAX:%[^ ]+]] = add i32 %len, -2147483647
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[LEN_MINUS_SMAX]], -13
; CHECK-NEXT:      [[SMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[LEN_MINUS_SMAX]], i32 -13
; CHECK-NEXT:      [[ADD1:%[^ ]+]] = add i32 [[SMAX1]], -1
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 [[ADD1]], %len
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp sgt i32 [[SUB1]], -102
; CHECK-NEXT:      [[SMAX2:%[^ ]+]] = select i1 [[CMP2]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, [[SMAX2]]
; CHECK-NEXT:      [[CMP3:%[^ ]+]] = icmp sgt i32 [[SUB2]], 0
; CHECK-NEXT:      %exit.mainloop.at = select i1 [[CMP3]], i32 [[SUB2]], i32 0
; CHECK-NEXT:      br i1 true, label %loop.preloop.preheader
; CHECK:         in.bounds.preloop:
; CHECK-NEXT:      %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:      store i32 0, i32* %addr.preloop
; CHECK-NEXT:      %next.preloop = icmp slt i32 %idx.next.preloop, 101
; CHECK-NEXT:      [[PRELOOP_COND:%[^ ]+]] = icmp slt i32 %idx.next.preloop, 13
; CHECK-NEXT:      br i1 [[PRELOOP_COND]], label %loop.preloop, label %preloop.exit.selector
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = sub i32 %idx, 13
  %abc = icmp slt i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Unsigned latch, Unsigned RC, negative offset. Same as test_03.
define void @test_07(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_07(
; CHECK-NOT:     preloop
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 -2, %len
; CHECK-NEXT:      [[SUB2:%[^ ]+]] = sub i32 -1, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp sgt i32 [[SUB2]], -14
; CHECK-NEXT:      [[SMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB2]], i32 -14
; CHECK-NEXT:      [[SUB3:%[^ ]+]] = sub i32 [[SUB1]], [[SMAX1]]
; CHECK-NEXT:      [[CMP2:%[^ ]+]] = icmp ugt i32 [[SUB3]], -102
; CHECK-NEXT:      [[UMAX1:%[^ ]+]] = select i1 [[CMP2]], i32 [[SUB3]], i32 -102
; CHECK-NEXT:      %exit.mainloop.at = sub i32 -1, [[UMAX1]]
; CHECK-NEXT:      [[CMP3:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:      br i1 [[CMP3]], label %loop.preheader, label %main.pseudo.exit
; CHECK:         loop
; CHECK:           br i1 true, label %in.bounds
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = add i32 %idx, 13
  %abc = icmp ult i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Unsigned latch, Unsigned RC, negative offset. Same as test_04.
define void @test_08(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK-LABEL: test_08(
; CHECK:         entry:
; CHECK-NEXT:      %len = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:      [[SUB1:%[^ ]+]] = sub i32 -14, %len
; CHECK-NEXT:      [[CMP1:%[^ ]+]] = icmp ugt i32 [[SUB1]], -102
; CHECK-NEXT:      [[UMAX1:%[^ ]+]] = select i1 [[CMP1]], i32 [[SUB1]], i32 -102
; CHECK-NEXT:      %exit.mainloop.at = sub i32 -1, [[UMAX1]]
; CHECK-NEXT:      br i1 true, label %loop.preloop.preheader
; CHECK:         in.bounds.preloop:
; CHECK-NEXT:      %addr.preloop = getelementptr i32, i32* %arr, i32 %idx.preloop
; CHECK-NEXT:      store i32 0, i32* %addr.preloop
; CHECK-NEXT:      %next.preloop = icmp ult i32 %idx.next.preloop, 101
; CHECK-NEXT:      [[PRELOOP_COND:%[^ ]+]] = icmp ult i32 %idx.next.preloop, 13
; CHECK-NEXT:      br i1 [[PRELOOP_COND]], label %loop.preloop, label %preloop.exit.selector
; CHECK:         postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.offset = sub i32 %idx, 13
  %abc = icmp ult i32 %idx.offset, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 101
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
