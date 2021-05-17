; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_03: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_04: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_05: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_06: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_07: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_08: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_09: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; ULT condition for increasing loop.
define void @test_01(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_01
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:     %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add nuw nsw i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp ult i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; ULT condition for decreasing loops.
define void @test_02(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK: test_02(
; CHECK:        entry:
; CHECK-NEXT:     %len = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[UMIN:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %len, i32 1)
; CHECK-NEXT:     %exit.preloop.at = add nsw i32 [[UMIN]], -1
; CHECK-NEXT:     [[COND2:%[^ ]+]] = icmp ugt i32 100, %exit.preloop.at
; CHECK-NEXT:     br i1 [[COND2]], label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:        mainloop:
; CHECK-NEXT:     br label %loop
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:     %idx.next = add i32 %idx, -1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %len
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.postloop:
; CHECK:        loop.preloop:
; CHECK-NEXT:     %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ 100, %loop.preloop.preheader ]
; CHECK-NEXT:     %idx.next.preloop = add i32 %idx.preloop, -1
; CHECK-NEXT:     %abc.preloop = icmp ult i32 %idx.preloop, %len
; CHECK-NEXT:     br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 1
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Check SINT_MAX.
define void @test_03(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_03
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:     %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add nuw nsw i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp ult i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 2147483647
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Check SINT_MAX + 1, test is similar to test_01.
define void @test_04(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_04
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:     %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add nuw nsw i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp ult i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 2147483648
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Check SINT_MAX + 1, test is similar to test_02.
define void @test_05(i32* %arr, i32* %a_len_ptr) #0 {
; CHECK: test_05(
; CHECK:        entry:
; CHECK-NEXT:     %len = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[UMIN:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %len, i32 1)
; CHECK-NEXT:     %exit.preloop.at = add nsw i32 [[UMIN]], -1
; CHECK-NEXT:     [[COND2:%[^ ]+]] = icmp ugt i32 -2147483648, %exit.preloop.at
; CHECK-NEXT:     br i1 [[COND2]], label %loop.preloop.preheader, label %preloop.pseudo.exit
; CHECK:        mainloop:
; CHECK-NEXT:     br label %loop
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.preloop.copy, %mainloop ], [ %idx.next, %in.bounds ]
; CHECK-NEXT:     %idx.next = add i32 %idx, -1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %len
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.postloop:
; CHECK:        loop.preloop:
; CHECK-NEXT:     %idx.preloop = phi i32 [ %idx.next.preloop, %in.bounds.preloop ], [ -2147483648, %loop.preloop.preheader ]
; CHECK-NEXT:     %idx.next.preloop = add i32 %idx.preloop, -1
; CHECK-NEXT:     %abc.preloop = icmp ult i32 %idx.preloop, %len
; CHECK-NEXT:     br i1 %abc.preloop, label %in.bounds.preloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 2147483648, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 1
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Increasing loop, UINT_MAX. Positive test.
define void @test_06(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_06
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:     %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add nuw nsw i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp ult i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 4294967295
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Decreasing loop, UINT_MAX. Negative test: we cannot substract -1 from 0.
define void @test_07(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK: test_07(
; CHECK-NOT:    loop.preloop:
; CHECK-NOT:    loop.postloop:

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 4294967295, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nuw i32 %idx, -1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 0
  br i1 %next, label %exit, label %loop

out.of.bounds:
  ret void

exit:
  ret void
}

; Unsigned walking through signed border is allowed.
; Iteration space [0; UINT_MAX - 99), the fact that SINT_MAX is within this
; range does not prevent us from performing IRCE.

define void @test_08(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_08
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, align 4, !range !0
; CHECK-NEXT:     [[COND:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK-NOT:    loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:     %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp ult i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, -100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Walking through the border of unsigned range is not allowed
; (iteration space [-100; 100)). Negative test.

define void @test_09(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_09
; CHECK-NOT:  preloop
; CHECK-NOT:  postloop
; CHECK-NOT:  br i1 false
; CHECK-NOT:  br i1 true

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ -100, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp ult i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx.next, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
