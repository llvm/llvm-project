; RUN: opt -verify-loop-info -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s

; Check that IRCE is able to deal with loops where the latch comparison is
; done against current value of the IV, not the IV.next.

; CHECK: irce: in function test_01: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK: irce: in function test_02: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_03: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>
; CHECK-NOT: irce: in function test_04: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

; SLT condition for increasing loop from 0 to 100.
define void @test_01(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_01
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:     [[COND2:%[^ ]+]] = icmp slt i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND2]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:        in.bounds:
; CHECK-NEXT:     %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:     store i32 0, i32* %addr
; CHECK-NEXT:     %next = icmp slt i32 %idx, 100
; CHECK-NEXT:     [[COND3:%[^ ]+]] = icmp slt i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND3]], label %loop, label %main.exit.selector
; CHECK:        main.exit.selector:
; CHECK-NEXT:     %idx.lcssa = phi i32 [ %idx, %in.bounds ]
; CHECK-NEXT:     [[COND4:%[^ ]+]] = icmp slt i32 %idx.lcssa, 100
; CHECK-NEXT:     br i1 [[COND4]], label %main.pseudo.exit, label %exit
; CHECK-NOT: loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:    %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
; CHECK-NEXT:     %idx.next.postloop = add nuw nsw i32 %idx.postloop, 1
; CHECK-NEXT:     %abc.postloop = icmp slt i32 %idx.postloop, %exit.mainloop.at
; CHECK-NEXT:     br i1 %abc.postloop, label %in.bounds.postloop, label %out.of.bounds.loopexit

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; ULT condition for increasing loop from 0 to 100.
define void @test_02(i32* %arr, i32* %a_len_ptr) #0 {

; CHECK:      test_02
; CHECK:        entry:
; CHECK-NEXT:     %exit.mainloop.at = load i32, i32* %a_len_ptr, !range !0
; CHECK-NEXT:     [[COND2:%[^ ]+]] = icmp ult i32 0, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND2]], label %loop.preheader, label %main.pseudo.exit
; CHECK:        loop:
; CHECK-NEXT:     %idx = phi i32 [ %idx.next, %in.bounds ], [ 0, %loop.preheader ]
; CHECK-NEXT:     %idx.next = add nuw nsw i32 %idx, 1
; CHECK-NEXT:     %abc = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 true, label %in.bounds, label %out.of.bounds.loopexit1
; CHECK:        in.bounds:
; CHECK-NEXT:     %addr = getelementptr i32, i32* %arr, i32 %idx
; CHECK-NEXT:     store i32 0, i32* %addr
; CHECK-NEXT:     %next = icmp ult i32 %idx, 100
; CHECK-NEXT:     [[COND3:%[^ ]+]] = icmp ult i32 %idx, %exit.mainloop.at
; CHECK-NEXT:     br i1 [[COND3]], label %loop, label %main.exit.selector
; CHECK:        main.exit.selector:
; CHECK-NEXT:     %idx.lcssa = phi i32 [ %idx, %in.bounds ]
; CHECK-NEXT:     [[COND4:%[^ ]+]] = icmp ult i32 %idx.lcssa, 100
; CHECK-NEXT:     br i1 [[COND4]], label %main.pseudo.exit, label %exit
; CHECK-NOT: loop.preloop:
; CHECK:        loop.postloop:
; CHECK-NEXT:    %idx.postloop = phi i32 [ %idx.copy, %postloop ], [ %idx.next.postloop, %in.bounds.postloop ]
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
  %next = icmp ult i32 %idx, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Same as test_01, but comparison happens against IV extended to a wider type.
; This test ensures that IRCE rejects it and does not falsely assume that it was
; a comparison against iv.next.
; TODO: We can actually extend the recognition to cover this case.
define void @test_03(i32* %arr, i64* %a_len_ptr) #0 {

; CHECK:      test_03

entry:
  %len = load i64, i64* %a_len_ptr, !range !1
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %idx.ext = sext i32 %idx to i64
  %abc = icmp slt i64 %idx.ext, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

; Same as test_02, but comparison happens against IV extended to a wider type.
; This test ensures that IRCE rejects it and does not falsely assume that it was
; a comparison against iv.next.
; TODO: We can actually extend the recognition to cover this case.
define void @test_04(i32* %arr, i64* %a_len_ptr) #0 {

; CHECK:      test_04

entry:
  %len = load i64, i64* %a_len_ptr, !range !1
  br label %loop

loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add nsw nuw i32 %idx, 1
  %idx.ext = sext i32 %idx to i64
  %abc = icmp ult i64 %idx.ext, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds

in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp ult i32 %idx, 100
  br i1 %next, label %loop, label %exit

out.of.bounds:
  ret void

exit:
  ret void
}

!0 = !{i32 0, i32 50}
!1 = !{i64 0, i64 50}
