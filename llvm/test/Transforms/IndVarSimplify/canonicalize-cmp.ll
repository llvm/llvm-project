; RUN: opt -S -indvars < %s | FileCheck %s

; Check that we replace signed comparisons between non-negative values with
; unsigned comparisons if we can.

target datalayout = "n8:16:32:64"

define i32 @test_01(i32 %a, i32 %b, i32* %p) {

; CHECK-LABEL: @test_01(
; CHECK-NOT:   icmp slt
; CHECK:       %cmp1 = icmp ult i32 %iv, 100
; CHECK:       %cmp2 = icmp ult i32 %iv, 100
; CHECK-NOT:   %cmp3
; CHECK:       %exitcond = icmp ne i32 %iv.next, 1000

entry:
  br label %loop.entry

loop.entry:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.be ]
  %cmp1 = icmp slt i32 %iv, 100
  br i1 %cmp1, label %b1, label %b2

b1:
  store i32 %iv, i32* %p
  br label %merge

b2:
  store i32 %a, i32* %p
  br label %merge

merge:
  %cmp2 = icmp ult i32 %iv, 100
  br i1 %cmp2, label %b3, label %b4

b3:
  store i32 %iv, i32* %p
  br label %loop.be

b4:
  store i32 %b, i32* %p
  br label %loop.be

loop.be:
  %iv.next = add i32 %iv, 1
  %cmp3 = icmp slt i32 %iv.next, 1000
  br i1 %cmp3, label %loop.entry, label %exit

exit:
  ret i32 %iv
}

define i32 @test_02(i32 %a, i32 %b, i32* %p) {

; CHECK-LABEL: @test_02(
; CHECK-NOT:   icmp sgt
; CHECK:       %cmp1 = icmp ugt i32 100, %iv
; CHECK:       %cmp2 = icmp ugt i32 100, %iv
; CHECK-NOT:   %cmp3
; CHECK:       %exitcond = icmp ne i32 %iv.next, 1000

entry:
  br label %loop.entry

loop.entry:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.be ]
  %cmp1 = icmp sgt i32 100, %iv
  br i1 %cmp1, label %b1, label %b2

b1:
  store i32 %iv, i32* %p
  br label %merge

b2:
  store i32 %a, i32* %p
  br label %merge

merge:
  %cmp2 = icmp ugt i32 100, %iv
  br i1 %cmp2, label %b3, label %b4

b3:
  store i32 %iv, i32* %p
  br label %loop.be

b4:
  store i32 %b, i32* %p
  br label %loop.be

loop.be:
  %iv.next = add i32 %iv, 1
  %cmp3 = icmp sgt i32 1000, %iv.next
  br i1 %cmp3, label %loop.entry, label %exit

exit:
  ret i32 %iv
}
