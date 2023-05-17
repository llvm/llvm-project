; RUN: llc -O2 < %s | FileCheck %s

target triple = "powerpc64le-grtev4-linux-gnu"

; No duplication of loop header into entry block.
define void @no_duplicate1(i64 %a) {
; CHECK-LABEL: no_duplicate1
; CHECK:        mr 30, 3
; CHECK-NEXT:   std 0, 64(1)
; CHECK-NEXT:   b .LBB0_2

; CHECK:      .LBB0_2:
; CHECK-NEXT:   # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:   cmpldi 30, 100
; CHECK-NEXT:   bne 0, .LBB0_1
entry:
  br label %header

header:
  %ind = phi i64 [%a, %entry], [%val3, %latch]
  %cond1 = icmp eq i64 %ind, 100
  br i1 %cond1, label %middle, label %latch

middle:
  %condx = call i1 @foo()
  %val1 = xor i64 %ind, 2
  br label %latch

latch:
  %val2 = phi i64 [%ind, %header], [%val1, %middle]
  %val3 = add i64 %val2, 1
  %cond2 = call i1 @foo()
  br i1 %cond2, label %end, label %header

end:
  ret void
}

; No duplication of loop header into latches.
define void @no_duplicate2(i64 %a) {
; CHECK-LABEL: no_duplicate2
; CHECK:        mr 30, 3
; CHECK-NEXT:   std 0, 64(1)
; CHECK-NEXT:   b .LBB1_2

; CHECK:      .LBB1_2:
; CHECK-NEXT:   # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:   cmpldi 30, 100
; CHECK-NEXT:   bne 0, .LBB1_1

; CHECK:      %latch2
; CHECK:        b .LBB1_2

; CHECK:      %latch3
; CHECK:        b .LBB1_2
entry:
  br label %header

header:
  %ind = phi i64 [%a, %entry], [%val1, %latch1], [%val2, %latch2], [%val2, %latch3]
  %cond1 = icmp eq i64 %ind, 100
  br i1 %cond1, label %middle1, label %latch1

latch1:
  %cond2 = call i1 @foo()
  %val1 = xor i64 %ind, 2
  br i1 %cond2, label %end, label %header

middle1:
  %cond3 = call i1 @foo()
  br i1 %cond3, label %latch1, label %middle2

middle2:
  %cond4 = call i1 @foo()
  %val2 = add i64 %ind, 1
  br i1 %cond4, label %latch2, label %latch3

latch2:
  call void @a()
  br label %header

latch3:
  call void @b()
  br label %header

end:
  ret void
}


declare i1 @foo()
declare void @a()
declare void @b()
