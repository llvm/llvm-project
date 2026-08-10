; RUN: llc -mtriple=i686-linux < %s | FileCheck %s

define i1 @block_filter() !prof !22{
; CHECK-LABEL: block_filter
; CHECK: %.entry
; CHECK: %.header1
; CHECK: %.bb1
; CHECK: %.header2
; CHECK: %.latch2
; CHECK: %.cold
; CHECK: %.pred
; CHECK: %.problem
; CHECK: %.latch1
; CHECK: %.exit
.entry:
  %val0 = call i1 @bar()
  br label %.header1

.header1:
  %val1 = call i1 @foo()
  br i1 %val1, label %.bb1, label %.pred, !prof !2

.bb1:
  %val11 = call i1 @foo()
  br i1 %val11, label %.header2, label %.pred, !prof !2

.header2:
  %val2 = call i1 @foo()
  br i1 %val2, label %.latch2, label %.cold, !prof !10

.cold:
  %val4 = call i1 @bar()
  br i1 %val4, label %.latch2, label %.problem

.latch2:
  %val5 = call i1 @foo()
  br i1 %val5, label %.header2, label %.latch1, !prof !1

.pred:
  %valp = call i1 @foo()
  br label %.problem

.problem:
  %val3 = call i1 @foo()
  br label %.latch1

.latch1:
  %val6 = call i1 @foo()
  br i1 %val6, label %.header1, label %.exit, !prof !1

.exit:
  %val7 = call i1 @foo()
  ret i1 %val7
}

declare i1 @foo()
declare i1 @bar()

!1 = !{!"branch_weights", i32 5, i32 5}
!2 = !{!"branch_weights", i32 60, i32 40}
!3 = !{!"branch_weights", i32 90, i32 10}
!10 = !{!"branch_weights", i32 90, i32 10}

!22 = !{!"function_entry_count", i64 100}
