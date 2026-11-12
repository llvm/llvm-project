; RUN: llc -mtriple=i686-linux < %s | FileCheck %s

define i1 @no_viable_top_fallthrough() {
; CHECK-LABEL: no_viable_top_fallthrough
; CHECK: %.entry
; CHECK: %.bb1
; CHECK: %.stop
; CHECK: %.bb2
; CHECK: %.middle
; CHECK: %.backedge
; CHECK: %.bb3
; CHECK: %.header
; CHECK: %.exit
.entry:
  %val1 = call i1 @foo()
  br i1 %val1, label %.bb1, label %.header, !prof !10

.bb1:
  %val2 = call i1 @foo()
  br i1 %val2, label %.stop, label %.exit, !prof !10

.header:
  %val3 = call i1 @foo()
  br i1 %val3, label %.bb2, label %.exit

.bb2:
  %val4 = call i1 @foo()
  br i1 %val4, label %.middle, label %.bb3, !prof !10

.middle:
  %val5 = call i1 @foo()
  br i1 %val5, label %.header, label %.backedge

.backedge:
  %val6 = call i1 @foo()
  br label %.header

.bb3:
  %val7 = call i1 @foo()
  br label %.middle

.exit:
  %val8 = call i1 @foo()
  br label %.stop

.stop:
  %result = call i1 @foo()
  ret i1 %result
}

declare i1 @foo()

!10 = !{!"branch_weights", i32 90, i32 10}
