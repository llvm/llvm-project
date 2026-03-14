; RUN: llc -mtriple=i686-linux < %s | FileCheck %s


define i32 @bar(i32 %count) {
; Test checks that basic block backedge2 is not moved before header,
; because it can't reduce taken branches.
; Later backedge1 and backedge2 is rotated before loop header.
; CHECK-LABEL: bar
; CHECK: %.entry
; CHECK: %.header
; CHECK: %.backedge1
; CHECK: %.backedge2
; CHECK: %.exit
.entry:
  %c = shl nsw i32 %count, 2
  br label %.header

.header:
  %val1 = call i32 @foo()
  %cond1 = icmp sgt i32 %val1, 1
  br i1 %cond1, label %.exit, label %.backedge1

.backedge1:
  %val2 = call i32 @foo()
  %cond2 = icmp sgt i32 %val2, 1
  br i1 %cond2, label %.header, label %.backedge2

.backedge2:
  %val3 = call i32 @foo()
  br label %.header

.exit:
  ret i32 %c
}

declare i32 @foo()
