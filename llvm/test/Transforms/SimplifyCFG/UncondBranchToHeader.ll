; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; Check that we can get rid of empty block leading to header
; if it does not introduce new edge.
define i32 @test(i32 %c) {
entry:
  br label %header
header:
  %i = phi i32 [0, %entry], [%i.1, %backedge]
  %i.1 = add i32 %i, 1
  %cmp = icmp slt i32 %i.1, %c
  br i1 %cmp, label %backedge, label %exit
; CHECK-NOT: backedge:
backedge:
  br label %header
exit:
  ret i32 %i
}
