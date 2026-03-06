; RUN: opt -S -passes=loop-reduce < %s | FileCheck %s
;
; Make sure loop-reduce doesn't crash with infinite recursion in
; getZeroExtendExpr via getAddExpr calling getZeroExtendExpr without
; propagating the depth argument.
; See https://github.com/llvm/llvm-project/issues/184947

; Corresponds to:
;   a() {
;     int b = 0, c = 0;
;     for (;; a) {
;       c++;
;       if (c <= 14) continue;
;       b++;
;       if (b <= 45) continue;
;       return;
;     }
;   }

; CHECK-LABEL: @a(
define void @a() {
entry:
  br label %for.body

for.body:
  %b.0 = phi i32 [ 0, %entry ], [ %b.next, %for.latch ]
  %c.0 = phi i32 [ 0, %entry ], [ %c.inc, %for.latch ]
  %c.inc = add i32 %c.0, 1
  %cmp.c = icmp sle i32 %c.inc, 14
  br i1 %cmp.c, label %for.latch, label %b.inc.blk

b.inc.blk:
  %b.inc = add i32 %b.0, 1
  %cmp.b = icmp sle i32 %b.inc, 45
  br i1 %cmp.b, label %for.latch, label %return

for.latch:
  %b.next = phi i32 [ %b.0, %for.body ], [ %b.inc, %b.inc.blk ]
  br label %for.body

return:
  ret void
}
