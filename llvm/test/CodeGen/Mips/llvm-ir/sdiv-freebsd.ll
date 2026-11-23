; RUN: llc --mtriple=mips-unknown-freebsd -mcpu=mips2 -filetype=asm < %s -mcpu=mips2 | FileCheck %s -check-prefixes=MIPS2
;
; Created from the following test case (PR121463) with
; clang -cc1 -triple mips-unknown-freebsd -target-cpu mips2 -O2 -emit-llvm test.c -o test.ll
; int l2arc_feed_secs, l2arc_feed_min_ms, l2arc_write_interval_wrote, l2arc_write_interval_next;
; void l2arc_write_interval() {
;   int interval = 0;
;   if (l2arc_write_interval_wrote)
;     interval = l2arc_feed_min_ms / l2arc_feed_secs;
;   l2arc_write_interval_next = interval;
; }

@l2arc_write_interval_wrote = local_unnamed_addr global i32 0, align 4
@l2arc_feed_min_ms = local_unnamed_addr global i32 0, align 4
@l2arc_feed_secs = local_unnamed_addr global i32 0, align 4
@l2arc_write_interval_next = local_unnamed_addr global i32 0, align 4

define dso_local void @l2arc_write_interval() local_unnamed_addr #0 {
; MIPS2-LABEL: l2arc_write_interval:
; MIPS2:       # %bb.0: # %entry
; MIPS2-NEXT:    lui $1, %hi(l2arc_write_interval_wrote)
; MIPS2-NEXT:    lw $1, %lo(l2arc_write_interval_wrote)($1)
; MIPS2-NEXT:    beqz $1, $BB0_2
; MIPS2-NEXT:    nop
; MIPS2-NEXT:  # %bb.1: # %if.then
; MIPS2-NEXT:    lui $1, %hi(l2arc_feed_secs)
; MIPS2-NEXT:    lw $1, %lo(l2arc_feed_secs)($1)
; MIPS2-NEXT:    lui $2, %hi(l2arc_feed_min_ms)
; MIPS2-NEXT:    lw $2, %lo(l2arc_feed_min_ms)($2)
; MIPS2-NEXT:    div $zero, $2, $1
; MIPS2-NEXT:    teq $1, $zero, 7
; MIPS2-NEXT:    mflo $2
; MIPS2-NEXT:    j $BB0_3
; MIPS2-NEXT:    nop
entry:
  %0 = load i32, ptr @l2arc_write_interval_wrote, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %1 = load i32, ptr @l2arc_feed_min_ms, align 4
  %2 = load i32, ptr @l2arc_feed_secs, align 4
  %div = sdiv i32 %1, %2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %interval.0 = phi i32 [ %div, %if.then ], [ 0, %entry ]
  store i32 %interval.0, ptr @l2arc_write_interval_next, align 4
  ret void
}
