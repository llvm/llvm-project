; RUN: llc -mtriple=mipsel-elf < %s | FileCheck %s

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4

; Check that LSR doesn't choose a solution with a formula "reg + 4*reg".
;
; CHECK:      $BB0_2:
; CHECK-NOT:  sll ${{[0-9]+}}, ${{[0-9]+}}, 2

define i32 @f0(i32 %n, i32 %m, ptr nocapture %a, ptr nocapture %b) nounwind readonly {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %s.022 = phi i32 [ 0, %entry ], [ %add7, %for.inc9 ]
  %i.021 = phi i32 [ 0, %entry ], [ %add10, %for.inc9 ]
  br label %for.body3

for.body3:
  %s.120 = phi i32 [ %s.022, %for.cond1.preheader ], [ %add7, %for.body3 ]
  %j.019 = phi i32 [ 0, %for.cond1.preheader ], [ %add8, %for.body3 ]
  %arrayidx4 = getelementptr inbounds [256 x i32], ptr %a, i32 %i.021, i32 %j.019
  %0 = load i32, ptr %arrayidx4, align 4
  %arrayidx6 = getelementptr inbounds [256 x i32], ptr %b, i32 %i.021, i32 %j.019
  %1 = load i32, ptr %arrayidx6, align 4
  %add = add i32 %0, %s.120
  %add7 = add i32 %add, %1
  %add8 = add nsw i32 %j.019, %m
  %cmp2 = icmp slt i32 %add8, 64
  br i1 %cmp2, label %for.body3, label %for.inc9

for.inc9:
  %add10 = add nsw i32 %i.021, %n
  %cmp = icmp slt i32 %add10, 64
  br i1 %cmp, label %for.cond1.preheader, label %for.end11

for.end11:
  ret i32 %add7
}

