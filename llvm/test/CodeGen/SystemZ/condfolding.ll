; Test that a conditional branch with a discoverably trivial condition
; does not result in an invalid conditional branch instruction.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   --stop-after=systemz-isel | FileCheck %s

@g_1 = dso_local local_unnamed_addr global i64 0, align 8
@g_2 = dso_local local_unnamed_addr global i32 0, align 4

define dso_local void @f1() local_unnamed_addr #1 {
entry:
;CHECK-LABEL: f1
;CHECK-NOT: BRC 14, 0, %bb.2
  %0 = load i64, ptr @g_1, align 8
  %tobool.not = icmp eq i64 %0, 0
  %sub.i = select i1 %tobool.not, i8 4, i8 3
  %conv1 = zext nneg i8 %sub.i to i32
  store i32 %conv1, ptr @g_2, align 4
  %.pr = load i32, ptr @g_2, align 4
  %tobool5.not = icmp eq i32 %.pr, 0
  br i1 %tobool5.not, label %for.cond, label %lbl_1

lbl_1:
  br label %lbl_1

for.cond:
  br label %for.cond
}
