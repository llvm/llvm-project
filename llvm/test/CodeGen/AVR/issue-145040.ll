; RUN: llc < %s -O=2 -mtriple=avr-none --mcpu=avr128db28 -verify-machineinstrs | FileCheck %s

declare dso_local void @nil(i16 noundef) local_unnamed_addr addrspace(1) #1
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

define void @complex_sbi() {
; CHECK: sbi 1, 7
entry:
  br label %while.cond
while.cond:                                       ; preds = %while.cond, %entry
  %s.0 = phi i16 [ 0, %entry ], [ %inc, %while.cond ]
  %inc = add nuw nsw i16 %s.0, 1
  %0 = load volatile i8, ptr inttoptr (i16 1 to ptr), align 1, !tbaa !3
  %or = or i8 %0, -128
  store volatile i8 %or, ptr inttoptr (i16 1 to ptr), align 1, !tbaa !3
  %and = and i16 %inc, 15
  %add = add nuw nsw i16 %and, 1
  tail call addrspace(1) void @nil(i16 noundef %add) #2
  br label %while.cond
}

