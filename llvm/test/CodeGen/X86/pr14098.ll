; RUN: llc -mtriple i386-unknown-linux-gnu -relocation-model=pic -verify-machineinstrs < %s
; We used to crash on this.

declare void @foo()
declare void @foo3(i1 %x)
define void @bar(i1 %a1, i16 %a2) nounwind align 2 {
bb0:
  %a3 = trunc i16 %a2 to i8
  %a4 = lshr i16 %a2, 8
  %a5 = trunc i16 %a4 to i8
  br i1 %a1, label %bb1, label %bb2
bb1:
  br label %bb2
bb2:
  %a6 = phi i8 [ 3, %bb0 ], [ %a5, %bb1 ]
  %a7 = phi i8 [ 9, %bb0 ], [ %a3, %bb1 ]
  %a8 = icmp eq i8 %a6, 1
  call void @foo()
  %a9 = icmp eq i8 %a7, 0
  call void @foo3(i1 %a9)
  call void @foo3(i1 %a8)
  ret void
}
