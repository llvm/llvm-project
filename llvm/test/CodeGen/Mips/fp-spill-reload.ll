; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s
; check that $fp is not reserved. 

define void @foo0(i32* nocapture %b) nounwind {
entry:
; CHECK: sw  $fp
; CHECK: lw  $fp
  %0 = load i32, i32* %b, align 4
  %arrayidx.1 = getelementptr inbounds i32, i32* %b, i32 1
  %1 = load i32, i32* %arrayidx.1, align 4
  %add.1 = add nsw i32 %1, 1
  %arrayidx.2 = getelementptr inbounds i32, i32* %b, i32 2
  %2 = load i32, i32* %arrayidx.2, align 4
  %add.2 = add nsw i32 %2, 2
  %arrayidx.3 = getelementptr inbounds i32, i32* %b, i32 3
  %3 = load i32, i32* %arrayidx.3, align 4
  %add.3 = add nsw i32 %3, 3
  %arrayidx.4 = getelementptr inbounds i32, i32* %b, i32 4
  %4 = load i32, i32* %arrayidx.4, align 4
  %add.4 = add nsw i32 %4, 4
  %arrayidx.5 = getelementptr inbounds i32, i32* %b, i32 5
  %5 = load i32, i32* %arrayidx.5, align 4
  %add.5 = add nsw i32 %5, 5
  %arrayidx.6 = getelementptr inbounds i32, i32* %b, i32 6
  %6 = load i32, i32* %arrayidx.6, align 4
  %add.6 = add nsw i32 %6, 6
  %arrayidx.7 = getelementptr inbounds i32, i32* %b, i32 7
  %7 = load i32, i32* %arrayidx.7, align 4
  %add.7 = add nsw i32 %7, 7
  call void @foo2(i32 %0, i32 %add.1, i32 %add.2, i32 %add.3, i32 %add.4, i32 %add.5, i32 %add.6, i32 %add.7) nounwind
  call void bitcast (void (...)* @foo1 to void ()*)() nounwind
  call void @foo2(i32 %0, i32 %add.1, i32 %add.2, i32 %add.3, i32 %add.4, i32 %add.5, i32 %add.6, i32 %add.7) nounwind
  ret void
}

declare void @foo2(i32, i32, i32, i32, i32, i32, i32, i32)

declare void @foo1(...)

