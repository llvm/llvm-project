; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: minu

define zeroext i8 @f(ptr noalias nocapture %src) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds i8, ptr %src, i32 1
  %0 = load i8, ptr %arrayidx, align 1
  %cmp = icmp ult i8 %0, 127
  %. = select i1 %cmp, i8 %0, i8 127
  ret i8 %.
}
