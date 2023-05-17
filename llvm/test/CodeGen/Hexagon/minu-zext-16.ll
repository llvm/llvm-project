; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: minu

define zeroext i16 @f(ptr noalias nocapture %src) nounwind readonly {
entry:
  %arrayidx = getelementptr inbounds i16, ptr %src, i32 1
  %0 = load i16, ptr %arrayidx, align 1
  %cmp = icmp ult i16 %0, 32767
  %. = select i1 %cmp, i16 %0, i16 32767
  ret i16 %.
}
