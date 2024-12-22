; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: min

define i64 @f(i64 %src, i64 %maxval) nounwind readnone {
entry:
  %cmp = icmp sgt i64 %maxval, %src
  %cond = select i1 %cmp, i64 %src, i64 %maxval
  ret i64 %cond
}
