; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: max

define i64 @f(i64 %src, i64 %maxval) nounwind readnone {
entry:
  %cmp = icmp slt i64 %maxval, %src
  %cond = select i1 %cmp, i64 %src, i64 %maxval
  ret i64 %cond
}
