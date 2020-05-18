; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
;
; Test that only one vperm of the vector compare is needed for both extracts.

define void @fun() {
; CHECK-LABEL: fun
; CHECK: vperm
; CHECK-NOT: vperm
bb:
  %tmp = load <4 x i8>, <4 x i8>* undef
  %tmp1 = icmp eq <4 x i8> zeroinitializer, %tmp
  %tmp2 = extractelement <4 x i1> %tmp1, i32 0
  br i1 %tmp2, label %bb1, label %bb2

bb1:
  unreachable

bb2:
  %tmp3 = extractelement <4 x i1> %tmp1, i32 1
  br i1 %tmp3, label %bb3, label %bb4

bb3:
  unreachable

bb4:
  unreachable
}
