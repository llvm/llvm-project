; Test that the dag combiner can understand that some vector operands are
; all-zeros and then optimize the logical operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define void @f1() {
; CHECK-LABEL: f1:
; CHECK: vno
; CHECK-NOT: vno

bb:
  %tmp = shufflevector <2 x i64> undef, <2 x i64> undef, <2 x i32> zeroinitializer
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = load i64, i64* undef, align 8
  %tmp3 = insertelement <2 x i64> undef, i64 %tmp2, i32 1
  %tmp4 = icmp ne <2 x i64> undef, zeroinitializer
  %tmp5 = xor <2 x i1> %tmp4, zeroinitializer
  %tmp6 = xor <2 x i1> zeroinitializer, %tmp5
  %tmp7 = and <2 x i64> %tmp3, %tmp
  %tmp8 = icmp ne <2 x i64> %tmp7, zeroinitializer
  %tmp9 = xor <2 x i1> zeroinitializer, %tmp8
  %tmp10 = icmp ne <2 x i64> undef, zeroinitializer
  %tmp11 = xor <2 x i1> %tmp10, %tmp9
  %tmp12 = and <2 x i1> %tmp6, %tmp11
  %tmp13 = extractelement <2 x i1> %tmp12, i32 0
  br i1 %tmp13, label %bb14, label %bb15

bb14:                                             ; preds = %bb1
  store i64 undef, i64* undef, align 8
  br label %bb15

bb15:                                             ; preds = %bb14, %bb1
  unreachable
}
