; RUN: opt < %s -instcombine -S | FileCheck %s

define <4 x i32> @psignd_3(<4 x i32> %a, <4 x i32> %b) nounwind ssp {
entry:
  %cmp = icmp slt <4 x i32> %b, zeroinitializer
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %sub = sub nsw <4 x i32> zeroinitializer, %a
  %0 = icmp slt <4 x i32> %sext, zeroinitializer
  %sext3 = sext <4 x i1> %0 to <4 x i32>
  %1 = xor <4 x i32> %sext3, <i32 -1, i32 -1, i32 -1, i32 -1>
  %2 = and <4 x i32> %a, %1
  %3 = and <4 x i32> %sext3, %sub
  %cond = or <4 x i32> %2, %3
  ret <4 x i32> %cond

; CHECK-LABEL: @psignd_3
; CHECK:   ashr <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
; CHECK:   sub nsw <4 x i32> zeroinitializer, %a
; CHECK:   xor <4 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK:   and <4 x i32> %a, %0
; CHECK:   and <4 x i32> %b.lobit, %sub
; CHECK:   or <4 x i32> %1, %2
}

define <4 x i32> @test1(<4 x i32> %a, <4 x i32> %b) nounwind ssp {
entry:
  %cmp = icmp sgt <4 x i32> %b, <i32 -1, i32 -1, i32 -1, i32 -1>
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %sub = sub nsw <4 x i32> zeroinitializer, %a
  %0 = icmp slt <4 x i32> %sext, zeroinitializer
  %sext3 = sext <4 x i1> %0 to <4 x i32>
  %1 = xor <4 x i32> %sext3, <i32 -1, i32 -1, i32 -1, i32 -1>
  %2 = and <4 x i32> %a, %1
  %3 = and <4 x i32> %sext3, %sub
  %cond = or <4 x i32> %2, %3
  ret <4 x i32> %cond

; CHECK-LABEL: @test1
; CHECK:   ashr <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
; CHECK:   xor <4 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1>
; CHECK:   sub nsw <4 x i32> zeroinitializer, %a
; CHECK:   and <4 x i32> %b.lobit, %a
; CHECK:   and <4 x i32> %b.lobit.not, %sub
; CHECK:   or <4 x i32> %0, %1
}
