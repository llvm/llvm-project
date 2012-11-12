; RUN: llc < %s -march=x86 -mcpu=nehalem | FileCheck %s

define <4 x i32> @signd(<4 x i32> %a, <4 x i32> %b) nounwind {
entry:
; CHECK: signd:
; CHECK: psignd
; CHECK-NOT: sub
; CHECK: ret
  %b.lobit = ashr <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
  %sub = sub nsw <4 x i32> zeroinitializer, %a
  %0 = xor <4 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1>
  %1 = and <4 x i32> %a, %0
  %2 = and <4 x i32> %b.lobit, %sub
  %cond = or <4 x i32> %1, %2
  ret <4 x i32> %cond
}

define <4 x i32> @blendvb(<4 x i32> %b, <4 x i32> %a, <4 x i32> %c) nounwind {
entry:
; CHECK: blendvb:
; CHECK: pblendvb
; CHECK: ret
  %b.lobit = ashr <4 x i32> %b, <i32 31, i32 31, i32 31, i32 31>
  %sub = sub nsw <4 x i32> zeroinitializer, %a
  %0 = xor <4 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1>
  %1 = and <4 x i32> %c, %0
  %2 = and <4 x i32> %a, %b.lobit
  %cond = or <4 x i32> %1, %2
  ret <4 x i32> %cond
}
