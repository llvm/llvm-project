; RUN: opt < %s -instcombine -S | FileCheck %s
; CHECK: bitcast

define void @foo(<16 x i8> %a, <16 x i8> %b, <4 x i32>* %c) {
  %aa = bitcast <16 x i8> %a to <4 x i32>
  %bb = bitcast <16 x i8> %b to <4 x i32>
  %select_v = select <4 x i1> zeroinitializer, <4 x i32> %aa, <4 x i32> %bb
  store <4 x i32> %select_v, <4 x i32>* %c, align 4
  ret void
}

