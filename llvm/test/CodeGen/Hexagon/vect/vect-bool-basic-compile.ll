; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that we can compile these functions. Don't check anything else for now.
; CHECK-LABEL: test_0:
; CHECK: tstbit
; CHECK-LABEL: test_1:
; CHECK: tstbit
; CHECK-LABEL: test_2:
; CHECK: tstbit

define i32 @test_0(i32 %a0, i32 %a1) #0 {
  %t0 = trunc i32 %a0 to i1
  %t1 = trunc i32 %a1 to i1

  %t2 = insertelement <2 x i1> undef, i1 %t0, i32 0
  %t3 = insertelement <2 x i1> %t2, i1 %t1, i32 1

  %t4 = shufflevector <2 x i1> %t3, <2 x i1> undef, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %t5 = bitcast <8 x i1> %t4 to i8
  %t6 = zext i8 %t5 to i32
  ret i32 %t6
}

define i32 @test_1(i32 %a0, i32 %a1, i32 %a2, i32 %a3) #0 {
  %t0 = trunc i32 %a0 to i1
  %t1 = trunc i32 %a1 to i1
  %t2 = trunc i32 %a2 to i1
  %t3 = trunc i32 %a3 to i1

  %t4 = insertelement <4 x i1> undef, i1 %t0, i32 0
  %t5 = insertelement <4 x i1> %t4, i1 %t1, i32 1
  %t6 = insertelement <4 x i1> %t5, i1 %t2, i32 2
  %t7 = insertelement <4 x i1> %t6, i1 %t3, i32 3

  %t8 = shufflevector <4 x i1> %t7, <4 x i1> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %t9 = bitcast <8 x i1> %t8 to i8
  %ta = zext i8 %t9 to i32
  ret i32 %ta
}

define i32 @test_2(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) #0 {
  %t0 = trunc i32 %a0 to i1
  %t1 = trunc i32 %a1 to i1
  %t2 = trunc i32 %a2 to i1
  %t3 = trunc i32 %a3 to i1
  %t4 = trunc i32 %a4 to i1
  %t5 = trunc i32 %a5 to i1
  %t6 = trunc i32 %a6 to i1
  %t7 = trunc i32 %a7 to i1

  %t8 = insertelement <8 x i1> undef, i1 %t0, i32 0
  %t9 = insertelement <8 x i1> %t8, i1 %t1, i32 1
  %ta = insertelement <8 x i1> %t9, i1 %t2, i32 2
  %tb = insertelement <8 x i1> %ta, i1 %t3, i32 3
  %tc = insertelement <8 x i1> %tb, i1 %t4, i32 4
  %td = insertelement <8 x i1> %tc, i1 %t5, i32 5
  %te = insertelement <8 x i1> %td, i1 %t6, i32 6
  %tf = insertelement <8 x i1> %te, i1 %t7, i32 7

  %tg = bitcast <8 x i1> %tf to i8
  %th = zext i8 %tg to i32
  ret i32 %th
}

attributes #0 = { nounwind readnone }
