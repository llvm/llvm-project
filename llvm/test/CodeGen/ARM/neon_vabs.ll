; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <4 x i32> @test1(<4 x i32> %a) nounwind {
; CHECK-LABEL: test1:
; CHECK: vabs.s32 q
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sgt <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <4 x i32> @test2(<4 x i32> %a) nounwind {
; CHECK-LABEL: test2:
; CHECK: vabs.s32 q
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sge <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %a, <4 x i32> %tmp1neg
        ret <4 x i32> %abs
}

define <8 x i16> @test3(<8 x i16> %a) nounwind {
; CHECK-LABEL: test3:
; CHECK: vabs.s16 q
        %tmp1neg = sub <8 x i16> zeroinitializer, %a
        %b = icmp sgt <8 x i16> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i16> %a, <8 x i16> %tmp1neg
        ret <8 x i16> %abs
}

define <16 x i8> @test4(<16 x i8> %a) nounwind {
; CHECK-LABEL: test4:
; CHECK: vabs.s8 q
        %tmp1neg = sub <16 x i8> zeroinitializer, %a
        %b = icmp slt <16 x i8> %a, zeroinitializer
        %abs = select <16 x i1> %b, <16 x i8> %tmp1neg, <16 x i8> %a
        ret <16 x i8> %abs
}

define <4 x i32> @test5(<4 x i32> %a) nounwind {
; CHECK-LABEL: test5:
; CHECK: vabs.s32 q
        %tmp1neg = sub <4 x i32> zeroinitializer, %a
        %b = icmp sle <4 x i32> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i32> %tmp1neg, <4 x i32> %a
        ret <4 x i32> %abs
}

define <2 x i32> @test6(<2 x i32> %a) nounwind {
; CHECK-LABEL: test6:
; CHECK: vabs.s32 d
        %tmp1neg = sub <2 x i32> zeroinitializer, %a
        %b = icmp sgt <2 x i32> %a, <i32 -1, i32 -1>
        %abs = select <2 x i1> %b, <2 x i32> %a, <2 x i32> %tmp1neg
        ret <2 x i32> %abs
}

define <2 x i32> @test7(<2 x i32> %a) nounwind {
; CHECK-LABEL: test7:
; CHECK: vabs.s32 d
        %tmp1neg = sub <2 x i32> zeroinitializer, %a
        %b = icmp sge <2 x i32> %a, zeroinitializer
        %abs = select <2 x i1> %b, <2 x i32> %a, <2 x i32> %tmp1neg
        ret <2 x i32> %abs
}

define <4 x i16> @test8(<4 x i16> %a) nounwind {
; CHECK-LABEL: test8:
; CHECK: vabs.s16 d
        %tmp1neg = sub <4 x i16> zeroinitializer, %a
        %b = icmp sgt <4 x i16> %a, zeroinitializer
        %abs = select <4 x i1> %b, <4 x i16> %a, <4 x i16> %tmp1neg
        ret <4 x i16> %abs
}

define <8 x i8> @test9(<8 x i8> %a) nounwind {
; CHECK-LABEL: test9:
; CHECK: vabs.s8 d
        %tmp1neg = sub <8 x i8> zeroinitializer, %a
        %b = icmp slt <8 x i8> %a, zeroinitializer
        %abs = select <8 x i1> %b, <8 x i8> %tmp1neg, <8 x i8> %a
        ret <8 x i8> %abs
}

define <2 x i32> @test10(<2 x i32> %a) nounwind {
; CHECK-LABEL: test10:
; CHECK: vabs.s32 d
        %tmp1neg = sub <2 x i32> zeroinitializer, %a
        %b = icmp sle <2 x i32> %a, zeroinitializer
        %abs = select <2 x i1> %b, <2 x i32> %tmp1neg, <2 x i32> %a
        ret <2 x i32> %abs
}
