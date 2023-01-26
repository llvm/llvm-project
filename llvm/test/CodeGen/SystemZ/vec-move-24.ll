; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s
;
; Test that vperm is not used if a single unpack is enough.

define <4 x i32> @fun0(ptr %Src) nounwind {
; CHECK-LABEL: fun0:
; CHECK-NOT: vperm
  %tmp = load <4 x i32>, ptr %Src
  %tmp2 = shufflevector <4 x i32> zeroinitializer, <4 x i32> %tmp, <4 x i32> <i32 0, i32 4, i32 2, i32 5>
  ret <4 x i32> %tmp2
}

define  void @fun1(i8 %Src, ptr %Dst) nounwind {
; CHECK-LABEL: fun1:
; CHECK-NOT: vperm
  %I0 = insertelement <16 x i8> undef, i8 %Src, i32 0
  %I1 = insertelement <16 x i8> %I0, i8 %Src, i32 1
  %I2 = insertelement <16 x i8> %I1, i8 %Src, i32 2
  %I3 = insertelement <16 x i8> %I2, i8 %Src, i32 3
  %I4 = insertelement <16 x i8> %I3, i8 %Src, i32 4
  %I5 = insertelement <16 x i8> %I4, i8 %Src, i32 5
  %I6 = insertelement <16 x i8> %I5, i8 %Src, i32 6
  %I7 = insertelement <16 x i8> %I6, i8 %Src, i32 7
  %I8 = insertelement <16 x i8> %I7, i8 %Src, i32 8
  %I9 = insertelement <16 x i8> %I8, i8 %Src, i32 9
  %I10 = insertelement <16 x i8> %I9, i8 %Src, i32 10
  %I11 = insertelement <16 x i8> %I10, i8 %Src, i32 11
  %I12 = insertelement <16 x i8> %I11, i8 %Src, i32 12
  %I13 = insertelement <16 x i8> %I12, i8 %Src, i32 13
  %I14 = insertelement <16 x i8> %I13, i8 %Src, i32 14
  %I15 = insertelement <16 x i8> %I14, i8 %Src, i32 15

  %tmp = shufflevector <16 x i8> zeroinitializer,
                       <16 x i8> %I15,
                       <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                                   i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
                                   i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23,
                                   i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %tmp9 = shufflevector <32 x i8> undef,
                        <32 x i8> %tmp,
                        <32 x i32> <i32 33, i32 32, i32 48, i32 49, i32 1, i32 17, i32 50, i32 51,
                                    i32 2, i32 18, i32 52, i32 53, i32 3, i32 19, i32 54, i32 55,
                                    i32 4, i32 20, i32 56, i32 57, i32 5, i32 21, i32 58, i32 59,
                                    i32 6, i32 22, i32 60, i32 61, i32 7, i32 62, i32 55, i32 63>

  store <32 x i8> %tmp9, ptr %Dst
  ret void
}

