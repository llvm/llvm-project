; Verify that we do not create illegal scalar urems after type legalization.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13

define <16 x i8> @main(i16 %arg) {
bb:
  %tmp6 = insertelement <16 x i16> undef, i16 %arg, i32 0
  %tmp7 = shufflevector <16 x i16> %tmp6, <16 x i16> undef, <16 x i32> zeroinitializer
  %tmp8 = insertelement <16 x i8> <i8 undef, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>, i8 undef, i32 0
  %tmp11 = urem <16 x i16> %tmp7, <i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265, i16 29265>
  %tmp12 = trunc <16 x i16> %tmp11 to <16 x i8>
  ret <16 x i8> %tmp12
}
