; Test that SystemZTargetLowering::combineSTORE() does not crash due to not
; checking if store is actually a truncating store before calling
; combineTruncateExtract().
;
; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 < %s

@g_348 = external dso_local unnamed_addr global [6 x [10 x i16]], align 2

define void @main() local_unnamed_addr {
bb:
  %tmp = load i16, i16* getelementptr inbounds ([6 x [10 x i16]], [6 x [10 x i16]]* @g_348, i64 0, i64 1, i64 6), align 2
  %tmp1 = xor i16 %tmp, 0
  %tmp2 = insertelement <2 x i16> <i16 undef, i16 0>, i16 %tmp1, i32 0
  %tmp3 = shufflevector <2 x i16> %tmp2, <2 x i16> undef, <2 x i32> <i32 1, i32 undef>
  %tmp4 = xor <2 x i16> %tmp2, %tmp3
  %tmp5 = extractelement <2 x i16> %tmp4, i32 0
  store i16 %tmp5, i16* getelementptr inbounds ([6 x [10 x i16]], [6 x [10 x i16]]* @g_348, i64 0, i64 1, i64 6), align 2
  unreachable
}
