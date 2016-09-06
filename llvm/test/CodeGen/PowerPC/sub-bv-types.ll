; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD10521() {
BB:
  %Shuff7 = shufflevector <16 x i16> zeroinitializer, <16 x i16> zeroinitializer, <16 x i32> <i32 undef, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 undef, i32 22, i32 undef, i32 26, i32 undef, i32 30>
  br label %CF

CF:                                               ; preds = %CF78, %CF, %BB
  %I27 = insertelement <16 x i16> %Shuff7, i16 1360, i32 8
  %B28 = sub <16 x i16> %I27, %Shuff7
  br i1 undef, label %CF, label %CF78

CF78:                                             ; preds = %CF
  %B42 = xor <16 x i16> %B28, %Shuff7
  br label %CF
}
