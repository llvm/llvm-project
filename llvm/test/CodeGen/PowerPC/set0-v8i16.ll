; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD367951() {
BB:
  %Shuff = shufflevector <16 x i16> zeroinitializer, <16 x i16> zeroinitializer, <16 x i32> <i32 26, i32 28, i32 30, i32 undef, i32 2, i32 4, i32 undef, i32 undef, i32 10, i32 undef, i32 14, i32 16, i32 undef, i32 20, i32 undef, i32 24>
  %Shuff7 = shufflevector <16 x i16> zeroinitializer, <16 x i16> %Shuff, <16 x i32> <i32 20, i32 undef, i32 24, i32 26, i32 28, i32 undef, i32 0, i32 undef, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18>
  %Cmp11 = icmp ugt <16 x i16> %Shuff7, zeroinitializer
  %E27 = extractelement <16 x i1> %Cmp11, i32 5
  br label %CF76

CF76:                                             ; preds = %CF80, %CF76, %BB
  br i1 undef, label %CF76, label %CF80

CF80:                                             ; preds = %CF76
  %Sl37 = select i1 %E27, <16 x i16> undef, <16 x i16> %Shuff
  br label %CF76
}
