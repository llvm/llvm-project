; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD4357(i8) {
BB:
  br label %CF

CF:                                               ; preds = %CF, %BB
  br i1 undef, label %CF, label %CF77

CF77:                                             ; preds = %CF81, %CF77, %CF
  %Shuff12 = shufflevector <2 x i8> <i8 -1, i8 -1>, <2 x i8> <i8 -1, i8 -1>, <2 x i32> <i32 0, i32 undef>
  br i1 undef, label %CF77, label %CF80

CF80:                                             ; preds = %CF80, %CF77
  %B21 = mul <2 x i8> %Shuff12, <i8 -1, i8 -1>
  %Cmp24 = fcmp une ppc_fp128 0xM00000000000000000000000000000000, 0xM00000000000000000000000000000000
  br i1 %Cmp24, label %CF80, label %CF81

CF81:                                             ; preds = %CF80
  %I36 = insertelement <2 x i8> %B21, i8 %0, i32 0
  br label %CF77
}
