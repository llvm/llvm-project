; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD13() {
BB:
  br label %CF78

CF78:                                             ; preds = %CF87, %CF86, %CF78, %BB
  %Cmp = icmp ule <16 x i64> zeroinitializer, zeroinitializer
  br i1 undef, label %CF78, label %CF86

CF86:                                             ; preds = %CF78
  br i1 undef, label %CF78, label %CF84

CF84:                                             ; preds = %CF84, %CF86
  br i1 undef, label %CF84, label %CF87

CF87:                                             ; preds = %CF84
  br i1 undef, label %CF78, label %CF82

CF82:                                             ; preds = %CF82, %CF87
  br i1 undef, label %CF82, label %CF83

CF83:                                             ; preds = %CF82
  br label %CF

CF:                                               ; preds = %CF80, %CF81, %CF, %CF83
  br i1 undef, label %CF, label %CF81

CF81:                                             ; preds = %CF
  %Se = sext <16 x i1> %Cmp to <16 x i16>
  br i1 undef, label %CF, label %CF80

CF80:                                             ; preds = %CF81
  br i1 undef, label %CF, label %CF76

CF76:                                             ; preds = %CF76, %CF80
  %Sl58 = select i1 undef, <16 x i16> %Se, <16 x i16> %Se
  br label %CF76
}

define void @autogen_SD1067() {
BB:
  %FC = sitofp <4 x i32> zeroinitializer to <4 x ppc_fp128>
  br label %CF77

CF77:                                             ; preds = %CF77, %BB
  %brmerge = or i1 false, undef
  br i1 %brmerge, label %CF77, label %CF85

CF85:                                             ; preds = %CF77
  %Shuff19 = shufflevector <4 x ppc_fp128> %FC, <4 x ppc_fp128> %FC, <4 x i32> <i32 7, i32 1, i32 3, i32 5>
  br label %CF75

CF75:                                             ; preds = %CF75, %CF85
  br label %CF75
}
