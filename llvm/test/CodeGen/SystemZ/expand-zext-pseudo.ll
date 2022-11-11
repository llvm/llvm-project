; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s
;
; Test that a def operand of super-reg is not dropped during post RA pseudo
; expansion in expandZExtPseudo().

define void @fun_llvm_stress_reduced(ptr, ptr, ptr, i32) {
; CHECK: .text
BB:
  %A = alloca i32
  %Sl24 = select i1 undef, ptr %1, ptr %1
  %L26 = load i16, ptr undef
  %L32 = load i32, ptr %Sl24
  br label %CF847

CF847:                                            ; preds = %CF878, %BB
  %L61 = load i16, ptr undef
  br label %CF878

CF878:                                            ; preds = %CF847
  %Sl67 = select i1 undef, <2 x i32> undef, <2 x i32> undef
  %Cmp68 = icmp ugt i32 undef, %3
  br i1 %Cmp68, label %CF847, label %CF863

CF863:                                            ; preds = %CF878
  %L84 = load i16, ptr undef
  br label %CF825

CF825:                                            ; preds = %CF825, %CF863
  %Sl105 = select i1 undef, i1 undef, i1 undef
  br i1 %Sl105, label %CF825, label %CF856

CF856:                                            ; preds = %CF856, %CF825
  %Cmp114 = icmp ult i16 -24837, %L61
  br i1 %Cmp114, label %CF856, label %CF875

CF875:                                            ; preds = %CF856
  %Shuff124 = shufflevector <2 x i32> undef, <2 x i32> undef, <2 x i32> <i32 1, i32 3>
  br label %CF827

CF827:                                            ; preds = %CF923, %CF911, %CF875
  %Sl142 = select i1 undef, i64 undef, i64 -1
  %B148 = sdiv i32 409071, 409071
  %E153 = extractelement <2 x i32> %Shuff124, i32 1
  br label %CF911

CF911:                                            ; preds = %CF827
  br i1 undef, label %CF827, label %CF867

CF867:                                            ; preds = %CF911
  br label %CF870

CF870:                                            ; preds = %CF870, %CF867
  store i8 0, ptr %0
  %FC176 = fptoui double undef to i1
  br i1 %FC176, label %CF870, label %CF923

CF923:                                            ; preds = %CF870
  %L179 = load i16, ptr undef
  %Sl191 = select i1 undef, ptr %A, ptr %A
  br i1 false, label %CF827, label %CF828

CF828:                                            ; preds = %CF905, %CF923
  %B205 = urem i16 -7553, undef
  %E209 = extractelement <2 x i32> %Sl67, i32 1
  %Cmp215 = icmp ugt i16 %L179, 0
  br label %CF905

CF905:                                            ; preds = %CF828
  %E231 = extractelement <4 x i1> undef, i32 1
  br i1 %E231, label %CF828, label %CF829

CF829:                                            ; preds = %CF909, %CF829, %CF905
  %B234 = udiv i16 %L26, %L84
  br i1 undef, label %CF829, label %CF894

CF894:                                            ; preds = %CF894, %CF829
  store i64 %Sl142, ptr %Sl191
  %Sl241 = select i1 %Cmp114, i1 false, i1 %Cmp215
  br i1 %Sl241, label %CF894, label %CF907

CF907:                                            ; preds = %CF894
  %B247 = udiv i32 0, %E153
  br label %CF909

CF909:                                            ; preds = %CF907
  store i1 %FC176, ptr undef
  %Cmp263 = icmp ugt i1 undef, %Sl241
  br i1 %Cmp263, label %CF829, label %CF830

CF830:                                            ; preds = %CF909
  %B304 = urem i16 %L84, %B205
  %I311 = insertelement <2 x i32> %Shuff124, i32 %B247, i32 1
  store i8 0, ptr %0
  %Sl373 = select i1 %Cmp68, i32 0, i32 %E153
  br label %CF833

CF833:                                            ; preds = %CF880, %CF830
  br label %CF880

CF880:                                            ; preds = %CF833
  %Cmp412 = icmp ne i16 %B234, -18725
  br i1 %Cmp412, label %CF833, label %CF865

CF865:                                            ; preds = %CF880
  store double 0.000000e+00, ptr %Sl24
  br label %CF860

CF860:                                            ; preds = %CF860, %CF865
  store i8 0, ptr %2
  %Cmp600 = icmp sge i32 %B148, undef
  br i1 %Cmp600, label %CF860, label %CF913

CF913:                                            ; preds = %CF860
  store i32 %E209, ptr undef
  store i32 %Sl373, ptr undef
  %Cmp771 = icmp ule i32 undef, %L32
  br label %CF842

CF842:                                            ; preds = %CF925, %CF913
  br label %CF925

CF925:                                            ; preds = %CF842
  %Cmp778 = icmp sgt i1 %Cmp771, %Sl241
  br i1 %Cmp778, label %CF842, label %CF898

CF898:                                            ; preds = %CF925
  %Sl785 = select i1 %Cmp600, i16 undef, i16 %B304
  unreachable
}
