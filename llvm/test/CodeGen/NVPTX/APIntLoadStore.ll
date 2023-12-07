; RUN: llc -march=nvptx  < %s > %t
; RUN: llc -march=nvptx64  < %s > %t

@i1_l = external global i1		; <ptr> [#uses=1]
@i1_s = external global i1		; <ptr> [#uses=1]
@i2_l = external global i2		; <ptr> [#uses=1]
@i2_s = external global i2		; <ptr> [#uses=1]
@i3_l = external global i3		; <ptr> [#uses=1]
@i3_s = external global i3		; <ptr> [#uses=1]
@i4_l = external global i4		; <ptr> [#uses=1]
@i4_s = external global i4		; <ptr> [#uses=1]
@i5_l = external global i5		; <ptr> [#uses=1]
@i5_s = external global i5		; <ptr> [#uses=1]
@i6_l = external global i6		; <ptr> [#uses=1]
@i6_s = external global i6		; <ptr> [#uses=1]
@i7_l = external global i7		; <ptr> [#uses=1]
@i7_s = external global i7		; <ptr> [#uses=1]
@i8_l = external global i8		; <ptr> [#uses=1]
@i8_s = external global i8		; <ptr> [#uses=1]
@i9_l = external global i9		; <ptr> [#uses=1]
@i9_s = external global i9		; <ptr> [#uses=1]
@i10_l = external global i10		; <ptr> [#uses=1]
@i10_s = external global i10		; <ptr> [#uses=1]
@i11_l = external global i11		; <ptr> [#uses=1]
@i11_s = external global i11		; <ptr> [#uses=1]
@i12_l = external global i12		; <ptr> [#uses=1]
@i12_s = external global i12		; <ptr> [#uses=1]
@i13_l = external global i13		; <ptr> [#uses=1]
@i13_s = external global i13		; <ptr> [#uses=1]
@i14_l = external global i14		; <ptr> [#uses=1]
@i14_s = external global i14		; <ptr> [#uses=1]
@i15_l = external global i15		; <ptr> [#uses=1]
@i15_s = external global i15		; <ptr> [#uses=1]
@i16_l = external global i16		; <ptr> [#uses=1]
@i16_s = external global i16		; <ptr> [#uses=1]
@i17_l = external global i17		; <ptr> [#uses=1]
@i17_s = external global i17		; <ptr> [#uses=1]
@i18_l = external global i18		; <ptr> [#uses=1]
@i18_s = external global i18		; <ptr> [#uses=1]
@i19_l = external global i19		; <ptr> [#uses=1]
@i19_s = external global i19		; <ptr> [#uses=1]
@i20_l = external global i20		; <ptr> [#uses=1]
@i20_s = external global i20		; <ptr> [#uses=1]
@i21_l = external global i21		; <ptr> [#uses=1]
@i21_s = external global i21		; <ptr> [#uses=1]
@i22_l = external global i22		; <ptr> [#uses=1]
@i22_s = external global i22		; <ptr> [#uses=1]
@i23_l = external global i23		; <ptr> [#uses=1]
@i23_s = external global i23		; <ptr> [#uses=1]
@i24_l = external global i24		; <ptr> [#uses=1]
@i24_s = external global i24		; <ptr> [#uses=1]
@i25_l = external global i25		; <ptr> [#uses=1]
@i25_s = external global i25		; <ptr> [#uses=1]
@i26_l = external global i26		; <ptr> [#uses=1]
@i26_s = external global i26		; <ptr> [#uses=1]
@i27_l = external global i27		; <ptr> [#uses=1]
@i27_s = external global i27		; <ptr> [#uses=1]
@i28_l = external global i28		; <ptr> [#uses=1]
@i28_s = external global i28		; <ptr> [#uses=1]
@i29_l = external global i29		; <ptr> [#uses=1]
@i29_s = external global i29		; <ptr> [#uses=1]
@i30_l = external global i30		; <ptr> [#uses=1]
@i30_s = external global i30		; <ptr> [#uses=1]
@i31_l = external global i31		; <ptr> [#uses=1]
@i31_s = external global i31		; <ptr> [#uses=1]
@i32_l = external global i32		; <ptr> [#uses=1]
@i32_s = external global i32		; <ptr> [#uses=1]
@i33_l = external global i33		; <ptr> [#uses=1]
@i33_s = external global i33		; <ptr> [#uses=1]
@i34_l = external global i34		; <ptr> [#uses=1]
@i34_s = external global i34		; <ptr> [#uses=1]
@i35_l = external global i35		; <ptr> [#uses=1]
@i35_s = external global i35		; <ptr> [#uses=1]
@i36_l = external global i36		; <ptr> [#uses=1]
@i36_s = external global i36		; <ptr> [#uses=1]
@i37_l = external global i37		; <ptr> [#uses=1]
@i37_s = external global i37		; <ptr> [#uses=1]
@i38_l = external global i38		; <ptr> [#uses=1]
@i38_s = external global i38		; <ptr> [#uses=1]
@i39_l = external global i39		; <ptr> [#uses=1]
@i39_s = external global i39		; <ptr> [#uses=1]
@i40_l = external global i40		; <ptr> [#uses=1]
@i40_s = external global i40		; <ptr> [#uses=1]
@i41_l = external global i41		; <ptr> [#uses=1]
@i41_s = external global i41		; <ptr> [#uses=1]
@i42_l = external global i42		; <ptr> [#uses=1]
@i42_s = external global i42		; <ptr> [#uses=1]
@i43_l = external global i43		; <ptr> [#uses=1]
@i43_s = external global i43		; <ptr> [#uses=1]
@i44_l = external global i44		; <ptr> [#uses=1]
@i44_s = external global i44		; <ptr> [#uses=1]
@i45_l = external global i45		; <ptr> [#uses=1]
@i45_s = external global i45		; <ptr> [#uses=1]
@i46_l = external global i46		; <ptr> [#uses=1]
@i46_s = external global i46		; <ptr> [#uses=1]
@i47_l = external global i47		; <ptr> [#uses=1]
@i47_s = external global i47		; <ptr> [#uses=1]
@i48_l = external global i48		; <ptr> [#uses=1]
@i48_s = external global i48		; <ptr> [#uses=1]
@i49_l = external global i49		; <ptr> [#uses=1]
@i49_s = external global i49		; <ptr> [#uses=1]
@i50_l = external global i50		; <ptr> [#uses=1]
@i50_s = external global i50		; <ptr> [#uses=1]
@i51_l = external global i51		; <ptr> [#uses=1]
@i51_s = external global i51		; <ptr> [#uses=1]
@i52_l = external global i52		; <ptr> [#uses=1]
@i52_s = external global i52		; <ptr> [#uses=1]
@i53_l = external global i53		; <ptr> [#uses=1]
@i53_s = external global i53		; <ptr> [#uses=1]
@i54_l = external global i54		; <ptr> [#uses=1]
@i54_s = external global i54		; <ptr> [#uses=1]
@i55_l = external global i55		; <ptr> [#uses=1]
@i55_s = external global i55		; <ptr> [#uses=1]
@i56_l = external global i56		; <ptr> [#uses=1]
@i56_s = external global i56		; <ptr> [#uses=1]
@i57_l = external global i57		; <ptr> [#uses=1]
@i57_s = external global i57		; <ptr> [#uses=1]
@i58_l = external global i58		; <ptr> [#uses=1]
@i58_s = external global i58		; <ptr> [#uses=1]
@i59_l = external global i59		; <ptr> [#uses=1]
@i59_s = external global i59		; <ptr> [#uses=1]
@i60_l = external global i60		; <ptr> [#uses=1]
@i60_s = external global i60		; <ptr> [#uses=1]
@i61_l = external global i61		; <ptr> [#uses=1]
@i61_s = external global i61		; <ptr> [#uses=1]
@i62_l = external global i62		; <ptr> [#uses=1]
@i62_s = external global i62		; <ptr> [#uses=1]
@i63_l = external global i63		; <ptr> [#uses=1]
@i63_s = external global i63		; <ptr> [#uses=1]
@i64_l = external global i64		; <ptr> [#uses=1]
@i64_s = external global i64		; <ptr> [#uses=1]

define void @i1_ls() nounwind  {
	%tmp = load i1, ptr @i1_l		; <i1> [#uses=1]
	store i1 %tmp, ptr @i1_s
	ret void
}

define void @i2_ls() nounwind  {
	%tmp = load i2, ptr @i2_l		; <i2> [#uses=1]
	store i2 %tmp, ptr @i2_s
	ret void
}

define void @i3_ls() nounwind  {
	%tmp = load i3, ptr @i3_l		; <i3> [#uses=1]
	store i3 %tmp, ptr @i3_s
	ret void
}

define void @i4_ls() nounwind  {
	%tmp = load i4, ptr @i4_l		; <i4> [#uses=1]
	store i4 %tmp, ptr @i4_s
	ret void
}

define void @i5_ls() nounwind  {
	%tmp = load i5, ptr @i5_l		; <i5> [#uses=1]
	store i5 %tmp, ptr @i5_s
	ret void
}

define void @i6_ls() nounwind  {
	%tmp = load i6, ptr @i6_l		; <i6> [#uses=1]
	store i6 %tmp, ptr @i6_s
	ret void
}

define void @i7_ls() nounwind  {
	%tmp = load i7, ptr @i7_l		; <i7> [#uses=1]
	store i7 %tmp, ptr @i7_s
	ret void
}

define void @i8_ls() nounwind  {
	%tmp = load i8, ptr @i8_l		; <i8> [#uses=1]
	store i8 %tmp, ptr @i8_s
	ret void
}

define void @i9_ls() nounwind  {
	%tmp = load i9, ptr @i9_l		; <i9> [#uses=1]
	store i9 %tmp, ptr @i9_s
	ret void
}

define void @i10_ls() nounwind  {
	%tmp = load i10, ptr @i10_l		; <i10> [#uses=1]
	store i10 %tmp, ptr @i10_s
	ret void
}

define void @i11_ls() nounwind  {
	%tmp = load i11, ptr @i11_l		; <i11> [#uses=1]
	store i11 %tmp, ptr @i11_s
	ret void
}

define void @i12_ls() nounwind  {
	%tmp = load i12, ptr @i12_l		; <i12> [#uses=1]
	store i12 %tmp, ptr @i12_s
	ret void
}

define void @i13_ls() nounwind  {
	%tmp = load i13, ptr @i13_l		; <i13> [#uses=1]
	store i13 %tmp, ptr @i13_s
	ret void
}

define void @i14_ls() nounwind  {
	%tmp = load i14, ptr @i14_l		; <i14> [#uses=1]
	store i14 %tmp, ptr @i14_s
	ret void
}

define void @i15_ls() nounwind  {
	%tmp = load i15, ptr @i15_l		; <i15> [#uses=1]
	store i15 %tmp, ptr @i15_s
	ret void
}

define void @i16_ls() nounwind  {
	%tmp = load i16, ptr @i16_l		; <i16> [#uses=1]
	store i16 %tmp, ptr @i16_s
	ret void
}

define void @i17_ls() nounwind  {
	%tmp = load i17, ptr @i17_l		; <i17> [#uses=1]
	store i17 %tmp, ptr @i17_s
	ret void
}

define void @i18_ls() nounwind  {
	%tmp = load i18, ptr @i18_l		; <i18> [#uses=1]
	store i18 %tmp, ptr @i18_s
	ret void
}

define void @i19_ls() nounwind  {
	%tmp = load i19, ptr @i19_l		; <i19> [#uses=1]
	store i19 %tmp, ptr @i19_s
	ret void
}

define void @i20_ls() nounwind  {
	%tmp = load i20, ptr @i20_l		; <i20> [#uses=1]
	store i20 %tmp, ptr @i20_s
	ret void
}

define void @i21_ls() nounwind  {
	%tmp = load i21, ptr @i21_l		; <i21> [#uses=1]
	store i21 %tmp, ptr @i21_s
	ret void
}

define void @i22_ls() nounwind  {
	%tmp = load i22, ptr @i22_l		; <i22> [#uses=1]
	store i22 %tmp, ptr @i22_s
	ret void
}

define void @i23_ls() nounwind  {
	%tmp = load i23, ptr @i23_l		; <i23> [#uses=1]
	store i23 %tmp, ptr @i23_s
	ret void
}

define void @i24_ls() nounwind  {
	%tmp = load i24, ptr @i24_l		; <i24> [#uses=1]
	store i24 %tmp, ptr @i24_s
	ret void
}

define void @i25_ls() nounwind  {
	%tmp = load i25, ptr @i25_l		; <i25> [#uses=1]
	store i25 %tmp, ptr @i25_s
	ret void
}

define void @i26_ls() nounwind  {
	%tmp = load i26, ptr @i26_l		; <i26> [#uses=1]
	store i26 %tmp, ptr @i26_s
	ret void
}

define void @i27_ls() nounwind  {
	%tmp = load i27, ptr @i27_l		; <i27> [#uses=1]
	store i27 %tmp, ptr @i27_s
	ret void
}

define void @i28_ls() nounwind  {
	%tmp = load i28, ptr @i28_l		; <i28> [#uses=1]
	store i28 %tmp, ptr @i28_s
	ret void
}

define void @i29_ls() nounwind  {
	%tmp = load i29, ptr @i29_l		; <i29> [#uses=1]
	store i29 %tmp, ptr @i29_s
	ret void
}

define void @i30_ls() nounwind  {
	%tmp = load i30, ptr @i30_l		; <i30> [#uses=1]
	store i30 %tmp, ptr @i30_s
	ret void
}

define void @i31_ls() nounwind  {
	%tmp = load i31, ptr @i31_l		; <i31> [#uses=1]
	store i31 %tmp, ptr @i31_s
	ret void
}

define void @i32_ls() nounwind  {
	%tmp = load i32, ptr @i32_l		; <i32> [#uses=1]
	store i32 %tmp, ptr @i32_s
	ret void
}

define void @i33_ls() nounwind  {
	%tmp = load i33, ptr @i33_l		; <i33> [#uses=1]
	store i33 %tmp, ptr @i33_s
	ret void
}

define void @i34_ls() nounwind  {
	%tmp = load i34, ptr @i34_l		; <i34> [#uses=1]
	store i34 %tmp, ptr @i34_s
	ret void
}

define void @i35_ls() nounwind  {
	%tmp = load i35, ptr @i35_l		; <i35> [#uses=1]
	store i35 %tmp, ptr @i35_s
	ret void
}

define void @i36_ls() nounwind  {
	%tmp = load i36, ptr @i36_l		; <i36> [#uses=1]
	store i36 %tmp, ptr @i36_s
	ret void
}

define void @i37_ls() nounwind  {
	%tmp = load i37, ptr @i37_l		; <i37> [#uses=1]
	store i37 %tmp, ptr @i37_s
	ret void
}

define void @i38_ls() nounwind  {
	%tmp = load i38, ptr @i38_l		; <i38> [#uses=1]
	store i38 %tmp, ptr @i38_s
	ret void
}

define void @i39_ls() nounwind  {
	%tmp = load i39, ptr @i39_l		; <i39> [#uses=1]
	store i39 %tmp, ptr @i39_s
	ret void
}

define void @i40_ls() nounwind  {
	%tmp = load i40, ptr @i40_l		; <i40> [#uses=1]
	store i40 %tmp, ptr @i40_s
	ret void
}

define void @i41_ls() nounwind  {
	%tmp = load i41, ptr @i41_l		; <i41> [#uses=1]
	store i41 %tmp, ptr @i41_s
	ret void
}

define void @i42_ls() nounwind  {
	%tmp = load i42, ptr @i42_l		; <i42> [#uses=1]
	store i42 %tmp, ptr @i42_s
	ret void
}

define void @i43_ls() nounwind  {
	%tmp = load i43, ptr @i43_l		; <i43> [#uses=1]
	store i43 %tmp, ptr @i43_s
	ret void
}

define void @i44_ls() nounwind  {
	%tmp = load i44, ptr @i44_l		; <i44> [#uses=1]
	store i44 %tmp, ptr @i44_s
	ret void
}

define void @i45_ls() nounwind  {
	%tmp = load i45, ptr @i45_l		; <i45> [#uses=1]
	store i45 %tmp, ptr @i45_s
	ret void
}

define void @i46_ls() nounwind  {
	%tmp = load i46, ptr @i46_l		; <i46> [#uses=1]
	store i46 %tmp, ptr @i46_s
	ret void
}

define void @i47_ls() nounwind  {
	%tmp = load i47, ptr @i47_l		; <i47> [#uses=1]
	store i47 %tmp, ptr @i47_s
	ret void
}

define void @i48_ls() nounwind  {
	%tmp = load i48, ptr @i48_l		; <i48> [#uses=1]
	store i48 %tmp, ptr @i48_s
	ret void
}

define void @i49_ls() nounwind  {
	%tmp = load i49, ptr @i49_l		; <i49> [#uses=1]
	store i49 %tmp, ptr @i49_s
	ret void
}

define void @i50_ls() nounwind  {
	%tmp = load i50, ptr @i50_l		; <i50> [#uses=1]
	store i50 %tmp, ptr @i50_s
	ret void
}

define void @i51_ls() nounwind  {
	%tmp = load i51, ptr @i51_l		; <i51> [#uses=1]
	store i51 %tmp, ptr @i51_s
	ret void
}

define void @i52_ls() nounwind  {
	%tmp = load i52, ptr @i52_l		; <i52> [#uses=1]
	store i52 %tmp, ptr @i52_s
	ret void
}

define void @i53_ls() nounwind  {
	%tmp = load i53, ptr @i53_l		; <i53> [#uses=1]
	store i53 %tmp, ptr @i53_s
	ret void
}

define void @i54_ls() nounwind  {
	%tmp = load i54, ptr @i54_l		; <i54> [#uses=1]
	store i54 %tmp, ptr @i54_s
	ret void
}

define void @i55_ls() nounwind  {
	%tmp = load i55, ptr @i55_l		; <i55> [#uses=1]
	store i55 %tmp, ptr @i55_s
	ret void
}

define void @i56_ls() nounwind  {
	%tmp = load i56, ptr @i56_l		; <i56> [#uses=1]
	store i56 %tmp, ptr @i56_s
	ret void
}

define void @i57_ls() nounwind  {
	%tmp = load i57, ptr @i57_l		; <i57> [#uses=1]
	store i57 %tmp, ptr @i57_s
	ret void
}

define void @i58_ls() nounwind  {
	%tmp = load i58, ptr @i58_l		; <i58> [#uses=1]
	store i58 %tmp, ptr @i58_s
	ret void
}

define void @i59_ls() nounwind  {
	%tmp = load i59, ptr @i59_l		; <i59> [#uses=1]
	store i59 %tmp, ptr @i59_s
	ret void
}

define void @i60_ls() nounwind  {
	%tmp = load i60, ptr @i60_l		; <i60> [#uses=1]
	store i60 %tmp, ptr @i60_s
	ret void
}

define void @i61_ls() nounwind  {
	%tmp = load i61, ptr @i61_l		; <i61> [#uses=1]
	store i61 %tmp, ptr @i61_s
	ret void
}

define void @i62_ls() nounwind  {
	%tmp = load i62, ptr @i62_l		; <i62> [#uses=1]
	store i62 %tmp, ptr @i62_s
	ret void
}

define void @i63_ls() nounwind  {
	%tmp = load i63, ptr @i63_l		; <i63> [#uses=1]
	store i63 %tmp, ptr @i63_s
	ret void
}

define void @i64_ls() nounwind  {
	%tmp = load i64, ptr @i64_l		; <i64> [#uses=1]
	store i64 %tmp, ptr @i64_s
	ret void
}
