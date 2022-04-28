; RUN: llc -march=nvptx  < %s > %t
; RUN: llc -march=nvptx64  < %s > %t

@i1_l = external global i1		; <i1*> [#uses=1]
@i1_s = external global i1		; <i1*> [#uses=1]
@i2_l = external global i2		; <i2*> [#uses=1]
@i2_s = external global i2		; <i2*> [#uses=1]
@i3_l = external global i3		; <i3*> [#uses=1]
@i3_s = external global i3		; <i3*> [#uses=1]
@i4_l = external global i4		; <i4*> [#uses=1]
@i4_s = external global i4		; <i4*> [#uses=1]
@i5_l = external global i5		; <i5*> [#uses=1]
@i5_s = external global i5		; <i5*> [#uses=1]
@i6_l = external global i6		; <i6*> [#uses=1]
@i6_s = external global i6		; <i6*> [#uses=1]
@i7_l = external global i7		; <i7*> [#uses=1]
@i7_s = external global i7		; <i7*> [#uses=1]
@i8_l = external global i8		; <i8*> [#uses=1]
@i8_s = external global i8		; <i8*> [#uses=1]
@i9_l = external global i9		; <i9*> [#uses=1]
@i9_s = external global i9		; <i9*> [#uses=1]
@i10_l = external global i10		; <i10*> [#uses=1]
@i10_s = external global i10		; <i10*> [#uses=1]
@i11_l = external global i11		; <i11*> [#uses=1]
@i11_s = external global i11		; <i11*> [#uses=1]
@i12_l = external global i12		; <i12*> [#uses=1]
@i12_s = external global i12		; <i12*> [#uses=1]
@i13_l = external global i13		; <i13*> [#uses=1]
@i13_s = external global i13		; <i13*> [#uses=1]
@i14_l = external global i14		; <i14*> [#uses=1]
@i14_s = external global i14		; <i14*> [#uses=1]
@i15_l = external global i15		; <i15*> [#uses=1]
@i15_s = external global i15		; <i15*> [#uses=1]
@i16_l = external global i16		; <i16*> [#uses=1]
@i16_s = external global i16		; <i16*> [#uses=1]
@i17_l = external global i17		; <i17*> [#uses=1]
@i17_s = external global i17		; <i17*> [#uses=1]
@i18_l = external global i18		; <i18*> [#uses=1]
@i18_s = external global i18		; <i18*> [#uses=1]
@i19_l = external global i19		; <i19*> [#uses=1]
@i19_s = external global i19		; <i19*> [#uses=1]
@i20_l = external global i20		; <i20*> [#uses=1]
@i20_s = external global i20		; <i20*> [#uses=1]
@i21_l = external global i21		; <i21*> [#uses=1]
@i21_s = external global i21		; <i21*> [#uses=1]
@i22_l = external global i22		; <i22*> [#uses=1]
@i22_s = external global i22		; <i22*> [#uses=1]
@i23_l = external global i23		; <i23*> [#uses=1]
@i23_s = external global i23		; <i23*> [#uses=1]
@i24_l = external global i24		; <i24*> [#uses=1]
@i24_s = external global i24		; <i24*> [#uses=1]
@i25_l = external global i25		; <i25*> [#uses=1]
@i25_s = external global i25		; <i25*> [#uses=1]
@i26_l = external global i26		; <i26*> [#uses=1]
@i26_s = external global i26		; <i26*> [#uses=1]
@i27_l = external global i27		; <i27*> [#uses=1]
@i27_s = external global i27		; <i27*> [#uses=1]
@i28_l = external global i28		; <i28*> [#uses=1]
@i28_s = external global i28		; <i28*> [#uses=1]
@i29_l = external global i29		; <i29*> [#uses=1]
@i29_s = external global i29		; <i29*> [#uses=1]
@i30_l = external global i30		; <i30*> [#uses=1]
@i30_s = external global i30		; <i30*> [#uses=1]
@i31_l = external global i31		; <i31*> [#uses=1]
@i31_s = external global i31		; <i31*> [#uses=1]
@i32_l = external global i32		; <i32*> [#uses=1]
@i32_s = external global i32		; <i32*> [#uses=1]
@i33_l = external global i33		; <i33*> [#uses=1]
@i33_s = external global i33		; <i33*> [#uses=1]
@i34_l = external global i34		; <i34*> [#uses=1]
@i34_s = external global i34		; <i34*> [#uses=1]
@i35_l = external global i35		; <i35*> [#uses=1]
@i35_s = external global i35		; <i35*> [#uses=1]
@i36_l = external global i36		; <i36*> [#uses=1]
@i36_s = external global i36		; <i36*> [#uses=1]
@i37_l = external global i37		; <i37*> [#uses=1]
@i37_s = external global i37		; <i37*> [#uses=1]
@i38_l = external global i38		; <i38*> [#uses=1]
@i38_s = external global i38		; <i38*> [#uses=1]
@i39_l = external global i39		; <i39*> [#uses=1]
@i39_s = external global i39		; <i39*> [#uses=1]
@i40_l = external global i40		; <i40*> [#uses=1]
@i40_s = external global i40		; <i40*> [#uses=1]
@i41_l = external global i41		; <i41*> [#uses=1]
@i41_s = external global i41		; <i41*> [#uses=1]
@i42_l = external global i42		; <i42*> [#uses=1]
@i42_s = external global i42		; <i42*> [#uses=1]
@i43_l = external global i43		; <i43*> [#uses=1]
@i43_s = external global i43		; <i43*> [#uses=1]
@i44_l = external global i44		; <i44*> [#uses=1]
@i44_s = external global i44		; <i44*> [#uses=1]
@i45_l = external global i45		; <i45*> [#uses=1]
@i45_s = external global i45		; <i45*> [#uses=1]
@i46_l = external global i46		; <i46*> [#uses=1]
@i46_s = external global i46		; <i46*> [#uses=1]
@i47_l = external global i47		; <i47*> [#uses=1]
@i47_s = external global i47		; <i47*> [#uses=1]
@i48_l = external global i48		; <i48*> [#uses=1]
@i48_s = external global i48		; <i48*> [#uses=1]
@i49_l = external global i49		; <i49*> [#uses=1]
@i49_s = external global i49		; <i49*> [#uses=1]
@i50_l = external global i50		; <i50*> [#uses=1]
@i50_s = external global i50		; <i50*> [#uses=1]
@i51_l = external global i51		; <i51*> [#uses=1]
@i51_s = external global i51		; <i51*> [#uses=1]
@i52_l = external global i52		; <i52*> [#uses=1]
@i52_s = external global i52		; <i52*> [#uses=1]
@i53_l = external global i53		; <i53*> [#uses=1]
@i53_s = external global i53		; <i53*> [#uses=1]
@i54_l = external global i54		; <i54*> [#uses=1]
@i54_s = external global i54		; <i54*> [#uses=1]
@i55_l = external global i55		; <i55*> [#uses=1]
@i55_s = external global i55		; <i55*> [#uses=1]
@i56_l = external global i56		; <i56*> [#uses=1]
@i56_s = external global i56		; <i56*> [#uses=1]
@i57_l = external global i57		; <i57*> [#uses=1]
@i57_s = external global i57		; <i57*> [#uses=1]
@i58_l = external global i58		; <i58*> [#uses=1]
@i58_s = external global i58		; <i58*> [#uses=1]
@i59_l = external global i59		; <i59*> [#uses=1]
@i59_s = external global i59		; <i59*> [#uses=1]
@i60_l = external global i60		; <i60*> [#uses=1]
@i60_s = external global i60		; <i60*> [#uses=1]
@i61_l = external global i61		; <i61*> [#uses=1]
@i61_s = external global i61		; <i61*> [#uses=1]
@i62_l = external global i62		; <i62*> [#uses=1]
@i62_s = external global i62		; <i62*> [#uses=1]
@i63_l = external global i63		; <i63*> [#uses=1]
@i63_s = external global i63		; <i63*> [#uses=1]
@i64_l = external global i64		; <i64*> [#uses=1]
@i64_s = external global i64		; <i64*> [#uses=1]

define void @i1_ls() nounwind  {
	%tmp = load i1, i1* @i1_l		; <i1> [#uses=1]
	store i1 %tmp, i1* @i1_s
	ret void
}

define void @i2_ls() nounwind  {
	%tmp = load i2, i2* @i2_l		; <i2> [#uses=1]
	store i2 %tmp, i2* @i2_s
	ret void
}

define void @i3_ls() nounwind  {
	%tmp = load i3, i3* @i3_l		; <i3> [#uses=1]
	store i3 %tmp, i3* @i3_s
	ret void
}

define void @i4_ls() nounwind  {
	%tmp = load i4, i4* @i4_l		; <i4> [#uses=1]
	store i4 %tmp, i4* @i4_s
	ret void
}

define void @i5_ls() nounwind  {
	%tmp = load i5, i5* @i5_l		; <i5> [#uses=1]
	store i5 %tmp, i5* @i5_s
	ret void
}

define void @i6_ls() nounwind  {
	%tmp = load i6, i6* @i6_l		; <i6> [#uses=1]
	store i6 %tmp, i6* @i6_s
	ret void
}

define void @i7_ls() nounwind  {
	%tmp = load i7, i7* @i7_l		; <i7> [#uses=1]
	store i7 %tmp, i7* @i7_s
	ret void
}

define void @i8_ls() nounwind  {
	%tmp = load i8, i8* @i8_l		; <i8> [#uses=1]
	store i8 %tmp, i8* @i8_s
	ret void
}

define void @i9_ls() nounwind  {
	%tmp = load i9, i9* @i9_l		; <i9> [#uses=1]
	store i9 %tmp, i9* @i9_s
	ret void
}

define void @i10_ls() nounwind  {
	%tmp = load i10, i10* @i10_l		; <i10> [#uses=1]
	store i10 %tmp, i10* @i10_s
	ret void
}

define void @i11_ls() nounwind  {
	%tmp = load i11, i11* @i11_l		; <i11> [#uses=1]
	store i11 %tmp, i11* @i11_s
	ret void
}

define void @i12_ls() nounwind  {
	%tmp = load i12, i12* @i12_l		; <i12> [#uses=1]
	store i12 %tmp, i12* @i12_s
	ret void
}

define void @i13_ls() nounwind  {
	%tmp = load i13, i13* @i13_l		; <i13> [#uses=1]
	store i13 %tmp, i13* @i13_s
	ret void
}

define void @i14_ls() nounwind  {
	%tmp = load i14, i14* @i14_l		; <i14> [#uses=1]
	store i14 %tmp, i14* @i14_s
	ret void
}

define void @i15_ls() nounwind  {
	%tmp = load i15, i15* @i15_l		; <i15> [#uses=1]
	store i15 %tmp, i15* @i15_s
	ret void
}

define void @i16_ls() nounwind  {
	%tmp = load i16, i16* @i16_l		; <i16> [#uses=1]
	store i16 %tmp, i16* @i16_s
	ret void
}

define void @i17_ls() nounwind  {
	%tmp = load i17, i17* @i17_l		; <i17> [#uses=1]
	store i17 %tmp, i17* @i17_s
	ret void
}

define void @i18_ls() nounwind  {
	%tmp = load i18, i18* @i18_l		; <i18> [#uses=1]
	store i18 %tmp, i18* @i18_s
	ret void
}

define void @i19_ls() nounwind  {
	%tmp = load i19, i19* @i19_l		; <i19> [#uses=1]
	store i19 %tmp, i19* @i19_s
	ret void
}

define void @i20_ls() nounwind  {
	%tmp = load i20, i20* @i20_l		; <i20> [#uses=1]
	store i20 %tmp, i20* @i20_s
	ret void
}

define void @i21_ls() nounwind  {
	%tmp = load i21, i21* @i21_l		; <i21> [#uses=1]
	store i21 %tmp, i21* @i21_s
	ret void
}

define void @i22_ls() nounwind  {
	%tmp = load i22, i22* @i22_l		; <i22> [#uses=1]
	store i22 %tmp, i22* @i22_s
	ret void
}

define void @i23_ls() nounwind  {
	%tmp = load i23, i23* @i23_l		; <i23> [#uses=1]
	store i23 %tmp, i23* @i23_s
	ret void
}

define void @i24_ls() nounwind  {
	%tmp = load i24, i24* @i24_l		; <i24> [#uses=1]
	store i24 %tmp, i24* @i24_s
	ret void
}

define void @i25_ls() nounwind  {
	%tmp = load i25, i25* @i25_l		; <i25> [#uses=1]
	store i25 %tmp, i25* @i25_s
	ret void
}

define void @i26_ls() nounwind  {
	%tmp = load i26, i26* @i26_l		; <i26> [#uses=1]
	store i26 %tmp, i26* @i26_s
	ret void
}

define void @i27_ls() nounwind  {
	%tmp = load i27, i27* @i27_l		; <i27> [#uses=1]
	store i27 %tmp, i27* @i27_s
	ret void
}

define void @i28_ls() nounwind  {
	%tmp = load i28, i28* @i28_l		; <i28> [#uses=1]
	store i28 %tmp, i28* @i28_s
	ret void
}

define void @i29_ls() nounwind  {
	%tmp = load i29, i29* @i29_l		; <i29> [#uses=1]
	store i29 %tmp, i29* @i29_s
	ret void
}

define void @i30_ls() nounwind  {
	%tmp = load i30, i30* @i30_l		; <i30> [#uses=1]
	store i30 %tmp, i30* @i30_s
	ret void
}

define void @i31_ls() nounwind  {
	%tmp = load i31, i31* @i31_l		; <i31> [#uses=1]
	store i31 %tmp, i31* @i31_s
	ret void
}

define void @i32_ls() nounwind  {
	%tmp = load i32, i32* @i32_l		; <i32> [#uses=1]
	store i32 %tmp, i32* @i32_s
	ret void
}

define void @i33_ls() nounwind  {
	%tmp = load i33, i33* @i33_l		; <i33> [#uses=1]
	store i33 %tmp, i33* @i33_s
	ret void
}

define void @i34_ls() nounwind  {
	%tmp = load i34, i34* @i34_l		; <i34> [#uses=1]
	store i34 %tmp, i34* @i34_s
	ret void
}

define void @i35_ls() nounwind  {
	%tmp = load i35, i35* @i35_l		; <i35> [#uses=1]
	store i35 %tmp, i35* @i35_s
	ret void
}

define void @i36_ls() nounwind  {
	%tmp = load i36, i36* @i36_l		; <i36> [#uses=1]
	store i36 %tmp, i36* @i36_s
	ret void
}

define void @i37_ls() nounwind  {
	%tmp = load i37, i37* @i37_l		; <i37> [#uses=1]
	store i37 %tmp, i37* @i37_s
	ret void
}

define void @i38_ls() nounwind  {
	%tmp = load i38, i38* @i38_l		; <i38> [#uses=1]
	store i38 %tmp, i38* @i38_s
	ret void
}

define void @i39_ls() nounwind  {
	%tmp = load i39, i39* @i39_l		; <i39> [#uses=1]
	store i39 %tmp, i39* @i39_s
	ret void
}

define void @i40_ls() nounwind  {
	%tmp = load i40, i40* @i40_l		; <i40> [#uses=1]
	store i40 %tmp, i40* @i40_s
	ret void
}

define void @i41_ls() nounwind  {
	%tmp = load i41, i41* @i41_l		; <i41> [#uses=1]
	store i41 %tmp, i41* @i41_s
	ret void
}

define void @i42_ls() nounwind  {
	%tmp = load i42, i42* @i42_l		; <i42> [#uses=1]
	store i42 %tmp, i42* @i42_s
	ret void
}

define void @i43_ls() nounwind  {
	%tmp = load i43, i43* @i43_l		; <i43> [#uses=1]
	store i43 %tmp, i43* @i43_s
	ret void
}

define void @i44_ls() nounwind  {
	%tmp = load i44, i44* @i44_l		; <i44> [#uses=1]
	store i44 %tmp, i44* @i44_s
	ret void
}

define void @i45_ls() nounwind  {
	%tmp = load i45, i45* @i45_l		; <i45> [#uses=1]
	store i45 %tmp, i45* @i45_s
	ret void
}

define void @i46_ls() nounwind  {
	%tmp = load i46, i46* @i46_l		; <i46> [#uses=1]
	store i46 %tmp, i46* @i46_s
	ret void
}

define void @i47_ls() nounwind  {
	%tmp = load i47, i47* @i47_l		; <i47> [#uses=1]
	store i47 %tmp, i47* @i47_s
	ret void
}

define void @i48_ls() nounwind  {
	%tmp = load i48, i48* @i48_l		; <i48> [#uses=1]
	store i48 %tmp, i48* @i48_s
	ret void
}

define void @i49_ls() nounwind  {
	%tmp = load i49, i49* @i49_l		; <i49> [#uses=1]
	store i49 %tmp, i49* @i49_s
	ret void
}

define void @i50_ls() nounwind  {
	%tmp = load i50, i50* @i50_l		; <i50> [#uses=1]
	store i50 %tmp, i50* @i50_s
	ret void
}

define void @i51_ls() nounwind  {
	%tmp = load i51, i51* @i51_l		; <i51> [#uses=1]
	store i51 %tmp, i51* @i51_s
	ret void
}

define void @i52_ls() nounwind  {
	%tmp = load i52, i52* @i52_l		; <i52> [#uses=1]
	store i52 %tmp, i52* @i52_s
	ret void
}

define void @i53_ls() nounwind  {
	%tmp = load i53, i53* @i53_l		; <i53> [#uses=1]
	store i53 %tmp, i53* @i53_s
	ret void
}

define void @i54_ls() nounwind  {
	%tmp = load i54, i54* @i54_l		; <i54> [#uses=1]
	store i54 %tmp, i54* @i54_s
	ret void
}

define void @i55_ls() nounwind  {
	%tmp = load i55, i55* @i55_l		; <i55> [#uses=1]
	store i55 %tmp, i55* @i55_s
	ret void
}

define void @i56_ls() nounwind  {
	%tmp = load i56, i56* @i56_l		; <i56> [#uses=1]
	store i56 %tmp, i56* @i56_s
	ret void
}

define void @i57_ls() nounwind  {
	%tmp = load i57, i57* @i57_l		; <i57> [#uses=1]
	store i57 %tmp, i57* @i57_s
	ret void
}

define void @i58_ls() nounwind  {
	%tmp = load i58, i58* @i58_l		; <i58> [#uses=1]
	store i58 %tmp, i58* @i58_s
	ret void
}

define void @i59_ls() nounwind  {
	%tmp = load i59, i59* @i59_l		; <i59> [#uses=1]
	store i59 %tmp, i59* @i59_s
	ret void
}

define void @i60_ls() nounwind  {
	%tmp = load i60, i60* @i60_l		; <i60> [#uses=1]
	store i60 %tmp, i60* @i60_s
	ret void
}

define void @i61_ls() nounwind  {
	%tmp = load i61, i61* @i61_l		; <i61> [#uses=1]
	store i61 %tmp, i61* @i61_s
	ret void
}

define void @i62_ls() nounwind  {
	%tmp = load i62, i62* @i62_l		; <i62> [#uses=1]
	store i62 %tmp, i62* @i62_s
	ret void
}

define void @i63_ls() nounwind  {
	%tmp = load i63, i63* @i63_l		; <i63> [#uses=1]
	store i63 %tmp, i63* @i63_s
	ret void
}

define void @i64_ls() nounwind  {
	%tmp = load i64, i64* @i64_l		; <i64> [#uses=1]
	store i64 %tmp, i64* @i64_s
	ret void
}
