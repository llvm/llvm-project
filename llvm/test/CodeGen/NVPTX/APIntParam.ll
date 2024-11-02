; RUN: llc -march=nvptx  < %s > %t
; RUN: llc -march=nvptx64  < %s > %t

@i1_s = external global i1		; <i1*> [#uses=1]
@i2_s = external global i2		; <i2*> [#uses=1]
@i3_s = external global i3		; <i3*> [#uses=1]
@i4_s = external global i4		; <i4*> [#uses=1]
@i5_s = external global i5		; <i5*> [#uses=1]
@i6_s = external global i6		; <i6*> [#uses=1]
@i7_s = external global i7		; <i7*> [#uses=1]
@i8_s = external global i8		; <i8*> [#uses=1]
@i9_s = external global i9		; <i9*> [#uses=1]
@i10_s = external global i10		; <i10*> [#uses=1]
@i11_s = external global i11		; <i11*> [#uses=1]
@i12_s = external global i12		; <i12*> [#uses=1]
@i13_s = external global i13		; <i13*> [#uses=1]
@i14_s = external global i14		; <i14*> [#uses=1]
@i15_s = external global i15		; <i15*> [#uses=1]
@i16_s = external global i16		; <i16*> [#uses=1]
@i17_s = external global i17		; <i17*> [#uses=1]
@i18_s = external global i18		; <i18*> [#uses=1]
@i19_s = external global i19		; <i19*> [#uses=1]
@i20_s = external global i20		; <i20*> [#uses=1]
@i21_s = external global i21		; <i21*> [#uses=1]
@i22_s = external global i22		; <i22*> [#uses=1]
@i23_s = external global i23		; <i23*> [#uses=1]
@i24_s = external global i24		; <i24*> [#uses=1]
@i25_s = external global i25		; <i25*> [#uses=1]
@i26_s = external global i26		; <i26*> [#uses=1]
@i27_s = external global i27		; <i27*> [#uses=1]
@i28_s = external global i28		; <i28*> [#uses=1]
@i29_s = external global i29		; <i29*> [#uses=1]
@i30_s = external global i30		; <i30*> [#uses=1]
@i31_s = external global i31		; <i31*> [#uses=1]
@i32_s = external global i32		; <i32*> [#uses=1]
@i33_s = external global i33		; <i33*> [#uses=1]
@i34_s = external global i34		; <i34*> [#uses=1]
@i35_s = external global i35		; <i35*> [#uses=1]
@i36_s = external global i36		; <i36*> [#uses=1]
@i37_s = external global i37		; <i37*> [#uses=1]
@i38_s = external global i38		; <i38*> [#uses=1]
@i39_s = external global i39		; <i39*> [#uses=1]
@i40_s = external global i40		; <i40*> [#uses=1]
@i41_s = external global i41		; <i41*> [#uses=1]
@i42_s = external global i42		; <i42*> [#uses=1]
@i43_s = external global i43		; <i43*> [#uses=1]
@i44_s = external global i44		; <i44*> [#uses=1]
@i45_s = external global i45		; <i45*> [#uses=1]
@i46_s = external global i46		; <i46*> [#uses=1]
@i47_s = external global i47		; <i47*> [#uses=1]
@i48_s = external global i48		; <i48*> [#uses=1]
@i49_s = external global i49		; <i49*> [#uses=1]
@i50_s = external global i50		; <i50*> [#uses=1]
@i51_s = external global i51		; <i51*> [#uses=1]
@i52_s = external global i52		; <i52*> [#uses=1]
@i53_s = external global i53		; <i53*> [#uses=1]
@i54_s = external global i54		; <i54*> [#uses=1]
@i55_s = external global i55		; <i55*> [#uses=1]
@i56_s = external global i56		; <i56*> [#uses=1]
@i57_s = external global i57		; <i57*> [#uses=1]
@i58_s = external global i58		; <i58*> [#uses=1]
@i59_s = external global i59		; <i59*> [#uses=1]
@i60_s = external global i60		; <i60*> [#uses=1]
@i61_s = external global i61		; <i61*> [#uses=1]
@i62_s = external global i62		; <i62*> [#uses=1]
@i63_s = external global i63		; <i63*> [#uses=1]
@i64_s = external global i64		; <i64*> [#uses=1]

define void @i1_ls(i1 %x) nounwind  {
	store i1 %x, i1* @i1_s
	ret void
}

define void @i2_ls(i2 %x) nounwind  {
	store i2 %x, i2* @i2_s
	ret void
}

define void @i3_ls(i3 %x) nounwind  {
	store i3 %x, i3* @i3_s
	ret void
}

define void @i4_ls(i4 %x) nounwind  {
	store i4 %x, i4* @i4_s
	ret void
}

define void @i5_ls(i5 %x) nounwind  {
	store i5 %x, i5* @i5_s
	ret void
}

define void @i6_ls(i6 %x) nounwind  {
	store i6 %x, i6* @i6_s
	ret void
}

define void @i7_ls(i7 %x) nounwind  {
	store i7 %x, i7* @i7_s
	ret void
}

define void @i8_ls(i8 %x) nounwind  {
	store i8 %x, i8* @i8_s
	ret void
}

define void @i9_ls(i9 %x) nounwind  {
	store i9 %x, i9* @i9_s
	ret void
}

define void @i10_ls(i10 %x) nounwind  {
	store i10 %x, i10* @i10_s
	ret void
}

define void @i11_ls(i11 %x) nounwind  {
	store i11 %x, i11* @i11_s
	ret void
}

define void @i12_ls(i12 %x) nounwind  {
	store i12 %x, i12* @i12_s
	ret void
}

define void @i13_ls(i13 %x) nounwind  {
	store i13 %x, i13* @i13_s
	ret void
}

define void @i14_ls(i14 %x) nounwind  {
	store i14 %x, i14* @i14_s
	ret void
}

define void @i15_ls(i15 %x) nounwind  {
	store i15 %x, i15* @i15_s
	ret void
}

define void @i16_ls(i16 %x) nounwind  {
	store i16 %x, i16* @i16_s
	ret void
}

define void @i17_ls(i17 %x) nounwind  {
	store i17 %x, i17* @i17_s
	ret void
}

define void @i18_ls(i18 %x) nounwind  {
	store i18 %x, i18* @i18_s
	ret void
}

define void @i19_ls(i19 %x) nounwind  {
	store i19 %x, i19* @i19_s
	ret void
}

define void @i20_ls(i20 %x) nounwind  {
	store i20 %x, i20* @i20_s
	ret void
}

define void @i21_ls(i21 %x) nounwind  {
	store i21 %x, i21* @i21_s
	ret void
}

define void @i22_ls(i22 %x) nounwind  {
	store i22 %x, i22* @i22_s
	ret void
}

define void @i23_ls(i23 %x) nounwind  {
	store i23 %x, i23* @i23_s
	ret void
}

define void @i24_ls(i24 %x) nounwind  {
	store i24 %x, i24* @i24_s
	ret void
}

define void @i25_ls(i25 %x) nounwind  {
	store i25 %x, i25* @i25_s
	ret void
}

define void @i26_ls(i26 %x) nounwind  {
	store i26 %x, i26* @i26_s
	ret void
}

define void @i27_ls(i27 %x) nounwind  {
	store i27 %x, i27* @i27_s
	ret void
}

define void @i28_ls(i28 %x) nounwind  {
	store i28 %x, i28* @i28_s
	ret void
}

define void @i29_ls(i29 %x) nounwind  {
	store i29 %x, i29* @i29_s
	ret void
}

define void @i30_ls(i30 %x) nounwind  {
	store i30 %x, i30* @i30_s
	ret void
}

define void @i31_ls(i31 %x) nounwind  {
	store i31 %x, i31* @i31_s
	ret void
}

define void @i32_ls(i32 %x) nounwind  {
	store i32 %x, i32* @i32_s
	ret void
}

define void @i33_ls(i33 %x) nounwind  {
	store i33 %x, i33* @i33_s
	ret void
}

define void @i34_ls(i34 %x) nounwind  {
	store i34 %x, i34* @i34_s
	ret void
}

define void @i35_ls(i35 %x) nounwind  {
	store i35 %x, i35* @i35_s
	ret void
}

define void @i36_ls(i36 %x) nounwind  {
	store i36 %x, i36* @i36_s
	ret void
}

define void @i37_ls(i37 %x) nounwind  {
	store i37 %x, i37* @i37_s
	ret void
}

define void @i38_ls(i38 %x) nounwind  {
	store i38 %x, i38* @i38_s
	ret void
}

define void @i39_ls(i39 %x) nounwind  {
	store i39 %x, i39* @i39_s
	ret void
}

define void @i40_ls(i40 %x) nounwind  {
	store i40 %x, i40* @i40_s
	ret void
}

define void @i41_ls(i41 %x) nounwind  {
	store i41 %x, i41* @i41_s
	ret void
}

define void @i42_ls(i42 %x) nounwind  {
	store i42 %x, i42* @i42_s
	ret void
}

define void @i43_ls(i43 %x) nounwind  {
	store i43 %x, i43* @i43_s
	ret void
}

define void @i44_ls(i44 %x) nounwind  {
	store i44 %x, i44* @i44_s
	ret void
}

define void @i45_ls(i45 %x) nounwind  {
	store i45 %x, i45* @i45_s
	ret void
}

define void @i46_ls(i46 %x) nounwind  {
	store i46 %x, i46* @i46_s
	ret void
}

define void @i47_ls(i47 %x) nounwind  {
	store i47 %x, i47* @i47_s
	ret void
}

define void @i48_ls(i48 %x) nounwind  {
	store i48 %x, i48* @i48_s
	ret void
}

define void @i49_ls(i49 %x) nounwind  {
	store i49 %x, i49* @i49_s
	ret void
}

define void @i50_ls(i50 %x) nounwind  {
	store i50 %x, i50* @i50_s
	ret void
}

define void @i51_ls(i51 %x) nounwind  {
	store i51 %x, i51* @i51_s
	ret void
}

define void @i52_ls(i52 %x) nounwind  {
	store i52 %x, i52* @i52_s
	ret void
}

define void @i53_ls(i53 %x) nounwind  {
	store i53 %x, i53* @i53_s
	ret void
}

define void @i54_ls(i54 %x) nounwind  {
	store i54 %x, i54* @i54_s
	ret void
}

define void @i55_ls(i55 %x) nounwind  {
	store i55 %x, i55* @i55_s
	ret void
}

define void @i56_ls(i56 %x) nounwind  {
	store i56 %x, i56* @i56_s
	ret void
}

define void @i57_ls(i57 %x) nounwind  {
	store i57 %x, i57* @i57_s
	ret void
}

define void @i58_ls(i58 %x) nounwind  {
	store i58 %x, i58* @i58_s
	ret void
}

define void @i59_ls(i59 %x) nounwind  {
	store i59 %x, i59* @i59_s
	ret void
}

define void @i60_ls(i60 %x) nounwind  {
	store i60 %x, i60* @i60_s
	ret void
}

define void @i61_ls(i61 %x) nounwind  {
	store i61 %x, i61* @i61_s
	ret void
}

define void @i62_ls(i62 %x) nounwind  {
	store i62 %x, i62* @i62_s
	ret void
}

define void @i63_ls(i63 %x) nounwind  {
	store i63 %x, i63* @i63_s
	ret void
}

define void @i64_ls(i64 %x) nounwind  {
	store i64 %x, i64* @i64_s
	ret void
}
