; RUN: llc -march=nvptx  < %s > %t
; RUN: llc -march=nvptx64  < %s > %t

@i1_s = external global i1		; <ptr> [#uses=1]
@i2_s = external global i2		; <ptr> [#uses=1]
@i3_s = external global i3		; <ptr> [#uses=1]
@i4_s = external global i4		; <ptr> [#uses=1]
@i5_s = external global i5		; <ptr> [#uses=1]
@i6_s = external global i6		; <ptr> [#uses=1]
@i7_s = external global i7		; <ptr> [#uses=1]
@i8_s = external global i8		; <ptr> [#uses=1]
@i9_s = external global i9		; <ptr> [#uses=1]
@i10_s = external global i10		; <ptr> [#uses=1]
@i11_s = external global i11		; <ptr> [#uses=1]
@i12_s = external global i12		; <ptr> [#uses=1]
@i13_s = external global i13		; <ptr> [#uses=1]
@i14_s = external global i14		; <ptr> [#uses=1]
@i15_s = external global i15		; <ptr> [#uses=1]
@i16_s = external global i16		; <ptr> [#uses=1]
@i17_s = external global i17		; <ptr> [#uses=1]
@i18_s = external global i18		; <ptr> [#uses=1]
@i19_s = external global i19		; <ptr> [#uses=1]
@i20_s = external global i20		; <ptr> [#uses=1]
@i21_s = external global i21		; <ptr> [#uses=1]
@i22_s = external global i22		; <ptr> [#uses=1]
@i23_s = external global i23		; <ptr> [#uses=1]
@i24_s = external global i24		; <ptr> [#uses=1]
@i25_s = external global i25		; <ptr> [#uses=1]
@i26_s = external global i26		; <ptr> [#uses=1]
@i27_s = external global i27		; <ptr> [#uses=1]
@i28_s = external global i28		; <ptr> [#uses=1]
@i29_s = external global i29		; <ptr> [#uses=1]
@i30_s = external global i30		; <ptr> [#uses=1]
@i31_s = external global i31		; <ptr> [#uses=1]
@i32_s = external global i32		; <ptr> [#uses=1]
@i33_s = external global i33		; <ptr> [#uses=1]
@i34_s = external global i34		; <ptr> [#uses=1]
@i35_s = external global i35		; <ptr> [#uses=1]
@i36_s = external global i36		; <ptr> [#uses=1]
@i37_s = external global i37		; <ptr> [#uses=1]
@i38_s = external global i38		; <ptr> [#uses=1]
@i39_s = external global i39		; <ptr> [#uses=1]
@i40_s = external global i40		; <ptr> [#uses=1]
@i41_s = external global i41		; <ptr> [#uses=1]
@i42_s = external global i42		; <ptr> [#uses=1]
@i43_s = external global i43		; <ptr> [#uses=1]
@i44_s = external global i44		; <ptr> [#uses=1]
@i45_s = external global i45		; <ptr> [#uses=1]
@i46_s = external global i46		; <ptr> [#uses=1]
@i47_s = external global i47		; <ptr> [#uses=1]
@i48_s = external global i48		; <ptr> [#uses=1]
@i49_s = external global i49		; <ptr> [#uses=1]
@i50_s = external global i50		; <ptr> [#uses=1]
@i51_s = external global i51		; <ptr> [#uses=1]
@i52_s = external global i52		; <ptr> [#uses=1]
@i53_s = external global i53		; <ptr> [#uses=1]
@i54_s = external global i54		; <ptr> [#uses=1]
@i55_s = external global i55		; <ptr> [#uses=1]
@i56_s = external global i56		; <ptr> [#uses=1]
@i57_s = external global i57		; <ptr> [#uses=1]
@i58_s = external global i58		; <ptr> [#uses=1]
@i59_s = external global i59		; <ptr> [#uses=1]
@i60_s = external global i60		; <ptr> [#uses=1]
@i61_s = external global i61		; <ptr> [#uses=1]
@i62_s = external global i62		; <ptr> [#uses=1]
@i63_s = external global i63		; <ptr> [#uses=1]
@i64_s = external global i64		; <ptr> [#uses=1]

define void @i1_ls(i1 %x) nounwind  {
	store i1 %x, ptr @i1_s
	ret void
}

define void @i2_ls(i2 %x) nounwind  {
	store i2 %x, ptr @i2_s
	ret void
}

define void @i3_ls(i3 %x) nounwind  {
	store i3 %x, ptr @i3_s
	ret void
}

define void @i4_ls(i4 %x) nounwind  {
	store i4 %x, ptr @i4_s
	ret void
}

define void @i5_ls(i5 %x) nounwind  {
	store i5 %x, ptr @i5_s
	ret void
}

define void @i6_ls(i6 %x) nounwind  {
	store i6 %x, ptr @i6_s
	ret void
}

define void @i7_ls(i7 %x) nounwind  {
	store i7 %x, ptr @i7_s
	ret void
}

define void @i8_ls(i8 %x) nounwind  {
	store i8 %x, ptr @i8_s
	ret void
}

define void @i9_ls(i9 %x) nounwind  {
	store i9 %x, ptr @i9_s
	ret void
}

define void @i10_ls(i10 %x) nounwind  {
	store i10 %x, ptr @i10_s
	ret void
}

define void @i11_ls(i11 %x) nounwind  {
	store i11 %x, ptr @i11_s
	ret void
}

define void @i12_ls(i12 %x) nounwind  {
	store i12 %x, ptr @i12_s
	ret void
}

define void @i13_ls(i13 %x) nounwind  {
	store i13 %x, ptr @i13_s
	ret void
}

define void @i14_ls(i14 %x) nounwind  {
	store i14 %x, ptr @i14_s
	ret void
}

define void @i15_ls(i15 %x) nounwind  {
	store i15 %x, ptr @i15_s
	ret void
}

define void @i16_ls(i16 %x) nounwind  {
	store i16 %x, ptr @i16_s
	ret void
}

define void @i17_ls(i17 %x) nounwind  {
	store i17 %x, ptr @i17_s
	ret void
}

define void @i18_ls(i18 %x) nounwind  {
	store i18 %x, ptr @i18_s
	ret void
}

define void @i19_ls(i19 %x) nounwind  {
	store i19 %x, ptr @i19_s
	ret void
}

define void @i20_ls(i20 %x) nounwind  {
	store i20 %x, ptr @i20_s
	ret void
}

define void @i21_ls(i21 %x) nounwind  {
	store i21 %x, ptr @i21_s
	ret void
}

define void @i22_ls(i22 %x) nounwind  {
	store i22 %x, ptr @i22_s
	ret void
}

define void @i23_ls(i23 %x) nounwind  {
	store i23 %x, ptr @i23_s
	ret void
}

define void @i24_ls(i24 %x) nounwind  {
	store i24 %x, ptr @i24_s
	ret void
}

define void @i25_ls(i25 %x) nounwind  {
	store i25 %x, ptr @i25_s
	ret void
}

define void @i26_ls(i26 %x) nounwind  {
	store i26 %x, ptr @i26_s
	ret void
}

define void @i27_ls(i27 %x) nounwind  {
	store i27 %x, ptr @i27_s
	ret void
}

define void @i28_ls(i28 %x) nounwind  {
	store i28 %x, ptr @i28_s
	ret void
}

define void @i29_ls(i29 %x) nounwind  {
	store i29 %x, ptr @i29_s
	ret void
}

define void @i30_ls(i30 %x) nounwind  {
	store i30 %x, ptr @i30_s
	ret void
}

define void @i31_ls(i31 %x) nounwind  {
	store i31 %x, ptr @i31_s
	ret void
}

define void @i32_ls(i32 %x) nounwind  {
	store i32 %x, ptr @i32_s
	ret void
}

define void @i33_ls(i33 %x) nounwind  {
	store i33 %x, ptr @i33_s
	ret void
}

define void @i34_ls(i34 %x) nounwind  {
	store i34 %x, ptr @i34_s
	ret void
}

define void @i35_ls(i35 %x) nounwind  {
	store i35 %x, ptr @i35_s
	ret void
}

define void @i36_ls(i36 %x) nounwind  {
	store i36 %x, ptr @i36_s
	ret void
}

define void @i37_ls(i37 %x) nounwind  {
	store i37 %x, ptr @i37_s
	ret void
}

define void @i38_ls(i38 %x) nounwind  {
	store i38 %x, ptr @i38_s
	ret void
}

define void @i39_ls(i39 %x) nounwind  {
	store i39 %x, ptr @i39_s
	ret void
}

define void @i40_ls(i40 %x) nounwind  {
	store i40 %x, ptr @i40_s
	ret void
}

define void @i41_ls(i41 %x) nounwind  {
	store i41 %x, ptr @i41_s
	ret void
}

define void @i42_ls(i42 %x) nounwind  {
	store i42 %x, ptr @i42_s
	ret void
}

define void @i43_ls(i43 %x) nounwind  {
	store i43 %x, ptr @i43_s
	ret void
}

define void @i44_ls(i44 %x) nounwind  {
	store i44 %x, ptr @i44_s
	ret void
}

define void @i45_ls(i45 %x) nounwind  {
	store i45 %x, ptr @i45_s
	ret void
}

define void @i46_ls(i46 %x) nounwind  {
	store i46 %x, ptr @i46_s
	ret void
}

define void @i47_ls(i47 %x) nounwind  {
	store i47 %x, ptr @i47_s
	ret void
}

define void @i48_ls(i48 %x) nounwind  {
	store i48 %x, ptr @i48_s
	ret void
}

define void @i49_ls(i49 %x) nounwind  {
	store i49 %x, ptr @i49_s
	ret void
}

define void @i50_ls(i50 %x) nounwind  {
	store i50 %x, ptr @i50_s
	ret void
}

define void @i51_ls(i51 %x) nounwind  {
	store i51 %x, ptr @i51_s
	ret void
}

define void @i52_ls(i52 %x) nounwind  {
	store i52 %x, ptr @i52_s
	ret void
}

define void @i53_ls(i53 %x) nounwind  {
	store i53 %x, ptr @i53_s
	ret void
}

define void @i54_ls(i54 %x) nounwind  {
	store i54 %x, ptr @i54_s
	ret void
}

define void @i55_ls(i55 %x) nounwind  {
	store i55 %x, ptr @i55_s
	ret void
}

define void @i56_ls(i56 %x) nounwind  {
	store i56 %x, ptr @i56_s
	ret void
}

define void @i57_ls(i57 %x) nounwind  {
	store i57 %x, ptr @i57_s
	ret void
}

define void @i58_ls(i58 %x) nounwind  {
	store i58 %x, ptr @i58_s
	ret void
}

define void @i59_ls(i59 %x) nounwind  {
	store i59 %x, ptr @i59_s
	ret void
}

define void @i60_ls(i60 %x) nounwind  {
	store i60 %x, ptr @i60_s
	ret void
}

define void @i61_ls(i61 %x) nounwind  {
	store i61 %x, ptr @i61_s
	ret void
}

define void @i62_ls(i62 %x) nounwind  {
	store i62 %x, ptr @i62_s
	ret void
}

define void @i63_ls(i63 %x) nounwind  {
	store i63 %x, ptr @i63_s
	ret void
}

define void @i64_ls(i64 %x) nounwind  {
	store i64 %x, ptr @i64_s
	ret void
}
