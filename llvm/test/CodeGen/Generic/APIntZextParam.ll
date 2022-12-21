; RUN: llc < %s > %t

; NVPTX does not support arbitrary integer types and has acceptable subset tested in NVPTX/APIntZextParam.ll
; UNSUPPORTED: target=nvptx{{.*}}

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
@i65_s = external global i65		; <ptr> [#uses=1]
@i66_s = external global i66		; <ptr> [#uses=1]
@i67_s = external global i67		; <ptr> [#uses=1]
@i68_s = external global i68		; <ptr> [#uses=1]
@i69_s = external global i69		; <ptr> [#uses=1]
@i70_s = external global i70		; <ptr> [#uses=1]
@i71_s = external global i71		; <ptr> [#uses=1]
@i72_s = external global i72		; <ptr> [#uses=1]
@i73_s = external global i73		; <ptr> [#uses=1]
@i74_s = external global i74		; <ptr> [#uses=1]
@i75_s = external global i75		; <ptr> [#uses=1]
@i76_s = external global i76		; <ptr> [#uses=1]
@i77_s = external global i77		; <ptr> [#uses=1]
@i78_s = external global i78		; <ptr> [#uses=1]
@i79_s = external global i79		; <ptr> [#uses=1]
@i80_s = external global i80		; <ptr> [#uses=1]
@i81_s = external global i81		; <ptr> [#uses=1]
@i82_s = external global i82		; <ptr> [#uses=1]
@i83_s = external global i83		; <ptr> [#uses=1]
@i84_s = external global i84		; <ptr> [#uses=1]
@i85_s = external global i85		; <ptr> [#uses=1]
@i86_s = external global i86		; <ptr> [#uses=1]
@i87_s = external global i87		; <ptr> [#uses=1]
@i88_s = external global i88		; <ptr> [#uses=1]
@i89_s = external global i89		; <ptr> [#uses=1]
@i90_s = external global i90		; <ptr> [#uses=1]
@i91_s = external global i91		; <ptr> [#uses=1]
@i92_s = external global i92		; <ptr> [#uses=1]
@i93_s = external global i93		; <ptr> [#uses=1]
@i94_s = external global i94		; <ptr> [#uses=1]
@i95_s = external global i95		; <ptr> [#uses=1]
@i96_s = external global i96		; <ptr> [#uses=1]
@i97_s = external global i97		; <ptr> [#uses=1]
@i98_s = external global i98		; <ptr> [#uses=1]
@i99_s = external global i99		; <ptr> [#uses=1]
@i100_s = external global i100		; <ptr> [#uses=1]
@i101_s = external global i101		; <ptr> [#uses=1]
@i102_s = external global i102		; <ptr> [#uses=1]
@i103_s = external global i103		; <ptr> [#uses=1]
@i104_s = external global i104		; <ptr> [#uses=1]
@i105_s = external global i105		; <ptr> [#uses=1]
@i106_s = external global i106		; <ptr> [#uses=1]
@i107_s = external global i107		; <ptr> [#uses=1]
@i108_s = external global i108		; <ptr> [#uses=1]
@i109_s = external global i109		; <ptr> [#uses=1]
@i110_s = external global i110		; <ptr> [#uses=1]
@i111_s = external global i111		; <ptr> [#uses=1]
@i112_s = external global i112		; <ptr> [#uses=1]
@i113_s = external global i113		; <ptr> [#uses=1]
@i114_s = external global i114		; <ptr> [#uses=1]
@i115_s = external global i115		; <ptr> [#uses=1]
@i116_s = external global i116		; <ptr> [#uses=1]
@i117_s = external global i117		; <ptr> [#uses=1]
@i118_s = external global i118		; <ptr> [#uses=1]
@i119_s = external global i119		; <ptr> [#uses=1]
@i120_s = external global i120		; <ptr> [#uses=1]
@i121_s = external global i121		; <ptr> [#uses=1]
@i122_s = external global i122		; <ptr> [#uses=1]
@i123_s = external global i123		; <ptr> [#uses=1]
@i124_s = external global i124		; <ptr> [#uses=1]
@i125_s = external global i125		; <ptr> [#uses=1]
@i126_s = external global i126		; <ptr> [#uses=1]
@i127_s = external global i127		; <ptr> [#uses=1]
@i128_s = external global i128		; <ptr> [#uses=1]
@i129_s = external global i129		; <ptr> [#uses=1]
@i130_s = external global i130		; <ptr> [#uses=1]
@i131_s = external global i131		; <ptr> [#uses=1]
@i132_s = external global i132		; <ptr> [#uses=1]
@i133_s = external global i133		; <ptr> [#uses=1]
@i134_s = external global i134		; <ptr> [#uses=1]
@i135_s = external global i135		; <ptr> [#uses=1]
@i136_s = external global i136		; <ptr> [#uses=1]
@i137_s = external global i137		; <ptr> [#uses=1]
@i138_s = external global i138		; <ptr> [#uses=1]
@i139_s = external global i139		; <ptr> [#uses=1]
@i140_s = external global i140		; <ptr> [#uses=1]
@i141_s = external global i141		; <ptr> [#uses=1]
@i142_s = external global i142		; <ptr> [#uses=1]
@i143_s = external global i143		; <ptr> [#uses=1]
@i144_s = external global i144		; <ptr> [#uses=1]
@i145_s = external global i145		; <ptr> [#uses=1]
@i146_s = external global i146		; <ptr> [#uses=1]
@i147_s = external global i147		; <ptr> [#uses=1]
@i148_s = external global i148		; <ptr> [#uses=1]
@i149_s = external global i149		; <ptr> [#uses=1]
@i150_s = external global i150		; <ptr> [#uses=1]
@i151_s = external global i151		; <ptr> [#uses=1]
@i152_s = external global i152		; <ptr> [#uses=1]
@i153_s = external global i153		; <ptr> [#uses=1]
@i154_s = external global i154		; <ptr> [#uses=1]
@i155_s = external global i155		; <ptr> [#uses=1]
@i156_s = external global i156		; <ptr> [#uses=1]
@i157_s = external global i157		; <ptr> [#uses=1]
@i158_s = external global i158		; <ptr> [#uses=1]
@i159_s = external global i159		; <ptr> [#uses=1]
@i160_s = external global i160		; <ptr> [#uses=1]
@i161_s = external global i161		; <ptr> [#uses=1]
@i162_s = external global i162		; <ptr> [#uses=1]
@i163_s = external global i163		; <ptr> [#uses=1]
@i164_s = external global i164		; <ptr> [#uses=1]
@i165_s = external global i165		; <ptr> [#uses=1]
@i166_s = external global i166		; <ptr> [#uses=1]
@i167_s = external global i167		; <ptr> [#uses=1]
@i168_s = external global i168		; <ptr> [#uses=1]
@i169_s = external global i169		; <ptr> [#uses=1]
@i170_s = external global i170		; <ptr> [#uses=1]
@i171_s = external global i171		; <ptr> [#uses=1]
@i172_s = external global i172		; <ptr> [#uses=1]
@i173_s = external global i173		; <ptr> [#uses=1]
@i174_s = external global i174		; <ptr> [#uses=1]
@i175_s = external global i175		; <ptr> [#uses=1]
@i176_s = external global i176		; <ptr> [#uses=1]
@i177_s = external global i177		; <ptr> [#uses=1]
@i178_s = external global i178		; <ptr> [#uses=1]
@i179_s = external global i179		; <ptr> [#uses=1]
@i180_s = external global i180		; <ptr> [#uses=1]
@i181_s = external global i181		; <ptr> [#uses=1]
@i182_s = external global i182		; <ptr> [#uses=1]
@i183_s = external global i183		; <ptr> [#uses=1]
@i184_s = external global i184		; <ptr> [#uses=1]
@i185_s = external global i185		; <ptr> [#uses=1]
@i186_s = external global i186		; <ptr> [#uses=1]
@i187_s = external global i187		; <ptr> [#uses=1]
@i188_s = external global i188		; <ptr> [#uses=1]
@i189_s = external global i189		; <ptr> [#uses=1]
@i190_s = external global i190		; <ptr> [#uses=1]
@i191_s = external global i191		; <ptr> [#uses=1]
@i192_s = external global i192		; <ptr> [#uses=1]
@i193_s = external global i193		; <ptr> [#uses=1]
@i194_s = external global i194		; <ptr> [#uses=1]
@i195_s = external global i195		; <ptr> [#uses=1]
@i196_s = external global i196		; <ptr> [#uses=1]
@i197_s = external global i197		; <ptr> [#uses=1]
@i198_s = external global i198		; <ptr> [#uses=1]
@i199_s = external global i199		; <ptr> [#uses=1]
@i200_s = external global i200		; <ptr> [#uses=1]
@i201_s = external global i201		; <ptr> [#uses=1]
@i202_s = external global i202		; <ptr> [#uses=1]
@i203_s = external global i203		; <ptr> [#uses=1]
@i204_s = external global i204		; <ptr> [#uses=1]
@i205_s = external global i205		; <ptr> [#uses=1]
@i206_s = external global i206		; <ptr> [#uses=1]
@i207_s = external global i207		; <ptr> [#uses=1]
@i208_s = external global i208		; <ptr> [#uses=1]
@i209_s = external global i209		; <ptr> [#uses=1]
@i210_s = external global i210		; <ptr> [#uses=1]
@i211_s = external global i211		; <ptr> [#uses=1]
@i212_s = external global i212		; <ptr> [#uses=1]
@i213_s = external global i213		; <ptr> [#uses=1]
@i214_s = external global i214		; <ptr> [#uses=1]
@i215_s = external global i215		; <ptr> [#uses=1]
@i216_s = external global i216		; <ptr> [#uses=1]
@i217_s = external global i217		; <ptr> [#uses=1]
@i218_s = external global i218		; <ptr> [#uses=1]
@i219_s = external global i219		; <ptr> [#uses=1]
@i220_s = external global i220		; <ptr> [#uses=1]
@i221_s = external global i221		; <ptr> [#uses=1]
@i222_s = external global i222		; <ptr> [#uses=1]
@i223_s = external global i223		; <ptr> [#uses=1]
@i224_s = external global i224		; <ptr> [#uses=1]
@i225_s = external global i225		; <ptr> [#uses=1]
@i226_s = external global i226		; <ptr> [#uses=1]
@i227_s = external global i227		; <ptr> [#uses=1]
@i228_s = external global i228		; <ptr> [#uses=1]
@i229_s = external global i229		; <ptr> [#uses=1]
@i230_s = external global i230		; <ptr> [#uses=1]
@i231_s = external global i231		; <ptr> [#uses=1]
@i232_s = external global i232		; <ptr> [#uses=1]
@i233_s = external global i233		; <ptr> [#uses=1]
@i234_s = external global i234		; <ptr> [#uses=1]
@i235_s = external global i235		; <ptr> [#uses=1]
@i236_s = external global i236		; <ptr> [#uses=1]
@i237_s = external global i237		; <ptr> [#uses=1]
@i238_s = external global i238		; <ptr> [#uses=1]
@i239_s = external global i239		; <ptr> [#uses=1]
@i240_s = external global i240		; <ptr> [#uses=1]
@i241_s = external global i241		; <ptr> [#uses=1]
@i242_s = external global i242		; <ptr> [#uses=1]
@i243_s = external global i243		; <ptr> [#uses=1]
@i244_s = external global i244		; <ptr> [#uses=1]
@i245_s = external global i245		; <ptr> [#uses=1]
@i246_s = external global i246		; <ptr> [#uses=1]
@i247_s = external global i247		; <ptr> [#uses=1]
@i248_s = external global i248		; <ptr> [#uses=1]
@i249_s = external global i249		; <ptr> [#uses=1]
@i250_s = external global i250		; <ptr> [#uses=1]
@i251_s = external global i251		; <ptr> [#uses=1]
@i252_s = external global i252		; <ptr> [#uses=1]
@i253_s = external global i253		; <ptr> [#uses=1]
@i254_s = external global i254		; <ptr> [#uses=1]
@i255_s = external global i255		; <ptr> [#uses=1]
@i256_s = external global i256		; <ptr> [#uses=1]

define void @i1_ls(i1 zeroext %x) nounwind  {
	store i1 %x, ptr @i1_s
	ret void
}

define void @i2_ls(i2 zeroext %x) nounwind  {
	store i2 %x, ptr @i2_s
	ret void
}

define void @i3_ls(i3 zeroext %x) nounwind  {
	store i3 %x, ptr @i3_s
	ret void
}

define void @i4_ls(i4 zeroext %x) nounwind  {
	store i4 %x, ptr @i4_s
	ret void
}

define void @i5_ls(i5 zeroext %x) nounwind  {
	store i5 %x, ptr @i5_s
	ret void
}

define void @i6_ls(i6 zeroext %x) nounwind  {
	store i6 %x, ptr @i6_s
	ret void
}

define void @i7_ls(i7 zeroext %x) nounwind  {
	store i7 %x, ptr @i7_s
	ret void
}

define void @i8_ls(i8 zeroext %x) nounwind  {
	store i8 %x, ptr @i8_s
	ret void
}

define void @i9_ls(i9 zeroext %x) nounwind  {
	store i9 %x, ptr @i9_s
	ret void
}

define void @i10_ls(i10 zeroext %x) nounwind  {
	store i10 %x, ptr @i10_s
	ret void
}

define void @i11_ls(i11 zeroext %x) nounwind  {
	store i11 %x, ptr @i11_s
	ret void
}

define void @i12_ls(i12 zeroext %x) nounwind  {
	store i12 %x, ptr @i12_s
	ret void
}

define void @i13_ls(i13 zeroext %x) nounwind  {
	store i13 %x, ptr @i13_s
	ret void
}

define void @i14_ls(i14 zeroext %x) nounwind  {
	store i14 %x, ptr @i14_s
	ret void
}

define void @i15_ls(i15 zeroext %x) nounwind  {
	store i15 %x, ptr @i15_s
	ret void
}

define void @i16_ls(i16 zeroext %x) nounwind  {
	store i16 %x, ptr @i16_s
	ret void
}

define void @i17_ls(i17 zeroext %x) nounwind  {
	store i17 %x, ptr @i17_s
	ret void
}

define void @i18_ls(i18 zeroext %x) nounwind  {
	store i18 %x, ptr @i18_s
	ret void
}

define void @i19_ls(i19 zeroext %x) nounwind  {
	store i19 %x, ptr @i19_s
	ret void
}

define void @i20_ls(i20 zeroext %x) nounwind  {
	store i20 %x, ptr @i20_s
	ret void
}

define void @i21_ls(i21 zeroext %x) nounwind  {
	store i21 %x, ptr @i21_s
	ret void
}

define void @i22_ls(i22 zeroext %x) nounwind  {
	store i22 %x, ptr @i22_s
	ret void
}

define void @i23_ls(i23 zeroext %x) nounwind  {
	store i23 %x, ptr @i23_s
	ret void
}

define void @i24_ls(i24 zeroext %x) nounwind  {
	store i24 %x, ptr @i24_s
	ret void
}

define void @i25_ls(i25 zeroext %x) nounwind  {
	store i25 %x, ptr @i25_s
	ret void
}

define void @i26_ls(i26 zeroext %x) nounwind  {
	store i26 %x, ptr @i26_s
	ret void
}

define void @i27_ls(i27 zeroext %x) nounwind  {
	store i27 %x, ptr @i27_s
	ret void
}

define void @i28_ls(i28 zeroext %x) nounwind  {
	store i28 %x, ptr @i28_s
	ret void
}

define void @i29_ls(i29 zeroext %x) nounwind  {
	store i29 %x, ptr @i29_s
	ret void
}

define void @i30_ls(i30 zeroext %x) nounwind  {
	store i30 %x, ptr @i30_s
	ret void
}

define void @i31_ls(i31 zeroext %x) nounwind  {
	store i31 %x, ptr @i31_s
	ret void
}

define void @i32_ls(i32 zeroext %x) nounwind  {
	store i32 %x, ptr @i32_s
	ret void
}

define void @i33_ls(i33 zeroext %x) nounwind  {
	store i33 %x, ptr @i33_s
	ret void
}

define void @i34_ls(i34 zeroext %x) nounwind  {
	store i34 %x, ptr @i34_s
	ret void
}

define void @i35_ls(i35 zeroext %x) nounwind  {
	store i35 %x, ptr @i35_s
	ret void
}

define void @i36_ls(i36 zeroext %x) nounwind  {
	store i36 %x, ptr @i36_s
	ret void
}

define void @i37_ls(i37 zeroext %x) nounwind  {
	store i37 %x, ptr @i37_s
	ret void
}

define void @i38_ls(i38 zeroext %x) nounwind  {
	store i38 %x, ptr @i38_s
	ret void
}

define void @i39_ls(i39 zeroext %x) nounwind  {
	store i39 %x, ptr @i39_s
	ret void
}

define void @i40_ls(i40 zeroext %x) nounwind  {
	store i40 %x, ptr @i40_s
	ret void
}

define void @i41_ls(i41 zeroext %x) nounwind  {
	store i41 %x, ptr @i41_s
	ret void
}

define void @i42_ls(i42 zeroext %x) nounwind  {
	store i42 %x, ptr @i42_s
	ret void
}

define void @i43_ls(i43 zeroext %x) nounwind  {
	store i43 %x, ptr @i43_s
	ret void
}

define void @i44_ls(i44 zeroext %x) nounwind  {
	store i44 %x, ptr @i44_s
	ret void
}

define void @i45_ls(i45 zeroext %x) nounwind  {
	store i45 %x, ptr @i45_s
	ret void
}

define void @i46_ls(i46 zeroext %x) nounwind  {
	store i46 %x, ptr @i46_s
	ret void
}

define void @i47_ls(i47 zeroext %x) nounwind  {
	store i47 %x, ptr @i47_s
	ret void
}

define void @i48_ls(i48 zeroext %x) nounwind  {
	store i48 %x, ptr @i48_s
	ret void
}

define void @i49_ls(i49 zeroext %x) nounwind  {
	store i49 %x, ptr @i49_s
	ret void
}

define void @i50_ls(i50 zeroext %x) nounwind  {
	store i50 %x, ptr @i50_s
	ret void
}

define void @i51_ls(i51 zeroext %x) nounwind  {
	store i51 %x, ptr @i51_s
	ret void
}

define void @i52_ls(i52 zeroext %x) nounwind  {
	store i52 %x, ptr @i52_s
	ret void
}

define void @i53_ls(i53 zeroext %x) nounwind  {
	store i53 %x, ptr @i53_s
	ret void
}

define void @i54_ls(i54 zeroext %x) nounwind  {
	store i54 %x, ptr @i54_s
	ret void
}

define void @i55_ls(i55 zeroext %x) nounwind  {
	store i55 %x, ptr @i55_s
	ret void
}

define void @i56_ls(i56 zeroext %x) nounwind  {
	store i56 %x, ptr @i56_s
	ret void
}

define void @i57_ls(i57 zeroext %x) nounwind  {
	store i57 %x, ptr @i57_s
	ret void
}

define void @i58_ls(i58 zeroext %x) nounwind  {
	store i58 %x, ptr @i58_s
	ret void
}

define void @i59_ls(i59 zeroext %x) nounwind  {
	store i59 %x, ptr @i59_s
	ret void
}

define void @i60_ls(i60 zeroext %x) nounwind  {
	store i60 %x, ptr @i60_s
	ret void
}

define void @i61_ls(i61 zeroext %x) nounwind  {
	store i61 %x, ptr @i61_s
	ret void
}

define void @i62_ls(i62 zeroext %x) nounwind  {
	store i62 %x, ptr @i62_s
	ret void
}

define void @i63_ls(i63 zeroext %x) nounwind  {
	store i63 %x, ptr @i63_s
	ret void
}

define void @i64_ls(i64 zeroext %x) nounwind  {
	store i64 %x, ptr @i64_s
	ret void
}

define void @i65_ls(i65 zeroext %x) nounwind  {
	store i65 %x, ptr @i65_s
	ret void
}

define void @i66_ls(i66 zeroext %x) nounwind  {
	store i66 %x, ptr @i66_s
	ret void
}

define void @i67_ls(i67 zeroext %x) nounwind  {
	store i67 %x, ptr @i67_s
	ret void
}

define void @i68_ls(i68 zeroext %x) nounwind  {
	store i68 %x, ptr @i68_s
	ret void
}

define void @i69_ls(i69 zeroext %x) nounwind  {
	store i69 %x, ptr @i69_s
	ret void
}

define void @i70_ls(i70 zeroext %x) nounwind  {
	store i70 %x, ptr @i70_s
	ret void
}

define void @i71_ls(i71 zeroext %x) nounwind  {
	store i71 %x, ptr @i71_s
	ret void
}

define void @i72_ls(i72 zeroext %x) nounwind  {
	store i72 %x, ptr @i72_s
	ret void
}

define void @i73_ls(i73 zeroext %x) nounwind  {
	store i73 %x, ptr @i73_s
	ret void
}

define void @i74_ls(i74 zeroext %x) nounwind  {
	store i74 %x, ptr @i74_s
	ret void
}

define void @i75_ls(i75 zeroext %x) nounwind  {
	store i75 %x, ptr @i75_s
	ret void
}

define void @i76_ls(i76 zeroext %x) nounwind  {
	store i76 %x, ptr @i76_s
	ret void
}

define void @i77_ls(i77 zeroext %x) nounwind  {
	store i77 %x, ptr @i77_s
	ret void
}

define void @i78_ls(i78 zeroext %x) nounwind  {
	store i78 %x, ptr @i78_s
	ret void
}

define void @i79_ls(i79 zeroext %x) nounwind  {
	store i79 %x, ptr @i79_s
	ret void
}

define void @i80_ls(i80 zeroext %x) nounwind  {
	store i80 %x, ptr @i80_s
	ret void
}

define void @i81_ls(i81 zeroext %x) nounwind  {
	store i81 %x, ptr @i81_s
	ret void
}

define void @i82_ls(i82 zeroext %x) nounwind  {
	store i82 %x, ptr @i82_s
	ret void
}

define void @i83_ls(i83 zeroext %x) nounwind  {
	store i83 %x, ptr @i83_s
	ret void
}

define void @i84_ls(i84 zeroext %x) nounwind  {
	store i84 %x, ptr @i84_s
	ret void
}

define void @i85_ls(i85 zeroext %x) nounwind  {
	store i85 %x, ptr @i85_s
	ret void
}

define void @i86_ls(i86 zeroext %x) nounwind  {
	store i86 %x, ptr @i86_s
	ret void
}

define void @i87_ls(i87 zeroext %x) nounwind  {
	store i87 %x, ptr @i87_s
	ret void
}

define void @i88_ls(i88 zeroext %x) nounwind  {
	store i88 %x, ptr @i88_s
	ret void
}

define void @i89_ls(i89 zeroext %x) nounwind  {
	store i89 %x, ptr @i89_s
	ret void
}

define void @i90_ls(i90 zeroext %x) nounwind  {
	store i90 %x, ptr @i90_s
	ret void
}

define void @i91_ls(i91 zeroext %x) nounwind  {
	store i91 %x, ptr @i91_s
	ret void
}

define void @i92_ls(i92 zeroext %x) nounwind  {
	store i92 %x, ptr @i92_s
	ret void
}

define void @i93_ls(i93 zeroext %x) nounwind  {
	store i93 %x, ptr @i93_s
	ret void
}

define void @i94_ls(i94 zeroext %x) nounwind  {
	store i94 %x, ptr @i94_s
	ret void
}

define void @i95_ls(i95 zeroext %x) nounwind  {
	store i95 %x, ptr @i95_s
	ret void
}

define void @i96_ls(i96 zeroext %x) nounwind  {
	store i96 %x, ptr @i96_s
	ret void
}

define void @i97_ls(i97 zeroext %x) nounwind  {
	store i97 %x, ptr @i97_s
	ret void
}

define void @i98_ls(i98 zeroext %x) nounwind  {
	store i98 %x, ptr @i98_s
	ret void
}

define void @i99_ls(i99 zeroext %x) nounwind  {
	store i99 %x, ptr @i99_s
	ret void
}

define void @i100_ls(i100 zeroext %x) nounwind  {
	store i100 %x, ptr @i100_s
	ret void
}

define void @i101_ls(i101 zeroext %x) nounwind  {
	store i101 %x, ptr @i101_s
	ret void
}

define void @i102_ls(i102 zeroext %x) nounwind  {
	store i102 %x, ptr @i102_s
	ret void
}

define void @i103_ls(i103 zeroext %x) nounwind  {
	store i103 %x, ptr @i103_s
	ret void
}

define void @i104_ls(i104 zeroext %x) nounwind  {
	store i104 %x, ptr @i104_s
	ret void
}

define void @i105_ls(i105 zeroext %x) nounwind  {
	store i105 %x, ptr @i105_s
	ret void
}

define void @i106_ls(i106 zeroext %x) nounwind  {
	store i106 %x, ptr @i106_s
	ret void
}

define void @i107_ls(i107 zeroext %x) nounwind  {
	store i107 %x, ptr @i107_s
	ret void
}

define void @i108_ls(i108 zeroext %x) nounwind  {
	store i108 %x, ptr @i108_s
	ret void
}

define void @i109_ls(i109 zeroext %x) nounwind  {
	store i109 %x, ptr @i109_s
	ret void
}

define void @i110_ls(i110 zeroext %x) nounwind  {
	store i110 %x, ptr @i110_s
	ret void
}

define void @i111_ls(i111 zeroext %x) nounwind  {
	store i111 %x, ptr @i111_s
	ret void
}

define void @i112_ls(i112 zeroext %x) nounwind  {
	store i112 %x, ptr @i112_s
	ret void
}

define void @i113_ls(i113 zeroext %x) nounwind  {
	store i113 %x, ptr @i113_s
	ret void
}

define void @i114_ls(i114 zeroext %x) nounwind  {
	store i114 %x, ptr @i114_s
	ret void
}

define void @i115_ls(i115 zeroext %x) nounwind  {
	store i115 %x, ptr @i115_s
	ret void
}

define void @i116_ls(i116 zeroext %x) nounwind  {
	store i116 %x, ptr @i116_s
	ret void
}

define void @i117_ls(i117 zeroext %x) nounwind  {
	store i117 %x, ptr @i117_s
	ret void
}

define void @i118_ls(i118 zeroext %x) nounwind  {
	store i118 %x, ptr @i118_s
	ret void
}

define void @i119_ls(i119 zeroext %x) nounwind  {
	store i119 %x, ptr @i119_s
	ret void
}

define void @i120_ls(i120 zeroext %x) nounwind  {
	store i120 %x, ptr @i120_s
	ret void
}

define void @i121_ls(i121 zeroext %x) nounwind  {
	store i121 %x, ptr @i121_s
	ret void
}

define void @i122_ls(i122 zeroext %x) nounwind  {
	store i122 %x, ptr @i122_s
	ret void
}

define void @i123_ls(i123 zeroext %x) nounwind  {
	store i123 %x, ptr @i123_s
	ret void
}

define void @i124_ls(i124 zeroext %x) nounwind  {
	store i124 %x, ptr @i124_s
	ret void
}

define void @i125_ls(i125 zeroext %x) nounwind  {
	store i125 %x, ptr @i125_s
	ret void
}

define void @i126_ls(i126 zeroext %x) nounwind  {
	store i126 %x, ptr @i126_s
	ret void
}

define void @i127_ls(i127 zeroext %x) nounwind  {
	store i127 %x, ptr @i127_s
	ret void
}

define void @i128_ls(i128 zeroext %x) nounwind  {
	store i128 %x, ptr @i128_s
	ret void
}

define void @i129_ls(i129 zeroext %x) nounwind  {
	store i129 %x, ptr @i129_s
	ret void
}

define void @i130_ls(i130 zeroext %x) nounwind  {
	store i130 %x, ptr @i130_s
	ret void
}

define void @i131_ls(i131 zeroext %x) nounwind  {
	store i131 %x, ptr @i131_s
	ret void
}

define void @i132_ls(i132 zeroext %x) nounwind  {
	store i132 %x, ptr @i132_s
	ret void
}

define void @i133_ls(i133 zeroext %x) nounwind  {
	store i133 %x, ptr @i133_s
	ret void
}

define void @i134_ls(i134 zeroext %x) nounwind  {
	store i134 %x, ptr @i134_s
	ret void
}

define void @i135_ls(i135 zeroext %x) nounwind  {
	store i135 %x, ptr @i135_s
	ret void
}

define void @i136_ls(i136 zeroext %x) nounwind  {
	store i136 %x, ptr @i136_s
	ret void
}

define void @i137_ls(i137 zeroext %x) nounwind  {
	store i137 %x, ptr @i137_s
	ret void
}

define void @i138_ls(i138 zeroext %x) nounwind  {
	store i138 %x, ptr @i138_s
	ret void
}

define void @i139_ls(i139 zeroext %x) nounwind  {
	store i139 %x, ptr @i139_s
	ret void
}

define void @i140_ls(i140 zeroext %x) nounwind  {
	store i140 %x, ptr @i140_s
	ret void
}

define void @i141_ls(i141 zeroext %x) nounwind  {
	store i141 %x, ptr @i141_s
	ret void
}

define void @i142_ls(i142 zeroext %x) nounwind  {
	store i142 %x, ptr @i142_s
	ret void
}

define void @i143_ls(i143 zeroext %x) nounwind  {
	store i143 %x, ptr @i143_s
	ret void
}

define void @i144_ls(i144 zeroext %x) nounwind  {
	store i144 %x, ptr @i144_s
	ret void
}

define void @i145_ls(i145 zeroext %x) nounwind  {
	store i145 %x, ptr @i145_s
	ret void
}

define void @i146_ls(i146 zeroext %x) nounwind  {
	store i146 %x, ptr @i146_s
	ret void
}

define void @i147_ls(i147 zeroext %x) nounwind  {
	store i147 %x, ptr @i147_s
	ret void
}

define void @i148_ls(i148 zeroext %x) nounwind  {
	store i148 %x, ptr @i148_s
	ret void
}

define void @i149_ls(i149 zeroext %x) nounwind  {
	store i149 %x, ptr @i149_s
	ret void
}

define void @i150_ls(i150 zeroext %x) nounwind  {
	store i150 %x, ptr @i150_s
	ret void
}

define void @i151_ls(i151 zeroext %x) nounwind  {
	store i151 %x, ptr @i151_s
	ret void
}

define void @i152_ls(i152 zeroext %x) nounwind  {
	store i152 %x, ptr @i152_s
	ret void
}

define void @i153_ls(i153 zeroext %x) nounwind  {
	store i153 %x, ptr @i153_s
	ret void
}

define void @i154_ls(i154 zeroext %x) nounwind  {
	store i154 %x, ptr @i154_s
	ret void
}

define void @i155_ls(i155 zeroext %x) nounwind  {
	store i155 %x, ptr @i155_s
	ret void
}

define void @i156_ls(i156 zeroext %x) nounwind  {
	store i156 %x, ptr @i156_s
	ret void
}

define void @i157_ls(i157 zeroext %x) nounwind  {
	store i157 %x, ptr @i157_s
	ret void
}

define void @i158_ls(i158 zeroext %x) nounwind  {
	store i158 %x, ptr @i158_s
	ret void
}

define void @i159_ls(i159 zeroext %x) nounwind  {
	store i159 %x, ptr @i159_s
	ret void
}

define void @i160_ls(i160 zeroext %x) nounwind  {
	store i160 %x, ptr @i160_s
	ret void
}

define void @i161_ls(i161 zeroext %x) nounwind  {
	store i161 %x, ptr @i161_s
	ret void
}

define void @i162_ls(i162 zeroext %x) nounwind  {
	store i162 %x, ptr @i162_s
	ret void
}

define void @i163_ls(i163 zeroext %x) nounwind  {
	store i163 %x, ptr @i163_s
	ret void
}

define void @i164_ls(i164 zeroext %x) nounwind  {
	store i164 %x, ptr @i164_s
	ret void
}

define void @i165_ls(i165 zeroext %x) nounwind  {
	store i165 %x, ptr @i165_s
	ret void
}

define void @i166_ls(i166 zeroext %x) nounwind  {
	store i166 %x, ptr @i166_s
	ret void
}

define void @i167_ls(i167 zeroext %x) nounwind  {
	store i167 %x, ptr @i167_s
	ret void
}

define void @i168_ls(i168 zeroext %x) nounwind  {
	store i168 %x, ptr @i168_s
	ret void
}

define void @i169_ls(i169 zeroext %x) nounwind  {
	store i169 %x, ptr @i169_s
	ret void
}

define void @i170_ls(i170 zeroext %x) nounwind  {
	store i170 %x, ptr @i170_s
	ret void
}

define void @i171_ls(i171 zeroext %x) nounwind  {
	store i171 %x, ptr @i171_s
	ret void
}

define void @i172_ls(i172 zeroext %x) nounwind  {
	store i172 %x, ptr @i172_s
	ret void
}

define void @i173_ls(i173 zeroext %x) nounwind  {
	store i173 %x, ptr @i173_s
	ret void
}

define void @i174_ls(i174 zeroext %x) nounwind  {
	store i174 %x, ptr @i174_s
	ret void
}

define void @i175_ls(i175 zeroext %x) nounwind  {
	store i175 %x, ptr @i175_s
	ret void
}

define void @i176_ls(i176 zeroext %x) nounwind  {
	store i176 %x, ptr @i176_s
	ret void
}

define void @i177_ls(i177 zeroext %x) nounwind  {
	store i177 %x, ptr @i177_s
	ret void
}

define void @i178_ls(i178 zeroext %x) nounwind  {
	store i178 %x, ptr @i178_s
	ret void
}

define void @i179_ls(i179 zeroext %x) nounwind  {
	store i179 %x, ptr @i179_s
	ret void
}

define void @i180_ls(i180 zeroext %x) nounwind  {
	store i180 %x, ptr @i180_s
	ret void
}

define void @i181_ls(i181 zeroext %x) nounwind  {
	store i181 %x, ptr @i181_s
	ret void
}

define void @i182_ls(i182 zeroext %x) nounwind  {
	store i182 %x, ptr @i182_s
	ret void
}

define void @i183_ls(i183 zeroext %x) nounwind  {
	store i183 %x, ptr @i183_s
	ret void
}

define void @i184_ls(i184 zeroext %x) nounwind  {
	store i184 %x, ptr @i184_s
	ret void
}

define void @i185_ls(i185 zeroext %x) nounwind  {
	store i185 %x, ptr @i185_s
	ret void
}

define void @i186_ls(i186 zeroext %x) nounwind  {
	store i186 %x, ptr @i186_s
	ret void
}

define void @i187_ls(i187 zeroext %x) nounwind  {
	store i187 %x, ptr @i187_s
	ret void
}

define void @i188_ls(i188 zeroext %x) nounwind  {
	store i188 %x, ptr @i188_s
	ret void
}

define void @i189_ls(i189 zeroext %x) nounwind  {
	store i189 %x, ptr @i189_s
	ret void
}

define void @i190_ls(i190 zeroext %x) nounwind  {
	store i190 %x, ptr @i190_s
	ret void
}

define void @i191_ls(i191 zeroext %x) nounwind  {
	store i191 %x, ptr @i191_s
	ret void
}

define void @i192_ls(i192 zeroext %x) nounwind  {
	store i192 %x, ptr @i192_s
	ret void
}

define void @i193_ls(i193 zeroext %x) nounwind  {
	store i193 %x, ptr @i193_s
	ret void
}

define void @i194_ls(i194 zeroext %x) nounwind  {
	store i194 %x, ptr @i194_s
	ret void
}

define void @i195_ls(i195 zeroext %x) nounwind  {
	store i195 %x, ptr @i195_s
	ret void
}

define void @i196_ls(i196 zeroext %x) nounwind  {
	store i196 %x, ptr @i196_s
	ret void
}

define void @i197_ls(i197 zeroext %x) nounwind  {
	store i197 %x, ptr @i197_s
	ret void
}

define void @i198_ls(i198 zeroext %x) nounwind  {
	store i198 %x, ptr @i198_s
	ret void
}

define void @i199_ls(i199 zeroext %x) nounwind  {
	store i199 %x, ptr @i199_s
	ret void
}

define void @i200_ls(i200 zeroext %x) nounwind  {
	store i200 %x, ptr @i200_s
	ret void
}

define void @i201_ls(i201 zeroext %x) nounwind  {
	store i201 %x, ptr @i201_s
	ret void
}

define void @i202_ls(i202 zeroext %x) nounwind  {
	store i202 %x, ptr @i202_s
	ret void
}

define void @i203_ls(i203 zeroext %x) nounwind  {
	store i203 %x, ptr @i203_s
	ret void
}

define void @i204_ls(i204 zeroext %x) nounwind  {
	store i204 %x, ptr @i204_s
	ret void
}

define void @i205_ls(i205 zeroext %x) nounwind  {
	store i205 %x, ptr @i205_s
	ret void
}

define void @i206_ls(i206 zeroext %x) nounwind  {
	store i206 %x, ptr @i206_s
	ret void
}

define void @i207_ls(i207 zeroext %x) nounwind  {
	store i207 %x, ptr @i207_s
	ret void
}

define void @i208_ls(i208 zeroext %x) nounwind  {
	store i208 %x, ptr @i208_s
	ret void
}

define void @i209_ls(i209 zeroext %x) nounwind  {
	store i209 %x, ptr @i209_s
	ret void
}

define void @i210_ls(i210 zeroext %x) nounwind  {
	store i210 %x, ptr @i210_s
	ret void
}

define void @i211_ls(i211 zeroext %x) nounwind  {
	store i211 %x, ptr @i211_s
	ret void
}

define void @i212_ls(i212 zeroext %x) nounwind  {
	store i212 %x, ptr @i212_s
	ret void
}

define void @i213_ls(i213 zeroext %x) nounwind  {
	store i213 %x, ptr @i213_s
	ret void
}

define void @i214_ls(i214 zeroext %x) nounwind  {
	store i214 %x, ptr @i214_s
	ret void
}

define void @i215_ls(i215 zeroext %x) nounwind  {
	store i215 %x, ptr @i215_s
	ret void
}

define void @i216_ls(i216 zeroext %x) nounwind  {
	store i216 %x, ptr @i216_s
	ret void
}

define void @i217_ls(i217 zeroext %x) nounwind  {
	store i217 %x, ptr @i217_s
	ret void
}

define void @i218_ls(i218 zeroext %x) nounwind  {
	store i218 %x, ptr @i218_s
	ret void
}

define void @i219_ls(i219 zeroext %x) nounwind  {
	store i219 %x, ptr @i219_s
	ret void
}

define void @i220_ls(i220 zeroext %x) nounwind  {
	store i220 %x, ptr @i220_s
	ret void
}

define void @i221_ls(i221 zeroext %x) nounwind  {
	store i221 %x, ptr @i221_s
	ret void
}

define void @i222_ls(i222 zeroext %x) nounwind  {
	store i222 %x, ptr @i222_s
	ret void
}

define void @i223_ls(i223 zeroext %x) nounwind  {
	store i223 %x, ptr @i223_s
	ret void
}

define void @i224_ls(i224 zeroext %x) nounwind  {
	store i224 %x, ptr @i224_s
	ret void
}

define void @i225_ls(i225 zeroext %x) nounwind  {
	store i225 %x, ptr @i225_s
	ret void
}

define void @i226_ls(i226 zeroext %x) nounwind  {
	store i226 %x, ptr @i226_s
	ret void
}

define void @i227_ls(i227 zeroext %x) nounwind  {
	store i227 %x, ptr @i227_s
	ret void
}

define void @i228_ls(i228 zeroext %x) nounwind  {
	store i228 %x, ptr @i228_s
	ret void
}

define void @i229_ls(i229 zeroext %x) nounwind  {
	store i229 %x, ptr @i229_s
	ret void
}

define void @i230_ls(i230 zeroext %x) nounwind  {
	store i230 %x, ptr @i230_s
	ret void
}

define void @i231_ls(i231 zeroext %x) nounwind  {
	store i231 %x, ptr @i231_s
	ret void
}

define void @i232_ls(i232 zeroext %x) nounwind  {
	store i232 %x, ptr @i232_s
	ret void
}

define void @i233_ls(i233 zeroext %x) nounwind  {
	store i233 %x, ptr @i233_s
	ret void
}

define void @i234_ls(i234 zeroext %x) nounwind  {
	store i234 %x, ptr @i234_s
	ret void
}

define void @i235_ls(i235 zeroext %x) nounwind  {
	store i235 %x, ptr @i235_s
	ret void
}

define void @i236_ls(i236 zeroext %x) nounwind  {
	store i236 %x, ptr @i236_s
	ret void
}

define void @i237_ls(i237 zeroext %x) nounwind  {
	store i237 %x, ptr @i237_s
	ret void
}

define void @i238_ls(i238 zeroext %x) nounwind  {
	store i238 %x, ptr @i238_s
	ret void
}

define void @i239_ls(i239 zeroext %x) nounwind  {
	store i239 %x, ptr @i239_s
	ret void
}

define void @i240_ls(i240 zeroext %x) nounwind  {
	store i240 %x, ptr @i240_s
	ret void
}

define void @i241_ls(i241 zeroext %x) nounwind  {
	store i241 %x, ptr @i241_s
	ret void
}

define void @i242_ls(i242 zeroext %x) nounwind  {
	store i242 %x, ptr @i242_s
	ret void
}

define void @i243_ls(i243 zeroext %x) nounwind  {
	store i243 %x, ptr @i243_s
	ret void
}

define void @i244_ls(i244 zeroext %x) nounwind  {
	store i244 %x, ptr @i244_s
	ret void
}

define void @i245_ls(i245 zeroext %x) nounwind  {
	store i245 %x, ptr @i245_s
	ret void
}

define void @i246_ls(i246 zeroext %x) nounwind  {
	store i246 %x, ptr @i246_s
	ret void
}

define void @i247_ls(i247 zeroext %x) nounwind  {
	store i247 %x, ptr @i247_s
	ret void
}

define void @i248_ls(i248 zeroext %x) nounwind  {
	store i248 %x, ptr @i248_s
	ret void
}

define void @i249_ls(i249 zeroext %x) nounwind  {
	store i249 %x, ptr @i249_s
	ret void
}

define void @i250_ls(i250 zeroext %x) nounwind  {
	store i250 %x, ptr @i250_s
	ret void
}

define void @i251_ls(i251 zeroext %x) nounwind  {
	store i251 %x, ptr @i251_s
	ret void
}

define void @i252_ls(i252 zeroext %x) nounwind  {
	store i252 %x, ptr @i252_s
	ret void
}

define void @i253_ls(i253 zeroext %x) nounwind  {
	store i253 %x, ptr @i253_s
	ret void
}

define void @i254_ls(i254 zeroext %x) nounwind  {
	store i254 %x, ptr @i254_s
	ret void
}

define void @i255_ls(i255 zeroext %x) nounwind  {
	store i255 %x, ptr @i255_s
	ret void
}

define void @i256_ls(i256 zeroext %x) nounwind  {
	store i256 %x, ptr @i256_s
	ret void
}
