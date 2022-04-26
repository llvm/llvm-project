; RUN: llc < %s > %t

; NVPTX does not support arbitrary integer types and has acceptable subset tested in NVPTX/APIntParam.ll
; UNSUPPORTED: nvptx

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
@i65_s = external global i65		; <i65*> [#uses=1]
@i66_s = external global i66		; <i66*> [#uses=1]
@i67_s = external global i67		; <i67*> [#uses=1]
@i68_s = external global i68		; <i68*> [#uses=1]
@i69_s = external global i69		; <i69*> [#uses=1]
@i70_s = external global i70		; <i70*> [#uses=1]
@i71_s = external global i71		; <i71*> [#uses=1]
@i72_s = external global i72		; <i72*> [#uses=1]
@i73_s = external global i73		; <i73*> [#uses=1]
@i74_s = external global i74		; <i74*> [#uses=1]
@i75_s = external global i75		; <i75*> [#uses=1]
@i76_s = external global i76		; <i76*> [#uses=1]
@i77_s = external global i77		; <i77*> [#uses=1]
@i78_s = external global i78		; <i78*> [#uses=1]
@i79_s = external global i79		; <i79*> [#uses=1]
@i80_s = external global i80		; <i80*> [#uses=1]
@i81_s = external global i81		; <i81*> [#uses=1]
@i82_s = external global i82		; <i82*> [#uses=1]
@i83_s = external global i83		; <i83*> [#uses=1]
@i84_s = external global i84		; <i84*> [#uses=1]
@i85_s = external global i85		; <i85*> [#uses=1]
@i86_s = external global i86		; <i86*> [#uses=1]
@i87_s = external global i87		; <i87*> [#uses=1]
@i88_s = external global i88		; <i88*> [#uses=1]
@i89_s = external global i89		; <i89*> [#uses=1]
@i90_s = external global i90		; <i90*> [#uses=1]
@i91_s = external global i91		; <i91*> [#uses=1]
@i92_s = external global i92		; <i92*> [#uses=1]
@i93_s = external global i93		; <i93*> [#uses=1]
@i94_s = external global i94		; <i94*> [#uses=1]
@i95_s = external global i95		; <i95*> [#uses=1]
@i96_s = external global i96		; <i96*> [#uses=1]
@i97_s = external global i97		; <i97*> [#uses=1]
@i98_s = external global i98		; <i98*> [#uses=1]
@i99_s = external global i99		; <i99*> [#uses=1]
@i100_s = external global i100		; <i100*> [#uses=1]
@i101_s = external global i101		; <i101*> [#uses=1]
@i102_s = external global i102		; <i102*> [#uses=1]
@i103_s = external global i103		; <i103*> [#uses=1]
@i104_s = external global i104		; <i104*> [#uses=1]
@i105_s = external global i105		; <i105*> [#uses=1]
@i106_s = external global i106		; <i106*> [#uses=1]
@i107_s = external global i107		; <i107*> [#uses=1]
@i108_s = external global i108		; <i108*> [#uses=1]
@i109_s = external global i109		; <i109*> [#uses=1]
@i110_s = external global i110		; <i110*> [#uses=1]
@i111_s = external global i111		; <i111*> [#uses=1]
@i112_s = external global i112		; <i112*> [#uses=1]
@i113_s = external global i113		; <i113*> [#uses=1]
@i114_s = external global i114		; <i114*> [#uses=1]
@i115_s = external global i115		; <i115*> [#uses=1]
@i116_s = external global i116		; <i116*> [#uses=1]
@i117_s = external global i117		; <i117*> [#uses=1]
@i118_s = external global i118		; <i118*> [#uses=1]
@i119_s = external global i119		; <i119*> [#uses=1]
@i120_s = external global i120		; <i120*> [#uses=1]
@i121_s = external global i121		; <i121*> [#uses=1]
@i122_s = external global i122		; <i122*> [#uses=1]
@i123_s = external global i123		; <i123*> [#uses=1]
@i124_s = external global i124		; <i124*> [#uses=1]
@i125_s = external global i125		; <i125*> [#uses=1]
@i126_s = external global i126		; <i126*> [#uses=1]
@i127_s = external global i127		; <i127*> [#uses=1]
@i128_s = external global i128		; <i128*> [#uses=1]
@i129_s = external global i129		; <i129*> [#uses=1]
@i130_s = external global i130		; <i130*> [#uses=1]
@i131_s = external global i131		; <i131*> [#uses=1]
@i132_s = external global i132		; <i132*> [#uses=1]
@i133_s = external global i133		; <i133*> [#uses=1]
@i134_s = external global i134		; <i134*> [#uses=1]
@i135_s = external global i135		; <i135*> [#uses=1]
@i136_s = external global i136		; <i136*> [#uses=1]
@i137_s = external global i137		; <i137*> [#uses=1]
@i138_s = external global i138		; <i138*> [#uses=1]
@i139_s = external global i139		; <i139*> [#uses=1]
@i140_s = external global i140		; <i140*> [#uses=1]
@i141_s = external global i141		; <i141*> [#uses=1]
@i142_s = external global i142		; <i142*> [#uses=1]
@i143_s = external global i143		; <i143*> [#uses=1]
@i144_s = external global i144		; <i144*> [#uses=1]
@i145_s = external global i145		; <i145*> [#uses=1]
@i146_s = external global i146		; <i146*> [#uses=1]
@i147_s = external global i147		; <i147*> [#uses=1]
@i148_s = external global i148		; <i148*> [#uses=1]
@i149_s = external global i149		; <i149*> [#uses=1]
@i150_s = external global i150		; <i150*> [#uses=1]
@i151_s = external global i151		; <i151*> [#uses=1]
@i152_s = external global i152		; <i152*> [#uses=1]
@i153_s = external global i153		; <i153*> [#uses=1]
@i154_s = external global i154		; <i154*> [#uses=1]
@i155_s = external global i155		; <i155*> [#uses=1]
@i156_s = external global i156		; <i156*> [#uses=1]
@i157_s = external global i157		; <i157*> [#uses=1]
@i158_s = external global i158		; <i158*> [#uses=1]
@i159_s = external global i159		; <i159*> [#uses=1]
@i160_s = external global i160		; <i160*> [#uses=1]
@i161_s = external global i161		; <i161*> [#uses=1]
@i162_s = external global i162		; <i162*> [#uses=1]
@i163_s = external global i163		; <i163*> [#uses=1]
@i164_s = external global i164		; <i164*> [#uses=1]
@i165_s = external global i165		; <i165*> [#uses=1]
@i166_s = external global i166		; <i166*> [#uses=1]
@i167_s = external global i167		; <i167*> [#uses=1]
@i168_s = external global i168		; <i168*> [#uses=1]
@i169_s = external global i169		; <i169*> [#uses=1]
@i170_s = external global i170		; <i170*> [#uses=1]
@i171_s = external global i171		; <i171*> [#uses=1]
@i172_s = external global i172		; <i172*> [#uses=1]
@i173_s = external global i173		; <i173*> [#uses=1]
@i174_s = external global i174		; <i174*> [#uses=1]
@i175_s = external global i175		; <i175*> [#uses=1]
@i176_s = external global i176		; <i176*> [#uses=1]
@i177_s = external global i177		; <i177*> [#uses=1]
@i178_s = external global i178		; <i178*> [#uses=1]
@i179_s = external global i179		; <i179*> [#uses=1]
@i180_s = external global i180		; <i180*> [#uses=1]
@i181_s = external global i181		; <i181*> [#uses=1]
@i182_s = external global i182		; <i182*> [#uses=1]
@i183_s = external global i183		; <i183*> [#uses=1]
@i184_s = external global i184		; <i184*> [#uses=1]
@i185_s = external global i185		; <i185*> [#uses=1]
@i186_s = external global i186		; <i186*> [#uses=1]
@i187_s = external global i187		; <i187*> [#uses=1]
@i188_s = external global i188		; <i188*> [#uses=1]
@i189_s = external global i189		; <i189*> [#uses=1]
@i190_s = external global i190		; <i190*> [#uses=1]
@i191_s = external global i191		; <i191*> [#uses=1]
@i192_s = external global i192		; <i192*> [#uses=1]
@i193_s = external global i193		; <i193*> [#uses=1]
@i194_s = external global i194		; <i194*> [#uses=1]
@i195_s = external global i195		; <i195*> [#uses=1]
@i196_s = external global i196		; <i196*> [#uses=1]
@i197_s = external global i197		; <i197*> [#uses=1]
@i198_s = external global i198		; <i198*> [#uses=1]
@i199_s = external global i199		; <i199*> [#uses=1]
@i200_s = external global i200		; <i200*> [#uses=1]
@i201_s = external global i201		; <i201*> [#uses=1]
@i202_s = external global i202		; <i202*> [#uses=1]
@i203_s = external global i203		; <i203*> [#uses=1]
@i204_s = external global i204		; <i204*> [#uses=1]
@i205_s = external global i205		; <i205*> [#uses=1]
@i206_s = external global i206		; <i206*> [#uses=1]
@i207_s = external global i207		; <i207*> [#uses=1]
@i208_s = external global i208		; <i208*> [#uses=1]
@i209_s = external global i209		; <i209*> [#uses=1]
@i210_s = external global i210		; <i210*> [#uses=1]
@i211_s = external global i211		; <i211*> [#uses=1]
@i212_s = external global i212		; <i212*> [#uses=1]
@i213_s = external global i213		; <i213*> [#uses=1]
@i214_s = external global i214		; <i214*> [#uses=1]
@i215_s = external global i215		; <i215*> [#uses=1]
@i216_s = external global i216		; <i216*> [#uses=1]
@i217_s = external global i217		; <i217*> [#uses=1]
@i218_s = external global i218		; <i218*> [#uses=1]
@i219_s = external global i219		; <i219*> [#uses=1]
@i220_s = external global i220		; <i220*> [#uses=1]
@i221_s = external global i221		; <i221*> [#uses=1]
@i222_s = external global i222		; <i222*> [#uses=1]
@i223_s = external global i223		; <i223*> [#uses=1]
@i224_s = external global i224		; <i224*> [#uses=1]
@i225_s = external global i225		; <i225*> [#uses=1]
@i226_s = external global i226		; <i226*> [#uses=1]
@i227_s = external global i227		; <i227*> [#uses=1]
@i228_s = external global i228		; <i228*> [#uses=1]
@i229_s = external global i229		; <i229*> [#uses=1]
@i230_s = external global i230		; <i230*> [#uses=1]
@i231_s = external global i231		; <i231*> [#uses=1]
@i232_s = external global i232		; <i232*> [#uses=1]
@i233_s = external global i233		; <i233*> [#uses=1]
@i234_s = external global i234		; <i234*> [#uses=1]
@i235_s = external global i235		; <i235*> [#uses=1]
@i236_s = external global i236		; <i236*> [#uses=1]
@i237_s = external global i237		; <i237*> [#uses=1]
@i238_s = external global i238		; <i238*> [#uses=1]
@i239_s = external global i239		; <i239*> [#uses=1]
@i240_s = external global i240		; <i240*> [#uses=1]
@i241_s = external global i241		; <i241*> [#uses=1]
@i242_s = external global i242		; <i242*> [#uses=1]
@i243_s = external global i243		; <i243*> [#uses=1]
@i244_s = external global i244		; <i244*> [#uses=1]
@i245_s = external global i245		; <i245*> [#uses=1]
@i246_s = external global i246		; <i246*> [#uses=1]
@i247_s = external global i247		; <i247*> [#uses=1]
@i248_s = external global i248		; <i248*> [#uses=1]
@i249_s = external global i249		; <i249*> [#uses=1]
@i250_s = external global i250		; <i250*> [#uses=1]
@i251_s = external global i251		; <i251*> [#uses=1]
@i252_s = external global i252		; <i252*> [#uses=1]
@i253_s = external global i253		; <i253*> [#uses=1]
@i254_s = external global i254		; <i254*> [#uses=1]
@i255_s = external global i255		; <i255*> [#uses=1]
@i256_s = external global i256		; <i256*> [#uses=1]

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

define void @i65_ls(i65 %x) nounwind  {
	store i65 %x, i65* @i65_s
	ret void
}

define void @i66_ls(i66 %x) nounwind  {
	store i66 %x, i66* @i66_s
	ret void
}

define void @i67_ls(i67 %x) nounwind  {
	store i67 %x, i67* @i67_s
	ret void
}

define void @i68_ls(i68 %x) nounwind  {
	store i68 %x, i68* @i68_s
	ret void
}

define void @i69_ls(i69 %x) nounwind  {
	store i69 %x, i69* @i69_s
	ret void
}

define void @i70_ls(i70 %x) nounwind  {
	store i70 %x, i70* @i70_s
	ret void
}

define void @i71_ls(i71 %x) nounwind  {
	store i71 %x, i71* @i71_s
	ret void
}

define void @i72_ls(i72 %x) nounwind  {
	store i72 %x, i72* @i72_s
	ret void
}

define void @i73_ls(i73 %x) nounwind  {
	store i73 %x, i73* @i73_s
	ret void
}

define void @i74_ls(i74 %x) nounwind  {
	store i74 %x, i74* @i74_s
	ret void
}

define void @i75_ls(i75 %x) nounwind  {
	store i75 %x, i75* @i75_s
	ret void
}

define void @i76_ls(i76 %x) nounwind  {
	store i76 %x, i76* @i76_s
	ret void
}

define void @i77_ls(i77 %x) nounwind  {
	store i77 %x, i77* @i77_s
	ret void
}

define void @i78_ls(i78 %x) nounwind  {
	store i78 %x, i78* @i78_s
	ret void
}

define void @i79_ls(i79 %x) nounwind  {
	store i79 %x, i79* @i79_s
	ret void
}

define void @i80_ls(i80 %x) nounwind  {
	store i80 %x, i80* @i80_s
	ret void
}

define void @i81_ls(i81 %x) nounwind  {
	store i81 %x, i81* @i81_s
	ret void
}

define void @i82_ls(i82 %x) nounwind  {
	store i82 %x, i82* @i82_s
	ret void
}

define void @i83_ls(i83 %x) nounwind  {
	store i83 %x, i83* @i83_s
	ret void
}

define void @i84_ls(i84 %x) nounwind  {
	store i84 %x, i84* @i84_s
	ret void
}

define void @i85_ls(i85 %x) nounwind  {
	store i85 %x, i85* @i85_s
	ret void
}

define void @i86_ls(i86 %x) nounwind  {
	store i86 %x, i86* @i86_s
	ret void
}

define void @i87_ls(i87 %x) nounwind  {
	store i87 %x, i87* @i87_s
	ret void
}

define void @i88_ls(i88 %x) nounwind  {
	store i88 %x, i88* @i88_s
	ret void
}

define void @i89_ls(i89 %x) nounwind  {
	store i89 %x, i89* @i89_s
	ret void
}

define void @i90_ls(i90 %x) nounwind  {
	store i90 %x, i90* @i90_s
	ret void
}

define void @i91_ls(i91 %x) nounwind  {
	store i91 %x, i91* @i91_s
	ret void
}

define void @i92_ls(i92 %x) nounwind  {
	store i92 %x, i92* @i92_s
	ret void
}

define void @i93_ls(i93 %x) nounwind  {
	store i93 %x, i93* @i93_s
	ret void
}

define void @i94_ls(i94 %x) nounwind  {
	store i94 %x, i94* @i94_s
	ret void
}

define void @i95_ls(i95 %x) nounwind  {
	store i95 %x, i95* @i95_s
	ret void
}

define void @i96_ls(i96 %x) nounwind  {
	store i96 %x, i96* @i96_s
	ret void
}

define void @i97_ls(i97 %x) nounwind  {
	store i97 %x, i97* @i97_s
	ret void
}

define void @i98_ls(i98 %x) nounwind  {
	store i98 %x, i98* @i98_s
	ret void
}

define void @i99_ls(i99 %x) nounwind  {
	store i99 %x, i99* @i99_s
	ret void
}

define void @i100_ls(i100 %x) nounwind  {
	store i100 %x, i100* @i100_s
	ret void
}

define void @i101_ls(i101 %x) nounwind  {
	store i101 %x, i101* @i101_s
	ret void
}

define void @i102_ls(i102 %x) nounwind  {
	store i102 %x, i102* @i102_s
	ret void
}

define void @i103_ls(i103 %x) nounwind  {
	store i103 %x, i103* @i103_s
	ret void
}

define void @i104_ls(i104 %x) nounwind  {
	store i104 %x, i104* @i104_s
	ret void
}

define void @i105_ls(i105 %x) nounwind  {
	store i105 %x, i105* @i105_s
	ret void
}

define void @i106_ls(i106 %x) nounwind  {
	store i106 %x, i106* @i106_s
	ret void
}

define void @i107_ls(i107 %x) nounwind  {
	store i107 %x, i107* @i107_s
	ret void
}

define void @i108_ls(i108 %x) nounwind  {
	store i108 %x, i108* @i108_s
	ret void
}

define void @i109_ls(i109 %x) nounwind  {
	store i109 %x, i109* @i109_s
	ret void
}

define void @i110_ls(i110 %x) nounwind  {
	store i110 %x, i110* @i110_s
	ret void
}

define void @i111_ls(i111 %x) nounwind  {
	store i111 %x, i111* @i111_s
	ret void
}

define void @i112_ls(i112 %x) nounwind  {
	store i112 %x, i112* @i112_s
	ret void
}

define void @i113_ls(i113 %x) nounwind  {
	store i113 %x, i113* @i113_s
	ret void
}

define void @i114_ls(i114 %x) nounwind  {
	store i114 %x, i114* @i114_s
	ret void
}

define void @i115_ls(i115 %x) nounwind  {
	store i115 %x, i115* @i115_s
	ret void
}

define void @i116_ls(i116 %x) nounwind  {
	store i116 %x, i116* @i116_s
	ret void
}

define void @i117_ls(i117 %x) nounwind  {
	store i117 %x, i117* @i117_s
	ret void
}

define void @i118_ls(i118 %x) nounwind  {
	store i118 %x, i118* @i118_s
	ret void
}

define void @i119_ls(i119 %x) nounwind  {
	store i119 %x, i119* @i119_s
	ret void
}

define void @i120_ls(i120 %x) nounwind  {
	store i120 %x, i120* @i120_s
	ret void
}

define void @i121_ls(i121 %x) nounwind  {
	store i121 %x, i121* @i121_s
	ret void
}

define void @i122_ls(i122 %x) nounwind  {
	store i122 %x, i122* @i122_s
	ret void
}

define void @i123_ls(i123 %x) nounwind  {
	store i123 %x, i123* @i123_s
	ret void
}

define void @i124_ls(i124 %x) nounwind  {
	store i124 %x, i124* @i124_s
	ret void
}

define void @i125_ls(i125 %x) nounwind  {
	store i125 %x, i125* @i125_s
	ret void
}

define void @i126_ls(i126 %x) nounwind  {
	store i126 %x, i126* @i126_s
	ret void
}

define void @i127_ls(i127 %x) nounwind  {
	store i127 %x, i127* @i127_s
	ret void
}

define void @i128_ls(i128 %x) nounwind  {
	store i128 %x, i128* @i128_s
	ret void
}

define void @i129_ls(i129 %x) nounwind  {
	store i129 %x, i129* @i129_s
	ret void
}

define void @i130_ls(i130 %x) nounwind  {
	store i130 %x, i130* @i130_s
	ret void
}

define void @i131_ls(i131 %x) nounwind  {
	store i131 %x, i131* @i131_s
	ret void
}

define void @i132_ls(i132 %x) nounwind  {
	store i132 %x, i132* @i132_s
	ret void
}

define void @i133_ls(i133 %x) nounwind  {
	store i133 %x, i133* @i133_s
	ret void
}

define void @i134_ls(i134 %x) nounwind  {
	store i134 %x, i134* @i134_s
	ret void
}

define void @i135_ls(i135 %x) nounwind  {
	store i135 %x, i135* @i135_s
	ret void
}

define void @i136_ls(i136 %x) nounwind  {
	store i136 %x, i136* @i136_s
	ret void
}

define void @i137_ls(i137 %x) nounwind  {
	store i137 %x, i137* @i137_s
	ret void
}

define void @i138_ls(i138 %x) nounwind  {
	store i138 %x, i138* @i138_s
	ret void
}

define void @i139_ls(i139 %x) nounwind  {
	store i139 %x, i139* @i139_s
	ret void
}

define void @i140_ls(i140 %x) nounwind  {
	store i140 %x, i140* @i140_s
	ret void
}

define void @i141_ls(i141 %x) nounwind  {
	store i141 %x, i141* @i141_s
	ret void
}

define void @i142_ls(i142 %x) nounwind  {
	store i142 %x, i142* @i142_s
	ret void
}

define void @i143_ls(i143 %x) nounwind  {
	store i143 %x, i143* @i143_s
	ret void
}

define void @i144_ls(i144 %x) nounwind  {
	store i144 %x, i144* @i144_s
	ret void
}

define void @i145_ls(i145 %x) nounwind  {
	store i145 %x, i145* @i145_s
	ret void
}

define void @i146_ls(i146 %x) nounwind  {
	store i146 %x, i146* @i146_s
	ret void
}

define void @i147_ls(i147 %x) nounwind  {
	store i147 %x, i147* @i147_s
	ret void
}

define void @i148_ls(i148 %x) nounwind  {
	store i148 %x, i148* @i148_s
	ret void
}

define void @i149_ls(i149 %x) nounwind  {
	store i149 %x, i149* @i149_s
	ret void
}

define void @i150_ls(i150 %x) nounwind  {
	store i150 %x, i150* @i150_s
	ret void
}

define void @i151_ls(i151 %x) nounwind  {
	store i151 %x, i151* @i151_s
	ret void
}

define void @i152_ls(i152 %x) nounwind  {
	store i152 %x, i152* @i152_s
	ret void
}

define void @i153_ls(i153 %x) nounwind  {
	store i153 %x, i153* @i153_s
	ret void
}

define void @i154_ls(i154 %x) nounwind  {
	store i154 %x, i154* @i154_s
	ret void
}

define void @i155_ls(i155 %x) nounwind  {
	store i155 %x, i155* @i155_s
	ret void
}

define void @i156_ls(i156 %x) nounwind  {
	store i156 %x, i156* @i156_s
	ret void
}

define void @i157_ls(i157 %x) nounwind  {
	store i157 %x, i157* @i157_s
	ret void
}

define void @i158_ls(i158 %x) nounwind  {
	store i158 %x, i158* @i158_s
	ret void
}

define void @i159_ls(i159 %x) nounwind  {
	store i159 %x, i159* @i159_s
	ret void
}

define void @i160_ls(i160 %x) nounwind  {
	store i160 %x, i160* @i160_s
	ret void
}

define void @i161_ls(i161 %x) nounwind  {
	store i161 %x, i161* @i161_s
	ret void
}

define void @i162_ls(i162 %x) nounwind  {
	store i162 %x, i162* @i162_s
	ret void
}

define void @i163_ls(i163 %x) nounwind  {
	store i163 %x, i163* @i163_s
	ret void
}

define void @i164_ls(i164 %x) nounwind  {
	store i164 %x, i164* @i164_s
	ret void
}

define void @i165_ls(i165 %x) nounwind  {
	store i165 %x, i165* @i165_s
	ret void
}

define void @i166_ls(i166 %x) nounwind  {
	store i166 %x, i166* @i166_s
	ret void
}

define void @i167_ls(i167 %x) nounwind  {
	store i167 %x, i167* @i167_s
	ret void
}

define void @i168_ls(i168 %x) nounwind  {
	store i168 %x, i168* @i168_s
	ret void
}

define void @i169_ls(i169 %x) nounwind  {
	store i169 %x, i169* @i169_s
	ret void
}

define void @i170_ls(i170 %x) nounwind  {
	store i170 %x, i170* @i170_s
	ret void
}

define void @i171_ls(i171 %x) nounwind  {
	store i171 %x, i171* @i171_s
	ret void
}

define void @i172_ls(i172 %x) nounwind  {
	store i172 %x, i172* @i172_s
	ret void
}

define void @i173_ls(i173 %x) nounwind  {
	store i173 %x, i173* @i173_s
	ret void
}

define void @i174_ls(i174 %x) nounwind  {
	store i174 %x, i174* @i174_s
	ret void
}

define void @i175_ls(i175 %x) nounwind  {
	store i175 %x, i175* @i175_s
	ret void
}

define void @i176_ls(i176 %x) nounwind  {
	store i176 %x, i176* @i176_s
	ret void
}

define void @i177_ls(i177 %x) nounwind  {
	store i177 %x, i177* @i177_s
	ret void
}

define void @i178_ls(i178 %x) nounwind  {
	store i178 %x, i178* @i178_s
	ret void
}

define void @i179_ls(i179 %x) nounwind  {
	store i179 %x, i179* @i179_s
	ret void
}

define void @i180_ls(i180 %x) nounwind  {
	store i180 %x, i180* @i180_s
	ret void
}

define void @i181_ls(i181 %x) nounwind  {
	store i181 %x, i181* @i181_s
	ret void
}

define void @i182_ls(i182 %x) nounwind  {
	store i182 %x, i182* @i182_s
	ret void
}

define void @i183_ls(i183 %x) nounwind  {
	store i183 %x, i183* @i183_s
	ret void
}

define void @i184_ls(i184 %x) nounwind  {
	store i184 %x, i184* @i184_s
	ret void
}

define void @i185_ls(i185 %x) nounwind  {
	store i185 %x, i185* @i185_s
	ret void
}

define void @i186_ls(i186 %x) nounwind  {
	store i186 %x, i186* @i186_s
	ret void
}

define void @i187_ls(i187 %x) nounwind  {
	store i187 %x, i187* @i187_s
	ret void
}

define void @i188_ls(i188 %x) nounwind  {
	store i188 %x, i188* @i188_s
	ret void
}

define void @i189_ls(i189 %x) nounwind  {
	store i189 %x, i189* @i189_s
	ret void
}

define void @i190_ls(i190 %x) nounwind  {
	store i190 %x, i190* @i190_s
	ret void
}

define void @i191_ls(i191 %x) nounwind  {
	store i191 %x, i191* @i191_s
	ret void
}

define void @i192_ls(i192 %x) nounwind  {
	store i192 %x, i192* @i192_s
	ret void
}

define void @i193_ls(i193 %x) nounwind  {
	store i193 %x, i193* @i193_s
	ret void
}

define void @i194_ls(i194 %x) nounwind  {
	store i194 %x, i194* @i194_s
	ret void
}

define void @i195_ls(i195 %x) nounwind  {
	store i195 %x, i195* @i195_s
	ret void
}

define void @i196_ls(i196 %x) nounwind  {
	store i196 %x, i196* @i196_s
	ret void
}

define void @i197_ls(i197 %x) nounwind  {
	store i197 %x, i197* @i197_s
	ret void
}

define void @i198_ls(i198 %x) nounwind  {
	store i198 %x, i198* @i198_s
	ret void
}

define void @i199_ls(i199 %x) nounwind  {
	store i199 %x, i199* @i199_s
	ret void
}

define void @i200_ls(i200 %x) nounwind  {
	store i200 %x, i200* @i200_s
	ret void
}

define void @i201_ls(i201 %x) nounwind  {
	store i201 %x, i201* @i201_s
	ret void
}

define void @i202_ls(i202 %x) nounwind  {
	store i202 %x, i202* @i202_s
	ret void
}

define void @i203_ls(i203 %x) nounwind  {
	store i203 %x, i203* @i203_s
	ret void
}

define void @i204_ls(i204 %x) nounwind  {
	store i204 %x, i204* @i204_s
	ret void
}

define void @i205_ls(i205 %x) nounwind  {
	store i205 %x, i205* @i205_s
	ret void
}

define void @i206_ls(i206 %x) nounwind  {
	store i206 %x, i206* @i206_s
	ret void
}

define void @i207_ls(i207 %x) nounwind  {
	store i207 %x, i207* @i207_s
	ret void
}

define void @i208_ls(i208 %x) nounwind  {
	store i208 %x, i208* @i208_s
	ret void
}

define void @i209_ls(i209 %x) nounwind  {
	store i209 %x, i209* @i209_s
	ret void
}

define void @i210_ls(i210 %x) nounwind  {
	store i210 %x, i210* @i210_s
	ret void
}

define void @i211_ls(i211 %x) nounwind  {
	store i211 %x, i211* @i211_s
	ret void
}

define void @i212_ls(i212 %x) nounwind  {
	store i212 %x, i212* @i212_s
	ret void
}

define void @i213_ls(i213 %x) nounwind  {
	store i213 %x, i213* @i213_s
	ret void
}

define void @i214_ls(i214 %x) nounwind  {
	store i214 %x, i214* @i214_s
	ret void
}

define void @i215_ls(i215 %x) nounwind  {
	store i215 %x, i215* @i215_s
	ret void
}

define void @i216_ls(i216 %x) nounwind  {
	store i216 %x, i216* @i216_s
	ret void
}

define void @i217_ls(i217 %x) nounwind  {
	store i217 %x, i217* @i217_s
	ret void
}

define void @i218_ls(i218 %x) nounwind  {
	store i218 %x, i218* @i218_s
	ret void
}

define void @i219_ls(i219 %x) nounwind  {
	store i219 %x, i219* @i219_s
	ret void
}

define void @i220_ls(i220 %x) nounwind  {
	store i220 %x, i220* @i220_s
	ret void
}

define void @i221_ls(i221 %x) nounwind  {
	store i221 %x, i221* @i221_s
	ret void
}

define void @i222_ls(i222 %x) nounwind  {
	store i222 %x, i222* @i222_s
	ret void
}

define void @i223_ls(i223 %x) nounwind  {
	store i223 %x, i223* @i223_s
	ret void
}

define void @i224_ls(i224 %x) nounwind  {
	store i224 %x, i224* @i224_s
	ret void
}

define void @i225_ls(i225 %x) nounwind  {
	store i225 %x, i225* @i225_s
	ret void
}

define void @i226_ls(i226 %x) nounwind  {
	store i226 %x, i226* @i226_s
	ret void
}

define void @i227_ls(i227 %x) nounwind  {
	store i227 %x, i227* @i227_s
	ret void
}

define void @i228_ls(i228 %x) nounwind  {
	store i228 %x, i228* @i228_s
	ret void
}

define void @i229_ls(i229 %x) nounwind  {
	store i229 %x, i229* @i229_s
	ret void
}

define void @i230_ls(i230 %x) nounwind  {
	store i230 %x, i230* @i230_s
	ret void
}

define void @i231_ls(i231 %x) nounwind  {
	store i231 %x, i231* @i231_s
	ret void
}

define void @i232_ls(i232 %x) nounwind  {
	store i232 %x, i232* @i232_s
	ret void
}

define void @i233_ls(i233 %x) nounwind  {
	store i233 %x, i233* @i233_s
	ret void
}

define void @i234_ls(i234 %x) nounwind  {
	store i234 %x, i234* @i234_s
	ret void
}

define void @i235_ls(i235 %x) nounwind  {
	store i235 %x, i235* @i235_s
	ret void
}

define void @i236_ls(i236 %x) nounwind  {
	store i236 %x, i236* @i236_s
	ret void
}

define void @i237_ls(i237 %x) nounwind  {
	store i237 %x, i237* @i237_s
	ret void
}

define void @i238_ls(i238 %x) nounwind  {
	store i238 %x, i238* @i238_s
	ret void
}

define void @i239_ls(i239 %x) nounwind  {
	store i239 %x, i239* @i239_s
	ret void
}

define void @i240_ls(i240 %x) nounwind  {
	store i240 %x, i240* @i240_s
	ret void
}

define void @i241_ls(i241 %x) nounwind  {
	store i241 %x, i241* @i241_s
	ret void
}

define void @i242_ls(i242 %x) nounwind  {
	store i242 %x, i242* @i242_s
	ret void
}

define void @i243_ls(i243 %x) nounwind  {
	store i243 %x, i243* @i243_s
	ret void
}

define void @i244_ls(i244 %x) nounwind  {
	store i244 %x, i244* @i244_s
	ret void
}

define void @i245_ls(i245 %x) nounwind  {
	store i245 %x, i245* @i245_s
	ret void
}

define void @i246_ls(i246 %x) nounwind  {
	store i246 %x, i246* @i246_s
	ret void
}

define void @i247_ls(i247 %x) nounwind  {
	store i247 %x, i247* @i247_s
	ret void
}

define void @i248_ls(i248 %x) nounwind  {
	store i248 %x, i248* @i248_s
	ret void
}

define void @i249_ls(i249 %x) nounwind  {
	store i249 %x, i249* @i249_s
	ret void
}

define void @i250_ls(i250 %x) nounwind  {
	store i250 %x, i250* @i250_s
	ret void
}

define void @i251_ls(i251 %x) nounwind  {
	store i251 %x, i251* @i251_s
	ret void
}

define void @i252_ls(i252 %x) nounwind  {
	store i252 %x, i252* @i252_s
	ret void
}

define void @i253_ls(i253 %x) nounwind  {
	store i253 %x, i253* @i253_s
	ret void
}

define void @i254_ls(i254 %x) nounwind  {
	store i254 %x, i254* @i254_s
	ret void
}

define void @i255_ls(i255 %x) nounwind  {
	store i255 %x, i255* @i255_s
	ret void
}

define void @i256_ls(i256 %x) nounwind  {
	store i256 %x, i256* @i256_s
	ret void
}
