; RUN: llc < %s > %t

; NVPTX does not support arbitrary integer types and has acceptable subset tested in NVPTX/APIntLoadStore.ll
; UNSUPPORTED: target=nvptx{{.*}}

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
@i65_l = external global i65		; <ptr> [#uses=1]
@i65_s = external global i65		; <ptr> [#uses=1]
@i66_l = external global i66		; <ptr> [#uses=1]
@i66_s = external global i66		; <ptr> [#uses=1]
@i67_l = external global i67		; <ptr> [#uses=1]
@i67_s = external global i67		; <ptr> [#uses=1]
@i68_l = external global i68		; <ptr> [#uses=1]
@i68_s = external global i68		; <ptr> [#uses=1]
@i69_l = external global i69		; <ptr> [#uses=1]
@i69_s = external global i69		; <ptr> [#uses=1]
@i70_l = external global i70		; <ptr> [#uses=1]
@i70_s = external global i70		; <ptr> [#uses=1]
@i71_l = external global i71		; <ptr> [#uses=1]
@i71_s = external global i71		; <ptr> [#uses=1]
@i72_l = external global i72		; <ptr> [#uses=1]
@i72_s = external global i72		; <ptr> [#uses=1]
@i73_l = external global i73		; <ptr> [#uses=1]
@i73_s = external global i73		; <ptr> [#uses=1]
@i74_l = external global i74		; <ptr> [#uses=1]
@i74_s = external global i74		; <ptr> [#uses=1]
@i75_l = external global i75		; <ptr> [#uses=1]
@i75_s = external global i75		; <ptr> [#uses=1]
@i76_l = external global i76		; <ptr> [#uses=1]
@i76_s = external global i76		; <ptr> [#uses=1]
@i77_l = external global i77		; <ptr> [#uses=1]
@i77_s = external global i77		; <ptr> [#uses=1]
@i78_l = external global i78		; <ptr> [#uses=1]
@i78_s = external global i78		; <ptr> [#uses=1]
@i79_l = external global i79		; <ptr> [#uses=1]
@i79_s = external global i79		; <ptr> [#uses=1]
@i80_l = external global i80		; <ptr> [#uses=1]
@i80_s = external global i80		; <ptr> [#uses=1]
@i81_l = external global i81		; <ptr> [#uses=1]
@i81_s = external global i81		; <ptr> [#uses=1]
@i82_l = external global i82		; <ptr> [#uses=1]
@i82_s = external global i82		; <ptr> [#uses=1]
@i83_l = external global i83		; <ptr> [#uses=1]
@i83_s = external global i83		; <ptr> [#uses=1]
@i84_l = external global i84		; <ptr> [#uses=1]
@i84_s = external global i84		; <ptr> [#uses=1]
@i85_l = external global i85		; <ptr> [#uses=1]
@i85_s = external global i85		; <ptr> [#uses=1]
@i86_l = external global i86		; <ptr> [#uses=1]
@i86_s = external global i86		; <ptr> [#uses=1]
@i87_l = external global i87		; <ptr> [#uses=1]
@i87_s = external global i87		; <ptr> [#uses=1]
@i88_l = external global i88		; <ptr> [#uses=1]
@i88_s = external global i88		; <ptr> [#uses=1]
@i89_l = external global i89		; <ptr> [#uses=1]
@i89_s = external global i89		; <ptr> [#uses=1]
@i90_l = external global i90		; <ptr> [#uses=1]
@i90_s = external global i90		; <ptr> [#uses=1]
@i91_l = external global i91		; <ptr> [#uses=1]
@i91_s = external global i91		; <ptr> [#uses=1]
@i92_l = external global i92		; <ptr> [#uses=1]
@i92_s = external global i92		; <ptr> [#uses=1]
@i93_l = external global i93		; <ptr> [#uses=1]
@i93_s = external global i93		; <ptr> [#uses=1]
@i94_l = external global i94		; <ptr> [#uses=1]
@i94_s = external global i94		; <ptr> [#uses=1]
@i95_l = external global i95		; <ptr> [#uses=1]
@i95_s = external global i95		; <ptr> [#uses=1]
@i96_l = external global i96		; <ptr> [#uses=1]
@i96_s = external global i96		; <ptr> [#uses=1]
@i97_l = external global i97		; <ptr> [#uses=1]
@i97_s = external global i97		; <ptr> [#uses=1]
@i98_l = external global i98		; <ptr> [#uses=1]
@i98_s = external global i98		; <ptr> [#uses=1]
@i99_l = external global i99		; <ptr> [#uses=1]
@i99_s = external global i99		; <ptr> [#uses=1]
@i100_l = external global i100		; <ptr> [#uses=1]
@i100_s = external global i100		; <ptr> [#uses=1]
@i101_l = external global i101		; <ptr> [#uses=1]
@i101_s = external global i101		; <ptr> [#uses=1]
@i102_l = external global i102		; <ptr> [#uses=1]
@i102_s = external global i102		; <ptr> [#uses=1]
@i103_l = external global i103		; <ptr> [#uses=1]
@i103_s = external global i103		; <ptr> [#uses=1]
@i104_l = external global i104		; <ptr> [#uses=1]
@i104_s = external global i104		; <ptr> [#uses=1]
@i105_l = external global i105		; <ptr> [#uses=1]
@i105_s = external global i105		; <ptr> [#uses=1]
@i106_l = external global i106		; <ptr> [#uses=1]
@i106_s = external global i106		; <ptr> [#uses=1]
@i107_l = external global i107		; <ptr> [#uses=1]
@i107_s = external global i107		; <ptr> [#uses=1]
@i108_l = external global i108		; <ptr> [#uses=1]
@i108_s = external global i108		; <ptr> [#uses=1]
@i109_l = external global i109		; <ptr> [#uses=1]
@i109_s = external global i109		; <ptr> [#uses=1]
@i110_l = external global i110		; <ptr> [#uses=1]
@i110_s = external global i110		; <ptr> [#uses=1]
@i111_l = external global i111		; <ptr> [#uses=1]
@i111_s = external global i111		; <ptr> [#uses=1]
@i112_l = external global i112		; <ptr> [#uses=1]
@i112_s = external global i112		; <ptr> [#uses=1]
@i113_l = external global i113		; <ptr> [#uses=1]
@i113_s = external global i113		; <ptr> [#uses=1]
@i114_l = external global i114		; <ptr> [#uses=1]
@i114_s = external global i114		; <ptr> [#uses=1]
@i115_l = external global i115		; <ptr> [#uses=1]
@i115_s = external global i115		; <ptr> [#uses=1]
@i116_l = external global i116		; <ptr> [#uses=1]
@i116_s = external global i116		; <ptr> [#uses=1]
@i117_l = external global i117		; <ptr> [#uses=1]
@i117_s = external global i117		; <ptr> [#uses=1]
@i118_l = external global i118		; <ptr> [#uses=1]
@i118_s = external global i118		; <ptr> [#uses=1]
@i119_l = external global i119		; <ptr> [#uses=1]
@i119_s = external global i119		; <ptr> [#uses=1]
@i120_l = external global i120		; <ptr> [#uses=1]
@i120_s = external global i120		; <ptr> [#uses=1]
@i121_l = external global i121		; <ptr> [#uses=1]
@i121_s = external global i121		; <ptr> [#uses=1]
@i122_l = external global i122		; <ptr> [#uses=1]
@i122_s = external global i122		; <ptr> [#uses=1]
@i123_l = external global i123		; <ptr> [#uses=1]
@i123_s = external global i123		; <ptr> [#uses=1]
@i124_l = external global i124		; <ptr> [#uses=1]
@i124_s = external global i124		; <ptr> [#uses=1]
@i125_l = external global i125		; <ptr> [#uses=1]
@i125_s = external global i125		; <ptr> [#uses=1]
@i126_l = external global i126		; <ptr> [#uses=1]
@i126_s = external global i126		; <ptr> [#uses=1]
@i127_l = external global i127		; <ptr> [#uses=1]
@i127_s = external global i127		; <ptr> [#uses=1]
@i128_l = external global i128		; <ptr> [#uses=1]
@i128_s = external global i128		; <ptr> [#uses=1]
@i129_l = external global i129		; <ptr> [#uses=1]
@i129_s = external global i129		; <ptr> [#uses=1]
@i130_l = external global i130		; <ptr> [#uses=1]
@i130_s = external global i130		; <ptr> [#uses=1]
@i131_l = external global i131		; <ptr> [#uses=1]
@i131_s = external global i131		; <ptr> [#uses=1]
@i132_l = external global i132		; <ptr> [#uses=1]
@i132_s = external global i132		; <ptr> [#uses=1]
@i133_l = external global i133		; <ptr> [#uses=1]
@i133_s = external global i133		; <ptr> [#uses=1]
@i134_l = external global i134		; <ptr> [#uses=1]
@i134_s = external global i134		; <ptr> [#uses=1]
@i135_l = external global i135		; <ptr> [#uses=1]
@i135_s = external global i135		; <ptr> [#uses=1]
@i136_l = external global i136		; <ptr> [#uses=1]
@i136_s = external global i136		; <ptr> [#uses=1]
@i137_l = external global i137		; <ptr> [#uses=1]
@i137_s = external global i137		; <ptr> [#uses=1]
@i138_l = external global i138		; <ptr> [#uses=1]
@i138_s = external global i138		; <ptr> [#uses=1]
@i139_l = external global i139		; <ptr> [#uses=1]
@i139_s = external global i139		; <ptr> [#uses=1]
@i140_l = external global i140		; <ptr> [#uses=1]
@i140_s = external global i140		; <ptr> [#uses=1]
@i141_l = external global i141		; <ptr> [#uses=1]
@i141_s = external global i141		; <ptr> [#uses=1]
@i142_l = external global i142		; <ptr> [#uses=1]
@i142_s = external global i142		; <ptr> [#uses=1]
@i143_l = external global i143		; <ptr> [#uses=1]
@i143_s = external global i143		; <ptr> [#uses=1]
@i144_l = external global i144		; <ptr> [#uses=1]
@i144_s = external global i144		; <ptr> [#uses=1]
@i145_l = external global i145		; <ptr> [#uses=1]
@i145_s = external global i145		; <ptr> [#uses=1]
@i146_l = external global i146		; <ptr> [#uses=1]
@i146_s = external global i146		; <ptr> [#uses=1]
@i147_l = external global i147		; <ptr> [#uses=1]
@i147_s = external global i147		; <ptr> [#uses=1]
@i148_l = external global i148		; <ptr> [#uses=1]
@i148_s = external global i148		; <ptr> [#uses=1]
@i149_l = external global i149		; <ptr> [#uses=1]
@i149_s = external global i149		; <ptr> [#uses=1]
@i150_l = external global i150		; <ptr> [#uses=1]
@i150_s = external global i150		; <ptr> [#uses=1]
@i151_l = external global i151		; <ptr> [#uses=1]
@i151_s = external global i151		; <ptr> [#uses=1]
@i152_l = external global i152		; <ptr> [#uses=1]
@i152_s = external global i152		; <ptr> [#uses=1]
@i153_l = external global i153		; <ptr> [#uses=1]
@i153_s = external global i153		; <ptr> [#uses=1]
@i154_l = external global i154		; <ptr> [#uses=1]
@i154_s = external global i154		; <ptr> [#uses=1]
@i155_l = external global i155		; <ptr> [#uses=1]
@i155_s = external global i155		; <ptr> [#uses=1]
@i156_l = external global i156		; <ptr> [#uses=1]
@i156_s = external global i156		; <ptr> [#uses=1]
@i157_l = external global i157		; <ptr> [#uses=1]
@i157_s = external global i157		; <ptr> [#uses=1]
@i158_l = external global i158		; <ptr> [#uses=1]
@i158_s = external global i158		; <ptr> [#uses=1]
@i159_l = external global i159		; <ptr> [#uses=1]
@i159_s = external global i159		; <ptr> [#uses=1]
@i160_l = external global i160		; <ptr> [#uses=1]
@i160_s = external global i160		; <ptr> [#uses=1]
@i161_l = external global i161		; <ptr> [#uses=1]
@i161_s = external global i161		; <ptr> [#uses=1]
@i162_l = external global i162		; <ptr> [#uses=1]
@i162_s = external global i162		; <ptr> [#uses=1]
@i163_l = external global i163		; <ptr> [#uses=1]
@i163_s = external global i163		; <ptr> [#uses=1]
@i164_l = external global i164		; <ptr> [#uses=1]
@i164_s = external global i164		; <ptr> [#uses=1]
@i165_l = external global i165		; <ptr> [#uses=1]
@i165_s = external global i165		; <ptr> [#uses=1]
@i166_l = external global i166		; <ptr> [#uses=1]
@i166_s = external global i166		; <ptr> [#uses=1]
@i167_l = external global i167		; <ptr> [#uses=1]
@i167_s = external global i167		; <ptr> [#uses=1]
@i168_l = external global i168		; <ptr> [#uses=1]
@i168_s = external global i168		; <ptr> [#uses=1]
@i169_l = external global i169		; <ptr> [#uses=1]
@i169_s = external global i169		; <ptr> [#uses=1]
@i170_l = external global i170		; <ptr> [#uses=1]
@i170_s = external global i170		; <ptr> [#uses=1]
@i171_l = external global i171		; <ptr> [#uses=1]
@i171_s = external global i171		; <ptr> [#uses=1]
@i172_l = external global i172		; <ptr> [#uses=1]
@i172_s = external global i172		; <ptr> [#uses=1]
@i173_l = external global i173		; <ptr> [#uses=1]
@i173_s = external global i173		; <ptr> [#uses=1]
@i174_l = external global i174		; <ptr> [#uses=1]
@i174_s = external global i174		; <ptr> [#uses=1]
@i175_l = external global i175		; <ptr> [#uses=1]
@i175_s = external global i175		; <ptr> [#uses=1]
@i176_l = external global i176		; <ptr> [#uses=1]
@i176_s = external global i176		; <ptr> [#uses=1]
@i177_l = external global i177		; <ptr> [#uses=1]
@i177_s = external global i177		; <ptr> [#uses=1]
@i178_l = external global i178		; <ptr> [#uses=1]
@i178_s = external global i178		; <ptr> [#uses=1]
@i179_l = external global i179		; <ptr> [#uses=1]
@i179_s = external global i179		; <ptr> [#uses=1]
@i180_l = external global i180		; <ptr> [#uses=1]
@i180_s = external global i180		; <ptr> [#uses=1]
@i181_l = external global i181		; <ptr> [#uses=1]
@i181_s = external global i181		; <ptr> [#uses=1]
@i182_l = external global i182		; <ptr> [#uses=1]
@i182_s = external global i182		; <ptr> [#uses=1]
@i183_l = external global i183		; <ptr> [#uses=1]
@i183_s = external global i183		; <ptr> [#uses=1]
@i184_l = external global i184		; <ptr> [#uses=1]
@i184_s = external global i184		; <ptr> [#uses=1]
@i185_l = external global i185		; <ptr> [#uses=1]
@i185_s = external global i185		; <ptr> [#uses=1]
@i186_l = external global i186		; <ptr> [#uses=1]
@i186_s = external global i186		; <ptr> [#uses=1]
@i187_l = external global i187		; <ptr> [#uses=1]
@i187_s = external global i187		; <ptr> [#uses=1]
@i188_l = external global i188		; <ptr> [#uses=1]
@i188_s = external global i188		; <ptr> [#uses=1]
@i189_l = external global i189		; <ptr> [#uses=1]
@i189_s = external global i189		; <ptr> [#uses=1]
@i190_l = external global i190		; <ptr> [#uses=1]
@i190_s = external global i190		; <ptr> [#uses=1]
@i191_l = external global i191		; <ptr> [#uses=1]
@i191_s = external global i191		; <ptr> [#uses=1]
@i192_l = external global i192		; <ptr> [#uses=1]
@i192_s = external global i192		; <ptr> [#uses=1]
@i193_l = external global i193		; <ptr> [#uses=1]
@i193_s = external global i193		; <ptr> [#uses=1]
@i194_l = external global i194		; <ptr> [#uses=1]
@i194_s = external global i194		; <ptr> [#uses=1]
@i195_l = external global i195		; <ptr> [#uses=1]
@i195_s = external global i195		; <ptr> [#uses=1]
@i196_l = external global i196		; <ptr> [#uses=1]
@i196_s = external global i196		; <ptr> [#uses=1]
@i197_l = external global i197		; <ptr> [#uses=1]
@i197_s = external global i197		; <ptr> [#uses=1]
@i198_l = external global i198		; <ptr> [#uses=1]
@i198_s = external global i198		; <ptr> [#uses=1]
@i199_l = external global i199		; <ptr> [#uses=1]
@i199_s = external global i199		; <ptr> [#uses=1]
@i200_l = external global i200		; <ptr> [#uses=1]
@i200_s = external global i200		; <ptr> [#uses=1]
@i201_l = external global i201		; <ptr> [#uses=1]
@i201_s = external global i201		; <ptr> [#uses=1]
@i202_l = external global i202		; <ptr> [#uses=1]
@i202_s = external global i202		; <ptr> [#uses=1]
@i203_l = external global i203		; <ptr> [#uses=1]
@i203_s = external global i203		; <ptr> [#uses=1]
@i204_l = external global i204		; <ptr> [#uses=1]
@i204_s = external global i204		; <ptr> [#uses=1]
@i205_l = external global i205		; <ptr> [#uses=1]
@i205_s = external global i205		; <ptr> [#uses=1]
@i206_l = external global i206		; <ptr> [#uses=1]
@i206_s = external global i206		; <ptr> [#uses=1]
@i207_l = external global i207		; <ptr> [#uses=1]
@i207_s = external global i207		; <ptr> [#uses=1]
@i208_l = external global i208		; <ptr> [#uses=1]
@i208_s = external global i208		; <ptr> [#uses=1]
@i209_l = external global i209		; <ptr> [#uses=1]
@i209_s = external global i209		; <ptr> [#uses=1]
@i210_l = external global i210		; <ptr> [#uses=1]
@i210_s = external global i210		; <ptr> [#uses=1]
@i211_l = external global i211		; <ptr> [#uses=1]
@i211_s = external global i211		; <ptr> [#uses=1]
@i212_l = external global i212		; <ptr> [#uses=1]
@i212_s = external global i212		; <ptr> [#uses=1]
@i213_l = external global i213		; <ptr> [#uses=1]
@i213_s = external global i213		; <ptr> [#uses=1]
@i214_l = external global i214		; <ptr> [#uses=1]
@i214_s = external global i214		; <ptr> [#uses=1]
@i215_l = external global i215		; <ptr> [#uses=1]
@i215_s = external global i215		; <ptr> [#uses=1]
@i216_l = external global i216		; <ptr> [#uses=1]
@i216_s = external global i216		; <ptr> [#uses=1]
@i217_l = external global i217		; <ptr> [#uses=1]
@i217_s = external global i217		; <ptr> [#uses=1]
@i218_l = external global i218		; <ptr> [#uses=1]
@i218_s = external global i218		; <ptr> [#uses=1]
@i219_l = external global i219		; <ptr> [#uses=1]
@i219_s = external global i219		; <ptr> [#uses=1]
@i220_l = external global i220		; <ptr> [#uses=1]
@i220_s = external global i220		; <ptr> [#uses=1]
@i221_l = external global i221		; <ptr> [#uses=1]
@i221_s = external global i221		; <ptr> [#uses=1]
@i222_l = external global i222		; <ptr> [#uses=1]
@i222_s = external global i222		; <ptr> [#uses=1]
@i223_l = external global i223		; <ptr> [#uses=1]
@i223_s = external global i223		; <ptr> [#uses=1]
@i224_l = external global i224		; <ptr> [#uses=1]
@i224_s = external global i224		; <ptr> [#uses=1]
@i225_l = external global i225		; <ptr> [#uses=1]
@i225_s = external global i225		; <ptr> [#uses=1]
@i226_l = external global i226		; <ptr> [#uses=1]
@i226_s = external global i226		; <ptr> [#uses=1]
@i227_l = external global i227		; <ptr> [#uses=1]
@i227_s = external global i227		; <ptr> [#uses=1]
@i228_l = external global i228		; <ptr> [#uses=1]
@i228_s = external global i228		; <ptr> [#uses=1]
@i229_l = external global i229		; <ptr> [#uses=1]
@i229_s = external global i229		; <ptr> [#uses=1]
@i230_l = external global i230		; <ptr> [#uses=1]
@i230_s = external global i230		; <ptr> [#uses=1]
@i231_l = external global i231		; <ptr> [#uses=1]
@i231_s = external global i231		; <ptr> [#uses=1]
@i232_l = external global i232		; <ptr> [#uses=1]
@i232_s = external global i232		; <ptr> [#uses=1]
@i233_l = external global i233		; <ptr> [#uses=1]
@i233_s = external global i233		; <ptr> [#uses=1]
@i234_l = external global i234		; <ptr> [#uses=1]
@i234_s = external global i234		; <ptr> [#uses=1]
@i235_l = external global i235		; <ptr> [#uses=1]
@i235_s = external global i235		; <ptr> [#uses=1]
@i236_l = external global i236		; <ptr> [#uses=1]
@i236_s = external global i236		; <ptr> [#uses=1]
@i237_l = external global i237		; <ptr> [#uses=1]
@i237_s = external global i237		; <ptr> [#uses=1]
@i238_l = external global i238		; <ptr> [#uses=1]
@i238_s = external global i238		; <ptr> [#uses=1]
@i239_l = external global i239		; <ptr> [#uses=1]
@i239_s = external global i239		; <ptr> [#uses=1]
@i240_l = external global i240		; <ptr> [#uses=1]
@i240_s = external global i240		; <ptr> [#uses=1]
@i241_l = external global i241		; <ptr> [#uses=1]
@i241_s = external global i241		; <ptr> [#uses=1]
@i242_l = external global i242		; <ptr> [#uses=1]
@i242_s = external global i242		; <ptr> [#uses=1]
@i243_l = external global i243		; <ptr> [#uses=1]
@i243_s = external global i243		; <ptr> [#uses=1]
@i244_l = external global i244		; <ptr> [#uses=1]
@i244_s = external global i244		; <ptr> [#uses=1]
@i245_l = external global i245		; <ptr> [#uses=1]
@i245_s = external global i245		; <ptr> [#uses=1]
@i246_l = external global i246		; <ptr> [#uses=1]
@i246_s = external global i246		; <ptr> [#uses=1]
@i247_l = external global i247		; <ptr> [#uses=1]
@i247_s = external global i247		; <ptr> [#uses=1]
@i248_l = external global i248		; <ptr> [#uses=1]
@i248_s = external global i248		; <ptr> [#uses=1]
@i249_l = external global i249		; <ptr> [#uses=1]
@i249_s = external global i249		; <ptr> [#uses=1]
@i250_l = external global i250		; <ptr> [#uses=1]
@i250_s = external global i250		; <ptr> [#uses=1]
@i251_l = external global i251		; <ptr> [#uses=1]
@i251_s = external global i251		; <ptr> [#uses=1]
@i252_l = external global i252		; <ptr> [#uses=1]
@i252_s = external global i252		; <ptr> [#uses=1]
@i253_l = external global i253		; <ptr> [#uses=1]
@i253_s = external global i253		; <ptr> [#uses=1]
@i254_l = external global i254		; <ptr> [#uses=1]
@i254_s = external global i254		; <ptr> [#uses=1]
@i255_l = external global i255		; <ptr> [#uses=1]
@i255_s = external global i255		; <ptr> [#uses=1]
@i256_l = external global i256		; <ptr> [#uses=1]
@i256_s = external global i256		; <ptr> [#uses=1]

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

define void @i65_ls() nounwind  {
	%tmp = load i65, ptr @i65_l		; <i65> [#uses=1]
	store i65 %tmp, ptr @i65_s
	ret void
}

define void @i66_ls() nounwind  {
	%tmp = load i66, ptr @i66_l		; <i66> [#uses=1]
	store i66 %tmp, ptr @i66_s
	ret void
}

define void @i67_ls() nounwind  {
	%tmp = load i67, ptr @i67_l		; <i67> [#uses=1]
	store i67 %tmp, ptr @i67_s
	ret void
}

define void @i68_ls() nounwind  {
	%tmp = load i68, ptr @i68_l		; <i68> [#uses=1]
	store i68 %tmp, ptr @i68_s
	ret void
}

define void @i69_ls() nounwind  {
	%tmp = load i69, ptr @i69_l		; <i69> [#uses=1]
	store i69 %tmp, ptr @i69_s
	ret void
}

define void @i70_ls() nounwind  {
	%tmp = load i70, ptr @i70_l		; <i70> [#uses=1]
	store i70 %tmp, ptr @i70_s
	ret void
}

define void @i71_ls() nounwind  {
	%tmp = load i71, ptr @i71_l		; <i71> [#uses=1]
	store i71 %tmp, ptr @i71_s
	ret void
}

define void @i72_ls() nounwind  {
	%tmp = load i72, ptr @i72_l		; <i72> [#uses=1]
	store i72 %tmp, ptr @i72_s
	ret void
}

define void @i73_ls() nounwind  {
	%tmp = load i73, ptr @i73_l		; <i73> [#uses=1]
	store i73 %tmp, ptr @i73_s
	ret void
}

define void @i74_ls() nounwind  {
	%tmp = load i74, ptr @i74_l		; <i74> [#uses=1]
	store i74 %tmp, ptr @i74_s
	ret void
}

define void @i75_ls() nounwind  {
	%tmp = load i75, ptr @i75_l		; <i75> [#uses=1]
	store i75 %tmp, ptr @i75_s
	ret void
}

define void @i76_ls() nounwind  {
	%tmp = load i76, ptr @i76_l		; <i76> [#uses=1]
	store i76 %tmp, ptr @i76_s
	ret void
}

define void @i77_ls() nounwind  {
	%tmp = load i77, ptr @i77_l		; <i77> [#uses=1]
	store i77 %tmp, ptr @i77_s
	ret void
}

define void @i78_ls() nounwind  {
	%tmp = load i78, ptr @i78_l		; <i78> [#uses=1]
	store i78 %tmp, ptr @i78_s
	ret void
}

define void @i79_ls() nounwind  {
	%tmp = load i79, ptr @i79_l		; <i79> [#uses=1]
	store i79 %tmp, ptr @i79_s
	ret void
}

define void @i80_ls() nounwind  {
	%tmp = load i80, ptr @i80_l		; <i80> [#uses=1]
	store i80 %tmp, ptr @i80_s
	ret void
}

define void @i81_ls() nounwind  {
	%tmp = load i81, ptr @i81_l		; <i81> [#uses=1]
	store i81 %tmp, ptr @i81_s
	ret void
}

define void @i82_ls() nounwind  {
	%tmp = load i82, ptr @i82_l		; <i82> [#uses=1]
	store i82 %tmp, ptr @i82_s
	ret void
}

define void @i83_ls() nounwind  {
	%tmp = load i83, ptr @i83_l		; <i83> [#uses=1]
	store i83 %tmp, ptr @i83_s
	ret void
}

define void @i84_ls() nounwind  {
	%tmp = load i84, ptr @i84_l		; <i84> [#uses=1]
	store i84 %tmp, ptr @i84_s
	ret void
}

define void @i85_ls() nounwind  {
	%tmp = load i85, ptr @i85_l		; <i85> [#uses=1]
	store i85 %tmp, ptr @i85_s
	ret void
}

define void @i86_ls() nounwind  {
	%tmp = load i86, ptr @i86_l		; <i86> [#uses=1]
	store i86 %tmp, ptr @i86_s
	ret void
}

define void @i87_ls() nounwind  {
	%tmp = load i87, ptr @i87_l		; <i87> [#uses=1]
	store i87 %tmp, ptr @i87_s
	ret void
}

define void @i88_ls() nounwind  {
	%tmp = load i88, ptr @i88_l		; <i88> [#uses=1]
	store i88 %tmp, ptr @i88_s
	ret void
}

define void @i89_ls() nounwind  {
	%tmp = load i89, ptr @i89_l		; <i89> [#uses=1]
	store i89 %tmp, ptr @i89_s
	ret void
}

define void @i90_ls() nounwind  {
	%tmp = load i90, ptr @i90_l		; <i90> [#uses=1]
	store i90 %tmp, ptr @i90_s
	ret void
}

define void @i91_ls() nounwind  {
	%tmp = load i91, ptr @i91_l		; <i91> [#uses=1]
	store i91 %tmp, ptr @i91_s
	ret void
}

define void @i92_ls() nounwind  {
	%tmp = load i92, ptr @i92_l		; <i92> [#uses=1]
	store i92 %tmp, ptr @i92_s
	ret void
}

define void @i93_ls() nounwind  {
	%tmp = load i93, ptr @i93_l		; <i93> [#uses=1]
	store i93 %tmp, ptr @i93_s
	ret void
}

define void @i94_ls() nounwind  {
	%tmp = load i94, ptr @i94_l		; <i94> [#uses=1]
	store i94 %tmp, ptr @i94_s
	ret void
}

define void @i95_ls() nounwind  {
	%tmp = load i95, ptr @i95_l		; <i95> [#uses=1]
	store i95 %tmp, ptr @i95_s
	ret void
}

define void @i96_ls() nounwind  {
	%tmp = load i96, ptr @i96_l		; <i96> [#uses=1]
	store i96 %tmp, ptr @i96_s
	ret void
}

define void @i97_ls() nounwind  {
	%tmp = load i97, ptr @i97_l		; <i97> [#uses=1]
	store i97 %tmp, ptr @i97_s
	ret void
}

define void @i98_ls() nounwind  {
	%tmp = load i98, ptr @i98_l		; <i98> [#uses=1]
	store i98 %tmp, ptr @i98_s
	ret void
}

define void @i99_ls() nounwind  {
	%tmp = load i99, ptr @i99_l		; <i99> [#uses=1]
	store i99 %tmp, ptr @i99_s
	ret void
}

define void @i100_ls() nounwind  {
	%tmp = load i100, ptr @i100_l		; <i100> [#uses=1]
	store i100 %tmp, ptr @i100_s
	ret void
}

define void @i101_ls() nounwind  {
	%tmp = load i101, ptr @i101_l		; <i101> [#uses=1]
	store i101 %tmp, ptr @i101_s
	ret void
}

define void @i102_ls() nounwind  {
	%tmp = load i102, ptr @i102_l		; <i102> [#uses=1]
	store i102 %tmp, ptr @i102_s
	ret void
}

define void @i103_ls() nounwind  {
	%tmp = load i103, ptr @i103_l		; <i103> [#uses=1]
	store i103 %tmp, ptr @i103_s
	ret void
}

define void @i104_ls() nounwind  {
	%tmp = load i104, ptr @i104_l		; <i104> [#uses=1]
	store i104 %tmp, ptr @i104_s
	ret void
}

define void @i105_ls() nounwind  {
	%tmp = load i105, ptr @i105_l		; <i105> [#uses=1]
	store i105 %tmp, ptr @i105_s
	ret void
}

define void @i106_ls() nounwind  {
	%tmp = load i106, ptr @i106_l		; <i106> [#uses=1]
	store i106 %tmp, ptr @i106_s
	ret void
}

define void @i107_ls() nounwind  {
	%tmp = load i107, ptr @i107_l		; <i107> [#uses=1]
	store i107 %tmp, ptr @i107_s
	ret void
}

define void @i108_ls() nounwind  {
	%tmp = load i108, ptr @i108_l		; <i108> [#uses=1]
	store i108 %tmp, ptr @i108_s
	ret void
}

define void @i109_ls() nounwind  {
	%tmp = load i109, ptr @i109_l		; <i109> [#uses=1]
	store i109 %tmp, ptr @i109_s
	ret void
}

define void @i110_ls() nounwind  {
	%tmp = load i110, ptr @i110_l		; <i110> [#uses=1]
	store i110 %tmp, ptr @i110_s
	ret void
}

define void @i111_ls() nounwind  {
	%tmp = load i111, ptr @i111_l		; <i111> [#uses=1]
	store i111 %tmp, ptr @i111_s
	ret void
}

define void @i112_ls() nounwind  {
	%tmp = load i112, ptr @i112_l		; <i112> [#uses=1]
	store i112 %tmp, ptr @i112_s
	ret void
}

define void @i113_ls() nounwind  {
	%tmp = load i113, ptr @i113_l		; <i113> [#uses=1]
	store i113 %tmp, ptr @i113_s
	ret void
}

define void @i114_ls() nounwind  {
	%tmp = load i114, ptr @i114_l		; <i114> [#uses=1]
	store i114 %tmp, ptr @i114_s
	ret void
}

define void @i115_ls() nounwind  {
	%tmp = load i115, ptr @i115_l		; <i115> [#uses=1]
	store i115 %tmp, ptr @i115_s
	ret void
}

define void @i116_ls() nounwind  {
	%tmp = load i116, ptr @i116_l		; <i116> [#uses=1]
	store i116 %tmp, ptr @i116_s
	ret void
}

define void @i117_ls() nounwind  {
	%tmp = load i117, ptr @i117_l		; <i117> [#uses=1]
	store i117 %tmp, ptr @i117_s
	ret void
}

define void @i118_ls() nounwind  {
	%tmp = load i118, ptr @i118_l		; <i118> [#uses=1]
	store i118 %tmp, ptr @i118_s
	ret void
}

define void @i119_ls() nounwind  {
	%tmp = load i119, ptr @i119_l		; <i119> [#uses=1]
	store i119 %tmp, ptr @i119_s
	ret void
}

define void @i120_ls() nounwind  {
	%tmp = load i120, ptr @i120_l		; <i120> [#uses=1]
	store i120 %tmp, ptr @i120_s
	ret void
}

define void @i121_ls() nounwind  {
	%tmp = load i121, ptr @i121_l		; <i121> [#uses=1]
	store i121 %tmp, ptr @i121_s
	ret void
}

define void @i122_ls() nounwind  {
	%tmp = load i122, ptr @i122_l		; <i122> [#uses=1]
	store i122 %tmp, ptr @i122_s
	ret void
}

define void @i123_ls() nounwind  {
	%tmp = load i123, ptr @i123_l		; <i123> [#uses=1]
	store i123 %tmp, ptr @i123_s
	ret void
}

define void @i124_ls() nounwind  {
	%tmp = load i124, ptr @i124_l		; <i124> [#uses=1]
	store i124 %tmp, ptr @i124_s
	ret void
}

define void @i125_ls() nounwind  {
	%tmp = load i125, ptr @i125_l		; <i125> [#uses=1]
	store i125 %tmp, ptr @i125_s
	ret void
}

define void @i126_ls() nounwind  {
	%tmp = load i126, ptr @i126_l		; <i126> [#uses=1]
	store i126 %tmp, ptr @i126_s
	ret void
}

define void @i127_ls() nounwind  {
	%tmp = load i127, ptr @i127_l		; <i127> [#uses=1]
	store i127 %tmp, ptr @i127_s
	ret void
}

define void @i128_ls() nounwind  {
	%tmp = load i128, ptr @i128_l		; <i128> [#uses=1]
	store i128 %tmp, ptr @i128_s
	ret void
}

define void @i129_ls() nounwind  {
	%tmp = load i129, ptr @i129_l		; <i129> [#uses=1]
	store i129 %tmp, ptr @i129_s
	ret void
}

define void @i130_ls() nounwind  {
	%tmp = load i130, ptr @i130_l		; <i130> [#uses=1]
	store i130 %tmp, ptr @i130_s
	ret void
}

define void @i131_ls() nounwind  {
	%tmp = load i131, ptr @i131_l		; <i131> [#uses=1]
	store i131 %tmp, ptr @i131_s
	ret void
}

define void @i132_ls() nounwind  {
	%tmp = load i132, ptr @i132_l		; <i132> [#uses=1]
	store i132 %tmp, ptr @i132_s
	ret void
}

define void @i133_ls() nounwind  {
	%tmp = load i133, ptr @i133_l		; <i133> [#uses=1]
	store i133 %tmp, ptr @i133_s
	ret void
}

define void @i134_ls() nounwind  {
	%tmp = load i134, ptr @i134_l		; <i134> [#uses=1]
	store i134 %tmp, ptr @i134_s
	ret void
}

define void @i135_ls() nounwind  {
	%tmp = load i135, ptr @i135_l		; <i135> [#uses=1]
	store i135 %tmp, ptr @i135_s
	ret void
}

define void @i136_ls() nounwind  {
	%tmp = load i136, ptr @i136_l		; <i136> [#uses=1]
	store i136 %tmp, ptr @i136_s
	ret void
}

define void @i137_ls() nounwind  {
	%tmp = load i137, ptr @i137_l		; <i137> [#uses=1]
	store i137 %tmp, ptr @i137_s
	ret void
}

define void @i138_ls() nounwind  {
	%tmp = load i138, ptr @i138_l		; <i138> [#uses=1]
	store i138 %tmp, ptr @i138_s
	ret void
}

define void @i139_ls() nounwind  {
	%tmp = load i139, ptr @i139_l		; <i139> [#uses=1]
	store i139 %tmp, ptr @i139_s
	ret void
}

define void @i140_ls() nounwind  {
	%tmp = load i140, ptr @i140_l		; <i140> [#uses=1]
	store i140 %tmp, ptr @i140_s
	ret void
}

define void @i141_ls() nounwind  {
	%tmp = load i141, ptr @i141_l		; <i141> [#uses=1]
	store i141 %tmp, ptr @i141_s
	ret void
}

define void @i142_ls() nounwind  {
	%tmp = load i142, ptr @i142_l		; <i142> [#uses=1]
	store i142 %tmp, ptr @i142_s
	ret void
}

define void @i143_ls() nounwind  {
	%tmp = load i143, ptr @i143_l		; <i143> [#uses=1]
	store i143 %tmp, ptr @i143_s
	ret void
}

define void @i144_ls() nounwind  {
	%tmp = load i144, ptr @i144_l		; <i144> [#uses=1]
	store i144 %tmp, ptr @i144_s
	ret void
}

define void @i145_ls() nounwind  {
	%tmp = load i145, ptr @i145_l		; <i145> [#uses=1]
	store i145 %tmp, ptr @i145_s
	ret void
}

define void @i146_ls() nounwind  {
	%tmp = load i146, ptr @i146_l		; <i146> [#uses=1]
	store i146 %tmp, ptr @i146_s
	ret void
}

define void @i147_ls() nounwind  {
	%tmp = load i147, ptr @i147_l		; <i147> [#uses=1]
	store i147 %tmp, ptr @i147_s
	ret void
}

define void @i148_ls() nounwind  {
	%tmp = load i148, ptr @i148_l		; <i148> [#uses=1]
	store i148 %tmp, ptr @i148_s
	ret void
}

define void @i149_ls() nounwind  {
	%tmp = load i149, ptr @i149_l		; <i149> [#uses=1]
	store i149 %tmp, ptr @i149_s
	ret void
}

define void @i150_ls() nounwind  {
	%tmp = load i150, ptr @i150_l		; <i150> [#uses=1]
	store i150 %tmp, ptr @i150_s
	ret void
}

define void @i151_ls() nounwind  {
	%tmp = load i151, ptr @i151_l		; <i151> [#uses=1]
	store i151 %tmp, ptr @i151_s
	ret void
}

define void @i152_ls() nounwind  {
	%tmp = load i152, ptr @i152_l		; <i152> [#uses=1]
	store i152 %tmp, ptr @i152_s
	ret void
}

define void @i153_ls() nounwind  {
	%tmp = load i153, ptr @i153_l		; <i153> [#uses=1]
	store i153 %tmp, ptr @i153_s
	ret void
}

define void @i154_ls() nounwind  {
	%tmp = load i154, ptr @i154_l		; <i154> [#uses=1]
	store i154 %tmp, ptr @i154_s
	ret void
}

define void @i155_ls() nounwind  {
	%tmp = load i155, ptr @i155_l		; <i155> [#uses=1]
	store i155 %tmp, ptr @i155_s
	ret void
}

define void @i156_ls() nounwind  {
	%tmp = load i156, ptr @i156_l		; <i156> [#uses=1]
	store i156 %tmp, ptr @i156_s
	ret void
}

define void @i157_ls() nounwind  {
	%tmp = load i157, ptr @i157_l		; <i157> [#uses=1]
	store i157 %tmp, ptr @i157_s
	ret void
}

define void @i158_ls() nounwind  {
	%tmp = load i158, ptr @i158_l		; <i158> [#uses=1]
	store i158 %tmp, ptr @i158_s
	ret void
}

define void @i159_ls() nounwind  {
	%tmp = load i159, ptr @i159_l		; <i159> [#uses=1]
	store i159 %tmp, ptr @i159_s
	ret void
}

define void @i160_ls() nounwind  {
	%tmp = load i160, ptr @i160_l		; <i160> [#uses=1]
	store i160 %tmp, ptr @i160_s
	ret void
}

define void @i161_ls() nounwind  {
	%tmp = load i161, ptr @i161_l		; <i161> [#uses=1]
	store i161 %tmp, ptr @i161_s
	ret void
}

define void @i162_ls() nounwind  {
	%tmp = load i162, ptr @i162_l		; <i162> [#uses=1]
	store i162 %tmp, ptr @i162_s
	ret void
}

define void @i163_ls() nounwind  {
	%tmp = load i163, ptr @i163_l		; <i163> [#uses=1]
	store i163 %tmp, ptr @i163_s
	ret void
}

define void @i164_ls() nounwind  {
	%tmp = load i164, ptr @i164_l		; <i164> [#uses=1]
	store i164 %tmp, ptr @i164_s
	ret void
}

define void @i165_ls() nounwind  {
	%tmp = load i165, ptr @i165_l		; <i165> [#uses=1]
	store i165 %tmp, ptr @i165_s
	ret void
}

define void @i166_ls() nounwind  {
	%tmp = load i166, ptr @i166_l		; <i166> [#uses=1]
	store i166 %tmp, ptr @i166_s
	ret void
}

define void @i167_ls() nounwind  {
	%tmp = load i167, ptr @i167_l		; <i167> [#uses=1]
	store i167 %tmp, ptr @i167_s
	ret void
}

define void @i168_ls() nounwind  {
	%tmp = load i168, ptr @i168_l		; <i168> [#uses=1]
	store i168 %tmp, ptr @i168_s
	ret void
}

define void @i169_ls() nounwind  {
	%tmp = load i169, ptr @i169_l		; <i169> [#uses=1]
	store i169 %tmp, ptr @i169_s
	ret void
}

define void @i170_ls() nounwind  {
	%tmp = load i170, ptr @i170_l		; <i170> [#uses=1]
	store i170 %tmp, ptr @i170_s
	ret void
}

define void @i171_ls() nounwind  {
	%tmp = load i171, ptr @i171_l		; <i171> [#uses=1]
	store i171 %tmp, ptr @i171_s
	ret void
}

define void @i172_ls() nounwind  {
	%tmp = load i172, ptr @i172_l		; <i172> [#uses=1]
	store i172 %tmp, ptr @i172_s
	ret void
}

define void @i173_ls() nounwind  {
	%tmp = load i173, ptr @i173_l		; <i173> [#uses=1]
	store i173 %tmp, ptr @i173_s
	ret void
}

define void @i174_ls() nounwind  {
	%tmp = load i174, ptr @i174_l		; <i174> [#uses=1]
	store i174 %tmp, ptr @i174_s
	ret void
}

define void @i175_ls() nounwind  {
	%tmp = load i175, ptr @i175_l		; <i175> [#uses=1]
	store i175 %tmp, ptr @i175_s
	ret void
}

define void @i176_ls() nounwind  {
	%tmp = load i176, ptr @i176_l		; <i176> [#uses=1]
	store i176 %tmp, ptr @i176_s
	ret void
}

define void @i177_ls() nounwind  {
	%tmp = load i177, ptr @i177_l		; <i177> [#uses=1]
	store i177 %tmp, ptr @i177_s
	ret void
}

define void @i178_ls() nounwind  {
	%tmp = load i178, ptr @i178_l		; <i178> [#uses=1]
	store i178 %tmp, ptr @i178_s
	ret void
}

define void @i179_ls() nounwind  {
	%tmp = load i179, ptr @i179_l		; <i179> [#uses=1]
	store i179 %tmp, ptr @i179_s
	ret void
}

define void @i180_ls() nounwind  {
	%tmp = load i180, ptr @i180_l		; <i180> [#uses=1]
	store i180 %tmp, ptr @i180_s
	ret void
}

define void @i181_ls() nounwind  {
	%tmp = load i181, ptr @i181_l		; <i181> [#uses=1]
	store i181 %tmp, ptr @i181_s
	ret void
}

define void @i182_ls() nounwind  {
	%tmp = load i182, ptr @i182_l		; <i182> [#uses=1]
	store i182 %tmp, ptr @i182_s
	ret void
}

define void @i183_ls() nounwind  {
	%tmp = load i183, ptr @i183_l		; <i183> [#uses=1]
	store i183 %tmp, ptr @i183_s
	ret void
}

define void @i184_ls() nounwind  {
	%tmp = load i184, ptr @i184_l		; <i184> [#uses=1]
	store i184 %tmp, ptr @i184_s
	ret void
}

define void @i185_ls() nounwind  {
	%tmp = load i185, ptr @i185_l		; <i185> [#uses=1]
	store i185 %tmp, ptr @i185_s
	ret void
}

define void @i186_ls() nounwind  {
	%tmp = load i186, ptr @i186_l		; <i186> [#uses=1]
	store i186 %tmp, ptr @i186_s
	ret void
}

define void @i187_ls() nounwind  {
	%tmp = load i187, ptr @i187_l		; <i187> [#uses=1]
	store i187 %tmp, ptr @i187_s
	ret void
}

define void @i188_ls() nounwind  {
	%tmp = load i188, ptr @i188_l		; <i188> [#uses=1]
	store i188 %tmp, ptr @i188_s
	ret void
}

define void @i189_ls() nounwind  {
	%tmp = load i189, ptr @i189_l		; <i189> [#uses=1]
	store i189 %tmp, ptr @i189_s
	ret void
}

define void @i190_ls() nounwind  {
	%tmp = load i190, ptr @i190_l		; <i190> [#uses=1]
	store i190 %tmp, ptr @i190_s
	ret void
}

define void @i191_ls() nounwind  {
	%tmp = load i191, ptr @i191_l		; <i191> [#uses=1]
	store i191 %tmp, ptr @i191_s
	ret void
}

define void @i192_ls() nounwind  {
	%tmp = load i192, ptr @i192_l		; <i192> [#uses=1]
	store i192 %tmp, ptr @i192_s
	ret void
}

define void @i193_ls() nounwind  {
	%tmp = load i193, ptr @i193_l		; <i193> [#uses=1]
	store i193 %tmp, ptr @i193_s
	ret void
}

define void @i194_ls() nounwind  {
	%tmp = load i194, ptr @i194_l		; <i194> [#uses=1]
	store i194 %tmp, ptr @i194_s
	ret void
}

define void @i195_ls() nounwind  {
	%tmp = load i195, ptr @i195_l		; <i195> [#uses=1]
	store i195 %tmp, ptr @i195_s
	ret void
}

define void @i196_ls() nounwind  {
	%tmp = load i196, ptr @i196_l		; <i196> [#uses=1]
	store i196 %tmp, ptr @i196_s
	ret void
}

define void @i197_ls() nounwind  {
	%tmp = load i197, ptr @i197_l		; <i197> [#uses=1]
	store i197 %tmp, ptr @i197_s
	ret void
}

define void @i198_ls() nounwind  {
	%tmp = load i198, ptr @i198_l		; <i198> [#uses=1]
	store i198 %tmp, ptr @i198_s
	ret void
}

define void @i199_ls() nounwind  {
	%tmp = load i199, ptr @i199_l		; <i199> [#uses=1]
	store i199 %tmp, ptr @i199_s
	ret void
}

define void @i200_ls() nounwind  {
	%tmp = load i200, ptr @i200_l		; <i200> [#uses=1]
	store i200 %tmp, ptr @i200_s
	ret void
}

define void @i201_ls() nounwind  {
	%tmp = load i201, ptr @i201_l		; <i201> [#uses=1]
	store i201 %tmp, ptr @i201_s
	ret void
}

define void @i202_ls() nounwind  {
	%tmp = load i202, ptr @i202_l		; <i202> [#uses=1]
	store i202 %tmp, ptr @i202_s
	ret void
}

define void @i203_ls() nounwind  {
	%tmp = load i203, ptr @i203_l		; <i203> [#uses=1]
	store i203 %tmp, ptr @i203_s
	ret void
}

define void @i204_ls() nounwind  {
	%tmp = load i204, ptr @i204_l		; <i204> [#uses=1]
	store i204 %tmp, ptr @i204_s
	ret void
}

define void @i205_ls() nounwind  {
	%tmp = load i205, ptr @i205_l		; <i205> [#uses=1]
	store i205 %tmp, ptr @i205_s
	ret void
}

define void @i206_ls() nounwind  {
	%tmp = load i206, ptr @i206_l		; <i206> [#uses=1]
	store i206 %tmp, ptr @i206_s
	ret void
}

define void @i207_ls() nounwind  {
	%tmp = load i207, ptr @i207_l		; <i207> [#uses=1]
	store i207 %tmp, ptr @i207_s
	ret void
}

define void @i208_ls() nounwind  {
	%tmp = load i208, ptr @i208_l		; <i208> [#uses=1]
	store i208 %tmp, ptr @i208_s
	ret void
}

define void @i209_ls() nounwind  {
	%tmp = load i209, ptr @i209_l		; <i209> [#uses=1]
	store i209 %tmp, ptr @i209_s
	ret void
}

define void @i210_ls() nounwind  {
	%tmp = load i210, ptr @i210_l		; <i210> [#uses=1]
	store i210 %tmp, ptr @i210_s
	ret void
}

define void @i211_ls() nounwind  {
	%tmp = load i211, ptr @i211_l		; <i211> [#uses=1]
	store i211 %tmp, ptr @i211_s
	ret void
}

define void @i212_ls() nounwind  {
	%tmp = load i212, ptr @i212_l		; <i212> [#uses=1]
	store i212 %tmp, ptr @i212_s
	ret void
}

define void @i213_ls() nounwind  {
	%tmp = load i213, ptr @i213_l		; <i213> [#uses=1]
	store i213 %tmp, ptr @i213_s
	ret void
}

define void @i214_ls() nounwind  {
	%tmp = load i214, ptr @i214_l		; <i214> [#uses=1]
	store i214 %tmp, ptr @i214_s
	ret void
}

define void @i215_ls() nounwind  {
	%tmp = load i215, ptr @i215_l		; <i215> [#uses=1]
	store i215 %tmp, ptr @i215_s
	ret void
}

define void @i216_ls() nounwind  {
	%tmp = load i216, ptr @i216_l		; <i216> [#uses=1]
	store i216 %tmp, ptr @i216_s
	ret void
}

define void @i217_ls() nounwind  {
	%tmp = load i217, ptr @i217_l		; <i217> [#uses=1]
	store i217 %tmp, ptr @i217_s
	ret void
}

define void @i218_ls() nounwind  {
	%tmp = load i218, ptr @i218_l		; <i218> [#uses=1]
	store i218 %tmp, ptr @i218_s
	ret void
}

define void @i219_ls() nounwind  {
	%tmp = load i219, ptr @i219_l		; <i219> [#uses=1]
	store i219 %tmp, ptr @i219_s
	ret void
}

define void @i220_ls() nounwind  {
	%tmp = load i220, ptr @i220_l		; <i220> [#uses=1]
	store i220 %tmp, ptr @i220_s
	ret void
}

define void @i221_ls() nounwind  {
	%tmp = load i221, ptr @i221_l		; <i221> [#uses=1]
	store i221 %tmp, ptr @i221_s
	ret void
}

define void @i222_ls() nounwind  {
	%tmp = load i222, ptr @i222_l		; <i222> [#uses=1]
	store i222 %tmp, ptr @i222_s
	ret void
}

define void @i223_ls() nounwind  {
	%tmp = load i223, ptr @i223_l		; <i223> [#uses=1]
	store i223 %tmp, ptr @i223_s
	ret void
}

define void @i224_ls() nounwind  {
	%tmp = load i224, ptr @i224_l		; <i224> [#uses=1]
	store i224 %tmp, ptr @i224_s
	ret void
}

define void @i225_ls() nounwind  {
	%tmp = load i225, ptr @i225_l		; <i225> [#uses=1]
	store i225 %tmp, ptr @i225_s
	ret void
}

define void @i226_ls() nounwind  {
	%tmp = load i226, ptr @i226_l		; <i226> [#uses=1]
	store i226 %tmp, ptr @i226_s
	ret void
}

define void @i227_ls() nounwind  {
	%tmp = load i227, ptr @i227_l		; <i227> [#uses=1]
	store i227 %tmp, ptr @i227_s
	ret void
}

define void @i228_ls() nounwind  {
	%tmp = load i228, ptr @i228_l		; <i228> [#uses=1]
	store i228 %tmp, ptr @i228_s
	ret void
}

define void @i229_ls() nounwind  {
	%tmp = load i229, ptr @i229_l		; <i229> [#uses=1]
	store i229 %tmp, ptr @i229_s
	ret void
}

define void @i230_ls() nounwind  {
	%tmp = load i230, ptr @i230_l		; <i230> [#uses=1]
	store i230 %tmp, ptr @i230_s
	ret void
}

define void @i231_ls() nounwind  {
	%tmp = load i231, ptr @i231_l		; <i231> [#uses=1]
	store i231 %tmp, ptr @i231_s
	ret void
}

define void @i232_ls() nounwind  {
	%tmp = load i232, ptr @i232_l		; <i232> [#uses=1]
	store i232 %tmp, ptr @i232_s
	ret void
}

define void @i233_ls() nounwind  {
	%tmp = load i233, ptr @i233_l		; <i233> [#uses=1]
	store i233 %tmp, ptr @i233_s
	ret void
}

define void @i234_ls() nounwind  {
	%tmp = load i234, ptr @i234_l		; <i234> [#uses=1]
	store i234 %tmp, ptr @i234_s
	ret void
}

define void @i235_ls() nounwind  {
	%tmp = load i235, ptr @i235_l		; <i235> [#uses=1]
	store i235 %tmp, ptr @i235_s
	ret void
}

define void @i236_ls() nounwind  {
	%tmp = load i236, ptr @i236_l		; <i236> [#uses=1]
	store i236 %tmp, ptr @i236_s
	ret void
}

define void @i237_ls() nounwind  {
	%tmp = load i237, ptr @i237_l		; <i237> [#uses=1]
	store i237 %tmp, ptr @i237_s
	ret void
}

define void @i238_ls() nounwind  {
	%tmp = load i238, ptr @i238_l		; <i238> [#uses=1]
	store i238 %tmp, ptr @i238_s
	ret void
}

define void @i239_ls() nounwind  {
	%tmp = load i239, ptr @i239_l		; <i239> [#uses=1]
	store i239 %tmp, ptr @i239_s
	ret void
}

define void @i240_ls() nounwind  {
	%tmp = load i240, ptr @i240_l		; <i240> [#uses=1]
	store i240 %tmp, ptr @i240_s
	ret void
}

define void @i241_ls() nounwind  {
	%tmp = load i241, ptr @i241_l		; <i241> [#uses=1]
	store i241 %tmp, ptr @i241_s
	ret void
}

define void @i242_ls() nounwind  {
	%tmp = load i242, ptr @i242_l		; <i242> [#uses=1]
	store i242 %tmp, ptr @i242_s
	ret void
}

define void @i243_ls() nounwind  {
	%tmp = load i243, ptr @i243_l		; <i243> [#uses=1]
	store i243 %tmp, ptr @i243_s
	ret void
}

define void @i244_ls() nounwind  {
	%tmp = load i244, ptr @i244_l		; <i244> [#uses=1]
	store i244 %tmp, ptr @i244_s
	ret void
}

define void @i245_ls() nounwind  {
	%tmp = load i245, ptr @i245_l		; <i245> [#uses=1]
	store i245 %tmp, ptr @i245_s
	ret void
}

define void @i246_ls() nounwind  {
	%tmp = load i246, ptr @i246_l		; <i246> [#uses=1]
	store i246 %tmp, ptr @i246_s
	ret void
}

define void @i247_ls() nounwind  {
	%tmp = load i247, ptr @i247_l		; <i247> [#uses=1]
	store i247 %tmp, ptr @i247_s
	ret void
}

define void @i248_ls() nounwind  {
	%tmp = load i248, ptr @i248_l		; <i248> [#uses=1]
	store i248 %tmp, ptr @i248_s
	ret void
}

define void @i249_ls() nounwind  {
	%tmp = load i249, ptr @i249_l		; <i249> [#uses=1]
	store i249 %tmp, ptr @i249_s
	ret void
}

define void @i250_ls() nounwind  {
	%tmp = load i250, ptr @i250_l		; <i250> [#uses=1]
	store i250 %tmp, ptr @i250_s
	ret void
}

define void @i251_ls() nounwind  {
	%tmp = load i251, ptr @i251_l		; <i251> [#uses=1]
	store i251 %tmp, ptr @i251_s
	ret void
}

define void @i252_ls() nounwind  {
	%tmp = load i252, ptr @i252_l		; <i252> [#uses=1]
	store i252 %tmp, ptr @i252_s
	ret void
}

define void @i253_ls() nounwind  {
	%tmp = load i253, ptr @i253_l		; <i253> [#uses=1]
	store i253 %tmp, ptr @i253_s
	ret void
}

define void @i254_ls() nounwind  {
	%tmp = load i254, ptr @i254_l		; <i254> [#uses=1]
	store i254 %tmp, ptr @i254_s
	ret void
}

define void @i255_ls() nounwind  {
	%tmp = load i255, ptr @i255_l		; <i255> [#uses=1]
	store i255 %tmp, ptr @i255_s
	ret void
}

define void @i256_ls() nounwind  {
	%tmp = load i256, ptr @i256_l		; <i256> [#uses=1]
	store i256 %tmp, ptr @i256_s
	ret void
}
