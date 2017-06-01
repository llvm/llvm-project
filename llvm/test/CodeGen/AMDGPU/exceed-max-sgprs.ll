; RUN: not llc -march=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: scalar registers limit of 104 exceeded (106) in use_too_many_sgprs_tahiti
define void @use_too_many_sgprs_tahiti() #0 {
  call void asm sideeffect "", "~{SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7}" ()
  call void asm sideeffect "", "~{SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}" ()
  call void asm sideeffect "", "~{SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23}" ()
  call void asm sideeffect "", "~{SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}" ()
  call void asm sideeffect "", "~{SGPR32_SGPR33_SGPR34_SGPR35_SGPR36_SGPR37_SGPR38_SGPR39}" ()
  call void asm sideeffect "", "~{SGPR40_SGPR41_SGPR42_SGPR43_SGPR44_SGPR45_SGPR46_SGPR47}" ()
  call void asm sideeffect "", "~{SGPR48_SGPR49_SGPR50_SGPR51_SGPR52_SGPR53_SGPR54_SGPR55}" ()
  call void asm sideeffect "", "~{SGPR56_SGPR57_SGPR58_SGPR59_SGPR60_SGPR61_SGPR62_SGPR63}" ()
  call void asm sideeffect "", "~{SGPR64_SGPR65_SGPR66_SGPR67_SGPR68_SGPR69_SGPR70_SGPR71}" ()
  call void asm sideeffect "", "~{SGPR72_SGPR73_SGPR74_SGPR75_SGPR76_SGPR77_SGPR78_SGPR79}" ()
  call void asm sideeffect "", "~{SGPR80_SGPR81_SGPR82_SGPR83_SGPR84_SGPR85_SGPR86_SGPR87}" ()
  call void asm sideeffect "", "~{SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95}" ()
  call void asm sideeffect "", "~{SGPR96_SGPR97_SGPR98_SGPR99_SGPR100_SGPR101_SGPR102_SGPR103}" ()
  call void asm sideeffect "", "~{VCC}" ()
  ret void
}

; ERROR: error: scalar registers limit of 104 exceeded (106) in use_too_many_sgprs_bonaire
define void @use_too_many_sgprs_bonaire() #1 {
  call void asm sideeffect "", "~{SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7}" ()
  call void asm sideeffect "", "~{SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}" ()
  call void asm sideeffect "", "~{SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23}" ()
  call void asm sideeffect "", "~{SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}" ()
  call void asm sideeffect "", "~{SGPR32_SGPR33_SGPR34_SGPR35_SGPR36_SGPR37_SGPR38_SGPR39}" ()
  call void asm sideeffect "", "~{SGPR40_SGPR41_SGPR42_SGPR43_SGPR44_SGPR45_SGPR46_SGPR47}" ()
  call void asm sideeffect "", "~{SGPR48_SGPR49_SGPR50_SGPR51_SGPR52_SGPR53_SGPR54_SGPR55}" ()
  call void asm sideeffect "", "~{SGPR56_SGPR57_SGPR58_SGPR59_SGPR60_SGPR61_SGPR62_SGPR63}" ()
  call void asm sideeffect "", "~{SGPR64_SGPR65_SGPR66_SGPR67_SGPR68_SGPR69_SGPR70_SGPR71}" ()
  call void asm sideeffect "", "~{SGPR72_SGPR73_SGPR74_SGPR75_SGPR76_SGPR77_SGPR78_SGPR79}" ()
  call void asm sideeffect "", "~{SGPR80_SGPR81_SGPR82_SGPR83_SGPR84_SGPR85_SGPR86_SGPR87}" ()
  call void asm sideeffect "", "~{SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95}" ()
  call void asm sideeffect "", "~{SGPR96_SGPR97_SGPR98_SGPR99_SGPR100_SGPR101_SGPR102_SGPR103}" ()
  call void asm sideeffect "", "~{VCC}" ()
  ret void
}

; ERROR: error: scalar registers limit of 104 exceeded (106) in use_too_many_sgprs_bonaire_flat_scr
define void @use_too_many_sgprs_bonaire_flat_scr() #1 {
  call void asm sideeffect "", "~{SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7}" ()
  call void asm sideeffect "", "~{SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}" ()
  call void asm sideeffect "", "~{SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23}" ()
  call void asm sideeffect "", "~{SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}" ()
  call void asm sideeffect "", "~{SGPR32_SGPR33_SGPR34_SGPR35_SGPR36_SGPR37_SGPR38_SGPR39}" ()
  call void asm sideeffect "", "~{SGPR40_SGPR41_SGPR42_SGPR43_SGPR44_SGPR45_SGPR46_SGPR47}" ()
  call void asm sideeffect "", "~{SGPR48_SGPR49_SGPR50_SGPR51_SGPR52_SGPR53_SGPR54_SGPR55}" ()
  call void asm sideeffect "", "~{SGPR56_SGPR57_SGPR58_SGPR59_SGPR60_SGPR61_SGPR62_SGPR63}" ()
  call void asm sideeffect "", "~{SGPR64_SGPR65_SGPR66_SGPR67_SGPR68_SGPR69_SGPR70_SGPR71}" ()
  call void asm sideeffect "", "~{SGPR72_SGPR73_SGPR74_SGPR75_SGPR76_SGPR77_SGPR78_SGPR79}" ()
  call void asm sideeffect "", "~{SGPR80_SGPR81_SGPR82_SGPR83_SGPR84_SGPR85_SGPR86_SGPR87}" ()
  call void asm sideeffect "", "~{SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95}" ()
  call void asm sideeffect "", "~{SGPR96_SGPR97_SGPR98_SGPR99_SGPR100_SGPR101_SGPR102_SGPR103}" ()
  call void asm sideeffect "", "~{VCC}" ()
  call void asm sideeffect "", "~{FLAT_SCR}" ()
  ret void
}

; ERROR: error: scalar registers limit of 96 exceeded (98) in use_too_many_sgprs_iceland
define void @use_too_many_sgprs_iceland() #2 {
  call void asm sideeffect "", "~{VCC}" ()
  call void asm sideeffect "", "~{SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7}" ()
  call void asm sideeffect "", "~{SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}" ()
  call void asm sideeffect "", "~{SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23}" ()
  call void asm sideeffect "", "~{SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}" ()
  call void asm sideeffect "", "~{SGPR32_SGPR33_SGPR34_SGPR35_SGPR36_SGPR37_SGPR38_SGPR39}" ()
  call void asm sideeffect "", "~{SGPR40_SGPR41_SGPR42_SGPR43_SGPR44_SGPR45_SGPR46_SGPR47}" ()
  call void asm sideeffect "", "~{SGPR48_SGPR49_SGPR50_SGPR51_SGPR52_SGPR53_SGPR54_SGPR55}" ()
  call void asm sideeffect "", "~{SGPR56_SGPR57_SGPR58_SGPR59_SGPR60_SGPR61_SGPR62_SGPR63}" ()
  call void asm sideeffect "", "~{SGPR64_SGPR65_SGPR66_SGPR67_SGPR68_SGPR69_SGPR70_SGPR71}" ()
  call void asm sideeffect "", "~{SGPR72_SGPR73_SGPR74_SGPR75_SGPR76_SGPR77_SGPR78_SGPR79}" ()
  call void asm sideeffect "", "~{SGPR80_SGPR81_SGPR82_SGPR83_SGPR84_SGPR85_SGPR86_SGPR87}" ()
  call void asm sideeffect "", "~{SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95}" ()
  ret void
}

; ERROR: error: addressable scalar registers limit of 102 exceeded (103) in use_too_many_sgprs_fiji
define void @use_too_many_sgprs_fiji() #3 {
  call void asm sideeffect "", "~{SGPR0_SGPR1_SGPR2_SGPR3_SGPR4_SGPR5_SGPR6_SGPR7}" ()
  call void asm sideeffect "", "~{SGPR8_SGPR9_SGPR10_SGPR11_SGPR12_SGPR13_SGPR14_SGPR15}" ()
  call void asm sideeffect "", "~{SGPR16_SGPR17_SGPR18_SGPR19_SGPR20_SGPR21_SGPR22_SGPR23}" ()
  call void asm sideeffect "", "~{SGPR24_SGPR25_SGPR26_SGPR27_SGPR28_SGPR29_SGPR30_SGPR31}" ()
  call void asm sideeffect "", "~{SGPR32_SGPR33_SGPR34_SGPR35_SGPR36_SGPR37_SGPR38_SGPR39}" ()
  call void asm sideeffect "", "~{SGPR40_SGPR41_SGPR42_SGPR43_SGPR44_SGPR45_SGPR46_SGPR47}" ()
  call void asm sideeffect "", "~{SGPR48_SGPR49_SGPR50_SGPR51_SGPR52_SGPR53_SGPR54_SGPR55}" ()
  call void asm sideeffect "", "~{SGPR56_SGPR57_SGPR58_SGPR59_SGPR60_SGPR61_SGPR62_SGPR63}" ()
  call void asm sideeffect "", "~{SGPR64_SGPR65_SGPR66_SGPR67_SGPR68_SGPR69_SGPR70_SGPR71}" ()
  call void asm sideeffect "", "~{SGPR72_SGPR73_SGPR74_SGPR75_SGPR76_SGPR77_SGPR78_SGPR79}" ()
  call void asm sideeffect "", "~{SGPR80_SGPR81_SGPR82_SGPR83_SGPR84_SGPR85_SGPR86_SGPR87}" ()
  call void asm sideeffect "", "~{SGPR88_SGPR89_SGPR90_SGPR91_SGPR92_SGPR93_SGPR94_SGPR95}" ()
  call void asm sideeffect "", "~{SGPR96_SGPR97_SGPR98_SGPR99}" ()
  call void asm sideeffect "", "~{SGPR100_SGPR101}" ()
  call void asm sideeffect "", "~{SGPR102}" ()
  ret void
}

attributes #0 = { nounwind "target-cpu"="tahiti" }
attributes #1 = { nounwind "target-cpu"="bonaire" }
attributes #2 = { nounwind "target-cpu"="iceland" }
attributes #3 = { nounwind "target-cpu"="fiji" }
