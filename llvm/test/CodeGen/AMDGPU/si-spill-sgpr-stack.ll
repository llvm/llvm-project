; RUN: llc -march=amdgcn -mcpu=fiji < %s | FileCheck %s

; Make sure this doesn't crash.
; CHECK: {{^}}test:
; Make sure we are handling hazards correctly.
; CHECK: buffer_load_dword [[VHI:v[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], s{{[0-9]+}} offset:12
; CHECK-NEXT: s_waitcnt vmcnt(0)
; CHECK-NEXT: v_readfirstlane_b32 s[[HI:[0-9]+]], [[VHI]]
; CHECK-NEXT: s_nop 4
; CHECK-NEXT: buffer_store_dword v0, off, s[0:[[HI]]{{\]}}, 0
; CHECK: s_endpgm
define void @test(i32 addrspace(1)* %out, i32 %in) {
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
  call void asm sideeffect "", "~{VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7}" ()
  call void asm sideeffect "", "~{VGPR8_VGPR9_VGPR10_VGPR11_VGPR12_VGPR13_VGPR14_VGPR15}" ()
  call void asm sideeffect "", "~{VGPR16_VGPR17_VGPR18_VGPR19_VGPR20_VGPR21_VGPR22_VGPR23}" ()
  call void asm sideeffect "", "~{VGPR24_VGPR25_VGPR26_VGPR27_VGPR28_VGPR29_VGPR30_VGPR31}" ()
  call void asm sideeffect "", "~{VGPR32_VGPR33_VGPR34_VGPR35_VGPR36_VGPR37_VGPR38_VGPR39}" ()
  call void asm sideeffect "", "~{VGPR40_VGPR41_VGPR42_VGPR43_VGPR44_VGPR45_VGPR46_VGPR47}" ()
  call void asm sideeffect "", "~{VGPR48_VGPR49_VGPR50_VGPR51_VGPR52_VGPR53_VGPR54_VGPR55}" ()
  call void asm sideeffect "", "~{VGPR56_VGPR57_VGPR58_VGPR59_VGPR60_VGPR61_VGPR62_VGPR63}" ()
  call void asm sideeffect "", "~{VGPR64_VGPR65_VGPR66_VGPR67_VGPR68_VGPR69_VGPR70_VGPR71}" ()
  call void asm sideeffect "", "~{VGPR72_VGPR73_VGPR74_VGPR75_VGPR76_VGPR77_VGPR78_VGPR79}" ()
  call void asm sideeffect "", "~{VGPR80_VGPR81_VGPR82_VGPR83_VGPR84_VGPR85_VGPR86_VGPR87}" ()
  call void asm sideeffect "", "~{VGPR88_VGPR89_VGPR90_VGPR91_VGPR92_VGPR93_VGPR94_VGPR95}" ()
  call void asm sideeffect "", "~{VGPR96_VGPR97_VGPR98_VGPR99_VGPR100_VGPR101_VGPR102_VGPR103}" ()
  call void asm sideeffect "", "~{VGPR104_VGPR105_VGPR106_VGPR107_VGPR108_VGPR109_VGPR110_VGPR111}" ()
  call void asm sideeffect "", "~{VGPR112_VGPR113_VGPR114_VGPR115_VGPR116_VGPR117_VGPR118_VGPR119}" ()
  call void asm sideeffect "", "~{VGPR120_VGPR121_VGPR122_VGPR123_VGPR124_VGPR125_VGPR126_VGPR127}" ()
  call void asm sideeffect "", "~{VGPR128_VGPR129_VGPR130_VGPR131_VGPR132_VGPR133_VGPR134_VGPR135}" ()
  call void asm sideeffect "", "~{VGPR136_VGPR137_VGPR138_VGPR139_VGPR140_VGPR141_VGPR142_VGPR143}" ()
  call void asm sideeffect "", "~{VGPR144_VGPR145_VGPR146_VGPR147_VGPR148_VGPR149_VGPR150_VGPR151}" ()
  call void asm sideeffect "", "~{VGPR152_VGPR153_VGPR154_VGPR155_VGPR156_VGPR157_VGPR158_VGPR159}" ()
  call void asm sideeffect "", "~{VGPR160_VGPR161_VGPR162_VGPR163_VGPR164_VGPR165_VGPR166_VGPR167}" ()
  call void asm sideeffect "", "~{VGPR168_VGPR169_VGPR170_VGPR171_VGPR172_VGPR173_VGPR174_VGPR175}" ()
  call void asm sideeffect "", "~{VGPR176_VGPR177_VGPR178_VGPR179_VGPR180_VGPR181_VGPR182_VGPR183}" ()
  call void asm sideeffect "", "~{VGPR184_VGPR185_VGPR186_VGPR187_VGPR188_VGPR189_VGPR190_VGPR191}" ()
  call void asm sideeffect "", "~{VGPR192_VGPR193_VGPR194_VGPR195_VGPR196_VGPR197_VGPR198_VGPR199}" ()
  call void asm sideeffect "", "~{VGPR200_VGPR201_VGPR202_VGPR203_VGPR204_VGPR205_VGPR206_VGPR207}" ()
  call void asm sideeffect "", "~{VGPR208_VGPR209_VGPR210_VGPR211_VGPR212_VGPR213_VGPR214_VGPR215}" ()
  call void asm sideeffect "", "~{VGPR216_VGPR217_VGPR218_VGPR219_VGPR220_VGPR221_VGPR222_VGPR223}" ()
  call void asm sideeffect "", "~{VGPR224_VGPR225_VGPR226_VGPR227_VGPR228_VGPR229_VGPR230_VGPR231}" ()
  call void asm sideeffect "", "~{VGPR232_VGPR233_VGPR234_VGPR235_VGPR236_VGPR237_VGPR238_VGPR239}" ()
  call void asm sideeffect "", "~{VGPR240_VGPR241_VGPR242_VGPR243_VGPR244_VGPR245_VGPR246_VGPR247}" ()
  call void asm sideeffect "", "~{VGPR248_VGPR249_VGPR250_VGPR251_VGPR252_VGPR253_VGPR254_VGPR255}" ()

  store i32 %in, i32 addrspace(1)* %out
  ret void
}
