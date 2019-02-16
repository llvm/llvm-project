; RUN: llc -O0 -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=VGPR -check-prefix=GCN %s

; FIXME: we should disable sdwa peephole because dead-code elimination, that
; runs after peephole, ruins this test (different register numbers)

; Spill all SGPRs so multiple VGPRs are required for spilling all of them.

; Ideally we only need 2 VGPRs for all spilling. The VGPRs are
; allocated per-frame index, so it's possible to get up with more.

; GCN-LABEL: {{^}}spill_sgprs_to_multiple_vgprs:

; GCN: def s[4:11]
; GCN: def s[12:19]
; GCN: def s[20:27]
; GCN: def s[28:35]
; GCN: def s[36:43]
; GCN: def s[44:51]
; GCN: def s[52:59]
; GCN: def s[60:67]
; GCN: def s[68:75]
; GCN: def s[76:83]
; GCN: def s[84:91]

; GCN: v_writelane_b32 v0, s4, 0
; GCN-NEXT: v_writelane_b32 v0, s5, 1
; GCN-NEXT: v_writelane_b32 v0, s6, 2
; GCN-NEXT: v_writelane_b32 v0, s7, 3
; GCN-NEXT: v_writelane_b32 v0, s8, 4
; GCN-NEXT: v_writelane_b32 v0, s9, 5
; GCN-NEXT: v_writelane_b32 v0, s10, 6
; GCN-NEXT: v_writelane_b32 v0, s11, 7

; GCN: def s{{\[}}[[TMP_LO:[0-9]+]]:[[TMP_HI:[0-9]+]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 8
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 9
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 10
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 11
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 12
; GCN-NEXT: v_writelane_b32 v0, s9, 13
; GCN-NEXT: v_writelane_b32 v0, s10, 14
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 15

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 16
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 17
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 18
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 19
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 20
; GCN-NEXT: v_writelane_b32 v0, s9, 21
; GCN-NEXT: v_writelane_b32 v0, s10, 22
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 23

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 24
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 25
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 26
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 27
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 28
; GCN-NEXT: v_writelane_b32 v0, s9, 29
; GCN-NEXT: v_writelane_b32 v0, s10, 30
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 31

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 32
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 33
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 34
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 35
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 36
; GCN-NEXT: v_writelane_b32 v0, s9, 37
; GCN-NEXT: v_writelane_b32 v0, s10, 38
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 39

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 40
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 41
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 42
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 43
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 44
; GCN-NEXT: v_writelane_b32 v0, s9, 45
; GCN-NEXT: v_writelane_b32 v0, s10, 46
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 47

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s12, 48
; GCN-NEXT: v_writelane_b32 v0, s13, 49
; GCN-NEXT: v_writelane_b32 v0, s14, 50
; GCN-NEXT: v_writelane_b32 v0, s15, 51
; GCN-NEXT: v_writelane_b32 v0, s16, 52
; GCN-NEXT: v_writelane_b32 v0, s17, 53
; GCN-NEXT: v_writelane_b32 v0, s18, 54
; GCN-NEXT: v_writelane_b32 v0, s19, 55

; GCN-NEXT: v_writelane_b32 v0, s20, 56
; GCN-NEXT: v_writelane_b32 v0, s21, 57
; GCN-NEXT: v_writelane_b32 v0, s22, 58
; GCN-NEXT: v_writelane_b32 v0, s23, 59
; GCN-NEXT: v_writelane_b32 v0, s24, 60
; GCN-NEXT: v_writelane_b32 v0, s25, 61
; GCN-NEXT: v_writelane_b32 v0, s26, 62
; GCN-NEXT: v_writelane_b32 v0, s27, 63
; GCN-NEXT: v_writelane_b32 v1, s28, 0
; GCN-NEXT: v_writelane_b32 v1, s29, 1
; GCN-NEXT: v_writelane_b32 v1, s30, 2
; GCN-NEXT: v_writelane_b32 v1, s31, 3
; GCN-NEXT: v_writelane_b32 v1, s32, 4
; GCN-NEXT: v_writelane_b32 v1, s33, 5
; GCN-NEXT: v_writelane_b32 v1, s34, 6
; GCN-NEXT: v_writelane_b32 v1, s35, 7
; GCN-NEXT: v_writelane_b32 v1, s36, 8
; GCN-NEXT: v_writelane_b32 v1, s37, 9
; GCN-NEXT: v_writelane_b32 v1, s38, 10
; GCN-NEXT: v_writelane_b32 v1, s39, 11
; GCN-NEXT: v_writelane_b32 v1, s40, 12
; GCN-NEXT: v_writelane_b32 v1, s41, 13
; GCN-NEXT: v_writelane_b32 v1, s42, 14
; GCN-NEXT: v_writelane_b32 v1, s43, 15
; GCN-NEXT: v_writelane_b32 v1, s44, 16
; GCN-NEXT: v_writelane_b32 v1, s45, 17
; GCN-NEXT: v_writelane_b32 v1, s46, 18
; GCN-NEXT: v_writelane_b32 v1, s47, 19
; GCN-NEXT: v_writelane_b32 v1, s48, 20
; GCN-NEXT: v_writelane_b32 v1, s49, 21
; GCN-NEXT: v_writelane_b32 v1, s50, 22
; GCN-NEXT: v_writelane_b32 v1, s51, 23
; GCN-NEXT: v_writelane_b32 v1, s52, 24
; GCN-NEXT: v_writelane_b32 v1, s53, 25
; GCN-NEXT: v_writelane_b32 v1, s54, 26
; GCN-NEXT: v_writelane_b32 v1, s55, 27
; GCN-NEXT: v_writelane_b32 v1, s56, 28
; GCN-NEXT: v_writelane_b32 v1, s57, 29
; GCN-NEXT: v_writelane_b32 v1, s58, 30
; GCN-NEXT: v_writelane_b32 v1, s59, 31
; GCN-NEXT: v_writelane_b32 v1, s60, 32
; GCN-NEXT: v_writelane_b32 v1, s61, 33
; GCN-NEXT: v_writelane_b32 v1, s62, 34
; GCN-NEXT: v_writelane_b32 v1, s63, 35
; GCN-NEXT: v_writelane_b32 v1, s64, 36
; GCN-NEXT: v_writelane_b32 v1, s65, 37
; GCN-NEXT: v_writelane_b32 v1, s66, 38
; GCN-NEXT: v_writelane_b32 v1, s67, 39
; GCN-NEXT: v_writelane_b32 v1, s68, 40
; GCN-NEXT: v_writelane_b32 v1, s69, 41
; GCN-NEXT: v_writelane_b32 v1, s70, 42
; GCN-NEXT: v_writelane_b32 v1, s71, 43
; GCN-NEXT: v_writelane_b32 v1, s72, 44
; GCN-NEXT: v_writelane_b32 v1, s73, 45
; GCN-NEXT: v_writelane_b32 v1, s74, 46
; GCN-NEXT: v_writelane_b32 v1, s75, 47
; GCN-NEXT: v_writelane_b32 v1, s76, 48
; GCN-NEXT: v_writelane_b32 v1, s77, 49
; GCN-NEXT: v_writelane_b32 v1, s78, 50
; GCN-NEXT: v_writelane_b32 v1, s79, 51
; GCN-NEXT: v_writelane_b32 v1, s80, 52
; GCN-NEXT: v_writelane_b32 v1, s81, 53
; GCN-NEXT: v_writelane_b32 v1, s82, 54
; GCN-NEXT: v_writelane_b32 v1, s83, 55
; GCN-NEXT: v_writelane_b32 v1, s84, 56
; GCN-NEXT: v_writelane_b32 v1, s85, 57
; GCN-NEXT: v_writelane_b32 v1, s86, 58
; GCN-NEXT: v_writelane_b32 v1, s87, 59
; GCN-NEXT: v_writelane_b32 v1, s88, 60
; GCN-NEXT: v_writelane_b32 v1, s89, 61
; GCN-NEXT: v_writelane_b32 v1, s90, 62
; GCN-NEXT: v_writelane_b32 v1, s91, 63
; GCN-NEXT: v_writelane_b32 v2, s4, 0
; GCN-NEXT: v_writelane_b32 v2, s5, 1
; GCN-NEXT: v_writelane_b32 v2, s6, 2
; GCN-NEXT: v_writelane_b32 v2, s7, 3
; GCN-NEXT: v_writelane_b32 v2, s8, 4
; GCN-NEXT: v_writelane_b32 v2, s9, 5
; GCN-NEXT: v_writelane_b32 v2, s10, 6
; GCN-NEXT: v_writelane_b32 v2, s11, 7
; GCN: s_cbranch_scc1


; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v0, 0
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 1
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 2
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 3
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 4
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 5
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 6
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v0, 7
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v0, 48
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 49
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 50
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 51
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 52
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 53
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 54
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v0, 55
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v0, 56
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 57
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 58
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 59
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 60
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 61
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 62
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v0, 63
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 0
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 1
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 2
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 3
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 4
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 5
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 6
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 7
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 8
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 9
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 10
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 11
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 12
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 13
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 14
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 15
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 16
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 17
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 18
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 19
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 20
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 21
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 22
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 23
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 24
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 25
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 26
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 27
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 28
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 29
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 30
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 31
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 32
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 33
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 34
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 35
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 36
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 37
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 38
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 39
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 40
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 41
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 42
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 43
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 44
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 45
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 46
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 47
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 48
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 49
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 50
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 51
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 52
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 53
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 54
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 55
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO]], v1, 56
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 57
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 58
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 59
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 60
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 61
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v1, 62
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI]], v1, 63
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 8
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 9
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 10
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 11
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 12
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 13
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 14
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 15
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 16
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 17
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 18
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 19
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 20
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 21
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 22
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 23
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 24
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 25
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 26
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 27
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 28
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 29
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 30
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 31
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 32
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 33
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 34
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 35
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 36
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 37
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 38
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 39
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 40
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 41
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 42
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 43
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 44
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 45
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 46
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 47
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v2, 0
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 1
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 2
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 3
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 4
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 5
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 6
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 7
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}
define amdgpu_kernel void @spill_sgprs_to_multiple_vgprs(i32 addrspace(1)* %out, i32 %in) #0 {
  %wide.sgpr0 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr5 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr6 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr7 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr8 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr9 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr10 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr11 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr12 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr13 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr14 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr15 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr16 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret

bb0:
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr4) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr5) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr6) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr7) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr8) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr9) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr10) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr11) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr12) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr13) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr14) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr15) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr16) #0
  br label %ret

ret:
  ret void
}

; Some of the lanes of an SGPR spill are in one VGPR and some forced
; into the next available VGPR.

; GCN-LABEL: {{^}}split_sgpr_spill_2_vgprs:
; GCN: def s[4:19]
; GCN: def s[20:35]

; GCN: v_writelane_b32 v0, s4, 48
; GCN-NEXT: v_writelane_b32 v0, s5, 49
; GCN-NEXT: v_writelane_b32 v0, s6, 50
; GCN-NEXT: v_writelane_b32 v0, s7, 51
; GCN-NEXT: v_writelane_b32 v0, s8, 52
; GCN-NEXT: v_writelane_b32 v0, s9, 53
; GCN-NEXT: v_writelane_b32 v0, s10, 54
; GCN-NEXT: v_writelane_b32 v0, s11, 55
; GCN-NEXT: v_writelane_b32 v0, s12, 56
; GCN-NEXT: v_writelane_b32 v0, s13, 57
; GCN-NEXT: v_writelane_b32 v0, s14, 58
; GCN-NEXT: v_writelane_b32 v0, s15, 59
; GCN-NEXT: v_writelane_b32 v0, s16, 60
; GCN-NEXT: v_writelane_b32 v0, s17, 61
; GCN-NEXT: v_writelane_b32 v0, s18, 62
; GCN-NEXT: v_writelane_b32 v0, s19, 63

; GCN: v_readlane_b32 s4, v0, 48
; GCN-NEXT: v_readlane_b32 s5, v0, 49
; GCN-NEXT: v_readlane_b32 s6, v0, 50
; GCN-NEXT: v_readlane_b32 s7, v0, 51
; GCN-NEXT: v_readlane_b32 s8, v0, 52
; GCN-NEXT: v_readlane_b32 s9, v0, 53
; GCN-NEXT: v_readlane_b32 s10, v0, 54
; GCN-NEXT: v_readlane_b32 s11, v0, 55
; GCN-NEXT: v_readlane_b32 s12, v0, 56
; GCN-NEXT: v_readlane_b32 s13, v0, 57
; GCN-NEXT: v_readlane_b32 s14, v0, 58
; GCN-NEXT: v_readlane_b32 s15, v0, 59
; GCN-NEXT: v_readlane_b32 s16, v0, 60
; GCN-NEXT: v_readlane_b32 s17, v0, 61
; GCN-NEXT: v_readlane_b32 s18, v0, 62
; GCN-NEXT: v_readlane_b32 s19, v0, 63
define amdgpu_kernel void @split_sgpr_spill_2_vgprs(i32 addrspace(1)* %out, i32 %in) #1 {
  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr5 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <8 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0

  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret

bb0:
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<8 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %wide.sgpr4) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr5) #0
  br label %ret

ret:
  ret void
}

; The first 64 SGPR spills can go to a VGPR, but there isn't a second
; so some spills must be to memory. The last 16 element spill runs out of lanes at the 15th element.

; GCN-LABEL: {{^}}no_vgprs_last_sgpr_spill:

; GCN: v_writelane_b32 v23, s{{[0-9]+}}, 0
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 1
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 2
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 3
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 4
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 5
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 6
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 7
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 8
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 9
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 10
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 11
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 12
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 13
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 14
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 15

; GCN: v_writelane_b32 v23, s{{[0-9]+}}, 16
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 17
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 18
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 19
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 20
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 21
; GCN-NEXT: v_writelane_b32 v23, s{{[0-9]+}}, 22
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 23
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 24
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 25
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 26
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 27
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 28
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 29
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 30
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 31

; GCN: def s[0:1]
; GCN:      v_writelane_b32 v23, s20, 32
; GCN-NEXT: v_writelane_b32 v23, s21, 33

; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 34
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 35
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 36
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 37
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 38
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 39
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 40
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 41
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 42
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 43
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 44
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 45
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 46
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 47
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 48
; GCN-NEXT: v_writelane_b32 v23, s{{[[0-9]+}}, 49

; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: s_cbranch_scc1


; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v23, 0
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 1
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 2
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 3
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 4
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 5
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 6
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 7
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 8
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 9
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 10
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 11
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 12
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 13
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 14
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v23, 15
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}


; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v23, 32
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 33
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 34
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 35
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 36
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 37
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 38
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 39
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 40
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 41
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 42
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 43
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 44
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 45
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 46
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v23, 47
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v23, 16
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 17
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 18
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 19
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 20
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 21
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 22
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 23
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 24
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 25
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 26
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 27
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 28
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 29
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 30
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v23, 31
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}

; GCN: v_readfirstlane_b32 s1, v0
; GCN: ;;#ASMSTART
; GCN: ; use s[0:1]
define amdgpu_kernel void @no_vgprs_last_sgpr_spill(i32 addrspace(1)* %out, i32 %in) #1 {
  call void asm sideeffect "", "~{v[0:7]}" () #0
  call void asm sideeffect "", "~{v[8:15]}" () #0
  call void asm sideeffect "", "~{v[16:19]}"() #0
  call void asm sideeffect "", "~{v[20:21]}"() #0
  call void asm sideeffect "", "~{v22}"() #0

  %wide.sgpr0 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <16 x i32> asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr4 = call <2 x i32> asm sideeffect "; def $0", "=s" () #0
  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret

bb0:
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<16 x i32> %wide.sgpr3) #0
  call void asm sideeffect "; use $0", "s"(<2 x i32> %wide.sgpr4) #0
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="10,10" }
