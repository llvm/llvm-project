; RUN: llc -O0 -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=VGPR -check-prefix=GCN %s

; FIXME: we should disable sdwa peephole because dead-code elimination, that
; runs after peephole, ruins this test (different register numbers)

; Spill all SGPRs so multiple VGPRs are required for spilling all of them.

; Ideally we only need 2 VGPRs for all spilling. The VGPRs are
; allocated per-frame index, so it's possible to get up with more.

; GCN-LABEL: {{^}}spill_sgprs_to_multiple_vgprs:

; GCN: def s[8:15]
; GCN: def s[16:23]
; GCN: def s[24:31]
; GCN: def s[32:39]
; GCN: def s[40:47]
; GCN: def s[48:55]
; GCN: def s[56:63]
; GCN: def s[64:71]
; GCN: def s[72:79]
; GCN: def s[80:87]
; GCN: def s[88:95]

; GCN: v_writelane_b32 v0, s8, 0
; GCN-NEXT: v_writelane_b32 v0, s9, 1
; GCN-NEXT: v_writelane_b32 v0, s10, 2
; GCN-NEXT: v_writelane_b32 v0, s11, 3
; GCN-NEXT: v_writelane_b32 v0, s12, 4
; GCN-NEXT: v_writelane_b32 v0, s13, 5
; GCN-NEXT: v_writelane_b32 v0, s14, 6
; GCN-NEXT: v_writelane_b32 v0, s15, 7

; GCN: def s{{\[}}[[TMP_LO:[0-9]+]]:[[TMP_HI:[0-9]+]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 8
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 9
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 10
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 11
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 12
; GCN-NEXT: v_writelane_b32 v0, s13, 13
; GCN-NEXT: v_writelane_b32 v0, s14, 14
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 15

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 16
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 17
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 18
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 19
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 20
; GCN-NEXT: v_writelane_b32 v0, s13, 21
; GCN-NEXT: v_writelane_b32 v0, s14, 22
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 23

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 24
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 25
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 26
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 27
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 28
; GCN-NEXT: v_writelane_b32 v0, s13, 29
; GCN-NEXT: v_writelane_b32 v0, s14, 30
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 31

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 32
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 33
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 34
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 35
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 36
; GCN-NEXT: v_writelane_b32 v0, s13, 37
; GCN-NEXT: v_writelane_b32 v0, s14, 38
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 39

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 40
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 41
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 42
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 43
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 44
; GCN-NEXT: v_writelane_b32 v0, s13, 45
; GCN-NEXT: v_writelane_b32 v0, s14, 46
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 47

; GCN: def s{{\[}}[[TMP_LO]]:[[TMP_HI]]{{\]}}
; GCN: v_writelane_b32 v0, s[[TMP_LO]], 48
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 49
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 50
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 51
; GCN-NEXT: v_writelane_b32 v0, s{{[0-9]+}}, 52
; GCN-NEXT: v_writelane_b32 v0, s13, 53
; GCN-NEXT: v_writelane_b32 v0, s14, 54
; GCN-NEXT: v_writelane_b32 v0, s[[TMP_HI]], 55

; GCN-NEXT: v_writelane_b32 v0, s88, 56
; GCN-NEXT: v_writelane_b32 v0, s89, 57
; GCN-NEXT: v_writelane_b32 v0, s90, 58
; GCN-NEXT: v_writelane_b32 v0, s91, 59
; GCN-NEXT: v_writelane_b32 v0, s92, 60
; GCN-NEXT: v_writelane_b32 v0, s93, 61
; GCN-NEXT: v_writelane_b32 v0, s94, 62
; GCN-NEXT: v_writelane_b32 v0, s95, 63
; GCN-NEXT: v_writelane_b32 v1, s16, 0
; GCN-NEXT: v_writelane_b32 v1, s17, 1
; GCN-NEXT: v_writelane_b32 v1, s18, 2
; GCN-NEXT: v_writelane_b32 v1, s19, 3
; GCN-NEXT: v_writelane_b32 v1, s20, 4
; GCN-NEXT: v_writelane_b32 v1, s21, 5
; GCN-NEXT: v_writelane_b32 v1, s22, 6
; GCN-NEXT: v_writelane_b32 v1, s23, 7
; GCN-NEXT: v_writelane_b32 v1, s24, 8
; GCN-NEXT: v_writelane_b32 v1, s25, 9
; GCN-NEXT: v_writelane_b32 v1, s26, 10
; GCN-NEXT: v_writelane_b32 v1, s27, 11
; GCN-NEXT: v_writelane_b32 v1, s28, 12
; GCN-NEXT: v_writelane_b32 v1, s29, 13
; GCN-NEXT: v_writelane_b32 v1, s30, 14
; GCN-NEXT: v_writelane_b32 v1, s31, 15
; GCN-NEXT: v_writelane_b32 v1, s32, 16
; GCN-NEXT: v_writelane_b32 v1, s33, 17
; GCN-NEXT: v_writelane_b32 v1, s34, 18
; GCN-NEXT: v_writelane_b32 v1, s35, 19
; GCN-NEXT: v_writelane_b32 v1, s36, 20
; GCN-NEXT: v_writelane_b32 v1, s37, 21
; GCN-NEXT: v_writelane_b32 v1, s38, 22
; GCN-NEXT: v_writelane_b32 v1, s39, 23
; GCN-NEXT: v_writelane_b32 v1, s40, 24
; GCN-NEXT: v_writelane_b32 v1, s41, 25
; GCN-NEXT: v_writelane_b32 v1, s42, 26
; GCN-NEXT: v_writelane_b32 v1, s43, 27
; GCN-NEXT: v_writelane_b32 v1, s44, 28
; GCN-NEXT: v_writelane_b32 v1, s45, 29
; GCN-NEXT: v_writelane_b32 v1, s46, 30
; GCN-NEXT: v_writelane_b32 v1, s47, 31
; GCN-NEXT: v_writelane_b32 v1, s48, 32
; GCN-NEXT: v_writelane_b32 v1, s49, 33
; GCN-NEXT: v_writelane_b32 v1, s50, 34
; GCN-NEXT: v_writelane_b32 v1, s51, 35
; GCN-NEXT: v_writelane_b32 v1, s52, 36
; GCN-NEXT: v_writelane_b32 v1, s53, 37
; GCN-NEXT: v_writelane_b32 v1, s54, 38
; GCN-NEXT: v_writelane_b32 v1, s55, 39
; GCN-NEXT: v_writelane_b32 v1, s56, 40
; GCN-NEXT: v_writelane_b32 v1, s57, 41
; GCN-NEXT: v_writelane_b32 v1, s58, 42
; GCN-NEXT: v_writelane_b32 v1, s59, 43
; GCN-NEXT: v_writelane_b32 v1, s60, 44
; GCN-NEXT: v_writelane_b32 v1, s61, 45
; GCN-NEXT: v_writelane_b32 v1, s62, 46
; GCN-NEXT: v_writelane_b32 v1, s63, 47
; GCN-NEXT: v_writelane_b32 v1, s64, 48
; GCN-NEXT: v_writelane_b32 v1, s65, 49
; GCN-NEXT: v_writelane_b32 v1, s66, 50
; GCN-NEXT: v_writelane_b32 v1, s67, 51
; GCN-NEXT: v_writelane_b32 v1, s68, 52
; GCN-NEXT: v_writelane_b32 v1, s69, 53
; GCN-NEXT: v_writelane_b32 v1, s70, 54
; GCN-NEXT: v_writelane_b32 v1, s71, 55
; GCN-NEXT: v_writelane_b32 v1, s72, 56
; GCN-NEXT: v_writelane_b32 v1, s73, 57
; GCN-NEXT: v_writelane_b32 v1, s74, 58
; GCN-NEXT: v_writelane_b32 v1, s75, 59
; GCN-NEXT: v_writelane_b32 v1, s76, 60
; GCN-NEXT: v_writelane_b32 v1, s77, 61
; GCN-NEXT: v_writelane_b32 v1, s78, 62
; GCN-NEXT: v_writelane_b32 v1, s79, 63
; GCN-NEXT: v_writelane_b32 v2, s80, 0
; GCN-NEXT: v_writelane_b32 v2, s81, 1
; GCN-NEXT: v_writelane_b32 v2, s82, 2
; GCN-NEXT: v_writelane_b32 v2, s83, 3
; GCN-NEXT: v_writelane_b32 v2, s84, 4
; GCN-NEXT: v_writelane_b32 v2, s85, 5
; GCN-NEXT: v_writelane_b32 v2, s86, 6
; GCN-NEXT: v_writelane_b32 v2, s87, 7
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

; GCN: v_readlane_b32 s{{[0-9]+}}, v2, 0
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 1
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 2
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 3
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 4
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 5
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 6
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v2, 7
; GCN: ; use s{{\[}}[[USE_TMP_LO]]:[[USE_TMP_HI]]{{\]}}

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 56
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 57
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 58
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 59
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 60
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 61
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 62
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 63
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

; GCN: v_readlane_b32 s{{[0-9]+}}, v0, 48
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 49
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 50
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 51
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 52
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 53
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 54
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v0, 55
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
; GCN: def s[24:39]

; GCN: v_writelane_b32 v0, s24, 50
; GCN-NEXT: v_writelane_b32 v0, s25, 51
; GCN-NEXT: v_writelane_b32 v0, s26, 52
; GCN-NEXT: v_writelane_b32 v0, s27, 53
; GCN-NEXT: v_writelane_b32 v0, s28, 54
; GCN-NEXT: v_writelane_b32 v0, s29, 55
; GCN-NEXT: v_writelane_b32 v0, s30, 56
; GCN-NEXT: v_writelane_b32 v0, s31, 57
; GCN-NEXT: v_writelane_b32 v0, s32, 58
; GCN-NEXT: v_writelane_b32 v0, s33, 59
; GCN-NEXT: v_writelane_b32 v0, s34, 60
; GCN-NEXT: v_writelane_b32 v0, s35, 61
; GCN-NEXT: v_writelane_b32 v0, s36, 62
; GCN-NEXT: v_writelane_b32 v0, s37, 63
; GCN-NEXT: v_writelane_b32 v1, s38, 0
; GCN-NEXT: v_writelane_b32 v1, s39, 1

; GCN: v_readlane_b32 s4, v0, 50
; GCN-NEXT: v_readlane_b32 s5, v0, 51
; GCN-NEXT: v_readlane_b32 s6, v0, 52
; GCN-NEXT: v_readlane_b32 s7, v0, 53
; GCN-NEXT: v_readlane_b32 s8, v0, 54
; GCN-NEXT: v_readlane_b32 s9, v0, 55
; GCN-NEXT: v_readlane_b32 s10, v0, 56
; GCN-NEXT: v_readlane_b32 s11, v0, 57
; GCN-NEXT: v_readlane_b32 s12, v0, 58
; GCN-NEXT: v_readlane_b32 s13, v0, 59
; GCN-NEXT: v_readlane_b32 s14, v0, 60
; GCN-NEXT: v_readlane_b32 s15, v0, 61
; GCN-NEXT: v_readlane_b32 s16, v0, 62
; GCN-NEXT: v_readlane_b32 s17, v0, 63
; GCN-NEXT: v_readlane_b32 s18, v1, 0
; GCN-NEXT: v_readlane_b32 s19, v1, 1
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
; GCN:      v_writelane_b32 v23, s0, 32
; GCN-NEXT: v_writelane_b32 v23, s1, 33

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
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
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


; GCN: v_readlane_b32 s[[USE_TMP_LO:[0-9]+]], v23, 34
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
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 47
; GCN-NEXT: v_readlane_b32 s{{[0-9]+}}, v23, 48
; GCN-NEXT: v_readlane_b32 s[[USE_TMP_HI:[0-9]+]], v23, 49
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
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}
; GCN: buffer_load_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}

; GCN: v_readlane_b32 s0, v23, 32
; GCN: v_readlane_b32 s1, v23, 33
; GCN: ;;#ASMSTART
; GCN: ; use s[0:1]
define amdgpu_kernel void @no_vgprs_last_sgpr_spill(i32 addrspace(1)* %out, i32 %in) #1 {
  call void asm sideeffect "", "~{VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7}" () #0
  call void asm sideeffect "", "~{VGPR8_VGPR9_VGPR10_VGPR11_VGPR12_VGPR13_VGPR14_VGPR15}" () #0
  call void asm sideeffect "", "~{VGPR16_VGPR17_VGPR18_VGPR19}"() #0
  call void asm sideeffect "", "~{VGPR20_VGPR21}"() #0
  call void asm sideeffect "", "~{VGPR22}"() #0

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
