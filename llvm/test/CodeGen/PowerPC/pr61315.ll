; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mcpu=pwr9 -mtriple=powerpc64 < %s | FileCheck %s
define dso_local <16 x i8> @ConvertExtractedMaskBitsToVect(<16 x i8> noundef %0) local_unnamed_addr #0 {
; CHECK: .LCPI0_0:
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-LABEL: ConvertExtractedMaskBitsToVect:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addis r3, r2, .LCPI0_0@toc@ha
; CHECK-NEXT:    xxlxor v3, v3, v3
; CHECK-NEXT:    addi r3, r3, .LCPI0_0@toc@l
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    addis r3, r2, .LCPI0_1@toc@ha
; CHECK-NEXT:    addi r3, r3, .LCPI0_1@toc@l
; CHECK-NEXT:    xxperm v2, v3, vs0
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    xxland v2, v2, vs0
; CHECK-NEXT:    vcmpequb v2, v2, v3
; CHECK-NEXT:    xxlnor v2, v2, v2
; CHECK-NEXT:    blr
  %a4 = extractelement <16 x i8> %0, i64 7
  %a5 = zext i8 %a4 to i16
  %a6 = insertelement <8 x i16> poison, i16 %a5, i64 0
  %a7 = bitcast <8 x i16> %a6 to <16 x i8>
  %a8 = shufflevector <16 x i8> %a7, <16 x i8> undef, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a9 = and <16 x i8> %a8, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a10 = icmp eq <16 x i8> %a9, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a11 = sext <16 x i1> %a10 to <16 x i8>
  ret <16 x i8> %a11
}

define dso_local <16 x i8> @ConvertExtractedMaskBitsToVect2(<16 x i8> noundef %0) local_unnamed_addr #0 {
; CHECK: .LCPI1_0:
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	23                               # 0x17
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-NEXT: .byte	0                                # 0x0
; CHECK-LABEL: ConvertExtractedMaskBitsToVect2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addis r3, r2, .LCPI1_0@toc@ha
; CHECK-NEXT:    xxlxor v3, v3, v3
; CHECK-NEXT:    addi r3, r3, .LCPI1_0@toc@l
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    addis r3, r2, .LCPI1_1@toc@ha
; CHECK-NEXT:    addi r3, r3, .LCPI1_1@toc@l
; CHECK-NEXT:    xxperm v2, v3, vs0
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    xxland v2, v2, vs0
; CHECK-NEXT:    vcmpequb v2, v2, v3
; CHECK-NEXT:    xxlnor v2, v2, v2
; CHECK-NEXT:    blr
  %a4 = extractelement <16 x i8> %0, i64 7
  %a5 = zext i8 %a4 to i32
  %a6 = insertelement <4 x i32> poison, i32 %a5, i64 0
  %a7 = bitcast <4 x i32> %a6 to <16 x i8>
  %a8 = shufflevector <16 x i8> %a7, <16 x i8> undef, <16 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a9 = and <16 x i8> %a8, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a10 = icmp eq <16 x i8> %a9, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a11 = sext <16 x i1> %a10 to <16 x i8>
  ret <16 x i8> %a11
}

define dso_local <16 x i8> @ConvertExtractedMaskBitsToVect3(<8 x i16> noundef %0) local_unnamed_addr #0 {
; CHECK: .LCPI2_0:
; CHECK-NEXT: .byte	22                                # 0x16
; CHECK-NEXT: .byte	23                                # 0x17
; CHECK-NEXT: .byte	22                                # 0x16
; CHECK-NEXT: .byte	23                                # 0x17
; CHECK-NEXT: .byte	22                                # 0x16
; CHECK-NEXT: .byte	23                                # 0x17
; CHECK-NEXT: .byte	22                                # 0x16
; CHECK-NEXT: .byte	23                                # 0x17
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-NEXT: .byte	0                                 # 0x0
; CHECK-LABEL: ConvertExtractedMaskBitsToVect3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addis r3, r2, .LCPI2_0@toc@ha
; CHECK-NEXT:    xxlxor v3, v3, v3
; CHECK-NEXT:    addi r3, r3, .LCPI2_0@toc@l
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    addis r3, r2, .LCPI2_1@toc@ha
; CHECK-NEXT:    addi r3, r3, .LCPI2_1@toc@l
; CHECK-NEXT:    xxperm v2, v3, vs0
; CHECK-NEXT:    lxv vs0, 0(r3)
; CHECK-NEXT:    xxland v2, v2, vs0
; CHECK-NEXT:    vcmpequb v2, v2, v3
; CHECK-NEXT:    xxlnor v2, v2, v2
; CHECK-NEXT:    blr
  %a4 = extractelement <8 x i16> %0, i64 3
  %a5 = zext i16 %a4 to i32
  %a6 = insertelement <4 x i32> poison, i32 %a5, i64 0
  %a7 = bitcast <4 x i32> %a6 to <16 x i8>
  %a8 = shufflevector <16 x i8> %a7, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 2, i32 3, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  %a9 = and <16 x i8> %a8, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a10 = icmp eq <16 x i8> %a9, <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>
  %a11 = sext <16 x i1> %a10 to <16 x i8>
  ret <16 x i8> %a11
}
