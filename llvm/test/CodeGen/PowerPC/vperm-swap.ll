; RUN: llc -verify-machineinstrs -mcpu=pwr8 -ppc-asm-full-reg-names \
; RUN:   -ppc-vsr-nums-as-vr -mtriple=powerpc64le-unknown-linux-gnu < %s | \
; RUN: FileCheck %s --check-prefix=CHECK-LE-P8

define <16 x i8> @test_none_v16i8(i8 %arg, ptr nocapture noundef readonly %b) {
; CHECK-LE-P8: .LCPI0_0:
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   22                              # 0x16
; CHECK-LE-P8-NEXT: .byte   7                               # 0x7
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-NEXT: .byte   23                              # 0x17
; CHECK-LE-P8-LABEL: test_none_v16i8:
; CHECK-LE-P8:       # %bb.0: # %entry
; CHECK-LE-P8-NEXT:    addis r5, r2, .LCPI0_0@toc@ha
; CHECK-LE-P8-NEXT:    lxvd2x v3, 0, r4
; CHECK-LE-P8-NEXT:    mtvsrd v4, r3
; CHECK-LE-P8-NEXT:    addi r5, r5, .LCPI0_0@toc@l
; CHECK-LE-P8-NEXT:    lxvd2x vs0, 0, r5
; CHECK-LE-P8-NEXT:    xxswapd v2, vs0
; CHECK-LE-P8-NEXT:    vperm v2, v4, v3, v2
; CHECK-LE-P8-NEXT:    blr
entry:
  %lhs = load <16 x i8>, ptr %b, align 4
  %rhs = insertelement <16 x i8> undef, i8 %arg, i32 0
  %shuffle = shufflevector <16 x i8> %lhs, <16 x i8> %rhs, <16 x i32> <i32 0, i32 1, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %shuffle
}
