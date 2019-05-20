@ RUN: llvm-mc < %s -triple armv7-none-linux-gnueabi -filetype=obj  | llvm-objdump -triple armv7-none-linux-gnueabi -r - | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
@ RUN: llvm-mc < %s -triple thumbv7-none-linux-gnueabi -filetype=obj  | llvm-objdump -triple thumbv7-none-linux-gnueabi -r - | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB

@ CHECK-LABEL: RELOCATION RECORDS FOR [.rel.text]
.Lsym:

@ empty
.word f00
.word f01
@CHECK: 0 R_ARM_ABS32 f00
@CHECK: 4 R_ARM_ABS32 f01

@ none
.word f02(NONE)
.word f03(none)
@CHECK: 8 R_ARM_NONE f02
@CHECK: c R_ARM_NONE f03

@ plt
bl f04(PLT)
bl f05(plt)
@ARM: 10 R_ARM_CALL f04
@ARM: 14 R_ARM_CALL f05
@THUMB: 10 R_ARM_THM_CALL f04
@THUMB: 14 R_ARM_THM_CALL f05

@ got
.word f06(GOT)
.word f07(got)
@CHECK: 18 R_ARM_GOT_BREL f06
@CHECK: 1c R_ARM_GOT_BREL f07

@ gotoff
.word f08(GOTOFF)
.word f09(gotoff)
@CHECK: 20 R_ARM_GOTOFF32 f08
@CHECK: 24 R_ARM_GOTOFF32 f09

@ tpoff
.word f10(TPOFF)
.word f11(tpoff)
@CHECK: 28 R_ARM_TLS_LE32 f10
@CHECK: 2c R_ARM_TLS_LE32 f11

@ tlsgd
.word f12(TLSGD)
.word f13(tlsgd)
@CHECK: 30 R_ARM_TLS_GD32 f12
@CHECK: 34 R_ARM_TLS_GD32 f13

@ target1
.word f14(TARGET1)
.word f15(target1)
@CHECK: 38 R_ARM_TARGET1 f14
@CHECK: 3c R_ARM_TARGET1 f15

@ target2
.word f16(TARGET2)
.word f17(target2)
@CHECK: 40 R_ARM_TARGET2 f16
@CHECK: 44 R_ARM_TARGET2 f17

@ prel31
.word f18(PREL31)
.word f19(prel31)
@CHECK: 48 R_ARM_PREL31 f18
@CHECK: 4c R_ARM_PREL31 f19

@ tlsldo
.word f20(TLSLDO)
.word f21(tlsldo)
@CHECK: 50 R_ARM_TLS_LDO32 f20
@CHECK: 54 R_ARM_TLS_LDO32 f21

@ tlscall
.word f22(TLSCALL)
.word f23(tlscall)
@ CHECK: 58 R_ARM_TLS_CALL f22
@ CHECK: 5c R_ARM_TLS_CALL f23

@ tlsdesc
.word f24(TLSDESC)
.word f25(tlsdesc)
@ CHECK: 60 R_ARM_TLS_GOTDESC f24
@ CHECK: 64 R_ARM_TLS_GOTDESC f25

@ prel31 (relative)
.word f26(PREL31)-.
.word f27(prel31)-.
@CHECK: 68 R_ARM_PREL31 f26
@CHECK: 6c R_ARM_PREL31 f27

@ tlsldm
.word f28(TLSLDM)
.word f29(tlsldm)
@CHECK: 70 R_ARM_TLS_LDM32 f28
@CHECK: 74 R_ARM_TLS_LDM32 f29

@ got_prel
.word	f30(GOT_PREL) + (. - .Lsym)
	ldr r3, =f31(GOT_PREL)
@ CHECK: 78 R_ARM_GOT_PREL f30
@ CHECK: 80 R_ARM_GOT_PREL f31
