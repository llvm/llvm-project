@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readelf -s - | FileCheck %s

@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s \
@ RUN:   | FileCheck --check-prefix=ASM %s

@ CHECK:      Num:    Value  Size Type    Bind   Vis      Ndx Name
@ CHECK-NEXT:   0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
@ CHECK-NEXT:   1: 00000000     0 FUNC    LOCAL  DEFAULT    2 arm_func
@ CHECK-NEXT:   2: 00000000     0 NOTYPE  LOCAL  DEFAULT    2 $a.0
@ CHECK-NEXT:   3: 00000001     0 FUNC    LOCAL  DEFAULT    2 alias_arm_func
@ CHECK-NEXT:   4: 00000001     0 FUNC    LOCAL  DEFAULT    2 alias_arm_func2
@ CHECK-NEXT:   5: 00000001     0 FUNC    LOCAL  DEFAULT    2 alias_arm_func3
@ CHECK-NEXT:   6: 00000005     0 FUNC    LOCAL  DEFAULT    2 thumb_func
@ CHECK-NEXT:   7: 00000004     0 NOTYPE  LOCAL  DEFAULT    2 $t.1
@ CHECK-NEXT:   8: 00000005     0 FUNC    LOCAL  DEFAULT    2 alias_thumb_func
@ CHECK-NEXT:   9: 5eed1e55     0 FUNC    LOCAL  DEFAULT  ABS seedless
@ CHECK-NEXT:  10: e665a1ad     0 FUNC    LOCAL  DEFAULT  ABS eggsalad
@ CHECK-NEXT:  11: face1e55     0 FUNC    LOCAL  DEFAULT  ABS faceless
@ CHECK-NEXT:  12: 00000000     0 OBJECT  LOCAL  DEFAULT    3 alias_undefined_data
@ CHECK-NEXT:  13: 00000000     0 OBJECT  LOCAL  DEFAULT    3 badblood
@ CHECK-NEXT:  14: 00000004     0 OBJECT  LOCAL  DEFAULT    3 bedazzle
@ CHECK-NEXT:  15: 00000005     0 FUNC    LOCAL  DEFAULT    3 alias_defined_data
@ CHECK-NEXT:  16: 00000007     0 FUNC    LOCAL  DEFAULT    2 alpha
@ CHECK-NEXT:  17: 00000007     0 FUNC    LOCAL  DEFAULT    2 beta

	.syntax unified

	.arm

	.type arm_func,%function
arm_func:
	nop

	.thumb_set alias_arm_func, arm_func

        alias_arm_func2 = alias_arm_func
        alias_arm_func3 = alias_arm_func2

@ ASM: .thumb_set alias_arm_func, arm_func

	.thumb

	.type thumb_func,%function
	.thumb_func
thumb_func:
	nop

	.thumb_set alias_thumb_func, thumb_func

	.thumb_set seedless, 0x5eed1e55
	.thumb_set eggsalad, seedless + 0x87788358
	.thumb_set faceless, ~eggsalad + 0xe133c002

	.thumb_set alias_undefined_data, badblood

	.data

	.type badblood,%object
badblood:
	.long 0xbadb100d

	.type bedazzle,%object
bedazzle:
	.long 0xbeda221e

	.text
	.thumb

	.thumb_set alias_defined_data, bedazzle

	.type alpha,%function
alpha:
	nop

        .type beta,%function

	.thumb_set beta, alpha
