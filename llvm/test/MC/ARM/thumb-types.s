@ RUN: llvm-mc -triple armv7-elf -filetype obj -o - %s | llvm-readelf -s - | FileCheck %s

@ CHECK:      Num:    Value  Size Type    Bind   Vis      Ndx Name
@ CHECK-NEXT:   0: 00000000     0 NOTYPE  LOCAL  DEFAULT  UND
@ CHECK-NEXT:   1: 00000001     0 FUNC    LOCAL  DEFAULT    2 implicit_function
@ CHECK-NEXT:   2: 00000000     0 NOTYPE  LOCAL  DEFAULT    2 $t.0
@ CHECK-NEXT:   3: 00000002     0 OBJECT  LOCAL  DEFAULT    2 implicit_data
@ CHECK-NEXT:   4: 00000002     0 NOTYPE  LOCAL  DEFAULT    2 $d.1
@ CHECK-NEXT:   5: 00000008     0 FUNC    LOCAL  DEFAULT    2 arm_function
@ CHECK-NEXT:   6: 00000008     0 NOTYPE  LOCAL  DEFAULT    2 $a.2
@ CHECK-NEXT:   7: 0000000c     0 NOTYPE  LOCAL  DEFAULT    2 untyped_text_label
@ CHECK-NEXT:   8: 0000000c     0 NOTYPE  LOCAL  DEFAULT    2 $t.3
@ CHECK-NEXT:   9: 0000000f     0 FUNC    LOCAL  DEFAULT    2 explicit_function
@ CHECK-NEXT:  10: 00000010     0 NOTYPE  LOCAL  DEFAULT    2 $d.4
@ CHECK-NEXT:  11: 00000000     4 TLS     LOCAL  DEFAULT    5 tls
@ CHECK-NEXT:  12: 00000015     0 IFUNC   LOCAL  DEFAULT    2 indirect_function
@ CHECK-NEXT:  13: 00000014     0 NOTYPE  LOCAL  DEFAULT    2 $t.5
@ CHECK-NEXT:  14: 00000000     0 NOTYPE  LOCAL  DEFAULT    4 untyped_data_label
@ CHECK-NEXT:  15: 00000000     0 NOTYPE  LOCAL  DEFAULT    4 $t.6
@ CHECK-NEXT:  16: 00000002     0 OBJECT  LOCAL  DEFAULT    4 explicit_data
@ CHECK-NEXT:  17: 00000002     0 NOTYPE  LOCAL  DEFAULT    4 $d.7


	.syntax unified

	.thumb

	.type implicit_function,%function
implicit_function:
	nop

	.type implicit_data,%object
implicit_data:
	.long 0

	.arm
	.type arm_function,%function
arm_function:
	nop

	.thumb

	.text

untyped_text_label:
	nop

	.type explicit_function,%function
explicit_function:
	nop

	.long	tls(TPOFF)

	.type indirect_function,%gnu_indirect_function
indirect_function:
	nop

	.data

untyped_data_label:
	nop

	.type explicit_data,%object
explicit_data:
	.long 0

	.section	.tdata,"awT",%progbits
	.type	tls,%object
	.align	2
tls:
	.long	42
	.size	tls, 4

