@ RUN: llvm-mc %s -triple=armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -S --sd --sr > %t
@ RUN: FileCheck %s < %t
@ RUN: FileCheck --check-prefix=RELOC %s < %t

@ Check the .personality directive.

	.syntax unified

@-------------------------------------------------------------------------------
@ TEST1
@-------------------------------------------------------------------------------
	.section	.TEST1
	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	bx	lr
	.personality __gxx_personality_v0
	.handlerdata
	.fnend


@ CHECK: Section {
@ CHECK:   Name: .TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 1EFF2FE1                             |../.|
@ CHECK:   )
@ CHECK: }
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0B000                    |........|
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.extab.TEST1
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_PREL31 __gxx_personality_v0 0x0
@ RELOC:   ]
@ RELOC: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.TEST1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000                    |........|
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.TEST1
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_PREL31 .TEST1 0x0
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.TEST1 0x0
@ RELOC:   ]
@ RELOC: }


@-------------------------------------------------------------------------------
@ TEST2
@-------------------------------------------------------------------------------
	.section	.TEST2
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	bx	lr
	.personality __gxx_personality_v0
	@ The .handlerdata directive is intentionally ignored.  The .fnend		@ directive should create the EXTAB entry and flush the unwind opcodes.
	.fnend


@ CHECK: Section {
@ CHECK:   Name: .TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 1EFF2FE1                             |../.|
@ CHECK:   )
@ CHECK: }
@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0B000                    |........|
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.extab.TEST2
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_PREL31 __gxx_personality_v0 0x0
@ RELOC:   ]
@ RELOC: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.TEST2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000                    |........|
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.TEST2
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_PREL31 .TEST2 0x0
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.TEST2 0x0
@ RELOC:   ]
@ RELOC: }
