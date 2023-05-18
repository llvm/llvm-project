@ RUN: llvm-mc -triple armv7-linux-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj -S --sd --sr - > %t
@ RUN: FileCheck %s < %t
@ RUN: FileCheck --check-prefix=RELOC %s < %t

	.syntax unified
	.thumb


	.section .pr0

	.global pr0
	.type pr0,%function
	.thumb_func
pr0:
	.fnstart
	.personalityindex 0
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr0
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B0B080
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr0
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0
@ RELOC:     0x0 R_ARM_PREL31 .pr0
@ RELOC:   ]
@ RELOC: }

	.section .pr0.nontrivial

	.global pr0_nontrivial
	.type pr0_nontrivial,%function
	.thumb_func
pr0_nontrivial:
	.fnstart
	.personalityindex 0
	.pad #0x10
	sub sp, sp, #0x10
	add sp, sp, #0x10
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr0.nontrivial
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B0B00380
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr0.nontrivial
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr0
@ RELOC:     0x0 R_ARM_PREL31 .pr0.nontrivial
@ RELOC:   ]
@ RELOC: }

	.section .pr1

	.global pr1
	.type pr1,%function
	.thumb_func
pr1:
	.fnstart
	.personalityindex 1
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.pr1
@ CHECK:   SectionData (
@ CHECK:     0000: B0B00081 00000000
@ CHECK:   )
@ CHECK: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr1
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr1
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr1
@ RELOC:     0x0 R_ARM_PREL31 .pr1
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.pr1
@ RELOC:   ]
@ RELOC: }

	.section .pr1.nontrivial

	.global pr1_nontrivial
	.type pr1_nontrivial,%function
	.thumb_func
pr1_nontrivial:
	.fnstart
	.personalityindex 1
	.pad #0x10
	sub sp, sp, #0x10
	add sp, sp, #0x10
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.pr1.nontrivial
@ CHECK:   SectionData (
@ CHECK:     0000: B0030081 00000000
@ CHECK:   )
@ CHECK: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr1.nontrivial
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr1.nontrivial
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr1
@ RELOC:     0x0 R_ARM_PREL31 .pr1.nontrivial
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.pr1.nontrivial
@ RELOC:   ]
@ RELOC: }

	.section .pr2

	.global pr2
	.type pr2,%function
	.thumb_func
pr2:
	.fnstart
	.personalityindex 2
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.pr2
@ CHECK:   SectionData (
@ CHECK:     0000: B0B00082 00000000
@ CHECK:   )
@ CHECK: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr2
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr2
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr2
@ RELOC:     0x0 R_ARM_PREL31 .pr2
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.pr2
@ RELOC:   ]
@ RELOC: }

	.section .pr2.nontrivial
	.type pr2_nontrivial,%function
	.thumb_func
pr2_nontrivial:
	.fnstart
	.personalityindex 2
	.pad #0x10
	sub sp, sp, #0x10
	add sp, sp, #0x10
	bx lr
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.extab.pr2.nontrivial
@ CHECK:   SectionData (
@ CHECK:     0000: B0030082 00000000
@ CHECK:   )
@ CHECK: }

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.pr2.nontrivial
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 00000000
@ CHECK:   )
@ CHECK: }

@ RELOC: Section {
@ RELOC:   Name: .rel.ARM.exidx.pr2.nontrivial
@ RELOC:   Relocations [
@ RELOC:     0x0 R_ARM_NONE __aeabi_unwind_cpp_pr2
@ RELOC:     0x0 R_ARM_PREL31 .pr2.nontrivial
@ RELOC:     0x4 R_ARM_PREL31 .ARM.extab.pr2.nontrivial
@ RELOC:   ]
@ RELOC: }
