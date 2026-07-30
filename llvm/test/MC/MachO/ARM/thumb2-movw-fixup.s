@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7-apple-darwin10 -filetype=obj -o - < %s | llvm-readobj -r --expand-relocs - | FileCheck %s

@ rdar://10038370

	.syntax unified
  .text
	.align	2
	.code	16           
	.thumb_func	_foo
  movw	r2, :lower16:L1
	movt	r2, :upper16:L1
  movw	r12, :lower16:L2
	movt	r12, :upper16:L2
  .space 70000
  
  .data
L1: .long 0
L2: .long 0

@ CHECK: Format: Mach-O arm
@ CHECK: Arch: arm
@ CHECK: AddressSize: 32bit
@ CHECK: Relocations [
@ CHECK:   Section __text {
@ CHECK:     Relocation {
@ CHECK:       Offset: 0xC
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_HALF (8)
@ CHECK:       Section: __data (2)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x1184
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_PAIR (1)
@ CHECK:       Section: - (16777215)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x8
@ CHECK:       PCRel: 0
@ CHECK:       Length: 2
@ CHECK:       Type: ARM_RELOC_HALF (8)
@ CHECK:       Section: __data (2)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x1
@ CHECK:       PCRel: 0
@ CHECK:       Length: 2
@ CHECK:       Type: ARM_RELOC_PAIR (1)
@ CHECK:       Section: - (16777215)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x4
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_HALF (8)
@ CHECK:       Section: __data (2)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x1180
@ CHECK:       PCRel: 0
@ CHECK:       Length: 3
@ CHECK:       Type: ARM_RELOC_PAIR (1)
@ CHECK:       Section: - (16777215)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x0
@ CHECK:       PCRel: 0
@ CHECK:       Length: 2
@ CHECK:       Type: ARM_RELOC_HALF (8)
@ CHECK:       Section: __data (2)
@ CHECK:     }
@ CHECK:     Relocation {
@ CHECK:       Offset: 0x1
@ CHECK:       PCRel: 0
@ CHECK:       Length: 2
@ CHECK:       Type: ARM_RELOC_PAIR (1)
@ CHECK:       Section: - (16777215)
@ CHECK:     }
@ CHECK:   }
@ CHECK: ]
